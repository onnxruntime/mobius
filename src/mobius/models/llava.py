# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LLaVA multimodal model (vision + text) — 3-model split.

Splits the LLaVA architecture into three ONNX models for
onnxruntime-genai:

- **decoder**: text decoder taking ``inputs_embeds``
- **vision**: CLIP/SigLIP vision tower + MLP projector
- **embedding**: token embedding + image feature fusion

Used by: llava, llava_next, llava_onevision, molmo, paligemma, pixtral,
video_llava, and other models with the CLIP+MLP+LLM pattern.

HuggingFace weight names:
- ``vision_tower.vision_model.*``
- ``multi_modal_projector.linear_1/2.*``
- ``language_model.model.* / language_model.lm_head.*``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import (
    vlm_decoder_weights,
    vlm_embedding_weights,
    vlm_vision_weights,
)
from mobius.components import (
    Embedding,
    Linear,
    MLPMultiModalProjector,
    VisionModel,
)
from mobius.models.base import TextModel

if TYPE_CHECKING:
    import onnx_ir as ir


class _LLaVADecoderModel(nn.Module):
    """LLaVA text decoder taking inputs_embeds."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return vlm_decoder_weights(state_dict, tie=self.config.tie_word_embeddings)


class _LLaVAVisionEncoderModel(nn.Module):
    """LLaVA vision encoder: CLIP/SigLIP + MLP projector."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.vision_tower = VisionModel(config)
        self.multi_modal_projector = MLPMultiModalProjector(
            vision_hidden_size=config.vision.hidden_size,
            text_hidden_size=config.hidden_size,
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        vision_features = self.vision_tower(op, pixel_values)
        return self.multi_modal_projector(op, vision_features)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return vlm_vision_weights(state_dict, ("vision_tower.", "multi_modal_projector."))


class _LLaVAEmbeddingModel(nn.Module):
    """LLaVA embedding: token lookup + image feature fusion."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.image_token_id = config.image_token_id or 0

    def forward(self, op: OpBuilder, input_ids: ir.Value, image_features: ir.Value):
        text_embeds = self.embed_tokens(op, input_ids)

        image_mask = op.Equal(
            input_ids,
            op.Constant(value_int=self.image_token_id),
        )
        image_mask_3d = op.Unsqueeze(image_mask, [-1])

        mask_int = op.Cast(image_mask, to=7)
        cumsum = op.CumSum(mask_int, 1)
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        # Pad image_features with one zero row so Gather is valid even when
        # image_features is empty (text-only input: num_image_tokens == 0).
        # The Where mask ensures the padding row is never used in the output.
        pad_row = op.Expand(
            op.CastLike(0.0, image_features),
            op.Concat(
                op.Constant(value_ints=[1]),
                op.Shape(image_features, start=1, end=2),
                axis=0,
            ),
        )
        padded_features = op.Concat(image_features, pad_row, axis=0)

        gathered = op.Gather(padded_features, indices, axis=0)
        return op.Where(image_mask_3d, gathered, text_embeds)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return vlm_embedding_weights(state_dict)


class _PixtralVisionEncoderModel(nn.Module):
    """Pixtral vision encoder: 2D RoPE vision tower + Mistral3 projector.

    Used for ``mistral3`` / ``pixtral`` model types that use the
    Pixtral vision architecture instead of CLIP/SigLIP.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        from mobius.components import (
            Mistral3MultiModalProjector,
            PixtralVisionTower,
        )

        self.vision_tower = PixtralVisionTower(config)
        self.multi_modal_projector = Mistral3MultiModalProjector(
            vision_hidden_size=config.vision.hidden_size,
            text_hidden_size=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            norm_eps=config.rms_norm_eps,
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        hidden_states, grid_h, grid_w = self.vision_tower(op, pixel_values)
        return self.multi_modal_projector(op, hidden_states, grid_h, grid_w)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in state_dict.items()
            if key.startswith(("vision_tower.", "multi_modal_projector."))
        }


class LLaVAModel(nn.Module):
    """LLaVA vision-language model (3-model split).

    Builds three separate ONNX models:
    - decoder: text decoder taking inputs_embeds
    - vision_encoder: CLIP/SigLIP + MLP projector
    - embedding: token embedding + image feature fusion
    """

    default_task: str = "vision-language"
    category: str = "Multimodal"

    # Runtime HF ``named_modules()`` sub-trees for the LLaVA/PaliGemma/Mistral3
    # layout. Decoder paths exclude token embeddings, which are a separate graph.
    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "decoder": (
            "model.language_model.layers",
            "model.language_model.norm",
            "model.language_model.rotary_emb",
            "lm_head",
        ),
        "vision_encoder": ("model.vision_tower", "model.multi_modal_projector"),
        "embedding": ("model.language_model.embed_tokens",),
    }
    _IDEFICS_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "decoder": (
            "model.text_model.layers",
            "model.text_model.norm",
            "model.text_model.rotary_emb",
            "lm_head",
        ),
        "vision_encoder": ("model.vision_model", "model.connector"),
        "embedding": ("model.text_model.embed_tokens",),
    }

    @classmethod
    def get_hf_component_sources(
        cls,
        *,
        model_type: str,
        hf_config: object,
    ) -> dict[str, tuple[str, ...]]:
        """Return static runtime paths for verified HF layouts served by this class."""
        del hf_config
        if model_type in {"llava", "llava_next", "llava_onevision", "paligemma", "mistral3"}:
            return cls.HF_COMPONENT_SOURCES
        if model_type in {"idefics2", "idefics3", "smolvlm"}:
            return cls._IDEFICS_COMPONENT_SOURCES
        # This generic mobius class is registered for additional HF families
        # whose runtime trees differ. Empty paths prevent tools from slicing
        # against a guessed checkpoint-style prefix.
        return {}

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _LLaVADecoderModel(config)
        # Dispatch: use Pixtral vision encoder for pixtral-based models,
        # CLIP/SigLIP for everything else.
        self._is_pixtral = (
            config.vision and getattr(config.vision, "model_type", None) == "pixtral"
        )
        if self._is_pixtral:
            self.vision_encoder = _PixtralVisionEncoderModel(config)
        else:
            self.vision_encoder = _LLaVAVisionEncoderModel(config)
        self.embedding = _LLaVAEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "LLaVAModel uses VisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Mistral-3 / Pixtral models prefix decoder weights with
        # ``language_model.`` — strip it so the decoder/embedding
        # sub-models can find their weights.
        if self._is_pixtral:
            return _preprocess_pixtral_weights(state_dict, self.config.tie_word_embeddings)
        # Default LLaVA: only handle weight tying
        if self.config.tie_word_embeddings:
            embed_key = "language_model.model.embed_tokens.weight"
            head_key = "language_model.lm_head.weight"
            if head_key not in state_dict and embed_key in state_dict:
                state_dict[head_key] = state_dict[embed_key]
        return state_dict


def _preprocess_pixtral_weights(
    state_dict: dict[str, torch.Tensor],
    tie_word_embeddings: bool,
) -> dict[str, torch.Tensor]:
    """Remap HF Mistral-3/Pixtral weight names to ONNX sub-model names.

    HF wraps everything under ``model.`` (Mistral3Model) and prefixes
    decoder weights with ``language_model.``.  This function strips
    the ``model.`` prefix, adds ``vision_encoder.`` for vision/projector
    weights, and strips ``language_model.`` for decoder/embedding.
    """
    renamed: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        # Strip outer model. prefix (Mistral3ForConditionalGeneration.model)
        k = key[len("model.") :] if key.startswith("model.") else key

        if k.startswith(("vision_tower.", "multi_modal_projector.")):
            renamed[f"vision_encoder.{k}"] = value
        elif k.startswith("language_model.model.embed_tokens."):
            suffix = k[len("language_model.model.") :]
            renamed[f"decoder.model.{suffix}"] = value
            renamed[f"embedding.{suffix}"] = value
            if tie_word_embeddings:
                renamed["decoder.lm_head.weight"] = value
        elif k.startswith("language_model.lm_head."):
            suffix = k[len("language_model.") :]
            renamed[f"decoder.{suffix}"] = value
        elif k.startswith("language_model."):
            suffix = k[len("language_model.") :]
            renamed[f"decoder.{suffix}"] = value
        elif k == "lm_head.weight" or k.startswith("lm_head."):
            renamed[f"decoder.{k}"] = value
    return renamed
