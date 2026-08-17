# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GLM-OCR vision-language model.

Replicates HuggingFace ``GlmOcrForConditionalGeneration`` as standardized
``decoder`` / ``vision_encoder`` / ``embedding`` ONNX components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import Embedding, GlmOcrVisionModel, Linear
from mobius.models.glm import Glm4TextModel

if TYPE_CHECKING:
    import onnx_ir as ir


class GlmOcrDecoderModel(nn.Module):
    """GLM-OCR four-norm fused-MLP decoder taking precomputed embeddings."""

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.model = Glm4TextModel(config)
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
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        return self.lm_head(op, hidden_states), present_key_values


class GlmOcrVisionEncoderModel(nn.Module):
    """GLM-OCR packed vision tower and learned spatial merger."""

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        vision = config.vision
        assert vision is not None
        assert vision.hidden_size is not None
        assert vision.intermediate_size is not None
        assert vision.num_hidden_layers is not None
        assert vision.num_attention_heads is not None
        assert vision.patch_size is not None
        assert vision.out_hidden_size is not None
        self.visual = GlmOcrVisionModel(
            depth=vision.num_hidden_layers,
            hidden_size=vision.hidden_size,
            intermediate_size=vision.intermediate_size,
            num_heads=vision.num_attention_heads,
            patch_size=vision.patch_size,
            temporal_patch_size=vision.temporal_patch_size,
            in_channels=vision.in_channels,
            out_hidden_size=vision.out_hidden_size,
            spatial_merge_size=vision.spatial_merge_size,
            norm_eps=vision.norm_eps,
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ) -> ir.Value:
        return self.visual(op, pixel_values, image_grid_thw)


class GlmOcrEmbeddingModel(nn.Module):
    """Inject flattened vision features at GLM-OCR image placeholder tokens."""

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self._image_token_id = (
            config.image_token_id if config.image_token_id is not None else 59280
        )

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
    ) -> ir.Value:
        text_embeddings = self.embed_tokens(op, input_ids)  # [B, S, H]
        image_mask = op.Equal(input_ids, op.Constant(value_int=self._image_token_id))
        flat_mask = op.Reshape(image_mask, [-1])
        flat_indices = op.Sub(
            op.CumSum(op.Cast(flat_mask, to=7), op.Constant(value_int=0)),
            op.Constant(value_int=1),
        )
        flat_indices = op.Clip(flat_indices, op.Constant(value_int=0))

        # Keep text-only calls valid when image_features has zero rows.
        zero_row = op.Expand(
            op.CastLike(0.0, image_features),
            op.Concat(
                op.Constant(value_ints=[1]),
                op.Shape(image_features, start=1, end=2),
                axis=0,
            ),
        )
        padded_features = op.Concat(image_features, zero_row, axis=0)
        gathered = op.Gather(padded_features, flat_indices, axis=0)
        gathered = op.Reshape(gathered, op.Shape(text_embeddings))
        return op.Where(op.Unsqueeze(image_mask, [-1]), gathered, text_embeddings)


class GlmOcrForConditionalGeneration(nn.Module):
    """GLM-OCR document vision-language model with M-RoPE and cached decoding."""

    default_task: str = "glm-ocr"
    category: str = "Multimodal"
    config_class: type = ArchitectureConfig

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self._num_hidden_layers = config.num_hidden_layers
        self.decoder = GlmOcrDecoderModel(config)
        self.vision_encoder = GlmOcrVisionEncoderModel(config)
        self.embedding = GlmOcrEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "GLM-OCR is exported as decoder, vision_encoder, and embedding models."
        )

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Route pinned HuggingFace checkpoint tensors to the three sub-models."""
        if any(
            key.startswith(("decoder.", "vision_encoder.", "embedding."))
            for key in state_dict
        ):
            return dict(state_dict)

        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("model.visual."):
                stripped = key.removeprefix("model.")
                renamed[f"vision_encoder.{stripped}"] = value
            elif key == "model.language_model.embed_tokens.weight":
                renamed["embedding.embed_tokens.weight"] = value
            elif key.startswith("model.language_model."):
                stripped = key.removeprefix("model.language_model.")
                if stripped.startswith("layers."):
                    layer = stripped.split(".", 2)[1]
                    if layer.isdigit() and int(layer) >= self._num_hidden_layers:
                        # Auxiliary next-token predictor layers are trained with
                        # the checkpoint but are not used by the base forward pass.
                        continue
                renamed[f"decoder.model.{stripped}"] = value
            elif key == "lm_head.weight":
                renamed["decoder.lm_head.weight"] = value
        return renamed
