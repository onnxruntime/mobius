# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""NemotronH_Nano_VL_V2 multimodal model — RADIO vision + NemotronH text decoder.

Splits the NemotronH_Nano_VL_V2 architecture into three ONNX models for
onnxruntime-genai:

- **decoder**: NemotronH hybrid (Mamba2 + Attention + MLP) text decoder taking
  ``inputs_embeds``
- **vision**: RADIO ViT-H/16 encoder + pixel shuffle + MLP projector
- **embedding**: token embedding + image feature fusion

Architecture notes:
- Identical RADIO vision encoder to ``Llama_Nemotron_Nano_VL`` (RADIO ViT-H/16 CPE)
- NemotronH text decoder (hybrid Mamba2 + Attention + MLP layers)
- Larger MLP projector: 5120 → 20480 (projector_hidden_size) → 5120 (llm_hidden)
- ``ps_version='v2'`` pixel shuffle

HuggingFace reference: ``nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16``
(model_type ``NemotronH_Nano_VL_V2``).

HuggingFace weight names:
- ``vision_model.radio_model.model.patch_generator.*``
- ``vision_model.radio_model.model.blocks.N.*``
- ``vision_model.radio_model.input_conditioner.*`` (skipped)
- ``mlp1.{0,1,3}.*``
- ``language_model.backbone.*`` / ``language_model.lm_head.*``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import Linear
from mobius.components._vision import VisionLayerNorm
from mobius.models.internvl import _GELUPlaceholder, _InternVL2EmbeddingModel
from mobius.models.llama_nemotron_nano_vl import _LlamaNemotronNanoVLVisionEncoderModel
from mobius.models.nemotron_h import _NemotronHTextModel, _rename_nemotron_h_weight

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# NemotronH text decoder for VL — takes inputs_embeds
# ---------------------------------------------------------------------------


class _NemotronHVLDecoderModel(nn.Module):
    """NemotronH text decoder for VL use — takes ``inputs_embeds`` instead of ``input_ids``.

    Wraps ``_NemotronHTextModel`` and exposes the VL-compatible interface (same
    as ``_InternVL2DecoderModel``) so the ``VisionLanguageTask`` can drive it.

    HF weight prefix: ``language_model.backbone.*`` / ``language_model.lm_head.*``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = _NemotronHTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,  # type: ignore[arg-type]
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values


# ---------------------------------------------------------------------------
# Three-model split
# ---------------------------------------------------------------------------


class NemotronHNanoVLModel(nn.Module):
    """NemotronH_Nano_VL_V2 vision-language model (3-model split).

    Builds three separate ONNX models:
    - decoder: NemotronH hybrid text decoder taking inputs_embeds
    - vision: RADIO ViT-H/16 + pixel shuffle + MLP projector
    - embedding: token embedding + image feature fusion

    HF reference: ``nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16``
    (model_type ``NemotronH_Nano_VL_V2``).
    """

    default_task: str = "hybrid-vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _NemotronHVLDecoderModel(config)
        self.vision_encoder = _LlamaNemotronNanoVLVisionEncoderModel(config)
        self.embedding = _InternVL2EmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "NemotronHNanoVLModel uses HybridVisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route HF weights to the correct ONNX sub-model initializer names.

        HF prefixes → ONNX prefixes:
        - ``vision_model.radio_model.model.patch_generator.*``
            → ``vision_encoder.vision_model.*`` (window-select pos_embed)
        - ``vision_model.radio_model.model.blocks.N.*``
            → ``vision_encoder.vision_model.blocks.N.*``
        - ``vision_model.radio_model.input_conditioner.*`` → SKIP
        - ``mlp1.*`` → ``vision_encoder.mlp1.*``
        - ``language_model.*`` → apply NemotronH weight renaming → ``decoder.*``
        - ``language_model.backbone.embeddings.weight``
            → ``embedding.embed_tokens.weight`` (dual copy)
        """
        renamed: dict[str, torch.Tensor] = {}

        vc = self.config.vision
        assert vc is not None
        image_size = vc.image_size
        patch_size = vc.patch_size
        h_in = w_in = image_size // patch_size

        pg = "vision_model.radio_model.model.patch_generator."
        blocks_prefix = "vision_model.radio_model.model.blocks."
        ic = "vision_model.radio_model.input_conditioner."

        layer_types = self.config.layer_types or []

        for key, value in state_dict.items():
            if key.startswith(ic):
                continue  # skip image normalization stats
            elif key == pg + "embedder.weight":
                renamed["vision_encoder.vision_model.patch_embed.weight"] = value
            elif key == pg + "cls_token.token":
                renamed["vision_encoder.vision_model.cls_token"] = value
            elif key == pg + "pos_embed":
                # Window-select from max-resolution table to inference resolution
                n, max_patches, c = value.shape
                h_max = w_max = int(max_patches**0.5)
                if h_max == h_in:
                    renamed["vision_encoder.vision_model.pos_embed"] = value
                else:
                    grid = value.reshape(n, h_max, w_max, c)
                    sliced = grid[:, :h_in, :w_in, :].reshape(n, h_in * w_in, c)
                    renamed["vision_encoder.vision_model.pos_embed"] = sliced.contiguous()
            elif key.startswith(blocks_prefix):
                suffix = key[len("vision_model.radio_model.model."):]
                renamed[f"vision_encoder.vision_model.{suffix}"] = value
            elif key.startswith("mlp1."):
                renamed[f"vision_encoder.{key}"] = value
            elif key.startswith("language_model."):
                # Strip "language_model." prefix, then apply NemotronH weight renaming
                raw = key[len("language_model."):]
                # NemotronH renaming: backbone.* → model.*, etc.
                onnx_key = _rename_nemotron_h_weight(raw, layer_types)
                renamed[f"decoder.{onnx_key}"] = value
                # Embedding model also needs embed_tokens
                if raw == "backbone.embeddings.weight":
                    renamed["embedding.embed_tokens.weight"] = value

        # Weight tying: copy embed_tokens → lm_head when tie_word_embeddings=True
        if self.config.tie_word_embeddings:
            embed_key = "embedding.embed_tokens.weight"
            head_key = "decoder.lm_head.weight"
            if head_key not in renamed and embed_key in renamed:
                renamed[head_key] = renamed[embed_key]

        return renamed
