# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Phi-3-Vision / Phi-3.5-Vision multimodal model (vision + text) — 3-model split.

Builds three ONNX models:

- **decoder**: Phi3 text decoder taking ``inputs_embeds``
- **vision**: CLIP ViT-L/14-336 + MLP projector
- **embedding**: token embedding + image feature fusion

Architecture:
    pixel_values (num_crops, 3, 336, 336)
        → CLIP ViT-L/14-336 → (num_crops, 576, 1024)
        → MLP projector    → (num_crops, 576, 3072)
        → concat + insert into input sequence
        → Phi3 text decoder

The vision encoder uses CLIP ViT-L/14-336 (patch_size=14, image_size=336),
which produces 576 patches per crop. The MLP projector maps from the vision
hidden size (1024) to the text hidden size (3072).

HD transform note: the HuggingFace phi3_v model supports high-definition
tiling (sub_glb ordering with learnable separators). This ONNX implementation
processes a batch of pre-tiled crops uniformly — the HD tiling and separator
insertion is expected to happen in the preprocessing step outside ONNX.

HuggingFace weight name prefixes::

    model.embed_tokens.*
        → decoder.model.embed_tokens.*
        → embedding.embed_tokens.*
    model.layers.N.* (fused qkv_proj, gate_up_proj)
        → decoder.model.layers.N.* (split q/k/v, gate/up)
    model.norm.* / lm_head.*
        → decoder.model.norm.* / decoder.lm_head.*
    model.vision_embed_tokens.img_processor.vision_model.*
        → vision_encoder.vision_tower.vision_model.*
    model.vision_embed_tokens.img_projection.0.*  (fc1)
        → vision_encoder.multi_modal_projector.fc1.*
    model.vision_embed_tokens.img_projection.2.*  (fc2)
        → vision_encoder.multi_modal_projector.fc2.*

Reference: ``microsoft/Phi-3-vision-128k-instruct``,
``microsoft/Phi-3.5-vision-instruct``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import split_fused_qkv, split_gate_up_proj
from mobius.components import (
    Embedding,
    Linear,
    MLPMultiModalProjector,
    VisionModel,
)
from mobius.models.base import TextModel

if TYPE_CHECKING:
    import onnx_ir as ir

# Phi-3-Vision: <|image|> token id
_IMAGE_TOKEN_ID = 32044


class _Phi3VDecoderModel(nn.Module):
    """Phi3-V text decoder taking inputs_embeds.

    Identical to Phi3CausalLMModel but accepts inputs_embeds instead of
    input_ids, allowing the vision features to be fused before decoding.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = TextModel(config)
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
        """Map Phi3-V decoder weights, splitting fused QKV and gate_up."""
        # phi3_v has no `language_model.` prefix — decoder weights live under `model.*`
        prefix = "model."
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                renamed["model." + key[len(prefix) :]] = value
            elif key == "lm_head.weight":
                renamed["lm_head.weight"] = value

        # Split fused QKV
        for key in list(renamed.keys()):
            if "qkv_proj" in key:
                q, k, v = split_fused_qkv(
                    renamed.pop(key),
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
                renamed[key.replace("qkv_proj", "q_proj")] = q
                renamed[key.replace("qkv_proj", "k_proj")] = k
                renamed[key.replace("qkv_proj", "v_proj")] = v
            elif "gate_up_proj" in key:
                gate, up = split_gate_up_proj(
                    renamed.pop(key),
                    self.config.intermediate_size,
                )
                renamed[key.replace("gate_up_proj", "gate_proj")] = gate
                renamed[key.replace("gate_up_proj", "up_proj")] = up

        return renamed


class _Phi3VVisionEncoderModel(nn.Module):
    """Phi3-V vision encoder: CLIP ViT-L/14-336 + MLP projector.

    Accepts a batch of image crops (pre-tiled), applies the CLIP encoder to
    each, and projects from the vision hidden size to the text hidden size.

    Input shape:  (num_crops, 3, 336, 336)  — NCHW
    Output shape: (num_crops * num_patches, text_hidden_size)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.vision_tower = VisionModel(config)
        self.multi_modal_projector = MLPMultiModalProjector(
            vision_hidden_size=config.vision.hidden_size,
            text_hidden_size=config.hidden_size,
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # pixel_values: (num_crops, 3, H, W)
        vision_features = self.vision_tower(op, pixel_values)
        # vision_features: (num_crops, num_patches, vision_hidden)
        # Flatten crops and patches into a single token sequence
        num_crops = op.Shape(pixel_values, start=0, end=1)  # scalar
        num_patches = op.Shape(vision_features, start=1, end=2)
        total = op.Mul(num_crops, num_patches)
        vision_dim = op.Shape(vision_features, start=2, end=3)
        flat_shape = op.Concat(total, vision_dim, axis=0)
        vision_features = op.Reshape(vision_features, flat_shape)
        # vision_features: (num_crops * num_patches, vision_hidden)
        return self.multi_modal_projector(op, vision_features)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Extract vision encoder and projector weights from the full checkpoint.

        HF path → ONNX path::

            model.vision_embed_tokens.img_processor.vision_model.*
                → vision_tower.vision_model.*
            model.vision_embed_tokens.img_projection.0.*
                → multi_modal_projector.fc1.*
            model.vision_embed_tokens.img_projection.2.*
                → multi_modal_projector.fc2.*
        """
        renamed: dict[str, torch.Tensor] = {}

        vision_pfx = "model.vision_embed_tokens.img_processor."
        proj_0_pfx = "model.vision_embed_tokens.img_projection.0."
        proj_2_pfx = "model.vision_embed_tokens.img_projection.2."

        for key, value in state_dict.items():
            if key.startswith(vision_pfx):
                renamed["vision_tower." + key[len(vision_pfx) :]] = value
            elif key.startswith(proj_0_pfx):
                renamed["multi_modal_projector.fc1." + key[len(proj_0_pfx) :]] = value
            elif key.startswith(proj_2_pfx):
                renamed["multi_modal_projector.fc2." + key[len(proj_2_pfx) :]] = value

        return renamed


class _Phi3VEmbeddingModel(nn.Module):
    """Phi3-V embedding: token lookup + image feature fusion.

    Replaces <|image|> token positions with projected vision features
    from the vision encoder.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.image_token_id = (
            config.image_token_id if config.image_token_id is not None else _IMAGE_TOKEN_ID
        )

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value, image_features: ir.Value):
        # (batch, seq_len, hidden_size)
        image_mask = op.Equal(input_ids, op.Constant(value_int=self.image_token_id))
        # Replace image token positions with index 0 before the Gather so that
        # negative or out-of-range image token IDs don't cause undefined behavior.
        # These positions will be overwritten by image features via op.Where below.
        safe_ids = op.Where(image_mask, op.Constant(value_int=0), input_ids)
        text_embeds = self.embed_tokens(op, safe_ids)

        image_mask_3d = op.Unsqueeze(image_mask, [-1])

        mask_int = op.Cast(image_mask, to=7)  # INT64
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        gathered = op.Gather(image_features, indices, axis=0)
        return op.Where(image_mask_3d, gathered, text_embeds)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Extract embed_tokens weights for the embedding sub-model."""
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if "embed_tokens" in key:
                # model.embed_tokens.weight → embed_tokens.weight
                for pfx in ("model.", "language_model.model.", "language_model."):
                    if key.startswith(pfx):
                        renamed[key[len(pfx) :]] = value
                        break
        return renamed


class Phi3VModel(nn.Module):
    """Phi-3-Vision / Phi-3.5-Vision model (3-model split).

    Builds three separate ONNX models:
    - decoder: Phi3 text decoder taking inputs_embeds
    - vision_encoder: CLIP ViT-L/14-336 + MLP projector
    - embedding: token embedding + image feature fusion

    Supports ``microsoft/Phi-3-vision-128k-instruct`` and
    ``microsoft/Phi-3.5-vision-instruct``.
    """

    default_task: str = "vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _Phi3VDecoderModel(config)
        self.vision_encoder = _Phi3VVisionEncoderModel(config)
        self.embedding = _Phi3VEmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "Phi3VModel uses VisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HuggingFace phi3_v weights to ONNX initializer names.

        HF checkpoint layout → ONNX initializer names::

            model.embed_tokens.*
                → decoder.model.embed_tokens.*
                → embedding.embed_tokens.*      (duplicated)
            model.layers.N.*  (fused qkv/gate_up)
                → decoder.model.layers.N.*       (split)
            model.norm.* / lm_head.*
                → decoder.model.norm.* / decoder.lm_head.*
            model.vision_embed_tokens.img_processor.*
                → vision_encoder.vision_tower.*
            model.vision_embed_tokens.img_projection.0.*
                → vision_encoder.multi_modal_projector.fc1.*
            model.vision_embed_tokens.img_projection.2.*
                → vision_encoder.multi_modal_projector.fc2.*
        """
        renamed: dict[str, torch.Tensor] = {}

        vision_pfx = "model.vision_embed_tokens.img_processor."
        proj_0_pfx = "model.vision_embed_tokens.img_projection.0."
        proj_2_pfx = "model.vision_embed_tokens.img_projection.2."

        for key, value in state_dict.items():
            if key.startswith(vision_pfx):
                new_key = "vision_encoder.vision_tower." + key[len(vision_pfx) :]
                renamed[new_key] = value
            elif key.startswith(proj_0_pfx):
                sfx = key[len(proj_0_pfx) :]
                renamed[f"vision_encoder.multi_modal_projector.fc1.{sfx}"] = value
            elif key.startswith(proj_2_pfx):
                sfx = key[len(proj_2_pfx) :]
                renamed[f"vision_encoder.multi_modal_projector.fc2.{sfx}"] = value
            elif key.startswith("model.vision_embed_tokens."):
                # Learnable separators (sub_GN, glb_GN) and other VE state
                # are not used in the ONNX decoder sub-model — skip them.
                pass
            elif key == "model.embed_tokens.weight":
                renamed["decoder.model.embed_tokens.weight"] = value
                renamed["embedding.embed_tokens.weight"] = value
            elif key.startswith("model."):
                renamed["decoder." + key] = value
            elif key == "lm_head.weight":
                renamed["decoder.lm_head.weight"] = value
            else:
                renamed[key] = value

        # Split fused QKV and gate_up in decoder weights
        for key in list(renamed.keys()):
            if not key.startswith("decoder."):
                continue
            if "qkv_proj" in key:
                q, k, v = split_fused_qkv(
                    renamed.pop(key),
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
                renamed[key.replace("qkv_proj", "q_proj")] = q
                renamed[key.replace("qkv_proj", "k_proj")] = k
                renamed[key.replace("qkv_proj", "v_proj")] = v
            elif "gate_up_proj" in key:
                gate, up = split_gate_up_proj(
                    renamed.pop(key),
                    self.config.intermediate_size,
                )
                renamed[key.replace("gate_up_proj", "gate_proj")] = gate
                renamed[key.replace("gate_up_proj", "up_proj")] = up

        return renamed
