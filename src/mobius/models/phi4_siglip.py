# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Phi-4-Reasoning-Vision (phi4-siglip) multimodal model — 3-model split.

Builds three ONNX models:

- **decoder**: Phi4 text decoder taking ``inputs_embeds``
- **vision**: SigLIP-2 vision tower + MLP projector
- **embedding**: token embedding + image feature fusion

Architecture:
    pixel_values (1, 3, H, W)  [NaFlex or fixed-resolution image]
        → SigLIP-2 (ViT, patch_size=16)  → (num_patches, 1152)
        → mlp2x_gelu projector           → (num_patches, 5120)
        → fused into input sequence at image-token positions
        → Phi4 text decoder (40-layer, 5120-dim, GQA 40/10)

Vision config (from HF ``vision_config`` sub-dict):
    model_type: siglip2_vision_model
    hidden_size: 1152, intermediate_size: 4304
    num_hidden_layers: 27, num_attention_heads: 16
    image_size: 384 (default), patch_size: 16

Text config (from ``microsoft/Phi-4-reasoning-vision-15B``):
    hidden_size: 5120, intermediate_size: 17920
    num_hidden_layers: 40, num_attention_heads: 40 / num_key_value_heads: 10

Projector: mlp2x_gelu — 2-layer MLP (1152 → 5120) with GELU activation.

HuggingFace weight name prefixes::

    model.embed_tokens.*
        → decoder.model.embed_tokens.*
        → embedding.embed_tokens.*
    model.layers.N.* (fused qkv_proj, gate_up_proj)
        → decoder.model.layers.N.* (split q/k/v, gate/up)
    model.norm.* / lm_head.*
        → decoder.model.norm.* / decoder.lm_head.*
    model.vision_tower.* (SigLIP-2 vision model)
        → vision_encoder.vision_tower.*
    model.mm_projector.0.*  (mlp2x_gelu layer 1)
        → vision_encoder.multi_modal_projector.fc1.*
    model.mm_projector.2.*  (mlp2x_gelu layer 2)
        → vision_encoder.multi_modal_projector.fc2.*

Reference: ``microsoft/Phi-4-reasoning-vision-15B``.
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

# Phi-4-reasoning-vision: image placeholder token index.
# phi4-siglip uses -200 following the LLaVA IMAGE_TOKEN_INDEX convention.
_IMAGE_TOKEN_ID = -200


class _Phi4SigLIPDecoderModel(nn.Module):
    """Phi4 text decoder taking inputs_embeds.

    The Phi4-reasoning-vision text decoder is architecturally identical to
    Phi3 but with larger width (5120-dim, 40 layers, 40/10 GQA). It accepts
    pre-fused inputs_embeds instead of raw input_ids.
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
        """Map Phi4 decoder weights, splitting fused QKV and gate_up."""
        # Decoder weights sit under `model.*` (no `language_model.` prefix)
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if (
                key.startswith("model.")
                and not key.startswith("model.vision_tower.")
                and not key.startswith("model.mm_projector.")
            ):
                renamed[key] = value
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


class _Phi4SigLIPVisionEncoderModel(nn.Module):
    """Phi4-SigLIP vision encoder: SigLIP-2 + mlp2x_gelu projector.

    Input shape:  (batch, 3, H, W) — NaFlex or fixed resolution
    Output shape: (batch * num_patches, text_hidden_size)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.vision_tower = VisionModel(config)
        self.multi_modal_projector = MLPMultiModalProjector(
            vision_hidden_size=config.vision.hidden_size,
            text_hidden_size=config.hidden_size,
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # pixel_values: (batch, 3, H, W)
        vision_features = self.vision_tower(op, pixel_values)
        # vision_features: (batch, num_patches, vision_hidden)
        # Flatten images and patches into a single token sequence.
        batch_size = op.Shape(pixel_values, start=0, end=1)
        num_patches = op.Shape(vision_features, start=1, end=2)
        total = op.Mul(batch_size, num_patches)
        vision_dim = op.Shape(vision_features, start=2, end=3)
        flat_shape = op.Concat(total, vision_dim, axis=0)
        vision_features = op.Reshape(vision_features, flat_shape)
        return self.multi_modal_projector(op, vision_features)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Extract vision tower and projector weights.

        HF path → ONNX path::

            model.vision_tower.*
                → vision_tower.*
            model.mm_projector.0.*
                → multi_modal_projector.fc1.*
            model.mm_projector.2.*
                → multi_modal_projector.fc2.*
        """
        renamed: dict[str, torch.Tensor] = {}
        vt_pfx = "model.vision_tower."
        mm_0_pfx = "model.mm_projector.0."
        mm_2_pfx = "model.mm_projector.2."

        for key, value in state_dict.items():
            if key.startswith(vt_pfx):
                renamed["vision_tower." + key[len(vt_pfx) :]] = value
            elif key.startswith(mm_0_pfx):
                renamed["multi_modal_projector.fc1." + key[len(mm_0_pfx) :]] = value
            elif key.startswith(mm_2_pfx):
                renamed["multi_modal_projector.fc2." + key[len(mm_2_pfx) :]] = value

        return renamed


class _Phi4SigLIPEmbeddingModel(nn.Module):
    """Phi4-SigLIP embedding: token lookup + image feature fusion.

    Replaces image placeholder token positions with projected SigLIP-2 features.
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
        image_mask = op.Equal(input_ids, op.Constant(value_int=self.image_token_id))
        # Replace image token positions with index 0 before the Gather so that
        # negative or out-of-range image token IDs (e.g. -200) don't cause
        # undefined behavior in ONNX Gather. These positions are overwritten
        # by image features via op.Where below.
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
                for pfx in ("model.", "language_model.model.", "language_model."):
                    if key.startswith(pfx):
                        renamed[key[len(pfx) :]] = value
                        break
        return renamed


class Phi4SigLIPModel(nn.Module):
    """Phi-4-Reasoning-Vision (phi4-siglip) model (3-model split).

    Builds three separate ONNX models:
    - decoder: Phi4 text decoder taking inputs_embeds
    - vision_encoder: SigLIP-2 + mlp2x_gelu projector
    - embedding: token embedding + image feature fusion

    Supports ``microsoft/Phi-4-reasoning-vision-15B``.
    """

    default_task: str = "vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _Phi4SigLIPDecoderModel(config)
        self.vision_encoder = _Phi4SigLIPVisionEncoderModel(config)
        self.embedding = _Phi4SigLIPEmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "Phi4SigLIPModel uses VisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HuggingFace phi4-siglip weights to ONNX initializer names.

        HF checkpoint layout → ONNX initializer names::

            model.embed_tokens.*
                → decoder.model.embed_tokens.*
                → embedding.embed_tokens.*      (duplicated)
            model.layers.N.*  (fused qkv/gate_up)
                → decoder.model.layers.N.*       (split)
            model.norm.* / lm_head.*
                → decoder.model.norm.* / decoder.lm_head.*
            model.vision_tower.*
                → vision_encoder.vision_tower.*
            model.mm_projector.0.*
                → vision_encoder.multi_modal_projector.fc1.*
            model.mm_projector.2.*
                → vision_encoder.multi_modal_projector.fc2.*
        """
        renamed: dict[str, torch.Tensor] = {}
        vt_pfx = "model.vision_tower."
        mm_0_pfx = "model.mm_projector.0."
        mm_2_pfx = "model.mm_projector.2."

        for key, value in state_dict.items():
            if key.startswith(vt_pfx):
                renamed["vision_encoder.vision_tower." + key[len(vt_pfx) :]] = value
            elif key.startswith(mm_0_pfx):
                sfx = key[len(mm_0_pfx) :]
                renamed[f"vision_encoder.multi_modal_projector.fc1.{sfx}"] = value
            elif key.startswith(mm_2_pfx):
                sfx = key[len(mm_2_pfx) :]
                renamed[f"vision_encoder.multi_modal_projector.fc2.{sfx}"] = value
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
