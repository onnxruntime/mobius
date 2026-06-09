# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ViT (Vision Transformer) model for image feature extraction."""

from __future__ import annotations

import re

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    FCMLP,
    EncoderAttention,
    LayerNorm,
)
from mobius.components import (
    Conv2d as _Conv2d,
)


class ViTModel(nn.Module):
    """Vision Transformer for image feature extraction.

    Pre-norm encoder with patch embeddings, CLS token, and learned
    position embeddings. Output is the last hidden state including the
    CLS token at position 0.
    """

    default_task = "image-classification"
    category = "vision"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        image_size = getattr(config, "image_size", 224)
        patch_size = getattr(config, "patch_size", 16)
        num_channels = getattr(config, "num_channels", 3)
        num_patches = (image_size // patch_size) ** 2

        self.embeddings = _ViTEmbeddings(config, num_patches, patch_size, num_channels)
        self.encoder = _ViTEncoder(config)
        self.layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        hidden_states = self.embeddings(op, pixel_values)
        hidden_states = self.encoder(op, hidden_states)
        hidden_states = self.layernorm(op, hidden_states)
        return hidden_states

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_vit_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


class _ViTEmbeddings(nn.Module):
    """ViT embeddings: Conv2d patch embed + CLS token + position embeddings."""

    def __init__(self, config, num_patches, patch_size, num_channels):
        super().__init__()
        self.patch_embeddings = _Conv2dPatchEmbed(num_channels, config.hidden_size, patch_size)
        # CLS token and position embeddings as parameters with pre-computed data
        self.cls_token = nn.Parameter(
            [1, 1, config.hidden_size],
            data=ir.tensor(np.zeros((1, 1, config.hidden_size), dtype=np.float32)),
        )
        self.position_embeddings = nn.Parameter(
            [1, num_patches + 1, config.hidden_size],
            data=ir.tensor(
                np.zeros((1, num_patches + 1, config.hidden_size), dtype=np.float32)
            ),
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        patch_embeds = self.patch_embeddings(op, pixel_values)
        batch_size = op.Shape(patch_embeds, start=0, end=1)

        # Expand CLS token to batch size
        cls_tokens = op.Expand(
            self.cls_token,
            op.Concat(batch_size, [1], [1], axis=0),
        )
        # Prepend CLS token to patch embeddings
        hidden_states = op.Concat(cls_tokens, patch_embeds, axis=1)
        # Add position embeddings
        hidden_states = op.Add(hidden_states, self.position_embeddings)
        return hidden_states


class _Conv2dPatchEmbed(nn.Module):
    """Conv2d-based patch embedding."""

    def __init__(self, in_channels, hidden_size, patch_size):
        super().__init__()
        self.projection = _Conv2d(in_channels, hidden_size, patch_size, patch_size)

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        # Conv2d: [batch, channels, H, W] -> [batch, hidden, H/patch, W/patch]
        x = self.projection(op, pixel_values)
        # Flatten spatial dims and transpose: [batch, hidden, num_patches] -> [batch, num_patches, hidden]
        batch = op.Shape(x, start=0, end=1)
        hidden = op.Shape(x, start=1, end=2)
        x = op.Reshape(x, op.Concat(batch, hidden, [-1], axis=0))
        x = op.Transpose(x, perm=[0, 2, 1])
        return x


class _ViTEncoder(nn.Module):
    """ViT encoder: pre-norm encoder layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [_ViTEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        for layer in self.layer:
            hidden_states = layer(op, hidden_states)
        return hidden_states


class _ViTEncoderLayer(nn.Module):
    """ViT pre-norm encoder layer: norm → attn → residual → norm → mlp → residual."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.layernorm_before = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = EncoderAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            bias=True,
        )
        self.layernorm_after = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act,
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        residual = hidden_states
        hidden_states = self.layernorm_before(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.layernorm_after(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states


# Weight name mapping
_VIT_LAYER_RENAMES = {
    "attention.attention.query": "self_attn.q_proj",
    "attention.attention.key": "self_attn.k_proj",
    "attention.attention.value": "self_attn.v_proj",
    "attention.output.dense": "self_attn.out_proj",
    "intermediate.dense": "mlp.up_proj",
    "output.dense": "mlp.down_proj",
}

_LAYER_PATTERN = re.compile(r"^encoder\.layer\.(\d+)\.(.+)$")

# transformers >=5.x flattened the ViT state dict: encoder layers are
# ``layers.N.<sub>`` (no ``encoder.`` prefix) with consolidated attention
# (``attention.{q,k,v,o}_proj``) and MLP (``mlp.fc1``/``mlp.fc2``) names.
_LAYER_PATTERN_NEW = re.compile(r"^layers\.(\d+)\.(.+)$")
_VIT_NEW_LAYER_RENAMES = {
    "attention.q_proj": "self_attn.q_proj",
    "attention.k_proj": "self_attn.k_proj",
    "attention.v_proj": "self_attn.v_proj",
    "attention.o_proj": "self_attn.out_proj",
    "mlp.fc1": "mlp.up_proj",
    "mlp.fc2": "mlp.down_proj",
}


def _rename_vit_weight(name: str) -> str | None:
    """Rename HF ViT weight to our naming convention."""
    # Strip model-type prefix (e.g. vit., beit., deit., dinov2., swin., hiera.)
    # HF safetensors use the model class prefix before embeddings/encoder.
    first_dot = name.find(".")
    if first_dot > 0:
        after = name[first_dot + 1 :]
        if after.startswith(
            ("embeddings.", "encoder.", "layernorm.", "pooler.", "classifier.")
        ):
            name = after

    # Skip most pooler/classifier weights, but keep pooler.layernorm
    # (BeiT uses pooler.layernorm as the final layernorm)
    if name.startswith("pooler.layernorm."):
        suffix = name[len("pooler.layernorm.") :]
        return f"layernorm.{suffix}"
    if name.startswith(("pooler.", "classifier.")):
        return None

    # Embeddings
    if name == "embeddings.cls_token":
        return "embeddings.cls_token"
    if name == "embeddings.position_embeddings":
        return "embeddings.position_embeddings"
    # DINOv2 mask_token: used during pre-training only; not needed for inference
    if name == "embeddings.mask_token":
        return None
    if name.startswith("embeddings.patch_embeddings.projection."):
        suffix = name[len("embeddings.patch_embeddings.projection.") :]
        return f"embeddings.patch_embeddings.projection.{suffix}"

    # Final layernorm
    if name.startswith("layernorm."):
        return name  # Already correct

    # transformers >=5.x flattened encoder layers: ``layers.N.<sub>``.
    m_new = _LAYER_PATTERN_NEW.match(name)
    if m_new:
        layer_idx, suffix = m_new.group(1), m_new.group(2)
        if suffix.startswith("layernorm_"):
            return f"encoder.layer.{layer_idx}.{suffix}"
        for old, new in _VIT_NEW_LAYER_RENAMES.items():
            if suffix.startswith(old):
                remainder = suffix[len(old) :]
                return f"encoder.layer.{layer_idx}.{new}{remainder}"
        return None

    # Encoder layers
    m = _LAYER_PATTERN.match(name)
    if m:
        layer_idx, suffix = m.group(1), m.group(2)
        # layernorm_before / layernorm_after pass through
        if suffix.startswith("layernorm_"):
            return f"encoder.layer.{layer_idx}.{suffix}"
        # DINOv2/DeiT-style norm1/norm2 → layernorm_before/layernorm_after
        if suffix.startswith("norm1"):
            remainder = suffix[len("norm1") :]
            return f"encoder.layer.{layer_idx}.layernorm_before{remainder}"
        if suffix.startswith("norm2"):
            remainder = suffix[len("norm2") :]
            return f"encoder.layer.{layer_idx}.layernorm_after{remainder}"
        # DINOv2-style MLP: fc1 → up_proj, fc2 → down_proj
        if suffix.startswith("mlp.fc1"):
            remainder = suffix[len("mlp.fc1") :]
            return f"encoder.layer.{layer_idx}.mlp.up_proj{remainder}"
        if suffix.startswith("mlp.fc2"):
            remainder = suffix[len("mlp.fc2") :]
            return f"encoder.layer.{layer_idx}.mlp.down_proj{remainder}"
        # DINOv2 layer_scale: layer_scale1/layer_scale2 (skip — not
        # used in our ViT implementation)
        if suffix.startswith("layer_scale"):
            return None
        # BeiT lambda_1/lambda_2 (layer scale — not implemented)
        if suffix in ("lambda_1", "lambda_2"):
            return None
        # BeiT relative_position_bias (not implemented in base ViT)
        if "relative_position_bias" in suffix:
            return None
        for old, new in _VIT_LAYER_RENAMES.items():
            if suffix.startswith(old):
                remainder = suffix[len(old) :]
                return f"encoder.layer.{layer_idx}.{new}{remainder}"

    return None
