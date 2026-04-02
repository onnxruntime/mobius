# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""DPT (Dense Prediction Transformer) for monocular depth estimation.

DPT embeds a ViT backbone directly in the model (no separate backbone_config)
and adds:
  1. A readout ("project") stage: the CLS token is concatenated with patch
     tokens and projected via Linear(2H→H) + GELU before reassembly.
  2. A reassemble neck: per-scale channel projection + spatial resize.
  3. A feature fusion neck: coarse-to-fine merge with pre-act residual blocks.
  4. A 3-conv depth head producing a (B, H, W) disparity map.

Reference: https://huggingface.co/Intel/dpt-large
Paper: "Vision Transformers for Dense Prediction" (Ranftl et al., 2021)
"""

from __future__ import annotations

import re

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import DPTConfig
from mobius.components import (
    FCMLP,
    Conv2d,
    Conv2dNoBias,
    EncoderAttention,
    LayerNorm,
    Linear,
)
from mobius.models.depth_anything import (
    _Conv2dPatchEmbed,
    _FeatureFusionLayer,
    _ReassembleLayer,
)

# ---------------------------------------------------------------------------
# ViT Backbone (no intermediate layernorm — DPT passes raw encoder outputs)
# ---------------------------------------------------------------------------


class _DPTViTEncoderLayer(nn.Module):
    """Pre-norm ViT encoder layer."""

    def __init__(self, config: DPTConfig):
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

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        residual = hidden_states
        hidden_states = self.layernorm_before(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.layernorm_after(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class _DPTViTBackbone(nn.Module):
    """ViT backbone extracting raw (un-normed) hidden states at target layers.

    Unlike DepthAnything, DPT passes the raw encoder layer output to the neck
    without per-feature layernorm.  The final ``layernorm`` weight is stored
    for weight-loading completeness but is not used in forward().
    """

    def __init__(self, config: DPTConfig):
        super().__init__()
        image_size = config.image_size
        patch_size = config.patch_size
        num_channels = config.num_channels
        hidden_size = config.hidden_size
        num_patches = (image_size // patch_size) ** 2

        self.patch_embeddings = _Conv2dPatchEmbed(num_channels, hidden_size, patch_size)
        self.cls_token = nn.Parameter(
            [1, 1, hidden_size],
            data=ir.tensor(np.zeros((1, 1, hidden_size), dtype=np.float32)),
        )
        self.position_embeddings = nn.Parameter(
            [1, num_patches + 1, hidden_size],
            data=ir.tensor(np.zeros((1, num_patches + 1, hidden_size), dtype=np.float32)),
        )
        self.encoder = nn.ModuleList(
            [_DPTViTEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # Kept for weight-loading — not applied to intermediate features in forward()
        self.layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 0-indexed out_indices: 5 means after encoder layer index 5
        self.out_indices: list[int] = config.backbone_out_indices or []

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # Patch embedding: (B, C, H, W) → (B, S, hidden_size)
        patch_embeds = self.patch_embeddings(op, pixel_values)
        batch_size = op.Shape(patch_embeds, start=0, end=1)

        # Prepend CLS token and add positional embedding
        cls_tokens = op.Expand(
            self.cls_token,
            op.Concat(batch_size, op.Constant(value_ints=[1, 1]), axis=0),
        )
        # (B, S+1, hidden_size)
        hidden_states = op.Concat(cls_tokens, patch_embeds, axis=1)
        hidden_states = op.Add(hidden_states, self.position_embeddings)

        # Run encoder, collecting hidden states at target layers
        # out_indices are 0-indexed (backbone_out_indices=[5,11,17,23] for dpt-large)
        feature_maps = []
        for i, layer in enumerate(self.encoder):
            hidden_states = layer(op, hidden_states)
            if i in self.out_indices:
                feature_maps.append(hidden_states)

        return feature_maps  # list of (B, S+1, hidden_size) with CLS token


# ---------------------------------------------------------------------------
# DPT Neck: readout projection + reassemble + fusion
# ---------------------------------------------------------------------------


class _DPTReadoutProjection(nn.Module):
    """Project CLS token concat with patch tokens: Linear(2H→H) + activation."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = Linear(hidden_size * 2, hidden_size, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        patch_tokens: ir.Value,  # (B, S, H) — spatial tokens, no CLS
        cls_token: ir.Value,  # (B, H) — CLS token
    ) -> ir.Value:
        # Expand CLS to (B, S, H) and concatenate: (B, S, 2H)
        cls_expanded = op.Unsqueeze(cls_token, [1])
        s = op.Shape(patch_tokens, start=1, end=2)
        cls_expanded = op.Expand(
            cls_expanded,
            op.Concat(
                op.Shape(patch_tokens, start=0, end=1),
                s,
                op.Shape(patch_tokens, start=2, end=3),
                axis=0,
            ),
        )
        concat = op.Concat(patch_tokens, cls_expanded, axis=-1)  # (B, S, 2H)
        return op.Gelu(self.projection(op, concat))  # (B, S, H)


class _DPTNeck(nn.Module):
    """DPT neck: readout projection + reassemble + channel alignment + fusion."""

    def __init__(self, config: DPTConfig):
        super().__init__()
        neck_sizes = config.neck_hidden_sizes or [256, 512, 1024, 1024]
        factors = config.reassemble_factors or [4.0, 2.0, 1.0, 0.5]
        fusion_size = config.fusion_hidden_size
        hidden_size = config.hidden_size

        # Readout projection: one Linear(2H→H) per feature scale
        self.readout_projections = nn.ModuleList(
            [_DPTReadoutProjection(hidden_size) for _ in neck_sizes]
        )
        self.reassemble_layers = nn.ModuleList(
            [_ReassembleLayer(hidden_size, ch, f) for ch, f in zip(neck_sizes, factors)]
        )
        # 3×3 convs mapping reassemble output channels → fusion_size
        self.convs = nn.ModuleList(
            [Conv2dNoBias(ch, fusion_size, kernel_size=3, padding=1) for ch in neck_sizes]
        )
        self.fusion_layers = nn.ModuleList(
            [_FeatureFusionLayer(fusion_size) for _ in neck_sizes]
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: list[ir.Value],  # each (B, S+1, H) — includes CLS at [:, 0, :]
        patch_height: ir.Value,  # (1,) int
        patch_width: ir.Value,  # (1,) int
    ) -> list[ir.Value]:
        reassembled = []
        for i, hs in enumerate(hidden_states):
            # Split CLS token from patch tokens
            # cls_token: (B, H), patch_tokens: (B, S, H)
            cls_token = op.Squeeze(
                op.Slice(
                    hs,
                    op.Constant(value_ints=[0]),
                    op.Constant(value_ints=[1]),
                    op.Constant(value_ints=[1]),
                ),
                [1],
            )
            patch_tokens = op.Slice(
                hs,
                op.Constant(value_ints=[1]),
                op.Constant(value_ints=[2**31 - 1]),
                op.Constant(value_ints=[1]),
            )  # (B, S, H)

            # Apply readout projection (CLS token fusion)
            patch_tokens = self.readout_projections[i](op, patch_tokens, cls_token)

            # Reshape to spatial grid: (B, H, pH, pW)
            batch = op.Shape(patch_tokens, start=0, end=1)
            channels = op.Shape(patch_tokens, start=2, end=3)
            hs_2d = op.Transpose(patch_tokens, perm=[0, 2, 1])  # (B, H, S)
            hs_2d = op.Reshape(
                hs_2d,
                op.Concat(batch, channels, patch_height, patch_width, axis=0),
            )  # (B, H, pH, pW)

            # Per-scale projection and resize
            hs_2d = self.reassemble_layers[i](op, hs_2d)
            # Channel alignment: neck_hidden_size → fusion_size
            hs_2d = self.convs[i](op, hs_2d)
            reassembled.append(hs_2d)

        # Coarse-to-fine fusion (process from last/coarsest to first/finest)
        reassembled.reverse()
        fused = None
        fused_list = []
        for feature, layer in zip(reassembled, self.fusion_layers):
            if fused is None:
                fused = layer(op, feature)
            else:
                fused = layer(op, fused, feature)
            fused_list.append(fused)

        return fused_list


# ---------------------------------------------------------------------------
# Depth estimation head
# ---------------------------------------------------------------------------


class _DPTDepthHead(nn.Module):
    """DPT depth head: Conv(f→f/2) + Upsample(2x) + Conv(f/2→32) + ReLU + Conv(32→1) + ReLU.

    Weight names use Sequential indices (head.0, head.2, head.4) to match HF.
    """

    def __init__(self, config: DPTConfig):
        super().__init__()
        features = config.fusion_hidden_size  # 256

        # Named to match HF: head.head.0.*, head.head.2.*, head.head.4.*
        # preprocess_weights remaps conv1→0, conv2→2, conv3→4
        self.head = _DPTHeadConvs(features)

    def forward(
        self,
        op: builder.OpBuilder,
        fused_list: list[ir.Value],
    ) -> ir.Value:
        # Use the last (finest) fused feature map
        x = fused_list[-1]  # (B, fusion_size, H, W)
        x = self.head(op, x)
        # Squeeze channel dim: (B, 1, H, W) → (B, H, W)
        return op.Squeeze(x, [1])


class _DPTHeadConvs(nn.Module):
    """Three-conv sequential head matching HF weight names head.head.{0,2,4}.*."""

    def __init__(self, features: int):
        super().__init__()
        # Named conv1/conv2/conv3; preprocess_weights maps 0→conv1, 2→conv2, 4→conv3
        self.conv1 = Conv2d(features, features // 2, kernel_size=3, padding=1)
        self.conv2 = Conv2d(features // 2, 32, kernel_size=3, padding=1)
        self.conv3 = Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        # Conv → Upsample 2x → Conv → ReLU → Conv → ReLU
        x = self.conv1(op, x)
        x = op.Resize(
            x,
            None,
            op.Constant(value_floats=[1.0, 1.0, 2.0, 2.0]),
            mode="linear",
        )
        x = self.conv2(op, x)
        x = op.Relu(x)
        x = self.conv3(op, x)
        return op.Relu(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class DPTForDepthEstimation(nn.Module):
    """DPT (Dense Prediction Transformer) for monocular depth estimation.

    Wraps a ViT encoder with DPT-style readout projection, multi-scale
    reassembly, feature fusion, and a dense depth prediction head.
    Output is a (batch, height, width) disparity map.

    HuggingFace model type: ``dpt``
    Reference: Intel/dpt-large, Intel/dpt-hybrid-midas
    """

    default_task = "depth-estimation"

    def __init__(self, config: DPTConfig):
        super().__init__()
        self.config = config
        self._patch_size = config.patch_size

        self.backbone = _DPTViTBackbone(config)
        self.neck = _DPTNeck(config)
        self.head = _DPTDepthHead(config)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value) -> ir.Value:
        # Extract multi-scale features from ViT backbone
        feature_maps = self.backbone(op, pixel_values)  # list of (B, S+1, H)

        # Static patch grid dimensions (known at export time)
        ph = self.config.image_size // self._patch_size
        patch_height = op.Reshape(op.Constant(value_int=ph), op.Constant(value_ints=[1]))
        patch_width = op.Reshape(op.Constant(value_int=ph), op.Constant(value_ints=[1]))

        fused = self.neck(op, feature_maps, patch_height, patch_width)
        return self.head(op, fused)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        result = {}
        for name, tensor in state_dict.items():
            new_name = _rename_dpt_weight(name)
            if new_name is not None:
                result[new_name] = tensor
        return result


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------

# Mapping for ViT encoder layer sub-modules
_ENCODER_LAYER_RENAMES = {
    "attention.attention.query": "self_attn.q_proj",
    "attention.attention.key": "self_attn.k_proj",
    "attention.attention.value": "self_attn.v_proj",
    "attention.output.dense": "self_attn.out_proj",
    "intermediate.dense": "mlp.up_proj",
    "output.dense": "mlp.down_proj",
}

# dpt.encoder.layer.N.<suffix>
_ENCODER_LAYER_PATTERN = re.compile(r"^dpt\.encoder\.layer\.(\d+)\.(.+)$")


def _rename_dpt_weight(name: str) -> str | None:
    """Map HuggingFace DPT weight names to our module naming convention.

    DPTForDepthEstimation uses ``dpt.`` prefix for backbone but NOT for
    neck/head.  DPTModel uses ``dpt.`` for everything.  Handle both.
    """
    # Backbone embeddings -------------------------------------------------
    if name == "dpt.embeddings.cls_token":
        return "backbone.cls_token"
    if name == "dpt.embeddings.position_embeddings":
        return "backbone.position_embeddings"
    if name.startswith("dpt.embeddings.patch_embeddings.projection."):
        suffix = name[len("dpt.embeddings.patch_embeddings.projection."):]
        return f"backbone.patch_embeddings.projection.{suffix}"

    # Backbone final layernorm — used only for feature extraction; our
    # model doesn't include a separate layernorm after the backbone.
    if name.startswith("dpt.layernorm."):
        return None

    # Backbone encoder layers ---------------------------------------------
    m = _ENCODER_LAYER_PATTERN.match(name)
    if m:
        layer_idx, suffix = m.group(1), m.group(2)
        if suffix.startswith("layernorm_"):
            return f"backbone.encoder.{layer_idx}.{suffix}"
        for old, new in _ENCODER_LAYER_RENAMES.items():
            if suffix.startswith(old):
                remainder = suffix[len(old):]
                return f"backbone.encoder.{layer_idx}.{new}{remainder}"
        return None

    # Strip optional dpt.neck. or dpt. prefix for neck weights
    neck_name = name
    if neck_name.startswith("dpt.neck."):
        neck_name = neck_name[len("dpt."):]
    elif neck_name.startswith("dpt."):
        neck_name = neck_name[len("dpt."):]

    # Neck: readout projections -------------------------------------------
    # HF: neck.reassemble_stage.readout_projects.N.0.weight/bias
    # Ours: neck.readout_projections.N.projection.weight/bias
    m_ro = re.match(
        r"^neck\.reassemble_stage\.readout_projects\.(\d+)\.0\.(weight|bias)$",
        neck_name,
    )
    if m_ro:
        idx, param = m_ro.group(1), m_ro.group(2)
        return f"neck.readout_projections.{idx}.projection.{param}"

    # Neck: reassemble layers projection and resize -----------------------
    if neck_name.startswith("neck.reassemble_stage.layers."):
        suffix = neck_name[len("neck.reassemble_stage.layers."):]
        return f"neck.reassemble_layers.{suffix}"

    # Neck: channel alignment convs (Conv2dNoBias — no bias) --------------
    if neck_name.startswith("neck.convs."):
        return neck_name

    # Neck: feature fusion stage ------------------------------------------
    # HF: neck.fusion_stage.layers.N.{projection,residual_layer1,residual_layer2}.*
    # Ours: neck.fusion_layers.N.*
    # Note: Layer 0 has no residual_layer1 (no input from previous layer);
    # HF creates it but fills with random values (MISSING in load report).
    if neck_name.startswith("neck.fusion_stage.layers."):
        if "layers.0.residual_layer1." in neck_name:
            return None  # Skip randomly-initialized unused weights
        suffix = neck_name[len("neck.fusion_stage.layers."):]
        return f"neck.fusion_layers.{suffix}"

    # Head ----------------------------------------------------------------
    head_map = {"0": "conv1", "2": "conv2", "4": "conv3"}
    m_head = re.match(r"^head\.head\.(\d+)\.(.+)$", name)
    if m_head:
        idx, param = m_head.group(1), m_head.group(2)
        if idx in head_map:
            return f"head.head.{head_map[idx]}.{param}"
        return None

    return None
