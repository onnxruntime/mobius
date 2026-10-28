# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GLM-OCR packed vision encoder components.

Replicates the HuggingFace ``GlmOcrVisionModel``: Conv3d patch embedding,
bidirectional packed attention with Q/K RMSNorm and 2D RoPE, a post-transformer
RMSNorm, learned 2x2 downsampling, and the GLM gated patch merger.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._common import LayerNorm, Linear
from mobius.components._conv import Conv2d
from mobius.components._mlp import GatedMLP
from mobius.components._qwen25_vl_vision import (
    Qwen25VLVisionAttention,
    Qwen25VLVisionModel,
)
from mobius.components._rms_norm import RMSNorm


class GlmOcrVisionPatchEmbed(nn.Module):
    """Biased Conv3d patch embedding for packed GLM-OCR processor patches."""

    def __init__(
        self,
        *,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self._patch_size = patch_size
        self._temporal_patch_size = temporal_patch_size
        self._in_channels = in_channels
        self._hidden_size = hidden_size
        self.weight = nn.Parameter(
            [hidden_size, in_channels, temporal_patch_size, patch_size, patch_size],
            name="proj.weight",
        )
        self.bias = nn.Parameter([hidden_size], name="proj.bias")

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # [N, C*T*P*P] -> [N, C, T, P, P] -> [N, hidden]
        hidden_states = op.Reshape(
            hidden_states,
            [
                -1,
                self._in_channels,
                self._temporal_patch_size,
                self._patch_size,
                self._patch_size,
            ],
        )
        hidden_states = op.Conv(
            hidden_states,
            self.weight,
            self.bias,
            kernel_shape=[
                self._temporal_patch_size,
                self._patch_size,
                self._patch_size,
            ],
            strides=[
                self._temporal_patch_size,
                self._patch_size,
                self._patch_size,
            ],
        )
        return op.Reshape(hidden_states, [-1, self._hidden_size])


class GlmOcrVisionRotaryEmbedding(nn.Module):
    """GLM-OCR 2D rotary embeddings with dynamic float32 frequencies."""

    def __init__(self, dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        self._dim = dim
        inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        self.inv_freq = nn.Parameter([dim // 2], data=ir.tensor(inv_freq))
        self.inv_freq._keep_float32 = True

    def forward(
        self,
        op: OpBuilder,
        rotary_pos_ids: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        # Match HF's unbounded (N, 2, 1) * (dim/2,) formulation instead of
        # indexing a fixed lookup table, so extreme-aspect-ratio pages remain valid.
        position_ids = op.Cast(rotary_pos_ids, to=ir.DataType.FLOAT)
        inv_freq = op.Cast(self.inv_freq, to=ir.DataType.FLOAT)
        freqs = op.Mul(op.Unsqueeze(position_ids, [-1]), inv_freq)
        freqs = op.Reshape(freqs, [0, self._dim])  # (N, dim)
        embeddings = op.Concat(freqs, freqs, axis=-1)

        # HF evaluates GLM-OCR's rotary trigonometry in float32. Keeping Cos/Sin
        # out of bf16 also avoids unsupported ORT CUDA kernels.
        return op.Cos(embeddings), op.Sin(embeddings)


class GlmOcrVisionAttention(Qwen25VLVisionAttention):
    """GLM-OCR vision attention with per-head Q/K RMSNorm."""

    def __init__(self, hidden_size: int, num_heads: int, norm_eps: float) -> None:
        super().__init__(hidden_size, num_heads)
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
        cos: ir.Value,
        sin: ir.Value,
    ) -> ir.Value:
        seq_len = op.Shape(hidden_states, start=0, end=1)
        qkv = self.qkv(op, hidden_states)
        qkv = op.Reshape(
            qkv,
            op.Concat(seq_len, [3, self.num_heads, self.head_dim], axis=0),
        )
        qkv = op.Transpose(qkv, perm=[1, 0, 2, 3])  # [3, N, heads, head_dim]
        query = op.Squeeze(op.Gather(qkv, [0], axis=0), [0])
        key = op.Squeeze(op.Gather(qkv, [1], axis=0), [0])
        value = op.Squeeze(op.Gather(qkv, [2], axis=0), [0])

        # Normalize each head before applying 2D rotary embeddings.
        query = self.q_norm(op, query)
        key = self.k_norm(op, key)
        cos = op.CastLike(cos, query)
        sin = op.CastLike(sin, query)
        query = self._apply_rotary(op, query, cos, sin)
        key = self._apply_rotary(op, key, cos, sin)

        from mobius._build_context import ep_capabilities

        if ep_capabilities().supports_packed_multi_head_attention:
            output = self._emit_packed_mha(op, query, key, value, cu_seqlens, seq_len)
        else:
            output = self._emit_standard_attention(op, query, key, value, cu_seqlens, seq_len)
        return self.proj(op, output)


class GlmOcrVisionBlock(nn.Module):
    """GLM-OCR pre-norm vision transformer block."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = GlmOcrVisionAttention(hidden_size, num_heads, norm_eps)
        self.mlp = GatedMLP(
            hidden_size,
            intermediate_size,
            activation="silu",
            bias=True,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
        cos: ir.Value,
        sin: ir.Value,
    ) -> ir.Value:
        residual = hidden_states
        hidden_states = self.attn(
            op,
            self.norm1(op, hidden_states),
            cu_seqlens=cu_seqlens,
            cos=cos,
            sin=sin,
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(op, self.norm2(op, hidden_states))
        return op.Add(residual, hidden_states)


class GlmOcrVisionPatchMerger(nn.Module):
    """GLM-OCR projection, LayerNorm, and gated MLP merger."""

    def __init__(self, hidden_size: int, context_size: int) -> None:
        super().__init__()
        self.proj = Linear(hidden_size, hidden_size, bias=False)
        self.post_projection_norm = LayerNorm(hidden_size, eps=1e-5)
        self.gate_proj = Linear(hidden_size, context_size, bias=False)
        self.up_proj = Linear(hidden_size, context_size, bias=False)
        self.down_proj = Linear(context_size, hidden_size, bias=False)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = self.proj(op, hidden_states)
        hidden_states = op.Gelu(self.post_projection_norm(op, hidden_states))
        gate = self.gate_proj(op, hidden_states)
        gate = op.Mul(op.Sigmoid(gate), gate)
        return self.down_proj(
            op,
            op.Mul(gate, self.up_proj(op, hidden_states)),
        )


class GlmOcrVisionModel(Qwen25VLVisionModel):
    """Complete GLM-OCR packed vision tower."""

    def __init__(
        self,
        *,
        depth: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        out_hidden_size: int,
        spatial_merge_size: int,
        norm_eps: float,
    ) -> None:
        nn.Module.__init__(self)
        self._spatial_merge_size = spatial_merge_size
        self._spatial_merge_unit = spatial_merge_size * spatial_merge_size
        self._hidden_size = hidden_size
        self._out_hidden_size = out_hidden_size

        self.patch_embed = GlmOcrVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
        )
        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = GlmOcrVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                GlmOcrVisionBlock(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    norm_eps,
                )
                for _ in range(depth)
            ]
        )
        self.post_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        self.downsample = Conv2d(
            hidden_size,
            out_hidden_size,
            kernel_size=spatial_merge_size,
            stride=spatial_merge_size,
        )
        self.merger = GlmOcrVisionPatchMerger(
            out_hidden_size,
            out_hidden_size * in_channels,
        )

    def _compute_rotary_pos_ids(
        self,
        op: OpBuilder,
        image_grid_thw: ir.Value,
    ) -> ir.Value:
        """Vectorize packed 2D positions across variable-size images."""
        temporal = op.Squeeze(
            op.Slice(image_grid_thw, [0], [1], [1], [1]),
            [1],
        )  # [num_images]
        height = op.Squeeze(
            op.Slice(image_grid_thw, [1], [2], [1], [1]),
            [1],
        )  # [num_images]
        width = op.Squeeze(
            op.Slice(image_grid_thw, [2], [3], [1], [1]),
            [1],
        )  # [num_images]
        patch_counts = op.Mul(temporal, op.Mul(height, width))
        max_patches = op.ReduceMax(patch_counts, keepdims=False)
        patch_index = op.Range(
            op.Constant(value_int=0),
            max_patches,
            op.Constant(value_int=1),
        )
        patch_index = op.Unsqueeze(patch_index, [0])  # [1, max_patches]

        merge = op.Constant(value_int=self._spatial_merge_size)
        merge_unit = op.Constant(value_int=self._spatial_merge_unit)
        width_blocks = op.Unsqueeze(op.Div(width, merge), [1])
        height_blocks = op.Unsqueeze(op.Div(height, merge), [1])

        # Processor order is [T, H/merge, W/merge, merge, merge]. Recover
        # the original H/W coordinate for every packed patch without Scan.
        width_in_block = op.Mod(patch_index, merge)
        height_in_block = op.Mod(op.Div(patch_index, merge), merge)
        block_index = op.Div(patch_index, merge_unit)
        width_block = op.Mod(block_index, width_blocks)
        height_block = op.Mod(op.Div(block_index, width_blocks), height_blocks)
        height_pos = op.Add(op.Mul(height_block, merge), height_in_block)
        width_pos = op.Add(op.Mul(width_block, merge), width_in_block)
        position_ids = op.Concat(
            op.Unsqueeze(height_pos, [2]),
            op.Unsqueeze(width_pos, [2]),
            axis=2,
        )  # [num_images, max_patches, 2]

        valid = op.Less(
            patch_index,
            op.Unsqueeze(patch_counts, [1]),
        )  # [num_images, max_patches]
        flat_valid = op.Reshape(valid, [-1])
        valid_indices = op.Squeeze(op.NonZero(flat_valid), [0])
        return op.Gather(
            op.Reshape(position_ids, [-1, 2]),
            valid_indices,
            axis=0,
        )

    def _compute_cu_seqlens(
        self,
        op: OpBuilder,
        image_grid_thw: ir.Value,
    ) -> ir.Value:
        """Vectorize per-frame packed-attention boundaries across images."""
        temporal = op.Squeeze(
            op.Slice(image_grid_thw, [0], [1], [1], [1]),
            [1],
        )  # [num_images]
        height = op.Squeeze(
            op.Slice(image_grid_thw, [1], [2], [1], [1]),
            [1],
        )
        width = op.Squeeze(
            op.Slice(image_grid_thw, [2], [3], [1], [1]),
            [1],
        )
        max_temporal = op.ReduceMax(temporal, keepdims=False)
        frame_index = op.Range(
            op.Constant(value_int=0),
            max_temporal,
            op.Constant(value_int=1),
        )
        frame_index = op.Unsqueeze(frame_index, [0])  # [1, max_temporal]
        frame_shape = op.Concat(
            op.Shape(temporal),
            op.Reshape(max_temporal, [1]),
            axis=0,
        )
        frame_lengths = op.Expand(
            op.Unsqueeze(op.Mul(height, width), [1]),
            frame_shape,
        )  # [num_images, max_temporal]
        valid = op.Less(frame_index, op.Unsqueeze(temporal, [1]))
        flat_valid = op.Reshape(valid, [-1])
        valid_indices = op.Squeeze(op.NonZero(flat_valid), [0])
        frame_lengths = op.Gather(
            op.Reshape(frame_lengths, [-1]),
            valid_indices,
            axis=0,
        )
        cumulative = op.CumSum(frame_lengths, op.Constant(value_int=0))
        return op.Pad(
            cumulative,
            op.Constant(value_ints=[1, 0]),
            op.Constant(value_int=0),
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ) -> ir.Value:
        # The processor emits packed [N, C*T*P*P] patches in spatial-merge order.
        hidden_states = self.patch_embed(op, pixel_values)  # [N, vision_hidden]
        rotary_pos_ids = self._compute_rotary_pos_ids(op, image_grid_thw)
        cu_seqlens = self._compute_cu_seqlens(op, image_grid_thw)
        cos, sin = self.rotary_pos_emb(op, rotary_pos_ids)

        for block in self.blocks:
            hidden_states = block(
                op,
                hidden_states,
                cu_seqlens=cu_seqlens,
                cos=cos,
                sin=sin,
            )
        hidden_states = self.post_layernorm(op, hidden_states)

        # [N, C] -> [N/merge^2, C, merge, merge] -> [N/merge^2, out_hidden]
        merge = self._spatial_merge_size
        hidden_states = op.Reshape(
            hidden_states,
            [-1, merge, merge, self._hidden_size],
        )
        hidden_states = op.Transpose(hidden_states, perm=[0, 3, 1, 2])
        hidden_states = self.downsample(op, hidden_states)
        hidden_states = op.Reshape(hidden_states, [-1, self._out_hidden_size])
        return self.merger(op, hidden_states)
