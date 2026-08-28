# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable Qwen-like standalone CLIP sidecar graph components.

The modules reproduce the graph math used by llama.cpp's ``exaone4_5``,
``kimik25``, and ``kimivl`` projectors at commit
8d9af256337d1a501250f9bbf4c0859a654bddd6.
"""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._common import LayerNorm, Linear
from mobius.components._mlp import FCMLP, GatedMLP
from mobius.components._qwen25_vl_vision import (
    Qwen25VLVisionModel,
    Qwen25VLVisionRotaryEmbedding,
)
from mobius.components._rms_norm import RMSNorm


class DualTemporalPatchEmbedding(nn.Module):
    """Sum independent spatial convolutions over two temporal patch halves.

    Input is the Qwen processor contract ``(N, C*T*P*P)`` with ``T=2``.
    Unlike a Conv3d patch projection, EXAONE stores one ``(D,C,P,P)`` kernel
    for each temporal half and adds their outputs.
    """

    def __init__(self, hidden_size: int, in_channels: int, patch_size: int):
        super().__init__()
        self._hidden_size = hidden_size
        self._in_channels = in_channels
        self._patch_size = patch_size
        shape = [hidden_size, in_channels, patch_size, patch_size]
        self.weight_0 = nn.Parameter(shape)
        self.weight_1 = nn.Parameter(shape)

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        p = self._patch_size
        pixel_values = op.CastLike(pixel_values, self.weight_0)
        # (N, C*2*P*P) -> (N, 2, C, P, P), preserving processor flatten order.
        patches = op.Reshape(pixel_values, [-1, 2, self._in_channels, p, p])
        first = op.Squeeze(op.Gather(patches, [0], axis=1), [1])
        second = op.Squeeze(op.Gather(patches, [1], axis=1), [1])
        first = op.Conv(first, self.weight_0, kernel_shape=[p, p], strides=[p, p])
        second = op.Conv(second, self.weight_1, kernel_shape=[p, p], strides=[p, p])
        return op.Reshape(op.Add(first, second), [-1, self._hidden_size])


class SplitVisionRotaryEmbedding(nn.Module):
    """Two-dimensional split-half RoPE frequencies for converted Q/K tensors.

    ``position_ids`` is ``(N, 2)`` in ``[height, width]`` order. The resulting
    frequency layout is ``[width, height, width, height]``. Kimi-K2.5 is
    interleaved natively, but llama.cpp conversion permutes each Q/K head to
    this split layout before serialization.
    """

    def __init__(self, head_dim: int, theta: float = 10_000.0):
        super().__init__()
        if head_dim % 4:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
        axis_dim = head_dim // 2
        inv_freq = 1.0 / (theta ** (np.arange(0, axis_dim, 2, dtype=np.float32) / axis_dim))
        self.inv_freq = nn.Parameter([head_dim // 4], data=ir.tensor(inv_freq))
        self.inv_freq._keep_float32 = True  # type: ignore[attr-defined]

    def forward(self, op: OpBuilder, position_ids: ir.Value):
        positions = op.Cast(position_ids, to=ir.DataType.FLOAT)
        height = op.Squeeze(op.Gather(positions, [0], axis=1), [1])
        width = op.Squeeze(op.Gather(positions, [1], axis=1), [1])
        inv_freq = op.Cast(self.inv_freq, to=ir.DataType.FLOAT)
        width_freq = op.Mul(op.Unsqueeze(width, [1]), inv_freq)
        height_freq = op.Mul(op.Unsqueeze(height, [1]), inv_freq)
        # Conversion groups native [x0,y0,x1,y1,...] complex pairs into
        # [x0,x1,...,y0,y1,...]. Each scalar frequency rotates an adjacent
        # real/imaginary pair, hence the explicit final duplication.
        scalars = op.Concat(width_freq, height_freq, axis=-1)
        frequencies = op.Reshape(
            op.Concat(op.Unsqueeze(scalars, [2]), op.Unsqueeze(scalars, [2]), axis=2),
            [0, -1],
        )
        return op.Cos(frequencies), op.Sin(frequencies)


class InterleavedVisionRotaryEmbedding(SplitVisionRotaryEmbedding):
    """Native Kimi-VL ``[x0,y0,x1,y1,...]`` adjacent-pair 2-D RoPE."""

    def forward(self, op: OpBuilder, position_ids: ir.Value):
        positions = op.Cast(position_ids, to=ir.DataType.FLOAT)
        height = op.Squeeze(op.Gather(positions, [0], axis=1), [1])
        width = op.Squeeze(op.Gather(positions, [1], axis=1), [1])
        inv_freq = op.Cast(self.inv_freq, to=ir.DataType.FLOAT)
        width_freq = op.Mul(op.Unsqueeze(width, [1]), inv_freq)
        height_freq = op.Mul(op.Unsqueeze(height, [1]), inv_freq)
        # (N,F,2 axes) -> (N,2F), retaining the native alternating axes.
        scalars = op.Reshape(
            op.Concat(
                op.Unsqueeze(width_freq, [2]),
                op.Unsqueeze(height_freq, [2]),
                axis=2,
            ),
            [0, -1],
        )
        frequencies = op.Reshape(
            op.Concat(op.Unsqueeze(scalars, [2]), op.Unsqueeze(scalars, [2]), axis=2),
            [0, -1],
        )
        return op.Cos(frequencies), op.Sin(frequencies)


class _SplitRotaryAttentionBase(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        if hidden_size % num_heads or num_heads % num_kv_heads:
            raise ValueError("attention head counts must divide hidden_size and each other")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.kv_size = num_kv_heads * self.head_dim
        self.proj = Linear(hidden_size, hidden_size, bias=True)

    def _project(self, op: OpBuilder, hidden_states: ir.Value):
        raise NotImplementedError

    def _build_block_diagonal_bias(
        self,
        op: OpBuilder,
        cu_seqlens: ir.Value,
        seq_len: ir.Value,
        dtype_ref: ir.Value,
    ) -> ir.Value:
        indices = op.Range(0, op.Squeeze(seq_len, [0]), 1)
        segment_ids = op.Sub(
            op.ReduceSum(
                op.Cast(
                    op.GreaterOrEqual(
                        op.Unsqueeze(indices, [1]),
                        op.Unsqueeze(op.Cast(cu_seqlens, to=ir.DataType.INT64), [0]),
                    ),
                    to=ir.DataType.INT64,
                ),
                [1],
                keepdims=False,
            ),
            1,
        )
        same = op.Equal(op.Unsqueeze(segment_ids, [1]), op.Unsqueeze(segment_ids, [0]))
        return op.Where(same, op.CastLike(0.0, dtype_ref), op.CastLike(-1e9, dtype_ref))

    def _rotate(
        self, op: OpBuilder, value: ir.Value, cos: ir.Value, sin: ir.Value
    ) -> ir.Value:
        half = self.head_dim // 2
        first = op.Slice(value, [0], [half], axes=[2])
        second = op.Slice(value, [half], [self.head_dim], axes=[2])
        cos = op.Unsqueeze(op.CastLike(cos, value), [1])
        sin = op.Unsqueeze(op.CastLike(sin, value), [1])
        return op.Concat(
            op.Sub(
                op.Mul(first, op.Slice(cos, [0], [half], axes=[2])),
                op.Mul(second, op.Slice(sin, [0], [half], axes=[2])),
            ),
            op.Add(
                op.Mul(first, op.Slice(sin, [half], [self.head_dim], axes=[2])),
                op.Mul(second, op.Slice(cos, [half], [self.head_dim], axes=[2])),
            ),
            axis=-1,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        cos: ir.Value,
        sin: ir.Value,
        attention_bias: ir.Value | None = None,
    ) -> ir.Value:
        query, key, value = self._project(op, hidden_states)
        query = self._rotate(op, query, cos, sin)
        key = self._rotate(op, key, cos, sin)
        # ONNX Attention's 4-D form directly expresses GQA: Hq may be a
        # multiple of Hkv, matching EXAONE's production 32Q/8KV layout.
        query = op.Unsqueeze(op.Transpose(query, perm=[1, 0, 2]), [0])
        key = op.Unsqueeze(op.Transpose(key, perm=[1, 0, 2]), [0])
        value = op.Unsqueeze(op.Transpose(value, perm=[1, 0, 2]), [0])
        if attention_bias is not None:
            attention_bias = op.Unsqueeze(attention_bias, [0, 1])
        output = op.Attention(
            query,
            key,
            value,
            attention_bias,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        output = op.Transpose(op.Squeeze(output, [0]), perm=[1, 0, 2])
        return self.proj(op, op.Reshape(output, [-1, self.hidden_size]))


class GroupedQueryVisionAttention(_SplitRotaryAttentionBase):
    """Fused-QKV split-RoPE attention supporting grouped-query heads."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__(hidden_size, num_heads, num_kv_heads)
        self.qkv = Linear(hidden_size, hidden_size + 2 * self.kv_size, bias=True)

    def _project(self, op: OpBuilder, hidden_states: ir.Value):
        qkv = self.qkv(op, hidden_states)
        query = op.Slice(qkv, [0], [self.hidden_size], axes=[1])
        key = op.Slice(qkv, [self.hidden_size], [self.hidden_size + self.kv_size], axes=[1])
        value = op.Slice(
            qkv,
            [self.hidden_size + self.kv_size],
            [self.hidden_size + 2 * self.kv_size],
            axes=[1],
        )
        query = op.Reshape(query, [-1, self.num_heads, self.head_dim])
        key = op.Reshape(key, [-1, self.num_kv_heads, self.head_dim])
        value = op.Reshape(value, [-1, self.num_kv_heads, self.head_dim])
        return query, key, value


class SplitQKVVisionAttention(_SplitRotaryAttentionBase):
    """Separate-Q/K/V full vision attention, as stored by Kimi-VL."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__(hidden_size, num_heads, num_heads)
        self.q_proj = Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = Linear(hidden_size, hidden_size, bias=True)
        self.proj = Linear(hidden_size, hidden_size, bias=True)

    def _rotate(
        self, op: OpBuilder, value: ir.Value, cos: ir.Value, sin: ir.Value
    ) -> ir.Value:
        # Kimi uses complex adjacent pairs (..., real, imag), unlike Qwen's
        # rotate-half layout. K2.5 conversion only reorders which 2-D axis owns
        # each pair; it does not separate real and imaginary channels.
        even = op.Slice(value, [0], [self.head_dim], axes=[2], steps=[2])
        odd = op.Slice(value, [1], [self.head_dim], axes=[2], steps=[2])
        cos_even = op.Unsqueeze(
            op.Slice(op.CastLike(cos, value), [0], [self.head_dim], axes=[1], steps=[2]),
            [1],
        )
        sin_even = op.Unsqueeze(
            op.Slice(op.CastLike(sin, value), [0], [self.head_dim], axes=[1], steps=[2]),
            [1],
        )
        rotated_even = op.Sub(op.Mul(even, cos_even), op.Mul(odd, sin_even))
        rotated_odd = op.Add(op.Mul(even, sin_even), op.Mul(odd, cos_even))
        paired = op.Concat(
            op.Unsqueeze(rotated_even, [3]),
            op.Unsqueeze(rotated_odd, [3]),
            axis=3,
        )
        return op.Reshape(paired, [-1, self.num_heads, self.head_dim])

    def _project(self, op: OpBuilder, hidden_states: ir.Value):
        shape = [-1, self.num_heads, self.head_dim]
        return (
            op.Reshape(self.q_proj(op, hidden_states), shape),
            op.Reshape(self.k_proj(op, hidden_states), shape),
            op.Reshape(self.v_proj(op, hidden_states), shape),
        )


class FusedQKVVisionAttention(GroupedQueryVisionAttention):
    """Fused-QKV full attention used by converted Kimi-K2.5 weights."""

    def __init__(self, hidden_size: int, num_heads: int):
        _SplitRotaryAttentionBase.__init__(self, hidden_size, num_heads, num_heads)
        self.qkv = Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = Linear(hidden_size, hidden_size, bias=True)

    _rotate = SplitQKVVisionAttention._rotate


class VisionTransformerLayer(nn.Module):
    """LayerNorm, split-RoPE attention, and GELU MLP vision block."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        *,
        fused_qkv: bool,
    ):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps=norm_eps)
        self.norm2 = LayerNorm(hidden_size, eps=norm_eps)
        attention_type = FusedQKVVisionAttention if fused_qkv else SplitQKVVisionAttention
        self.attn = attention_type(hidden_size, num_heads)
        self.mlp = FCMLP(hidden_size, intermediate_size, activation="gelu", bias=True)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        cos: ir.Value,
        sin: ir.Value,
    ) -> ir.Value:
        hidden_states = op.Add(
            hidden_states, self.attn(op, self.norm1(op, hidden_states), cos, sin)
        )
        return op.Add(hidden_states, self.mlp(op, self.norm2(op, hidden_states)))


class PatchMergeMLPProjector(nn.Module):
    """Per-patch LayerNorm, 2-D patch merge, and GELU two-layer projector."""

    def __init__(
        self,
        hidden_size: int,
        projector_hidden_size: int,
        output_size: int,
        merge_size: int,
        norm_eps: float,
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self._merge_size = merge_size
        merged_size = hidden_size * merge_size * merge_size
        self.input_norm = LayerNorm(hidden_size, eps=norm_eps)
        self.linear_1 = Linear(merged_size, projector_hidden_size, bias=True)
        self.linear_2 = Linear(projector_hidden_size, output_size, bias=True)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        grid_height: ir.Value,
        grid_width: ir.Value,
    ) -> ir.Value:
        hidden_states = self.input_norm(op, hidden_states)
        merge = self._merge_size
        aligned_h = op.Mul(op.Div(op.Add(grid_height, merge - 1), merge), merge)
        aligned_w = op.Mul(op.Div(op.Add(grid_width, merge - 1), merge), merge)
        merged_h = op.Div(aligned_h, merge)
        merged_w = op.Div(aligned_w, merge)
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                op.Reshape(grid_height, [1]),
                op.Reshape(grid_width, [1]),
                [self._hidden_size],
                axis=0,
            ),
        )
        pads = op.Concat(
            [0, 0, 0],
            op.Reshape(op.Sub(aligned_h, grid_height), [1]),
            op.Reshape(op.Sub(aligned_w, grid_width), [1]),
            [0],
            axis=0,
        )
        hidden_states = op.Pad(hidden_states, pads)
        shape = op.Concat(
            op.Reshape(merged_h, [1]),
            [merge],
            op.Reshape(merged_w, [1]),
            [merge, self._hidden_size],
            axis=0,
        )
        hidden_states = op.Reshape(hidden_states, shape)
        # (H/m,m,W/m,m,C) -> (H/m,W/m,m,m,C): consecutive merged tokens
        # contain [top-left, top-right, bottom-left, bottom-right] channel rows.
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1, 3, 4])
        hidden_states = op.Reshape(hidden_states, [-1, merge * merge * self._hidden_size])
        return self.linear_2(op, op.Gelu(self.linear_1(op, hidden_states)))


class LearnedPositionGrid3D(nn.Module):
    """Stored ``(C,W,H)`` learned positions with bicubic spatial resize."""

    def __init__(self, hidden_size: int, stored_height: int, stored_width: int):
        super().__init__()
        self._hidden_size = hidden_size
        self.position_embeddings = nn.Parameter([hidden_size, stored_width, stored_height])

    def forward(self, op: OpBuilder, grid_height: ir.Value, grid_width: ir.Value) -> ir.Value:
        # llama.cpp stores C,W,H; convert to N,C,H,W before bicubic interpolation.
        table = op.Unsqueeze(op.Transpose(self.position_embeddings, perm=[0, 2, 1]), [0])
        sizes = op.Concat(
            [1, self._hidden_size],
            op.Reshape(grid_height, [1]),
            op.Reshape(grid_width, [1]),
            axis=0,
        )
        resized = op.Resize(
            table,
            None,
            None,
            sizes,
            mode="cubic",
            coordinate_transformation_mode="half_pixel",
        )
        return op.Transpose(op.Reshape(resized, [self._hidden_size, -1]), perm=[1, 0])


class LearnedPositionGrid2D(nn.Module):
    """Flattened learned 2-D positions with bicubic spatial resize."""

    def __init__(self, hidden_size: int, stored_height: int, stored_width: int):
        super().__init__()
        self._hidden_size = hidden_size
        self._stored_height = stored_height
        self._stored_width = stored_width
        self.position_embeddings = nn.Parameter([stored_height * stored_width, hidden_size])

    def forward(self, op: OpBuilder, grid_height: ir.Value, grid_width: ir.Value) -> ir.Value:
        table = op.Reshape(
            self.position_embeddings,
            [self._stored_height, self._stored_width, self._hidden_size],
        )
        table = op.Unsqueeze(op.Transpose(table, perm=[2, 0, 1]), [0])
        sizes = op.Concat(
            [1, self._hidden_size],
            op.Reshape(grid_height, [1]),
            op.Reshape(grid_width, [1]),
            axis=0,
        )
        resized = op.Resize(
            table,
            None,
            None,
            sizes,
            mode="cubic",
            coordinate_transformation_mode="half_pixel",
        )
        return op.Transpose(op.Reshape(resized, [self._hidden_size, -1]), perm=[1, 0])


class SpatialPatchEmbedding(nn.Module):
    """Bias-free spatial Conv2d patch embedding for processor pixel tensors."""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self._hidden_size = hidden_size
        self._patch_size = patch_size
        self.proj = nn.Parameter([hidden_size, in_channels, patch_size, patch_size])
        self.bias = nn.Parameter([hidden_size])

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        p = self._patch_size
        pixel_values = op.CastLike(pixel_values, self.proj)
        hidden_states = op.Conv(
            pixel_values, self.proj, self.bias, kernel_shape=[p, p], strides=[p, p]
        )
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 3, 1])
        return op.Reshape(hidden_states, [-1, self._hidden_size])


def _position_ids(op: OpBuilder, height: ir.Value, width: ir.Value) -> ir.Value:
    count = op.Mul(height, width)
    indices = op.Range(0, count, 1)
    return op.Concat(
        op.Unsqueeze(op.Div(indices, width), [1]),
        op.Unsqueeze(op.Mod(indices, width), [1]),
        axis=1,
    )


class _KimiVisionSidecar(nn.Module):
    _patch_size: int
    patch_embed: nn.Module
    position_embedding: nn.Module
    rotary_embedding: nn.Module
    layers: nn.ModuleList
    final_layernorm: nn.Module
    projector: nn.Module

    def _forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        image_height = op.Squeeze(op.Shape(pixel_values, start=2, end=3), [0])
        image_width = op.Squeeze(op.Shape(pixel_values, start=3, end=4), [0])
        grid_height = op.Div(image_height, self._patch_size)
        grid_width = op.Div(image_width, self._patch_size)
        hidden_states = self.patch_embed(op, pixel_values)
        hidden_states = op.Add(
            hidden_states, self.position_embedding(op, grid_height, grid_width)
        )
        positions = _position_ids(op, grid_height, grid_width)
        cos, sin = self.rotary_embedding(op, positions)
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, cos, sin)
        hidden_states = self.final_layernorm(op, hidden_states)
        return self.projector(op, hidden_states, grid_height, grid_width)


class KimiK25VisionSidecar(_KimiVisionSidecar):
    """Kimi-K2.5 standalone MoonViT3d image-side graph."""

    def __init__(
        self,
        *,
        depth: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        patch_size: int,
        in_channels: int,
        stored_height: int,
        stored_width: int,
        projector_hidden_size: int,
        output_size: int,
        merge_size: int = 2,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self._patch_size = patch_size
        self.patch_embed = SpatialPatchEmbedding(in_channels, hidden_size, patch_size)
        self.position_embedding = LearnedPositionGrid3D(
            hidden_size, stored_height, stored_width
        )
        self.rotary_embedding = SplitVisionRotaryEmbedding(hidden_size // num_heads)
        self.layers = nn.ModuleList(
            [
                VisionTransformerLayer(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    norm_eps,
                    fused_qkv=True,
                )
                for _ in range(depth)
            ]
        )
        self.final_layernorm = LayerNorm(hidden_size, eps=norm_eps)
        self.projector = PatchMergeMLPProjector(
            hidden_size,
            projector_hidden_size,
            output_size,
            merge_size,
            norm_eps,
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        return self._forward(op, pixel_values)


class KimiVLVisionSidecar(_KimiVisionSidecar):
    """Kimi-VL standalone image-side graph with learned 2-D positions."""

    def __init__(
        self,
        *,
        depth: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        patch_size: int,
        in_channels: int,
        stored_height: int,
        stored_width: int,
        projector_hidden_size: int,
        output_size: int,
        merge_size: int = 2,
    ):
        super().__init__()
        self._patch_size = patch_size
        self.patch_embed = SpatialPatchEmbedding(in_channels, hidden_size, patch_size)
        self.position_embedding = LearnedPositionGrid2D(
            hidden_size, stored_height, stored_width
        )
        self.rotary_embedding = InterleavedVisionRotaryEmbedding(hidden_size // num_heads)
        self.layers = nn.ModuleList(
            [
                VisionTransformerLayer(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    1e-5,
                    fused_qkv=False,
                )
                for _ in range(depth)
            ]
        )
        self.final_layernorm = LayerNorm(hidden_size, eps=1e-5)
        self.projector = PatchMergeMLPProjector(
            hidden_size,
            projector_hidden_size,
            output_size,
            merge_size,
            1e-5,
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        return self._forward(op, pixel_values)


class ExaoneVisionLayer(nn.Module):
    """EXAONE RMS pre-norm transformer layer with GQA and gated SiLU MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = GroupedQueryVisionAttention(hidden_size, num_heads, num_kv_heads)
        self.mlp = GatedMLP(hidden_size, intermediate_size, activation="silu", bias=True)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
        cos: ir.Value,
        sin: ir.Value,
    ) -> ir.Value:
        seq_len = op.Shape(hidden_states, start=0, end=1)
        bias = self.attn._build_block_diagonal_bias(op, cu_seqlens, seq_len, hidden_states)
        hidden_states = op.Add(
            hidden_states,
            self.attn(op, self.norm1(op, hidden_states), cos, sin, bias),
        )
        return op.Add(hidden_states, self.mlp(op, self.norm2(op, hidden_states)))


class ExaonePatchMerger(nn.Module):
    """EXAONE post-RMSNorm and consecutive 2x2 merger MLP."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, output_size: int, norm_eps: float
    ):
        super().__init__()
        self.post_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        self.linear_1 = Linear(hidden_size * 4, intermediate_size, bias=True)
        self.linear_2 = Linear(intermediate_size, output_size, bias=True)
        self._hidden_size = hidden_size

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = self.post_layernorm(op, hidden_states)
        hidden_states = op.Reshape(hidden_states, [-1, 4 * self._hidden_size])
        return self.linear_2(op, op.Gelu(self.linear_1(op, hidden_states)))


class Exaone45VisionSidecar(Qwen25VLVisionModel):
    """EXAONE-4.5 standalone packed-patch sidecar with 2-D MRoPE and windows."""

    def __init__(
        self,
        *,
        depth: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        patch_size: int,
        in_channels: int,
        output_size: int,
        fullatt_block_indexes: list[int],
        window_size: int,
        norm_eps: float = 1e-6,
    ):
        nn.Module.__init__(self)
        self._fullatt_block_indexes = set(fullatt_block_indexes)
        self._spatial_merge_size = 2
        self._spatial_merge_unit = 4
        self._patch_size = patch_size
        self._hidden_size = hidden_size
        self._vit_merger_window_size = window_size // 2 // patch_size
        self.patch_embed = DualTemporalPatchEmbedding(  # type: ignore[assignment]
            hidden_size, in_channels, patch_size
        )
        # llama.cpp serializes this as four-section vision MRoPE; the equivalent
        # processor-facing graph is Qwen2.5's tested [h,w] frequency layout.
        self.rotary_pos_emb = Qwen25VLVisionRotaryEmbedding(hidden_size // num_heads // 2)
        self.blocks = nn.ModuleList(
            [
                ExaoneVisionLayer(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    num_kv_heads,
                    norm_eps,
                )
                for _ in range(depth)
            ]
        )
        self.merger = ExaonePatchMerger(  # type: ignore[assignment]
            hidden_size,
            hidden_size * 4,
            output_size,
            norm_eps,
        )
