# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable vision-side components for MiMoVL and MiniMax-M3 CLIP graphs."""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._common import LayerNorm, Linear
from mobius.components._rms_norm import RMSNorm


class F32AccumulationLinear(nn.Module):
    """Linear whose matrix product is accumulated in float32."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool):
        super().__init__()
        self.weight = nn.Parameter([out_features, in_features])
        self.bias = nn.Parameter([out_features]) if bias else None

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        x_f32 = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        weight_f32 = op.Cast(self.weight, to=ir.DataType.FLOAT)
        output = op.MatMul(x_f32, op.Transpose(weight_f32, perm=[1, 0]))
        if self.bias is not None:
            output = op.Add(output, op.Cast(self.bias, to=ir.DataType.FLOAT))
        return op.CastLike(output, hidden_states)


class DualTemporalPatchEmbedding(nn.Module):
    """A temporal-depth-two Conv3D represented by two bias-free Conv2D slices."""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.weight_0 = nn.Parameter(
            [hidden_size, in_channels, patch_size, patch_size],
            name="patch_embeddings_0",
        )
        self.weight_1 = nn.Parameter(
            [hidden_size, in_channels, patch_size, patch_size],
            name="patch_embeddings_1",
        )
        self._in_channels = in_channels
        self._hidden_size = hidden_size
        self._patch_size = patch_size
        self._half = in_channels * patch_size * patch_size

    def forward(self, op: OpBuilder, pixel_patches: ir.Value) -> ir.Value:
        # Flat temporal patches are [t0(C,P,P), t1(C,P,P)].
        pixel_patches = op.CastLike(pixel_patches, self.weight_0)
        first = op.Slice(pixel_patches, [0], [self._half], [1])
        second = op.Slice(pixel_patches, [self._half], [2 * self._half], [1])
        shape = [-1, self._in_channels, self._patch_size, self._patch_size]
        first = op.Reshape(first, shape)
        second = op.Reshape(second, shape)
        conv_args = {
            "kernel_shape": [self._patch_size, self._patch_size],
            "strides": [self._patch_size, self._patch_size],
        }
        first = op.Conv(first, self.weight_0, **conv_args)
        second = op.Conv(second, self.weight_1, **conv_args)
        return op.Reshape(op.Add(first, second), [-1, self._hidden_size])


class MergeUnitReorder(nn.Module):
    """Reorder whole merge units while retaining the four patches inside each unit."""

    def __init__(self, hidden_size: int, merge_size: int = 2):
        super().__init__()
        self._hidden_size = hidden_size
        self._merge_unit = merge_size * merge_size

    def forward(
        self, op: OpBuilder, hidden_states: ir.Value, unit_indices: ir.Value
    ) -> ir.Value:
        grouped = op.Reshape(hidden_states, [-1, self._hidden_size * self._merge_unit])
        grouped = op.Gather(grouped, unit_indices, axis=0)
        return op.Reshape(grouped, [-1, self._hidden_size])


class MiMoVLRotaryEmbedding(nn.Module):
    """Qwen-style 2D vision RoPE over equal row and column rotary sections."""

    def __init__(self, head_dim: int, theta: float = 10000.0, max_grid_size: int = 512):
        super().__init__()
        if head_dim % 4:
            raise ValueError("MiMoVL head_dim must be divisible by four")
        quarter = head_dim // 4
        inv_freq = 1.0 / (
            theta ** (np.arange(0, quarter, dtype=np.float32) * 2.0 / (2 * quarter))
        )
        angles = np.outer(np.arange(max_grid_size, dtype=np.float32), inv_freq)
        self.cos_table = nn.Parameter(list(angles.shape), data=ir.tensor(np.cos(angles)))
        self.sin_table = nn.Parameter(list(angles.shape), data=ir.tensor(np.sin(angles)))
        self._head_dim = head_dim

    def forward(
        self, op: OpBuilder, hidden_states: ir.Value, position_ids: ir.Value
    ) -> ir.Value:
        row = op.Squeeze(op.Gather(position_ids, [0], axis=1), [1])
        column = op.Squeeze(op.Gather(position_ids, [1], axis=1), [1])
        cos = op.Concat(
            op.Gather(self.cos_table, row), op.Gather(self.cos_table, column), axis=-1
        )
        sin = op.Concat(
            op.Gather(self.sin_table, row), op.Gather(self.sin_table, column), axis=-1
        )
        cos = op.Unsqueeze(op.CastLike(cos, hidden_states), [1])
        sin = op.Unsqueeze(op.CastLike(sin, hidden_states), [1])
        first, second = op.Split(
            hidden_states,
            [self._head_dim // 2, self._head_dim // 2],
            axis=-1,
            _outputs=2,
        )
        return op.Concat(
            op.Sub(op.Mul(first, cos), op.Mul(second, sin)),
            op.Add(op.Mul(first, sin), op.Mul(second, cos)),
            axis=-1,
        )


class MiMoVLAttentionCore(nn.Module):
    """Manual GQA core supporting the virtual zero-value attention sink."""

    def __init__(self, num_query_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        if num_query_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        self._q_heads = num_query_heads
        self._kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._groups = num_query_heads // num_kv_heads
        self._scale = head_dim**-0.5

    def _repeat_kv(self, op: OpBuilder, states: ir.Value) -> ir.Value:
        tokens = op.Shape(states, start=0, end=1)
        expanded = op.Expand(
            op.Unsqueeze(states, [2]),
            op.Concat(tokens, [self._kv_heads, self._groups, self._head_dim], axis=0),
        )
        return op.Reshape(expanded, [-1, self._q_heads, self._head_dim])

    def forward(
        self,
        op: OpBuilder,
        query: ir.Value,
        key: ir.Value,
        value: ir.Value,
        attention_bias: ir.Value | None = None,
        sinks: ir.Value | None = None,
    ) -> ir.Value:
        key = self._repeat_kv(op, key)
        value = self._repeat_kv(op, value)
        query = op.Transpose(query, perm=[1, 0, 2])
        key = op.Transpose(key, perm=[1, 0, 2])
        value = op.Transpose(value, perm=[1, 0, 2])
        scores = op.Mul(
            op.MatMul(query, op.Transpose(key, perm=[0, 2, 1])),
            op.CastLike(self._scale, query),
        )
        if attention_bias is not None:
            scores = op.Add(scores, op.Unsqueeze(op.CastLike(attention_bias, scores), [0]))
        if sinks is not None:
            sink_logits = op.Expand(
                op.Reshape(op.CastLike(sinks, scores), [self._q_heads, 1, 1]),
                op.Concat(
                    [self._q_heads],
                    op.Shape(scores, start=1, end=2),
                    [1],
                    axis=0,
                ),
            )
            scores = op.Concat(scores, sink_logits, axis=-1)
            zero_value = op.Expand(
                op.CastLike(0.0, value),
                [self._q_heads, 1, self._head_dim],
            )
            value = op.Concat(value, zero_value, axis=1)
        attended = op.MatMul(op.Softmax(scores, axis=-1), value)
        return op.Transpose(attended, perm=[1, 0, 2])


class MiMoVLAttention(nn.Module):
    """Fused-QKV MiMoVL GQA with 2D RoPE and optional window sinks."""

    def __init__(
        self,
        hidden_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        windowed: bool,
    ):
        super().__init__()
        qkv_size = (num_query_heads + 2 * num_kv_heads) * head_dim
        self.qkv = F32AccumulationLinear(hidden_size, qkv_size, bias=True)
        self.proj = F32AccumulationLinear(num_query_heads * head_dim, hidden_size, bias=True)
        self.rotary = MiMoVLRotaryEmbedding(head_dim)
        self.core = MiMoVLAttentionCore(num_query_heads, num_kv_heads, head_dim)
        self.attn_sinks = nn.Parameter([num_query_heads]) if windowed else None
        self._q_size = num_query_heads * head_dim
        self._kv_size = num_kv_heads * head_dim
        self._q_heads = num_query_heads
        self._kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._windowed = windowed

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_ids: ir.Value,
        window_bias: ir.Value | None = None,
    ) -> ir.Value:
        qkv = self.qkv(op, hidden_states)
        query, key, value = op.Split(
            qkv,
            [self._q_size, self._kv_size, self._kv_size],
            axis=-1,
            _outputs=3,
        )
        query = op.Reshape(query, [-1, self._q_heads, self._head_dim])
        key = op.Reshape(key, [-1, self._kv_heads, self._head_dim])
        value = op.Reshape(value, [-1, self._kv_heads, self._head_dim])
        query = self.rotary(op, query, position_ids)
        key = self.rotary(op, key, position_ids)
        if self._windowed and window_bias is None:
            raise ValueError("windowed MiMoVL attention requires window_bias")
        attended = self.core(
            op,
            query,
            key,
            value,
            window_bias if self._windowed else None,
            self.attn_sinks,
        )
        return self.proj(op, op.Reshape(attended, [-1, self._q_size]))


class MiMoVLSwiGLU(nn.Module):
    """Bias-bearing SwiGLU with llama.cpp-compatible float32 accumulation."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = F32AccumulationLinear(hidden_size, intermediate_size, bias=True)
        self.gate_proj = F32AccumulationLinear(hidden_size, intermediate_size, bias=True)
        self.down_proj = F32AccumulationLinear(intermediate_size, hidden_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # SiLU(gate) * up; spell out SiLU to preserve the input to Sigmoid.
        gate = self.gate_proj(op, hidden_states)
        gated = op.Mul(op.Mul(gate, op.Sigmoid(gate)), self.up_proj(op, hidden_states))
        return self.down_proj(op, gated)


class MiMoVLBlock(nn.Module):
    """One row-, column-, or full-attention MiMoVL transformer layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        window_mode: int,
        merge_size: int = 2,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if window_mode not in (-1, 0, 1):
            raise ValueError("window_mode must be -1 (full), 0 (row), or 1 (column)")
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiMoVLAttention(
            hidden_size,
            num_query_heads,
            num_kv_heads,
            head_dim,
            windowed=window_mode != -1,
        )
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.mlp = MiMoVLSwiGLU(hidden_size, intermediate_size)
        self.reorder = MergeUnitReorder(hidden_size, merge_size)
        self._mode = window_mode

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        row_position_ids: ir.Value,
        column_position_ids: ir.Value,
        window_bias: ir.Value,
        column_indices: ir.Value,
        inverse_column_indices: ir.Value,
    ) -> ir.Value:
        if self._mode == 1:
            hidden_states = self.reorder(op, hidden_states, column_indices)
        position_ids = column_position_ids if self._mode == 1 else row_position_ids
        residual = hidden_states
        hidden_states = op.Add(
            residual,
            self.attn(
                op,
                self.norm1(op, hidden_states),
                position_ids,
                window_bias if self._mode != -1 else None,
            ),
        )
        hidden_states = op.Add(
            hidden_states,
            self.mlp(op, self.norm2(op, hidden_states)),
        )
        if self._mode == 1:
            hidden_states = self.reorder(op, hidden_states, inverse_column_indices)
        return hidden_states


class MiMoVLProjector(nn.Module):
    """Post-RMSNorm, merge-four, two-layer GELU projector."""

    def __init__(self, hidden_size: int, intermediate_size: int, output_size: int):
        super().__init__()
        self.post_ln = RMSNorm(hidden_size, eps=1e-6)
        self.fc1 = F32AccumulationLinear(hidden_size * 4, intermediate_size, bias=False)
        self.fc2 = F32AccumulationLinear(intermediate_size, output_size, bias=False)
        self._merged = hidden_size * 4

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        merged = op.Reshape(self.post_ln(op, hidden_states), [-1, self._merged])
        return self.fc2(op, op.Gelu(self.fc1(op, merged)))


class MiMoVLVisionSidecar(nn.Module):
    """Complete MiMoVL packed-patch tower with explicit window metadata inputs."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        patch_size: int,
        window_modes: list[int],
        projector_hidden_size: int,
        output_size: int,
        merge_size: int = 2,
    ):
        super().__init__()
        self.patch_embed = DualTemporalPatchEmbedding(3, hidden_size, patch_size)
        self.blocks = nn.ModuleList(
            [
                MiMoVLBlock(
                    hidden_size,
                    intermediate_size,
                    num_query_heads,
                    num_kv_heads,
                    head_dim,
                    window_mode=mode,
                    merge_size=merge_size,
                )
                for mode in window_modes
            ]
        )
        self.projector = MiMoVLProjector(
            hidden_size,
            projector_hidden_size,
            output_size,
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        row_position_ids: ir.Value,
        column_position_ids: ir.Value,
        window_bias: ir.Value,
        column_indices: ir.Value,
        inverse_column_indices: ir.Value,
    ) -> ir.Value:
        hidden_states = self.patch_embed(op, pixel_values)
        for block in self.blocks:
            hidden_states = block(
                op,
                hidden_states,
                row_position_ids,
                column_position_ids,
                window_bias,
                column_indices,
                inverse_column_indices,
            )
        return self.projector(op, hidden_states)


def minimax_m3_qk_permutation(head_dim: int) -> list[int]:
    """Converter permutation from interleaved axis halves to contiguous axes."""
    axis_dim = 2 * ((2 * (head_dim // 2) // 3) // 2)
    axis_half = axis_dim // 2
    half = 3 * axis_half
    return (
        list(range(axis_half))
        + list(range(half, half + axis_half))
        + list(range(axis_half, 2 * axis_half))
        + list(range(half + axis_half, half + 2 * axis_half))
        + list(range(2 * axis_half, 3 * axis_half))
        + list(range(half + 2 * axis_half, half + 3 * axis_half))
        + list(range(2 * half, head_dim))
    )


class MiniMaxM3PartialRotaryEmbedding(nn.Module):
    """Partial two-axis NEOX RoPE for converter-permuted MiniMax-M3 Q/K."""

    def __init__(self, head_dim: int, theta: float = 10000.0, max_grid_size: int = 512):
        super().__init__()
        self._axis_dim = 2 * ((2 * (head_dim // 2) // 3) // 2)
        if 3 * self._axis_dim > head_dim:
            raise ValueError("MiniMax-M3 axis dimensions exceed head_dim")
        inv_freq = 1.0 / (
            theta ** (np.arange(0, self._axis_dim, 2, dtype=np.float32) / self._axis_dim)
        )
        angles = np.outer(np.arange(max_grid_size, dtype=np.float32), inv_freq)
        self.cos_table = nn.Parameter(list(angles.shape), data=ir.tensor(np.cos(angles)))
        self.sin_table = nn.Parameter(list(angles.shape), data=ir.tensor(np.sin(angles)))
        self._head_dim = head_dim

    def _rotate(self, op: OpBuilder, states: ir.Value, positions: ir.Value) -> ir.Value:
        half = self._axis_dim // 2
        first, second = op.Split(states, [half, half], axis=-1, _outputs=2)
        cos = op.Unsqueeze(op.CastLike(op.Gather(self.cos_table, positions), states), [0, 2])
        sin = op.Unsqueeze(op.CastLike(op.Gather(self.sin_table, positions), states), [0, 2])
        return op.Concat(
            op.Sub(op.Mul(first, cos), op.Mul(second, sin)),
            op.Add(op.Mul(first, sin), op.Mul(second, cos)),
            axis=-1,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_h: ir.Value,
        position_w: ir.Value,
    ) -> ir.Value:
        time = op.Slice(hidden_states, [0], [self._axis_dim], [-1])
        height = op.Slice(hidden_states, [self._axis_dim], [2 * self._axis_dim], [-1])
        width = op.Slice(hidden_states, [2 * self._axis_dim], [3 * self._axis_dim], [-1])
        padding = op.Slice(hidden_states, [3 * self._axis_dim], [self._head_dim], [-1])
        return op.Concat(
            time,
            self._rotate(op, height, position_h),
            self._rotate(op, width, position_w),
            padding,
            axis=-1,
        )


class SpatialMergeOrder(nn.Module):
    """Put each spatial merge tile's patches consecutively without reducing tokens."""

    def __init__(self, grid_height: int, grid_width: int, hidden_size: int, merge_size: int):
        super().__init__()
        if grid_height % merge_size or grid_width % merge_size:
            raise ValueError("patch grid must be divisible by merge_size")
        self._height = grid_height
        self._width = grid_width
        self._hidden = hidden_size
        self._merge = merge_size

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        batch = op.Shape(hidden_states, start=0, end=1)
        merged_h = self._height // self._merge
        merged_w = self._width // self._merge
        shape = op.Concat(
            batch,
            [merged_h, self._merge, merged_w, self._merge, self._hidden],
            axis=0,
        )
        states = op.Reshape(hidden_states, shape)
        states = op.Transpose(states, perm=[0, 1, 3, 2, 4, 5])
        return op.Reshape(states, [-1, self._height * self._width, self._hidden])


class MiniMaxM3Attention(nn.Module):
    """MiniMax-M3 ViT attention with converter-permuted partial two-axis RoPE."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.q_proj = Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = Linear(hidden_size, hidden_size, bias=True)
        self.rotary = MiniMaxM3PartialRotaryEmbedding(hidden_size // num_heads)
        self._heads = num_heads
        self._head_dim = hidden_size // num_heads

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_h: ir.Value,
        position_w: ir.Value,
    ) -> ir.Value:
        shape = [0, 0, self._heads, self._head_dim]
        query = self.rotary(
            op, op.Reshape(self.q_proj(op, hidden_states), shape), position_h, position_w
        )
        key = self.rotary(
            op, op.Reshape(self.k_proj(op, hidden_states), shape), position_h, position_w
        )
        value = op.Reshape(self.v_proj(op, hidden_states), shape)
        query = op.Reshape(query, [0, 0, -1])
        key = op.Reshape(key, [0, 0, -1])
        value = op.Reshape(value, [0, 0, -1])
        attended = op.Attention(
            query,
            key,
            value,
            q_num_heads=self._heads,
            kv_num_heads=self._heads,
            scale=1.0 / math.sqrt(self._head_dim),
        )
        return self.out_proj(op, op.Reshape(attended, [0, 0, -1]))


class MiniMaxM3MLP(nn.Module):
    """Bias-bearing GELU-erf MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int, output_size: int):
        super().__init__()
        self.fc1 = Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = Linear(intermediate_size, output_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return self.fc2(op, op.Gelu(self.fc1(op, hidden_states), approximate="none"))


class MiniMaxM3VisionBlock(nn.Module):
    """Normal-LayerNorm MiniMax-M3 ViT block with exact GELU-erf."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxM3Attention(hidden_size, num_heads)
        self.norm2 = LayerNorm(hidden_size, eps=norm_eps)
        self.mlp = MiniMaxM3MLP(hidden_size, intermediate_size, hidden_size)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_h: ir.Value,
        position_w: ir.Value,
    ) -> ir.Value:
        hidden_states = op.Add(
            hidden_states,
            self.attn(op, self.norm1(op, hidden_states), position_h, position_w),
        )
        return op.Add(hidden_states, self.mlp(op, self.norm2(op, hidden_states)))


class MiniMaxM3Projector(nn.Module):
    """Per-patch GELU MLP followed by merge-four and a second GELU MLP."""

    def __init__(
        self,
        hidden_size: int,
        patch_mlp_size: int,
        projected_size: int,
        merger_mlp_size: int,
        output_size: int,
        merge_size: int = 2,
    ):
        super().__init__()
        self.patch_mlp = MiniMaxM3MLP(hidden_size, patch_mlp_size, projected_size)
        merged_size = projected_size * merge_size * merge_size
        self.merger_mlp = MiniMaxM3MLP(merged_size, merger_mlp_size, output_size)
        self._merged_size = merged_size

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = self.patch_mlp(op, hidden_states)
        hidden_states = op.Reshape(hidden_states, [-1, self._merged_size])
        return self.merger_mlp(op, hidden_states)


class MiniMaxM3VisionSidecar(nn.Module):
    """Complete MiniMax-M3 packed-patch tower with explicit spatial positions."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_layers: int,
        patch_size: int,
        grid_height: int,
        grid_width: int,
        patch_mlp_size: int,
        projected_size: int,
        merger_mlp_size: int,
        output_size: int,
        merge_size: int = 2,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.patch_embed = DualTemporalPatchEmbedding(3, hidden_size, patch_size)
        self.pre_vit_merge = SpatialMergeOrder(
            grid_height,
            grid_width,
            hidden_size,
            merge_size,
        )
        self.blocks = nn.ModuleList(
            [
                MiniMaxM3VisionBlock(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.projector = MiniMaxM3Projector(
            hidden_size,
            patch_mlp_size,
            projected_size,
            merger_mlp_size,
            output_size,
            merge_size,
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        position_h: ir.Value,
        position_w: ir.Value,
    ) -> ir.Value:
        hidden_states = self.pre_vit_merge(op, self.patch_embed(op, pixel_values))
        for block in self.blocks:
            hidden_states = block(op, hidden_states, position_h, position_w)
        return self.projector(op, hidden_states)
