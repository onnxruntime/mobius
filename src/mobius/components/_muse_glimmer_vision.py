# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Muse Glimmer dynamic-resolution vision encoder."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._common import LayerNorm, Linear
from mobius.components._qwen25_vl_vision import (
    Qwen25VLVisionAttention,
    Qwen25VLVisionModel,
)
from mobius.components._scan_utils import (
    compact_scan_output,
    create_body_graph,
    rename_subgraph_values,
)


class MuseGlimmerVisionPatchEmbedder(nn.Module):
    """Linear patch projection plus learned, interpolated 2D position embeddings."""

    def __init__(self, pixel_dim: int, hidden_size: int, position_grid_size: int):
        super().__init__()
        self.patch_embedding = Linear(pixel_dim, hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            [position_grid_size * position_grid_size, hidden_size],
            name="position_embedding_table.weight",
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        return self.patch_embedding(op, pixel_values)


class MuseGlimmerVisionRotaryEmbedding(nn.Module):
    """Muse 2D RoPE with the frequency layout ``[w, h, w, h]``."""

    def __init__(self, head_dim: int, theta: float = 10_000.0, max_grid_size: int = 4096):
        super().__init__()
        spatial_dim = head_dim // 2
        inv_freq = 1.0 / (
            theta ** (np.arange(0, spatial_dim, 2, dtype=np.float32) / spatial_dim)
        )
        positions = np.arange(max_grid_size, dtype=np.float32)
        frequencies = np.outer(positions, inv_freq).astype(np.float32)
        self.freq_table = nn.Parameter(
            list(frequencies.shape),
            data=ir.tensor(frequencies),
        )

    def forward(self, op: OpBuilder, position_ids: ir.Value):
        w_pos = op.Squeeze(op.Gather(position_ids, [0], axis=1), [1])
        h_pos = op.Squeeze(op.Gather(position_ids, [1], axis=1), [1])
        w_freq = op.Gather(self.freq_table, w_pos)
        h_freq = op.Gather(self.freq_table, h_pos)
        frequencies = op.Concat(w_freq, h_freq, w_freq, h_freq, axis=-1)
        return op.Cos(frequencies), op.Sin(frequencies)


class MuseGlimmerVisionAttention(Qwen25VLVisionAttention):
    """Packed bidirectional attention with separate Q/K/V checkpoint weights."""

    def __init__(self, hidden_size: int, num_heads: int):
        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = Linear(hidden_size, hidden_size, bias=True)
        self.proj = Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
        cos: ir.Value,
        sin: ir.Value,
    ):
        seq_len = op.Shape(hidden_states, start=0, end=1)
        head_shape = op.Concat(
            seq_len,
            op.Constant(value_ints=[self.num_heads, self.head_dim]),
            axis=0,
        )
        query = op.Reshape(self.q_proj(op, hidden_states), head_shape)
        key = op.Reshape(self.k_proj(op, hidden_states), head_shape)
        value = op.Reshape(self.v_proj(op, hidden_states), head_shape)

        query = self._apply_rotary(op, query, cos, sin)
        key = self._apply_rotary(op, key, cos, sin)

        from mobius._build_context import ep_capabilities

        if ep_capabilities().supports_packed_multi_head_attention:
            output = self._emit_packed_mha(op, query, key, value, cu_seqlens, seq_len)
        else:
            output = self._emit_standard_attention(op, query, key, value, cu_seqlens, seq_len)
        return self.proj(op, output)


class MuseGlimmerVisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        return self.fc2(op, op.Gelu(self.fc1(op, hidden_states)))


class MuseGlimmerVisionEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
    ):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps=norm_eps)
        self.norm2 = LayerNorm(hidden_size, eps=norm_eps)
        self.attn = MuseGlimmerVisionAttention(hidden_size, num_heads)
        self.mlp = MuseGlimmerVisionMLP(hidden_size, intermediate_size)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
        cos: ir.Value,
        sin: ir.Value,
    ):
        attention_output = self.attn(
            op,
            self.norm1(op, hidden_states),
            cu_seqlens,
            cos,
            sin,
        )
        hidden_states = op.Add(hidden_states, attention_output)
        return op.Add(hidden_states, self.mlp(op, self.norm2(op, hidden_states)))


class MuseGlimmerVisionModel(Qwen25VLVisionModel):
    """Muse Glimmer ViT with window attention and channel-first pixel shuffle."""

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
        merge_size: int,
        position_grid_size: int,
        fullatt_block_indexes: list[int],
        norm_eps: float,
        rope_theta: float = 10_000.0,
    ):
        nn.Module.__init__(self)
        self._fullatt_block_indexes = set(fullatt_block_indexes)
        # Windowing operates on individual patches. The final 2x2 merge is a
        # separate channel-first pixel shuffle after the transformer.
        self._spatial_merge_size = 1
        self._spatial_merge_unit = 1
        self._patch_size = patch_size
        self._hidden_size = hidden_size
        self._vit_merger_window_size = position_grid_size
        self._merge_size = merge_size
        self._position_grid_size = position_grid_size

        pixel_dim = in_channels * temporal_patch_size * patch_size * patch_size
        self.patch_embedder = MuseGlimmerVisionPatchEmbedder(
            pixel_dim,
            hidden_size,
            position_grid_size,
        )
        self.rotary_emb = MuseGlimmerVisionRotaryEmbedding(
            hidden_size // num_heads,
            theta=rope_theta,
        )
        self.ln_pre = LayerNorm(hidden_size, eps=norm_eps)
        self.layers = nn.ModuleList(
            [
                MuseGlimmerVisionEncoderLayer(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    norm_eps,
                )
                for _ in range(depth)
            ]
        )
        self.ln_post = LayerNorm(hidden_size, eps=norm_eps)

    def _interpolate_pos_embed(
        self,
        op: OpBuilder,
        grid_thw: ir.Value,
    ) -> ir.Value:
        """Bilinear interpolation matching grid_sample(align_corners=False, zeros)."""
        side = self._position_grid_size

        t_col = op.Squeeze(op.Slice(grid_thw, [0], [1], [1], [1]), [1])
        h_col = op.Squeeze(op.Slice(grid_thw, [1], [2], [1], [1]), [1])
        w_col = op.Squeeze(op.Slice(grid_thw, [2], [3], [1], [1]), [1])
        patches_per_image = op.Mul(t_col, op.Mul(h_col, w_col))
        max_patches = op.ReduceMax(patches_per_image, keepdims=False)

        body_thw = ir.Value(
            name="body_thw",
            shape=ir.Shape([3]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        body_graph, body_builder = create_body_graph([], [body_thw])
        body_op = body_builder.op

        b_t = body_op.Squeeze(body_op.Gather(body_thw, body_op.Constant(value_int=0)))
        b_h = body_op.Squeeze(body_op.Gather(body_thw, body_op.Constant(value_int=1)))
        b_w = body_op.Squeeze(body_op.Gather(body_thw, body_op.Constant(value_int=2)))
        h_range = body_op.Cast(
            body_op.Range(0, b_h, 1),
            to=ir.DataType.FLOAT,
        )
        w_range = body_op.Cast(
            body_op.Range(0, b_w, 1),
            to=ir.DataType.FLOAT,
        )
        h_grid = body_op.Sub(
            body_op.Mul(
                body_op.Add(h_range, 0.5),
                body_op.Div(float(side), body_op.Cast(b_h, to=ir.DataType.FLOAT)),
            ),
            0.5,
        )
        w_grid = body_op.Sub(
            body_op.Mul(
                body_op.Add(w_range, 0.5),
                body_op.Div(float(side), body_op.Cast(b_w, to=ir.DataType.FLOAT)),
            ),
            0.5,
        )

        h_floor_raw = body_op.Cast(body_op.Floor(h_grid), to=ir.DataType.INT64)
        w_floor_raw = body_op.Cast(body_op.Floor(w_grid), to=ir.DataType.INT64)
        h_ceil_raw = body_op.Add(h_floor_raw, 1)
        w_ceil_raw = body_op.Add(w_floor_raw, 1)
        lower = body_op.Constant(value_int=0)
        upper = body_op.Constant(value_int=side - 1)
        h_floor = body_op.Clip(h_floor_raw, lower, upper)
        w_floor = body_op.Clip(w_floor_raw, lower, upper)
        h_ceil = body_op.Clip(h_ceil_raw, lower, upper)
        w_ceil = body_op.Clip(w_ceil_raw, lower, upper)

        h_frac = body_op.Sub(h_grid, body_op.Cast(h_floor_raw, to=ir.DataType.FLOAT))
        w_frac = body_op.Sub(w_grid, body_op.Cast(w_floor_raw, to=ir.DataType.FLOAT))
        h_floor_valid = body_op.And(
            body_op.GreaterOrEqual(h_floor_raw, lower),
            body_op.LessOrEqual(h_floor_raw, upper),
        )
        h_ceil_valid = body_op.And(
            body_op.GreaterOrEqual(h_ceil_raw, lower),
            body_op.LessOrEqual(h_ceil_raw, upper),
        )
        w_floor_valid = body_op.And(
            body_op.GreaterOrEqual(w_floor_raw, lower),
            body_op.LessOrEqual(w_floor_raw, upper),
        )
        w_ceil_valid = body_op.And(
            body_op.GreaterOrEqual(w_ceil_raw, lower),
            body_op.LessOrEqual(w_ceil_raw, upper),
        )

        def corner(
            h_index: ir.Value,
            w_index: ir.Value,
            h_weight: ir.Value,
            w_weight: ir.Value,
            h_valid: ir.Value,
            w_valid: ir.Value,
        ) -> ir.Value:
            indices = body_op.Reshape(
                body_op.Add(
                    body_op.Unsqueeze(body_op.Mul(h_index, side), [1]),
                    body_op.Unsqueeze(w_index, [0]),
                ),
                [-1],
            )
            weights = body_op.Mul(
                body_op.Mul(
                    body_op.Unsqueeze(h_weight, [1]),
                    body_op.Unsqueeze(w_weight, [0]),
                ),
                body_op.Cast(
                    body_op.And(
                        body_op.Unsqueeze(h_valid, [1]),
                        body_op.Unsqueeze(w_valid, [0]),
                    ),
                    to=ir.DataType.FLOAT,
                ),
            )
            embeddings = body_op.Cast(
                body_op.Gather(
                    self.patch_embedder.position_embedding_table,
                    indices,
                ),
                to=ir.DataType.FLOAT,
            )
            return body_op.Mul(embeddings, body_op.Reshape(weights, [-1, 1]))

        one_minus_h = body_op.Sub(1.0, h_frac)
        one_minus_w = body_op.Sub(1.0, w_frac)
        pos_embeds = body_op.Add(
            body_op.Add(
                corner(
                    h_floor,
                    w_floor,
                    one_minus_h,
                    one_minus_w,
                    h_floor_valid,
                    w_floor_valid,
                ),
                corner(
                    h_floor,
                    w_ceil,
                    one_minus_h,
                    w_frac,
                    h_floor_valid,
                    w_ceil_valid,
                ),
            ),
            body_op.Add(
                corner(
                    h_ceil,
                    w_floor,
                    h_frac,
                    one_minus_w,
                    h_ceil_valid,
                    w_floor_valid,
                ),
                corner(
                    h_ceil,
                    w_ceil,
                    h_frac,
                    w_frac,
                    h_ceil_valid,
                    w_ceil_valid,
                ),
            ),
        )
        pos_embeds = body_op.Tile(
            pos_embeds,
            body_op.Concat(body_op.Reshape(b_t, [1]), [1], axis=0),
        )

        num_patches = body_op.Mul(b_t, body_op.Mul(b_h, b_w))
        pads = body_op.Concat(
            [0, 0],
            body_op.Reshape(body_op.Sub(max_patches, num_patches), [1]),
            [0],
            axis=0,
        )
        padded = body_op.Pad(pos_embeds, pads, 0.0)
        padded.name = "padded_pos_embeds"
        body_graph.outputs.append(padded)
        rename_subgraph_values(body_graph, "muse_pos_body_")

        scanned = op.Scan(
            grid_thw,
            body=body_graph,
            num_scan_inputs=1,
            _outputs=1,
        )
        return compact_scan_output(op, scanned, patches_per_image)

    def _compute_pixel_shuffle_index(
        self,
        op: OpBuilder,
        grid_thw: ir.Value,
    ) -> ir.Value:
        """Create the per-image, per-frame 2x2 grouping permutation."""
        merge = self._merge_size
        t_col = op.Squeeze(op.Slice(grid_thw, [0], [1], [1], [1]), [1])
        h_col = op.Squeeze(op.Slice(grid_thw, [1], [2], [1], [1]), [1])
        w_col = op.Squeeze(op.Slice(grid_thw, [2], [3], [1], [1]), [1])
        patches = op.Mul(t_col, op.Mul(h_col, w_col))
        max_patches = op.ReduceMax(patches, keepdims=False)

        body_offset = ir.Value(
            name="body_offset",
            shape=ir.Shape([]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        body_thw = ir.Value(
            name="body_thw",
            shape=ir.Shape([3]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        body_graph, body_builder = create_body_graph(
            [body_offset],
            [body_thw],
            name="muse_pixel_shuffle_body",
        )
        body_op = body_builder.op
        b_t = body_op.Squeeze(body_op.Gather(body_thw, 0))
        b_h = body_op.Squeeze(body_op.Gather(body_thw, 1))
        b_w = body_op.Squeeze(body_op.Gather(body_thw, 2))
        count = body_op.Mul(b_t, body_op.Mul(b_h, b_w))
        indices = body_op.Range(0, count, 1)
        shape = body_op.Concat(
            body_op.Reshape(b_t, [1]),
            body_op.Reshape(body_op.Div(b_h, merge), [1]),
            [merge],
            body_op.Reshape(body_op.Div(b_w, merge), [1]),
            [merge],
            axis=0,
        )
        indices = body_op.Reshape(indices, shape)
        indices = body_op.Reshape(
            body_op.Transpose(indices, perm=[0, 1, 3, 2, 4]),
            [-1],
        )
        indices = body_op.Add(indices, body_offset)
        pads = body_op.Concat(
            [0],
            body_op.Reshape(body_op.Sub(max_patches, count), [1]),
            axis=0,
        )
        padded = body_op.Pad(indices, pads, -1)
        new_offset = body_op.Add(body_offset, count)
        new_offset.name = "new_offset"
        padded.name = "padded_indices"
        body_graph.outputs.extend([new_offset, padded])
        rename_subgraph_values(body_graph, "muse_shuffle_body_")

        _, scanned = op.Scan(
            op.Constant(value_int=0),
            grid_thw,
            body=body_graph,
            num_scan_inputs=1,
            _outputs=2,
        )
        return compact_scan_output(op, scanned, patches)

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ) -> ir.Value:
        hidden_states = self.patch_embedder(op, pixel_values)
        pos_embeds = op.CastLike(
            self._interpolate_pos_embed(op, image_grid_thw),
            hidden_states,
        )
        hidden_states = self.ln_pre(op, op.Add(hidden_states, pos_embeds))

        window_index, cu_window_seqlens = self._compute_window_index(
            op,
            image_grid_thw,
        )
        cu_seqlens = self._compute_cu_seqlens(op, image_grid_thw)
        hidden_states = op.Gather(hidden_states, window_index)

        position_ids = self._compute_rotary_pos_ids(op, image_grid_thw)
        position_ids = op.Add(op.Gather(position_ids, [1, 0], axis=1), 1)
        position_ids = op.Gather(position_ids, window_index)
        cos, sin = self.rotary_emb(op, position_ids)

        for layer_idx, layer in enumerate(self.layers):
            layer_cu_seqlens = (
                cu_seqlens if layer_idx in self._fullatt_block_indexes else cu_window_seqlens
            )
            hidden_states = layer(
                op,
                hidden_states,
                layer_cu_seqlens,
                cos,
                sin,
            )

        k = op.Shape(window_index, start=0, end=1)
        _, reverse_index = op.TopK(
            op.Cast(window_index, to=ir.DataType.FLOAT),
            k,
            largest=0,
            sorted=1,
            _outputs=2,
        )
        hidden_states = self.ln_post(
            op,
            op.Gather(hidden_states, reverse_index),
        )

        shuffle_index = self._compute_pixel_shuffle_index(op, image_grid_thw)
        hidden_states = op.Gather(hidden_states, shuffle_index)
        hidden_states = op.Reshape(
            hidden_states,
            [-1, self._merge_size * self._merge_size, self._hidden_size],
        )
        # Muse concatenates the four spatial positions inside each channel.
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        return op.Reshape(
            hidden_states,
            [-1, self._hidden_size * self._merge_size * self._merge_size],
        )
