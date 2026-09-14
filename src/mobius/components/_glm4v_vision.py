# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GLM4V packed vision encoder and gated projector."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._common import LayerNorm, Linear
from mobius.components._conv import Conv2d
from mobius.components._glm_ocr_vision import (
    GlmOcrVisionModel,
    GlmOcrVisionPatchEmbed,
    GlmOcrVisionRotaryEmbedding,
)
from mobius.components._mlp import GatedMLP
from mobius.components._qwen25_vl_vision import Qwen25VLVisionAttention
from mobius.components._rms_norm import RMSNorm
from mobius.components._scan_utils import (
    compact_scan_output,
    create_body_graph,
    rename_subgraph_values,
)

if TYPE_CHECKING:
    pass


class Glm4VVisionAttention(Qwen25VLVisionAttention):
    """GLM4V packed attention with bias-free QKV and output projections."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = Linear(hidden_size, hidden_size, bias=False)


class Glm4VVisionBlock(nn.Module):
    """RMS-pre-norm GLM4V vision transformer block."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Glm4VVisionAttention(hidden_size, num_heads)
        self.mlp = GatedMLP(
            hidden_size,
            intermediate_size,
            activation=hidden_act,
            bias=False,
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


class Glm4VVisionPatchMerger(nn.Module):
    """GLM4V post-downsample projection, norm, GELU, and gated MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.proj = Linear(hidden_size, hidden_size, bias=False)
        self.post_projection_norm = LayerNorm(hidden_size, eps=1e-5)
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self._hidden_act = hidden_act

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = self.proj(op, hidden_states)
        hidden_states = op.Gelu(
            self.post_projection_norm(op, hidden_states),
            approximate="none",
        )
        gate = self.gate_proj(op, hidden_states)
        gate = (
            op.Mul(gate, op.Sigmoid(gate))
            if self._hidden_act == "silu"
            else op.Gelu(gate, approximate="none")
        )
        return self.down_proj(op, op.Mul(gate, self.up_proj(op, hidden_states)))


class Glm4VVisionModel(GlmOcrVisionModel):
    """Dynamic-resolution GLM4V ViT, 2x2 downsampler, and gated projector."""

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
        hidden_act: str = "silu",
        num_position_embeddings: int | None = None,
        projector_intermediate_size: int | None = None,
    ) -> None:
        nn.Module.__init__(self)
        if spatial_merge_size != 2:
            raise ValueError("GLM4V requires a 2x2 spatial merge")
        if hidden_size % num_heads:
            raise ValueError("GLM4V hidden size must be divisible by its head count")
        self._spatial_merge_size = spatial_merge_size
        self._spatial_merge_unit = spatial_merge_size**2
        self._hidden_size = hidden_size
        self._out_hidden_size = out_hidden_size

        self.patch_embed: Any = GlmOcrVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
        )
        self.post_conv_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        self.position_embeddings: nn.Parameter | None
        if num_position_embeddings is None:
            self.position_embeddings = None
            self._position_grid_size = None
        else:
            position_grid = math.isqrt(num_position_embeddings)
            if position_grid * position_grid != num_position_embeddings:
                raise ValueError("GLM4V learned position rows must form a square grid")
            self.position_embeddings = nn.Parameter([num_position_embeddings, hidden_size])
            self._position_grid_size = position_grid

        head_dim = hidden_size // num_heads
        self.rotary_pos_emb: Any = GlmOcrVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                Glm4VVisionBlock(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    norm_eps,
                    hidden_act,
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
        self.merger: Any = Glm4VVisionPatchMerger(
            out_hidden_size,
            projector_intermediate_size or intermediate_size,
            hidden_act=hidden_act,
        )

    def _guard_grid_contract(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        grid_thw: ir.Value,
    ) -> ir.Value:
        temporal = op.Slice(grid_thw, [0], [1], [1], [1])
        height = op.Slice(grid_thw, [1], [2], [1], [1])
        width = op.Slice(grid_thw, [2], [3], [1], [1])
        merge = op.Constant(value_int=self._spatial_merge_size)
        invalid = op.Or(
            op.LessOrEqual(grid_thw, op.Constant(value_int=0)),
            op.Concat(
                op.Equal(temporal, op.Constant(value_int=-1)),
                op.Not(op.Equal(op.Mod(height, merge), op.Constant(value_int=0))),
                op.Not(op.Equal(op.Mod(width, merge), op.Constant(value_int=0))),
                axis=1,
            ),
        )
        invalid_count = op.ReduceSum(op.Cast(invalid, to=ir.DataType.INT64), keepdims=False)
        expected_patches = op.ReduceSum(
            op.Mul(temporal, op.Mul(height, width)),
            keepdims=False,
        )
        actual_patches = op.Squeeze(op.Shape(pixel_values, start=0, end=1), [0])
        invalid_count = op.Add(
            invalid_count,
            op.Cast(op.Not(op.Equal(expected_patches, actual_patches)), to=ir.DataType.INT64),
        )
        guard = op.Gather(
            op.Constant(value_ints=[0]),
            invalid_count,
            axis=0,
        )
        return op.Add(pixel_values, op.CastLike(guard, pixel_values))

    def _interpolate_position_embeddings(
        self,
        op: OpBuilder,
        image_grid_thw: ir.Value,
    ) -> ir.Value | None:
        if self.position_embeddings is None or self._position_grid_size is None:
            return None

        hidden_size = self._hidden_size
        merge = self._spatial_merge_size
        source_grid = self._position_grid_size
        temporal = op.Squeeze(
            op.Slice(image_grid_thw, [0], [1], [1], [1]),
            [1],
        )
        height = op.Squeeze(
            op.Slice(image_grid_thw, [1], [2], [1], [1]),
            [1],
        )
        width = op.Squeeze(
            op.Slice(image_grid_thw, [2], [3], [1], [1]),
            [1],
        )
        patch_counts = op.Mul(temporal, op.Mul(height, width))
        max_patches = op.ReduceMax(patch_counts, keepdims=False)

        body_thw = ir.Value(
            name="body_thw",
            shape=ir.Shape([3]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        body_graph, body_builder = create_body_graph([], [body_thw])
        body_op = body_builder.op
        body_t = body_op.Squeeze(body_op.Gather(body_thw, 0))
        body_h = body_op.Squeeze(body_op.Gather(body_thw, 1))
        body_w = body_op.Squeeze(body_op.Gather(body_thw, 2))

        position_grid = body_op.Reshape(
            self.position_embeddings,
            [1, source_grid, source_grid, hidden_size],
        )
        position_grid = body_op.Transpose(position_grid, perm=[0, 3, 1, 2])
        resize_shape = body_op.Concat(
            body_op.Constant(value_ints=[1, hidden_size]),
            body_op.Reshape(body_h, [1]),
            body_op.Reshape(body_w, [1]),
            axis=0,
        )
        resized = body_op.Resize(
            position_grid,
            None,
            None,
            resize_shape,
            mode="cubic",
            coordinate_transformation_mode="half_pixel",
            cubic_coeff_a=-0.75,
        )
        resized = body_op.Reshape(
            body_op.Transpose(resized, perm=[0, 2, 3, 1]),
            [-1, hidden_size],
        )
        resized = body_op.Tile(
            resized,
            body_op.Concat(
                body_op.Reshape(body_t, [1]),
                body_op.Constant(value_ints=[1]),
                axis=0,
            ),
        )

        body_hm = body_op.Div(body_h, body_op.Constant(value_int=merge))
        body_wm = body_op.Div(body_w, body_op.Constant(value_int=merge))
        merged_shape = body_op.Concat(
            body_op.Reshape(body_t, [1]),
            body_op.Reshape(body_hm, [1]),
            body_op.Constant(value_ints=[merge]),
            body_op.Reshape(body_wm, [1]),
            body_op.Constant(value_ints=[merge, hidden_size]),
            axis=0,
        )
        resized = body_op.Reshape(resized, merged_shape)
        resized = body_op.Transpose(resized, perm=[0, 1, 3, 2, 4, 5])
        resized = body_op.Reshape(resized, [-1, hidden_size])

        body_count = body_op.Mul(body_t, body_op.Mul(body_h, body_w))
        pad_length = body_op.Reshape(body_op.Sub(max_patches, body_count), [1])
        pads = body_op.Concat(
            body_op.Constant(value_ints=[0, 0]),
            pad_length,
            body_op.Constant(value_ints=[0]),
            axis=0,
        )
        padded = body_op.Pad(resized, pads, 0.0)
        padded.name = "padded_glm4v_positions"
        body_graph.outputs.append(padded)
        rename_subgraph_values(body_graph, "glm4v_position_body_")

        scanned = op.Scan(
            image_grid_thw,
            body=body_graph,
            num_scan_inputs=1,
            _outputs=1,
        )
        return compact_scan_output(op, scanned, patch_counts)

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ) -> ir.Value:
        pixel_values = self._guard_grid_contract(op, pixel_values, image_grid_thw)
        hidden_states = self.patch_embed(op, pixel_values)
        hidden_states = self.post_conv_layernorm(op, hidden_states)

        position_ids = self._compute_rotary_pos_ids(op, image_grid_thw)
        cu_seqlens = self._compute_cu_seqlens(op, image_grid_thw)
        cos, sin = self.rotary_pos_emb(op, position_ids)
        learned_positions = self._interpolate_position_embeddings(op, image_grid_thw)
        if learned_positions is not None:
            hidden_states = op.Add(
                hidden_states,
                op.CastLike(learned_positions, hidden_states),
            )

        for block in self.blocks:
            hidden_states = block(
                op,
                hidden_states,
                cu_seqlens=cu_seqlens,
                cos=cos,
                sin=sin,
            )
        hidden_states = self.post_layernorm(op, hidden_states)

        merge = self._spatial_merge_size
        hidden_states = op.Reshape(
            hidden_states,
            [-1, merge, merge, self._hidden_size],
        )
        hidden_states = op.Transpose(hidden_states, perm=[0, 3, 1, 2])
        hidden_states = self.downsample(op, hidden_states)
        hidden_states = op.Reshape(hidden_states, [-1, self._out_hidden_size])
        return self.merger(op, hidden_states)
