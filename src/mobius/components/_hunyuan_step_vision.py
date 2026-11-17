# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standalone HunyuanVL and Step3VL CLIP sidecar components.

The data flow follows llama.cpp mtmd at revision
8d9af256337d1a501250f9bbf4c0859a654bddd6. HunyuanVL receives the
CPU-resized position table as an input. Step3VL resizes its learned table in
the graph and receives the processor's axial patch coordinates explicitly.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._activations import gelu_tanh, quick_gelu
from mobius.components._common import LayerNorm, Linear
from mobius.components._conv import Conv2d, Conv2dNoBias
from mobius.components._rms_norm import RMSNorm


class _PatchEmbedding(nn.Module):
    def __init__(self, channels: int, hidden_size: int, patch_size: int, *, bias: bool = True):
        super().__init__()
        conv_type = Conv2d if bias else Conv2dNoBias
        self.proj = conv_type(channels, hidden_size, patch_size, patch_size)

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        # BCHW -> BC(hw) -> B(hw)C, matching clip_graph::build_inp.
        pixel_values = op.CastLike(pixel_values, self.proj.weight)
        hidden_states = self.proj(op, pixel_values)
        batch = op.Shape(hidden_states, start=0, end=1)
        channels = op.Shape(hidden_states, start=1, end=2)
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(batch, channels, op.Constant(value_ints=[-1]), axis=0),
        )
        return op.Transpose(hidden_states, perm=[0, 2, 1])


class _VisionAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.in_proj = Linear(hidden_size, hidden_size * 3)
        self.out_proj = Linear(hidden_size, hidden_size)
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads

    def _qkv(self, op: OpBuilder, hidden_states: ir.Value):
        batch = op.Shape(hidden_states, start=0, end=1)
        sequence = op.Shape(hidden_states, start=1, end=2)
        qkv = self.in_proj(op, hidden_states)
        qkv = op.Reshape(
            qkv,
            op.Concat(
                batch,
                sequence,
                op.Constant(value_ints=[3, self._num_heads, self._head_dim]),
                axis=0,
            ),
        )
        qkv = op.Transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = op.Split(qkv, num_outputs=3, axis=0, _outputs=3)
        return op.Squeeze(q, [0]), op.Squeeze(k, [0]), op.Squeeze(v, [0])

    def _finish(self, op: OpBuilder, q: ir.Value, k: ir.Value, v: ir.Value) -> ir.Value:
        scores = op.Mul(
            op.MatMul(q, op.Transpose(k, perm=[0, 1, 3, 2])),
            self._head_dim**-0.5,
        )
        context = op.MatMul(op.Softmax(scores, axis=-1), v)
        context = op.Transpose(context, perm=[0, 2, 1, 3])
        batch = op.Shape(context, start=0, end=1)
        sequence = op.Shape(context, start=1, end=2)
        context = op.Reshape(
            context,
            op.Concat(
                batch,
                sequence,
                op.Constant(value_ints=[self._num_heads * self._head_dim]),
                axis=0,
            ),
        )
        return self.out_proj(op, context)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return self._finish(op, *self._qkv(op, hidden_states))


class _HunyuanVLBlock(nn.Module):
    """Normal-LayerNorm ViT block used by the HunyuanVL sidecar."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, eps: float):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps)
        self.attn = _VisionAttention(hidden_size, num_heads)
        self.norm2 = LayerNorm(hidden_size, eps)
        self.mlp_up = Linear(hidden_size, intermediate_size)
        self.mlp_down = Linear(intermediate_size, hidden_size)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = op.Add(hidden_states, self.attn(op, self.norm1(op, hidden_states)))
        mlp = self.mlp_down(op, gelu_tanh(op, self.mlp_up(op, self.norm2(op, hidden_states))))
        return op.Add(hidden_states, mlp)


class HunyuanVLClipSidecar(nn.Module):
    """HunyuanVL ViT and perceiver projector.

    Inputs:
        pixel_values: Normalized NCHW image tensor.

    Output token order is image-begin, rows of projected patches with one
    newline token after every row, image-end.
    """

    def __init__(
        self,
        *,
        vision_hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_layers: int,
        patch_size: int,
        grid_height: int,
        grid_width: int,
        position_grid_size: int,
        projector_hidden_size: int,
        output_size: int,
        merge_size: int = 2,
        eps: float = 1e-5,
    ):
        super().__init__()
        if grid_height % merge_size or grid_width % merge_size:
            raise ValueError("reference grid dimensions must be divisible by merge_size")
        self.patch_embedding = _PatchEmbedding(3, vision_hidden_size, patch_size)
        self.position_embedding = nn.Parameter(
            [position_grid_size * position_grid_size, vision_hidden_size]
        )
        self.layers = nn.ModuleList(
            [
                _HunyuanVLBlock(vision_hidden_size, intermediate_size, num_heads, eps)
                for _ in range(num_layers)
            ]
        )
        self.pre_projector_norm = RMSNorm(vision_hidden_size, eps)
        self.projector_conv1 = Conv2d(
            vision_hidden_size,
            projector_hidden_size,
            kernel_size=merge_size,
            stride=merge_size,
        )
        self.projector_conv2 = Conv2d(
            projector_hidden_size, projector_hidden_size * 2, kernel_size=1
        )
        self.projector = Linear(projector_hidden_size * 2, output_size)
        self.image_newline = nn.Parameter([projector_hidden_size * 2])
        self.image_begin = nn.Parameter([output_size])
        self.image_end = nn.Parameter([output_size])
        self.post_projector_norm = RMSNorm(output_size, eps)
        self._patch_size = patch_size
        self._merge_size = merge_size
        self._projector_channels = projector_hidden_size * 2
        self._output_size = output_size
        self._position_grid_size = position_grid_size
        self._vision_hidden_size = vision_hidden_size

    def _resize_positions(
        self,
        op: OpBuilder,
        grid_height: ir.Value,
        grid_width: ir.Value,
    ) -> ir.Value:
        # [S*S,C] -> [1,C,S,S] -> bilinear/antialiased [1,C,H,W] -> [1,HW,C].
        positions = op.Reshape(
            self.position_embedding,
            [
                self._position_grid_size,
                self._position_grid_size,
                self._vision_hidden_size,
            ],
        )
        positions = op.Unsqueeze(op.Transpose(positions, perm=[2, 0, 1]), [0])
        positions = op.Resize(
            positions,
            None,
            None,
            op.Concat(
                op.Constant(value_ints=[1, self._vision_hidden_size]),
                op.Reshape(grid_height, [1]),
                op.Reshape(grid_width, [1]),
                axis=0,
            ),
            mode="linear",
            coordinate_transformation_mode="half_pixel",
            antialias=1,
            exclude_outside=1,
        )
        return op.Reshape(
            op.Transpose(positions, perm=[0, 2, 3, 1]), [1, -1, self._vision_hidden_size]
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
    ) -> ir.Value:
        grid_height = op.Div(
            op.Squeeze(op.Shape(pixel_values, start=2, end=3), [0]),
            self._patch_size,
        )
        grid_width = op.Div(
            op.Squeeze(op.Shape(pixel_values, start=3, end=4), [0]),
            self._patch_size,
        )
        hidden_states = self.patch_embedding(op, pixel_values)
        hidden_states = op.Add(
            hidden_states,
            op.CastLike(self._resize_positions(op, grid_height, grid_width), hidden_states),
        )
        for layer in self.layers:
            hidden_states = layer(op, hidden_states)
        hidden_states = self.pre_projector_norm(op, hidden_states)

        # B(HW)C -> BCHW; convolution merges each non-overlapping spatial tile.
        batch = op.Shape(hidden_states, start=0, end=1)
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                batch,
                op.Reshape(grid_height, [1]),
                op.Reshape(grid_width, [1]),
                op.Constant(value_ints=[-1]),
                axis=0,
            ),
        )
        hidden_states = op.Transpose(hidden_states, perm=[0, 3, 1, 2])
        hidden_states = self.projector_conv1(op, hidden_states)
        hidden_states = gelu_tanh(op, hidden_states)
        hidden_states = self.projector_conv2(op, hidden_states)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 3, 1])

        # Append newline along width before flattening: row-major
        # [patch(0,0), ..., patch(0,W-1), newline, patch(1,0), ...].
        out_height = op.Div(grid_height, self._merge_size)
        channels = self._projector_channels
        newline = op.Reshape(self.image_newline, [1, 1, 1, channels])
        newline = op.Expand(
            newline,
            op.Concat(
                batch,
                op.Reshape(out_height, [1]),
                op.Constant(value_ints=[1, channels]),
                axis=0,
            ),
        )
        hidden_states = op.Concat(hidden_states, newline, axis=2)
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(batch, op.Constant(value_ints=[-1, channels]), axis=0),
        )
        hidden_states = self.projector(op, hidden_states)

        begin = op.Expand(
            op.Reshape(self.image_begin, [1, 1, -1]),
            op.Concat(batch, op.Constant(value_ints=[1, self._output_size]), axis=0),
        )
        end = op.Expand(
            op.Reshape(self.image_end, [1, 1, -1]),
            op.Concat(batch, op.Constant(value_ints=[1, self._output_size]), axis=0),
        )
        hidden_states = op.Concat(begin, hidden_states, end, axis=1)
        return self.post_projector_norm(op, hidden_states)


class _Step3VLAttention(_VisionAttention):
    """Step3VL attention with adjacent-pair, two-axis RoPE on Q and K."""

    def __init__(self, hidden_size: int, num_heads: int, rope_theta: float):
        super().__init__(hidden_size, num_heads)
        if self._head_dim % 4:
            raise ValueError("Step3VL head dimension must be divisible by four")
        self._rope_theta = rope_theta

    def _rope_axis(self, op: OpBuilder, x: ir.Value, positions: ir.Value) -> ir.Value:
        pair_count = self._head_dim // 4
        shape = op.Shape(x)
        paired = op.Reshape(
            x,
            op.Concat(
                op.Slice(shape, [0], [3]),
                op.Constant(value_ints=[pair_count, 2]),
                axis=0,
            ),
        )
        even, odd = op.Split(paired, num_outputs=2, axis=-1, _outputs=2)
        even = op.Squeeze(even, [-1])
        odd = op.Squeeze(odd, [-1])
        freq = op.Pow(
            self._rope_theta,
            op.Div(
                op.Cast(op.Range(0, pair_count, 1), to=ir.DataType.FLOAT),
                float(-pair_count),
            ),
        )
        angles = op.Mul(
            op.Cast(op.Reshape(positions, [-1, 1]), to=ir.DataType.FLOAT),
            op.Reshape(freq, [1, -1]),
        )
        cos = op.CastLike(
            op.Reshape(op.Cos(angles), [1, 1, -1, pair_count]),
            even,
        )
        sin = op.CastLike(
            op.Reshape(op.Sin(angles), [1, 1, -1, pair_count]),
            even,
        )
        rot_even = op.Sub(op.Mul(even, cos), op.Mul(odd, sin))
        rot_odd = op.Add(op.Mul(even, sin), op.Mul(odd, cos))
        return op.Reshape(
            op.Concat(op.Unsqueeze(rot_even, [-1]), op.Unsqueeze(rot_odd, [-1]), axis=-1),
            shape,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        pos_h: ir.Value | None = None,
        pos_w: ir.Value | None = None,
    ) -> ir.Value:
        if pos_h is None or pos_w is None:
            raise ValueError("Step3VL attention requires both height and width positions.")
        q, k, v = self._qkv(op, hidden_states)
        q_w, q_h = op.Split(q, num_outputs=2, axis=-1, _outputs=2)
        k_w, k_h = op.Split(k, num_outputs=2, axis=-1, _outputs=2)
        q = op.Concat(
            self._rope_axis(op, q_w, pos_w),
            self._rope_axis(op, q_h, pos_h),
            axis=-1,
        )
        k = op.Concat(
            self._rope_axis(op, k_w, pos_w),
            self._rope_axis(op, k_h, pos_h),
            axis=-1,
        )
        return self._finish(op, q, k, v)


class _Step3VLBlock(nn.Module):
    """Quick-GELU Step3VL block with checkpoint layer-scale vectors."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        eps: float,
        rope_theta: float,
    ):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps)
        self.attn = _Step3VLAttention(hidden_size, num_heads, rope_theta)
        self.ls_1 = nn.Parameter([hidden_size])
        self.norm2 = LayerNorm(hidden_size, eps)
        self.mlp_up = Linear(hidden_size, intermediate_size)
        self.mlp_down = Linear(intermediate_size, hidden_size)
        self.ls_2 = nn.Parameter([hidden_size])

    def forward(
        self, op: OpBuilder, hidden_states: ir.Value, pos_h: ir.Value, pos_w: ir.Value
    ) -> ir.Value:
        attention = self.attn(op, self.norm1(op, hidden_states), pos_h, pos_w)
        hidden_states = op.Add(hidden_states, op.Mul(attention, self.ls_1))
        mlp = self.mlp_down(op, quick_gelu(op, self.mlp_up(op, self.norm2(op, hidden_states))))
        return op.Add(hidden_states, op.Mul(mlp, self.ls_2))


class Step3VLClipSidecar(nn.Module):
    """Step3VL ViT plus two biased spatial downsamplers and biasless projection.

    ``pixel_values`` is the processor-produced normalized NCHW patch canvas.
    ``pos_h`` and ``pos_w`` are the processor's flattened int64 patch
    coordinates, making the non-raster preprocessing contract explicit.
    """

    def __init__(
        self,
        *,
        vision_hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_layers: int,
        patch_size: int,
        grid_height: int,
        grid_width: int,
        position_grid_size: int,
        downsample_hidden_size: int,
        output_size: int,
        eps: float = 1e-5,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.patch_embedding = _PatchEmbedding(
            3,
            vision_hidden_size,
            patch_size,
            bias=False,
        )
        self.position_embedding = nn.Parameter(
            [position_grid_size * position_grid_size, vision_hidden_size]
        )
        self.pre_layer_norm = LayerNorm(vision_hidden_size, eps)
        self.layers = nn.ModuleList(
            [
                _Step3VLBlock(
                    vision_hidden_size,
                    intermediate_size,
                    num_heads,
                    eps,
                    rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.downsample1 = Conv2d(vision_hidden_size, downsample_hidden_size, 3, 2, 1)
        self.downsample2 = Conv2d(downsample_hidden_size, downsample_hidden_size * 2, 3, 2, 1)
        self.projector = Linear(downsample_hidden_size * 2, output_size, bias=False)
        self._grid_height = grid_height
        self._grid_width = grid_width
        self._patch_size = patch_size
        self._position_grid_size = position_grid_size
        self._vision_hidden_size = vision_hidden_size
        self._projector_channels = downsample_hidden_size * 2

    def _resize_positions(
        self,
        op: OpBuilder,
        grid_height: ir.Value,
        grid_width: ir.Value,
    ) -> ir.Value:
        # [S*S,C] -> [1,C,S,S] -> bilinear/antialiased [1,C,H,W] -> [1,HW,C].
        positions = op.Reshape(
            self.position_embedding,
            [
                self._position_grid_size,
                self._position_grid_size,
                self._vision_hidden_size,
            ],
        )
        positions = op.Transpose(positions, perm=[2, 0, 1])
        positions = op.Unsqueeze(positions, [0])
        positions = op.Resize(
            positions,
            None,
            None,
            op.Concat(
                op.Constant(value_ints=[1, self._vision_hidden_size]),
                op.Reshape(grid_height, [1]),
                op.Reshape(grid_width, [1]),
                axis=0,
            ),
            mode="linear",
            coordinate_transformation_mode="half_pixel",
            antialias=1,
            exclude_outside=1,
        )
        positions = op.Transpose(positions, perm=[0, 2, 3, 1])
        return op.Reshape(
            positions,
            op.Concat(
                op.Constant(value_ints=[1]),
                op.Reshape(op.Mul(grid_height, grid_width), [1]),
                op.Constant(value_ints=[self._vision_hidden_size]),
                axis=0,
            ),
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        pos_h: ir.Value,
        pos_w: ir.Value,
    ) -> ir.Value:
        grid_height = op.Div(
            op.Squeeze(op.Shape(pixel_values, start=2, end=3), [0]),
            self._patch_size,
        )
        grid_width = op.Div(
            op.Squeeze(op.Shape(pixel_values, start=3, end=4), [0]),
            self._patch_size,
        )
        hidden_states = self.pre_layer_norm(
            op,
            op.Add(
                self.patch_embedding(op, pixel_values),
                self._resize_positions(op, grid_height, grid_width),
            ),
        )
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, pos_h, pos_w)

        batch = op.Shape(hidden_states, start=0, end=1)
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                batch,
                op.Reshape(grid_height, [1]),
                op.Reshape(grid_width, [1]),
                op.Constant(value_ints=[-1]),
                axis=0,
            ),
        )
        hidden_states = op.Transpose(hidden_states, perm=[0, 3, 1, 2])
        hidden_states = self.downsample1(op, hidden_states)
        hidden_states = self.downsample2(op, hidden_states)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 3, 1])
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                batch,
                op.Constant(value_ints=[-1, self._projector_channels]),
                axis=0,
            ),
        )
        return self.projector(op, hidden_states)
