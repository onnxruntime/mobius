# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable projector heads used by architecture-specific GGUF VLM sidecars."""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius.components._common import LayerNorm, Linear

if TYPE_CHECKING:
    import onnx_ir as ir


class _GELU(nn.Module):
    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return op.Gelu(hidden_states)


class SpatialPixelUnshuffle(nn.Module):
    """Merge adjacent image patches while preserving the upstream raster order.

    Idefics3, InternVL, and Llama4 use the same two-permutation pixel-unshuffle
    topology. A ``scale_factor`` by ``scale_factor`` patch group becomes one
    token whose channel width is multiplied by ``scale_factor**2``.
    """

    def __init__(self, grid_height: int, grid_width: int, scale_factor: int):
        super().__init__()
        if grid_height <= 0 or grid_width <= 0 or scale_factor <= 1:
            raise ValueError("Pixel-unshuffle grid dimensions and scale factor must be positive")
        if grid_height % scale_factor or grid_width % scale_factor:
            raise ValueError(
                f"{grid_height}x{grid_width} patch grid is not divisible by "
                f"scale factor {scale_factor}"
            )
        self._grid_height = grid_height
        self._grid_width = grid_width
        self._scale_factor = scale_factor

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # [B, H*W, C] -> [B, H, W, C].
        batch = op.Shape(hidden_states, start=0, end=1)
        channels = op.Shape(hidden_states, start=2, end=3)
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(batch, [self._grid_height, self._grid_width], channels, axis=0),
        )

        scale = self._scale_factor
        # Match HF pixel_shuffle: fold width into channels, swap H/W, then fold
        # the remaining spatial factor and swap back to raster order.
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                batch,
                [self._grid_height, self._grid_width // scale],
                op.Mul(channels, scale),
                axis=0,
            ),
        )
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1, 3])
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                batch,
                [self._grid_width // scale, self._grid_height // scale],
                op.Mul(channels, scale * scale),
                axis=0,
            ),
        )
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1, 3])
        return op.Reshape(
            hidden_states,
            op.Concat(
                batch,
                [self._grid_height * self._grid_width // (scale * scale)],
                op.Mul(channels, scale * scale),
                axis=0,
            ),
        )


class Idefics3Projector(nn.Module):
    """Idefics3 ``pixel_shuffle -> bias-free Linear`` projector."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        *,
        grid_size: int,
        scale_factor: int,
    ):
        super().__init__()
        self.pixel_shuffle = SpatialPixelUnshuffle(grid_size, grid_size, scale_factor)
        self.model_fc = Linear(
            vision_hidden_size * scale_factor * scale_factor,
            text_hidden_size,
            bias=False,
        )

    def forward(self, op: OpBuilder, vision_features: ir.Value) -> ir.Value:
        return self.model_fc(op, self.pixel_shuffle(op, vision_features))


class InternVLProjector(nn.Module):
    """InternVL ``pixel_shuffle -> LayerNorm -> Linear -> GELU -> Linear`` head."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        *,
        grid_size: int,
        scale_factor: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        merged_hidden = vision_hidden_size * scale_factor * scale_factor
        self.pixel_shuffle = SpatialPixelUnshuffle(grid_size, grid_size, scale_factor)
        self.mlp = nn.Sequential(
            LayerNorm(merged_hidden, eps=eps),
            Linear(merged_hidden, text_hidden_size, bias=True),
            _GELU(),
            Linear(text_hidden_size, text_hidden_size, bias=True),
        )

    def forward(self, op: OpBuilder, vision_features: ir.Value) -> ir.Value:
        return self.mlp(op, self.pixel_shuffle(op, vision_features))


class Llama4Projector(nn.Module):
    """Llama4 pixel-shuffle adapter and final multimodal projection."""

    def __init__(
        self,
        vision_hidden_size: int,
        adapter_hidden_size: int,
        text_hidden_size: int,
        *,
        grid_size: int,
        scale_factor: int,
    ):
        super().__init__()
        merged_hidden = vision_hidden_size * scale_factor * scale_factor
        self.pixel_shuffle = SpatialPixelUnshuffle(grid_size, grid_size, scale_factor)
        self.model_mlp_1 = Linear(merged_hidden, adapter_hidden_size, bias=False)
        self.model_mlp_2 = Linear(adapter_hidden_size, adapter_hidden_size, bias=False)
        self.model_fc = Linear(adapter_hidden_size, text_hidden_size, bias=False)

    def forward(self, op: OpBuilder, vision_features: ir.Value) -> ir.Value:
        hidden_states = self.pixel_shuffle(op, vision_features)
        hidden_states = op.Gelu(self.model_mlp_1(op, hidden_states))
        hidden_states = op.Gelu(self.model_mlp_2(op, hidden_states))
        return self.model_fc(op, hidden_states)


class PixtralProjector(nn.Module):
    """Original Pixtral MLP plus row-separating ``[IMG_BREAK]`` embeddings."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        *,
        with_image_break: bool,
    ):
        super().__init__()
        self.linear_1 = Linear(vision_hidden_size, text_hidden_size, bias=True)
        self.linear_2 = Linear(text_hidden_size, text_hidden_size, bias=True)
        self.image_break = nn.Parameter([text_hidden_size]) if with_image_break else None
        self._text_hidden_size = text_hidden_size

    def forward(
        self,
        op: OpBuilder,
        vision_features: ir.Value,
        grid_height: ir.Value,
        grid_width: ir.Value,
    ) -> ir.Value:
        # The GGUF Pixtral sidecar processes one image per invocation.
        hidden_states = self.linear_2(op, op.Gelu(self.linear_1(op, vision_features)))
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                op.Reshape(grid_height, [1]),
                op.Reshape(grid_width, [1]),
                [self._text_hidden_size],
                axis=0,
            ),
        )
        if self.image_break is None:
            return op.Reshape(hidden_states, [-1, self._text_hidden_size])

        # Append a break to every row, flatten, then remove the final row's break.
        breaks = op.Expand(
            op.Reshape(self.image_break, [1, 1, self._text_hidden_size]),
            op.Concat(
                op.Reshape(grid_height, [1]),
                [1, self._text_hidden_size],
                axis=0,
            ),
        )
        hidden_states = op.Concat(hidden_states, breaks, axis=1)
        hidden_states = op.Reshape(hidden_states, [-1, self._text_hidden_size])
        output_rows = op.Sub(
            op.Mul(grid_height, op.Add(grid_width, op.CastLike(1, grid_width))),
            op.CastLike(1, grid_height),
        )
        return op.Slice(
            hidden_states,
            op.Constant(value_ints=[0]),
            op.Reshape(output_rows, [1]),
            op.Constant(value_ints=[0]),
        )
