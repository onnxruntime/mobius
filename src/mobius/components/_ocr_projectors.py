# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable projector blocks used by document and OCR multimodal encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius.components._common import LayerNorm, Linear
from mobius.components._pixtral_vision import Mistral3PatchMerger
from mobius.components._rms_norm import RMSNorm

if TYPE_CHECKING:
    import onnx_ir as ir


def _merge_consecutive_patches(
    op: OpBuilder,
    hidden_states: ir.Value,
    *,
    hidden_size: int,
    merge_size: int,
) -> ir.Value:
    """Flatten already merge-ordered patch groups along the feature axis."""
    merge_unit = merge_size * merge_size
    return op.Reshape(hidden_states, [-1, hidden_size * merge_unit])


def _merge_raster_patches(
    op: OpBuilder,
    hidden_states: ir.Value,
    grid_h: ir.Value,
    grid_w: ir.Value,
    *,
    hidden_size: int,
    merge_size: int,
) -> ir.Value:
    """Group spatially adjacent raster-order patches into feature vectors."""
    merge = op.Constant(value_int=merge_size)
    grouped_h = op.Div(grid_h, merge)
    grouped_w = op.Div(grid_w, merge)
    shape = op.Concat(
        op.Reshape(grouped_h, [1]),
        [merge_size],
        op.Reshape(grouped_w, [1]),
        [merge_size, hidden_size],
        axis=0,
    )
    hidden_states = op.Reshape(hidden_states, shape)
    # (H/m, m, W/m, m, D) -> (H/m, W/m, m, m, D).
    hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1, 3, 4])
    return op.Reshape(hidden_states, [-1, hidden_size * merge_size * merge_size])


class DeepSeekOCRProjector(nn.Module):
    """Linear fusion of aligned CLIP and SAM patch features."""

    def __init__(self, clip_hidden_size: int, sam_hidden_size: int, output_size: int):
        super().__init__()
        self.linear = Linear(clip_hidden_size + sam_hidden_size, output_size, bias=True)

    def forward(
        self,
        op: OpBuilder,
        clip_features: ir.Value,
        sam_features: ir.Value,
    ) -> ir.Value:
        # Both towers preserve the same spatial patch order: (B, N, D).
        return self.linear(op, op.Concat(clip_features, sam_features, axis=-1))


class DotsOCRProjector(nn.Module):
    """Dots OCR/vision merger: LayerNorm -> 2-D merge -> exact GELU MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        *,
        merge_size: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.input_norm = LayerNorm(hidden_size, eps=eps)
        self.linear_0 = Linear(
            hidden_size * merge_size * merge_size,
            intermediate_size,
            bias=True,
        )
        self.linear_2 = Linear(intermediate_size, output_size, bias=True)
        self._hidden_size = hidden_size
        self._merge_size = merge_size

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # Qwen-style processors serialize each spatial merge group consecutively.
        hidden_states = self.input_norm(op, hidden_states)
        hidden_states = _merge_consecutive_patches(
            op,
            hidden_states,
            hidden_size=self._hidden_size,
            merge_size=self._merge_size,
        )
        hidden_states = op.Gelu(self.linear_0(op, hidden_states))
        return self.linear_2(op, hidden_states)


class PaddleOCRProjector(nn.Module):
    """PaddleOCR raster merger: LayerNorm -> spatial merge -> tanh-GELU MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        *,
        merge_size: int = 2,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.input_norm = LayerNorm(hidden_size, eps=eps)
        self.linear_1 = Linear(
            hidden_size * merge_size * merge_size,
            intermediate_size,
            bias=True,
        )
        self.linear_2 = Linear(intermediate_size, output_size, bias=True)
        self._hidden_size = hidden_size
        self._merge_size = merge_size

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        grid_h: ir.Value,
        grid_w: ir.Value,
    ) -> ir.Value:
        hidden_states = self.input_norm(op, hidden_states)
        hidden_states = _merge_raster_patches(
            op,
            hidden_states,
            grid_h,
            grid_w,
            hidden_size=self._hidden_size,
            merge_size=self._merge_size,
        )
        hidden_states = op.Gelu(
            self.linear_1(op, hidden_states),
            approximate="tanh",
        )
        return self.linear_2(op, hidden_states)


class YouTuVLProjector(nn.Module):
    """YouTu-VL merger: RMSNorm -> 2x2 flatten -> tanh-GELU MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        *,
        merge_size: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.input_norm = RMSNorm(hidden_size, eps=eps)
        self.linear_0 = Linear(
            hidden_size * merge_size * merge_size,
            intermediate_size,
            bias=True,
        )
        self.linear_2 = Linear(intermediate_size, output_size, bias=True)
        self._hidden_size = hidden_size
        self._merge_size = merge_size

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = self.input_norm(op, hidden_states)
        hidden_states = _merge_consecutive_patches(
            op,
            hidden_states,
            hidden_size=self._hidden_size,
            merge_size=self._merge_size,
        )
        hidden_states = op.Gelu(
            self.linear_0(op, hidden_states),
            approximate="tanh",
        )
        return self.linear_2(op, hidden_states)


class LightOnOCRProjector(nn.Module):
    """LightOnOCR Pixtral merger followed by a two-layer GELU projection."""

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        *,
        merge_size: int = 2,
        eps: float = 1e-5,
        first_bias: bool = False,
        second_bias: bool = False,
    ):
        super().__init__()
        self.input_norm = RMSNorm(hidden_size, eps=eps)
        self.patch_merger = Mistral3PatchMerger(hidden_size, merge_size)
        self.linear_1 = Linear(hidden_size, hidden_size, bias=first_bias)
        self.linear_2 = Linear(hidden_size, output_size, bias=second_bias)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        grid_h: ir.Value,
        grid_w: ir.Value,
    ) -> ir.Value:
        hidden_states = self.input_norm(op, hidden_states)
        hidden_states = self.patch_merger(op, hidden_states, grid_h, grid_w)
        hidden_states = self.linear_1(op, hidden_states)
        hidden_states = op.Gelu(hidden_states, approximate="tanh")
        return self.linear_2(op, hidden_states)


class Dots3NoteAudioProjector(nn.Module):
    """Dots3Note audio adapter: LayerNorm -> exact GELU -> two affine layers."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        *,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.norm_pre = LayerNorm(hidden_size, eps=eps)
        self.linear_1 = Linear(hidden_size, intermediate_size, bias=True)
        self.linear_3 = Linear(intermediate_size, output_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = self.norm_pre(op, hidden_states)
        hidden_states = op.Gelu(self.linear_1(op, hidden_states))
        return self.linear_3(op, hidden_states)
