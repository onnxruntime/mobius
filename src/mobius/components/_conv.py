# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Convolution building blocks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import onnx_ir as ir
from onnxscript import OpBuilder, nn

if TYPE_CHECKING:
    pass


def _pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value!r}")
    return int(value[0]), int(value[1])


class Conv2d(nn.Module):
    """2D convolution with bias.

    Matches ``torch.nn.Conv2d`` with ``bias=True``.  The default ``padding=0``
    follows PyTorch convention; callers should specify padding explicitly.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
    ):
        super().__init__()
        kernel_h, kernel_w = _pair(kernel_size, "kernel_size")
        self.weight = nn.Parameter((out_channels, in_channels // groups, kernel_h, kernel_w))
        self.bias = nn.Parameter((out_channels,))
        self._kernel_size = (kernel_h, kernel_w)
        self._stride = stride
        self._strides = _pair(stride, "stride")
        self._padding = padding
        self._pads = _pair(padding, "padding")
        self._groups = groups

    def forward(self, op: OpBuilder, x: ir.Value):
        pad_h, pad_w = self._pads
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=list(self._kernel_size),
            strides=list(self._strides),
            pads=[pad_h, pad_w, pad_h, pad_w],
            group=self._groups,
        )


class Conv2dNoBias(nn.Module):
    """2D convolution without bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
    ):
        super().__init__()
        kernel_h, kernel_w = _pair(kernel_size, "kernel_size")
        self.weight = nn.Parameter((out_channels, in_channels // groups, kernel_h, kernel_w))
        self._kernel_size = (kernel_h, kernel_w)
        self._stride = stride
        self._strides = _pair(stride, "stride")
        self._padding = padding
        self._pads = _pair(padding, "padding")
        self._groups = groups

    def forward(self, op: OpBuilder, x: ir.Value):
        pad_h, pad_w = self._pads
        return op.Conv(
            x,
            self.weight,
            kernel_shape=list(self._kernel_size),
            strides=list(self._strides),
            pads=[pad_h, pad_w, pad_h, pad_w],
            group=self._groups,
        )


class BatchNorm2d(nn.Module):
    """2D batch normalization.

    Matches ``torch.nn.BatchNorm2d``.  Uses ONNX ``BatchNormalization`` op
    with frozen running statistics (inference mode).
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter((num_features,))
        self.bias = nn.Parameter((num_features,))
        self.running_mean = nn.Parameter((num_features,))
        self.running_var = nn.Parameter((num_features,))
        self._eps = eps

    def forward(self, op: OpBuilder, x: ir.Value):
        return op.BatchNormalization(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            epsilon=self._eps,
        )


class ConvTranspose2d(nn.Module):
    """2D transposed convolution (deconvolution) with bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.weight = nn.Parameter((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter((out_channels,))
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

    def forward(self, op: OpBuilder, x: ir.Value):
        p = self._padding
        return op.ConvTranspose(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self._kernel_size, self._kernel_size],
            strides=[self._stride, self._stride],
            pads=[p, p, p, p],
        )


class CausalDepthwiseConv1d(nn.Module):
    """Causal depthwise 1-D convolution (left-pad only, no bias).

    Left-pads by ``kernel_size - 1`` so that each output frame depends only
    on the current and past input frames (causal). The padding is folded into
    the ONNX Conv ``pads`` attribute (rather than a separate Pad node) so that
    static shape inference can propagate through the layer.

    Weight layout: ``[channels, 1, kernel_size]`` (depthwise: ``groups=channels``).

    Args:
        channels: Number of input (and output) channels.
        kernel_size: Convolution kernel size along the time axis.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.weight = nn.Parameter([channels, 1, kernel_size])
        self._channels = channels
        self._kernel_size = kernel_size

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: [B, C, T]
        left_pad = self._kernel_size - 1
        return op.Conv(
            x,
            self.weight,
            kernel_shape=[self._kernel_size],
            strides=[1],
            pads=[left_pad, 0],  # [begin, end] on T — causal (past only)
            group=self._channels,
        )
