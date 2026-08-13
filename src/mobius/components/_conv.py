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


def _resolve_pads(padding: int | tuple[int, int, int, int]) -> list[int]:
    """Normalise a ``padding`` argument to the ONNX ``pads`` attribute layout.

    An ``int`` pads all four edges equally (PyTorch semantics).  A 4-tuple is
    taken verbatim as ONNX's ``[top, left, bottom, right]``, which is what
    TensorFlow-style ``SAME`` padding needs when the total pad is odd (the
    extra pixel goes on the end).  See
    :func:`mobius.components._mobilenetv5._same_padding`.

    A PyTorch-style ``(h, w)`` 2-tuple is rejected rather than accepted and
    reinterpreted: the sequence form here is ONNX order, so silently padding
    it out would put the values on the wrong edges and only show up as a
    numerical mismatch much later.

    Raises:
        ValueError: If ``padding`` is a sequence of any length other than 4.
        TypeError: If ``padding`` is neither an int nor a sequence of ints.
    """
    if isinstance(padding, int):
        return [padding, padding, padding, padding]
    try:
        pads = [int(p) for p in padding]
    except TypeError as exc:
        raise TypeError(
            "padding must be an int or a sequence of 4 ints in ONNX order "
            f"[top, left, bottom, right], got {padding!r}."
        ) from exc
    if len(pads) != 4:
        raise ValueError(
            "A sequence padding must have exactly 4 elements in ONNX order "
            f"[top, left, bottom, right], got {len(pads)}: {padding!r}. "
            "PyTorch-style (h, w) is not accepted -- pass an int for uniform "
            "padding, or spell out all four edges."
        )
    return pads


class Conv2d(nn.Module):
    """2D convolution with bias.

    Matches ``torch.nn.Conv2d`` with ``bias=True``.  The default ``padding=0``
    follows PyTorch convention; callers should specify padding explicitly.
    ``padding`` also accepts an ONNX-order ``[top, left, bottom, right]``
    4-tuple for asymmetric (``SAME``-style) padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int, int, int] = 0,
        groups: int = 1,
    ):
        super().__init__()
        kernel_h, kernel_w = _pair(kernel_size, "kernel_size")
        self.weight = nn.Parameter((out_channels, in_channels // groups, kernel_h, kernel_w))
        self.bias = nn.Parameter((out_channels,))
        self._kernel_size = (kernel_h, kernel_w)
        self._stride = stride
        self._strides = _pair(stride, "stride")
        self._pads = _resolve_pads(padding)
        self._groups = groups

    def forward(self, op: OpBuilder, x: ir.Value):
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=list(self._kernel_size),
            strides=list(self._strides),
            pads=self._pads,
            group=self._groups,
        )


class Conv2dNoBias(nn.Module):
    """2D convolution without bias.

    ``padding`` accepts an ``int`` or an ONNX-order
    ``[top, left, bottom, right]`` 4-tuple, as for :class:`Conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int, int, int] = 0,
        groups: int = 1,
    ):
        super().__init__()
        kernel_h, kernel_w = _pair(kernel_size, "kernel_size")
        self.weight = nn.Parameter((out_channels, in_channels // groups, kernel_h, kernel_w))
        self._kernel_size = (kernel_h, kernel_w)
        self._stride = stride
        self._strides = _pair(stride, "stride")
        self._pads = _resolve_pads(padding)
        self._groups = groups

    def forward(self, op: OpBuilder, x: ir.Value):
        return op.Conv(
            x,
            self.weight,
            kernel_shape=list(self._kernel_size),
            strides=list(self._strides),
            pads=self._pads,
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


class RmsNorm2d(nn.Module):
    """Channel-axis RMS normalization for NCHW tensors, scale-only.

    Matches timm's ``RmsNorm2d``: reduces over the channel axis (``dim=1``)
    of an NCHW tensor and applies a learnable per-channel scale.  There is
    no bias and no running statistics.

    Used by the MobileNet-V5 vision tower (Gemma 3n), where timm names these
    modules ``.bn`` for backwards weight compatibility even though they are
    RMSNorm, not BatchNorm.  Do not substitute :class:`BatchNorm2d`: the
    checkpoint ships only ``weight``, so the four extra initializers
    ``BatchNormalization`` requires (bias, running mean/var) do not exist.

    ``op.RMSNormalization`` normalizes over the *last* axis, so this reduces
    manually over axis 1 rather than transposing to NHWC and back.

    Args:
        num_features: Number of channels (the ``C`` of NCHW).
        eps: Added to the mean square before the reciprocal square root.
    """

    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter((num_features,))
        self._eps = eps

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # Accumulate the mean square in float32: squaring activations > 256
        # overflows float16 (max 65504), and this norm runs on raw conv output.
        x_f32 = op.Cast(x, to=ir.DataType.FLOAT)
        mean_sq = op.ReduceMean(op.Mul(x_f32, x_f32), [1], keepdims=1)
        normed = op.Mul(x_f32, op.Reciprocal(op.Sqrt(op.Add(mean_sq, self._eps))))
        # Broadcast the per-channel scale over NCHW: [C] -> [1, C, 1, 1].
        scale = op.Reshape(op.Cast(self.weight, to=ir.DataType.FLOAT), [1, -1, 1, 1])
        return op.CastLike(op.Mul(normed, scale), x)


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
