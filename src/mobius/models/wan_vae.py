# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Wan 3D causal video VAE (``diffusers.AutoencoderKLWan``).

Replicates the public HuggingFace diffusers class ``AutoencoderKLWan``
(``diffusers/models/autoencoders/autoencoder_kl_wan.py``) as a from-scratch
``onnxscript.nn`` graph builder.  This is the video VAE used by Wan 2.1 /
Wan 2.2 and, verbatim, by NVIDIA Cosmos3 (``nvidia/Cosmos3-Nano/vae``).

Pipeline (Cosmos3 / Wan 2.2 shapes, ``patch_size=2``, ``z_dim=48``)::

    video   (B, 3,  4k+1, H,    W)      pixel space, [-1, 1]
      patchify                          fold 2x2 spatial patches into channels
            (B, 12, 4k+1, H/2,  W/2)
      encoder                           3 spatial stages, 2 temporal stages
            (B, 96, k+1,  H/16, W/16)   96 = 2 * z_dim (mean ‖ logvar)
      quant_conv                        1x1x1 causal conv
            (B, 96, k+1,  H/16, W/16)
      chunk(2, dim=1)                   DiagonalGaussianDistribution moments
      mean/logvar (B, 48, k+1, H/16, W/16)

    latent  (B, 48, k+1,  H/16, W/16)   normalised: (z - latents_mean)/latents_std
      denormalise -> post_quant_conv -> decoder
            (B, 12, 4k+1, H/2,  W/2)
      unpatchify + clamp(-1, 1)
            (B, 3,  4k+1, H,    W)

**Whole-sequence formulation.**  Upstream runs the encoder/decoder in temporal
chunks, threading a two-frame ``feat_cache`` through every ``WanCausalConv3d``.
A static ONNX graph cannot carry that Python-side cache, so every block here is
written in *whole-sequence* form.  The two are numerically identical:

* ``WanCausalConv3d`` (kernel 3, temporal pad ``(2, 0)``) over ``[cache(2), chunk]``
  is exactly the full-sequence causal convolution restricted to that chunk's
  output positions.
* ``downsample3d``'s ``time_conv`` (kernel 3, stride 2, no padding) is skipped
  for chunk 0 and, from chunk 1 on, is fed ``[last frame of previous chunk, chunk]``.
  Concatenated, its windows start at global frames ``0, 2, 4, ...`` — i.e.
  ``concat(x[:, :, :1], time_conv(x))`` over the whole sequence.
* ``upsample3d``'s ``time_conv`` is skipped for chunk 0 (no temporal doubling)
  and restarts the causal sequence with zero padding at chunk 1 — i.e.
  ``concat(x[:, :, :1], interleave(time_conv(x[:, :, 1:])))``.
* ``AvgDown3D`` front-pads the temporal axis to a multiple of ``factor_t``;
  chunk 0 (1 frame) pads 1, and a whole ``4k+1``-frame sequence also pads 1,
  producing the same groups.
* ``DupUp3D`` with ``first_chunk=True`` drops the first duplicated frame, which
  is exactly what chunk 0 contributes when the per-chunk outputs are concatenated.

The equivalence holds for the supported frame counts: ``T_video = 4k + 1``
(encoder) and ``T_latent = k + 1`` (decoder), for ``k >= 0``.

**Single-frame (image) mode.**  ``T_video = 1`` / ``T_latent = 1`` is a first
class upstream case — ``_encode`` runs ``1 + (T_video - 1) // 4`` chunks and
``_decode`` one chunk per latent frame, so a lone frame is decoded as chunk 0,
whose ``time_conv`` is skipped entirely (``feat_cache[idx] is None`` -> ``"Rep"``
for ``upsample3d``, ``feat_cache[idx] = x`` for ``downsample3d``).  A static
graph cannot skip a node, and the naive whole-sequence form would hand ``Conv``
a temporal extent shorter than its kernel (ONNX Runtime rejects that outright),
so both temporal resamplers use a *safe window*: the temporal axis is
zero-padded on the **right** by exactly the number of frames the kernel needs,
and the surplus trailing outputs are sliced off afterwards.  Because each
``time_conv`` is causal in time, the retained outputs are bit-identical to
the unpadded convolution, so multi-frame numerics are untouched and the
single-frame case degenerates to "keep frame 0 only", exactly like chunk 0
upstream.  The final ``Slice`` is applied *after* the frame-0 concatenation so
no zero-length tensor is ever materialised.

Weight names match the HuggingFace ``AutoencoderKLWan`` ``state_dict`` exactly,
so :meth:`AutoencoderKLWanModel.preprocess_weights` performs no renaming.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs._wan_vae import WanVAEConfig
from mobius.components import INT64_MAX
from mobius.components import Conv2d as _Conv2d
from mobius.components import SiLU as _SiLU

if TYPE_CHECKING:
    import torch

__all__ = ["AutoencoderKLWanModel"]

#: ``F.normalize`` clamps the L2 norm at this epsilon before dividing.
_NORMALIZE_EPS = 1e-12

#: ``DiagonalGaussianDistribution`` clamps ``logvar`` to this range.
_LOGVAR_MIN = -30.0
_LOGVAR_MAX = 20.0


def _dim(op: OpBuilder, x: ir.Value, axis: int) -> ir.Value:
    """Return ``x.shape[axis]`` as a 1-D int64 tensor of length 1."""
    return op.Shape(x, start=axis, end=axis + 1)


def _dims5(op: OpBuilder, x: ir.Value) -> tuple[ir.Value, ...]:
    """Return the five dimensions of a ``(B, C, T, H, W)`` tensor."""
    return tuple(_dim(op, x, axis) for axis in range(5))


# ---------------------------------------------------------------------------
# Primitive layers
# ---------------------------------------------------------------------------


class _WanCausalConv3d(nn.Module):
    """3D convolution with causal (left-only) padding on the temporal axis.

    Mirrors ``WanCausalConv3d``, which subclasses ``nn.Conv3d`` and rewrites the
    padding as ``(W, W, H, H, 2 * T, 0)`` in ``F.pad`` order before delegating to
    the unpadded convolution.  Doubling the temporal padding and placing it all
    on the left makes the convolution causal while preserving the frame count.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: ``(kT, kH, kW)`` or a single int applied to all three axes.
        stride: ``(sT, sH, sW)`` or a single int.
        padding: ``(pT, pH, pW)`` or a single int, in ``nn.Conv3d`` semantics.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__()
        kernel = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
        strides = (stride,) * 3 if isinstance(stride, int) else tuple(stride)
        pads = (padding,) * 3 if isinstance(padding, int) else tuple(padding)

        self.weight = nn.Parameter((out_channels, in_channels, *kernel))
        self.bias = nn.Parameter((out_channels,))
        self._kernel_shape = list(kernel)
        self._strides = list(strides)
        # ONNX Pad order for a 5D (B, C, T, H, W) tensor:
        # [B_beg, C_beg, T_beg, H_beg, W_beg, B_end, C_end, T_end, H_end, W_end].
        # Temporal padding is 2 * pT on the left and 0 on the right (causal).
        self._pads = [0, 0, 2 * pads[0], pads[1], pads[2], 0, 0, 0, pads[1], pads[2]]
        self._needs_pad = any(self._pads)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: (B, C_in, T, H, W)
        if self._needs_pad:
            # Pad's constant_value defaults to 0 for any float dtype.
            x = op.Pad(x, self._pads)
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=self._kernel_shape,
            strides=self._strides,
            pads=[0, 0, 0, 0, 0, 0],
            dilations=[1, 1, 1],
            group=1,
        )


class _WanRMSNorm(nn.Module):
    """Channel-wise RMS normalisation (``WanRMS_norm``).

    Computes ``F.normalize(x, dim=1) * dim ** 0.5 * gamma``.  ``F.normalize``
    divides by ``max(||x||_2, 1e-12)`` along the channel axis.  Upstream forces
    the normalisation itself to float32 for half-precision inputs and casts back
    before applying ``scale``/``gamma``; the ``Cast``/``CastLike`` pair below
    reproduces that and folds away for float32 graphs.

    Upstream's optional ``bias`` term is never enabled by ``AutoencoderKLWan``
    (every call site uses the default ``bias=False``), so no bias is emitted.

    Args:
        dim: Channel count.
        images: ``True`` broadcasts ``gamma`` over ``(H, W)`` for 4D activations
            (the attention block); ``False`` broadcasts over ``(T, H, W)`` for
            5D activations.  This matches the upstream parameter shapes
            ``(dim, 1, 1)`` and ``(dim, 1, 1, 1)`` respectively.
    """

    def __init__(self, dim: int, images: bool = True) -> None:
        super().__init__()
        broadcast = (1, 1) if images else (1, 1, 1)
        self.gamma = nn.Parameter((dim, *broadcast))
        self._scale = dim**0.5

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # Normalise in float32, matching WanRMS_norm's fp16/bf16 upcast.
        x_f32 = op.Cast(x, to=ir.DataType.FLOAT)
        norm = op.ReduceL2(x_f32, [1], keepdims=1)
        normalized = op.CastLike(op.Div(x_f32, op.Max(norm, _NORMALIZE_EPS)), x)
        scaled = op.Mul(normalized, op.CastLike(self._scale, x))
        return op.Mul(scaled, self.gamma)


class _ZeroPad2d(nn.Module):
    """``nn.ZeroPad2d((0, 1, 0, 1))`` — one row/column of zeros at bottom/right.

    Emitted as index ``0`` of the ``resample`` ``Sequential`` so the following
    ``Conv2d`` keeps the HuggingFace name ``resample.1.{weight,bias}``.
    """

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: (N, C, H, W) -> pads [N_beg, C_beg, H_beg, W_beg, N_end, C_end, H_end, W_end]
        return op.Pad(x, [0, 0, 0, 0, 0, 0, 1, 1])


class _NearestUpsample2d(nn.Module):
    """2x nearest-neighbour spatial upsampling (``WanUpsample``).

    Upstream uses ``mode="nearest-exact"``, which maps output index ``i`` to
    ``floor((i + 0.5) / 2)``.  For an exact integer scale that is identical to
    ``floor(i / 2)``, which ONNX expresses as ``asymmetric`` coordinate
    transformation with ``nearest_mode="floor"``.
    """

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: (N, C, H, W) -> (N, C, 2H, 2W)
        return op.Resize(
            x,
            None,
            [1.0, 1.0, 2.0, 2.0],
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )


# ---------------------------------------------------------------------------
# Residual shortcuts (parameter-free)
# ---------------------------------------------------------------------------


class _AvgDown3D(nn.Module):
    """Grouped average-pooling shortcut for Wan 2.2 encoder stages (``AvgDown3D``).

    Folds ``factor_t`` frames and ``factor_s x factor_s`` spatial positions into
    the channel axis, then averages consecutive groups of ``group_size``
    channels.  The temporal axis is front-padded with zeros so its length is a
    multiple of ``factor_t``, exactly as upstream's ``F.pad(x, (0, 0, 0, 0, pad_t, 0))``.

    Args:
        in_channels: Shortcut input channels.
        out_channels: Shortcut output channels.
        factor_t: Temporal reduction factor (1 or 2).
        factor_s: Spatial reduction factor (1 or 2).
    """

    def __init__(
        self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1
    ) -> None:
        super().__init__()
        factor = factor_t * factor_s * factor_s
        if in_channels * factor % out_channels != 0:
            raise ValueError(
                f"AvgDown3D requires in_channels * factor ({in_channels} * {factor}) "
                f"to be divisible by out_channels ({out_channels})"
            )
        self._out_channels = out_channels
        self._factor_t = factor_t
        self._factor_s = factor_s
        self._factor = factor
        self._group_size = in_channels * factor // out_channels

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: (B, C, T, H, W)
        if self._factor == 1 and self._group_size == 1:
            # No folding and a single-element group mean: algebraically the identity.
            return x

        if self._factor_t > 1:
            # pad_t = (factor_t - T % factor_t) % factor_t, applied at the front of T.
            t_len = _dim(op, x, 2)
            pad_t = op.Mod(
                op.Sub(self._factor_t, op.Mod(t_len, self._factor_t)), self._factor_t
            )
            pads = op.Concat(
                op.Constant(value_ints=[0, 0]),
                pad_t,
                op.Constant(value_ints=[0, 0, 0, 0, 0, 0, 0]),
                axis=0,
            )
            x = op.Pad(x, pads)

        batch, channels, t_len, height, width = _dims5(op, x)
        t_out = op.Div(t_len, self._factor_t)
        h_out = op.Div(height, self._factor_s)
        w_out = op.Div(width, self._factor_s)
        ft = op.Constant(value_ints=[self._factor_t])
        fs = op.Constant(value_ints=[self._factor_s])

        # (B, C, T/ft, ft, H/fs, fs, W/fs, fs)
        x = op.Reshape(x, op.Concat(batch, channels, t_out, ft, h_out, fs, w_out, fs, axis=0))
        # -> (B, C, ft, fs, fs, T/ft, H/fs, W/fs)
        x = op.Transpose(x, perm=[0, 1, 3, 5, 7, 2, 4, 6])
        # Flatten (C, ft, fs, fs) == (out_channels, group_size) and average the group.
        x = op.Reshape(
            x,
            op.Concat(
                batch,
                op.Constant(value_ints=[self._out_channels, self._group_size]),
                t_out,
                h_out,
                w_out,
                axis=0,
            ),
        )
        # (B, out_channels, T/ft, H/fs, W/fs)
        return op.ReduceMean(x, [2], keepdims=0)


class _DupUp3D(nn.Module):
    """Channel-duplication shortcut for Wan 2.2 decoder stages (``DupUp3D``).

    Repeats each input channel ``repeats`` times, then unfolds those copies into
    ``factor_t`` temporal and ``factor_s x factor_s`` spatial positions.

    Args:
        in_channels: Shortcut input channels.
        out_channels: Shortcut output channels.
        factor_t: Temporal expansion factor (1 or 2).
        factor_s: Spatial expansion factor (1 or 2).
    """

    def __init__(
        self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1
    ) -> None:
        super().__init__()
        factor = factor_t * factor_s * factor_s
        if out_channels * factor % in_channels != 0:
            raise ValueError(
                f"DupUp3D requires out_channels * factor ({out_channels} * {factor}) "
                f"to be divisible by in_channels ({in_channels})"
            )
        self._out_channels = out_channels
        self._factor_t = factor_t
        self._factor_s = factor_s
        self._repeats = out_channels * factor // in_channels

    def forward(self, op: OpBuilder, x: ir.Value, first_chunk: bool = True) -> ir.Value:
        # x: (B, C, T, H, W)
        batch, channels, t_len, height, width = _dims5(op, x)

        # repeat_interleave(repeats, dim=1): (B, C, 1, T, H, W) -> (B, C, r, T, H, W)
        x = op.Expand(
            op.Unsqueeze(x, [2]),
            op.Concat(
                batch,
                channels,
                op.Constant(value_ints=[self._repeats]),
                t_len,
                height,
                width,
                axis=0,
            ),
        )
        # -> (B, out_channels, ft, fs, fs, T, H, W)
        x = op.Reshape(
            x,
            op.Concat(
                batch,
                op.Constant(
                    value_ints=[
                        self._out_channels,
                        self._factor_t,
                        self._factor_s,
                        self._factor_s,
                    ]
                ),
                t_len,
                height,
                width,
                axis=0,
            ),
        )
        # -> (B, out_channels, T, ft, H, fs, W, fs)
        x = op.Transpose(x, perm=[0, 1, 5, 2, 6, 3, 7, 4])
        # -> (B, out_channels, T * ft, H * fs, W * fs)
        x = op.Reshape(
            x,
            op.Concat(
                batch,
                op.Constant(value_ints=[self._out_channels]),
                op.Mul(t_len, self._factor_t),
                op.Mul(height, self._factor_s),
                op.Mul(width, self._factor_s),
                axis=0,
            ),
        )
        if first_chunk and self._factor_t > 1:
            # Upstream drops the leading (factor_t - 1) duplicated frames of the
            # very first chunk so the shortcut lines up with the main path, whose
            # first latent frame is *not* temporally upsampled.
            x = op.Slice(x, [self._factor_t - 1], [INT64_MAX], [2])
        return x


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


class _WanResample(nn.Module):
    """Spatial (and optionally temporal) resampling block (``WanResample``).

    ``resample`` is a two-entry ``Sequential`` whose ``Conv2d`` sits at index 1,
    matching the HuggingFace parameter names ``resample.1.{weight,bias}``.  The
    convolution is applied per frame by folding ``T`` into the batch axis.

    Args:
        dim: Channel count entering the block.
        mode: One of ``upsample2d``, ``upsample3d``, ``downsample2d``,
            ``downsample3d``.
        upsample_out_dim: Output channels of the upsampling convolution.
            ``None`` means ``dim // 2`` (the Wan 2.1 default); the Wan 2.2
            residual decoder passes ``out_dim`` explicitly.
    """

    def __init__(self, dim: int, mode: str, upsample_out_dim: int | None = None) -> None:
        super().__init__()
        if mode not in ("upsample2d", "upsample3d", "downsample2d", "downsample3d"):
            raise ValueError(f"Unsupported WanResample mode {mode!r}")
        self._mode = mode
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        if mode in ("upsample2d", "upsample3d"):
            self.resample = nn.Sequential(
                _NearestUpsample2d(),
                _Conv2d(dim, upsample_out_dim, kernel_size=3, padding=1),
            )
            if mode == "upsample3d":
                self.time_conv = _WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        else:
            self.resample = nn.Sequential(
                _ZeroPad2d(),
                _Conv2d(dim, dim, kernel_size=3, stride=2, padding=0),
            )
            if mode == "downsample3d":
                self.time_conv = _WanCausalConv3d(
                    dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
                )

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: (B, C, T, H, W)
        if self._mode == "upsample3d":
            x = self._temporal_upsample(op, x)
        x = self._spatial_resample(op, x)
        if self._mode == "downsample3d":
            x = self._temporal_downsample(op, x)
        return x

    def _temporal_upsample(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Duplicate every frame after the first via ``time_conv`` + interleave.

        Whole-sequence form of upstream's per-chunk ``upsample3d``:
        ``concat(x[:, :, :1], interleave(time_conv(x[:, :, 1:])))``, i.e. frame 0
        passes through untouched (upstream's first decode chunk never runs
        ``time_conv``) and the causal sequence restarts, zero-padded, at frame 1.

        ``x[:, :, 1:]`` is empty for a single latent frame, which no ``Conv``
        kernel accepts.  Instead the temporal axis is zero-padded by one frame on
        the right *before* dropping frame 0, so the convolution always sees
        ``T >= 1`` frames; the surplus pair of interleaved output frames is then
        sliced off.  ``time_conv`` is causal, so the retained frames are exactly
        those of the unpadded convolution.
        """
        batch, channels, t_len, height, width = _dims5(op, x)
        first = op.Slice(x, [0], [1], [2])

        # (B, C, T, H, W) -> (B, C, T + 1, H, W) -> drop frame 0 -> T frames.
        padded = op.Pad(x, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        rest = op.Slice(padded, [1], [INT64_MAX], [2])

        # (B, C, T, H, W) -> (B, 2C, T, H, W)
        doubled = self.time_conv(op, rest)
        # Split the 2C channels into the two interleaved half-frames.
        even, odd = op.Split(doubled, num_outputs=2, axis=1, _outputs=2)
        # Interleave along a new axis: (B, C, T, 2, H, W) -> (B, C, 2T, H, W)
        stacked = op.Concat(op.Unsqueeze(even, [3]), op.Unsqueeze(odd, [3]), axis=3)
        rest = op.Reshape(
            stacked,
            op.Concat(batch, channels, op.Mul(t_len, 2), height, width, axis=0),
        )
        # (B, C, 1 + 2T, H, W) -> keep 2T - 1 frames (== 1 for a single frame).
        out = op.Concat(first, rest, axis=2)
        return op.Slice(out, [0], op.Sub(op.Mul(t_len, 2), 1), [2])

    def _temporal_downsample(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Stride-2 causal-free temporal convolution, keeping frame 0 verbatim.

        Upstream skips ``time_conv`` for chunk 0 and, from chunk 1 on, prepends
        the previous chunk's last frame; concatenated, the kernel-3/stride-2
        windows start at global frames ``0, 2, 4, ...`` over the whole sequence,
        i.e. ``concat(x[:, :, :1], time_conv(x))``.

        A single-frame clip is shorter than the kernel, so the temporal axis is
        zero-padded by two frames on the right and the surplus trailing output
        frame is sliced off.  The convolution only ever looks backwards in time,
        so the retained frames match the unpadded convolution exactly.
        """
        t_len = _dim(op, x, 2)
        first = op.Slice(x, [0], [1], [2])

        # (B, C, T, H, W) -> (B, C, T + 2, H, W) -> stride-2 conv -> (T - 1) // 2 + 1.
        padded = op.Pad(x, [0, 0, 0, 0, 0, 0, 0, 2, 0, 0])
        strided = self.time_conv(op, padded)
        # Keep 1 + (T - 1) // 2 frames, dropping the one window that read padding.
        out = op.Concat(first, strided, axis=2)
        keep = op.Add(op.Div(op.Sub(t_len, 1), 2), 1)
        return op.Slice(out, [0], keep, [2])

    def _spatial_resample(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Apply the 2D ``resample`` Sequential frame-by-frame."""
        batch, channels, t_len, height, width = _dims5(op, x)
        # (B, C, T, H, W) -> (B*T, C, H, W)
        frames = op.Reshape(
            op.Transpose(x, perm=[0, 2, 1, 3, 4]),
            op.Concat(op.Mul(batch, t_len), channels, height, width, axis=0),
        )
        frames = self.resample(op, frames)
        out_c = _dim(op, frames, 1)
        out_h = _dim(op, frames, 2)
        out_w = _dim(op, frames, 3)
        # (B*T, C', H', W') -> (B, C', T, H', W')
        return op.Transpose(
            op.Reshape(frames, op.Concat(batch, t_len, out_c, out_h, out_w, axis=0)),
            perm=[0, 2, 1, 3, 4],
        )


# ---------------------------------------------------------------------------
# Residual / attention blocks
# ---------------------------------------------------------------------------


class _WanResidualBlock(nn.Module):
    """Pre-norm residual block with two causal 3D convolutions (``WanResidualBlock``).

    ``norm1 -> SiLU -> conv1 -> norm2 -> SiLU -> conv2`` plus a shortcut that is
    a 1x1x1 causal convolution when the channel count changes and the identity
    otherwise (upstream uses ``nn.Identity``, which contributes no weights).

    Upstream's ``nn.Dropout`` between ``norm2`` and ``conv2`` is an identity in
    eval mode and therefore not emitted.

    Args:
        in_dim: Input channels.
        out_dim: Output channels.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm1 = _WanRMSNorm(in_dim, images=False)
        self.conv1 = _WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = _WanRMSNorm(out_dim, images=False)
        self.conv2 = _WanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            _WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else None
        )
        self.nonlinearity = _SiLU()

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: (B, C_in, T, H, W)
        shortcut = x if self.conv_shortcut is None else self.conv_shortcut(op, x)
        x = self.conv1(op, self.nonlinearity(op, self.norm1(op, x)))
        x = self.conv2(op, self.nonlinearity(op, self.norm2(op, x)))
        return op.Add(x, shortcut)


class _WanAttentionBlock(nn.Module):
    """Single-head spatial self-attention over each frame (``WanAttentionBlock``).

    Frames are folded into the batch axis, so attention is computed
    independently per frame over the ``H * W`` spatial positions.

    Args:
        dim: Channel count (also the single head's dimension).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = _WanRMSNorm(dim, images=True)
        self.to_qkv = _Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = _Conv2d(dim, dim, kernel_size=1)
        self._scale = dim**-0.5

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: (B, C, T, H, W)
        identity = x
        batch, channels, t_len, height, width = _dims5(op, x)
        batch_frames = op.Mul(batch, t_len)

        # (B, C, T, H, W) -> (B*T, C, H, W)
        frames = op.Reshape(
            op.Transpose(x, perm=[0, 2, 1, 3, 4]),
            op.Concat(batch_frames, channels, height, width, axis=0),
        )
        frames = self.norm(op, frames)

        # (B*T, 3C, H, W) -> (B*T, 1, H*W, 3C), then split into q, k, v
        qkv = self.to_qkv(op, frames)
        qkv = op.Reshape(
            qkv,
            op.Concat(
                batch_frames,
                op.Constant(value_ints=[1]),
                op.Mul(channels, 3),
                op.Constant(value_ints=[-1]),
                axis=0,
            ),
        )
        qkv = op.Transpose(qkv, perm=[0, 1, 3, 2])
        query, key, value = op.Split(qkv, num_outputs=3, axis=-1, _outputs=3)

        # Single-head scaled dot-product attention -> (B*T, 1, H*W, C)
        attn = op.Attention(
            query, key, value, scale=self._scale, q_num_heads=1, kv_num_heads=1
        )

        # (B*T, 1, H*W, C) -> (B*T, C, H, W)
        attn = op.Transpose(
            op.Reshape(attn, op.Concat(batch_frames, op.Mul(height, width), channels, axis=0)),
            perm=[0, 2, 1],
        )
        attn = op.Reshape(attn, op.Concat(batch_frames, channels, height, width, axis=0))
        attn = self.proj(op, attn)

        # (B*T, C, H, W) -> (B, C, T, H, W)
        out = op.Transpose(
            op.Reshape(attn, op.Concat(batch, t_len, channels, height, width, axis=0)),
            perm=[0, 2, 1, 3, 4],
        )
        return op.Add(out, identity)


class _WanMidBlock(nn.Module):
    """Bottleneck block: ``resnet -> (attention -> resnet) * num_layers``.

    Args:
        dim: Channel count.
        num_layers: Number of attention/resnet pairs after the first resnet.
    """

    def __init__(self, dim: int, num_layers: int = 1) -> None:
        super().__init__()
        self.attentions = nn.ModuleList([_WanAttentionBlock(dim) for _ in range(num_layers)])
        self.resnets = nn.ModuleList(
            [_WanResidualBlock(dim, dim) for _ in range(num_layers + 1)]
        )

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        x = self.resnets[0](op, x)
        for index, attention in enumerate(self.attentions):
            x = attention(op, x)
            x = self.resnets[index + 1](op, x)
        return x


# ---------------------------------------------------------------------------
# Encoder / decoder stages
# ---------------------------------------------------------------------------


class _WanResidualDownBlock(nn.Module):
    """Wan 2.2 encoder stage: residual blocks + downsampler + ``AvgDown3D`` shortcut.

    Args:
        in_dim: Stage input channels.
        out_dim: Stage output channels.
        num_res_blocks: Residual blocks in the main path.
        temporal_downsample: Halve the temporal axis as well as the spatial axes.
        down_flag: Whether this stage downsamples at all (the last stage does not).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temporal_downsample: bool = False,
        down_flag: bool = False,
    ) -> None:
        super().__init__()
        self.avg_shortcut = _AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temporal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks):
            resnets.append(_WanResidualBlock(current_dim, out_dim))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.downsampler = (
            _WanResample(out_dim, "downsample3d" if temporal_downsample else "downsample2d")
            if down_flag
            else None
        )

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: (B, C_in, T, H, W)
        residual = x
        for resnet in self.resnets:
            x = resnet(op, x)
        if self.downsampler is not None:
            x = self.downsampler(op, x)
        return op.Add(x, self.avg_shortcut(op, residual))


class _WanResidualUpBlock(nn.Module):
    """Wan 2.2 decoder stage: residual blocks + upsampler + ``DupUp3D`` shortcut.

    Args:
        in_dim: Stage input channels.
        out_dim: Stage output channels.
        num_res_blocks: ``num_res_blocks + 1`` residual blocks are created,
            matching upstream.
        temporal_upsample: Double the temporal axis as well as the spatial axes.
        up_flag: Whether this stage upsamples at all (the last stage does not).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temporal_upsample: bool = False,
        up_flag: bool = False,
    ) -> None:
        super().__init__()
        self.avg_shortcut = (
            _DupUp3D(in_dim, out_dim, factor_t=2 if temporal_upsample else 1, factor_s=2)
            if up_flag
            else None
        )
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(_WanResidualBlock(current_dim, out_dim))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsampler = (
            _WanResample(
                out_dim,
                "upsample3d" if temporal_upsample else "upsample2d",
                upsample_out_dim=out_dim,
            )
            if up_flag
            else None
        )

    def forward(self, op: OpBuilder, x: ir.Value, first_chunk: bool = True) -> ir.Value:
        # x: (B, C_in, T, H, W)
        residual = x
        for resnet in self.resnets:
            x = resnet(op, x)
        if self.upsampler is not None:
            x = self.upsampler(op, x)
        if self.avg_shortcut is not None:
            x = op.Add(x, self.avg_shortcut(op, residual, first_chunk=first_chunk))
        return x


class _WanUpBlock(nn.Module):
    """Wan 2.1 decoder stage: residual blocks + optional upsampler, no shortcut.

    Args:
        in_dim: Stage input channels.
        out_dim: Stage output channels.
        num_res_blocks: ``num_res_blocks + 1`` residual blocks are created.
        upsample_mode: ``"upsample2d"``, ``"upsample3d"`` or ``None``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        upsample_mode: str | None = None,
    ) -> None:
        super().__init__()
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(_WanResidualBlock(current_dim, out_dim))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        # Upstream stores this as a one-element ModuleList named ``upsamplers``.
        self.upsamplers = (
            nn.ModuleList([_WanResample(out_dim, upsample_mode)])
            if upsample_mode is not None
            else None
        )

    def forward(self, op: OpBuilder, x: ir.Value, first_chunk: bool = True) -> ir.Value:
        # ``first_chunk`` is accepted for signature parity with
        # ``_WanResidualUpBlock`` (upstream's ``WanUpBlock.forward`` does the same);
        # Wan 2.1 has no ``DupUp3D`` shortcut, so it has no effect here.
        del first_chunk
        for resnet in self.resnets:
            x = resnet(op, x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](op, x)
        return x


class _WanEncoder3d(nn.Module):
    """3D encoder (``WanEncoder3d``): video -> ``2 * z_dim`` posterior moments.

    Note:
        The Wan 2.1 (``is_residual=False``) branch inserts a
        :class:`_WanAttentionBlock` after any residual block whose spatial scale
        appears in ``attn_scales``.  That branch is currently unreachable in
        diffusers itself — ``WanEncoder3d.forward`` forwards ``feat_cache`` to
        every ``down_blocks`` entry, and ``WanAttentionBlock.forward`` does not
        accept it — so it has no runnable upstream reference to compare against.
        The layout and weight names implemented here follow the upstream
        constructor exactly.

    Args:
        config: Parsed :class:`~mobius._configs._wan_vae.WanVAEConfig`.
    """

    def __init__(self, config: WanVAEConfig) -> None:
        super().__init__()
        dims = config.encoder_dims
        last_stage = len(config.dim_mult) - 1

        self.conv_in = _WanCausalConv3d(config.in_channels, dims[0], 3, padding=1)

        self.down_blocks = nn.ModuleList([])
        scale = 1.0
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            if config.is_residual:
                self.down_blocks.append(
                    _WanResidualDownBlock(
                        in_dim,
                        out_dim,
                        config.num_res_blocks,
                        temporal_downsample=(
                            config.temporal_downsample[i] if i != last_stage else False
                        ),
                        down_flag=i != last_stage,
                    )
                )
                continue
            # Wan 2.1: flat list of residual (and optional attention) blocks.
            stage_in = in_dim
            for _ in range(config.num_res_blocks):
                self.down_blocks.append(_WanResidualBlock(stage_in, out_dim))
                if scale in config.attn_scales:
                    self.down_blocks.append(_WanAttentionBlock(out_dim))
                stage_in = out_dim
            if i != last_stage:
                mode = "downsample3d" if config.temporal_downsample[i] else "downsample2d"
                self.down_blocks.append(_WanResample(out_dim, mode=mode))
                scale /= 2.0

        self.mid_block = _WanMidBlock(dims[-1], num_layers=1)
        self.norm_out = _WanRMSNorm(dims[-1], images=False)
        # The encoder emits mean ‖ logvar, hence 2 * z_dim output channels.
        self.conv_out = _WanCausalConv3d(dims[-1], config.z_dim * 2, 3, padding=1)
        self.nonlinearity = _SiLU()

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: (B, in_channels, T, H, W)
        x = self.conv_in(op, x)
        for block in self.down_blocks:
            x = block(op, x)
        x = self.mid_block(op, x)
        x = self.nonlinearity(op, self.norm_out(op, x))
        # (B, 2 * z_dim, T', H', W')
        return self.conv_out(op, x)


class _WanDecoder3d(nn.Module):
    """3D decoder (``WanDecoder3d``): ``z_dim`` latents -> ``out_channels`` frames.

    Args:
        config: Parsed :class:`~mobius._configs._wan_vae.WanVAEConfig`.
    """

    def __init__(self, config: WanVAEConfig) -> None:
        super().__init__()
        dims = config.decoder_dims
        temporal_upsample = config.temporal_upsample
        last_stage = len(config.dim_mult) - 1

        self.conv_in = _WanCausalConv3d(config.z_dim, dims[0], 3, padding=1)
        self.mid_block = _WanMidBlock(dims[0], num_layers=1)

        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            up_flag = i != last_stage
            if config.is_residual:
                self.up_blocks.append(
                    _WanResidualUpBlock(
                        in_dim,
                        out_dim,
                        config.num_res_blocks,
                        temporal_upsample=temporal_upsample[i] if up_flag else False,
                        up_flag=up_flag,
                    )
                )
                continue
            # Wan 2.1 halves the incoming width from the second stage on, because
            # its upsampling convolution emits ``dim // 2`` channels.
            stage_in = in_dim // 2 if i > 0 else in_dim
            upsample_mode = None
            if up_flag:
                upsample_mode = "upsample3d" if temporal_upsample[i] else "upsample2d"
            self.up_blocks.append(
                _WanUpBlock(stage_in, out_dim, config.num_res_blocks, upsample_mode)
            )

        self.norm_out = _WanRMSNorm(dims[-1], images=False)
        self.conv_out = _WanCausalConv3d(dims[-1], config.out_channels, 3, padding=1)
        self.nonlinearity = _SiLU()

    def forward(self, op: OpBuilder, x: ir.Value, first_chunk: bool = True) -> ir.Value:
        # x: (B, z_dim, T, H, W)
        x = self.conv_in(op, x)
        x = self.mid_block(op, x)
        for block in self.up_blocks:
            x = block(op, x, first_chunk=first_chunk)
        x = self.nonlinearity(op, self.norm_out(op, x))
        # (B, out_channels, T'', H'', W'')
        return self.conv_out(op, x)


# ---------------------------------------------------------------------------
# Full autoencoder
# ---------------------------------------------------------------------------


class AutoencoderKLWanModel(nn.Module):
    """Wan 3D causal video VAE (``diffusers.AutoencoderKLWan``).

    Exposes the four HuggingFace sub-modules (``encoder``, ``quant_conv``,
    ``post_quant_conv``, ``decoder``) plus the pipeline-level helpers that live
    *outside* those sub-modules upstream: :meth:`patchify` / :meth:`unpatchify`
    (applied in ``AutoencoderKLWan._encode`` / ``._decode``) and
    :meth:`normalize_latents` / :meth:`denormalize_latents` (applied by the
    ``WanPipeline``, not by the VAE itself).  Keeping them off the sub-modules
    is what makes the exported initializer names match the checkpoint exactly.

    Args:
        config: Parsed :class:`~mobius._configs._wan_vae.WanVAEConfig`.
    """

    default_task: str = "wan-vae"
    category: str = "autoencoder"

    def __init__(self, config: WanVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _WanEncoder3d(config)
        self.quant_conv = _WanCausalConv3d(config.z_dim * 2, config.z_dim * 2, 1)
        self.post_quant_conv = _WanCausalConv3d(config.z_dim, config.z_dim, 1)
        self.decoder = _WanDecoder3d(config)

    # ------------------------------------------------------------------
    # Patch folding (``AutoencoderKLWan._encode`` / ``._decode``)
    # ------------------------------------------------------------------

    def patchify(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Fold ``patch_size x patch_size`` spatial patches into the channel axis.

        Args:
            op: The ONNX op builder.
            x: Video tensor ``(B, C, T, H, W)``.

        Returns:
            ``(B, C * p * p, T, H / p, W / p)``, or *x* unchanged when
            ``patch_size`` is ``None`` or 1.
        """
        patch = self.config.patch_size
        if patch is None or patch == 1:
            return x
        batch, channels, t_len, height, width = _dims5(op, x)
        p = op.Constant(value_ints=[patch])
        # (B, C, T, H/p, p, W/p, p)
        x = op.Reshape(
            x,
            op.Concat(
                batch,
                channels,
                t_len,
                op.Div(height, patch),
                p,
                op.Div(width, patch),
                p,
                axis=0,
            ),
        )
        # -> (B, C, p_w, p_h, T, H/p, W/p)
        x = op.Transpose(x, perm=[0, 1, 6, 4, 2, 3, 5])
        return op.Reshape(
            x,
            op.Concat(
                batch,
                op.Mul(channels, patch * patch),
                t_len,
                op.Div(height, patch),
                op.Div(width, patch),
                axis=0,
            ),
        )

    def unpatchify(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Unfold channel-packed patches back into spatial resolution.

        Args:
            op: The ONNX op builder.
            x: Decoder output ``(B, C * p * p, T, H, W)``.

        Returns:
            ``(B, C, T, H * p, W * p)``, or *x* unchanged when ``patch_size``
            is ``None`` or 1.
        """
        patch = self.config.patch_size
        if patch is None or patch == 1:
            return x
        batch, channels, t_len, height, width = _dims5(op, x)
        p = op.Constant(value_ints=[patch])
        # (B, C, p_w, p_h, T, H, W)
        x = op.Reshape(
            x,
            op.Concat(
                batch, op.Div(channels, patch * patch), p, p, t_len, height, width, axis=0
            ),
        )
        # -> (B, C, T, H, p_h, W, p_w)
        x = op.Transpose(x, perm=[0, 1, 4, 5, 3, 6, 2])
        return op.Reshape(
            x,
            op.Concat(
                batch,
                op.Div(channels, patch * patch),
                t_len,
                op.Mul(height, patch),
                op.Mul(width, patch),
                axis=0,
            ),
        )

    # ------------------------------------------------------------------
    # Latent statistics (``WanPipeline``-level, not part of the VAE weights)
    # ------------------------------------------------------------------

    def _latent_stat(self, op: OpBuilder, values: tuple[float, ...], name: str) -> ir.Value:
        """Materialise a per-channel latent statistic as a ``(1, z, 1, 1, 1)`` constant."""
        array = np.asarray(values, dtype=np.float32).reshape(1, len(values), 1, 1, 1)
        dtype = self.config.dtype
        tensor = ir.tensor(array.astype(dtype.numpy()), dtype=dtype, name=name)
        return op.initializer(tensor, name)

    def normalize_latents(self, op: OpBuilder, latents: ir.Value) -> ir.Value:
        """Apply ``(z - latents_mean) / latents_std``.

        This is the diffusers ``WanPipeline`` convention (which stores the
        reciprocal in a variable also called ``latents_std`` and multiplies).
        It happens outside ``AutoencoderKLWan`` upstream, so it is emitted at the
        graph boundary rather than inside :attr:`encoder`.

        Args:
            op: The ONNX op builder.
            latents: Raw posterior latents ``(B, z_dim, T, H, W)``.

        Returns:
            Normalised latents with the same shape.
        """
        mean = self._latent_stat(op, self.config.latents_mean, "latents_mean")
        std = self._latent_stat(op, self.config.latents_std, "latents_std")
        return op.Div(op.Sub(latents, mean), std)

    def denormalize_latents(self, op: OpBuilder, latents: ir.Value) -> ir.Value:
        """Apply ``z * latents_std + latents_mean`` (inverse of :meth:`normalize_latents`).

        Args:
            op: The ONNX op builder.
            latents: Normalised latents ``(B, z_dim, T, H, W)``.

        Returns:
            Raw latents in the VAE's own scale, ready for ``post_quant_conv``.
        """
        mean = self._latent_stat(op, self.config.latents_mean, "latents_mean")
        std = self._latent_stat(op, self.config.latents_std, "latents_std")
        return op.Add(op.Mul(latents, std), mean)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, op: OpBuilder, sample: ir.Value) -> tuple[ir.Value, ir.Value]:
        """Encode a video into deterministic posterior moments.

        Mirrors ``AutoencoderKLWan._encode`` followed by
        ``DiagonalGaussianDistribution``: patchify, run the encoder, apply
        ``quant_conv``, split the ``2 * z_dim`` channels into mean and logvar and
        clamp the logvar to ``[-30, 20]``.  No sampling happens in the graph.

        Args:
            op: The ONNX op builder.
            sample: Video tensor ``(B, video_channels, T, H, W)`` in ``[-1, 1]``.

        Returns:
            ``(mean, logvar)``, each ``(B, z_dim, T', H', W')``.
        """
        hidden = self.patchify(op, sample)
        hidden = self.encoder(op, hidden)
        moments = self.quant_conv(op, hidden)
        mean, logvar = op.Split(moments, num_outputs=2, axis=1, _outputs=2)
        logvar = op.Clip(
            logvar, op.CastLike(_LOGVAR_MIN, logvar), op.CastLike(_LOGVAR_MAX, logvar)
        )
        return mean, logvar

    def decode(self, op: OpBuilder, latents: ir.Value) -> ir.Value:
        """Decode raw (un-normalised) latents into a video.

        Mirrors ``AutoencoderKLWan._decode``: ``post_quant_conv``, decoder,
        unpatchify and the unconditional clamp to ``[-1, 1]``.

        Args:
            op: The ONNX op builder.
            latents: Raw latents ``(B, z_dim, T, H, W)``.

        Returns:
            Video tensor ``(B, decoded_video_channels, 4 * (T - 1) + 1, H', W')``.
        """
        hidden = self.post_quant_conv(op, latents)
        hidden = self.decoder(op, hidden)
        hidden = self.unpatchify(op, hidden)
        # Upstream clamps unconditionally; ``clip_output`` from the public
        # config is not a parameter of AutoencoderKLWan and is ignored there.
        return op.Clip(hidden, op.CastLike(-1.0, hidden), op.CastLike(1.0, hidden))

    def forward(self, op: OpBuilder, latents: ir.Value) -> ir.Value:
        """Decode normalised latents into a video (the generation-time entry point)."""
        return self.decode(op, self.denormalize_latents(op, latents))

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return the HuggingFace state dict unchanged.

        Every ``nn.Module`` attribute name here mirrors the corresponding
        ``AutoencoderKLWan`` attribute, so the generated initializer names
        (``encoder.down_blocks.1.resnets.0.conv1.weight``,
        ``decoder.up_blocks.0.upsampler.resample.1.bias``, ...) already match the
        checkpoint keys and no renaming is required.
        """
        return state_dict
