# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Cosmos3 AVAE audio tokenizer — direct ONNX graph construction.

Replicates ``diffusers.models.autoencoders.autoencoder_cosmos3_audio``
(``Cosmos3AVAEAudioTokenizer``), the ``sound_tokenizer`` component of
``nvidia/Cosmos3-Nano`` / ``nvidia/Cosmos3-Super``.

Pipeline::

    waveform [B, C, N]
      -> (optional) peak volume normalization
      -> (optional) right zero-pad to a multiple of hop_size
      -> STFT front-end  [B, (n_fft+2)*C, N/hop_length]
      -> encoder.layers  (1x1 conv -> {ConvNeXt x k, strided conv} x S -> 1x1 conv)
      -> moments [B, 2*z, T]
      -> bottleneck (VAE): mean, scale = split(moments); std = softplus(scale) + 1e-4

    latents [B, z, T]
      -> decoder.conv1 -> decoder.block.{i} (Snake -> ConvTranspose -> 3x ResidualUnit)
      -> decoder.snake1 -> decoder.conv2 -> clamp(-1, 1)
      -> waveform [B, 2, T * hop_size]

Module attribute names mirror the upstream ``nn.Module`` tree exactly
(``encoder.layers.3.weight``, ``decoder.block.0.res_unit2.conv1.weight``, ...)
so ONNX initializer names line up with the HuggingFace checkpoint after the
weight-norm fold performed by :func:`fold_weight_norm`.

Published checkpoints (all FP32, all using legacy ``weight_g``/``weight_v``)::

    nvidia/Cosmos3-Nano             249 tensors = 67 encoder + 182 decoder
    nvidia/Cosmos3-Super            249 tensors = 67 encoder + 182 decoder
    nvidia/Cosmos3-Super-Text2Image 182 tensors =  0 encoder + 182 decoder

After folding (5 weight-norm pairs in the encoder, 37 in the decoder) that is
62 + 145 = 207 initializers for a full build and 145 for a decoder-only build.
Because all three ship the *same* config JSON, encoder presence must be read
from the weights — see :func:`state_dict_has_encoder` and the two tokenizer
classes below.

Two structural differences from the PyTorch reference, both required by ONNX:

* ``weight_norm`` is folded offline. PyTorch stores ``weight_g``/``weight_v``
  (or ``parametrizations.weight.original0/original1``); ONNX ``Conv`` takes a
  single weight tensor, so :meth:`preprocess_weights` recombines them into
  ``<path>.weight``. Everything above the leaf name is unchanged, and no
  ``g * v / ||v||`` math is left in the graph.
* Posterior sampling stays out of the graph. The encoder graph emits
  ``moments``/``latent_mean``/``latent_std`` and the caller draws the sample,
  keeping the ONNX model deterministic.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs._cosmos3_audio import Cosmos3AudioConfig, state_dict_has_encoder

if TYPE_CHECKING:
    import torch

__all__ = [
    "Cosmos3AVAEAudioDecoderOnlyTokenizer",
    "Cosmos3AVAEAudioTokenizer",
    "Cosmos3AudioConvNeXtBlock",
    "Cosmos3AudioDecoder",
    "Cosmos3AudioDecoderBlock",
    "Cosmos3AudioResidualUnit",
    "Cosmos3AudioSpectrogramConvNeXtEncoder",
    "Cosmos3AudioVAEBottleneck",
    "create_cosmos3_avae_audio_tokenizer",
    "fold_weight_norm",
    "state_dict_has_encoder",
]

# Matches ``OobleckDiagonalGaussianDistribution``: std = softplus(scale) + 1e-4.
_POSTERIOR_STD_EPS = 1e-4
# Matches ``Snake1d``: x + (beta + 1e-9).reciprocal() * sin(alpha * x)^2.
_SNAKE_EPS = 1e-9
# Matches ``Cosmos3AVAEAudioTokenizer.encode``: x / (|x|.max() + 1e-5) * 0.95.
_VOLUME_EPS = 1e-5
_VOLUME_PEAK = 0.95


def _scalar(op: OpBuilder, value: float, like: ir.Value) -> ir.Value:
    """Build a scalar constant cast to the dtype of *like*."""
    return op.CastLike(op.Constant(value_float=float(value)), like)


# ---------------------------------------------------------------------------
# Primitive layers (parameter holders mirroring torch.nn leaf modules)
# ---------------------------------------------------------------------------


class _Conv1d(nn.Module):
    """``nn.Conv1d`` with symmetric zero padding.

    Parameter names are ``weight`` / ``bias``, matching PyTorch after the
    weight-norm fold (``weight_g``/``weight_v`` -> ``weight``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: ir.DataType = ir.DataType.FLOAT,
    ):
        super().__init__()
        # Conv1d weight layout: (out_channels, in_channels / groups, kernel_size)
        self.weight = nn.Parameter([out_channels, in_channels // groups, kernel_size], dtype)
        self.bias = nn.Parameter([out_channels], dtype) if bias else None
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Convolve ``(B, C_in, L)`` into ``(B, C_out, L')``."""
        inputs = [x, self.weight] + ([self.bias] if self.bias is not None else [])
        return op.Conv(
            *inputs,
            kernel_shape=[self._kernel_size],
            strides=[self._stride],
            pads=[self._padding, self._padding],
            dilations=[self._dilation],
            group=self._groups,
        )


class _ConvTranspose1d(nn.Module):
    """``nn.ConvTranspose1d`` with symmetric padding and right output padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        bias: bool = True,
        dtype: ir.DataType = ir.DataType.FLOAT,
    ):
        super().__init__()
        # ConvTranspose1d weight layout: (in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter([in_channels, out_channels, kernel_size], dtype)
        self.bias = nn.Parameter([out_channels], dtype) if bias else None
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._output_padding = output_padding

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Upsample ``(B, C_in, L)`` to ``(B, C_out, L * stride)``.

        With ``kernel_size = 2 * stride``, ``padding = ceil(stride / 2)`` and
        ``output_padding = stride % 2`` the ONNX length formula
        ``stride * (L - 1) + output_padding + kernel_size - 2 * padding``
        collapses to exactly ``L * stride`` for both even and odd strides.
        """
        inputs = [x, self.weight] + ([self.bias] if self.bias is not None else [])
        return op.ConvTranspose(
            *inputs,
            kernel_shape=[self._kernel_size],
            strides=[self._stride],
            pads=[self._padding, self._padding],
            output_padding=[self._output_padding],
            dilations=[1],
            group=1,
        )


class _ConstantPad1d(nn.Module):
    """Parameter-free ``nn.ConstantPad1d`` over the temporal axis.

    Kept as its own module so the enclosing ``nn.Sequential`` assigns index
    ``0`` here and index ``1`` to the depthwise conv, reproducing the upstream
    ``dwconv.1.weight`` parameter names.
    """

    def __init__(self, pad_left: int, pad_right: int):
        super().__init__()
        # ONNX Pad on rank-3 input: [b0, b1, b2, e0, e1, e2]
        self._pads = [0, 0, pad_left, 0, 0, pad_right]
        self._needs_pad = pad_left > 0 or pad_right > 0

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Zero-pad ``(B, C, L)`` to ``(B, C, L + left + right)``."""
        if not self._needs_pad:
            return x
        return op.Pad(x, op.Constant(value_ints=self._pads))


class _Snake1d(nn.Module):
    """SnakeBeta activation ``x + (beta + 1e-9)^-1 * sin(alpha * x)^2``.

    Replicates ``Snake1d`` from the upstream module. ``alpha``/``beta`` keep the
    checkpoint's ``(1, C, 1)`` shape so they broadcast over ``(B, C, T)``
    without a reshape, and are exponentiated when ``logscale`` is set.
    """

    def __init__(
        self,
        hidden_dim: int,
        logscale: bool = True,
        dtype: ir.DataType = ir.DataType.FLOAT,
    ):
        super().__init__()
        self.alpha = nn.Parameter([1, hidden_dim, 1], dtype)
        self.beta = nn.Parameter([1, hidden_dim, 1], dtype)
        self._logscale = logscale

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Apply SnakeBeta elementwise to ``(B, C, T)``."""
        alpha = op.Exp(self.alpha) if self._logscale else self.alpha
        beta = op.Exp(self.beta) if self._logscale else self.beta
        # sin^2(alpha * x) — (B, C, T) broadcast against (1, C, 1)
        sin_val = op.Sin(op.Mul(alpha, x))
        sin_sq = op.Mul(sin_val, sin_val)
        inv_beta = op.Reciprocal(op.Add(beta, _scalar(op, _SNAKE_EPS, beta)))
        return op.Add(x, op.Mul(inv_beta, sin_sq))


class _GELU(nn.Module):
    """Exact (erf) GELU, matching ``nn.GELU()`` with default approximation."""

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Apply GELU elementwise."""
        return op.Gelu(x, approximate="none")


class _FP32LayerNorm(nn.Module):
    """Weight-only LayerNorm evaluated in float32.

    Mirrors ``diffusers.models.normalization.FP32LayerNorm`` constructed with
    ``bias=False``: the input and weight are upcast to float32, normalized over
    the last axis, then cast back to the original dtype. For float32 graphs the
    casts are elided.
    """

    def __init__(
        self,
        hidden_dim: int,
        eps: float = 1e-5,
        dtype: ir.DataType = ir.DataType.FLOAT,
    ):
        super().__init__()
        self.weight = nn.Parameter([hidden_dim], dtype)
        self._eps = eps
        self._needs_cast = dtype != ir.DataType.FLOAT

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Normalize the last axis of ``(B, T, C)``."""
        if not self._needs_cast:
            return op.LayerNormalization(x, self.weight, axis=-1, epsilon=self._eps)
        x_fp32 = op.Cast(x, to=ir.DataType.FLOAT)
        weight_fp32 = op.Cast(self.weight, to=ir.DataType.FLOAT)
        normalized = op.LayerNormalization(x_fp32, weight_fp32, axis=-1, epsilon=self._eps)
        return op.CastLike(normalized, x)


# ---------------------------------------------------------------------------
# Encoder: spectrogram ConvNeXt
# ---------------------------------------------------------------------------


class Cosmos3AudioConvNeXtBlock(nn.Module):
    """1-D ConvNeXt block used by the Cosmos3 SpecConvNeXt encoder.

    ``residual + pwconv2(act(pwconv1(norm(dwconv(x)))))`` where ``dwconv`` is a
    depthwise k=7 convolution wrapped in an explicit constant pad, and ``norm``
    is a channels-last FP32 LayerNorm.

    HF class: ``Cosmos3AudioConvNeXtBlock``.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        *,
        use_snake: bool = True,
        snake_logscale: bool = True,
        dtype: ir.DataType = ir.DataType.FLOAT,
    ):
        super().__init__()
        # Non-causal variant: symmetric (3, 3) pad around a k=7 depthwise conv,
        # so the temporal length is preserved.
        self.dwconv = nn.Sequential(
            _ConstantPad1d(3, 3),
            _Conv1d(hidden_dim, hidden_dim, 7, groups=hidden_dim, dtype=dtype),
        )
        self.norm = _FP32LayerNorm(hidden_dim, eps=1e-5, dtype=dtype)
        self.pwconv1 = _Conv1d(hidden_dim, intermediate_dim, 1, dtype=dtype)
        self.act: nn.Module = (
            _Snake1d(intermediate_dim, snake_logscale, dtype) if use_snake else _GELU()
        )
        self.pwconv2 = _Conv1d(intermediate_dim, hidden_dim, 1, dtype=dtype)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        """Transform ``(B, C, T)`` in place (channel count unchanged)."""
        residual = hidden_states
        # (B, C, T) -> (B, C, T)
        hidden_states = self.dwconv(op, hidden_states)
        # LayerNorm is channels-last upstream: (B, C, T) -> (B, T, C) -> (B, C, T)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = self.norm(op, hidden_states)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        # (B, C, T) -> (B, 4C, T) -> act -> (B, C, T)
        hidden_states = self.pwconv1(op, hidden_states)
        hidden_states = self.act(op, hidden_states)
        hidden_states = self.pwconv2(op, hidden_states)
        return op.Add(residual, hidden_states)


class Cosmos3AudioSpectrogramConvNeXtEncoder(nn.Module):
    """Waveform → latent-moments encoder (STFT front-end + ConvNeXt stack).

    The ``layers`` attribute is a flat ``nn.Sequential`` whose indices match the
    checkpoint exactly for ``enc_num_blocks=2`` and three stages::

        layers.0                1x1 conv, (n_fft+2)*C -> c_mults[0] * enc_dim
        layers.1, layers.2      ConvNeXt blocks
        layers.3                strided conv, stage 0 -> stage 1
        layers.4, layers.5      ConvNeXt blocks
        layers.6                strided conv, stage 1 -> stage 2
        layers.7, layers.8      ConvNeXt blocks
        layers.9                strided conv, stage 2 -> stage 2
        layers.10               1x1 conv, c_mults[-1] * enc_dim -> enc_latent_dim

    HF class: ``Cosmos3AudioSpectrogramConvNeXtEncoder``.

    Inputs: ``audio`` ``(B, encoder_input_channels, N)``.
    Outputs: ``(B, T, enc_latent_dim)`` with ``T = N / (prod(enc_strides) * enc_hop_length)``
    — channels-last, matching the upstream ``forward`` return.
    """

    def __init__(self, config: Cosmos3AudioConfig):
        super().__init__()
        config.validate()
        self._config = config
        dtype = config.dtype
        channels = config.enc_dim
        multiples = config.enc_c_mults
        strides = config.enc_strides

        self.input_channels = config.encoder_input_channels
        self.n_fft = config.enc_n_fft
        self.hop_length = config.enc_hop_length

        layers: list[nn.Module] = [
            # Packed real/imag spectrogram bins -> first stage width.
            _Conv1d(
                config.spectrogram_channels,
                multiples[0] * channels,
                1,
                bias=False,
                dtype=dtype,
            )
        ]
        for index, stride in enumerate(strides):
            input_dim = multiples[index] * channels
            # The last stage keeps its width (there is no multiples[index + 1]).
            output_dim = (
                multiples[index + 1] * channels
                if index < len(multiples) - 1
                else multiples[-1] * channels
            )
            for _ in range(config.enc_num_blocks):
                layers.append(
                    Cosmos3AudioConvNeXtBlock(
                        input_dim,
                        input_dim * 4,
                        use_snake=config.enc_use_snake,
                        snake_logscale=config.snake_logscale,
                        dtype=dtype,
                    )
                )
            # Downsample by `stride`: k = 2 * stride, pad = ceil(stride / 2).
            layers.append(
                _Conv1d(
                    input_dim,
                    output_dim,
                    2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                    dtype=dtype,
                )
            )
        layers.append(
            _Conv1d(
                multiples[-1] * channels, config.enc_latent_dim, 1, bias=False, dtype=dtype
            )
        )
        self.layers = nn.Sequential(*layers)

        # Periodic Hann window, identical to torch.hann_window(n_fft).
        window = np.hanning(self.n_fft + 1)[:-1].astype(np.float32)
        self._window = ir.tensor(window, name="stft_hann_window")

    def spectrogram(self, op: OpBuilder, audio: ir.Value) -> ir.Value:
        """Compute the packed real/imaginary STFT front-end.

        Replicates ``_spectrogram`` + the channel packing in the upstream
        ``forward``. The transform is fixed (no learnable parameters) but it
        lives inside ``encoder.forward`` upstream, so it is kept inside the
        ONNX graph to preserve the documented ``waveform -> moments`` contract.

        .. note::
            ``STFT`` has no CUDA kernel in onnxruntime; on the CUDA EP this
            node falls back to CPU and introduces a host/device copy at the
            graph entry. Callers that want a pure-GPU encoder should feed the
            spectrogram in directly instead.

        Args:
            audio: ``(B, C, N)`` waveform.

        Returns:
            ``(B, C * (n_fft + 2), N / hop_length)`` spectrogram, real bins of
            each channel followed by that channel's imaginary bins.
        """
        config = self._config
        channels = self.input_channels

        # (B, C, N) -> (B * C, N): the STFT is applied per waveform channel.
        num_samples = op.Shape(audio, start=2, end=3)
        flat = audio
        if channels > 1:
            flat_shape = op.Concat(op.Constant(value_ints=[-1]), num_samples, axis=0)
            flat = op.Reshape(flat, flat_shape)
        else:
            flat = op.Squeeze(flat, op.Constant(value_ints=[1]))

        # torch.stft(center=False) after an explicit symmetric pad.
        padded = op.Pad(
            flat,
            op.Constant(value_ints=[0, config.stft_pad_left, 0, config.stft_pad_right]),
        )
        # The reference computes the transform in float32 regardless of dtype.
        padded = op.Cast(padded, to=ir.DataType.FLOAT)

        # ONNX STFT wants a trailing "real signal" axis: (B*C, N', 1).
        signal = op.Unsqueeze(padded, op.Constant(value_ints=[-1]))
        spec = op.STFT(
            signal,
            op.Constant(value=ir.tensor(np.array(self.hop_length, dtype=np.int64))),
            op.Constant(value=self._window),
            op.Constant(value=ir.tensor(np.array(self.n_fft, dtype=np.int64))),
            onesided=1,
        )
        # (B*C, frames, bins, 2) -> (B*C, 2, bins, frames) so that a flatten of
        # axes 1-2 yields [real bins ..., imaginary bins ...] like
        # torch.cat([real, imaginary], dim=1).
        spec = op.Transpose(spec, perm=[0, 3, 2, 1])
        num_frames = op.Shape(spec, start=3, end=4)
        packed_shape = op.Concat(
            op.Constant(value_ints=[-1, config.enc_n_fft + 2]), num_frames, axis=0
        )
        spec = op.Reshape(spec, packed_shape)

        spec = op.CastLike(spec, audio)
        if channels > 1:
            # (B*C, n_fft+2, frames) -> (B, C*(n_fft+2), frames)
            merged_shape = op.Concat(
                op.Constant(value_ints=[-1, config.spectrogram_channels]), num_frames, axis=0
            )
            spec = op.Reshape(spec, merged_shape)
        return spec

    def forward(self, op: OpBuilder, audio: ir.Value) -> ir.Value:
        """Encode a waveform into channels-last latent moments.

        Args:
            audio: ``(B, encoder_input_channels, N)`` waveform.

        Returns:
            ``(B, T, enc_latent_dim)`` moments.
        """
        spec = self.spectrogram(op, audio)
        # (B, spectrogram_channels, T) -> (B, enc_latent_dim, T)
        hidden_states = self.layers(op, spec)
        # Upstream returns channels-last; the tokenizer transposes it back.
        return op.Transpose(hidden_states, perm=[0, 2, 1])


# ---------------------------------------------------------------------------
# Decoder: Oobleck
# ---------------------------------------------------------------------------


class Cosmos3AudioResidualUnit(nn.Module):
    """Oobleck residual unit: Snake → dilated k=7 conv → Snake → k=1 conv.

    HF class: ``Cosmos3AudioResidualUnit``.
    """

    def __init__(
        self,
        dimension: int = 16,
        dilation: int = 1,
        *,
        snake_logscale: bool = True,
        dtype: ir.DataType = ir.DataType.FLOAT,
    ):
        super().__init__()
        # pad = ((7 - 1) * dilation) // 2 exactly preserves the temporal length,
        # so the upstream centre-crop of the skip branch is always a no-op.
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = _Snake1d(dimension, snake_logscale, dtype)
        self.conv1 = _Conv1d(
            dimension, dimension, 7, dilation=dilation, padding=pad, dtype=dtype
        )
        self.snake2 = _Snake1d(dimension, snake_logscale, dtype)
        self.conv2 = _Conv1d(dimension, dimension, 1, dtype=dtype)

    def forward(self, op: OpBuilder, hidden_state: ir.Value) -> ir.Value:
        """Apply the residual unit to ``(B, C, T)``; shape is preserved."""
        output = self.conv1(op, self.snake1(op, hidden_state))
        output = self.conv2(op, self.snake2(op, output))
        return op.Add(hidden_state, output)


class Cosmos3AudioDecoderBlock(nn.Module):
    """Oobleck decoder block: Snake → transposed conv → 3 dilated residual units.

    HF class: ``Cosmos3AudioDecoderBlock``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int = 1,
        output_padding: int = 0,
        *,
        snake_logscale: bool = True,
        dtype: ir.DataType = ir.DataType.FLOAT,
    ):
        super().__init__()
        self.snake1 = _Snake1d(input_dim, snake_logscale, dtype)
        self.conv_t1 = _ConvTranspose1d(
            input_dim,
            output_dim,
            2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
            output_padding=output_padding,
            dtype=dtype,
        )
        self.res_unit1 = Cosmos3AudioResidualUnit(
            output_dim, dilation=1, snake_logscale=snake_logscale, dtype=dtype
        )
        self.res_unit2 = Cosmos3AudioResidualUnit(
            output_dim, dilation=3, snake_logscale=snake_logscale, dtype=dtype
        )
        self.res_unit3 = Cosmos3AudioResidualUnit(
            output_dim, dilation=9, snake_logscale=snake_logscale, dtype=dtype
        )

    def forward(self, op: OpBuilder, hidden_state: ir.Value) -> ir.Value:
        """Upsample ``(B, C_in, T)`` to ``(B, C_out, T * stride)``."""
        hidden_state = self.snake1(op, hidden_state)
        hidden_state = self.conv_t1(op, hidden_state)
        hidden_state = self.res_unit1(op, hidden_state)
        hidden_state = self.res_unit2(op, hidden_state)
        return self.res_unit3(op, hidden_state)


class Cosmos3AudioDecoder(nn.Module):
    """Oobleck-style latent → waveform decoder.

    Layout::

        conv1        k=7 conv, vocoder_input_dim -> dec_dim * ([1] + dec_c_mults)[-1]
        block.{i}    Cosmos3AudioDecoderBlock, stride = reversed(dec_strides)[i]
        snake1       SnakeBeta at dec_dim
        conv2        k=7 conv, dec_dim -> dec_out_channels (no bias)

    HF class: ``Cosmos3AudioDecoder``.

    Inputs: ``(B, vocoder_input_dim, T)``.
    Outputs: ``(B, dec_out_channels, T * prod(dec_strides))``.
    """

    def __init__(self, config: Cosmos3AudioConfig):
        super().__init__()
        config.validate()
        self._config = config
        dtype = config.dtype
        channels = config.dec_dim
        strides = config.decoder_upsampling_ratios
        multiples = config.decoder_channel_multiples

        self.conv1 = _Conv1d(
            config.vocoder_input_dim, channels * multiples[-1], 7, padding=3, dtype=dtype
        )

        # Walk the multiplier table from widest to narrowest as we upsample.
        blocks: list[nn.Module] = []
        for stride_index, stride in enumerate(strides):
            blocks.append(
                Cosmos3AudioDecoderBlock(
                    input_dim=channels * multiples[len(strides) - stride_index],
                    output_dim=channels * multiples[len(strides) - stride_index - 1],
                    stride=stride,
                    output_padding=stride % 2,
                    snake_logscale=config.snake_logscale,
                    dtype=dtype,
                )
            )
        self.block = nn.ModuleList(blocks)

        self.snake1 = _Snake1d(channels, config.snake_logscale, dtype)
        self.conv2 = _Conv1d(
            channels, config.dec_out_channels, 7, padding=3, bias=False, dtype=dtype
        )

    def forward(self, op: OpBuilder, hidden_state: ir.Value) -> ir.Value:
        """Decode ``(B, z, T)`` latents into ``(B, audio_channels, T * hop)``."""
        # (B, z, T) -> (B, dec_dim * mults[-1], T)
        hidden_state = self.conv1(op, hidden_state)
        for layer in self.block:
            hidden_state = layer(op, hidden_state)
        # (B, dec_dim, T * hop) -> (B, audio_channels, T * hop)
        hidden_state = self.snake1(op, hidden_state)
        return self.conv2(op, hidden_state)


# ---------------------------------------------------------------------------
# VAE bottleneck
# ---------------------------------------------------------------------------


class Cosmos3AudioVAEBottleneck(nn.Module):
    """Parameter-free diagonal-Gaussian bottleneck (``bottleneck_type="vae"``).

    Replicates ``OobleckDiagonalGaussianDistribution``: the encoder moments are
    split in half along the channel axis into ``mean`` and ``scale``, and the
    standard deviation is ``softplus(scale) + 1e-4``.

    Sampling (``mean + std * eps``) is deliberately *not* emitted — the ONNX
    graph stays deterministic and the caller draws ``eps`` itself. ``mode()``
    is simply ``mean``.
    """

    def __init__(self, latent_channels: int):
        super().__init__()
        self._latent_channels = latent_channels

    def forward(
        self,
        op: OpBuilder,
        moments: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        """Split moments into ``(mean, std)``.

        Args:
            moments: ``(B, 2 * z, T)`` encoder output.

        Returns:
            ``(mean, std)``, each ``(B, z, T)``.
        """
        split = op.Constant(value_ints=[self._latent_channels, self._latent_channels])
        mean, scale = op.Split(moments, split, axis=1, _outputs=2)
        # std = softplus(scale) + 1e-4
        std = op.Add(op.Softplus(scale), _scalar(op, _POSTERIOR_STD_EPS, scale))
        return mean, std


# ---------------------------------------------------------------------------
# Weight-norm folding
# ---------------------------------------------------------------------------


def fold_weight_norm(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Recombine PyTorch ``weight_norm`` pairs into a single dense weight.

    ``torch.nn.utils.weight_norm(module, dim=0)`` stores a magnitude ``g`` and a
    direction ``v``; the effective weight is ``g * v / ||v||`` where the norm is
    taken over every axis except axis 0. ONNX ``Conv``/``ConvTranspose`` accept
    only the combined tensor, so the fold happens here, at weight-load time.

    Both spellings are handled:

    * legacy — ``<path>.weight_g`` / ``<path>.weight_v`` (what the published
      Cosmos3 ``sound_tokenizer`` checkpoint uses);
    * ``torch.nn.utils.parametrizations.weight_norm`` —
      ``<path>.parametrizations.weight.original0`` / ``...original1``.

    Keys that are already dense are passed through untouched.

    Args:
        state_dict: Raw checkpoint tensors.

    Returns:
        A new dict with ``<path>.weight`` in place of each ``g``/``v`` pair.

    Raises:
        ValueError: If a magnitude tensor has no matching direction tensor.
    """
    suffix_pairs = (
        (".weight_g", ".weight_v", ".weight"),
        (
            ".parametrizations.weight.original0",
            ".parametrizations.weight.original1",
            ".weight",
        ),
    )
    folded: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        matched = False
        for g_suffix, v_suffix, target_suffix in suffix_pairs:
            if not key.endswith(g_suffix):
                continue
            prefix = key[: -len(g_suffix)]
            v_key = prefix + v_suffix
            if v_key not in state_dict:
                raise ValueError(
                    f"Weight-norm magnitude {key!r} has no matching direction tensor {v_key!r}."
                )
            direction = state_dict[v_key]
            # norm over every axis except axis 0, keeping (out, 1, 1) broadcast shape
            reduce_dims = tuple(range(1, direction.dim()))
            norm = direction.float().pow(2).sum(dim=reduce_dims, keepdim=True).sqrt()
            weight = value.float() * direction.float() / norm
            folded[prefix + target_suffix] = weight.to(direction.dtype)
            matched = True
            break
        if matched:
            continue
        # Direction tensors are consumed alongside their magnitude tensor.
        if any(key.endswith(v_suffix) for _, v_suffix, _ in suffix_pairs):
            continue
        folded[key] = value

    return folded


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------


class Cosmos3AVAEAudioDecoderOnlyTokenizer(nn.Module):
    """Decoder-only Cosmos3 AVAE sound tokenizer.

    Use this path when the checkpoint ships without ``encoder.*`` weights (the
    common case for sound *generation*). No encoder or bottleneck sub-module is
    created, so the exported graph can never contain an initializer that has no
    weight to fill it.

    HF class: ``Cosmos3AVAEAudioTokenizer`` with ``encoder_enabled=False``.
    """

    #: Whether :meth:`encode` is available on this module.
    encoder_available: bool = False

    def __init__(self, config: Cosmos3AudioConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.decoder = Cosmos3AudioDecoder(config)

    def decode(self, op: OpBuilder, latents: ir.Value) -> ir.Value:
        """Decode sound latents into a waveform.

        Args:
            latents: ``(B, vocoder_input_dim, T)`` diffusion-model latents.

        Returns:
            ``(B, dec_out_channels, T * hop_size)`` waveform clamped to
            ``[-1, 1]``.
        """
        audio = self.decoder(op, latents)
        return op.Clip(audio, _scalar(op, -1.0, audio), _scalar(op, 1.0, audio))

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Fold ``weight_norm`` pairs; module paths are already HF-aligned."""
        return fold_weight_norm(state_dict)


class Cosmos3AVAEAudioTokenizer(Cosmos3AVAEAudioDecoderOnlyTokenizer):
    """Full encoder + VAE bottleneck + decoder Cosmos3 AVAE sound tokenizer.

    HF class: ``Cosmos3AVAEAudioTokenizer`` with ``encoder_enabled=True``.
    """

    encoder_available: bool = True

    def __init__(self, config: Cosmos3AudioConfig):
        super().__init__(config)
        self.encoder = Cosmos3AudioSpectrogramConvNeXtEncoder(config)
        self.bottleneck = Cosmos3AudioVAEBottleneck(config.latent_channels)

    def normalize_volume(self, op: OpBuilder, waveform: ir.Value) -> ir.Value:
        """Peak-normalize a waveform, matching ``encode``'s pre-processing.

        ``x / (|x|.max() + 1e-5) * 0.95`` where the maximum is taken over the
        **entire tensor** (batch included), exactly as upstream. Callers that
        need per-sample normalization must batch size 1.

        Args:
            waveform: ``(B, C, N)`` waveform.

        Returns:
            The peak-normalized waveform, same shape.
        """
        peak = op.ReduceMax(op.Abs(waveform), keepdims=0)
        scaled = op.Div(waveform, op.Add(peak, _scalar(op, _VOLUME_EPS, waveform)))
        return op.Mul(scaled, _scalar(op, _VOLUME_PEAK, waveform))

    def pad_to_hop_size(self, op: OpBuilder, waveform: ir.Value) -> ir.Value:
        """Right zero-pad a waveform to a whole multiple of ``hop_size``.

        Mirrors the inference-mode padding in ``encode`` so that the number of
        latent frames is exactly ``ceil(N / hop_size)``.

        Args:
            waveform: ``(B, C, N)`` waveform.

        Returns:
            ``(B, C, ceil(N / hop) * hop)`` waveform.
        """
        hop = op.Constant(value=ir.tensor(np.array([self.config.resolved_hop_size], np.int64)))
        num_samples = op.Shape(waveform, start=2, end=3)
        # padding = (hop - (N % hop)) % hop
        remainder = op.Mod(num_samples, hop)
        padding = op.Mod(op.Sub(hop, remainder), hop)
        pads = op.Concat(op.Constant(value_ints=[0, 0, 0, 0, 0]), padding, axis=0)
        return op.Pad(waveform, pads)

    def encode(
        self,
        op: OpBuilder,
        sample: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        """Encode a waveform into deterministic posterior moments.

        Reproduces ``Cosmos3AVAEAudioTokenizer.encode`` in evaluation mode
        (``force_pad`` semantics): optional peak normalization, optional
        right-padding to ``hop_size``, the STFT ConvNeXt encoder, then the VAE
        bottleneck split. No sample is drawn.

        Args:
            sample: ``(B, encoder_input_channels, N)`` waveform.

        Returns:
            ``(moments, mean, std)`` where ``moments`` is ``(B, 2 * z, T)`` and
            ``mean``/``std`` are ``(B, z, T)``.
        """
        hidden_states = sample
        if self.config.normalize_volume:
            hidden_states = self.normalize_volume(op, hidden_states)
        hidden_states = self.pad_to_hop_size(op, hidden_states)

        # Upstream `_encode` transposes the channels-last encoder output back to
        # channels-first before handing it to the Gaussian distribution.
        encoded = self.encoder(op, hidden_states)
        moments = op.Transpose(encoded, perm=[0, 2, 1])
        mean, std = self.bottleneck(op, moments)
        return moments, mean, std


def create_cosmos3_avae_audio_tokenizer(
    config: Cosmos3AudioConfig,
) -> Cosmos3AVAEAudioDecoderOnlyTokenizer:
    """Instantiate the tokenizer variant that matches ``config.encoder_enabled``.

    Args:
        config: A validated :class:`Cosmos3AudioConfig`. Use
            :meth:`Cosmos3AudioConfig.with_encoder_from_state_dict` first when
            encoder presence must be derived from the checkpoint.

    Returns:
        :class:`Cosmos3AVAEAudioTokenizer` when the encoder is enabled,
        otherwise :class:`Cosmos3AVAEAudioDecoderOnlyTokenizer`.
    """
    if config.encoder_enabled:
        return Cosmos3AVAEAudioTokenizer(config)
    return Cosmos3AVAEAudioDecoderOnlyTokenizer(config)
