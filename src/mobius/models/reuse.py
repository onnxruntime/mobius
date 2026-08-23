# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""RE-USE / SEMamba universal speech enhancement (NVIDIA).

Replicates the forward pass of NVIDIA's ``SEMamba`` generator as published
in `nvidia/RE-USE <https://huggingface.co/nvidia/RE-USE>`_.  The exported
ONNX graph consumes the magnitude and phase of a noisy STFT and produces
the enhanced magnitude, phase, and complex spectrogram.  The STFT/ISTFT
themselves stay outside the graph, matching how every other audio model in
mobius is exported.

Pipeline (matching ``SEMamba.forward``)::

    noisy_mag [B, F, T], noisy_pha [B, F, T]
      -> stack as 2 channels, zero-pad time and freq by 2  [B, 2, T+2, F+2]
      -> DenseEncoder (1x1 conv, dilated dense block, strided conv)
                                                           [B, C, T', F']
      -> num_tfmamba x TFMambaBlock (bidirectional Mamba over time,
         then over frequency)                              [B, C, T', F']
      -> MagDecoder   -> denoised_mag [B, F, T]
      -> PhaseDecoder -> denoised_pha [B, F, T]
      -> denoised_com = stack(mag*cos(pha), mag*sin(pha))  [B, F, T, 2]

``T' = floor((T + 1) / 4) + 1`` and ``F' = floor((F - 1) / 2) + 1`` follow
from the encoder's ``stride=(4, 2)`` convolution; the decoders undo both
strides and the result is cropped back to the input ``(F, T)``.

The SSM layers are the original (Mamba1) selective scan run over the whole
sequence in both directions, so they use
:class:`~mobius.components.SequenceMambaBlock` rather than the decode-time
:class:`~mobius.components.MambaBlock`.

Reference implementation: ``models/generator_SEMamba_time_d4.py``,
``models/codec_module_time_d4.py``, and ``models/mamba_block2_SEMamba.py``
in the ``nvidia/RE-USE`` repository.
"""

from __future__ import annotations

import dataclasses
import math

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.components import Conv2d, LayerNorm, Linear, SequenceMambaBlock

# Slice "start" sentinel for a reverse (negative-step) slice.
_INT64_MIN = -9223372036854775808

# Frequency-axis geometry of the encoder, kept here because two places depend on
# it: ``SEMambaSpeechEnhancementModel.forward`` emits the tail pad, and
# ``DenseEncoder`` builds the strided convolution, while
# ``ReUseConfig.encoder_freq_bins`` predicts the resulting extent at build time.
# ``TestEncoderFreqBins`` pins the prediction against the graph the model
# actually builds, so these cannot drift apart silently.
_ENCODER_FREQ_TAIL_PAD = 2
_ENCODER_FREQ_KERNEL = 3
_ENCODER_FREQ_STRIDE = 2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ReUseConfig(BaseModelConfig):
    """Configuration for the RE-USE / SEMamba speech-enhancement model.

    Fields mirror the ``model_cfg`` and ``stft_cfg`` sections of the
    ``nvidia/RE-USE`` ``config.json``.
    """

    # --- model_cfg ---
    #: Input channels of the encoder (magnitude + phase).
    input_channel: int = 2
    #: Output channels of each decoder head.
    output_channel: int = 1
    #: Encoder/decoder feature width (Mamba ``d_model``).
    hid_feature: int = 64
    #: Number of stacked time-frequency Mamba blocks.
    num_tfmamba: int = 30
    #: SSM state dimension.
    d_state: int = 16
    #: Causal Conv1D kernel size inside each Mamba block.
    d_conv: int = 4
    #: Inner expansion factor (``d_inner = expand * hid_feature``).
    expand: int = 4
    #: Depth of each dense block (number of dilated convolutions).
    dense_depth: int = 4
    #: Epsilon for the instance / layer normalizations.
    norm_epsilon: float = 1e-5

    # --- stft_cfg ---
    #: FFT size; the frequency axis has ``n_fft // 2 + 1`` bins.
    n_fft: int = 320
    #: STFT hop size, in samples.
    hop_size: int = 40
    #: STFT window size, in samples.
    win_size: int = 320
    #: Audio sample rate the model was trained for.
    sampling_rate: int = 8000
    #: Magnitude compression applied before the model (informational only —
    #: it is applied by the STFT front-end, outside this graph).
    compress_factor: str = "relu_log1p"

    model_type: str | None = "reuse"

    @property
    def num_freq_bins(self) -> int:
        """Number of frequency bins on the model's input/output."""
        return self.n_fft // 2 + 1

    @property
    def encoder_freq_bins(self) -> int:
        """Frequency extent of the encoder output, i.e. what the TF blocks see.

        Statically derivable, which is the point of computing it here rather than
        with ``Shape`` at run time: the frequency axis is a fixed function of
        ``n_fft``, so an execution provider can claim the frequency-axis scans even
        when the time axis is fully dynamic.

        The derivation mirrors what the graph actually does, in order:

        1. ``forward`` zero-pads the tail of the frequency axis by
           ``_ENCODER_FREQ_TAIL_PAD``, so the strided convolution never drops a
           partial window.
        2. ``DenseEncoder.dense_conv_2`` is a ``kernel=(1, 3)``, ``stride=(4, 2)``
           convolution with no padding, giving the usual
           ``floor((in - kernel) / stride) + 1``.

        ``n_fft`` is even, so ``num_freq_bins`` is odd and the floor is exact.
        The decoder's ``up_conv1`` doubles this back to ``2 * encoder_freq_bins``,
        which is ``>= num_freq_bins``; ``forward`` crops off the overshoot.
        """
        padded = self.num_freq_bins + _ENCODER_FREQ_TAIL_PAD
        return (padded - _ENCODER_FREQ_KERNEL) // _ENCODER_FREQ_STRIDE + 1

    @property
    def d_inner(self) -> int:
        """Expanded Mamba inner dimension."""
        return self.expand * self.hid_feature

    @property
    def dt_rank(self) -> int:
        """Rank of the SSM time-step projection (``mamba_ssm`` "auto")."""
        return math.ceil(self.hid_feature / 16)

    def validate(self) -> None:
        if self.hid_feature <= 0:
            raise ValueError("hid_feature must be positive")
        if self.num_tfmamba <= 0:
            raise ValueError("num_tfmamba must be positive")
        if self.n_fft <= 0 or self.n_fft % 2 != 0:
            raise ValueError("n_fft must be a positive even number")

    @classmethod
    def from_json(cls, cfg: dict) -> ReUseConfig:
        """Build a config from the parsed ``nvidia/RE-USE`` ``config.json``."""
        model_cfg = cfg.get("model_cfg", {})
        stft_cfg = cfg.get("stft_cfg", {})
        return cls(
            input_channel=int(model_cfg.get("input_channel", 2)),
            output_channel=int(model_cfg.get("output_channel", 1)),
            hid_feature=int(model_cfg.get("hid_feature", 64)),
            num_tfmamba=int(model_cfg.get("num_tfmamba", 4)),
            d_state=int(model_cfg.get("d_state", 16)),
            d_conv=int(model_cfg.get("d_conv", 4)),
            expand=int(model_cfg.get("expand", 4)),
            norm_epsilon=float(model_cfg.get("norm_epsilon", 1e-5)),
            n_fft=int(stft_cfg.get("n_fft", 320)),
            hop_size=int(stft_cfg.get("hop_size", 40)),
            win_size=int(stft_cfg.get("win_size", 320)),
            sampling_rate=int(stft_cfg.get("sampling_rate", 8000)),
            compress_factor=str(model_cfg.get("compress_factor", "relu_log1p")),
            model_type="reuse",
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "nvidia/RE-USE",
        *,
        revision: str | None = None,
    ) -> ReUseConfig:
        """Read ``config.json`` from a local directory or the HuggingFace Hub.

        RE-USE ships a bespoke ``config.json`` with no ``model_type`` or
        ``architectures`` field, so ``transformers.AutoConfig`` cannot read
        it and the generic :func:`mobius.build` entry point does not apply.
        """
        import json
        import os

        local = os.path.join(model_id, "config.json")
        if os.path.isfile(local):
            path = local
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_id, "config.json", revision=revision)
        with open(path) as f:
            return cls.from_json(json.load(f))


# ---------------------------------------------------------------------------
# Low-level ONNX helpers
# ---------------------------------------------------------------------------


class _InstanceNorm2d(nn.Module):
    """``torch.nn.InstanceNorm2d(affine=True, track_running_stats=False)``.

    ONNX's ``InstanceNormalization`` normalizes each (batch, channel) plane
    over its spatial extent using the batch's own statistics, which is
    exactly PyTorch's behaviour when running statistics are disabled.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter([num_features])
        self.bias = nn.Parameter([num_features])
        self._eps = eps

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return op.InstanceNormalization(x, self.weight, self.bias, epsilon=self._eps)


class _PReLU2d(nn.Module):
    """``torch.nn.PReLU(num_parameters=channels)`` for NCHW activations."""

    def __init__(self, num_parameters: int):
        super().__init__()
        self.weight = nn.Parameter([num_parameters])

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # slope: (C,) → (C, 1, 1) so it broadcasts across H and W but not N.
        return op.PRelu(x, op.Unsqueeze(self.weight, [-1, -2]))


def _norm_act_stage(conv: nn.Module, channels: int, eps: float) -> nn.ModuleList:
    """``Sequential(conv, InstanceNorm2d, PReLU)`` shared by encoder/decoder.

    A plain :class:`~onnxscript.nn.ModuleList` (rather than a wrapper module)
    keeps the checkpoint's ``nn.Sequential`` indices — ``.0``/``.1``/``.2`` —
    as the parameter-name segments, so no weight renaming is needed.
    """
    return nn.ModuleList([conv, _InstanceNorm2d(channels, eps), _PReLU2d(channels)])


def _apply_stage(op: OpBuilder, stage: nn.ModuleList, x: ir.Value) -> ir.Value:
    """Run a :func:`_norm_act_stage` list in order."""
    for layer in stage:
        x = layer(op, x)
    return x


def _atan2(op: OpBuilder, y: ir.Value, x: ir.Value) -> ir.Value:
    """Two-argument arctangent, matching ``torch.atan2``.

    ONNX has no ``Atan2``, so the quadrant correction is spelled out:
    ``atan(y/x)`` is only valid for ``x > 0``, needs a ``±pi`` shift for
    ``x < 0``, and degenerates to ``±pi/2`` on the ``x == 0`` axis.

    Signed zeros are not distinguished: ``atan2(-0.0, x<0)`` returns ``+pi``
    where IEEE specifies ``-pi``.  Both name the same angle, so the phase
    decoder's ``cos``/``sin`` consumers cannot tell them apart.
    """
    zero = op.CastLike(op.Constant(value_float=0.0), x)
    one = op.CastLike(op.Constant(value_float=1.0), x)
    pi = op.CastLike(op.Constant(value_float=math.pi), x)
    half_pi = op.CastLike(op.Constant(value_float=math.pi / 2.0), x)

    x_is_zero = op.Equal(x, zero)
    # Substitute 1 for x where it is zero so the division never produces a
    # NaN/Inf that a later Where could not mask out.
    safe_x = op.Where(x_is_zero, one, x)
    base = op.Atan(op.Div(y, safe_x))

    # On the x == 0 axis the angle is sign(y) * pi/2 (and 0 at the origin,
    # which Sign(0) == 0 gives for free).
    base = op.Where(x_is_zero, op.Mul(op.Sign(y), half_pi), base)

    # For x < 0, atan(y/x) lands in the wrong half-plane: shift by +pi when
    # y >= 0 and by -pi otherwise.
    shift = op.Where(
        op.Less(x, zero),
        op.Where(op.GreaterOrEqual(y, zero), pi, op.Neg(pi)),
        zero,
    )
    return op.Add(base, shift)


class _SPConvTranspose2d(nn.Module):
    """Sub-pixel "transposed" convolution used by both decoders.

    Produces ``out_channels * r`` maps with an ordinary convolution and then
    interleaves the ``r`` copies along the last axis, expanding it by ``r``.
    The input is first padded by one column on each side of the last axis so
    the ``(1, 3)`` kernel preserves that axis before expansion.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size, r: int):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels * r, kernel_size)
        self._out_channels = out_channels
        self._r = r

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # Pad last axis by 1 on each side: pads = [begin..., end...] over 4 dims.
        x = op.Pad(x, op.Constant(value_ints=[0, 0, 0, 1, 0, 0, 0, 1]))
        out = self.conv(op, x)  # (B, out_channels * r, H, W)

        batch = op.Shape(out, start=0, end=1)
        height = op.Shape(out, start=2, end=3)
        width = op.Shape(out, start=3, end=4)

        # (B, r, C, H, W) -> (B, C, H, W, r) -> (B, C, H, W * r).
        out = op.Reshape(
            out,
            op.Concat(
                batch,
                op.Constant(value_ints=[self._r, self._out_channels]),
                height,
                width,
                axis=0,
            ),
        )
        out = op.Transpose(out, perm=[0, 2, 3, 4, 1])
        return op.Reshape(
            out,
            op.Concat(
                batch,
                op.Constant(value_ints=[self._out_channels]),
                height,
                op.Constant(value_ints=[-1]),
                axis=0,
            ),
        )


# ---------------------------------------------------------------------------
# Dense encoder / decoders
# ---------------------------------------------------------------------------


class DenseBlock(nn.Module):
    """Densely connected stack of dilated 2D convolutions.

    Each step convolves the concatenation of all previous outputs, so the
    ``i``-th convolution reads ``hid_feature * (i + 1)`` channels.  Dilation
    doubles per step along the time axis, widening the receptive field
    without changing the spatial dimensions.
    """

    def __init__(self, config: ReUseConfig, depth: int = 4):
        super().__init__()
        self._depth = depth
        hid = config.hid_feature
        blocks = []
        for i in range(depth):
            dilation = 2**i
            conv = Conv2d(
                hid * (i + 1),
                hid,
                kernel_size=(3, 3),
                # ONNX pad order is [top, left, bottom, right]; the reference
                # uses PyTorch padding=(dilation, 1) on a (3, 3) kernel.
                padding=(dilation, 1, dilation, 1),
                dilation=(dilation, 1),
            )
            blocks.append(_norm_act_stage(conv, hid, config.norm_epsilon))
        self.dense_block = nn.ModuleList(blocks)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        skip = x
        for block in self.dense_block:
            # x: (B, hid, H, W); skip grows by hid channels each step.
            x = _apply_stage(op, block, skip)
            skip = op.Concat(x, skip, axis=1)
        return x


class DenseEncoder(nn.Module):
    """Channel lift → dense block → strided downsample.

    ``dense_conv_2`` strides the time axis by 4 and the frequency axis by 2,
    which is what makes the Mamba stack affordable.
    """

    def __init__(self, config: ReUseConfig):
        super().__init__()
        hid = config.hid_feature
        eps = config.norm_epsilon
        self.dense_conv_1 = _norm_act_stage(
            Conv2d(config.input_channel, hid, kernel_size=(1, 1)), hid, eps
        )
        self.dense_block = DenseBlock(config, depth=config.dense_depth)
        self.dense_conv_2 = _norm_act_stage(
            Conv2d(
                hid,
                hid,
                kernel_size=(1, _ENCODER_FREQ_KERNEL),
                stride=(4, _ENCODER_FREQ_STRIDE),
            ),
            hid,
            eps,
        )

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        x = _apply_stage(op, self.dense_conv_1, x)  # (B, hid, T, F)
        x = self.dense_block(op, x)  # (B, hid, T, F)
        return _apply_stage(op, self.dense_conv_2, x)  # (B, hid, T', F')


class _Decoder(nn.Module):
    """Shared decoder trunk: dense block → freq upsample → time upsample.

    ``up_conv1`` expands the frequency axis by 2; ``up_conv2`` is applied to
    a time/frequency-transposed view so that it expands the time axis by 4,
    undoing the encoder's ``stride=(4, 2)``.
    """

    def __init__(self, config: ReUseConfig):
        super().__init__()
        hid = config.hid_feature
        eps = config.norm_epsilon
        self.dense_block = DenseBlock(config, depth=config.dense_depth)
        self.up_conv1 = _norm_act_stage(_SPConvTranspose2d(hid, hid, (1, 3), 2), hid, eps)
        self.up_conv2 = _norm_act_stage(_SPConvTranspose2d(hid, hid, (1, 3), 4), hid, eps)

    def _trunk(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        x = self.dense_block(op, x)
        x = _apply_stage(op, self.up_conv1, x)  # (B, hid, T', F' * 2)
        # Swap time and frequency so up_conv2 expands the time axis, then
        # swap back.
        x = op.Transpose(x, perm=[0, 1, 3, 2])
        x = _apply_stage(op, self.up_conv2, x)  # (B, hid, F' * 2, T' * 4)
        return op.Transpose(x, perm=[0, 1, 3, 2])


class MagDecoder(_Decoder):
    """Decoder head producing the enhanced magnitude spectrogram."""

    def __init__(self, config: ReUseConfig):
        super().__init__(config)
        self.final_conv = Conv2d(config.hid_feature, config.output_channel, kernel_size=(1, 1))

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return self.final_conv(op, self._trunk(op, x))  # (B, 1, T'*4, F'*2)


class PhaseDecoder(_Decoder):
    """Decoder head producing the enhanced phase via ``atan2(imag, real)``."""

    def __init__(self, config: ReUseConfig):
        super().__init__(config)
        self.phase_conv_r = Conv2d(
            config.hid_feature, config.output_channel, kernel_size=(1, 1)
        )
        self.phase_conv_i = Conv2d(
            config.hid_feature, config.output_channel, kernel_size=(1, 1)
        )

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        x = self._trunk(op, x)
        # Predict a unit-vector-like (real, imag) pair and read off its angle,
        # which keeps the output inherently wrapped to (-pi, pi].
        return _atan2(op, self.phase_conv_i(op, x), self.phase_conv_r(op, x))


# ---------------------------------------------------------------------------
# Bidirectional Mamba
# ---------------------------------------------------------------------------


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba over one axis, with a residual inside each branch.

    Runs the forward branch on the sequence and the backward branch on its
    reverse, re-reverses the latter, concatenates both, projects back to
    ``d_model``, and layer-normalizes.
    """

    def __init__(self, config: ReUseConfig):
        super().__init__()
        d_model = config.hid_feature
        self.forward_blocks = SequenceMambaBlock(
            d_model,
            config.d_inner,
            config.d_state,
            config.dt_rank,
            config.d_conv,
        )
        self.backward_blocks = SequenceMambaBlock(
            d_model,
            config.d_inner,
            config.d_state,
            config.dt_rank,
            config.d_conv,
        )
        self.output_proj = Linear(2 * d_model, d_model, bias=True)
        self.norm = LayerNorm(d_model, eps=1e-5)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: (batch, seq_len, d_model)
        out_fw = op.Add(self.forward_blocks(op, x), x)

        # Reverse along the sequence axis. ONNX has no Flip, and a
        # negative-step Slice is the standard spelling for it.
        #
        # ReverseSequence would also work and needs no negative-step support,
        # but it costs an extra Expand to build the per-row lengths and says
        # something the model does not mean: nothing here is padded, so every
        # row is reversed in full. Slice states the intent directly.
        x_rev = op.Slice(x, [-1], [_INT64_MIN], [1], [-1])
        out_bw = op.Add(self.backward_blocks(op, x_rev), x_rev)
        out_bw = op.Slice(out_bw, [-1], [_INT64_MIN], [1], [-1])

        out = op.Concat(out_fw, out_bw, axis=-1)  # (batch, seq_len, 2*d_model)
        return self.norm(op, self.output_proj(op, out))


class TFMambaBlock(nn.Module):
    """Bidirectional Mamba along time, then along frequency.

    The feature map ``(B, C, T, F)`` is folded so that each axis in turn
    becomes the sequence dimension of a :class:`BiMambaBlock`, with the other
    axis absorbed into the batch.
    """

    def __init__(self, config: ReUseConfig):
        super().__init__()
        self.time_mamba = BiMambaBlock(config)
        self.freq_mamba = BiMambaBlock(config)
        self._channels = config.hid_feature
        self._freq = config.encoder_freq_bins

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        channels = op.Constant(value_ints=[self._channels])
        batch = op.Shape(x, start=0, end=1)
        time = op.Shape(x, start=2, end=3)
        # The frequency extent is a fixed function of n_fft, so it is emitted as a
        # constant rather than read back with Shape. That keeps the frequency-axis
        # Scan's extent statically known even when the time axis is dynamic, which
        # is what lets an execution provider claim it — a Shape-derived extent
        # leaves the scan unclaimable and, since an unclaimed node is a partition
        # boundary, fragments the surrounding graph too.
        freq = op.Constant(value_ints=[self._freq])
        minus_one = op.Constant(value_ints=[-1])

        # --- Time branch: (B, C, T, F) -> (B*F, T, C) ---
        h = op.Transpose(x, perm=[0, 3, 2, 1])  # (B, F, T, C)
        h = op.Reshape(h, op.Concat(minus_one, time, channels, axis=0))
        h = op.Add(self.time_mamba(op, h), h)

        # --- Frequency branch: (B*F, T, C) -> (B*T, F, C) ---
        h = op.Reshape(h, op.Concat(batch, freq, time, channels, axis=0))
        h = op.Transpose(h, perm=[0, 2, 1, 3])  # (B, T, F, C)
        h = op.Reshape(h, op.Concat(minus_one, freq, channels, axis=0))
        h = op.Add(self.freq_mamba(op, h), h)

        # --- Back to (B, C, T, F) ---
        h = op.Reshape(h, op.Concat(batch, time, freq, channels, axis=0))
        return op.Transpose(h, perm=[0, 3, 1, 2])


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class SEMambaSpeechEnhancementModel(nn.Module):
    """NVIDIA RE-USE (SEMamba) universal speech enhancement generator.

    Consumes the magnitude and phase of a noisy STFT and predicts the
    enhanced magnitude, phase, and complex spectrogram.  The magnitude is
    expected to already be compressed by the STFT front-end (RE-USE uses
    ``log1p``); decompression and the ISTFT happen after this graph.
    """

    default_task: str = "speech-enhancement"
    category: str = "Audio"

    def __init__(self, config: ReUseConfig):
        super().__init__()
        self.config = config
        self.dense_encoder = DenseEncoder(config)
        self.TSMamba = nn.ModuleList([TFMambaBlock(config) for _ in range(config.num_tfmamba)])
        self.mask_decoder = MagDecoder(config)
        self.phase_decoder = PhaseDecoder(config)

    def forward(
        self,
        op: OpBuilder,
        noisy_mag: ir.Value,
        noisy_pha: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        """Enhance a noisy spectrogram.

        Args:
            op: ONNX op builder.
            noisy_mag: (batch, freq, time) — compressed noisy magnitude.
            noisy_pha: (batch, freq, time) — noisy phase, in radians.

        Returns:
            ``(denoised_mag, denoised_pha, denoised_com)`` with shapes
            ``(batch, freq, time)``, ``(batch, freq, time)`` and
            ``(batch, freq, time, 2)``.
        """
        num_freq = self.config.num_freq_bins
        # Remember the input time extent; the decoders overshoot it and the
        # result is cropped back at the end.
        time = op.Shape(noisy_mag, start=2, end=3)

        # (B, F, T) -> (B, 1, T, F) for each of magnitude and phase, then
        # stack them as the encoder's two input channels.
        mag = op.Unsqueeze(op.Transpose(noisy_mag, perm=[0, 2, 1]), [1])
        pha = op.Unsqueeze(op.Transpose(noisy_pha, perm=[0, 2, 1]), [1])
        x = op.Concat(mag, pha, axis=1)  # (B, 2, T, F)

        # Zero-pad the tail of both the time and frequency axes. The reference
        # does this so the strided encoder convolution never has to drop a
        # partial window. The frequency pad is part of how
        # ``ReUseConfig.encoder_freq_bins`` predicts the encoder's output extent.
        x = op.Pad(
            x,
            op.Constant(
                value_ints=[0, 0, 0, 0, 0, 0, _ENCODER_FREQ_TAIL_PAD, _ENCODER_FREQ_TAIL_PAD]
            ),
        )

        x = self.dense_encoder(op, x)  # (B, C, T', F')
        for block in self.TSMamba:
            x = block(op, x)

        # (B, 1, T'*4, F'*2) -> (B, F'*2, T'*4)
        denoised_mag = op.Squeeze(
            op.Transpose(self.mask_decoder(op, x), perm=[0, 3, 2, 1]), [-1]
        )
        denoised_pha = op.Squeeze(
            op.Transpose(self.phase_decoder(op, x), perm=[0, 3, 2, 1]), [-1]
        )

        # Crop the upsampled output back to the input (freq, time) extent.
        starts = op.Constant(value_ints=[0, 0])
        ends = op.Concat(op.Constant(value_ints=[num_freq]), time, axis=0)
        axes = op.Constant(value_ints=[1, 2])
        denoised_mag = op.Slice(denoised_mag, starts, ends, axes)
        denoised_pha = op.Slice(denoised_pha, starts, ends, axes)

        # Complex spectrogram as a trailing (real, imag) pair.
        denoised_com = op.Concat(
            op.Unsqueeze(op.Mul(denoised_mag, op.Cos(denoised_pha)), [-1]),
            op.Unsqueeze(op.Mul(denoised_mag, op.Sin(denoised_pha)), [-1]),
            axis=-1,
        )

        return denoised_mag, denoised_pha, denoised_com

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Map ``nvidia/RE-USE`` checkpoint names onto this module tree.

        The checkpoint stores the selective-scan parameters flat on each
        Mamba module (``forward_blocks.A_log``), while
        :class:`~mobius.components.SequenceMambaBlock` nests them under an
        ``ssm`` submodule (``forward_blocks.ssm.A_log``) — the same offset
        the Mamba causal-LM models correct for.  Everything else, including
        the encoder/decoder ``nn.Sequential`` indices, already lines up.

        The rename is idempotent: a state dict that already uses the nested
        names is returned unchanged, so re-running this (or loading an
        already-converted checkpoint) cannot produce ``ssm.ssm.A_log``.
        """
        renames = {}
        for key in list(state_dict):
            for param in _SSM_PARAMS:
                suffix = f".{param}"
                if not key.endswith(suffix):
                    continue
                prefix = key[: -len(suffix)]
                # Already nested — leave it alone.
                if prefix.endswith(".ssm") or prefix == "ssm":
                    break
                renames[key] = f"{prefix}.ssm{suffix}"
                break
        for old_key, new_key in renames.items():
            state_dict[new_key] = state_dict.pop(old_key)
        return state_dict


#: Selective-scan parameters the checkpoint keeps flat on the Mamba module
#: but :class:`~mobius.components.SequenceMambaBlock` nests under ``ssm``.
_SSM_PARAMS: tuple[str, ...] = (
    "A_log",
    "D",
    "x_proj.weight",
    "dt_proj.weight",
    "dt_proj.bias",
)


def build_reuse(
    model_id: str = "nvidia/RE-USE",
    *,
    revision: str | None = None,
    dtype: str | None = None,
    execution_provider: str = "default",
    load_weights: bool = True,
) -> ModelPackage:
    """Build an ONNX :class:`ModelPackage` for a RE-USE / SEMamba checkpoint.

    RE-USE is published with a bespoke ``config.json`` (no ``model_type``,
    no ``architectures``) and a ``PyTorchModelHubMixin`` checkpoint, so
    :func:`mobius.build` — which goes through ``transformers.AutoConfig`` —
    cannot discover it.  This function is the equivalent entry point.

    Args:
        model_id: Local directory or HuggingFace Hub repo holding
            ``config.json`` and ``model.safetensors``.
        revision: Optional Hub revision (branch, tag, or commit SHA) to pin
            downloads. Ignored for local directories.
        dtype: Override model dtype (e.g. ``"f16"``). Defaults to float32.
        execution_provider: Target execution provider for EP-aware
            optimizations.
        load_weights: When false, build the graph structure only.

    Returns:
        A :class:`ModelPackage` whose ``"model"`` entry is the enhancement
        network.
    """
    from mobius._builder import build_from_module, resolve_dtype
    from mobius.integrations._weight_loading import apply_weights

    config = ReUseConfig.from_pretrained(model_id, revision=revision)
    resolved = resolve_dtype(dtype)
    if resolved is not None:
        config.dtype = resolved

    module = SEMambaSpeechEnhancementModel(config)
    package = build_from_module(
        module,
        config,
        task="speech-enhancement",
        execution_provider=execution_provider,
    )
    if load_weights:
        apply_weights(
            package["model"],
            module.preprocess_weights(_load_state_dict(model_id, revision)),
        )
    return package


def _load_state_dict(model_id: str, revision: str | None) -> dict[str, torch.Tensor]:
    """Read ``model.safetensors`` from a local directory or the Hub."""
    import os

    from safetensors.torch import load_file

    local = os.path.join(model_id, "model.safetensors")
    if not os.path.isfile(local):
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(model_id, "model.safetensors", revision=revision)
    return load_file(local)
