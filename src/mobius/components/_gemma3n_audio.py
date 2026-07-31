# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma 3n audio encoder (Universal Speech Model conformer).

The 269 tensors under ``model.audio_tower.`` in ``google/gemma-3n-E4B-it``: a
two-stage strided 2-D convolutional subsampler ("SSCP", 5 tensors) followed by
12 conformer blocks (22 tensors each).  Unlike Gemma 4's audio tower there is
no output projection here — the projection into text space lives in
``model.embed_audio`` (:class:`~mobius.components.Gemma3nMultimodalEmbedder`).

HF reference: ``Gemma3nAudioEncoder`` and friends in
``transformers.models.gemma3n.modeling_gemma3n``.

Public component::

    Gemma3nAudioEncoder   - subsample -> 12 conformer blocks -> stride-4 reduce

Default config (google/gemma-3n-E4B-it ``audio_config``)::

    hidden_size=1536, conf_num_attention_heads=8, conf_num_hidden_layers=12,
    conf_attention_chunk_size=12, conf_attention_context_left=13,
    conf_attention_context_right=0, conf_attention_logit_cap=50.0,
    conf_conv_kernel_size=5, conf_reduction_factor=4, conf_residual_weight=0.5,
    input_feat_size=128, sscp_conv_channel_size=[128, 32],
    sscp_conv_kernel_size=[[3, 3], [3, 3]], sscp_conv_stride_size=[[2, 2], [2, 2]],
    sscp_conv_group_norm_eps=1e-3, gradient_clipping=1e10, rms_norm_eps=1e-6

Reuse from :mod:`mobius.components._gemma4_audio`: the conformer feed-forward
and light-conv1d sub-blocks are identical in both weight names and arithmetic,
so :class:`Gemma4FeedForward` and :class:`Gemma4LightConv1d` are instantiated
directly with ``linear_cls=Linear``.  Gemma 3n's checkpoint ships no learned
activation-clipping bounds, so the ``ClippableLinear`` default would demand
four initializers per projection that do not exist.

Three things genuinely diverge from Gemma 4 and are implemented here:

* **Cumulative group norm** (:class:`_CumulativeGroupNorm`) in the SSCP
  blocks — group statistics accumulated over the *time* axis, where Gemma 4
  uses a plain per-frame LayerNorm.
* **Reverse-causal time padding** ``(0, kernel_h - 1)`` plus fixed frequency
  padding ``(1, 1)`` in the SSCP convolutions, rather than symmetric padding.
* **Attention**: Gemma 3n has an explicit relative-position projection
  (``relative_position_embedding.pos_proj``), applies no rescaling to the keys,
  and admits ``conf_attention_context_left`` = 13 keys where
  :class:`Gemma4Attention` admits 12 — so that class cannot be subclassed.

**Chunked attention is flattened to full T×T attention here**, which is exactly
equivalent for offline inference — see :class:`_Gemma3nAudioAttention`.

**Mask polarity**: HF's ``audio_mel_mask`` is True for *padded* frames.  These
components follow the mobius convention instead (True = **valid**, matching
:class:`~mobius.components.Gemma4AudioEncoder` and every other encoder here),
so callers must not forward HF's mask unchanged.

Weight name alignment (HF -> ONNX), with ``model.audio_tower.`` stripped::

    subsample_conv_projection.conv_{0,1}.conv.weight     (unchanged)
    subsample_conv_projection.conv_{0,1}.norm.weight     (unchanged)
    subsample_conv_projection.input_proj_linear.weight   (unchanged)
    conformer.{i}.ffw_layer_{start,end}.*                (unchanged)
    conformer.{i}.attention.*                            (unchanged)
    conformer.{i}.lconv1d.*                              (unchanged)
    conformer.{i}.norm.weight                            (unchanged)

i.e. every learned tensor keeps its HF name; the only extra initializers are
the ``relative_position_embedding.sin_emb`` constants, which are derived from
hyperparameters and are not checkpoint tensors.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._common import INT64_MAX, Linear
from mobius.components._conv import Conv2dNoBias
from mobius.components._gemma4_audio import (
    Gemma4FeedForward,
    Gemma4LightConv1d,
    _gradient_clip,
)
from mobius.components._rms_norm import RMSNorm

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


# Fixed frequency-axis padding of the SSCP convolutions, on both edges. HF
# hard-codes it: it is what JAX 'SAME' works out to for the shipped 3-wide
# kernel at stride 2, and is not recomputed for other kernel/stride pairs.
_FREQ_PAD = 1

# Replacement score for blocked attention positions. Applied to float32 scores
# that the logit cap has already squeezed into [-cap, cap], so this is
# functionally -inf while staying finite (HF uses ``finfo.min``; both softmax
# to zero, and both leave an all-blocked row uniform).
_MASK_SCORE = -1e9


def _sscp_freq_out_dim(input_freq_dim: int, kernel_w: int, stride_w: int) -> int:
    """Frequency bins after one SSCP convolution stage.

    The flattened width of the subsampler's output projection
    (``input_proj_linear``) is ``channels * freq``, so the conv block, the
    projection and the tests all have to agree on this.
    """
    padded = input_freq_dim + 2 * _FREQ_PAD
    return (padded - kernel_w) // stride_w + 1


def _timing_signal(positions: np.ndarray, channels: int) -> np.ndarray:
    """Sinusoidal position embeddings for arbitrary (signed) *positions*.

    Matches ``Gemma3nAudioRelativePositionEmbedding._get_timing_signal_1d_pos``:
    ``channels // 2`` geometrically spaced timescales from 1 to 1e4, with the
    sines and cosines concatenated (not interleaved).

    Args:
        positions: 1-D position values, one per output row.
        channels: Embedding width; must be even.

    Returns:
        ``[len(positions), channels]`` float32.
    """
    num_timescales = channels // 2
    log_timescale_increment = math.log(1.0e4) / max(num_timescales - 1, 1)
    inv_timescales = np.exp(
        np.arange(num_timescales, dtype=np.float32) * -log_timescale_increment
    )
    scaled_time = positions.astype(np.float32)[:, None] * inv_timescales[None, :]
    return np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=-1).astype(
        np.float32
    )


class _CumulativeGroupNorm(nn.Module):
    """Group norm (one group) with statistics accumulated over the time axis.

    Matches ``Gemma3nAudioCumulativeGroupNorm``: for input ``[B, T, F, C]`` the
    mean and variance at time ``t`` are taken over *all* feature elements at
    times ``0..t``, so frame ``t``'s output depends on the whole prefix.  That
    running behaviour is what makes the audio tower streaming-compatible, and
    it is the one part of the SSCP subsampler with no Gemma 4 counterpart.

    HF's masking support is vestigial — the layer builds an all-ones mask
    unconditionally — so the element count is simply ``(t + 1) * F * C``, and
    HF's zero-count guard and trailing mask multiply are both no-ops.  They are
    therefore not reproduced.

    Statistics are accumulated in float32, as in HF: at f16 the running sum of
    squared deviations over a long prefix overflows the 65504 maximum well
    before the mean does.

    Args:
        num_channels: Size of the trailing (channel) axis.
        num_features: Size of the single non-channel feature axis (frequency
            bins after the convolution).  Only its product with
            ``num_channels`` is used, but both are named to document the
            expected ``[B, T, F, C]`` layout.
        eps: Added to the cumulative variance before the reciprocal sqrt.
    """

    def __init__(self, num_channels: int, num_features: int, eps: float = 1e-3):
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.weight = nn.Parameter([num_channels])
        self._eps = eps
        self._group_size = float(num_features * num_channels)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Normalize ``[B, T, F, C]`` cumulatively over ``T``."""
        time_axis = op.Constant(value_int=1)
        x_f32 = op.Cast(x, to=ir.DataType.FLOAT)

        # Elements folded into the statistics by frame t: (t + 1) * F * C.
        # A Range is cheaper than a CumSum over a broadcast constant.
        one = op.Constant(value_int=1)
        num_frames = op.Squeeze(op.Shape(x, start=1, end=2), [0])
        steps = op.Range(one, op.Add(num_frames, one), one)  # [T]: 1..T
        counts = op.Mul(
            op.Cast(steps, to=ir.DataType.FLOAT),
            op.Constant(value_float=self._group_size),
        )
        counts = op.Reshape(counts, op.Constant(value_ints=[1, -1, 1, 1]))

        sums = op.ReduceSum(x_f32, [2, 3], keepdims=True)  # [B, T, 1, 1]
        mean = op.Div(op.CumSum(sums, time_axis), counts)  # [B, T, 1, 1]

        # Deviations are measured against the *cumulative* mean, so the squared
        # sums cannot be accumulated once up front.
        centered = op.Sub(x_f32, mean)  # [B, T, F, C]
        squared = op.ReduceSum(op.Mul(centered, centered), [2, 3], keepdims=True)
        variance = op.Div(op.CumSum(squared, time_axis), counts)  # [B, T, 1, 1]

        normed = op.Mul(centered, op.Reciprocal(op.Sqrt(op.Add(variance, self._eps))))
        # Scale is per-channel: [C] -> [1, 1, 1, C].
        scale = op.Reshape(
            op.Cast(self.weight, to=ir.DataType.FLOAT),
            op.Constant(value_ints=[1, 1, 1, -1]),
        )
        return op.CastLike(op.Mul(normed, scale), x)


class _Gemma3nSSCPConvBlock(nn.Module):
    """One SSCP stage: padded ``Conv2d`` -> cumulative group norm -> ReLU.

    Matches ``Gemma3nAudioSSCPConvBlock``.  The convolution sees the input as
    ``[B, C, T, F]``, i.e. time is the *height* axis and the mel bins are the
    width, and HF pads it manually before an unpadded ``nn.Conv2d``:

    * time: ``(0, kernel_h - 1)`` — JAX ``reverse_causal``, so an output frame
      depends only on itself and *later* input frames at this stage;
    * frequency: a fixed ``(1, 1)`` (see :data:`_FREQ_PAD`).

    Both are folded into the ONNX ``Conv`` ``pads`` attribute rather than a
    separate ``Pad`` node, following
    :class:`~mobius.components.CausalDepthwiseConv1d`, so static shape
    inference still propagates through the layer.

    Args:
        in_channels: Input channels (1 for the first stage).
        out_channels: Convolution output channels.
        input_freq_dim: Unpadded frequency bins entering this stage.
        kernel_size: ``(kernel_h, kernel_w)`` over (time, frequency).
        stride: ``(stride_h, stride_w)`` over (time, frequency).
        norm_eps: Epsilon for the cumulative group norm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_freq_dim: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (2, 2),
        norm_eps: float = 1e-3,
    ):
        super().__init__()
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        if kernel_h != kernel_w or stride_h != stride_w:
            raise NotImplementedError(
                "Gemma 3n SSCP blocks only support square kernels and strides "
                f"(got kernel {kernel_size}, stride {stride}); Conv2dNoBias "
                "takes a single int for each."
            )

        self.conv = Conv2dNoBias(
            in_channels,
            out_channels,
            kernel_size=kernel_h,
            stride=stride_h,
            # ONNX pads are [t_begin, f_begin, t_end, f_end].
            padding=(0, _FREQ_PAD, kernel_h - 1, _FREQ_PAD),
        )
        self.norm = _CumulativeGroupNorm(
            out_channels,
            _sscp_freq_out_dim(input_freq_dim, kernel_w, stride_w),
            eps=norm_eps,
        )

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Subsample ``[B, C_in, T, F]`` to ``[B, C_out, T', F']``."""
        x = self.conv(op, x)  # [B, C_out, T', F']
        # The norm reduces over (F', C_out), so channels must be last.
        x = op.Transpose(x, perm=[0, 2, 3, 1])  # [B, T', F', C_out]
        x = self.norm(op, x)
        x = op.Transpose(x, perm=[0, 3, 1, 2])  # [B, C_out, T', F']
        return op.Relu(x)


class _Gemma3nSubSampleConvProjection(nn.Module):
    """Two SSCP stages plus a linear projection to the conformer width.

    Matches ``Gemma3nAudioSubSampleConvProjection``.  Time and frequency are
    each halved twice, so ``T' = ceil(ceil(T / 2) / 2)`` and the E4B frequency
    flow is ``128 -> 64 -> 32``, giving a ``32 * 32 = 1024``-wide projection
    input.

    Args:
        input_feat_size: Mel bins per frame (128 for E4B).
        hidden_size: Conformer width to project onto.
        conv_channel_size: Per-stage output channels, e.g. ``[128, 32]``.
        conv_kernel_size: Per-stage ``(kernel_h, kernel_w)``.
        conv_stride_size: Per-stage ``(stride_h, stride_w)``.
        norm_eps: Epsilon for the cumulative group norms.
    """

    def __init__(
        self,
        input_feat_size: int = 128,
        hidden_size: int = 1536,
        conv_channel_size: list[int] | None = None,
        conv_kernel_size: list[list[int]] | None = None,
        conv_stride_size: list[list[int]] | None = None,
        norm_eps: float = 1e-3,
    ):
        super().__init__()
        channels = list(conv_channel_size or [128, 32])
        kernels = [tuple(k) for k in (conv_kernel_size or [[3, 3], [3, 3]])]
        strides = [tuple(s) for s in (conv_stride_size or [[2, 2], [2, 2]])]

        freq_dims = []
        freq = input_feat_size
        for (_, kernel_w), (_, stride_w) in zip(kernels, strides, strict=True):
            freq = _sscp_freq_out_dim(freq, kernel_w, stride_w)
            freq_dims.append(freq)

        self.conv_0 = _Gemma3nSSCPConvBlock(
            1, channels[0], input_feat_size, kernels[0], strides[0], norm_eps
        )
        self.conv_1 = _Gemma3nSSCPConvBlock(
            channels[0], channels[1], freq_dims[0], kernels[1], strides[1], norm_eps
        )
        self.input_proj_in_features = channels[-1] * freq_dims[-1]
        self.input_proj_linear = Linear(
            self.input_proj_in_features, hidden_size, bias=False
        )
        # Original frames per subsampled frame; the encoder needs it to
        # subsample the mask.
        self.time_stride_product = math.prod(stride_h for stride_h, _ in strides)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Project mel frames ``[B, T, F]`` to ``[B, T', hidden_size]``."""
        x = op.Unsqueeze(x, [1])  # [B, 1, T, F]
        x = self.conv_0(op, x)
        x = self.conv_1(op, x)  # [B, C, T', F']
        # Flatten to [B, T', F' * C] — frequency-major, matching HF's permute
        # then view (a channel-major flatten would shuffle the projection).
        x = op.Transpose(x, perm=[0, 2, 3, 1])  # [B, T', F', C]
        x = op.Reshape(x, op.Constant(value_ints=[0, 0, -1]))
        return self.input_proj_linear(op, x)


class _Gemma3nAudioRelativePositionEmbedding(nn.Module):
    """Relative-position attention bias with a learned projection.

    Matches ``Gemma3nAudioRelativePositionEmbedding``, but produces the bias
    directly in ``[B, H, T, T]`` layout instead of HF's blocked
    ``[B, H, U, W, C]`` — see :class:`_Gemma3nAudioAttention` for why the two
    are equivalent.

    HF embeds the *span* of reachable relative distances ``[L, ..., -R]`` as a
    sinusoidal signal, projects it per head, dots it with the (already scaled)
    queries, then reindexes that ``[..., span]`` result into key positions with
    a pad / reshape / slice / reshape trick.  Working that trick through, query
    ``i`` attending key ``j`` reads span row ``f = L - (i - j)``, which is the
    explicit ``GatherElements`` used here.

    ``sin_emb`` is derived from hyperparameters and is *not* a checkpoint
    tensor, so it is materialized as a constant initializer (as Gemma 4's
    ``pos_embed`` is).  ``pos_proj`` is the one learned weight.

    Args:
        hidden_size: Model width (also the sinusoid width).
        num_heads: Attention heads.
        max_backward: Reachable history in frames (``L``).
        max_forward: Reachable lookahead in frames (``R``, 0 for E4B).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_backward: int,
        max_forward: int,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads
        self.max_backward = max_backward
        self.max_forward = max_forward

        self.pos_proj = Linear(hidden_size, num_heads * self._head_dim, bias=False)

        # Distances, descending: [L, L-1, ..., -R]. Row f holds distance L - f.
        positions = np.arange(max_backward, -max_forward - 1, -1)
        self.span = len(positions)
        self.sin_emb = nn.Parameter(
            [self.span, hidden_size],
            data=ir.Tensor(
                _timing_signal(positions, hidden_size), dtype=ir.DataType.FLOAT
            ),
        )

    def forward(self, op: OpBuilder, queries: ir.Value, seq_len: ir.Value) -> ir.Value:
        """Compute the additive bias for scaled queries.

        Args:
            queries: ``[B, H, T, head_dim]`` float32, already scaled.
            seq_len: 1-D ``[1]`` int64 holding ``T``.

        Returns:
            ``[B, H, T, T]`` float32 bias.
        """
        num_heads, head_dim = self._num_heads, self._head_dim

        # Project in the weight's dtype (sin_emb is a float32 constant, the
        # weight may be f16), then accumulate in float32.
        sin_emb = op.CastLike(self.sin_emb, self.pos_proj.weight)
        projected = self.pos_proj(op, sin_emb)  # [span, H * head_dim]
        projected = op.Cast(projected, to=ir.DataType.FLOAT)
        projected = op.Reshape(
            projected, op.Constant(value_ints=[-1, num_heads, head_dim])
        )
        projected = op.Transpose(projected, perm=[1, 2, 0])  # [H, head_dim, span]
        span_scores = op.MatMul(queries, op.Unsqueeze(projected, [0]))  # [B,H,T,span]

        # f = clip(L - (i - j), 0, span - 1). Out-of-window entries clip to an
        # arbitrary in-range row; the attention mask discards them.
        zero = op.Constant(value_int=0)
        positions = op.Range(zero, op.Squeeze(seq_len, [0]), op.Constant(value_int=1))
        distance = op.Sub(op.Unsqueeze(positions, [1]), op.Unsqueeze(positions, [0]))
        index = op.Clip(
            op.Sub(op.Constant(value_int=self.max_backward), distance),
            zero,
            op.Constant(value_int=self.span - 1),
        )  # [T, T] int64

        # GatherElements needs the index expanded to the full output shape.
        batch_heads = op.Shape(span_scores, start=0, end=2)  # [B, H]
        index = op.Expand(
            op.Unsqueeze(index, [0, 1]),
            op.Concat(batch_heads, seq_len, seq_len, axis=0),
        )
        return op.GatherElements(span_scores, index, axis=3)  # [B, H, T, T]


class _Gemma3nAudioAttention(nn.Module):
    """Gemma 3n audio self-attention (chunked attention, flattened to T×T).

    Matches ``Gemma3nAudioAttention``.  HF computes this in chunks: queries are
    grouped into blocks of ``conf_attention_chunk_size`` (``W``) frames, each
    block gets a key/value context window of ``W + L + R`` frames, and a
    ``[W, context]`` validity mask restricts each query inside it.  Working
    that mask through the block indexing, query ``i`` ends up attending exactly
    the keys ``j`` with ``-R <= i - j <= L``, and every masked-out score
    contributes nothing to its softmax — so the blocked form and a full ``T×T``
    attention under the same local mask are numerically identical for offline
    (whole-utterance) inference.  The flattened form is used here: it avoids
    the pad / unfold / relative-shift chain, which in ONNX would mean dynamic
    reshapes that block shape inference.

    Note ``L = conf_attention_context_left - 1``, so the default 13 admits 13
    keys (12 of history plus the query's own frame) — one more than
    :class:`Gemma4Attention`'s window, which is why that class cannot simply be
    subclassed.  The other blockers: Gemma 3n's checkpoint has no
    activation-clipping bounds (so ``ClippableLinear`` is wrong), it applies no
    rescaling to the keys, its output projection lives one level up in the
    conformer block, and the relative-position weight is nested under
    ``relative_position_embedding``.

    The logit cap is applied *after* the relative-position bias and *before*
    the mask, which is why the ONNX ``Attention`` op's ``softcap`` attribute
    cannot be used here either (its pipeline caps before any mask or bias);
    see :class:`Gemma4Attention` for the full argument.

    Args:
        hidden_size: Model width.
        num_heads: Attention heads.
        attention_chunk_size: HF's query block size.  It cancels out of the
            flattened form, and is accepted only so callers can pass the config
            field through uniformly.
        attention_context_left: History window *including* the current frame.
        attention_context_right: Lookahead window in frames.
        attention_logit_cap: ``tanh``-cap applied to the scores.
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        num_heads: int = 8,
        attention_chunk_size: int = 12,
        attention_context_left: int = 13,
        attention_context_right: int = 0,
        attention_logit_cap: float = 50.0,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads
        self._chunk_size = attention_chunk_size
        self._max_backward = max(0, attention_context_left - 1)
        self._max_forward = attention_context_right
        self._attention_logit_cap = attention_logit_cap
        # HF: head_dim**-0.5 / softplus(0), and softplus(0) == log(2).
        self._q_scale = (self._head_dim**-0.5) / math.log(2.0)

        self.q_proj = Linear(hidden_size, num_heads * self._head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_heads * self._head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_heads * self._head_dim, bias=False)
        # Learned per-head-dimension query scale, passed through softplus.
        self.per_dim_scale = nn.Parameter([self._head_dim])
        self.relative_position_embedding = _Gemma3nAudioRelativePositionEmbedding(
            hidden_size, num_heads, self._max_backward, self._max_forward
        )

    def _window_mask(self, op: OpBuilder, seq_len: ir.Value) -> ir.Value:
        """``[1, 1, T, T]`` bool: True where key ``j`` is in query ``i``'s window."""
        zero = op.Constant(value_int=0)
        positions = op.Range(zero, op.Squeeze(seq_len, [0]), op.Constant(value_int=1))
        distance = op.Sub(op.Unsqueeze(positions, [1]), op.Unsqueeze(positions, [0]))
        in_window = op.And(
            op.LessOrEqual(distance, op.Constant(value_int=self._max_backward)),
            op.GreaterOrEqual(distance, op.Constant(value_int=-self._max_forward)),
        )
        return op.Unsqueeze(in_window, [0, 1])

    def forward(
        self,
        op: OpBuilder,
        x: ir.Value,
        attention_mask: ir.Value,
    ) -> ir.Value:
        """Attend over the time axis.

        Args:
            x: ``[B, T, hidden_size]``.
            attention_mask: bool ``[B, T]``, True = **valid** (note this is the
                negation of HF's ``audio_mel_mask``).

        Returns:
            ``[B, T, hidden_size]``.  HF returns ``[B, T, H, head_dim]`` and
            leaves the flatten to its caller; it happens here instead.
        """
        num_heads, head_dim = self._num_heads, self._head_dim
        seq_len = op.Shape(x, start=1, end=2)

        q = self.q_proj(op, x)
        k = self.k_proj(op, x)
        v = self.v_proj(op, x)

        # float32 attention, matching HF's float32 softmax and guarding f16
        # exports against overflow in the score matmul.
        q = op.Cast(q, to=ir.DataType.FLOAT)
        k = op.Cast(k, to=ir.DataType.FLOAT)
        v = op.Cast(v, to=ir.DataType.FLOAT)

        # 0 means "copy this dim from the input", keeping the reshape static.
        qkv_shape = op.Constant(value_ints=[0, 0, num_heads, head_dim])
        q = op.Reshape(q, qkv_shape)  # [B, T, H, head_dim]
        k = op.Reshape(k, qkv_shape)
        v = op.Reshape(v, qkv_shape)

        per_dim = op.Softplus(op.Cast(self.per_dim_scale, to=ir.DataType.FLOAT))
        q = op.Mul(q, op.Mul(op.Constant(value_float=self._q_scale), per_dim))
        # Gemma 3n applies no scale to the keys (Gemma 4 does).

        q = op.Transpose(q, perm=[0, 2, 1, 3])  # [B, H, T, head_dim]
        k = op.Transpose(k, perm=[0, 2, 1, 3])
        v = op.Transpose(v, perm=[0, 2, 1, 3])

        scores = op.MatMul(q, op.Transpose(k, perm=[0, 1, 3, 2]))  # [B, H, T, T]
        scores = op.Add(scores, self.relative_position_embedding(op, q, seq_len))

        cap = op.Constant(value_float=self._attention_logit_cap)
        scores = op.Mul(op.Tanh(op.Div(scores, cap)), cap)

        # Blocked positions are *replaced*, not offset, so the cap above cannot
        # pull them back into range.
        allowed = op.And(
            self._window_mask(op, seq_len),
            op.Unsqueeze(attention_mask, [1, 2]),  # [B, 1, 1, T]
        )
        scores = op.Where(allowed, scores, op.Constant(value_float=_MASK_SCORE))

        context = op.MatMul(op.Softmax(scores, axis=-1), v)  # [B, H, T, head_dim]
        context = op.Transpose(context, perm=[0, 2, 1, 3])  # [B, T, H, head_dim]
        context = op.Reshape(context, op.Constant(value_ints=[0, 0, -1]))
        return op.CastLike(context, x)


class _Gemma3nAudioConformerAttention(nn.Module):
    """Pre-norm attention sub-block with a post projection and residual.

    Matches ``Gemma3nAudioConformerAttention``::

        residual = x
        x = clip(x) -> pre_attn_norm(x) -> attn(x, mask) -> post(x) -> clip(x)
        return residual + post_norm(x)

    Note the residual is the *unclipped* input, and ``post_norm`` sits inside
    the residual branch rather than wrapping the sum.

    Args:
        hidden_size: Model width.
        num_heads: Attention heads.
        attention_chunk_size: HF's query block size (see
            :class:`_Gemma3nAudioAttention`).
        attention_context_left: History window including the current frame.
        attention_context_right: Lookahead window.
        attention_logit_cap: ``tanh``-cap on the attention scores.
        rms_norm_eps: Epsilon for both RMSNorms.
        gradient_clipping: Activation clamp for numerical stability.
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        num_heads: int = 8,
        attention_chunk_size: int = 12,
        attention_context_left: int = 13,
        attention_context_right: int = 0,
        attention_logit_cap: float = 50.0,
        rms_norm_eps: float = 1e-6,
        gradient_clipping: float = 1e10,
    ):
        super().__init__()
        self._gradient_clipping = gradient_clipping

        self.pre_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn = _Gemma3nAudioAttention(
            hidden_size,
            num_heads,
            attention_chunk_size,
            attention_context_left,
            attention_context_right,
            attention_logit_cap,
        )
        self.post = Linear(hidden_size, hidden_size, bias=False)
        self.post_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, op: OpBuilder, x: ir.Value, attention_mask: ir.Value) -> ir.Value:
        """Attend over ``[B, T, hidden_size]`` with a ``[B, T]`` valid mask."""
        residual = x
        x = _gradient_clip(op, x, self._gradient_clipping)
        x = self.pre_attn_norm(op, x)
        x = self.attn(op, x, attention_mask)
        x = self.post(op, x)
        x = _gradient_clip(op, x, self._gradient_clipping)
        return op.Add(residual, self.post_norm(op, x))


class _Gemma3nAudioConformerBlock(nn.Module):
    """One conformer block: FF -> attention -> light conv1d -> FF -> norm.

    Matches ``Gemma3nAudioConformerBlock``.  The feed-forward and light-conv1d
    sub-blocks are Gemma 4's, identical here in both weight names and
    arithmetic — only with plain :class:`Linear` projections, since Gemma 3n
    ships no activation-clipping bounds.

    Padded frames are zeroed between the attention and the conv, because the
    depthwise convolution mixes neighbouring frames and would otherwise smear
    padding into valid positions.

    Args:
        hidden_size: Model width.
        num_heads: Attention heads.
        conv_kernel_size: Depthwise conv kernel (5 for E4B).
        attention_chunk_size: HF's query block size.
        attention_context_left: History window including the current frame.
        attention_context_right: Lookahead window.
        attention_logit_cap: ``tanh``-cap on the attention scores.
        rms_norm_eps: Epsilon for every RMSNorm in the block.
        residual_weight: Feed-forward output scale (``conf_residual_weight``).
        gradient_clipping: Activation clamp for numerical stability.
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        num_heads: int = 8,
        conv_kernel_size: int = 5,
        attention_chunk_size: int = 12,
        attention_context_left: int = 13,
        attention_context_right: int = 0,
        attention_logit_cap: float = 50.0,
        rms_norm_eps: float = 1e-6,
        residual_weight: float = 0.5,
        gradient_clipping: float = 1e10,
    ):
        super().__init__()
        self._gradient_clipping = gradient_clipping

        self.ffw_layer_start = Gemma4FeedForward(
            hidden_size,
            rms_norm_eps,
            residual_weight,
            gradient_clipping,
            linear_cls=Linear,
        )
        self.attention = _Gemma3nAudioConformerAttention(
            hidden_size,
            num_heads,
            attention_chunk_size,
            attention_context_left,
            attention_context_right,
            attention_logit_cap,
            rms_norm_eps,
            gradient_clipping,
        )
        self.lconv1d = Gemma4LightConv1d(
            hidden_size,
            conv_kernel_size,
            rms_norm_eps,
            gradient_clipping,
            linear_cls=Linear,
        )
        self.ffw_layer_end = Gemma4FeedForward(
            hidden_size,
            rms_norm_eps,
            residual_weight,
            gradient_clipping,
            linear_cls=Linear,
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, op: OpBuilder, x: ir.Value, attention_mask: ir.Value) -> ir.Value:
        """Run one block over ``[B, T, hidden_size]`` with a ``[B, T]`` valid mask."""
        x = self.ffw_layer_start(op, x)
        x = self.attention(op, x, attention_mask)
        x = op.Mul(x, op.CastLike(op.Unsqueeze(attention_mask, [-1]), x))
        x = self.lconv1d(op, x)
        x = self.ffw_layer_end(op, x)
        x = _gradient_clip(op, x, self._gradient_clipping)
        return self.norm(op, x)


class Gemma3nAudioEncoder(nn.Module):
    """Gemma 3n USM conformer audio tower.

    Matches ``Gemma3nAudioEncoder``::

        input_features [B, T, input_feat_size]
        -> subsample_conv_projection      (T/4, -> hidden_size)
        -> conformer x num_layers
        -> stride-reduction_factor subsample, padded frames zeroed
        [B, T/16, hidden_size]

    There is no output projection: the audio tower's 269 checkpoint tensors end
    at the conformer stack, and the projection into the text embedding space is
    ``model.embed_audio``
    (:class:`~mobius.components.Gemma3nMultimodalEmbedder`).

    The mask is required, not optional as in
    :class:`~mobius.components.Gemma4AudioEncoder`: it gates the light conv,
    zeroes padded outputs, and is returned for the caller to reuse, so an
    all-valid default would silently change results for batched audio.

    Args:
        input_feat_size: Mel bins per frame.
        hidden_size: Conformer width.
        num_heads: Attention heads (``conf_num_attention_heads``).
        num_layers: Conformer blocks (``conf_num_hidden_layers``).
        conv_kernel_size: Light-conv kernel (``conf_conv_kernel_size``).
        conv_channel_size: SSCP per-stage channels (``sscp_conv_channel_size``).
        conv_kernel_size_2d: SSCP per-stage kernels (``sscp_conv_kernel_size``).
        conv_stride_size: SSCP per-stage strides (``sscp_conv_stride_size``).
        conv_group_norm_eps: SSCP cumulative group-norm epsilon.
        attention_chunk_size: HF's query block size (see
            :class:`_Gemma3nAudioAttention`).
        attention_context_left: History window including the current frame.
        attention_context_right: Lookahead window.
        attention_logit_cap: ``tanh``-cap on the attention scores.
        reduction_factor: Final time subsampling (``conf_reduction_factor``).
        rms_norm_eps: Epsilon for every RMSNorm. HF hard-codes 1e-6 outside the
            light conv and reads ``rms_norm_eps`` inside it; the two coincide
            for every published Gemma 3n config, so one value is used here.
        residual_weight: Feed-forward output scale.
        gradient_clipping: Activation clamp for numerical stability.
    """

    def __init__(
        self,
        input_feat_size: int = 128,
        hidden_size: int = 1536,
        num_heads: int = 8,
        num_layers: int = 12,
        conv_kernel_size: int = 5,
        conv_channel_size: list[int] | None = None,
        conv_kernel_size_2d: list[list[int]] | None = None,
        conv_stride_size: list[list[int]] | None = None,
        conv_group_norm_eps: float = 1e-3,
        attention_chunk_size: int = 12,
        attention_context_left: int = 13,
        attention_context_right: int = 0,
        attention_logit_cap: float = 50.0,
        reduction_factor: int = 4,
        rms_norm_eps: float = 1e-6,
        residual_weight: float = 0.5,
        gradient_clipping: float = 1e10,
    ):
        super().__init__()
        self._reduction_factor = reduction_factor

        self.subsample_conv_projection = _Gemma3nSubSampleConvProjection(
            input_feat_size,
            hidden_size,
            conv_channel_size,
            conv_kernel_size_2d,
            conv_stride_size,
            conv_group_norm_eps,
        )
        self.conformer = nn.ModuleList(
            [
                _Gemma3nAudioConformerBlock(
                    hidden_size,
                    num_heads,
                    conv_kernel_size,
                    attention_chunk_size,
                    attention_context_left,
                    attention_context_right,
                    attention_logit_cap,
                    rms_norm_eps,
                    residual_weight,
                    gradient_clipping,
                )
                for _ in range(num_layers)
            ]
        )

    def _subsample_mask(
        self, op: OpBuilder, mask: ir.Value, encodings: ir.Value
    ) -> ir.Value:
        """Pick the mask entry starting each subsampled frame's receptive field.

        Matches HF: ``clamp(arange(T') * time_stride_product, max=T - 1)``,
        gathered from the full-resolution mask.  The clamp matters because the
        reverse-causal time padding makes ``T'`` large enough that the last few
        start indices can run past ``T``.
        """
        stride = self.subsample_conv_projection.time_stride_product
        sub_len = op.Shape(encodings, start=1, end=2)  # [1]
        full_len = op.Shape(mask, start=1, end=2)  # [1]
        indices = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(sub_len, [0]),
            op.Constant(value_int=1),
        )
        indices = op.Mul(indices, op.Constant(value_int=stride))
        last = op.Squeeze(op.Sub(full_len, op.Constant(value_ints=[1])), [0])
        indices = op.Min(indices, last)
        # The indices are shared across the batch, so a plain axis-1 Gather
        # does what HF's batch-expanded torch.gather does.
        return op.Gather(mask, indices, axis=1)  # [B, T']

    def forward(
        self,
        op: OpBuilder,
        input_features: ir.Value,
        input_features_mask: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        """Encode mel-spectrogram frames.

        Args:
            input_features: ``[B, T, input_feat_size]`` mel features.
            input_features_mask: bool ``[B, T]``, True = **valid** frame (the
                negation of HF's ``audio_mel_mask``).

        Returns:
            ``(encodings, mask)`` with encodings ``[B, T', hidden_size]`` and
            mask bool ``[B, T']``, where ``T'`` is ``T`` reduced by the SSCP
            strides and then by ``reduction_factor``.  Padded positions in
            ``encodings`` are zero.
        """
        x = self.subsample_conv_projection(op, input_features)  # [B, T/4, hidden]
        mask = self._subsample_mask(op, input_features_mask, x)  # [B, T/4]

        for block in self.conformer:
            x = block(op, x, mask)

        if self._reduction_factor > 1:
            # x[:, ::reduction_factor]. INT64_MAX as ``ends`` means "to the
            # end"; shape inference normalizes it for strided slices.
            step = [self._reduction_factor]
            x = op.Slice(x, [0], [INT64_MAX], [1], step)
            mask = op.Slice(mask, [0], [INT64_MAX], [1], step)

        # HF masked_fill's the padded frames to zero after the reduction.
        x = op.Mul(x, op.CastLike(op.Unsqueeze(mask, [-1]), x))
        return x, mask
