# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""FastConformer-RNNT (Transducer) model for NeMo speech recognition.

Replicates NVIDIA NeMo's ``EncDecRNNTBPEModel`` (e.g.
``nvidia/nemotron-speech-streaming-en-0.6b``), a streaming Cache-Aware
FastConformer encoder paired with an RNN-T (transducer) prediction +
joint network.

The architecture is split into three ONNX sub-models (see
:class:`mobius.tasks._rnnt.RNNTTask`):

* ``encoder`` — mel features ``(B, feat_in, T)`` → encoded ``(B, d_model, T')``.
  A FastConformer encoder: causal depthwise-striding Conv2d subsampling
  (8x time reduction) followed by ``num_hidden_layers`` Conformer blocks with
  Transformer-XL relative-position multi-head attention, Macaron-style
  feed-forwards, and a causal depthwise convolution module.
* ``decoder`` — token ids ``(B, U)`` + LSTM state → prediction ``(B, d_pred, U)``.
  An embedding followed by a multi-layer LSTM ("prediction network").
* ``joint`` — encoder ``(B, d_model, T')`` + prediction ``(B, d_pred, U)`` →
  logits ``(B, T', U, vocab+1)``.  Combines the two projections and applies
  the joint network.

HuggingFace/NeMo correspondence (state-dict prefixes):

* ``encoder.*``            → :class:`FastConformerEncoder` (names match exactly)
* ``decoder.prediction.*`` → :class:`RNNTPrediction`
* ``joint.*``              → :class:`RNNTJoint`
"""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import Embedding, LayerNorm, Linear

_INF_VAL = 10000.0
_NEG_INF = -10000.0  # NeMo masks attention scores with -INF_VAL (= -10000.0)


def _swish(op: OpBuilder, x: ir.Value) -> ir.Value:
    """Swish / SiLU activation: ``x * sigmoid(x)``."""
    return op.Mul(x, op.Sigmoid(x))


def _dim(op: OpBuilder, x: ir.Value, axis: int) -> ir.Value:
    """Return a single dimension of *x* as a 1-D INT64 tensor of length 1."""
    return op.Shape(x, start=axis, end=axis + 1)


def _scalar_like(op: OpBuilder, value: float, ref: ir.Value) -> ir.Value:
    """Emit a scalar constant cast to the dtype of *ref*.

    Float scalar Constants are kept at float32 by the builder, so they must be
    cast to the model's compute dtype (f16/bf16) before combining with
    dtype-cast tensors; otherwise ONNX rejects the mismatched element types.
    """
    return op.CastLike(op.Constant(value=ir.tensor(np.float32(value))), ref)


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------


class _Conv2d(nn.Module):
    """Conv2d with optional asymmetric (causal) padding applied via op.Pad.

    NeMo's ``CausalConv2D`` pads both spatial dims by ``(k-1, s-1)`` on the
    (left, right) of each axis, then runs a zero-padded convolution.  When
    ``causal`` is ``False`` this is an ordinary Conv2d (used for the 1x1
    pointwise stages).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        groups: int = 1,
        causal: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            [out_channels, in_channels // groups, kernel_size, kernel_size]
        )
        self.bias = nn.Parameter([out_channels])
        self._k = kernel_size
        self._stride = stride
        self._groups = groups
        self._causal = causal

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        if self._causal:
            left = self._k - 1
            right = self._stride - 1
            # x is (N, C, H=time, W=freq); pad both spatial axes by (left, right)
            x = op.Pad(
                x,
                op.Constant(value_ints=[0, 0, left, left, 0, 0, right, right]),
            )
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self._k, self._k],
            strides=[self._stride, self._stride],
            pads=[0, 0, 0, 0],
            group=self._groups,
        )


class _Conv1d(nn.Module):
    """Conv1d wrapper (weight + optional bias), optional causal left padding."""

    def __init__(
        self,
        out_channels: int,
        in_channels_per_group: int,
        kernel_size: int,
        *,
        groups: int = 1,
        bias: bool = False,
        left_pad: int = 0,
    ):
        super().__init__()
        self.weight = nn.Parameter([out_channels, in_channels_per_group, kernel_size])
        self.bias = nn.Parameter([out_channels]) if bias else None
        self._k = kernel_size
        self._groups = groups
        self._left_pad = left_pad

    def forward(
        self, op: OpBuilder, x: ir.Value, *, cache: ir.Value | None = None
    ) -> ir.Value | tuple[ir.Value, ir.Value]:
        if cache is not None:
            return self._forward_with_cache(op, x, cache)
        if self._left_pad:
            # x is (N, C, T); causal: pad only the left of the time axis
            x = op.Pad(
                x,
                op.Constant(value_ints=[0, 0, self._left_pad, 0, 0, 0]),
            )
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self._k],
            strides=[1],
            pads=[0, 0],
            group=self._groups,
        )

    def _forward_with_cache(
        self, op: OpBuilder, x: ir.Value, cache: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        """Causal conv using a left-context *cache* instead of zero padding.

        ``x`` is ``(N, C, T)`` and ``cache`` is ``(N, C, left_pad)`` carrying
        the previous chunk's last ``left_pad`` input frames.  Returns the
        convolution output ``(N, C, T)`` and the updated cache ``(N, C,
        left_pad)`` (the last ``left_pad`` frames of *x*), matching NeMo's
        ``CausalConv1D.update_cache``.
        """
        x_cat = op.Concat(cache, x, axis=-1)  # (N, C, left_pad + T)
        out = op.Conv(
            x_cat,
            self.weight,
            self.bias,
            kernel_shape=[self._k],
            strides=[1],
            pads=[0, 0],
            group=self._groups,
        )
        # next cache = last left_pad frames of concat(cache, x) (NeMo update_cache)
        new_cache = op.Slice(
            x_cat,
            op.Constant(value_ints=[-self._left_pad]),
            op.Constant(value_ints=[int(np.iinfo(np.int64).max)]),
            op.Constant(value_ints=[2]),
        )
        return out, new_cache


class _ReLU(nn.Module):
    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return op.Relu(x)


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------


class ConvSubsampling(nn.Module):
    """Causal depthwise-striding Conv2d subsampling (NeMo ``dw_striding``).

    Reduces the time dimension by 8x via three stride-2 stages.  The first
    stage is a full Conv2d (1 → C); the remaining two are depthwise (CxC,
    groups=C) followed by a 1x1 pointwise Conv2d.  All stride-2 convolutions
    use causal padding (left=k-1, right=s-1 on each spatial axis).

    Input:  ``(B, feat_in, T)``
    Output: ``(B, T', d_model)`` where ``T' = ceil(T / 8)``
    """

    def __init__(self, feat_in: int, conv_channels: int, d_model: int):
        super().__init__()
        # Frequency dim after three causal stride-2 stages (pad left=2,right=1)
        freq = feat_in
        for _ in range(3):
            freq = (freq + 3 - 3) // 2 + 1
        c = conv_channels
        # Indices match NeMo: conv.0/2/3/5/6 are conv layers; 1/4/7 are ReLU.
        self.conv = nn.ModuleList(
            [
                _Conv2d(1, c, 3, stride=2, causal=True),  # 0
                _ReLU(),  # 1
                _Conv2d(c, c, 3, stride=2, groups=c, causal=True),  # 2 depthwise
                _Conv2d(c, c, 1),  # 3 pointwise
                _ReLU(),  # 4
                _Conv2d(c, c, 3, stride=2, groups=c, causal=True),  # 5 depthwise
                _Conv2d(c, c, 1),  # 6 pointwise
                _ReLU(),  # 7
            ]
        )
        self.out = Linear(conv_channels * freq, d_model)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: (B, feat_in, T) -> (B, T, feat_in) -> (B, 1, T, feat_in)
        x = op.Transpose(x, perm=[0, 2, 1])
        x = op.Unsqueeze(x, op.Constant(value_ints=[1]))
        for layer in self.conv:
            x = layer(op, x)
        # (B, C, T', F') -> (B, T', C, F') -> (B, T', C*F')
        x = op.Transpose(x, perm=[0, 2, 1, 3])
        x = op.Reshape(x, op.Constant(value_ints=[0, 0, -1]))
        return self.out(op, x)


# ---------------------------------------------------------------------------
# Relative-position attention
# ---------------------------------------------------------------------------


class RelPositionMultiHeadAttention(nn.Module):
    """Transformer-XL relative-position multi-head attention (NeMo ``rel_pos``).

    Implements the non-SDPA path::

        matrix_ac = (q + pos_bias_u) @ k^T
        matrix_bd = rel_shift((q + pos_bias_v) @ p^T)[..., :T]
        scores    = (matrix_ac + matrix_bd) / sqrt(d_k)

    where ``p = linear_pos(pos_emb)`` are the projected relative position
    embeddings.  ``att_bias`` is a boolean keep-mask broadcast to
    ``(1, 1, T, T)`` (True = attend).
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.linear_q = Linear(d_model, d_model, bias=False)
        self.linear_k = Linear(d_model, d_model, bias=False)
        self.linear_v = Linear(d_model, d_model, bias=False)
        self.linear_out = Linear(d_model, d_model, bias=False)
        self.linear_pos = Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter([n_heads, self.d_k])
        self.pos_bias_v = nn.Parameter([n_heads, self.d_k])

    def _split_heads(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # (B, T, d_model) -> (B, T, h, d_k)
        b = _dim(op, x, 0)
        t = _dim(op, x, 1)
        shape = op.Concat(b, t, op.Constant(value_ints=[self.h, self.d_k]), axis=0)
        return op.Reshape(x, shape)

    def _rel_shift(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: (B, h, T, L) with L = 2T-1 -> shifted (B, h, T, L)
        b = _dim(op, x, 0)
        h = _dim(op, x, 1)
        t = _dim(op, x, 2)
        # pad a zero column on the left of the last dim -> (B, h, T, L+1)
        x = op.Pad(x, op.Constant(value_ints=[0, 0, 0, 1, 0, 0, 0, 0]))
        # view (B, h, L+1, T)
        shape1 = op.Concat(b, h, op.Constant(value_ints=[-1]), t, axis=0)
        x = op.Reshape(x, shape1)
        # drop the first row along axis 2 -> (B, h, L, T)
        x = op.Slice(
            x,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[int(np.iinfo(np.int64).max)]),
            op.Constant(value_ints=[2]),
        )
        # view back to (B, h, T, L)
        ll = op.Constant(value_ints=[-1])
        shape2 = op.Concat(b, h, t, ll, axis=0)
        return op.Reshape(x, shape2)

    def forward(
        self,
        op: OpBuilder,
        x: ir.Value,
        pos_emb: ir.Value,
        att_bias: ir.Value,
        *,
        kv_x: ir.Value | None = None,
    ) -> ir.Value:
        """Relative-position attention.

        Offline (``kv_x is None``): queries and keys are the same sequence, so
        the score matrices are square.  Cache-aware streaming (``kv_x`` given):
        queries come from the current chunk ``x`` ``(B, Tq, d)`` while keys and
        values come from ``concat(cache, current)`` ``(B, Tk, d)`` with ``Tk =
        cache_len + Tq``, mirroring NeMo's non-SDPA streaming path where the
        relative-position embedding spans the full ``Tk`` window.
        """
        return self._attend(op, x, x if kv_x is None else kv_x, pos_emb, att_bias)

    def _attend(
        self,
        op: OpBuilder,
        q_x: ir.Value,
        kv_x: ir.Value,
        pos_emb: ir.Value,
        att_bias: ir.Value,
    ) -> ir.Value:
        q = self._split_heads(op, self.linear_q(op, q_x))  # (B, Tq, h, d_k)
        k = self._split_heads(op, self.linear_k(op, kv_x))  # (B, Tk, h, d_k)
        v = self._split_heads(op, self.linear_v(op, kv_x))
        # to (B, h, Tk, d_k)
        k = op.Transpose(k, perm=[0, 2, 1, 3])
        v = op.Transpose(v, perm=[0, 2, 1, 3])

        # p = linear_pos(pos_emb): (1, L, d_model) -> (1, h, L, d_k)
        p = self._split_heads(op, self.linear_pos(op, pos_emb))
        p = op.Transpose(p, perm=[0, 2, 1, 3])

        # pos_bias_* broadcast over (B, T): reshape to (1, 1, h, d_k)
        bias_u = op.Reshape(self.pos_bias_u, op.Constant(value_ints=[1, 1, self.h, self.d_k]))
        bias_v = op.Reshape(self.pos_bias_v, op.Constant(value_ints=[1, 1, self.h, self.d_k]))
        q_u = op.Transpose(op.Add(q, bias_u), perm=[0, 2, 1, 3])  # (B, h, T, d_k)
        q_v = op.Transpose(op.Add(q, bias_v), perm=[0, 2, 1, 3])

        # matrix_ac: (B, h, T, T)
        matrix_ac = op.MatMul(q_u, op.Transpose(k, perm=[0, 1, 3, 2]))
        # matrix_bd: (B, h, T, L) -> rel_shift -> slice to T
        matrix_bd = op.MatMul(q_v, op.Transpose(p, perm=[0, 1, 3, 2]))
        matrix_bd = self._rel_shift(op, matrix_bd)
        t = _dim(op, matrix_ac, 3)
        matrix_bd = op.Slice(
            matrix_bd,
            op.Constant(value_ints=[0]),
            t,
            op.Constant(value_ints=[3]),
        )

        scale = _scalar_like(op, 1.0 / math.sqrt(self.d_k), matrix_ac)
        scores = op.Mul(op.Add(matrix_ac, matrix_bd), scale)

        # Apply keep-mask: where not allowed, set score to -INF_VAL
        neg_inf = _scalar_like(op, _NEG_INF, scores)
        scores = op.Where(att_bias, scores, neg_inf)
        attn = op.Softmax(scores, axis=-1)

        out = op.MatMul(attn, v)  # (B, h, T, d_k)
        out = op.Transpose(out, perm=[0, 2, 1, 3])  # (B, T, h, d_k)
        b = _dim(op, out, 0)
        tt = _dim(op, out, 1)
        out = op.Reshape(
            out,
            op.Concat(b, tt, op.Constant(value_ints=[self.h * self.d_k]), axis=0),
        )
        return self.linear_out(op, out)


# ---------------------------------------------------------------------------
# Conformer feed-forward and convolution
# ---------------------------------------------------------------------------


class ConformerFeedForward(nn.Module):
    """Macaron feed-forward: ``Linear -> Swish -> Linear`` (no bias)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, bias=False)
        self.linear2 = Linear(d_ff, d_model, bias=False)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return self.linear2(op, _swish(op, self.linear1(op, x)))


class ConformerConvolution(nn.Module):
    """NeMo Conformer convolution module (causal, layer-norm variant).

    ``pointwise_conv1 -> GLU -> depthwise (causal) -> LayerNorm -> Swish ->
    pointwise_conv2``.  Operates on ``(B, T, d_model)`` internally transposing
    to channels-first for the convolutions.
    """

    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.pointwise_conv1 = _Conv1d(2 * d_model, d_model, 1)
        # Causal depthwise: left pad = kernel_size - 1, right = 0
        self.depthwise_conv = _Conv1d(
            d_model, 1, kernel_size, groups=d_model, left_pad=kernel_size - 1
        )
        self.batch_norm = LayerNorm(d_model)
        self.pointwise_conv2 = _Conv1d(d_model, d_model, 1)

    def forward(
        self, op: OpBuilder, x: ir.Value, *, cache_time: ir.Value | None = None
    ) -> ir.Value | tuple[ir.Value, ir.Value]:
        """Conformer convolution.

        Offline (``cache_time is None``): the depthwise conv uses zero causal
        left-padding.  Cache-aware streaming (``cache_time`` given, shape
        ``(B, C, kernel_size - 1)``): the depthwise conv's left context comes
        from ``cache_time`` (the previous chunk's last GLU-output frames) and
        the updated time cache is returned alongside the output.
        """
        x = op.Transpose(x, perm=[0, 2, 1])  # (B, C, T)
        x = self.pointwise_conv1(op, x)  # (B, 2C, T)
        a, b = op.Split(x, axis=1, num_outputs=2, _outputs=2)
        x = op.Mul(a, op.Sigmoid(b))  # GLU over channel axis -> depthwise input
        new_cache = None
        if cache_time is None:
            x = self.depthwise_conv(op, x)
        else:
            x, new_cache = self.depthwise_conv(op, x, cache=cache_time)
        # LayerNorm over channels: transpose to (B, T, C)
        x = op.Transpose(x, perm=[0, 2, 1])
        x = self.batch_norm(op, x)
        x = _swish(op, x)
        x = op.Transpose(x, perm=[0, 2, 1])  # (B, C, T)
        x = self.pointwise_conv2(op, x)
        x = op.Transpose(x, perm=[0, 2, 1])  # (B, T, C)
        if new_cache is None:
            return x
        return x, new_cache


class ConformerLayer(nn.Module):
    """A single FastConformer block (Macaron FF + rel-pos MHA + conv + FF)."""

    _FC_FACTOR = 0.5

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        d = config.hidden_size
        d_ff = config.intermediate_size
        self.norm_feed_forward1 = LayerNorm(d)
        self.feed_forward1 = ConformerFeedForward(d, d_ff)
        self.norm_self_att = LayerNorm(d)
        self.self_attn = RelPositionMultiHeadAttention(d, config.num_attention_heads)
        self.norm_conv = LayerNorm(d)
        self.conv = ConformerConvolution(d, config.fastconformer_conv_kernel_size)
        self.norm_feed_forward2 = LayerNorm(d)
        self.feed_forward2 = ConformerFeedForward(d, d_ff)
        self.norm_out = LayerNorm(d)

    def forward(
        self,
        op: OpBuilder,
        x: ir.Value,
        pos_emb: ir.Value,
        att_bias: ir.Value,
        *,
        cache_channel: ir.Value | None = None,
        cache_time: ir.Value | None = None,
        cache_len: int | None = None,
    ) -> ir.Value | tuple[ir.Value, ir.Value, ir.Value]:
        """Conformer block (Macaron FF + rel-pos MHA + conv + FF).

        Offline (no cache args): standard full-context forward returning ``x``.
        Cache-aware streaming (``cache_channel`` ``(B, cache_len, d)`` and
        ``cache_time`` ``(B, d, k-1)`` given): queries from the current chunk
        attend over ``concat(cache, current)``; returns ``(x, new_channel,
        new_time)`` carrying the updated caches (NeMo ``ConformerLayer.forward``
        streaming path; the cached attention state is the normed input).
        """
        streaming = cache_channel is not None
        fc = _scalar_like(op, self._FC_FACTOR, x)
        residual = op.Add(
            x, op.Mul(self.feed_forward1(op, self.norm_feed_forward1(op, x)), fc)
        )
        att_in = self.norm_self_att(op, residual)
        if streaming:
            assert cache_len is not None
            # Keys/values come from concat(cache, current); the normed input is
            # what gets cached for the next chunk.
            kv = op.Concat(cache_channel, att_in, axis=1)  # (B, cache_len + T, d)
            att = self.self_attn(op, att_in, pos_emb, att_bias, kv_x=kv)
            new_channel = op.Slice(
                kv,
                op.Constant(value_ints=[-cache_len]),
                op.Constant(value_ints=[int(np.iinfo(np.int64).max)]),
                op.Constant(value_ints=[1]),
            )  # last cache_len frames of concat(cache, att_in)
        else:
            att = self.self_attn(op, att_in, pos_emb, att_bias)
        residual = op.Add(residual, att)
        conv_in = self.norm_conv(op, residual)
        if streaming:
            conv, new_time = self.conv(op, conv_in, cache_time=cache_time)
        else:
            conv = self.conv(op, conv_in)
        residual = op.Add(residual, conv)
        residual = op.Add(
            residual,
            op.Mul(self.feed_forward2(op, self.norm_feed_forward2(op, residual)), fc),
        )
        out = self.norm_out(op, residual)
        if streaming:
            return out, new_channel, new_time
        return out


# Encoder
# ---------------------------------------------------------------------------


class FastConformerEncoder(nn.Module):
    """FastConformer encoder: subsampling + N Conformer layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self._dtype = config.dtype
        self.pre_encode = ConvSubsampling(
            config.fastconformer_feat_in,
            config.fastconformer_subsampling_conv_channels,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [ConformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # Relative-position divisor term: exp(arange(0,d,2) * -log(1e4)/d)
        d = config.hidden_size
        self._div_term = np.exp(
            np.arange(0, d, 2, dtype=np.float32) * -(math.log(_INF_VAL) / d)
        )
        # chunked_limited mask parameters (att_context_size = [left, right])
        left, right = config.fastconformer_att_context_size
        self._chunk_size = right + 1
        self._left_chunks = left // self._chunk_size if left >= 0 else 10000
        # Cache-aware streaming sizes.
        self._cache_size = config.fastconformer_streaming_cache_size
        self._drop_extra = config.fastconformer_streaming_drop_extra

    def _pos_emb(self, op: OpBuilder, t: ir.Value) -> ir.Value:
        # positions = arange(T-1, -T, -1) (length 2T-1), as float
        one = op.Constant(value=ir.tensor(np.float32(1.0)))
        t_f = op.Cast(op.Squeeze(t), to=ir.DataType.FLOAT)
        start = op.Sub(t_f, one)
        limit = op.Neg(t_f)
        positions = op.Range(start, limit, op.Neg(one))  # (L,)
        positions = op.Unsqueeze(positions, op.Constant(value_ints=[1]))  # (L, 1)
        div = op.Constant(value=ir.tensor(self._div_term))  # (d/2,)
        div = op.Unsqueeze(div, op.Constant(value_ints=[0]))  # (1, d/2)
        angles = op.Mul(positions, div)  # (L, d/2)
        sin = op.Unsqueeze(op.Sin(angles), op.Constant(value_ints=[-1]))
        cos = op.Unsqueeze(op.Cos(angles), op.Constant(value_ints=[-1]))
        pe = op.Concat(sin, cos, axis=-1)  # (L, d/2, 2)
        ll = _dim(op, angles, 0)
        d_full = op.Constant(value_ints=[self.config.hidden_size])
        pe = op.Reshape(pe, op.Concat(ll, d_full, axis=0))  # (L, d)
        pe = op.Unsqueeze(pe, op.Constant(value_ints=[0]))  # (1, L, d)
        # Sin/Cos run in float32; cast to the compute dtype for f16/bf16 models
        # so the projection MatMul (linear_pos) sees matching element types.
        if self._dtype != ir.DataType.FLOAT:
            pe = op.Cast(pe, to=self._dtype)
        return pe

    def _subsampled_length(self, op: OpBuilder, length: ir.Value) -> ir.Value:
        """Apply the 8x causal subsampling length formula to *length*.

        Each stride-2 causal stage maps ``n -> floor(n / 2) + 1`` (NeMo
        ``ConvSubsampling.calc_length`` with ``all_paddings == kernel_size``).
        """
        x = op.Cast(length, to=ir.DataType.FLOAT)
        two = op.Constant(value=ir.tensor(np.float32(2.0)))
        one = op.Constant(value=ir.tensor(np.float32(1.0)))
        for _ in range(3):
            x = op.Add(op.Floor(op.Div(x, two)), one)
        return op.Cast(x, to=ir.DataType.INT64)

    def _att_bias(self, op: OpBuilder, t: ir.Value, valid: ir.Value) -> ir.Value:
        # chunk_idx[i] = floor(i / chunk_size); keep iff
        # 0 <= chunk_idx[i] - chunk_idx[j] <= left_chunks
        t_s = op.Squeeze(t)
        zero = op.Constant(value=ir.tensor(np.int64(0)))
        one = op.Constant(value=ir.tensor(np.int64(1)))
        idx = op.Range(zero, t_s, one)  # (T,)
        idx_f = op.Cast(idx, to=ir.DataType.FLOAT)
        cs = op.Constant(value=ir.tensor(np.float32(self._chunk_size)))
        chunk = op.Floor(op.Div(idx_f, cs))  # (T,)
        chunk_i = op.Unsqueeze(chunk, op.Constant(value_ints=[1]))  # (T, 1)
        chunk_j = op.Unsqueeze(chunk, op.Constant(value_ints=[0]))  # (1, T)
        diff = op.Sub(chunk_i, chunk_j)  # (T, T)
        ge0 = op.GreaterOrEqual(diff, op.Constant(value=ir.tensor(np.float32(0.0))))
        le = op.LessOrEqual(
            diff, op.Constant(value=ir.tensor(np.float32(float(self._left_chunks))))
        )
        keep = op.And(ge0, le)  # (T, T) bool, True = attend
        # -> (1, 1, T, T)
        keep = op.Unsqueeze(keep, op.Constant(value_ints=[0, 1]))
        # Combine with the key-padding mask so queries never attend to padded
        # frames: valid is (B, T) -> (B, 1, 1, T).
        valid_key = op.Unsqueeze(valid, op.Constant(value_ints=[1, 2]))
        return op.And(keep, valid_key)  # (B, 1, T, T)

    def _att_bias_streaming(
        self,
        op: OpBuilder,
        tq: ir.Value,
        tk: ir.Value,
        cache_last_channel_len: ir.Value,
        length_sub: ir.Value,
    ) -> ir.Value:
        """Streaming chunked-limited keep-mask of shape ``(B, 1, Tq, Tk)``.

        Queries are the current chunk frames at global indices
        ``cache_size .. cache_size + Tq - 1``; keys span the full
        ``[cache | chunk]`` window of length ``Tk = cache_size + Tq``.  Combines
        the chunked-limited context rule with a per-sample key-padding mask:
        valid key columns are ``[cache_size - cache_len, length_sub +
        cache_size)`` (the populated cache tail plus the current chunk).
        """
        cache_size = self._cache_size
        zero = op.Constant(value=ir.tensor(np.int64(0)))
        one = op.Constant(value=ir.tensor(np.int64(1)))
        cs64 = op.Constant(value=ir.tensor(np.int64(cache_size)))
        # Global key / query indices.
        kj = op.Range(zero, op.Squeeze(tk), one)  # (Tk,)
        qi = op.Range(cs64, op.Add(cs64, op.Squeeze(tq)), one)  # (Tq,)
        cs = op.Constant(value=ir.tensor(np.float32(self._chunk_size)))
        chunk_q = op.Floor(op.Div(op.Cast(qi, to=ir.DataType.FLOAT), cs))  # (Tq,)
        chunk_k = op.Floor(op.Div(op.Cast(kj, to=ir.DataType.FLOAT), cs))  # (Tk,)
        diff = op.Sub(
            op.Unsqueeze(chunk_q, op.Constant(value_ints=[1])),  # (Tq, 1)
            op.Unsqueeze(chunk_k, op.Constant(value_ints=[0])),  # (1, Tk)
        )  # (Tq, Tk)
        ge0 = op.GreaterOrEqual(diff, op.Constant(value=ir.tensor(np.float32(0.0))))
        le = op.LessOrEqual(
            diff, op.Constant(value=ir.tensor(np.float32(float(self._left_chunks))))
        )
        keep = op.And(ge0, le)  # (Tq, Tk)
        keep = op.Unsqueeze(keep, op.Constant(value_ints=[0, 1]))  # (1, 1, Tq, Tk)
        # Per-sample key-padding mask over the [cache | chunk] axis.
        offset = op.Sub(cs64, cache_last_channel_len)  # (B,)
        pad_len = op.Add(length_sub, cs64)  # (B,)
        kj_b = op.Unsqueeze(kj, op.Constant(value_ints=[0]))  # (1, Tk)
        ge_off = op.GreaterOrEqual(kj_b, op.Unsqueeze(offset, op.Constant(value_ints=[1])))
        lt_len = op.Less(kj_b, op.Unsqueeze(pad_len, op.Constant(value_ints=[1])))
        valid_k = op.And(ge_off, lt_len)  # (B, Tk)
        valid_k = op.Unsqueeze(valid_k, op.Constant(value_ints=[1, 2]))  # (B, 1, 1, Tk)
        return op.And(keep, valid_k)  # (B, 1, Tq, Tk)

    def forward(
        self,
        op: OpBuilder,
        audio_signal: ir.Value,
        length: ir.Value,
        *,
        cache_last_channel: ir.Value | None = None,
        cache_last_time: ir.Value | None = None,
        cache_last_channel_len: ir.Value | None = None,
    ):
        """FastConformer encoder forward.

        Offline (no cache args): consumes the full feature sequence and returns
        ``(encoder_output (B, d, T'), encoder_length (B,))``.

        Cache-aware streaming (cache args given): consumes a single feature
        chunk plus NeMo's per-layer caches and returns ``(encoder_output,
        encoder_length, cache_last_channel_next, cache_last_time_next,
        cache_last_channel_len_next)``.
        """
        if cache_last_channel is not None:
            return self._forward_streaming(
                op,
                audio_signal,
                length,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            )
        x = self.pre_encode(op, audio_signal)  # (B, T', d)
        t = _dim(op, x, 1)
        # Subsampled per-sample lengths and the (B, T') validity mask.
        enc_len = self._subsampled_length(op, length)  # (B,)
        t_s = op.Squeeze(t)
        frame_idx = op.Range(
            op.Constant(value=ir.tensor(np.int64(0))),
            t_s,
            op.Constant(value=ir.tensor(np.int64(1))),
        )  # (T',)
        # valid[b, i] = i < enc_len[b]
        valid = op.Less(
            op.Unsqueeze(frame_idx, op.Constant(value_ints=[0])),  # (1, T')
            op.Unsqueeze(enc_len, op.Constant(value_ints=[1])),  # (B, 1)
        )  # (B, T') bool
        pos_emb = self._pos_emb(op, t)
        att_bias = self._att_bias(op, t, valid)
        for layer in self.layers:
            x = layer(op, x, pos_emb, att_bias)
        # Zero padded frames so the padded region carries no garbage (matches
        # NeMo, which masks padded encoder outputs to zero).
        zero = _scalar_like(op, 0.0, x)
        x = op.Where(op.Unsqueeze(valid, op.Constant(value_ints=[2])), x, zero)
        # (B, T', d) -> (B, d, T')
        return op.Transpose(x, perm=[0, 2, 1]), enc_len

    def _forward_streaming(
        self,
        op: OpBuilder,
        audio_signal: ir.Value,
        length: ir.Value,
        cache_last_channel: ir.Value,
        cache_last_time: ir.Value,
        cache_last_channel_len: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value, ir.Value]:
        """Cache-aware streaming step over a single feature chunk.

        Inputs (NeMo ``ConformerEncoder.forward`` streaming signature):

        * ``audio_signal``           ``(B, feat_in, T_chunk)``
        * ``length``                 ``(B,)`` valid feature frames per sample
        * ``cache_last_channel``     ``(L, B, cache_size, d)`` per-layer attention cache
        * ``cache_last_time``        ``(L, B, d, k-1)`` per-layer conv cache
        * ``cache_last_channel_len`` ``(B,)`` populated cache length per sample

        Returns ``(encoder_output (B, d, T_out), encoder_length (B,),
        cache_last_channel_next, cache_last_time_next,
        cache_last_channel_len_next)``.
        """
        de = self._drop_extra
        cache_size = self._cache_size
        intmax = int(np.iinfo(np.int64).max)
        x = self.pre_encode(op, audio_signal)  # (B, T_sub, d)
        length_sub = self._subsampled_length(op, length)  # (B,)
        # Drop the leading subsampled frames corrupted by chunk-boundary padding
        # (NeMo ``drop_extra_pre_encoded``), and shrink the lengths accordingly.
        x = op.Slice(
            x,
            op.Constant(value_ints=[de]),
            op.Constant(value_ints=[intmax]),
            op.Constant(value_ints=[1]),
        )  # (B, T_out, d)
        length_sub = op.Max(
            op.Sub(length_sub, op.Constant(value=ir.tensor(np.int64(de)))),
            op.Constant(value=ir.tensor(np.int64(0))),
        )
        tq = _dim(op, x, 1)  # (1,) = T_out
        # Relative-position embedding spans the full [cache | chunk] window.
        tk = op.Add(tq, op.Constant(value_ints=[cache_size]))  # (1,) = cache_size + T_out
        pos_emb = self._pos_emb(op, tk)
        att_bias = self._att_bias_streaming(op, tq, tk, cache_last_channel_len, length_sub)
        new_channels = []
        new_times = []
        for i, layer in enumerate(self.layers):
            ch = op.Squeeze(
                op.Slice(
                    cache_last_channel,
                    op.Constant(value_ints=[i]),
                    op.Constant(value_ints=[i + 1]),
                    op.Constant(value_ints=[0]),
                ),
                op.Constant(value_ints=[0]),
            )  # (B, cache_size, d)
            ct = op.Squeeze(
                op.Slice(
                    cache_last_time,
                    op.Constant(value_ints=[i]),
                    op.Constant(value_ints=[i + 1]),
                    op.Constant(value_ints=[0]),
                ),
                op.Constant(value_ints=[0]),
            )  # (B, d, k-1)
            x, nch, nct = layer(
                op,
                x,
                pos_emb,
                att_bias,
                cache_channel=ch,
                cache_time=ct,
                cache_len=cache_size,
            )
            new_channels.append(op.Unsqueeze(nch, op.Constant(value_ints=[0])))
            new_times.append(op.Unsqueeze(nct, op.Constant(value_ints=[0])))
        cache_channel_next = op.Concat(*new_channels, axis=0)  # (L, B, cache_size, d)
        cache_time_next = op.Concat(*new_times, axis=0)  # (L, B, d, k-1)
        # Grow the populated cache length by this chunk's frame count, capped.
        new_len = op.Min(
            op.Add(cache_last_channel_len, op.Squeeze(tq)),
            op.Constant(value=ir.tensor(np.int64(cache_size))),
        )
        out = op.Transpose(x, perm=[0, 2, 1])  # (B, d, T_out)
        return out, length_sub, cache_channel_next, cache_time_next, new_len


# ---------------------------------------------------------------------------
# RNN-T prediction network
# ---------------------------------------------------------------------------


class _LSTMWeights(nn.Module):
    """ONNX LSTM weights for a single layer: W (1,4H,in), R (1,4H,H), B (1,8H)."""

    def __init__(self, hidden: int, input_size: int):
        super().__init__()
        self.W = nn.Parameter([1, 4 * hidden, input_size])
        self.R = nn.Parameter([1, 4 * hidden, hidden])
        self.B = nn.Parameter([1, 8 * hidden])

    def forward(self, op: OpBuilder):
        # Returning the parameters here ensures __call__ realizes them as
        # graph initializers with fully-qualified names.
        return self.W, self.R, self.B


class RNNTPrediction(nn.Module):
    """RNN-T prediction network: embedding + multi-layer LSTM.

    The embedding table is extended by one row (a zero "start-of-sequence"
    vector at index ``vocab+1``) so the SOS step NeMo prepends as a zero
    embedding can be expressed by feeding the SOS id.  Inputs carry explicit
    LSTM state so the decoder can be stepped incrementally at inference time.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.hidden = config.rnnt_pred_hidden
        self.num_layers = config.rnnt_pred_rnn_layers
        # vocab (incl. blank) + 1 zero SOS row
        self.sos_id = config.rnnt_num_classes + 1
        self.embed = Embedding(config.rnnt_num_classes + 2, self.hidden)
        h = self.hidden
        # ONNX LSTM weights per layer: W/R (1, 4H, in/H), B (1, 8H)
        self.weights = nn.ModuleList([_LSTMWeights(h, h) for _ in range(self.num_layers)])

    def forward(
        self,
        op: OpBuilder,
        targets: ir.Value,
        state_h: ir.Value,
        state_c: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        # targets: (B, U) -> embed -> (B, U, H) -> (U, B, H)
        y = self.embed(op, targets)
        x = op.Transpose(y, perm=[1, 0, 2])  # (U, B, H)
        h_out = []
        c_out = []
        for i, layer in enumerate(self.weights):
            w, r, b = layer(op)
            init_h = op.Slice(
                state_h,
                op.Constant(value_ints=[i]),
                op.Constant(value_ints=[i + 1]),
                op.Constant(value_ints=[0]),
            )  # (1, B, H)
            init_c = op.Slice(
                state_c,
                op.Constant(value_ints=[i]),
                op.Constant(value_ints=[i + 1]),
                op.Constant(value_ints=[0]),
            )
            y_all, y_h, y_c = op.LSTM(
                x,
                w,
                r,
                b,
                None,
                init_h,
                init_c,
                hidden_size=self.hidden,
                _outputs=3,
            )
            # y_all: (U, 1, B, H) -> (U, B, H) for next layer
            x = op.Squeeze(y_all, op.Constant(value_ints=[1]))
            h_out.append(y_h)
            c_out.append(y_c)
        new_h = op.Concat(*h_out, axis=0) if self.num_layers > 1 else h_out[0]
        new_c = op.Concat(*c_out, axis=0) if self.num_layers > 1 else c_out[0]
        # x: (U, B, H) -> (B, H, U)
        g = op.Transpose(x, perm=[1, 2, 0])
        return g, new_h, new_c


# ---------------------------------------------------------------------------
# RNN-T joint network
# ---------------------------------------------------------------------------


class RNNTJoint(nn.Module):
    """RNN-T joint network: project encoder + prediction, add, ReLU, project."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        d_joint = config.rnnt_joint_hidden
        self.enc = Linear(config.hidden_size, d_joint, bias=True)
        self.pred = Linear(config.rnnt_pred_hidden, d_joint, bias=True)
        # vocab + blank
        self.out = Linear(d_joint, config.rnnt_num_classes + 1, bias=True)

    def forward(
        self,
        op: OpBuilder,
        encoder_outputs: ir.Value,
        decoder_outputs: ir.Value,
    ) -> ir.Value:
        # encoder_outputs: (B, d_model, T') -> (B, T', d_model)
        f = op.Transpose(encoder_outputs, perm=[0, 2, 1])
        f = self.enc(op, f)  # (B, T', d_joint)
        # decoder_outputs: (B, d_pred, U) -> (B, U, d_pred)
        g = op.Transpose(decoder_outputs, perm=[0, 2, 1])
        g = self.pred(op, g)  # (B, U, d_joint)
        f = op.Unsqueeze(f, op.Constant(value_ints=[2]))  # (B, T', 1, d_joint)
        g = op.Unsqueeze(g, op.Constant(value_ints=[1]))  # (B, 1, U, d_joint)
        x = op.Relu(op.Add(f, g))  # (B, T', U, d_joint)
        logits = self.out(op, x)  # (B, T', U, vocab+1)
        # NeMo's joint applies log_softmax over the vocabulary axis for
        # inference (standard RNN-T log-prob output; argmax-invariant).
        return op.LogSoftmax(logits, axis=-1)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class EncDecRNNTModel(nn.Module):
    """NeMo FastConformer-RNNT (transducer) speech-to-text model.

    Holds the three sub-modules (:attr:`encoder`, :attr:`prediction`,
    :attr:`joint`) that the :class:`~mobius.tasks._rnnt.RNNTTask` wires into
    three separate ONNX graphs.
    """

    default_task: str = "fastconformer-rnnt"
    category: str = "Speech-to-Text"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.encoder = FastConformerEncoder(config)
        self.prediction = RNNTPrediction(config)
        self.joint = RNNTJoint(config)

    def preprocess_weights(self, state_dict):
        """Map NeMo state-dict names to this module's ONNX parameter names.

        * ``encoder.*`` names already match :class:`FastConformerEncoder`.
        * ``decoder.prediction.embed.weight`` is extended by a zero SOS row.
        * ``decoder.prediction.dec_rnn.lstm.*`` PyTorch LSTM gates (i,f,g,o)
          are converted to ONNX layout (i,o,f,g) and packed into W/R/B.
        * ``joint.joint_net.2.*`` is renamed to ``joint.out.*``.
        """
        import torch

        def reorder_gates(t: torch.Tensor) -> torch.Tensor:
            # PyTorch LSTM gate order (i, f, g, o) -> ONNX (i, o, f, g)
            i, f, g, o = torch.chunk(t, 4, dim=0)
            return torch.cat([i, o, f, g], dim=0)

        out: dict[str, torch.Tensor] = {}
        lstm: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                out[key] = value
            elif key == "decoder.prediction.embed.weight":
                zero_sos = torch.zeros(1, value.shape[1], dtype=value.dtype)
                out["prediction.embed.weight"] = torch.cat([value, zero_sos], dim=0)
            elif key.startswith("decoder.prediction.dec_rnn.lstm."):
                lstm[key.rsplit(".", 1)[-1]] = value
            elif key.startswith(("joint.enc.", "joint.pred.")):
                out[key] = value
            elif key.startswith("joint.joint_net.2."):
                out["joint.out." + key.split(".")[-1]] = value
            # other keys (preprocessor.*) are dropped — not part of the graph

        num_layers = self.config.rnnt_pred_rnn_layers
        for layer in range(num_layers):
            w_ih = reorder_gates(lstm[f"weight_ih_l{layer}"])  # (4H, in)
            w_hh = reorder_gates(lstm[f"weight_hh_l{layer}"])  # (4H, H)
            b_ih = reorder_gates(lstm[f"bias_ih_l{layer}"])  # (4H,)
            b_hh = reorder_gates(lstm[f"bias_hh_l{layer}"])  # (4H,)
            out[f"prediction.weights.{layer}.W"] = w_ih.unsqueeze(0)
            out[f"prediction.weights.{layer}.R"] = w_hh.unsqueeze(0)
            out[f"prediction.weights.{layer}.B"] = torch.cat([b_ih, b_hh]).unsqueeze(0)

        return out
