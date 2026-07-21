# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Streaming multi-talker Parakeet RNN-T ASR model (NVIDIA NeMo).

Replicates the cache-aware streaming forward pass of NeMo's
``EncDecMultiTalkerRNNTBPEModel`` (``nvidia/multitalker-parakeet-streaming-0.6b-v1``),
a FastConformer + RNN-T transducer derived from ``nemotron-speech-streaming``
with learnable *speaker kernels* injected at the pre-encode layer.

The export produces **three** ONNX graphs, matching the
``nemotron-speech-streaming`` deployment contract:

1. ``encoder`` — cache-aware streaming FastConformer.
   Inputs: ``audio_signal`` ``[B, T_mel, feat]``, ``length`` ``[B]``,
   ``cache_last_channel`` ``[B, L, C, D]``, ``cache_last_time``
   ``[B, L, D, K-1]``, ``cache_last_channel_len`` ``[B]``, plus the two
   speaker-activity masks ``spk_mask`` / ``bg_mask`` ``[B, T]``.
   Outputs: ``encoder_outputs`` ``[B, T, D]``, ``encoded_lengths`` ``[B]``
   and the three updated caches.
2. ``decoder`` — RNN-T prediction network (embedding + 2-layer LSTM).
   Inputs: ``targets`` ``[B, U]``, ``h_in`` / ``c_in`` ``[2, B, H]``.
   Outputs: ``decoder_outputs`` ``[B, H, U]``, ``h_out`` / ``c_out``.
3. ``joint`` — RNN-T joint network.
   Inputs: ``encoder_outputs`` ``[B, T, D]``, ``decoder_outputs``
   ``[B, U, H]``.  Output: ``logits`` ``[B, T, U, V]``.

The autoregressive greedy-decode loop that ties the three graphs together
runs in host code, outside the static graphs.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import BaseModelConfig
from mobius.components import LayerNorm, Linear

# The Parakeet subsampling is implemented locally (it uses
# ``causal_downsampling=True`` — asymmetric conv padding).


def _swish(op: OpBuilder, x: ir.Value) -> ir.Value:
    """Swish / SiLU activation: ``x * sigmoid(x)``."""
    return op.Mul(x, op.Sigmoid(x))


class _ReLU(nn.Module):
    """ReLU activation as a module (used inside the subsampling ModuleList)."""

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return op.Relu(x)


class _Conv2d(nn.Module):
    """Conv2d with weight + bias and explicit asymmetric padding.

    ``pads`` follows the ONNX layout ``[H_begin, W_begin, H_end, W_end]``
    (``H`` = time, ``W`` = frequency) so causal (left-heavy) padding can be
    expressed for the ``dw_striding`` subsampling stages.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        pads: tuple[int, int, int, int] = (0, 0, 0, 0),
        groups: int = 1,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            [out_channels, in_channels // groups, kernel_size, kernel_size]
        )
        self.bias = nn.Parameter([out_channels])
        self._kernel_size = kernel_size
        self._stride = stride
        self._pads = list(pads)
        self._groups = groups

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self._kernel_size, self._kernel_size],
            strides=[self._stride, self._stride],
            pads=self._pads,
            group=self._groups,
        )


class _CausalConvSubsampling(nn.Module):
    """NeMo ``dw_striding`` conv subsampling with ``causal_downsampling=True``.

    Three stride-2 stages (regular Conv2d then two depthwise-separable
    depthwise+pointwise stages, each with a ReLU).  The stride-2 convs use
    NeMo ``CausalConv2D`` padding ``left = kernel - 1 = 2``,
    ``right = stride - 1 = 1`` on **both** the time and frequency axes, so the
    subsampled frequency dimension is ``floor(F / 2) + 1`` per stage
    (128 -> 65 -> 33 -> 17), not the symmetric-padding ``F / 2``.
    The final Linear projects the flattened channel/frequency axis to
    ``feat_out``.  Weight names match NeMo ``pre_encode.conv.{0,2,3,5,6}``.

    Input:  ``[B, T_mel, feat_in]``
    Output: ``[B, T', feat_out]`` with ``T' = floor(T_mel / 2 ... )`` (8x).
    """

    def __init__(self, feat_in: int, conv_channels: int, feat_out: int):
        super().__init__()
        # Frequency after three causal stride-2 stages: floor(F/2)+1 each.
        freq = feat_in
        for _ in range(3):
            freq = freq // 2 + 1

        c = conv_channels
        # Causal stride-2 padding (ONNX [H_begin, W_begin, H_end, W_end]).
        cpad = (2, 2, 1, 1)
        self.conv = nn.ModuleList(
            [
                _Conv2d(1, c, 3, stride=2, pads=cpad),  # 0: regular
                _ReLU(),  # 1
                _Conv2d(c, c, 3, stride=2, pads=cpad, groups=c),  # 2: depthwise
                _Conv2d(c, c, 1),  # 3: pointwise
                _ReLU(),  # 4
                _Conv2d(c, c, 3, stride=2, pads=cpad, groups=c),  # 5: depthwise
                _Conv2d(c, c, 1),  # 6: pointwise
                _ReLU(),  # 7
            ]
        )
        self.out = Linear(conv_channels * freq, feat_out)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: [B, T_mel, feat_in] -> add channel dim -> [B, 1, T_mel, feat_in]
        x = op.Unsqueeze(x, [1])
        for layer in self.conv:
            x = layer(op, x)
        # [B, C, T', F'] -> [B, T', C, F'] -> [B, T', C * F']
        x = op.Transpose(x, perm=[0, 2, 1, 3])
        x = op.Reshape(x, [0, 0, -1])
        return self.out(op, x)


class _Conv1d(nn.Module):
    """Conv1d with explicit weight and optional bias.

    Unlike the Sortformer ``_Conv1d`` (always biased), the Parakeet encoder
    convolutions are bias-free (``use_bias=false``), so bias is optional.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter([out_channels, in_channels // groups, kernel_size])
        self.bias = nn.Parameter([out_channels]) if bias else None
        self._kernel_size = kernel_size
        self._padding = padding
        self._groups = groups

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        args = [x, self.weight] + ([self.bias] if self.bias is not None else [])
        return op.Conv(
            *args,
            kernel_shape=[self._kernel_size],
            strides=[1],
            pads=[self._padding, self._padding],
            group=self._groups,
        )


if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

# Slice "end" sentinel meaning "to the end of the axis".
_INT64_MAX = 9223372036854775807


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ParakeetMultiTalkerConfig(BaseModelConfig):
    """Configuration for the streaming multi-talker Parakeet RNN-T model.

    Fields mirror the NeMo ``.nemo`` ``model_config.yaml`` (``encoder``,
    ``decoder`` and ``joint`` sections) plus the streaming cache sizes.
    """

    # Mel feature dimension (preprocessor ``features``).
    feat_in: int = 128
    # FastConformer encoder.
    d_model: int = 1024
    num_layers: int = 24
    num_heads: int = 8
    ff_expansion: int = 4
    conv_kernel: int = 9
    subsampling_conv_channels: int = 256
    subsampling_factor: int = 8
    # Streaming cache sizes (att left-context and causal-conv context).
    att_left_context: int = 70
    att_right_context: int = 13
    # RNN-T prediction network.
    pred_hidden: int = 640
    pred_rnn_layers: int = 2
    # RNN-T joint network.
    joint_hidden: int = 640
    vocab_size: int = 1025  # includes the RNN-T blank token

    @property
    def last_channel_cache_size(self) -> int:
        """Attention left-context cache length (``att_left_context``)."""
        return self.att_left_context

    @property
    def conv_cache_size(self) -> int:
        """Causal depthwise-conv cache length (``kernel_size - 1``)."""
        return self.conv_kernel - 1

    def validate(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

    @classmethod
    def from_nemo_yaml(cls, cfg: dict) -> ParakeetMultiTalkerConfig:
        """Build a config from a parsed NeMo ``model_config.yaml`` mapping."""
        enc = cfg.get("encoder", {})
        dec = cfg.get("decoder", {})
        joint = cfg.get("joint", {})
        pre = cfg.get("preprocessor", {})
        # att_context_size is a list of [left, right] modes; the first mode is
        # used at inference.
        att_ctx = enc.get("att_context_size", [[70, 13]])
        if att_ctx and isinstance(att_ctx[0], list):
            left, right = att_ctx[0]
        else:
            left, right = att_ctx
        pred_cfg = dec.get("prednet", {})
        joint_cfg = joint.get("jointnet", {})
        vocab = len(cfg.get("labels", [])) or joint.get("num_classes", 1024)
        return cls(
            feat_in=int(enc.get("feat_in", pre.get("features", 128))),
            d_model=int(enc.get("d_model", 1024)),
            num_layers=int(enc.get("n_layers", 24)),
            num_heads=int(enc.get("n_heads", 8)),
            ff_expansion=int(enc.get("ff_expansion_factor", 4)),
            conv_kernel=int(enc.get("conv_kernel_size", 9)),
            subsampling_conv_channels=int(enc.get("subsampling_conv_channels", 256)),
            subsampling_factor=int(enc.get("subsampling_factor", 8)),
            att_left_context=int(left),
            att_right_context=int(right),
            pred_hidden=int(pred_cfg.get("pred_hidden", 640)),
            pred_rnn_layers=int(pred_cfg.get("pred_rnn_layers", 2)),
            joint_hidden=int(joint_cfg.get("joint_hidden", 640)),
            vocab_size=int(vocab) + 1,  # + blank
        )


# ---------------------------------------------------------------------------
# Streaming FastConformer encoder
# ---------------------------------------------------------------------------


class _StreamingConformerFeedForward(nn.Module):
    """Conformer feed-forward (Linear -> Swish -> Linear), bias-free.

    ``use_bias=false`` in the Parakeet encoder, so both projections omit bias.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, bias=False)
        self.linear2 = Linear(d_ff, d_model, bias=False)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return self.linear2(op, _swish(op, self.linear1(op, x)))


class _StreamingConformerConvolution(nn.Module):
    """Causal Conformer convolution with a streaming conv cache.

    Structure (channels-first internally, all convs bias-free):
        pointwise_conv1 (C -> 2C) -> GLU -> [prepend cache] -> depthwise_conv
        (kernel k, groups=C, causal) -> LayerNorm -> Swish -> pointwise_conv2.

    The depthwise conv is *causal*: the left ``k-1`` context comes from the
    conv cache (``cache_last_time``) rather than zero padding, and the updated
    cache is the last ``k-1`` frames of ``[cache, glu_out]``.  ``batch_norm`` is
    the NeMo attribute name for the ``layer_norm`` normalization.
    """

    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self._kernel_size = kernel_size
        self._cache_size = kernel_size - 1
        self.pointwise_conv1 = _Conv1d(d_model, d_model * 2, 1, bias=False)
        # Causal depthwise conv: no internal padding (cache supplies left ctx).
        self.depthwise_conv = _Conv1d(
            d_model, d_model, kernel_size, padding=0, groups=d_model, bias=False
        )
        # NeMo names the layer-norm ``batch_norm`` even for conv_norm_type=layer_norm.
        self.batch_norm = LayerNorm(d_model, eps=1e-5)
        self.pointwise_conv2 = _Conv1d(d_model, d_model, 1, bias=False)

    def forward(
        self, op: OpBuilder, x: ir.Value, cache: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        # x: [B, T, C], cache: [B, C, k-1]
        x = op.Transpose(x, perm=[0, 2, 1])  # [B, C, T]
        x = self.pointwise_conv1(op, x)  # [B, 2C, T]
        # GLU over the channel axis: first * sigmoid(second)
        first, second = op.Split(x, axis=1, num_outputs=2, _outputs=2)
        x = op.Mul(first, op.Sigmoid(second))  # [B, C, T]
        # Prepend the conv cache along time -> [B, C, (k-1) + T].
        x_cat = op.Concat(cache, x, axis=2)
        # New cache = last (k-1) frames of [cache, glu_out].
        new_cache = op.Slice(
            x_cat,
            op.Constant(value_ints=[-self._cache_size]),
            op.Constant(value_ints=[_INT64_MAX]),
            op.Constant(value_ints=[2]),
        )
        # Causal depthwise conv consumes the cached left context (no padding).
        x = self.depthwise_conv(op, x_cat)  # [B, C, T]
        # LayerNorm over channels: [B, C, T] -> [B, T, C] -> LN -> [B, C, T].
        x = op.Transpose(x, perm=[0, 2, 1])
        x = self.batch_norm(op, x)
        x = op.Transpose(x, perm=[0, 2, 1])
        x = _swish(op, x)
        x = self.pointwise_conv2(op, x)
        return op.Transpose(x, perm=[0, 2, 1]), new_cache  # [B, T, C]


class _StreamingRelPositionMultiHeadAttention(nn.Module):
    """Transformer-XL relative-position attention with a streaming KV cache.

    The keys/values are ``concat([cache_last_channel, x])`` so each chunk
    attends to ``C`` frames of left context plus the current ``T`` frames; the
    queries are the current ``T`` frames only.  An additive mask implements the
    ``chunked_limited`` band and the cache-length padding.  The updated cache is
    ``concat([cache[:, T:], x])`` (the most recent ``C`` frames).  All
    projections are bias-free (``use_bias=false``).
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self.linear_q = Linear(d_model, d_model, bias=False)
        self.linear_k = Linear(d_model, d_model, bias=False)
        self.linear_v = Linear(d_model, d_model, bias=False)
        self.linear_out = Linear(d_model, d_model, bias=False)
        self.linear_pos = Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter([num_heads, self._head_dim])
        self.pos_bias_v = nn.Parameter([num_heads, self._head_dim])

    def _to_heads(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # [B, T, D] -> [B, T, H, d_k] -> [B, H, T, d_k]
        x = op.Reshape(x, [0, 0, self._num_heads, self._head_dim])
        return op.Transpose(x, perm=[0, 2, 1, 3])

    def _rel_shift(
        self, op: OpBuilder, x: ir.Value, q_dim: ir.Value, key_dim: ir.Value
    ) -> ir.Value:
        """Relative shift of ``[B, H, T, P]`` scores, sliced to ``key_len``.

        Reproduces NeMo's ``rel_shift`` for the cache-aware case where the
        positional length ``P = 2*(T+C)-1`` differs from the key length
        ``key_len = T + C``: left-pad, reshape/drop to realign, then keep the
        first ``key_len`` columns.
        """
        bh = op.Shape(x, start=0, end=2)  # [B, H]
        # 1) left-pad last dim by 1 -> [B, H, T, P+1]
        x = op.Pad(x, op.Constant(value_ints=[0, 0, 0, 1, 0, 0, 0, 0]))
        # 2) reshape to [B, H, P+1, T]
        shape_a = op.Concat(bh, op.Constant(value_ints=[-1]), q_dim, axis=0)
        x = op.Reshape(x, shape_a)
        # 3) drop the first row along axis 2 -> [B, H, P, T]
        x = op.Slice(
            x,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[_INT64_MAX]),
            op.Constant(value_ints=[2]),
        )
        # 4) reshape back to [B, H, T, P]
        shape_b = op.Concat(bh, q_dim, op.Constant(value_ints=[-1]), axis=0)
        x = op.Reshape(x, shape_b)
        # 5) keep the first key_len columns -> [B, H, T, key_len]
        return op.Slice(x, op.Constant(value_ints=[0]), key_dim, op.Constant(value_ints=[3]))

    def forward(
        self,
        op: OpBuilder,
        x: ir.Value,
        cache: ir.Value,
        pos_emb: ir.Value,
        att_bias: ir.Value,
        q_dim: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        # x: [B, T, D], cache: [B, C, D], pos_emb: [1, 2(T+C)-1, D]
        # Keys/values attend to the left-context cache + current frames.
        kv = op.Concat(cache, x, axis=1)  # [B, C+T, D]
        key_dim = op.Shape(kv, start=1, end=2)  # [C+T]
        # New cache: most recent C frames = concat([cache[:, T:], x]).
        new_cache = op.Concat(
            op.Slice(
                cache,
                q_dim,
                op.Constant(value_ints=[_INT64_MAX]),
                op.Constant(value_ints=[1]),
            ),
            x,
            axis=1,
        )

        q = self._to_heads(op, self.linear_q(op, x))  # [B, H, T, d_k]
        k = self._to_heads(op, self.linear_k(op, kv))  # [B, H, C+T, d_k]
        v = self._to_heads(op, self.linear_v(op, kv))
        p = self._to_heads(op, self.linear_pos(op, pos_emb))  # [1, H, P, d_k]

        bias_u = op.Unsqueeze(self.pos_bias_u, [0, 2])  # [1, H, 1, d_k]
        bias_v = op.Unsqueeze(self.pos_bias_v, [0, 2])
        q_u = op.Add(q, bias_u)
        q_v = op.Add(q, bias_v)

        matrix_ac = op.MatMul(q_u, op.Transpose(k, perm=[0, 1, 3, 2]))  # [B,H,T,C+T]
        matrix_bd = op.MatMul(q_v, op.Transpose(p, perm=[0, 1, 3, 2]))  # [B,H,T,P]
        matrix_bd = self._rel_shift(op, matrix_bd, q_dim, key_dim)  # [B,H,T,C+T]

        scale = op.Constant(value=ir.tensor(np.array(self._head_dim**-0.5, np.float32)))
        scores = op.Mul(op.Add(matrix_ac, matrix_bd), scale)
        scores = op.Add(scores, att_bias)  # additive chunked/pad mask
        attn = op.Softmax(scores, axis=-1)
        out = op.MatMul(attn, v)  # [B, H, T, d_k]
        out = op.Transpose(out, perm=[0, 2, 1, 3])  # [B, T, H, d_k]
        out = op.Reshape(out, [0, 0, -1])  # [B, T, D]
        return self.linear_out(op, out), new_cache


class _StreamingConformerLayer(nn.Module):
    """Single streaming FastConformer block (Macaron feed-forward)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, kernel_size: int):
        super().__init__()
        self.norm_feed_forward1 = LayerNorm(d_model, eps=1e-5)
        self.feed_forward1 = _StreamingConformerFeedForward(d_model, d_ff)
        self.norm_self_att = LayerNorm(d_model, eps=1e-5)
        self.self_attn = _StreamingRelPositionMultiHeadAttention(d_model, num_heads)
        self.norm_conv = LayerNorm(d_model, eps=1e-5)
        self.conv = _StreamingConformerConvolution(d_model, kernel_size)
        self.norm_feed_forward2 = LayerNorm(d_model, eps=1e-5)
        self.feed_forward2 = _StreamingConformerFeedForward(d_model, d_ff)
        self.norm_out = LayerNorm(d_model, eps=1e-5)

    def forward(
        self,
        op: OpBuilder,
        x: ir.Value,
        cache_channel: ir.Value,
        cache_time: ir.Value,
        pos_emb: ir.Value,
        att_bias: ir.Value,
        q_dim: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        # Macaron feed-forward (half-step residual)
        ff = self.feed_forward1(op, self.norm_feed_forward1(op, x))
        x = op.Add(x, op.Mul(ff, op.Constant(value_float=0.5)))
        # Relative-position self-attention with streaming KV cache
        normed = self.norm_self_att(op, x)
        att, new_channel = self.self_attn(op, normed, cache_channel, pos_emb, att_bias, q_dim)
        x = op.Add(x, att)
        # Causal convolution module with streaming conv cache
        conv_out, new_time = self.conv(op, self.norm_conv(op, x), cache_time)
        x = op.Add(x, conv_out)
        # Macaron feed-forward (half-step residual)
        ff = self.feed_forward2(op, self.norm_feed_forward2(op, x))
        x = op.Add(x, op.Mul(ff, op.Constant(value_float=0.5)))
        return self.norm_out(op, x), new_channel, new_time


class _SpeakerKernel(nn.Module):
    """Speaker kernel (``ff`` type): Linear -> ReLU -> (Dropout) -> Linear.

    The two linears carry bias (unlike the encoder body).  Module indices ``0``
    and ``3`` match NeMo's ``nn.Sequential(Linear, ReLU, Dropout, Linear)``.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Placeholder indices 1 (ReLU) and 2 (Dropout) carry no weights; only
        # indices 0 and 3 are Linear layers, matching the NeMo weight names.
        self.layers = nn.ModuleList(
            [
                Linear(d_model, d_model),
                _ReLU(),
                _Identity(),
                Linear(d_model, d_model),
            ]
        )

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        for layer in self.layers:
            x = layer(op, x)
        return x


class _Identity(nn.Module):
    """No-op module occupying the Dropout slot in the speaker kernel."""

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return x


class _StreamingFastConformerEncoder(nn.Module):
    """Cache-aware streaming FastConformer encoder with speaker kernels."""

    def __init__(self, config: ParakeetMultiTalkerConfig):
        super().__init__()
        d_model = config.d_model
        self._d_model = d_model
        self._num_layers = config.num_layers
        self._cache_channel = config.last_channel_cache_size
        self._chunk_size = config.att_right_context + 1
        self._left_chunks = config.att_left_context // self._chunk_size
        self.pre_encode = _CausalConvSubsampling(
            config.feat_in, config.subsampling_conv_channels, d_model
        )
        d_ff = d_model * config.ff_expansion
        self.layers = nn.ModuleList(
            [
                _StreamingConformerLayer(d_model, config.num_heads, d_ff, config.conv_kernel)
                for _ in range(config.num_layers)
            ]
        )
        # Speaker kernels injected at layer 0 (pre-hook in NeMo).
        self.spk_kernels = _SpeakerKernel(d_model)
        self.bg_spk_kernels = _SpeakerKernel(d_model)
        # Sinusoidal frequency divisors: exp(arange(0, d, 2) * -log(10000)/d).
        self._div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model)
        )

    def _build_pos_emb(self, op: OpBuilder, input_len: ir.Value) -> ir.Value:
        """Build Transformer-XL relative positional embeddings.

        ``input_len`` is a scalar ``int64`` equal to ``T + C`` (current chunk
        plus the attention cache).  Returns ``pos_emb`` of shape
        ``[1, 2*input_len - 1, d_model]`` for positions ``input_len-1, ...,
        -(input_len-1)`` (sin even columns, cos odd columns).
        """
        one = op.Constant(value_int=1)
        start = op.Sub(input_len, one)  # input_len - 1
        limit = op.Neg(input_len)  # last emitted value is -(input_len-1)
        positions = op.Range(start, limit, op.Constant(value_int=-1))  # [2L-1]
        pos_f = op.Unsqueeze(op.Cast(positions, to=ir.DataType.FLOAT), [-1])
        div_term = op.Constant(value=ir.tensor(self._div_term))  # [d/2]
        angles = op.Mul(pos_f, div_term)  # [2L-1, d/2]
        sin = op.Unsqueeze(op.Sin(angles), [-1])
        cos = op.Unsqueeze(op.Cos(angles), [-1])
        interleaved = op.Concat(sin, cos, axis=-1)  # [2L-1, d/2, 2]
        pe = op.Reshape(interleaved, [0, -1])  # [2L-1, d]
        return op.Unsqueeze(pe, [0])  # [1, 2L-1, d]

    def _build_att_bias(
        self,
        op: OpBuilder,
        t_dim: ir.Value,
        length: ir.Value,
        cache_len: ir.Value,
    ) -> ir.Value:
        """Additive attention mask for chunked_limited streaming attention.

        Builds a ``[B, 1, T, C+T]`` additive bias (0 for visible, -inf for
        masked) combining the chunked-limited band with the cache-length
        padding mask.  ``t_dim`` is ``[T]``, ``length`` is ``[B]`` (valid
        current frames), ``cache_len`` is ``[B]`` (valid cache frames).
        """
        c = op.Constant(value_int=self._cache_channel)
        # max_audio = C + T (total keys after cache concat).
        t_scalar = op.Squeeze(t_dim)
        max_audio = op.Add(c, t_scalar)  # scalar
        max_audio_1d = op.Unsqueeze(max_audio, [0])  # [1]
        positions = op.Range(
            op.Constant(value_int=0), max_audio, op.Constant(value_int=1)
        )  # [C+T]
        chunk = op.Constant(value_int=self._chunk_size)
        chunk_idx = op.Div(positions, chunk)  # [C+T] integer trunc div
        diff = op.Sub(op.Unsqueeze(chunk_idx, [1]), op.Unsqueeze(chunk_idx, [0]))  # [C+T, C+T]
        left = op.Constant(value_int=self._left_chunks)
        zero = op.Constant(value_int=0)
        chunked = op.And(
            op.LessOrEqual(diff, left), op.GreaterOrEqual(diff, zero)
        )  # [C+T, C+T] bool
        # Pad-validity per batch: position in [C - cache_len, length + C).
        pos_row = op.Unsqueeze(positions, [0])  # [1, C+T]
        upper = op.Add(op.Unsqueeze(length, [1]), c)  # [B, 1]
        lower = op.Sub(c, op.Unsqueeze(cache_len, [1]))  # [B, 1]
        valid = op.And(op.Less(pos_row, upper), op.GreaterOrEqual(pos_row, lower))  # [B, C+T]
        pad_att = op.And(op.Unsqueeze(valid, [1]), op.Unsqueeze(valid, [2]))  # [B, C+T, C+T]
        allowed = op.And(op.Unsqueeze(chunked, [0]), pad_att)  # [B, C+T, C+T]
        # Keep query rows for the current chunk: rows [C:].
        allowed = op.Slice(
            allowed, op.Unsqueeze(c, [0]), max_audio_1d, op.Constant(value_ints=[1])
        )  # [B, T, C+T]
        # Additive bias: 0 where allowed, -inf where masked.
        neg_inf = op.Constant(value=ir.tensor(np.array(-1e9, np.float32)))
        zero_f = op.Constant(value=ir.tensor(np.array(0.0, np.float32)))
        bias = op.Where(allowed, zero_f, neg_inf)  # [B, T, C+T]
        return op.Unsqueeze(bias, [1])  # [B, 1, T, C+T]

    def forward(
        self,
        op: OpBuilder,
        audio_signal: ir.Value,
        length: ir.Value,
        cache_last_channel: ir.Value,
        cache_last_time: ir.Value,
        cache_last_channel_len: ir.Value,
        spk_mask: ir.Value,
        bg_mask: ir.Value,
    ):
        # audio_signal: [B, T_mel, feat] -> subsample -> [B, T_pe, D]
        x = self.pre_encode(op, audio_signal)
        # Drop the extra pre-encoded frames produced by cache overlap (2).
        drop = self._drop_extra_pre_encoded()
        x = op.Slice(
            x,
            op.Constant(value_ints=[drop]),
            op.Constant(value_ints=[_INT64_MAX]),
            op.Constant(value_ints=[1]),
        )  # [B, T, D]
        t_dim = op.Shape(x, start=1, end=2)  # [T]
        t_scalar = op.Squeeze(t_dim)
        cache_len_scalar = op.Constant(value_int=self._cache_channel)
        input_len = op.Add(t_scalar, cache_len_scalar)  # T + C
        pos_emb = self._build_pos_emb(op, input_len)

        # Encoded length: subsample the mel length by 8 (three causal
        # stride-2 stages, each ``floor(L / 2) + 1``), then drop the extra
        # pre-encoded frames.  Clamped at 0.
        enc_len = length
        for _ in range(3):
            enc_len = op.Add(
                op.Div(enc_len, op.Constant(value_ints=[2])),
                op.Constant(value_ints=[1]),
            )
        enc_len = op.Max(
            op.Sub(enc_len, op.Constant(value_ints=[drop])),
            op.Constant(value_ints=[0]),
        )
        att_bias = self._build_att_bias(op, t_dim, enc_len, cache_last_channel_len)

        # Speaker kernel injection at layer 0 (pre-hook):
        #   x = x + spk(x * spk_mask);  x = x + bg(x * bg_mask)
        spk_m = op.Unsqueeze(spk_mask, [2])  # [B, T, 1]
        bg_m = op.Unsqueeze(bg_mask, [2])
        x = op.Add(x, self.spk_kernels(op, op.Mul(x, spk_m)))
        x = op.Add(x, self.bg_spk_kernels(op, op.Mul(x, bg_m)))

        # Split the per-layer caches: [B, L, ...] -> L x [B, ...].
        channel_layers = op.Split(
            cache_last_channel,
            axis=1,
            num_outputs=self._num_layers,
            _outputs=self._num_layers,
        )
        time_layers = op.Split(
            cache_last_time,
            axis=1,
            num_outputs=self._num_layers,
            _outputs=self._num_layers,
        )
        if self._num_layers == 1:
            channel_layers = [channel_layers]
            time_layers = [time_layers]

        new_channels: list[ir.Value] = []
        new_times: list[ir.Value] = []
        for i, layer in enumerate(self.layers):
            cc = op.Squeeze(channel_layers[i], [1])  # [B, C, D]
            ct = op.Squeeze(time_layers[i], [1])  # [B, D, k-1]
            x, nc, nt = layer(op, x, cc, ct, pos_emb, att_bias, t_dim)
            new_channels.append(op.Unsqueeze(nc, [1]))  # [B, 1, C, D]
            new_times.append(op.Unsqueeze(nt, [1]))  # [B, 1, D, k-1]

        cc_next = op.Concat(*new_channels, axis=1)  # [B, L, C, D]
        ct_next = op.Concat(*new_times, axis=1)  # [B, L, D, k-1]
        # Updated channel-cache fill level: min(cache_size, prev_len + T).
        ccl_next = op.Min(
            op.Add(cache_last_channel_len, enc_len),
            op.Constant(value_ints=[self._cache_channel]),
        )
        return x, enc_len, cc_next, ct_next, ccl_next

    def _drop_extra_pre_encoded(self) -> int:
        # dw_striding subsampling with pre_encode_cache overlap drops the first
        # ``subsampling_factor // 4`` re-encoded frames (2 for factor 8).
        return 2


# ---------------------------------------------------------------------------
# RNN-T prediction network (decoder)
# ---------------------------------------------------------------------------


class _PredictionNetwork(nn.Module):
    """RNN-T prediction network: embedding + stacked LSTM.

    Mirrors NeMo's ``RNNTDecoder`` export path (``predict`` with
    ``add_sos=False``): ``targets [B, U] -> embed -> LSTM -> g [B, U, H]`` with
    LSTM state ``(h, c)`` each ``[L, B, H]``.
    """

    def __init__(self, config: ParakeetMultiTalkerConfig):
        super().__init__()
        self._hidden = config.pred_hidden
        self._num_layers = config.pred_rnn_layers
        self.embed = _Embedding(config.vocab_size, config.pred_hidden)
        # ONNX LSTM weights are supplied per layer as initializers.
        self.dec_rnn = _StackedLSTM(config.pred_hidden, config.pred_rnn_layers)

    def forward(
        self, op: OpBuilder, targets: ir.Value, h_in: ir.Value, c_in: ir.Value
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        # targets: [B, U] int64 -> [B, U, H]
        emb = self.embed(op, targets)
        # ONNX LSTM expects [seq, batch, input]; NeMo transposes to [U, B, H].
        emb = op.Transpose(emb, perm=[1, 0, 2])
        g, h_out, c_out = self.dec_rnn(op, emb, h_in, c_in)
        # g: [U, B, H] -> [B, U, H]
        g = op.Transpose(g, perm=[1, 0, 2])
        return g, h_out, c_out


class _Embedding(nn.Module):
    """Token embedding lookup (``embed.weight`` [vocab, hidden])."""

    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self.weight = nn.Parameter([vocab_size, hidden])

    def forward(self, op: OpBuilder, tokens: ir.Value) -> ir.Value:
        return op.Gather(self.weight, tokens, axis=0)


class _StackedLSTM(nn.Module):
    """Stacked uni-directional LSTM built from per-layer ONNX ``LSTM`` ops.

    The PyTorch ``nn.LSTM`` gate order ``(i, f, g, o)`` is reordered to the ONNX
    ``(i, o, f, g)`` order in :meth:`preprocess_weights`.  Each layer keeps its
    own ``weight_ih/weight_hh/bias`` initializers so the parameter names match
    the NeMo ``dec_rnn.lstm.*_l{n}`` checkpoint keys (via preprocessing).
    """

    def __init__(self, hidden: int, num_layers: int):
        super().__init__()
        self._hidden = hidden
        self._num_layers = num_layers
        self.lstm = nn.ModuleList([_LSTMLayerWeights(hidden) for _ in range(num_layers)])

    def forward(
        self, op: OpBuilder, x: ir.Value, h_in: ir.Value, c_in: ir.Value
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        # x: [seq, batch, input]; h_in/c_in: [num_layers, batch, hidden]
        h_layers = op.Split(
            h_in, axis=0, num_outputs=self._num_layers, _outputs=self._num_layers
        )
        c_layers = op.Split(
            c_in, axis=0, num_outputs=self._num_layers, _outputs=self._num_layers
        )
        if self._num_layers == 1:
            h_layers = [h_layers]
            c_layers = [c_layers]

        h_outs: list[ir.Value] = []
        c_outs: list[ir.Value] = []
        for i in range(self._num_layers):
            # Delegate to the layer module so its parameters are realized
            # (qualified + registered as initializers) under the ``lstm.{i}``
            # scope; accessing nested params without invoking ``__call__``
            # leaves them unregistered.
            x, y_h, y_c = self.lstm[i](op, x, h_layers[i], c_layers[i])
            h_outs.append(y_h)  # [1, batch, H]
            c_outs.append(y_c)
        h_out = op.Concat(*h_outs, axis=0)  # [num_layers, batch, H]
        c_out = op.Concat(*c_outs, axis=0)
        return x, h_out, c_out


class _LSTMLayerWeights(nn.Module):
    """One uni-directional ONNX ``LSTM`` layer with its own initializers.

    ``weight_ih`` [1, 4H, in], ``weight_hh`` [1, 4H, H], ``bias`` [1, 8H]
    (concatenated ``[Wb, Rb]``).  Populated by ``preprocess_weights``.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self._hidden = hidden
        self.weight_ih = nn.Parameter([1, 4 * hidden, hidden])
        self.weight_hh = nn.Parameter([1, 4 * hidden, hidden])
        self.bias = nn.Parameter([1, 8 * hidden])

    def forward(
        self, op: OpBuilder, x: ir.Value, h0: ir.Value, c0: ir.Value
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        # ONNX LSTM: Y [seq, num_dir, batch, H], Y_h/Y_c [num_dir, batch, H]
        y, y_h, y_c = op.LSTM(
            x,
            self.weight_ih,  # [1, 4H, in]
            self.weight_hh,  # [1, 4H, H]
            self.bias,  # [1, 8H]
            None,  # sequence_lens
            h0,  # initial_h [num_dir=1, batch, H]
            c0,  # initial_c
            hidden_size=self._hidden,
            direction="forward",
            _outputs=3,
        )
        # Squeeze the num_directions axis -> [seq, batch, H].
        return op.Squeeze(y, [1]), y_h, y_c


# ---------------------------------------------------------------------------
# RNN-T joint network
# ---------------------------------------------------------------------------


class _JointNetwork(nn.Module):
    """RNN-T joint network.

    ``enc(f) [B,T,J]`` + ``pred(g) [B,U,J]`` -> broadcast-sum ``[B,T,U,J]`` ->
    ReLU -> ``joint_net.2`` Linear -> ``[B, T, U, V]``.  Raw logits are emitted
    (no log-softmax); argmax is unchanged.
    """

    def __init__(self, config: ParakeetMultiTalkerConfig):
        super().__init__()
        self.enc = Linear(config.d_model, config.joint_hidden)
        self.pred = Linear(config.pred_hidden, config.joint_hidden)
        self.joint_net = _JointFinal(config.joint_hidden, config.vocab_size)

    def forward(
        self, op: OpBuilder, encoder_outputs: ir.Value, decoder_outputs: ir.Value
    ) -> ir.Value:
        # encoder_outputs: [B, T, D]; decoder_outputs: [B, U, H]
        f = self.enc(op, encoder_outputs)  # [B, T, J]
        g = self.pred(op, decoder_outputs)  # [B, U, J]
        f = op.Unsqueeze(f, [2])  # [B, T, 1, J]
        g = op.Unsqueeze(g, [1])  # [B, 1, U, J]
        inp = op.Add(f, g)  # [B, T, U, J]
        inp = op.Relu(inp)
        return self.joint_net(op, inp)  # [B, T, U, V]


class _JointFinal(nn.Module):
    """Final joint Linear, exposed at index ``2`` to match ``joint_net.2.*``."""

    def __init__(self, joint_hidden: int, vocab_size: int):
        super().__init__()
        # Indices 0 (ReLU) and 1 (Dropout) carry no weights; only index 2 is a
        # Linear, matching NeMo's Sequential(ReLU, Dropout, Linear) naming.
        self.layers = nn.ModuleList([_ReLU(), _Identity(), Linear(joint_hidden, vocab_size)])

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # ReLU already applied by the caller; index 2 is the projection.
        return self.layers[2](op, x)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class ParakeetMultiTalkerModel(nn.Module):
    """Streaming multi-talker Parakeet RNN-T model (NVIDIA NeMo).

    Replicates NVIDIA NeMo's ``EncDecMultiTalkerRNNTBPEModel`` cache-aware
    streaming inference: a FastConformer encoder with speaker-kernel injection,
    an RNN-T LSTM prediction network, and an RNN-T joint network.  Exported as
    three ONNX graphs (``encoder``, ``decoder``, ``joint``) matching the
    ``nemotron-speech-streaming`` deployment contract.
    """

    default_task: str = "multitalker-rnnt"
    category: str = "Speech-to-Text"

    def __init__(self, config: ParakeetMultiTalkerConfig):
        super().__init__()
        self.config = config
        self.encoder = _StreamingFastConformerEncoder(config)
        self.decoder = _PredictionNetwork(config)
        self.joint = _JointNetwork(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map NeMo checkpoint names onto this module's parameter names.

        Handles three name families:

        * ``encoder.*`` — near 1:1; speaker kernels ``spk_kernels.0.{0,3}`` and
          ``bg_spk_kernels.0.{0,3}`` map to ``…kernels.layers.{0,3}``.
        * ``decoder.prediction.*`` — the embedding maps directly; the stacked
          LSTM weights are reshaped/gate-reordered to the ONNX ``LSTM`` layout
          (PyTorch ``(i,f,g,o)`` -> ONNX ``(i,o,f,g)``) and concatenated biases.
        * ``joint.*`` — 1:1 (``enc``/``pred``/``joint_net.2``).
        """
        import torch

        cfg = self.config
        h = cfg.pred_hidden
        renamed: dict[str, torch.Tensor] = {}

        def reorder_gates(t: torch.Tensor) -> torch.Tensor:
            # [4H, ...] gate blocks (i, f, g, o) -> ONNX order (i, o, f, g).
            i, f, g, o = torch.split(t, h, dim=0)
            return torch.cat([i, o, f, g], dim=0)

        for key, value in state_dict.items():
            if key.endswith("num_batches_tracked"):
                continue
            if key.startswith("preprocessor."):
                continue

            # ---- Speaker kernels (top-level in the NeMo checkpoint) ----
            # spk_kernels.0.N -> encoder.spk_kernels.layers.N (the ``.replace``
            # also matches inside ``bg_spk_kernels.0.``).
            if key.startswith(("spk_kernels.", "bg_spk_kernels.")):
                new = "encoder." + key.replace("spk_kernels.0.", "spk_kernels.layers.")
                renamed[new] = value
                continue

            # ---- Encoder ----
            if key.startswith("encoder."):
                renamed[key] = value
                continue

            # ---- Prediction network (LSTM) ----
            if key.startswith("decoder.prediction."):
                sub = key[len("decoder.prediction.") :]
                if sub == "embed.weight":
                    renamed["decoder.embed.weight"] = value
                    continue
                # dec_rnn.lstm.{weight,bias}_{ih,hh}_l{n}
                if sub.startswith("dec_rnn.lstm."):
                    name = sub[len("dec_rnn.lstm.") :]
                    kind, layer = name.rsplit("_l", 1)
                    layer = int(layer)
                    if kind == "weight_ih":
                        w = reorder_gates(value).unsqueeze(0)  # [1, 4H, in]
                        renamed[f"decoder.dec_rnn.lstm.{layer}.weight_ih"] = w
                    elif kind == "weight_hh":
                        w = reorder_gates(value).unsqueeze(0)  # [1, 4H, H]
                        renamed[f"decoder.dec_rnn.lstm.{layer}.weight_hh"] = w
                    # Biases combined below once both ih and hh are seen.
                    continue
                renamed[f"decoder.{sub}"] = value
                continue

            # ---- Joint ----
            if key.startswith("joint."):
                sub = key[len("joint.") :]
                if sub.startswith("joint_net.2."):
                    renamed["joint.joint_net.layers.2." + sub[len("joint_net.2.") :]] = value
                else:
                    renamed[f"joint.{sub}"] = value
                continue

            renamed[key] = value

        # Combine LSTM biases: ONNX B = concat([Wb(i,o,f,g), Rb(i,o,f,g)]).
        for layer in range(cfg.pred_rnn_layers):
            bih = state_dict.get(f"decoder.prediction.dec_rnn.lstm.bias_ih_l{layer}")
            bhh = state_dict.get(f"decoder.prediction.dec_rnn.lstm.bias_hh_l{layer}")
            if bih is not None and bhh is not None:
                bih = reorder_gates(bih)
                bhh = reorder_gates(bhh)
                b = torch.cat([bih, bhh], dim=0).unsqueeze(0)  # [1, 8H]
                renamed[f"decoder.dec_rnn.lstm.{layer}.bias"] = b

        return renamed


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def build_parakeet_multitalker(nemo_path: str, *, dtype: str = "float32"):
    """Build a Parakeet multi-talker RNN-T ``ModelPackage`` from ``.nemo``.

    Extracts the NeMo archive (``model_config.yaml`` + ``model_weights.ckpt``),
    constructs a :class:`ParakeetMultiTalkerConfig`, builds the three streaming
    ONNX graphs (``encoder``/``decoder``/``joint``) and applies the checkpoint
    weights.

    Args:
        nemo_path: Path to a ``.nemo`` archive or an already-extracted
            directory containing ``model_config.yaml`` and
            ``model_weights.ckpt``.
        dtype: Target compute dtype (``"float32"``, ``"float16"`` or
            ``"bfloat16"``).

    Returns:
        A ``ModelPackage`` with ``"encoder"``, ``"decoder"`` and ``"joint"``.
    """
    import os
    import tarfile
    import tempfile

    import torch
    import yaml

    from mobius import build_from_module

    dtype_map = {
        "float32": ir.DataType.FLOAT,
        "float16": ir.DataType.FLOAT16,
        "bfloat16": ir.DataType.BFLOAT16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype}'. Choose from {sorted(dtype_map)}.")

    extract_dir = nemo_path
    tmp: tempfile.TemporaryDirectory | None = None
    if os.path.isfile(nemo_path):
        tmp = tempfile.TemporaryDirectory()
        with tarfile.open(nemo_path, "r:*") as tar:
            # Trusted local checkpoint archive.
            tar.extractall(tmp.name)
        extract_dir = tmp.name

    try:
        with open(os.path.join(extract_dir, "model_config.yaml"), encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)
        state_dict = torch.load(
            os.path.join(extract_dir, "model_weights.ckpt"),
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        config = ParakeetMultiTalkerConfig.from_nemo_yaml(yaml_cfg)
        config.dtype = dtype_map[dtype]
        module = ParakeetMultiTalkerModel(config)
        pkg = build_from_module(module, config, task="multitalker-rnnt")
        pkg.apply_weights(module.preprocess_weights(dict(state_dict)))
        return pkg
    finally:
        if tmp is not None:
            tmp.cleanup()
