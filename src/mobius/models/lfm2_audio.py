# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LFM2-Audio: audio-to-audio model with hybrid conv+attention backbone.

Architecture (4-model ONNX split):
1. **audio_encoder**: ConformerEncoder + adapter MLP
   mel (B, n_mels, T) -> audio embeddings (B, T', hidden_size)
2. **embedding**: text token embed + audio codebook embed
   text_ids + audio_features -> inputs_embeds
3. **decoder**: LFM2 backbone (takes inputs_embeds, not input_ids)
   inputs_embeds -> text_logits + hybrid KV cache
4. **audio_decoder**: depthformer (per-codebook autoregressive transformer)
   backbone_hidden -> codebook_logits (one codebook at a time)

The decoder uses hybrid cache: "conv" layers carry conv_state,
"full_attention" layers carry standard KV cache.

HuggingFace weight name prefixes::

    lfm.              -> decoder sub-model (LFM2 backbone)
    conformer.        -> audio_encoder.encoder (ConformerEncoder)
    audio_adapter.    -> audio_encoder.adapter (projection MLP)
    depthformer.      -> audio_decoder.depthformer
    depth_linear.     -> audio_decoder.depth_linear
    depth_embeddings. -> audio_decoder.depth_embeddings
    embedding_norm.   -> audio_decoder.embedding_norm

Reference: ``liquid_audio.model.lfm2_audio.LFM2AudioModel``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import Lfm2AudioConfig
from mobius._weight_utils import tie_word_embeddings
from mobius.components import (
    MLP,
    Attention,
    Embedding,
    LayerNorm,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.models.lfm2 import Lfm2AttentionDecoderLayer, Lfm2ConvDecoderLayer

if TYPE_CHECKING:
    pass

# Slice "end" sentinel = INT64_MAX, meaning "to the end of axis".
_INT64_MAX = 9223372036854775807


# ---------------------------------------------------------------------------
# NeMo-style Conformer encoder (Liquid LFM2-Audio)
# ---------------------------------------------------------------------------
#
# The LFM2-Audio checkpoint ships a NeMo Conformer encoder whose state-dict
# layout differs from the Sherpa/k2 ``ConformerEncoder`` in
# :mod:`mobius.components`. To match the HF weight names exactly we keep the
# encoder local to this file rather than altering the shared component.
#
# Per-block layout (matches ``nemo.collections.asr.modules.ConformerLayer``):
#
#     norm_feed_forward1  → linear1 → SiLU → linear2     ⇒ x += 0.5·FF1
#     norm_self_att       → relative-position MHA         ⇒ x += attn
#     norm_conv           → pointwise → GLU → depthwise →
#                            BatchNorm → SiLU → pointwise ⇒ x += conv
#     norm_feed_forward2  → linear1 → SiLU → linear2     ⇒ x += 0.5·FF2
#     norm_out            (final LayerNorm)


def _silu(op: OpBuilder, x: ir.Value) -> ir.Value:
    """SiLU/Swish activation: x * sigmoid(x)."""
    return op.Mul(x, op.Sigmoid(x))


class _Conv2dPreEncode(nn.Module):
    """Single Conv2d with bias for the ``pre_encode`` subsampling stack."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            [out_channels, in_channels // groups, kernel_size, kernel_size]
        )
        self.bias = nn.Parameter([out_channels])
        self._kernel_shape = [kernel_size, kernel_size]
        self._strides = [stride, stride]
        self._pads = [padding, padding, padding, padding]
        self._groups = groups

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=self._kernel_shape,
            strides=self._strides,
            pads=self._pads,
            group=self._groups,
        )


class _NeMoSubsampling(nn.Module):
    """NeMo ``pre_encode`` (depthwise-separable conv subsampling).

    Mirrors ``conformer.pre_encode`` in the HF checkpoint::

        conv.0  Conv2d(1, C, 3, stride=2, pad=1)       — initial stride-2
        conv.1  ReLU                                    — activation
        conv.2  Conv2d(C, C, 3, stride=2, pad=1, dw)   — depthwise stride-2
        conv.3  Conv2d(C, C, 1)                         — pointwise mix
        conv.4  ReLU
        conv.5  Conv2d(C, C, 3, stride=2, pad=1, dw)   — depthwise stride-2
        conv.6  Conv2d(C, C, 1)                         — pointwise mix
        conv.7  ReLU
        out     Linear(C * freq_out, d_model)

    The depthwise stages use ``groups=C`` and ``in_channels // groups = 1`` so
    their stored weight shape is ``[C, 1, 3, 3]`` (matches the checkpoint).

    Input shape:  ``[B, T, n_mels]`` (channel = 1 added internally).
    Output shape: ``[B, ceil(T/8), d_model]``.

    Padding-frame masking
    ---------------------
    HF wraps the conv stack in a ``MaskedConvSequential`` that re-zeros
    padded time frames before *every* layer call. This matters because each
    Conv2d carries a bias: an all-zero input frame would otherwise produce
    a non-zero (bias-only) output that then leaks into neighbouring valid
    frames via subsequent stride-2 convolutions. Skipping the mask causes
    a few-tenths-scale parity gap at the trailing edge that attention then
    smears across every output frame (~0.12 max-abs-diff with HF on a
    1-second clip).

    The mobius graph does not receive an explicit ``feat_lengths`` input.
    Instead we *derive* a time mask from the mel itself: padded frames are
    written as exact zeros by the mel preprocessor (after normalization),
    so ``ReduceMax(|mel|, dim=freq) > 0`` recovers the validity mask. After
    each stride-2 layer the per-batch valid length ``L`` is updated via
    ``L_next = (L - 1) // 2 + 1`` (same formula HF uses) and a fresh
    ``arange(T_new) < L_next`` mask is broadcast back over the conv output.
    """

    def __init__(self, n_mels: int, conv_channels: int, d_model: int):
        super().__init__()
        c = conv_channels
        # Three stride-2 stages collapse the frequency axis by 8x.
        freq = n_mels
        for _ in range(3):
            freq = (freq + 2 - 3) // 2 + 1

        # Conv layers as named attributes (no Sequential; we need to interleave
        # masking between them, and depthwise/pointwise convs have different
        # stride/group patterns we want to be explicit about).
        self.conv_0 = _Conv2dPreEncode(1, c, kernel_size=3, stride=2, padding=1)
        self.conv_2 = _Conv2dPreEncode(c, c, kernel_size=3, stride=2, padding=1, groups=c)
        self.conv_3 = _Conv2dPreEncode(c, c, kernel_size=1)
        self.conv_5 = _Conv2dPreEncode(c, c, kernel_size=3, stride=2, padding=1, groups=c)
        self.conv_6 = _Conv2dPreEncode(c, c, kernel_size=1)
        self.out = Linear(c * freq, d_model, bias=True)
        self._conv_channels = c
        self._freq_out = freq

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # -------- derive time-mask from input (padded frames are all zero) --
        # x: [B, T, n_mels]
        abs_x = op.Abs(x)  # [B, T, n_mels]
        frame_max = op.ReduceMax(abs_x, op.Constant(value_ints=[-1]), keepdims=0)  # [B, T]
        zero_like = op.CastLike(op.Constant(value_float=0.0), frame_max)
        valid_bool = op.Greater(frame_max, zero_like)  # [B, T] bool
        valid_f = op.CastLike(valid_bool, x)  # [B, T] in model dtype
        valid_len = op.Cast(
            op.ReduceSum(valid_f, op.Constant(value_ints=[1]), keepdims=0),
            to=ir.DataType.INT64,
        )  # [B] int64 — per-batch number of valid frames

        # x: [B, T, n_mels] → [B, 1, T, n_mels]
        x = op.Unsqueeze(x, [1])
        # mask shape [B, 1, T, 1] broadcast over (C, F)
        mask = op.Unsqueeze(op.Unsqueeze(valid_f, [1]), [-1])

        # Conv stages (3 stride-2 conv2ds + 2 pointwise + ReLU after each
        # depthwise group). Between layers we re-mask and (for stride-2)
        # downsample the validity length.
        x = self._apply_mask_and_conv(op, x, mask, self.conv_0)
        valid_len, mask = self._downsample_mask(op, x, valid_len)
        x = op.Relu(op.Mul(x, mask))  # mask-before-ReLU == mask-after-ReLU here

        x = self._apply_mask_and_conv(op, x, mask, self.conv_2)
        valid_len, mask = self._downsample_mask(op, x, valid_len)
        x = self._apply_mask_and_conv(op, x, mask, self.conv_3)  # 1x1, no stride
        x = op.Relu(op.Mul(x, mask))

        x = self._apply_mask_and_conv(op, x, mask, self.conv_5)
        valid_len, mask = self._downsample_mask(op, x, valid_len)
        x = self._apply_mask_and_conv(op, x, mask, self.conv_6)  # 1x1, no stride
        x = op.Relu(op.Mul(x, mask))

        # Final masking, then [B, C, T', F'] → [B, T', C*F']
        x = op.Mul(x, mask)
        x = op.Transpose(x, perm=[0, 2, 1, 3])
        x = op.Reshape(x, op.Constant(value_ints=[0, 0, self._conv_channels * self._freq_out]))
        return self.out(op, x)

    @staticmethod
    def _apply_mask_and_conv(
        op: OpBuilder, x: ir.Value, mask: ir.Value, conv: _Conv2dPreEncode
    ) -> ir.Value:
        """Zero padded frames, then run a single Conv2d layer."""
        return conv(op, op.Mul(x, mask))

    @staticmethod
    def _downsample_mask(
        op: OpBuilder, conv_out: ir.Value, valid_len: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        """Recompute (length, mask) after a stride-2/k=3/pad=1 Conv2d.

        New per-batch length follows ``L_new = (L - 1) // 2 + 1`` (same as
        HF's ``calculate_conv_output_size`` for k=3, stride=2, pad=(1,1)).
        New mask is ``arange(T_new) < L_new`` broadcast back to ``[B, 1, T_new, 1]``.
        """
        one_i = op.Constant(value_ints=[1])
        two_i = op.Constant(value_ints=[2])
        # L_new = (L - 1) // 2 + 1
        new_len = op.Add(op.Div(op.Sub(valid_len, one_i), two_i), one_i)  # [B]
        # T_new is dim-2 of the conv output ([B, C, T_new, F_new]).
        new_t = op.Shape(conv_out, start=2, end=3)  # [1] int64
        arange = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(new_t, [0]),
            op.Constant(value_int=1),
        )  # [T_new] int64
        # mask_2d[b, t] = t < L_new[b]
        mask_2d = op.Less(
            op.Unsqueeze(arange, [0]),  # [1, T_new]
            op.Unsqueeze(new_len, [1]),  # [B, 1]
        )  # [B, T_new]
        mask_f = op.CastLike(mask_2d, conv_out)
        mask = op.Unsqueeze(op.Unsqueeze(mask_f, [1]), [-1])  # [B, 1, T_new, 1]
        return new_len, mask


class _NeMoFeedForward(nn.Module):
    """Macaron feed-forward sub-block (``linear1 → SiLU → linear2``).

    The pre-LayerNorm and ``0.5`` residual scaling are applied at the
    Conformer-layer level, not inside this module.
    """

    def __init__(self, d_model: int, d_inner: int):
        super().__init__()
        self.linear1 = Linear(d_model, d_inner, bias=True)
        self.linear2 = Linear(d_inner, d_model, bias=True)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return self.linear2(op, _silu(op, self.linear1(op, x)))


class _NeMoConv1d(nn.Module):
    """Conv1d wrapper that mirrors ``torch.nn.Conv1d`` weight names."""

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
        self._kernel_shape = [kernel_size]
        self._pads = [padding, padding]
        self._groups = groups

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        if self.bias is not None:
            return op.Conv(
                x,
                self.weight,
                self.bias,
                kernel_shape=self._kernel_shape,
                pads=self._pads,
                group=self._groups,
            )
        return op.Conv(
            x,
            self.weight,
            kernel_shape=self._kernel_shape,
            pads=self._pads,
            group=self._groups,
        )


class _NeMoBatchNorm1d(nn.Module):
    """1D batch normalization with frozen running statistics.

    Stores the four learned/buffer tensors that ``torch.nn.BatchNorm1d``
    serializes (``weight``, ``bias``, ``running_mean``, ``running_var``) and
    forwards via ONNX ``BatchNormalization`` in inference mode.

    ``num_batches_tracked`` is a scalar bookkeeping buffer in the HF
    checkpoint — it is intentionally not stored here (and is skipped during
    weight loading).
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter([num_features])
        self.bias = nn.Parameter([num_features])
        self.running_mean = nn.Parameter([num_features])
        self.running_var = nn.Parameter([num_features])
        self._eps = eps

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return op.BatchNormalization(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            epsilon=self._eps,
            training_mode=0,
        )


class _NeMoConvBlock(nn.Module):
    """Conformer convolution module (NeMo layout).

    Pipeline (input arrives in ``[B, T, C]`` layout from the layer)::

        x → transpose to [B, C, T]
          → pointwise_conv1 (Conv1d C→2C, k=1)
          → GLU split (first half * sigmoid(second half))
          → depthwise_conv (Conv1d C→C, k=kernel, groups=C, pad=(k-1)/2)
          → batch_norm
          → SiLU
          → pointwise_conv2 (Conv1d C→C, k=1)
          → transpose back to [B, T, C]

    The pre-LayerNorm and residual addition live in
    :class:`_NeMoConformerLayer`.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.pointwise_conv1 = _NeMoConv1d(channels, 2 * channels, kernel_size=1)
        self.depthwise_conv = _NeMoConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
        )
        self.batch_norm = _NeMoBatchNorm1d(channels)
        self.pointwise_conv2 = _NeMoConv1d(channels, channels, kernel_size=1)
        self._channels = channels

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: [B, T, C]
        x = op.Transpose(x, perm=[0, 2, 1])  # [B, C, T]

        x = self.pointwise_conv1(op, x)  # [B, 2C, T]
        # GLU: split along channel dim, gate first half with sigmoid(second).
        first, second = op.Split(x, axis=1, num_outputs=2, _outputs=2)
        x = op.Mul(first, op.Sigmoid(second))  # [B, C, T]

        x = self.depthwise_conv(op, x)  # [B, C, T]
        x = self.batch_norm(op, x)  # [B, C, T]
        x = _silu(op, x)  # [B, C, T]
        x = self.pointwise_conv2(op, x)  # [B, C, T]

        return op.Transpose(x, perm=[0, 2, 1])  # [B, T, C]


class _NeMoRelPosAttention(nn.Module):
    """Relative-position multi-head attention (Transformer-XL / NeMo style).

    Implements the ``RelPositionMultiHeadAttention`` block used in NeMo
    Conformer. The attention scores are::

        AC[b,h,i,j] = (Q[b,h,i] + u[h]) · K[b,h,j]
        BD[b,h,i,j] = (Q[b,h,i] + v[h]) · R[h, i-j]
        scores       = (AC + BD) / sqrt(head_dim)

    where ``u = pos_bias_u`` and ``v = pos_bias_v`` are learned per-head
    biases (shape ``[num_heads, head_dim]``) and ``R = linear_pos(PE)`` is
    a learned projection of a sinusoidal relative-position embedding of
    length ``2T - 1``.

    The ``BD`` matrix is initially computed as ``[B, H, T, 2T-1]`` over all
    relative offsets and then folded down to ``[B, H, T, T]`` via the
    standard *matrix shift* trick.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self._d_model = d_model

        self.linear_q = Linear(d_model, d_model, bias=True)
        self.linear_k = Linear(d_model, d_model, bias=True)
        self.linear_v = Linear(d_model, d_model, bias=True)
        self.linear_out = Linear(d_model, d_model, bias=True)
        # Relative-position projection: no bias in the HF checkpoint.
        self.linear_pos = Linear(d_model, d_model, bias=False)
        # Learned per-head biases u, v.
        self.pos_bias_u = nn.Parameter([num_heads, self._head_dim])
        self.pos_bias_v = nn.Parameter([num_heads, self._head_dim])

        # Precomputed sinusoidal frequencies (1/10000^(2i/d_model)).
        # Stored as a frozen constant so we don't reissue an op.Constant on
        # every forward call. div_term has shape [d_model // 2].
        half = d_model // 2
        log_div = math.log(10000.0) / d_model
        div_term = np.exp(-log_div * np.arange(0, d_model, 2, dtype=np.float32))
        # Pad with a zero column if d_model is odd (NeMo always uses even
        # d_model, so this is purely defensive).
        if div_term.shape[0] != half:
            div_term = np.resize(div_term, (half,))
        self._div_term_np = div_term

    def _build_pos_emb(self, op: OpBuilder, seq_len: ir.Value, dtype) -> ir.Value:
        """Build sinusoidal pos embedding of shape ``[1, 2T-1, d_model]``."""
        one = op.Constant(value_ints=[1])
        two = op.Constant(value_ints=[2])
        two_t = op.Mul(seq_len, two)
        two_t_m1 = op.Sub(two_t, one)
        t_m1 = op.Sub(seq_len, one)

        # positions = [T-1, T-2, ..., -(T-1)] (length 2T-1).
        # Implemented as (T-1) - arange(2T-1) so we don't need a negative step.
        rng = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(two_t_m1, [0]),
            op.Constant(value_int=1),
        )  # [2T-1] int64
        positions = op.Sub(op.Squeeze(t_m1, [0]), rng)  # [2T-1] int64
        positions = op.Cast(positions, to=ir.DataType.FLOAT)
        positions = op.Unsqueeze(positions, [1])  # [2T-1, 1] float32

        div_term = op.Constant(
            value=ir.tensor(self._div_term_np, name="rel_pos_div_term")
        )  # [d_model // 2] float32
        scaled = op.Mul(positions, div_term)  # [2T-1, d_model // 2]

        sin_vals = op.Sin(scaled)
        cos_vals = op.Cos(scaled)
        # Interleave so result[:, 0::2] = sin, result[:, 1::2] = cos.
        sin_u = op.Unsqueeze(sin_vals, [-1])  # [2T-1, d_model//2, 1]
        cos_u = op.Unsqueeze(cos_vals, [-1])
        interleaved = op.Concat(sin_u, cos_u, axis=-1)  # [2T-1, d_model//2, 2]
        pe = op.Reshape(
            interleaved, op.Constant(value_ints=[-1, self._d_model])
        )  # [2T-1, d_model]
        pe = op.Unsqueeze(pe, [0])  # [1, 2T-1, d_model]
        # Cast to model dtype so linear_pos(weight in model dtype) is happy.
        pe = op.Cast(pe, to=dtype)
        return pe

    @staticmethod
    def _rel_shift(
        op: OpBuilder,
        x: ir.Value,
        batch: ir.Value,
        num_heads: int,
        seq_len: ir.Value,
    ) -> ir.Value:
        """Shift ``[B, H, T, 2T-1]`` → ``[B, H, T, T]`` (Transformer-XL trick).

        Standard reference implementation::

            zero_pad = zeros(B, H, T, 1)
            x_padded = cat([zero_pad, x], -1)             # [B, H, T, 2T]
            x_padded = x_padded.view(B, H, 2T, T)
            x = x_padded[:, :, 1:].view(B, H, T, 2T-1)
            return x[:, :, :, :T]
        """
        h_const = op.Constant(value_ints=[num_heads])
        one = op.Constant(value_ints=[1])
        two = op.Constant(value_ints=[2])
        two_t = op.Mul(seq_len, two)
        two_t_m1 = op.Sub(two_t, one)

        # zero_pad [B, H, T, 1] in the same dtype as x.
        pad_shape = op.Concat(batch, h_const, seq_len, one, axis=0)
        zero_scalar = op.CastLike(op.Constant(value_float=0.0), x)
        zero_pad = op.Expand(zero_scalar, pad_shape)

        x_padded = op.Concat(zero_pad, x, axis=-1)  # [B, H, T, 2T]

        # Reshape to [B, H, 2T, T] (same total element count: T * 2T).
        new_shape = op.Concat(batch, h_const, two_t, seq_len, axis=0)
        x_padded = op.Reshape(x_padded, new_shape)

        # Drop the first slot along axis 2: [B, H, 2T-1, T].
        x_padded = op.Slice(
            x_padded,
            starts=op.Constant(value_ints=[1]),
            ends=op.Constant(value_ints=[_INT64_MAX]),
            axes=op.Constant(value_ints=[2]),
        )

        # Reshape back to [B, H, T, 2T-1].
        flat_shape = op.Concat(batch, h_const, seq_len, two_t_m1, axis=0)
        x_padded = op.Reshape(x_padded, flat_shape)

        # Keep only the first T columns: [B, H, T, T].
        return op.Slice(
            x_padded,
            starts=op.Constant(value_ints=[0]),
            ends=seq_len,
            axes=op.Constant(value_ints=[3]),
        )

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: [B, T, d_model]
        num_heads = self._num_heads
        head_dim = self._head_dim
        scale = float(head_dim**-0.5)

        q = self.linear_q(op, x)  # [B, T, d_model]
        k = self.linear_k(op, x)
        v = self.linear_v(op, x)

        # Split heads: [B, T, H, D]. Using 0 for "copy" dims keeps the shape
        # spec static where it can be.
        q = op.Reshape(q, op.Constant(value_ints=[0, 0, num_heads, head_dim]))
        k = op.Reshape(k, op.Constant(value_ints=[0, 0, num_heads, head_dim]))
        v = op.Reshape(v, op.Constant(value_ints=[0, 0, num_heads, head_dim]))

        # q + u, q + v broadcast (H, D) over (B, T, H, D).
        q_u = op.Add(q, self.pos_bias_u)
        q_v = op.Add(q, self.pos_bias_v)

        # Transpose to [B, H, T, D] for batched matmul.
        q_u = op.Transpose(q_u, perm=[0, 2, 1, 3])
        q_v = op.Transpose(q_v, perm=[0, 2, 1, 3])
        k_t = op.Transpose(k, perm=[0, 2, 1, 3])
        v_t = op.Transpose(v, perm=[0, 2, 1, 3])

        # Content-based attention: AC = (Q+u) @ K^T → [B, H, T, T]
        ac = op.MatMul(q_u, op.Transpose(k_t, perm=[0, 1, 3, 2]))

        # Relative-position embedding R = linear_pos(PE)
        seq_len = op.Shape(x, start=1, end=2)
        batch = op.Shape(x, start=0, end=1)
        pos_emb = self._build_pos_emb(op, seq_len, dtype=x.dtype)  # [1, 2T-1, D]
        p = self.linear_pos(op, pos_emb)  # [1, 2T-1, d_model]
        p = op.Reshape(
            p, op.Constant(value_ints=[1, -1, num_heads, head_dim])
        )  # [1, 2T-1, H, D]
        p = op.Transpose(p, perm=[0, 2, 3, 1])  # [1, H, D, 2T-1]

        # BD = (Q+v) @ R → [B, H, T, 2T-1]
        bd = op.MatMul(q_v, p)
        bd = self._rel_shift(op, bd, batch, num_heads, seq_len)  # [B, H, T, T]

        scores = op.Add(ac, bd)
        scores = op.Mul(scores, op.CastLike(op.Constant(value_float=scale), scores))
        attn = op.Softmax(scores, axis=-1)

        context = op.MatMul(attn, v_t)  # [B, H, T, D]
        context = op.Transpose(context, perm=[0, 2, 1, 3])  # [B, T, H, D]
        context = op.Reshape(
            context, op.Constant(value_ints=[0, 0, num_heads * head_dim])
        )  # [B, T, d_model]
        return self.linear_out(op, context)


class _NeMoConformerLayer(nn.Module):
    """Single NeMo Conformer block.

    Forward (matches HF ``ConformerLayer``)::

        x = x + 0.5 * FF1(norm_feed_forward1(x))
        x = x +       attn(norm_self_att(x))
        x = x +       conv(norm_conv(x))
        x = x + 0.5 * FF2(norm_feed_forward2(x))
        x = norm_out(x)
    """

    def __init__(self, d_model: int, num_heads: int, d_inner: int, kernel_size: int):
        super().__init__()
        self.norm_feed_forward1 = LayerNorm(d_model, eps=1e-5)
        self.feed_forward1 = _NeMoFeedForward(d_model, d_inner)
        self.norm_self_att = LayerNorm(d_model, eps=1e-5)
        self.self_attn = _NeMoRelPosAttention(d_model, num_heads)
        self.norm_conv = LayerNorm(d_model, eps=1e-5)
        self.conv = _NeMoConvBlock(d_model, kernel_size)
        self.norm_feed_forward2 = LayerNorm(d_model, eps=1e-5)
        self.feed_forward2 = _NeMoFeedForward(d_model, d_inner)
        self.norm_out = LayerNorm(d_model, eps=1e-5)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        half = op.CastLike(op.Constant(value_float=0.5), x)

        # Macaron FF1: pre-norm, half-scaled residual.
        ff1 = self.feed_forward1(op, self.norm_feed_forward1(op, x))
        x = op.Add(x, op.Mul(ff1, half))

        # Self-attention with relative position embeddings.
        attn = self.self_attn(op, self.norm_self_att(op, x))
        x = op.Add(x, attn)

        # Conv module.
        conv = self.conv(op, self.norm_conv(op, x))
        x = op.Add(x, conv)

        # Macaron FF2.
        ff2 = self.feed_forward2(op, self.norm_feed_forward2(op, x))
        x = op.Add(x, op.Mul(ff2, half))

        return self.norm_out(op, x)


class _NeMoConformerEncoder(nn.Module):
    """Full NeMo Conformer encoder (subsampling + stacked Conformer blocks).

    Weight prefixes match ``conformer.`` in the HF checkpoint::

        pre_encode.*  — subsampling (Conv2d x 3 + Linear)
        layers.K.*    — Conformer blocks (norm_*, feed_forward*, self_attn, conv)

    There is no final encoder LayerNorm — each block's ``norm_out``
    serves as the per-layer output normalization, and the audio adapter
    has its own ``LayerNorm`` on top.

    Input:  ``[B, T, n_mels]``  (callers transpose from ``[B, n_mels, T]``)
    Output: ``[B, ceil(T/8), d_model]``
    """

    def __init__(
        self,
        n_mels: int,
        d_model: int,
        num_heads: int,
        d_inner: int,
        num_layers: int,
        kernel_size: int,
        conv_channels: int,
    ):
        super().__init__()
        self.pre_encode = _NeMoSubsampling(n_mels, conv_channels, d_model)
        self.layers = nn.ModuleList(
            [
                _NeMoConformerLayer(d_model, num_heads, d_inner, kernel_size)
                for _ in range(num_layers)
            ]
        )

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        x = self.pre_encode(op, x)
        for layer in self.layers:
            x = layer(op, x)
        return x


# ---------------------------------------------------------------------------
# Audio Encoder sub-model
# ---------------------------------------------------------------------------


class _Lfm2AudioAdapter(nn.Module):
    """Audio adapter MLP for LFM2-Audio.

    Matches the HuggingFace ``audio_adapter`` Sequential layout exactly::

        model.0 = LayerNorm(encoder_dim)        # weight + bias both [encoder_dim]
        model.1 = Linear(encoder_dim, hidden_size)
        model.2 = GELU                          # no parameters
        model.3 = Linear(hidden_size, hidden_size)

    Output dimension is ``hidden_size`` (the backbone hidden dim), not
    ``encoder_dim`` — i.e. this is *not* a residual MLP, it's a projection
    from the conformer's hidden width up to the LM backbone's hidden width
    followed by a hidden-size-square refinement Linear.
    """

    def __init__(self, encoder_dim: int, hidden_size: int):
        super().__init__()
        self.pre_norm = LayerNorm(encoder_dim, eps=1e-5)
        self.up_proj = Linear(encoder_dim, hidden_size, bias=True)
        self.out_proj = Linear(hidden_size, hidden_size, bias=True)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        x = self.pre_norm(op, x)
        x = self.up_proj(op, x)
        x = op.Gelu(x)
        return self.out_proj(op, x)


class _Lfm2AudioEncoder(nn.Module):
    """NeMo Conformer encoder + adapter MLP for mel → LFM hidden_size projection.

    HuggingFace weight mapping (handled by ``preprocess_weights``)::

        conformer.pre_encode.* -> encoder.pre_encode.*
        conformer.layers.K.*   -> encoder.layers.K.*
        audio_adapter.model.0.{weight,bias} -> adapter.pre_norm.{weight,bias}
        audio_adapter.model.1.{weight,bias} -> adapter.up_proj.{weight,bias}
        audio_adapter.model.3.{weight,bias} -> adapter.out_proj.{weight,bias}
    """

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        audio = config.audio
        assert audio is not None

        encoder_dim = audio.attention_dim or audio.d_model or 512
        num_heads = audio.attention_heads or audio.encoder_attention_heads or 8
        num_layers = audio.num_blocks or audio.encoder_layers or 17
        d_inner = audio.linear_units or audio.encoder_ffn_dim or 2048
        kernel_size = audio.kernel_size or 9
        conv_channels = audio.conv_channels or 256
        n_mels = audio.num_mel_bins or 128

        self.encoder = _NeMoConformerEncoder(
            n_mels=n_mels,
            d_model=encoder_dim,
            num_heads=num_heads,
            d_inner=d_inner,
            num_layers=num_layers,
            kernel_size=kernel_size,
            conv_channels=conv_channels,
        )
        # Adapter: encoder_dim -> hidden_size
        self.adapter = _Lfm2AudioAdapter(encoder_dim, config.hidden_size)

    def forward(self, op: OpBuilder, input_features: ir.Value):
        """Forward: mel (B, n_mels, T) -> (B, T', hidden_size)."""
        # Encoder expects (B, T, n_mels); transpose from (B, n_mels, T).
        input_features = op.Transpose(input_features, perm=[0, 2, 1])
        audio_features = self.encoder(op, input_features)
        return self.adapter(op, audio_features)


# ---------------------------------------------------------------------------
# Embedding sub-model
# ---------------------------------------------------------------------------


class _Lfm2AudioEmbedding(nn.Module):
    """Embedding model for LFM2-Audio.

    Returns text token embeddings for the backbone. Audio codebook embeddings
    are handled by the ``audio_decoder``'s ``depth_embeddings`` — not here.

    Weight names (HF)::

        lfm.embed_tokens.weight -> text_embed.weight
    """

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        self.text_embed = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
    ):
        """Forward: text_ids -> inputs_embeds.

        Returns text embeddings only. Audio codebook embeddings are
        handled separately by the audio_decoder's depth_embeddings.
        Runtime assembles text + audio at the sequence level.
        """
        return self.text_embed(op, input_ids)


# ---------------------------------------------------------------------------
# Decoder sub-model (LFM2 backbone without embed_tokens)
# ---------------------------------------------------------------------------


# Reuse the decoder layers from the base LFM2 model — they are identical
# for the audio backbone. Lfm2AudioConfig inherits from Lfm2Config, so
# the constructors accept it directly.
_Lfm2AudioDecoderLayer = Lfm2AttentionDecoderLayer
_Lfm2AudioConvLayer = Lfm2ConvDecoderLayer


class _Lfm2AudioDecoder(nn.Module):
    """LFM2 decoder backbone: takes inputs_embeds -> logits + cache.

    This is the LFM2 model minus the embedding layer. It takes
    pre-assembled inputs_embeds (from the embedding model) and runs
    the hybrid conv+attention backbone, then projects to vocab logits.

    The text LM head shares weights with embed_tokens (tied).
    """

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        self._dtype = config.dtype

        layer_types = config.layer_types or []
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            ltype = layer_types[i] if i < len(layer_types) else "full_attention"
            if ltype == "conv":
                self.layers.append(_Lfm2AudioConvLayer(config))
            else:
                self.layers.append(_Lfm2AudioDecoderLayer(config))

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        # LM head (tied with lfm.embed_tokens in preprocess_weights)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            # Use a dummy input_ids shape from inputs_embeds
            input_ids=position_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values


# ---------------------------------------------------------------------------
# Audio decoder sub-model (depthformer)
# ---------------------------------------------------------------------------


def _depthformer_intermediate_size(config: Lfm2AudioConfig) -> int:
    """Return the depthformer SwiGLU intermediate size.

    Uses ``config.depthformer_intermediate_size`` when set; otherwise
    derives it from the depthformer hidden dim using the same
    ``block_auto_adjust_ff_dim`` formula as the LFM2 backbone:
    ``round_up(2 * 4 * dim / 3, 256)``.

    For ``depthformer_dim=1024`` this yields ``2816``, matching the
    LFM2-Audio-1.5B checkpoint's ``feed_forward.w*`` rows.
    """
    if config.depthformer_intermediate_size is not None:
        return int(config.depthformer_intermediate_size)
    dim = config.depthformer_dim
    intermediate = int(2 * (4 * dim) / 3)
    multiple_of = 256
    return multiple_of * ((intermediate + multiple_of - 1) // multiple_of)


class _DepthformerLayer(nn.Module):
    """Single depthformer layer with RMSNorm, GQA Attention, and SwiGLU MLP.

    Architecture: RMSNorm -> GQA Attention (head_dim=32, kv_heads=8) ->
    residual -> RMSNorm -> SwiGLU MLP -> residual.

    Mirrors the HuggingFace ``depthformer.layers.K`` block layout::

        operator_norm    (RMSNorm)             -> operator_norm
        operator         (BoundedAttention)    -> self_attn
          .qkv_proj      [num_q*hd + 2*num_kv*hd, dim]
                                                -> q_proj, k_proj, v_proj
          .out_proj      [dim, num_q*hd]       -> o_proj
          .bounded_attention.q_layernorm [hd]  -> q_norm  (per-head RMSNorm)
          .bounded_attention.k_layernorm [hd]  -> k_norm
        ffn_norm         (RMSNorm)             -> ffn_norm
        feed_forward     (SwiGLU MLP)          -> feed_forward
          .w1 [I, dim]                          -> gate_proj
          .w3 [I, dim]                          -> up_proj
          .w2 [dim, I]                          -> down_proj

    Note: ``head_dim`` is **not** ``depthformer_dim // depthformer_heads``.
    LFM2-Audio hardcodes ``head_dim=32`` (so ``num_q = dim // 32``) with
    GQA ``kv_heads=8``.
    """

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        from mobius._configs import ArchitectureConfig

        depthformer_dim = config.depthformer_dim
        head_dim = config.depthformer_head_dim
        num_q_heads = depthformer_dim // head_dim
        num_kv_heads = config.depthformer_kv_heads
        intermediate = _depthformer_intermediate_size(config)

        attn_config = ArchitectureConfig(
            hidden_size=depthformer_dim,
            intermediate_size=intermediate,
            num_attention_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_act="silu",
            attn_qkv_bias=False,
            attn_o_bias=False,
            attn_qk_norm=True,
            attn_qk_norm_full=False,
            rms_norm_eps=1e-5,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.operator_norm = RMSNorm(depthformer_dim, eps=1e-5)
        self.self_attn = Attention(attn_config)
        self.ffn_norm = RMSNorm(depthformer_dim, eps=1e-5)
        self.feed_forward = MLP(attn_config)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        residual = hidden_states
        hidden_states = self.operator_norm(op, hidden_states)
        hidden_states, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.ffn_norm(op, hidden_states)
        hidden_states = self.feed_forward(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states, present_kv


class _DepthCodebookHead(nn.Module):
    """Per-codebook embedding + norm + output head triple.

    Mirrors the HuggingFace ``depth_embeddings.K`` layout::

        embedding       [audio_vocab_size, dim]    Embedding layer
        embedding_norm  [dim]                      RMSNorm before to_logits
        to_logits       [audio_vocab_size, dim]    Linear (tied with embedding
                                                   when ``depthformer_tie=True``)

    The ``embedding`` is used externally (host code) to build the
    ``prev_embedding`` input fed into the depthformer. The ``embedding_norm``
    and ``to_logits`` weights are gathered at runtime through stacked tensors
    so the depthformer forward can select them by ``codebook_idx``.
    """

    def __init__(self, vocab_size: int, dim: int, eps: float = 1e-5):
        super().__init__()
        self.embedding = Embedding(vocab_size, dim)
        self.embedding_norm = RMSNorm(dim, eps=eps)
        self.to_logits = Linear(dim, vocab_size, bias=False)


class _Lfm2AudioDecoderModule(nn.Module):
    """Depthformer audio decoder for per-codebook token prediction.

    Takes the backbone hidden state + the previous codebook embedding,
    runs through depthformer layers, and produces logits for the current
    codebook.

    Architecture::

        depth_linear(backbone_hidden) -> split by codebook_idx ->
        + prev_embedding -> depthformer layers ->
        per-codebook embedding_norm -> per-codebook to_logits -> logits

    Each codebook has its own ``embedding`` / ``embedding_norm`` /
    ``to_logits`` triple (``depth_embeddings.K``), all stored as separate
    state-dict entries to match the HF checkpoint. At runtime, the
    per-codebook norm and head are selected by a single ``Gather`` against
    stacked tensors assembled in :meth:`preprocess_weights`.
    """

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        depthformer_dim = config.depthformer_dim

        # Project backbone hidden -> per-codebook inputs.
        # depth_linear: (hidden_size) -> (codebooks * depthformer_dim).
        self.depth_linear = Linear(
            config.hidden_size,
            config.num_codebooks * depthformer_dim,
            bias=True,
        )

        # Depthformer layers.
        self.layers = nn.ModuleList([])
        for _ in range(config.depthformer_layers):
            self.layers.append(_DepthformerLayer(config))

        # Per-codebook embedding + norm + output head triples. These mirror
        # the HF ``depth_embeddings.K`` modules. The ``embedding`` weights
        # live here for host-side construction of ``prev_embedding`` even
        # though the audio_decoder forward never consumes them directly.
        self.depth_embeddings = nn.ModuleList(
            [
                _DepthCodebookHead(config.audio_vocab_size, depthformer_dim, eps=1e-5)
                for _ in range(config.num_codebooks)
            ]
        )

        # Stacked per-codebook tensors used by forward via Gather. Assembled
        # from the per-codebook triples in ``preprocess_weights``. Same
        # pattern as ``stacked_head_weights`` in :mod:`mobius.models.moshi`.
        self.stacked_norm_weights = nn.Parameter([config.num_codebooks, depthformer_dim])
        # Output head weights: tied with ``embedding.weight`` when
        # ``depthformer_tie=True``, but still shipped as a separate stacked
        # tensor so the ONNX graph remains tie-agnostic.
        self.stacked_head_weights = nn.Parameter(
            [config.num_codebooks, config.audio_vocab_size, depthformer_dim]
        )

        self._depthformer_dim = depthformer_dim
        self._num_codebooks = config.num_codebooks

        # Build a separate RoPE for depthformer (per-step head_dim=32).
        from mobius._configs import ArchitectureConfig

        head_dim = config.depthformer_head_dim
        num_q_heads = depthformer_dim // head_dim
        rope_config = ArchitectureConfig(
            hidden_size=depthformer_dim,
            num_attention_heads=num_q_heads,
            head_dim=head_dim,
            rope_theta=config.rope_theta,
            rope_type="default",
            max_position_embeddings=config.max_position_embeddings,
        )
        self.rotary_emb = initialize_rope(rope_config)

    def forward(
        self,
        op: OpBuilder,
        backbone_hidden: ir.Value,
        prev_embedding: ir.Value,
        codebook_idx: ir.Value,
        past_key_values: list | None = None,
    ):
        """Forward pass for single-codebook prediction.

        Args:
            backbone_hidden: (B, 1, hidden_size) from LFM2 decoder.
            prev_embedding: (B, 1, depthformer_dim) from previous codebook.
            codebook_idx: scalar int — which codebook to predict.
            past_key_values: depthformer KV cache.

        Returns:
            (codebook_logits, present_key_values)
        """
        # Project backbone hidden to all codebook inputs.
        # (B, 1, hidden_size) -> (B, 1, codebooks * depthformer_dim).
        projected = self.depth_linear(op, backbone_hidden)

        # Reshape to (B, codebooks, depthformer_dim) for gathering.
        projected_2d = op.Squeeze(projected, [1])
        projected_3d = op.Reshape(
            projected_2d,
            op.Constant(
                value_ints=[
                    -1,
                    self._num_codebooks,
                    self._depthformer_dim,
                ]
            ),
        )

        # Gather the codebook_idx slice along axis 1: (B, 1, depthformer_dim).
        idx_3d = op.Reshape(codebook_idx, op.Constant(value_ints=[1, 1, 1]))
        batch_dim = op.Shape(projected_3d, start=0, end=1)
        expand_shape = op.Concat(
            batch_dim,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[self._depthformer_dim]),
            axis=0,
        )
        idx_expanded = op.Expand(idx_3d, expand_shape)
        depthformer_input = op.GatherElements(projected_3d, idx_expanded, axis=1)

        # Add previous codebook embedding (depth autoregressive context).
        hidden_states = op.Add(depthformer_input, prev_embedding)

        # Position IDs for depthformer (single step: just codebook_idx),
        # shape (B, 1) derived from the runtime batch dim.
        batch_dim = op.Shape(hidden_states, start=0, end=1)
        one_dim = op.Constant(value_ints=[1])
        position_shape = op.Concat(batch_dim, one_dim, axis=0)
        position_ids = op.Reshape(codebook_idx, position_shape)
        position_embeddings = self.rotary_emb(op, position_ids)

        # Run depthformer layers.
        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=None,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        # Per-codebook RMSNorm: gather weight (depthformer_dim,) by codebook_idx.
        norm_weight = op.Gather(self.stacked_norm_weights, codebook_idx, axis=0)
        hidden_states = op.RMSNormalization(
            hidden_states,
            norm_weight,
            epsilon=1e-5,
            axis=-1,
        )

        # Per-codebook output head: gather to_logits weight
        # (audio_vocab_size, depthformer_dim) and apply.
        head_weight = op.Gather(self.stacked_head_weights, codebook_idx, axis=0)
        head_weight_3d = op.Unsqueeze(head_weight, [0])
        logits = op.MatMul(hidden_states, op.Transpose(head_weight_3d, perm=[0, 2, 1]))

        return logits, present_key_values


# ---------------------------------------------------------------------------
# Composite model
# ---------------------------------------------------------------------------


class Lfm2AudioModel(nn.Module):
    """LFM2-Audio: audio-to-audio model.

    Exports as 4 ONNX models via AudioToAudioTask:
    - audio_encoder: ConformerEncoder + adapter
    - embedding: text + audio embedding fusion
    - decoder: LFM2 hybrid backbone
    - audio_decoder: depthformer per-codebook decoder

    HuggingFace reference: ``liquid_audio.model.lfm2_audio.LFM2AudioModel``.
    """

    default_task: str = "audio-to-audio"
    category: str = "Audio-to-Audio"
    config_class: type = Lfm2AudioConfig

    def __init__(self, config: Lfm2AudioConfig):
        super().__init__()
        self.config = config

        self.audio_encoder = _Lfm2AudioEncoder(config)
        self.embedding = _Lfm2AudioEmbedding(config)
        self.decoder = _Lfm2AudioDecoder(config)
        self.audio_decoder = _Lfm2AudioDecoderModule(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map LFM2-Audio weights to ONNX sub-model parameters.

        Routes weights to sub-models by prefix:
            lfm.embed_tokens.*  -> embedding.text_embed.*
            lfm.*               -> decoder.* (backbone layers)
            conformer.*         -> audio_encoder.encoder.*
            audio_adapter.*     -> audio_encoder.adapter.*
            audio_embedding.*   -> skipped (host code consumes raw HF tensor)
            depthformer.*       -> audio_decoder.* (depthformer layers)
            depth_linear.*      -> audio_decoder.depth_linear.*
            depth_embeddings.K.{embedding,embedding_norm,to_logits}.weight
                                -> audio_decoder.depth_embeddings.K.*

        Special-case transforms:
            * Each depthformer layer's fused ``operator.qkv_proj.weight``
              is split into ``self_attn.{q,k,v}_proj.weight`` along the row
              axis using ``num_q*head_dim`` / ``num_kv*head_dim`` chunks.
            * Per-codebook ``embedding_norm.weight`` tensors are stacked
              into ``audio_decoder.stacked_norm_weights``.
            * Per-codebook ``to_logits.weight`` tensors (which are tied to
              the corresponding ``embedding.weight`` in HF when
              ``depthformer_tie=True``) are stacked into
              ``audio_decoder.stacked_head_weights``.
        """
        # LFM2-Audio doesn't carry an explicit ``tie_word_embeddings`` flag
        # in its custom config, but the checkpoint omits ``lfm.lm_head.weight``
        # so the embedding is always tied. Force the tie whenever lm_head is
        # missing and embed_tokens is present so the decoder lm_head gets a
        # weight regardless of how config.tie_word_embeddings was inferred.
        force_tie = (
            "lfm.lm_head.weight" not in state_dict and "lfm.embed_tokens.weight" in state_dict
        )
        if self.config.tie_word_embeddings or force_tie:
            tie_word_embeddings(
                state_dict,
                embed_key="lfm.embed_tokens.weight",
                head_key="lfm.lm_head.weight",
            )
            # In multi-model splits the embedding and lm_head end up in
            # *different* ONNX sub-graphs (``embedding`` and ``decoder``).
            # mobius's apply_weights deduplicates tensors that share Python
            # storage, which would drop the decoder copy entirely. Clone so
            # each sub-model gets its own initializer.
            state_dict["lfm.lm_head.weight"] = state_dict["lfm.lm_head.weight"].clone()

        # Split fused depthformer qkv_proj into q/k/v.
        head_dim = self.config.depthformer_head_dim
        num_q_heads = self.config.depthformer_dim // head_dim
        num_kv_heads = self.config.depthformer_kv_heads
        q_rows = num_q_heads * head_dim
        kv_rows = num_kv_heads * head_dim
        for i in range(self.config.depthformer_layers):
            qkv_key = f"depthformer.layers.{i}.operator.qkv_proj.weight"
            if qkv_key in state_dict:
                qkv = state_dict.pop(qkv_key)
                state_dict[f"depthformer.layers.{i}.operator.q_proj.weight"] = qkv[:q_rows]
                state_dict[f"depthformer.layers.{i}.operator.k_proj.weight"] = qkv[
                    q_rows : q_rows + kv_rows
                ]
                state_dict[f"depthformer.layers.{i}.operator.v_proj.weight"] = qkv[
                    q_rows + kv_rows : q_rows + 2 * kv_rows
                ]

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _rename_lfm2_audio_weight(key)
            if new_key is not None:
                new_state_dict[new_key] = value

        # Stack per-codebook norm + head weights for Gather-by-codebook in
        # the audio_decoder forward.
        norm_weights = []
        head_weights = []
        for i in range(self.config.num_codebooks):
            nkey = f"audio_decoder.depth_embeddings.{i}.embedding_norm.weight"
            hkey = f"audio_decoder.depth_embeddings.{i}.to_logits.weight"
            if nkey in new_state_dict:
                norm_weights.append(new_state_dict[nkey])
            if hkey in new_state_dict:
                head_weights.append(new_state_dict[hkey])
        if norm_weights:
            new_state_dict["audio_decoder.stacked_norm_weights"] = torch.stack(
                norm_weights, dim=0
            )
        if head_weights:
            new_state_dict["audio_decoder.stacked_head_weights"] = torch.stack(
                head_weights, dim=0
            )

        return new_state_dict


def _rename_lfm2_audio_weight(key: str) -> str | None:
    """Rename a single HF weight key to ONNX module structure.

    Returns None if the weight should be skipped.
    """
    import re

    # LFM backbone embed_tokens -> embedding.text_embed
    if key.startswith("lfm.embed_tokens."):
        return key.replace("lfm.embed_tokens.", "embedding.text_embed.")

    # LFM backbone layers -> decoder.layers
    if key.startswith("lfm."):
        rest = key[len("lfm.") :]
        # model.layers.N patterns
        m = re.match(r"^layers\.(\d+)\.(.+)$", rest)
        if m:
            idx = m.group(1)
            layer_rest = m.group(2)
            # Conv weight nesting
            layer_rest = layer_rest.replace("conv.conv.weight", "conv.conv_weight")
            layer_rest = layer_rest.replace("conv.conv.bias", "conv.conv_bias")
            # MLP: w1->gate_proj, w3->up_proj, w2->down_proj
            layer_rest = layer_rest.replace("feed_forward.w1.", "feed_forward.gate_proj.")
            layer_rest = layer_rest.replace("feed_forward.w3.", "feed_forward.up_proj.")
            layer_rest = layer_rest.replace("feed_forward.w2.", "feed_forward.down_proj.")
            # Attention: out_proj->o_proj, layernorm->norm
            layer_rest = layer_rest.replace("self_attn.out_proj.", "self_attn.o_proj.")
            layer_rest = layer_rest.replace("self_attn.q_layernorm.", "self_attn.q_norm.")
            layer_rest = layer_rest.replace("self_attn.k_layernorm.", "self_attn.k_norm.")
            return f"decoder.layers.{idx}.{layer_rest}"

        # HF stores the post-layers RMSNorm as ``lfm.embedding_norm.weight``,
        # but our :class:`_Lfm2AudioDecoder` exposes it as ``self.norm``.
        if rest == "embedding_norm.weight":
            return "decoder.norm.weight"

        # lfm.norm -> decoder.norm
        return f"decoder.{rest}"

    # Conformer -> audio_encoder.encoder. The mel preprocessor is implemented
    # host-side (numpy/librosa), so its frozen featurizer buffers
    # (``window``, ``fb``) have no ONNX home. Likewise the per-layer
    # BatchNorm's ``num_batches_tracked`` is a scalar bookkeeping buffer with
    # no parameter slot in our ``_NeMoBatchNorm1d``.
    if key.startswith("conformer."):
        if key.startswith("conformer.preprocessor."):
            return None
        if key.endswith(".num_batches_tracked"):
            return None
        rest = key[len("conformer.") :]
        # pre_encode.conv.{0,2,3,5,6}.{weight,bias} -> pre_encode.conv_{N}.{...}
        # (HF stores the conv stack as Sequential indices including ReLUs at
        # 1/4/7; mobius hoists each Conv2d into a named attribute so we can
        # interleave per-step masking — see _NeMoSubsampling.)
        m = re.match(r"^pre_encode\.conv\.(\d+)\.(.+)$", rest)
        if m:
            idx = m.group(1)
            tail = m.group(2)
            if idx in ("1", "4", "7"):  # ReLU slots have no params
                return None
            rest = f"pre_encode.conv_{idx}.{tail}"
        return f"audio_encoder.encoder.{rest}"

    # Audio adapter: HF Sequential -> our named modules
    #   model.0 = LayerNorm(encoder_dim)              -> pre_norm
    #   model.1 = Linear(encoder_dim, hidden_size)    -> up_proj (no bias)
    #   model.2 = GELU (no params)
    #   model.3 = Linear(hidden_size, hidden_size)    -> out_proj
    if key.startswith("audio_adapter."):
        rest = key[len("audio_adapter.") :]
        if rest.startswith("model.2."):
            return None  # GELU has no params
        rest = rest.replace("model.0.", "pre_norm.")
        rest = rest.replace("model.1.", "up_proj.")
        rest = rest.replace("model.3.", "out_proj.")
        return f"audio_encoder.adapter.{rest}"

    # Audio embedding weights live in audio_decoder.depth_embeddings at runtime.
    # The embedding sub-model only handles text tokens — skip these.
    if key.startswith("audio_embedding."):
        return None

    # Depthformer layers
    if key.startswith("depthformer.layers."):
        rest = key[len("depthformer.layers.") :]
        m = re.match(r"^(\d+)\.(.+)$", rest)
        if m:
            idx = m.group(1)
            layer_rest = m.group(2)
            # operator.{q,k,v,out}_proj -> self_attn.{q,k,v,o}_proj
            # (qkv_proj was already split into q/k/v_proj earlier).
            layer_rest = layer_rest.replace("operator.out_proj.", "self_attn.o_proj.")
            layer_rest = layer_rest.replace(
                "operator.bounded_attention.q_layernorm.",
                "self_attn.q_norm.",
            )
            layer_rest = layer_rest.replace(
                "operator.bounded_attention.k_layernorm.",
                "self_attn.k_norm.",
            )
            layer_rest = layer_rest.replace("operator.q_proj.", "self_attn.q_proj.")
            layer_rest = layer_rest.replace("operator.k_proj.", "self_attn.k_proj.")
            layer_rest = layer_rest.replace("operator.v_proj.", "self_attn.v_proj.")
            # MLP renames
            layer_rest = layer_rest.replace("feed_forward.w1.", "feed_forward.gate_proj.")
            layer_rest = layer_rest.replace("feed_forward.w3.", "feed_forward.up_proj.")
            layer_rest = layer_rest.replace("feed_forward.w2.", "feed_forward.down_proj.")
            return f"audio_decoder.layers.{idx}.{layer_rest}"
        return None

    # Depth linear
    if key.startswith("depth_linear."):
        return key.replace("depth_linear.", "audio_decoder.depth_linear.")

    # Per-codebook depth embedding triples: depth_embeddings.K.{embedding,
    # embedding_norm, to_logits}.weight -> audio_decoder.depth_embeddings.K.*.
    if key.startswith("depth_embeddings."):
        return key.replace("depth_embeddings.", "audio_decoder.depth_embeddings.")

    # Legacy top-level embedding_norm (not present in LFM2-Audio-1.5B but
    # tolerated for older checkpoints). The current model moves the
    # depthformer output norm into each per-codebook triple, so this key
    # has nowhere to land — drop it rather than throwing.
    if key.startswith("embedding_norm."):
        return None

    return key
