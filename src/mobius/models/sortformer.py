# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Streaming Sortformer speaker-diarization model (NVIDIA NeMo).

Replicates the offline forward pass of NeMo's
``SortformerEncLabelModel`` (e.g. ``nvidia/diar_streaming_sortformer_4spk``).
The exported ONNX graph consumes a mel-spectrogram feature sequence and
produces per-frame speaker-activity probabilities.

Pipeline (matching ``frontend_encoder`` + ``forward_infer`` in
``nemo.collections.asr.models.sortformer_diar_models``):

    input_features [B, feat, T_mel]
      -> transpose                       [B, T_mel, feat]
      -> FastConformer encoder           [B, T, fc_d_model]   (T = T_mel / 8)
      -> encoder_proj (Linear)           [B, T, tf_d_model]
      -> Transformer encoder (post-LN)   [B, T, tf_d_model]
      -> speaker sigmoid head            [B, T, num_spks]

The FastConformer encoder is NeMo's ``ConformerEncoder`` with
``self_attention_model='rel_pos'`` (Transformer-XL relative positional
attention), ``dw_striding`` conv subsampling (8x), Macaron feed-forward
blocks, and a batch-norm convolution module.  The Transformer encoder is
NeMo's post-LayerNorm ``TransformerEncoder``.

Only the offline (non-streaming) network is exported; the streaming
speaker-cache / FIFO bookkeeping is runtime state management outside the
static graph.
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

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

# Slice "end" sentinel meaning "to the end of the axis".
_INT64_MAX = 9223372036854775807


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SortformerConfig(BaseModelConfig):
    """Configuration for the Sortformer diarization model.

    Fields mirror the NeMo ``.nemo`` ``model_config.yaml`` (``encoder``,
    ``transformer_encoder`` and ``sortformer_modules`` sections).
    """

    # Mel feature dimension (preprocessor ``features``).
    feat_in: int = 128
    # FastConformer encoder.
    fc_d_model: int = 512
    fc_num_layers: int = 17
    fc_num_heads: int = 8
    fc_ff_expansion: int = 4
    fc_conv_kernel: int = 9
    fc_subsampling_conv_channels: int = 256
    fc_subsampling_factor: int = 8
    # Transformer encoder (operates on projected embeddings).
    tf_d_model: int = 192
    tf_num_layers: int = 18
    tf_num_heads: int = 8
    tf_inner_size: int = 768
    tf_hidden_act: str = "relu"
    # Diarization head.
    num_spks: int = 4

    # HuggingFace/registry model_type — consumed by the generic
    # ``build_from_nemo`` pipeline to resolve the model class and task.
    model_type: str | None = "sortformer"

    def validate(self) -> None:
        if self.fc_d_model % self.fc_num_heads != 0:
            raise ValueError("fc_d_model must be divisible by fc_num_heads")
        if self.tf_d_model % self.tf_num_heads != 0:
            raise ValueError("tf_d_model must be divisible by tf_num_heads")

    @classmethod
    def from_nemo_yaml(cls, cfg: dict) -> SortformerConfig:
        """Build a config from a parsed NeMo ``model_config.yaml`` mapping."""
        enc = cfg.get("encoder", {})
        tf = cfg.get("transformer_encoder", {})
        pre = cfg.get("preprocessor", {})
        return cls(
            feat_in=int(enc.get("feat_in", pre.get("features", 128))),
            fc_d_model=int(enc.get("d_model", 512)),
            fc_num_layers=int(enc.get("n_layers", 17)),
            fc_num_heads=int(enc.get("n_heads", 8)),
            fc_ff_expansion=int(enc.get("ff_expansion_factor", 4)),
            fc_conv_kernel=int(enc.get("conv_kernel_size", 9)),
            fc_subsampling_conv_channels=int(enc.get("subsampling_conv_channels", 256)),
            fc_subsampling_factor=int(enc.get("subsampling_factor", 8)),
            tf_d_model=int(tf.get("hidden_size", 192)),
            tf_num_layers=int(tf.get("num_layers", 18)),
            tf_num_heads=int(tf.get("num_attention_heads", 8)),
            tf_inner_size=int(tf.get("inner_size", 768)),
            tf_hidden_act=str(tf.get("hidden_act", "relu")),
            num_spks=int(cfg.get("max_num_of_spks", 4)),
            model_type="sortformer",
        )


# ---------------------------------------------------------------------------
# Low-level ONNX helpers
# ---------------------------------------------------------------------------


class _Conv1d(nn.Module):
    """Conv1d with explicit (possibly grouped) weight and bias.

    Stores ``weight``/``bias`` so parameter names match NeMo's
    ``nn.Conv1d`` layers (e.g. ``conv.pointwise_conv1.weight``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        padding: int = 0,
        groups: int = 1,
    ):
        super().__init__()
        self.weight = nn.Parameter([out_channels, in_channels // groups, kernel_size])
        self.bias = nn.Parameter([out_channels])
        self._kernel_size = kernel_size
        self._padding = padding
        self._groups = groups

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self._kernel_size],
            strides=[1],
            pads=[self._padding, self._padding],
            group=self._groups,
        )


class _Conv2d(nn.Module):
    """Conv2d with weight + bias (used by the dw_striding subsampling)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            [out_channels, in_channels // groups, kernel_size, kernel_size]
        )
        self.bias = nn.Parameter([out_channels])
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._groups = groups

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        p = self._padding
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self._kernel_size, self._kernel_size],
            strides=[self._stride, self._stride],
            pads=[p, p, p, p],
            group=self._groups,
        )


class _BatchNorm1d(nn.Module):
    """1D batch normalization with frozen running statistics (inference)."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter([num_features])
        self.bias = nn.Parameter([num_features])
        self.running_mean = nn.Parameter([num_features])
        self.running_var = nn.Parameter([num_features])
        self._eps = eps

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: [B, C, T]
        return op.BatchNormalization(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            epsilon=self._eps,
        )


class _ReLU(nn.Module):
    """ReLU activation as a module (used inside the subsampling ModuleList)."""

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return op.Relu(x)


# ---------------------------------------------------------------------------
# FastConformer subsampling (NeMo ``dw_striding``)
# ---------------------------------------------------------------------------


class _ConvSubsampling(nn.Module):
    """NeMo ``dw_striding`` conv subsampling (8x time reduction).

    Three stride-2 stages: a regular Conv2d followed by two
    depthwise-separable (depthwise + pointwise) stages, each with a ReLU.
    A final Linear projects the flattened channel/frequency axis to
    ``feat_out``.  The ``conv`` ModuleList indices match NeMo's
    ``pre_encode.conv.{0,2,3,5,6}`` weight names.

    Input:  ``[B, T_mel, feat_in]``
    Output: ``[B, ceil(T_mel / 8), feat_out]``
    """

    def __init__(self, feat_in: int, conv_channels: int, feat_out: int):
        super().__init__()
        # Frequency dimension after three stride-2, pad-1, kernel-3 convs.
        freq = feat_in
        for _ in range(3):
            freq = (freq + 2 - 3) // 2 + 1

        c = conv_channels
        self.conv = nn.ModuleList(
            [
                _Conv2d(1, c, 3, stride=2, padding=1),  # 0: regular
                _ReLU(),  # 1
                _Conv2d(c, c, 3, stride=2, padding=1, groups=c),  # 2: depthwise
                _Conv2d(c, c, 1),  # 3: pointwise
                _ReLU(),  # 4
                _Conv2d(c, c, 3, stride=2, padding=1, groups=c),  # 5: depthwise
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


# ---------------------------------------------------------------------------
# FastConformer conformer block (rel_pos attention)
# ---------------------------------------------------------------------------


class _ConformerFeedForward(nn.Module):
    """Conformer feed-forward: Linear -> Swish -> Linear."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # Swish / SiLU (opset-24 op): x * sigmoid(x).
        return self.linear2(op, op.Swish(self.linear1(op, x)))


class _ConformerConvolution(nn.Module):
    """Conformer convolution module (GLU + depthwise + batch-norm + Swish).

    Structure (channels-first internally):
        pointwise_conv1 (C -> 2C) -> GLU -> depthwise_conv (k, groups=C)
        -> batch_norm -> Swish -> pointwise_conv2 (C -> C)
    """

    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        pad = (kernel_size - 1) // 2
        self.pointwise_conv1 = _Conv1d(d_model, d_model * 2, 1)
        self.depthwise_conv = _Conv1d(
            d_model, d_model, kernel_size, padding=pad, groups=d_model
        )
        self.batch_norm = _BatchNorm1d(d_model)
        self.pointwise_conv2 = _Conv1d(d_model, d_model, 1)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # x: [B, T, C] -> [B, C, T]
        x = op.Transpose(x, perm=[0, 2, 1])
        x = self.pointwise_conv1(op, x)  # [B, 2C, T]
        # GLU over the channel axis: first * sigmoid(second)
        first, second = op.Split(x, axis=1, num_outputs=2, _outputs=2)
        x = op.Mul(first, op.Sigmoid(second))
        x = self.depthwise_conv(op, x)
        x = self.batch_norm(op, x)
        x = op.Swish(x)  # Swish / SiLU (opset-24 op)
        x = self.pointwise_conv2(op, x)
        return op.Transpose(x, perm=[0, 2, 1])  # [B, T, C]


class _RelPositionMultiHeadAttention(nn.Module):
    """Transformer-XL relative-position multi-head attention (NeMo rel_pos).

    Computes ``scores = (matrix_ac + rel_shift(matrix_bd)) / sqrt(d_k)`` where
    ``matrix_ac = (q + u) @ k^T`` and ``matrix_bd = (q + v) @ p^T``, with
    ``u``/``v`` the learned positional biases and ``p`` the projected
    relative positional embeddings.  No attention mask is applied (the
    exported graph assumes an unpadded single-utterance input, matching
    ``att_context_size=[-1, -1]``).
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model)
        self.linear_v = Linear(d_model, d_model)
        self.linear_out = Linear(d_model, d_model)
        self.linear_pos = Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter([num_heads, self._head_dim])
        self.pos_bias_v = nn.Parameter([num_heads, self._head_dim])

    def _to_heads(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # [B, T, D] -> [B, T, H, d_k] -> [B, H, T, d_k]
        x = op.Reshape(x, [0, 0, self._num_heads, self._head_dim])
        return op.Transpose(x, perm=[0, 2, 1, 3])

    def _rel_shift(self, op: OpBuilder, x: ir.Value, t_dim: ir.Value) -> ir.Value:
        """Relative shift of ``[B, H, T, 2T-1]`` scores.

        Reproduces NeMo's ``rel_shift``: left-pad the last axis with a zero
        column, reshape/drop to realign, then keep the first ``T`` columns so
        entry ``(i, j)`` holds the bias for relative distance ``i - j``.
        """
        # bh = [B, H] (dynamic batch, static heads); t_dim = [T].
        bh = op.Shape(x, start=0, end=2)
        # 1) left-pad last dim by 1 -> [B, H, T, 2T]
        x = op.Pad(x, op.Constant(value_ints=[0, 0, 0, 1, 0, 0, 0, 0]))
        # 2) reshape to [B, H, 2T, T]
        shape_a = op.Concat(bh, op.Constant(value_ints=[-1]), t_dim, axis=0)
        x = op.Reshape(x, shape_a)
        # 3) drop the first row along axis 2 -> [B, H, 2T-1, T]
        x = op.Slice(
            x,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[_INT64_MAX]),
            op.Constant(value_ints=[2]),
        )
        # 4) reshape back to [B, H, T, 2T-1]
        shape_b = op.Concat(bh, t_dim, op.Constant(value_ints=[-1]), axis=0)
        x = op.Reshape(x, shape_b)
        # 5) keep the first T columns -> [B, H, T, T]
        return op.Slice(x, op.Constant(value_ints=[0]), t_dim, op.Constant(value_ints=[3]))

    def forward(
        self, op: OpBuilder, x: ir.Value, pos_emb: ir.Value, t_dim: ir.Value
    ) -> ir.Value:
        # x: [B, T, D], pos_emb: [1, 2T-1, D], t_dim: [T]
        q = self._to_heads(op, self.linear_q(op, x))  # [B, H, T, d_k]
        k = self._to_heads(op, self.linear_k(op, x))
        v = self._to_heads(op, self.linear_v(op, x))
        # p: [1, H, 2T-1, d_k]
        p = self._to_heads(op, self.linear_pos(op, pos_emb))

        # q + bias, broadcast [H, d_k] over [B, H, T, d_k].
        bias_u = op.Unsqueeze(self.pos_bias_u, [0, 2])  # [1, H, 1, d_k]
        bias_v = op.Unsqueeze(self.pos_bias_v, [0, 2])
        q_u = op.Add(q, bias_u)
        q_v = op.Add(q, bias_v)

        matrix_ac = op.MatMul(q_u, op.Transpose(k, perm=[0, 1, 3, 2]))  # [B,H,T,T]
        matrix_bd = op.MatMul(q_v, op.Transpose(p, perm=[0, 1, 3, 2]))  # [B,H,T,2T-1]
        matrix_bd = self._rel_shift(op, matrix_bd, t_dim)  # [B,H,T,T]

        scale = op.CastLike(
            op.Constant(value=ir.tensor(np.array(self._head_dim**-0.5, dtype=np.float32))),
            matrix_ac,
        )
        scores = op.Mul(op.Add(matrix_ac, matrix_bd), scale)
        attn = op.Softmax(scores, axis=-1)
        out = op.MatMul(attn, v)  # [B, H, T, d_k]
        out = op.Transpose(out, perm=[0, 2, 1, 3])  # [B, T, H, d_k]
        out = op.Reshape(out, [0, 0, -1])  # [B, T, D]
        return self.linear_out(op, out)


class _ConformerLayer(nn.Module):
    """Single FastConformer block (Macaron feed-forward structure)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, kernel_size: int):
        super().__init__()
        self.norm_feed_forward1 = LayerNorm(d_model, eps=1e-5)
        self.feed_forward1 = _ConformerFeedForward(d_model, d_ff)
        self.norm_self_att = LayerNorm(d_model, eps=1e-5)
        self.self_attn = _RelPositionMultiHeadAttention(d_model, num_heads)
        self.norm_conv = LayerNorm(d_model, eps=1e-5)
        self.conv = _ConformerConvolution(d_model, kernel_size)
        self.norm_feed_forward2 = LayerNorm(d_model, eps=1e-5)
        self.feed_forward2 = _ConformerFeedForward(d_model, d_ff)
        self.norm_out = LayerNorm(d_model, eps=1e-5)

    def forward(
        self, op: OpBuilder, x: ir.Value, pos_emb: ir.Value, t_dim: ir.Value
    ) -> ir.Value:
        # Macaron feed-forward (half-step residual)
        ff = self.feed_forward1(op, self.norm_feed_forward1(op, x))
        half = op.CastLike(op.Constant(value_float=0.5), x)
        x = op.Add(x, op.Mul(ff, half))
        # Relative-position self-attention
        att = self.self_attn(op, self.norm_self_att(op, x), pos_emb, t_dim)
        x = op.Add(x, att)
        # Convolution module
        x = op.Add(x, self.conv(op, self.norm_conv(op, x)))
        # Macaron feed-forward (half-step residual)
        ff = self.feed_forward2(op, self.norm_feed_forward2(op, x))
        x = op.Add(x, op.Mul(ff, op.Constant(value_float=0.5)))
        return self.norm_out(op, x)


class _FastConformerEncoder(nn.Module):
    """NeMo FastConformer encoder (rel_pos, dw_striding subsampling)."""

    def __init__(self, config: SortformerConfig):
        super().__init__()
        d_model = config.fc_d_model
        self._d_model = d_model
        self._xscale = math.sqrt(d_model)
        self.pre_encode = _ConvSubsampling(
            config.feat_in, config.fc_subsampling_conv_channels, d_model
        )
        d_ff = d_model * config.fc_ff_expansion
        self.layers = nn.ModuleList(
            [
                _ConformerLayer(d_model, config.fc_num_heads, d_ff, config.fc_conv_kernel)
                for _ in range(config.fc_num_layers)
            ]
        )
        # Sinusoidal frequency divisors: exp(arange(0, d, 2) * -log(10000)/d).
        # INF_VAL in NeMo's positional encoding is 10000.0.
        self._div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model)
        )

    def _build_pos_emb(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Build Transformer-XL relative positional embeddings.

        Returns ``pos_emb`` of shape ``[1, 2T-1, d_model]`` for positions
        ``T-1, T-2, ..., -(T-1)`` (sin in even columns, cos in odd columns),
        matching NeMo's ``RelPositionalEncoding``.
        """
        t_scalar = op.Squeeze(op.Shape(x, start=1, end=2))  # scalar T
        one = op.Constant(value_int=1)
        start = op.Sub(t_scalar, one)  # T - 1
        limit = op.Neg(t_scalar)  # -T (exclusive) -> last value -(T-1)
        positions = op.Range(start, limit, op.Constant(value_int=-1))  # [2T-1]
        pos_f = op.Unsqueeze(op.Cast(positions, to=ir.DataType.FLOAT), [-1])  # [2T-1,1]
        div_term = op.Constant(value=ir.tensor(self._div_term))  # [d/2]
        angles = op.Mul(pos_f, div_term)  # [2T-1, d/2]
        sin = op.Unsqueeze(op.Sin(angles), [-1])
        cos = op.Unsqueeze(op.Cos(angles), [-1])
        interleaved = op.Concat(sin, cos, axis=-1)  # [2T-1, d/2, 2]
        pe = op.Reshape(interleaved, [0, -1])  # [2T-1, d]
        return op.Unsqueeze(pe, [0])  # [1, 2T-1, d]

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        # input_features: [B, feat, T_mel] -> [B, T_mel, feat]
        x = op.Transpose(input_features, perm=[0, 2, 1])
        x = self.pre_encode(op, x)  # [B, T, d_model]
        # Scale inputs to the attention layers by sqrt(d_model) (xscaling).
        x = op.Mul(x, op.Constant(value=ir.tensor(np.array(self._xscale, np.float32))))
        pos_emb = self._build_pos_emb(op, x)
        t_dim = op.Shape(x, start=1, end=2)  # [T]
        for layer in self.layers:
            x = layer(op, x, pos_emb, t_dim)
        return x  # [B, T, d_model]


# ---------------------------------------------------------------------------
# Transformer encoder (NeMo post-LayerNorm)
# ---------------------------------------------------------------------------


class _TransformerAttention(nn.Module):
    """Standard bidirectional multi-head attention (NeMo transformer)."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self.query_net = Linear(d_model, d_model)
        self.key_net = Linear(d_model, d_model)
        self.value_net = Linear(d_model, d_model)
        self.out_projection = Linear(d_model, d_model)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # Standard scaled-dot-product attention via the opset-24 Attention op,
        # which splits heads internally from the [B, T, D] projections.  No mask
        # (bidirectional, unpadded single-utterance input).
        q = self.query_net(op, x)
        k = self.key_net(op, x)
        v = self.value_net(op, x)
        out = op.Attention(
            q,
            k,
            v,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            scale=float(self._head_dim**-0.5),
        )  # [B, T, D]
        return self.out_projection(op, out)


class _PositionWiseFF(nn.Module):
    """Position-wise feed-forward: Linear -> ReLU -> Linear."""

    def __init__(self, d_model: int, inner_size: int):
        super().__init__()
        self.dense_in = Linear(d_model, inner_size)
        self.dense_out = Linear(inner_size, d_model)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return self.dense_out(op, op.Relu(self.dense_in(op, x)))


class _TransformerBlock(nn.Module):
    """Post-LayerNorm Transformer encoder block."""

    def __init__(self, d_model: int, num_heads: int, inner_size: int):
        super().__init__()
        self.layer_norm_1 = LayerNorm(d_model, eps=1e-5)
        self.first_sub_layer = _TransformerAttention(d_model, num_heads)
        self.layer_norm_2 = LayerNorm(d_model, eps=1e-5)
        self.second_sub_layer = _PositionWiseFF(d_model, inner_size)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # Post-LN: Self-Attn -> Residual -> LN -> FFN -> Residual -> LN
        h = op.Add(self.first_sub_layer(op, x), x)
        h = self.layer_norm_1(op, h)
        out = op.Add(self.second_sub_layer(op, h), h)
        return self.layer_norm_2(op, out)


class _TransformerEncoder(nn.Module):
    """Stack of post-LayerNorm Transformer encoder blocks."""

    def __init__(self, config: SortformerConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TransformerBlock(config.tf_d_model, config.tf_num_heads, config.tf_inner_size)
                for _ in range(config.tf_num_layers)
            ]
        )

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        for layer in self.layers:
            x = layer(op, x)
        return x


# ---------------------------------------------------------------------------
# Sortformer head + top-level model
# ---------------------------------------------------------------------------


class _SortformerModules(nn.Module):
    """Projection + speaker sigmoid head.

    ``encoder_proj`` maps the FastConformer output to the Transformer width;
    the head applies ``relu -> Linear -> relu -> Linear -> sigmoid`` to emit
    per-speaker activity probabilities.
    """

    def __init__(self, config: SortformerConfig):
        super().__init__()
        self.encoder_proj = Linear(config.fc_d_model, config.tf_d_model)
        self.first_hidden_to_hidden = Linear(config.tf_d_model, config.tf_d_model)
        self.single_hidden_to_spks = Linear(config.tf_d_model, config.num_spks)

    def forward(self, op: OpBuilder, x: ir.Value, stage: str = "project") -> ir.Value:
        # ``stage`` selects which sub-computation to run so that both the
        # projection and the head share this module's name scope (invoking the
        # module via ``__call__`` is what registers the ``sortformer_modules.*``
        # parameter prefix).
        if stage == "project":
            return self.encoder_proj(op, x)
        # stage == "head": relu -> Linear -> relu -> Linear -> sigmoid
        x = op.Relu(x)
        x = self.first_hidden_to_hidden(op, x)
        x = op.Relu(x)
        x = self.single_hidden_to_spks(op, x)
        return op.Sigmoid(x)


class SortformerDiarizationModel(nn.Module):
    """Streaming Sortformer speaker-diarization model (offline forward).

    Replicates NVIDIA NeMo's ``SortformerEncLabelModel`` offline inference:
    a FastConformer encoder followed by a Transformer encoder and a speaker
    sigmoid head.  Consumes mel-spectrogram features and returns per-frame
    speaker-activity probabilities in ``[0, 1]``.
    """

    default_task: str = "diarization"
    category: str = "Speech-to-Text"

    def __init__(self, config: SortformerConfig):
        super().__init__()
        self.config = config
        self.encoder = _FastConformerEncoder(config)
        self.transformer_encoder = _TransformerEncoder(config)
        self.sortformer_modules = _SortformerModules(config)

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        # input_features: [B, feat_in, T_mel]
        emb = self.encoder(op, input_features)  # [B, T, fc_d_model]
        emb = self.sortformer_modules(op, emb, stage="project")  # [B, T, tf_d_model]
        emb = self.transformer_encoder(op, emb)  # [B, T, tf_d_model]
        return self.sortformer_modules(op, emb, stage="head")  # [B, T, num_spks]

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map NeMo checkpoint names onto module parameter names.

        NeMo prefixes align 1:1 with this module tree (``encoder.*``,
        ``transformer_encoder.*``, ``sortformer_modules.*``).  Only the mel
        preprocessor, batch-norm ``num_batches_tracked`` counters and the
        unused streaming ``hidden_to_spks`` head are dropped.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("preprocessor."):
                continue
            if key.endswith("num_batches_tracked"):
                continue
            # Unused in the offline forward path (streaming FIFO head).
            if key.startswith("sortformer_modules.hidden_to_spks."):
                continue
            renamed[key] = value
        return renamed
