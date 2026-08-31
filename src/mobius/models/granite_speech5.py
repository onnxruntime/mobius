# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native Hugging Face Granite Speech 5 encoder and CTC head."""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import GraniteSpeech5CTCConfig
from mobius.components import Conv1d, Embedding, LayerNorm, Linear, get_activation


def _dim(op: OpBuilder, value: ir.Value, axis: int) -> ir.Value:
    return op.Shape(value, start=axis, end=axis + 1)


def _scalar_like(op: OpBuilder, value: float, reference: ir.Value) -> ir.Value:
    scalar = op.Constant(value=ir.tensor(np.float32(value)))
    return op.CastLike(scalar, reference)


class _GraniteSpeech5FeedForward(nn.Module):
    """Macaron feed-forward branch used twice in each conformer block."""

    def __init__(self, config: GraniteSpeech5CTCConfig):
        super().__init__()
        self.linear1 = Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.attention_bias,
        )
        self.linear2 = Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.act_fn = get_activation(config.hidden_act)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # (B, T, hidden) -> (B, T, intermediate) -> (B, T, hidden)
        return self.linear2(op, self.act_fn(op, self.linear1(op, hidden_states)))


class _GraniteSpeech5ChunkedAttention(nn.Module):
    """Block-local self-attention with Shaw relative position embeddings."""

    def __init__(self, config: GraniteSpeech5CTCConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self._context_size = config.context_size
        self._num_heads = config.num_attention_heads
        self._head_dim = config.head_dim
        self._inner_dim = self._num_heads * self._head_dim
        self._scale = float(self._head_dim**-0.5)
        self._mask_value = (
            -65504.0 if config.dtype == ir.DataType.FLOAT16 else np.finfo(np.float32).min
        )

        self.q_proj = Linear(hidden_size, self._inner_dim, bias=False)
        self.k_proj = Linear(hidden_size, self._inner_dim, bias=False)
        self.v_proj = Linear(hidden_size, self._inner_dim, bias=False)
        self.o_proj = Linear(self._inner_dim, hidden_size, bias=True)
        self.rel_pos_emb = Embedding(
            2 * config.max_position_embeddings + 1,
            self._head_dim,
        )

        positions = np.arange(self._context_size, dtype=np.int64)
        distances = positions[:, None] - positions[None, :]
        self._position_indices = (
            np.clip(distances, -self._context_size, self._context_size)
            + config.max_position_embeddings
        ).astype(np.int64)

    def _pad_to_context(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        sequence_length = _dim(op, hidden_states, 1)
        num_padded = op.Mod(
            op.Neg(sequence_length),
            op.Constant(value=ir.tensor(np.array([self._context_size], dtype=np.int64))),
        )
        hidden_padding = op.Expand(
            _scalar_like(op, 0.0, hidden_states),
            op.Concat(
                _dim(op, hidden_states, 0),
                num_padded,
                _dim(op, hidden_states, 2),
                axis=0,
            ),
        )
        mask_padding = op.Expand(
            op.Constant(value=ir.tensor(np.array(False))),
            op.Concat(_dim(op, attention_mask, 0), num_padded, axis=0),
        )
        return (
            op.Concat(hidden_states, hidden_padding, axis=1),
            op.Concat(attention_mask, mask_padding, axis=1),
            sequence_length,
        )

    def _relative_bias(self, op: OpBuilder, query: ir.Value) -> ir.Value:
        # Query: (chunks, context, heads*head_dim) -> (chunks, heads, context, head_dim).
        query = op.Reshape(
            query,
            op.Constant(
                value=ir.tensor(
                    np.array(
                        [-1, self._context_size, self._num_heads, self._head_dim],
                        dtype=np.int64,
                    )
                )
            ),
        )
        query = op.Transpose(query, perm=[0, 2, 1, 3])

        # Shaw embeddings are query-position-specific: (context, context, head_dim).
        relative = self.rel_pos_emb(
            op,
            op.Constant(value=ir.tensor(self._position_indices)),
        )
        relative = op.Mul(relative, _scalar_like(op, self._scale, relative))

        # Match the upstream batched MatMul exactly:
        # (context, chunks*heads, head_dim) @ (context, head_dim, context).
        query_by_position = op.Transpose(query, perm=[2, 0, 1, 3])
        query_by_position = op.Reshape(
            query_by_position,
            op.Constant(value_ints=[self._context_size, -1, self._head_dim]),
        )
        position_bias = op.MatMul(
            query_by_position,
            op.Transpose(relative, perm=[0, 2, 1]),
        )
        position_bias = op.Reshape(
            position_bias,
            op.Concat(
                op.Constant(value_ints=[self._context_size]),
                _dim(op, query, 0),
                op.Constant(value_ints=[self._num_heads, self._context_size]),
                axis=0,
            ),
        )
        return op.Transpose(position_bias, perm=[1, 2, 0, 3])

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
    ) -> ir.Value:
        # Each context block is folded into the batch axis; blocks never attend
        # across boundaries. The tail is right-padded and sliced away afterward.
        padded_states, padded_mask, sequence_length = self._pad_to_context(
            op, hidden_states, attention_mask
        )
        chunk_shape = op.Constant(value_ints=[-1, self._context_size, self._inner_dim])
        query = op.Reshape(self.q_proj(op, padded_states), chunk_shape)
        key = op.Reshape(self.k_proj(op, padded_states), chunk_shape)
        value = op.Reshape(self.v_proj(op, padded_states), chunk_shape)

        position_bias = self._relative_bias(op, query)
        chunk_mask = op.Reshape(
            padded_mask,
            op.Constant(value_ints=[-1, self._context_size]),
        )
        chunk_mask = op.Unsqueeze(chunk_mask, op.Constant(value_ints=[1, 2]))
        attention_bias = op.Where(
            chunk_mask,
            position_bias,
            _scalar_like(op, self._mask_value, position_bias),
        )
        output = op.Attention(
            query,
            key,
            value,
            attention_bias,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            scale=self._scale,
        )
        output = op.Reshape(
            output,
            op.Concat(
                _dim(op, padded_states, 0),
                _dim(op, padded_states, 1),
                op.Constant(value_ints=[self._inner_dim]),
                axis=0,
            ),
        )
        output = op.Slice(
            output,
            op.Constant(value_ints=[0]),
            sequence_length,
            op.Constant(value_ints=[1]),
        )
        return self.o_proj(op, output)


class _Float32BatchNorm1d(nn.Module):
    """Frozen BatchNorm kept in fp32, matching the upstream strict dtype policy."""

    def __init__(self, num_features: int):
        super().__init__()
        self.weight = nn.Parameter((num_features,), dtype=ir.DataType.FLOAT)
        self.bias = nn.Parameter((num_features,), dtype=ir.DataType.FLOAT)
        self.running_mean = nn.Parameter((num_features,), dtype=ir.DataType.FLOAT)
        self.running_var = nn.Parameter((num_features,), dtype=ir.DataType.FLOAT)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        reference = hidden_states
        hidden_states = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        channel_shape = op.Constant(value_ints=[1, -1, 1])
        mean = op.Cast(op.Reshape(self.running_mean, channel_shape), to=ir.DataType.FLOAT)
        variance = op.Cast(
            op.Reshape(self.running_var, channel_shape), to=ir.DataType.FLOAT
        )
        scale = op.Cast(op.Reshape(self.weight, channel_shape), to=ir.DataType.FLOAT)
        bias = op.Cast(op.Reshape(self.bias, channel_shape), to=ir.DataType.FLOAT)
        hidden_states = op.Div(
            op.Sub(hidden_states, mean),
            op.Sqrt(op.Add(variance, _scalar_like(op, 1e-5, variance))),
        )
        hidden_states = op.Add(op.Mul(hidden_states, scale), bias)
        return op.CastLike(hidden_states, reference)


class _GraniteSpeech5Convolution(nn.Module):
    """Linear-GLU conformer convolution with optional stride-2 subsampling."""

    def __init__(self, config: GraniteSpeech5CTCConfig, *, stride: int):
        super().__init__()
        inner_dim = config.hidden_size * config.conv_expansion_factor
        self.pointwise_lin1 = Linear(config.hidden_size, 2 * inner_dim, bias=True)
        self.depthwise_conv = Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=config.conv_kernel_size,
            stride=stride,
            padding=(config.conv_kernel_size - 1) // 2,
            bias=False,
            groups=inner_dim,
        )
        self.norm = _Float32BatchNorm1d(inner_dim)
        self.pointwise_lin2 = Linear(inner_dim, config.hidden_size, bias=True)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
    ) -> ir.Value:
        # (B, T, hidden) -> Linear+GLU -> (B, inner, T) depthwise Conv1d.
        hidden_states = self.pointwise_lin1(op, hidden_states)
        first, gate = op.Split(hidden_states, axis=-1, num_outputs=2, _outputs=2)
        hidden_states = op.Mul(first, op.Sigmoid(gate))
        hidden_states = op.Where(
            op.Unsqueeze(attention_mask, op.Constant(value_ints=[-1])),
            hidden_states,
            _scalar_like(op, 0.0, hidden_states),
        )
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = self.depthwise_conv(op, hidden_states)
        hidden_states = op.Swish(self.norm(op, hidden_states))
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        return self.pointwise_lin2(op, hidden_states)


class _GraniteSpeech5EncoderBlock(nn.Module):
    """Pre-norm Macaron conformer block with optional time subsampling."""

    def __init__(self, config: GraniteSpeech5CTCConfig, *, subsample: bool):
        super().__init__()
        hidden_size = config.hidden_size
        eps = config.layer_norm_eps
        self.feed_forward1 = _GraniteSpeech5FeedForward(config)
        self.self_attn = _GraniteSpeech5ChunkedAttention(config)
        self.conv = _GraniteSpeech5Convolution(config, stride=2 if subsample else 1)
        self.feed_forward2 = _GraniteSpeech5FeedForward(config)
        self.norm_feed_forward1 = LayerNorm(hidden_size, eps=eps)
        self.norm_self_att = LayerNorm(hidden_size, eps=eps)
        self.norm_conv = LayerNorm(hidden_size, eps=eps)
        self.norm_feed_forward2 = LayerNorm(hidden_size, eps=eps)
        self.norm_out = LayerNorm(hidden_size, eps=eps)
        self._subsample = subsample

    def _pool_residual(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        half_length = op.Div(
            _dim(op, hidden_states, 1),
            op.Constant(value=ir.tensor(np.array([2], dtype=np.int64))),
        )
        paired_length = op.Mul(
            half_length,
            op.Constant(value=ir.tensor(np.array([2], dtype=np.int64))),
        )
        hidden_states = op.Slice(
            hidden_states,
            op.Constant(value_ints=[0]),
            paired_length,
            op.Constant(value_ints=[1]),
        )
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                _dim(op, hidden_states, 0),
                half_length,
                op.Constant(value_ints=[2]),
                _dim(op, hidden_states, 2),
                axis=0,
            ),
        )
        first, second = op.Split(hidden_states, axis=2, num_outputs=2, _outputs=2)
        reference = first
        first = op.Cast(first, to=ir.DataType.FLOAT)
        second = op.Cast(second, to=ir.DataType.FLOAT)
        pooled = op.Add(first, second)
        pooled = op.Mul(pooled, _scalar_like(op, 0.5, pooled))
        pooled = op.CastLike(pooled, reference)
        return op.Squeeze(pooled, op.Constant(value_ints=[2]))

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
    ) -> ir.Value:
        half = _scalar_like(op, 0.5, hidden_states)
        residual = hidden_states
        hidden_states = self.feed_forward1(
            op, self.norm_feed_forward1(op, hidden_states)
        )
        hidden_states = op.Add(residual, op.Mul(hidden_states, half))

        residual = hidden_states
        hidden_states = self.self_attn(
            op,
            self.norm_self_att(op, hidden_states),
            attention_mask,
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        conv_output = self.conv(
            op,
            self.norm_conv(op, hidden_states),
            attention_mask,
        )
        if self._subsample:
            # The stride-2 convolution emits ceil(T/2); the mean-pooled residual
            # defines the true floor(T/2) length, so trim the convolution to it.
            residual = self._pool_residual(op, residual)
            conv_output = op.Slice(
                conv_output,
                op.Constant(value_ints=[0]),
                _dim(op, residual, 1),
                op.Constant(value_ints=[1]),
            )
        hidden_states = op.Add(residual, conv_output)

        residual = hidden_states
        hidden_states = self.feed_forward2(
            op, self.norm_feed_forward2(op, hidden_states)
        )
        hidden_states = op.Add(residual, op.Mul(hidden_states, half))
        return self.norm_out(op, hidden_states)


def _downsample_mask(op: OpBuilder, attention_mask: ir.Value) -> ir.Value:
    """Pairwise-AND a valid-frame mask, dropping an odd trailing frame."""
    half_length = op.Div(
        _dim(op, attention_mask, 1),
        op.Constant(value=ir.tensor(np.array([2], dtype=np.int64))),
    )
    paired_length = op.Mul(
        half_length,
        op.Constant(value=ir.tensor(np.array([2], dtype=np.int64))),
    )
    attention_mask = op.Slice(
        attention_mask,
        op.Constant(value_ints=[0]),
        paired_length,
        op.Constant(value_ints=[1]),
    )
    attention_mask = op.Reshape(
        attention_mask,
        op.Concat(
            _dim(op, attention_mask, 0),
            half_length,
            op.Constant(value_ints=[2]),
            axis=0,
        ),
    )
    attention_mask = op.ReduceMin(
        op.Cast(attention_mask, to=ir.DataType.INT64),
        axes=[2],
        keepdims=0,
    )
    return op.Cast(attention_mask, to=ir.DataType.BOOL)


class _GraniteSpeech5Encoder(nn.Module):
    """Chunked conformer encoder with block subsampling and self-conditioned CTC."""

    def __init__(self, config: GraniteSpeech5CTCConfig):
        super().__init__()
        self.input_linear = Linear(config.input_feature_size, config.hidden_size, bias=True)
        self.layers = nn.ModuleList(
            [
                _GraniteSpeech5EncoderBlock(
                    config,
                    subsample=index in config.subsample_layers,
                )
                for index in range(config.num_hidden_layers)
            ]
        )
        self.out = Linear(config.hidden_size, config.vocab_size, bias=True)
        self.out_mid = Linear(config.vocab_size, config.hidden_size, bias=True)
        self._subsample_layers = frozenset(config.subsample_layers)
        self._self_conditioning_layer = config.num_hidden_layers // 2

    def forward(
        self,
        op: OpBuilder,
        input_features: ir.Value,
        attention_mask: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        hidden_states = self.input_linear(op, input_features)  # (B, T, hidden)
        hidden_states = op.Where(
            op.Unsqueeze(attention_mask, op.Constant(value_ints=[-1])),
            hidden_states,
            _scalar_like(op, 0.0, hidden_states),
        )

        for index, layer in enumerate(self.layers):
            hidden_states = layer(op, hidden_states, attention_mask)
            if index in self._subsample_layers:
                attention_mask = _downsample_mask(op, attention_mask)

            if index + 1 == self._self_conditioning_layer:
                # Mid-encoder CTC posteriors are projected back into hidden space.
                mid_logits = self.out(op, hidden_states)  # (B, T/4, vocab)
                mid_injection = self.out_mid(op, op.Softmax(mid_logits, axis=-1))
                hidden_states = op.Add(hidden_states, mid_injection)

            # Some CUDA attention kernels return NaN for an all-masked block.
            # Invalid queries are semantically outside the CTC sequence, and
            # clearing them prevents their projected K/V from contaminating a
            # later layer before that layer applies its key mask.
            hidden_states = op.Where(
                op.Unsqueeze(attention_mask, op.Constant(value_ints=[-1])),
                hidden_states,
                _scalar_like(op, 0.0, hidden_states),
            )

        return hidden_states, attention_mask


class GraniteSpeech5ForCTCModel(nn.Module):
    """Replicate ``transformers.GraniteSpeech5ForCTC`` as one ONNX graph."""

    default_task = "feature-ctc-asr"
    category = "Speech-to-Text"
    config_class = GraniteSpeech5CTCConfig

    def __init__(self, config: GraniteSpeech5CTCConfig):
        super().__init__()
        self._dtype = config.dtype
        self._subsample_count = len(config.subsample_layers)
        self.encoder = _GraniteSpeech5Encoder(config)

    def forward(
        self,
        op: OpBuilder,
        input_features: ir.Value,
        attention_mask: ir.Value,
    ) -> ir.Value:
        # The native processor emits float32 features and an int64 valid mask.
        # Reduced-precision models cast features once at graph entry.
        if self._dtype != ir.DataType.FLOAT:
            input_features = op.Cast(input_features, to=self._dtype)
        valid_frames = op.Cast(attention_mask, to=ir.DataType.BOOL)
        hidden_states, _ = self.encoder(op, input_features, valid_frames)

        # The top-level ctc_head is tied to encoder.out in Transformers, so reuse
        # the same module and initializer for final logits.
        return self.encoder.out(op, hidden_states)  # (B, floor(T/4), vocab)

    def frame_lengths(self, op: OpBuilder, attention_mask: ir.Value) -> ir.Value:
        """Return valid CTC frame counts after pairwise block subsampling."""
        lengths = op.ReduceSum(
            op.Cast(attention_mask, to=ir.DataType.INT64),
            axes=[1],
            keepdims=0,
        )
        two = op.Constant(value=ir.tensor(np.int64(2)))
        for _ in range(self._subsample_count):
            lengths = op.Div(lengths, two)
        return lengths

    def preprocess_weights(self, state_dict: dict[str, object]) -> dict[str, object]:
        """Keep native ``encoder.*`` names and remove tied/runtime-only entries."""
        return {
            name: value
            for name, value in state_dict.items()
            if not name.startswith("ctc_head.") and not name.endswith(".num_batches_tracked")
        }
