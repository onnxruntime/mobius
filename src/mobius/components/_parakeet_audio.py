# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline FastConformer components used by Hugging Face Parakeet encoders."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._activations import get_activation
from mobius.components._common import LayerNorm, Linear
from mobius.components._conv import BatchNorm1d, Conv2d
from mobius.components._whisper import Conv1d

if TYPE_CHECKING:
    from mobius._configs import ParakeetCTCConfig


def _dim(op: OpBuilder, value: ir.Value, axis: int) -> ir.Value:
    return op.Shape(value, start=axis, end=axis + 1)


def _scalar_like(op: OpBuilder, value: float, reference: ir.Value) -> ir.Value:
    scalar = op.Constant(value=ir.tensor(np.float32(value)))
    return op.CastLike(scalar, reference)


class _ReLU(nn.Module):
    def forward(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        return op.Relu(value)


class _ParakeetSubsampling(nn.Module):
    """Symmetric depthwise Conv2d subsampling from mel frames to encoder states."""

    def __init__(self, config: ParakeetCTCConfig):
        super().__init__()
        kernel = config.subsampling_conv_kernel_size
        stride = config.subsampling_conv_stride
        padding = (kernel - 1) // 2
        channels = config.subsampling_conv_channels
        num_layers = int(math.log2(config.subsampling_factor))

        layers: list[nn.Module] = [
            Conv2d(
                1,
                channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            ),
            _ReLU(),
        ]
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    Conv2d(
                        channels,
                        channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                        groups=channels,
                    ),
                    Conv2d(channels, channels, kernel_size=1),
                    _ReLU(),
                ]
            )
        self.layers = nn.ModuleList(layers)
        self._conv_indices = tuple(
            index for index, layer in enumerate(layers) if isinstance(layer, Conv2d)
        )
        self._strided_conv_indices = tuple(
            index
            for index, layer in enumerate(layers)
            if isinstance(layer, Conv2d) and layer._stride != 1
        )

        output_frequency = config.num_mel_bins // config.subsampling_factor
        self.linear = Linear(channels * output_frequency, config.hidden_size, bias=True)
        self._kernel = kernel
        self._stride = stride
        self._padding = padding

    def _downsample_lengths(self, op: OpBuilder, lengths: ir.Value) -> ir.Value:
        numerator = op.Add(
            lengths,
            op.Constant(value=ir.tensor(np.int64(2 * self._padding - self._kernel))),
        )
        return op.Add(
            op.Div(numerator, op.Constant(value=ir.tensor(np.int64(self._stride)))),
            op.Constant(value=ir.tensor(np.int64(1))),
        )

    def _mask_conv_output(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        lengths: ir.Value,
    ) -> ir.Value:
        frame_ids = op.Range(
            op.Constant(value=ir.tensor(np.int64(0))),
            op.Squeeze(_dim(op, hidden_states, 2)),
            op.Constant(value=ir.tensor(np.int64(1))),
        )
        valid = op.Less(
            op.Unsqueeze(frame_ids, op.Constant(value_ints=[0])),
            op.Unsqueeze(lengths, op.Constant(value_ints=[1])),
        )
        valid = op.Unsqueeze(valid, op.Constant(value_ints=[1, 3]))
        return op.Where(valid, hidden_states, _scalar_like(op, 0.0, hidden_states))

    def forward(
        self,
        op: OpBuilder,
        input_features: ir.Value,
        attention_mask: ir.Value,
    ) -> ir.Value:
        # (B, T, mel) -> (B, 1, T, mel)
        hidden_states = op.Unsqueeze(input_features, op.Constant(value_ints=[1]))
        lengths = op.ReduceSum(
            op.Cast(attention_mask, to=ir.DataType.INT64),
            axes=[1],
            keepdims=0,
        )

        for index, layer in enumerate(self.layers):
            hidden_states = layer(op, hidden_states)
            if index in self._strided_conv_indices:
                lengths = self._downsample_lengths(op, lengths)
            if index in self._conv_indices:
                hidden_states = self._mask_conv_output(op, hidden_states, lengths)

        # (B, C, T', F') -> (B, T', C*F') -> (B, T', hidden)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1, 3])
        hidden_states = op.Reshape(hidden_states, [0, 0, -1])
        return self.linear(op, hidden_states)


class _ParakeetRelativePositionEncoding(nn.Module):
    """Interleaved sinusoidal relative positions over ``[-T+1, T-1]``."""

    def __init__(self, hidden_size: int, dtype: ir.DataType):
        super().__init__()
        self._hidden_size = hidden_size
        self._dtype = dtype
        self._inv_freq = 1.0 / (
            10_000.0 ** (np.arange(0, hidden_size, 2, dtype=np.float32) / hidden_size)
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        seq_length = _dim(op, hidden_states, 1)
        seq_float = op.Cast(op.Squeeze(seq_length), to=ir.DataType.FLOAT)
        one = op.Constant(value=ir.tensor(np.float32(1.0)))
        positions = op.Range(op.Sub(seq_float, one), op.Neg(seq_float), op.Neg(one))
        frequencies = op.Mul(
            op.Unsqueeze(positions, op.Constant(value_ints=[1])),
            op.Unsqueeze(
                op.Constant(value=ir.tensor(self._inv_freq)),
                op.Constant(value_ints=[0]),
            ),
        )
        sin = op.Unsqueeze(op.Sin(frequencies), op.Constant(value_ints=[-1]))
        cos = op.Unsqueeze(op.Cos(frequencies), op.Constant(value_ints=[-1]))
        positions = op.Reshape(
            op.Concat(sin, cos, axis=-1),
            op.Concat(
                _dim(op, frequencies, 0),
                op.Constant(value_ints=[self._hidden_size]),
                axis=0,
            ),
        )
        positions = op.Unsqueeze(positions, op.Constant(value_ints=[0]))
        if self._dtype != ir.DataType.FLOAT:
            positions = op.Cast(positions, to=self._dtype)
        return positions


class _ParakeetFeedForward(nn.Module):
    """Macaron feed-forward projection with the configured activation."""

    def __init__(self, config: ParakeetCTCConfig):
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
        return self.linear2(op, self.act_fn(op, self.linear1(op, hidden_states)))


class _ParakeetAttention(nn.Module):
    """Transformer-XL relative-position self-attention used by Parakeet."""

    def __init__(self, config: ParakeetCTCConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self._num_heads = config.num_attention_heads
        self._head_dim = hidden_size // self._num_heads
        self.q_proj = Linear(hidden_size, hidden_size, bias=config.attention_bias)
        self.k_proj = Linear(hidden_size, hidden_size, bias=config.attention_bias)
        self.v_proj = Linear(hidden_size, hidden_size, bias=config.attention_bias)
        self.o_proj = Linear(hidden_size, hidden_size, bias=config.attention_bias)
        self.relative_k_proj = Linear(hidden_size, hidden_size, bias=False)
        self.bias_u = nn.Parameter([self._num_heads, self._head_dim])
        self.bias_v = nn.Parameter([self._num_heads, self._head_dim])

    def _split_heads(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        shape = op.Concat(
            _dim(op, value, 0),
            _dim(op, value, 1),
            op.Constant(value_ints=[self._num_heads, self._head_dim]),
            axis=0,
        )
        return op.Reshape(value, shape)

    def _relative_shift(self, op: OpBuilder, scores: ir.Value) -> ir.Value:
        # (B, H, T, 2T-1) -> pad/reshape/shift -> (B, H, T, 2T-1)
        zero_column = op.Expand(
            _scalar_like(op, 0.0, scores),
            op.Concat(
                _dim(op, scores, 0),
                _dim(op, scores, 1),
                _dim(op, scores, 2),
                op.Constant(value_ints=[1]),
                axis=0,
            ),
        )
        scores = op.Concat(zero_column, scores, axis=-1)
        scores = op.Reshape(
            scores,
            op.Concat(
                _dim(op, scores, 0),
                _dim(op, scores, 1),
                op.Constant(value_ints=[-1]),
                _dim(op, scores, 2),
                axis=0,
            ),
        )
        scores = op.Slice(
            scores,
            op.Constant(value_ints=[1]),
            _dim(op, scores, 2),
            op.Constant(value_ints=[2]),
        )
        return op.Reshape(
            scores,
            op.Concat(
                _dim(op, scores, 0),
                _dim(op, scores, 1),
                _dim(op, scores, 3),
                op.Constant(value_ints=[-1]),
                axis=0,
            ),
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: ir.Value,
        attention_mask: ir.Value,
    ) -> ir.Value:
        scale = float(self._head_dim**-0.5)
        query = self.q_proj(op, hidden_states)  # (B, T, H*D)
        key = self.k_proj(op, hidden_states)
        value = self.v_proj(op, hidden_states)

        query_u = op.Add(
            query,
            op.Reshape(
                self.bias_u,
                op.Constant(value_ints=[self._num_heads * self._head_dim]),
            ),
        )
        query_v = self._split_heads(op, query)
        query_v = op.Add(
            query_v,
            op.Reshape(
                self.bias_v,
                op.Constant(value_ints=[1, 1, self._num_heads, self._head_dim]),
            ),
        )
        query_v = op.Transpose(query_v, perm=[0, 2, 1, 3])

        relative_key = self._split_heads(op, self.relative_k_proj(op, position_embeddings))
        relative_key = op.Transpose(relative_key, perm=[0, 2, 1, 3])
        relative_scores = op.MatMul(
            query_v,
            op.Transpose(relative_key, perm=[0, 1, 3, 2]),
        )
        relative_scores = self._relative_shift(op, relative_scores)
        relative_scores = op.Slice(
            relative_scores,
            op.Constant(value_ints=[0]),
            _dim(op, key, 1),
            op.Constant(value_ints=[3]),
        )
        relative_scores = op.Mul(relative_scores, _scalar_like(op, scale, relative_scores))

        additive_mask = op.Where(
            attention_mask,
            _scalar_like(op, 0.0, relative_scores),
            _scalar_like(op, float("-inf"), relative_scores),
        )
        attention_bias = op.Add(relative_scores, additive_mask)
        output = op.Attention(
            query_u,
            key,
            value,
            attention_bias,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            scale=scale,
        )
        return self.o_proj(op, output)


class _ParakeetConvolution(nn.Module):
    """Bi-directional depthwise Conformer convolution with BatchNorm."""

    def __init__(self, config: ParakeetCTCConfig):
        super().__init__()
        hidden_size = config.hidden_size
        kernel_size = config.conv_kernel_size
        self.pointwise_conv1 = Conv1d(
            hidden_size,
            2 * hidden_size,
            kernel_size=1,
            bias=config.convolution_bias,
        )
        self.depthwise_conv = Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=hidden_size,
            bias=config.convolution_bias,
        )
        self.norm = BatchNorm1d(hidden_size)
        self.act_fn = get_activation(config.hidden_act)
        self.pointwise_conv2 = Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=1,
            bias=config.convolution_bias,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        valid_frames: ir.Value,
    ) -> ir.Value:
        # (B, T, C) -> pointwise GLU -> depthwise convolution in (B, C, T).
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = self.pointwise_conv1(op, hidden_states)
        first, gate = op.Split(hidden_states, axis=1, num_outputs=2, _outputs=2)
        hidden_states = op.Mul(first, op.Sigmoid(gate))
        hidden_states = op.Where(
            op.Unsqueeze(valid_frames, op.Constant(value_ints=[1])),
            hidden_states,
            _scalar_like(op, 0.0, hidden_states),
        )
        hidden_states = self.depthwise_conv(op, hidden_states)
        hidden_states = self.norm(op, hidden_states)
        hidden_states = self.act_fn(op, hidden_states)
        hidden_states = self.pointwise_conv2(op, hidden_states)
        return op.Transpose(hidden_states, perm=[0, 2, 1])


class _ParakeetEncoderLayer(nn.Module):
    """Macaron FastConformer block with relative attention and convolution."""

    def __init__(self, config: ParakeetCTCConfig):
        super().__init__()
        hidden_size = config.hidden_size
        eps = config.layer_norm_eps
        self.feed_forward1 = _ParakeetFeedForward(config)
        self.self_attn = _ParakeetAttention(config)
        self.conv = _ParakeetConvolution(config)
        self.feed_forward2 = _ParakeetFeedForward(config)
        self.norm_feed_forward1 = LayerNorm(hidden_size, eps=eps)
        self.norm_self_att = LayerNorm(hidden_size, eps=eps)
        self.norm_conv = LayerNorm(hidden_size, eps=eps)
        self.norm_feed_forward2 = LayerNorm(hidden_size, eps=eps)
        self.norm_out = LayerNorm(hidden_size, eps=eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: ir.Value,
        attention_mask: ir.Value,
        valid_frames: ir.Value,
    ) -> ir.Value:
        half = _scalar_like(op, 0.5, hidden_states)
        hidden_states = op.Add(
            hidden_states,
            op.Mul(
                self.feed_forward1(op, self.norm_feed_forward1(op, hidden_states)),
                half,
            ),
        )
        hidden_states = op.Add(
            hidden_states,
            self.self_attn(
                op,
                self.norm_self_att(op, hidden_states),
                position_embeddings,
                attention_mask,
            ),
        )
        hidden_states = op.Add(
            hidden_states,
            self.conv(op, self.norm_conv(op, hidden_states), valid_frames),
        )
        hidden_states = op.Add(
            hidden_states,
            op.Mul(
                self.feed_forward2(op, self.norm_feed_forward2(op, hidden_states)),
                half,
            ),
        )
        return self.norm_out(op, hidden_states)


class ParakeetFastConformerEncoder(nn.Module):
    """Hugging Face Parakeet offline FastConformer audio encoder.

    Inputs are normalized log-mel features ``(batch, frames, mel_bins)`` and a
    boolean valid-frame mask. The output is ``(batch, ceil(frames / 8), hidden)``.
    """

    def __init__(self, config: ParakeetCTCConfig):
        super().__init__()
        self.subsampling = _ParakeetSubsampling(config)
        self.encode_positions = _ParakeetRelativePositionEncoding(
            config.hidden_size, config.dtype
        )
        self.layers = nn.ModuleList(
            [_ParakeetEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self._num_subsampling_layers = int(math.log2(config.subsampling_factor))
        self._subsampling_kernel = config.subsampling_conv_kernel_size
        self._subsampling_stride = config.subsampling_conv_stride
        self._subsampling_padding = (config.subsampling_conv_kernel_size - 1) // 2
        self._input_scale = math.sqrt(config.hidden_size) if config.scale_input else 1.0

    def _output_mask(
        self,
        op: OpBuilder,
        attention_mask: ir.Value,
        target_length: ir.Value,
    ) -> ir.Value:
        lengths = op.ReduceSum(
            op.Cast(attention_mask, to=ir.DataType.INT64),
            axes=[1],
            keepdims=0,
        )
        add_pad = 2 * self._subsampling_padding - self._subsampling_kernel
        stride = op.Constant(value=ir.tensor(np.int64(self._subsampling_stride)))
        for _ in range(self._num_subsampling_layers):
            lengths = op.Add(
                op.Div(
                    op.Add(
                        lengths,
                        op.Constant(value=ir.tensor(np.int64(add_pad))),
                    ),
                    stride,
                ),
                op.Constant(value=ir.tensor(np.int64(1))),
            )
        frame_ids = op.Range(
            op.Constant(value=ir.tensor(np.int64(0))),
            op.Squeeze(target_length),
            op.Constant(value=ir.tensor(np.int64(1))),
        )
        return op.Less(
            op.Unsqueeze(frame_ids, op.Constant(value_ints=[0])),
            op.Unsqueeze(lengths, op.Constant(value_ints=[1])),
        )

    def forward(
        self,
        op: OpBuilder,
        input_features: ir.Value,
        attention_mask: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        hidden_states = self.subsampling(op, input_features, attention_mask)
        hidden_states = op.Mul(
            hidden_states, _scalar_like(op, self._input_scale, hidden_states)
        )
        position_embeddings = self.encode_positions(op, hidden_states)
        output_mask = self._output_mask(op, attention_mask, _dim(op, hidden_states, 1))
        attention_mask_4d = op.And(
            op.Unsqueeze(output_mask, op.Constant(value_ints=[1, 2])),
            op.Unsqueeze(output_mask, op.Constant(value_ints=[1, 3])),
        )

        for layer in self.layers:
            hidden_states = layer(
                op,
                hidden_states,
                position_embeddings,
                attention_mask_4d,
                output_mask,
            )
        return hidden_states, output_mask
