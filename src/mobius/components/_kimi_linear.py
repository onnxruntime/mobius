# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Kimi Linear's KDA recurrence and NoPE MLA attention components."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components._common import Linear
from mobius.components._rms_norm import RMSNorm


class _KimiDepthwiseConv1d(nn.Module):
    """HF-aligned ShortConvolution weight wrapped by the stateful ONNX function."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.weight = nn.Parameter([channels, kernel_size])
        self._channels = channels

    def forward(self, op: OpBuilder, x: ir.Value, state: ir.Value):
        weight = op.Unsqueeze(self.weight, [1])
        bias = op.Expand(op.CastLike(0.0, self.weight), [self._channels])
        return op.CausalConvWithState(
            x,
            weight,
            bias,
            state,
            activation="silu",
            _domain="com.microsoft",
            _outputs=2,
        )


class _KimiGatedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self._eps = eps

    def forward(self, op: OpBuilder, hidden_states: ir.Value, gate: ir.Value):
        hidden_states = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        weight = op.Cast(self.weight, to=ir.DataType.FLOAT)
        normed = op.RMSNormalization(
            hidden_states, weight, axis=-1, epsilon=self._eps, stash_type=1
        )
        return op.Mul(normed, op.Sigmoid(op.Cast(gate, to=ir.DataType.FLOAT)))


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention with three convolution histories and FP32 recurrence."""

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__()
        linear_class = linear_class or Linear
        self._heads = config.linear_num_key_heads
        self._head_dim = config.linear_key_head_dim
        self._projection_size = self._heads * self._head_dim
        kernel = config.linear_conv_kernel_dim

        self.q_proj = linear_class(config.hidden_size, self._projection_size, bias=False)
        self.k_proj = linear_class(config.hidden_size, self._projection_size, bias=False)
        self.v_proj = linear_class(config.hidden_size, self._projection_size, bias=False)
        self.q_conv1d = _KimiDepthwiseConv1d(self._projection_size, kernel)
        self.k_conv1d = _KimiDepthwiseConv1d(self._projection_size, kernel)
        self.v_conv1d = _KimiDepthwiseConv1d(self._projection_size, kernel)

        self.A_log = nn.Parameter([1, 1, self._heads, 1])
        self.f_a_proj = linear_class(config.hidden_size, self._head_dim, bias=False)
        self.f_b_proj = linear_class(self._head_dim, self._projection_size, bias=False)
        self.dt_bias = nn.Parameter([self._projection_size])
        self.b_proj = linear_class(config.hidden_size, self._heads, bias=False)
        self.g_a_proj = linear_class(config.hidden_size, self._head_dim, bias=False)
        self.g_b_proj = linear_class(self._head_dim, self._projection_size, bias=False)
        self.o_norm = _KimiGatedRMSNorm(self._head_dim, config.rms_norm_eps)
        self.o_proj = linear_class(self._projection_size, config.hidden_size, bias=False)
        self._eps = config.rms_norm_eps

    def _project_conv(
        self,
        op: OpBuilder,
        projection: nn.Module,
        convolution: nn.Module,
        hidden_states: ir.Value,
        state: ir.Value,
    ):
        projected = op.Transpose(projection(op, hidden_states), perm=[0, 2, 1])
        value, present = convolution(op, projected, state)
        return op.Transpose(value, perm=[0, 2, 1]), present

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        q_conv_state: ir.Value,
        k_conv_state: ir.Value,
        v_conv_state: ir.Value,
        recurrent_state: ir.Value,
    ):
        seq_len = op.Shape(hidden_states, start=1, end=2)
        current_mask = op.Slice(
            attention_mask,
            op.Neg(seq_len),
            op.Constant(value_ints=[9223372036854775807]),
            op.Constant(value_ints=[1]),
        )
        current_mask = op.Unsqueeze(op.CastLike(current_mask, hidden_states), [-1])
        masked_hidden_states = op.Mul(hidden_states, current_mask)

        q, present_q = self._project_conv(
            op, self.q_proj, self.q_conv1d, masked_hidden_states, q_conv_state
        )
        k, present_k = self._project_conv(
            op, self.k_proj, self.k_conv1d, masked_hidden_states, k_conv_state
        )
        v, present_v = self._project_conv(
            op, self.v_proj, self.v_conv1d, masked_hidden_states, v_conv_state
        )

        shape = [0, 0, self._heads, self._head_dim]
        q4 = op.Cast(op.Reshape(q, shape), to=ir.DataType.FLOAT)
        k4 = op.Cast(op.Reshape(k, shape), to=ir.DataType.FLOAT)
        q4 = op.Div(
            q4,
            op.Sqrt(op.Add(op.ReduceSumSquare(q4, [-1], keepdims=True), self._eps)),
        )
        k4 = op.Div(
            k4,
            op.Sqrt(op.Add(op.ReduceSumSquare(k4, [-1], keepdims=True), self._eps)),
        )
        q = op.Reshape(q4, [0, 0, self._projection_size])
        k = op.Reshape(k4, [0, 0, self._projection_size])
        v = op.Cast(v, to=ir.DataType.FLOAT)

        decay = self.f_b_proj(op, self.f_a_proj(op, masked_hidden_states))
        decay = op.Cast(decay, to=ir.DataType.FLOAT)
        decay = op.Softplus(op.Add(decay, op.Cast(self.dt_bias, to=ir.DataType.FLOAT)))
        a = op.Neg(op.Exp(op.Cast(self.A_log, to=ir.DataType.FLOAT)))
        decay = op.Reshape(decay, [0, 0, self._heads, self._head_dim])
        decay = op.Reshape(op.Mul(decay, a), [0, 0, self._projection_size])
        decay = op.Mul(decay, current_mask)
        beta = op.Sigmoid(
            op.Cast(self.b_proj(op, masked_hidden_states), to=ir.DataType.FLOAT)
        )
        beta = op.Mul(beta, current_mask)

        output, present_recurrent = op.LinearAttention(
            q,
            k,
            v,
            recurrent_state,
            decay,
            beta,
            update_rule="gated_delta",
            scale=self._head_dim**-0.5,
            q_num_heads=self._heads,
            kv_num_heads=self._heads,
            _domain="com.microsoft",
            _outputs=2,
        )
        output = op.Reshape(output, shape)
        gate = self.g_b_proj(op, self.g_a_proj(op, masked_hidden_states))
        gate = op.Reshape(gate, shape)
        output = self.o_norm(op, output, gate)
        output = op.CastLike(
            op.Reshape(output, [0, 0, self._projection_size]),
            hidden_states,
        )
        output = op.Mul(output, current_mask)
        output = self.o_proj(op, output)
        return output, (present_q, present_k, present_v, present_recurrent)


class KimiMLAAttention(nn.Module):
    """Kimi's 192-wide MLA path where the nominal PE dimensions are not rotated."""

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__()
        linear_class = linear_class or Linear
        self._heads = config.num_attention_heads
        self._nope = config.qk_nope_head_dim
        self._extra = config.qk_rope_head_dim
        self._qk_dim = self._nope + self._extra
        self._value_dim = config.v_head_dim
        self._kv_rank = config.kv_lora_rank
        self.q_proj = linear_class(
            config.hidden_size, self._heads * self._qk_dim, bias=False
        )
        self.kv_a_proj_with_mqa = linear_class(
            config.hidden_size, self._kv_rank + self._extra, bias=False
        )
        self.kv_a_layernorm = RMSNorm(self._kv_rank, eps=config.rms_norm_eps)
        # llama.cpp serializes these as separate MatMul roles. Keeping them
        # separate preserves quantized GGUF weights without a lossy fuse/split.
        self.k_b_proj = linear_class(
            self._kv_rank, self._heads * self._nope, bias=False
        )
        self.v_b_proj = linear_class(
            self._kv_rank, self._heads * self._value_dim, bias=False
        )
        self.o_proj = linear_class(
            self._heads * self._value_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        past_key_value: tuple[ir.Value, ir.Value],
    ):
        query = op.Reshape(
            self.q_proj(op, hidden_states), [0, 0, self._heads, self._qk_dim]
        )
        compressed, key_extra = op.Split(
            self.kv_a_proj_with_mqa(op, hidden_states),
            [self._kv_rank, self._extra],
            axis=-1,
            _outputs=2,
        )
        compressed = self.kv_a_layernorm(op, compressed)
        key_nope = op.Reshape(
            self.k_b_proj(op, compressed), [0, 0, self._heads, self._nope]
        )
        value = op.Reshape(
            self.v_b_proj(op, compressed), [0, 0, self._heads, self._value_dim]
        )
        key_extra = op.Expand(
            op.Reshape(key_extra, [0, 0, 1, self._extra]),
            [1, 1, self._heads, 1],
        )
        key = op.Reshape(op.Concat(key_nope, key_extra, axis=-1), [0, 0, -1])
        query = op.Reshape(query, [0, 0, -1])
        value = op.Reshape(value, [0, 0, -1])
        output, present_key, present_value = op.Attention(
            query,
            key,
            value,
            attention_bias,
            past_key_value[0],
            past_key_value[1],
            q_num_heads=self._heads,
            kv_num_heads=self._heads,
            scale=self._qk_dim**-0.5,
            _outputs=3,
        )
        return self.o_proj(op, output), (present_key, present_value)
