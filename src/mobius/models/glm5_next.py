# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GLM-5.3-Flash multimodal NoPE KDA/DSA mixture-of-experts model.

This module replicates Hugging Face ``Glm5NextForConditionalGeneration``:
four-stream manifold-constrained hyper-connections feed a hybrid text decoder
whose layers alternate Kimi Delta Attention and k-pool-compressed DeepSeek
Sparse Attention, while a packed dynamic-resolution ViT supplies image and
video features to the shared-token embedding mixer.
"""

from __future__ import annotations

import logging
import re
from typing import ClassVar

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import Glm5NextConfig
from mobius.components import (
    Embedding,
    Glm5NextVisionModel,
    LayerNorm,
    Linear,
    RMSNorm,
)
from mobius.models.base import linear_class_for_config
from mobius.models.deepseek import DeepSeekMoEGate

logger = logging.getLogger(__name__)

_INT64_MAX = 9223372036854775807
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


class Glm5NextClampedMLP(nn.Module):
    """Checkpoint-aligned clamped SwiGLU feed-forward network."""

    def __init__(
        self,
        config: Glm5NextConfig,
        intermediate_size: int | None = None,
        linear_class: type | None = None,
    ) -> None:
        super().__init__()
        linear_class = linear_class or Linear
        width = intermediate_size or config.intermediate_size
        self.gate_proj = linear_class(config.hidden_size, width, bias=False)
        self.up_proj = linear_class(config.hidden_size, width, bias=False)
        self.down_proj = linear_class(width, config.hidden_size, bias=False)
        self._limit = config.swiglu_limit

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        gate = op.Clip(self.gate_proj(op, hidden_states), None, self._limit)
        up = op.Clip(self.up_proj(op, hidden_states), -self._limit, self._limit)
        return self.down_proj(op, op.Mul(op.Swish(gate), up))


class Glm5NextExperts(nn.Module):
    """Packed expert bank with exact selected-expert SwiGLU evaluation."""

    def __init__(self, config: Glm5NextConfig) -> None:
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        assert config.moe_intermediate_size is not None
        self._num_experts = config.num_local_experts
        self._top_k = config.num_experts_per_tok
        self._hidden_size = config.hidden_size
        self._intermediate_size = config.moe_intermediate_size
        self._limit = config.swiglu_limit
        self.gate_up_proj = nn.Parameter(
            [
                self._num_experts,
                2 * self._intermediate_size,
                self._hidden_size,
            ]
        )
        self.down_proj = nn.Parameter(
            [
                self._num_experts,
                self._hidden_size,
                self._intermediate_size,
            ]
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        selected_experts: ir.Value,
        routing_weights: ir.Value,
    ) -> ir.Value:
        original_shape = op.Shape(hidden_states)
        flat_hidden = op.Reshape(hidden_states, [-1, self._hidden_size])
        flat_selected = op.Reshape(selected_experts, [-1, self._top_k])
        flat_weights = op.Reshape(routing_weights, [-1, self._top_k])
        result = None
        for expert_index in range(self._num_experts):
            gate_up_weight = op.Squeeze(
                op.Gather(
                    self.gate_up_proj,
                    op.Constant(value_ints=[expert_index]),
                    axis=0,
                ),
                [0],
            )
            projected = op.MatMul(flat_hidden, op.Transpose(gate_up_weight))
            gate, up = op.Split(
                projected,
                [self._intermediate_size, self._intermediate_size],
                axis=-1,
                _outputs=2,
            )
            gate = op.Clip(gate, None, self._limit)
            up = op.Clip(up, -self._limit, self._limit)
            activated = op.Mul(op.Swish(gate), up)
            down_weight = op.Squeeze(
                op.Gather(
                    self.down_proj,
                    op.Constant(value_ints=[expert_index]),
                    axis=0,
                ),
                [0],
            )
            expert_output = op.MatMul(activated, op.Transpose(down_weight))
            selected = op.Equal(
                flat_selected,
                op.Constant(value_int=expert_index),
            )
            weight = op.ReduceSum(
                op.Mul(
                    flat_weights,
                    op.CastLike(selected, flat_weights),
                ),
                [-1],
                keepdims=True,
            )
            contribution = op.CastLike(
                op.Mul(
                    op.Cast(expert_output, to=ir.DataType.FLOAT),
                    weight,
                ),
                expert_output,
            )
            result = contribution if result is None else op.Add(result, contribution)
        assert result is not None
        return op.Reshape(result, original_shape)


class Glm5NextRouter(DeepSeekMoEGate):
    """DeepSeek-style router with the selection-only bias pinned to fp32."""

    def __init__(self, config: Glm5NextConfig) -> None:
        super().__init__(config)
        self.e_score_correction_bias._keep_float32 = True


class Glm5NextMoE(nn.Module):
    """Sigmoid noaux-tc routed experts plus one always-active shared expert."""

    def __init__(self, config: Glm5NextConfig) -> None:
        super().__init__()
        assert config.moe_intermediate_size is not None
        self.gate = Glm5NextRouter(config)
        self.experts = Glm5NextExperts(config)
        self.shared_experts = Glm5NextClampedMLP(
            config,
            config.moe_intermediate_size * (config.n_shared_experts or 1),
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        routing_weights, selected_experts = self.gate(op, hidden_states)
        routed = self.experts(
            op,
            hidden_states,
            selected_experts,
            routing_weights,
        )
        return op.Add(routed, self.shared_experts(op, hidden_states))


class _Glm5NextDepthwiseConv1d(nn.Module):
    """One checkpoint-aligned depthwise causal convolution."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter([channels, 1, kernel_size])
        self._channels = channels
        self._kernel_size = kernel_size

    def forward(
        self,
        op: OpBuilder,
        projected: ir.Value,
        state: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        # The cache stores K values like Transformers' LinearAttentionLayer.
        # Dropping its oldest entry leaves K-1 history values for this chunk.
        projected = op.Transpose(projected, perm=[0, 2, 1])
        history = op.Concat(state, projected, axis=2)
        conv_input = op.Slice(
            history,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[_INT64_MAX]),
            op.Constant(value_ints=[2]),
        )
        output = op.Conv(
            conv_input,
            self.weight,
            group=self._channels,
            kernel_shape=[self._kernel_size],
        )
        output = op.Swish(output)
        present = op.Slice(
            history,
            op.Constant(value_ints=[-self._kernel_size]),
            op.Constant(value_ints=[_INT64_MAX]),
            op.Constant(value_ints=[2]),
        )
        return op.Transpose(output, perm=[0, 2, 1]), present


class _Glm5NextGatedRMSNorm(nn.Module):
    """Strict-fp32 RMS normalization followed by a sigmoid output gate."""

    def __init__(self, head_dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter([head_dim])
        self._eps = eps

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        gate: ir.Value,
    ) -> ir.Value:
        hidden_f32 = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        variance = op.ReduceMean(
            op.Mul(hidden_f32, hidden_f32),
            [-1],
            keepdims=True,
        )
        normalized = op.Mul(
            hidden_f32,
            op.Reciprocal(op.Sqrt(op.Add(variance, self._eps))),
        )
        normalized = op.Mul(normalized, op.Cast(self.weight, to=ir.DataType.FLOAT))
        gated = op.Mul(normalized, op.Sigmoid(op.Cast(gate, to=ir.DataType.FLOAT)))
        return op.CastLike(gated, hidden_states)


class Glm5NextLinearAttention(nn.Module):
    """Kimi Delta Attention with fused semantic convolution/cache state."""

    _QK_NORM_EPS = 1e-6

    def __init__(
        self,
        config: Glm5NextConfig,
        linear_class: type | None = None,
    ) -> None:
        super().__init__()
        linear_class = linear_class or Linear
        assert config.linear_num_heads is not None
        assert config.linear_head_dim is not None
        assert config.linear_lower_bound is not None
        self._heads = config.linear_num_heads
        self._head_dim = config.linear_head_dim
        self._projection_size = self._heads * self._head_dim
        self._lower_bound = config.linear_lower_bound

        self.q_proj = linear_class(config.hidden_size, self._projection_size, bias=False)
        self.k_proj = linear_class(config.hidden_size, self._projection_size, bias=False)
        self.v_proj = linear_class(config.hidden_size, self._projection_size, bias=False)
        self.q_conv1d = _Glm5NextDepthwiseConv1d(
            self._projection_size, config.linear_conv_kernel_dim
        )
        self.k_conv1d = _Glm5NextDepthwiseConv1d(
            self._projection_size, config.linear_conv_kernel_dim
        )
        self.v_conv1d = _Glm5NextDepthwiseConv1d(
            self._projection_size, config.linear_conv_kernel_dim
        )
        self.f_a_proj = linear_class(config.hidden_size, self._head_dim, bias=False)
        self.f_b_proj = linear_class(self._head_dim, self._projection_size, bias=False)
        self.dt_bias = nn.Parameter([self._projection_size])
        self.A_log = nn.Parameter([self._heads])
        self.b_proj = linear_class(config.hidden_size, self._heads, bias=False)
        self.g_a_proj = linear_class(config.hidden_size, self._head_dim, bias=False)
        self.g_b_proj = linear_class(self._head_dim, self._projection_size, bias=False)
        self.o_norm = _Glm5NextGatedRMSNorm(self._head_dim, config.rms_norm_eps)
        self.o_proj = linear_class(self._projection_size, config.hidden_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        current_mask: ir.Value,
        past_state: tuple[ir.Value, ir.Value],
    ) -> tuple[ir.Value, tuple[ir.Value, ir.Value]]:
        mask = op.Unsqueeze(op.CastLike(current_mask, hidden_states), [-1])
        masked_hidden = op.Mul(hidden_states, mask)
        conv_state, recurrent_state = past_state
        q_state, k_state, v_state = op.Split(
            conv_state,
            [self._projection_size] * 3,
            axis=1,
            _outputs=3,
        )
        query, present_q = self.q_conv1d(op, self.q_proj(op, masked_hidden), q_state)
        key, present_k = self.k_conv1d(op, self.k_proj(op, masked_hidden), k_state)
        value, present_v = self.v_conv1d(op, self.v_proj(op, masked_hidden), v_state)

        head_shape = [0, 0, self._heads, self._head_dim]
        query_f32 = op.Cast(op.Reshape(query, head_shape), to=ir.DataType.FLOAT)
        key_f32 = op.Cast(op.Reshape(key, head_shape), to=ir.DataType.FLOAT)
        query_f32 = op.Div(
            query_f32,
            op.Sqrt(
                op.Add(
                    op.ReduceSumSquare(query_f32, [-1], keepdims=True),
                    self._QK_NORM_EPS,
                )
            ),
        )
        key_f32 = op.Div(
            key_f32,
            op.Sqrt(
                op.Add(
                    op.ReduceSumSquare(key_f32, [-1], keepdims=True),
                    self._QK_NORM_EPS,
                )
            ),
        )
        query = op.Reshape(query_f32, [0, 0, self._projection_size])
        key = op.Reshape(key_f32, [0, 0, self._projection_size])
        value = op.Cast(value, to=ir.DataType.FLOAT)

        forget = op.Add(
            op.Cast(self.f_b_proj(op, self.f_a_proj(op, masked_hidden)), to=ir.DataType.FLOAT),
            op.Cast(self.dt_bias, to=ir.DataType.FLOAT),
        )
        decay_rate = op.Reshape(
            op.Exp(op.Cast(self.A_log, to=ir.DataType.FLOAT)),
            [1, 1, self._heads, 1],
        )
        forget = op.Reshape(forget, head_shape)
        decay = op.Mul(self._lower_bound, op.Sigmoid(op.Mul(decay_rate, forget)))
        decay = op.Reshape(decay, [0, 0, self._projection_size])
        beta = op.Sigmoid(op.Cast(self.b_proj(op, masked_hidden), to=ir.DataType.FLOAT))

        output, present_recurrent = op.LinearAttention(
            query,
            key,
            value,
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
        output = op.Reshape(op.CastLike(output, hidden_states), head_shape)
        gate = op.Reshape(
            self.g_b_proj(op, self.g_a_proj(op, masked_hidden)),
            head_shape,
        )
        output = self.o_norm(op, output, gate)
        output = op.Reshape(output, [0, 0, self._projection_size])
        output = self.o_proj(op, op.Mul(output, mask))
        return output, (
            op.Concat(present_q, present_k, present_v, axis=1),
            op.Cast(present_recurrent, to=ir.DataType.FLOAT),
        )


class Glm5NextIndexer(nn.Module):
    """NoPE DSA indexer with learned k-pool compression and tail inclusion."""

    def __init__(
        self,
        config: Glm5NextConfig,
        linear_class: type | None = None,
    ) -> None:
        super().__init__()
        linear_class = linear_class or Linear
        assert config.q_lora_rank is not None
        assert config.index_head_dim is not None
        assert config.index_n_heads is not None
        assert config.index_topk is not None
        self._head_dim = config.index_head_dim
        self._heads = config.index_n_heads
        self._topk = config.index_topk
        self._pool = config.index_kpool
        self._always_tail = config.index_kpool_always_select_tail
        self._scale = self._head_dim**-0.5

        self.wq_b = linear_class(
            config.q_lora_rank,
            self._heads * self._head_dim,
            bias=False,
        )
        self.wk = linear_class(config.hidden_size, self._head_dim, bias=False)
        self.k_norm = LayerNorm(self._head_dim, eps=1e-6)
        self.weights_proj = linear_class(config.hidden_size, self._heads, bias=False)
        self.index_kpool_compress_ape = nn.Parameter([self._pool, self._head_dim])
        self.index_kpool_compress_gate = nn.Parameter([self._head_dim, config.hidden_size])

    @staticmethod
    def _gather_batched_rows(
        op: OpBuilder,
        values: ir.Value,
        indices: ir.Value,
    ) -> ir.Value:
        batch = op.Squeeze(op.Shape(values, start=0, end=1), [0])
        batch_ids = op.Range(
            op.Constant(value_int=0),
            batch,
            op.Constant(value_int=1),
        )
        batch_ids = op.Expand(
            op.Reshape(batch_ids, [-1, 1, 1]),
            op.Shape(indices),
        )
        gather_indices = op.Concat(
            op.Unsqueeze(batch_ids, [-1]),
            op.Unsqueeze(indices, [-1]),
            axis=-1,
        )
        return op.GatherND(values, gather_indices)

    def _pooled_states(
        self,
        op: OpBuilder,
        packed_states: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
        keys, gate_scores, valid_column = op.Split(
            packed_states,
            [self._head_dim, self._head_dim, 1],
            axis=-1,
            _outputs=3,
        )
        valid_keys = op.Not(
            op.Equal(
                op.Cast(op.Squeeze(valid_column, [-1]), to=ir.DataType.FLOAT),
                0.0,
            )
        )
        total_length = op.Squeeze(op.Shape(keys, start=1, end=2), [0])
        valid_int = op.Cast(valid_keys, to=ir.DataType.INT64)
        any_valid = op.Greater(op.ReduceMax(valid_int, [1], keepdims=False), 0)
        first_key = op.ArgMax(valid_int, axis=1, keepdims=False)
        first_key = op.Where(
            any_valid,
            first_key,
            op.Expand(total_length, op.Shape(first_key)),
        )

        pool_count = op.Div(
            op.Add(total_length, self._pool - 1),
            self._pool,
        )
        pool_offsets = op.Range(
            op.Constant(value_int=0),
            op.Mul(pool_count, self._pool),
            op.Constant(value_int=1),
        )
        pool_offsets = op.Reshape(pool_offsets, [1, -1, self._pool])
        pool_indices = op.Add(
            op.Reshape(first_key, [-1, 1, 1]),
            pool_offsets,
        )
        safe_indices = op.Clip(
            pool_indices,
            op.Constant(value_int=0),
            op.Sub(total_length, 1),
        )
        grouped_keys = self._gather_batched_rows(op, keys, safe_indices)
        grouped_scores = self._gather_batched_rows(op, gate_scores, safe_indices)
        grouped_valid = self._gather_batched_rows(
            op,
            op.Unsqueeze(valid_keys, [-1]),
            safe_indices,
        )
        grouped_valid = op.Squeeze(grouped_valid, [-1])
        grouped_valid = op.And(
            grouped_valid,
            op.Less(pool_indices, total_length),
        )
        pool_valid = op.Equal(
            op.ReduceSum(
                op.Cast(grouped_valid, to=ir.DataType.INT64),
                [-1],
                keepdims=False,
            ),
            self._pool,
        )

        # Stable masked softmax across each complete pool in float32.
        logits = op.Add(
            op.Cast(grouped_scores, to=ir.DataType.FLOAT),
            op.Cast(self.index_kpool_compress_ape, to=ir.DataType.FLOAT),
        )
        masked_logits = op.Where(
            op.Unsqueeze(grouped_valid, [-1]),
            logits,
            op.CastLike(-1e30, logits),
        )
        shifted = op.Sub(
            masked_logits,
            op.ReduceMax(masked_logits, [2], keepdims=True),
        )
        exponentials = op.Mul(
            op.Exp(shifted),
            op.CastLike(op.Unsqueeze(grouped_valid, [-1]), shifted),
        )
        denominator = op.ReduceSum(exponentials, [2], keepdims=True)
        probabilities = op.Where(
            op.Greater(denominator, 0.0),
            op.Div(exponentials, op.Max(denominator, 1e-20)),
            op.Expand(op.CastLike(0.0, exponentials), op.Shape(exponentials)),
        )
        pool_keys = op.ReduceSum(
            op.Mul(probabilities, op.Cast(grouped_keys, to=ir.DataType.FLOAT)),
            [2],
            keepdims=False,
        )
        return pool_keys, pool_indices, pool_valid, valid_keys

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        q_resid: ir.Value,
        current_mask: ir.Value,
        past_indexer_state: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        query = op.Reshape(
            self.wq_b(op, q_resid),
            [0, 0, self._heads, self._head_dim],
        )
        key = self.k_norm(op, self.wk(op, hidden_states))
        gate_scores = op.MatMul(
            hidden_states,
            op.Transpose(self.index_kpool_compress_gate, perm=[1, 0]),
        )
        packed_current = op.Concat(
            key,
            gate_scores,
            op.Unsqueeze(op.CastLike(current_mask, key), [-1]),
            axis=-1,
        )
        packed_states = op.Concat(past_indexer_state, packed_current, axis=1)
        pool_keys, pool_indices, pool_valid, valid_keys = self._pooled_states(
            op, packed_states
        )

        scores = op.MatMul(
            op.Cast(query, to=ir.DataType.FLOAT),
            op.Unsqueeze(op.Transpose(pool_keys, perm=[0, 2, 1]), [1]),
        )
        scores = op.Relu(op.Mul(scores, self._scale))
        head_weights = op.Mul(
            op.Cast(self.weights_proj(op, hidden_states), to=ir.DataType.FLOAT),
            self._heads**-0.5,
        )
        index_scores = op.ReduceSum(
            op.Mul(scores, op.Unsqueeze(head_weights, [3])),
            [2],
            keepdims=False,
        )

        total_length = op.Squeeze(op.Shape(packed_states, start=1, end=2), [0])
        current_positions = op.Sub(
            op.CumSum(
                op.Expand(
                    op.Constant(value_int=1),
                    op.Shape(current_mask),
                ),
                op.Constant(value_int=1),
            ),
            1,
        )
        query_positions = op.Add(
            current_positions,
            op.Shape(past_indexer_state, start=1, end=2),
        )
        pool_end = op.Gather(
            pool_indices,
            op.Constant(value_int=self._pool - 1),
            axis=2,
        )
        pool_end = op.Clip(pool_end, op.Constant(value_int=0), op.Sub(total_length, 1))
        visible = op.LessOrEqual(
            op.Unsqueeze(pool_end, [1]),
            op.Unsqueeze(query_positions, [-1]),
        )
        valid_candidates = op.And(
            visible,
            op.Unsqueeze(pool_valid, [1]),
        )
        index_scores = op.Where(
            valid_candidates,
            index_scores,
            op.CastLike(-1e30, index_scores),
        )
        select_k = op.Min(
            op.Shape(index_scores, start=2, end=3),
            op.Constant(value_ints=[self._topk // self._pool]),
        )
        _, selected = op.TopK(index_scores, select_k, axis=-1, _outputs=2)
        selected_valid = op.GatherElements(valid_candidates, selected, axis=2)

        expanded_pool_shape = op.Concat(
            op.Shape(selected, start=0, end=2),
            op.Shape(pool_indices, start=1, end=3),
            axis=0,
        )
        expanded_pools = op.Expand(op.Unsqueeze(pool_indices, [1]), expanded_pool_shape)
        gather_indices = op.Expand(
            op.Unsqueeze(selected, [-1]),
            op.Concat(
                op.Shape(selected),
                op.Constant(value_ints=[self._pool]),
                axis=0,
            ),
        )
        selected_indices = op.GatherElements(
            expanded_pools,
            gather_indices,
            axis=2,
        )
        selected_indices = op.Reshape(selected_indices, [0, 0, -1])
        selected_indices = op.Where(
            op.Reshape(
                op.Expand(
                    op.Unsqueeze(selected_valid, [-1]),
                    op.Shape(gather_indices),
                ),
                op.Shape(selected_indices),
            ),
            selected_indices,
            op.Expand(op.Constant(value_int=-1), op.Shape(selected_indices)),
        )

        output_width = self._topk
        if self._always_tail and self._pool > 1:
            first_key = op.ArgMax(
                op.Cast(valid_keys, to=ir.DataType.INT64),
                axis=1,
                keepdims=False,
            )
            causal = op.LessOrEqual(
                op.Reshape(
                    op.Range(
                        op.Constant(value_int=0),
                        total_length,
                        op.Constant(value_int=1),
                    ),
                    [1, 1, -1],
                ),
                op.Unsqueeze(query_positions, [-1]),
            )
            token_visible = op.And(causal, op.Unsqueeze(valid_keys, [1]))
            visible_count = op.ReduceSum(
                op.Cast(token_visible, to=ir.DataType.INT64),
                [-1],
                keepdims=False,
            )
            tail_count = op.Mod(visible_count, self._pool)
            tail_start = op.Add(
                op.Unsqueeze(first_key, [1]),
                op.Sub(visible_count, tail_count),
            )
            tail_offsets = op.Range(
                op.Constant(value_int=0),
                op.Constant(value_int=self._pool - 1),
                op.Constant(value_int=1),
            )
            tail_indices = op.Add(
                op.Unsqueeze(tail_start, [-1]),
                op.Reshape(tail_offsets, [1, 1, -1]),
            )
            tail_valid = op.And(
                op.Less(
                    op.Reshape(tail_offsets, [1, 1, -1]),
                    op.Unsqueeze(tail_count, [-1]),
                ),
                op.Less(tail_indices, total_length),
            )
            safe_tail = op.Clip(tail_indices, 0, op.Sub(total_length, 1))
            tail_valid = op.And(
                tail_valid,
                op.GatherElements(token_visible, safe_tail, axis=2),
            )
            tail_indices = op.Where(
                tail_valid,
                tail_indices,
                op.Expand(op.Constant(value_int=-1), op.Shape(tail_indices)),
            )
            selected_indices = op.Concat(selected_indices, tail_indices, axis=-1)
            output_width += self._pool - 1

        width = op.Shape(selected_indices, start=2, end=3)
        pad_width = op.Sub(op.Constant(value_ints=[output_width]), width)
        padding = op.Expand(
            op.Constant(value_int=-1),
            op.Concat(op.Shape(selected_indices, start=0, end=2), pad_width, axis=0),
        )
        selected_indices = op.Concat(selected_indices, padding, axis=-1)
        selected_indices = op.Slice(
            selected_indices,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[output_width]),
            op.Constant(value_ints=[2]),
        )
        selected_indices = op.Where(
            op.Unsqueeze(current_mask, [-1]),
            selected_indices,
            op.Expand(op.Constant(value_int=-1), op.Shape(selected_indices)),
        )
        return selected_indices, packed_states


class Glm5NextSparseAttention(nn.Module):
    """NoPE MLA restricted by the exact k-pool indexer selection."""

    def __init__(
        self,
        config: Glm5NextConfig,
        linear_class: type | None = None,
    ) -> None:
        super().__init__()
        linear_class = linear_class or Linear
        assert config.q_lora_rank is not None
        assert config.kv_lora_rank is not None
        assert config.qk_nope_head_dim is not None
        assert config.v_head_dim is not None
        self._heads = config.num_attention_heads
        self._dtype = config.dtype
        self._q_rank = config.q_lora_rank
        self._kv_rank = config.kv_lora_rank
        self._qk_dim = config.qk_nope_head_dim
        self._value_dim = config.v_head_dim
        self._scale = self._qk_dim**-0.5
        self.q_a_proj = linear_class(config.hidden_size, self._q_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self._q_rank, eps=config.rms_norm_eps)
        self.q_b_proj = linear_class(
            self._q_rank,
            self._heads * self._qk_dim,
            bias=False,
        )
        self.kv_a_proj_with_mqa = linear_class(
            config.hidden_size,
            self._kv_rank,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(self._kv_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = linear_class(
            self._kv_rank,
            self._heads * (self._qk_dim + self._value_dim),
            bias=False,
        )
        self.o_proj = linear_class(
            self._heads * self._value_dim,
            config.hidden_size,
            bias=False,
        )
        self.indexer = Glm5NextIndexer(config, linear_class)

    def _sparse_bias(
        self,
        op: OpBuilder,
        indices: ir.Value,
        total_length: ir.Value,
        dtype_like: ir.Value,
    ) -> ir.Value:
        valid = op.And(op.GreaterOrEqual(indices, 0), op.Less(indices, total_length))
        safe = op.Clip(indices, op.Constant(value_int=0), op.Sub(total_length, 1))
        mask_shape = op.Concat(
            op.Shape(indices, start=0, end=2),
            op.Reshape(total_length, [1]),
            axis=0,
        )
        selected_counts = op.ScatterElements(
            op.Expand(op.Constant(value_int=0), mask_shape),
            safe,
            op.Cast(valid, to=ir.DataType.INT64),
            axis=2,
            reduction="max",
        )
        selected = op.Greater(selected_counts, 0)
        minimum = float(self._dtype.min)
        bias = op.Where(
            selected,
            op.CastLike(0.0, dtype_like),
            op.CastLike(minimum, dtype_like),
        )
        return op.Unsqueeze(bias, [1])

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        current_mask: ir.Value,
        past_state: tuple[ir.Value, ir.Value, ir.Value],
    ) -> tuple[ir.Value, tuple[ir.Value, ir.Value, ir.Value]]:
        q_resid = self.q_a_layernorm(op, self.q_a_proj(op, hidden_states))
        query = self.q_b_proj(op, q_resid)
        compressed = self.kv_a_layernorm(
            op,
            self.kv_a_proj_with_mqa(op, hidden_states),
        )
        expanded = op.Reshape(
            self.kv_b_proj(op, compressed),
            [0, 0, self._heads, self._qk_dim + self._value_dim],
        )
        key, value = op.Split(
            expanded,
            [self._qk_dim, self._value_dim],
            axis=-1,
            _outputs=2,
        )
        key = op.Reshape(key, [0, 0, self._heads * self._qk_dim])
        value = op.Reshape(value, [0, 0, self._heads * self._value_dim])
        selected, present_indexer = self.indexer(
            op,
            hidden_states,
            q_resid,
            current_mask,
            past_state[2],
        )
        total_length = op.Add(
            op.Shape(past_state[0], start=2, end=3),
            op.Shape(hidden_states, start=1, end=2),
        )
        sparse_bias = self._sparse_bias(
            op,
            selected,
            op.Squeeze(total_length, [0]),
            query,
        )
        if self._dtype == ir.DataType.BFLOAT16:
            attention_inputs = (
                op.Cast(query, to=ir.DataType.FLOAT),
                op.Cast(key, to=ir.DataType.FLOAT),
                op.Cast(value, to=ir.DataType.FLOAT),
                op.Cast(sparse_bias, to=ir.DataType.FLOAT),
                op.Cast(past_state[0], to=ir.DataType.FLOAT),
                op.Cast(past_state[1], to=ir.DataType.FLOAT),
            )
        else:
            attention_inputs = (
                query,
                key,
                value,
                sparse_bias,
                past_state[0],
                past_state[1],
            )
        output, present_key, present_value = op.Attention(
            *attention_inputs,
            q_num_heads=self._heads,
            kv_num_heads=self._heads,
            scale=self._scale,
            _outputs=3,
        )
        output = op.CastLike(output, query)
        present_key = op.CastLike(present_key, key)
        present_value = op.CastLike(present_value, value)
        return self.o_proj(op, output), (
            present_key,
            present_value,
            present_indexer,
        )


class Glm5NextDecoderLayer(nn.Module):
    """One mHC-wrapped KDA/DSA attention and dense/MoE feed-forward layer."""

    def __init__(self, config: Glm5NextConfig, layer_idx: int) -> None:
        super().__init__()
        linear_class = linear_class_for_config(config)
        assert config.layer_types is not None
        assert config.mlp_layer_types is not None
        self._hc_mult = config.hc_mult
        self._hc_eps = config.hc_eps
        self._hc_iters = config.hc_sinkhorn_iters
        self._norm_eps = config.rms_norm_eps
        self.self_attn = (
            Glm5NextLinearAttention(config, linear_class)
            if config.layer_types[layer_idx] == "linear_attention"
            else Glm5NextSparseAttention(config, linear_class)
        )
        self.mlp = (
            Glm5NextMoE(config)
            if config.mlp_layer_types[layer_idx] == "sparse"
            else Glm5NextClampedMLP(config, linear_class=linear_class)
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        mix = (2 + self._hc_mult) * self._hc_mult
        hidden = self._hc_mult * config.hidden_size
        # Keep the checkpoint's raw names rather than introducing attn_hc/ffn_hc
        # wrapper prefixes that would require avoidable weight renames.
        self.hc_attn_fn = nn.Parameter([mix, hidden])
        self.hc_attn_base = nn.Parameter([mix])
        self.hc_attn_scale = nn.Parameter([3])
        self.hc_ffn_fn = nn.Parameter([mix, hidden])
        self.hc_ffn_base = nn.Parameter([mix])
        self.hc_ffn_scale = nn.Parameter([3])

    def _hyper_connection(
        self,
        op: OpBuilder,
        hidden_streams: ir.Value,
        fn: ir.Value,
        base: ir.Value,
        scale: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        flat = op.Reshape(hidden_streams, [0, 0, -1])
        flat_f32 = op.Cast(flat, to=ir.DataType.FLOAT)
        variance = op.ReduceMean(
            op.Mul(flat_f32, flat_f32),
            [-1],
            keepdims=True,
        )
        normalized = op.Mul(
            flat_f32,
            op.Reciprocal(op.Sqrt(op.Add(variance, self._norm_eps))),
        )
        logits = op.MatMul(normalized, op.Transpose(op.Cast(fn, to=ir.DataType.FLOAT)))
        pre_logits, post_logits, combine_logits = op.Split(
            logits,
            [self._hc_mult, self._hc_mult, self._hc_mult * self._hc_mult],
            axis=-1,
            _outputs=3,
        )
        pre_base, post_base, combine_base = op.Split(
            op.Cast(base, to=ir.DataType.FLOAT),
            [self._hc_mult, self._hc_mult, self._hc_mult * self._hc_mult],
            axis=-1,
            _outputs=3,
        )
        scale_f32 = op.Cast(scale, to=ir.DataType.FLOAT)
        pre = op.Add(
            op.Sigmoid(
                op.Add(
                    op.Mul(
                        pre_logits,
                        op.Gather(scale_f32, op.Constant(value_int=0)),
                    ),
                    pre_base,
                )
            ),
            self._hc_eps,
        )
        post = op.Mul(
            2.0,
            op.Sigmoid(
                op.Add(
                    op.Mul(
                        post_logits,
                        op.Gather(scale_f32, op.Constant(value_int=1)),
                    ),
                    post_base,
                )
            ),
        )
        combine = op.Reshape(
            op.Add(
                op.Mul(
                    combine_logits,
                    op.Gather(scale_f32, op.Constant(value_int=2)),
                ),
                combine_base,
            ),
            [0, 0, self._hc_mult, self._hc_mult],
        )
        combine = op.Add(op.Softmax(combine, axis=-1), self._hc_eps)
        combine = op.Div(
            combine,
            op.Add(op.ReduceSum(combine, [-2], keepdims=True), self._hc_eps),
        )
        for _ in range(self._hc_iters - 1):
            combine = op.Div(
                combine,
                op.Add(op.ReduceSum(combine, [-1], keepdims=True), self._hc_eps),
            )
            combine = op.Div(
                combine,
                op.Add(op.ReduceSum(combine, [-2], keepdims=True), self._hc_eps),
            )
        collapsed = op.ReduceSum(
            op.Mul(op.Unsqueeze(pre, [-1]), op.Cast(hidden_streams, to=ir.DataType.FLOAT)),
            [2],
            keepdims=False,
        )
        return post, combine, op.CastLike(collapsed, hidden_streams)

    @staticmethod
    def _inject(
        op: OpBuilder,
        output: ir.Value,
        residual: ir.Value,
        post: ir.Value,
        combine: ir.Value,
    ) -> ir.Value:
        injected = op.Mul(
            op.Unsqueeze(post, [-1]),
            op.Cast(op.Unsqueeze(output, [-2]), to=ir.DataType.FLOAT),
        )
        mixed_residual = op.MatMul(
            op.Transpose(combine, perm=[0, 1, 3, 2]),
            op.Cast(residual, to=ir.DataType.FLOAT),
        )
        return op.CastLike(
            op.Add(op.Cast(injected, to=ir.DataType.FLOAT), mixed_residual), residual
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        current_mask: ir.Value,
        past_state: tuple[ir.Value, ...],
    ) -> tuple[ir.Value, tuple[ir.Value, ...]]:
        residual = hidden_states
        post, combine, collapsed = self._hyper_connection(
            op,
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_base,
            self.hc_attn_scale,
        )
        output, present = self.self_attn(
            op,
            self.input_layernorm(op, collapsed),
            current_mask,
            past_state,
        )
        hidden_states = self._inject(op, output, residual, post, combine)

        residual = hidden_states
        post, combine, collapsed = self._hyper_connection(
            op,
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_base,
            self.hc_ffn_scale,
        )
        output = self.mlp(op, self.post_attention_layernorm(op, collapsed))
        hidden_states = self._inject(op, output, residual, post, combine)
        return hidden_states, present


class Glm5NextTextModel(nn.Module):
    """GLM-5.3 text backbone with heterogeneous KDA and pooled-DSA state."""

    def __init__(self, config: Glm5NextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [
                Glm5NextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value | None,
        attention_mask: ir.Value,
        past_key_values: list[tuple[ir.Value, ...]],
        position_ids: ir.Value | None = None,
        *,
        inputs_embeds: ir.Value | None = None,
    ) -> tuple[ir.Value, list[tuple[ir.Value, ...]]]:
        del position_ids  # Accepted for the Transformers API; GLM-5.3 is strictly NoPE.
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("GLM-5.3 requires input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(op, input_ids)
        hidden_states = op.Expand(
            op.Unsqueeze(inputs_embeds, [2]),
            op.Concat(
                op.Shape(inputs_embeds, start=0, end=2),
                op.Constant(value_ints=[self.config.hc_mult, self.config.hidden_size]),
                axis=0,
            ),
        )
        current_length = op.Shape(inputs_embeds, start=1, end=2)
        total_length = op.Shape(attention_mask, start=1, end=2)
        current_mask = op.Not(
            op.Equal(
                op.Slice(
                    attention_mask,
                    op.Neg(current_length),
                    total_length,
                    op.Constant(value_ints=[1]),
                ),
                0,
            )
        )

        presents = []
        for layer, past_state in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                op,
                hidden_states,
                current_mask,
                past_state,
            )
            presents.append(present)
        hidden_states = op.CastLike(
            op.ReduceMean(
                op.Cast(hidden_states, to=ir.DataType.FLOAT),
                [2],
                keepdims=False,
            ),
            hidden_states,
        )
        return self.norm(op, hidden_states), presents


class Glm5NextDecoderModel(nn.Module):
    """GLM-5.3 decoder component consuming pre-computed multimodal embeddings."""

    def __init__(self, config: Glm5NextConfig) -> None:
        super().__init__()
        self.model = Glm5NextTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        past_key_values: list[tuple[ir.Value, ...]],
        position_ids: ir.Value | None = None,
    ) -> tuple[ir.Value, list[tuple[ir.Value, ...]]]:
        hidden_states, presents = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        return self.lm_head(op, hidden_states), presents


class Glm5NextVisionEncoderModel(nn.Module):
    """Packed GLM-5.3 image/video encoder component."""

    def __init__(self, config: Glm5NextConfig) -> None:
        super().__init__()
        vision = config.vision
        if vision is None:
            raise ValueError("GLM-5.3 requires vision_config")
        required = (
            vision.hidden_size,
            vision.intermediate_size,
            vision.num_hidden_layers,
            vision.num_attention_heads,
            vision.patch_size,
            vision.out_hidden_size,
            vision.projector_intermediate_size,
        )
        if any(value is None for value in required):
            raise ValueError("GLM-5.3 vision dimensions must be complete")
        assert vision.hidden_size is not None
        assert vision.intermediate_size is not None
        assert vision.num_hidden_layers is not None
        assert vision.num_attention_heads is not None
        assert vision.patch_size is not None
        assert vision.out_hidden_size is not None
        assert vision.projector_intermediate_size is not None
        self.visual = Glm5NextVisionModel(
            depth=vision.num_hidden_layers,
            hidden_size=vision.hidden_size,
            intermediate_size=vision.intermediate_size,
            num_heads=vision.num_attention_heads,
            patch_size=vision.patch_size,
            temporal_patch_size=vision.temporal_patch_size,
            in_channels=vision.in_channels,
            out_hidden_size=vision.out_hidden_size,
            spatial_merge_size=vision.spatial_merge_size,
            norm_eps=vision.norm_eps,
            projector_intermediate_size=vision.projector_intermediate_size,
            swiglu_limit=vision.swiglu_limit,
        )
        if config.dtype == ir.DataType.BFLOAT16:
            # ORT CUDA does not provide a complete BF16 kernel set for this
            # dynamic packed ViT. Preserve semantic output by explicitly
            # computing the vision stage in float32; the BF16 text stages still
            # retain the checkpoint's reduced precision.
            for parameter in self.visual.parameters():
                parameter._keep_float32 = True

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        grid_thw: ir.Value,
    ) -> ir.Value:
        pixels = op.CastLike(pixel_values, self.visual.patch_embed.weight)
        return self.visual(op, pixels, grid_thw)


class Glm5NextEmbeddingModel(nn.Module):
    """Token embedding and shared-placeholder image/video feature mixer."""

    def __init__(self, config: Glm5NextConfig) -> None:
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        if config.image_token_id is None or config.video_start_token_id is None:
            raise ValueError("GLM-5.3 requires image and video boundary token IDs")
        if config.video_end_token_id is None:
            raise ValueError("GLM-5.3 requires video_end_token_id")
        self._image_token_id = config.image_token_id
        self._video_start_token_id = config.video_start_token_id
        self._video_end_token_id = config.video_end_token_id

    @staticmethod
    def _scatter(
        op: OpBuilder,
        inputs_embeds: ir.Value,
        features: ir.Value,
        mask: ir.Value,
    ) -> ir.Value:
        flat_mask = op.Reshape(op.Cast(mask, to=ir.DataType.INT64), [-1])
        indices = op.Reshape(
            op.Clip(
                op.Sub(
                    op.CumSum(flat_mask, op.Constant(value_int=0)),
                    1,
                ),
                0,
            ),
            op.Shape(mask),
        )
        pad = op.Expand(
            op.CastLike(0.0, features),
            op.Concat(
                op.Constant(value_ints=[1]),
                op.Shape(features, start=1, end=2),
                axis=0,
            ),
        )
        gathered = op.Gather(op.Concat(features, pad, axis=0), indices, axis=0)
        return op.Where(
            op.Unsqueeze(mask, [-1]),
            op.CastLike(gathered, inputs_embeds),
            inputs_embeds,
        )

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
        video_features: ir.Value,
    ) -> ir.Value:
        inputs_embeds = self.embed_tokens(op, input_ids)
        image_tokens = op.Equal(input_ids, self._image_token_id)
        video_depth = op.Greater(
            op.CumSum(
                op.Cast(op.Equal(input_ids, self._video_start_token_id), to=ir.DataType.INT64),
                op.Constant(value_int=1),
            ),
            op.CumSum(
                op.Cast(op.Equal(input_ids, self._video_end_token_id), to=ir.DataType.INT64),
                op.Constant(value_int=1),
            ),
        )
        video_mask = op.And(image_tokens, video_depth)
        image_mask = op.And(image_tokens, op.Not(video_depth))
        inputs_embeds = self._scatter(op, inputs_embeds, image_features, image_mask)
        return self._scatter(op, inputs_embeds, video_features, video_mask)


def _pack_expert_weights(
    state_dict: dict[str, torch.Tensor],
    num_experts: int,
) -> dict[str, torch.Tensor]:
    result = dict(state_dict)
    layer_prefixes = {
        key.split(".experts.", 1)[0]
        for key in state_dict
        if ".mlp.experts." in key and ".experts." in key
    }
    for layer_prefix in layer_prefixes:
        packed_gate = f"{layer_prefix}.experts.gate_up_proj"
        packed_down = f"{layer_prefix}.experts.down_proj"
        if packed_gate not in result:
            gate_keys = [
                f"{layer_prefix}.experts.{index}.gate_proj.weight"
                for index in range(num_experts)
            ]
            up_keys = [
                f"{layer_prefix}.experts.{index}.up_proj.weight"
                for index in range(num_experts)
            ]
            if all(key in result for key in (*gate_keys, *up_keys)):
                result[packed_gate] = torch.stack(
                    [
                        torch.cat((result.pop(gate), result.pop(up)), dim=0)
                        for gate, up in zip(gate_keys, up_keys)
                    ]
                )
        if packed_down not in result:
            down_keys = [
                f"{layer_prefix}.experts.{index}.down_proj.weight"
                for index in range(num_experts)
            ]
            if all(key in result for key in down_keys):
                result[packed_down] = torch.stack([result.pop(key) for key in down_keys])
    return result


def _preprocess_text_weights(
    state_dict: dict[str, torch.Tensor],
    config: Glm5NextConfig,
) -> dict[str, torch.Tensor]:
    stripped: dict[str, torch.Tensor] = {}
    dropped_mtp = 0
    for source_name, value in state_dict.items():
        name = source_name
        if name.startswith("model.visual."):
            continue
        if name.startswith("model.language_model."):
            name = f"model.{name[len('model.language_model.') :]}"
        match = _LAYER_RE.match(name)
        if match is not None and int(match.group(1)) >= config.num_hidden_layers:
            dropped_mtp += 1
            continue
        for site, checkpoint_prefix in (
            ("attn_hc", "hc_attn"),
            ("ffn_hc", "hc_ffn"),
        ):
            for field in ("fn", "base", "scale"):
                name = name.replace(
                    f".{site}.{field}",
                    f".{checkpoint_prefix}_{field}",
                )
        name = name.replace(".self_attn.forget_gate.", ".self_attn.")
        if name.endswith(".self_attn.conv1d.weight"):
            q_weight, k_weight, v_weight = value.chunk(3, dim=0)
            prefix = name[: -len("conv1d.weight")]
            stripped[f"{prefix}q_conv1d.weight"] = q_weight
            stripped[f"{prefix}k_conv1d.weight"] = k_weight
            stripped[f"{prefix}v_conv1d.weight"] = v_weight
            continue
        stripped[name] = value
    if dropped_mtp:
        logger.warning(
            "Dropping %d GLM-5.3 MTP tensor(s): Transformers 5.16 exposes no "
            "authoritative MTP forward or cache ABI.",
            dropped_mtp,
        )
    return _pack_expert_weights(stripped, config.num_local_experts or 0)


class Glm5NextCausalLMModel(nn.Module):
    """Text-only GLM-5.3-Flash decoder using the exact hybrid state ABI."""

    default_task: str = "glm5-next-text-generation"
    category: str = "Mixture of Experts"
    config_class: type = Glm5NextConfig

    def __init__(self, config: Glm5NextConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Glm5NextTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        past_key_values: list[tuple[ir.Value, ...]],
        position_ids: ir.Value | None = None,
    ) -> tuple[ir.Value, list[tuple[ir.Value, ...]]]:
        hidden_states, presents = self.model(
            op,
            input_ids,
            attention_mask,
            past_key_values,
            position_ids,
        )
        return self.lm_head(op, hidden_states), presents

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        parameter_names = {name for name, _ in self.named_parameters()}
        if set(state_dict) <= parameter_names:
            return dict(state_dict)
        return _preprocess_text_weights(state_dict, self.config)


class Glm5NextForConditionalGeneration(nn.Module):
    """GLM-5.3-Flash image/video-language model with a three-model ONNX split."""

    default_task: str = "glm5-next-vision-language"
    category: str = "Multimodal"
    config_class: type = Glm5NextConfig
    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "decoder": (
            "model.language_model.layers",
            "model.language_model.norm",
            "lm_head",
        ),
        "vision_encoder": ("model.visual",),
        "embedding": ("model.language_model.embed_tokens",),
    }

    def __init__(self, config: Glm5NextConfig) -> None:
        super().__init__()
        self.config = config
        self.decoder = Glm5NextDecoderModel(config)
        self.vision_encoder = Glm5NextVisionEncoderModel(config)
        self.embedding = Glm5NextEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "Glm5NextForConditionalGeneration is exported as decoder, "
            "vision_encoder, and embedding graphs."
        )

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        parameter_names = {name for name, _ in self.named_parameters()}
        if set(state_dict) <= parameter_names:
            return dict(state_dict)

        text = _preprocess_text_weights(state_dict, self.config)
        routed: dict[str, torch.Tensor] = {}
        for name, value in text.items():
            if name == "model.embed_tokens.weight":
                routed["embedding.embed_tokens.weight"] = value
            elif name.startswith(("model.", "lm_head.")):
                routed[f"decoder.{name}"] = value
        for name, value in state_dict.items():
            if name.startswith("model.visual."):
                routed[f"vision_encoder.visual.{name[len('model.visual.') :]}"] = value
        return routed
