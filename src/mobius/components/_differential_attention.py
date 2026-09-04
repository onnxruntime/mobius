# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Portable grouped-query differential attention."""

from __future__ import annotations

import math
from collections.abc import Callable

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._common import create_padding_mask, create_sliding_window_mask


class _DifferentialRMSNorm(nn.Module):
    """RMS normalization with source-compatible fp32 variance accumulation."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self._eps = eps

    def forward(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        value_f32 = op.Cast(value, to=ir.DataType.FLOAT)
        variance = op.ReduceMean(
            op.Mul(value_f32, value_f32),
            axes=[-1],
            keepdims=True,
        )
        normalized = op.Mul(
            value_f32,
            op.Reciprocal(op.Sqrt(op.Add(variance, self._eps))),
        )
        normalized = op.CastLike(normalized, value)
        return op.Mul(normalized, op.CastLike(self.weight, normalized))


class DifferentialGQAAttention(nn.Module):
    """Apply differential GQA to pre-projected query, key, and value tensors.

    Query heads and KV heads are striped into two equal groups. Each query
    stripe reads both value stripes, then the second result is subtracted using
    a learned lambda before 2*head_dim RMS normalization. This is the
    differential-attention equation used by architectures that store one fused
    ``[Q | K | V]`` projection externally to the attention primitive.
    """

    def __init__(
        self,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        depth: int,
        eps: float = 1e-5,
        local_window_size: int | None = None,
    ):
        super().__init__()
        if num_attention_heads % 2 or num_key_value_heads % 2:
            raise ValueError("Differential GQA requires an even query and KV head count")
        if num_attention_heads % num_key_value_heads:
            raise ValueError("Differential GQA requires query heads divisible by KV heads")
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self._query_pairs = num_attention_heads // 2
        self._kv_pairs = num_key_value_heads // 2
        self._scale = head_dim**-0.5
        self._lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self._local_window_size = local_window_size
        # The four vectors intentionally remain float parameters. The
        # reference takes their dot products in fp32 before returning to the
        # attention activation dtype.
        self.lambda_q1 = nn.Parameter([head_dim])
        self.lambda_k1 = nn.Parameter([head_dim])
        self.lambda_q2 = nn.Parameter([head_dim])
        self.lambda_k2 = nn.Parameter([head_dim])
        self.lambda_q1._keep_float32 = True
        self.lambda_k1._keep_float32 = True
        self.lambda_q2._keep_float32 = True
        self.lambda_k2._keep_float32 = True
        self.subln = _DifferentialRMSNorm(2 * head_dim, eps)

    def _split_stripes(
        self,
        op: OpBuilder,
        value: ir.Value,
        *,
        pairs: int,
    ) -> tuple[ir.Value, ir.Value]:
        """Map ``[B, T, 2*pairs, D]`` to its alternating head stripes."""
        striped = op.Reshape(value, [0, 0, pairs, 2, self.head_dim])
        return (
            op.Gather(striped, op.Constant(value_int=0), axis=3),
            op.Gather(striped, op.Constant(value_int=1), axis=3),
        )

    @staticmethod
    def _flatten_heads(op: OpBuilder, value: ir.Value) -> ir.Value:
        return op.Reshape(value, [0, 0, -1])

    def _attend(
        self,
        op: OpBuilder,
        query: ir.Value,
        key: ir.Value,
        value: ir.Value,
        attention_mask: ir.Value,
        past_key: ir.Value | None,
        past_value: ir.Value | None,
    ) -> ir.Value:
        """Run one causal/padded GQA branch without materializing a Q-by-K mask."""
        # The source's FlashAttention unpads left-padded batches. ORT GQA's
        # per-batch sequence lengths instead describe valid *prefixes*, so it
        # can only be used directly when every input token is valid. Preserve
        # that compact path for long unpadded prompts and select standard ONNX
        # Attention with an explicit mask for source-faithful padded batches.
        no_padding = op.Equal(
            op.ReduceMin(attention_mask, keepdims=False),
            op.Constant(value_int=1),
        )
        branch_scope = f"{query.name}_{key.name}_{value.name}"
        unpadded_branch = op.builder.subgraph(
            lambda branch_op: self._scoped_attention(
                branch_op,
                f"{branch_scope}.unpadded",
                (
                    lambda: (
                        self._gqa_attention(
                            branch_op, query, key, value, attention_mask, past_key, past_value
                        )
                        if self._local_window_size is not None
                        else self._standard_attention(
                            branch_op, query, key, value, None, past_key, past_value
                        )
                    )
                ),
            ),
            inputs=[],
            outputs=[ir.Value(name=f"{branch_scope}.unpadded_attention")],
            name="phi4flash_unpadded_attention",
        )
        padded_branch = op.builder.subgraph(
            lambda branch_op: self._scoped_attention(
                branch_op,
                f"{branch_scope}.padded",
                lambda: self._standard_attention(
                    branch_op,
                    query,
                    key,
                    value,
                    (
                        create_sliding_window_mask(
                            branch_op, query, attention_mask, self._local_window_size
                        )
                        if self._local_window_size is not None
                        else create_padding_mask(branch_op, query, attention_mask)
                    ),
                    past_key,
                    past_value,
                ),
            ),
            inputs=[],
            outputs=[ir.Value(name=f"{branch_scope}.padded_attention")],
            name="phi4flash_padded_attention",
        )
        return op.If(no_padding, then_branch=unpadded_branch, else_branch=padded_branch)

    @staticmethod
    def _scoped_attention(
        op: OpBuilder,
        scope: str,
        build: Callable[[], ir.Value],
    ) -> ir.Value:
        """Keep generated values in sibling If bodies globally SSA-safe."""
        op.builder.push_module(scope)
        try:
            return build()
        finally:
            op.builder.pop_module()

    def _gqa_attention(
        self,
        op: OpBuilder,
        query: ir.Value,
        key: ir.Value,
        value: ir.Value,
        attention_mask: ir.Value,
        past_key: ir.Value | None,
        past_value: ir.Value | None,
    ) -> ir.Value:
        """Apply source-windowed GQA when no padding needs Flash-style unpadding."""
        seqlens_k = op.Cast(
            op.Sub(
                op.ReduceSum(attention_mask, axes=[1], keepdims=False),
                op.Constant(value_int=1),
            ),
            to=ir.DataType.INT32,
        )
        total_sequence_length = op.Cast(
            op.Gather(op.Shape(attention_mask), 1),
            to=ir.DataType.INT32,
        )
        output, _, _ = op.GroupQueryAttention(
            self._flatten_heads(op, query),
            self._flatten_heads(op, key),
            self._flatten_heads(op, value),
            (
                op.Transpose(past_key, perm=[0, 2, 1, 3])
                if past_key is not None
                else None
            ),
            (
                op.Transpose(past_value, perm=[0, 2, 1, 3])
                if past_value is not None
                else None
            ),
            seqlens_k,
            total_sequence_length,
            None,
            None,
            num_heads=self._query_pairs,
            kv_num_heads=self._kv_pairs,
            scale=self._scale,
            do_rotary=0,
            local_window_size=self._local_window_size,
            _domain="com.microsoft",
            _outputs=3,
        )
        return output

    def _standard_attention(
        self,
        op: OpBuilder,
        query: ir.Value,
        key: ir.Value,
        value: ir.Value,
        attention_mask: ir.Value | None,
        past_key: ir.Value | None,
        past_value: ir.Value | None,
    ) -> ir.Value:
        """Apply causal ONNX Attention with a source-faithful padded-batch mask."""
        output, _, _ = op.Attention(
            self._flatten_heads(op, query),
            self._flatten_heads(op, key),
            self._flatten_heads(op, value),
            attention_mask,
            op.Transpose(past_key, perm=[0, 2, 1, 3]) if past_key is not None else None,
            op.Transpose(past_value, perm=[0, 2, 1, 3]) if past_value is not None else None,
            q_num_heads=self._query_pairs,
            kv_num_heads=self._kv_pairs,
            scale=self._scale,
            is_causal=1,
            _outputs=3,
        )
        return output

    def forward(
        self,
        op: OpBuilder,
        query: ir.Value,
        key: ir.Value,
        value: ir.Value,
        attention_mask: ir.Value,
        past_key_value: tuple[ir.Value, ir.Value] | None = None,
    ) -> ir.Value:
        """Return ``[B, T_q, num_attention_heads, head_dim]`` differential output."""
        query_1, query_2 = self._split_stripes(op, query, pairs=self._query_pairs)
        key_1, key_2 = self._split_stripes(op, key, pairs=self._kv_pairs)
        value_1, value_2 = self._split_stripes(op, value, pairs=self._kv_pairs)
        if past_key_value is None:
            past_key_1 = past_key_2 = past_value_1 = past_value_2 = None
        else:
            past_key_1, past_key_2 = self._split_stripes(
                op, past_key_value[0], pairs=self._kv_pairs
            )
            past_value_1, past_value_2 = self._split_stripes(
                op, past_key_value[1], pairs=self._kv_pairs
            )

        # A1/A2 concatenate the two value stripes after independent GQA reads.
        attn_1 = op.Concat(
            self._attend(op, query_1, key_1, value_1, attention_mask, past_key_1, past_value_1),
            self._attend(op, query_1, key_1, value_2, attention_mask, past_key_1, past_value_2),
            axis=-1,
        )
        attn_1 = op.Reshape(
            attn_1,
            [0, 0, self._query_pairs, 2 * self.head_dim],
        )
        attn_2 = op.Concat(
            self._attend(op, query_2, key_2, value_1, attention_mask, past_key_2, past_value_1),
            self._attend(op, query_2, key_2, value_2, attention_mask, past_key_2, past_value_2),
            axis=-1,
        )
        attn_2 = op.Reshape(
            attn_2,
            [0, 0, self._query_pairs, 2 * self.head_dim],
        )

        lambda_1 = op.CastLike(
            op.Exp(
                op.ReduceSum(
                    op.Mul(
                        op.Cast(self.lambda_q1, to=ir.DataType.FLOAT),
                        op.Cast(self.lambda_k1, to=ir.DataType.FLOAT),
                    ),
                    axes=[0],
                    keepdims=False,
                )
            ),
            query,
        )
        lambda_2 = op.CastLike(
            op.Exp(
                op.ReduceSum(
                    op.Mul(
                        op.Cast(self.lambda_q2, to=ir.DataType.FLOAT),
                        op.Cast(self.lambda_k2, to=ir.DataType.FLOAT),
                    ),
                    axes=[0],
                    keepdims=False,
                )
            ),
            query,
        )
        lambda_full = op.Add(
            op.Sub(lambda_1, lambda_2),
            op.CastLike(op.Constant(value_float=self._lambda_init), query),
        )
        result = self.subln(
            op,
            op.Sub(attn_1, op.Mul(lambda_full, attn_2)),
        )
        result = op.Mul(result, 1.0 - self._lambda_init)

        # [B, T, query_pairs, 2*D] -> [B, T, query_heads, D].
        result = op.Reshape(
            result,
            [0, 0, self._query_pairs, 2, self.head_dim],
        )
        return op.Reshape(result, [0, 0, self.num_attention_heads, self.head_dim])
