# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dense, separate-K/V PagedAttention for packed GQA decoders."""

from __future__ import annotations

from dataclasses import dataclass

import onnx_ir as ir
from onnxscript import OpBuilder

PAGED_BLOCK_SIZE = 256


@dataclass(frozen=True)
class DensePagedState:
    """Caller-owned page pools and scheduler inputs for one layer.

    attention_metadata holds host-side INT32 replay bounds [max_query_length,
    max_kv_length, optional max_kv_len_lower_bound]. The first two values
    bound all sequences from above; the optional third bounds their maximum
    KV length from below across graph-capture replays.
    """

    key_cache: ir.Value
    value_cache: ir.Value
    cumulative_sequence_lengths: ir.Value
    past_sequence_lengths: ir.Value
    block_table: ir.Value
    attention_metadata: ir.Value
    cos_cache: ir.Value | None = None
    sin_cache: ir.Value | None = None


def paged_dense_attention(
    op: OpBuilder,
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    state: DensePagedState,
    *,
    num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    scale: float,
    rotary_interleaved: bool,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """Emit v1 SEPARATE, deriving write slots from the scheduler's block table."""
    output, key_cache_out, value_cache_out = op.PagedAttention(
        op.Reshape(query, [-1, num_heads * head_dim]),
        op.Reshape(key, [-1, kv_num_heads * head_dim]),
        op.Reshape(value, [-1, kv_num_heads * head_dim]),
        state.key_cache,
        state.value_cache,
        state.cumulative_sequence_lengths,
        state.past_sequence_lengths,
        state.block_table,
        state.cos_cache,
        state.sin_cache,
        None,  # slot_mapping: operator derives slots
        None,  # head_sink
        None,  # q_norm_weight: explicit Qwen3 RMSNorm precedes RoPE
        None,  # k_norm_weight
        None,  # k_scale
        None,  # v_scale
        state.attention_metadata,
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        scale=scale,
        do_rotary=1,
        rotary_interleaved=int(rotary_interleaved),
        _domain="com.microsoft",
        _outputs=3,
    )
    output.shape = ir.Shape(["num_tokens", num_heads * head_dim])
    output.type = state.key_cache.type
    return output, key_cache_out, value_cache_out
