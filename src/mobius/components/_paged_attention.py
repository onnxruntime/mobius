# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model-agnostic adapters for ORT's packed hybrid serving operators."""

from __future__ import annotations

from typing import NamedTuple

import onnx_ir as ir
from onnxscript import OpBuilder

DOMAIN = "com.microsoft"
ORT_REVISION = "f38538cd5a4b5945a4c839565a8eebc65e1e2ef8"
GENAI_REVISION = "d5b40851ba80ffa8e95b6b01f921dbb9008fac80"


class PagedAttentionState(NamedTuple):
    """Per-layer SEPARATE paged KV state plus request-level packed metadata."""

    key_cache: ir.Value
    value_cache: ir.Value
    cumulative_sequence_lengths: ir.Value
    past_sequence_lengths: ir.Value
    block_table: ir.Value
    attention_metadata: ir.Value


class PagedHybridContext(NamedTuple):
    """Shared packed-request context passed through a hybrid decoder stack."""

    cumulative_sequence_lengths: ir.Value
    past_sequence_lengths: ir.Value
    block_table: ir.Value
    attention_metadata: ir.Value
    last_token_indices: ir.Value | None


def paged_attention(
    op: OpBuilder,
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    state: PagedAttentionState,
    *,
    num_heads: int,
    kv_num_heads: int,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """Emit the pinned 17-input SEPARATE PagedAttention ABI."""
    output, key_cache, value_cache = op.PagedAttention(
        query,
        key,
        value,
        state.key_cache,
        state.value_cache,
        state.cumulative_sequence_lengths,
        state.past_sequence_lengths,
        state.block_table,
        None,  # cos cache: RoPE is applied externally after Q/K normalization
        None,  # sin cache
        None,  # slot mapping: the GenAI page manager derives packed writes
        None,  # attention sinks
        None,  # q norm: external OffsetRMSNorm must not be repeated
        None,  # k norm
        None,  # k scale
        None,  # v scale
        state.attention_metadata,
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        kv_cache_layout="SEPARATE",
        do_rotary=0,
        _domain=DOMAIN,
        _outputs=3,
    )
    # Generic ONNX shape inference does not know these pinned contrib schemas.
    output.type, output.shape = query.type, query.shape
    key_cache.type, key_cache.shape = state.key_cache.type, state.key_cache.shape
    value_cache.type, value_cache.shape = state.value_cache.type, state.value_cache.shape
    return output, key_cache, value_cache


def varlen_causal_conv_with_state(
    op: OpBuilder,
    packed: ir.Value,
    weight: ir.Value,
    cumulative_sequence_lengths: ir.Value,
    bias: ir.Value,
    past_conv: ir.Value,
) -> tuple[ir.Value, ir.Value]:
    """Emit packed depthwise convolution with fixed per-sequence carry state."""
    output, present = op.VarlenCausalConvWithState(
        packed,
        weight,
        cumulative_sequence_lengths,
        bias,
        past_conv,
        activation="silu",
        _domain=DOMAIN,
        _outputs=2,
    )
    output.type, output.shape = packed.type, packed.shape
    present.type, present.shape = past_conv.type, past_conv.shape
    return output, present


def gated_delta_net(
    op: OpBuilder,
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    cumulative_sequence_lengths: ir.Value,
    raw_a: ir.Value,
    raw_b: ir.Value,
    past_recurrent: ir.Value,
    a_log: ir.Value,
    dt_bias: ir.Value,
) -> tuple[ir.Value, ir.Value]:
    """Emit the native packed Qwen GatedDeltaNet recurrence."""
    output, present = op.GatedDeltaNet(
        query,
        key,
        value,
        cumulative_sequence_lengths,
        raw_a,
        raw_b,
        past_recurrent,
        a_log,
        dt_bias,
        gate_activation="qwen",
        beta_activation="sigmoid",
        qk_l2_norm=1,
        update_rule="gated_delta",
        scale=0.0,
        _domain=DOMAIN,
        _outputs=2,
    )
    output.type, output.shape = value.type, value.shape
    present.type, present.shape = past_recurrent.type, past_recurrent.shape
    return output, present
