# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dense, separate-K/V PagedAttention for text-only Qwen GQA decoders."""

from __future__ import annotations

import math
from dataclasses import dataclass

import onnx_ir as ir
from onnxscript import OpBuilder

from mobius._configs import ArchitectureConfig

PAGED_BLOCK_SIZE = 256
DENSE_PAGED_MODEL_TYPES = frozenset({"qwen2", "qwen3"})


def dense_paged_rejection(config: ArchitectureConfig) -> str | None:
    """Return the reason this config cannot use the dense packed-token ABI."""
    if config.model_type not in DENSE_PAGED_MODEL_TYPES:
        return "dense PagedAttention supports only text-only qwen2/Qwen2.5 and qwen3."
    if config.dtype != ir.DataType.FLOAT16:
        return (
            "dense PagedAttention requires float16; bfloat16 is disabled pending "
            "full-logit numerical parity validation."
        )
    if (
        config.num_attention_heads <= 0
        or config.num_key_value_heads <= 0
        or config.num_attention_heads % config.num_key_value_heads
        or config.head_dim <= 0
        or config.head_dim % 16
    ):
        return "dense PagedAttention requires positive GQA heads and head_dim divisible by 16."
    if (
        not math.isclose(config.partial_rotary_factor or 0, 1.0)
        or config.rope_type != "default"
        or (config.qk_rope_head_dim and config.qk_rope_head_dim != config.head_dim)
    ):
        return "dense PagedAttention requires full standard RoPE."
    if config.rope_scaling and (
        config.rope_scaling.get("rope_type", "default") != "default"
        or set(config.rope_scaling) - {"rope_type", "rope_theta"}
    ):
        return "dense PagedAttention does not support scaled RoPE."
    if getattr(config, "mrope_section", None):
        return "dense PagedAttention requires 1D text RoPE, not multimodal RoPE."
    layer_types = getattr(config, "layer_types", None)
    if getattr(config, "sliding_window", None) or (
        layer_types is not None
        and (
            len(layer_types) != config.num_hidden_layers
            or any(layer != "full_attention" for layer in layer_types)
        )
    ):
        return "dense PagedAttention does not support hybrid or sliding-window attention."
    if getattr(config, "attn_logit_softcapping", None):
        return "dense PagedAttention does not support attention softcapping."
    if getattr(config, "output_layer_indices", None) or getattr(
        config, "output_final_hidden_state", False
    ):
        return "dense PagedAttention does not support auxiliary hidden-state outputs."
    if getattr(config, "num_nextn_predict_layers", 0):
        return "dense PagedAttention does not support Multi-Token Prediction."
    if config.model_type == "qwen2" and config.attn_qk_norm:
        return "Qwen2 dense PagedAttention does not support Q/K normalization."
    if config.model_type == "qwen3" and (not config.attn_qk_norm or config.attn_qk_norm_full):
        return "Qwen3 dense PagedAttention requires per-head Q/K RMSNorm."
    return None


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
