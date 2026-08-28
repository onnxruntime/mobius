# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact tensor contracts for Grok, GroveMoE, and Hunyuan-MoE GGUF graphs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from mobius.integrations.gguf._tensor_mapping import is_known_skip

_ARCHITECTURES = frozenset({"grok", "grovemoe", "hunyuan-moe"})


def _positive_int(metadata: Mapping[str, Any], key: str) -> int:
    value = int(metadata[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _vocab_size(metadata: Mapping[str, Any], arch: str) -> int:
    value = int(metadata.get(f"{arch}.vocab_size", 0))
    if not value:
        tokens = metadata.get("tokenizer.ggml.tokens")
        value = len(tokens) if isinstance(tokens, (list, tuple)) else 0
    if value <= 0:
        raise ValueError(f"{arch} GGUF must declare a positive vocabulary size")
    return value


def _attention_tensors(
    actual: dict[str, tuple[int, ...]],
    *,
    prefix: str,
    hidden: int,
    query_width: int,
    kv_width: int,
) -> tuple[dict[str, tuple[int, ...]], bool]:
    fused_weight = prefix + "attn_qkv.weight"
    split_weights: dict[str, tuple[int, ...]] = {
        prefix + "attn_q.weight": (query_width, hidden),
        prefix + "attn_k.weight": (kv_width, hidden),
        prefix + "attn_v.weight": (kv_width, hidden),
    }
    has_fused = fused_weight in actual
    has_split = bool(set(split_weights) & set(actual))
    if has_fused == has_split:
        raise ValueError(
            f"{prefix[:-1]} must contain exactly one fused or split Q/K/V weight layout"
        )

    required: dict[str, tuple[int, ...]]
    if has_fused:
        required = {fused_weight: (query_width + 2 * kv_width, hidden)}
        fused_bias = prefix + "attn_qkv.bias"
        has_bias = fused_bias in actual
        if has_bias:
            required[fused_bias] = (query_width + 2 * kv_width,)
        forbidden_biases = {
            prefix + f"attn_{projection}.bias" for projection in ("q", "k", "v")
        }
        if forbidden_biases & set(actual):
            raise ValueError(f"{prefix[:-1]} mixes fused weights with split Q/K/V biases")
    else:
        required = split_weights
        split_biases = {
            prefix + "attn_q.bias": (query_width,),
            prefix + "attn_k.bias": (kv_width,),
            prefix + "attn_v.bias": (kv_width,),
        }
        present_biases = set(split_biases) & set(actual)
        if present_biases and present_biases != set(split_biases):
            raise ValueError(f"{prefix[:-1]} has a partial split Q/K/V bias family")
        has_bias = bool(present_biases)
        if has_bias:
            required.update(split_biases)
        if prefix + "attn_qkv.bias" in actual:
            raise ValueError(f"{prefix[:-1]} has a fused QKV bias without fused weights")
    return required, has_bias


def _validate_shapes(
    architecture: str,
    actual: dict[str, tuple[int, ...]],
    required: dict[str, tuple[int, ...]],
    optional: dict[str, tuple[int, ...]],
) -> None:
    allowed = set(required) | set(optional)
    missing = sorted(set(required) - set(actual))
    unexpected = sorted(set(actual) - allowed)
    malformed = {
        name: (required.get(name, optional.get(name)), actual[name])
        for name in allowed & set(actual)
        if actual[name] != required.get(name, optional.get(name))
    }
    if missing or unexpected or malformed:
        raise ValueError(
            f"Invalid {architecture} GGUF tensor closure: missing={missing}, "
            f"unexpected={unexpected}, malformed={malformed}"
        )


def _require_uniform(label: str, values: Iterable[bool]) -> None:
    values = tuple(values)
    if values and any(value != values[0] for value in values[1:]):
        raise ValueError(f"{label} must be uniform across all layers")


def validate_remaining_moe_tensor_contract(gguf_model) -> None:
    """Validate metadata, conditional tensor families, and exact tensor shapes."""
    architecture = gguf_model.architecture
    if architecture not in _ARCHITECTURES:
        return

    metadata = gguf_model.metadata
    required_suffixes = {
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "block_count",
        "attention.head_count",
        "attention.layer_norm_rms_epsilon",
        "expert_count",
        "expert_used_count",
    }
    if architecture in {"grovemoe", "hunyuan-moe"}:
        required_suffixes.add("expert_feed_forward_length")
    if architecture == "grovemoe":
        required_suffixes.update({"expert_group_scale", "experts_per_group"})
    missing_metadata = sorted(
        f"{architecture}.{suffix}"
        for suffix in required_suffixes
        if f"{architecture}.{suffix}" not in metadata
    )
    if missing_metadata:
        raise ValueError(
            f"{architecture} GGUF is missing required MoE metadata: {missing_metadata}"
        )

    hidden = _positive_int(metadata, f"{architecture}.embedding_length")
    dense_width = _positive_int(metadata, f"{architecture}.feed_forward_length")
    layers = _positive_int(metadata, f"{architecture}.block_count")
    heads = _positive_int(metadata, f"{architecture}.attention.head_count")
    kv_heads = int(metadata.get(f"{architecture}.attention.head_count_kv", heads))
    experts = _positive_int(metadata, f"{architecture}.expert_count")
    top_k = _positive_int(metadata, f"{architecture}.expert_used_count")
    if kv_heads <= 0 or heads % kv_heads or top_k > experts:
        raise ValueError(f"{architecture} GGUF has invalid attention or MoE geometry")
    head_dim = int(metadata.get(f"{architecture}.attention.key_length", hidden // heads))
    value_dim = int(metadata.get(f"{architecture}.attention.value_length", hidden // heads))
    rope_dim = int(metadata.get(f"{architecture}.rope.dimension_count", head_dim))
    if (
        min(head_dim, value_dim, rope_dim) <= 0
        or value_dim != head_dim
        or rope_dim != head_dim
    ):
        raise ValueError(
            f"{architecture} requires equal positive Q/K/V and full-RoPE head dimensions"
        )
    query_width = heads * head_dim
    kv_width = kv_heads * head_dim
    if architecture in {"grok", "hunyuan-moe"} and query_width != hidden:
        raise ValueError(
            f"{architecture} requires head_count * key_length == embedding_length"
        )
    eps = float(metadata[f"{architecture}.attention.layer_norm_rms_epsilon"])
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"{architecture} GGUF has invalid normalization epsilon")

    vocab = _vocab_size(metadata, architecture)
    actual = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
        if not is_known_skip(name)
    }
    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {"output.weight": (vocab, hidden)}
    attention_biases: list[bool] = []

    if architecture == "grok":
        expert_width = int(metadata.get("grok.expert_feed_forward_length", 0))
        if expert_width < 0:
            raise ValueError("grok.expert_feed_forward_length must be non-negative")
        expert_width = expert_width or dense_width
        dense_layers: list[bool] = []
        gated_expert_layers: list[bool] = []
        for layer in range(layers):
            prefix = f"blk.{layer}."
            attention, has_bias = _attention_tensors(
                actual,
                prefix=prefix,
                hidden=hidden,
                query_width=query_width,
                kv_width=kv_width,
            )
            required.update(attention)
            attention_biases.append(has_bias)
            required.update(
                {
                    prefix + "attn_norm.weight": (hidden,),
                    prefix + "attn_output.weight": (hidden, query_width),
                    prefix + "attn_output_norm.weight": (hidden,),
                    prefix + "ffn_norm.weight": (hidden,),
                    prefix + "ffn_gate_inp.weight": (experts, hidden),
                    prefix + "ffn_up_exps.weight": (experts, expert_width, hidden),
                    prefix + "ffn_down_exps.weight": (experts, hidden, expert_width),
                }
            )
            expert_gate = prefix + "ffn_gate_exps.weight"
            has_expert_gate = expert_gate in actual
            gated_expert_layers.append(has_expert_gate)
            if has_expert_gate:
                required[expert_gate] = (experts, expert_width, hidden)

            post_norms = {
                prefix + "layer_output_norm.weight",
                prefix + "post_ffw_norm.weight",
            }
            present_post_norms = post_norms & set(actual)
            if len(present_post_norms) != 1:
                raise ValueError(
                    f"{architecture} layer {layer} must contain exactly one post-FFN norm"
                )
            selected_post_norm = present_post_norms.pop()
            required[selected_post_norm] = (hidden,)

            dense_shapes = {
                prefix + "ffn_gate.weight": (dense_width, hidden),
                prefix + "ffn_up.weight": (dense_width, hidden),
                prefix + "ffn_down.weight": (hidden, dense_width),
            }
            present_dense = set(dense_shapes) & set(actual)
            if present_dense and present_dense != set(dense_shapes):
                raise ValueError(
                    f"{architecture} layer {layer} has a partial dense FFN tensor family"
                )
            dense_layers.append(bool(present_dense))
            if present_dense:
                required.update(dense_shapes)
        _require_uniform("grok attention projection biases", attention_biases)
        _require_uniform("grok dense FFN topology", dense_layers)
        _require_uniform("grok expert gating topology", gated_expert_layers)
        _validate_shapes(architecture, actual, required, optional)
        return

    expert_width = _positive_int(
        metadata,
        f"{architecture}.expert_feed_forward_length",
    )
    if architecture == "hunyuan-moe" and expert_width != dense_width:
        raise ValueError(
            "hunyuan-moe.expert_feed_forward_length must equal feed_forward_length"
        )
    if architecture == "grovemoe":
        group_size = _positive_int(metadata, "grovemoe.experts_per_group")
        if experts % group_size:
            raise ValueError("grovemoe.experts_per_group must divide expert_count")
        chunk_experts = experts // group_size
        chunk_width = int(metadata.get("grovemoe.expert_chunk_feed_forward_length", 0))
        chunk_width = chunk_width or head_dim
        if chunk_width <= 0:
            raise ValueError(
                "grovemoe.expert_chunk_feed_forward_length must be positive when present"
            )
        group_scale = float(metadata["grovemoe.expert_group_scale"])
        if not math.isfinite(group_scale):
            raise ValueError("grovemoe.expert_group_scale must be finite")
    else:
        shared_count = int(metadata.get("hunyuan-moe.expert_shared_count", 1))
        if shared_count != 1:
            raise ValueError("hunyuan-moe.expert_shared_count must be one")
        shared_width = int(metadata.get("hunyuan-moe.expert_shared_feed_forward_length", 0))
        shared_width = shared_width or dense_width
        if shared_width <= 0:
            raise ValueError(
                "hunyuan-moe.expert_shared_feed_forward_length must be positive when present"
            )

    for layer in range(layers):
        prefix = f"blk.{layer}."
        attention, has_bias = _attention_tensors(
            actual,
            prefix=prefix,
            hidden=hidden,
            query_width=query_width,
            kv_width=kv_width,
        )
        required.update(attention)
        attention_biases.append(has_bias)
        required.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_q_norm.weight": (head_dim,),
                prefix + "attn_k_norm.weight": (head_dim,),
                prefix + "attn_output.weight": (hidden, query_width),
                prefix + "ffn_norm.weight": (hidden,),
                prefix + "ffn_gate_inp.weight": (experts, hidden),
                prefix + "ffn_gate_exps.weight": (experts, expert_width, hidden),
                prefix + "ffn_up_exps.weight": (experts, expert_width, hidden),
                prefix + "ffn_down_exps.weight": (experts, hidden, expert_width),
            }
        )
        if architecture == "grovemoe":
            required.update(
                {
                    prefix + "ffn_gate_chexps.weight": (
                        chunk_experts,
                        chunk_width,
                        hidden,
                    ),
                    prefix + "ffn_up_chexps.weight": (
                        chunk_experts,
                        chunk_width,
                        hidden,
                    ),
                    prefix + "ffn_down_chexps.weight": (
                        chunk_experts,
                        hidden,
                        chunk_width,
                    ),
                }
            )
        else:
            required.update(
                {
                    prefix + "ffn_gate_shexp.weight": (shared_width, hidden),
                    prefix + "ffn_up_shexp.weight": (shared_width, hidden),
                    prefix + "ffn_down_shexp.weight": (hidden, shared_width),
                }
            )

    _require_uniform(f"{architecture} attention projection biases", attention_biases)
    _validate_shapes(architecture, actual, required, optional)
