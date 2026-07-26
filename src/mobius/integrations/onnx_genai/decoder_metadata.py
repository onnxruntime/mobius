# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Emit onnx-genai ``inference_metadata`` for decoder-only language models.

onnx-genai's runtime is driven by a declared ``inference_metadata`` document
(onnx/onnx#8184) rather than a runtime-specific config. For an autoregressive
LLM the relevant sections are ``model.attention`` (head counts, head dim,
sliding window) and ``kv_cache`` (native dtype). This module maps a Mobius
``BaseModelConfig``/``ArchitectureConfig`` (or explicit values) onto that
document so a Mobius-built LLM is directly loadable by onnx-genai.

Reads only plain config fields (no torch state); cheap to unit-test.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

# Mobius' unset-int sentinel.
_UNSET = -42

_FLOAT_DTYPE_ALIASES = {
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
}


def _clean_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _canonical_float_dtype(value: Any) -> str | None:
    """Return the canonical metadata spelling for a floating-point dtype."""
    if value is None:
        return None
    name = getattr(value, "name", None)
    token = str(name if name is not None else value).strip().lower().rsplit(".", 1)[-1]
    return _FLOAT_DTYPE_ALIASES.get(token)


def _infer_kv_native_dtype(config: Any) -> str | None:
    """Infer KV storage dtype from the model's activation/compute dtype."""
    for name in ("activation_dtype", "compute_dtype", "dtype", "torch_dtype"):
        dtype = _canonical_float_dtype(getattr(config, name, None))
        if dtype is not None:
            return dtype
    return None


def build_decoder_metadata(
    *,
    num_attention_heads: int,
    head_dim: int,
    num_kv_heads: int | None = None,
    max_sequence_length: int | None = None,
    kv_native_dtype: str | None = None,
    attention_type: str | None = None,
    sliding_window: int | None = None,
    sink_tokens: int | None = None,
    architecture: str | None = None,
    mixture_of_experts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the onnx-genai ``inference_metadata`` dict for a decoder LLM.

    Args:
        num_attention_heads: Number of query/attention heads.
        head_dim: Per-head hidden dimension.
        num_kv_heads: Number of key/value heads (defaults to
            ``num_attention_heads`` = multi-head; a smaller value = GQA).
        max_sequence_length: Maximum total sequence length in tokens.
        kv_native_dtype: KV-cache storage dtype (e.g. ``"float16"``,
            ``"bfloat16"``).
        attention_type: Override the derived attention type (``multi_head`` /
            ``grouped_query_attention``).
        sliding_window: Sliding-window length in tokens (None = full context).
        sink_tokens: Attention-sink tokens retained with the sliding window.
        architecture: Optional architecture hint (e.g. ``"llama"``).
        mixture_of_experts: Explicit, architecture-neutral MoE graph contract.

    Returns:
        A dict ready to serialize to ``inference_metadata.yaml``.

    """
    if num_attention_heads < 1 or head_dim < 1:
        raise ValueError("num_attention_heads and head_dim must be >= 1")
    kv = num_kv_heads or num_attention_heads
    if kv < 1 or kv > num_attention_heads or num_attention_heads % kv != 0:
        raise ValueError(
            f"num_kv_heads ({kv}) must divide num_attention_heads ({num_attention_heads})"
        )
    if attention_type is None:
        attention_type = (
            "grouped_query_attention" if kv < num_attention_heads else "multi_head"
        )
    else:
        normalized_attention_type = (
            attention_type.strip().lower().replace("-", "_").replace(" ", "_")
        )
        if normalized_attention_type in {
            "grouped_query",
            "group_query_attention",
            "grouped_query_attention",
            "gqa",
        }:
            attention_type = "grouped_query_attention"

    capabilities = ["kv_cache"]
    capabilities.append(
        "grouped_query_attention"
        if attention_type == "grouped_query_attention"
        else "multi_head_attention"
    )

    attention: dict[str, Any] = {
        "type": attention_type,
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": kv,
        "head_dim": head_dim,
    }
    if sliding_window:
        attention["sliding_window"] = sliding_window
    if sink_tokens:
        attention["sink_tokens"] = sink_tokens

    model: dict[str, Any] = {"attention": attention}
    if architecture:
        model["architecture"] = architecture
    if max_sequence_length:
        model["max_sequence_length"] = max_sequence_length
    if mixture_of_experts is not None:
        model["mixture_of_experts"] = mixture_of_experts

    metadata: dict[str, Any] = {"required_capabilities": capabilities, "model": model}
    if kv_native_dtype:
        metadata["kv_cache"] = {
            "native_dtype": _canonical_float_dtype(kv_native_dtype) or kv_native_dtype
        }
    return metadata


def moe_metadata_from_config(
    config: Any, *, representation: str = "dense_fallback"
) -> dict[str, Any] | None:
    """Build generic MoE metadata from explicit architecture configuration.

    Returns ``None`` for dense models. The emitted contract describes graph
    structure and router semantics only; it never dispatches on a model name.
    """
    routed_experts = _clean_int(getattr(config, "num_local_experts", None))
    if routed_experts is None or routed_experts <= 1:
        return None

    experts_per_token = _clean_int(getattr(config, "num_experts_per_tok", None))
    expert_intermediate_size = _clean_int(
        getattr(config, "moe_intermediate_size", None)
    ) or _clean_int(getattr(config, "intermediate_size", None))
    if experts_per_token is None:
        raise ValueError(
            "cannot emit mixture_of_experts metadata: config is missing num_experts_per_tok"
        )
    if experts_per_token > routed_experts:
        raise ValueError(
            "cannot emit mixture_of_experts metadata: num_experts_per_tok "
            f"({experts_per_token}) exceeds num_local_experts ({routed_experts})"
        )
    if expert_intermediate_size is None:
        raise ValueError(
            "cannot emit mixture_of_experts metadata: config lacks both "
            "moe_intermediate_size and intermediate_size"
        )

    shared_expert_value = None
    for name in (
        "n_shared_experts",
        "moe_num_shared_experts",
        "num_shared_expert",
    ):
        value = getattr(config, name, None)
        if isinstance(value, (list, tuple)):
            value = next((item for item in value if _clean_int(item) is not None), None)
        if _clean_int(value) is not None:
            shared_expert_value = value
            break
    shared_experts = _clean_int(shared_expert_value) or 0
    shared_intermediate_size = _clean_int(
        getattr(config, "shared_expert_intermediate_size", None)
    )
    if shared_intermediate_size is None:
        shared_intermediate_size = shared_experts * expert_intermediate_size

    score_function = str(getattr(config, "scoring_func", "softmax")).lower()
    topk_method = str(getattr(config, "topk_method", "greedy")).lower()
    group_count = _clean_int(getattr(config, "n_group", None)) or 1
    groups_per_token = _clean_int(getattr(config, "topk_group", None)) or 1
    selection_method = (
        "grouped_top_k" if group_count > 1 and topk_method != "greedy" else "top_k"
    )

    router: dict[str, Any] = {
        "score_function": score_function,
        "selection_method": selection_method,
        "normalize_weights": bool(getattr(config, "norm_topk_prob", True)),
        "scaling_factor": float(getattr(config, "routed_scaling_factor", 1.0)),
    }
    if selection_method == "grouped_top_k":
        router["group_count"] = group_count
        router["groups_per_token"] = groups_per_token
        router["group_score"] = "top_2_sum" if topk_method == "noaux_tc" else "maximum"

    return {
        "representation": representation,
        "routed_expert_count": routed_experts,
        "shared_expert_count": shared_experts,
        "experts_per_token": experts_per_token,
        "expert_intermediate_size": expert_intermediate_size,
        "shared_expert_intermediate_size": shared_intermediate_size,
        "activation": str(getattr(config, "hidden_act", "silu")).lower(),
        "router": router,
    }


def decoder_metadata_from_config(
    config: Any, *, kv_native_dtype: str | None = None
) -> dict[str, Any]:
    """Build decoder metadata from a Mobius ``BaseModelConfig``/``ArchitectureConfig``.

    Unset fields (Mobius' ``DEFAULT_INT`` sentinel) are dropped. ``head_dim``
    falls back to ``hidden_size // num_attention_heads`` when not declared.
    """
    num_heads = _clean_int(getattr(config, "num_attention_heads", None))
    if num_heads is None:
        raise ValueError("config is missing num_attention_heads")
    num_kv = _clean_int(getattr(config, "num_key_value_heads", None)) or num_heads
    head_dim = _clean_int(getattr(config, "head_dim", None))
    if head_dim is None:
        hidden = _clean_int(getattr(config, "hidden_size", None))
        if hidden is None:
            raise ValueError("config lacks head_dim and hidden_size")
        head_dim = hidden // num_heads

    sliding = getattr(config, "sliding_window", None)
    sliding = _clean_int(sliding) if sliding not in (None, _UNSET) else None

    if kv_native_dtype is None:
        kv_native_dtype = _infer_kv_native_dtype(config)

    return build_decoder_metadata(
        num_attention_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv,
        max_sequence_length=_clean_int(getattr(config, "max_position_embeddings", None)),
        sliding_window=sliding,
        kv_native_dtype=kv_native_dtype,
        architecture=getattr(config, "architecture", None)
        or getattr(config, "model_type", None),
        mixture_of_experts=moe_metadata_from_config(config),
    )


def write_decoder_metadata(
    directory: str,
    *,
    filename: str = "inference_metadata.yaml",
    config: Any | None = None,
    **kwargs: Any,
) -> str:
    """Build and write decoder ``inference_metadata.yaml`` into ``directory``.

    Provide either ``config`` (a Mobius model config) or the explicit keyword
    args accepted by :func:`build_decoder_metadata`.
    """
    if config is not None:
        metadata = decoder_metadata_from_config(config, **kwargs)
    else:
        metadata = build_decoder_metadata(**kwargs)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path
