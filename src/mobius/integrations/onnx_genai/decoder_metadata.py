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


def _clean_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


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
) -> dict[str, Any]:
    """Build the onnx-genai ``inference_metadata`` dict for a decoder LLM.

    Args:
        num_attention_heads: Number of query/attention heads.
        head_dim: Per-head hidden dimension.
        num_kv_heads: Number of key/value heads (defaults to
            ``num_attention_heads`` = multi-head; a smaller value = GQA).
        max_sequence_length: Maximum total sequence length in tokens.
        kv_native_dtype: KV-cache storage dtype (e.g. ``"fp16"``, ``"bf16"``).
        attention_type: Override the derived attention type (``multi_head`` /
            ``grouped_query``).
        sliding_window: Sliding-window length in tokens (None = full context).
        sink_tokens: Attention-sink tokens retained with the sliding window.
        architecture: Optional architecture hint (e.g. ``"llama"``).

    Returns:
        A dict ready to serialize to ``inference_metadata.yaml``.
    """
    if num_attention_heads < 1 or head_dim < 1:
        raise ValueError("num_attention_heads and head_dim must be >= 1")
    kv = num_kv_heads or num_attention_heads
    if kv < 1 or kv > num_attention_heads or num_attention_heads % kv != 0:
        raise ValueError(
            f"num_kv_heads ({kv}) must divide num_attention_heads "
            f"({num_attention_heads})"
        )
    if attention_type is None:
        attention_type = "grouped_query" if kv < num_attention_heads else "multi_head"

    caps = ["kv_cache"]
    caps.append(
        "grouped_query_attention"
        if attention_type == "grouped_query"
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

    metadata: dict[str, Any] = {"required_capabilities": caps, "model": model}
    if kv_native_dtype:
        metadata["kv_cache"] = {"native_dtype": kv_native_dtype}
    return metadata


def decoder_metadata_from_config(config: Any, *, kv_native_dtype: str | None = None) -> dict[str, Any]:
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

    return build_decoder_metadata(
        num_attention_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv,
        max_sequence_length=_clean_int(getattr(config, "max_position_embeddings", None)),
        sliding_window=sliding,
        kv_native_dtype=kv_native_dtype,
        architecture=getattr(config, "architecture", None) or getattr(config, "model_type", None),
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
