# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate onnx-genai ``inference_metadata.yaml`` sidecars."""

from __future__ import annotations

import os
from typing import Any

import onnx_ir as ir

from mobius._model_package import ModelPackage

_DTYPE_NAMES = {
    ir.DataType.FLOAT: "float32",
    ir.DataType.FLOAT16: "float16",
    ir.DataType.BFLOAT16: "bfloat16",
}
_DEFAULT_MAX_SEQUENCE_LENGTH = 4096


def _positive_int(config: object, name: str) -> int:
    value = getattr(config, name, None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"onnx-genai inference metadata requires a positive {name}, got {value!r}."
        )
    return value


def _kv_dtype(config: object) -> str:
    dtype = getattr(config, "dtype", None)
    try:
        return _DTYPE_NAMES[dtype]
    except KeyError:
        raise ValueError(
            "onnx-genai inference metadata supports float32, float16, or bfloat16 "
            f"KV caches, got {dtype!r}."
        ) from None


def _max_sequence_length(config: object, requested: int | None) -> int:
    model_max = _positive_int(config, "max_position_embeddings")
    if requested is None:
        return min(model_max, _DEFAULT_MAX_SEQUENCE_LENGTH)
    if not isinstance(requested, int) or requested <= 0:
        raise ValueError(
            f"onnx-genai max_sequence_length must be a positive integer, got {requested!r}."
        )
    if requested > model_max:
        raise ValueError(
            f"onnx-genai max_sequence_length {requested} exceeds the model limit {model_max}."
        )
    return requested


def generate_inference_metadata(
    config: object, *, max_sequence_length: int | None = None
) -> dict[str, Any]:
    """Map a decoder config to metadata with a conservative serving KV capacity."""
    num_attention_heads = _positive_int(config, "num_attention_heads")
    num_kv_heads = _positive_int(config, "num_key_value_heads")
    head_dim = _positive_int(config, "head_dim")
    max_sequence_length = _max_sequence_length(config, max_sequence_length)
    kv_dtype = _kv_dtype(config)

    is_gqa = num_kv_heads != num_attention_heads
    capabilities = ["grouped_query_attention" if is_gqa else "multi_head_attention"]

    attention: dict[str, Any] = {
        "type": "group_query_attention" if is_gqa else "multi_head_attention",
        "num_kv_heads": num_kv_heads,
        "num_attention_heads": num_attention_heads,
        "head_dim": head_dim,
    }
    sliding_window = getattr(config, "sliding_window", None)
    if isinstance(sliding_window, int) and sliding_window > 0:
        attention["sliding_window"] = sliding_window

    return {
        "required_capabilities": capabilities,
        "model": {
            "attention": attention,
            "max_sequence_length": max_sequence_length,
            "runtime_configurable": {"kv_cache": {"dtype": [kv_dtype]}},
        },
        "kv_cache": {"native_dtype": kv_dtype},
    }


def _to_yaml(metadata: dict[str, Any]) -> str:
    capabilities = metadata["required_capabilities"]
    attention = metadata["model"]["attention"]
    kv_dtypes = metadata["model"]["runtime_configurable"]["kv_cache"]["dtype"]

    lines = ["required_capabilities:"]
    if capabilities:
        lines.extend(f"  - {capability}" for capability in capabilities)
    else:
        lines[-1] += " []"

    lines.extend(
        [
            "model:",
            "  attention:",
            f"    type: {attention['type']}",
            f"    num_kv_heads: {attention['num_kv_heads']}",
            f"    num_attention_heads: {attention['num_attention_heads']}",
            f"    head_dim: {attention['head_dim']}",
        ]
    )
    if "sliding_window" in attention:
        lines.append(f"    sliding_window: {attention['sliding_window']}")
    lines.extend(
        [
            f"  max_sequence_length: {metadata['model']['max_sequence_length']}",
            "  runtime_configurable:",
            "    kv_cache:",
            "      dtype:",
            *(f"        - {dtype}" for dtype in kv_dtypes),
            "kv_cache:",
            f"  native_dtype: {metadata['kv_cache']['native_dtype']}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_inference_metadata(
    pkg: ModelPackage,
    directory: str,
    *,
    max_sequence_length: int | None = None,
) -> str:
    """Write ``inference_metadata.yaml`` for an already-built model package."""
    config = getattr(pkg, "config", None)
    if config is None:
        raise ValueError(
            "write_inference_metadata requires ModelPackage.config to be set. "
            "This is set automatically when building with mobius.build()."
        )

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "inference_metadata.yaml")
    with open(path, "w", encoding="utf-8") as file:
        file.write(
            _to_yaml(
                generate_inference_metadata(config, max_sequence_length=max_sequence_length)
            )
        )
    return path
