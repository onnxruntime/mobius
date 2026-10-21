# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Internal fixed-schema identity for independently persisted split components."""

from __future__ import annotations

from collections.abc import Mapping

SPLIT_ABI_PROPERTY = "mobius.qwen4_exp.split_abi"
COMPONENT_KEY_PROPERTY = "mobius.qwen4_exp.component_key"
SPLIT_ABI = "qwen4-exp-ngram-main.v1"
TEXT_COMPONENT_KEYS = frozenset({"ngram", "main"})
VLM_COMPONENT_KEYS = TEXT_COMPONENT_KEYS | {"embedding", "vision_encoder"}


def split_component_key(metadata: Mapping[str, str]) -> str | None:
    """Recover a marked graph's identity; legacy graphs have neither property."""
    abi = metadata.get(SPLIT_ABI_PROPERTY)
    key = metadata.get(COMPONENT_KEY_PROPERTY)
    if abi is None and key is None:
        return None
    if abi is None or key is None:
        raise ValueError("Split identity requires both split_abi and component_key")
    if abi != SPLIT_ABI:
        raise ValueError(f"Unsupported split ABI {abi!r}")
    if key not in VLM_COMPONENT_KEYS:
        raise ValueError(f"Unknown split component key {key!r}")
    return key
