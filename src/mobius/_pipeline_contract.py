# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Declarations carried by exported ONNX graphs for pipeline metadata emission."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import onnx_ir as ir

_COMPONENT_PRESENCE = "mobius.pipeline.when_present"
_OPTIONAL_INPUT_PRESENCE = "mobius.pipeline.optional_input.presence"
_OPTIONAL_INPUT_ABSENT_SHAPE = "mobius.pipeline.optional_input.absent_shape"
_ARBITRARY_ATTENTION_MASK = "mobius.attention.requires_arbitrary_mask"


def declare_component_presence(graph: ir.Graph, presence: str) -> None:
    """Declare the opaque presence key required to execute a component graph."""
    if not presence:
        raise ValueError("Component presence key must be non-empty")
    graph.metadata_props[_COMPONENT_PRESENCE] = presence


def component_presence(graph: Any) -> str | None:
    """Return a component graph's declared presence key, if any."""
    presence = getattr(graph, "metadata_props", {}).get(_COMPONENT_PRESENCE)
    return presence or None


def declare_optional_input(
    value: ir.Value,
    *,
    presence: str,
    absent_shape: Sequence[int | str],
) -> None:
    """Declare a zero fallback for an optional ONNX graph input."""
    if not presence:
        raise ValueError("Optional-input presence key must be non-empty")
    shape = list(absent_shape)
    if not shape or any(isinstance(dim, int) and dim < 0 for dim in shape):
        raise ValueError("Optional-input absent shape must contain non-negative dimensions")
    value.metadata_props[_OPTIONAL_INPUT_PRESENCE] = presence
    value.metadata_props[_OPTIONAL_INPUT_ABSENT_SHAPE] = json.dumps(shape)


def optional_input_contract(value: Any) -> dict[str, Any] | None:
    """Return the executable optional-input contract declared on a graph input."""
    metadata = getattr(value, "metadata_props", {})
    presence = metadata.get(_OPTIONAL_INPUT_PRESENCE)
    absent_shape = metadata.get(_OPTIONAL_INPUT_ABSENT_SHAPE)
    if presence is None and absent_shape is None:
        return None
    if not presence or absent_shape is None:
        raise ValueError(
            f"ONNX input {value.name!r} has an incomplete optional-input declaration"
        )
    try:
        shape = json.loads(absent_shape)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"ONNX input {value.name!r} has an invalid optional-input absent shape"
        ) from error
    if not isinstance(shape, list) or not shape:
        raise ValueError(
            f"ONNX input {value.name!r} optional-input absent shape must be a non-empty list"
        )
    return {
        "presence": presence,
        "absent": {
            "kind": "zeros",
            "shape": shape,
        },
    }


def declare_arbitrary_attention_mask(graph: ir.Graph) -> None:
    """Prevent attention fusions that only support prefix-valid masks."""
    graph.metadata_props[_ARBITRARY_ATTENTION_MASK] = "true"


def requires_arbitrary_attention_mask(graph: Any) -> bool:
    """Return whether a decoder needs its full additive attention mask."""
    return getattr(graph, "metadata_props", {}).get(_ARBITRARY_ATTENTION_MASK) == "true"
