# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deterministic serialization helpers for ONNX-GenAI metadata."""

from __future__ import annotations

import copy
from typing import Any

import yaml


class _NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _published_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return metadata with tensor contracts in the current serialized form."""
    published = copy.deepcopy(metadata)

    def normalize(value: Any, path: str) -> None:
        if isinstance(value, dict):
            if path == "pipeline.workflow.manifest":
                value.pop("capabilities", None)
            if "dtype" in value and "shape" in value and "rank" in value:
                shape = value["shape"]
                rank = value["rank"]
                if not isinstance(shape, list):
                    raise ValueError(f"tensor contract at {path} has non-list shape {shape!r}")
                if rank != len(shape):
                    raise ValueError(
                        f"tensor contract at {path} declares rank {rank!r} but "
                        f"shape {shape!r} has rank {len(shape)}"
                    )
                del value["rank"]
            for key, nested in value.items():
                normalize(nested, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                normalize(nested, f"{path}[{index}]")

    normalize(published, "")
    return published


def _dump_yaml(metadata: dict[str, Any], handle: Any) -> None:
    yaml.dump(
        _published_metadata(metadata),
        handle,
        Dumper=_NoAliasSafeDumper,
        sort_keys=False,
    )
