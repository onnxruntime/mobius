# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deterministic serialization helpers for ONNX-GenAI metadata."""

from __future__ import annotations

from typing import Any

import yaml


class _NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _dump_yaml(metadata: dict[str, Any], handle: Any) -> None:
    yaml.dump(metadata, handle, Dumper=_NoAliasSafeDumper, sort_keys=False)
