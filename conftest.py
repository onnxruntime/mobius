# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Prefer worktree imports and provide shared, explicit experimental opt-ins."""

from __future__ import annotations

import importlib
import os
import sys

import pytest

# Insert the worktree src/ at the front of sys.path so it takes priority
# over any installed (editable or otherwise) copy of the package.
_src = os.path.join(os.path.dirname(__file__), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Force reload of mobius modules if they were loaded from a different location
# (e.g., from the main repo's editable install rather than this worktree).
_mobius_file = sys.modules.get("mobius", None)
if _mobius_file is not None and not getattr(_mobius_file, "__file__", "").startswith(_src):
    # Stale import from a different path — purge and reload from worktree.
    for key in list(sys.modules.keys()):
        if key == "mobius" or key.startswith("mobius."):
            del sys.modules[key]
    importlib.import_module("mobius")


@pytest.fixture
def experimental_cuda_qkv_packing(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    import onnx_ir as ir

    from mobius._execution_providers import ep_registry

    monkeypatch.setitem(
        ep_registry._entries,
        "cuda",
        dataclasses.replace(
            ep_registry.require("cuda"),
            matmul_nbits_qkv_pack_dtypes=frozenset({ir.DataType.FLOAT16}),
        ),
    )
