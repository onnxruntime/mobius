# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dependency and serialization boundaries for ONNX-GenAI metadata."""

from __future__ import annotations

import ast
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mobius.integrations.onnx_genai._metadata_io import _dump_yaml

_PACKAGE = "mobius.integrations.onnx_genai"
_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def _import_edges(source: str, package: str) -> set[str]:
    """Return normalized module edges for imports written inside *package*."""
    edges: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            edges.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        if node.level:
            package_parts = package.split(".")
            base_length = len(package_parts) - node.level + 1
            if base_length < 0:
                continue
            module_parts = package_parts[:base_length]
            if node.module:
                module_parts.extend(node.module.split("."))
            module = ".".join(module_parts)
        else:
            module = node.module
        if not module:
            continue

        edges.add(module)
        edges.update(f"{module}.{alias.name}" for alias in node.names if alias.name != "*")
    return edges


@pytest.mark.parametrize(
    "module",
    [
        f"{_PACKAGE}.workflow_metadata",
        f"{_PACKAGE}.inference_metadata",
    ],
)
def test_metadata_modules_import_in_fresh_process(module: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_SOURCE_ROOT)
    subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        check=True,
        env=environment,
    )


@pytest.mark.parametrize(
    ("source", "prohibited"),
    [
        (f"from {_PACKAGE} import workflow_metadata", True),
        (f"import {_PACKAGE}.workflow_metadata", True),
        ("from . import workflow_metadata", True),
        ("from ._workflow_contract import _port", False),
    ],
)
def test_import_edges_cover_metadata_dependency_syntax(source: str, prohibited: bool) -> None:
    target = f"{_PACKAGE}.workflow_metadata"
    assert (target in _import_edges(source, _PACKAGE)) is prohibited


@pytest.mark.parametrize(
    ("filename", "forbidden"),
    [
        ("inference_metadata.py", {f"{_PACKAGE}.workflow_metadata"}),
        (
            "_workflow_contract.py",
            {
                f"{_PACKAGE}.inference_metadata",
                f"{_PACKAGE}.workflow_metadata",
            },
        ),
    ],
)
def test_metadata_modules_keep_one_way_dependencies(
    filename: str, forbidden: set[str]
) -> None:
    path = Path(__file__).with_name(filename)
    imports = _import_edges(path.read_text(encoding="utf-8"), _PACKAGE)
    assert imports.isdisjoint(forbidden)


def test_metadata_yaml_is_deterministic_and_suppresses_aliases() -> None:
    shared = [1, 2]
    output = io.StringIO()

    _dump_yaml({"first": shared, "second": shared}, output)

    assert output.getvalue().encode() == (b"first:\n- 1\n- 2\nsecond:\n- 1\n- 2\n")
