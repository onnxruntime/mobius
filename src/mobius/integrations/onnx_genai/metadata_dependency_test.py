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


def test_inference_metadata_has_no_workflow_metadata_dependency() -> None:
    inference_path = Path(__file__).with_name("inference_metadata.py")
    contract_path = Path(__file__).with_name("_workflow_contract.py")

    inference_imports = {
        node.module
        for node in ast.walk(ast.parse(inference_path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
    }
    contract_imports = {
        node.module
        for node in ast.walk(ast.parse(contract_path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
    }

    assert f"{_PACKAGE}.workflow_metadata" not in inference_imports
    assert f"{_PACKAGE}.workflow_metadata" not in contract_imports
    assert f"{_PACKAGE}.inference_metadata" not in contract_imports


def test_metadata_yaml_is_deterministic_and_suppresses_aliases() -> None:
    shared = [1, 2]
    output = io.StringIO()

    _dump_yaml({"first": shared, "second": shared}, output)

    assert output.getvalue().encode() == (
        b"first:\n"
        b"- 1\n"
        b"- 2\n"
        b"second:\n"
        b"- 1\n"
        b"- 2\n"
    )
