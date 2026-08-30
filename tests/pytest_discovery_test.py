# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regression tests for the repository's default pytest discovery."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_default_discovery_includes_repository_tool_tests(pytestconfig: pytest.Config):
    assert "scripts" in pytestconfig.getini("testpaths")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--color=no", "scripts"],
        cwd=pytestconfig.rootpath,
        capture_output=True,
        text=True,
    )
    collection_output = result.stdout + result.stderr
    assert result.returncode == 0, collection_output

    expected_modules = {
        "detect_affected_models_test.py",
        "generate_golden_test.py",
    }
    assert all(f"{module}::" in collection_output for module in expected_modules)
