# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regression tests for the repository's default pytest discovery."""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest


def test_default_discovery_includes_repository_tool_tests(pytestconfig: pytest.Config):
    assert "scripts" in pytestconfig.getini("testpaths")

    python_files = pytestconfig.getini("python_files")
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    collected_script_modules = {
        path.name
        for path in scripts_dir.glob("*.py")
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in python_files)
    }

    assert collected_script_modules == {
        "detect_affected_models_test.py",
        "generate_golden_test.py",
    }
