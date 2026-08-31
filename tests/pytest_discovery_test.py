# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regression tests for the repository's default pytest discovery."""

from __future__ import annotations

import subprocess
import sys
from pathlib import PurePosixPath

import pytest


def _normalize_repo_path(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def _collected_test_paths(output: str) -> set[str]:
    return {
        _normalize_repo_path(line.split("::", 1)[0].strip())
        for line in output.splitlines()
        if "::" in line
    }


@pytest.mark.parametrize(
    ("nodeid", "expected"),
    [
        ("scripts/nested/tool_test.py::test_case", "scripts/nested/tool_test.py"),
        (r"scripts\nested\tool_test.py::TestTool::test_case", "scripts/nested/tool_test.py"),
    ],
)
def test_collected_test_paths_normalizes_platform_separators(nodeid: str, expected: str):
    assert _collected_test_paths(nodeid) == {expected}


def test_default_discovery_includes_repository_tool_tests(pytestconfig: pytest.Config):
    assert "scripts" in pytestconfig.getini("testpaths")

    tracked_result = subprocess.run(
        ["git", "ls-files", "-z", "--", "scripts"],
        cwd=pytestconfig.rootpath,
        capture_output=True,
        check=True,
        text=True,
    )
    tracked_script_tests = {
        normalized_path
        for path in tracked_result.stdout.split("\0")
        if path
        if (normalized_path := _normalize_repo_path(path))
        if PurePosixPath(normalized_path).name.endswith("_test.py")
    }
    known_modules = {
        "scripts/detect_affected_models_test.py",
        "scripts/generate_golden_test.py",
    }
    assert known_modules <= tracked_script_tests

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--color=no", "scripts"],
        cwd=pytestconfig.rootpath,
        capture_output=True,
        text=True,
    )
    collection_output = result.stdout + result.stderr
    assert result.returncode == 0, collection_output

    collected_script_tests = _collected_test_paths(collection_output)
    missing_script_tests = tracked_script_tests - collected_script_tests
    assert not missing_script_tests, (
        f"pytest did not collect tracked script tests: {sorted(missing_script_tests)}"
    )
