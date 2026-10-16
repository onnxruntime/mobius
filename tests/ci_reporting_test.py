# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Keep failed-test diagnostics available when a later test kills the CI job."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


def _run_case(tmp_path: Path, *, ci: bool, phase: str, hard_exit: bool):
    shutil.copyfile(Path(__file__).with_name("conftest.py"), tmp_path / "conftest.py")
    (tmp_path / "case_test.py").write_text(
        f"""
import os
import pytest

@pytest.fixture
def failure():
    if {phase!r} == "setup":
        pytest.fail("EARLY_FAILURE_TRACE")
    yield
    if {phase!r} == "teardown":
        pytest.fail("EARLY_FAILURE_TRACE")

def test_pass():
    print("PASS_CAPTURE_MUST_STAY_HIDDEN")

def test_failure(failure):
    print("FAIL_CAPTURE_ONLY_IN_FINAL_SUMMARY")
    if {phase!r} == "call":
        pytest.fail("EARLY_FAILURE_TRACE")

def test_later():
    if {hard_exit!r}:
        os._exit(23)
"""
    )
    env = dict(os.environ, GITHUB_ACTIONS="true" if ci else "false")
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTHONUNBUFFERED", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "case_test.py",
            "-q",
            "--tb=short",
            "--color=no",
            "--junitxml=results.xml",
            "-p",
            "no:cacheprovider",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
@pytest.mark.parametrize("ci", [True, False])
def test_failure_trace_survives_later_hard_exit(tmp_path, phase, ci):
    result = _run_case(tmp_path, ci=ci, phase=phase, hard_exit=True)

    assert result.returncode == 23, result.stdout + result.stderr
    assert ("Immediate CI failure:" in result.stdout) is ci
    assert ("Failed: EARLY_FAILURE_TRACE" in result.stdout) is ci
    if ci:
        assert f"({phase})" in result.stdout
    assert "short test summary info" not in result.stdout
    assert "PASS_CAPTURE_MUST_STAY_HIDDEN" not in result.stdout
    assert "FAIL_CAPTURE_ONLY_IN_FINAL_SUMMARY" not in result.stdout


@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
@pytest.mark.parametrize("ci", [True, False])
def test_final_summary_status_and_junit_are_preserved(tmp_path, phase, ci):
    result = _run_case(tmp_path, ci=ci, phase=phase, hard_exit=False)

    assert result.returncode == 1, result.stdout + result.stderr
    assert ("Immediate CI failure:" in result.stdout) is ci
    assert "short test summary info" in result.stdout
    assert "PASS_CAPTURE_MUST_STAY_HIDDEN" not in result.stdout
    if phase != "setup":
        assert result.stdout.count("FAIL_CAPTURE_ONLY_IN_FINAL_SUMMARY") == 1
    suite = ET.parse(tmp_path / "results.xml").getroot().find("testsuite")
    assert suite is not None
    assert suite.attrib["failures"] == ("1" if phase == "call" else "0")
    assert suite.attrib["errors"] == ("0" if phase == "call" else "1")
    assert suite.attrib["tests"] == "3"
    if ci:
        assert result.stdout.index("Failed: EARLY_FAILURE_TRACE") < result.stdout.index(
            "short test summary info"
        )


@pytest.mark.parametrize("ci", [True, False])
def test_passing_run_is_unchanged(tmp_path, ci):
    result = _run_case(tmp_path, ci=ci, phase="none", hard_exit=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout
    assert "Immediate CI failure:" not in result.stdout
    assert "PASS_CAPTURE_MUST_STAY_HIDDEN" not in result.stdout
    assert "FAIL_CAPTURE_ONLY_IN_FINAL_SUMMARY" not in result.stdout
