# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Guards the repo rule that ONNX Runtime telemetry is disabled.

The rule is enforced in two places that share no code — the rootdir ``conftest.py`` and every CI
workflow — because a workflow step that reaches ORT without going through pytest never loads the
conftest. Config duplicated across files with nothing tying it together is what drifts, so it is
checked here.
"""

from __future__ import annotations

import os
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"
_WORKFLOWS = sorted(p.name for p in _WORKFLOW_DIR.glob("*.yml"))


def test_telemetry_is_disabled_in_this_process() -> None:
    """The rootdir conftest must set this before anything imports ``onnxruntime``.

    ORT reads the variable when its native library loads, so a fixture — even session-scoped and
    autouse — is too late: test modules import ORT at module scope. Asserted on the environment
    because ORT exposes no way to read the setting back.
    """
    assert os.environ.get("ORT_DISABLE_TELEMETRY") == "1", (
        "ORT_DISABLE_TELEMETRY is not set for the test process; it belongs in the rootdir "
        "conftest.py, before any onnxruntime import"
    )


def test_every_workflow_exists() -> None:
    """Guard the guard: if the glob stopped matching, the parametrized test would silently pass."""
    assert _WORKFLOWS, f"no workflows found under {_WORKFLOW_DIR}"


@pytest.mark.parametrize("workflow", _WORKFLOWS)
def test_workflow_disables_telemetry(workflow: str) -> None:
    """Set at workflow level, so all jobs and steps inherit it.

    Job-level would leave any ORT-invoking step in an unpatched job uncovered, and there is no
    reason for the setting to vary by job.
    """
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((_WORKFLOW_DIR / workflow).read_text())
    assert config.get("env", {}).get("ORT_DISABLE_TELEMETRY") == "1", (
        f"{workflow} does not set ORT_DISABLE_TELEMETRY at workflow level"
    )
