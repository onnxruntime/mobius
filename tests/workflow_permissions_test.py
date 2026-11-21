# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"


def _load_workflow(name: str) -> dict:
    workflow = (_WORKFLOWS / name).read_text(encoding="utf-8")
    return yaml.load(workflow, Loader=yaml.BaseLoader)


def _steps(job: dict) -> list[dict]:
    return job["steps"]


def test_pr_workflows_do_not_have_write_permissions() -> None:
    architecture_diff = _load_workflow("arch_diff.yml")
    benchmark = _load_workflow("benchmark.yml")

    assert architecture_diff["permissions"] == {"contents": "read"}
    assert architecture_diff["jobs"]["arch-diff"].get("permissions", {}) == {}
    architecture_steps = _steps(architecture_diff["jobs"]["arch-diff"])
    assert all("github-script" not in step.get("uses", "") for step in architecture_steps)

    assert benchmark["permissions"] == {"contents": "read"}
    for job in benchmark["jobs"].values():
        assert "write" not in job.get("permissions", {}).values()
        assert all("github-script" not in step.get("uses", "") for step in _steps(job))
