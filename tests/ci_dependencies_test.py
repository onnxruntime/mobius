# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Keep pinned Transformers dependency resolution separate from VibeVoice."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_TRANSFORMERS = "requirements/ci/transformers.txt"
_VIBEVOICE = "requirements/ci/vibevoice.txt"
_WORKFLOWS = sorted(
    path
    for path in (_ROOT / ".github/workflows").glob("*.yml")
    if _VIBEVOICE in path.read_text(encoding="utf-8")
)


@pytest.mark.parametrize("path", _WORKFLOWS, ids=lambda path: path.name)
def test_vibevoice_install_resolves_transformers_first(path):
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    checked = 0
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            script = step.get("run", "")
            if _VIBEVOICE not in script:
                continue
            commands = [shlex.split(line, comments=True) for line in script.splitlines()]
            for index, command in enumerate(commands):
                if _VIBEVOICE not in command:
                    continue
                assert "--no-deps" in command
                assert command[command.index("-r") + 1] == _VIBEVOICE
                resolved = [prior for prior in commands[:index] if _TRANSFORMERS in prior]
                assert resolved, f"{path.name}: resolve pinned Transformers first"
                assert all("--no-deps" not in prior for prior in resolved)
                assert all("-e" in prior for prior in resolved)
                assert any(
                    "from transformers import AutoConfig" in later
                    for later in commands[index + 1 :]
                )
                checked += 1
    assert checked, f"{path.name}: no VibeVoice installation was checked"


@pytest.mark.parametrize(
    ("requirements", "package"), [(_TRANSFORMERS, "transformers"), (_VIBEVOICE, "vibevoice")]
)
def test_ci_source_requirements_contain_only_the_intended_package(requirements, package):
    lines = [
        line.strip()
        for line in (_ROOT / requirements).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(lines) == 1
    assert lines[0].startswith(f"{package} @ git+https://github.com/")
    revision = lines[0].rsplit("@", 1)[1]
    assert len(revision) == 40
    assert all(character in "0123456789abcdef" for character in revision)
