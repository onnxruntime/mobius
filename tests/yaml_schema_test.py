# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Schema validation tests for YAML test case files in testdata/cases/.

Validates every ``.yaml`` file in ``testdata/cases/`` against the JSON Schema
at ``testdata/cases/schema.json``. These tests are fast (no model downloads,
no ONNX inference) and run as part of the standard unit test suite.

Run::

    pytest tests/yaml_schema_test.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

from mobius.integrations.gguf._arch_registry import iter_arch_specs
from mobius.integrations.gguf._runtime_evidence import runtime_evidence
from mobius.integrations.gguf._spec import Support

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CASES_DIR = _REPO_ROOT / "testdata" / "cases"
_SCHEMA_PATH = _CASES_DIR / "schema.json"


def _load_schema() -> dict[str, Any]:
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _all_yaml_files() -> list[Path]:
    """Return all .yaml files under testdata/cases/, sorted for stable ordering."""
    return sorted(_CASES_DIR.rglob("*.yaml"))


# ---------------------------------------------------------------------------
# Parametrized validation test
# ---------------------------------------------------------------------------

_YAML_FILES = _all_yaml_files()
_SCHEMA = _load_schema()
_VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA)


@pytest.mark.parametrize(
    "yaml_path",
    _YAML_FILES,
    ids=[f.relative_to(_CASES_DIR).as_posix() for f in _YAML_FILES],
)
def test_yaml_validates_against_schema(yaml_path: Path) -> None:
    """Each YAML test case must conform to testdata/cases/schema.json."""
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    errors = sorted(_VALIDATOR.iter_errors(data), key=lambda e: e.json_path)
    if errors:
        messages = "\n".join(f"  [{e.json_path}] {e.message}" for e in errors)
        pytest.fail(f"{yaml_path.relative_to(_REPO_ROOT)}:\n{messages}")


# ---------------------------------------------------------------------------
# Schema self-consistency tests
# ---------------------------------------------------------------------------


def test_schema_file_exists() -> None:
    """schema.json must be present at the expected path."""
    assert _SCHEMA_PATH.exists(), f"Schema file not found: {_SCHEMA_PATH}"


def test_schema_is_valid_json_schema() -> None:
    """schema.json itself must be a valid JSON Schema (meta-validation)."""
    meta_validator = jsonschema.Draft202012Validator(
        jsonschema.Draft202012Validator.META_SCHEMA
    )
    errors = list(meta_validator.iter_errors(_SCHEMA))
    if errors:
        messages = "\n".join(f"  {e.message}" for e in errors)
        pytest.fail(f"schema.json is not a valid JSON Schema:\n{messages}")


def test_at_least_one_yaml_found() -> None:
    """Sanity check: the discovery function must find at least one test case."""
    assert len(_YAML_FILES) > 0, f"No .yaml files found in {_CASES_DIR}"


def test_all_yaml_task_types_are_in_schema() -> None:
    """Every task_type value used in actual YAML files must be in the schema enum.

    This catches the case where a new YAML uses a task_type that the schema
    doesn't know about yet — the schema enum needs to be updated.
    """
    schema_task_types: set[str] = set(_SCHEMA["properties"]["task_type"]["enum"])
    missing: list[str] = []
    for yaml_path in _YAML_FILES:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        task_type = data.get("task_type") if isinstance(data, dict) else None
        if task_type and task_type not in schema_task_types:
            missing.append(f"  {yaml_path.relative_to(_REPO_ROOT)}: task_type={task_type!r}")
    if missing:
        pytest.fail(
            "These YAML files use task_type values not listed in schema.json. "
            "Add them to the task_type enum in testdata/cases/schema.json:\n"
            + "\n".join(missing)
        )


def test_every_runtime_supported_route_has_ort_genai_e2e_enrollment() -> None:
    """Runtime support cannot outgrow pinned downstream generation coverage."""
    enrolled: dict[str, dict[str, Any]] = {}
    for yaml_path in _YAML_FILES:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        marker = data.get("ort_genai")
        if marker:
            evidence_id = marker["runtime_evidence_id"]
            assert evidence_id not in enrolled, (
                f"Duplicate ORT GenAI evidence ID: {evidence_id}"
            )
            enrolled[evidence_id] = marker

    required_routes = {
        (
            evidence.architecture,
            evidence.repository,
            evidence.revision,
            evidence.filename,
            evidence.import_route,
        )
        for spec in iter_arch_specs()
        if spec.runtime is Support.SUPPORTED
        for evidence_id in spec.runtime_evidence_ids
        if (evidence := runtime_evidence(evidence_id)) is not None
    }
    enrolled_routes = set()
    for evidence_id, marker in enrolled.items():
        evidence = runtime_evidence(evidence_id)
        assert evidence is not None
        assert evidence.runtime == "ort-genai"
        assert evidence.runtime_version in marker["runtime_versions"]
        enrolled_routes.add(
            (
                evidence.architecture,
                evidence.repository,
                evidence.revision,
                evidence.filename,
                evidence.import_route,
            )
        )
        assert marker["execution_provider"] == "cpu"
        assert marker["model_type"] == "decoder"
        assert marker["runtime_versions"] == ["0.15.2"]
        assert marker["max_download_bytes"] >= evidence.size
    assert enrolled_routes == required_routes
