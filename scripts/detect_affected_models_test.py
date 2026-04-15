# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for scripts/detect_affected_models.py.

Tests the AST-based detection logic without importing the actual
model registry — all analysis is done via AST parsing of source files.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Import the detection module directly
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from detect_affected_models import (  # noqa: E402
    _SRC_ROOT,
    _build_class_to_source_module,
    _build_import_graph,
    _build_registry_class_to_types,
    _build_source_module_to_types,
    _find_reverse_dependents,
    classify_file,
    detect_affected_models,
)

# ----------------------------------------------------------------
# classify_file tests
# ----------------------------------------------------------------


class TestClassifyFile:
    def test_model_file(self):
        assert classify_file("src/mobius/models/falcon.py") == "model"

    def test_model_init_is_shared_infra(self):
        """models/__init__.py is the re-export hub — classify as shared_infra."""
        assert classify_file("src/mobius/models/__init__.py") == "shared_infra"

    def test_component_file(self):
        assert classify_file("src/mobius/components/_attention.py") == "traceable"

    def test_task_file(self):
        assert classify_file("src/mobius/tasks/_causal_lm.py") == "shared_infra"

    def test_configs_file(self):
        assert classify_file("src/mobius/_configs.py") == "shared_infra"

    def test_registry_file(self):
        assert classify_file("src/mobius/_registry.py") == "shared_infra"

    def test_builder_file(self):
        assert classify_file("src/mobius/_builder.py") == "shared_infra"

    def test_exporter_file(self):
        assert classify_file("src/mobius/_exporter.py") == "shared_infra"

    def test_test_file_in_src(self):
        assert classify_file("src/mobius/models/_models_test.py") == "test"

    def test_test_file_in_tests(self):
        assert classify_file("tests/build_graph_test.py") == "test"

    def test_test_infra_conftest(self):
        assert classify_file("tests/conftest.py") == "shared_infra"

    def test_test_infra_configs(self):
        assert classify_file("tests/_test_configs.py") == "shared_infra"

    def test_readme(self):
        assert classify_file("README.md") == "other"

    def test_pyproject(self):
        assert classify_file("pyproject.toml") == "other"

    def test_windows_paths(self):
        assert classify_file("src\\mobius\\models\\falcon.py") == "model"


# ----------------------------------------------------------------
# AST registry parsing tests
# ----------------------------------------------------------------


class TestRegistryParsing:
    """Tests that AST-based registry parsing finds real mappings."""

    def test_class_to_source_module_has_entries(self):
        mapping = _build_class_to_source_module()
        # CausalLMModel should map to models.base
        assert "CausalLMModel" in mapping
        assert mapping["CausalLMModel"] == "mobius.models.base"

    def test_class_to_source_falcon(self):
        mapping = _build_class_to_source_module()
        assert "FalconCausalLMModel" in mapping
        assert mapping["FalconCausalLMModel"] == "mobius.models.falcon"

    def test_registry_class_to_types_has_entries(self):
        mapping = _build_registry_class_to_types()
        assert len(mapping) > 10

    def test_registry_has_causal_lm_model(self):
        mapping = _build_registry_class_to_types()
        assert "CausalLMModel" in mapping
        types = mapping["CausalLMModel"]
        assert "llama" in types
        assert "qwen2" in types

    def test_registry_has_falcon(self):
        mapping = _build_registry_class_to_types()
        assert "FalconCausalLMModel" in mapping
        types = mapping["FalconCausalLMModel"]
        assert "falcon" in types
        assert "falcon_h1" in types

    def test_source_module_to_types(self):
        mapping = _build_source_module_to_types()
        # falcon.py should map to falcon, bloom, mpt, falcon_h1
        falcon_key = "mobius.models.falcon"
        assert falcon_key in mapping
        types = mapping[falcon_key]
        assert "falcon" in types
        assert "bloom" in types

    def test_source_module_base_has_many_types(self):
        mapping = _build_source_module_to_types()
        base_key = "mobius.models.base"
        assert base_key in mapping
        # CausalLMModel is used by many model_types
        assert len(mapping[base_key]) > 20


# ----------------------------------------------------------------
# Import graph tests
# ----------------------------------------------------------------


class TestImportGraph:
    def test_build_import_graph_finds_modules(self):
        src_root = Path(__file__).resolve().parent.parent / "src" / "mobius"
        graph = _build_import_graph(src_root)
        assert len(graph) > 50  # many modules expected

    def test_reverse_dependents_simple(self):
        graph = {
            "a": {"b", "c"},
            "b": {"c"},
            "c": set(),
            "d": {"a"},
        }
        # Modules that depend on 'c': a (directly), b (directly),
        # d (transitively through a)
        deps = _find_reverse_dependents("c", graph)
        assert "a" in deps
        assert "b" in deps
        assert "d" in deps

    def test_reverse_dependents_no_self(self):
        graph = {"a": {"b"}, "b": set()}
        deps = _find_reverse_dependents("b", graph)
        assert "b" not in deps
        assert "a" in deps


# ----------------------------------------------------------------
# End-to-end detection tests
# ----------------------------------------------------------------


class TestDetectAffectedModels:
    def test_component_change_traces_affected_models(self):
        """A component change traces through the import graph to find affected models."""
        result = detect_affected_models(["src/mobius/components/_attention.py"])
        assert result["run_all"] is False
        # _attention.py is imported by many models — should find affected types
        assert len(result["affected"]) > 0

    def test_task_change_triggers_run_all(self):
        """Task files use string-based lookup, not imports — must trigger run_all."""
        result = detect_affected_models(["src/mobius/tasks/_causal_lm.py"])
        assert result["run_all"] is True

    def test_configs_change_triggers_run_all(self):
        result = detect_affected_models(["src/mobius/_configs.py"])
        assert result["run_all"] is True

    def test_unrelated_file_no_affected(self):
        result = detect_affected_models(["README.md"])
        assert result["run_all"] is False
        assert result["affected"] == []

    def test_test_file_no_affected(self):
        result = detect_affected_models(["tests/build_graph_test.py"])
        assert result["run_all"] is False
        assert result["affected"] == []

    def test_falcon_model_file(self):
        result = detect_affected_models(["src/mobius/models/falcon.py"])
        assert result["run_all"] is False
        assert "falcon" in result["affected"]
        assert "bloom" in result["affected"]

    def test_base_model_affects_many(self):
        result = detect_affected_models(["src/mobius/models/base.py"])
        assert result["run_all"] is False
        assert "llama" in result["affected"]
        assert "qwen2" in result["affected"]
        assert len(result["affected"]) > 20

    def test_moe_model_file(self):
        result = detect_affected_models(["src/mobius/models/moe.py"])
        assert result["run_all"] is False
        assert "mixtral" in result["affected"]
        assert "arctic" in result["affected"]

    def test_multiple_model_files(self):
        result = detect_affected_models(
            [
                "src/mobius/models/falcon.py",
                "src/mobius/models/gemma.py",
            ]
        )
        assert result["run_all"] is False
        assert "falcon" in result["affected"]
        assert "gemma" in result["affected"]

    def test_mixed_model_and_unrelated(self):
        result = detect_affected_models(
            [
                "README.md",
                "src/mobius/models/falcon.py",
                "docs/index.md",
            ]
        )
        assert result["run_all"] is False
        assert "falcon" in result["affected"]

    def test_shared_infra_overrides_model(self):
        """If any shared infra changes, run_all even with model files."""
        result = detect_affected_models(
            [
                "src/mobius/models/falcon.py",
                "src/mobius/_configs.py",
            ]
        )
        assert result["run_all"] is True

    def test_models_init_triggers_run_all(self):
        """models/__init__.py is the re-export hub — must trigger run_all."""
        result = detect_affected_models(["src/mobius/models/__init__.py"])
        assert result["run_all"] is True

    def test_deleted_model_file_triggers_run_all(self):
        """A model file that doesn't exist on disk triggers run_all."""
        result = detect_affected_models(["src/mobius/models/nonexistent_model.py"])
        assert result["run_all"] is True

    def test_empty_input(self):
        result = detect_affected_models([])
        assert result["run_all"] is False
        assert result["affected"] == []

    def test_component_common_affects_many_models(self):
        """_common.py is foundational — tracing should find many models."""
        result = detect_affected_models(["src/mobius/components/_common.py"])
        assert result["run_all"] is False
        # _common.py defines Linear, Embedding, LayerNorm — used everywhere
        assert len(result["affected"]) > 10

    def test_shared_infra_still_triggers_run_all(self):
        """True shared_infra files (_configs, _registry, etc.) still trigger run_all."""
        for path in [
            "src/mobius/_configs.py",
            "src/mobius/_registry.py",
            "src/mobius/_builder.py",
            "src/mobius/_weight_loading.py",
            "src/mobius/_model_package.py",
            "src/mobius/_exporter.py",
            "src/mobius/models/__init__.py",
            "tests/conftest.py",
            "tests/_test_configs.py",
        ]:
            result = detect_affected_models([path])
            assert result["run_all"] is True, f"{path} should trigger run_all but didn't"

    def test_traceable_and_model_combined(self):
        """A component + model file change returns union of affected types."""
        result = detect_affected_models(
            [
                "src/mobius/models/falcon.py",
                "src/mobius/components/_attention.py",
            ]
        )
        assert result["run_all"] is False
        assert "falcon" in result["affected"]
        # _attention.py dependents should also be included
        assert len(result["affected"]) > 2

    def test_traceable_overridden_by_shared_infra(self):
        """If both traceable and shared_infra change, run_all wins."""
        result = detect_affected_models(
            [
                "src/mobius/components/_attention.py",
                "src/mobius/_configs.py",
            ]
        )
        assert result["run_all"] is True

    def test_deleted_traceable_file_triggers_run_all(self):
        """A deleted component file triggers run_all (conservative)."""
        result = detect_affected_models(["src/mobius/components/_nonexistent_component.py"])
        assert result["run_all"] is True


# ----------------------------------------------------------------
# Traceable tracing integration tests
# ----------------------------------------------------------------


class TestTraceableTracing:
    """Verify the import graph tracing for component/task files."""

    def test_attention_component_finds_model_dependents(self):
        """_attention.py should trace to models that import it."""
        import_graph = _build_import_graph(_SRC_ROOT)
        registry_map = _build_source_module_to_types()

        dependents = _find_reverse_dependents("mobius.components._attention", import_graph)
        # At minimum, models that use Attention should appear
        affected_types: set[str] = set()
        for dep in dependents:
            if dep in registry_map:
                affected_types.update(registry_map[dep])
        assert len(affected_types) > 0, "Expected _attention.py to affect at least one model"

    def test_traceable_result_is_subset_of_all_models(self):
        """Traceable tracing should return a subset, not all models."""
        # A niche component should affect fewer models than _common.py
        result_common = detect_affected_models(["src/mobius/components/_common.py"])
        result_niche = detect_affected_models(["src/mobius/components/_sam_vision.py"])
        assert result_common["run_all"] is False
        assert result_niche["run_all"] is False
        # Niche component should affect fewer models
        assert len(result_niche["affected"]) <= len(result_common["affected"]), (
            f"_sam_vision.py ({len(result_niche['affected'])} models) should "
            f"affect <= models than _common.py ({len(result_common['affected'])})"
        )


# ----------------------------------------------------------------
# CLI tests
# ----------------------------------------------------------------


class TestCLI:
    def test_json_output(self):
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS_DIR / "detect_affected_models.py"),
                "--changed-files",
                "src/mobius/models/falcon.py",
                "--output-format",
                "json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "affected" in data
        assert "run_all" in data
        assert "falcon" in data["affected"]

    def test_github_output(self):
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS_DIR / "detect_affected_models.py"),
                "--changed-files",
                "src/mobius/models/falcon.py",
                "--output-format",
                "github",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        lines = result.stdout.strip().split("\n")
        assert any(line.startswith("affected=") for line in lines)
        assert any(line.startswith("run_all=") for line in lines)
        assert any(line.startswith("has_affected=") for line in lines)

    def test_stdin_mode(self):
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS_DIR / "detect_affected_models.py"),
                "--stdin",
                "--output-format",
                "json",
            ],
            input="src/mobius/models/falcon.py\n",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "falcon" in data["affected"]
