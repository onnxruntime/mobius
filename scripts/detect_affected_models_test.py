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
    _build_reexport_map,
    _build_registry_class_to_types,
    _build_source_module_to_types,
    _find_reverse_dependents,
    _parse_imports,
    classify_file,
    detect_affected_models,
)

# ----------------------------------------------------------------
# classify_file tests
# ----------------------------------------------------------------


class TestClassifyFile:
    def test_model_file(self):
        assert classify_file("src/mobius/models/falcon.py") == "model"

    def test_model_init_is_model(self):
        """models/__init__.py classifies as model (shared_infra disabled)."""
        assert classify_file("src/mobius/models/__init__.py") == "model"

    def test_component_file(self):
        assert classify_file("src/mobius/components/_attention.py") == "traceable"

    def test_task_file(self):
        assert classify_file("src/mobius/tasks/_causal_lm.py") == "traceable"

    def test_configs_file(self):
        assert classify_file("src/mobius/_configs.py") == "other"

    def test_registry_file(self):
        assert classify_file("src/mobius/_registry.py") == "other"

    def test_builder_file(self):
        assert classify_file("src/mobius/_builder.py") == "other"

    def test_exporter_file(self):
        assert classify_file("src/mobius/_exporter.py") == "other"

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

    def test_task_change_does_not_trigger_run_all(self):
        """Task files are traceable but produce an empty affected set.

        No model imports ``mobius.tasks`` directly (tasks are looked up at
        runtime by string keys), so tracing through the import graph finds
        no dependents. Documented limitation — see PR description.
        """
        result = detect_affected_models(["src/mobius/tasks/_causal_lm.py"])
        assert result["run_all"] is False
        assert result["affected"] == []

    def test_configs_change_no_run_all(self):
        """_configs.py no longer triggers run_all (shared_infra disabled)."""
        result = detect_affected_models(["src/mobius/_configs.py"])
        assert result["run_all"] is False
        assert result["affected"] == []

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

    def test_infra_does_not_override_model(self):
        """Infra files no longer trigger run_all (shared_infra disabled)."""
        result = detect_affected_models(
            [
                "src/mobius/models/falcon.py",
                "src/mobius/_configs.py",
            ]
        )
        assert result["run_all"] is False
        assert "falcon" in result["affected"]

    def test_models_init_no_run_all(self):
        """models/__init__.py no longer triggers run_all (shared_infra disabled)."""
        result = detect_affected_models(["src/mobius/models/__init__.py"])
        assert result["run_all"] is False

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

    def test_former_shared_infra_no_run_all(self):
        """Former shared_infra files no longer trigger run_all."""
        for path in [
            "src/mobius/_configs.py",
            "src/mobius/_registry.py",
            "src/mobius/_builder.py",
            "src/mobius/_weight_loading.py",
            "src/mobius/_model_package.py",
            "src/mobius/models/__init__.py",
        ]:
            result = detect_affected_models([path])
            assert result["run_all"] is False, (
                f"{path} should NOT trigger run_all"
            )

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

    def test_traceable_and_infra_no_run_all(self):
        """Component + infra files no longer trigger run_all."""
        result = detect_affected_models(
            [
                "src/mobius/components/_attention.py",
                "src/mobius/_configs.py",
            ]
        )
        assert result["run_all"] is False
        assert len(result["affected"]) > 0

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
# Re-export resolution tests
#
# These tests use synthetic source trees in a temp directory so they
# are isolated from the real mobius package layout.
# ----------------------------------------------------------------


class TestReexportResolution:
    """Tests for _parse_imports + _build_reexport_map.

    The resolver must record dependencies on the *source* module that
    actually defines a symbol, not on re-export hubs like
    ``components/__init__.py``. Wildcard and unknown symbols fall back
    to depending on the hub package.
    """

    @staticmethod
    def _write_pkg(tmp_path: Path, monkeypatch) -> Path:
        """Create a synthetic ``src/mobius`` tree.

        Layout::

            src/mobius/__init__.py
            src/mobius/components/__init__.py  # re-exports Attention, MLP
            src/mobius/components/_attention.py  # defines Attention
            src/mobius/components/_mlp.py        # defines MLP
        """
        src = tmp_path / "src" / "mobius"
        (src / "components").mkdir(parents=True)
        (src / "__init__.py").write_text("")
        (src / "components" / "__init__.py").write_text(
            "from ._attention import Attention\nfrom ._mlp import MLP\n"
        )
        (src / "components" / "_attention.py").write_text("class Attention: ...\n")
        (src / "components" / "_mlp.py").write_text("class MLP: ...\n")
        # _module_name_from_path uses _PROJECT_ROOT to resolve dotted names;
        # point it at our synthetic tree for the duration of the test.
        monkeypatch.setattr("detect_affected_models._PROJECT_ROOT", tmp_path)
        return src

    def test_resolves_symbol_to_source_module(self, tmp_path: Path, monkeypatch) -> None:
        """``from mobius.components import Attention`` → depends on _attention."""
        src = self._write_pkg(tmp_path, monkeypatch)
        importer = tmp_path / "importer.py"
        importer.write_text("from mobius.components import Attention\n")

        reexport = _build_reexport_map(src)
        imports = _parse_imports(importer, reexport)

        assert "mobius.components._attention" in imports
        # The hub package itself is NOT recorded when every symbol resolves.
        assert "mobius.components" not in imports

    def test_resolves_multiple_symbols_from_same_hub(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Each symbol in a multi-import resolves to its own source module."""
        src = self._write_pkg(tmp_path, monkeypatch)
        importer = tmp_path / "importer.py"
        importer.write_text("from mobius.components import Attention, MLP\n")

        reexport = _build_reexport_map(src)
        imports = _parse_imports(importer, reexport)

        assert "mobius.components._attention" in imports
        assert "mobius.components._mlp" in imports
        assert "mobius.components" not in imports

    def test_unknown_symbol_falls_back_to_package(self, tmp_path: Path, monkeypatch) -> None:
        """Symbols not in the re-export map fall back to the hub package."""
        src = self._write_pkg(tmp_path, monkeypatch)
        importer = tmp_path / "importer.py"
        importer.write_text("from mobius.components import NotExported\n")

        reexport = _build_reexport_map(src)
        imports = _parse_imports(importer, reexport)

        # Unknown symbol → conservative fallback on the package itself
        assert "mobius.components" in imports
        # And no spurious source-module resolution
        assert "mobius.components._attention" not in imports

    def test_wildcard_import_falls_back_to_package(self, tmp_path: Path, monkeypatch) -> None:
        """``from mobius.components import *`` cannot be resolved — fall back."""
        src = self._write_pkg(tmp_path, monkeypatch)
        importer = tmp_path / "importer.py"
        importer.write_text("from mobius.components import *\n")

        reexport = _build_reexport_map(src)
        imports = _parse_imports(importer, reexport)

        assert "mobius.components" in imports

    def test_mixed_resolved_and_unresolved_records_both(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Mix of known + unknown symbols records resolved sources AND the hub."""
        src = self._write_pkg(tmp_path, monkeypatch)
        importer = tmp_path / "importer.py"
        importer.write_text("from mobius.components import Attention, NotExported\n")

        reexport = _build_reexport_map(src)
        imports = _parse_imports(importer, reexport)

        assert "mobius.components._attention" in imports
        # Unresolved symbol keeps the hub as a conservative dependency
        assert "mobius.components" in imports

    def test_plain_import_statement_records_module(self, tmp_path: Path, monkeypatch) -> None:
        """``import mobius.components`` (no ``from``) records the module directly."""
        src = self._write_pkg(tmp_path, monkeypatch)
        importer = tmp_path / "importer.py"
        importer.write_text("import mobius.components\n")

        reexport = _build_reexport_map(src)
        imports = _parse_imports(importer, reexport)

        assert "mobius.components" in imports

    def test_no_reexport_map_records_module(self, tmp_path: Path, monkeypatch) -> None:
        """When no map is passed, the hub is always recorded (legacy behavior)."""
        self._write_pkg(tmp_path, monkeypatch)
        importer = tmp_path / "importer.py"
        importer.write_text("from mobius.components import Attention\n")

        imports = _parse_imports(importer, reexport_map=None)

        assert "mobius.components" in imports
        assert "mobius.components._attention" not in imports

    def test_non_mobius_imports_ignored(self, tmp_path: Path, monkeypatch) -> None:
        """Imports outside the mobius package are not recorded."""
        src = self._write_pkg(tmp_path, monkeypatch)
        importer = tmp_path / "importer.py"
        importer.write_text(
            "import os\nfrom typing import Any\nfrom mobius.components import Attention\n"
        )

        reexport = _build_reexport_map(src)
        imports = _parse_imports(importer, reexport)

        assert imports == {"mobius.components._attention"}

    def test_reexport_map_built_from_relative_import(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """``from .sub import X`` in __init__.py produces correct (pkg, X) entry."""
        src = self._write_pkg(tmp_path, monkeypatch)
        reexport = _build_reexport_map(src)

        assert reexport[("mobius.components", "Attention")] == ("mobius.components._attention")
        assert reexport[("mobius.components", "MLP")] == "mobius.components._mlp"


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
