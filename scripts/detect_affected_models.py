#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

r"""Detect which model or golden-case selectors are affected by changed files.

Uses AST-based static import analysis (no actual imports) to determine
which model types need testing when source files change. Designed for
CI diff-based scoping — outputs JSON for downstream workflow consumption.

Usage::

    # From git diff
    git diff --name-only origin/main...HEAD | \\
        python scripts/detect_affected_models.py --stdin

    # Explicit file list
    python scripts/detect_affected_models.py \\
        --changed-files "src/mobius/models/qwen2.py
    src/mobius/models/llama.py"

    # Output: {"affected": ["llama", "qwen2", ...], "run_all": false}
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src" / "mobius"

# ----------------------------------------------------------------
# Shared graph/runtime infrastructure paths — any change triggers run_all.
# This is intentionally limited to surfaces that can alter exported model
# graphs, runtime metadata, tokenizer assets, or GGUF qtype handling.
# ----------------------------------------------------------------
_SHARED_INFRA_PATTERNS: tuple[str, ...] = (
    "src/mobius/integrations/gguf/_runtime_evidence.py",
    "src/mobius/integrations/gguf/_runtime_package.py",
    "src/mobius/integrations/gguf/_tokenizer.py",
    "src/mobius/integrations/gguf/_builder.py",
    "src/mobius/integrations/gguf/_quant_registry.py",
    "src/mobius/integrations/gguf/_repacker.py",
    "src/mobius/integrations/gguf/_reader.py",
    "src/mobius/_builder.py",
    "src/mobius/_model_package.py",
    "src/mobius/_optimizations.py",
    "src/mobius/_weight_loading.py",
    "tests/ort_genai_e2e_test.py",
    "tests/gguf_small_model_runtime_integration_test.py",
    "testdata/cases/schema.json",
    ".github/workflows/ort_genai_e2e.yml",
    "pyproject.toml",
)

_SHARED_INFRA_PREFIXES: tuple[str, ...] = (
    "src/mobius/integrations/ort_genai/",
    "src/mobius/tasks/",
)

# Traceable infrastructure: component files that are analyzed via the
# import graph to find which models they actually affect, rather than
# triggering run_all unconditionally.
_TRACEABLE_PREFIXES = (
    "src/mobius/components/",
    "src/mobius/tasks/",
)


def classify_file(path: str) -> str:
    """Classify a changed file path.

    Returns one of: 'model', 'traceable', 'shared_infra',
    'test_config', 'golden_case', 'golden_data', 'test', 'other'.
    """
    normalized = path.replace("\\", "/")

    if normalized in _SHARED_INFRA_PATTERNS:
        return "shared_infra"
    if any(normalized.startswith(prefix) for prefix in _SHARED_INFRA_PREFIXES):
        return "shared_infra"

    if not normalized.startswith("src/mobius/"):
        # Test infrastructure files that affect all models
        if normalized == "tests/conftest.py":
            return "shared_infra"
        if normalized == "tests/_test_configs.py":
            # A config-only change is broad, but new model implementations also
            # add an entry here. Defer the run-all decision until model files
            # have been collected so those PRs can use import-graph scoping.
            return "test_config"
        if normalized.startswith("testdata/cases/") and normalized.endswith(".yaml"):
            return "golden_case"
        if normalized.startswith("testdata/golden/") and normalized.endswith(".json"):
            return "golden_data"
        if normalized.endswith("_test.py") or normalized.startswith("tests/"):
            return "test"
        return "other"

    rel = normalized[len("src/mobius/") :]

    # Test files within the source tree (check before infra prefixes)
    if rel.endswith("_test.py"):
        return "test"

    # Shared infrastructure patterns
    if normalized in _SHARED_INFRA_PATTERNS:
        return "shared_infra"
    for prefix in _SHARED_INFRA_PREFIXES:
        if normalized.startswith(prefix):
            return "shared_infra"

    # Traceable infrastructure (components) — traced via import graph
    for prefix in _TRACEABLE_PREFIXES:
        if normalized.startswith(prefix):
            return "traceable"

    # Model files
    if rel.startswith("models/") and not rel.endswith("_test.py"):
        return "model"

    return "other"


# ----------------------------------------------------------------
# AST-based import analysis
# ----------------------------------------------------------------


def _parse_imports(
    filepath: Path,
    reexport_map: dict[tuple[str, str], str] | None = None,
) -> set[str]:
    """Extract imported module names from a Python file using AST.

    Returns a set of dotted module names that appear in import
    statements. Only collects imports from within the
    mobius package.

    When ``reexport_map`` is provided, ``from pkg import sym`` statements
    are resolved through the re-export map to the underlying source
    module that defines ``sym``. This avoids spurious dependencies on
    re-export hubs like ``mobius.components/__init__.py``: a model that
    imports ``Attention`` from ``mobius.components`` is recorded as
    depending on ``mobius.components._attention`` (the actual source),
    not on the package itself. Symbols not found in the re-export map
    fall back to recording the package name.
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mobius"):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("mobius"):
                unresolved = False
                for alias in node.names:
                    if alias.name == "*":
                        # Wildcard imports can't be resolved — fall back
                        # to depending on the package itself.
                        unresolved = True
                        continue
                    source_mod = (
                        reexport_map.get((node.module, alias.name))
                        if reexport_map is not None
                        else None
                    )
                    if source_mod:
                        imports.add(source_mod)
                    else:
                        unresolved = True
                # Only record the package itself when at least one
                # imported symbol could not be resolved through the
                # re-export map. This avoids spurious dependencies on
                # re-export hubs like ``mobius.components/__init__.py``.
                if unresolved:
                    imports.add(node.module)
    return imports


def _build_reexport_map(search_dir: Path) -> dict[tuple[str, str], str]:
    """Build a (package, symbol) → source_module map from ``__init__.py`` files.

    Parses each ``__init__.py`` in the source tree for ``from .submodule
    import Symbol`` and ``from mobius.pkg.submodule import Symbol``
    statements. The resulting map lets us resolve re-exported symbols
    back to their defining module so changes to a re-export hub don't
    spuriously invalidate every importer.
    """
    reexport: dict[tuple[str, str], str] = {}
    for init_file in search_dir.rglob("__init__.py"):
        package = _module_name_from_path(init_file)
        if not package:
            continue
        try:
            source = init_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(init_file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            # Resolve relative imports like ``from . import x`` or
            # ``from .sub import X`` against the current package.
            if node.level:
                base_parts = package.split(".") if package else []
                # ``from .`` keeps us at the same package; ``from ..`` goes up.
                if node.level - 1 > len(base_parts):
                    continue
                base = ".".join(base_parts[: len(base_parts) - (node.level - 1)])
                if node.module:
                    src_module = f"{base}.{node.module}" if base else node.module
                else:
                    src_module = base
            else:
                src_module = node.module or ""
            if not src_module.startswith("mobius"):
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                exported_name = alias.asname or alias.name
                reexport[(package, exported_name)] = src_module
    return reexport


def _module_name_from_path(filepath: Path) -> str | None:
    """Convert a file path to a dotted module name.

    Returns None if the file is not under the src/ tree.
    """
    try:
        rel = filepath.resolve().relative_to(_PROJECT_ROOT / "src")
    except ValueError:
        return None

    parts = list(rel.with_suffix("").parts)
    # __init__.py represents the package itself, not a submodule
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _build_import_graph(
    search_dir: Path,
) -> dict[str, set[str]]:
    """Build a module → set[imported_modules] graph for all .py files.

    The graph maps each module name to the set of mobius
    modules it directly imports.
    """
    graph: dict[str, set[str]] = {}
    reexport_map = _build_reexport_map(search_dir)
    for pyfile in search_dir.rglob("*.py"):
        if pyfile.name.endswith("_test.py"):
            continue
        mod_name = _module_name_from_path(pyfile)
        if mod_name:
            graph[mod_name] = _parse_imports(pyfile, reexport_map)
    return graph


def _find_reverse_dependents(
    target_module: str,
    import_graph: dict[str, set[str]],
) -> set[str]:
    """Find all modules that transitively depend on target_module.

    Uses BFS on the reverse dependency graph.
    """
    # Build reverse graph: module → set of modules that import it
    reverse: dict[str, set[str]] = {}
    for mod, deps in import_graph.items():
        for dep in deps:
            reverse.setdefault(dep, set()).add(mod)

    visited: set[str] = set()
    queue: collections.deque[str] = collections.deque([target_module])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for dependent in reverse.get(current, set()):
            if dependent not in visited:
                queue.append(dependent)

    # Don't include the target itself
    visited.discard(target_module)
    return visited


def _model_file_to_module(rel_path: str) -> str | None:
    """Convert a relative model file path to a module name.

    Example: 'models/qwen.py' → 'mobius.models.qwen'
    """
    if not rel_path.endswith(".py"):
        return None
    module_path = rel_path[:-3].replace("/", ".")
    return f"mobius.{module_path}"


# ----------------------------------------------------------------
# Registry mapping: source module → model_types
#
# The registry imports classes from mobius.models (the
# package __init__), but the actual definitions live in submodules
# like models.base, models.falcon, etc. We parse models/__init__.py
# to resolve class → source submodule, then parse _registry.py to
# map class → model_types. Combined: submodule → model_types.
# ----------------------------------------------------------------


def _build_class_to_source_module() -> dict[str, str]:
    """Parse models/__init__.py to map class names to source submodules.

    E.g. CausalLMModel → mobius.models.base
         FalconCausalLMModel → mobius.models.falcon
    """
    init_file = _SRC_ROOT / "models" / "__init__.py"
    class_to_source: dict[str, str] = {}

    try:
        source = init_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(init_file))
    except (SyntaxError, UnicodeDecodeError):
        return class_to_source

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("mobius.models."):
                for alias in node.names:
                    class_to_source[alias.name] = node.module

    return class_to_source


def _build_registry_class_to_types() -> dict[str, list[str]]:
    """Parse _registry.py to map class names to registered model_types.

    Parses the declarative ``_REGISTRATIONS`` dict::

        _REGISTRATIONS = {"name": ModelRegistration(ClassName, ...)}
    """
    registry_file = _SRC_ROOT / "_registry.py"
    class_to_types: dict[str, list[str]] = {}

    try:
        source = registry_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(registry_file))
    except (SyntaxError, UnicodeDecodeError):
        return class_to_types

    for node in ast.walk(tree):
        # _REGISTRATIONS = {"name": ModelRegistration(Cls, ...)}
        # Handles both plain assignment and type-annotated assignment
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_REGISTRATIONS":
                    if isinstance(node.value, ast.Dict):
                        _process_registrations_dict(node.value, class_to_types)
        if isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "_REGISTRATIONS"
                and isinstance(node.value, ast.Dict)
            ):
                _process_registrations_dict(node.value, class_to_types)

    return {c: sorted(set(t)) for c, t in class_to_types.items()}


def _process_registrations_dict(
    dict_node: ast.Dict,
    class_to_types: dict[str, list[str]],
) -> None:
    """Extract model_type → class from _REGISTRATIONS = {"name": ModelRegistration(Cls)}."""
    for key, value in zip(dict_node.keys, dict_node.values):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        arch_name = key.value
        # value is ModelRegistration(ClassName, ...) — extract the first arg
        if isinstance(value, ast.Call) and value.args:
            cls_arg = value.args[0]
            if isinstance(cls_arg, ast.Name):
                class_to_types.setdefault(cls_arg.id, []).append(arch_name)


def _build_source_module_to_types() -> dict[str, list[str]]:
    """Build the final source_module → [model_types] mapping.

    Combines __init__.py class→source resolution with _registry.py
    class→model_types mapping to produce submodule→model_types.
    """
    class_to_source = _build_class_to_source_module()
    class_to_types = _build_registry_class_to_types()

    # Also collect direct imports from _registry.py itself
    # (for classes imported from submodules directly, not via __init__)
    registry_file = _SRC_ROOT / "_registry.py"
    try:
        source = registry_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(registry_file))
    except (SyntaxError, UnicodeDecodeError):
        tree = None

    if tree:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("mobius.models."):
                    for alias in node.names:
                        # Only add if not already resolved
                        if alias.name not in class_to_source:
                            class_to_source[alias.name] = node.module

    # Now combine: source_module → model_types
    module_to_types: dict[str, list[str]] = {}
    for cls_name, types in class_to_types.items():
        source_mod = class_to_source.get(cls_name)
        if source_mod:
            module_to_types.setdefault(source_mod, []).extend(types)

    return {m: sorted(set(t)) for m, t in module_to_types.items()}


# ----------------------------------------------------------------
# Main detection logic
# ----------------------------------------------------------------


def detect_affected_models(
    changed_files: list[str],
) -> dict[str, list[str] | bool]:
    """Determine which model_types are affected by file changes.

    Args:
        changed_files: List of file paths relative to repository root.

    Returns:
        Dict with keys:
            - "affected": sorted list of affected model_type strings
            - "run_all": True if all models should be tested
    """
    affected: set[str] = set()
    run_all = False
    test_config_changed = False
    golden_case_files: list[str] = []
    _, _, golden_detection_failed = _detect_changed_golden_cases(changed_files)
    if golden_detection_failed:
        return {"affected": [], "run_all": True}

    # Classify files
    model_files: list[str] = []
    traceable_files: list[str] = []
    for path in changed_files:
        category = classify_file(path)
        if category == "shared_infra":
            run_all = True
            break
        elif category == "test_config":
            test_config_changed = True
        elif category == "golden_case":
            golden_case_files.append(path)
        elif category == "model":
            # Deleted model files could break dependents — run all
            full_path = _PROJECT_ROOT / path
            if not full_path.exists():
                run_all = True
                break
            model_files.append(path)
        elif category == "traceable":
            full_path = _PROJECT_ROOT / path
            if not full_path.exists():
                run_all = True
                break
            traceable_files.append(path)

    if run_all:
        return {"affected": [], "run_all": True}

    if test_config_changed and not model_files:
        return {"affected": [], "run_all": True}

    for path in golden_case_files:
        full_path = _PROJECT_ROOT / path
        if not full_path.exists():
            return {"affected": [], "run_all": True}

    if not model_files and not traceable_files:
        return {"affected": sorted(affected), "run_all": False}

    # Build the registry map: source_module → [model_types]
    registry_map = _build_source_module_to_types()

    # Build import graph for transitive analysis
    import_graph = _build_import_graph(_SRC_ROOT)

    # Process model files: direct mapping + transitive dependents
    for path in model_files:
        normalized = path.replace("\\", "/")
        rel = normalized[len("src/mobius/") :]
        module_name = _model_file_to_module(rel)
        if not module_name:
            continue

        # A concrete model module should only scope to the model_types it
        # directly registers. Reverse-dependency expansion is intentionally
        # skipped here so a change in one model implementation does not
        # incorrectly cascade to sibling models that happen to import shared
        # helpers from it. Shared-base modules (e.g. ``models/base.py``) do
        # not map directly to registered model_types, so they still fall
        # through to the transitive scan below.
        if module_name in registry_map:
            affected.update(registry_map[module_name])

            # Also include directly registered model modules that import this one.
            # Some architectures are implemented as thin wrappers over another
            # model module (e.g. importing a base model class).
            direct_dependents = {
                mod for mod, deps in import_graph.items() if module_name in deps
            }
            for dep_module in direct_dependents:
                if dep_module in registry_map:
                    affected.update(registry_map[dep_module])
            continue

        # Transitive: find modules that import from this model file.
        # This is only used for infrastructure/shared-model modules without a
        # direct registration. Example: a base class file used by many models.
        dependents = _find_reverse_dependents(module_name, import_graph)
        for dep_module in dependents:
            if dep_module in registry_map:
                affected.update(registry_map[dep_module])

    # Process traceable files (components, tasks): find which models
    # transitively import them, then map to registered model_types.
    for path in traceable_files:
        normalized = path.replace("\\", "/")
        # Convert path to module name: src/mobius/components/_attention.py
        # → mobius.components._attention
        # Special case: __init__.py → package name (mobius.components)
        rel = normalized[len("src/") :]
        if rel.endswith("/__init__.py"):
            module_name = rel[: -len("/__init__.py")].replace("/", ".")
        else:
            module_name = rel[:-3].replace("/", ".")  # strip .py
        if not module_name:
            continue

        dependents = _find_reverse_dependents(module_name, import_graph)
        for dep_module in dependents:
            if dep_module in registry_map:
                affected.update(registry_map[dep_module])

    return {"affected": sorted(affected), "run_all": False}


def _read_golden_case_levels(yaml_path: Path) -> set[str] | None:
    """Read one top-level L4/L5 declaration without requiring PyYAML."""
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("level:"):
            continue
        value = line.partition(":")[2].strip().strip("\"'")
        if value == "L4":
            return {"L4"}
        if value == "L5":
            return {"L5"}
        if value == "L4+L5":
            return {"L4", "L5"}
        return None
    return None


def _detect_changed_golden_cases(
    changed_files: list[str],
) -> tuple[list[str], list[str], bool]:
    """Return level-specific exact case IDs and whether detection failed closed."""
    l4_cases: set[str] = set()
    l5_cases: set[str] = set()
    cases_root = _PROJECT_ROOT / "testdata" / "cases"

    for path in changed_files:
        category = classify_file(path)
        if category not in {"golden_case", "golden_data"}:
            continue
        full_path = _PROJECT_ROOT / path
        if not full_path.exists():
            return [], [], True

        if category == "golden_case":
            yaml_path = full_path
            case_id = full_path.stem
            is_generation_data = False
            if len(list(cases_root.rglob(f"{case_id}.yaml"))) != 1:
                return [], [], True
        else:
            stem = full_path.stem
            is_generation_data = stem.endswith("_generation")
            case_id = stem.removesuffix("_generation")
            matches = sorted(cases_root.rglob(f"{case_id}.yaml"))
            if len(matches) != 1:
                return [], [], True
            yaml_path = matches[0]

        levels = _read_golden_case_levels(yaml_path)
        if levels is None or (is_generation_data and "L5" not in levels):
            return [], [], True
        if "L4" in levels and not is_generation_data:
            l4_cases.add(case_id)
        if "L5" in levels:
            l5_cases.add(case_id)

    return sorted(l4_cases), sorted(l5_cases), False


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Detect model_types affected by changed files"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--changed-files",
        help="Newline-separated list of changed file paths",
    )
    group.add_argument(
        "--stdin",
        action="store_true",
        help="Read changed file paths from stdin (one per line)",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "github"],
        default="json",
        help=(
            "Output format. 'json' prints the full result dict. "
            "'github' sets GitHub Actions output variables."
        ),
    )
    args = parser.parse_args()

    if args.stdin:
        changed = [line.strip() for line in sys.stdin if line.strip()]
    else:
        changed = [line.strip() for line in args.changed_files.split("\n") if line.strip()]

    result = detect_affected_models(changed)
    golden_l4_cases, golden_l5_cases, golden_detection_failed = _detect_changed_golden_cases(
        changed
    )
    if golden_detection_failed:
        result = {"affected": [], "run_all": True}
    if result["run_all"]:
        golden_l4_cases = []
        golden_l5_cases = []

    if args.output_format == "github":
        # Output for GitHub Actions
        affected_json = json.dumps(result["affected"])
        golden_l4_cases_json = json.dumps(golden_l4_cases)
        golden_l5_cases_json = json.dumps(golden_l5_cases)
        run_all = "true" if result["run_all"] else "false"
        has_model_affected = bool(result["run_all"] or result["affected"])
        has_l4_affected = "true" if has_model_affected or golden_l4_cases else "false"
        has_l5_affected = "true" if has_model_affected or golden_l5_cases else "false"
        has_affected = (
            "true"
            if (result["run_all"] or result["affected"] or golden_l4_cases or golden_l5_cases)
            else "false"
        )
        print(f"affected={affected_json}")
        print(f"golden_l4_cases={golden_l4_cases_json}")
        print(f"golden_l5_cases={golden_l5_cases_json}")
        print(f"run_all={run_all}")
        print(f"has_affected={has_affected}")
        print(f"has_l4_affected={has_l4_affected}")
        print(f"has_l5_affected={has_l5_affected}")
    else:
        print(
            json.dumps(
                {
                    **result,
                    "golden_l4_cases": golden_l4_cases,
                    "golden_l5_cases": golden_l5_cases,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
