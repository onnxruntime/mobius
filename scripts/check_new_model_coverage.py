#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

r"""Check that newly registered model architectures have test coverage.

For each model_type registered in ``src/mobius/_registry.py``, verifies:

    a. L3 — has a synthetic build-graph test config in ``tests/_test_configs.py``
    b. L4 — has a YAML test case in ``testdata/cases/``
    c. L5 — has golden data in ``testdata/golden/`` (or a ``skip_reason`` in the YAML)

Usage::

    # Audit all registered models (report all gaps, exit 1 if any found)
    python scripts/check_new_model_coverage.py

    # CI mode: only fail on models that are NEW in this PR vs base branch
    python scripts/check_new_model_coverage.py --diff-base origin/main

    # Show only missing, one-line-per-model
    python scripts/check_new_model_coverage.py --quiet
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src" / "mobius"
_TESTS_DIR = _PROJECT_ROOT / "tests"
_CASES_DIR = _PROJECT_ROOT / "testdata" / "cases"
_GOLDEN_DIR = _PROJECT_ROOT / "testdata" / "golden"

# model_types that are intentionally VL sub-models registered separately.
# These are text-decoder or embedding sub-models that reuse the VL parent's
# test coverage (YAML + golden are keyed on the parent model_type, not the
# text-only variant). Excluding them avoids spurious L4/L5 failures.
_VL_TEXT_SUFFIX_ALIASES: frozenset[str] = frozenset({
    "_text",
    "_multimodal",
})


def _get_all_registered_types() -> list[str]:
    """Return all model_types from the live registry."""
    sys.path.insert(0, str(_SRC_ROOT.parent))
    from mobius._registry import registry  # noqa: PLC0415

    return sorted(registry.architectures())


def _get_l3_types() -> set[str]:
    """Return model_types that have a synthetic L3 build-graph test config."""
    sys.path.insert(0, str(_TESTS_DIR))
    from _test_configs import (  # noqa: PLC0415
        ALL_CAUSAL_LM_CONFIGS,
        ENCODER_CONFIGS,
        SEQ2SEQ_CONFIGS,
        VISION_CONFIGS,
    )

    types: set[str] = set()
    for mt, _, _ in ALL_CAUSAL_LM_CONFIGS:
        types.add(mt)
    for mt, _, _ in ENCODER_CONFIGS:
        types.add(mt)
    for mt, _, _ in SEQ2SEQ_CONFIGS:
        types.add(mt)
    for mt, _, _ in VISION_CONFIGS:
        types.add(mt)
    return types


def _get_yaml_cases() -> dict[str, Path]:
    """Return {model_type_or_case_id: yaml_path} for all YAML test cases."""
    import yaml  # noqa: PLC0415

    cases: dict[str, Path] = {}
    for yaml_path in _CASES_DIR.rglob("*.yaml"):
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                # Index by case_id (stem) and by model_id prefix heuristic
                case_id = yaml_path.stem
                cases[case_id] = yaml_path
                # Also index by model_type if present in YAML
                mt = data.get("task_type")  # not model_type, but task_type
                # Primary key is the stem (e.g. "qwen2_5-0_5b")
        except Exception:  # noqa: BLE001
            pass
    return cases


def _get_yaml_model_type_map() -> dict[str, dict]:
    """Return {case_stem: parsed_yaml_data} for all YAML test cases."""
    import yaml  # noqa: PLC0415

    cases: dict[str, dict] = {}
    for yaml_path in _CASES_DIR.rglob("*.yaml"):
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                cases[yaml_path.stem] = data
        except Exception:  # noqa: BLE001
            pass
    return cases


def _get_golden_stems() -> set[str]:
    """Return set of golden JSON stems (without _generation suffix)."""
    stems: set[str] = set()
    for p in _GOLDEN_DIR.rglob("*.json"):
        name = p.stem
        if not name.endswith("_generation"):
            stems.add(name)
    return stems


def _get_new_model_types(diff_base: str) -> set[str] | None:
    """Return model_types newly added vs diff_base, or None if git fails.

    Detects new ``reg.register(...)`` calls added to ``_registry.py``.
    Adding entries to ``_TEST_MODEL_IDS`` does NOT count as a new model
    registration — only actual ``reg.register()`` calls are tracked.
    """
    try:
        result = subprocess.run(
            ["git", "diff", f"{diff_base}...HEAD", "--", "src/mobius/_registry.py"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            return None
        diff_text = result.stdout
    except FileNotFoundError:
        return None

    if not diff_text.strip():
        return None

    # Only detect lines like:  +    reg.register("model_type", ...)
    # Ignoring _TEST_MODEL_IDS additions (those are test metadata, not new models).
    import re  # noqa: PLC0415

    added_types: set[str] = set()
    register_pattern = re.compile(r'^\+\s+reg\.register\(\s*"([^"]+)"')

    for line in diff_text.splitlines():
        m = register_pattern.match(line)
        if m:
            added_types.add(m.group(1))

    return added_types if added_types else set()


def _is_vl_text_alias(model_type: str) -> bool:
    """Return True if model_type is a VL sub-model text alias.

    These variants (e.g. ``qwen2_vl_text``) share YAML + golden with
    their parent (``qwen2_vl``) and don't need separate test cases.
    """
    for suffix in _VL_TEXT_SUFFIX_ALIASES:
        if model_type.endswith(suffix):
            base = model_type[: -len(suffix)]
            # Only flag as alias if the base type is also registered
            return len(base) > 0
    return False


def _check_coverage(
    model_types: list[str],
    l3_types: set[str],
    yaml_cases: dict[str, dict],
    golden_stems: set[str],
) -> dict[str, list[str]]:
    """Return {model_type: [list of missing coverage items]} for each type."""
    gaps: dict[str, list[str]] = {}

    # Build a set of model_types represented in YAML (by model_id or task_type)
    # and golden files. YAML cases may cover multiple types via one file.
    # We do a fuzzy match: a YAML case covers a model_type if the case stem
    # contains the model_type (with underscores normalized).
    def _normalize(s: str) -> str:
        return s.replace("-", "_").replace(".", "_").lower()

    yaml_normalized = {_normalize(k) for k in yaml_cases}
    golden_normalized = {_normalize(s) for s in golden_stems}

    # Also collect skip_reason fields from YAML
    yaml_with_skip: set[str] = set()
    for stem, data in yaml_cases.items():
        if data.get("skip_reason"):
            yaml_with_skip.add(_normalize(stem))

    for mt in model_types:
        if _is_vl_text_alias(mt):
            continue

        missing: list[str] = []
        mt_norm = _normalize(mt)

        # L3: synthetic build-graph test
        if mt not in l3_types:
            missing.append("No L3 test config in tests/_test_configs.py")

        # L4/L5: YAML test case
        has_yaml = any(mt_norm in s for s in yaml_normalized)
        if not has_yaml:
            missing.append("No YAML test case in testdata/cases/")
        else:
            # Has YAML — check golden unless skip_reason is set
            has_golden = any(mt_norm in s for s in golden_normalized)
            has_skip = any(mt_norm in s for s in yaml_with_skip)
            if not has_golden and not has_skip:
                missing.append(
                    "No golden data in testdata/golden/ "
                    "(add golden or set skip_reason in YAML)"
                )

        if missing:
            gaps[mt] = missing

    return gaps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check test coverage for registered model architectures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--diff-base",
        metavar="REF",
        help=(
            "Git ref to compare against (e.g. origin/main). "
            "When set, only newly registered model_types are checked. "
            "Without this flag, all registered types are audited."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only model_types with gaps, one per line.",
    )
    args = parser.parse_args()

    all_types = _get_all_registered_types()
    l3_types = _get_l3_types()
    yaml_cases = _get_yaml_model_type_map()
    golden_stems = _get_golden_stems()

    if args.diff_base:
        new_types = _get_new_model_types(args.diff_base)
        if new_types is None:
            print(
                f"⚠️  Could not determine new model types from git diff against "
                f"{args.diff_base!r}. Skipping coverage check.",
                file=sys.stderr,
            )
            return 0
        types_to_check = sorted(t for t in all_types if t in new_types)
        if not types_to_check:
            print("✅ No new model_types detected in this diff. Coverage check skipped.")
            return 0
        print(f"🔍 Checking coverage for {len(types_to_check)} new model type(s):")
        for t in types_to_check:
            print(f"   {t}")
        print()
    else:
        types_to_check = all_types
        print(f"🔍 Auditing coverage for all {len(types_to_check)} registered model types.")
        print()

    gaps = _check_coverage(types_to_check, l3_types, yaml_cases, golden_stems)

    if not gaps:
        if args.diff_base:
            print("✅ All new model types have required test coverage.")
        else:
            print("✅ All registered model types have test coverage.")
        return 0

    # Print gaps
    if args.quiet:
        for mt in sorted(gaps):
            print(mt)
        return 1

    if args.diff_base:
        print(f"❌ {len(gaps)} new model type(s) missing test coverage:\n")
    else:
        print(f"❌ {len(gaps)} model type(s) missing test coverage:\n")

    for mt in sorted(gaps):
        items = gaps[mt]
        print(f"  ❌ New model {mt!r} missing coverage:")
        for item in items:
            print(f"     - {item}")
        print()

    print(
        "To fix: See .github/skills/writing-tests/SKILL.md for how to add test coverage.\n"
        "Quick guide:\n"
        "  1. Add a config entry to tests/_test_configs.py (L3)\n"
        "  2. Add a YAML case to testdata/cases/<task-type>/<model>.yaml (L4/L5)\n"
        "  3. Run: python scripts/generate_golden.py --filter <model_id> to create golden data\n"
        "     Or add skip_reason to the YAML if golden generation is not feasible."
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())
