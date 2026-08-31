#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

r"""Generate a static HTML confidence dashboard for model testing coverage.

Scans the model registry, test configurations, and test files to determine
confidence levels per model type. Outputs a self-contained HTML dashboard
with no external dependencies.

Usage::

    python scripts/generate_dashboard.py --output docs/dashboard/index.html
    python scripts/generate_dashboard.py --output docs/dashboard/index.html \
        --commit $(git rev-parse --short HEAD)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jinja2


def _json_safe(obj: Any) -> Any:
    """Convert an object to a JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(v) for v in obj)
    if isinstance(obj, type):
        return obj.__name__
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _json_safe(dataclasses.asdict(obj))
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# Ensure the source package is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "tests"))


@dataclasses.dataclass
class ModelInfo:
    """Collected information about a single registered model type."""

    model_type: str
    module_class_name: str
    task: str
    category: str
    family: str
    # Confidence levels (True if covered at that level).
    # L1 is a single boolean; L2/L3/L4/L5 use a "test configured" boolean
    # PLUS a richer ``<level>_status`` field ("pass" / "xfail" / "skip" /
    # None) so the dashboard can distinguish passes, expected failures,
    # documented skips, and awaiting-data states uniformly.
    l1_graph_build: bool = False
    l2_arch_validation: bool = False
    l3_synthetic_parity: bool = False
    # Code paths exercised by test configs
    code_paths: set[str] = dataclasses.field(default_factory=set)
    # Config overrides from test configs (for drill-down)
    config_overrides: list[dict[str, Any]] = dataclasses.field(
        default_factory=list,
    )
    # Whether the model has integration tests
    has_integration_test: bool = False
    # Whether the model has a test_model_id
    test_model_id: str | None = None
    # Golden test case coverage (from testdata/cases/ YAML files).
    # ``l{4,5}_has_test_case`` is the "test configured" signal that
    # parallels ``l2_arch_validation`` / ``l3_synthetic_parity`` and is
    # True when a YAML case exists for that level (skipped cases do NOT
    # set this; the skipped state is represented via ``l{4,5}_status``).
    l4_has_test_case: bool = False
    l5_has_test_case: bool = False
    yaml_test_case_file: str | None = None
    yaml_test_case_skip_reason: str | None = None
    yaml_test_case_ci_skip_reason: str | None = None
    yaml_min_token_match_ratio: float | None = None
    # YAML ``task_type`` field — used to construct the pytest test ID
    # ``"{task_type}/{case_id}"`` for matching against L4/L5 xfail dicts
    # in ``e2e_golden_test.py``. May differ from the parent directory name.
    yaml_task_type: str | None = None
    # Per-level status fields. Each follows the same {"pass", "xfail",
    # "skip", None} contract so the template can dispatch uniformly.
    # L2 also admits "xfail_graph_only" (the config-parse subtest passes
    # but the full-graph build subtest xfails). All flavors of xfail
    # count as not-passing.
    l2_status: str | None = None
    l2_status_reason: str | None = None
    l3_status: str | None = None
    l3_status_reason: str | None = None
    # L4 status comes from yaml skip_reason, the e2e xfail dicts, or the
    # presence of a matching golden file on disk (in that precedence
    # order — skip > xfail > pass). ``None`` with ``l4_has_test_case=True``
    # means the YAML case exists but no golden has been generated yet
    # (the "awaiting data" state).
    l4_status: str | None = None
    l4_status_reason: str | None = None
    l5_status: str | None = None
    l5_status_reason: str | None = None

    @property
    def l2_passes(self) -> bool:
        """True iff L2 architecture validation is configured and expected to pass.

        L2 distinguishes "test exists" (``l2_arch_validation``) from "test
        passes" (``l2_status``) because ``arch_validation_test.py`` xfails
        subtests for models with known config / graph-build limitations
        (e.g. VL models needing ``trust_remote_code``).
        """
        return self.l2_arch_validation and self.l2_status == "pass"

    @property
    def l3_passes(self) -> bool:
        """True iff L3 synthetic parity is configured and expected to pass."""
        return self.l3_synthetic_parity and self.l3_status == "pass"

    @property
    def l4_passes(self) -> bool:
        """True iff L4 golden test is configured and expected to pass.

        Mirrors :attr:`l2_passes` / :attr:`l3_passes`. An xfailed L4 test
        has a golden file on disk (it is the HF reference output) but the
        test is expected to fail, so it must not count as passing.
        """
        return self.l4_has_test_case and self.l4_status == "pass"

    @property
    def l5_passes(self) -> bool:
        """True iff L5 generation test is configured and expected to pass."""
        return self.l5_has_test_case and self.l5_status == "pass"

    @property
    def confidence_level(self) -> int:
        """The highest confidence level achieved (0-5)."""
        if self.l5_passes:
            return 5
        if self.l4_passes:
            return 4
        if self.l3_passes:
            return 3
        if self.l2_passes:
            return 2
        if self.l1_graph_build:
            return 1
        return 0

    @property
    def confidence_label(self) -> str:
        """Human-readable confidence label."""
        return _CONFIDENCE_LABELS.get(self.confidence_level, "Not Tested")


_CONFIDENCE_LABELS = {
    0: "Not Tested",
    1: "L1: Graph Builds",
    2: "L2: Config Compatible",
    3: "L3: Synthetic Parity",
    4: "L4: Golden Match",
    5: "L5: Generation Verified",
}

# Task strings from the registry to dashboard category mapping.
# These include model-specific task names (e.g. "qwen-vl") because the
# registry uses custom task strings for specialized pipelines.
_TASK_CATEGORY_MAP = {
    "text-generation": "Causal LM",
    "feature-extraction": "Encoder",
    "seq2seq": "Seq2Seq",
    "image-classification": "Vision",
    "object-detection": "Detection",
    "audio-feature-extraction": "Audio",
    "speech-to-text": "Speech",
    "speech-language": "Speech",
    "codec": "Speech",
    "vision-language": "Vision-Language",
    "qwen-vl": "Vision-Language",
    "qwen3-vl-vision-language": "Vision-Language",
    "hybrid-qwen-vl": "Vision-Language",
    "mllama-vision-language": "Vision-Language",
    "multimodal": "Multimodal",
}


def _derive_family(model_type: str) -> str:
    """Derive a family name from a model type using prefix heuristic.

    Groups model types that share a common prefix into families.
    E.g., qwen2, qwen2_moe, qwen2_vl → "qwen2".
    """
    for prefix in _FAMILY_PREFIXES:
        if model_type.startswith(prefix):
            return prefix
    # Default: use the whole model_type as the family
    return model_type


# Common prefixes for model families (ordered longest-first so that
# "qwen3_5" matches before "qwen3").
_FAMILY_PREFIXES = [
    "qwen3_5",
    "qwen3",
    "qwen2_5",
    "qwen2",
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_vl",
    "gemma3n",
    "gemma3",
    "gemma2",
    "internlm",
    "internvl",
    "phi4",
    "phi3",
    "olmo",
    "llava_next",
    "llava",
    "glm4v",
    "glm4",
    "wav2vec2",
    "data2vec",
    "falcon",
    "blenderbot",
    "roberta",
    "xlm",
    "gpt_neo",
    "gpt",
    "vit",
    "swin",
    "dinov2",
    "siglip",
]


def _scan_registry() -> dict[str, ModelInfo]:
    """Scan the model registry and build initial ModelInfo entries."""
    from mobius._registry import registry

    models: dict[str, ModelInfo] = {}
    for arch in registry.architectures():
        reg = registry.get_registration(arch)
        task = reg.task or "text-generation"
        category = _TASK_CATEGORY_MAP.get(task, "Other")
        family_override = getattr(reg, "family", None)
        family = family_override or _derive_family(arch)
        test_model_id = getattr(reg, "test_model_id", None)

        models[arch] = ModelInfo(
            model_type=arch,
            module_class_name=reg.module_class.__name__,
            task=task,
            category=category,
            family=family,
            test_model_id=test_model_id,
        )
    return models


def _scan_l1_configs(models: dict[str, ModelInfo]) -> None:
    """Mark L1 coverage from test config presence in _test_configs.py.

    Also marks models in ``_SPECIALIZED_TEST_MODEL_TYPES`` (VLM/audio models
    tested via dedicated test methods rather than the parametrized config loop).
    """
    from mobius._testing.code_paths import (
        detect_code_paths,
    )

    try:
        from _test_configs import ALL_CONFIGS
    except ImportError:
        print(
            "Warning: Could not import _test_configs. L1 detection skipped.",
            file=sys.stderr,
        )
        return

    for model_type, config_overrides, _is_repr in ALL_CONFIGS:
        if model_type in models:
            models[model_type].l1_graph_build = True
            models[model_type].config_overrides.append(config_overrides)
            paths = detect_code_paths(config_overrides)
            models[model_type].code_paths.update(paths)

    # Specialized models have dedicated graph-construction tests but are not
    # in ALL_CONFIGS. They still build a graph.
    try:
        from build_graph._support import _SPECIALIZED_TEST_MODEL_TYPES
    except ImportError:
        return

    for model_type in _SPECIALIZED_TEST_MODEL_TYPES:
        if model_type in models:
            models[model_type].l1_graph_build = True


def _parse_status_dict(
    content: str,
    dict_name: str,
    key_pattern: str = r"[\w-]+",
) -> dict[str, str]:
    r"""Extract entries from a ``{name: dict[str, str] = {...}}`` literal.

    Uses a regex pass over the source text so the dashboard generator does
    not have to import the test file (avoiding side effects from pytest
    collection). Keys may contain hyphens (e.g. ``"xlm-roberta"``,
    ``"data2vec-text"``) so the default key character class is ``[\w-]+``.
    Callers that expect richer keys (e.g. ``"text-generation/helium-1-2b"``
    in the e2e golden xfail dicts) can pass a custom ``key_pattern`` such
    as ``[^"]+``.

    The dict name must appear at the start of a line (after optional
    leading whitespace) so a stray docstring mention of the name does not
    cause the regex to skip ahead and match a different dict's body.
    Only ``[^=\n]*`` is allowed between the name and ``=`` (i.e. type
    annotations on the same line), which prevents bridging across
    multiple statements.
    """
    import re

    block = re.search(
        rf"^\s*{dict_name}\b[^=\n]*=\s*\{{(.*?)\}}",
        content,
        re.DOTALL | re.MULTILINE,
    )
    if block is None:
        return {}
    return {
        m.group(1): m.group(2)
        for m in re.finditer(rf'"({key_pattern})":\s*"([^"]+)"', block.group(1))
    }


def _scan_l2_arch_tests(models: dict[str, ModelInfo]) -> None:
    """Mark L2 coverage from arch_validation_test.py presence.

    Sets ``l2_arch_validation`` for models that are *parametrized into* the
    L2 test (either via ``test_model_id`` in the registry or by appearing
    by name in the test file). This is the "test exists" signal; the
    "test passes" signal is set by :func:`_scan_l2_arch_status`.
    """
    arch_test = _REPO_ROOT / "tests" / "arch_validation_test.py"
    if not arch_test.exists():
        return

    content = arch_test.read_text(encoding="utf-8")
    for model_type, info in models.items():
        # L2 requires test_model_id in registry
        if info.test_model_id:
            info.l2_arch_validation = True
        # Also check if model_type appears in the test file
        elif f'"{model_type}"' in content or f"'{model_type}'" in content:
            info.l2_arch_validation = True


def _scan_l2_arch_status(models: dict[str, ModelInfo]) -> None:
    """Extract L2 xfail status from arch_validation_test.py.

    Parses ``_PARSE_AND_GRAPH_XFAILS`` and ``_GRAPH_ONLY_XFAILS`` dicts to
    determine per-model status without running the tests. Models that have
    L2 coverage but are not in either dict are marked ``"pass"``.
    """
    arch_test = _REPO_ROOT / "tests" / "arch_validation_test.py"
    if not arch_test.exists():
        return

    content = arch_test.read_text(encoding="utf-8")

    parse_and_graph_xfails = _parse_status_dict(content, "_PARSE_AND_GRAPH_XFAILS")
    graph_only_xfails = _parse_status_dict(content, "_GRAPH_ONLY_XFAILS")

    for model_type, info in models.items():
        if not info.l2_arch_validation:
            continue
        if model_type in parse_and_graph_xfails:
            info.l2_status = "xfail"
            info.l2_status_reason = parse_and_graph_xfails[model_type]
        elif model_type in graph_only_xfails:
            info.l2_status = "xfail_graph_only"
            info.l2_status_reason = graph_only_xfails[model_type]
        else:
            info.l2_status = "pass"


def _scan_l3_synthetic_parity(models: dict[str, ModelInfo]) -> None:
    """Mark L3 coverage from synthetic_parity_test.py.

    Imports the actual test config lists used by the parametrized tests
    (``ALL_CAUSAL_LM_CONFIGS``, ``ENCODER_CONFIGS``, ``SEQ2SEQ_CONFIGS``)
    to detect which model_types have L3 coverage. This avoids false
    positives from string matching (e.g. skipped models mentioned in
    ``_SKIP_REASONS``) and false negatives from models parametrized via
    imported config lists rather than inline ``pytest.mark.parametrize``
    strings.
    """
    parity_test = _REPO_ROOT / "tests" / "synthetic_parity_test.py"
    if not parity_test.exists():
        return

    # Import actual configs from the test support module.
    # This is the authoritative source of which model_types are tested.
    try:
        sys.path.insert(0, str(_REPO_ROOT / "tests"))
        from _test_configs import (
            ALL_CAUSAL_LM_CONFIGS,
            ENCODER_CONFIGS,
            SEQ2SEQ_CONFIGS,
        )

        l3_model_types = {mt for mt, _ov, _rep in ALL_CAUSAL_LM_CONFIGS}
        l3_model_types.update(mt for mt, _ov, _rep in ENCODER_CONFIGS)
        l3_model_types.update(mt for mt, _ov, _rep in SEQ2SEQ_CONFIGS)
    except ImportError:
        # Fallback: if import fails, do nothing rather than false-positive
        return
    finally:
        sys.path.pop(0)

    for model_type in models:
        if model_type in l3_model_types:
            models[model_type].l3_synthetic_parity = True


def _scan_l4_golden_files(models: dict[str, ModelInfo]) -> None:
    """Mark L4 ``status='pass'`` for models with a golden file on disk.

    Two matching strategies:
    1. Direct: ``golden/<category>/<model_type>.json`` — works when the golden
       file stem equals the registry model_type.
    2. Indirect: when a model has a YAML test case, derive the expected golden
       path from the case_id (the YAML file stem). This handles cases like
       ``golden/vision-language/qwen2_5-vl-3b.json`` → model_type ``qwen2_5_vl``.

    Skipped cases (``l4_status='skip'`` set earlier by
    :func:`_scan_yaml_test_cases`) are left untouched: a golden may exist
    on disk for a skipped model, but the skip wins.
    """
    golden_dir = _REPO_ROOT / "testdata" / "golden"
    if not golden_dir.exists():
        return

    # NOTE: relies on _scan_yaml_test_cases having run first (see
    # collect_all_model_info call order) so any skip status is in place.

    # Strategy 1: direct stem → model_type match
    for golden_file in golden_dir.rglob("*.json"):
        if "_generation" in golden_file.name:
            continue
        model_type = golden_file.stem
        info = models.get(model_type)
        if info is None or info.l4_status is not None:
            continue
        info.l4_status = "pass"

    # Strategy 2: YAML-derived path (case_id may differ from model_type)
    for info in models.values():
        if info.l4_status is not None or not info.yaml_test_case_file:
            continue
        case_path = _REPO_ROOT / info.yaml_test_case_file
        case_id = case_path.stem
        task_dir = case_path.parent.name
        golden_path = golden_dir / task_dir / f"{case_id}.json"
        if golden_path.exists():
            info.l4_status = "pass"


def _scan_l5_generation_golden(models: dict[str, ModelInfo]) -> None:
    """Mark L5 ``status='pass'`` for models with a generation golden on disk.

    Uses the same two-strategy matching as :func:`_scan_l4_golden_files`.
    """
    golden_dir = _REPO_ROOT / "testdata" / "golden"
    if not golden_dir.exists():
        return

    # Strategy 1: direct stem → model_type match
    for golden_file in golden_dir.rglob("*_generation.json"):
        model_type = golden_file.stem.removesuffix("_generation")
        info = models.get(model_type)
        if info is None or info.l5_status is not None:
            continue
        info.l5_status = "pass"

    # Strategy 2: YAML-derived path (case_id may differ from model_type)
    for info in models.values():
        if info.l5_status is not None or not info.yaml_test_case_file:
            continue
        case_path = _REPO_ROOT / info.yaml_test_case_file
        case_id = case_path.stem
        task_dir = case_path.parent.name
        gen_path = golden_dir / task_dir / f"{case_id}_generation.json"
        if gen_path.exists():
            info.l5_status = "pass"


def _scan_integration_tests(models: dict[str, ModelInfo]) -> None:
    """Mark models that have integration tests."""
    tests_dir = _REPO_ROOT / "tests"
    integration_files = list(tests_dir.glob("*integration*.py"))

    for test_file in integration_files:
        content = test_file.read_text(encoding="utf-8")
        for model_type in models:
            if f'"{model_type}"' in content or f"'{model_type}'" in content:
                models[model_type].has_integration_test = True


def _scan_yaml_test_cases(models: dict[str, ModelInfo]) -> None:
    """Scan testdata/cases/ for YAML test case files and mark L4/L5 coverage.

    Resolution order for mapping a YAML case to one or more registry
    model_types:

    1. **YAML ``model_type`` field (authoritative)** — if present and the
       referenced model_type is registered, attach the case to *only* that
       model_type. This disambiguates cases where multiple registry entries
       share a ``test_model_id`` (e.g. ``dinov2`` and ``dinov3_vit`` both
       point at ``facebook/dinov2-small``).
    2. **``model_id`` reverse index (fallback)** — if no ``model_type``
       field, look up all registry entries whose ``test_model_id`` equals
       the YAML ``model_id``. Preserves legacy behavior for cases without
       the explicit field.
    """
    cases_dir = _REPO_ROOT / "testdata" / "cases"
    if not cases_dir.exists():
        return

    try:
        import yaml
    except ImportError:
        print(
            "Warning: PyYAML not installed. YAML test case scanning skipped.",
            file=sys.stderr,
        )
        return

    # Build reverse index: HF model_id → list of model_types (fallback only).
    model_id_to_types: dict[str, list[str]] = {}
    for model_type, info in models.items():
        if info.test_model_id:
            model_id_to_types.setdefault(info.test_model_id, []).append(model_type)

    for yaml_file in sorted(cases_dir.rglob("*.yaml")):
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        model_id = data.get("model_id", "")
        yaml_model_type = data.get("model_type")
        yaml_task_type = data.get("task_type")
        level = data.get("level", "")
        rel_path = str(yaml_file.relative_to(_REPO_ROOT))

        # Resolve target model_types: prefer YAML model_type field; fall
        # back to model_id reverse index for legacy cases without it.
        if yaml_model_type and yaml_model_type in models:
            matched_types: list[str] = [yaml_model_type]
        else:
            matched_types = model_id_to_types.get(model_id, [])

        # Skip test cases that are explicitly skipped — they don't count as coverage,
        # but we still record them so the dashboard can show "skipped" status.
        skip_reason = data.get("skip_reason")
        ci_skip_reason = data.get("ci_skip_reason")
        min_token_match_ratio = data.get("min_token_match_ratio")
        if skip_reason:
            for model_type in matched_types:
                if model_type in models:
                    models[model_type].yaml_test_case_file = rel_path
                    models[model_type].yaml_test_case_skip_reason = skip_reason
                    if yaml_task_type:
                        models[model_type].yaml_task_type = yaml_task_type
                    # Deliberately do NOT record min_token_match_ratio for
                    # skipped cases: the test never runs, so surfacing its
                    # configured pass threshold on the dashboard would
                    # misrepresent declared config as a measured result.
                    # Skipped cases set status="skip" (not l*_has_test_case)
                    # so the "configured" signal stays consistent with L2/L3
                    # ("configured" means a runnable test exists).
                    if "L4" in level:
                        models[model_type].l4_status = "skip"
                        models[model_type].l4_status_reason = skip_reason
                    if "L5" in level:
                        models[model_type].l5_status = "skip"
                        models[model_type].l5_status_reason = skip_reason
            continue

        for model_type in matched_types:
            if model_type in models:
                models[model_type].yaml_test_case_file = rel_path
                if yaml_task_type:
                    models[model_type].yaml_task_type = yaml_task_type
                if ci_skip_reason:
                    models[model_type].yaml_test_case_ci_skip_reason = ci_skip_reason
                if min_token_match_ratio is not None:
                    models[model_type].yaml_min_token_match_ratio = float(
                        min_token_match_ratio
                    )
                if "L4" in level:
                    models[model_type].l4_has_test_case = True
                if "L5" in level:
                    models[model_type].l5_has_test_case = True


def _scan_l3_parity_status(models: dict[str, ModelInfo]) -> None:
    """Extract L3 synthetic parity skip/xfail status from test file.

    Parses all six per-category status dicts from
    ``synthetic_parity_test.py`` (causal-LM, encoder, seq2seq x
    skip/xfail) without importing the test file. Hyphenated registry
    keys (e.g. ``"xlm-roberta"``, ``"data2vec-text"``,
    ``"nllb-moe"``) are handled by :func:`_parse_status_dict`.
    """
    parity_test = _REPO_ROOT / "tests" / "synthetic_parity_test.py"
    if not parity_test.exists():
        return

    content = parity_test.read_text(encoding="utf-8")

    # Merge per-category dicts: a given model_type appears in at most one
    # parametrized test, so per-category dicts never collide.
    skip_reasons: dict[str, str] = {}
    for dict_name in ("_SKIP_REASONS", "_ENCODER_SKIP_REASONS", "_SEQ2SEQ_SKIP_REASONS"):
        skip_reasons.update(_parse_status_dict(content, dict_name))

    xfail_reasons: dict[str, str] = {}
    for dict_name in ("_XFAIL_REASONS", "_ENCODER_XFAIL_REASONS", "_SEQ2SEQ_XFAIL_REASONS"):
        xfail_reasons.update(_parse_status_dict(content, dict_name))

    for model_type, info in models.items():
        if not info.l3_synthetic_parity:
            continue
        if model_type in skip_reasons:
            info.l3_status = "skip"
            info.l3_status_reason = skip_reasons[model_type]
        elif model_type in xfail_reasons:
            info.l3_status = "xfail"
            info.l3_status_reason = xfail_reasons[model_type]
        else:
            # Has L3 test and not skipped/xfailed → expected to pass
            info.l3_status = "pass"


def _scan_l4_l5_xfail_status(models: dict[str, ModelInfo]) -> None:
    """Extract L4 / L5 xfail status from e2e_golden_test.py.

    Parses ``_XFAIL_REASONS`` (applies to L4 *and* L5) and
    ``_L5_ONLY_XFAIL_REASONS`` (L5-only). Keys are pytest test IDs in the
    form ``"<task_type>/<case_id>"``; the case_id is the YAML file stem.

    For each xfail entry, the corresponding model's ``l4_status`` /
    ``l5_status`` is set to ``"xfail"`` (with reason), overriding any
    prior ``"pass"`` status. The golden file may exist on disk (it is the
    HuggingFace reference output) but an xfailed test is not a passing
    test. Skip status is NOT overridden \u2014 a skipped test does not run, so
    the xfail marker is irrelevant.

    Must run AFTER :func:`_scan_yaml_test_cases` (for the test_id \u2192
    model_type reverse map) and AFTER :func:`_scan_l4_golden_files` /
    :func:`_scan_l5_generation_golden` (so xfail can override their
    ``"pass"`` writes).
    """
    e2e_test = _REPO_ROOT / "tests" / "e2e_golden_test.py"
    if not e2e_test.exists():
        return

    content = e2e_test.read_text(encoding="utf-8")
    # Test IDs contain '/' so the default [\w-]+ key class won't match;
    # widen to [^"] to capture the full "<task>/<case_id>" key.
    l4_l5_xfails = _parse_status_dict(content, "_XFAIL_REASONS", key_pattern=r"[^\"]+")
    l5_only_xfails = _parse_status_dict(
        content, "_L5_ONLY_XFAIL_REASONS", key_pattern=r"[^\"]+"
    )

    # Build (task_type, case_id) -> model_type reverse map from already-
    # scanned YAML files. test_id uses the YAML ``task_type`` field, which
    # may differ from the on-disk directory name (e.g. the directory is
    # ``causal-lm`` but task_type is ``text-generation``).
    test_id_to_model: dict[str, str] = {}
    for model_type, info in models.items():
        if not info.yaml_test_case_file or not info.yaml_task_type:
            continue
        case_id = Path(info.yaml_test_case_file).stem
        test_id = f"{info.yaml_task_type}/{case_id}"
        test_id_to_model[test_id] = model_type

    def _apply_xfail(info: ModelInfo, level: int, reason: str) -> None:
        attr = f"l{level}_status"
        # Do not overwrite "skip" \u2014 the test does not run, so xfail is moot.
        if getattr(info, attr) == "skip":
            return
        setattr(info, attr, "xfail")
        setattr(info, f"l{level}_status_reason", reason)

    # _XFAIL_REASONS applies to both L4 and L5 (e2e_golden_test._discover_cases
    # passes it as xfails= for both levels).
    for test_id, reason in l4_l5_xfails.items():
        model_type = test_id_to_model.get(test_id)
        if model_type is None:
            continue
        _apply_xfail(models[model_type], 4, reason)
        _apply_xfail(models[model_type], 5, reason)

    # _L5_ONLY_XFAIL_REASONS applies only to L5; L4 prefill still passes.
    for test_id, reason in l5_only_xfails.items():
        model_type = test_id_to_model.get(test_id)
        if model_type is None:
            continue
        _apply_xfail(models[model_type], 5, reason)


def collect_all_model_info() -> dict[str, ModelInfo]:
    """Collect all model information by scanning registry and tests."""
    models = _scan_registry()
    _scan_l1_configs(models)
    _scan_l2_arch_tests(models)
    _scan_l2_arch_status(models)
    _scan_l3_synthetic_parity(models)
    _scan_l3_parity_status(models)
    # YAML test cases must be scanned before golden files so that the
    # YAML-derived golden paths can be used for indirect model_type matching.
    _scan_yaml_test_cases(models)
    _scan_l4_golden_files(models)
    _scan_l5_generation_golden(models)
    # L4/L5 xfail status must run after both golden scanners so it can
    # clear the pass flag for xfailed tests (golden file exists on disk
    # but the test is expected to fail).
    _scan_l4_l5_xfail_status(models)
    _scan_integration_tests(models)
    return models


def _group_by_family(
    models: dict[str, ModelInfo],
) -> dict[str, list[ModelInfo]]:
    """Group models by family, sorted."""
    families: dict[str, list[ModelInfo]] = {}
    for info in sorted(models.values(), key=lambda m: m.model_type):
        families.setdefault(info.family, []).append(info)
    return dict(sorted(families.items()))


def _compute_summary(
    models: dict[str, ModelInfo],
) -> dict[str, Any]:
    """Compute summary statistics for the dashboard."""
    total = len(models)
    by_level = dict.fromkeys(range(6), 0)
    by_category: dict[str, int] = {}
    all_code_paths: set[str] = set()
    code_path_coverage: dict[str, int] = {}
    # Per-level status breakdowns. All levels share the same five-bucket
    # taxonomy so the template can dispatch uniformly. ``awaiting`` and
    # ``untested`` are derived per-level (see notes below); the explicit
    # ``pass`` / ``xfail`` / ``skip`` come from the ``l<N>_status`` field.
    # By construction the five buckets sum to ``total`` for every level.
    l2_status_counts: dict[str, int] = {
        "pass": 0,
        "xfail": 0,
        "xfail_graph_only": 0,
        "untested": 0,
    }
    l3_status_counts: dict[str, int] = {
        "pass": 0,
        "xfail": 0,
        "skip": 0,
        "untested": 0,
    }
    l4_status_counts: dict[str, int] = {
        "pass": 0,
        "xfail": 0,
        "skip": 0,
        "awaiting": 0,
        "untested": 0,
    }
    l5_status_counts: dict[str, int] = {
        "pass": 0,
        "xfail": 0,
        "skip": 0,
        "awaiting": 0,
        "untested": 0,
    }
    ci_skip_count = 0

    for info in models.values():
        # Per-flag counts: how many models have each level flag set, independently.
        # These are NOT exclusive (a model counted in L3 may also be in L1/L2).
        # by_level[0] = not-tested (no *passing* coverage at any level — a
        # model that is L2/L3-configured but xfailed/skipped still lands here).
        # L2/L3/L4/L5 use their ``_passes`` properties so xfailed/skipped
        # tests do not count as coverage — matching their card counts and the
        # confidence_level property.
        if not any(
            [
                info.l1_graph_build,
                info.l2_passes,
                info.l3_passes,
                info.l4_passes,
                info.l5_passes,
            ]
        ):
            by_level[0] += 1
        if info.l1_graph_build:
            by_level[1] += 1
        if info.l2_passes:
            by_level[2] += 1
        if info.l3_passes:
            by_level[3] += 1
        if info.l4_passes:
            by_level[4] += 1
        if info.l5_passes:
            by_level[5] += 1
        by_category[info.category] = by_category.get(info.category, 0) + 1
        all_code_paths.update(info.code_paths)
        for cp in info.code_paths:
            code_path_coverage[cp] = code_path_coverage.get(cp, 0) + 1
        # L2 status (only counted for models that have an L2 test configured).
        if info.l2_arch_validation:
            key = info.l2_status or "untested"
            l2_status_counts[key] = l2_status_counts.get(key, 0) + 1
        # L3 status
        if info.l3_status:
            l3_status_counts[info.l3_status] = l3_status_counts.get(info.l3_status, 0) + 1
        else:
            l3_status_counts["untested"] += 1
        # L4 / L5 status. The dataclass field uses {"pass","xfail","skip",None};
        # ``None`` splits into two card buckets:
        #   - ``awaiting`` = yaml case exists but no golden generated yet
        #   - ``untested`` = no yaml case at all
        for level, counts in ((4, l4_status_counts), (5, l5_status_counts)):
            status = getattr(info, f"l{level}_status")
            has_case = getattr(info, f"l{level}_has_test_case")
            if status:
                counts[status] = counts.get(status, 0) + 1
            elif has_case:
                counts["awaiting"] += 1
            else:
                counts["untested"] += 1
        if info.yaml_test_case_ci_skip_reason:
            ci_skip_count += 1

    return {
        "total": total,
        "by_level": by_level,
        "by_category": dict(sorted(by_category.items())),
        "code_path_coverage": dict(sorted(code_path_coverage.items())),
        "all_code_paths": sorted(all_code_paths),
        "l2_status_counts": l2_status_counts,
        "l3_status_counts": l3_status_counts,
        "l4_status_counts": l4_status_counts,
        "l5_status_counts": l5_status_counts,
        "ci_skip_count": ci_skip_count,
    }


def _build_component_matrix(
    models: dict[str, ModelInfo],
) -> dict[str, Any]:
    """Build component x family matrix for the heatmap visualization.

    Returns a dict suitable for JSON serialization with:
    - ``families``: sorted list of family names that have at least one component.
    - ``rows``: one entry per component, with per-family max confidence levels.
    """
    from mobius._testing.code_paths import CODE_PATH_INDICATORS

    # Gather only families that exercise at least one component.
    families_with_paths: set[str] = set()
    for info in models.values():
        if info.code_paths:
            families_with_paths.add(info.family)
    sorted_families = sorted(families_with_paths)

    # matrix[feature_label][family] = max confidence level among all models
    # in that family that exercise this feature.
    matrix: dict[str, dict[str, int]] = {ind.label: {} for ind in CODE_PATH_INDICATORS}
    for info in models.values():
        if not info.code_paths:
            continue
        for path in info.code_paths:
            if path in matrix:
                cur = matrix[path].get(info.family, -1)
                matrix[path][info.family] = max(cur, info.confidence_level)

    # Build rows — one per indicator, with a cell value per family.
    rows = []
    for ind in CODE_PATH_INDICATORS:
        fam_cells = matrix[ind.label]
        cells = [fam_cells.get(fam, -1) for fam in sorted_families]
        family_count = sum(1 for c in cells if c >= 0)
        # Total individual model count from summary (recomputed here for simplicity).
        model_count = sum(1 for info in models.values() if ind.label in info.code_paths)
        best_level = max((c for c in cells if c >= 0), default=-1)
        rows.append(
            {
                "label": ind.label,
                "description": ind.description,
                "model_count": model_count,
                "family_count": family_count,
                "best_level": best_level,
                "cells": cells,
            }
        )

    return {"families": sorted_families, "rows": rows}


_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _to_js_json(obj: Any) -> str:
    """Serialize obj to JSON safe for inline <script> injection.

    The ``</`` replacement prevents the string from accidentally closing
    a ``<script>`` tag when embedded in HTML.
    """
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def _render_html(
    models: dict[str, ModelInfo],
    commit: str | None = None,
) -> str:
    """Render the self-contained HTML dashboard via Jinja2 template."""
    summary = _compute_summary(models)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build model data as JSON for JavaScript consumption.
    model_data = []
    for info in models.values():
        model_data.append(
            {
                "model_type": info.model_type,
                "module_class": info.module_class_name,
                "task": info.task,
                "category": info.category,
                "family": info.family,
                "confidence_level": info.confidence_level,
                "confidence_label": info.confidence_label,
                # Per-level passing booleans. The template uses these for
                # the "passed" dot color. xfail / skip / awaiting / untested
                # are derived from the status fields below.
                "l1": info.l1_graph_build,
                "l2": info.l2_passes,
                "l3": info.l3_passes,
                "l4": info.l4_passes,
                "l5": info.l5_passes,
                # "Test configured" signal (parallels l2_configured).
                "l2_configured": info.l2_arch_validation,
                "l3_configured": info.l3_synthetic_parity,
                "l4_configured": info.l4_has_test_case,
                "l5_configured": info.l5_has_test_case,
                # Unified status fields ({pass, xfail, skip, None}; L2 also
                # admits xfail_graph_only). Status drives dot color, badges,
                # and the per-card bucket annotations.
                "l2_status": info.l2_status,
                "l2_reason": info.l2_status_reason,
                "l3_status": info.l3_status,
                "l3_reason": info.l3_status_reason,
                "l4_status": info.l4_status,
                "l4_reason": info.l4_status_reason,
                "l5_status": info.l5_status,
                "l5_reason": info.l5_status_reason,
                "yaml_case": info.yaml_test_case_file,
                "yaml_skip_reason": info.yaml_test_case_skip_reason,
                "yaml_ci_skip_reason": info.yaml_test_case_ci_skip_reason,
                "min_token_match_ratio": info.yaml_min_token_match_ratio,
                "code_paths": sorted(info.code_paths),
                "config_overrides": _json_safe(info.config_overrides),
                "has_integration_test": info.has_integration_test,
                "test_model_id": info.test_model_id,
            }
        )

    model_data_json = _to_js_json(sorted(model_data, key=lambda m: m["model_type"]))

    from mobius._testing.code_paths import CODE_PATH_INDICATORS

    code_path_info = [
        {
            "label": ind.label,
            "description": ind.description,
            "example_config": ind.example_config,
        }
        for ind in CODE_PATH_INDICATORS
    ]

    # Commit string is plain text; Jinja2 autoescape handles HTML encoding.
    component_matrix = _build_component_matrix(models)
    context = {
        "timestamp": timestamp,
        "commit": commit if commit else "unknown",
        "total_models": summary["total"],
        # JSON blobs injected into <script> tags: marked |safe in the template
        # because json.dumps already produces valid JS values and the </
        # replacement prevents premature script-tag closure.
        "model_data_json": model_data_json,
        "code_path_json": _to_js_json(code_path_info),
        "component_matrix_json": _to_js_json(component_matrix),
        "summary_json": _to_js_json(summary),
        "labels_json": _to_js_json(_CONFIDENCE_LABELS),
    }

    # autoescape=True: Jinja2 HTML-escapes all {{ var }} by default.
    # Variables containing pre-serialized JSON are marked |safe in the template
    # to bypass escaping — they are already safe for <script> injection.
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
        keep_trailing_newline=True,
    )
    template = env.get_template("dashboard.html.j2")
    return template.render(**context)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the testing confidence dashboard.")
    parser.add_argument(
        "--output",
        type=str,
        default="docs/dashboard/index.html",
        help="Output HTML file path (default: docs/dashboard/index.html)",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Git commit SHA to display in the dashboard",
    )
    args = parser.parse_args()

    models = collect_all_model_info()
    html_content = _render_html(models, commit=args.commit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)

    # Print summary
    summary = _compute_summary(models)
    total = summary["total"]
    by_level = summary["by_level"]
    print(f"Dashboard generated: {output_path}")
    print(f"  Total Models: {total}")
    for level, count in sorted(by_level.items()):
        pct = round(count / total * 100, 1) if total > 0 else 0
        label = _CONFIDENCE_LABELS.get(level, f"L{level}")
        print(f"  {label}: {count} ({pct}%)")


if __name__ == "__main__":
    main()
