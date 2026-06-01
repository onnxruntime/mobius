# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regression tests for scripts/generate_dashboard.py.

These tests pin down the bugs described in issue #211: skipped L4/L5 test
cases were being counted as passing coverage in several places.

The defect lives in the data-collection layer — ``_scan_l4_golden_files``
and ``_scan_l5_generation_golden`` set their flags whenever a JSON file
exists on disk, even when the corresponding YAML test case has a
``skip_reason``. Every downstream consumer (``confidence_level`` property,
``_compute_summary`` card counts, ``_render_html`` JSON emission,
``_build_component_matrix`` heatmap) then trusted that wrong flag and
amplified the bug.

The tests drive the real scanners through a tmp filesystem so they
faithfully exercise the production code path, then assert both the
upstream flag state and the downstream observable behavior.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "generate_dashboard.py"


def _load_dashboard_module():
    """Load generate_dashboard.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "_generate_dashboard_under_test", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gd():
    return _load_dashboard_module()


def _write_yaml(path: Path, *, model_id: str, level: str, skip_reason: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'model_id: "{model_id}"', f'level: "{level}"']
    if skip_reason is not None:
        lines.append(f'skip_reason: "{skip_reason}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {"placeholder": True}), encoding="utf-8")


def _build_model(
    gd,
    *,
    model_type: str = "dinov3_vit",
    family: str = "dinov3",
    test_model_id: str = "fake/dinov3-vit",
    l1: bool = True,
    l2: bool = False,
):
    """Build a ModelInfo as the registry + L1/L2 scanners would leave it.

    State is captured before the golden + YAML scanners run.
    """
    return gd.ModelInfo(
        model_type=model_type,
        module_class_name="FakeModel",
        task="image-classification",
        category="Vision",
        family=family,
        l1_graph_build=l1,
        l2_arch_validation=l2,
        test_model_id=test_model_id,
    )


@pytest.fixture
def tmp_repo(tmp_path, monkeypatch, gd):
    """Tmp filesystem with testdata/{cases,golden} dirs; patches _REPO_ROOT."""
    (tmp_path / "testdata" / "cases").mkdir(parents=True)
    (tmp_path / "testdata" / "golden").mkdir(parents=True)
    monkeypatch.setattr(gd, "_REPO_ROOT", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Upstream scanner fix — root-cause assertions.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["direct", "indirect"])
def test_scan_l4_does_not_set_flag_when_yaml_skipped(gd, tmp_repo, strategy):
    """``_scan_l4_golden_files`` must not mark coverage when YAML is skipped.

    Strategy 1 (direct): golden file stem == model_type.
    Strategy 2 (indirect): golden file stem == YAML case_id (differs from model_type).
    """
    case_stem = "dinov3_vit" if strategy == "direct" else "dinov3-vit-small"
    _write_yaml(
        tmp_repo / "testdata" / "cases" / "vision" / f"{case_stem}.yaml",
        model_id="fake/dinov3-vit",
        level="L4",
        skip_reason="tracing fails on op X",
    )
    _write_json(tmp_repo / "testdata" / "golden" / "vision" / f"{case_stem}.json")

    models = {"dinov3_vit": _build_model(gd, l2=True)}
    gd._scan_yaml_test_cases(models)
    gd._scan_l4_golden_files(models)

    info = models["dinov3_vit"]
    assert info.l4_test_case_skipped is True
    assert info.l4_golden_files is False, (
        f"L4 golden flag must remain False when YAML has skip_reason "
        f"(strategy={strategy}); skip_reason was set to "
        f"{info.yaml_test_case_skip_reason!r}"
    )


@pytest.mark.parametrize("strategy", ["direct", "indirect"])
def test_scan_l5_does_not_set_flag_when_yaml_skipped(gd, tmp_repo, strategy):
    """Same as the L4 test, for ``_scan_l5_generation_golden``."""
    case_stem = "skipped_only" if strategy == "direct" else "skipped-only-case"
    _write_yaml(
        tmp_repo / "testdata" / "cases" / "text" / f"{case_stem}.yaml",
        model_id="fake/skipped-only",
        level="L5",
        skip_reason="generation diverges from HF",
    )
    _write_json(tmp_repo / "testdata" / "golden" / "text" / f"{case_stem}_generation.json")

    models = {
        "skipped_only": _build_model(
            gd,
            model_type="skipped_only",
            family="skipped",
            test_model_id="fake/skipped-only",
        )
    }
    gd._scan_yaml_test_cases(models)
    gd._scan_l5_generation_golden(models)

    info = models["skipped_only"]
    assert info.l5_test_case_skipped is True
    assert info.l5_generation_golden is False


def test_scan_l4_still_sets_flag_when_yaml_not_skipped(gd, tmp_repo):
    """Sanity: the fix must not regress the happy path."""
    _write_yaml(
        tmp_repo / "testdata" / "cases" / "vision" / "dinov3_vit.yaml",
        model_id="fake/dinov3-vit",
        level="L4",
        skip_reason=None,
    )
    _write_json(tmp_repo / "testdata" / "golden" / "vision" / "dinov3_vit.json")

    models = {"dinov3_vit": _build_model(gd)}
    gd._scan_yaml_test_cases(models)
    gd._scan_l4_golden_files(models)

    info = models["dinov3_vit"]
    assert info.l4_test_case_skipped is False
    assert info.l4_golden_files is True


# ---------------------------------------------------------------------------
# Downstream consumer behavior after the upstream fix.
# Each test maps to one or more user-visible bugs from issue #211.
# ---------------------------------------------------------------------------
def _setup_skipped_l4_pipeline(gd, tmp_repo, *, l2: bool = True):
    """Skipped-L4 model: lay out YAML+JSON and run the relevant scanners."""
    _write_yaml(
        tmp_repo / "testdata" / "cases" / "vision" / "dinov3_vit.yaml",
        model_id="fake/dinov3-vit",
        level="L4",
        skip_reason="known issue: tracing fails",
    )
    _write_json(tmp_repo / "testdata" / "golden" / "vision" / "dinov3_vit.json")
    models = {"dinov3_vit": _build_model(gd, l2=l2)}
    gd._scan_yaml_test_cases(models)
    gd._scan_l4_golden_files(models)
    gd._scan_l5_generation_golden(models)
    return models


def test_bug1_confidence_level_reflects_real_passing_level(gd, tmp_repo):
    """Bug 1: confidence_level returned 4 even when only L2 was actually passing."""
    models = _setup_skipped_l4_pipeline(gd, tmp_repo, l2=True)
    info = models["dinov3_vit"]
    assert info.confidence_level == 2
    assert info.confidence_label == "L2: Config compatible"


def test_bug2_compute_summary_l4_card_excludes_skipped(gd, tmp_repo):
    """Bug 2: summary L4/L5 card numbers were too high."""
    models = _setup_skipped_l4_pipeline(gd, tmp_repo, l2=True)
    summary = gd._compute_summary(models)
    assert summary["by_level"][4] == 0
    assert summary["l4_skipped_count"] == 1


def test_bug3_compute_summary_not_tested_includes_skipped_only(gd, tmp_repo):
    """Bug 3: 'Not tested' card undercounted skipped-only models."""
    _write_yaml(
        tmp_repo / "testdata" / "cases" / "text" / "skipped_only.yaml",
        model_id="fake/skipped-only",
        level="L5",
        skip_reason="generation diverges",
    )
    _write_json(tmp_repo / "testdata" / "golden" / "text" / "skipped_only_generation.json")
    models = {
        "skipped_only": _build_model(
            gd,
            model_type="skipped_only",
            family="skipped",
            test_model_id="fake/skipped-only",
            l1=False,
            l2=False,
        )
    }
    gd._scan_yaml_test_cases(models)
    gd._scan_l4_golden_files(models)
    gd._scan_l5_generation_golden(models)

    summary = gd._compute_summary(models)
    assert summary["by_level"][0] == 1, (
        f"Skipped-only model with no real coverage should land in 'Not tested'; "
        f"got by_level={summary['by_level']}"
    )
    assert summary["by_level"][5] == 0


def _extract_model_data_json(html: str) -> list[dict]:
    """Pull the MODEL_DATA = [...]; literal out of the rendered HTML."""
    match = re.search(r"MODEL_DATA\s*=\s*(\[.*?\]);", html, re.DOTALL)
    assert match is not None, "Could not find MODEL_DATA in rendered HTML"
    return json.loads(match.group(1))


def test_bugs_4_6_7_render_html_l4_json_false_when_skipped(gd, tmp_repo):
    """Bugs 4/6/7: JSON ``m.l4`` was True for skipped models.

    This poisoned the level dots (4), the golden-status detail row (6),
    and the family histogram (7).
    """
    models = _setup_skipped_l4_pipeline(gd, tmp_repo, l2=True)
    html = gd._render_html(models)
    data = _extract_model_data_json(html)
    assert len(data) == 1
    m = data[0]
    assert m["l4"] is False
    assert m["l4_skipped"] is True
    assert m["confidence_level"] == 2


def test_bug8_component_matrix_cell_excludes_skipped_level(gd, tmp_repo):
    """Bug 8: heatmap cell came from max(confidence_level) per family.

    A family whose only L4 was skipped showed as a darker (L4) cell.
    """
    from mobius._testing.code_paths import CODE_PATH_INDICATORS

    models = _setup_skipped_l4_pipeline(gd, tmp_repo, l2=True)
    indicator_label = CODE_PATH_INDICATORS[0].label
    models["dinov3_vit"].code_paths.add(indicator_label)

    matrix = gd._build_component_matrix(models)
    row = next(r for r in matrix["rows"] if r["label"] == indicator_label)
    fam_idx = matrix["families"].index("dinov3")
    cell = row["cells"][fam_idx]
    assert cell == 2, (
        f"Heatmap cell should reflect L2 (real passing level), not L4 "
        f"(inflated by skipped golden). Got {cell}."
    )


# ---------------------------------------------------------------------------
# L3 analogs — same bug pattern as L4/L5, partially unfixed in current code.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("l3_status", ["skip", "xfail"])
def test_l3_skip_or_xfail_not_counted_as_confidence_3(gd, l3_status):
    """L3 analog of Bug 1.

    ``confidence_level`` returns 3 when the L3 test is in the parametrize
    list but explicitly skipped or xfailed. The only thing actually
    passing here is L2, so confidence should be 2.
    """
    info = _build_model(gd, l1=True, l2=True)
    info.l3_synthetic_parity = True
    info.l3_status = l3_status
    info.l3_status_reason = "known failure"
    assert info.confidence_level == 2, (
        f"L3 {l3_status!r} should not count as L3 passing; "
        f"got confidence_level={info.confidence_level}"
    )
    assert info.confidence_label == "L2: Config compatible"


@pytest.mark.parametrize("l3_status", ["skip", "xfail"])
def test_compute_summary_l3_skip_xfail_counted_as_not_tested(gd, l3_status):
    """L3 analog of Bug 3.

    A model whose only signal is an L3 skip/xfail escapes the 'Not tested'
    bucket because the ``any([...])`` check uses the raw
    ``l3_synthetic_parity`` flag instead of ``status == "pass"``.
    """
    info = _build_model(
        gd,
        model_type="l3_skip_only",
        family="l3only",
        test_model_id="fake/l3-skip-only",
        l1=False,
        l2=False,
    )
    info.l3_synthetic_parity = True
    info.l3_status = l3_status
    info.l3_status_reason = "known failure"
    summary = gd._compute_summary({"l3_skip_only": info})
    assert summary["by_level"][0] == 1, (
        f"Model with only an L3 {l3_status!r} should land in 'Not tested'; "
        f"got by_level={summary['by_level']}"
    )
    assert summary["by_level"][3] == 0
