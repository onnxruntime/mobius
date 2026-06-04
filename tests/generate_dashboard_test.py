# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regression tests for ``scripts/generate_dashboard.py``.

The dashboard generator is a three-stage pipeline:

1. **Scanners** populate a ``dict[str, ModelInfo]`` from the registry,
   test-config Python files, YAML test cases under ``testdata/cases/``,
   and golden JSON files under ``testdata/golden/``. Each scanner sets
   one or more ``ModelInfo`` flags.
2. **Aggregation** functions (``ModelInfo.confidence_level``,
   ``_compute_summary``, ``_build_component_matrix``) derive summary
   counts and matrix cells from the populated ``ModelInfo`` dicts.
3. **Rendering** (``_render_html``) emits an HTML page that embeds the
   aggregated state as a JSON ``MODEL_DATA`` blob consumed by the
   inline JavaScript.

Tests are grouped by the stage they exercise. Most use ``tmp_repo`` to
drive the real scanners through a synthetic filesystem so the
production code path runs end-to-end. A small set of tests run against
the real repository to verify the dashboard generates without error
and that no model_type in a known skip/xfail dict appears as passing.

When adding a new ``ModelInfo`` field, a new scanner, or a new
status-dict in a test file, extend the corresponding section here so
the regression net keeps the same shape as the data flow.
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

    State is captured before the golden + YAML scanners run. Passing
    ``l2=True`` mirrors the production scanner: both the configured flag
    and the ``"pass"`` status are set.
    """
    return gd.ModelInfo(
        model_type=model_type,
        module_class_name="FakeModel",
        task="image-classification",
        category="Vision",
        family=family,
        l1_graph_build=l1,
        l2_arch_validation=l2,
        l2_status="pass" if l2 else None,
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
# Scanner tests: golden JSON discovery must honor YAML skip_reason.
# ---------------------------------------------------------------------------
# ``_scan_l4_golden_files`` and ``_scan_l5_generation_golden`` find golden
# JSON files on disk and set the corresponding coverage flag. They must
# defer to ``_scan_yaml_test_cases``: if a model's YAML case is skipped,
# its golden file is reference data only and does not represent passing
# coverage. Both match strategies (direct stem and YAML-derived stem)
# need the same guard.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["direct", "indirect"])
def test_scan_l4_does_not_set_flag_when_yaml_skipped(gd, tmp_repo, strategy):
    """L4 coverage flag must not be set when the YAML case is skipped.

    Both stem-matching strategies are exercised:

    * direct: ``golden/<task>/<model_type>.json``
    * indirect: ``golden/<task>/<yaml_case_id>.json``, where the YAML
      case_id differs from the registry model_type.
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
    assert info.l4_status == "skip"
    assert info.l4_passes is False, (
        f"L4 must not pass when YAML has skip_reason "
        f"(strategy={strategy}); skip_reason was set to "
        f"{info.yaml_test_case_skip_reason!r}"
    )


@pytest.mark.parametrize("strategy", ["direct", "indirect"])
def test_scan_l5_does_not_set_flag_when_yaml_skipped(gd, tmp_repo, strategy):
    """L5 coverage flag must not be set when the YAML case is skipped.

    Same contract as the L4 counterpart, applied to generation goldens
    (``<stem>_generation.json``).
    """
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
    assert info.l5_status == "skip"
    assert info.l5_passes is False


def test_scan_l4_still_sets_flag_when_yaml_not_skipped(gd, tmp_repo):
    """L4 flag is set normally when the YAML case has no skip_reason.

    Guards against an over-zealous skip filter accidentally suppressing
    legitimate coverage.
    """
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
    assert info.l4_status == "pass"
    assert info.l4_passes is True


# ---------------------------------------------------------------------------
# Aggregation tests: skipped/xfail flags must not inflate downstream views.
# ---------------------------------------------------------------------------
# Every consumer of the L4/L5 coverage flags (``confidence_level``,
# ``_compute_summary`` card counts, ``_render_html`` JSON emission,
# ``_build_component_matrix`` heatmap cells) must reflect the real
# passing level, not the disk presence of a golden file.
# ---------------------------------------------------------------------------
def _setup_skipped_l4_pipeline(gd, tmp_repo, *, l2: bool = True):
    """Fixture builder: a skipped-L4 model wired through every scanner.

    Lays out a YAML test case with ``skip_reason`` plus the matching
    golden JSON, then runs the YAML, L4, and L5 scanners. Returns the
    populated ``ModelInfo`` dict ready for aggregation/rendering tests.
    """
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
    """``confidence_level`` returns the highest *actually-passing* level.

    A model with a skipped L4 YAML and a green L2 has confidence 2,
    not 4.
    """
    models = _setup_skipped_l4_pipeline(gd, tmp_repo, l2=True)
    info = models["dinov3_vit"]
    assert info.confidence_level == 2
    assert info.confidence_label == "L2: Config Compatible"


def test_bug2_compute_summary_l4_card_excludes_skipped(gd, tmp_repo):
    """Summary L4 card counts only models with real L4 coverage.

    Skipped models contribute to ``l4_status_counts["skip"]`` instead.
    """
    models = _setup_skipped_l4_pipeline(gd, tmp_repo, l2=True)
    summary = gd._compute_summary(models)
    assert summary["by_level"][4] == 0
    assert summary["l4_status_counts"]["skip"] == 1


def test_bug3_compute_summary_not_tested_includes_skipped_only(gd, tmp_repo):
    """A model whose only signal is a skipped case lands in 'Not tested'.

    The L0 bucket counts models with no *passing* coverage at any
    level — skip/xfail signals do not lift them out of it.
    """
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
    """Rendered JSON ``m.l4`` reports passing coverage, not file presence.

    The template's dot logic, golden-status block, and family histogram
    all read this field; emitting ``true`` for skipped models leaks the
    bug into every one of those views.
    """
    models = _setup_skipped_l4_pipeline(gd, tmp_repo, l2=True)
    html = gd._render_html(models)
    data = _extract_model_data_json(html)
    assert len(data) == 1
    m = data[0]
    assert m["l4"] is False
    assert m["l4_status"] == "skip"
    assert m["confidence_level"] == 2


def test_bug8_component_matrix_cell_excludes_skipped_level(gd, tmp_repo):
    """Component matrix cell reflects the family's real passing level.

    ``_build_component_matrix`` takes ``max(confidence_level)`` per
    (component, family) pair; cells must not be inflated by skipped or
    xfailed levels.
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
# Property tests: L3 skip/xfail must not count as L3 passing.
# ---------------------------------------------------------------------------
# L3 distinguishes "test exists" (``l3_synthetic_parity``) from "test
# passes" (``l3_status``) via two fields. Every consumer must combine
# them — a raw read of ``l3_synthetic_parity`` is a bug.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("l3_status", ["skip", "xfail"])
def test_l3_skip_or_xfail_not_counted_as_confidence_3(gd, l3_status):
    """``confidence_level`` excludes L3 unless ``l3_status == "pass"``.

    A model parametrized into the L3 test but listed in
    ``_SKIP_REASONS`` or ``_XFAIL_REASONS`` has ``l3_synthetic_parity``
    True but is not passing; confidence falls through to the next lower
    real level.
    """
    info = _build_model(gd, l1=True, l2=True)
    info.l3_synthetic_parity = True
    info.l3_status = l3_status
    info.l3_status_reason = "known failure"
    assert info.confidence_level == 2, (
        f"L3 {l3_status!r} should not count as L3 passing; "
        f"got confidence_level={info.confidence_level}"
    )
    assert info.confidence_label == "L2: Config Compatible"


@pytest.mark.parametrize("l3_status", ["skip", "xfail"])
def test_compute_summary_l3_skip_xfail_counted_as_not_tested(gd, l3_status):
    """``_compute_summary`` excludes L3 skip/xfail from L0 'Not tested' check.

    The 'any flag set' check that gates the L0 bucket must use
    ``l3_passes``, not raw ``l3_synthetic_parity``, so a skipped-only
    L3 model still counts as untested.
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


# ---------------------------------------------------------------------------
# Template integrity: CSS / JS / legend in sync.
# ---------------------------------------------------------------------------
# The template emits dot classes from JavaScript. Each class needs a
# matching ``.dot.X`` CSS rule and a legend entry, or it renders as an
# unstyled dot. The class-assignment logic is reimplemented in Python
# so dot rendering can be asserted without spinning up a browser.
# ---------------------------------------------------------------------------
_TEMPLATE_PATH = _REPO_ROOT / "scripts" / "templates" / "dashboard.html.j2"


@pytest.mark.parametrize(
    "css_class",
    ["xfail", "skipped", "pending", "untested", "failed"],
)
def test_template_has_css_and_legend_for_dot_class(css_class):
    """Each dot state has a matching CSS rule and a legend entry.

    Missing either side produces an unstyled dot or an undocumented
    state in the legend.
    """
    text = _TEMPLATE_PATH.read_text(encoding="utf-8")
    assert f".dot.{css_class}" in text, f"CSS rule for .dot.{css_class} is missing"
    # Legend uses inline classes like ``dot xfail``; match either spelling.
    assert f"dot {css_class}" in text or f"dot.{css_class}" in text, (
        f"Legend entry referencing dot.{css_class} not found"
    )


def test_template_dot_class_assignments_match_css():
    """Every dot class assigned by JS is defined in CSS.

    Scans the template for ``cls += ' X'`` assignments and asserts each
    one has a matching ``.dot.X`` rule. The dynamic ``active-N`` family
    is checked explicitly for ``N \u2208 [1, 5]``.
    """
    import re

    text = _TEMPLATE_PATH.read_text(encoding="utf-8")
    # Find all class names the JS adds to dots via `cls += ' XXX'`.
    js_classes = set(re.findall(r"cls\s*\+=\s*[\"']\s+([\w-]+)\s*[\"']", text))
    # Drop the dynamic-suffix base (active-1..active-5 are checked below).
    js_classes.discard("active-")
    css_classes = set(re.findall(r"\.dot\.([\w-]+)\s*\{", text))
    missing = js_classes - css_classes
    assert not missing, f"JS assigns dot classes that have no CSS rule: {sorted(missing)}"
    # Active-N CSS rules must exist for all 5 levels.
    for level in range(1, 6):
        assert f".dot.active-{level}" in text, f"CSS rule for .dot.active-{level} is missing"


@pytest.mark.parametrize(
    "l_status,expected_class",
    [
        # L2 xfail → dot xfail
        (("l2", "xfail"), "xfail"),
        (("l2", "xfail_graph_only"), "xfail"),
        # L3 statuses
        (("l3", "xfail"), "xfail"),
        (("l3", "skip"), "skipped"),
    ],
)
def test_dot_class_assignment_logic(l_status, expected_class):
    """Python re-implementation of the JS dot-class chooser.

    Lets us assert the chosen class for any ``ModelInfo`` state without
    a headless browser. Kept structurally parallel to ``renderModelRow``
    so reviewers can spot drift by reading the two side by side.
    """
    level_name, status = l_status
    # Minimal model dict mirroring what the template's JS sees.
    m = {
        "l1": True,
        "l2": False,
        "l3": False,
        "l4": False,
        "l5": False,
        "l2_configured": False,
        "l2_status": None,
        "l3_status": None,
        "l4_case": False,
        "l5_case": False,
        "l4_skipped": False,
        "l5_skipped": False,
        "config_overrides": [],
        "test_model_id": None,
    }
    if level_name == "l2":
        m["l2_configured"] = True
        m["l2_status"] = status
    elif level_name == "l3":
        m["l3"] = False
        m["l3_status"] = status

    # The level index we are asking about.
    i = 2 if level_name == "l2" else 3

    # Reproduce template's dot-class logic.
    if i == 2:
        active = m["l2"] and m["l2_status"] == "pass"
    elif i == 3:
        active = m["l3"] and m["l3_status"] == "pass"
    else:
        active = m[f"l{i}"]
    is_l2_xfail = i == 2 and m["l2_status"] in ("xfail", "xfail_graph_only")
    is_xfail = is_l2_xfail or (i == 3 and m["l3_status"] == "xfail")
    is_l3_skip = i == 3 and m["l3_status"] == "skip"
    is_skipped = (i == 4 and m["l4_skipped"]) or (i == 5 and m["l5_skipped"]) or is_l3_skip

    if active:
        chosen = f"active-{i}"
    elif is_xfail:
        chosen = "xfail"
    elif is_skipped:
        chosen = "skipped"
    else:
        chosen = "untested"

    assert chosen == expected_class, (
        f"Dot for L{i} with {level_name}_status={status!r} should be "
        f"{expected_class!r}, got {chosen!r}"
    )


# ---------------------------------------------------------------------------
# Scanner tests: L2 architecture-validation status.
# ---------------------------------------------------------------------------
# ``_scan_l2_arch_tests`` flags models that have an L2 test configured
# (a ``test_model_id`` in the registry). ``_scan_l2_arch_status`` then
# reads xfail dicts from ``arch_validation_test.py`` to distinguish
# "pass" / "xfail" / "xfail_graph_only". Both flavors of xfail must
# downgrade ``l2_passes``.
# ---------------------------------------------------------------------------
def _write_arch_validation_stub(
    tmp_repo: Path,
    *,
    parse_and_graph_xfails: dict[str, str] | None = None,
    graph_only_xfails: dict[str, str] | None = None,
) -> None:
    """Write a minimal ``arch_validation_test.py`` containing the two xfail dicts.

    Allows ``_scan_l2_arch_status`` to be exercised with a known set of
    xfail entries without depending on the real test file.
    """
    tests_dir = tmp_repo / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    pg = parse_and_graph_xfails or {}
    go = graph_only_xfails or {}

    def _fmt(d: dict[str, str]) -> str:
        if not d:
            return "{}"
        body = ",\n    ".join(f'"{k}": "{v}"' for k, v in d.items())
        return "{\n    " + body + ",\n}"

    (tests_dir / "arch_validation_test.py").write_text(
        f"_PARSE_AND_GRAPH_XFAILS: dict[str, str] = {_fmt(pg)}\n"
        f"_GRAPH_ONLY_XFAILS: dict[str, str] = {_fmt(go)}\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "dict_kind,expected_status",
    [
        ("parse_and_graph_xfails", "xfail"),
        ("graph_only_xfails", "xfail_graph_only"),
    ],
)
def test_scan_l2_arch_status_marks_xfail(gd, tmp_repo, dict_kind, expected_status):
    """L2 status records the two xfail flavors distinctly.

    ``arch_validation_test.py`` distinguishes "parse + graph both fail"
    from "only the graph-build subtest fails"; the scanner must preserve
    that so the template can render different annotations.
    """
    _write_arch_validation_stub(
        tmp_repo, **{dict_kind: {"fuyu": "FuyuConfig has no vision_config"}}
    )
    models = {
        "fuyu": _build_model(
            gd, model_type="fuyu", family="fuyu", test_model_id="adept/fuyu-8b"
        )
    }
    gd._scan_l2_arch_tests(models)
    gd._scan_l2_arch_status(models)
    info = models["fuyu"]
    assert info.l2_arch_validation is True
    assert info.l2_status == expected_status
    assert info.l2_status_reason == "FuyuConfig has no vision_config"
    assert info.l2_passes is False


def test_scan_l2_arch_status_marks_pass_when_no_xfail(gd, tmp_repo):
    """L2 status defaults to ``"pass"`` for configured models not in any xfail dict."""
    _write_arch_validation_stub(tmp_repo)
    models = {
        "llama": _build_model(
            gd, model_type="llama", family="llama", test_model_id="meta/llama"
        )
    }
    gd._scan_l2_arch_tests(models)
    gd._scan_l2_arch_status(models)
    info = models["llama"]
    assert info.l2_status == "pass"
    assert info.l2_passes is True


@pytest.mark.parametrize("l2_status", ["xfail", "xfail_graph_only"])
def test_confidence_level_excludes_l2_xfail(gd, l2_status):
    """``confidence_level`` excludes L2 unless ``l2_status == "pass"``."""
    info = _build_model(gd, l1=True, l2=True)
    info.l2_status = l2_status
    info.l2_status_reason = "config requires trust_remote_code"
    assert info.confidence_level == 1
    assert info.confidence_label == "L1: Graph Builds"


@pytest.mark.parametrize("l2_status", ["xfail", "xfail_graph_only"])
def test_compute_summary_l2_card_excludes_xfail(gd, l2_status):
    """Summary L2 card and L0 'Not tested' check exclude xfailed L2 models."""
    info = _build_model(gd, l1=False, l2=True)
    info.l2_status = l2_status
    summary = gd._compute_summary({"fuyu": info})
    assert summary["by_level"][2] == 0, (
        f"xfail L2 should not count in L2 card; got by_level={summary['by_level']}"
    )
    # No real coverage at all → Not tested
    assert summary["by_level"][0] == 1
    assert summary["l2_status_counts"][l2_status] == 1


def test_render_html_l2_json_false_when_xfail(gd):
    """Rendered JSON gates ``l2`` on ``l2_passes``, exposes raw status separately.

    The template needs both signals: ``l2`` (boolean, drives the dot)
    and ``l2_status`` / ``l2_configured`` / ``l2_reason`` (drive the
    xfail chip and detail row).
    """
    info = _build_model(gd, l1=True, l2=True)
    info.l2_status = "xfail_graph_only"
    info.l2_status_reason = "missing vision_config"
    html = gd._render_html({"fuyu": info})
    data = _extract_model_data_json(html)
    m = data[0]
    assert m["l2"] is False
    assert m["l2_configured"] is True
    assert m["l2_status"] == "xfail_graph_only"
    assert m["l2_reason"] == "missing vision_config"
    assert m["confidence_level"] == 1


# ---------------------------------------------------------------------------
# Scanner tests: L3 synthetic parity coverage and status.
# ---------------------------------------------------------------------------
# L3 spans causal-LM, encoder, and seq2seq parametrized tests, each
# with its own ``_SKIP_REASONS`` / ``_XFAIL_REASONS`` dict (six in
# total). ``_scan_l3_synthetic_parity`` must union all three config
# lists; ``_scan_l3_parity_status`` must parse all six dicts.
# ---------------------------------------------------------------------------
def test_scan_l3_includes_encoder_and_seq2seq(gd):
    """L3 coverage includes encoder and seq2seq, not only causal-LM.

    Three parametrized tests in ``synthetic_parity_test.py`` populate
    L3 coverage from three separate config lists; the scanner must
    union all of them.
    """
    from mobius._registry import registry

    # Pick a known encoder + a known seq2seq model that should be tested.
    arches = set(registry.architectures())
    encoder_pick = next(mt for mt in ("bert", "roberta", "albert") if mt in arches)
    seq2seq_pick = next(mt for mt in ("bart", "t5", "marian") if mt in arches)
    causal_pick = next(mt for mt in ("llama", "gpt2", "qwen2") if mt in arches)

    models = {
        mt: _build_model(gd, model_type=mt, family=mt, test_model_id=f"fake/{mt}")
        for mt in (encoder_pick, seq2seq_pick, causal_pick)
    }
    gd._scan_l3_synthetic_parity(models)
    assert models[encoder_pick].l3_synthetic_parity, (
        f"encoder model {encoder_pick!r} should be marked L3-tested"
    )
    assert models[seq2seq_pick].l3_synthetic_parity, (
        f"seq2seq model {seq2seq_pick!r} should be marked L3-tested"
    )
    assert models[causal_pick].l3_synthetic_parity, (
        f"causal-LM model {causal_pick!r} should still be marked L3-tested"
    )


# ---------------------------------------------------------------------------
# Regex helper: ``_parse_status_dict``.
# ---------------------------------------------------------------------------
# Used by every scanner that reads a ``{NAME}: dict[str, str] = {...}``
# literal from a test file's source. Must handle hyphenated keys
# (``xlm-roberta``, ``data2vec-text``, etc.) and ignore docstring
# mentions of the dict name.
# ---------------------------------------------------------------------------
def _write_parity_test_stub(
    tmp_repo: Path,
    *,
    dicts: dict[str, dict[str, str]] | None = None,
) -> None:
    """Write a minimal ``synthetic_parity_test.py`` containing the six status dicts.

    Allows ``_scan_l3_parity_status`` to be exercised with a known set
    of entries (defaulting to empty dicts) without depending on the
    real test file.
    """
    tests_dir = tmp_repo / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    all_dicts = {
        "_SKIP_REASONS": {},
        "_XFAIL_REASONS": {},
        "_ENCODER_SKIP_REASONS": {},
        "_ENCODER_XFAIL_REASONS": {},
        "_SEQ2SEQ_SKIP_REASONS": {},
        "_SEQ2SEQ_XFAIL_REASONS": {},
    }
    if dicts:
        all_dicts.update(dicts)

    def _fmt(d: dict[str, str]) -> str:
        if not d:
            return "{}"
        body = ",\n    ".join(f'"{k}": "{v}"' for k, v in d.items())
        return "{\n    " + body + ",\n}"

    lines = [f"{name}: dict[str, str] = {_fmt(d)}\n" for name, d in all_dicts.items()]
    (tests_dir / "synthetic_parity_test.py").write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.parametrize(
    "dict_name,expected_status",
    [
        ("_ENCODER_SKIP_REASONS", "skip"),
        ("_ENCODER_XFAIL_REASONS", "xfail"),
        ("_SEQ2SEQ_SKIP_REASONS", "skip"),
        ("_SEQ2SEQ_XFAIL_REASONS", "xfail"),
    ],
)
def test_l3_status_parses_encoder_and_seq2seq_dicts(gd, tmp_repo, dict_name, expected_status):
    """L3 status reflects all six skip/xfail dicts, not only the causal-LM pair."""
    _write_parity_test_stub(tmp_repo, dicts={dict_name: {"layoutlmv2": "needs bbox inputs"}})
    info = _build_model(gd, model_type="layoutlmv2", family="layoutlm")
    info.l3_synthetic_parity = True
    models = {"layoutlmv2": info}
    gd._scan_l3_parity_status(models)
    assert info.l3_status == expected_status
    assert info.l3_status_reason == "needs bbox inputs"


@pytest.mark.parametrize(
    "hyphenated_key",
    ["xlm-roberta", "data2vec-text", "nllb-moe", "megatron-bert", "roberta-prelayernorm"],
)
def test_l3_status_parses_hyphenated_keys(gd, tmp_repo, hyphenated_key):
    r"""L3 status parses hyphenated registry keys.

    Several encoder and seq2seq architectures register hyphenated
    ``model_type`` strings; the dict-parsing regex must use ``[\w-]+``
    rather than ``\w+`` for keys.
    """
    _write_parity_test_stub(
        tmp_repo, dicts={"_ENCODER_XFAIL_REASONS": {hyphenated_key: "position_ids offset"}}
    )
    info = _build_model(gd, model_type=hyphenated_key, family=hyphenated_key)
    info.l3_synthetic_parity = True
    models = {hyphenated_key: info}
    gd._scan_l3_parity_status(models)
    assert info.l3_status == "xfail", (
        f"hyphenated key {hyphenated_key!r} should have been parsed"
    )


def test_parse_status_dict_handles_underscores_and_hyphens(gd):
    """Direct unit test: ``_parse_status_dict`` extracts underscore and hyphen keys."""
    content = """
_FOO: dict[str, str] = {
    "plain": "underscore_value",
    "with-hyphen": "value with spaces",
    "snake_case": "ok",
}
"""
    parsed = gd._parse_status_dict(content, "_FOO")
    assert parsed == {
        "plain": "underscore_value",
        "with-hyphen": "value with spaces",
        "snake_case": "ok",
    }


def test_parse_status_dict_ignores_docstring_mention_of_dict_name(gd):
    """``_parse_status_dict`` matches the definition, not a docstring mention.

    The first textual occurrence of the dict name in a real test file
    is often inside a docstring or comment. The anchored regex must
    skip those and find the actual ``NAME: dict[str, str] = {...}``
    definition.
    """
    content = """
# * ``_GRAPH_ONLY_XFAILS`` — described in the docstring before any code.

_PARSE_AND_GRAPH_XFAILS: dict[str, str] = {}

_GRAPH_ONLY_XFAILS: dict[str, str] = {
    "fuyu": "FuyuConfig has no vision_config",
    "florence2": "DaViT multi-stage encoder",
}
"""
    assert gd._parse_status_dict(content, "_PARSE_AND_GRAPH_XFAILS") == {}
    parsed = gd._parse_status_dict(content, "_GRAPH_ONLY_XFAILS")
    assert parsed == {
        "fuyu": "FuyuConfig has no vision_config",
        "florence2": "DaViT multi-stage encoder",
    }


# ---------------------------------------------------------------------------
# Scanner tests: YAML case → ``ModelInfo`` mapping.
# ---------------------------------------------------------------------------
# YAML files name an HF ``model_id``; the scanner must map back to one
# (and only one) registry ``model_type``. When two entries share a
# ``test_model_id``, the YAML's own ``model_type`` field is
# authoritative.
# ---------------------------------------------------------------------------
def test_yaml_model_type_field_disambiguates_shared_test_model_id(gd, tmp_repo):
    """YAML ``model_type`` field takes precedence over the ``model_id`` index.

    When two registry entries share a ``test_model_id``, only the entry
    named by the YAML's ``model_type`` field receives the case.
    """
    (tmp_repo / "testdata" / "cases" / "vision" / "dinov2-small.yaml").parent.mkdir(
        parents=True, exist_ok=True
    )
    (tmp_repo / "testdata" / "cases" / "vision" / "dinov2-small.yaml").write_text(
        'model_id: "facebook/dinov2-small"\n'
        'model_type: "dinov2"\n'
        'level: "L4"\n'
        'skip_reason: "architecture diverges from generic ViT"\n',
        encoding="utf-8",
    )

    models = {
        "dinov2": _build_model(
            gd, model_type="dinov2", family="dinov2", test_model_id="facebook/dinov2-small"
        ),
        "dinov3_vit": _build_model(
            gd, model_type="dinov3_vit", family="dinov3", test_model_id="facebook/dinov2-small"
        ),
    }
    gd._scan_yaml_test_cases(models)

    # Only dinov2 should claim the YAML.
    assert models["dinov2"].l4_status == "skip"
    assert models["dinov2"].yaml_test_case_file is not None
    assert models["dinov3_vit"].l4_status is None, (
        "dinov3_vit must NOT inherit the dinov2 YAML case via shared test_model_id"
    )
    assert models["dinov3_vit"].yaml_test_case_file is None


def test_yaml_falls_back_to_model_id_index_when_no_model_type(gd, tmp_repo):
    """YAML without a ``model_type`` field falls back to the ``model_id`` index."""
    (tmp_repo / "testdata" / "cases" / "text" / "legacy.yaml").parent.mkdir(
        parents=True, exist_ok=True
    )
    (tmp_repo / "testdata" / "cases" / "text" / "legacy.yaml").write_text(
        'model_id: "fake/legacy"\nlevel: "L4"\n',
        encoding="utf-8",
    )

    models = {
        "legacy": _build_model(
            gd, model_type="legacy", family="legacy", test_model_id="fake/legacy"
        )
    }
    gd._scan_yaml_test_cases(models)
    assert models["legacy"].l4_has_test_case is True


# ---------------------------------------------------------------------------
# YAML metadata fields exposed through to summary and JSON.
# ---------------------------------------------------------------------------
def test_ci_skip_reason_surfaces_in_summary_and_json(gd, tmp_repo):
    """``ci_skip_reason`` flows through to ``summary['ci_skip_count']`` and JSON.

    The case still counts as L4-configured (it passes locally); the
    field is exposed so the template can flag it as CI-only.
    """
    (tmp_repo / "testdata" / "cases" / "text" / "ci-only.yaml").parent.mkdir(
        parents=True, exist_ok=True
    )
    (tmp_repo / "testdata" / "cases" / "text" / "ci-only.yaml").write_text(
        'model_id: "fake/ci-only"\n'
        'model_type: "ci_only"\n'
        'level: "L4"\n'
        'ci_skip_reason: "downloads 30GB weights"\n',
        encoding="utf-8",
    )
    models = {
        "ci_only": _build_model(
            gd, model_type="ci_only", family="ci_only", test_model_id="fake/ci-only"
        )
    }
    gd._scan_yaml_test_cases(models)
    info = models["ci_only"]
    assert info.yaml_test_case_ci_skip_reason == "downloads 30GB weights"
    assert info.l4_has_test_case is True  # local still counts
    summary = gd._compute_summary(models)
    assert summary["ci_skip_count"] == 1

    html = gd._render_html(models)
    m = _extract_model_data_json(html)[0]
    assert m["yaml_ci_skip_reason"] == "downloads 30GB weights"


# ---------------------------------------------------------------------------
# Meta-tests against the real repository.
# ---------------------------------------------------------------------------
# These run ``collect_all_model_info`` once against the live
# ``testdata/`` and ``tests/`` directories and assert global invariants.
# They guard against future skip/xfail dicts being added without a
# matching scanner update.
# ---------------------------------------------------------------------------
def test_meta_no_skipped_or_xfailed_model_is_counted_as_passing(gd):
    """Every model in a known skip/xfail dict is downgraded by the dashboard.

    Auto-discovers L2 and L3 skip/xfail dicts from the real test files
    and runs the full ``collect_all_model_info`` pipeline. Catches the
    case where a new dict is added without a matching scanner update.
    """
    parity_test = (_REPO_ROOT / "tests" / "synthetic_parity_test.py").read_text(
        encoding="utf-8"
    )
    arch_test = (_REPO_ROOT / "tests" / "arch_validation_test.py").read_text(encoding="utf-8")

    l3_dicts = [
        "_SKIP_REASONS",
        "_XFAIL_REASONS",
        "_ENCODER_SKIP_REASONS",
        "_ENCODER_XFAIL_REASONS",
        "_SEQ2SEQ_SKIP_REASONS",
        "_SEQ2SEQ_XFAIL_REASONS",
    ]
    l2_dicts = ["_PARSE_AND_GRAPH_XFAILS", "_GRAPH_ONLY_XFAILS"]

    l3_skipped_or_xfailed: set[str] = set()
    for name in l3_dicts:
        l3_skipped_or_xfailed.update(gd._parse_status_dict(parity_test, name))
    l2_xfailed: set[str] = set()
    for name in l2_dicts:
        l2_xfailed.update(gd._parse_status_dict(arch_test, name))

    if not l3_skipped_or_xfailed and not l2_xfailed:
        pytest.skip("No skip/xfail dicts populated — nothing to check")

    models = gd.collect_all_model_info()

    for mt in l3_skipped_or_xfailed:
        if mt not in models:
            continue
        info = models[mt]
        assert info.l3_passes is False, (
            f"{mt!r} is in an L3 skip/xfail dict yet l3_passes is True; "
            f"status={info.l3_status!r}"
        )

    for mt in l2_xfailed:
        if mt not in models:
            continue
        info = models[mt]
        assert info.l2_passes is False, (
            f"{mt!r} is in an L2 xfail dict yet l2_passes is True; status={info.l2_status!r}"
        )
        assert info.confidence_level < 2, (
            f"{mt!r} L2 xfail must not be counted toward confidence_level; "
            f"got {info.confidence_level}"
        )


# ---------------------------------------------------------------------------
# Property tests: ``confidence_level`` truth table.
# ---------------------------------------------------------------------------
# ``confidence_level`` returns the highest passing level. The contract:
# L5 > L4 > L3-pass > L2-pass > L1 > 0. Skip/xfail statuses at any level
# must transparently fall through to the next lower passing signal.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "flags,expected",
    [
        # (l1, l2_passes_arg, l3_status, l4, l5, expected_level)
        # Fully untested.
        (("none", "none", "none", "none", "none"), 0),
        # Each level alone.
        (("l1", "none", "none", "none", "none"), 1),
        (("l1", "pass", "none", "none", "none"), 2),
        (("l1", "pass", "pass", "none", "none"), 3),
        (("l1", "pass", "pass", "l4", "none"), 4),
        (("l1", "pass", "pass", "l4", "l5"), 5),
        # Higher-level pass wins over lower-level pass.
        (("l1", "pass", "pass", "l4", "none"), 4),
        # L5 alone is a strange-but-legal state (golden file with no
        # other coverage configured). Should still return 5.
        (("none", "none", "none", "none", "l5"), 5),
        # L3 skip falls through to L2.
        (("l1", "pass", "skip", "none", "none"), 2),
        # L3 xfail falls through to L2.
        (("l1", "pass", "xfail", "none", "none"), 2),
        # L2 xfail falls through to L1; L3 pass overrides regardless.
        (("l1", "xfail", "pass", "none", "none"), 3),
        # L2 xfail with no L3 falls to L1.
        (("l1", "xfail", "none", "none", "none"), 1),
        # L2 xfail_graph_only behaves like xfail for confidence purposes.
        (("l1", "xfail_graph_only", "none", "none", "none"), 1),
        # L4 set with L2 xfail still gives 4 (L4 is independent of L2 status).
        (("l1", "xfail", "none", "l4", "none"), 4),
    ],
)
def test_confidence_level_truth_table(gd, flags, expected):
    """``confidence_level`` returns the highest passing level across states.

    Exhaustively covers the interesting combinations of L1 / L2-status /
    L3-status / L4 / L5 to lock in the ordering rule.
    """
    l1_arg, l2_arg, l3_arg, l4_arg, l5_arg = flags
    info = gd.ModelInfo(
        model_type="x",
        module_class_name="X",
        task="text-generation",
        category="Causal LM",
        family="x",
        l1_graph_build=(l1_arg == "l1"),
        l2_arch_validation=(l2_arg != "none"),
        l2_status=l2_arg if l2_arg != "none" else None,
        l3_synthetic_parity=(l3_arg != "none"),
        l3_status=l3_arg if l3_arg != "none" else None,
        l4_has_test_case=(l4_arg == "l4"),
        l4_status=("pass" if l4_arg == "l4" else None),
        l5_has_test_case=(l5_arg == "l5"),
        l5_status=("pass" if l5_arg == "l5" else None),
    )
    assert info.confidence_level == expected, (
        f"flags={flags} should give level {expected}, got {info.confidence_level} "
        f"(label={info.confidence_label!r})"
    )


@pytest.mark.parametrize(
    "level,expected_label",
    [
        (0, "Not Tested"),
        (1, "L1: Graph Builds"),
        (2, "L2: Config Compatible"),
        (3, "L3: Synthetic Parity"),
        (4, "L4: Golden Match"),
        (5, "L5: Generation Verified"),
    ],
)
def test_confidence_label_matches_label_table(gd, level, expected_label):
    """``confidence_label`` reads the canonical ``_CONFIDENCE_LABELS`` table."""
    assert gd._CONFIDENCE_LABELS[level] == expected_label


# ---------------------------------------------------------------------------
# Aggregation tests: ``_compute_summary`` invariants over multi-model dicts.
# ---------------------------------------------------------------------------
def _make_model(gd, name: str, **kwargs):
    """Minimal ``ModelInfo`` with sensible defaults for summary tests."""
    defaults = dict(
        model_type=name,
        module_class_name="X",
        task="text-generation",
        category="Causal LM",
        family=name,
    )
    defaults.update(kwargs)
    return gd.ModelInfo(**defaults)


def test_compute_summary_level_flags_are_independent(gd):
    """Level flag counts are independent, not partitioned by highest level.

    A model passing L1, L2, L3, L4, and L5 contributes 1 to each of
    ``by_level[1..5]`` simultaneously.
    """
    info = _make_model(
        gd,
        "stack",
        l1_graph_build=True,
        l2_arch_validation=True,
        l2_status="pass",
        l3_synthetic_parity=True,
        l3_status="pass",
        l4_has_test_case=True,
        l4_status="pass",
        l5_has_test_case=True,
        l5_status="pass",
    )
    summary = gd._compute_summary({"stack": info})
    assert summary["by_level"][1] == 1
    assert summary["by_level"][2] == 1
    assert summary["by_level"][3] == 1
    assert summary["by_level"][4] == 1
    assert summary["by_level"][5] == 1
    # L0 = no flags set → 0 for this fully-covered model.
    assert summary["by_level"][0] == 0


def test_compute_summary_counts_sum_correctly_across_many_models(gd):
    """L0 + sum(positive flags) ≥ total; level counts are non-exclusive."""
    models = {
        "a": _make_model(gd, "a"),  # untested
        "b": _make_model(gd, "b", l1_graph_build=True),  # L1 only
        "c": _make_model(
            gd, "c", l1_graph_build=True, l2_arch_validation=True, l2_status="pass"
        ),  # L1+L2
        "d": _make_model(
            gd,
            "d",
            l1_graph_build=True,
            l3_synthetic_parity=True,
            l3_status="skip",
        ),  # L1 + L3-skip → not tested at L3
    }
    summary = gd._compute_summary(models)
    assert summary["total"] == 4
    assert summary["by_level"][0] == 1, "model 'a' has no flags"
    assert summary["by_level"][1] == 3, "models b, c, d all set l1_graph_build"
    assert summary["by_level"][2] == 1, "only c passes L2"
    assert summary["by_level"][3] == 0, "d's L3 is skipped, not passing"
    # L3 status histogram counts every model with any l3_status set.
    assert summary["l3_status_counts"]["skip"] == 1
    assert summary["l3_status_counts"]["untested"] == 3


def test_compute_summary_l2_status_counts_track_configured_models(gd):
    """``l2_status_counts`` only counts models that are L2-configured.

    Models without ``test_model_id`` (i.e. ``l2_arch_validation`` is
    False) should not appear in any status bucket.
    """
    models = {
        "pass": _make_model(gd, "pass", l2_arch_validation=True, l2_status="pass"),
        "xf": _make_model(gd, "xf", l2_arch_validation=True, l2_status="xfail"),
        "xfg": _make_model(gd, "xfg", l2_arch_validation=True, l2_status="xfail_graph_only"),
        "unconfigured": _make_model(gd, "unconfigured"),
    }
    summary = gd._compute_summary(models)
    assert summary["l2_status_counts"]["pass"] == 1
    assert summary["l2_status_counts"]["xfail"] == 1
    assert summary["l2_status_counts"]["xfail_graph_only"] == 1
    # 'unconfigured' has no l2_arch_validation → not counted at all.
    assert sum(summary["l2_status_counts"].values()) == 3, (
        f"unexpected status counts: {summary['l2_status_counts']}"
    )


# ---------------------------------------------------------------------------
# Aggregation tests: ``_build_component_matrix`` multi-family invariants.
# ---------------------------------------------------------------------------
def test_build_component_matrix_aggregates_max_level_per_family(gd):
    """Cells store ``max(confidence_level)`` over all models in (component, family).

    Two models in the same family contributing to the same component
    must yield a single cell at the higher of their confidence levels.
    """
    from mobius._testing.code_paths import CODE_PATH_INDICATORS

    indicator = CODE_PATH_INDICATORS[0].label
    fam = "famA"
    low = _make_model(gd, "lo", family=fam, l1_graph_build=True)
    low.code_paths.add(indicator)
    high = _make_model(
        gd,
        "hi",
        family=fam,
        l1_graph_build=True,
        l2_arch_validation=True,
        l2_status="pass",
        l3_synthetic_parity=True,
        l3_status="pass",
    )
    high.code_paths.add(indicator)

    matrix = gd._build_component_matrix({"lo": low, "hi": high})
    row = next(r for r in matrix["rows"] if r["label"] == indicator)
    fam_idx = matrix["families"].index(fam)
    assert row["cells"][fam_idx] == 3, f"max(L1, L3) should be 3; got {row['cells'][fam_idx]}"
    assert row["model_count"] == 2
    assert row["family_count"] == 1
    assert row["best_level"] == 3


def test_build_component_matrix_distinct_families_share_indicator(gd):
    """Two families exercising the same component produce two cells."""
    from mobius._testing.code_paths import CODE_PATH_INDICATORS

    indicator = CODE_PATH_INDICATORS[0].label
    a = _make_model(gd, "a", family="famA", l1_graph_build=True)
    a.code_paths.add(indicator)
    b = _make_model(
        gd, "b", family="famB", l1_graph_build=True, l2_arch_validation=True, l2_status="pass"
    )
    b.code_paths.add(indicator)

    matrix = gd._build_component_matrix({"a": a, "b": b})
    row = next(r for r in matrix["rows"] if r["label"] == indicator)
    assert matrix["families"] == ["famA", "famB"]
    assert row["cells"] == [1, 2]
    assert row["family_count"] == 2
    assert row["best_level"] == 2


def test_build_component_matrix_omits_families_without_code_paths(gd):
    """Families whose models exercise no indicators do not appear in the matrix.

    Keeps the heatmap focused on architectures with at least one tested
    component.
    """
    from mobius._testing.code_paths import CODE_PATH_INDICATORS

    a = _make_model(gd, "a", family="withpath", l1_graph_build=True)
    a.code_paths.add(CODE_PATH_INDICATORS[0].label)
    b = _make_model(gd, "b", family="nopath", l1_graph_build=True)

    matrix = gd._build_component_matrix({"a": a, "b": b})
    assert matrix["families"] == ["withpath"]


# ---------------------------------------------------------------------------
# Rendering tests: ``_render_html`` HTML/JSON well-formedness.
# ---------------------------------------------------------------------------
def test_render_html_emits_parseable_json_for_every_model(gd):
    """The embedded ``MODEL_DATA`` JSON parses and has one entry per model."""
    models = {
        "a": _make_model(gd, "a", l1_graph_build=True),
        "b": _make_model(gd, "b"),
    }
    html = gd._render_html(models)
    data = _extract_model_data_json(html)
    assert {m["model_type"] for m in data} == {"a", "b"}


def test_render_html_summary_json_matches_compute_summary(gd):
    """The embedded ``SUMMARY`` blob matches ``_compute_summary`` output."""
    models = {
        "a": _make_model(gd, "a", l1_graph_build=True),
        "b": _make_model(
            gd, "b", l1_graph_build=True, l2_arch_validation=True, l2_status="pass"
        ),
    }
    html = gd._render_html(models)
    summary = gd._compute_summary(models)
    embedded = re.search(r"SUMMARY\s*=\s*(\{.*?\});", html, re.DOTALL)
    assert embedded is not None, "SUMMARY blob missing from HTML"
    embedded_obj = json.loads(embedded.group(1))
    assert embedded_obj["total"] == summary["total"]
    # by_level dict keys round-trip through JSON as strings, so compare
    # via int conversion.
    assert {int(k): v for k, v in embedded_obj["by_level"].items()} == summary["by_level"]


def test_render_html_includes_commit_when_provided(gd):
    """``commit`` argument is HTML-escaped and embedded in the subtitle."""
    info = _make_model(gd, "x", l1_graph_build=True)
    html = gd._render_html({"x": info}, commit="abc1234")
    assert "abc1234" in html


def test_render_html_marks_unknown_commit_when_omitted(gd):
    """Default ``commit`` of ``None`` renders as the literal ``unknown``."""
    info = _make_model(gd, "x", l1_graph_build=True)
    html = gd._render_html({"x": info})
    assert "unknown" in html


def test_to_js_json_escapes_script_tag_close(gd):
    r"""``_to_js_json`` rewrites ``</`` to ``<\/`` to prevent script-tag escape."""
    payload = {"x": "</script><script>alert(1)</script>"}
    out = gd._to_js_json(payload)
    assert "</script>" not in out, (
        "raw </script> in serialized JSON would close the inline <script>"
    )
    assert "<\\/script>" in out


# ---------------------------------------------------------------------------
# YAML metadata: ``min_token_match_ratio`` propagation.
# ---------------------------------------------------------------------------
def test_yaml_min_token_match_ratio_propagates_to_model_info(gd, tmp_repo):
    """``min_token_match_ratio`` from YAML lands on ``ModelInfo`` and JSON."""
    (tmp_repo / "testdata" / "cases" / "text" / "case.yaml").parent.mkdir(
        parents=True, exist_ok=True
    )
    (tmp_repo / "testdata" / "cases" / "text" / "case.yaml").write_text(
        'model_id: "fake/x"\nmodel_type: "x"\nlevel: "L5"\nmin_token_match_ratio: 0.75\n',
        encoding="utf-8",
    )
    models = {"x": _build_model(gd, model_type="x", family="x", test_model_id="fake/x")}
    gd._scan_yaml_test_cases(models)
    assert models["x"].yaml_min_token_match_ratio == pytest.approx(0.75)

    html = gd._render_html(models)
    m = _extract_model_data_json(html)[0]
    assert m["min_token_match_ratio"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# End-to-end smoke tests against the real repository.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def live_models(gd):
    """Run the full collection pipeline against the real repository once."""
    return gd.collect_all_model_info()


def test_live_collect_yields_at_least_one_model(live_models):
    """Sanity: the pipeline returns non-empty model info."""
    assert len(live_models) > 0


def test_live_render_produces_valid_html_and_parseable_json(gd, live_models):
    """End-to-end: real data renders to HTML whose embedded JSON parses."""
    html = gd._render_html(live_models, commit="livetest")
    # Embedded MODEL_DATA must parse and have one entry per live model.
    data = _extract_model_data_json(html)
    assert len(data) == len(live_models)
    # Embedded SUMMARY must parse and report the same total.
    summary_blob = re.search(r"SUMMARY\s*=\s*(\{.*?\});", html, re.DOTALL)
    assert summary_blob is not None
    assert json.loads(summary_blob.group(1))["total"] == len(live_models)


def test_live_confidence_levels_all_in_range(live_models):
    """No model reports a ``confidence_level`` outside [0, 5]."""
    out_of_range = [
        (mt, info.confidence_level)
        for mt, info in live_models.items()
        if not (0 <= info.confidence_level <= 5)
    ]
    assert not out_of_range, f"models with bad confidence: {out_of_range}"


def test_live_passing_implies_configured(live_models):
    """An ``lN_passes``-True model is also configured at level N.

    Symmetry check: passing without being configured would indicate the
    status scanner ran ahead of the configuration scanner, or that the
    'pass' default leaked onto a model that isn't actually parametrized.
    """
    for mt, info in live_models.items():
        if info.l2_passes:
            assert info.l2_arch_validation, f"{mt} l2_passes without l2_arch_validation"
        if info.l3_passes:
            assert info.l3_synthetic_parity, f"{mt} l3_passes without l3_synthetic_parity"


def test_live_yaml_test_case_file_paths_exist(live_models):
    """Every YAML path recorded on a ``ModelInfo`` resolves on disk.

    Catches the case where a scanner records a path that doesn't exist
    (e.g. from a path-resolution bug or a partial filesystem state).
    """
    bad = [
        (mt, info.yaml_test_case_file)
        for mt, info in live_models.items()
        if info.yaml_test_case_file is not None
        and not (_REPO_ROOT / info.yaml_test_case_file).is_file()
    ]
    assert not bad, f"YAML paths recorded but missing: {bad}"
