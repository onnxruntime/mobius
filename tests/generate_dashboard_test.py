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


# ---------------------------------------------------------------------------
# Template integrity — dot icon CSS + legend + JS branches all in sync.
# ---------------------------------------------------------------------------
_TEMPLATE_PATH = _REPO_ROOT / "scripts" / "templates" / "dashboard.html.j2"


@pytest.mark.parametrize(
    "css_class",
    ["xfail", "skipped", "pending", "untested", "failed"],
)
def test_template_has_css_and_legend_for_dot_class(css_class):
    """Every dot state used by the JS must have CSS and a legend entry."""
    text = _TEMPLATE_PATH.read_text(encoding="utf-8")
    assert f".dot.{css_class}" in text, f"CSS rule for .dot.{css_class} is missing"
    # Legend uses inline classes like ``dot xfail``; match either spelling.
    assert f"dot {css_class}" in text or f"dot.{css_class}" in text, (
        f"Legend entry referencing dot.{css_class} not found"
    )


def test_template_dot_class_assignments_match_css():
    """Every JS branch that assigns a dot class must have matching CSS.

    Catches the case where a new status string is added to the JS without
    a corresponding CSS rule (resulting in an unstyled dot). The
    ``active-`` family is dynamic (``active-1`` .. ``active-5``) and is
    verified separately via :func:`test_template_has_css_and_legend_for_dot_class`-style
    checks on the active levels.
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
    """Hand-port of the JS dot logic for L2 / L3 states.

    This is a small re-implementation of the ``dots.map`` logic in
    ``renderModelRow`` so we can assert the chosen class for a given
    ``ModelInfo`` state without spinning up a headless browser. Kept in
    sync with the template by reading it side-by-side during review.
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
# L2 xfail handling (issue A from the follow-up plan).
# ---------------------------------------------------------------------------
def _write_arch_validation_stub(
    tmp_repo: Path,
    *,
    parse_and_graph_xfails: dict[str, str] | None = None,
    graph_only_xfails: dict[str, str] | None = None,
) -> None:
    """Write a minimal arch_validation_test.py stub.

    Provides just enough content for ``_scan_l2_arch_status`` to parse.
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
    """``_scan_l2_arch_status`` must record xfail status from both dicts.

    Mirrors the L3 status scanner: distinguishes "parse+graph" xfail from
    "graph-only" xfail so the template can show different annotations.
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
    """Models with L2 coverage and not in any xfail dict → status="pass"."""
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
    """L2 xfail must not inflate ``confidence_level`` past L1."""
    info = _build_model(gd, l1=True, l2=True)
    info.l2_status = l2_status
    info.l2_status_reason = "config requires trust_remote_code"
    assert info.confidence_level == 1
    assert info.confidence_label == "L1: Graph builds"


@pytest.mark.parametrize("l2_status", ["xfail", "xfail_graph_only"])
def test_compute_summary_l2_card_excludes_xfail(gd, l2_status):
    """Summary L2 card and Not-tested check must exclude xfailed L2 models."""
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
    """JSON ``m.l2`` (= ``l2_passes``) is False for xfailed L2 models.

    ``m.l2_configured`` and ``m.l2_status`` must still be exposed so the
    template can render the xfail chip.
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
# L3 scope (issue B): encoder + seq2seq must populate l3_synthetic_parity.
# ---------------------------------------------------------------------------
def test_scan_l3_includes_encoder_and_seq2seq(gd):
    """``_scan_l3_synthetic_parity`` must union all three config lists.

    Not just causal-LM. The production scanner is import-based, so this
    test exercises the real ``tests/_test_configs.py``.
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
# L3 status scope (issues C + D): all 6 dicts parsed; hyphenated keys.
# ---------------------------------------------------------------------------
def _write_parity_test_stub(
    tmp_repo: Path,
    *,
    dicts: dict[str, dict[str, str]] | None = None,
) -> None:
    """Write a minimal synthetic_parity_test.py stub with the six dicts.

    Each dict is emitted as an empty literal unless overridden.
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
    """``_scan_l3_parity_status`` must parse all six dicts, not just causal-LM."""
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
    r"""Hyphenated registry keys in encoder/seq2seq dicts must be parsed.

    Original regex used ``\w+`` which excludes hyphens, silently dropping
    these entries. Fix widens to ``[\w-]+``.
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
    """Direct unit test for the regex helper."""
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
    """Docstring mentions of the dict name must not derail the regex.

    Previous regex used lazy ``.*?=`` which would skip ahead to the next
    ``=`` after a docstring mention, capturing an empty body if the next
    dict happened to be empty. Mirrors the real
    ``tests/arch_validation_test.py`` layout where the first
    ``_GRAPH_ONLY_XFAILS`` token is inside a docstring and the next ``=``
    belongs to an empty ``_PARSE_AND_GRAPH_XFAILS``.
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
# E: YAML model_type field disambiguates colliding test_model_ids.
# ---------------------------------------------------------------------------
def test_yaml_model_type_field_disambiguates_shared_test_model_id(gd, tmp_repo):
    """YAML ``model_type`` field overrides the ``model_id`` reverse index.

    When two registry entries share ``test_model_id``, the YAML's own
    ``model_type`` field must determine which entry the case attaches to.
    Reproduces the dinov2 vs dinov3_vit collision from issue #211.
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
    assert models["dinov2"].l4_test_case_skipped is True
    assert models["dinov2"].yaml_test_case_file is not None
    assert models["dinov3_vit"].l4_test_case_skipped is False, (
        "dinov3_vit must NOT inherit the dinov2 YAML case via shared test_model_id"
    )
    assert models["dinov3_vit"].yaml_test_case_file is None


def test_yaml_falls_back_to_model_id_index_when_no_model_type(gd, tmp_repo):
    """Legacy YAML without ``model_type`` field falls back to model_id reverse index."""
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
# F: ci_skip_reason flows from YAML through summary and JSON.
# ---------------------------------------------------------------------------
def test_ci_skip_reason_surfaces_in_summary_and_json(gd, tmp_repo):
    """``ci_skip_reason`` is exposed as a summary count and JSON field.

    Local-only tests still count as passing coverage.
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
# G: Meta-test — every skip/xfail dict entry must be reflected in the
# dashboard (i.e. the corresponding lN_passes is False for those models).
# ---------------------------------------------------------------------------
def test_meta_no_skipped_or_xfailed_model_is_counted_as_passing(gd):
    """Meta-check: every skip/xfail dict entry must downgrade its model.

    Walks every skip/xfail dict declared in the real test files and asserts
    that ``_scan_*_status`` properly downgrades each one. Prevents a future
    skip/xfail dict from being added without a corresponding scanner update.
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
