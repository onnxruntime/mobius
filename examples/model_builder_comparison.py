#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

r"""Compare ONNX model graph structure between ORT GenAI's model builder and mobius.

Produces a presentation-quality report showing op-count parity, PASS/FAIL/PARTIAL
verdicts, and plain-English explanations for any observed differences.  Outputs both
a readable console summary and a self-contained Markdown file suitable for sharing
with stakeholders.

----------------------------------------------------------------------
Usage examples
----------------------------------------------------------------------

Compare a single model across all EPs (no weights downloaded — fast):

    python examples/model_builder_comparison.py \\
        --model meta-llama/Llama-3.2-1B \\
        --ep-list default,cuda,dml,webgpu \\
        --no-ort

Compare against ORT GenAI builder — ALL EPs by default (no --ep needed):

    ORT_GENAI_REPO=/path/to/onnxruntime-genai \\
    python examples/model_builder_comparison.py \\
        --model meta-llama/Llama-3.2-1B

  This shows a wide table: Op | ORT GenAI | default | cuda | dml | webgpu | ...
  with a per-EP verdict row at the bottom.

Compare a single specific EP against ORT GenAI:

    ORT_GENAI_REPO=/path/to/onnxruntime-genai \\
    python examples/model_builder_comparison.py \\
        --model meta-llama/Llama-3.2-1B \\
        --ep cuda

Compare against a pre-built ORT GenAI model dir:

    python examples/model_builder_comparison.py \\
        --model meta-llama/Llama-3.2-1B \\
        --ep cuda \\
        --ort-model /path/to/ort-output/

Compare multiple models and save the report:

    python examples/model_builder_comparison.py \\
        --model meta-llama/Llama-3.2-1B \\
        --model Qwen/Qwen3-0.6B \\
        --ep cuda \\
        --output report.md
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform as _platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import onnx_ir as ir
from onnx_ir.traversal import RecursiveGraphIterator

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"

STANDARD_SUITE = [
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen3-0.6B",
    "google/gemma-2-2b",
    "google/gemma-3-1b-pt",
    "mistralai/Mistral-7B-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
]

# ---------------------------------------------------------------------------
# Tracked ops and their descriptions
# ---------------------------------------------------------------------------

# (op_type, short_label, explanation_if_absent_from_mobius)
_OP_CATALOG: list[tuple[str, str, str]] = [
    (
        "GroupQueryAttention",
        "GQA",
        "GQA is emitted when the EP supports it (cuda, webgpu, dml, cpu). "
        "Default EP keeps standard Attention + RotaryEmbedding ops.",
    ),
    (
        "Attention",
        "Attention",
        "Standard ONNX opset-23 Attention. Used when GQA is not fused.",
    ),
    (
        "MultiHeadAttention",
        "MultiHeadAttention",
        "ORT custom MultiHeadAttention (cross-attention in encoder-decoder).",
    ),
    (
        "RotaryEmbedding",
        "RotaryEmbedding",
        "RoPE kept explicit (default EP) or after SeparateRoPE lowering (DML). "
        "Absent when fused into GQA (cuda).",
    ),
    (
        "SkipLayerNormalization",
        "SkipLayerNorm",
        "Add + LayerNorm fused. Absent for RMSNorm-only models.",
    ),
    # NOTE: ORT GenAI uses the name "SimplifiedLayerNormalization" for unfused RMSNorm
    # and "SkipSimplifiedLayerNormalization" for the fused Skip+RMSNorm.
    # Mobius uses the standard ONNX opset-23 name "RMSNormalization".
    # These represent the same mathematical operation.
    (
        "SkipSimplifiedLayerNormalization",
        "SkipSimplifiedLayerNorm",
        "Add + RMSNorm fused (com.microsoft custom op). "
        "One fewer than ORT GenAI if mobius keeps the final norm as standard ONNX.",
    ),
    (
        "LayerNormalization",
        "LayerNorm",
        "Standard ONNX LayerNorm. Unfused residual norm.",
    ),
    (
        "RMSNormalization",
        "RMSNorm (ONNX)",
        "Standard ONNX opset-23 RMSNorm. ORT GenAI uses 'SimplifiedLayerNormalization' for the same op. "
        "Mobius emits this for norms not covered by SkipSimplifiedLayerNorm fusion (embedding norm, final norm).",
    ),
    (
        "SimplifiedLayerNormalization",
        "SimplifiedLayerNorm (ORT)",
        # ORT GenAI uses this com.microsoft op for standalone (unfused) RMSNorm.
        # Mobius emits the standard ONNX 'RMSNormalization' for the same node.
        "ORT GenAI uses this for standalone (unfused) RMSNorm. "
        "Mobius emits 'RMSNormalization' for the same node — same math, different op name.",
    ),
    (
        "BiasGelu",
        "BiasGelu",
        "Fused Bias + GELU (com.microsoft). Absent when bias is zero or unfused.",
    ),
    (
        "FastGelu",
        "FastGelu",
        "ORT FastGelu approximation. Mobius uses standard Gelu.",
    ),
    (
        "Gelu",
        "Gelu",
        "Standard ONNX Gelu. Used when BiasGelu/FastGelu not applicable.",
    ),
    (
        "MatMul",
        "MatMul",
        "Plain matrix multiply. Count differs when QKV projections are packed "
        "differently (ORT GenAI sometimes packs Q/K/V into one MatMul).",
    ),
    (
        "Shape",
        "Shape",
        "Shape op. Should be 0 on WebGPU after EliminateShape lowering.",
    ),
    (
        "Cast",
        "Cast",
        "Type cast. Extra Casts appear for seqlen inputs in GQA mode.",
    ),
    (
        "Gather",
        "Gather",
        "Embedding lookup. Count should match across builders.",
    ),
    (
        "Softmax",
        "Softmax",
        "Softmax. Present when attention is not fully fused.",
    ),
    (
        "MoE",
        "MoE",
        "Fused Mixture-of-Experts op (com.microsoft).",
    ),
]

OP_TYPES = [op for op, _, _ in _OP_CATALOG]
OP_LABEL = {op: label for op, label, _ in _OP_CATALOG}
OP_EXPLANATION = {op: expl for op, _, expl in _OP_CATALOG}

# Ops whose count mismatch signals a real parity problem
_CRITICAL_OPS = {"GroupQueryAttention", "BiasGelu", "MoE"}


# ---------------------------------------------------------------------------
# QK-norm model families
# ---------------------------------------------------------------------------

# Models in these families use per-head Q and K normalization. This prevents
# QKV weight packing (ORT GenAI `use_packed_matmul=False`) but does NOT prevent
# GQA. Both builders should emit the same GQA count AND both should have
# separate Q/K/V MatMuls (higher total MatMul count vs packed-QKV models).
_QK_NORM_FAMILIES: tuple[str, ...] = (
    "Qwen/Qwen3",
    "Qwen/Qwen3.5",
    "Qwen3",
    "Qwen3.5",
    "qwen3",
    "qwen3.5",
)


def _is_qk_norm_model(model_id: str) -> bool:
    """Return True if model_id belongs to a known QK-norm family."""
    lower = model_id.lower()
    return any(lower.startswith(f.lower()) or f.lower() in lower for f in _QK_NORM_FAMILIES)


def _qk_norm_checks(report: ModelReport) -> list[str]:
    """Validate QK-norm invariants for a cross-builder report. Returns [] on success."""
    issues: list[str] = []
    if not _is_qk_norm_model(report.model_id) or len(report.columns) < 2:
        return issues

    # Find ORT GenAI and mobius columns
    ort_cols = [c for c in report.columns if c.source == "ort-genai"]
    mob_cols = [c for c in report.columns if c.source == "mobius"]
    if not ort_cols or not mob_cols:
        return issues

    ort_gqa = ort_cols[0].counts.get("GroupQueryAttention", 0)
    mob_gqa = mob_cols[0].counts.get("GroupQueryAttention", 0)

    if ort_gqa != mob_gqa:
        issues.append(
            f"GQA MISMATCH for QK-norm model: ORT GenAI={ort_gqa}, mobius={mob_gqa}. "
            "Expected equal counts — QK norm prevents QKV packing but NOT GQA."
        )

    # PackQKV absent means ORT GenAI uses 3 separate QKV MatMuls per layer,
    # same as mobius. The MatMul difference should be <= a few global nodes,
    # not 2x the QKV contribution.
    ort_mm = ort_cols[0].counts.get("MatMul", 0)
    mob_mm = mob_cols[0].counts.get("MatMul", 0)
    # Estimate layers from GQA count (1 GQA = 1 attention layer)
    est_layers = mob_gqa or 1
    # If ORT GenAI had packed QKV, matmul diff would be ~2*layers (saving 2 QKV MatMuls/layer).
    # For QK-norm models, diff should be much smaller.
    if mob_mm > 0 and (mob_mm - ort_mm) > 2 * est_layers:
        issues.append(
            f"Unexpected MatMul gap ({mob_mm} vs {ort_mm} = diff {mob_mm - ort_mm}) "
            f"for a QK-norm model ({est_layers} layers). "
            "If ORT GenAI is packing QKV for this model, that is a bug — "
            "QK norm must use separate Q/K/V projections."
        )

    # For QK-norm models, expect 3 separate QKV MatMuls per attention layer.
    # Total expected QKV MatMuls: 3 * num_layers
    if mob_mm > 0 and mob_mm < 3 * est_layers:
        issues.append(
            f"Unexpectedly low MatMul count ({mob_mm}) for {est_layers} layers. "
            "Expected at least 3 separate Q/K/V projections per layer."
        )
    # Check that ORT GenAI also has >= 3 * layers MatMuls (no packing)
    if ort_mm > 0 and ort_mm < 3 * est_layers:
        issues.append(
            f"ORT GenAI MatMul count ({ort_mm}) suggests packed QKV for {est_layers} layers. "
            "QK-norm models should use separate Q/K/V MatMuls (use_packed_matmul=False)."
        )

    return issues


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class OpCounts(NamedTuple):
    label: str  # e.g. "mobius/cuda" or "ort-genai/cuda"
    counts: dict[str, int]
    total_nodes: int
    source: str  # "mobius" | "ort-genai" | "file"


@dataclass
class ModelReport:
    model_id: str
    ep: str
    columns: list[OpCounts]
    verdict: str = ""  # "PASS" | "KNOWN_DIFF" | "FAIL" | "MOBIUS-ONLY"
    verdict_reason: str = ""
    differences: list[str] = field(default_factory=list)
    qk_norm_issues: list[str] = field(
        default_factory=list
    )  # Non-empty → QK-norm invariant violated
    expected_counts: dict[str, int] | None = None
    # Per-mobius-column verdict for ORT-all-EPs mode.
    # Each entry is (ep_name, verdict, reason) for one mobius column.
    per_ep_verdicts: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class ComparisonReport:
    generated_at: str
    mobius_version: str
    ort_genai_version: str
    python_version: str
    git_sha: str = ""  # mobius git SHA (short)
    platform_info: str = ""  # platform.platform(terse=True)
    models: list[ModelReport] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(1 for m in self.models if m.verdict == "PASS")

    @property
    def known_diff_count(self) -> int:
        return sum(1 for m in self.models if m.verdict == "KNOWN_DIFF")

    @property
    def fail_count(self) -> int:
        return sum(1 for m in self.models if m.verdict == "FAIL")


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _count_ops(model: ir.Model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in RecursiveGraphIterator(model.graph):
        op = node.op_type
        if op not in counts:
            counts[op] = 0
        counts[op] += 1
    return counts


def _op_counts_from_file(path: str | Path) -> OpCounts:
    model = ir.load(str(path))
    counts = _count_ops(model)
    total = sum(counts.values())
    return OpCounts(label="ort-genai (file)", counts=counts, total_nodes=total, source="file")


def _detect_model_dtype(model_id: str):
    """Return the ir.DataType for a HuggingFace model, defaulting to FLOAT."""
    import onnx_ir as ir
    import transformers

    from mobius._configs import _resolve_dtype

    try:
        hf_config = transformers.AutoConfig.from_pretrained(model_id)
        resolved = _resolve_dtype(hf_config)
    except Exception:
        return ir.DataType.FLOAT
    else:
        return resolved if resolved is not None else ir.DataType.FLOAT


def build_mobius(model_id: str, ep: str, load_weights: bool = False) -> OpCounts:
    from mobius import build

    pkg = build(model_id, execution_provider=ep, load_weights=load_weights)
    role = "model" if "model" in pkg else next(iter(pkg))
    counts = _count_ops(pkg[role])
    return OpCounts(
        label=f"mobius/{ep}",
        counts=counts,
        total_nodes=sum(counts.values()),
        source="mobius",
    )


def build_ort_genai(
    model_id: str,
    ep: str,
    precision: str,
    ort_genai_repo: str,
) -> OpCounts:
    builders_dir = os.path.join(ort_genai_repo, "src/python/py/models")
    builder_file = os.path.join(builders_dir, "builder.py")
    if not os.path.isfile(builder_file):
        raise FileNotFoundError(
            f"ORT GenAI builder not found at {builder_file}. "
            "Check --ort-genai-repo / $ORT_GENAI_REPO."
        )
    # Use spec_from_file_location to avoid polluting sys.path.
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "ort_genai_builder",
        builder_file,
        submodule_search_locations=[builders_dir],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load ORT GenAI builder from {builder_file}")
    builder_mod = _ilu.module_from_spec(spec)
    # The builder module imports sibling modules; temporarily add builders_dir
    # to sys.path for the duration of the load, then remove it.
    _added = builders_dir not in sys.path
    if _added:
        sys.path.insert(0, builders_dir)
    try:
        spec.loader.exec_module(builder_mod)  # type: ignore[union-attr]
    finally:
        if _added and builders_dir in sys.path:
            sys.path.remove(builders_dir)
    create_model = builder_mod.create_model

    out_dir = tempfile.mkdtemp(prefix="ort_genai_cmp_")
    try:
        create_model(
            model_name=model_id,
            input_path=model_id,
            output_dir=out_dir,
            precision=precision,
            execution_provider=ep,
            cache_dir=os.path.join(out_dir, "cache"),
        )
        candidates = list(Path(out_dir).glob("**/*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"No .onnx found in ORT GenAI output: {out_dir}")
        model = ir.load(str(candidates[0]))
        counts = _count_ops(model)
        return OpCounts(
            label=f"ort-genai/{ep}",
            counts=counts,
            total_nodes=sum(counts.values()),
            source="ort-genai",
        )
    finally:
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def _all_ops_in_order(cols: list[OpCounts]) -> list[tuple[str, str]]:
    """Return (op_type, display_label) for every op that appears in any column.

    Catalogued ops from _OP_CATALOG appear first (in their existing priority
    order), followed by any uncatalogued ops sorted alphabetically.
    """
    all_op_types: set[str] = {op for c in cols for op in c.counts}
    catalogued_set = set(OP_TYPES)
    # Catalogued ops first (preserve _OP_CATALOG ordering)
    result = [(op, OP_LABEL[op]) for op, _, _ in _OP_CATALOG if op in all_op_types]
    # Uncatalogued ops alphabetically
    uncatalogued = sorted(all_op_types - catalogued_set)
    result += [(op, op) for op in uncatalogued]
    return result


def _ep_differences(cols: list[OpCounts]) -> list[str]:
    """Describe the expected per-EP differences for a multi-EP comparison."""
    diffs = []
    for op, lbl in _all_ops_in_order(cols):
        counts = [c.counts.get(op, 0) for c in cols]
        if len(set(counts)) == 1:
            continue
        expl = OP_EXPLANATION.get(op, "uncatalogued op — count shown for transparency")
        vals = ", ".join(f"{c.label}={c.counts.get(op, 0)}" for c in cols)
        diffs.append(f"**{lbl}** ({op}): {vals}  \n  ↳ {expl}")
    return diffs


def _compute_verdict(
    cols: list[OpCounts],
    model_id: str = "",
    extra_known_diff_ops: frozenset[str] | None = None,
) -> tuple[str, str, list[str], list[str]]:
    """Return (verdict, reason, differences_list, qk_norm_issues).

    extra_known_diff_ops: additional ops to treat as KNOWN_DIFF rather than FAIL,
        e.g. when an EP is known not to support GQA at the build dtype.

    Verdicts:
      PASS         — all critical ops match across all columns
      KNOWN_DIFF   — critical ops match; benign/explained ops differ
      FAIL         — critical ops differ unexpectedly
      MOBIUS-ONLY  — only one column, no cross-builder comparison
    """
    if len(cols) < 2:
        return "MOBIUS-ONLY", "Single builder — no cross-builder comparison.", [], []

    effective_critical = _CRITICAL_OPS - (extra_known_diff_ops or frozenset())
    differences: list[str] = []
    critical_fail = False

    # Iterate ALL op types that appear in any column — catalogued first (in
    # priority order), then uncatalogued alphabetically.  Uncatalogued ops are
    # always benign (never trigger FAIL) but are included in the diff list for
    # full transparency.
    for op, lbl in _all_ops_in_order(cols):
        counts_for_op = [c.counts.get(op, 0) for c in cols]
        if len(set(counts_for_op)) == 1:
            continue  # All same — no difference
        vals = ", ".join(f"{c.label}={c.counts.get(op, 0)}" for c in cols)
        expl = OP_EXPLANATION.get(op, "uncatalogued op — count shown for transparency")
        differences.append(f"**{lbl}** ({op}): {vals}  \n  ↳ {expl}")
        if op in effective_critical:
            critical_fail = True

    # Run QK-norm invariant checks for known QK-norm model families.
    # Build a temporary ModelReport-like object so _qk_norm_checks can inspect it.
    _tmp = ModelReport(model_id=model_id, ep="", columns=cols)
    qk_norm_issues = _qk_norm_checks(_tmp)
    if qk_norm_issues:
        critical_fail = True

    if critical_fail:
        verdict = "FAIL"
        reason = (
            "Critical op counts differ (GQA / BiasGelu / MoE count mismatch"
            + ("; QK-norm invariant violated" if qk_norm_issues else "")
            + ")."
        )
    elif differences:
        verdict = "KNOWN_DIFF"
        reason = (
            "Key fused ops (GQA/Attention) match. Known differences: "
            "RMSNormalization vs SimplifiedLayerNormalization (same math, different name), "
            "RotaryEmbedding absent when fused into GQA (do_rotary=1), "
            "MatMul count differs for packed-QKV models."
        )
    else:
        verdict = "PASS"
        reason = "All tracked op counts match exactly."

    return verdict, reason, differences, qk_norm_issues


# ---------------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------------

_VERDICT_ICON = {"PASS": "✅", "KNOWN_DIFF": "🟡", "FAIL": "❌", "MOBIUS-ONLY": "\u2139\ufe0f"}
_VERDICT_COLOR = {
    "PASS": "\033[92m",  # green
    "KNOWN_DIFF": "\033[93m",  # yellow
    "FAIL": "\033[91m",  # red
    "MOBIUS-ONLY": "\033[94m",  # blue
}
_RESET = "\033[0m"


def _console_table(report: ModelReport, color: bool) -> str:
    cols = report.columns
    ops_in_order = _all_ops_in_order(cols)

    op_w = max((len(lbl) for _, lbl in ops_in_order), default=24)
    col_w = max(18, max(len(c.label) for c in cols))
    row_w = op_w + 4 + (col_w + 4) * len(cols)

    sep = "─" * row_w
    header = f"  {'Op':<{op_w}}  " + "  ".join(f"{c.label:^{col_w}}" for c in cols)
    lines = [sep, header, sep]

    for op, lbl in ops_in_order:
        counts = [c.counts.get(op, 0) for c in cols]
        differs = len(set(counts)) > 1
        marker = " ◀" if differs else "  "
        row = f"  {lbl:<{op_w}}  " + "  ".join(f"{n:^{col_w}}" for n in counts)
        lines.append(row + marker)

    lines.append(sep)
    total = f"  {'Total nodes':<{op_w}}  " + "  ".join(
        f"{c.total_nodes:^{col_w}}" for c in cols
    )
    lines.append(total)
    lines.append(sep)

    # Per-EP verdict row for ORT-all-EPs mode.
    # Columns: first is ORT GenAI (no verdict), rest are per-EP mobius verdicts.
    if report.per_ep_verdicts:
        use_color = color and sys.stdout.isatty()
        # Build per-column verdict cells (ORT GenAI column = blank)
        ort_cols = [c for c in cols if c.source == "ort-genai"]
        verdict_cells: list[str] = []
        if ort_cols:
            verdict_cells.append(f"{'(reference)':^{col_w}}")
        mob_ep_idx = 0
        for c in cols:
            if c.source != "ort-genai":
                if mob_ep_idx < len(report.per_ep_verdicts):
                    _ep_name, v, _reason = report.per_ep_verdicts[mob_ep_idx]
                    icon = _VERDICT_ICON.get(v, "?")
                    vc = _VERDICT_COLOR.get(v, "") if use_color else ""
                    rc = _RESET if use_color else ""
                    cell = f"{vc}{icon} {v}{rc}"
                    # Pad to col_w (icons are multi-byte; use label width approximation)
                    verdict_cells.append(f"{cell:^{col_w}}")
                    mob_ep_idx += 1
                else:
                    verdict_cells.append(f"{'':^{col_w}}")
        verdict_row = f"  {'Verdict':<{op_w}}  " + "  ".join(verdict_cells)
        lines += [verdict_row, sep]

    return "\n".join(lines)


def render_console(report: ComparisonReport, color: bool = True) -> str:
    use_color = color and sys.stdout.isatty()
    lines: list[str] = []

    # Header
    lines += [
        "",
        "═" * 72,
        "  ONNX MODEL BUILDER PARITY REPORT",
        "═" * 72,
        f"  Generated:     {report.generated_at}",
        f"  Platform:      {report.platform_info}",
        f"  Python:        {report.python_version}",
        f"  mobius:        {report.mobius_version}  ({report.git_sha})",
        f"  ORT GenAI:     {report.ort_genai_version}",
        f"  Models:        {len(report.models)}",
        "═" * 72,
        "",
    ]

    for mr in report.models:
        icon = _VERDICT_ICON.get(mr.verdict, "?")
        vc = _VERDICT_COLOR.get(mr.verdict, "") if use_color else ""
        rc = _RESET if use_color else ""
        lines += [
            f"┌─ Model: {mr.model_id}  EP: {mr.ep}",
            f"│  Verdict: {vc}{icon} {mr.verdict}{rc}  —  {mr.verdict_reason}",
            "│",
        ]
        for line in _console_table(mr, use_color).splitlines():
            lines.append("│  " + line)
        lines.append("│")
        if mr.expected_counts:
            lines.append("│  Expected counts (from HF config):")
            for op, expected_n in mr.expected_counts.items():
                lbl = OP_LABEL.get(op, op)
                mob_cols_ec = [c for c in mr.columns if c.source == "mobius"]
                actual_col = (
                    mob_cols_ec[0] if mob_cols_ec else mr.columns[0] if mr.columns else None
                )
                actual_n = actual_col.counts.get(op, 0) if actual_col else 0
                # Allow ±1 for norm ops: mobius keeps final/embedding norm as
                # ONNX RMSNormalization rather than fusing into SkipSimplifiedLayerNorm.
                norm_ops = {"SkipSimplifiedLayerNormalization", "SkipLayerNormalization"}
                tolerance = 1 if op in norm_ops else 0
                ok = "✓" if abs(actual_n - expected_n) <= tolerance else "✗"
                note = (
                    " (±1 ok: final norm kept as ONNX RMSNorm)"
                    if ok == "✓" and actual_n != expected_n
                    else ""
                )
                lines.append(
                    f"│    {ok} {lbl}: Expected={expected_n}  Actual={actual_n}{note}"
                )
            lines.append("│")
        if mr.differences:
            lines.append("│  Differences:")
            for diff in mr.differences:
                # Strip markdown for console
                plain = diff.replace("**", "").replace("  \n  ↳ ", "\n     ↳ ")
                for dline in plain.splitlines():
                    lines.append("│    " + dline)
            lines.append("│")
        if _is_qk_norm_model(mr.model_id):
            if mr.qk_norm_issues:
                lines.append("│  ⚠️  QK-norm invariant VIOLATED:")
                for issue in mr.qk_norm_issues:
                    lines.append("│    ✗ " + issue)
            else:
                gqa_vals = " | ".join(
                    f"{c.label}={c.counts.get('GroupQueryAttention', 0)}" for c in mr.columns
                )
                lines.append(
                    f"│  ✓ QK-norm model: GQA match [{gqa_vals}], PackQKV absent (separate QKV MatMuls)"
                )
            lines.append("│")
        lines += [
            "└" + "─" * 70,
            "",
        ]

    # Summary
    total = len(report.models)
    lines += [
        "═" * 72,
        "  SUMMARY",
        "═" * 72,
    ]
    if any(m.verdict != "MOBIUS-ONLY" for m in report.models):
        vc_p = _VERDICT_COLOR["PASS"] if use_color else ""
        vc_kd = _VERDICT_COLOR["KNOWN_DIFF"] if use_color else ""
        vc_f = _VERDICT_COLOR["FAIL"] if use_color else ""
        rc = _RESET if use_color else ""
        lines += [
            f"  {vc_p}✅ PASS{rc}:       {report.pass_count}/{total}",
            f"  {vc_kd}🟡 KNOWN_DIFF{rc}: {report.known_diff_count}/{total}",
            f"  {vc_f}❌ FAIL{rc}:       {report.fail_count}/{total}",
        ]
    else:
        lines.append(f"  {total} mobius-only comparison(s) — no cross-builder parity check.")
    lines += ["═" * 72, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(report: ComparisonReport) -> str:
    lines: list[str] = []
    pass_count = report.pass_count
    known_diff_count = report.known_diff_count
    fail_count = report.fail_count
    lines += [
        "---",
        'title: "ONNX Model Builder Parity Report"',
        f'date: "{report.generated_at}"',
        f'mobius: "{report.mobius_version} ({report.git_sha})"',
        f'ort_genai: "{report.ort_genai_version}"',
        f'platform: "{report.platform_info}"',
        f"models_compared: {len(report.models)}",
        f"verdict_pass: {pass_count}",
        f"verdict_known_diff: {known_diff_count}",
        f"verdict_fail: {fail_count}",
        "---",
        "",
        "# ONNX Model Builder Parity Report",
        "",
        "| Attribute | Value |",
        "|-----------|-------|",
        f"| Generated | {report.generated_at} |",
        f"| Platform | {report.platform_info} |",
        f"| Python | {report.python_version} |",
        f"| mobius | `{report.mobius_version}` ({report.git_sha}) |",
        f"| ORT GenAI | `{report.ort_genai_version}` |",
        f"| Models compared | {len(report.models)} |",
        "",
        "---",
        "",
    ]

    for mr in report.models:
        icon = _VERDICT_ICON.get(mr.verdict, "?")
        lines += [
            f"## {mr.model_id}  ·  EP: `{mr.ep}`",
            "",
            f"**Verdict: {icon} {mr.verdict}**",
            "",
            f"> {mr.verdict_reason}",
            "",
        ]

        # Op count table — all ops in union of both models
        cols = mr.columns
        ops_in_order = _all_ops_in_order(cols)

        header = "| Op |" + "".join(f" {c.label} |" for c in cols) + " Notes |"
        sep_row = "|" + "---|" * (2 + len(cols))
        lines += [header, sep_row]

        for op, lbl in ops_in_order:
            counts = [c.counts.get(op, 0) for c in cols]
            differs = len(set(counts)) > 1
            expl = OP_EXPLANATION.get(op, "")
            if differs:
                note = f"⚠️ differs — {expl}" if expl else "⚠️ differs"
            else:
                note = expl if expl else ""
            count_cells = "".join(f" {n} |" for n in counts)
            lines.append(f"| `{lbl}` |{count_cells} {note} |")

        # Total row
        total_cells = "".join(f" **{c.total_nodes}** |" for c in cols)
        lines.append(f"| **Total nodes** |{total_cells}  |")

        # Per-EP verdict row for ORT-all-EPs mode
        if mr.per_ep_verdicts:
            ort_cols_md = [c for c in cols if c.source == "ort-genai"]
            verdict_cells_md = ["*(reference)*"] if ort_cols_md else []
            for _, v, _r in mr.per_ep_verdicts:
                icon = _VERDICT_ICON.get(v, "?")
                verdict_cells_md.append(f"{icon} **{v}**")
            # Pad to match non-ep columns
            while len(verdict_cells_md) < len(cols):
                verdict_cells_md.append("")
            vc_row = "".join(f" {cell} |" for cell in verdict_cells_md)
            lines.append(f"| **Verdict** |{vc_row}  |")

        lines.append("")

        if mr.differences:
            lines += ["### Differences", ""]
            for diff in mr.differences:
                lines.append(f"- {diff}")
            lines += [""]

        if mr.expected_counts:
            lines += ["### Expected Counts (from HF config)", ""]
            lines += [
                "| Op | Expected | Actual | Status |",
                "|-----|----------|--------|--------|",
            ]
            mob_cols_ec = [c for c in mr.columns if c.source == "mobius"]
            actual_col_ec = (
                mob_cols_ec[0] if mob_cols_ec else mr.columns[0] if mr.columns else None
            )
            norm_ops = {"SkipSimplifiedLayerNormalization", "SkipLayerNormalization"}
            for op, expected_n in mr.expected_counts.items():
                lbl = OP_LABEL.get(op, op)
                actual_n = actual_col_ec.counts.get(op, 0) if actual_col_ec else 0
                tolerance = 1 if op in norm_ops else 0
                ok = "✓" if abs(actual_n - expected_n) <= tolerance else "✗"
                note = (
                    " *(±1: final norm as ONNX RMSNorm)*"
                    if ok == "✓" and actual_n != expected_n
                    else ""
                )
                lines.append(f"| `{lbl}` | {expected_n} | {actual_n} | {ok}{note} |")
            lines += [""]

        # QK-norm invariant section
        if _is_qk_norm_model(mr.model_id):
            lines += ["### QK-Norm Invariant Check", ""]
            if mr.qk_norm_issues:
                lines.append("**⚠️ QK-norm invariant VIOLATED:**")
                for issue in mr.qk_norm_issues:
                    lines.append(f"- ✗ {issue}")
            else:
                gqa_vals = ", ".join(
                    f"`{c.label}`={c.counts.get('GroupQueryAttention', 0)}" for c in mr.columns
                )
                lines.append(
                    f"**✓ Invariants satisfied** for this QK-norm model family ({mr.model_id}):"
                )
                lines.append(f"- GQA count matches across builders: {gqa_vals}")
                lines.append(
                    "- PackQKV absent: both builders use separate Q/K/V MatMul projections "
                    "(higher MatMul count vs packed-QKV models is expected)."
                )
            lines += [""]

        lines += ["---", ""]

    # Summary
    lines += [
        "## Summary",
        "",
        "| Verdict | Count |",
        "|---------|-------|",
        f"| ✅ PASS | {report.pass_count} |",
        f"| 🟡 KNOWN_DIFF | {report.known_diff_count} |",
        f"| ❌ FAIL | {report.fail_count} |",
        f"| \u2139\ufe0f MOBIUS-ONLY | {sum(1 for m in report.models if m.verdict == 'MOBIUS-ONLY')} |",
        "",
    ]

    lines += [
        "---",
        "",
        "### Op Glossary",
        "",
        "| Op | Meaning |",
        "|----|---------|",
    ]
    for _op, label, expl in _OP_CATALOG:
        safe = expl.replace("|", "\\|")
        lines.append(f"| `{label}` | {safe} |")

    lines += [""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


def _expected_counts(model_id: str) -> dict[str, int] | None:
    """Load HuggingFace config and compute expected op counts for validation.

    Returns a dict of {op_type: expected_count}. Only includes the norm op
    type that the model actually uses (RMSNorm models → SkipSimplifiedLayerNorm,
    LayerNorm models → SkipLayerNorm).
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        text_cfg = getattr(cfg, "text_config", cfg)
        num_layers = getattr(text_cfg, "num_hidden_layers", None)
        if num_layers is None:
            return None

        expected: dict[str, int] = {"GroupQueryAttention": num_layers}

        # Detect norm type from config: models with rms_norm_eps use
        # SkipSimplifiedLayerNorm; models with layer_norm_eps use SkipLayerNorm.
        rms = getattr(text_cfg, "rms_norm_eps", None)
        layer_norm = getattr(text_cfg, "layer_norm_eps", None)
        norm_type = getattr(text_cfg, "norm_type", "")

        if rms is not None or "rms" in norm_type.lower():
            # RMSNorm model (Llama, Qwen, Gemma, Mistral, etc.)
            # 2 skip norms per layer; final + embedding norms are ONNX RMSNormalization
            expected["SkipSimplifiedLayerNormalization"] = num_layers * 2
        elif layer_norm is not None:
            # LayerNorm model (BERT, GPT-2, Falcon, etc.)
            expected["SkipLayerNormalization"] = num_layers * 2
    except Exception:
        return None
    else:
        return expected
    return None


def _mobius_version() -> str:
    try:
        return importlib.metadata.version("mobius-ai")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        return importlib.metadata.version("mobius")
    except importlib.metadata.PackageNotFoundError:
        pass
    return "dev"


def _ort_genai_version(repo: str | None) -> str:
    if repo and os.path.isdir(repo):
        try:
            sha = subprocess.check_output(
                ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            pass
        else:
            return f"git@{sha}"
    try:
        return importlib.metadata.version("onnxruntime-genai")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _mobius_git_sha() -> str:
    try:
        import importlib.resources

        mobius_src = Path(importlib.resources.files("mobius").__str__()).parent
        return subprocess.check_output(
            ["git", "-C", str(mobius_src), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    from mobius._execution_providers import ep_registry

    all_eps = sorted(ep_registry.names())

    parser = argparse.ArgumentParser(
        description="Presentation-quality parity report: ORT GenAI builder vs mobius.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        dest="models_csv",
        metavar="MODEL_ID[,MODEL_ID,...]",
        help="Comma-separated HuggingFace model IDs to compare.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="model_list",
        metavar="MODEL_ID",
        help="HuggingFace model ID (repeat for multiple). Kept for backward compat.",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run the standard suite of models (STANDARD_SUITE).",
    )
    parser.add_argument(
        "--ep",
        default=None,
        metavar="EP",
        help=(
            f"Target EP (default: unset). When --ort-genai-repo is provided and --ep is "
            f"not set, all registered EPs are compared side-by-side. "
            f"Available: {', '.join(all_eps)}."
        ),
    )
    parser.add_argument(
        "--ep-list",
        default=None,
        metavar="EP1,EP2,...",
        help="Compare mobius output across multiple EPs (e.g. default,cuda,dml,webgpu). "
        "Overrides --ep and implies --no-ort.",
    )
    parser.add_argument(
        "--no-ort",
        action="store_true",
        help="Skip ORT GenAI comparison — only compare mobius across EPs.",
    )
    parser.add_argument(
        "--ort-model",
        action="append",
        dest="ort_models",
        metavar="DIR_OR_FILE",
        help="Pre-built ORT GenAI model dir or .onnx file (repeat to match --model list).",
    )
    parser.add_argument(
        "--ort-genai-repo",
        default=os.environ.get("ORT_GENAI_REPO"),
        metavar="REPO_PATH",
        help="Path to onnxruntime-genai checkout for in-process building. "
        "Defaults to $ORT_GENAI_REPO env var.",
    )
    parser.add_argument(
        "--ort-precision",
        default="fp16",
        choices=["fp32", "fp16", "bf16", "int4"],
        help="Precision for ORT GenAI builder (default: fp16).",
    )
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Download and apply weights when building with mobius (slower).",
    )
    parser.add_argument(
        "--format",
        choices=["console", "markdown"],
        default="console",
        help="Output format (default: console).",
    )
    parser.add_argument(
        "--output",
        dest="output_file",
        default=None,
        metavar="FILE",
        help="Save output to file.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-model build progress output.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color in console output.",
    )
    args = parser.parse_args()

    if args.suite:
        models = STANDARD_SUITE
    elif args.models_csv:
        models = [m.strip() for m in args.models_csv.split(",")]
    elif args.model_list:
        models = args.model_list
    else:
        models = [DEFAULT_MODEL]

    # Determine operating mode:
    #   ort_all_eps_mode — --ort-genai-repo set, --ep NOT explicitly given:
    #       Build ORT GenAI once (cuda) + all mobius EPs → one wide table per model.
    #   ep_list_mode — --ep-list given: mobius-only multi-EP (no ORT GenAI).
    #   single_ep_mode — explicit --ep or --no-ort: one (model, ep) pair.
    has_ort = bool(args.ort_genai_repo and os.path.isdir(args.ort_genai_repo)) or bool(
        args.ort_models
    )
    ort_all_eps_mode = has_ort and args.ep is None and not args.ep_list and not args.no_ort
    no_ort = args.no_ort or bool(args.ep_list)

    if args.ep_list:
        eps: list[str] = args.ep_list.split(",")
    elif ort_all_eps_mode:
        eps = all_eps  # all registered EPs for mobius columns
    else:
        eps = [args.ep or "cuda"]  # default to cuda for single-EP mode

    report = ComparisonReport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        mobius_version=_mobius_version(),
        ort_genai_version=_ort_genai_version(args.ort_genai_repo if not no_ort else None),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        git_sha=_mobius_git_sha(),
        platform_info=_platform.platform(terse=True),
    )

    for i, model_id in enumerate(models):
        if ort_all_eps_mode:
            # ORT-all-EPs mode: build ORT GenAI once (cuda), mobius for every EP.
            if not args.quiet:
                print(
                    f"\n→ {model_id}  (ORT GenAI + all EPs: {', '.join(all_eps)})", flush=True
                )

            all_columns: list[OpCounts] = []

            # ORT GenAI column (built with cuda — serves as reference)
            ort_ep = "cuda"
            ort_models_list = args.ort_models or []
            ort_path = ort_models_list[i] if i < len(ort_models_list) else None
            if ort_path:
                p = Path(ort_path)
                found = (
                    list(p.glob("**/*.onnx")) if p.is_dir() else ([p] if p.is_file() else [])
                )
                if found:
                    if not args.quiet:
                        print(f"  Loading ORT GenAI model: {found[0]}", flush=True)
                    try:
                        all_columns.append(_op_counts_from_file(found[0]))
                    except Exception as e:
                        print(f"  WARNING: {e}")
            else:
                if not args.quiet:
                    print(
                        f"  Building with ORT GenAI ({args.ort_precision}/{ort_ep}) ...",
                        end="",
                        flush=True,
                    )
                try:
                    all_columns.append(
                        build_ort_genai(
                            model_id,
                            ep=ort_ep,
                            precision=args.ort_precision,
                            ort_genai_repo=args.ort_genai_repo,
                        )
                    )
                    if not args.quiet:
                        print(" done.")
                except Exception as e:
                    print(f"\n  WARNING: ORT GenAI build failed: {e}")

            # Mobius columns — one per EP
            for ep_target in all_eps:
                if not args.quiet:
                    print(f"  Building with mobius/{ep_target} ...", end="", flush=True)
                try:
                    all_columns.append(
                        build_mobius(model_id, ep=ep_target, load_weights=args.load_weights)
                    )
                    if not args.quiet:
                        print(" done.")
                except Exception as e:
                    print(f"\n  ERROR building mobius/{ep_target}: {e}")

            if not all_columns:
                continue

            # Compute per-EP verdict: each mobius column vs ORT GenAI column.
            # GQA is downgraded to KNOWN_DIFF for EPs where the model's dtype
            # is not in the EP's gqa_dtypes (e.g. cpu doesn't support bf16 GQA,
            # dml/webgpu don't support bf16 GQA, etc.)
            from mobius._execution_providers import ep_registry as _ep_reg

            model_dtype = _detect_model_dtype(model_id)

            ort_ref = [c for c in all_columns if c.source == "ort-genai"]
            per_ep_verdicts: list[tuple[str, str, str]] = []
            all_diffs: list[str] = []
            all_qk_issues: list[str] = []
            overall_verdict = "PASS"
            for mob_col in [c for c in all_columns if c.source == "mobius"]:
                ep_name = mob_col.label.removeprefix("mobius/")
                ep_caps = _ep_reg.get(ep_name)
                # GQA mismatch vs ORT GenAI/cuda is KNOWN_DIFF when the
                # model's dtype is not in this EP's gqa_dtypes — the EP
                # simply doesn't support GQA at that precision.
                gqa_known_diff: frozenset[str] = frozenset()
                if ep_caps is None or model_dtype not in ep_caps.gqa_dtypes:
                    gqa_known_diff = frozenset({"GroupQueryAttention"})
                pair = [*ort_ref, mob_col] if ort_ref else [mob_col]
                ep_v, ep_r, ep_diffs, ep_qk = _compute_verdict(
                    pair, model_id=model_id, extra_known_diff_ops=gqa_known_diff
                )
                per_ep_verdicts.append((ep_name, ep_v, ep_r))
                all_diffs.extend(d for d in ep_diffs if d not in all_diffs)
                all_qk_issues.extend(q for q in ep_qk if q not in all_qk_issues)
                if ep_v == "FAIL":
                    overall_verdict = "FAIL"
                elif ep_v == "KNOWN_DIFF" and overall_verdict == "PASS":
                    overall_verdict = "KNOWN_DIFF"

            # Overall report verdict = worst of per-EP verdicts
            if not ort_ref:
                overall_verdict = "MOBIUS-ONLY"
                overall_reason = "No ORT GenAI column — mobius-only multi-EP comparison."
            else:
                verdict_counts = {v for _, v, _ in per_ep_verdicts}
                if "FAIL" in verdict_counts:
                    overall_reason = "One or more EPs have FAIL verdict."
                elif "KNOWN_DIFF" in verdict_counts:
                    overall_reason = "All EPs have KNOWN_DIFF or better."
                else:
                    overall_reason = "All EPs PASS."

            report.models.append(
                ModelReport(
                    model_id=model_id,
                    ep="all",
                    columns=all_columns,
                    verdict=overall_verdict,
                    verdict_reason=overall_reason,
                    differences=all_diffs,
                    qk_norm_issues=all_qk_issues,
                    expected_counts=_expected_counts(model_id),
                    per_ep_verdicts=per_ep_verdicts,
                )
            )

        elif args.ep_list:
            # Multi-EP mode: one ModelReport per model, one column per EP.
            if not args.quiet:
                print(f"\n→ {model_id}  (multi-EP: {', '.join(eps)})", flush=True)
            columns: list[OpCounts] = []
            for ep_target in eps:
                if not args.quiet:
                    print(f"  Building with mobius/{ep_target} ...", end="", flush=True)
                try:
                    columns.append(
                        build_mobius(model_id, ep=ep_target, load_weights=args.load_weights)
                    )
                    if not args.quiet:
                        print(" done.")
                except Exception as e:
                    print(f"\n  ERROR: {e}")
                    import traceback

                    traceback.print_exc()

            if columns:
                # Multi-EP mode: verdict is MOBIUS-ONLY (no cross-builder comparison)
                report.models.append(
                    ModelReport(
                        model_id=model_id,
                        ep="multi",
                        columns=columns,
                        verdict="MOBIUS-ONLY",
                        verdict_reason=(
                            "Mobius-only multi-EP comparison. Differences shown are "
                            "by design — each EP applies different fusions and lowerings."
                        ),
                        differences=_ep_differences(columns),
                        expected_counts=_expected_counts(model_id),
                    )
                )
        else:
            # Cross-builder mode: one ModelReport per (model, ep).
            for ep in eps:
                if not args.quiet:
                    print(f"\n→ {model_id}  EP: {ep}", flush=True)
                columns_: list[OpCounts] = []

                # --- ORT GenAI column ---
                if not no_ort:
                    ort_models = args.ort_models or []
                    ort_path = ort_models[i] if i < len(ort_models) else None

                    if ort_path:
                        p = Path(ort_path)
                        if p.is_dir():
                            found = list(p.glob("**/*.onnx"))
                            if not found:
                                print(f"  WARNING: No .onnx in {p}, skipping ORT GenAI.")
                            else:
                                print(f"  Loading ORT GenAI model: {found[0]}")
                                try:
                                    columns_.append(_op_counts_from_file(found[0]))
                                except Exception as e:
                                    print(f"  WARNING: {e}")
                        elif p.is_file():
                            print(f"  Loading ORT GenAI model: {p}")
                            try:
                                columns_.append(_op_counts_from_file(p))
                            except Exception as e:
                                print(f"  WARNING: {e}")
                    elif args.ort_genai_repo and os.path.isdir(args.ort_genai_repo):
                        if not args.quiet:
                            print(
                                f"  Building with ORT GenAI ({args.ort_precision}/{ep}) ...",
                                end="",
                                flush=True,
                            )
                        try:
                            columns_.append(
                                build_ort_genai(
                                    model_id,
                                    ep=ep,
                                    precision=args.ort_precision,
                                    ort_genai_repo=args.ort_genai_repo,
                                )
                            )
                            if not args.quiet:
                                print(" done.")
                        except Exception as e:
                            print(f"\n  WARNING: ORT GenAI build failed: {e}")
                            print("  Use --ort-model or --no-ort to proceed without it.")
                    else:
                        repo_note = args.ort_genai_repo or "(not set)"
                        print(
                            f"  NOTE: ORT GenAI repo not found ({repo_note}). "
                            "Use --ort-genai-repo or set $ORT_GENAI_REPO. "
                            "Skipping ORT GenAI column."
                        )

                # --- mobius column ---
                if not args.quiet:
                    print(f"  Building with mobius/{ep} ...", end="", flush=True)
                try:
                    columns_.append(
                        build_mobius(model_id, ep=ep, load_weights=args.load_weights)
                    )
                    if not args.quiet:
                        print(" done.")
                except Exception as e:
                    print(f"\n  ERROR: {e}")
                    import traceback

                    traceback.print_exc()

                if not columns_:
                    continue

                verdict, reason, diffs, qk_issues = _compute_verdict(
                    columns_, model_id=model_id
                )
                report.models.append(
                    ModelReport(
                        model_id=model_id,
                        ep=ep,
                        columns=columns_,
                        verdict=verdict,
                        verdict_reason=reason,
                        differences=diffs,
                        qk_norm_issues=qk_issues,
                        expected_counts=_expected_counts(model_id),
                    )
                )

    if not report.models:
        print("No models built successfully.")
        sys.exit(1)

    output_format = getattr(args, "format", "console")
    output_file = getattr(args, "output_file", None)

    if output_format == "markdown":
        md = render_markdown(report)
        if output_file:
            with open(output_file, "w") as f:
                f.write(md)
            print(f"  Markdown report saved to: {output_file}\n")
        else:
            print(md)
    else:
        # Console output (default)
        print(render_console(report, color=not args.no_color))
        if output_file:
            md = render_markdown(report)
            with open(output_file, "w") as f:
                f.write(md)
            print(f"  Markdown report saved to: {output_file}\n")
        else:
            print("  Tip: use --output report.md to save a markdown report.\n")


if __name__ == "__main__":
    main()
