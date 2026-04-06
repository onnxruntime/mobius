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

Compare ALL EPs side-by-side (mobius-only, no ORT GenAI required):

    python examples/model_builder_comparison.py \\
        --model meta-llama/Llama-3.2-1B \\
        --no-ort

  Shows: Op | default | cuda | dml | webgpu | trt-rtx | ...

Compare against ORT GenAI builder — ALL EPs by default (requires 'pip install onnxruntime-genai'):

    python examples/model_builder_comparison.py \\
        --model meta-llama/Llama-3.2-1B

  Produces one table per EP, each showing ORT GenAI vs mobius side-by-side:
    EP: cuda (fp16)  → ORT GenAI/cuda | mobius/cuda
    EP: cpu  (fp32)  → ORT GenAI/cpu  | mobius/cpu
    EP: dml  (fp16)  → ORT GenAI/dml  | mobius/dml
    ...
  "default" EP (mobius-only, no ORT GenAI equivalent) is shown separately.

Compare a single specific EP against ORT GenAI:

    python examples/model_builder_comparison.py \\
        --model meta-llama/Llama-3.2-1B \\
        --ep cuda

Compare against a pre-built ORT GenAI model dir:

    python examples/model_builder_comparison.py \\
        --model meta-llama/Llama-3.2-1B \\
        --ep cuda \\
        --ort-model /path/to/ort-output/

Compare multiple models and save the report (auto-saved by default):

    python examples/model_builder_comparison.py \\
        --model meta-llama/Llama-3.2-1B \\
        --model Qwen/Qwen3-0.6B \\
        --ep cuda \\
        --output custom_report.md   # override default parity_report_YYYYMMDD_HHMM.md
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


# ---------------------------------------------------------------------------
# ORT GenAI EP / precision mapping
# ---------------------------------------------------------------------------

# Mobius EP name → ORT GenAI execution_provider argument.
# "default" has no ORT GenAI equivalent (it's mobius-specific portable ONNX).
_ORT_EP_MAP: dict[str, str] = {
    "cpu": "cpu",
    "cuda": "cuda",
    "dml": "dml",
    "webgpu": "webgpu",
    "trt-rtx": "NvTensorRtRtx",
}

# Default ORT GenAI precision per EP when --dtype is not explicitly set.
# cpu/webgpu default to fp32; GPU EPs default to fp16.
_ORT_DEFAULT_PRECISION: dict[str, str] = {
    "cpu": "fp32",
    "cuda": "fp16",
    "dml": "fp16",
    "webgpu": "fp32",
    "NvTensorRtRtx": "fp16",
}


def _download_model_once(model_id: str, cache_root: str) -> str:
    """Download a HuggingFace model to a local cache dir and return the local path.

    Reuses the cached snapshot on subsequent calls — no re-download per EP.
    Falls back to returning model_id as-is if huggingface_hub is not available.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_id  # Let ORT GenAI builder handle download itself
    local_dir = os.path.join(cache_root, model_id.replace("/", "--"))
    if os.path.isdir(local_dir) and any(Path(local_dir).iterdir()):
        return local_dir  # Already cached
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
    return local_dir


def build_ort_genai(
    model_id: str,
    ep: str,
    precision: str,
    *,
    input_path: str | None = None,
    mobius_ep_label: str | None = None,
) -> OpCounts:
    """Build an ORT GenAI model using the installed onnxruntime_genai package.

    Args:
        model_id: HuggingFace model ID.
        ep: ORT GenAI execution_provider argument (e.g. "cuda", "NvTensorRtRtx").
        precision: ORT GenAI precision argument (e.g. "fp16", "fp32").
        input_path: Local directory containing the downloaded model weights.
            When provided, ORT GenAI uses this instead of re-downloading from HF.
            Pass the same path for every EP build of the same model to avoid
            repeated downloads.
        mobius_ep_label: Mobius EP name to use in the column label (e.g. "trt-rtx").
            Defaults to ep when not given.

    Requires: pip install onnxruntime-genai
    """
    try:
        from onnxruntime_genai.models.builder import create_model
    except ImportError as exc:
        raise ImportError(
            "ORT GenAI comparison requires onnxruntime-genai to be installed.\n"
            "  pip install onnxruntime-genai"
        ) from exc

    label_ep = mobius_ep_label or ep
    # Use a per-(ep, precision) temp dir for ONNX output; the model weights in
    # input_path are shared and never deleted by this function.
    out_dir = tempfile.mkdtemp(prefix=f"ort_genai_{label_ep}_")
    try:
        create_model(
            model_name=model_id,
            input_path=input_path or model_id,
            output_dir=out_dir,
            precision=precision,
            execution_provider=ep,
            # cache_dir is only used when input_path is not a local dir;
            # point it inside out_dir so it's cleaned up with the ONNX output.
            cache_dir=os.path.join(out_dir, "cache"),
        )
        candidates = list(Path(out_dir).glob("**/*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"No .onnx found in ORT GenAI output: {out_dir}")
        model = ir.load(str(candidates[0]))
        counts = _count_ops(model)
        return OpCounts(
            label=f"ort-genai/{label_ep}",
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
    from mobius._execution_providers import ep_registry

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
            lines.append("│  Expected counts (from HF config, per EP):")
            norm_ops = {"SkipSimplifiedLayerNormalization", "SkipLayerNormalization"}
            mob_cols = [c for c in mr.columns if c.source == "mobius"]
            for op, base_expected in mr.expected_counts.items():
                lbl = OP_LABEL.get(op, op)
                per_ep_parts = []
                for col in mob_cols:
                    ep_name = col.label.removeprefix("mobius/")
                    ep_caps = ep_registry.get(ep_name)
                    # GQA expected = num_layers only if EP supports GQA at the model dtype
                    if op == "GroupQueryAttention":
                        model_dtype = _detect_model_dtype(mr.model_id)
                        ep_expected = (
                            base_expected
                            if ep_caps is not None and model_dtype in ep_caps.gqa_dtypes
                            else 0
                        )
                    else:
                        ep_expected = base_expected
                    actual_n = col.counts.get(op, 0)
                    tolerance = 1 if op in norm_ops else 0
                    ok = "✓" if abs(actual_n - ep_expected) <= tolerance else "✗"
                    note = "\u00b11" if ok == "\u2713" and actual_n != ep_expected else ""
                    per_ep_parts.append(
                        f"{ok} {ep_name}: exp={ep_expected} act={actual_n}"
                        + (f" ({note})" if note else "")
                    )
                lines.append(f"│    {lbl}: " + ",  ".join(per_ep_parts))
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
    from mobius._execution_providers import ep_registry

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
            lines += ["### Expected Counts (from HF config, per EP)", ""]
            mob_cols_ec = [c for c in mr.columns if c.source == "mobius"]
            # Build column headers: one column per mobius EP
            ep_names_ec = [c.label.removeprefix("mobius/") for c in mob_cols_ec]
            hdr = "| Op |" + "".join(f" {ep} |" for ep in ep_names_ec)
            sep = "|-----|" + "---|" * len(mob_cols_ec)
            lines += [hdr, sep]
            norm_ops = {"SkipSimplifiedLayerNormalization", "SkipLayerNormalization"}
            for op, base_expected in mr.expected_counts.items():
                lbl = OP_LABEL.get(op, op)
                cells = []
                for col in mob_cols_ec:
                    ep_name = col.label.removeprefix("mobius/")
                    ep_caps = ep_registry.get(ep_name)
                    if op == "GroupQueryAttention":
                        model_dtype = _detect_model_dtype(mr.model_id)
                        ep_expected = (
                            base_expected
                            if ep_caps is not None and model_dtype in ep_caps.gqa_dtypes
                            else 0
                        )
                    else:
                        ep_expected = base_expected
                    actual_n = col.counts.get(op, 0)
                    tolerance = 1 if op in norm_ops else 0
                    ok = "✓" if abs(actual_n - ep_expected) <= tolerance else "✗"
                    note = " *(±1)*" if ok == "✓" and actual_n != ep_expected else ""
                    cells.append(f" {ok} exp={ep_expected} act={actual_n}{note} |")
                lines.append(f"| `{lbl}` |" + "".join(cells))
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


def _ort_genai_version() -> str:
    try:
        return importlib.metadata.version("onnxruntime-genai")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _ort_genai_available() -> bool:
    """Return True if onnxruntime_genai.models.builder can be imported."""
    try:
        import importlib as _il

        return _il.util.find_spec("onnxruntime_genai") is not None
    except Exception:
        return False


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
            "Target EP (default: unset). When --ep is not set, all registered EPs are "
            "compared side-by-side (with ORT GenAI if installed, mobius-only with --no-ort). "
            f"Available: {', '.join(all_eps)}."
        ),
    )
    parser.add_argument(
        "--ep-list",
        default=None,
        metavar="EP1,EP2,...",
        help="Compare mobius output across an explicit EP subset (e.g. default,cuda,dml). "
        "Overrides --ep and implies --no-ort.",
    )
    parser.add_argument(
        "--no-ort",
        action="store_true",
        help="Skip ORT GenAI comparison. Without --ep, shows all registered EPs side-by-side.",
    )
    parser.add_argument(
        "--ort-model",
        action="append",
        dest="ort_models",
        metavar="DIR_OR_FILE",
        help="Pre-built ORT GenAI model dir or .onnx file (repeat to match --model list).",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.expanduser("~"), ".cache", "mobius_comparison"),
        metavar="DIR",
        help="Directory for caching downloaded HuggingFace model weights. "
        "The same weights are reused for every EP build of the same model. "
        "Default: ~/.cache/mobius_comparison",
    )
    parser.add_argument(
        "--dtype",
        "--ort-precision",
        dest="ort_precision",
        default=None,
        choices=["fp32", "float32", "fp16", "float16", "bf16", "bfloat16", "int4"],
        help="Model dtype / ORT GenAI builder precision. "
        "Accepts both short (fp16, bf16, fp32) and long (float16, bfloat16, float32) forms. "
        "Default: auto-selected per EP (fp32 for cpu/webgpu, fp16 for cuda/dml/trt-rtx).",
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
        help="Override the auto-generated markdown report filename "
        "(default: parity_report_YYYYMMDD_HHMM.md).",
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

    # Normalize long-form dtype aliases to the short forms that ORT GenAI expects.
    _dtype_normalize = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}
    if args.ort_precision is not None:
        args.ort_precision = _dtype_normalize.get(args.ort_precision, args.ort_precision)

    if args.suite:
        models = STANDARD_SUITE
    elif args.models_csv:
        models = [m.strip() for m in args.models_csv.split(",")]
    elif args.model_list:
        models = args.model_list
    else:
        models = [DEFAULT_MODEL]

    # Determine operating mode:
    #   ort_all_eps_mode — ORT GenAI available/provided, --ep NOT given:
    #       Build ORT GenAI once (cuda) + all mobius EPs → one wide table.
    #   no_ort_all_eps_mode — --no-ort and --ep NOT given:
    #       mobius-only, but still show ALL EPs side-by-side.
    #   ep_list_mode — --ep-list given: explicit EP set (mobius-only).
    #   single_ep_mode — explicit --ep: one (model, ep) pair.
    has_ort = bool(args.ort_models) or _ort_genai_available()
    ort_all_eps_mode = has_ort and args.ep is None and not args.ep_list and not args.no_ort
    no_ort = args.no_ort or bool(args.ep_list)

    if args.ep_list:
        eps: list[str] = args.ep_list.split(",")
    elif ort_all_eps_mode:
        eps = all_eps  # all registered EPs for mobius columns
    elif args.no_ort and args.ep is None:
        # --no-ort without an explicit --ep: show all EPs side-by-side (mobius-only)
        eps = all_eps
    else:
        eps = [args.ep or "cuda"]  # default to cuda for single-EP mode

    report = ComparisonReport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        mobius_version=_mobius_version(),
        ort_genai_version=_ort_genai_version() if not no_ort else "n/a",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        git_sha=_mobius_git_sha(),
        platform_info=_platform.platform(terse=True),
    )

    for i, model_id in enumerate(models):
        if ort_all_eps_mode:
            # ORT-all-EPs mode: one ModelReport per EP, each with two columns:
            #   [ort-genai/{ep}  |  mobius/{ep}]
            # ORT GenAI is built for each EP it supports; "default" is mobius-only.
            if not args.quiet:
                print(
                    f"\n→ {model_id}  (ORT GenAI + all EPs: {', '.join(all_eps)})", flush=True
                )

            from mobius._execution_providers import ep_registry as _ep_reg

            model_dtype = _detect_model_dtype(model_id)
            ort_models_list = args.ort_models or []
            ort_path = ort_models_list[i] if i < len(ort_models_list) else None

            # Download model weights ONCE, reuse across all EP builds.
            # _download_model_once returns a local path; create_model skips download
            # when input_path is an existing local directory.
            if not ort_path and not no_ort:
                if not args.quiet:
                    print(
                        f"  Downloading {model_id} to cache (once for all EPs) ...",
                        end="",
                        flush=True,
                    )
                try:
                    cached_model_path: str | None = _download_model_once(
                        model_id, args.cache_dir
                    )
                    if not args.quiet:
                        print(" done.")
                except Exception as e:
                    print(f"\n  WARNING: model download failed: {e} — will retry per EP")
                    cached_model_path = None
            else:
                cached_model_path = None

            for ep_target in all_eps:
                ort_ep_name = _ORT_EP_MAP.get(ep_target)  # None for "default"
                precision = args.ort_precision or _ORT_DEFAULT_PRECISION.get(
                    ort_ep_name or "", "fp16"
                )
                ep_columns: list[OpCounts] = []

                # --- ORT GenAI column (skipped for "default" EP) ---
                if ort_ep_name and not no_ort:
                    if ort_path:
                        p = Path(ort_path)
                        found = (
                            list(p.glob("**/*.onnx"))
                            if p.is_dir()
                            else ([p] if p.is_file() else [])
                        )
                        if found:
                            if not args.quiet:
                                print(f"  Loading ORT GenAI model: {found[0]}", flush=True)
                            try:
                                ep_columns.append(_op_counts_from_file(found[0]))
                            except Exception as e:
                                print(f"  WARNING: {e}")
                    else:
                        if not args.quiet:
                            print(
                                f"  Building ort-genai/{ep_target} ({precision}) ...",
                                end="",
                                flush=True,
                            )
                        try:
                            ep_columns.append(
                                build_ort_genai(
                                    model_id,
                                    ep=ort_ep_name,
                                    precision=precision,
                                    input_path=cached_model_path,
                                    mobius_ep_label=ep_target,
                                )
                            )
                            if not args.quiet:
                                print(" done.")
                        except Exception as e:
                            print(f"\n  WARNING: ORT GenAI {ep_target} failed: {e}")

                # --- mobius column ---
                if not args.quiet:
                    print(f"  Building mobius/{ep_target} ...", end="", flush=True)
                try:
                    ep_columns.append(
                        build_mobius(model_id, ep=ep_target, load_weights=args.load_weights)
                    )
                    if not args.quiet:
                        print(" done.")
                except Exception as e:
                    print(f"\n  ERROR building mobius/{ep_target}: {e}")

                if not ep_columns:
                    continue

                # --- per-EP verdict ---
                ep_caps = _ep_reg.get(ep_target)
                gqa_known_diff: frozenset[str] = frozenset()
                if ep_caps is None or model_dtype not in ep_caps.gqa_dtypes:
                    gqa_known_diff = frozenset({"GroupQueryAttention"})
                ep_v, ep_r, ep_diffs, ep_qk = _compute_verdict(
                    ep_columns, model_id=model_id, extra_known_diff_ops=gqa_known_diff
                )

                report.models.append(
                    ModelReport(
                        model_id=model_id,
                        ep=ep_target,
                        columns=ep_columns,
                        verdict=ep_v,
                        verdict_reason=ep_r,
                        differences=ep_diffs,
                        qk_norm_issues=ep_qk,
                        expected_counts=_expected_counts(model_id),
                    )
                )

        elif args.ep_list or (args.no_ort and args.ep is None):
            # Multi-EP mobius-only mode: one ModelReport per model, one column per EP.
            # Triggered by --ep-list (explicit EP set) or --no-ort without --ep (all EPs).
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
            # Download model weights once for all EP iterations.
            import contextlib

            _single_ep_cached_path: str | None = None
            if not no_ort and _ort_genai_available() and not (args.ort_models or []):
                with contextlib.suppress(Exception):
                    _single_ep_cached_path = _download_model_once(model_id, args.cache_dir)

            for ep in eps:
                if not args.quiet:
                    print(f"\n→ {model_id}  EP: {ep}", flush=True)
                columns_: list[OpCounts] = []

                ort_ep_name = _ORT_EP_MAP.get(ep)  # None for "default"
                precision = args.ort_precision or _ORT_DEFAULT_PRECISION.get(
                    ort_ep_name or "", "fp16"
                )

                # --- ORT GenAI column ---
                if not no_ort and ort_ep_name:
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
                    elif _ort_genai_available():
                        if not args.quiet:
                            print(
                                f"  Building with ORT GenAI ({precision}/{ep}) ...",
                                end="",
                                flush=True,
                            )
                        try:
                            columns_.append(
                                build_ort_genai(
                                    model_id,
                                    ep=ort_ep_name,
                                    precision=precision,
                                    input_path=_single_ep_cached_path,
                                    mobius_ep_label=ep,
                                )
                            )
                            if not args.quiet:
                                print(" done.")
                        except Exception as e:
                            print(f"\n  WARNING: ORT GenAI build failed: {e}")
                            print("  Use --ort-model or --no-ort to proceed without it.")
                    else:
                        print(
                            "  NOTE: onnxruntime-genai not installed. "
                            "Run 'pip install onnxruntime-genai' to enable ORT GenAI comparison. "
                            "Skipping ORT GenAI column."
                        )
                elif not no_ort and ep == "default":
                    if not args.quiet:
                        print(
                            "  NOTE: ORT GenAI has no 'default' EP — "
                            "showing mobius/default only."
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

    # Auto-generate a filename when --output is not explicitly set.
    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = f"parity_report_{ts}.md"

    if output_format == "markdown":
        md = render_markdown(report)
        with open(output_file, "w") as f:
            f.write(md)
        print(f"  Markdown report saved to: {output_file}\n")
    else:
        # Console output (default) + always save markdown report
        print(render_console(report, color=not args.no_color))
        md = render_markdown(report)
        with open(output_file, "w") as f:
            f.write(md)
        print(f"  Markdown report saved to: {output_file}\n")


if __name__ == "__main__":
    main()
