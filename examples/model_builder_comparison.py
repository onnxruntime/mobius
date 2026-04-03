#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare ONNX model graph structure between ORT GenAI's model builder and mobius.

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

Compare against ORT GenAI builder (builds on-the-fly, downloads weights):

    python examples/model_builder_comparison.py \\
        --model meta-llama/Llama-3.2-1B \\
        --ep cuda \\
        --ort-genai-repo /home/justinchu/dev/onnxruntime-genai

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
        --save-report report.md
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import onnx_ir as ir

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
        "Standard ONNX opset-23 RMSNorm. "
        "Mobius emits this for norms not covered by SkipSimplifiedLayerNorm fusion "
        "(e.g. embedding norm, final pre-lm-head norm).",
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

# Ops whose mismatch is expected / known-benign
_KNOWN_BENIGN_OPS = {
    # Norm fusion: ORT GenAI packs the final pre-lm-head RMSNorm into a Skip
    # fusion; mobius emits it as a standard ONNX RMSNormalization. Total norm
    # count may differ by ±2. Differences are annotated in the diff section.
    "SkipLayerNormalization",
    "SkipSimplifiedLayerNormalization",
    "RMSNormalization",  # mobius keeps final norm as standard ONNX
    # GQA/Attention: counts are complementary — when GQA is emitted Attention=0.
    # The sum GQA+Attention should equal num_layers in both builders.
    "Attention",
    "RotaryEmbedding",  # absent when fused into GQA
    "MatMul",  # ORT GenAI packs Q/K/V into one MatMul
    "Cast",  # minor seqlen cast differences
    "Shape",  # webgpu-specific lowering
    "Gather",  # implementation detail
}


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
    verdict: str = ""  # "PASS" | "PARTIAL" | "FAIL" | "MOBIUS-ONLY"
    verdict_reason: str = ""
    differences: list[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    generated_at: str
    mobius_version: str
    ort_genai_version: str
    python_version: str
    models: list[ModelReport] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(1 for m in self.models if m.verdict == "PASS")

    @property
    def partial_count(self) -> int:
        return sum(1 for m in self.models if m.verdict == "PARTIAL")

    @property
    def fail_count(self) -> int:
        return sum(1 for m in self.models if m.verdict == "FAIL")


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _count_ops(model: ir.Model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in model.graph:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def _op_counts_from_file(path: str | Path) -> OpCounts:
    model = ir.load(str(path))
    counts = _count_ops(model)
    total = sum(counts.values())
    return OpCounts(label="ort-genai (file)", counts=counts, total_nodes=total, source="file")


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
    if builders_dir not in sys.path:
        sys.path.insert(0, builders_dir)
    from builder import create_model  # type: ignore[import]

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


def _ep_differences(cols: list[OpCounts]) -> list[str]:
    """Describe the expected per-EP differences for a multi-EP comparison."""
    diffs = []
    for op in OP_TYPES:
        counts = [c.counts.get(op, 0) for c in cols]
        if len(set(counts)) == 1:
            continue
        label = OP_LABEL.get(op, op)
        expl = OP_EXPLANATION.get(op, "")
        vals = ", ".join(f"{c.label}={c.counts.get(op, 0)}" for c in cols)
        diffs.append(f"**{label}** ({op}): {vals}  \n  ↳ {expl}")
    return diffs


def _compute_verdict(cols: list[OpCounts]) -> tuple[str, str, list[str]]:
    """Return (verdict, reason, differences_list).

    Verdicts:
      PASS    — all critical ops match across all columns, no unexplained diff
      PARTIAL — critical ops match but benign ops differ (known acceptable)
      FAIL    — critical ops differ across columns
      MOBIUS-ONLY — only one column, no comparison possible
    """
    if len(cols) < 2:
        return "MOBIUS-ONLY", "Single builder — no cross-builder comparison.", []

    differences: list[str] = []
    critical_fail = False

    for op in OP_TYPES:
        counts_for_op = [c.counts.get(op, 0) for c in cols]
        if len(set(counts_for_op)) == 1:
            continue  # All same — no difference
        vals = ", ".join(f"{c.label}={c.counts.get(op, 0)}" for c in cols)
        label = OP_LABEL.get(op, op)
        expl = OP_EXPLANATION.get(op, "")
        differences.append(f"**{label}** ({op}): {vals}  \n  ↳ {expl}")
        if op in _CRITICAL_OPS:
            critical_fail = True

    if critical_fail:
        verdict = "FAIL"
        reason = "Critical op counts differ (GQA / BiasGelu / MoE count mismatch)."
    elif differences:
        verdict = "PARTIAL"
        reason = (
            "Key fused ops (GQA/Attention) match. Differences in norm fusion "
            "strategy, MatMul packing, and Cast/Shape count are expected between builders."
        )
    else:
        verdict = "PASS"
        reason = "All tracked op counts match exactly."

    return verdict, reason, differences


# ---------------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------------

_VERDICT_ICON = {"PASS": "✅", "PARTIAL": "🟡", "FAIL": "❌", "MOBIUS-ONLY": "\u2139\ufe0f"}
_VERDICT_COLOR = {
    "PASS": "\033[92m",  # green
    "PARTIAL": "\033[93m",  # yellow
    "FAIL": "\033[91m",  # red
    "MOBIUS-ONLY": "\033[94m",  # blue
}
_RESET = "\033[0m"


def _console_table(report: ModelReport, color: bool) -> str:
    cols = report.columns
    present_ops = {op for c in cols for op in c.counts if op in set(OP_TYPES)}
    ops_in_order = [(op, OP_LABEL[op]) for op, _, _ in _OP_CATALOG if op in present_ops]

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
        f"  Python:        {report.python_version}",
        f"  mobius:        {report.mobius_version}",
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
        if mr.differences:
            lines.append("│  Differences:")
            for diff in mr.differences:
                # Strip markdown for console
                plain = diff.replace("**", "").replace("  \n  ↳ ", "\n     ↳ ")
                for dline in plain.splitlines():
                    lines.append("│    " + dline)
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
        vc_pa = _VERDICT_COLOR["PARTIAL"] if use_color else ""
        vc_f = _VERDICT_COLOR["FAIL"] if use_color else ""
        rc = _RESET if use_color else ""
        lines += [
            f"  {vc_p}✅ PASS{rc}:    {report.pass_count}/{total}",
            f"  {vc_pa}🟡 PARTIAL{rc}: {report.partial_count}/{total}",
            f"  {vc_f}❌ FAIL{rc}:    {report.fail_count}/{total}",
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
    lines += [
        "# ONNX Model Builder Parity Report",
        "",
        "| Attribute | Value |",
        "|-----------|-------|",
        f"| Generated | {report.generated_at} |",
        f"| Python | {report.python_version} |",
        f"| mobius | `{report.mobius_version}` |",
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

        # Op count table
        cols = mr.columns
        present_ops = {op for c in cols for op in c.counts if op in set(OP_TYPES)}
        ops_in_order = [(op, OP_LABEL[op]) for op, _, _ in _OP_CATALOG if op in present_ops]

        header = "| Op |" + "".join(f" {c.label} |" for c in cols) + " Notes |"
        sep_row = "|" + "---|" * (2 + len(cols))
        lines += [header, sep_row]

        for op, lbl in ops_in_order:
            counts = [c.counts.get(op, 0) for c in cols]
            differs = len(set(counts)) > 1
            note = "⚠️ differs" if differs else ""
            count_cells = "".join(f" {n} |" for n in counts)
            lines.append(f"| `{lbl}` |{count_cells} {note} |")

        # Total row
        total_cells = "".join(f" **{c.total_nodes}** |" for c in cols)
        lines.append(f"| **Total nodes** |{total_cells}  |")
        lines.append("")

        if mr.differences:
            lines += ["### Differences", ""]
            for diff in mr.differences:
                lines.append(f"- {diff}")
            lines += ["", "---", ""]
        else:
            lines += ["---", ""]

    # Summary
    lines += [
        "## Summary",
        "",
        "| Verdict | Count |",
        "|---------|-------|",
        f"| ✅ PASS | {report.pass_count} |",
        f"| 🟡 PARTIAL | {report.partial_count} |",
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
        "--model",
        action="append",
        dest="models",
        metavar="MODEL_ID",
        help="HuggingFace model ID to compare. Repeat for multiple models.",
    )
    parser.add_argument(
        "--ep",
        default="cuda",
        metavar="EP",
        help=f"Target EP for both builders (default: cuda). Available: {', '.join(all_eps)}.",
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
        default="/home/justinchu/dev/onnxruntime-genai",
        metavar="REPO_PATH",
        help="Path to onnxruntime-genai checkout for in-process building "
        "(default: %(default)s).",
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
        "--save-report",
        default=None,
        metavar="FILE",
        help="Save Markdown report to FILE (e.g. report.md).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color in console output.",
    )
    args = parser.parse_args()

    if not args.models:
        args.models = ["meta-llama/Llama-3.2-1B"]

    eps: list[str] = args.ep_list.split(",") if args.ep_list else [args.ep]
    no_ort = args.no_ort or bool(args.ep_list)

    report = ComparisonReport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        mobius_version=_mobius_version(),
        ort_genai_version=_ort_genai_version(args.ort_genai_repo if not no_ort else None),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )

    for i, model_id in enumerate(args.models):
        if args.ep_list:
            # Multi-EP mode: one ModelReport per model, one column per EP.
            print(f"\n→ {model_id}  (multi-EP: {', '.join(eps)})", flush=True)
            columns: list[OpCounts] = []
            for ep_target in eps:
                print(f"  Building with mobius/{ep_target} ...", end="", flush=True)
                try:
                    columns.append(
                        build_mobius(model_id, ep=ep_target, load_weights=args.load_weights)
                    )
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
                    )
                )
        else:
            # Cross-builder mode: one ModelReport per (model, ep).
            for ep in eps:
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
                    elif os.path.isdir(args.ort_genai_repo):
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
                            print(" done.")
                        except Exception as e:
                            print(f"\n  WARNING: ORT GenAI build failed: {e}")
                            print("  Use --ort-model or --no-ort to proceed without it.")
                    else:
                        print(
                            f"  NOTE: ORT GenAI repo not at {args.ort_genai_repo}. "
                            "Use --ort-genai-repo to specify. Skipping ORT GenAI column."
                        )

                # --- mobius column ---
                print(f"  Building with mobius/{ep} ...", end="", flush=True)
                try:
                    columns_.append(
                        build_mobius(model_id, ep=ep, load_weights=args.load_weights)
                    )
                    print(" done.")
                except Exception as e:
                    print(f"\n  ERROR: {e}")
                    import traceback

                    traceback.print_exc()

                if not columns_:
                    continue

                verdict, reason, diffs = _compute_verdict(columns_)
                report.models.append(
                    ModelReport(
                        model_id=model_id,
                        ep=ep,
                        columns=columns_,
                        verdict=verdict,
                        verdict_reason=reason,
                        differences=diffs,
                    )
                )

    if not report.models:
        print("No models built successfully.")
        sys.exit(1)

    # Console output
    print(render_console(report, color=not args.no_color))

    # Markdown output
    md = render_markdown(report)
    save_path = args.save_report
    if save_path is None:
        save_path = f"parity_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.md"
    with open(save_path, "w") as f:
        f.write(md)
    print(f"  Markdown report saved to: {save_path}\n")


if __name__ == "__main__":
    main()
