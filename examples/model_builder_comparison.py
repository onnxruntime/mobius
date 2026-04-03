#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare ONNX model graph structure between ORT GenAI's model builder and mobius.

Shows a side-by-side op-count table for the key ops that differ across builders
and execution providers — GroupQueryAttention, SkipLayerNormalization, RotaryEmbedding,
MatMul, etc.

Usage — compare pre-built ORT GenAI model against fresh mobius build::

    # Point at an existing ORT GenAI output dir and a HuggingFace model ID
    python examples/model_builder_comparison.py \\
        --ort-model /path/to/ort-genai-output/ \\
        --model meta-llama/Llama-3.2-1B \\
        --ep cuda

Usage — build with ORT GenAI builder on-the-fly (requires ort-genai checkout)::

    python examples/model_builder_comparison.py \\
        --model meta-llama/Llama-3.2-1B \\
        --ep cuda \\
        --ort-genai-repo /home/justinchu/dev/onnxruntime-genai

Usage — mobius-only (compare across EPs)::

    python examples/model_builder_comparison.py \\
        --model meta-llama/Llama-3.2-1B \\
        --ep-list default,cuda,dml,webgpu \\
        --no-ort

Supported model architectures (in both builders):
  llama / LlamaForCausalLM, mistral / MistralForCausalLM,
  qwen2 / Qwen2ForCausalLM, gemma / GemmaForCausalLM, phi / PhiForCausalLM

Notes:
  - mobius is built with ``load_weights=False`` by default (fast, no HF download).
  - ORT GenAI builder loads weights from HuggingFace (slower, requires download).
  - To avoid downloading, provide ``--ort-model <dir>`` with a pre-built model.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import onnx_ir as ir

# ---------------------------------------------------------------------------
# The ops we track in the comparison table
# ---------------------------------------------------------------------------

TRACKED_OPS: list[tuple[str, str]] = [
    # (op_type, description)
    ("GroupQueryAttention", "GQA (fused attention)"),
    ("Attention", "Attention (unfused)"),
    ("MultiHeadAttention", "MultiHeadAttention"),
    ("RotaryEmbedding", "RotaryEmbedding (fused RoPE)"),
    ("SkipLayerNormalization", "SkipLayerNorm"),
    ("SkipSimplifiedLayerNormalization", "SkipSimplifiedLayerNorm"),
    ("LayerNormalization", "LayerNormalization"),
    ("RMSNormalization", "RMSNorm (standard ONNX)"),
    ("BiasGelu", "BiasGelu (fused)"),
    ("FastGelu", "FastGelu"),
    ("Gelu", "Gelu (unfused)"),
    ("MatMul", "MatMul"),
    ("Shape", "Shape (should be 0 on WebGPU)"),
    ("Cast", "Cast"),
]

TRACKED_OP_TYPES = {op for op, _ in TRACKED_OPS}


# ---------------------------------------------------------------------------
# Model stats
# ---------------------------------------------------------------------------


class ModelStats(NamedTuple):
    name: str
    op_counts: dict[str, int]
    total_nodes: int


def _count_ops(model: ir.Model) -> dict[str, int]:
    """Count all op types in the graph (including subgraphs)."""
    counts: dict[str, int] = {}
    for node in model.graph:
        key = node.op_type
        counts[key] = counts.get(key, 0) + 1
    return counts


def stats_from_model(name: str, model: ir.Model) -> ModelStats:
    counts = _count_ops(model)
    total = sum(counts.values())
    return ModelStats(name=name, op_counts=counts, total_nodes=total)


def stats_from_file(name: str, path: str | Path) -> ModelStats:
    """Load an ONNX file and return ModelStats."""
    model = ir.load(str(path))
    return stats_from_model(name, model)


# ---------------------------------------------------------------------------
# mobius builder
# ---------------------------------------------------------------------------


def build_with_mobius(
    model_id: str,
    ep: str = "default",
    load_weights: bool = False,
) -> ModelStats:
    """Build a model with mobius and return ModelStats for the decoder."""
    from mobius import build

    pkg = build(
        model_id,
        execution_provider=ep,
        load_weights=load_weights,
    )
    # For VLMs, compare the decoder; otherwise the only model
    role = "model" if "model" in pkg else next(iter(pkg))
    label = f"mobius/{ep}"
    return stats_from_model(label, pkg[role])


# ---------------------------------------------------------------------------
# ORT GenAI builder (in-process via sys.path manipulation)
# ---------------------------------------------------------------------------


def build_with_ort_genai(
    model_id: str,
    ep: str = "cuda",
    precision: str = "fp16",
    ort_genai_repo: str | None = None,
    output_dir: str | None = None,
) -> ModelStats:
    """Build with ORT GenAI's model builder and return ModelStats.

    Requires either:
    - ``ort_genai_repo``: path to onnxruntime-genai checkout
    - ORT GenAI model builder already importable
    """
    builders_dir = None
    if ort_genai_repo:
        builders_dir = os.path.join(ort_genai_repo, "src/python/py/models")
        if not os.path.isdir(builders_dir):
            raise FileNotFoundError(
                f"ORT GenAI builders dir not found: {builders_dir}\n"
                "Expected: <ort_genai_repo>/src/python/py/models/builders/"
            )
        if builders_dir not in sys.path:
            sys.path.insert(0, builders_dir)

    try:
        from builder import create_model  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "Cannot import ORT GenAI model builder. "
            "Provide --ort-genai-repo pointing to the onnxruntime-genai checkout, "
            "or run from the models/ directory.\n"
            f"Original error: {e}"
        ) from e

    cleanup = output_dir is None
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="ort_genai_")

    try:
        create_model(
            model_name=model_id,
            input_path=model_id,
            output_dir=output_dir,
            precision=precision,
            execution_provider=ep,
            cache_dir=os.path.join(output_dir, "cache"),
        )
        onnx_path = Path(output_dir) / "model.onnx"
        if not onnx_path.exists():
            # Some models save to a subdir
            candidates = list(Path(output_dir).glob("**/*.onnx"))
            if not candidates:
                raise FileNotFoundError(
                    f"No .onnx file found in ORT GenAI output dir: {output_dir}"
                )
            onnx_path = candidates[0]
        return stats_from_file(f"ort-genai/{ep}", onnx_path)
    finally:
        if cleanup and os.path.isdir(output_dir):
            import shutil

            shutil.rmtree(output_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Comparison table rendering
# ---------------------------------------------------------------------------


def _render_table(stats_list: list[ModelStats]) -> str:
    """Render a comparison table as plain text."""
    # Find ops that appear in at least one model
    present_ops = {op for s in stats_list for op in s.op_counts if op in TRACKED_OP_TYPES}
    # Sort by TRACKED_OPS order
    ops_in_order = [(op, desc) for op, desc in TRACKED_OPS if op in present_ops]

    # Column widths
    name_w = max(len(s.name) for s in stats_list)
    col_w = max(name_w, 12)
    op_w = max((len(desc) for _, desc in ops_in_order), default=20)

    header = f"{'Op':<{op_w}} " + "  ".join(f"{s.name:>{col_w}}" for s in stats_list)
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for op_type, desc in ops_in_order:
        counts = [s.op_counts.get(op_type, 0) for s in stats_list]
        # Highlight rows where counts differ across models
        differs = len(set(counts)) > 1
        marker = " ◀" if differs else ""
        row = f"{desc:<{op_w}} " + "  ".join(f"{c:>{col_w}}" for c in counts)
        lines.append(row + marker)

    lines.append(sep)
    total_row = (
        f"{'Total nodes':<{op_w}} "
        + "  ".join(f"{s.total_nodes:>{col_w}}" for s in stats_list)
    )
    lines.append(total_row)
    lines.append(sep)

    legend = "\n  ◀ = counts differ between builders/EPs"
    return "\n".join(lines) + legend


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    from mobius._execution_providers import ep_registry

    all_eps = sorted(ep_registry.names())

    parser = argparse.ArgumentParser(
        description="Compare ONNX op counts between ORT GenAI builder and mobius.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model ID to compare (default: %(default)s).",
    )
    parser.add_argument(
        "--ep",
        default="cuda",
        metavar="EP",
        help=f"Target EP for mobius (default: cuda). Available: {', '.join(all_eps)}.",
    )
    parser.add_argument(
        "--ep-list",
        default=None,
        metavar="EP1,EP2,...",
        help="Compare mobius output across multiple EPs (e.g. default,cuda,dml,webgpu). "
        "Overrides --ep.",
    )
    parser.add_argument(
        "--ort-model",
        default=None,
        metavar="DIR_OR_FILE",
        help="Pre-built ORT GenAI model dir or .onnx file. "
        "Skips building with ORT GenAI builder.",
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
        "--no-ort",
        action="store_true",
        help="Skip ORT GenAI comparison (mobius-only, useful with --ep-list).",
    )
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Download and apply weights when building with mobius (slower).",
    )
    args = parser.parse_args()

    eps: list[str] = args.ep_list.split(",") if args.ep_list else [args.ep]

    print(f"\nModel: {args.model}")
    print(f"Mobius EPs: {', '.join(eps)}")
    print()

    all_stats: list[ModelStats] = []

    # --- ORT GenAI ---
    if not args.no_ort:
        if args.ort_model:
            # Load pre-built model
            ort_path = Path(args.ort_model)
            if ort_path.is_dir():
                candidates = list(ort_path.glob("**/*.onnx"))
                if not candidates:
                    print(f"ERROR: No .onnx file found in {ort_path}")
                    sys.exit(1)
                ort_path = candidates[0]
            print(f"Loading ORT GenAI model from: {ort_path}")
            try:
                all_stats.append(stats_from_file("ort-genai", ort_path))
            except Exception as e:
                print(f"WARNING: Failed to load ORT GenAI model: {e}")
        else:
            # Try building with ORT GenAI builder
            ort_repo = args.ort_genai_repo
            if os.path.isdir(ort_repo):
                print(f"Building with ORT GenAI builder ({args.ort_precision}/{args.ep}) ...")
                try:
                    s = build_with_ort_genai(
                        args.model,
                        ep=args.ep,
                        precision=args.ort_precision,
                        ort_genai_repo=ort_repo,
                    )
                    all_stats.append(s)
                    print("  Done.")
                except Exception as e:
                    print(f"WARNING: ORT GenAI build failed: {e}")
                    print("  Use --ort-model to provide a pre-built model, or --no-ort to skip.")
            else:
                print(
                    f"NOTE: ORT GenAI repo not found at {ort_repo}. "
                    "Skipping ORT GenAI comparison.\n"
                    "      Provide --ort-genai-repo or --ort-model to enable it."
                )

    # --- mobius ---
    for ep in eps:
        print(f"Building with mobius (ep={ep}, load_weights={args.load_weights}) ...")
        try:
            s = build_with_mobius(args.model, ep=ep, load_weights=args.load_weights)
            all_stats.append(s)
            print("  Done.")
        except Exception as e:
            print(f"ERROR: mobius build failed for ep={ep}: {e}")
            import traceback

            traceback.print_exc()

    if not all_stats:
        print("No models built successfully. Exiting.")
        sys.exit(1)

    print()
    print(_render_table(all_stats))
    print()


if __name__ == "__main__":
    main()
