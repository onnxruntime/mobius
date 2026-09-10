#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""EP comparison example — inspect construction differences across EPs.

Builds a small Qwen3.5 model (graph only, no weights) for each registered
execution provider and shows construction-time operator differences. The
finalization trace covers cleanup and local-function inlining, not downstream
Olive graph transformations.

Usage::

    # Compare all EPs (no weights downloaded — fast)
    python examples/ep_comparison.py

    # Compare a single EP
    python examples/ep_comparison.py --ep cuda

    # Use a different model
    python examples/ep_comparison.py --model Qwen/Qwen3.5-0.8B

    # Register a custom EP, then compare
    python examples/ep_comparison.py --ep my-ep

Design notes
------------
- ``--no-weights`` (default) avoids downloading large checkpoints.
  The graph structure is identical with or without weights.
- ``trace_optimization=True`` logs each finalization stage at INFO level.
  Set ``--verbose`` to see the full per-stage log.
- ``ep_registry.names()`` is the live registry — any EP registered via
  ``register_ep()`` automatically appears in ``--ep`` choices.

Custom EP example::

    from mobius._execution_providers import EpCapabilities, register_ep
    import onnx_ir as ir

    register_ep(EpCapabilities(
        name="my-ep",
        gqa_dtypes=frozenset({ir.DataType.FLOAT16}),
    ))

    # Now usable:
    pkg = build("Qwen/Qwen3.5-2B", execution_provider="my-ep")
"""

from __future__ import annotations

import argparse
import logging

import onnx_ir as ir

from mobius import build
from mobius._execution_providers import ep_registry

# ---------------------------------------------------------------------------
# EP descriptions — shown alongside trace output for context
# ---------------------------------------------------------------------------

_EP_NOTES: dict[str, str] = {
    "default": "Portable construction path; local function bodies are retained.",
    "cpu": "CPU contract; compatible FLOAT decoders emit GQA directly.",
    "cuda": "CUDA contract; compatible decoders and vision models emit fused ops directly.",
    "dml": "DirectML contract; standard Attention is retained when fused RoPE is unavailable.",
    "webgpu": (
        "WebGPU contract; compatible decoders emit GQA directly and "
        "graph construction avoids unsupported dynamic empty-KV shapes."
    ),
    "trt-rtx": (
        "TensorRT-RTX contract; compatible FLOAT16/BFLOAT16 decoders emit GQA directly."
    ),
}


def _op_counts(model: ir.Model) -> dict[str, int]:
    """Return a dict of op_type → count for all nodes in the graph."""
    counts: dict[str, int] = {}
    for node in model.graph:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def _print_key_ops(model: ir.Model) -> None:
    """Print the counts of EP-diagnostic ops."""
    interesting = [
        "GroupQueryAttention",
        "Attention",
        "RotaryEmbedding",
        "SkipLayerNormalization",
        "SkipSimplifiedLayerNormalization",
        "Shape",
        "MatMul",
        "Cast",
    ]
    counts = _op_counts(model)
    relevant = {op: counts[op] for op in interesting if op in counts}
    if relevant:
        for op, count in sorted(relevant.items()):
            print(f"    {op}: {count}")
    else:
        print("    (no diagnostic ops found)")


def compare_eps(model_id: str, eps: list[str], *, dtype: str | None, verbose: bool) -> None:
    """Build *model_id* for each EP in *eps* and print a summary."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="  %(message)s", level=level)

    print(f"Model: {model_id}")
    if dtype:
        print(f"Dtype:  {dtype} (override)")
    print(f"EPs to compare: {eps}\n")

    for ep in eps:
        note = _EP_NOTES.get(ep, "(custom EP)")
        print("=" * 64)
        print(f"  EP: {ep}")
        print(f"  {note}")
        print("=" * 64)

        pkg = build(
            model_id,
            execution_provider=ep,
            dtype=dtype,
            load_weights=False,
            trace_optimization=True,
        )

        # Show key op counts for the decoder (or only model)
        role = "model" if "model" in pkg else next(iter(pkg))
        model = pkg[role]
        print(f"\n  Key ops in '{role}':")
        _print_key_ops(model)
        print()


def main() -> None:
    all_eps = sorted(ep_registry.names())

    parser = argparse.ArgumentParser(
        description="Compare Mobius graph construction across execution providers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace model ID (default: %(default)s). "
        "Pass any registered model; no weights are downloaded.",
    )
    parser.add_argument(
        "--ep",
        default="all",
        metavar="EP",
        help=(
            f"Target EP, or 'all' (default). Available: {', '.join(all_eps)}. "
            "Custom EPs registered via register_ep() also work."
        ),
    )
    parser.add_argument(
        "--dtype",
        default=None,
        metavar="DTYPE",
        help=(
            "Override model dtype (e.g. 'bfloat16', 'float16', 'float32'). "
            "Without this, dtype is auto-detected from the HuggingFace config. "
            "Useful when the HF config dtype is fp32 but you want to inspect "
            "GPU construction paths that require fp16/bf16."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show the full DEBUG-level finalization trace.",
    )
    args = parser.parse_args()

    eps = all_eps if args.ep == "all" else [args.ep]

    # Validate EP name (allow custom EPs not in the list)
    for ep in eps:
        if ep != "all" and ep not in ep_registry:
            parser.error(
                f"Unknown EP '{ep}'. "
                f"Available: {sorted(ep_registry.names())}. "
                "Register custom EPs with register_ep() before running."
            )

    compare_eps(args.model, eps, dtype=args.dtype, verbose=args.verbose)


if __name__ == "__main__":
    main()
