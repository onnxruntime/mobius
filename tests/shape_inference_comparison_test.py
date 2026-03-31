# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Comparison tests: mobius SymbolicShapeInferencePass vs onnx_ir ShapeInferencePass.

This module documents and verifies the behavioural differences between the two
shape inference strategies used in (or available to) mobius:

  1. **``SymbolicShapeInferencePass``** — mobius's primary pass, wrapping
     ``onnx_shape_inference.infer_symbolic_shapes``.  Works natively on
     ``ir.Model`` without serialization.  Produces *symbolic* dimensions
     (e.g. ``batch``, ``sequence_len``, ``past_seq_len + seq_len``) and
     handles both standard ONNX ops and ``com.microsoft.*`` contrib ops.

  2. **``onnx_ir.passes.common.ShapeInferencePass``** — wraps the ONNX C++
     inference engine (``onnx.shape_inference.infer_shapes``).  Round-trips
     through protobuf.  Produces only *concrete* (integer) shapes when all
     data is statically known.  Cannot start from models with unknown dtypes.

Run these tests with::

    pytest tests/shape_inference_comparison_test.py -v
"""

from __future__ import annotations

import copy
import sys
from typing import NamedTuple

import numpy as np
import onnx_ir as ir
import pytest
from _test_configs import _base_config
from onnx_ir.passes import common as common_passes

sys.path.insert(0, "tests")

from mobius._builder import SymbolicShapeInferencePass, build_from_module
from mobius.models import CausalLMModel
from mobius.tasks import CausalLMTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CoverageStats(NamedTuple):
    """Shape/dtype coverage statistics for a model's intermediate values."""

    total: int
    with_shape: int
    with_dtype: int
    missing_shape_ops: list[tuple[str, str]]  # [(op_type, domain), ...]

    @property
    def shape_pct(self) -> float:
        return self.with_shape / self.total if self.total else 0.0

    @property
    def dtype_pct(self) -> float:
        return self.with_dtype / self.total if self.total else 0.0


def _count_coverage(model: ir.Model) -> CoverageStats:
    """Count shape/dtype coverage over all intermediate node outputs."""
    total = with_shape = with_dtype = 0
    missing_shape_ops: list[tuple[str, str]] = []
    for node in model.graph:
        for v in node.outputs:
            total += 1
            if v.shape is not None:
                with_shape += 1
            else:
                missing_shape_ops.append((node.op_type, node.domain or "ai.onnx"))
            if v.dtype is not None:
                with_dtype += 1
    return CoverageStats(total, with_shape, with_dtype, missing_shape_ops)


def _strip_shapes(model: ir.Model) -> None:
    """Remove shape and dtype from all intermediate values (not graph I/O)."""
    for node in model.graph:
        for v in node.outputs:
            v.shape = None
            v.dtype = None


def _fill_dummy_weights(model: ir.Model) -> None:
    """Fill initializers that have no const_value with zero tensors.

    Required for ShapeInferencePass, which serializes the model to
    protobuf before calling the ONNX C++ inference engine.
    """
    for initializer in model.graph.initializers.values():
        if initializer.const_value is not None:
            continue
        shape = initializer.shape
        dims = [d if isinstance(d, int) else 1 for d in shape] if shape else [1]
        dtype = initializer.dtype or ir.DataType.FLOAT
        initializer.const_value = ir.Tensor(np.zeros(dims, dtype=dtype.numpy()))


def _unique_missing_ops(stats: CoverageStats) -> set[str]:
    return {f"{d}.{op}" for op, d in stats.missing_shape_ops}


def _collect_op_types(model: ir.Model) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for node in model.graph:
        key = (node.op_type, node.domain or "ai.onnx")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _build_raw(config=None) -> ir.Model:
    """Build a CausalLM model *without* the optimization pipeline."""
    if config is None:
        config = _base_config()
    module = CausalLMModel(config)
    task = CausalLMTask()
    pkg = task.build(module, config)
    return pkg["model"]


def _build_optimized(config=None) -> ir.Model:
    """Build a CausalLM model *with* the full optimization pipeline."""
    if config is None:
        config = _base_config()
    module = CausalLMModel(config)
    pkg = build_from_module(module, config, "text-generation")
    return pkg["model"]


# ---------------------------------------------------------------------------
# Test 1 — Symbolic pass achieves full coverage from scratch
# ---------------------------------------------------------------------------


def test_symbolic_pass_achieves_full_coverage_from_scratch():
    """SymbolicShapeInferencePass infers shapes/dtypes for every node output.

    No pre-existing shape info is required. The pass operates directly on the
    ir.Model without any serialization round-trip.
    """
    model = _build_raw()
    _strip_all_type_info(model)  # verify it truly starts from zero

    baseline = _count_coverage(model)
    assert baseline.with_shape == 0, "Baseline should have no shapes"
    assert baseline.with_dtype == 0, "Baseline should have no dtypes"

    SymbolicShapeInferencePass()(model)

    after = _count_coverage(model)
    # Allow at most 1 missing shape (Expand with fully dynamic broadcast)
    assert after.with_shape >= after.total - 1, (
        f"Expected near-100% shape coverage, got {after.with_shape}/{after.total}. "
        f"Missing ops: {_unique_missing_ops(after)}"
    )
    assert after.with_dtype == after.total, (
        f"Expected 100% dtype coverage, got {after.with_dtype}/{after.total}"
    )


# ---------------------------------------------------------------------------
# Test 2 — onnx_ir ShapeInferencePass fails on models with unknown dtypes
# ---------------------------------------------------------------------------


def _strip_all_type_info(model: ir.Model) -> None:
    """Strip both shape and dtype from all intermediate values."""
    for node in model.graph:
        for v in node.outputs:
            v.shape = None
            v.dtype = None


def test_onnx_ir_pass_fails_when_dtypes_unknown():
    """onnx_ir ShapeInferencePass requires dtype on all values to serialize.

    When intermediate values have no dtype, protobuf serialization fails and
    the pass leaves the model unchanged.  This is the fundamental bootstrap
    limitation: ShapeInferencePass cannot be the *first* pass applied to a
    model that has had all type information stripped.

    Root cause: ``onnx_ir.serde.serialize_value_into`` raises ``SerdeError``
    when it encounters ``Value(type=Tensor(None))``, because the protobuf
    ``ValueInfoProto.type`` field requires a concrete ``elem_type``.

    Note: The raw model from ``task.build()`` retains partial shape/dtype
    from onnxscript's own type propagation.  We strip everything explicitly
    here to demonstrate the failure mode.
    """
    model = _build_raw()
    # Strip all shapes and dtypes to simulate a truly blank model
    _strip_all_type_info(model)
    # Also strip initializer const_values to trigger the earliest failure path
    for initializer in model.graph.initializers.values():
        initializer.const_value = None

    result = common_passes.ShapeInferencePass()(model)

    stats = _count_coverage(model)
    # Pass must have left model unchanged (it catches the failure internally)
    assert not result.modified, "ShapeInferencePass should report modified=False on failure"
    assert stats.with_shape == 0, (
        "ShapeInferencePass should leave shapes unchanged when serialization fails"
    )


# ---------------------------------------------------------------------------
# Test 3 — onnx_ir ShapeInferencePass succeeds on a fully-typed model
# ---------------------------------------------------------------------------


def test_onnx_ir_pass_succeeds_after_symbolic_pass():
    """ShapeInferencePass runs successfully when all types are already known.

    After SymbolicShapeInferencePass fills in all dtypes, ShapeInferencePass
    can serialize and round-trip without error.  In practice it adds no new
    information (symbolic inference already covers everything), but it
    confirms the two passes are compatible as a pipeline.
    """
    model = _build_optimized()
    _fill_dummy_weights(model)

    stats_before = _count_coverage(model)
    # Full optimization already ran — expect 100% coverage
    assert stats_before.with_shape >= stats_before.total - 1

    common_passes.ShapeInferencePass()(model)

    stats_after = _count_coverage(model)
    # Coverage must not regress
    assert stats_after.with_shape >= stats_before.with_shape
    assert stats_after.with_dtype >= stats_before.with_dtype


# ---------------------------------------------------------------------------
# Test 4 — Symbolic shapes (not concrete integers) are preserved
# ---------------------------------------------------------------------------


def test_symbolic_pass_produces_named_symbolic_dimensions():
    """SymbolicShapeInferencePass produces named symbolic dimensions.

    Graph inputs carry dimension names like ``batch`` and ``sequence_len``.
    These propagate through the graph, enabling downstream tools to understand
    which dimension carries batch semantics vs. sequence semantics.

    onnx.shape_inference.infer_shapes cannot express this — it can only
    substitute concrete integer values when the data is statically known.
    """
    model = _build_optimized()

    # Collect all dimension objects from node outputs
    symbolic_dims: set[str] = set()
    for node in model.graph:
        for v in node.outputs:
            if v.shape is None:
                continue
            for dim in v.shape:
                dim_str = str(dim)
                if not dim_str.isdigit() and dim_str not in ("None", "?"):
                    symbolic_dims.add(dim_str)

    # We expect at least "batch" and "sequence_len" to propagate
    assert any("batch" in d for d in symbolic_dims), (
        f"Expected 'batch' symbolic dim to propagate. Found: {symbolic_dims}"
    )
    assert any("seq" in d.lower() for d in symbolic_dims), (
        f"Expected 'sequence_len' symbolic dim to propagate. Found: {symbolic_dims}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Symbolic dims carry through graph inputs
# ---------------------------------------------------------------------------


def test_graph_inputs_carry_semantic_dimension_names():
    """Graph inputs use human-readable symbolic dimension names.

    These names (``batch``, ``sequence_len``, ``past_sequence_len``) are set
    by the task builder and propagated by ``SymbolicShapeInferencePass``.
    ``onnx.shape_inference`` has no concept of named dimensions — it would
    leave dynamic axes as unnamed ``?``.
    """
    model = _build_optimized()

    all_dim_names: set[str] = set()
    for inp in model.graph.inputs:
        if inp.shape is None:
            continue
        for dim in inp.shape:
            all_dim_names.add(str(dim))

    assert "batch" in all_dim_names, f"Expected 'batch' in input dims, got: {all_dim_names}"
    assert any("seq" in d.lower() for d in all_dim_names), (
        f"Expected sequence dimension in inputs, got: {all_dim_names}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Op-type coverage comparison
# ---------------------------------------------------------------------------


def test_op_coverage_comparison():
    """Document which op types are covered by each pass.

    SymbolicShapeInferencePass registers handlers for:
      - All standard ONNX opset ops (199 in ai.onnx / '' domain)
      - ONNX opset 23 ops used by mobius: Attention, RMSNormalization,
        RotaryEmbedding (registered under '' domain)
      - Microsoft contrib ops: 51 ops in com.microsoft domain

    ShapeInferencePass delegates to ONNX C++ inference (onnx 1.20+, opset 25),
    which handles standard ONNX + opset 23 ops but NOT com.microsoft contrib ops.

    Since mobius currently emits ONNX opset 23 standard ops (not com.microsoft),
    both passes cover our current model graphs.  The gap matters for any future
    graph variant targeting ORT contrib ops directly.
    """
    import onnx
    import onnx_shape_inference as osi
    import onnx_shape_inference._ops  # trigger registration

    reg = osi.registry

    # Ops registered in onnx_shape_inference
    sym_ops_by_domain: dict[str, set[str]] = {}
    for domain, op_type in reg._registrations:
        sym_ops_by_domain.setdefault(domain, set()).add(op_type)

    # Key stats
    standard_count = len(sym_ops_by_domain.get("", set()))
    ms_contrib_count = len(sym_ops_by_domain.get("com.microsoft", set()))
    onnx_opset = onnx.defs.onnx_opset_version()

    # Verify key opset 23 ops are registered (used by all mobius models)
    for op in ("Attention", "RMSNormalization", "RotaryEmbedding"):
        assert op in sym_ops_by_domain.get("", set()), (
            f"SymbolicShapeInferencePass should register opset-23 op '{op}'"
        )

    # Verify opset 23 ops are known to ONNX C++ (onnx >= 1.17 for opset 23)
    for op in ("Attention", "RMSNormalization", "RotaryEmbedding"):
        try:
            schema = onnx.defs.get_schema(op, domain="")
            assert schema is not None
        except Exception:
            pytest.fail(
                f"ONNX C++ schema missing for '{op}' — onnx version {onnx.__version__} "
                f"may be too old (need >= 1.17 for opset 23)"
            )

    # Summary for human readers (always printed)
    print("\n=== Op Coverage Summary ===")
    print(f"onnx version: {onnx.__version__} (opset {onnx_opset})")
    print(
        f"SymbolicShapeInferencePass: {standard_count} standard ops, "
        f"{ms_contrib_count} com.microsoft contrib ops"
    )
    print(
        "ShapeInferencePass (ONNX C++): covers standard ONNX + opset 23, "
        "NO com.microsoft contrib ops"
    )
    print("Mobius model ops: all standard ONNX + opset 23 (no contrib ops currently)")

    assert standard_count >= 190, f"Expected 190+ standard ops, got {standard_count}"
    assert ms_contrib_count >= 40, f"Expected 40+ ms contrib ops, got {ms_contrib_count}"


# ---------------------------------------------------------------------------
# Test 7 — Full pipeline coverage across representative model types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type,extra_kwargs",
    [
        ("llama", {}),
        ("qwen2", {"sliding_window": 32}),
        ("gemma", {}),
        ("mistral", {}),
    ],
)
def test_symbolic_pass_full_coverage_representative_models(model_type, extra_kwargs):
    """SymbolicShapeInferencePass achieves ≥99% shape coverage for representative models."""
    config = _base_config(**extra_kwargs)
    module = CausalLMModel(config)
    pkg = build_from_module(module, config, "text-generation")
    model = pkg["model"]

    stats = _count_coverage(model)
    assert stats.shape_pct >= 0.99, (
        f"{model_type}: Expected ≥99% shape coverage, got {stats.with_shape}/{stats.total}. "
        f"Missing: {_unique_missing_ops(stats)}"
    )
    assert stats.dtype_pct >= 1.0, (
        f"{model_type}: Expected 100% dtype coverage, got {stats.with_dtype}/{stats.total}"
    )


# ---------------------------------------------------------------------------
# Test 8 — Comparison summary (always printed, never fails)
# ---------------------------------------------------------------------------


def test_print_comparison_summary():
    """Print a structured comparison between the two passes.

    This test always passes — its purpose is to emit a human-readable summary
    to the test output (``pytest -v -s``) documenting the findings.
    """
    model_raw = _build_raw()

    # ---- Pass A: Symbolic -----------------------------------------------
    model_sym = copy.deepcopy(model_raw)
    SymbolicShapeInferencePass()(model_sym)
    sym_stats = _count_coverage(model_sym)

    # ---- Pass B: ONNX C++ on stripped model (failure scenario) -----------
    model_onnx_stripped = copy.deepcopy(model_raw)
    _strip_all_type_info(model_onnx_stripped)
    for init in model_onnx_stripped.graph.initializers.values():
        init.const_value = None
    common_passes.ShapeInferencePass()(model_onnx_stripped)
    onnx_ir_stripped_stats = _count_coverage(model_onnx_stripped)

    # ---- Pass B: ONNX C++ on raw model (partial shapes from onnxscript) --
    model_onnx_ir = copy.deepcopy(model_raw)
    common_passes.ShapeInferencePass()(model_onnx_ir)
    onnx_ir_stats = _count_coverage(model_onnx_ir)

    # ---- Pass B after A --------------------------------------------------
    model_combined = copy.deepcopy(model_sym)
    _fill_dummy_weights(model_combined)
    result_combined = common_passes.ShapeInferencePass()(model_combined)
    combined_stats = _count_coverage(model_combined)

    # Collect symbolic dim names
    symbolic_dims: set[str] = set()
    for node in model_sym.graph:
        for v in node.outputs:
            if v.shape is None:
                continue
            for dim in v.shape:
                dim_str = str(dim)
                if not dim_str.isdigit():
                    symbolic_dims.add(dim_str)

    print("\n" + "=" * 70)
    print("SHAPE INFERENCE COMPARISON: SymbolicPass vs onnx_ir ShapeInferencePass")
    print("=" * 70)
    print("\nModel: CausalLMModel (LLaMA-style, tiny config — no weights)")
    print(f"Total intermediate values: {sym_stats.total}")
    print()
    print(f"{'Property':<48} {'Symbolic':>10} {'onnx_ir':>10}")
    print("-" * 70)
    print(f"{'Starting state required':<48} {'Any':>10} {'Typed model':>10}")
    print(
        f"{'Shape coverage (fully stripped model)':<48} "
        f"{sym_stats.with_shape:>9}/{sym_stats.total} "
        f"{onnx_ir_stripped_stats.with_shape:>9}/{onnx_ir_stripped_stats.total}"
    )
    print(
        f"{'Shape coverage (raw model from onnxscript)':<48} "
        f"{'N/A':>10} "
        f"{onnx_ir_stats.with_shape:>9}/{onnx_ir_stats.total}"
    )
    print(
        f"{'Dtype coverage (fully stripped model)':<48} "
        f"{sym_stats.with_dtype:>9}/{sym_stats.total} "
        f"{onnx_ir_stripped_stats.with_dtype:>9}/{onnx_ir_stripped_stats.total}"
    )
    print(f"{'Produces symbolic dims (batch, seq_len)':<48} {'Yes':>10} {'No':>10}")
    print(f"{'com.microsoft contrib op support':<48} {'Yes (51)':>10} {'No':>10}")
    print(f"{'Serialization round-trip required':<48} {'No':>10} {'Yes':>10}")
    print(f"{'Can run after optimization pipeline':<48} {'Yes':>10} {'Yes':>10}")
    print()
    print(f"Symbolic dim names found: {sorted(symbolic_dims)[:8]} ...")
    print()
    print("After SymbolicPass → ShapeInferencePass pipeline:")
    print(
        f"  Shape coverage: {combined_stats.with_shape}/{combined_stats.total}, "
        f"modified={result_combined.modified}"
    )
    print("  Conclusion: ShapeInferencePass adds nothing after SymbolicPass")
    print("=" * 70)
