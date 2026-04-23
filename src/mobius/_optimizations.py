# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""EP-aware model optimization pipeline.

Exposes :func:`optimize_model` which applies a four-stage pass pipeline to an
ONNX IR model:

1. **Cleanup** — identity elimination, CSE, dead-code removal, constant
   folding, shape inference. EP-agnostic; always applied.
2. **Fusion** — promote standard ops to EP-supported fused ops
   (GQA, SkipNorm, GeluFusion). Gated by ``(ep, dtype)`` and ``model_role``.
3. **Lowering** — decompose ops the EP cannot execute
   (SeparateRoPE).
4. **Fold** — final dead-node removal and constant folding.

All EP knowledge is encoded in :class:`~mobius._execution_providers.EpCapabilities`
entries in the :data:`~mobius._execution_providers.ep_registry`. Adding EP
support requires only a new registry entry — no changes to this module.

Post-weight passes
------------------
:func:`fold_initializers_after_weights` should be called after weights are loaded.
It runs :class:`~mobius._passes.FoldTransposedInitializerPass` and
:class:`~mobius._passes.FoldConcatInitializersPass` to fold runtime Transpose and
Concat nodes over initializers into pre-computed weights, then removes unused
nodes.
"""

from __future__ import annotations

__all__ = [
    # Public API
    "optimize_model",
    "fold_initializers_after_weights",
    # Passes (used by tests and _builder re-exports)
    "CleanupMetadataPass",
    "SymbolicShapeInferencePass",
    # Diagnostic helpers
    "_count_all_ops",
    "_count_ops",
]

import contextlib
import dataclasses
import logging
import warnings

import onnx_ir as ir
import onnx_shape_inference
import onnxscript.optimizer._constant_folding
from onnx_ir.passes import common as common_passes
from onnxscript.rewriter import rewrite

from mobius._execution_providers import EpCapabilities, ep_registry
from mobius._flags import flags
from mobius._passes import (
    FoldConcatInitializersPass,
    FoldTransposedInitializerPass,
    RemoveDeadGraphInputsPass,
)
from mobius.functions import register_function_bodies
from mobius.rewrite_rules import (
    gelu_fusion_rules,
    group_query_attention_rules,
    pack_qkv_for_gqa_rules,
    separate_rope_rules,
    skip_layer_norm_rules,
    skip_norm_rules,
    unpack_qkv_rules,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility passes
# ---------------------------------------------------------------------------


class SymbolicShapeInferencePass(ir.passes.InPlacePass):
    """ONNX IR pass that applies symbolic shape inference to all nodes."""

    def __init__(self, policy: onnx_shape_inference.ShapeMergePolicy = "refine"):
        super().__init__()
        self.policy = policy

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        try:
            onnx_shape_inference.infer_symbolic_shapes(model, policy=self.policy)
        except Exception:
            # Upstream onnx_shape_inference bugs: e.g. comparison between int
            # and SymbolicDim, or ShapeInferenceError on complex models.
            # Non-fatal — skip gracefully.
            logger.warning("Symbolic shape inference failed (upstream bug); skipping")
        return ir.passes.PassResult(model, modified=True)


class CleanupMetadataPass(ir.passes.InPlacePass):
    """ONNX IR pass that removes redundant metadata from all nodes."""

    def __init__(self):
        self.keys_to_remove = ["pkg.onnxscript.shape_inference_error"]

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        for node in model.graph.all_nodes():
            for key in self.keys_to_remove:
                if key in node.metadata_props:
                    modified = True
                    del node.metadata_props[key]
        return ir.passes.PassResult(model, modified=modified)


# Maximum number of elements allowed in a constant-folded output tensor.
# Large weight tensors (Transpose, Concat/QKV packing) are handled by
# FoldTransposedInitializerPass and FoldConcatInitializersPass after weight
# loading, so the general constant-fold pass no longer needs a high limit.
_FOLD_OUTPUT_SIZE_LIMIT = 262144

_DEFAULT_PASSES = [
    common_passes.IdentityEliminationPass(),
    common_passes.LiftConstantsToInitializersPass(),
    common_passes.DeduplicateInitializersPass(),
    common_passes.CommonSubexpressionEliminationPass(),
    common_passes.RemoveUnusedNodesPass(),
    common_passes.RemoveUnusedOpsetsPass(),
    SymbolicShapeInferencePass(),
    onnxscript.optimizer._constant_folding.FoldConstantsPass(
        shape_inference=False,
        input_size_limit=8192,
        output_size_limit=_FOLD_OUTPUT_SIZE_LIMIT,
    ),
    CleanupMetadataPass(),
]


class _SuppressNoConstValueWarning(logging.Filter):
    """Filter out 'has no constant value' warnings from initializer dedup.

    Mobius runs optimization passes before weight loading, so weight
    initializers intentionally have no const_value at that point.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "has no constant value" not in record.getMessage()


@contextlib.contextmanager
def _suppress_dedup_empty_initializer_warnings():
    """Temporarily suppress 'has no constant value' dedup warnings."""
    dedup_logger = logging.getLogger("onnx_ir.passes.common.initializer_deduplication")
    log_filter = _SuppressNoConstValueWarning()
    dedup_logger.addFilter(log_filter)
    try:
        yield
    finally:
        dedup_logger.removeFilter(log_filter)


# Standard ONNX domains — functions from these domains are never expanded by InlinePass.
_STANDARD_ONNX_DOMAINS: frozenset[str] = frozenset({"", "ai.onnx"})

# ---------------------------------------------------------------------------
# Op counting helpers
# ---------------------------------------------------------------------------


def _count_ops(model: ir.Model, op_type: str) -> int:
    """Count nodes of a given op_type in all model graph nodes."""
    return sum(1 for node in model.graph.all_nodes() if node.op_type == op_type)


def _count_all_ops(model: ir.Model) -> dict[str, int]:
    """Count all op types present in the model graph (including subgraphs)."""
    counts: dict[str, int] = {}
    for node in model.graph.all_nodes():
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Trace infrastructure
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _TraceEntry:
    """Per-stage diagnostic data collected during traced optimization."""

    name: str
    added: dict[str, int]  # op_type → count added
    removed: dict[str, int]  # op_type → count removed (positive values)

    @property
    def nodes_added(self) -> int:
        return sum(self.added.values())

    @property
    def nodes_removed(self) -> int:
        return sum(self.removed.values())

    @property
    def matched(self) -> int:
        """Nodes consumed/replaced by this stage (proxy for 'rules matched')."""
        return self.nodes_removed


def _make_trace_entry(name: str, before: dict[str, int], after: dict[str, int]) -> _TraceEntry:
    all_ops = set(before) | set(after)
    added = {
        op: after.get(op, 0) - before.get(op, 0)
        for op in all_ops
        if after.get(op, 0) > before.get(op, 0)
    }
    removed = {
        op: before.get(op, 0) - after.get(op, 0)
        for op in all_ops
        if before.get(op, 0) > after.get(op, 0)
    }
    return _TraceEntry(name=name, added=added, removed=removed)


def _apply_stage(model: ir.Model, rules_or_pass: list | ir.passes.InPlacePass) -> None:
    """Apply a single optimization stage — either a rewrite-rule list or an IR pass."""
    if isinstance(rules_or_pass, ir.passes.InPlacePass):
        rules_or_pass(model)
    elif rules_or_pass:
        rewrite(model, pattern_rewrite_rules=rules_or_pass)


def _log_trace_entry(entry: _TraceEntry) -> None:
    if not entry.added and not entry.removed:
        logger.info("[EP Trace]   %-25s: no matches (0 nodes affected)", entry.name)
        return
    parts = [f"+{count} {op}" for op, count in sorted(entry.added.items())]
    parts += [f"-{count} {op}" for op, count in sorted(entry.removed.items())]
    logger.info("[EP Trace]   %-25s: %s", entry.name, ", ".join(parts))


def _log_trace_summary(entries: list[_TraceEntry]) -> None:
    if not entries:
        return
    logger.info("[EP Trace] Summary:")
    logger.info("[EP Trace]   %-25s | %7s | %6s | %6s", "Rule", "Matched", "+Nodes", "-Nodes")
    logger.info("[EP Trace]   %s", "-" * 57)
    for e in entries:
        logger.info(
            "[EP Trace]   %-25s | %7d | %6d | %6d",
            e.name,
            e.matched,
            e.nodes_added,
            e.nodes_removed,
        )


# ---------------------------------------------------------------------------
# Optimization pass selection
# ---------------------------------------------------------------------------


def _get_optimization_passes(
    caps: EpCapabilities,
    dtype: ir.DataType,
    model_role: str = "decoder",
) -> tuple[list[tuple[str, list]], list[tuple[str, list | ir.passes.InPlacePass]]]:
    """Return ``(fuse_stages, lower_stages)`` for the given capabilities.

    Queries the :class:`EpCapabilities` object rather than branching on EP
    name strings. Adding a new EP requires only a new registry entry —
    no changes to this function.

    Args:
        caps: EP capability descriptor from :data:`~mobius._execution_providers.ep_registry`.
        dtype: Model dtype for GQA/PackedAttn support checks.
        model_role: Semantic role. GQA fusion only applies to ``"decoder"``.

    Returns:
        ``(fuse_stages, lower_stages)`` — each a list of ``(name, payload)``
        tuples where payload is a rule list or IR pass.
    """
    fuse: list[tuple[str, list]] = []
    lower: list[tuple[str, list | ir.passes.InPlacePass]] = []

    # --- Attention fusion (decoder only) ---
    if model_role == "decoder" and dtype in caps.gqa_dtypes:
        fuse.append(("GQAFusion", list(group_query_attention_rules())))

    # --- QKV packing (decoder only, gated by qkv_pack_dtypes) ---
    if model_role == "decoder" and dtype in caps.qkv_pack_dtypes:
        fuse.append(("PackQKV", list(pack_qkv_for_gqa_rules())))

    # --- Normalization fusions (all roles, all dtypes) ---
    # When supports_skip_layer_norm=False (e.g. trt-rtx), fusion is skipped;
    # InlinePass in optimize_model() expands any pre-existing nodes instead.
    if caps.supports_skip_layer_norm:
        fuse.append(("SkipNorm", list(skip_norm_rules())))
        fuse.append(("SkipLayerNorm", list(skip_layer_norm_rules())))

    # --- Activation fusions (all roles, all dtypes) ---
    fuse.append(("GeluFusion", list(gelu_fusion_rules())))

    # --- Lowering passes ---
    # SkipLayerNorm/SimplifiedLayerNorm decomposition is handled by InlinePass
    # using registered ir.Function bodies — not rewrite rules.
    if not caps.supports_fused_rope:
        lower.append(("SeparateRoPE", list(separate_rope_rules())))
        lower.append(("UnpackQKV", list(unpack_qkv_rules())))

    return fuse, lower


# ---------------------------------------------------------------------------
# Main optimization entry point
# ---------------------------------------------------------------------------


def optimize_model(
    model: ir.Model,
    ep: str = "default",
    dtype: ir.DataType = ir.DataType.FLOAT,
    model_role: str = "decoder",
    trace: bool = False,
) -> None:
    """Apply EP-aware optimization passes to *model* in-place.

    Runs a five-stage pipeline:

    1. **Cleanup** — identity elimination, CSE, dead-code removal, constant
       folding, shape inference (EP-agnostic; always applied).
    2. **Fusion** — promote standard ops to EP-supported fused ops
       (e.g. GQA, SkipNorm, GeluFusion). Gated by ``(ep, dtype)`` and role.
    3. **Lowering** — decompose ops the EP cannot execute
       (e.g. SeparateRoPE for DML).
    4. **Fold** — final dead-node removal and constant folding.

    After fusion, if GQA was expected for ``(ep, dtype)`` but zero
    ``GroupQueryAttention`` nodes were produced while ``Attention`` nodes
    remain, a warning is emitted. This catches silent rule-match failures.

    Args:
        model: The ONNX IR model to optimize in-place.
        ep: Target execution provider. Must be registered in
            :data:`~mobius._execution_providers.ep_registry`.
        dtype: Model dtype for support-matrix lookups.
        model_role: Semantic role of this model component.
        trace: When ``True``, emit per-stage diagnostic logs at INFO level.

    Raises:
        ValueError: If *ep* is not a registered execution provider.
    """
    # Stage 1: Base cleanup (EP-agnostic — always applied).
    if trace:
        before_total = sum(_count_all_ops(model).values())
        logger.info("[EP Trace] Target: %s, dtype: %s, role: %s", ep, dtype, model_role)
        logger.info("[EP Trace] Stage 1: Cleanup (%d passes)", len(_DEFAULT_PASSES))

    cleanup_pass = ir.passes.PassManager(_DEFAULT_PASSES, steps=2)
    if flags.suppress_dedup_warning:
        with _suppress_dedup_empty_initializer_warnings():
            cleanup_pass(model)
    else:
        cleanup_pass(model)

    if trace:
        after_total = sum(_count_all_ops(model).values())
        logger.info(
            "[EP Trace]   Cleanup: %d → %d nodes (%+d)",
            before_total,
            after_total,
            after_total - before_total,
        )

    # Stage 2+3: Fusion + InlinePass + Lowering — gated by EP capabilities.
    caps = ep_registry.get(ep)
    if caps is None:
        raise ValueError(
            f"Unknown execution provider {ep!r}. Supported: {sorted(ep_registry)}"
        )

    fuse_stages, lower_stages = _get_optimization_passes(caps, dtype, model_role)

    # Register standard-ONNX ir.Function bodies for all known custom ops.
    # InlinePass below uses these to expand ops the EP cannot execute.
    register_function_bodies(model)

    # Build InlinePass criteria: expand custom ops this EP doesn't support.
    def _should_inline(func: ir.Function) -> bool:
        # onnx-standard EP: inline every function whose domain is not a standard
        # ONNX domain. This covers both the well-known ops below AND parametric
        # ops like CausalConvWithState and LinearAttention registered per-model.
        if caps.name == "onnx-standard" and func.domain not in _STANDARD_ONNX_DOMAINS:
            return True
        if func.domain == "com.microsoft" and func.name in (
            "SkipLayerNormalization",
            "SkipSimplifiedLayerNormalization",
        ):
            return not caps.supports_skip_layer_norm
        if func.domain == "com.microsoft" and func.name == "PackedMultiHeadAttention":
            return not caps.supports_packed_multi_head_attention
        return False

    inline_pass = common_passes.InlinePass(criteria=_should_inline)

    trace_entries: list[_TraceEntry] = []

    if trace:
        logger.info("[EP Trace] Stage 2: Fusion (%d rule groups)", len(fuse_stages))
        for name, rules_or_pass in fuse_stages:
            before = _count_all_ops(model)
            _apply_stage(model, rules_or_pass)
            after = _count_all_ops(model)
            entry = _make_trace_entry(name, before, after)
            trace_entries.append(entry)
            _log_trace_entry(entry)

        logger.info("[EP Trace] Stage 2b: InlinePass (expand unsupported custom ops)")
        before = _count_all_ops(model)
        inline_pass(model)
        after = _count_all_ops(model)
        trace_entries.append(_make_trace_entry("InlinePass", before, after))
        _log_trace_entry(trace_entries[-1])

        logger.info(
            "[EP Trace] Stage 3: Lowering (%d rule groups for %s)", len(lower_stages), ep
        )
        for name, rules_or_pass in lower_stages:
            before = _count_all_ops(model)
            _apply_stage(model, rules_or_pass)
            after = _count_all_ops(model)
            entry = _make_trace_entry(name, before, after)
            trace_entries.append(entry)
            _log_trace_entry(entry)
    else:
        # Batch all rewrite rules for efficiency; apply IR passes separately.
        all_fuse_rules = [r for _, rp in fuse_stages if isinstance(rp, list) for r in rp]
        all_lower_rules = [r for _, rp in lower_stages if isinstance(rp, list) for r in rp]
        lower_ir_passes = [(n, rp) for n, rp in lower_stages if not isinstance(rp, list)]

        if all_fuse_rules:
            rewrite(model, pattern_rewrite_rules=all_fuse_rules)

        # Expand unsupported custom ops via registered ir.Function bodies.
        inline_pass(model)

        if all_lower_rules:
            rewrite(model, pattern_rewrite_rules=all_lower_rules)

        for _, ir_pass in lower_ir_passes:
            ir_pass(model)

    # Stage 4: Final dead-node removal, constant folding, and dead input
    # cleanup after rewrites.
    if trace:
        before_fold = sum(_count_all_ops(model).values())
        logger.info("[EP Trace] Stage 4: Constant folding")

    fold_pass = ir.passes.PassManager(
        [
            common_passes.RemoveUnusedNodesPass(),
            # CSE after lowering collapses duplicate Gather(cos/sin, position_ids)
            # nodes introduced by SeparateRoPE in Stage 3 (2 per layer → 2 total).
            common_passes.CommonSubexpressionEliminationPass(),
            onnxscript.optimizer._constant_folding.FoldConstantsPass(
                shape_inference=False,
                input_size_limit=8192,
                output_size_limit=_FOLD_OUTPUT_SIZE_LIMIT,
            ),
            # Remove graph inputs whose consumers were all eliminated by
            # fusion (e.g. position_ids when GQA absorbs RoPE).
            RemoveDeadGraphInputsPass(),
        ]
    )
    fold_pass(model)

    if trace:
        after_fold = sum(_count_all_ops(model).values())
        logger.info(
            "[EP Trace]   Fold: %d → %d nodes (%+d)",
            before_fold,
            after_fold,
            after_fold - before_fold,
        )
        logger.info(
            "[EP Trace] Summary: %d nodes total, ep=%s, dtype=%s",
            after_fold,
            ep,
            dtype.name,
        )

    # Fusion assertion: warn if GQA was expected but no GQA nodes produced.
    if model_role == "decoder" and dtype in caps.gqa_dtypes:
        gqa_count = _count_ops(model, "GroupQueryAttention")
        attn_count = _count_ops(model, "Attention")
        if gqa_count == 0 and attn_count > 0:
            warnings.warn(
                f"GQA fusion expected for ep={ep!r}/dtype={dtype} but found "
                f"0 GroupQueryAttention and {attn_count} Attention nodes. "
                f"The model may run slower than expected on this EP. "
                f"Check that the attention pattern matches the GQA rewrite rule.",
                stacklevel=4,
            )


def fold_initializers_after_weights(model: ir.Model) -> None:
    """Fold weight ``Transpose`` and ``Concat`` nodes after weights are loaded.

    Runs :class:`~onnx_ir.passes.common.LiftConstantsToInitializersPass`,
    :class:`~mobius._passes.FoldConcatInitializersPass`,
    :class:`~mobius._passes.FoldTransposedInitializerPass`, and
    :class:`~onnx_ir.passes.common.RemoveUnusedNodesPass` in order.
    FoldConcat must precede FoldTranspose so that packed QKV initializers are
    visible before the Transpose fold runs.
    """
    ir.passes.PassManager(
        [
            common_passes.LiftConstantsToInitializersPass(),
            FoldConcatInitializersPass(),
            FoldTransposedInitializerPass(),
            common_passes.RemoveUnusedNodesPass(),
        ]
    )(model)
