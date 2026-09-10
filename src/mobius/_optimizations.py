# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model finalization passes.

Exposes :func:`optimize_model`, which applies exporter-owned cleanup and
function inlining to an ONNX IR model. Graph fusions and EP-specific lowerings
are intentionally not performed here; those transformations are owned by
Olive after export.

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
    "StripDebugMetadataPass",
    "SymbolicShapeInferencePass",
    "strip_debug_metadata",
    "DEBUG_METADATA_PREFIXES",
    "DEBUG_METADATA_KEYS",
    "FUNCTIONAL_METADATA_PREFIX",
    # Diagnostic helpers
    "_count_all_ops",
    "_count_ops",
]

import contextlib
import logging
import warnings

import onnx_ir as ir
import onnx_shape_inference
import onnxscript.optimizer._constant_folding
from onnx_ir.passes import common as common_passes

from mobius._execution_providers import ep_registry
from mobius._flags import flags
from mobius._passes import (
    FoldConcatInitializersPass,
    FoldTransposedInitializerPass,
    Fp8KvCachePass,
    RemoveDeadGraphInputsPass,
)
from mobius.functions import register_function_bodies

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


#: Metadata written by the build toolchain to describe *where a node came from*.
#: Useful when debugging a graph, carried into every serialized model, and read
#: by nothing at inference time.
#:
#: Prefixes are matched with ``startswith``; bare names are matched exactly.
#: Keeping this explicit rather than "strip everything that is not ours" means a
#: new provenance key from a toolchain upgrade is *kept* until someone lists it,
#: which is the safe direction to fail: a graph that is larger than it needs to
#: be, rather than one missing metadata a runtime depended on.
DEBUG_METADATA_PREFIXES: tuple[str, ...] = (
    "pkg.onnxscript.",
    "pkg.onnx_shape_inference.",
)
DEBUG_METADATA_KEYS: tuple[str, ...] = ("namespace",)

#: Everything mobius itself writes for a runtime to read back is under this
#: prefix — pipeline component presence, optional-input presence, generation
#: policy contracts, conv-cache spatial scales, batch-padding sensitivity. The
#: strip pass must never touch it; ``StripDebugMetadataPass`` asserts as much.
FUNCTIONAL_METADATA_PREFIX = "mobius."


def _is_debug_metadata(key: str) -> bool:
    return key.startswith(DEBUG_METADATA_PREFIXES) or key in DEBUG_METADATA_KEYS


class StripDebugMetadataPass(ir.passes.InPlacePass):
    """Remove build-time provenance metadata from every node and value.

    ``onnxscript`` annotates each node with the ``nn.Module`` path that produced
    it (``namespace``, ``pkg.onnxscript.class_hierarchy``,
    ``pkg.onnxscript.name_scopes``), which graph transform created it
    (``pkg.onnxscript.rewriter.rule_name``), and symbolic-inference internals on
    values (``pkg.onnx_shape_inference.sym_data``). That is exactly what you want
    when reading a graph in Netron and tracing a node back to the source module,
    and it is dead weight in a shipped artifact: on the tiny test models it is
    **36-39% of the serialized graph** (weights excluded), and it scales with
    node count, so the fraction holds for real models.

    Deliberately *not* driven by "delete anything not prefixed ``mobius.``".
    Metadata this pass does not recognise is preserved, so an unrecognised key
    costs bytes rather than breaking a runtime that reads it.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        removed = 0

        def strip(props) -> None:
            nonlocal removed
            for key in [key for key in props if _is_debug_metadata(key)]:
                assert not key.startswith(FUNCTIONAL_METADATA_PREFIX), (
                    f"{key!r} is both functional and debug metadata; the two "
                    "namespaces must stay disjoint"
                )
                del props[key]
                removed += 1

        seen: set[int] = set()
        for node in model.graph.all_nodes():
            strip(node.metadata_props)
            for value in (*node.inputs, *node.outputs):
                if value is not None and id(value) not in seen:
                    seen.add(id(value))
                    strip(value.metadata_props)

        if removed:
            logger.debug("Stripped %d debug metadata entries", removed)
        return ir.passes.PassResult(model, modified=bool(removed))


def strip_debug_metadata(model: ir.Model) -> ir.Model:
    """Run :class:`StripDebugMetadataPass` over ``model`` in place."""
    return StripDebugMetadataPass()(model).model


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
# Main finalization entry point
# ---------------------------------------------------------------------------


def optimize_model(
    model: ir.Model,
    ep: str = "default",
    dtype: ir.DataType = ir.DataType.FLOAT,
    model_role: str = "decoder",
    trace: bool = False,
    fp8_kv_cache: bool = False,
    kv_cache_scales: dict[int, tuple[float, float]] | None = None,
) -> None:
    """Apply exporter-owned finalization passes to *model* in-place.

    Mobius performs graph cleanup, expands registered local functions that the
    selected EP cannot consume, and applies FP8 KV-cache typing when requested.
    Olive owns all post-export fusions and EP-specific graph lowerings.

    Args:
        model: The ONNX IR model to optimize in-place.
        ep: Target execution provider. Must be registered in
            :data:`~mobius._execution_providers.ep_registry`.
        dtype: Model dtype for FP8 KV-cache capability checks.
        model_role: Semantic role of this model component.
        trace: When ``True``, emit per-stage diagnostic logs at INFO level.
        fp8_kv_cache: When ``True``, convert decoder
            ``GroupQueryAttention`` KV caches to ``FLOAT8E4M3FN`` (per-tensor
            E4M3) via :class:`~mobius._passes.Fp8KvCachePass`. Only applied
            when the exported graph contains GroupQueryAttention,
            ``model_role == "decoder"``, and the EP ships the FP8 GQA kernel
            (``caps.supports_fp8_kv_cache`` — currently CUDA only); otherwise a
            warning is emitted and the request is ignored.
        kv_cache_scales: Optional ``layer_id -> (k_scale, v_scale)`` map of
            per-tensor FP8 scales (from offline calibration). Only used when
            ``fp8_kv_cache`` is ``True``; layers absent from the map use a unit
            scale of ``1.0``.

    Raises:
        ValueError: If *ep* is not a registered execution provider.
    """
    caps = ep_registry.get(ep)
    if caps is None:
        raise ValueError(
            f"Unknown execution provider {ep!r}. Supported: {sorted(ep_registry)}"
        )

    # Stage 1: Base cleanup.
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

    # Stage 2: Expand registered local functions that the target cannot consume.
    register_function_bodies(model)

    def _should_inline(func: ir.Function) -> bool:
        if caps.name == "onnx-standard" and func.domain not in _STANDARD_ONNX_DOMAINS:
            return True
        if (
            caps.name == "cuda"
            and func.domain == "com.microsoft"
            and func.name == "CausalConvWithState"
        ):
            return True
        if func.domain == "com.microsoft" and func.name in (
            "SkipLayerNormalization",
            "SkipSimplifiedLayerNormalization",
        ):
            return not caps.supports_skip_layer_norm
        if func.domain == "com.microsoft" and func.name == "PackedMultiHeadAttention":
            return not caps.supports_packed_multi_head_attention
        if func.domain == "com.microsoft" and func.name == "MatMulNBits":
            return not caps.supports_matmul_nbits
        return False

    inline_pass = common_passes.InlinePass(criteria=_should_inline)
    if trace:
        before_inline = sum(_count_all_ops(model).values())
        logger.info("[EP Trace] Stage 2: Inline unsupported local functions")
    inline_pass(model)
    if trace:
        after_inline = sum(_count_all_ops(model).values())
        logger.info(
            "[EP Trace]   Inline: %d → %d nodes (%+d)",
            before_inline,
            after_inline,
            after_inline - before_inline,
        )

    # Stage 3: Final dead-node removal, constant folding, and dead input cleanup.
    if trace:
        before_fold = sum(_count_all_ops(model).values())
        logger.info("[EP Trace] Stage 3: Constant folding")

    fold_pass = ir.passes.PassManager(
        [
            common_passes.RemoveUnusedNodesPass(),
            common_passes.CommonSubexpressionEliminationPass(),
            onnxscript.optimizer._constant_folding.FoldConstantsPass(
                shape_inference=False,
                input_size_limit=8192,
                output_size_limit=_FOLD_OUTPUT_SIZE_LIMIT,
            ),
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

    # FP8 KV cache: convert GroupQueryAttention KV caches to FLOAT8E4M3FN.
    if fp8_kv_cache:
        gqa_expected = model_role == "decoder" and dtype in caps.gqa_dtypes
        if not gqa_expected or not caps.supports_fp8_kv_cache:
            warnings.warn(
                f"fp8_kv_cache=True was requested but the FP8 GQA KV-cache kernel "
                f"is not available for ep={ep!r}/dtype={dtype}/role={model_role!r}. "
                f"FP8 KV cache requires a GroupQueryAttention decoder on an EP with "
                f"the FP8 kernel (currently only --execution-provider cuda with an "
                f"fp16/bf16 dtype). Ignoring the request.",
                stacklevel=4,
            )
        else:
            fp8_pass = Fp8KvCachePass(kv_cache_scales)
            fp8_pass(model)
            if fp8_pass.converted == 0:
                raise ValueError(
                    "fp8_kv_cache=True was requested but the exported graph "
                    "exposes no GroupQueryAttention KV cache to convert. FP8 KV "
                    "storage needs an attention operator with k_scale/v_scale "
                    "inputs to dequantize the cache on read; a static-cache "
                    "export scatters into fixed buffers read by ai.onnx "
                    "Attention, which has no such inputs. Build without "
                    "--features fp8-kv-cache, or without --features static-cache."
                )

    # Passes may insert producers after existing consumers.
    model.graph.sort()


def fold_initializers_after_weights(model: ir.Model) -> None:
    """Fold weight ``Transpose`` and ``Concat`` nodes after weights are loaded.

    Runs :class:`~onnx_ir.passes.common.LiftConstantsToInitializersPass`,
    :class:`~mobius._passes.FoldConcatInitializersPass`,
    :class:`~mobius._passes.FoldTransposedInitializerPass`, and
    :class:`~onnx_ir.passes.common.RemoveUnusedNodesPass` in order.
    FoldConcat must precede FoldTranspose so concatenated initializers are
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
