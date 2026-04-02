# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Model building API.

This module provides the core functions for constructing ONNX models from
``onnxscript.nn.Module`` instances:

- :func:`build_from_module` — Build from a module instance and config.
- :func:`build` — Build from a HuggingFace model ID.
- :func:`resolve_dtype` — Resolve dtype strings to ``ir.DataType``.
"""

from __future__ import annotations

__all__ = [
    "DTYPE_MAP",
    "build",
    "build_from_module",
    "resolve_dtype",
]

import contextlib
import dataclasses
import logging
import warnings

import onnx_ir as ir
import onnx_shape_inference
import onnxscript.optimizer._constant_folding  # TODO(justinchuby): Expose the FoldConstantsPass from onnxscript
import torch
from onnx_ir import tensor_adapters
from onnx_ir.passes import common as common_passes
from onnxscript import nn

from mobius._configs import (
    BaseModelConfig,
)
from mobius._flags import flags
from mobius._model_package import ModelPackage
from mobius._registry import registry
from mobius._weight_loading import _download_weights
from mobius.tasks import ModelTask, get_task

logger = logging.getLogger(__name__)


class _SuppressNoConstValueWarning(logging.Filter):
    """Filter out 'has no constant value' warnings from initializer dedup.

    Mobius runs optimization passes before weight loading, so weight
    initializers intentionally have no const_value at that point.
    Other warnings from the pass (e.g. hash collisions) are preserved.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "has no constant value" not in record.getMessage()


@contextlib.contextmanager
def _suppress_dedup_empty_initializer_warnings():
    """Temporarily suppress 'has no constant value' dedup warnings.

    Scoped to the optimization pass invocation only — the filter is
    removed when the context exits so it doesn't affect other code.
    """
    dedup_logger = logging.getLogger("onnx_ir.passes.common.initializer_deduplication")
    log_filter = _SuppressNoConstValueWarning()
    dedup_logger.addFilter(log_filter)
    try:
        yield
    finally:
        dedup_logger.removeFilter(log_filter)


# ---------------------------------------------------------------------------
# Public build API
# ---------------------------------------------------------------------------


class SymbolicShapeInferencePass(ir.passes.InPlacePass):
    """ONNX IR pass that applies symbolic shape inference to all nodes."""

    def __init__(self, policy: onnx_shape_inference.ShapeMergePolicy = "refine"):
        super().__init__()
        self.policy = policy

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        onnx_shape_inference.infer_symbolic_shapes(model, policy=self.policy)
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


_DEFAULT_PASSES = [
    common_passes.IdentityEliminationPass(),
    common_passes.LiftConstantsToInitializersPass(),
    common_passes.DeduplicateInitializersPass(),
    common_passes.CommonSubexpressionEliminationPass(),
    common_passes.RemoveUnusedNodesPass(),
    common_passes.RemoveUnusedOpsetsPass(),
    SymbolicShapeInferencePass(),
    onnxscript.optimizer._constant_folding.FoldConstantsPass(
        shape_inference=False, input_size_limit=8192, output_size_limit=512 * 512
    ),
    CleanupMetadataPass(),
]


# Mapping of short dtype names to ONNX IR dtypes
DTYPE_MAP: dict[str, ir.DataType] = {
    "f32": ir.DataType.FLOAT,
    "float32": ir.DataType.FLOAT,
    "f16": ir.DataType.FLOAT16,
    "float16": ir.DataType.FLOAT16,
    "bf16": ir.DataType.BFLOAT16,
    "bfloat16": ir.DataType.BFLOAT16,
}


# ---------------------------------------------------------------------------
# EP capability matrices
# ---------------------------------------------------------------------------

# EP+dtype combinations where GroupQueryAttention fusion is supported.
_GQA_SUPPORT: frozenset[tuple[str, ir.DataType]] = frozenset(
    [
        ("cpu", ir.DataType.FLOAT),
        ("cuda", ir.DataType.FLOAT16),
        ("cuda", ir.DataType.BFLOAT16),
        ("dml", ir.DataType.FLOAT16),
        ("webgpu", ir.DataType.FLOAT),
        ("webgpu", ir.DataType.FLOAT16),
        ("trt-rtx", ir.DataType.FLOAT16),
        ("trt-rtx", ir.DataType.BFLOAT16),
    ]
)

# EP+dtype combinations where PackedAttention fusion is supported.
# Used by Phase 2 PackedAttention rule activation.
_PACKED_ATTN_SUPPORT: frozenset[tuple[str, ir.DataType]] = frozenset(
    [
        ("cpu", ir.DataType.FLOAT),
        ("cuda", ir.DataType.FLOAT),
        ("cuda", ir.DataType.FLOAT16),
        ("cuda", ir.DataType.BFLOAT16),
        ("dml", ir.DataType.FLOAT),
        ("dml", ir.DataType.FLOAT16),
        ("webgpu", ir.DataType.FLOAT),
        ("webgpu", ir.DataType.FLOAT16),
        ("trt-rtx", ir.DataType.FLOAT),
        ("trt-rtx", ir.DataType.FLOAT16),
        ("trt-rtx", ir.DataType.BFLOAT16),
    ]
)

# Map ModelPackage entry names to semantic model roles.
# GQA fusion is only applied to "decoder" role models.
_MODEL_ROLE_MAP: dict[str, str] = {
    "model": "decoder",
    "decoder": "decoder",
    "vision": "vision",
    "embedding": "embedding",
    "encoder": "encoder",
}


def _count_ops(model: ir.Model, op_type: str) -> int:
    """Count nodes of a given op_type in all model graph nodes."""
    return sum(1 for node in model.graph.all_nodes() if node.op_type == op_type)


def _count_all_ops(model: ir.Model) -> dict[str, int]:
    """Count all op types present in the model graph (including subgraphs)."""
    counts: dict[str, int] = {}
    for node in model.graph.all_nodes():
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


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
        from onnxscript.rewriter import rewrite

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


def _get_optimization_passes(
    ep: str,
    dtype: ir.DataType,
    model_role: str = "decoder",
) -> tuple[list[tuple[str, list]], list[tuple[str, list | ir.passes.InPlacePass]]]:
    """Return ``(fuse_stages, lower_stages)`` for the given EP, dtype, and role.

    Each stage is a ``(name, rules_or_pass)`` pair where ``rules_or_pass``
    is either a list of onnxscript rewrite rules or an
    :class:`ir.passes.InPlacePass` instance (for IR-level passes that cannot
    be expressed as pattern rewrite rules).

    Only fusions the target EP supports are returned. Lowering stages
    decompose ops the EP does not support.

    Args:
        ep: Target execution provider (``"cpu"``, ``"cuda"``, ``"dml"``,
            ``"webgpu"``, ``"trt-rtx"``).
        dtype: Model dtype for support-matrix lookups.
        model_role: Semantic role of this model component (``"decoder"``,
            ``"vision"``, ``"embedding"``, ``"encoder"``). GQA fusion is
            only applied to the decoder role.

    Returns:
        ``(fuse_stages, lower_stages)`` — each a list of ``(name, payload)``
        tuples. The payload is either a list of rewrite rules or an
        :class:`ir.passes.InPlacePass`.
    """
    from mobius.rewrite_rules import (
        cast_int64_to_int32_rules,
        decompose_if_pass,
        decompose_skip_layer_norm_rules,
        eliminate_shape_rules,
        gelu_fusion_rules,
        group_query_attention_rules,
        separate_rope_rules,
        skip_layer_norm_rules,
        skip_norm_rules,
        unpack_qkv_rules,
    )

    fuse: list[tuple[str, list]] = []
    lower: list[tuple[str, list | ir.passes.InPlacePass]] = []

    # --- Attention fusion (decoder only) ---
    if model_role == "decoder" and (ep, dtype) in _GQA_SUPPORT:
        fuse.append(("GQAFusion", list(group_query_attention_rules())))

    # --- Normalization fusions (all roles, all dtypes) ---
    # TRT-RTX decomposes SkipNorm/SkipLayerNorm rather than fusing them.
    if ep != "trt-rtx":
        fuse.append(("SkipNorm", list(skip_norm_rules())))
        fuse.append(("SkipLayerNorm", list(skip_layer_norm_rules())))

    # --- Activation fusions (all roles, all dtypes) ---
    fuse.append(("BiasGelu", list(gelu_fusion_rules())))

    # --- TRT-RTX lowering: decompose fused skip-norm ops ---
    if ep == "trt-rtx":
        lower.append(("DecomposeSkipLayerNorm", list(decompose_skip_layer_norm_rules())))

    if ep == "dml":
        lower.append(
            ("SeparateRoPE", list(separate_rope_rules()))
        )  # BP-6: decompose fused RoPE
        lower.append(("UnpackQKV", list(unpack_qkv_rules())))  # BP-7: split packed QKV
        lower.append(("DecomposeIf", decompose_if_pass()))  # BP-10: If→Where for DML
    elif ep == "webgpu":
        lower.append(("EliminateShape", list(eliminate_shape_rules())))  # BP-13
        lower.append(("CastInt64ToInt32", list(cast_int64_to_int32_rules())))  # BP-12
        lower.append(("DecomposeIf", decompose_if_pass()))  # BP-10: If→Where for WebGPU

    return fuse, lower


def _optimize(
    model: ir.Model,
    ep: str = "cpu",
    dtype: ir.DataType = ir.DataType.FLOAT,
    model_role: str = "decoder",
    trace: bool = False,
) -> None:
    """Apply EP-aware optimization passes to a model in-place.

    Runs a four-stage pipeline:

    1. **Cleanup** — identity elimination, CSE, dead-code removal, constant
       folding, shape inference (EP-agnostic; always applied).
    2. **Fusion** — promote standard ops to EP-supported fused ops
       (e.g. GQA, SkipNorm, BiasGelu). Gated by ``(ep, dtype)`` support
       matrix and ``model_role``.
    3. **Lowering** — decompose ops the EP does not support
       (e.g. SeparateRoPE, UnpackQKV, DecomposeIf for DML/WebGPU).
    4. **Fold** — final dead-node removal and constant folding after
       rewrites.

    After the fusion stage, if GQA was expected for this ``(ep, dtype)``
    combination but zero ``GroupQueryAttention`` nodes were produced while
    ``Attention`` nodes remain, a warning is emitted. This helps catch
    silent rule-match failures.

    Args:
        model: The ONNX IR model to optimize in-place.
        ep: Target execution provider.
        dtype: Model dtype for support-matrix lookups.
        model_role: Semantic role of this model component.
        trace: When ``True``, emit per-stage diagnostic logs at INFO level
            showing which rules matched, how many nodes were added/removed,
            and a final summary table. Useful for debugging EP configuration.
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

    # Stage 2: Fusion / Stage 3: Lowering — gated by EP support matrix.
    fuse_stages, lower_stages = _get_optimization_passes(ep, dtype, model_role)

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
            from onnxscript.rewriter import rewrite

            rewrite(model, pattern_rewrite_rules=all_fuse_rules)

        if all_lower_rules:
            from onnxscript.rewriter import rewrite

            rewrite(model, pattern_rewrite_rules=all_lower_rules)

        for _, ir_pass in lower_ir_passes:
            ir_pass(model)

    # Stage 4: Final dead-node removal and constant folding after rewrites.
    if trace:
        before_fold = sum(_count_all_ops(model).values())
        logger.info("[EP Trace] Stage 4: Constant folding")

    fold_pass = ir.passes.PassManager(
        [
            common_passes.RemoveUnusedNodesPass(),
            onnxscript.optimizer._constant_folding.FoldConstantsPass(
                shape_inference=False,
                input_size_limit=8192,
                output_size_limit=512 * 512,
            ),
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
        _log_trace_summary(trace_entries)

    # Fusion assertion: warn if GQA was expected but no GQA nodes produced.
    # Only fires when Attention nodes are present — models with no attention
    # (Mamba, RWKV, etc.) are silently skipped.
    if model_role == "decoder" and (ep, dtype) in _GQA_SUPPORT:
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


def resolve_dtype(dtype: str | ir.DataType | None) -> ir.DataType | None:
    """Resolve a dtype string to an ``ir.DataType``.

    Args:
        dtype: A dtype string (e.g. ``"f16"``), an ``ir.DataType``, or ``None``.

    Returns:
        The resolved ``ir.DataType``, or ``None`` if *dtype* is ``None``.

    Raises:
        ValueError: If the dtype string is not recognised.
    """
    if dtype is None or isinstance(dtype, ir.DataType):
        return dtype
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{dtype}'. Available: {sorted(DTYPE_MAP)}")
    return DTYPE_MAP[dtype]


def _cast_module_dtype(module: nn.Module, dtype: ir.DataType) -> None:
    """Cast all FLOAT parameters in a module to the target dtype before graph building.

    This must be called **before** tracing/building the graph so that ONNX
    type inference propagates the correct dtype through all intermediate
    values. For parameters with precomputed ``const_value`` (e.g. RoPE
    caches), the underlying data is also cast.

    Only recasts parameters that are currently FLOAT (float32). Integer
    parameters and non-float types are left unchanged.
    """
    if dtype == ir.DataType.FLOAT:
        return
    torch_dtype = tensor_adapters.to_torch_dtype(dtype)
    for param in module.parameters():
        if param.dtype != ir.DataType.FLOAT:
            continue
        param.type = ir.TensorType(dtype)
        if param.const_value is not None:
            cast_tensor = torch.from_numpy(param.const_value.numpy()).to(torch_dtype)
            param.const_value = tensor_adapters.TorchTensor(cast_tensor)


def build_from_module(
    module: nn.Module,
    config: BaseModelConfig,
    task: str | ModelTask = "text-generation",
    *,
    execution_provider: str = "cpu",
    trace_optimization: bool = False,
) -> ModelPackage:
    """Build an ONNX :class:`ModelPackage` from a module instance and config.

    Use this when you have a custom :class:`onnxscript.nn.Module` or want
    full control over module construction. The model dtype is determined
    by ``config.dtype``.

    Args:
        module: An ``onnxscript.nn.Module`` instance. Its ``forward()``
            signature must be compatible with the task.
        config: Architecture configuration. The ``dtype`` field controls
            the target precision for model weights. If the config has a
            ``validate()`` method, it is called before graph construction
            to catch invalid fields early.
        task: The model task. Either a task name string
            (e.g. ``"text-generation"``) or a :class:`ModelTask` instance.
        execution_provider: Target execution provider for EP-aware
            optimizations. Accepted values: ``"cpu"`` (default),
            ``"cuda"``, ``"dml"``, ``"webgpu"``, ``"trt-rtx"``. Controls
            which fusion and lowering passes are applied during graph
            optimization.
        trace_optimization: When ``True``, log step-by-step diagnostic
            output at INFO level for each optimization stage, showing which
            rules matched and how many nodes were added/removed. Useful for
            debugging EP configuration and rule coverage.

    Returns:
        A :class:`ModelPackage` containing the built model(s).

    Raises:
        ValueError: If config validation fails (e.g. non-positive
            ``hidden_size``, ``num_attention_heads``, etc.).

    Example::

        from onnxscript import nn
        from mobius import ArchitectureConfig, build_from_module

        class MyModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                # ... define layers ...

            def forward(self, op, input_ids, attention_mask,
                        position_ids, past_key_values):
                # ... build graph ...
                return logits, present_key_values

        config = ArchitectureConfig(vocab_size=32000, hidden_size=4096, ...)
        pkg = build_from_module(MyModel(config), config)
    """
    if hasattr(config, "validate"):
        config.validate()
    dtype = getattr(config, "dtype", ir.DataType.FLOAT)
    _cast_module_dtype(module, dtype)
    resolved_task = get_task(task)

    # Translate EP string → structural flags for the task layer.
    # Tasks receive a boolean flag, never an EP string directly.
    use_concrete_dims = execution_provider == "webgpu"

    # Introspect the task's build() signature to pass use_concrete_dims only
    # to tasks that already accept it. This preserves backward compatibility
    # with the ~25 existing task classes that have not yet been updated to
    # accept the WebGPU structural flag — they continue to work unchanged.
    import inspect

    _build_sig = inspect.signature(resolved_task.build)
    if "use_concrete_dims" in _build_sig.parameters:
        pkg = resolved_task.build(module, config, use_concrete_dims=use_concrete_dims)
    else:
        pkg = resolved_task.build(module, config)

    for name, model in pkg.items():
        # Unknown model names default to decoder role for fusion purposes.
        role = _MODEL_ROLE_MAP.get(name, "decoder")
        _optimize(
            model,
            ep=execution_provider,
            dtype=dtype,
            model_role=role,
            trace=trace_optimization,
        )
    return pkg


def build(
    model_id: str,
    task: str | ModelTask | None = None,
    *,
    module_class: type[nn.Module] | None = None,
    dtype: str | ir.DataType | None = None,
    load_weights: bool = True,
    trust_remote_code: bool = False,
    execution_provider: str = "cpu",
    trace_optimization: bool = False,
) -> ModelPackage:
    """Build an ONNX :class:`ModelPackage` from a HuggingFace model ID.

    This is the main entry point for building models. It downloads the
    model configuration (and optionally weights) from HuggingFace Hub,
    selects the appropriate module class, and builds the ONNX graph(s).

    For single-component models (e.g. CausalLM), the package contains one
    ``"model"`` entry.  For multi-component models (e.g. encoder-decoder),
    it contains separate entries (``"encoder"``, ``"decoder"``).  For
    diffusers pipelines, each neural-network component gets its own entry.

    The model dtype is auto-detected from the HuggingFace config's
    ``torch_dtype`` field unless overridden by *dtype*.

    Args:
        model_id: HuggingFace model repository ID
            (e.g. ``"meta-llama/Llama-3-8B"``).
        task: The model task. Either a task name string
            (e.g. ``"text-generation"``) or a :class:`ModelTask` instance.
            When ``None``, the task is auto-detected from the model type.
        module_class: Custom module class to use instead of the auto-detected
            one. The class must accept an :class:`ArchitectureConfig` as its
            constructor argument and have a ``forward()`` method compatible
            with the task.
        dtype: Override the model dtype. Accepts short names (``"f32"``,
            ``"f16"``, ``"bf16"``) or :class:`ir.DataType` values.
            When ``None``, the dtype is auto-detected from the HuggingFace
            config.
        load_weights: Whether to download and apply weights from HuggingFace.
        trust_remote_code: Whether to trust remote code when loading the
            HuggingFace config.
        execution_provider: Target execution provider for EP-aware
            optimizations. Accepted values: ``"cpu"`` (default),
            ``"cuda"``, ``"dml"``, ``"webgpu"``, ``"trt-rtx"``. Controls
            which fusion and lowering passes are applied during graph
            optimization.
        trace_optimization: When ``True``, log step-by-step diagnostic
            output at INFO level for each optimization stage. See
            :func:`build_from_module` for details.

    Returns:
        A :class:`ModelPackage` containing the built model(s).

    Example::

        from mobius import build

        # Auto-detect architecture and task
        pkg = build("meta-llama/Llama-3-8B")

        # Save all components
        pkg.save("/output/llama/")

        # Access individual models
        model = pkg["model"]

    Example with static cache::

        from mobius import build, CausalLMTask

        task = CausalLMTask(static_cache=True, max_seq_len=2048)
        pkg = build("meta-llama/Llama-3-8B", task=task)
    """
    import transformers

    from mobius._config_resolver import (
        _config_from_hf,
        _default_task_for_model,
        _try_load_config_json,
    )
    from mobius._diffusers_builder import build_diffusers_pipeline

    try:
        hf_config = transformers.AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
    except (ValueError, KeyError, OSError):
        # AutoConfig failed — the model_type may not be in transformers,
        # or the HF config class has a bug (e.g. NemotronH with '-' pattern).
        # Try loading config.json directly if the model is in our registry.
        hf_config = _try_load_config_json(model_id)
        if hf_config is None or hf_config.model_type not in registry:
            # Not a model we support — try diffusers pipeline
            return build_diffusers_pipeline(
                model_id,
                dtype=dtype,
                load_weights=load_weights,
            )

    model_type = hf_config.model_type

    # Validate model/EP compatibility before graph construction
    from mobius._ep_validation import validate_ep_support

    validate_ep_support(model_type, execution_provider)

    parent_config = hf_config
    if hasattr(hf_config, "talker_config"):
        hf_config = hf_config.talker_config
    elif hasattr(hf_config, "thinker_config"):
        thinker = hf_config.thinker_config
        if hasattr(thinker, "text_config"):
            hf_config = thinker.text_config
    elif hasattr(hf_config, "decoder_config") and model_type == "qwen3_tts_tokenizer_12hz":
        # Codec tokenizer: use decoder_config as the primary config source
        dc = hf_config.decoder_config
        if isinstance(dc, dict):
            dc = type("DC", (), {**dc, "model_type": model_type})()
        else:
            dc.model_type = model_type
        hf_config = dc
    elif hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config

    if module_class is None:
        if model_type in registry:
            module_class = registry.get(model_type)
        else:
            from mobius._registry import _detect_fallback_registration

            fallback = _detect_fallback_registration(parent_config)
            if fallback is not None:
                module_class = fallback.module_class
                logger.warning(
                    "Model type '%s' is not registered. Auto-detected as compatible with %s.",
                    model_type,
                    module_class.__name__,
                )
                if task is None and fallback.task is not None:
                    task = fallback.task
            else:
                # No compatible fallback — raise the original error
                registry.get(model_type)  # raises KeyError

    config = _config_from_hf(hf_config, parent_config=parent_config, module_class=module_class)

    if dtype is not None:
        dtype = resolve_dtype(dtype)
        config = dataclasses.replace(config, dtype=dtype)

    if task is None:
        task = _default_task_for_model(model_type)

    model_module = module_class(config)
    pkg = build_from_module(
        model_module,
        config,
        task,
        execution_provider=execution_provider,
        trace_optimization=trace_optimization,
    )

    # Set graph names
    for name, model in pkg.items():
        model.graph.name = f"{model_id}/{name}"

    if load_weights:
        state_dict = _download_weights(model_id)
        if hasattr(model_module, "preprocess_weights"):
            state_dict = model_module.preprocess_weights(state_dict)
        prefix_map = getattr(model_module, "weight_prefix_map", None)
        pkg.apply_weights(state_dict, prefix_map=prefix_map)

    return pkg
