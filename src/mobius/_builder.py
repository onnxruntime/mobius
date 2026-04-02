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


def _get_optimization_passes(
    ep: str,
    dtype: ir.DataType,
    model_role: str = "decoder",
) -> tuple[list, list]:
    """Return ``(fuse_rules, lower_rules)`` for the given EP, dtype, and role.

    Only fusions the target EP supports are returned. Lowering passes
    decompose ops the EP does not support. Phase 2 stubs are left as
    comments for the next implementation phase.

    Args:
        ep: Target execution provider (``"cpu"``, ``"cuda"``, ``"dml"``,
            ``"webgpu"``, ``"trt-rtx"``).
        dtype: Model dtype for support-matrix lookups.
        model_role: Semantic role of this model component (``"decoder"``,
            ``"vision"``, ``"embedding"``, ``"encoder"``). GQA fusion is
            only applied to the decoder role.

    Returns:
        ``(fuse_rules, lower_rules)`` — each a flat list of rewrite-rule
        instances suitable for passing to ``onnxscript.rewriter.rewrite()``.
    """
    from mobius.rewrite_rules import (
        cast_int64_to_int32_rules,
        decompose_skip_layer_norm_rules,
        eliminate_shape_rules,
        gelu_fusion_rules,
        group_query_attention_rules,
        separate_rope_rules,
        skip_layer_norm_rules,
        skip_norm_rules,
        unpack_qkv_rules,
    )

    fuse: list = []
    lower: list = []

    # --- Attention fusion (decoder only) ---
    if model_role == "decoder" and (ep, dtype) in _GQA_SUPPORT:
        fuse.extend(group_query_attention_rules())

    # --- Normalization fusions (all roles, all dtypes) ---
    # TRT-RTX decomposes SkipNorm/SkipLayerNorm rather than fusing them.
    if ep != "trt-rtx":
        fuse.extend(skip_norm_rules())
        fuse.extend(skip_layer_norm_rules())

    # --- Activation fusions (all roles, all dtypes) ---
    fuse.extend(gelu_fusion_rules())

    # --- TRT-RTX lowering: decompose fused skip-norm ops ---
    if ep == "trt-rtx":
        lower.extend(decompose_skip_layer_norm_rules())

    if ep == "dml":
        lower.extend(separate_rope_rules())  # BP-6: decompose fused RoPE
        lower.extend(unpack_qkv_rules())  # BP-7: split packed QKV
    elif ep == "webgpu":
        lower.extend(eliminate_shape_rules())  # BP-13: Shape → ReduceSum+ReduceMax
        lower.extend(cast_int64_to_int32_rules())  # BP-12: INT64 → INT32 for Gather indices

    return fuse, lower


def _optimize(
    model: ir.Model,
    ep: str = "cpu",
    dtype: ir.DataType = ir.DataType.FLOAT,
    model_role: str = "decoder",
) -> None:
    """Apply EP-aware optimization passes to a model in-place.

    Runs a four-stage pipeline:

    1. **Cleanup** — identity elimination, CSE, dead-code removal, constant
       folding, shape inference (EP-agnostic; always applied).
    2. **Fusion** — promote standard ops to EP-supported fused ops
       (e.g. GQA, SkipNorm, BiasGelu). Gated by ``(ep, dtype)`` support
       matrix and ``model_role``.
    3. **Lowering** — decompose ops the EP does not support
       (Phase 2 stubs; currently empty for all EPs).
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
    """
    # Stage 1: Base cleanup (EP-agnostic — always applied).
    pass_ = ir.passes.PassManager(_DEFAULT_PASSES, steps=2)
    if flags.suppress_dedup_warning:
        with _suppress_dedup_empty_initializer_warnings():
            pass_(model)
    else:
        pass_(model)

    # Stage 2: Fusion — gated by EP support matrix and model_role.
    # Stage 3: Lowering — Phase 2 stubs; currently empty for all EPs.
    fuse_rules, lower_rules = _get_optimization_passes(ep, dtype, model_role)

    if fuse_rules:
        from onnxscript.rewriter import rewrite

        rewrite(model, pattern_rewrite_rules=fuse_rules)

    if lower_rules:
        from onnxscript.rewriter import rewrite

        rewrite(model, pattern_rewrite_rules=lower_rules)

    # Stage 4: Final dead-node removal and constant folding after rewrites.
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

    # Pass use_concrete_dims only to tasks that accept it (CausalLMTask and
    # future tasks). Other tasks are called without the kwarg.
    import inspect

    _build_sig = inspect.signature(resolved_task.build)
    if "use_concrete_dims" in _build_sig.parameters:
        pkg = resolved_task.build(module, config, use_concrete_dims=use_concrete_dims)
    else:
        pkg = resolved_task.build(module, config)

    for name, model in pkg.items():
        role = _MODEL_ROLE_MAP.get(name, "decoder")
        _optimize(model, ep=execution_provider, dtype=dtype, model_role=role)
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
    import dataclasses

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
    pkg = build_from_module(model_module, config, task, execution_provider=execution_provider)

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
