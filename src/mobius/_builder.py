# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Core ONNX graph construction.

This ecosystem-agnostic module builds :class:`~mobius._model_package.ModelPackage`
instances from ``onnxscript.nn.Module`` objects. Hugging Face Transformers and
Diffusers discovery, configuration, and weight loading live under
:mod:`mobius.integrations`.
"""

from __future__ import annotations

__all__ = [
    "DTYPE_MAP",
    "build_from_module",
    "resolve_dtype",
]

import logging

import onnx_ir as ir
import torch
from onnx_ir import tensor_adapters
from onnxscript import nn

from mobius._build_context import BuildContract, build_context
from mobius._component_manifest import ComponentManifest
from mobius._component_quantization import configure_component_quantization
from mobius._configs import BaseModelConfig
from mobius._execution_providers import ep_registry
from mobius._flags import flags
from mobius._model_package import ModelPackage
from mobius._optimizations import SymbolicShapeInferencePass
from mobius.functions import register_function_bodies
from mobius.tasks import ModelTask, get_task

logger = logging.getLogger(__name__)


DTYPE_MAP: dict[str, ir.DataType] = {
    "f32": ir.DataType.FLOAT,
    "float32": ir.DataType.FLOAT,
    "f16": ir.DataType.FLOAT16,
    "float16": ir.DataType.FLOAT16,
    "bf16": ir.DataType.BFLOAT16,
    "bfloat16": ir.DataType.BFLOAT16,
}


def resolve_dtype(dtype: str | ir.DataType | None) -> ir.DataType | None:
    """Resolve a dtype string to an ``ir.DataType``."""
    if dtype is None or isinstance(dtype, ir.DataType):
        return dtype
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{dtype}'. Available: {sorted(DTYPE_MAP)}")
    return DTYPE_MAP[dtype]


def _cast_module_dtype(module: nn.Module, dtype: ir.DataType) -> None:
    """Cast FLOAT module parameters to *dtype* before graph construction."""
    if dtype == ir.DataType.FLOAT:
        return
    torch_dtype = tensor_adapters.to_torch_dtype(dtype)
    for param in module.parameters():
        if param.dtype != ir.DataType.FLOAT or getattr(param, "_keep_float32", False):
            continue
        param.type = ir.TensorType(dtype)
        if param.const_value is not None:
            cast_tensor = torch.from_numpy(param.const_value.numpy()).to(torch_dtype)
            param.const_value = tensor_adapters.TorchTensor(cast_tensor)


def _enable_prefill_prefix_pruning_task(task: str | ModelTask) -> str | ModelTask:
    """Return a task equivalent to *task* with prefill-prefix pruning enabled."""
    from mobius.tasks import (
        CausalLMTask,
        Gemma4Task,
        Gemma4TextCausalLMTask,
        HybridCausalLMTask,
    )

    if task == "text-generation":
        return CausalLMTask(prune_prefill_prefix=True)
    if task == "hybrid-text-generation":
        return HybridCausalLMTask(prune_prefill_prefix=True)
    if task == "gemma4-text-generation":
        return Gemma4TextCausalLMTask(prune_prefill_prefix=True)
    if task == "gemma4":
        return Gemma4Task(prune_prefill_prefix=True)
    if isinstance(task, CausalLMTask):
        return CausalLMTask(
            static_cache=getattr(task, "_static_cache", False),
            max_seq_len=getattr(task, "_max_seq_len", None),
            prune_prefill_prefix=True,
        )
    if isinstance(task, HybridCausalLMTask):
        return HybridCausalLMTask(prune_prefill_prefix=True)
    if isinstance(task, Gemma4TextCausalLMTask):
        return Gemma4TextCausalLMTask(
            static_cache=getattr(task, "_static_cache", False),
            max_seq_len=getattr(task, "_max_seq_len", None),
            prune_prefill_prefix=True,
        )
    if isinstance(task, Gemma4Task):
        return Gemma4Task(
            static_cache=getattr(task, "_static_cache", False),
            max_seq_len=getattr(task, "_max_seq_len", None),
            prune_prefill_prefix=True,
        )
    raise ValueError(
        "prune_prefill_prefix=True is only supported for text-generation, "
        "hybrid-text-generation, gemma4-text-generation, and gemma4 tasks."
    )


def build_from_module(
    module: nn.Module,
    config: BaseModelConfig,
    task: str | ModelTask = "text-generation",
    *,
    execution_provider: str = "default",
    device: str | None = None,
    trace_optimization: bool = False,
    fp8_kv_cache: bool = False,
    kv_cache_scales: dict[int, tuple[float, float]] | None = None,
    prune_prefill_prefix: bool = False,
    component_manifest: ComponentManifest | None = None,
) -> ModelPackage:
    """Build an ONNX :class:`ModelPackage` from a module instance and config.

    Args:
        module: An ``onnxscript.nn.Module`` whose ``forward`` signature is
            compatible with *task*.
        config: Architecture configuration. Its ``dtype`` controls model
            precision and its ``validate`` method runs before build.
        task: Task name or :class:`ModelTask` instance.
        execution_provider: Target whose structural build contract is applied.
            Mobius does not apply graph rewrites based on this value.
        device: Optional target device recorded in the build contract.
        trace_optimization: Deprecated. Mobius export does not run graph rewrites.
        fp8_kv_cache: Deprecated. Apply FP8 KV-cache rewrites downstream.
        kv_cache_scales: Deprecated. Apply FP8 KV-cache rewrites downstream.
        prune_prefill_prefix: Retain only the final sequence position before
            the LM head for supported causal generation tasks.

    Returns:
        The built :class:`ModelPackage`.
    """
    if hasattr(config, "validate"):
        config.validate()
    if trace_optimization or fp8_kv_cache or kv_cache_scales is not None:
        raise ValueError(
            "Mobius export no longer applies graph rewrites. Apply execution-provider "
            "optimizations downstream (for example, with Olive)."
        )
    dtype = getattr(config, "dtype", ir.DataType.FLOAT)
    if prune_prefill_prefix:
        task = _enable_prefill_prefix_pruning_task(task)
    resolved_task = get_task(task)
    configure_component_quantization(
        module,
        config,
        resolved_task,
        manifest=component_manifest,
    )
    _cast_module_dtype(module, dtype)
    capabilities = ep_registry.require("default")
    target_capabilities = ep_registry.require(execution_provider)
    contract = BuildContract.from_capabilities(
        target_capabilities,
        target_device=device,
    )
    with build_context(capabilities, dtype, contract=contract):
        package = resolved_task.build(module, config)

    for model in package.values():
        register_function_bodies(model)
        SymbolicShapeInferencePass()(model)
    return package


# Attention input index for the optional ``nonpad_kv_seqlen`` operand. This
# operand and TensorScatter are defined only in opset 24.
_ATTENTION_NONPAD_KV_SEQLEN_INPUT_INDEX = 6


def _maybe_apply_opset_lowering(package: ModelPackage, execution_provider: str) -> None:
    """Lower default-domain opset 24 to 23 for sub-models where it is safe."""
    if not flags.ort_lower_opset_for_ep:
        return
    if execution_provider in ("default", "cpu"):
        return
    for name, model in package.items():
        if "" not in model.graph.opset_imports:
            continue
        if _graph_requires_opset24(model.graph):
            logger.info(
                "Skipped opset→23 lowering for '%s' (EP=%s): graph uses "
                "opset-24-only ops (TensorScatter / Attention nonpad_kv_seqlen). "
                "Preserving opset 24 to keep the static-cache Flash path valid.",
                name,
                execution_provider,
            )
            continue
        original = model.graph.opset_imports[""]
        model.graph.opset_imports[""] = 23
        logger.warning(
            "Lowered opset %d→23 for '%s' (EP=%s). "
            "ORT does not yet register opset %d kernels for this EP. "
            "Track https://github.com/microsoft/onnxruntime/issues/27729",
            original,
            name,
            execution_provider,
            original,
        )


def _graph_requires_opset24(graph: ir.Graph) -> bool:
    """Return whether *graph* uses opset-24-only default-domain semantics."""
    for node in ir.traversal.RecursiveGraphIterator(graph):
        if node.domain not in ("", "ai.onnx"):
            continue
        if node.op_type == "TensorScatter":
            return True
        if node.op_type == "Attention":
            inputs = node.inputs
            if (
                len(inputs) > _ATTENTION_NONPAD_KV_SEQLEN_INPUT_INDEX
                and inputs[_ATTENTION_NONPAD_KV_SEQLEN_INPUT_INDEX] is not None
            ):
                return True
    return False
