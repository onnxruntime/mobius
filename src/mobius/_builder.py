# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model building API.

This module provides the core functions for constructing ONNX models from
``onnxscript.nn.Module`` instances:

- :func:`build_from_module` — Build from a module instance and config.
- :func:`build` — Build from a HuggingFace model ID.
- :func:`resolve_dtype` — Resolve dtype strings to ``ir.DataType``.

EP capabilities are defined in :mod:`mobius._execution_providers`.
The optimization pipeline lives in :mod:`mobius._optimizations`.
"""

from __future__ import annotations

__all__ = [
    # Public build API
    "DTYPE_MAP",
    "build",
    "build_from_module",
    "resolve_dtype",
]

import dataclasses
import logging
from typing import Any

import onnx_ir as ir
import torch
from onnx_ir import tensor_adapters
from onnxscript import nn

from mobius._build_context import build_context
from mobius._configs import (
    BaseModelConfig,
)
from mobius._execution_providers import ep_registry
from mobius._flags import flags
from mobius._model_package import ModelPackage
from mobius._optimizations import optimize_model
from mobius._registry import registry
from mobius._weight_loading import _download_weights
from mobius.tasks import ModelTask, get_task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

# Mapping of short dtype names to ONNX IR dtypes
DTYPE_MAP: dict[str, ir.DataType] = {
    "f32": ir.DataType.FLOAT,
    "float32": ir.DataType.FLOAT,
    "f16": ir.DataType.FLOAT16,
    "float16": ir.DataType.FLOAT16,
    "bf16": ir.DataType.BFLOAT16,
    "bfloat16": ir.DataType.BFLOAT16,
}


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
    parameters, non-float types, and parameters marked ``_keep_float32`` are
    left unchanged.
    """
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


# Map ModelPackage entry names to semantic model roles.
# GQA fusion is only applied to "decoder" role models.
_MODEL_ROLE_MAP: dict[str, str] = {
    "model": "decoder",
    "decoder": "decoder",
    "vision_encoder": "vision",
    "embedding": "embedding",
    "encoder": "encoder",
    # Audio sub-models are encoder-role; must not receive decoder-only fusions.
    "audio_encoder": "encoder",
    # Backward compatibility aliases (deprecated — will be removed)
    "vision": "vision",
    "audio": "encoder",
    "speech": "encoder",
}


# ---------------------------------------------------------------------------
# Build API
# ---------------------------------------------------------------------------


def build_from_module(
    module: nn.Module,
    config: BaseModelConfig,
    task: str | ModelTask = "text-generation",
    *,
    execution_provider: str = "default",
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
            optimizations. Defaults to ``"default"``, which applies standard
            fusions (SkipNorm, Gelu) but no EP-specific vendor ops (no GQA,
            no PackQKV). Custom ops like
            ``com.microsoft::SkipLayerNormalization`` are present but carry
            portable ONNX function bodies as fallbacks. Accepted values are
            the names returned by ``ep_registry`` (e.g. ``"cpu"``,
            ``"cuda"``, ``"dml"``, ``"webgpu"``, ``"trt-rtx"``). Controls
            which fusion, lowering, and structural passes are applied; in
            particular, ``"webgpu"`` uses concrete (non-symbolic) input
            dimensions.
        trace_optimization: When ``True``, log step-by-step diagnostic
            output at INFO level for each optimization stage, showing which
            rules matched and how many nodes were added/removed.

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
    # Cast all parameters to the target dtype. Vision/audio encoder weights
    # are included — their graph inputs are kept at f32 (matching GenAI's
    # image processor output) with a Cast at the graph entry.
    _cast_module_dtype(module, dtype)
    resolved_task = get_task(task)
    capabilities = ep_registry.require(execution_provider)
    with build_context(capabilities, dtype):
        pkg = resolved_task.build(module, config)

    for name, model in pkg.items():
        # Resolve role from the task first, then fall back to the global name map.
        # This ensures encoder-only tasks (e.g. ViT, BERT) don't get GQA fusion.
        role = resolved_task.model_roles.get(name) or _MODEL_ROLE_MAP.get(name, "decoder")
        optimize_model(
            model,
            ep=execution_provider,
            dtype=dtype,
            model_role=role,
            trace=trace_optimization,
        )

    _maybe_apply_opset_lowering(pkg, execution_provider)
    return pkg


def _strip_to_text_only(config: Any, model_type: str) -> Any:
    """Return a copy of *config* reduced to a pure text-only decoder.

    Used by :func:`build` when ``text_only=True``. Overrides ``model_type``
    to the text-only registry sibling and nulls multimodal fields so the text
    backbone builds as a plain causal LM (no vision-block bidirectional
    overlay, no image/audio token routing). This lets the decoder use
    ``GroupQueryAttention`` on GQA-capable execution providers instead of the
    multimodal float-bias ``Attention`` path.

    Only fields that exist on the config dataclass are overridden, so the helper is
    safe across different config dataclasses. Raises :class:`TypeError` if *config*
    is not a dataclass instance.
    """
    if not dataclasses.is_dataclass(config):
        raise TypeError(
            f"_strip_to_text_only expects a dataclass config instance, got {type(config)!r}"
        )
    field_names = {f.name for f in dataclasses.fields(config)}
    # model_type drives downstream ORT-GenAI type selection and task defaults.
    overrides: dict[str, Any] = {"model_type": model_type}
    # Vision/audio routing fields: nulling them removes the bidirectional
    # image-block overlay and the per-layer image/audio token masking, leaving
    # a pure causal decoder. ``None`` is the "absent" value for each of these.
    for name in (
        "image_token_id",
        "use_bidirectional_attention",
        "audio_token_id",
        "boa_token_id",
        "vision",
        "audio",
    ):
        if name in field_names:
            overrides[name] = None
    return dataclasses.replace(
        config, **{k: v for k, v in overrides.items() if k in field_names}
    )


# Attention input index for the optional ``nonpad_kv_seqlen`` operand. This
# operand (external/static KV cache length) and the TensorScatter op are
# defined only in opset 24, so a graph using either must not declare opset 23.
_ATTENTION_NONPAD_KV_SEQLEN_INPUT_INDEX = 6


def _maybe_apply_opset_lowering(pkg: ModelPackage, execution_provider: str) -> None:
    """Lower the default-domain opset from 24 to 23 where it is safe to do so.

    Some EPs (older ORT builds) don't register opset 24 kernels for standard
    ops (Reshape, RMSNormalization, etc.). Without lowering, those ops fall to
    CPU and produce ~280 memcpy nodes that destroy performance. The
    ``MOBIUS_ORT_LOWER_OPSET_FOR_EP`` flag (default False) opts a deployment
    into the lowering; it is a no-op for the ``"default"`` and ``"cpu"`` EPs
    (the CPU EP already has opset-24 kernels), matching the inference-side gate
    ``ort_inference._should_lower_opset``.

    Any sub-model that uses opset-24-only semantics (TensorScatter, or
    Attention with a non-empty input #6 ``nonpad_kv_seqlen``) is left at opset
    24: declaring opset 23 on such a graph is invalid and would strip the
    static-cache Flash path. See :func:`_graph_requires_opset24`. The decision
    is made per sub-model, so a mixed package lowers its standard sub-models
    while preserving its static-cache sub-models.
    """
    if not flags.ort_lower_opset_for_ep:
        return
    # Mirror ort_inference._should_lower_opset: the "default" EP is a no-op, and
    # the CPU EP already registers opset-24 kernels, so lowering there is both
    # unnecessary and inconsistent with the inference-side gate.
    if execution_provider in ("default", "cpu"):
        return
    for name, model in pkg.items():
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
    """Return True if the graph uses opset-24-only default-domain semantics.

    Lowering the default-domain opset import to 23 on such a graph is invalid
    and would silently break the static-cache Flash-attention path. A graph
    requires opset 24 when it contains:

    - a ``TensorScatter`` node (default domain), or
    - an ``Attention`` node consuming a non-empty input #6 (``nonpad_kv_seqlen``).

    The scan is recursive: nodes nested inside ``If``/``Loop``/``Scan``
    subgraphs are inspected too, so a future graph that buries one of these ops
    in a control-flow body is still detected.
    """
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


def build(
    model_id: str,
    task: str | ModelTask | None = None,
    *,
    module_class: type[nn.Module] | None = None,
    dtype: str | ir.DataType | None = None,
    output_layer_indices: list[int] | None = None,
    load_weights: bool = True,
    trust_remote_code: bool = False,
    execution_provider: str = "default",
    trace_optimization: bool = False,
    text_only: bool = False,
    embedding_bits: int | None = None,
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
        output_layer_indices: Optional list of decoder layer indices for
            which to emit additional ``hidden_states.{k}`` ONNX outputs
            alongside the standard ``logits`` / ``present.*`` outputs.
            Each ``k`` follows the HF ``output_hidden_states`` convention
            and refers to the post-residual output of decoder layer ``k``
            (equivalent to ``model(...).hidden_states[k + 1]`` in
            transformers).  Used by speculative-decoding draft models
            such as DFlash that condition on intermediate target hidden
            states.  See
            :class:`mobius.ArchitectureConfig.output_layer_indices`.
        load_weights: Whether to download and apply weights from HuggingFace.
        trust_remote_code: Whether to trust remote code when loading the
            HuggingFace config.
        execution_provider: Target execution provider for EP-aware
            optimizations. Defaults to ``"default"``, which produces
            portable ONNX with no vendor-specific ops. Accepted values are
            the names returned by ``ep_registry`` (e.g. ``"cpu"``,
            ``"cuda"``, ``"dml"``, ``"webgpu"``, ``"trt-rtx"``). Controls
            which fusion and lowering passes are applied during graph
            optimization; ``"webgpu"`` additionally uses concrete (non-
            symbolic) input dimensions.
        trace_optimization: When ``True``, log step-by-step diagnostic
            output at INFO level for each optimization stage. See
            :func:`build_from_module` for details.
        text_only: When ``True``, export the text backbone of a multimodal
            checkpoint as a standalone decoder-only LLM. The resolved
            ``model_type`` is remapped to its text-only registry sibling (see
            ``_TEXT_ONLY_MODEL_TYPE``) and vision/audio config fields
            (``image_token_id``, ``use_bidirectional_attention``, ``vision``,
            ``audio``, ...) are stripped, yielding a pure-causal decoder that
            can use ``GroupQueryAttention`` on GQA-capable execution providers.
            Raises :class:`ValueError` if the resolved ``model_type`` has no
            text-only sibling. Currently supported for ``gemma4_unified``
            (``google/gemma-4-12B``).
        embedding_bits: Quantization bit-width for per-layer embedding tables
            (4 or 8). When set, large embedding tables that exceed the EP's
            buffer limit are block-quantized with ``GatherBlockQuantized``
            instead of stored at full precision. Only affects models that
            split per-layer embeddings (e.g. Gemma4 on WebGPU). When ``None``
            (default), the model task decides — currently defaults to INT4
            when splitting is required. Ignored by models without per-layer
            embedding tables.

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
        _dict_to_pretrained_config,
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
            if text_only:
                raise ValueError(
                    f"text_only=True is not supported for '{model_id}': it does "
                    "not resolve to a registered text-capable model_type (it "
                    "looks like a diffusers pipeline or an unsupported config)."
                )
            # Not a model we support — try diffusers pipeline
            return build_diffusers_pipeline(
                model_id,
                dtype=dtype,
                load_weights=load_weights,
            )

    model_type = hf_config.model_type

    parent_config = hf_config
    if hasattr(hf_config, "talker_config"):
        hf_config = hf_config.talker_config
    elif hasattr(hf_config, "thinker_config"):
        thinker = hf_config.thinker_config
        # Some checkpoints (e.g. Qwen3-ASR) ship ``thinker_config`` as a plain
        # dict rather than a nested ``PretrainedConfig``. Convert it so the
        # decoder ``text_config`` (and its scalar fields such as hidden_size)
        # is reachable via attribute access.
        if isinstance(thinker, dict):
            thinker = _dict_to_pretrained_config(thinker)
        if getattr(thinker, "text_config", None) is not None:
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
        # Qwen3.5-MoE-VL (Qwen3.6-35B-A3B etc.) ships ``model_type=qwen3_5_moe``
        # for *both* text-only and VL checkpoints. When the composite carries
        # a ``vision_config`` sub-object, override model_type so the registry
        # picks the 3-model VL class. The text backbone's per-layer fields
        # still live under ``text_config``, so we always unwrap; the original
        # composite stays available via ``parent_config`` for vision extraction.
        if (
            model_type == "qwen3_5_moe"
            and getattr(hf_config, "vision_config", None) is not None
        ):
            model_type = "qwen3_5_moe_vl"
        hf_config = hf_config.text_config

    # Wav2Vec2 / HuBERT / WavLM ship ``model_type="wav2vec2"`` (etc.) for both
    # feature-extraction and CTC checkpoints. Switch to the ``mms`` registration
    # (Wav2Vec2ForCTCModel + ctc-asr task) when the architecture indicates a
    # CTC head — this covers both MMS and vanilla Wav2Vec2ForCTC fine-tunes.
    if model_type in ("wav2vec2", "hubert", "wavlm"):
        architectures = getattr(parent_config, "architectures", None) or []
        if any("ForCTC" in arch for arch in architectures):
            model_type = "mms"

    if text_only:
        from mobius._registry import _TEXT_ONLY_MODEL_TYPE

        text_type = _TEXT_ONLY_MODEL_TYPE.get(model_type)
        if text_type is None:
            raise ValueError(
                f"text_only=True is not supported for model_type '{model_type}'. "
                "It is only available for multimodal checkpoints with a text-only "
                f"registry sibling: {sorted(_TEXT_ONLY_MODEL_TYPE)}."
            )
        model_type = text_type

    # DFlash speculative-decoding drafters ship ``model_type="qwen3"`` (the
    # base Qwen3 family) but declare ``architectures=["DFlashDraftModel"]``.
    # Re-route via the architectures field so build() picks the cross-
    # attending drafter class + dflash-draft task instead of the standard
    # CausalLMModel + text-generation task.
    architectures = getattr(parent_config, "architectures", None) or []
    if architectures and architectures[0] in registry:
        arch_key = architectures[0]
        # Only override when the architecture-keyed registration is *more
        # specific* than the model_type-keyed one (i.e. different class).
        model_type_class = registry.get(model_type) if model_type in registry else None
        arch_class = registry.get(arch_key)
        if model_type_class is not arch_class:
            model_type = arch_key

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

    if text_only:
        config = _strip_to_text_only(config, model_type)

    if dtype is not None:
        dtype = resolve_dtype(dtype)
        config = dataclasses.replace(config, dtype=dtype)

    if output_layer_indices is not None:
        # Opt-in: emit additional `hidden_states.{k}` ONNX outputs for each
        # listed decoder layer index.  Used by speculative-decoding draft
        # models (e.g. DFlash) that condition on intermediate target hidden
        # states.  See ``ArchitectureConfig.output_layer_indices``.
        config = dataclasses.replace(config, output_layer_indices=list(output_layer_indices))

    if embedding_bits is not None:
        if embedding_bits not in (4, 8):
            raise ValueError(f"embedding_bits must be 4 or 8, got {embedding_bits}")
        # ``embedding_bits`` is only for Gemma4's per-layer embedding table.  Do not
        # attach a QuantizationConfig to ordinary text models, because that changes
        # their regular token embedding/Linear modules.
        if (
            hasattr(config, "per_layer_embedding_bits")
            and getattr(config, "hidden_size_per_layer_input", 0)
            and getattr(config, "vocab_size_per_layer_input", 0)
        ):
            config = dataclasses.replace(
                config,
                per_layer_embedding_bits=embedding_bits,
                per_layer_embedding_group_size=32,
                per_layer_embedding_sym=False,
            )

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

    for name, model in pkg.items():
        model.graph.name = f"{model_id}/{name}"

    if load_weights:
        state_dict = _download_weights(model_id)
        if hasattr(model_module, "preprocess_weights"):
            state_dict = model_module.preprocess_weights(state_dict)
        prefix_map = getattr(model_module, "weight_prefix_map", None)
        pkg.apply_weights(state_dict, prefix_map=prefix_map)

    return pkg
