# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build ONNX model packages from Hugging Face Transformers checkpoints."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Any

import onnx_ir as ir
from onnxscript import nn

from mobius._builder import build_from_module, resolve_dtype
from mobius._configs import QuantizedWeightFormat
from mobius._model_package import ModelPackage
from mobius._registry import registry
from mobius.integrations._weight_loading import (
    _download_weights,
    stream_preprocessed_safetensors_to_model,
    stream_qdq_safetensors_to_model,
)
from mobius.integrations.compressed_tensors import (
    CompressedTensorsConfig,
    stream_compressed_tensors_to_package,
)
from mobius.tasks import ModelTask

logger = logging.getLogger(__name__)


def _is_native_gptoss_mxfp4(config) -> bool:
    quantization = getattr(config, "quantization", None)
    return bool(
        getattr(config, "model_type", None) == "gpt_oss"
        and quantization is not None
        and quantization.weight_format is QuantizedWeightFormat.MXFP4
    )


def _validate_native_gptoss_build_contract(
    config,
    *,
    execution_provider: str,
) -> None:
    """Fail before weight I/O when native FP4 QMoE cannot be exported."""
    if not _is_native_gptoss_mxfp4(config):
        return
    if execution_provider != "cuda":
        raise ValueError(
            "Native GPT-OSS MXFP4 requires explicit CUDA export. Pass "
            "--execution-provider cuda and --dtype f16 (or bf16); the default/CPU "
            "provider has no lossless fallback. ORT must be built with FP4 QMoE "
            "enabled (CUDA >=12.8); pre-Blackwell GPUs such as A100 may require "
            "the available SM80 fallback/runtime configuration."
        )
    if config.dtype not in {ir.DataType.FLOAT16, ir.DataType.BFLOAT16}:
        raise ValueError(
            "Native GPT-OSS MXFP4 requires dtype='f16' or 'bf16' with "
            "--execution-provider cuda. ORT must be built with FP4 QMoE enabled "
            "(CUDA >=12.8); pre-Blackwell GPUs such as A100 may require the "
            "available SM80 fallback/runtime configuration."
        )


def _is_qwen4_exp_composite(config) -> bool:
    """Return whether *config* describes the multimodal Qwen4-Exp wrapper."""
    return getattr(config, "model_type", None) == "qwen4_exp" or (
        "Qwen4ExpForConditionalGeneration" in set(getattr(config, "architectures", None) or [])
    )


def _strip_to_text_only(config: Any, model_type: str) -> Any:
    """Return a copy of *config* reduced to a pure text-only decoder."""
    if not dataclasses.is_dataclass(config):
        raise TypeError(
            f"_strip_to_text_only expects a dataclass config instance, got {type(config)!r}"
        )
    field_names = {field.name for field in dataclasses.fields(config)}
    overrides: dict[str, Any] = {"model_type": model_type}
    for name in (
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "unsupported_video_token_id",
        "deepstack_visual_indexes",
        "use_bidirectional_attention",
        "audio_token_id",
        "boa_token_id",
        "vision",
        "audio",
    ):
        if name in field_names:
            overrides[name] = None
    return dataclasses.replace(
        config, **{key: value for key, value in overrides.items() if key in field_names}
    )


def _load_transformers_config(
    model_id: str,
    *,
    revision: str | None,
    trust_remote_code: bool,
) -> tuple[object | None, bool]:
    """Load a Transformers config and report whether raw JSON was required."""
    import transformers

    from mobius.integrations.transformers._config_resolver import _try_load_config_json

    try:
        kwargs = {"trust_remote_code": trust_remote_code}
        if revision is not None:
            kwargs["revision"] = revision
        return transformers.AutoConfig.from_pretrained(model_id, **kwargs), False
    except (ValueError, KeyError, OSError):
        return _try_load_config_json(model_id, revision=revision), True


def _select_primary_config(hf_config):
    """Return ``(primary_config, parent_config, model_type)`` for composites."""
    from mobius.integrations.transformers._config_resolver import (
        _dict_to_pretrained_config,
    )

    parent_config = hf_config
    model_type = hf_config.model_type

    if hasattr(hf_config, "talker_config"):
        hf_config = hf_config.talker_config
    elif hasattr(hf_config, "thinker_config"):
        thinker = hf_config.thinker_config
        if isinstance(thinker, dict):
            thinker = _dict_to_pretrained_config(thinker)
        if getattr(thinker, "text_config", None) is not None:
            hf_config = thinker.text_config
    elif hasattr(hf_config, "decoder_config") and model_type == "qwen3_tts_tokenizer_12hz":
        decoder = hf_config.decoder_config
        if isinstance(decoder, dict):
            decoder = type("DecoderConfig", (), {**decoder, "model_type": model_type})()
        else:
            decoder.model_type = model_type
        hf_config = decoder
    elif hasattr(hf_config, "text_config"):
        if (
            model_type == "qwen3_5_moe"
            and getattr(hf_config, "vision_config", None) is not None
        ):
            model_type = "qwen3_5_moe_vl"
        hf_config = hf_config.text_config

    if model_type in ("wav2vec2", "hubert", "wavlm"):
        architectures = getattr(parent_config, "architectures", None) or []
        if any("ForCTC" in architecture for architecture in architectures):
            model_type = "mms"

    return hf_config, parent_config, model_type


def _resolve_module_class(
    model_type: str,
    parent_config,
    module_class: type[nn.Module] | None,
    task: str | ModelTask | None,
    *,
    allow_parent_architecture_override: bool = True,
) -> tuple[type[nn.Module], str | ModelTask | None, str]:
    """Resolve architecture aliases and structural fallback registrations."""
    architectures = getattr(parent_config, "architectures", None) or []
    if allow_parent_architecture_override and architectures and architectures[0] in registry:
        architecture_key = architectures[0]
        model_type_class = registry.get(model_type) if model_type in registry else None
        architecture_class = registry.get(architecture_key)
        if model_type_class is not architecture_class:
            model_type = architecture_key

    if module_class is not None:
        return module_class, task, model_type
    if model_type in registry:
        return registry.get(model_type), task, model_type

    from mobius._registry import _detect_fallback_registration

    fallback = _detect_fallback_registration(parent_config)
    if fallback is None:
        registry.get(model_type)  # Raise the registry's descriptive KeyError.
        raise AssertionError("unreachable")

    module_class = fallback.module_class
    logger.warning(
        "Model type '%s' is not registered. Auto-detected as compatible with %s.",
        model_type,
        module_class.__name__,
    )
    if task is None and fallback.task is not None:
        task = fallback.task
    return module_class, task, model_type


def build_transformers_model(
    model_id: str,
    task: str | ModelTask | None = None,
    *,
    revision: str | None = None,
    module_class: type[nn.Module] | None = None,
    dtype: str | ir.DataType | None = None,
    output_layer_indices: list[int] | None = None,
    load_weights: bool = True,
    keep_quantized: bool = True,
    trust_remote_code: bool = False,
    execution_provider: str = "default",
    trace_optimization: bool = False,
    text_only: bool = False,
    fp8_kv_cache: bool = False,
    kv_cache_scales: dict[int, tuple[float, float]] | None = None,
    prune_prefill_prefix: bool = False,
    glm_full_attention: bool = False,
    export_paged_attention: bool = False,
    input_sampling_rate: int | None = None,
    bwe_sampling_rate: int | None = None,
) -> ModelPackage:
    """Build a model package from a Transformers checkpoint.

    If the repository is not a supported Transformers checkpoint, dispatch to
    the Diffusers integration so :func:`mobius.build` remains ecosystem-agnostic.

    ``glm_full_attention`` is the ``--glm-full-attention`` CLI feature: it
    forces ``config.use_dsa=False`` so GLM-5.2 (``glm_moe_dsa``) exports plain
    dense causal attention (executable on stock ORT) instead of the DSA
    ``IndexShare`` path, which requires the native custom-op runtime kernel.
    It is only valid for ``model_type == "glm_moe_dsa"``.

    ``keep_quantized`` preserves supported compressed-tensors checkpoints in
    their native block-weight representation. Set it to ``False`` only to
    request explicit dense reconstruction.
    """
    if input_sampling_rate is not None and bwe_sampling_rate is not None:
        raise ValueError("input_sampling_rate and bwe_sampling_rate are mutually exclusive")

    from mobius.integrations._moshi import (
        _build_personaplex,
        _is_personaplex_checkpoint,
        _personaplex_revision,
    )

    if _is_personaplex_checkpoint(model_id):
        unsupported = {
            "task": task is not None,
            "module_class": module_class is not None,
            "output_layer_indices": output_layer_indices is not None,
            "trace_optimization": trace_optimization,
            "dequantize": not keep_quantized,
            "text_only": text_only,
            "fp8_kv_cache": fp8_kv_cache,
            "kv_cache_scales": kv_cache_scales is not None,
            "prune_prefill_prefix": prune_prefill_prefix,
            "glm_full_attention": glm_full_attention,
            "export_paged_attention": export_paged_attention,
            "input_sampling_rate": input_sampling_rate is not None,
            "bwe_sampling_rate": bwe_sampling_rate is not None,
        }
        selected = sorted(name for name, enabled in unsupported.items() if enabled)
        if selected:
            raise ValueError(
                "PersonaPlex checkpoints do not support these build options: "
                + ", ".join(selected)
            )
        return _build_personaplex(
            model_id,
            dtype=dtype,
            execution_provider=execution_provider,
            revision=_personaplex_revision(model_id, revision),
            load_weights=load_weights,
        )

    from mobius.integrations.diffusers import build_diffusers_pipeline
    from mobius.integrations.transformers._config_resolver import (
        _config_from_hf,
        _default_task_for_model,
    )

    detection_revision = revision
    if model_id == "nvidia/RE-USE" and detection_revision is None:
        # Pin the very first AutoConfig/raw-JSON probe, not only the later
        # bespoke loader. Otherwise mutable Hub main could change dispatch
        # before RE-USE's pinned default ever takes effect.
        from mobius.models.reuse import REUSE_REVISION

        detection_revision = REUSE_REVISION

    hf_config, loaded_from_raw_json = _load_transformers_config(
        model_id,
        revision=detection_revision,
        trust_remote_code=trust_remote_code,
    )
    if hf_config is None or (loaded_from_raw_json and hf_config.model_type not in registry):
        from mobius.models.reuse import _build_reuse, _is_reuse_checkpoint

        if module_class is None and _is_reuse_checkpoint(model_id, detection_revision):
            from mobius.tasks import SpeechEnhancementTask

            if task not in (None, "speech-enhancement") and not isinstance(
                task, SpeechEnhancementTask
            ):
                raise ValueError("RE-USE checkpoints only support task='speech-enhancement'.")
            unsupported = {
                "output_layer_indices": output_layer_indices is not None,
                "text_only": text_only,
                "fp8_kv_cache": fp8_kv_cache,
                "kv_cache_scales": kv_cache_scales is not None,
                "prune_prefill_prefix": prune_prefill_prefix,
                "glm_full_attention": glm_full_attention,
                "export_paged_attention": export_paged_attention,
            }
            selected = sorted(name for name, enabled in unsupported.items() if enabled)
            if selected:
                raise ValueError(
                    "RE-USE checkpoints do not support these decoder-only options: "
                    + ", ".join(selected)
                )
            return _build_reuse(
                model_id,
                revision=detection_revision,
                dtype=dtype,
                execution_provider=execution_provider,
                load_weights=load_weights,
                input_sampling_rate=input_sampling_rate,
                bwe_sampling_rate=bwe_sampling_rate,
            )
        if input_sampling_rate is not None or bwe_sampling_rate is not None:
            raise ValueError(
                "input_sampling_rate and bwe_sampling_rate are only supported "
                "for RE-USE speech-enhancement checkpoints"
            )
        if text_only:
            raise ValueError(
                f"text_only=True is not supported for '{model_id}': it does not "
                "resolve to a registered text-capable model_type (it looks like "
                "a diffusers pipeline or an unsupported config)."
            )
        if glm_full_attention:
            raise ValueError(
                f"glm_full_attention=True is not supported for '{model_id}': it "
                "does not resolve to a registered 'glm_moe_dsa' model_type (it "
                "looks like a diffusers pipeline or an unsupported config)."
            )
        return build_diffusers_pipeline(
            model_id,
            revision=revision,
            dtype=dtype,
            load_weights=load_weights,
            execution_provider=execution_provider,
        )

    if input_sampling_rate is not None or bwe_sampling_rate is not None:
        raise ValueError(
            "input_sampling_rate and bwe_sampling_rate are only supported "
            "for RE-USE speech-enhancement checkpoints"
        )

    hf_config, parent_config, model_type = _select_primary_config(hf_config)

    compressed_tensors_config = CompressedTensorsConfig.from_hf_config(parent_config)
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

    module_class, task, model_type = _resolve_module_class(
        model_type,
        parent_config,
        module_class,
        task,
        allow_parent_architecture_override=not text_only,
    )
    config = _config_from_hf(
        hf_config,
        parent_config=parent_config,
        module_class=module_class,
    )
    if (
        compressed_tensors_config is not None
        and fp8_kv_cache
        and compressed_tensors_config.kv_cache_scheme is not None
    ):
        layer_types = config.layer_types
        if not layer_types:
            raise ValueError(
                "Cannot validate this compressed-tensors checkpoint's FP8 KV-cache "
                "scales because the decoder does not declare per-layer attention types."
            )
        expected_scale_layers = {
            index
            for index, layer_type in enumerate(layer_types)
            if layer_type == "full_attention"
        }
        provided_scale_layers = set(kv_cache_scales or {})
        if provided_scale_layers != expected_scale_layers:
            missing = sorted(expected_scale_layers - provided_scale_layers)
            extra = sorted(provided_scale_layers - expected_scale_layers)
            raise ValueError(
                "fp8_kv_cache=True requires the checkpoint's complete per-layer "
                "k_scale/v_scale map; partial maps would silently use unit scales. "
                f"Missing layers: {missing}; unexpected layers: {extra}."
            )

    if text_only:
        config = _strip_to_text_only(config, model_type)
    if dtype is not None:
        config = dataclasses.replace(config, dtype=resolve_dtype(dtype))
    elif compressed_tensors_config is not None and keep_quantized:
        # The pinned Microsoft block-weight ABI is W4A16/W8A16 with FP16 A/Y.
        config = dataclasses.replace(config, dtype=ir.DataType.FLOAT16)
    if (
        compressed_tensors_config is not None
        and keep_quantized
        and config.dtype != ir.DataType.FLOAT16
    ):
        raise ValueError(
            "Storage-preserving compressed-tensors export requires dtype='f16' "
            "for the Microsoft W4A16/W8A16 custom-op ABI. Use dtype='f16' or "
            "set keep_quantized=False (--dequantize)."
        )
    _validate_native_gptoss_build_contract(
        config,
        execution_provider=execution_provider,
    )
    if output_layer_indices is not None:
        config = dataclasses.replace(
            config,
            output_layer_indices=list(output_layer_indices),
        )
    if glm_full_attention:
        if model_type != "glm_moe_dsa":
            raise ValueError(
                f"glm_full_attention=True is not supported for model_type "
                f"'{model_type}'. It is only available for GLM-5.2 "
                "('glm_moe_dsa') checkpoints."
            )
        config = dataclasses.replace(config, use_dsa=False)
    if export_paged_attention:
        from mobius.components._paged_mla import paged_attention_rejection

        config = dataclasses.replace(config, export_paged_attention=True)
        reason = paged_attention_rejection(config)
        if reason is not None:
            raise ValueError(
                "export_paged_attention=True (--features paged-attention) is not "
                f"supported for model_type '{model_type}': {reason}"
            )
    if task is None:
        task = _default_task_for_model(model_type)

    model_module = module_class(config)
    package = build_from_module(
        model_module,
        config,
        task,
        execution_provider=execution_provider,
        trace_optimization=trace_optimization,
        fp8_kv_cache=fp8_kv_cache,
        kv_cache_scales=kv_cache_scales,
        prune_prefill_prefix=prune_prefill_prefix,
    )
    graph_source_name = (
        model_type
        if _is_native_gptoss_mxfp4(config) and pathlib.Path(model_id).is_dir()
        else model_id
    )
    for name, model in package.items():
        model.graph.name = f"{graph_source_name}/{name}"
        if model_type in {"qwen4_exp", "qwen4_exp_text"}:
            model.metadata_props["mobius.source_revision"] = revision or "unpinned"

    if load_weights:
        if _is_native_gptoss_mxfp4(config):
            from mobius.integrations.transformers._gptoss_weights import (
                stream_gptoss_mxfp4_safetensors_to_package,
            )

            stream_gptoss_mxfp4_safetensors_to_package(
                package,
                model_id,
                config,
                revision=revision,
            )
        elif config.block_quant_scheme is not None and hasattr(
            model_module, "build_fp8_streaming_plan"
        ):
            if len(package) != 1:
                raise ValueError(
                    "Block-FP8 dense streaming currently requires one text-only "
                    "model component; multimodal Qwen4-Exp remains unsupported."
                )
            stream = (
                stream_qdq_safetensors_to_model
                if keep_quantized
                else stream_preprocessed_safetensors_to_model
            )
            report = stream(
                next(iter(package.values())),
                model_id,
                model_module.build_fp8_streaming_plan,
                revision=revision,
            )
            package.weight_loading_report = report
            if keep_quantized:
                logger.info(
                    "Preserved %s FP8 storage with standard QDQ. Current ORT "
                    "execution is not claimed; see weight-loading-report.json.",
                    model_id,
                )
            else:
                logger.warning(
                    "Loaded %s as an explicitly requested streaming dense fallback; "
                    "native FP8 was not preserved. See weight-loading-report.json.",
                    model_id,
                )
        elif model_type in {"qwen4_exp", "qwen4_exp_text"}:
            from mobius.integrations.transformers._qwen4_exp_weights import (
                stream_qwen4_exp_safetensors_to_package,
            )

            stream_qwen4_exp_safetensors_to_package(
                package,
                model_id,
                config,
                revision=revision,
            )
        elif compressed_tensors_config is not None:
            stream_compressed_tensors_to_package(
                package,
                model_id,
                compressed_tensors_config,
                preprocess_weights=getattr(model_module, "preprocess_weights", None),
                revision=revision,
                fp8_kv_cache=fp8_kv_cache,
                keep_quantized=keep_quantized,
            )
        else:
            state_dict = _download_weights(model_id, revision=revision)
            if hasattr(model_module, "preprocess_weights"):
                state_dict = model_module.preprocess_weights(state_dict)
            package.apply_weights(
                state_dict,
                prefix_map=getattr(model_module, "weight_prefix_map", None),
            )
    return package


__all__ = ["build_transformers_model"]
