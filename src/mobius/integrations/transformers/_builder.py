# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build ONNX model packages from Hugging Face Transformers checkpoints."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import onnx_ir as ir
from onnxscript import nn

from mobius._builder import build_from_module, resolve_dtype
from mobius._model_package import ModelPackage
from mobius._registry import registry
from mobius.integrations._weight_loading import _download_weights
from mobius.tasks import ModelTask

logger = logging.getLogger(__name__)


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
) -> tuple[type[nn.Module], str | ModelTask | None, str]:
    """Resolve architecture aliases and structural fallback registrations."""
    architectures = getattr(parent_config, "architectures", None) or []
    if architectures and architectures[0] in registry:
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
    trust_remote_code: bool = False,
    execution_provider: str = "default",
    trace_optimization: bool = False,
    text_only: bool = False,
    fp8_kv_cache: bool = False,
    kv_cache_scales: dict[int, tuple[float, float]] | None = None,
    prune_prefill_prefix: bool = False,
) -> ModelPackage:
    """Build a model package from a Transformers checkpoint.

    If the repository is not a supported Transformers checkpoint, dispatch to
    the Diffusers integration so :func:`mobius.build` remains ecosystem-agnostic.
    """
    from mobius.integrations.diffusers import build_diffusers_pipeline
    from mobius.integrations.transformers._config_resolver import (
        _config_from_hf,
        _default_task_for_model,
    )

    hf_config, loaded_from_raw_json = _load_transformers_config(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    if hf_config is None or (loaded_from_raw_json and hf_config.model_type not in registry):
        if text_only:
            raise ValueError(
                f"text_only=True is not supported for '{model_id}': it does not "
                "resolve to a registered text-capable model_type (it looks like "
                "a diffusers pipeline or an unsupported config)."
            )
        return build_diffusers_pipeline(
            model_id,
            revision=revision,
            dtype=dtype,
            load_weights=load_weights,
            execution_provider=execution_provider,
        )

    hf_config, parent_config, model_type = _select_primary_config(hf_config)

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
    )
    config = _config_from_hf(
        hf_config,
        parent_config=parent_config,
        module_class=module_class,
    )

    if text_only:
        config = _strip_to_text_only(config, model_type)
    if dtype is not None:
        config = dataclasses.replace(config, dtype=resolve_dtype(dtype))
    if output_layer_indices is not None:
        config = dataclasses.replace(
            config,
            output_layer_indices=list(output_layer_indices),
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
    for name, model in package.items():
        model.graph.name = f"{model_id}/{name}"

    if load_weights:
        state_dict = _download_weights(model_id, revision=revision)
        if hasattr(model_module, "preprocess_weights"):
            state_dict = model_module.preprocess_weights(state_dict)
        package.apply_weights(
            state_dict,
            prefix_map=getattr(model_module, "weight_prefix_map", None),
        )
    return package


__all__ = ["build_transformers_model"]
