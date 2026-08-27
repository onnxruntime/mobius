# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration resolution for HuggingFace models.

Resolves HuggingFace model configs to the internal ``BaseModelConfig``
subclasses used for ONNX graph construction.
"""

from __future__ import annotations

__all__ = [
    "_config_from_hf",
    "_default_task_for_model",
    "_dict_to_pretrained_config",
    "_try_load_config_json",
]

import dataclasses
import logging

from mobius._configs import (
    ArchitectureConfig,
    BaseModelConfig,
    _as_attribute_config,
)
from mobius._registry import registry

logger = logging.getLogger(__name__)


def _config_from_hf(hf_config, parent_config=None, module_class=None) -> BaseModelConfig:
    """Select the right config class for a HuggingFace config object.

    Resolution order:

    1. If *module_class* is given and has a non-default ``config_class``, use it.
    2. Query the :data:`registry` for a config class registered for the model type.
    3. Fall back to ``ArchitectureConfig``.
    """
    # `transformers` 5.x leaves nested sub-configs (a decoder, a vision tower) as
    # plain dicts, and callers pass those straight in. Every path below reads the
    # config with `getattr`, which on a dict yields the default for *every* field:
    # the model type resolves to None, so step 2 silently falls through to
    # ArchitectureConfig, and then a required field raises AttributeError far from
    # the cause. Normalising here fixes it once for every config class instead of
    # each one re-discovering it.
    hf_config = _as_attribute_config(hf_config)

    config_cls: type[BaseModelConfig] | None = None

    # 1. Module-level override
    if module_class is not None:
        config_cls = getattr(module_class, "config_class", None)

    # 2. Registry lookup (when module didn't specify a non-default class)
    if config_cls is None or config_cls is ArchitectureConfig:
        model_type = getattr(hf_config, "model_type", None)
        if model_type and model_type in registry:
            reg_cls = registry.get_config_class(model_type)
            if reg_cls is not None:
                config_cls = reg_cls

    # 3. Default
    if config_cls is None or config_cls is ArchitectureConfig:
        config_cls = ArchitectureConfig

    # Call from_transformers — pass parent_config for ArchitectureConfig tree
    if issubclass(config_cls, ArchitectureConfig):
        resolved = config_cls.from_transformers(hf_config, parent_config=parent_config)
    else:
        resolved = config_cls.from_transformers(hf_config)

    quantization = resolved.quantization
    if (
        module_class is not None
        and resolved.component_quantization is None
        and quantization is not None
        and quantization.has_module_plan
    ):
        source_config = parent_config or hf_config
        model_type = getattr(source_config, "model_type", getattr(hf_config, "model_type", ""))
        source_resolver = getattr(module_class, "get_hf_component_sources", None)
        if source_resolver is not None:
            component_sources = source_resolver(
                model_type=model_type,
                hf_config=source_config,
            )
        else:
            component_sources = getattr(module_class, "HF_COMPONENT_SOURCES", {})
        if component_sources:
            component_quantization = quantization.for_components(component_sources)
            resolved = dataclasses.replace(
                resolved,
                quantization=component_quantization.get(
                    "decoder",
                    component_quantization.get("model"),
                ),
                component_quantization=component_quantization,
            )
    return resolved


def _default_task_for_model(model_type: str) -> str:
    """Return the default task name for a HuggingFace model type.

    Reads from the :data:`registry` first, then falls back to the
    ``default_task`` class attribute on the registered model class.
    Falls back to ``"text-generation"`` if not set or unregistered.
    """
    if model_type not in registry:
        return "text-generation"
    task = registry.get_task(model_type)
    if task is not None:
        return task
    cls = registry.get(model_type)
    return getattr(cls, "default_task", "text-generation")


def _try_load_config_json(model_id: str, revision: str | None = None):
    """Try to load config.json directly for models not in transformers.

    Accepts a Hugging Face repo id or a local directory. Returns a
    ``PretrainedConfig``-like object with attribute access, or ``None`` if the
    file cannot be downloaded/parsed.
    """
    import json
    import os

    from huggingface_hub import hf_hub_download

    if os.path.isdir(model_id):
        path = os.path.join(model_id, "config.json")
        if not os.path.isfile(path):
            logger.debug("No config.json in local directory %s", model_id)
            return None
    else:
        try:
            kwargs = {"repo_id": model_id, "filename": "config.json"}
            if revision is not None:
                kwargs["revision"] = revision
            path = hf_hub_download(**kwargs)
        except (OSError, ValueError) as e:
            logger.debug("Failed to download config.json for %s: %s", model_id, e)
            return None

    try:
        with open(path) as f:
            config_dict = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to parse config.json for %s: %s", model_id, e)
        return None

    model_type = config_dict.get("model_type")
    if not model_type:
        model_type = _model_type_from_architectures(config_dict.get("architectures"))
        if not model_type:
            return None
        logger.info(
            "config.json for %s declares no model_type; inferred '%s' from architectures=%s",
            model_id,
            model_type,
            config_dict.get("architectures"),
        )
        config_dict = {**config_dict, "model_type": model_type}

    return _dict_to_pretrained_config(config_dict)


def _model_type_from_architectures(architectures) -> str | None:
    """Recover a HuggingFace ``model_type`` from a config's ``architectures``.

    Checkpoints published before ``model_type`` became mandatory omit the key
    (``Rostlab/prot_bert`` is one), which makes ``AutoConfig`` refuse the repo
    even though the architecture is a plain, fully supported one. The
    architecture class name is not a guess in that situation: ``transformers``
    exports it, and the class states its own config class, which states the
    model type. Anything that does not resolve through that chain returns
    ``None`` so an unknown repo still fails loudly.
    """
    if not architectures:
        return None

    import transformers

    for architecture in architectures:
        model_class = getattr(transformers, str(architecture), None)
        config_class = getattr(model_class, "config_class", None)
        model_type = getattr(config_class, "model_type", None)
        if model_type:
            return str(model_type)
    return None


def _dict_to_pretrained_config(d: dict):
    """Recursively convert a dict to a PretrainedConfig with attribute access.

    Nested config dicts (thinker_config, text_config, etc.) are also
    converted so that ``config.thinker_config.text_config.model_type``
    works correctly.
    """
    import transformers
    from huggingface_hub import errors as hub_errors

    # Introduced in newer huggingface_hub releases. Older supported versions
    # can still construct these configs and should not fail on the import.
    strict_validation_error = getattr(
        hub_errors,
        "StrictDataclassClassValidationError",
        TypeError,
    )

    # Composite configs (e.g. configs with text_config/thinker_config) may
    # duplicate rope_scaling at the top level.  PretrainedConfig's rope
    # standardization (__post_init__ → standardize_rope_params) can crash
    # with AttributeError when self.max_position_embeddings is not yet set.
    # Strip top-level rope fields ONLY for composite configs — the nested
    # text_config will carry its own rope_scaling with correct context.
    # Non-composite (flat) configs must keep rope fields intact.
    nested_config_keys = (
        "thinker_config",
        "talker_config",
        "text_config",
        "llm_config",
        "audio_config",
        "vision_config",
        "code_predictor_config",
        "speaker_encoder_config",
    )
    is_composite = any(isinstance(d.get(k), dict) for k in nested_config_keys)
    rope_keys = ("rope_scaling", "rope_parameters")
    if is_composite and any(k in d for k in rope_keys):
        logger.debug(
            "Stripping top-level rope fields from composite %s config",
            d.get("model_type", "unknown"),
        )
        d = {k: v for k, v in d.items() if k not in rope_keys}

    try:
        config = transformers.PretrainedConfig(**d)
    except (
        AttributeError,
        KeyError,
        TypeError,
        strict_validation_error,
    ) as e:
        # Newer transformers may crash during rope standardization
        # (e.g. Phi4-MM longrope format where PretrainedConfig doesn't
        # set max_position_embeddings before accessing it), or reject a
        # model-specific layer type that is newer than the installed
        # transformers (e.g. Muse Glimmer's ``window_attention``). Strip
        # only the fields validated by PretrainedConfig, construct the
        # attribute container, then restore their authoritative raw values.
        logger.warning(
            "Retrying %s config without validated model-specific fields after "
            "PretrainedConfig init failure: %s",
            d.get("model_type", "unknown"),
            e,
        )
        retry_keys = (*rope_keys, "layer_types")
        saved_fields = {k: d[k] for k in retry_keys if k in d}
        d_clean = {k: v for k, v in d.items() if k not in retry_keys}
        config = transformers.PretrainedConfig(**d_clean)
        for k, v in saved_fields.items():
            setattr(config, k, v)

    # Recursively convert known nested config keys
    rope_keys = ("rope_scaling", "rope_parameters")
    for key in nested_config_keys:
        val = getattr(config, key, None)
        if isinstance(val, dict):
            # Capture the raw rope fields before conversion: constructing a
            # nested PretrainedConfig runs HF rope standardization, which
            # silently drops non-standard rope_scaling formats (e.g. the
            # Qwen3-TTS talker's ``{"interleaved": True, "mrope_section": ...,
            # "type": "default"}``). Restore them onto the converted config so
            # _extract_mrope_fields / _extract_rope_config can still read them.
            raw_rope = {k: val[k] for k in rope_keys if k in val}
            nested = _dict_to_pretrained_config(val)
            for k, v in raw_rope.items():
                # The raw config.json value is authoritative for our extractors.
                # Restore it whenever HF standardization dropped (None) OR
                # rewrote it to a different value, so non-standard formats
                # (e.g. Qwen3-TTS's ``interleaved``/``mrope_section``) survive.
                if getattr(nested, k, None) != v:
                    setattr(nested, k, v)
            setattr(config, key, nested)
    return config
