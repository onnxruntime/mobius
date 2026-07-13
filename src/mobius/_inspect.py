# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Inspect a model's component layout without building it.

``inspect_components`` reports the components mobius would produce for a model
(their package keys and optimization roles) **without** constructing graphs or
loading weights. External tools such as Olive use this to plan per-component
work — e.g. optimizing a VLM's ``decoder`` differently from its
``vision_encoder`` — before calling :func:`mobius.build`.

The component names returned here are the same keys :func:`mobius.build`
produces in its :class:`~mobius._model_package.ModelPackage` (and therefore the
subfolder names ``ModelPackage.save`` writes for multi-component models).
"""

from __future__ import annotations

__all__ = ["ComponentInfo", "inspect_components"]

import dataclasses
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ComponentInfo:
    """A single component of a model.

    Attributes:
        name: Component name. This is the ``ModelPackage`` key mobius produces
            (and the subfolder name a multi-component export is saved under).
        role: Optimization role of the component, e.g. ``"decoder"``,
            ``"encoder"``, or ``"embedding"``. Mobius uses this to gate fusion
            passes (only ``"decoder"`` receives GQA / QKV-packing). It is the
            value declared in the task's ``model_roles``.
        source_paths: Dotted HuggingFace module paths that make up this
            component inside the full model. A single component may map to
            multiple disjoint sub-modules (e.g. a decoder assembled from
            ``model.layers``, ``model.norm`` and ``lm_head``), so this is a
            tuple. Empty when the component is the whole model or the layout is
            unknown. Tools such as Olive use these to optimize a submodule in
            place before exporting the full model.
    """

    name: str
    role: str
    source_paths: tuple[str, ...] = ()


def _resolve_task_and_model_type(
    model_id: str, task, trust_remote_code: bool
) -> tuple[str, str | None]:
    """Resolve the mobius task name for a model id without building it.

    Mirrors the model_type/task resolution in :func:`mobius.build`, limited to
    what is needed to pick a task (no module construction, no weight loading).
    """
    import transformers

    from mobius._config_resolver import _default_task_for_model, _try_load_config_json
    from mobius._registry import _detect_fallback_registration, registry

    if task is not None:
        return task, None

    try:
        hf_config = transformers.AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
    except (ValueError, KeyError, OSError):
        hf_config = _try_load_config_json(model_id)
        if hf_config is None:
            raise ValueError(
                f"Could not load a HuggingFace config for {model_id!r}. inspect_components supports "
                "transformers/registry models; diffusers pipelines are not supported."
            ) from None

    model_type = hf_config.model_type

    # model_type adjustments that affect task selection (subset of build()):
    # Qwen3.5-MoE ships the same model_type for text-only and VL checkpoints.
    if model_type == "qwen3_5_moe" and getattr(hf_config, "vision_config", None) is not None:
        model_type = "qwen3_5_moe_vl"
    # wav2vec2/hubert/wavlm with a CTC head map to the mms (CTC) registration.
    if model_type in ("wav2vec2", "hubert", "wavlm"):
        architectures = getattr(hf_config, "architectures", None) or []
        if any("ForCTC" in arch for arch in architectures):
            model_type = "mms"

    if model_type in registry:
        return _default_task_for_model(model_type), model_type

    fallback = _detect_fallback_registration(hf_config)
    if fallback is not None and fallback.task is not None:
        return fallback.task, model_type
    return _default_task_for_model(model_type), model_type


def inspect_components(
    model_id: str,
    task=None,
    trust_remote_code: bool = False,
) -> list[ComponentInfo]:
    """Return the components mobius would produce for a model.

    Args:
        model_id: HuggingFace model id or local path.
        task: Optional task name (e.g. ``"vision-language"``) or
            :class:`~mobius.tasks.ModelTask` instance. When ``None``, the task
            is auto-detected from the model type.
        trust_remote_code: Whether to trust remote code when loading the
            HuggingFace config.

    Returns:
        A list of :class:`ComponentInfo`. Single-component models (most LLMs)
        return a single entry named ``"model"``; multi-component models (VLMs,
        encoder-decoders, speech models) return one entry per component.

    Raises:
        ValueError: If a config cannot be resolved for ``model_id`` (e.g. a
            diffusers pipeline, which is not supported).
    """
    from mobius._registry import registry
    from mobius.tasks import get_task

    resolved_task, model_type = _resolve_task_and_model_type(model_id, task, trust_remote_code)
    task_obj = get_task(resolved_task)
    roles = task_obj.model_roles or {}

    # ``source_paths`` live as an ``HF_COMPONENT_SOURCES`` ClassVar on the
    # registered module class (next to that class's ``preprocess_weights``,
    # which routes the very same HF prefixes). Read it straight off the class
    # so inspection stays cheap: no module construction, no weight loading.
    component_sources: dict[str, tuple[str, ...]] = {}
    if model_type is not None and model_type in registry:
        module_class = registry.get(model_type)
        component_sources = getattr(module_class, "HF_COMPONENT_SOURCES", {})

    components = [
        ComponentInfo(
            name=name,
            role=role,
            source_paths=tuple(component_sources.get(name, ())),
        )
        for name, role in roles.items()
    ]
    logger.debug(
        "inspect_components(%s): task=%s components=%s",
        model_id,
        resolved_task,
        [c.name for c in components],
    )
    return components
