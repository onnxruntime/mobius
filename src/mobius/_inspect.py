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
        kind: Optimization role of the component, e.g. ``"decoder"``,
            ``"encoder"``, or ``"embedding"``.
    """

    name: str
    kind: str


def _resolve_task(model_id: str, task, trust_remote_code: bool) -> str:
    """Resolve the mobius task name for a model id without building it.

    Mirrors the model_type/task resolution in :func:`mobius.build`, limited to
    what is needed to pick a task (no module construction, no weight loading).
    """
    import transformers

    from mobius._config_resolver import _default_task_for_model, _try_load_config_json
    from mobius._registry import _detect_fallback_registration, registry

    if task is not None:
        return task

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
        return _default_task_for_model(model_type)

    fallback = _detect_fallback_registration(hf_config)
    if fallback is not None and fallback.task is not None:
        return fallback.task
    return _default_task_for_model(model_type)


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
    from mobius.tasks import get_task

    resolved_task = _resolve_task(model_id, task, trust_remote_code)
    task_obj = get_task(resolved_task)
    roles = task_obj.model_roles or {}
    components = [ComponentInfo(name=name, kind=kind) for name, kind in roles.items()]
    logger.debug(
        "inspect_components(%s): task=%s components=%s",
        model_id,
        resolved_task,
        [c.name for c in components],
    )
    return components
