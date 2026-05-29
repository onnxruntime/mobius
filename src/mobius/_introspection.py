# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Lightweight introspection of mobius-buildable models.

These helpers answer questions such as *"which ONNX component names will
``mobius.build()`` produce for this model?"* without actually building the
graph or downloading the model weights.

They are useful for tooling that needs to plan or schedule work on a per
component basis before paying the cost of a full build — for example, an
optimization pipeline that wants to assign different passes to the
``encoder`` and ``decoder`` components of an encoder-decoder model.

Only metadata files are downloaded:

- For transformer models: HuggingFace ``config.json`` (via
  ``transformers.AutoConfig``).
- For diffusers pipelines: ``model_index.json``.

No PyTorch weights are downloaded and no ONNX graphs are constructed.
"""

from __future__ import annotations

__all__ = [
    "list_components",
]

from mobius._registry import _detect_fallback_registration, registry
from mobius.tasks import ModelTask, get_task


def list_components(
    model_id: str,
    *,
    task: str | ModelTask | None = None,
    trust_remote_code: bool = False,
) -> list[str]:
    """Return the ONNX component names :func:`mobius.build` would produce.

    The result matches the keys of the :class:`~mobius.ModelPackage`
    returned by ``build(model_id)``. For single-component models this is
    ``["model"]``; for multi-component models (encoder-decoder, VAE,
    diffusers pipelines, multimodal) it is the list of sub-model names.

    Only HuggingFace metadata files (``config.json`` or ``model_index.json``)
    are downloaded; no weights are fetched and no graphs are built. The
    cost is roughly one HTTP request.

    Args:
        model_id: HuggingFace model repository ID (e.g.
            ``"meta-llama/Llama-3-8B"`` or
            ``"stabilityai/stable-diffusion-3-medium-diffusers"``).
        task: Override the auto-detected task. Either a task name string
            (e.g. ``"text-generation"``) or a :class:`ModelTask` instance.
            Ignored for diffusers pipelines.
        trust_remote_code: Whether to trust remote code when loading the
            HuggingFace config. Forwarded to ``transformers.AutoConfig``.

    Returns:
        A list of component names. Order matches the order
        :func:`mobius.build` produces (sorted by spec for transformer
        models, by ``model_index.json`` iteration order for diffusers).

    Raises:
        ValueError: If *model_id* is neither a supported transformer model
            nor a diffusers pipeline.

    Example::

        >>> import mobius
        >>> mobius.list_components("meta-llama/Llama-3-8B")
        ['model']
        >>> mobius.list_components("stabilityai/stable-diffusion-3-medium-diffusers")
        ['text_encoder', 'text_encoder_2', 'text_encoder_3', 'transformer', 'vae']
    """
    try:
        import transformers
    except ImportError as e:
        raise ImportError(
            "transformers is required for list_components(); "
            "install with `pip install transformers`."
        ) from e

    from mobius._config_resolver import _default_task_for_model, _try_load_config_json

    try:
        hf_config = transformers.AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
    except (ValueError, KeyError, OSError):
        hf_config = _try_load_config_json(model_id)
        if hf_config is None or hf_config.model_type not in registry:
            return _list_diffusers_components(model_id)

    model_type = _resolve_effective_model_type(hf_config)

    if model_type in registry:
        if task is None:
            task = _default_task_for_model(model_type)
        return _components_for_task(task)

    fallback = _detect_fallback_registration(hf_config)
    if fallback is not None:
        if task is None:
            task = fallback.task or "text-generation"
        return _components_for_task(task)

    return _list_diffusers_components(model_id)


def _resolve_effective_model_type(hf_config) -> str:
    """Return the model_type :func:`mobius.build` would actually dispatch on.

    Mirrors the composite-config unwrapping in :func:`mobius._builder.build`:
    for configs that wrap a text-only sub-config (``talker_config``,
    ``thinker_config``, ``text_config``), the dispatched model_type may
    differ from ``hf_config.model_type``. This is currently meaningful only
    for the ``qwen3_5_moe`` → ``qwen3_5_moe_vl`` override, but mirroring
    the full logic keeps the two paths in sync as ``build()`` evolves.
    """
    model_type = hf_config.model_type
    if hasattr(hf_config, "talker_config") or hasattr(hf_config, "thinker_config"):
        return model_type
    if hasattr(hf_config, "text_config"):
        if (
            model_type == "qwen3_5_moe"
            and getattr(hf_config, "vision_config", None) is not None
        ):
            return "qwen3_5_moe_vl"
    return model_type


def _components_for_task(task: str | ModelTask) -> list[str]:
    """Return the component names declared by *task*.

    Single-component tasks (with no :class:`ComponentSpec`) always
    produce a single ``"model"`` entry.
    """
    resolved = get_task(task)
    if resolved.components is None:
        return ["model"]
    return list(resolved.components.keys())


def _list_diffusers_components(model_id: str) -> list[str]:
    """List the component names :func:`build_diffusers_pipeline` would emit.

    Mirrors the iteration and flattening logic in
    :func:`mobius._diffusers_builder.build_diffusers_pipeline` so that the
    returned names exactly match the keys of the :class:`ModelPackage` it
    would produce. Components that ``build_diffusers_pipeline`` would skip
    (private entries, non-tuple values, unregistered classes) are skipped
    here too.
    """
    from mobius._diffusers_builder import (
        _DIFFUSERS_CLASS_MAP,
        _init_diffusers_class_map,
        _load_diffusers_pipeline_index,
    )

    pipeline_index = _load_diffusers_pipeline_index(model_id)
    if pipeline_index is None:
        raise ValueError(
            f"'{model_id}' is not a supported transformer model and is not "
            f"a diffusers pipeline (no model_index.json found)."
        )

    _init_diffusers_class_map()

    components: list[str] = []
    for component_name, component_info in pipeline_index.items():
        if component_name.startswith("_"):
            continue
        if not isinstance(component_info, list) or len(component_info) != 2:
            continue

        _, class_name = component_info
        if class_name not in _DIFFUSERS_CLASS_MAP:
            continue

        _, _, task_name = _DIFFUSERS_CLASS_MAP[class_name]
        sub_components = _components_for_task(task_name)

        if sub_components == ["model"]:
            components.append(component_name)
        else:
            components.extend(f"{component_name}_{sub_name}" for sub_name in sub_components)

    if not components:
        raise ValueError(f"No supported components found in diffusers pipeline '{model_id}'.")
    return components
