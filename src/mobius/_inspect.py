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

__all__ = [
    "ComponentInfo",
    "SharedWeightEndpoint",
    "SharedWeightInfo",
    "inspect_components",
]

import dataclasses
import logging
from collections.abc import Mapping
from typing import Any, cast

logger = logging.getLogger(__name__)

_INPUT_EMBEDDING_MODULE_NAMES = {
    "codec_embedding",
    "embed_tokens",
    "shared",
    "text_embedding",
    "tok_embeddings",
}
_OUTPUT_HEAD_MODULE_NAMES = {
    "codec_head",
    "lm_head",
    "output",
    "output_projection",
    "proj_out",
}


@dataclasses.dataclass(frozen=True)
class SharedWeightEndpoint:
    """One component-local consumer of a shared HuggingFace parameter."""

    component: str
    parameter: str

    def __post_init__(self) -> None:
        if not self.component:
            raise ValueError("shared-weight component must not be empty")
        if not self.parameter:
            raise ValueError("shared-weight parameter must not be empty")

    @classmethod
    def from_value(cls, value: object) -> SharedWeightEndpoint:
        """Normalize a mapping or duck-typed endpoint."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                component=str(value["component"]),
                parameter=str(value["parameter"]),
            )
        duck_value = cast(Any, value)
        return cls(
            component=str(duck_value.component),
            parameter=str(duck_value.parameter),
        )


@dataclasses.dataclass(frozen=True)
class SharedWeightInfo:
    """A logical HuggingFace parameter consumed by multiple components."""

    name: str
    canonical: SharedWeightEndpoint
    aliases: tuple[SharedWeightEndpoint, ...]
    kind: str = "parameter_alias"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("shared-weight name must not be empty")
        if not self.kind:
            raise ValueError(f"shared weight {self.name!r} kind must not be empty")
        if not self.aliases:
            raise ValueError(f"shared weight {self.name!r} must declare at least one alias")
        endpoints = (self.canonical, *self.aliases)
        if len(set(endpoints)) != len(endpoints):
            raise ValueError(f"shared weight {self.name!r} contains duplicate endpoints")

    @classmethod
    def from_value(cls, value: object) -> SharedWeightInfo:
        """Normalize a mapping or duck-typed shared-weight declaration."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                name=str(value["name"]),
                canonical=SharedWeightEndpoint.from_value(value["canonical"]),
                aliases=tuple(
                    SharedWeightEndpoint.from_value(alias)
                    for alias in value.get("aliases", ())
                ),
                kind=str(value.get("kind", "parameter_alias")),
            )
        duck_value = cast(Any, value)
        return cls(
            name=str(duck_value.name),
            canonical=SharedWeightEndpoint.from_value(duck_value.canonical),
            aliases=tuple(
                SharedWeightEndpoint.from_value(alias) for alias in duck_value.aliases
            ),
            kind=str(getattr(duck_value, "kind", "parameter_alias")),
        )

    @property
    def endpoints(self) -> tuple[SharedWeightEndpoint, ...]:
        """Canonical endpoint followed by every alias."""
        return (self.canonical, *self.aliases)


@dataclasses.dataclass(frozen=True)
class ComponentInfo:
    """A single component of a model.

    Attributes:
        name: Component name. This is the ``ModelPackage`` key mobius produces
            (and the subfolder name a multi-component export is saved under).
        role: Optimization role of the component, e.g. ``"decoder"``,
            ``"encoder"``, ``"embedding"``, or ``"glue"``. Mobius uses this to
            gate fusion passes (only ``"decoder"`` receives GQA / QKV-packing).
            ``"glue"`` marks a parameter-free graph that only wires a
            generation loop — it carries no weights and no fusion applies. It
            is the value declared in the task's ``model_roles``.
        source_paths: Runtime ``named_modules()`` paths that make up this
            component inside the full HuggingFace model. These are not
            checkpoint/state-dict key prefixes. A single component may map to
            multiple disjoint sub-modules (e.g. a decoder assembled from
            ``model.layers``, ``model.norm`` and ``lm_head``), so this is a
            tuple. Empty when the component is the whole model or the layout is
            unknown. Tools such as Olive use these to optimize a submodule in
            place before exporting the full model.
        shared_weights: Cross-component shared parameters involving this
            component. The same immutable declaration is attached to every
            participating component so each independently selected build keeps
            the complete relationship.
    """

    name: str
    role: str
    source_paths: tuple[str, ...] = ()
    shared_weights: tuple[SharedWeightInfo, ...] = ()


def _resolve_task_model_type_and_config(
    model_id: str, task, trust_remote_code: bool
) -> tuple[str, str | None, object | None]:
    """Resolve the mobius task name for a model id without building it.

    Mirrors the model_type/task resolution in :func:`mobius.build`, limited to
    what is needed to pick a task and inspect class-level component metadata
    (no module construction, no weight loading).
    """
    import transformers

    from mobius._registry import _detect_fallback_registration, registry
    from mobius.integrations.transformers._config_resolver import (
        _default_task_for_model,
        _try_load_config_json,
    )

    try:
        hf_config = transformers.AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
    except (ValueError, KeyError, OSError):
        hf_config = _try_load_config_json(model_id)
        if hf_config is None:
            # An explicit task is enough to report component names and roles,
            # but source paths require a model type and therefore stay empty.
            if task is not None:
                return task, None, None
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

    if task is not None:
        return task, model_type, hf_config
    if model_type in registry:
        return _default_task_for_model(model_type), model_type, hf_config

    fallback = _detect_fallback_registration(hf_config)
    if fallback is not None and fallback.task is not None:
        return fallback.task, model_type, hf_config
    return _default_task_for_model(model_type), model_type, hf_config


def _get_hf_component_sources(
    module_class: type,
    model_type: str,
    hf_config: object,
) -> dict[str, tuple[str, ...]]:
    """Read runtime HuggingFace component paths from a registered model class."""
    from mobius._component_manifest import get_hf_component_sources

    return get_hf_component_sources(module_class, model_type, hf_config)


def _config_ties_word_embeddings(hf_config: object) -> bool:
    """Whether the parent or nested text config declares tied embeddings."""
    configs = (
        hf_config,
        getattr(hf_config, "text_config", None),
        getattr(hf_config, "llm_config", None),
        getattr(hf_config, "language_config", None),
    )
    return any(
        bool(getattr(config, "tie_word_embeddings", False))
        for config in configs
        if config is not None
    )


def _aliased_component_endpoint(
    manifest,
    *,
    role: str,
    local_names: set[str],
) -> SharedWeightEndpoint | None:
    """Resolve one explicitly aliased source module for a component role."""
    candidates = {
        SharedWeightEndpoint(
            component=component.name,
            parameter=f"{source_path}.weight",
        )
        for component in manifest.values()
        if component.role == role
        for local_path, source_path in component.source_path_aliases
        if local_path.rsplit(".", 1)[-1] in local_names
    }
    if len(candidates) != 1:
        return None
    return candidates.pop()


def _infer_hf_shared_weights(
    hf_config: object,
    manifest,
) -> tuple[SharedWeightInfo, ...]:
    """Infer standard tied word embeddings from config plus explicit aliases."""
    if not _config_ties_word_embeddings(hf_config):
        return ()
    canonical = _aliased_component_endpoint(
        manifest,
        role="embedding",
        local_names=_INPUT_EMBEDDING_MODULE_NAMES,
    )
    output = _aliased_component_endpoint(
        manifest,
        role="decoder",
        local_names=_OUTPUT_HEAD_MODULE_NAMES,
    )
    if canonical is None or output is None or canonical.component == output.component:
        return ()
    return (
        SharedWeightInfo(
            name="word_embeddings",
            kind="tied_word_embeddings",
            canonical=canonical,
            aliases=(output,),
        ),
    )


def _validate_shared_weights(shared_weights, manifest) -> None:
    """Validate component ownership and runtime paths for shared parameters."""
    seen_names: set[str] = set()
    for shared_weight in shared_weights:
        if shared_weight.name in seen_names:
            raise ValueError(
                f"shared weight {shared_weight.name!r} is declared more than once"
            )
        seen_names.add(shared_weight.name)
        for endpoint in shared_weight.endpoints:
            if endpoint.component not in manifest:
                raise ValueError(
                    f"shared weight {shared_weight.name!r} references unknown "
                    f"component {endpoint.component!r}"
                )
            source_paths = manifest[endpoint.component].source_paths
            module_path = endpoint.parameter.rpartition(".")[0]
            if source_paths and not any(
                module_path == source_path or module_path.startswith(f"{source_path}.")
                for source_path in source_paths
            ):
                raise ValueError(
                    f"shared weight {shared_weight.name!r} parameter "
                    f"{endpoint.parameter!r} is outside component "
                    f"{endpoint.component!r} source paths {source_paths!r}"
                )


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

    resolved_task, model_type, hf_config = _resolve_task_model_type_and_config(
        model_id, task, trust_remote_code
    )
    task_obj = get_task(resolved_task)
    module_class = None
    if model_type is not None and hf_config is not None and model_type in registry:
        module_class = registry.get(model_type)
    manifest = task_obj.component_manifest(
        module_class=module_class,
        model_type=model_type,
        hf_config=hf_config,
    )
    shared_weights = (
        _infer_hf_shared_weights(hf_config, manifest) if hf_config is not None else ()
    )
    _validate_shared_weights(shared_weights, manifest)
    shared_by_component = {
        name: tuple(
            shared_weight
            for shared_weight in shared_weights
            if any(endpoint.component == name for endpoint in shared_weight.endpoints)
        )
        for name in manifest
    }

    components = [
        ComponentInfo(
            name=component.name,
            role=component.role,
            source_paths=component.source_paths,
            shared_weights=shared_by_component[component.name],
        )
        for component in manifest.values()
    ]
    logger.debug(
        "inspect_components(%s): task=%s components=%s",
        model_id,
        resolved_task,
        [c.name for c in components],
    )
    return components
