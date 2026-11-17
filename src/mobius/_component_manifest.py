# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Canonical metadata for the components of a model package."""

from __future__ import annotations

__all__ = [
    "ComponentDescriptor",
    "ComponentManifest",
    "SharedWeightEndpoint",
    "SharedWeightInfo",
    "get_hf_component_sources",
    "resolve_component_manifest",
]

import dataclasses
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

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

    @property
    def endpoints(self) -> tuple[SharedWeightEndpoint, ...]:
        """Canonical endpoint followed by every alias."""
        return (self.canonical, *self.aliases)


@dataclasses.dataclass(frozen=True)
class ComponentDescriptor:
    """One package component and all metadata needed to address it.

    Attributes:
        name: Key used by :class:`~mobius.ModelPackage`.
        module_attribute_path: Dotted Python attribute path from the root
            :class:`onnxscript.nn.Module` passed to ``task.build()`` to the
            sub-module that constructs this component. The empty string means
            the root module itself. This is not a package key or checkpoint
            prefix.
        role: Task-defined optimization category. Current roles include
            ``decoder``, ``encoder``, ``vision``, ``embedding``, and ``glue``.
        source_paths: Runtime HuggingFace ``named_modules()`` paths whose
            weights belong to this component.
        source_path_aliases: Pairs of ``(local_prefix, source_prefix)`` for
            component paths that cannot be aligned by a shared anchor segment.
    """

    name: str
    module_attribute_path: str
    role: str
    source_paths: tuple[str, ...] = ()
    source_path_aliases: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("component name must not be empty")
        if not self.role:
            raise ValueError(f"component {self.name!r} must declare a role")
        if any(not path for path in self.source_paths):
            raise ValueError(
                f"component {self.name!r} source_paths must not contain empty paths"
            )
        if any(not local or not source for local, source in self.source_path_aliases):
            raise ValueError(
                f"component {self.name!r} source_path_aliases must contain "
                "non-empty local/source prefixes"
            )

    def source_module_names(self, local_module_path: str) -> tuple[str, ...]:
        """Candidate HuggingFace names for a component-local module path.

        Source roots and Mobius paths commonly share an anchor segment even
        when their prefixes differ. For example, source root
        ``model.language_model.layers`` and local path
        ``model.layers.0.self_attn.q_proj`` share ``layers`` and resolve to
        ``model.language_model.layers.0.self_attn.q_proj``.
        """
        if not local_module_path:
            return self.source_paths

        local_parts = local_module_path.split(".")
        candidates: list[str] = []
        for local_prefix, source_prefix in self.source_path_aliases:
            if local_module_path == local_prefix:
                candidates.append(source_prefix)
            elif local_module_path.startswith(f"{local_prefix}."):
                suffix = local_module_path[len(local_prefix) + 1 :]
                candidates.append(f"{source_prefix}.{suffix}")
        for source_path in self.source_paths:
            source_parts = source_path.split(".")
            anchor = source_parts[-1]
            anchor_indices = [
                index for index, part in enumerate(local_parts) if part == anchor
            ]
            if anchor_indices:
                for index in anchor_indices:
                    suffix_parts = local_parts[index + 1 :]
                    candidates.append(".".join((*source_parts, *suffix_parts)))
        if not candidates:
            component_roots = {self.name, self.module_attribute_path.rsplit(".", 1)[-1]}
            for source_path in self.source_paths:
                if source_path.rsplit(".", 1)[-1] in component_roots:
                    # Use the component root only if no alias or separate
                    # source (such as a top-level output head) already owns it.
                    candidates.append(f"{source_path}.{local_module_path}")
        return tuple(dict.fromkeys((local_module_path, *candidates)))


@dataclasses.dataclass(frozen=True)
class ComponentManifest(Mapping[str, ComponentDescriptor]):
    """Ordered, immutable component metadata keyed by package component name."""

    components: tuple[ComponentDescriptor, ...]
    shared_weights: tuple[SharedWeightInfo, ...] = ()
    _by_name: Mapping[str, ComponentDescriptor] = dataclasses.field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        by_name: dict[str, ComponentDescriptor] = {}
        for component in self.components:
            if component.name in by_name:
                raise ValueError(
                    f"component manifest declares {component.name!r} more than once"
                )
            by_name[component.name] = component
        shared_names: set[str] = set()
        for shared_weight in self.shared_weights:
            if shared_weight.name in shared_names:
                raise ValueError(
                    f"shared weight {shared_weight.name!r} is declared more than once"
                )
            shared_names.add(shared_weight.name)
            for endpoint in shared_weight.endpoints:
                if endpoint.component not in by_name:
                    raise ValueError(
                        f"shared weight {shared_weight.name!r} references unknown "
                        f"component {endpoint.component!r}"
                    )
                source_paths = by_name[endpoint.component].source_paths
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
        object.__setattr__(self, "_by_name", MappingProxyType(by_name))

    def __getitem__(self, name: str) -> ComponentDescriptor:
        return self._by_name[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    @property
    def names(self) -> tuple[str, ...]:
        """Component names in task declaration order."""
        return tuple(self._by_name)


def get_hf_component_sources(
    module_class: type,
    model_type: str | None,
    hf_config: object,
) -> dict[str, tuple[str, ...]]:
    """Read component paths, invoking dynamic hooks only with a known model type."""
    resolver = getattr(module_class, "get_hf_component_sources", None)
    if resolver is not None:
        model_type = model_type or getattr(hf_config, "model_type", None)
        if not model_type:
            return {}
        resolved = resolver(model_type=model_type, hf_config=hf_config)
    else:
        resolved = getattr(module_class, "HF_COMPONENT_SOURCES", {})
    return {name: tuple(paths) for name, paths in resolved.items()}


def _config_ties_word_embeddings(hf_config: object) -> bool:
    """Honor text-config precedence, except for an explicit quantized tie."""

    def field(value: object, name: str) -> object | None:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    configs = (
        hf_config,
        field(hf_config, "text_config"),
        field(hf_config, "llm_config"),
        field(hf_config, "language_config"),
    )
    if any(
        bool(field(declaration, "tie_word_embeddings"))
        for config in configs
        if config is not None
        for declaration in (
            field(config, "quantization_config"),
            field(config, "quantization"),
        )
        if declaration is not None
    ):
        return True
    text_config = next((config for config in configs[1:] if config is not None), None)
    for config in (text_config, hf_config):
        if config is None:
            continue
        for name in ("tie_word_embeddings", "weight_tying"):
            value = field(config, name)
            if value is not None:
                return bool(value)
    return False


def _aliased_component_endpoint(
    manifest: ComponentManifest,
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


def _infer_shared_weights(
    hf_config: object,
    manifest: ComponentManifest,
) -> tuple[SharedWeightInfo, ...]:
    """Infer tied embeddings only for explicit, unique component source aliases."""
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


def resolve_component_manifest(
    task: object,
    *,
    module_class: type | None = None,
    model_type: str | None = None,
    hf_config: object | None = None,
) -> ComponentManifest:
    """Combine task roles/paths and model source ownership into one manifest."""
    roles = dict(getattr(task, "model_roles", {}) or {})
    component_spec = getattr(task, "components", None)
    module_paths = dict(component_spec.items()) if component_spec is not None else {}

    component_sources: dict[str, tuple[str, ...]] = {}
    component_aliases: dict[str, tuple[tuple[str, str], ...]] = {}
    if module_class is not None and hf_config is not None:
        component_sources = get_hf_component_sources(
            module_class,
            model_type,
            hf_config,
        )
        alias_resolver = getattr(module_class, "get_hf_component_module_aliases", None)
        raw_aliases = (
            alias_resolver(hf_config=hf_config)
            if alias_resolver is not None
            else getattr(module_class, "HF_COMPONENT_MODULE_ALIASES", {})
        )
        component_aliases = {
            name: tuple(aliases.items()) for name, aliases in raw_aliases.items()
        }

    ordered_names = tuple(dict.fromkeys((*roles, *module_paths)))
    descriptors = tuple(
        ComponentDescriptor(
            name=name,
            module_attribute_path=module_paths.get(
                name,
                "" if name == "model" else name,
            ),
            role=roles.get(name, "decoder"),
            source_paths=component_sources.get(name, ()),
            source_path_aliases=component_aliases.get(name, ()),
        )
        for name in ordered_names
    )
    manifest = ComponentManifest(descriptors)
    return ComponentManifest(
        descriptors,
        shared_weights=_infer_shared_weights(hf_config, manifest)
        if hf_config is not None
        else (),
    )
