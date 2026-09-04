# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Canonical metadata for the components of a model package."""

from __future__ import annotations

__all__ = [
    "ComponentDescriptor",
    "ComponentManifest",
    "get_hf_component_sources",
    "resolve_component_manifest",
]

import dataclasses
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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
        candidates = [local_module_path]
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
                    suffix = local_parts[index + 1 :]
                    candidates.append(".".join((*source_parts, *suffix)))
        return tuple(dict.fromkeys(candidates))


@dataclasses.dataclass(frozen=True)
class ComponentManifest(Mapping[str, ComponentDescriptor]):
    """Ordered, immutable component metadata keyed by package component name."""

    components: tuple[ComponentDescriptor, ...]
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
    model_type: str,
    hf_config: object,
) -> dict[str, tuple[str, ...]]:
    """Read runtime HuggingFace component paths from a registered model class."""
    resolver = getattr(module_class, "get_hf_component_sources", None)
    if resolver is not None:
        resolved = resolver(model_type=model_type, hf_config=hf_config)
    else:
        resolved = getattr(module_class, "HF_COMPONENT_SOURCES", {})
    return {name: tuple(paths) for name, paths in resolved.items()}


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
    if module_class is not None and model_type is not None and hf_config is not None:
        component_sources = get_hf_component_sources(
            module_class,
            model_type,
            hf_config,
        )
        raw_aliases = getattr(module_class, "HF_COMPONENT_MODULE_ALIASES", {})
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
    return ComponentManifest(descriptors)
