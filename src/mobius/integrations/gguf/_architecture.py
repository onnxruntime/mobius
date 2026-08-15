# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Architecture adapters for GGUF imports.

Adapters keep source-architecture details out of the generic builder. They
translate GGUF metadata and tensor records into Mobius config and initializer
contracts while the builder owns parsing, graph construction, and weight
application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from mobius._configs import ArchitectureConfig, QuantizationConfig
    from mobius._model_package import ModelPackage
    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.integrations.gguf._repacker import RepackedTensor


@dataclass(frozen=True)
class GGUFTensorTarget:
    """One state-dict and ONNX target produced from a GGUF source tensor."""

    state_dict_name: str
    initializer_name: str
    source_index: int | None = None


@dataclass(frozen=True)
class GGUFTensorMapping:
    """Disposition of one GGUF tensor."""

    targets: tuple[GGUFTensorTarget, ...] = ()
    exclusion: str | None = None

    def __post_init__(self) -> None:
        if bool(self.targets) == bool(self.exclusion):
            raise ValueError("A GGUF tensor mapping must have targets or one exclusion")

    @classmethod
    def excluded(cls, reason: str) -> GGUFTensorMapping:
        return cls(exclusion=reason)


@dataclass
class GGUFMappingAudit:
    """Completeness accounting collected while loading a GGUF tensor table."""

    mapped_sources: set[str] = field(default_factory=set)
    excluded_sources: dict[str, str] = field(default_factory=dict)
    unmapped_sources: set[str] = field(default_factory=set)
    target_sources: dict[str, str] = field(default_factory=dict)

    def record(self, source_name: str, mapping: GGUFTensorMapping | None) -> None:
        if mapping is None:
            self.unmapped_sources.add(source_name)
            return
        if mapping.exclusion is not None:
            self.excluded_sources[source_name] = mapping.exclusion
            return

        self.mapped_sources.add(source_name)
        for target in mapping.targets:
            previous = self.target_sources.setdefault(target.initializer_name, source_name)
            if previous != source_name:
                raise ValueError(
                    f"GGUF sources {previous!r} and {source_name!r} both map to "
                    f"initializer {target.initializer_name!r}"
                )


class GGUFArchitectureAdapter:
    """Base contract for source-architecture-specific GGUF behavior."""

    architecture: str
    model_type: str

    def __init__(self, model: GGUFModel) -> None:
        self.model = model

    def validate_model(self, *, source: str) -> None:
        """Validate the source tensor table and supported quantization."""
        raise NotImplementedError

    def build_config(self) -> ArchitectureConfig:
        raise NotImplementedError

    def quantization_config(self) -> QuantizationConfig:
        raise NotImplementedError

    def map_tensor(
        self,
        source_name: str,
        source_shape: tuple[int, ...],
    ) -> GGUFTensorMapping | None:
        raise NotImplementedError

    def transform_tensor(
        self,
        source_name: str,
        target: GGUFTensorTarget,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        return tensor

    def transform_repacked(
        self,
        source_name: str,
        target: GGUFTensorTarget,
        tensor: RepackedTensor,
    ) -> RepackedTensor:
        return tensor

    def validate_mapping_audit(self, audit: GGUFMappingAudit) -> None:
        if audit.unmapped_sources:
            examples = sorted(audit.unmapped_sources)[:10]
            raise ValueError(
                f"{self.architecture} GGUF has {len(audit.unmapped_sources)} "
                f"unmapped source tensor(s): {examples}"
            )


_ADAPTER_TYPES: dict[str, type[GGUFArchitectureAdapter]] = {}
_BUILTINS_LOADED = False
_BUILTIN_MODULES = ("mobius.integrations.gguf._nemotron_h_moe",)


def register_architecture_adapter(
    adapter_type: type[GGUFArchitectureAdapter],
) -> type[GGUFArchitectureAdapter]:
    """Register an architecture adapter class."""
    architecture = adapter_type.architecture
    if architecture in _ADAPTER_TYPES:
        raise ValueError(f"GGUF architecture adapter already registered: {architecture!r}")
    _ADAPTER_TYPES[architecture] = adapter_type
    return adapter_type


def _load_builtin_adapters() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    for module_name in _BUILTIN_MODULES:
        import_module(module_name)
    _BUILTINS_LOADED = True


def create_architecture_adapter(
    architecture: str,
    model: GGUFModel,
) -> GGUFArchitectureAdapter | None:
    """Create the registered adapter for *architecture*, if one exists."""
    _load_builtin_adapters()
    adapter_type = _ADAPTER_TYPES.get(architecture)
    return adapter_type(model) if adapter_type is not None else None


def validate_package_state_dict(
    package: ModelPackage,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Require exact state-dict coverage of all unset package initializers."""
    required: set[str] = set()
    for model in package.values():
        required.update(
            name
            for name, initializer in model.graph.initializers.items()
            if initializer.const_value is None
        )

    provided = set(state_dict)
    missing = required - provided
    unexpected = provided - required
    if not missing and not unexpected:
        return

    details = []
    if missing:
        details.append(f"{len(missing)} missing: {sorted(missing)[:10]}")
    if unexpected:
        details.append(f"{len(unexpected)} unexpected: {sorted(unexpected)[:10]}")
    raise ValueError("GGUF state-dict/initializer mismatch; " + "; ".join(details))
