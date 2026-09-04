# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Typed logical weights used between checkpoint readers and model adapters."""

from __future__ import annotations

__all__ = [
    "FloatWeight",
    "PackedWeight",
    "WeightBundle",
    "WeightRecord",
]

import dataclasses
from collections.abc import Iterator, Mapping
from types import MappingProxyType

import torch

from mobius._component_manifest import ComponentDescriptor


@dataclasses.dataclass(frozen=True)
class FloatWeight:
    """One ordinary floating-point checkpoint tensor."""

    value: torch.Tensor
    source_key: str


@dataclasses.dataclass(frozen=True)
class PackedWeight:
    """One logical affine-quantized weight grouped from checkpoint sidecars."""

    qweight: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor | None
    qweight_key: str
    scales_key: str
    zero_points_key: str | None
    method: str

    @property
    def source_keys(self) -> tuple[str, ...]:
        """Checkpoint keys consumed by this logical packed weight."""
        keys = [self.qweight_key, self.scales_key]
        if self.zero_points_key is not None:
            keys.append(self.zero_points_key)
        return tuple(keys)

    def as_state_dict(self) -> dict[str, torch.Tensor]:
        """Reconstruct the source sidecars for a compatibility codec."""
        tensors = {
            self.qweight_key: self.qweight,
            self.scales_key: self.scales,
        }
        if self.zero_points_key is not None and self.zero_points is not None:
            tensors[self.zero_points_key] = self.zero_points
        return tensors


WeightStorage = FloatWeight | PackedWeight


@dataclasses.dataclass(frozen=True)
class WeightRecord:
    """A named logical weight owned by exactly one package component."""

    name: str
    component: str
    storage: WeightStorage

    @property
    def is_quantized(self) -> bool:
        """Whether this record stores an existing packed weight."""
        return isinstance(self.storage, PackedWeight)

    @property
    def source_keys(self) -> tuple[str, ...]:
        """Checkpoint keys represented by this record."""
        if isinstance(self.storage, FloatWeight):
            return (self.storage.source_key,)
        return self.storage.source_keys


@dataclasses.dataclass(frozen=True)
class WeightBundle(Mapping[str, WeightRecord]):
    """Immutable records routed to one component descriptor."""

    component: ComponentDescriptor
    records: Mapping[str, WeightRecord]

    def __post_init__(self) -> None:
        for name, record in self.records.items():
            if name != record.name:
                raise ValueError(
                    f"weight bundle key {name!r} does not match record name {record.name!r}"
                )
            if record.component != self.component.name:
                raise ValueError(
                    f"weight {name!r} belongs to component "
                    f"{record.component!r}, expected {self.component.name!r}"
                )
        object.__setattr__(self, "records", MappingProxyType(dict(self.records)))

    def __getitem__(self, name: str) -> WeightRecord:
        return self.records[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def source_keys(self) -> frozenset[str]:
        """All checkpoint keys represented by this bundle."""
        return frozenset(
            source_key for record in self.records.values() for source_key in record.source_keys
        )
