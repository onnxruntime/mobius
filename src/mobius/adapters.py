# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model-agnostic low-rank adapter artifacts and per-request selection state."""

from __future__ import annotations

__all__ = [
    "AdapterApplication",
    "AdapterArtifact",
    "AdapterBatchSelection",
    "AdapterRowSelection",
    "AdapterTarget",
    "AdapterWeights",
    "compose_adapter_deltas",
    "fingerprint_model_weights",
]

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import onnx_ir as ir


def _validate_identifier(value: str, description: str) -> None:
    if not value or "/" in value or "\\" in value:
        raise ValueError(f"{description} must be a non-empty path segment")


def _tensor_bytes(tensor: ir.Tensor) -> bytes:
    array = np.ascontiguousarray(tensor.numpy())
    return array.tobytes(order="C")


def _update_tensor_hash(digest: Any, tensor: ir.Tensor) -> None:
    shape = [int(dimension) for dimension in tensor.shape]
    digest.update(
        json.dumps(
            {"dtype": tensor.dtype.name, "shape": shape},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(b"\0")
    digest.update(_tensor_bytes(tensor))


def fingerprint_model_weights(models: Mapping[str, ir.Model]) -> str:
    """Return a deterministic SHA-256 fingerprint of loaded base-model weights.

    Component and initializer names are part of the digest, so an adapter cannot
    silently bind to an equal-shaped parameter in a different model component.
    """
    digest = hashlib.sha256()
    for component_name, model in sorted(models.items()):
        digest.update(component_name.encode())
        digest.update(b"\0")
        for parameter_name, initializer in sorted(model.graph.initializers.items()):
            if initializer.const_value is None:
                raise ValueError(
                    f"cannot fingerprint unloaded initializer "
                    f"{component_name!r}/{parameter_name!r}"
                )
            digest.update(parameter_name.encode())
            digest.update(b"\0")
            _update_tensor_hash(digest, initializer.const_value)
            digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


@dataclasses.dataclass(frozen=True)
class AdapterTarget:
    """A base-model parameter addressed without architecture-specific aliases."""

    component: str
    parameter: str

    def __post_init__(self) -> None:
        _validate_identifier(self.component, "adapter target component")
        if not self.parameter:
            raise ValueError("adapter target parameter must be non-empty")


@dataclasses.dataclass(frozen=True)
class AdapterWeights:
    """LoRA factors for one target, computing ``B @ A * alpha / rank``."""

    target: AdapterTarget
    a: ir.Tensor
    b: ir.Tensor
    alpha: float

    def __post_init__(self) -> None:
        if len(self.a.shape) != 2 or len(self.b.shape) != 2:
            raise ValueError("adapter A and B factors must both be rank-2 tensors")
        if int(self.a.shape[0]) <= 0:
            raise ValueError("adapter rank must be positive")
        if int(self.b.shape[1]) != int(self.a.shape[0]):
            raise ValueError(
                "adapter B input dimension must equal adapter A rank "
                f"({self.b.shape[1]} != {self.a.shape[0]})"
            )
        if self.a.dtype != self.b.dtype:
            raise ValueError("adapter A and B factors must have the same dtype")
        if not math.isfinite(self.alpha):
            raise ValueError("adapter alpha must be finite")

    @property
    def rank(self) -> int:
        return int(self.a.shape[0])

    @property
    def dtype(self) -> ir.DataType:
        return self.a.dtype

    def delta(self) -> np.ndarray:
        """Materialize the reference LoRA update for validation and parity tests."""
        return (self.b.numpy() @ self.a.numpy()) * (self.alpha / self.rank)


@dataclasses.dataclass(frozen=True)
class AdapterArtifact:
    """Immutable low-rank adapter data bound to one exact base-model fingerprint."""

    name: str
    base_fingerprint: str
    weights: tuple[AdapterWeights, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "adapter name")
        if not self.base_fingerprint:
            raise ValueError("adapter base fingerprint must be non-empty")
        if not self.weights:
            raise ValueError("adapter artifact must contain at least one target")
        targets = [weight.target for weight in self.weights]
        if len(targets) != len(set(targets)):
            raise ValueError("adapter artifact contains duplicate targets")

    @property
    def checksum(self) -> str:
        """Return a deterministic checksum covering bindings, metadata, and tensors."""
        digest = hashlib.sha256()
        digest.update(self.name.encode())
        digest.update(b"\0")
        digest.update(self.base_fingerprint.encode())
        for weight in sorted(
            self.weights, key=lambda item: (item.target.component, item.target.parameter)
        ):
            metadata = {
                "alpha": weight.alpha,
                "component": weight.target.component,
                "parameter": weight.target.parameter,
                "rank": weight.rank,
            }
            digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\0")
            _update_tensor_hash(digest, weight.a)
            _update_tensor_hash(digest, weight.b)
        return f"sha256:{digest.hexdigest()}"

    def validate_base(self, models: Mapping[str, ir.Model]) -> None:
        """Validate fingerprint, target existence, dtype, and matrix dimensions."""
        actual_fingerprint = fingerprint_model_weights(models)
        if actual_fingerprint != self.base_fingerprint:
            raise ValueError(
                f"adapter {self.name!r} base fingerprint mismatch: "
                f"expected {self.base_fingerprint}, got {actual_fingerprint}"
            )
        for weight in self.weights:
            model = models.get(weight.target.component)
            if model is None:
                raise ValueError(
                    f"adapter {self.name!r} targets unknown component "
                    f"{weight.target.component!r}"
                )
            initializer = model.graph.initializers.get(weight.target.parameter)
            if initializer is None:
                raise ValueError(
                    f"adapter {self.name!r} targets unknown parameter "
                    f"{weight.target.component!r}/{weight.target.parameter!r}"
                )
            expected_shape = [int(weight.b.shape[0]), int(weight.a.shape[1])]
            actual_shape = [int(dimension) for dimension in initializer.shape]
            if actual_shape != expected_shape:
                raise ValueError(
                    f"adapter target {weight.target.component!r}/"
                    f"{weight.target.parameter!r} has shape {actual_shape}, "
                    f"but B @ A has shape {expected_shape}"
                )
            if initializer.dtype != weight.dtype:
                raise ValueError(
                    f"adapter target {weight.target.component!r}/"
                    f"{weight.target.parameter!r} has dtype {initializer.dtype.name}, "
                    f"but adapter factors have dtype {weight.dtype.name}"
                )

    def validate_checksum(self, expected: str) -> None:
        """Reject corrupted or substituted adapter tensor data."""
        if self.checksum != expected:
            raise ValueError(
                f"adapter {self.name!r} checksum mismatch: "
                f"expected {expected}, got {self.checksum}"
            )


@dataclasses.dataclass(frozen=True)
class AdapterApplication:
    """One adapter and its request-local scale in composition order."""

    adapter: str
    scale: float = 1.0

    def __post_init__(self) -> None:
        _validate_identifier(self.adapter, "adapter application name")
        if not math.isfinite(self.scale):
            raise ValueError("adapter application scale must be finite")


@dataclasses.dataclass(frozen=True)
class AdapterRowSelection:
    """Adapter composition for one stable semantic request row."""

    row_id: int
    request_epoch: int
    adapters: tuple[AdapterApplication, ...] = ()

    def __post_init__(self) -> None:
        if self.request_epoch < 0:
            raise ValueError("adapter request epoch must be non-negative")
        names = [application.adapter for application in self.adapters]
        if len(names) != len(set(names)):
            raise ValueError("an adapter may appear at most once in a row composition")


@dataclasses.dataclass(frozen=True)
class AdapterBatchSelection:
    """Fixed-shape, compaction-safe adapter state for a heterogeneous batch."""

    rows: tuple[AdapterRowSelection, ...]

    def __post_init__(self) -> None:
        row_ids = [row.row_id for row in self.rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("adapter batch row IDs must be unique")

    def validate_catalog(self, artifacts: Mapping[str, AdapterArtifact]) -> None:
        for row in self.rows:
            for application in row.adapters:
                if application.adapter not in artifacts:
                    raise ValueError(
                        f"row {row.row_id} selects unknown adapter {application.adapter!r}"
                    )

    def compact(self, permutation: Sequence[int]) -> AdapterBatchSelection:
        """Apply the same physical-row permutation used for all workflow state."""
        if sorted(permutation) != list(range(len(self.rows))):
            raise ValueError("adapter compaction must be a permutation of all batch rows")
        return AdapterBatchSelection(tuple(self.rows[index] for index in permutation))


def compose_adapter_deltas(
    row: AdapterRowSelection,
    artifacts: Mapping[str, AdapterArtifact],
) -> dict[AdapterTarget, np.ndarray]:
    """Compose a row's selected adapters into reference parameter updates."""
    deltas: dict[AdapterTarget, np.ndarray] = {}
    for application in row.adapters:
        try:
            artifact = artifacts[application.adapter]
        except KeyError as error:
            raise ValueError(
                f"row {row.row_id} selects unknown adapter {application.adapter!r}"
            ) from error
        for weight in artifact.weights:
            update = weight.delta() * application.scale
            if weight.target in deltas:
                deltas[weight.target] = deltas[weight.target] + update
            else:
                deltas[weight.target] = update
    return deltas
