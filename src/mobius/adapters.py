# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model-agnostic low-rank adapter artifacts and per-request selection state."""

from __future__ import annotations

__all__ = [
    "AdapterApplication",
    "AdapterArtifact",
    "AdapterBatchSelection",
    "AdapterRowSelection",
    "AdapterSelectionTensors",
    "AdapterServiceOptions",
    "AdapterSource",
    "AdapterTarget",
    "AdapterTargetDescriptor",
    "AdapterTargetManifest",
    "AdapterTargetSlice",
    "AdapterWeights",
    "compose_adapter_deltas",
    "fingerprint_model_weights",
]

import dataclasses
import hashlib
import itertools
import json
import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import onnx_ir as ir
import rfc8785


def _validate_identifier(value: str, description: str) -> None:
    if not value or "/" in value or "\\" in value:
        raise ValueError(f"{description} must be a non-empty path segment")


def _tensor_bytes(tensor: ir.Tensor) -> bytes:
    array = np.ascontiguousarray(tensor.numpy())
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and sys.byteorder == "big"
    ):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
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


def _canonical_attribute_value(attribute: ir.Attr) -> object:
    value = attribute.value
    if attribute.type in {
        ir.AttributeType.FLOAT,
        ir.AttributeType.INT,
        ir.AttributeType.STRING,
    }:
        return value
    if attribute.type in {
        ir.AttributeType.FLOATS,
        ir.AttributeType.INTS,
        ir.AttributeType.STRINGS,
    }:
        return list(value)
    if attribute.type == ir.AttributeType.TENSOR:
        return {
            "dtype": int(value.dtype),
            "shape": [int(dimension) for dimension in value.shape],
            "sha256": hashlib.sha256(_tensor_bytes(value)).hexdigest(),
        }
    if attribute.type == ir.AttributeType.TENSORS:
        return [
            {
                "dtype": int(tensor.dtype),
                "shape": [int(dimension) for dimension in tensor.shape],
                "sha256": hashlib.sha256(_tensor_bytes(tensor)).hexdigest(),
            }
            for tensor in value
        ]
    raise ValueError(
        f"cannot canonicalize adapter target consumer attribute "
        f"{attribute.name!r} of type {attribute.type.name}"
    )


def _target_fingerprint_record(
    models: Mapping[str, ir.Model],
    target: AdapterTarget,
) -> dict[str, object]:
    model = models.get(target.component)
    if model is None:
        raise ValueError(f"cannot fingerprint unknown component {target.component!r}")
    initializer = model.graph.initializers.get(target.parameter)
    if initializer is None:
        raise ValueError(
            f"cannot fingerprint unknown parameter {target.component!r}/{target.parameter!r}"
        )
    if initializer.const_value is None:
        raise ValueError(
            f"cannot fingerprint unloaded initializer "
            f"{target.component!r}/{target.parameter!r}"
        )
    consumers: list[dict[str, object]] = []
    for node_ordinal, node in enumerate(model.graph):
        for input_ordinal, node_input in enumerate(node.inputs):
            if node_input is not initializer:
                continue
            consumers.append(
                {
                    "attributes": {
                        name: _canonical_attribute_value(attribute)
                        for name, attribute in sorted(node.attributes.items())
                    },
                    "domain": node.domain or "ai.onnx",
                    "input_ordinal": input_ordinal,
                    "node_ordinal": node_ordinal,
                    "op_type": node.op_type,
                }
            )
    return {
        "component": target.component,
        "consumers": consumers,
        "dtype": int(initializer.dtype),
        "parameter": target.parameter,
        "shape": [int(dimension) for dimension in initializer.shape],
        "tensor_sha256": hashlib.sha256(_tensor_bytes(initializer.const_value)).hexdigest(),
    }


def fingerprint_model_weights(
    models: Mapping[str, ir.Model],
    targets: Sequence[AdapterTarget] | None = None,
) -> str:
    """Fingerprint the exact immutable base parameters targeted by adapters."""
    if targets is None:
        targets = tuple(
            AdapterTarget(component, parameter)
            for component, model in sorted(models.items())
            for parameter in sorted(model.graph.initializers)
        )
    unique_targets = sorted(set(targets), key=lambda item: (item.component, item.parameter))
    if not unique_targets:
        raise ValueError("adapter base fingerprint requires at least one target")
    canonical = rfc8785.dumps(
        {
            "schema": "onnx-genai-targeted-base-v1",
            "targets": [
                _target_fingerprint_record(models, target) for target in unique_targets
            ],
        }
    )
    return f"onnx-genai-targeted-base-v1:sha256:{hashlib.sha256(canonical).hexdigest()}"


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
class AdapterTargetSlice:
    """One semantic child occupying a contiguous fused-projection output slice."""

    role: str
    offset: int
    width: int
    rank: int | None = None
    alpha: float | None = None

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("adapter target slice role must be non-empty")
        if self.offset < 0 or self.width <= 0:
            raise ValueError("adapter target slice offset/width must be non-negative/positive")
        if self.rank is not None and self.rank <= 0:
            raise ValueError("adapter target slice rank must be positive")
        if self.alpha is not None and not math.isfinite(self.alpha):
            raise ValueError("adapter target slice alpha must be finite")


@dataclasses.dataclass(frozen=True)
class AdapterTargetDescriptor:
    """Authoritative producer binding from a semantic target to an exact ONNX value."""

    target: AdapterTarget
    semantic_name: str
    node_name: str
    output_name: str
    input_size: int
    output_size: int
    layer_index: int | None = None
    rank: int | None = None
    alpha: float | None = None
    slices: tuple[AdapterTargetSlice, ...] = ()

    def __post_init__(self) -> None:
        if not self.semantic_name or not self.node_name or not self.output_name:
            raise ValueError("adapter semantic, node, and output names must be non-empty")
        if self.input_size <= 0 or self.output_size <= 0:
            raise ValueError("adapter target input/output dimensions must be positive")
        if self.layer_index is not None and self.layer_index < 0:
            raise ValueError("adapter target layer index must be non-negative")
        if self.rank is not None and self.rank <= 0:
            raise ValueError("adapter target rank must be positive")
        if self.alpha is not None and not math.isfinite(self.alpha):
            raise ValueError("adapter target alpha must be finite")
        roles = [item.role for item in self.slices]
        if len(roles) != len(set(roles)):
            raise ValueError("adapter target slice roles must be unique")
        ordered = sorted(self.slices, key=lambda item: item.offset)
        for previous, current in itertools.pairwise(ordered):
            if previous.offset + previous.width > current.offset:
                raise ValueError("adapter target slices must not overlap")
        if ordered and ordered[-1].offset + ordered[-1].width > self.output_size:
            raise ValueError("adapter target slice exceeds the projection output dimension")


@dataclasses.dataclass(frozen=True)
class AdapterTargetManifest:
    """Authoritative model-export target map; runtimes need no family discovery."""

    base_fingerprint: str
    targets: tuple[AdapterTargetDescriptor, ...]

    def __post_init__(self) -> None:
        if not self.base_fingerprint:
            raise ValueError("adapter target manifest base fingerprint must be non-empty")
        if not self.targets:
            raise ValueError("adapter target manifest must contain at least one target")
        bindings = [descriptor.target for descriptor in self.targets]
        if len(bindings) != len(set(bindings)):
            raise ValueError("adapter target manifest contains duplicate bindings")
        semantics = [descriptor.semantic_name for descriptor in self.targets]
        if len(semantics) != len(set(semantics)):
            raise ValueError("adapter target manifest contains duplicate semantic names")

    def validate(self, models: Mapping[str, ir.Model]) -> None:
        for descriptor in self.targets:
            model = models.get(descriptor.target.component)
            if model is None:
                raise ValueError(
                    f"adapter manifest targets unknown component "
                    f"{descriptor.target.component!r}"
                )
            initializer = model.graph.initializers.get(descriptor.target.parameter)
            if initializer is None:
                raise ValueError(
                    f"adapter manifest targets unknown parameter "
                    f"{descriptor.target.component!r}/{descriptor.target.parameter!r}"
                )
            shape = [int(dimension) for dimension in initializer.shape]
            expected_shape = [descriptor.output_size, descriptor.input_size]
            if shape != expected_shape:
                raise ValueError(
                    f"adapter manifest parameter {descriptor.target.parameter!r} "
                    f"has shape {shape}, expected {expected_shape}"
                )
            nodes = [node for node in model.graph if node.name == descriptor.node_name]
            if len(nodes) != 1:
                raise ValueError(
                    f"adapter manifest node {descriptor.node_name!r} resolved "
                    f"{len(nodes)} times"
                )
            if initializer not in nodes[0].inputs:
                raise ValueError(
                    f"adapter manifest node {descriptor.node_name!r} does not consume "
                    f"parameter {descriptor.target.parameter!r}"
                )
            if descriptor.output_name not in {
                output.name for output in nodes[0].outputs if output.name is not None
            }:
                raise ValueError(
                    f"adapter manifest node {descriptor.node_name!r} does not produce "
                    f"{descriptor.output_name!r}"
                )
        actual_fingerprint = fingerprint_model_weights(
            models, tuple(descriptor.target for descriptor in self.targets)
        )
        if actual_fingerprint != self.base_fingerprint:
            raise ValueError(
                "adapter target manifest base fingerprint mismatch: "
                f"expected {self.base_fingerprint}, got {actual_fingerprint}"
            )

    @property
    def bindings(self) -> frozenset[AdapterTarget]:
        """Exact base parameters covered by the manifest."""
        return frozenset(descriptor.target for descriptor in self.targets)


@dataclasses.dataclass(frozen=True)
class AdapterSource:
    """Artifact provenance retained independently from runtime container choice."""

    format: Literal["in_memory", "peft_safetensors", "onnx_adapter"]
    path: str | None = None
    checksum: str | None = None
    base_model: str | None = None
    revision: str | None = None
    native_parameters: tuple[tuple[AdapterTarget, str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.format != "in_memory" and not self.path:
            raise ValueError(f"{self.format} adapter source requires a path")
        if self.checksum is not None and not self.checksum.startswith("sha256:"):
            raise ValueError("adapter source checksum must use sha256")
        targets = [target for target, _, _ in self.native_parameters]
        if len(targets) != len(set(targets)):
            raise ValueError("adapter source contains duplicate native parameter bindings")
        if any(not a or not b or a == b for _, a, b in self.native_parameters):
            raise ValueError(
                "adapter native parameters must contain distinct non-empty A/B names"
            )


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
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("adapter alpha must be finite and greater than zero")

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
    source: AdapterSource = dataclasses.field(
        default_factory=lambda: AdapterSource("in_memory")
    )
    identity: str | None = None
    version: str = "1"

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "adapter name")
        if not self.base_fingerprint:
            raise ValueError("adapter base fingerprint must be non-empty")
        if self.identity is not None and not self.identity:
            raise ValueError("adapter identity must be non-empty")
        if not self.version:
            raise ValueError("adapter version must be non-empty")
        if not self.weights:
            raise ValueError("adapter artifact must contain at least one target")
        targets = [weight.target for weight in self.weights]
        if len(targets) != len(set(targets)):
            raise ValueError("adapter artifact contains duplicate targets")

    @property
    def checksum(self) -> str:
        """Deterministic checksum covering bindings, metadata, and tensors."""
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

    @property
    def nbytes(self) -> int:
        """Resident tensor bytes required by this artifact's factor pages."""
        return sum(
            weight.a.numpy().nbytes + weight.b.numpy().nbytes for weight in self.weights
        )

    def validate_base(
        self,
        models: Mapping[str, ir.Model],
        *,
        fingerprint_targets: Sequence[AdapterTarget] | None = None,
    ) -> None:
        """Validate fingerprint, target existence, dtype, and matrix dimensions."""
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
        actual_fingerprint = fingerprint_model_weights(
            models,
            fingerprint_targets or tuple(weight.target for weight in self.weights),
        )
        if actual_fingerprint != self.base_fingerprint:
            raise ValueError(
                f"adapter {self.name!r} base fingerprint mismatch: "
                f"expected {self.base_fingerprint}, got {actual_fingerprint}"
            )

    @property
    def target_bindings(self) -> frozenset[AdapterTarget]:
        """Exact base parameters modified by this artifact."""
        return frozenset(weight.target for weight in self.weights)

    @property
    def stable_identity(self) -> str:
        """Stable identity used independently from the package catalog alias."""
        return self.identity or self.name

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
        if not math.isfinite(self.scale) or not -16.0 <= self.scale <= 16.0:
            raise ValueError("adapter application scale must be finite and within [-16, 16]")


@dataclasses.dataclass(frozen=True)
class AdapterServiceOptions:
    """Producer-neutral runtime lifecycle, planning, and artifact format options."""

    row_ids: str | None = None
    request_epochs: str | None = None
    adapter_ids: str = "request.adapter_ids"
    adapter_counts: str = "request.adapter_counts"
    scales: str = "request.adapter_scales"
    active: str | None = None
    max_adapters: int = 4
    application_capability: str = "onnx-genai.adapters@1"
    portable_fallback: bool = True
    cache_max_entries: int = 16
    bucket_by_adapter_set: bool = True
    stable_buffers: bool = True
    invalidate_capture_on_eviction: bool = True
    preserve_source_format: bool = False

    def __post_init__(self) -> None:
        if not self.application_capability:
            raise ValueError("adapter application capability must be non-empty")
        if self.cache_max_entries <= 0:
            raise ValueError("adapter cache max_entries must be greater than zero")
        if self.max_adapters <= 0:
            raise ValueError("adapter max_adapters must be greater than zero")
        if self.portable_fallback and self.preserve_source_format:
            raise ValueError(
                "portable fallback requires portable JSON artifacts; "
                "source-format preservation requires a native adapter capability"
            )


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
            raise ValueError("adapter row contains duplicate adapter")


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

    def to_tensors(
        self,
        artifacts: Mapping[str, AdapterArtifact],
        *,
        max_adapters: int,
        active: Sequence[bool] | None = None,
    ) -> AdapterSelectionTensors:
        """Lower aliases to fixed-shape request tensors without serializing numeric IDs."""
        if max_adapters <= 0:
            raise ValueError("adapter max_adapters must be greater than zero")
        if active is None:
            active = [True] * len(self.rows)
        if len(active) != len(self.rows):
            raise ValueError("adapter active rows must match the selection batch size")
        self.validate_catalog(artifacts)
        aliases = tuple(sorted(artifacts))
        indices = {alias: index for index, alias in enumerate(aliases)}
        adapter_ids = np.full((len(self.rows), max_adapters), -1, dtype=np.int64)
        scales = np.zeros((len(self.rows), max_adapters), dtype=np.float32)
        counts = np.zeros((len(self.rows),), dtype=np.int64)
        for row_index, (row, is_active) in enumerate(zip(self.rows, active)):
            if not is_active:
                continue
            if len(row.adapters) > max_adapters:
                raise ValueError(
                    f"adapter row {row.row_id} selects {len(row.adapters)} adapters, "
                    f"exceeding max_adapters {max_adapters}"
                )
            counts[row_index] = len(row.adapters)
            for slot, application in enumerate(row.adapters):
                adapter_ids[row_index, slot] = indices[application.adapter]
                scales[row_index, slot] = application.scale
        return AdapterSelectionTensors(
            row_ids=np.asarray([row.row_id for row in self.rows], dtype=np.int64),
            request_epochs=np.asarray(
                [row.request_epoch for row in self.rows], dtype=np.int64
            ),
            adapter_ids=adapter_ids,
            adapter_counts=counts,
            scales=scales,
            active=np.asarray(active, dtype=np.bool_),
            aliases=aliases,
        )

    @property
    def referenced_adapters(self) -> frozenset[str]:
        """Live adapter set that a paged runtime must pin against eviction."""
        return frozenset(
            application.adapter for row in self.rows for application in row.adapters
        )


@dataclasses.dataclass(frozen=True)
class AdapterSelectionTensors:
    """Fixed-shape SSA request buffers for the ``onnx-genai.adapters@1`` ABI."""

    row_ids: np.ndarray
    request_epochs: np.ndarray
    adapter_ids: np.ndarray
    adapter_counts: np.ndarray
    scales: np.ndarray
    active: np.ndarray
    aliases: tuple[str, ...]


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
