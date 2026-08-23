# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Preserve compatible GGUF tensor payloads as ONNX external data."""

from __future__ import annotations

__all__ = ["GGUFReuseCandidate", "GGUFReusePlan", "verify_gguf_reuse_manifest"]

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

import onnx_ir as ir

if TYPE_CHECKING:
    from mobius._model_package import ModelPackage

_MANIFEST_NAME = "gguf-reuse.json"
_SIDECAR_NAME = "model.onnx.data"
_EXTERNAL_WEIGHT_THRESHOLD = 256
_GENERATED_NAMES = frozenset({"model.onnx", _SIDECAR_NAME, _MANIFEST_NAME})
_FLOAT_QTYPE_DTYPES = {
    "F32": ir.DataType.FLOAT,
    "F16": ir.DataType.FLOAT16,
}


@dataclass(frozen=True)
class GGUFReuseCandidate:
    """A final state-dict tensor that can read its exact bytes from the GGUF."""

    source_name: str
    offset: int
    length: int
    qtype: str


@dataclass(frozen=True)
class GGUFReuseTensor:
    """A source range bound to an ONNX initializer."""

    initializer: str
    source_name: str
    offset: int
    length: int
    qtype: str


@dataclass(frozen=True)
class GGUFReusePlan:
    """Source identity and exact tensor ranges used by a mixed ONNX package."""

    source_path: Path
    size: int
    sha256: str
    tensors: tuple[GGUFReuseTensor, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attach_reused_initializers(
    package: ModelPackage,
    source_path: str | Path,
    candidates: dict[str, GGUFReuseCandidate],
) -> None:
    """Replace eligible in-memory initializers with GGUF ExternalTensors."""
    if len(package) != 1:
        raise ValueError(
            "reuse_gguf_weights=True currently supports only single-model, flat packages."
        )

    source = Path(source_path).absolute()
    if source.is_symlink():
        raise ValueError(
            "reuse_gguf_weights=True does not accept a symlinked GGUF. "
            "Use the real GGUF file in the ONNX output directory."
        )

    model = next(iter(package.values()))
    reused: list[GGUFReuseTensor] = []
    for name, candidate in candidates.items():
        initializer = model.graph.initializers.get(name)
        if initializer is None or initializer.const_value is None:
            continue
        tensor = initializer.const_value
        expected_dtype = _FLOAT_QTYPE_DTYPES.get(candidate.qtype, ir.DataType.UINT8)
        if tensor.dtype != expected_dtype:
            continue
        if tensor.nbytes != candidate.length:
            continue
        initializer.const_value = ir.ExternalTensor(
            source.name,
            candidate.offset,
            candidate.length,
            tensor.dtype,
            shape=tensor.shape,
            name=tensor.name or name,
            base_dir=source.parent,
        )
        reused.append(
            GGUFReuseTensor(
                initializer=name,
                source_name=candidate.source_name,
                offset=candidate.offset,
                length=candidate.length,
                qtype=candidate.qtype,
            )
        )

    if not reused:
        raise ValueError(
            "reuse_gguf_weights=True found no byte-compatible tensors. "
            "This initial implementation reuses unchanged F32/F16 tensors and "
            "runtime-native IQ/MXFP4 projection blocks; transformed or repacked "
            "weights use the ONNX sidecar. Some float transforms are graph-expressible "
            "but are deferred from this first implementation."
        )

    package.gguf_reuse_plan = GGUFReusePlan(
        source_path=source,
        size=source.stat().st_size,
        sha256=_sha256(source),
        tensors=tuple(sorted(reused, key=lambda tensor: tensor.initializer)),
    )


def _validate_source(plan: GGUFReusePlan, output_directory: Path) -> None:
    source = plan.source_path
    if source.is_symlink():
        raise ValueError("The GGUF source became a symlink; refusing an unsafe external path.")
    if source.parent.resolve() != output_directory.resolve():
        raise ValueError(
            "reuse_gguf_weights=True requires flat same-directory packaging: move the "
            f"GGUF to {output_directory} before building. Mobius will not copy, hardlink, "
            "or symlink a multi-GB source file."
        )
    if source.name in _GENERATED_NAMES:
        raise ValueError(
            f"The GGUF source name {source.name!r} collides with a generated package "
            "artifact. Rename the GGUF before building."
        )
    for generated_name in _GENERATED_NAMES:
        generated_path = output_directory / generated_name
        if generated_path.exists() and os.path.samefile(source, generated_path):
            raise ValueError(
                f"The GGUF source is hard-linked to generated artifact "
                f"{generated_name!r}. Use an independent real file."
            )
    if source.stat().st_size != plan.size or _sha256(source) != plan.sha256:
        raise ValueError(
            "The GGUF source no longer matches the file used to build this package "
            "(size or SHA-256 changed). Rebuild from the intended GGUF."
        )


def save_reuse_model(
    model: ir.Model,
    path: str | Path,
    plan: GGUFReusePlan,
    *,
    callback=None,
) -> tuple[str, ...]:
    """Save mixed GGUF references plus one converted-weight sidecar."""
    path = Path(path)
    _validate_source(plan, path.parent)

    memory_initializers = [
        value
        for graph in model.graphs()
        for value in graph.initializers.values()
        if value.const_value is not None
        and not isinstance(value.const_value, ir.ExternalTensor)
        and value.const_value.nbytes > _EXTERNAL_WEIGHT_THRESHOLD
    ]
    original_tensors = [value.const_value for value in memory_initializers]
    converted_names = tuple(sorted(value.name for value in memory_initializers))

    try:
        if memory_initializers:
            external_tensors = ir.external_data.convert_tensors_to_external(
                original_tensors,
                base_dir=path.parent,
                relative_path=_SIDECAR_NAME,
                callback=callback,
            )
            for value, external_tensor in zip(
                memory_initializers, external_tensors, strict=True
            ):
                value.const_value = external_tensor
        ir.save(model, path)
    finally:
        for value, tensor in zip(memory_initializers, original_tensors, strict=True):
            value.const_value = tensor

    return converted_names


def write_reuse_manifest(
    directory: str | Path,
    plan: GGUFReusePlan,
    converted_tensors: tuple[str, ...],
) -> None:
    """Write source identity and per-tensor routing without runtime claims."""
    payload = {
        "format": "mobius.gguf-external-data.v1",
        "source": {
            "location": plan.source_path.name,
            "size": plan.size,
            "sha256": plan.sha256,
        },
        "reused_tensors": [
            {
                "initializer": tensor.initializer,
                "source_tensor": tensor.source_name,
                "offset": tensor.offset,
                "length": tensor.length,
                "qtype": tensor.qtype,
            }
            for tensor in plan.tensors
        ],
        "converted_tensors": list(converted_tensors),
        "runtime_verification": (
            "ONNX runtimes resolve location/offset/length but do not enforce this SHA-256."
        ),
    }
    manifest_path = Path(directory) / _MANIFEST_NAME
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verify_gguf_reuse_manifest(directory: str | Path) -> None:
    """Verify the packaged GGUF's size, digest, and pinned tensor ranges."""
    root = Path(directory)
    manifest = json.loads((root / _MANIFEST_NAME).read_text())
    source_info = manifest["source"]
    location = source_info["location"]
    posix = PurePosixPath(location)
    windows = PureWindowsPath(location)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.parts != (location,)
        or windows.parts != (location,)
        or location in {".", ".."}
    ):
        raise ValueError(f"Unsafe GGUF manifest location: {location!r}")
    source = root / location
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"GGUF manifest source is missing or unsafe: {source}")
    if (
        source.stat().st_size != source_info["size"]
        or _sha256(source) != source_info["sha256"]
    ):
        raise ValueError("GGUF source identity mismatch (size or SHA-256).")
    for tensor in manifest["reused_tensors"]:
        if tensor["offset"] < 0 or tensor["length"] <= 0:
            raise ValueError(f"Invalid GGUF range for {tensor['initializer']!r}.")
        if tensor["offset"] + tensor["length"] > source_info["size"]:
            raise ValueError(f"GGUF range exceeds the source for {tensor['initializer']!r}.")

    model = ir.load(root / "model.onnx")
    for tensor in manifest["reused_tensors"]:
        initializer = model.graph.initializers.get(tensor["initializer"])
        external = None if initializer is None else initializer.const_value
        if not isinstance(external, ir.ExternalTensor):
            raise ValueError(
                f"Manifest initializer {tensor['initializer']!r} is not external."
            )
        if (
            external.location != location
            or external.offset != tensor["offset"]
            or external.length != tensor["length"]
        ):
            raise ValueError(
                f"Manifest range does not match ONNX initializer "
                f"{tensor['initializer']!r}."
            )
