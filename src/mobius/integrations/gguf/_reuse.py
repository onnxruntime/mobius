# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Preserve compatible GGUF tensor payloads as ONNX external data."""

from __future__ import annotations

__all__ = ["GGUFReuseCandidate", "GGUFReusePlan", "verify_gguf_reuse_manifest"]

import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import onnx_ir as ir

if TYPE_CHECKING:
    from mobius._model_package import ModelPackage

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_MANIFEST_NAME = "gguf-reuse.json"
_SIDECAR_NAME = "model.onnx.data"
_TRANSACTION_NAME = ".gguf-reuse.transaction.json"
_LOCK_NAME = ".gguf-reuse.lock"
_EXTERNAL_WEIGHT_THRESHOLD = 256
_GENERATED_NAMES = frozenset(
    {"model.onnx", _SIDECAR_NAME, _MANIFEST_NAME, _TRANSACTION_NAME, _LOCK_NAME}
)
_FLOAT_QTYPE_DTYPES = {
    "F32": ir.DataType.FLOAT,
    "F16": ir.DataType.FLOAT16,
}


class _TransactionEntry(TypedDict):
    final: str
    backup: str
    had_existing: bool


class _TransactionJournal(TypedDict):
    phase: str
    managed: list[_TransactionEntry]
    staged: list[str]


@dataclass(frozen=True)
class GGUFReuseCandidate:
    """A final state-dict tensor that can read its exact bytes from the GGUF."""

    source_name: str
    offset: int
    length: int
    qtype: str
    source_shape: tuple[int, ...]
    transform: str | None = None
    transform_parameter: int | None = None


@dataclass(frozen=True)
class GGUFReuseTensor:
    """A source range bound to an ONNX initializer."""

    initializer: str
    source_name: str
    offset: int
    length: int
    qtype: str
    transform: str | None
    source_shape: tuple[int, ...]
    final_shape: tuple[int, ...]
    transform_parameter: int | None


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
        final_shape = tuple(int(dim) for dim in tensor.shape)
        expected_dtype = _FLOAT_QTYPE_DTYPES.get(candidate.qtype, ir.DataType.UINT8)
        if tensor.dtype != expected_dtype:
            continue
        if tensor.nbytes != candidate.length:
            continue
        _insert_external_transform(model.graph, initializer, candidate)
        initializer.const_value = ir.ExternalTensor(
            source.name,
            candidate.offset,
            candidate.length,
            tensor.dtype,
            shape=ir.Shape(candidate.source_shape),
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
                transform=candidate.transform,
                source_shape=candidate.source_shape,
                final_shape=final_shape,
                transform_parameter=candidate.transform_parameter,
            )
        )

    if not reused:
        raise ValueError(
            "reuse_gguf_weights=True found no byte-compatible tensors. "
            "This initial implementation reuses unchanged F32/F16 tensors and "
            "runtime-native IQ/MXFP4 projection blocks, including supported "
            "graph-expressible float transforms. Repacked or otherwise unsupported "
            "weights use the ONNX sidecar."
        )

    package.gguf_reuse_plan = GGUFReusePlan(
        source_path=source,
        size=source.stat().st_size,
        sha256=_sha256(source),
        tensors=tuple(sorted(reused, key=lambda tensor: tensor.initializer)),
    )


def _shape_initializer(graph: ir.Graph, name: str, shape: tuple[int, ...]) -> ir.Value:
    value = ir.Value(name=name)
    value.const_value = ir.tensor(np.asarray(shape, dtype=np.int64), name=name)
    value.shape = ir.Shape([len(shape)])
    value.dtype = ir.DataType.INT64
    graph.initializers[name] = value
    return value


def _insert_external_transform(
    graph: ir.Graph,
    initializer: ir.Value,
    candidate: GGUFReuseCandidate,
) -> None:
    """Reproduce a byte-preserving GGUF-to-graph float transform at runtime."""
    if candidate.transform is None:
        return
    uses = list(initializer.uses())
    if not uses:
        return
    dtype = initializer.dtype
    final_shape = initializer.shape
    assert dtype is not None and final_shape is not None
    prefix = f"{initializer.name}.gguf_reuse"
    output = ir.Value(
        name=f"{prefix}.output",
        shape=final_shape,
        type=ir.TensorType(dtype),
    )
    initializer.replace_all_uses_with(output)
    initializer.shape = ir.Shape(candidate.source_shape)

    nodes: list[ir.Node]
    if candidate.transform == "transpose":
        nodes = [
            ir.Node(
                "",
                "Transpose",
                inputs=[initializer],
                outputs=[output],
                attributes=ir.convenience.convert_attributes(
                    {"perm": list(reversed(range(len(candidate.source_shape))))}
                ),
                name=f"{prefix}.Transpose",
            )
        ]
    elif candidate.transform == "subtract_one":
        np_dtype = np.float16 if dtype == ir.DataType.FLOAT16 else np.float32
        one = ir.Value(name=f"{prefix}.one")
        one.const_value = ir.tensor(np.asarray(1, dtype=np_dtype), name=one.name)
        one.shape = ir.Shape([])
        one.dtype = dtype
        graph.initializers[one.name] = one
        nodes = [
            ir.Node(
                "",
                "Sub",
                inputs=[initializer, one],
                outputs=[output],
                name=f"{prefix}.Sub",
            )
        ]
    elif candidate.transform == "log_neg":
        negated = ir.Value(
            name=f"{prefix}.negated",
            shape=ir.Shape(candidate.source_shape),
            type=ir.TensorType(dtype),
        )
        nodes = [
            ir.Node(
                "",
                "Neg",
                inputs=[initializer],
                outputs=[negated],
                name=f"{prefix}.Neg",
            ),
            ir.Node(
                "",
                "Log",
                inputs=[negated],
                outputs=[output],
                name=f"{prefix}.Log",
            ),
        ]
    elif candidate.transform in {"reshape", "llama_qk_permute"}:
        if candidate.transform == "reshape":
            shape = _shape_initializer(
                graph,
                f"{prefix}.shape",
                tuple(int(dim) for dim in final_shape),
            )
            nodes = [
                ir.Node(
                    "",
                    "Reshape",
                    inputs=[initializer, shape],
                    outputs=[output],
                    name=f"{prefix}.Reshape",
                )
            ]
        else:
            n_head = candidate.transform_parameter
            assert n_head is not None
            dim = candidate.source_shape[0] // n_head // 2
            expanded_shape = (
                n_head,
                dim,
                2,
                *candidate.source_shape[1:],
            )
            shape1 = _shape_initializer(graph, f"{prefix}.shape1", expanded_shape)
            shape2 = _shape_initializer(
                graph,
                f"{prefix}.shape2",
                candidate.source_shape,
            )
            expanded = ir.Value(
                name=f"{prefix}.expanded",
                shape=ir.Shape(expanded_shape),
                type=ir.TensorType(dtype),
            )
            permuted_shape = (
                n_head,
                2,
                dim,
                *candidate.source_shape[1:],
            )
            permuted = ir.Value(
                name=f"{prefix}.permuted",
                shape=ir.Shape(permuted_shape),
                type=ir.TensorType(dtype),
            )
            perm = [0, 2, 1, *range(3, len(expanded_shape))]
            nodes = [
                ir.Node(
                    "",
                    "Reshape",
                    inputs=[initializer, shape1],
                    outputs=[expanded],
                    name=f"{prefix}.Reshape1",
                ),
                ir.Node(
                    "",
                    "Transpose",
                    inputs=[expanded],
                    outputs=[permuted],
                    attributes=ir.convenience.convert_attributes({"perm": perm}),
                    name=f"{prefix}.Transpose",
                ),
                ir.Node(
                    "",
                    "Reshape",
                    inputs=[permuted, shape2],
                    outputs=[output],
                    name=f"{prefix}.Reshape2",
                ),
            ]
    else:
        raise ValueError(f"Unknown GGUF external transform {candidate.transform!r}.")

    graph.insert_before(uses[0][0], nodes)


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


def _manifest_payload(
    plan: GGUFReusePlan,
    converted_tensors: tuple[str, ...],
) -> dict[str, object]:
    return {
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
                "transform": tensor.transform,
                "source_shape": list(tensor.source_shape),
                "final_shape": list(tensor.final_shape),
                "transform_parameter": tensor.transform_parameter,
            }
            for tensor in plan.tensors
        ],
        "converted_tensors": list(converted_tensors),
        "runtime_verification": (
            "ONNX runtimes resolve location/offset/length but do not enforce this SHA-256."
        ),
    }


def _open_exclusive(path: Path) -> int:
    """Create a regular file without following a pre-existing link."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags, 0o600)


def _open_lock_path(path: Path) -> int:
    """Open or create the persistent lock without accepting an entry swap."""
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags, 0o600)
    except FileExistsError:
        entry_identity = path.lstat()
        if not stat.S_ISREG(entry_identity.st_mode):
            raise ValueError(f"Unsafe GGUF lock artifact: {path}")
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened_identity = os.fstat(descriptor)
        current_identity = path.lstat()
        identities = (
            (entry_identity.st_dev, entry_identity.st_ino),
            (opened_identity.st_dev, opened_identity.st_ino),
            (current_identity.st_dev, current_identity.st_ino),
        )
        if len(set(identities)) != 1 or not stat.S_ISREG(opened_identity.st_mode):
            os.close(descriptor)
            raise ValueError(f"Unsafe GGUF lock artifact: {path}")
        return descriptor


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = _open_exclusive(path)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _require_regular_or_missing(path: Path, *, artifact: str) -> None:
    """Reject links, directories, devices, and sockets at generated paths."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise ValueError(f"Unsafe GGUF {artifact} artifact: {path}")


def _acquire_file_lock(descriptor: int, *, shared: bool = False) -> None:
    try:
        if os.name == "nt":
            os.lseek(descriptor, 0, os.SEEK_SET)
            mode = msvcrt.LK_NBRLCK if shared else msvcrt.LK_NBLCK
            msvcrt.locking(descriptor, mode, 1)
        else:
            mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
    except OSError as error:
        raise ValueError("GGUF package is locked by active writer.") from error


def _release_file_lock(descriptor: int) -> None:
    if os.name == "nt":
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def _package_lock(root: Path) -> Iterator[None]:
    """Serialize package verification, recovery, and replacement."""
    lock_path = root / _LOCK_NAME
    _require_regular_or_missing(lock_path, artifact="lock")
    descriptor = _open_lock_path(lock_path)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"Unsafe GGUF lock artifact: {lock_path}")
        # Windows byte-range locks require the byte to exist.
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\n")
            os.fsync(descriptor)
        _acquire_file_lock(descriptor)
        _fsync_directory(root)
        try:
            yield
        finally:
            _release_file_lock(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _verification_lock(root: Path) -> Iterator[None]:
    """Hold a non-mutating shared lock while reading an installed package."""
    lock_path = root / _LOCK_NAME
    _require_regular_or_missing(lock_path, artifact="lock")
    if not lock_path.exists():
        try:
            descriptor = _open_lock_path(lock_path)
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\n")
                os.fsync(descriptor)
        except PermissionError:
            # A read-only legacy package cannot admit a writer either.
            yield
            return
    else:
        entry_identity = lock_path.lstat()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags)
        opened_identity = os.fstat(descriptor)
        current_identity = lock_path.lstat()
        identities = (
            (entry_identity.st_dev, entry_identity.st_ino),
            (opened_identity.st_dev, opened_identity.st_ino),
            (current_identity.st_dev, current_identity.st_ino),
        )
        if len(set(identities)) != 1 or not stat.S_ISREG(opened_identity.st_mode):
            os.close(descriptor)
            raise ValueError(f"Unsafe GGUF lock artifact: {lock_path}")
    try:
        _acquire_file_lock(descriptor, shared=True)
        try:
            yield
        finally:
            _release_file_lock(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_artifacts(
    replacements: dict[Path, Path],
    managed_paths: tuple[Path, ...],
) -> None:
    """Install staged files with a durable rollback journal."""
    root = managed_paths[0].parent
    with _package_lock(root):
        _recover_transaction_locked(root, preserve=frozenset(replacements.values()))
        _replace_artifacts_locked(replacements, managed_paths)


def _replace_artifacts_locked(
    replacements: dict[Path, Path],
    managed_paths: tuple[Path, ...],
) -> None:
    """Install staged files while the caller owns the package lock."""
    root = managed_paths[0].parent
    _require_hard_link_backups(root, managed_paths)
    token = uuid.uuid4().hex
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    transaction_path = root / _TRANSACTION_NAME
    committed_path = root / f".{_TRANSACTION_NAME}.{token}.tmp"
    _require_regular_or_missing(transaction_path, artifact="transaction journal")
    if transaction_path.exists():
        raise ValueError(f"Unexpected GGUF transaction journal: {transaction_path}")
    for final_path in managed_paths:
        _require_regular_or_missing(final_path, artifact="managed")
    for staged_path in replacements.values():
        _require_regular_or_missing(staged_path, artifact="staged")
        if not staged_path.is_file():
            raise ValueError(f"Missing GGUF staged artifact: {staged_path}")
    journal: _TransactionJournal = {
        "phase": "replacing",
        "managed": [
            {
                "final": path.name,
                "backup": f".{path.name}.{token}.backup",
                "had_existing": path.exists(),
            }
            for path in managed_paths
        ],
        "staged": [path.name for path in replacements.values()],
    }
    _publish_json(transaction_path, journal, token)
    try:
        for entry, final_path in zip(journal["managed"], managed_paths, strict=True):
            if final_path.exists():
                backup = root / entry["backup"]
                _require_regular_or_missing(backup, artifact="backup")
                os.link(final_path, backup)
                backups[final_path] = backup
        _fsync_directory(root)
        for final_path in managed_paths:
            if final_path not in replacements:
                final_path.unlink(missing_ok=True)
                installed.append(final_path)
        for final_path, staged_path in replacements.items():
            os.replace(staged_path, final_path)
            installed.append(final_path)
        _fsync_directory(root)
        committed_journal = dict(journal)
        committed_journal["phase"] = "committed"
        _write_json_exclusive(committed_path, committed_journal)
        os.replace(committed_path, transaction_path)
        _fsync_directory(root)
    except Exception:
        for final_path in reversed(installed):
            final_path.unlink(missing_ok=True)
        for final_path, backup in reversed(tuple(backups.items())):
            os.replace(backup, final_path)
            backup.unlink(missing_ok=True)
        transaction_path.unlink(missing_ok=True)
        committed_path.unlink(missing_ok=True)
        _fsync_directory(root)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
        _fsync_directory(root)
        transaction_path.unlink()
        _fsync_directory(root)


def _require_hard_link_backups(root: Path, managed_paths: tuple[Path, ...]) -> None:
    """Fail before a transaction when durable same-directory backups are unavailable."""
    if not any(path.exists() for path in managed_paths):
        return
    token = uuid.uuid4().hex
    probe = root / f".gguf-reuse.{token}.link-probe"
    linked = root / f".gguf-reuse.{token}.link-probe-copy"
    try:
        descriptor = _open_exclusive(probe)
        os.write(descriptor, b"probe")
        os.fsync(descriptor)
        os.close(descriptor)
        _require_regular_or_missing(linked, artifact="link probe")
        _fsync_file(probe)
        os.link(probe, linked)
        _fsync_directory(root)
    except OSError as error:
        raise ValueError(
            "Atomic GGUF package overwrite requires same-directory hard-link "
            "support for rollback backups. Save to a filesystem that supports "
            "hard links or choose a new empty output directory."
        ) from error
    finally:
        linked.unlink(missing_ok=True)
        probe.unlink(missing_ok=True)


def _recover_transaction(root: Path) -> None:
    """Roll back an interrupted artifact replacement before using the package."""
    with _package_lock(root):
        _recover_transaction_locked(root)


def _recover_transaction_locked(
    root: Path, *, preserve: frozenset[Path] = frozenset()
) -> None:
    """Roll back an interrupted replacement while owning the package lock."""
    transaction_path = root / _TRANSACTION_NAME
    _require_regular_or_missing(transaction_path, artifact="transaction journal")
    if not transaction_path.exists():
        _cleanup_stale_temporary_artifacts_locked(root, preserve=preserve)
        return
    journal = _validated_transaction_journal(transaction_path)
    if journal["phase"] == "committed":
        for entry in journal["managed"]:
            backup = root / entry["backup"]
            _require_regular_or_missing(backup, artifact="backup")
            backup.unlink(missing_ok=True)
        for staged_name in journal["staged"]:
            staged_path = root / staged_name
            _require_regular_or_missing(staged_path, artifact="staged")
            staged_path.unlink(missing_ok=True)
        transaction_path.unlink()
        _fsync_directory(root)
        _cleanup_stale_temporary_artifacts_locked(root)
        return
    for entry in journal["managed"]:
        final_path = root / entry["final"]
        backup = root / entry["backup"]
        _require_regular_or_missing(final_path, artifact="managed")
        _require_regular_or_missing(backup, artifact="backup")
        if backup.exists():
            os.replace(backup, final_path)
            backup.unlink(missing_ok=True)
        elif not entry["had_existing"]:
            final_path.unlink(missing_ok=True)
    for staged_name in journal["staged"]:
        staged_path = root / staged_name
        _require_regular_or_missing(staged_path, artifact="staged")
        staged_path.unlink(missing_ok=True)
    transaction_path.unlink(missing_ok=True)
    _fsync_directory(root)
    _cleanup_stale_temporary_artifacts_locked(root)


def _publish_json(path: Path, payload: Mapping[str, object], token: str) -> None:
    """Durably publish JSON without exposing an empty or partial final file."""
    temporary = path.with_name(f".{path.name}.{token}.tmp")
    _require_regular_or_missing(temporary, artifact="transaction journal temporary")
    try:
        _write_json_exclusive(temporary, payload)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _cleanup_stale_temporary_artifacts_locked(
    root: Path, *, preserve: frozenset[Path] = frozenset()
) -> None:
    """Remove only generated temporary files while no writer can be active."""
    final_names = {"model.onnx", _SIDECAR_NAME, _MANIFEST_NAME, _TRANSACTION_NAME}
    for path in root.iterdir():
        if path in preserve:
            continue
        if not any(
            re.fullmatch(rf"\.{re.escape(name)}\.[0-9a-f]{{32}}\.tmp", path.name)
            for name in final_names
        ):
            continue
        _require_regular_or_missing(path, artifact="stale temporary")
        path.unlink()
    _fsync_directory(root)


def _validated_transaction_journal(path: Path) -> _TransactionJournal:
    """Parse a journal without allowing paths outside the package directory."""
    journal = json.loads(path.read_text())
    if not isinstance(journal, dict):
        raise TypeError("Invalid GGUF transaction journal.")
    managed = journal.get("managed")
    staged = journal.get("staged")
    phase = journal.get("phase", "replacing")
    expected_finals = {"model.onnx", _SIDECAR_NAME, _MANIFEST_NAME}
    if (
        phase not in {"replacing", "committed"}
        or not isinstance(managed, list)
        or not isinstance(staged, list)
    ):
        raise TypeError("Invalid GGUF transaction journal.")
    if len(managed) != len(expected_finals):
        raise ValueError("Invalid GGUF transaction journal managed artifact set.")

    validated_managed: list[_TransactionEntry] = []
    seen_finals: set[str] = set()
    for entry in managed:
        if not isinstance(entry, dict):
            raise TypeError("Invalid GGUF transaction journal entry.")
        final = entry.get("final")
        backup = entry.get("backup")
        had_existing = entry.get("had_existing")
        if (
            not isinstance(final, str)
            or final not in expected_finals
            or final in seen_finals
            or not isinstance(backup, str)
            or re.fullmatch(
                rf"\.{re.escape(final)}\.[0-9a-f]{{32}}\.backup",
                backup,
            )
            is None
            or not isinstance(had_existing, bool)
        ):
            raise ValueError("Unsafe GGUF transaction journal entry.")
        seen_finals.add(final)
        validated_managed.append(
            {"final": final, "backup": backup, "had_existing": had_existing}
        )
    if seen_finals != expected_finals:
        raise ValueError("Invalid GGUF transaction journal managed artifact set.")

    validated_staged: list[str] = []
    staged_finals: set[str] = set()
    for staged_name in staged:
        if not isinstance(staged_name, str) or staged_name in validated_staged:
            raise ValueError("Unsafe GGUF transaction staged artifact.")
        matched_final = next(
            (
                final
                for final in expected_finals
                if re.fullmatch(
                    rf"\.{re.escape(final)}\.[0-9a-f]{{32}}\.tmp",
                    staged_name,
                )
            ),
            None,
        )
        if matched_final is None or matched_final in staged_finals:
            raise ValueError("Unsafe GGUF transaction staged artifact.")
        staged_finals.add(matched_final)
        validated_staged.append(staged_name)
    if not {"model.onnx", _MANIFEST_NAME}.issubset(staged_finals):
        raise ValueError("GGUF transaction journal omits required staged artifacts.")
    return {"phase": phase, "managed": validated_managed, "staged": validated_staged}


def save_reuse_package(
    model: ir.Model,
    path: str | Path,
    plan: GGUFReusePlan,
    *,
    callback=None,
) -> tuple[str, ...]:
    """Transactionally save mixed GGUF references plus a converted sidecar."""
    path = Path(path)
    with _package_lock(path.parent):
        _recover_transaction_locked(path.parent)
        _validate_source(plan, path.parent)
        token = uuid.uuid4().hex
        staged_model = path.with_name(f".{path.name}.{token}.tmp")
        staged_sidecar = path.with_name(f".{_SIDECAR_NAME}.{token}.tmp")
        staged_manifest = path.with_name(f".{_MANIFEST_NAME}.{token}.tmp")
        final_sidecar = path.parent / _SIDECAR_NAME
        final_manifest = path.parent / _MANIFEST_NAME
        for staged_path in (staged_model, staged_sidecar, staged_manifest):
            _require_regular_or_missing(staged_path, artifact="staged")
            if staged_path.exists():
                raise ValueError(f"Unexpected GGUF staged artifact: {staged_path}")

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
                    relative_path=staged_sidecar.name,
                    callback=callback,
                )
                for value, external_tensor in zip(
                    memory_initializers, external_tensors, strict=True
                ):
                    value.const_value = ir.ExternalTensor(
                        _SIDECAR_NAME,
                        external_tensor.offset,
                        external_tensor.length,
                        external_tensor.dtype,
                        shape=external_tensor.shape,
                        name=external_tensor.name,
                        base_dir=path.parent,
                    )
                _fsync_file(staged_sidecar)
            ir.save(model, staged_model)
            _fsync_file(staged_model)
            _write_json_exclusive(staged_manifest, _manifest_payload(plan, converted_names))
            verify_gguf_reuse_manifest(
                path.parent,
                model_path=staged_model,
                manifest_path=staged_manifest,
                sidecar_path=staged_sidecar if memory_initializers else None,
                _lock_held=True,
            )
            replacements = {
                path: staged_model,
                final_manifest: staged_manifest,
            }
            if memory_initializers:
                replacements[final_sidecar] = staged_sidecar
            _replace_artifacts_locked(
                replacements,
                (path, final_sidecar, final_manifest),
            )
        finally:
            for value, tensor in zip(memory_initializers, original_tensors, strict=True):
                value.const_value = tensor
            for staged_path in (staged_model, staged_sidecar, staged_manifest):
                _require_regular_or_missing(staged_path, artifact="staged")
                staged_path.unlink(missing_ok=True)

    return converted_names


def _safe_flat_location(location: str) -> bool:
    posix = PurePosixPath(location)
    windows = PureWindowsPath(location)
    return not (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.parts != (location,)
        or windows.parts != (location,)
        or location in {".", ".."}
    )


def verify_gguf_reuse_manifest(
    directory: str | Path,
    *,
    model_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    sidecar_path: str | Path | None = None,
    _lock_held: bool = False,
) -> None:
    """Verify the packaged GGUF's size, digest, and pinned tensor ranges."""
    root = Path(directory)
    if not _lock_held:
        recover = False
        with _verification_lock(root):
            transaction_path = root / _TRANSACTION_NAME
            _require_regular_or_missing(transaction_path, artifact="transaction journal")
            recover = (
                model_path is None
                and manifest_path is None
                and sidecar_path is None
                and transaction_path.exists()
            )
            if not recover:
                return verify_gguf_reuse_manifest(
                    root,
                    model_path=model_path,
                    manifest_path=manifest_path,
                    sidecar_path=sidecar_path,
                    _lock_held=True,
                )
        _recover_transaction(root)
        return verify_gguf_reuse_manifest(
            root,
            model_path=model_path,
            manifest_path=manifest_path,
            sidecar_path=sidecar_path,
        )
    manifest_file = Path(manifest_path) if manifest_path is not None else root / _MANIFEST_NAME
    _require_regular_or_missing(manifest_file, artifact="manifest")
    if not manifest_file.is_file():
        raise ValueError(f"GGUF reuse manifest is missing: {manifest_file}")
    manifest = json.loads(manifest_file.read_text())
    source_info = manifest["source"]
    location = source_info["location"]
    if not _safe_flat_location(location):
        raise ValueError(f"Unsafe GGUF manifest location: {location!r}")
    source = root / location
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"GGUF manifest source is missing or unsafe: {source}")
    if (
        source.stat().st_size != source_info["size"]
        or _sha256(source) != source_info["sha256"]
    ):
        raise ValueError("GGUF source identity mismatch (size or SHA-256).")
    reused_entries = manifest["reused_tensors"]
    converted_names = manifest["converted_tensors"]
    if len({entry["initializer"] for entry in reused_entries}) != len(reused_entries):
        raise ValueError("GGUF reuse manifest contains duplicate initializer routes.")
    if len(set(converted_names)) != len(converted_names):
        raise ValueError("GGUF reuse manifest contains duplicate converted tensors.")
    reused_names = {entry["initializer"] for entry in reused_entries}
    if reused_names.intersection(converted_names):
        raise ValueError("GGUF reused and converted initializer routes must be disjoint.")

    from mobius.integrations.gguf._reader import GGUFModel

    gguf_model = GGUFModel(source)
    reused_by_name = {entry["initializer"]: entry for entry in reused_entries}
    for tensor in reused_entries:
        if tensor["offset"] < 0 or tensor["length"] <= 0:
            raise ValueError(f"Invalid GGUF range for {tensor['initializer']!r}.")
        if tensor["offset"] + tensor["length"] > source_info["size"]:
            raise ValueError(f"GGUF range exceeds the source for {tensor['initializer']!r}.")
        actual_offset, actual_length, actual_qtype = gguf_model.tensor_storage_range(
            tensor["source_tensor"]
        )
        if (actual_offset, actual_length, actual_qtype) != (
            tensor["offset"],
            tensor["length"],
            tensor["qtype"],
        ):
            raise ValueError(
                f"Manifest GGUF route does not match source tensor "
                f"{tensor['source_tensor']!r}."
            )

    model_file = Path(model_path) if model_path is not None else root / "model.onnx"
    _require_regular_or_missing(model_file, artifact="model")
    if not model_file.is_file():
        raise ValueError(f"GGUF reuse model is missing: {model_file}")
    model = ir.load(model_file)
    external_by_name: dict[str, ir.ExternalTensor] = {}
    external_values_by_name: dict[str, ir.Value] = {}
    nodes_by_name: dict[str, ir.Node] = {}
    for graph in model.graphs():
        for node in graph:
            if node.name:
                nodes_by_name[node.name] = node
        for initializer in graph.initializers.values():
            external = initializer.const_value
            if not isinstance(external, ir.ExternalTensor):
                continue
            if initializer.name in external_by_name:
                raise ValueError(
                    f"External initializer name {initializer.name!r} is ambiguous."
                )
            external_by_name[initializer.name] = external
            external_values_by_name[initializer.name] = initializer
            if not _safe_flat_location(str(external.location)):
                raise ValueError(
                    f"Unsafe external initializer location: {external.location!r}."
                )

    sidecar_file = Path(sidecar_path) if sidecar_path is not None else root / _SIDECAR_NAME
    _require_regular_or_missing(sidecar_file, artifact="sidecar")
    sidecar_size = sidecar_file.stat().st_size if sidecar_file.is_file() else None
    sidecar_ranges: list[tuple[int, int, str]] = []
    for name, external in external_by_name.items():
        if external.location == location:
            tensor = reused_by_name.get(name)
            if tensor is None:
                raise ValueError(f"Unmanifested GGUF external initializer {name!r}.")
            expected_dtype = _FLOAT_QTYPE_DTYPES.get(tensor["qtype"], ir.DataType.UINT8)
            source_shape = tuple(tensor["source_shape"])
            if (
                external.dtype != expected_dtype
                or external.nbytes != tensor["length"]
                or tuple(int(dim) for dim in external.shape) != source_shape
            ):
                raise ValueError(
                    f"GGUF initializer {name!r} has incompatible dtype, shape, or byte length."
                )
            if external.offset != tensor["offset"] or external.length != tensor["length"]:
                raise ValueError(f"Manifest range does not match ONNX initializer {name!r}.")
            _verify_transform_graph(
                name,
                external_values_by_name[name],
                tensor,
                nodes_by_name,
            )
        elif external.location == _SIDECAR_NAME:
            if name not in converted_names:
                raise ValueError(f"Unmanifested sidecar initializer {name!r}.")
            shape = tuple(external.shape)
            if any(not isinstance(dim, int) or dim < 0 for dim in shape):
                raise ValueError(
                    f"Unsupported sidecar dtype or shape for initializer {name!r}."
                )
            try:
                expected_nbytes = external.nbytes
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Unsupported sidecar dtype or shape for initializer {name!r}."
                ) from error
            if (
                sidecar_size is None
                or external.offset is None
                or external.length is None
                or external.offset < 0
                or external.length <= 0
                or external.length != expected_nbytes
                or external.offset + external.length > sidecar_size
            ):
                raise ValueError(
                    f"Invalid sidecar range or byte length for initializer {name!r}."
                )
            sidecar_ranges.append((external.offset, external.offset + external.length, name))
        else:
            raise ValueError(
                f"External initializer {name!r} uses unapproved location "
                f"{external.location!r}."
            )

    for tensor in reused_entries:
        external = external_by_name.get(tensor["initializer"])
        if external is None or external.location != location:
            raise ValueError(
                f"Manifest initializer {tensor['initializer']!r} is not GGUF-backed."
            )
    for name in converted_names:
        external = external_by_name.get(name)
        if external is None or external.location != _SIDECAR_NAME:
            raise ValueError(
                f"Converted initializer {name!r} is not stored in {_SIDECAR_NAME!r}."
            )
    previous_end = 0
    for start, end, name in sorted(sidecar_ranges):
        if start < previous_end:
            raise ValueError(f"Overlapping sidecar range for initializer {name!r}.")
        previous_end = end


def _verify_transform_graph(
    initializer_name: str,
    initializer: ir.Value,
    route: dict,
    nodes_by_name: dict[str, ir.Node],
) -> None:
    transform = route["transform"]
    source_shape = tuple(route["source_shape"])
    final_shape = tuple(route["final_shape"])
    parameter = route["transform_parameter"]
    if transform is None:
        if source_shape != final_shape:
            raise ValueError(
                f"Untransformed GGUF initializer {initializer_name!r} changes shape."
            )
        return

    suffixes = {
        "transpose": (("Transpose", "Transpose"),),
        "subtract_one": (("Sub", "Sub"),),
        "log_neg": (("Neg", "Neg"), ("Log", "Log")),
        "reshape": (("Reshape", "Reshape"),),
        "llama_qk_permute": (
            ("Reshape1", "Reshape"),
            ("Transpose", "Transpose"),
            ("Reshape2", "Reshape"),
        ),
    }
    try:
        expected_nodes = suffixes[transform]
    except KeyError as error:
        raise ValueError(f"Unknown manifest transform {transform!r}.") from error

    prefix = f"{initializer_name}.gguf_reuse"
    nodes: list[ir.Node] = []
    for suffix, op_type in expected_nodes:
        node = nodes_by_name.get(f"{prefix}.{suffix}")
        if node is None or node.op_type != op_type:
            raise ValueError(f"Missing or invalid {transform} graph for {initializer_name!r}.")
        nodes.append(node)
    if not nodes[0].inputs or nodes[0].inputs[0] is None:
        raise ValueError(f"Transform graph for {initializer_name!r} has no source input.")
    if nodes[0].inputs[0].name != initializer_name:
        raise ValueError(f"Transform graph for {initializer_name!r} uses the wrong source.")
    uses = initializer.uses()
    if len(uses) != 1 or uses[0][0].name != nodes[0].name:
        raise ValueError(
            f"GGUF initializer {initializer_name!r} bypasses its transform graph."
        )
    output = nodes[-1].outputs[0]
    if output.shape is None or tuple(int(dim) for dim in output.shape) != final_shape:
        raise ValueError(f"Transform output shape is wrong for {initializer_name!r}.")

    if transform == "transpose":
        expected_perm = tuple(reversed(range(len(source_shape))))
        if (
            tuple(reversed(source_shape)) != final_shape
            or tuple(nodes[0].attributes["perm"].as_ints()) != expected_perm
        ):
            raise ValueError(f"Transpose semantics are wrong for {initializer_name!r}.")
    if transform in {"subtract_one", "log_neg", "llama_qk_permute"}:
        if source_shape != final_shape:
            raise ValueError(f"Transform shapes are inconsistent for {initializer_name!r}.")
    if transform == "subtract_one":
        constant = nodes[0].inputs[1]
        if (
            constant is None
            or constant.const_value is None
            or constant.const_value.numpy().shape != ()
            or not np.isclose(float(constant.const_value.numpy()), 1.0)
        ):
            raise ValueError(f"Norm offset constant is wrong for {initializer_name!r}.")
    if transform == "log_neg" and nodes[1].inputs[0] is not nodes[0].outputs[0]:
        raise ValueError(f"Log/Neg wiring is wrong for {initializer_name!r}.")
    if transform == "reshape":
        shape_input = nodes[0].inputs[1]
        if (
            np.prod(source_shape) != np.prod(final_shape)
            or shape_input is None
            or shape_input.const_value is None
            or tuple(int(value) for value in shape_input.const_value.numpy()) != final_shape
        ):
            raise ValueError(f"Reshape semantics are wrong for {initializer_name!r}.")
    if transform == "llama_qk_permute":
        if not isinstance(parameter, int) or source_shape[0] % (parameter * 2):
            raise ValueError(f"Invalid Q/K head count for {initializer_name!r}.")
        dim = source_shape[0] // parameter // 2
        expanded_shape = (parameter, dim, 2, *source_shape[1:])
        permuted_shape = (parameter, 2, dim, *source_shape[1:])
        expanded = nodes[0].outputs[0]
        permuted = nodes[1].outputs[0]
        shape1 = nodes[0].inputs[1]
        shape2 = nodes[2].inputs[1]
        expected_perm = (0, 2, 1, *range(3, len(expanded_shape)))
        if (
            expanded.shape is None
            or tuple(int(value) for value in expanded.shape) != expanded_shape
            or permuted.shape is None
            or tuple(int(value) for value in permuted.shape) != permuted_shape
            or nodes[1].inputs[0] is not expanded
            or nodes[2].inputs[0] is not permuted
            or tuple(nodes[1].attributes["perm"].as_ints()) != expected_perm
            or shape1 is None
            or shape1.const_value is None
            or tuple(int(value) for value in shape1.const_value.numpy()) != expanded_shape
            or shape2 is None
            or shape2.const_value is None
            or tuple(int(value) for value in shape2.const_value.numpy()) != source_shape
        ):
            raise ValueError(f"Q/K permutation shapes are wrong for {initializer_name!r}.")
