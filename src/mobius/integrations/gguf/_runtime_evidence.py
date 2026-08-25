# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Structured real-artifact evidence required for GGUF runtime support."""

from __future__ import annotations

__all__ = [
    "GGUFArtifactIdentity",
    "GGUFGraphPackageIdentity",
    "GGUFRuntimeEvidence",
    "gguf_artifact_identity",
    "gguf_graph_package_identity",
    "matching_runtime_evidence",
    "runtime_evidence",
    "validate_runtime_evidence_ids",
]

import dataclasses
import hashlib
import os
import stat
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFArtifactIdentity:
    """Immutable identity of the exact GGUF bytes used for graph construction."""

    architecture: str
    filename: str
    size: int
    sha256: str
    tensor_count: int
    tensor_qtypes: tuple[tuple[str, int], ...]


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFGraphPackageIdentity:
    """Canonical identity of the exact serialized ONNX graph package."""

    files: tuple[str, ...]
    sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFRuntimeEvidence:
    """One immutable end-to-end runtime evidence record."""

    evidence_id: str
    architecture: str
    repository: str
    revision: str
    filename: str
    size: int
    lfs_sha256: str
    config_repository: str
    config_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    tensor_count: int
    tensor_qtypes: tuple[tuple[str, int], ...]
    import_route: str
    graph_files: tuple[str, ...]
    graph_sha256: str
    runtime_package_files: tuple[str, ...]
    runtime_package_sha256: str
    parity_test: str
    parity_kind: str
    deterministic_test: str
    stateful_semantics: str
    runtime: str
    runtime_version: str

    def __post_init__(self) -> None:
        text_fields = (
            self.evidence_id,
            self.architecture,
            self.repository,
            self.revision,
            self.filename,
            self.lfs_sha256,
            self.config_repository,
            self.config_revision,
            self.tokenizer_repository,
            self.tokenizer_revision,
            self.import_route,
            self.graph_sha256,
            self.runtime_package_sha256,
            self.parity_test,
            self.parity_kind,
            self.deterministic_test,
            self.stateful_semantics,
            self.runtime,
            self.runtime_version,
        )
        if any(not value.strip() for value in text_fields):
            raise ValueError("GGUF runtime evidence fields must be non-empty")
        if self.size <= 0 or self.tensor_count <= 0 or not self.tensor_qtypes:
            raise ValueError("GGUF runtime evidence requires positive artifact/tensor census")
        revisions = (self.revision, self.config_revision, self.tokenizer_revision)
        if (
            any(len(value) != 40 for value in revisions)
            or any(
                len(value) != 64
                for value in (
                    self.lfs_sha256,
                    self.graph_sha256,
                    self.runtime_package_sha256,
                )
            )
            or any(
                not _is_hex(value)
                for value in (
                    *revisions,
                    self.lfs_sha256,
                    self.graph_sha256,
                    self.runtime_package_sha256,
                )
            )
        ):
            raise ValueError(
                "GGUF runtime evidence requires immutable 40-hex revisions and LFS SHA-256"
            )
        if self.parity_kind not in {"full-logit", "component"}:
            raise ValueError(
                "GGUF runtime evidence parity_kind must be full-logit or component"
            )
        if (
            not self.graph_files
            or tuple(sorted(self.graph_files)) != self.graph_files
            or len(set(self.graph_files)) != len(self.graph_files)
        ):
            raise ValueError("GGUF runtime evidence graph_files must be sorted and unique")
        if (
            not self.runtime_package_files
            or tuple(sorted(self.runtime_package_files)) != self.runtime_package_files
            or len(set(self.runtime_package_files)) != len(self.runtime_package_files)
        ):
            raise ValueError(
                "GGUF runtime evidence runtime_package_files must be sorted and unique"
            )


# Empty by design: graph/import execution evidence does not satisfy this schema.
_RUNTIME_EVIDENCE: MappingProxyType[str, GGUFRuntimeEvidence] = MappingProxyType({})


def _is_hex(value: str) -> bool:
    return all(character in "0123456789abcdefABCDEF" for character in value)


def runtime_evidence(evidence_id: str) -> GGUFRuntimeEvidence | None:
    """Return a structured evidence record by stable ID."""
    return _RUNTIME_EVIDENCE.get(evidence_id)


def validate_runtime_evidence_ids(architecture: str, evidence_ids: tuple[str, ...]) -> None:
    """Require every runtime evidence ID to resolve to a complete record."""
    if not evidence_ids:
        raise ValueError("runtime=SUPPORTED requires structured evidence IDs")
    unknown = sorted(
        evidence_id
        for evidence_id in evidence_ids
        if not evidence_id or runtime_evidence(evidence_id) is None
    )
    if unknown:
        raise ValueError(f"Unknown GGUF runtime evidence IDs: {unknown}")
    mismatched = sorted(
        evidence_id
        for evidence_id in evidence_ids
        if _RUNTIME_EVIDENCE[evidence_id].architecture != architecture
    )
    if mismatched:
        raise ValueError(
            f"GGUF runtime evidence IDs do not belong to {architecture!r}: {mismatched}"
        )


def matching_runtime_evidence(
    evidence_ids: tuple[str, ...],
    *,
    architecture: str,
    runtime: str,
    source_path: Path,
    gguf_model: Any,
    built_identity: GGUFArtifactIdentity,
    import_route: str,
    runtime_version: str | None,
) -> GGUFRuntimeEvidence:
    """Return exact evidence for the package source, route, and requested runtime."""
    validate_runtime_evidence_ids(architecture, evidence_ids)
    current_identity = gguf_artifact_identity(
        source_path,
        gguf_model,
        architecture=architecture,
        filename=built_identity.filename,
    )
    if current_identity != built_identity:
        raise ValueError(
            "The GGUF source no longer matches the exact artifact identity captured during "
            f"graph construction: built={built_identity!r}, current={current_identity!r}."
        )
    if runtime_version is None:
        raise ValueError(
            "Runtime packaging requires the exact runtime version covered by evidence."
        )
    identity = built_identity
    candidates = [
        _RUNTIME_EVIDENCE[evidence_id]
        for evidence_id in evidence_ids
        if _RUNTIME_EVIDENCE[evidence_id].runtime == runtime
        and _RUNTIME_EVIDENCE[evidence_id].runtime_version == runtime_version
        and _RUNTIME_EVIDENCE[evidence_id].filename == identity.filename
        and _RUNTIME_EVIDENCE[evidence_id].size == identity.size
        and _RUNTIME_EVIDENCE[evidence_id].lfs_sha256 == identity.sha256
        and _RUNTIME_EVIDENCE[evidence_id].tensor_count == identity.tensor_count
        and _RUNTIME_EVIDENCE[evidence_id].tensor_qtypes == identity.tensor_qtypes
        and _RUNTIME_EVIDENCE[evidence_id].import_route == import_route
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"No unique GGUF runtime evidence matches architecture={architecture!r}, "
            f"runtime={runtime!r} {runtime_version!r}, artifact={identity!r}, "
            f"import_route={import_route!r}."
        )
    return candidates[0]


def gguf_artifact_identity(
    source_path: Path,
    gguf_model: Any,
    *,
    architecture: str,
    filename: str | None = None,
) -> GGUFArtifactIdentity:
    """Fingerprint source bytes and parsed tensor census under a canonical architecture."""
    stat, sha256 = _hash_regular_file(source_path)
    qtypes = Counter(tensor.tensor_type.name for tensor in gguf_model._reader.tensors)
    return GGUFArtifactIdentity(
        architecture=architecture,
        filename=filename or source_path.name,
        size=stat.st_size,
        sha256=sha256,
        tensor_count=len(gguf_model._reader.tensors),
        tensor_qtypes=tuple(sorted(qtypes.items())),
    )


def gguf_graph_package_identity(package_dir: Path) -> GGUFGraphPackageIdentity:
    """Hash every regular graph-package file with its relative path."""
    paths: list[Path] = []
    for root, directories, filenames in os.walk(package_dir, followlinks=False):
        root_path = Path(root)
        entries = [root_path / name for name in (*directories, *filenames)]
        if any(path.is_symlink() for path in entries):
            raise ValueError("GGUF graph package must not contain symlinks")
        paths.extend(root_path / name for name in filenames)
    paths.sort()
    if not paths:
        raise ValueError("GGUF graph package must contain regular files and no symlinks")
    digest = hashlib.sha256()
    names: list[str] = []
    for path in paths:
        name = path.relative_to(package_dir).as_posix()
        names.append(name)
        encoded = name.encode("utf-8")
        stat, file_sha256 = _hash_regular_file(path)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(stat.st_size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_sha256))
    return GGUFGraphPackageIdentity(files=tuple(names), sha256=digest.hexdigest())


def _hash_regular_file(path: Path) -> tuple[os.stat_result, str]:
    """Hash one non-symlink regular file through the descriptor being validated."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or path.is_symlink():
            raise ValueError(f"Expected a non-symlink regular file: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)

        def identity(value: os.stat_result) -> tuple[int, int, int, int]:
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)

        if identity(before) != identity(after) or identity(after) != identity(path.stat()):
            raise ValueError(f"File changed while its immutable identity was computed: {path}")
        return after, digest.hexdigest()
    finally:
        os.close(descriptor)
