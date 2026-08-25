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
    tokenizer_metadata_sha256: str
    tokenizer_assets: tuple[tuple[str, int, str], ...]
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
            self.tokenizer_metadata_sha256,
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
                    self.tokenizer_metadata_sha256,
                    self.graph_sha256,
                    self.runtime_package_sha256,
                )
            )
            or any(
                not _is_hex(value)
                for value in (
                    *revisions,
                    self.lfs_sha256,
                    self.tokenizer_metadata_sha256,
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
        asset_names = tuple(asset[0] for asset in self.tokenizer_assets)
        if (
            not self.tokenizer_assets
            or "tokenizer.json" not in asset_names
            or asset_names != tuple(sorted(asset_names))
            or len(set(asset_names)) != len(asset_names)
            or any(
                filename != Path(filename).name
                or size <= 0
                or len(sha256) != 64
                or not _is_hex(sha256)
                for filename, size, sha256 in self.tokenizer_assets
            )
        ):
            raise ValueError(
                "GGUF runtime evidence tokenizer_assets must be sorted, unique, "
                "basename-only exact file identities including tokenizer.json"
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


def _is_hex(value: str) -> bool:
    return all(character in "0123456789abcdefABCDEF" for character in value)


_SMOLLM_F16_ROUTE = (
    '{"architecture":"llama","config_sha256":'
    '"134f95e6a635d978737d712ed61ac8959acebdf080eafae838cf97f12c416430",'
    '"execution_provider":"cpu","model_type":"llama","module_type":"llama",'
    '"preserve_quantization":false,"registry_import":{"config_key_map":null,'
    '"config_postprocessor":null,"llama_qk_permute":true,"offset_norm":false,'
    '"required_metadata":[],"rope_interleave":false,"tensor_processor":"llama",'
    '"v_head_reorder":false,"vlm_builder":null},"route_schema":1,"static_cache":false,'
    '"task":{"class":"builtins.str","state":"text-generation"},'
    '"tensor_map_recipe":["llama"]}'
)

_SMOLLM_F16_ONNX_RUNTIME = GGUFRuntimeEvidence(
    evidence_id="smollm-135m-f16-onnxruntime-1.29.0",
    architecture="llama",
    repository="neopolita/smollm-135m-gguf",
    revision="22cca988936eafe92908e7558907c3964e10bba7",
    filename="ggml-model-f16.gguf",
    size=270_885_504,
    lfs_sha256="ec8c775c16944a7e4b5251f97b3f848500dcc3e701b0d492ce9055cea42138a2",
    config_repository="HuggingFaceTB/SmolLM-135M",
    config_revision="1d461723eec654e65efdc40cf49301c89c0c92f4",
    tokenizer_repository="HuggingFaceTB/SmolLM-135M",
    tokenizer_revision="1d461723eec654e65efdc40cf49301c89c0c92f4",
    tokenizer_metadata_sha256="46646ba36ecae43de6f9f649d217774b889e0fd405af92205319b882927493fc",
    tokenizer_assets=(
        (
            "special_tokens_map.json",
            831,
            "e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3",
        ),
        (
            "tokenizer.json",
            2_104_556,
            "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
        ),
        (
            "tokenizer_config.json",
            3_685,
            "238ad6b60d48e471624ea70bc79e92f2611844d5016471fee8c167854bcb98e8",
        ),
    ),
    tensor_count=272,
    tensor_qtypes=(("F16", 211), ("F32", 61)),
    import_route=_SMOLLM_F16_ROUTE,
    graph_files=("model.onnx", "model.onnx.data"),
    graph_sha256="3d242b09fcb5041d71e5914084cf00780867b3b0e32f669f8733369b19b6ea9b",
    runtime_package_files=(
        "gguf_tokenizer_manifest.json",
        "inference_metadata.yaml",
        "model.onnx",
        "model.onnx.data",
        "policies/cache_length_update.onnx",
        "policies/decoder_state_initializer.onnx",
        "policies/decoder_step_update.onnx",
        "policies/generated_length_update.onnx",
        "policies/last_token_logits.onnx",
        "policies/termination.onnx",
        "policies/termination_batch_initializer.onnx",
        "policies/token_sampler.onnx",
        "policies/token_state_update.onnx",
        "policies/token_to_slot.onnx",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    runtime_package_sha256="5b6fdbdb1db7f7fb9423f2356820812712556abf783acb6bd63c572920031982",
    parity_test="test_small_f16_gguf_cli_full_logit_and_generation_parity[smollm-135m-f16]",
    parity_kind="full-logit",
    deterministic_test=(
        "test_small_f16_gguf_cli_full_logit_and_generation_parity[smollm-135m-f16]"
    ),
    stateful_semantics="dynamic KV cache prefill plus 20 cache-threaded decode steps",
    runtime="onnx-genai",
    runtime_version="1.29.0",
)

_RUNTIME_EVIDENCE: MappingProxyType[str, GGUFRuntimeEvidence] = MappingProxyType(
    {_SMOLLM_F16_ONNX_RUNTIME.evidence_id: _SMOLLM_F16_ONNX_RUNTIME}
)


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
    tokenizer_repository: str,
    tokenizer_revision: str,
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
        and _RUNTIME_EVIDENCE[evidence_id].tokenizer_repository == tokenizer_repository
        and _RUNTIME_EVIDENCE[evidence_id].tokenizer_revision == tokenizer_revision
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
