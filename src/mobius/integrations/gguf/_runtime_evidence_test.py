# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for exact GGUF runtime-evidence binding."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest

from mobius.integrations.gguf import _runtime_evidence
from mobius.integrations.gguf._runtime_evidence import (
    GGUFRuntimeEvidence,
    gguf_artifact_identity,
    gguf_graph_package_identity,
    matching_runtime_evidence,
    validate_runtime_evidence_ids,
)


def _record(payload: bytes) -> GGUFRuntimeEvidence:
    return GGUFRuntimeEvidence(
        evidence_id="llama-onnx-genai",
        architecture="llama",
        repository="owner/model",
        revision="a" * 40,
        filename="model.gguf",
        size=len(payload),
        lfs_sha256=hashlib.sha256(payload).hexdigest(),
        config_repository="owner/config",
        config_revision="b" * 40,
        tokenizer_repository="owner/tokenizer",
        tokenizer_revision="c" * 40,
        tokenizer_metadata_sha256="f" * 64,
        tokenizer_assets=(("tokenizer.json", 2, hashlib.sha256(b"{}").hexdigest()),),
        tensor_count=2,
        tensor_qtypes=(("F32", 1), ("Q4_K", 1)),
        import_route='{"route_schema":1}',
        graph_files=("model.onnx",),
        graph_sha256="d" * 64,
        runtime_package_files=("model.onnx", "tokenizer.json"),
        runtime_package_sha256="e" * 64,
        parity_test="test_full_logit_parity",
        parity_kind="full-logit",
        deterministic_test="test_cached_generation",
        stateful_semantics="dynamic KV cache with reorder and rollback",
        runtime="onnx-genai",
        runtime_version="1.0.0",
    )


def _model():
    tensors = [
        SimpleNamespace(tensor_type=SimpleNamespace(name="Q4_K")),
        SimpleNamespace(tensor_type=SimpleNamespace(name="F32")),
    ]
    return SimpleNamespace(
        _reader=SimpleNamespace(tensors=tensors),
        reader_tensors=lambda: tensors,
    )


def test_runtime_evidence_rejects_non_hex_tokenizer_metadata_digest() -> None:
    with pytest.raises(ValueError, match="immutable 40-hex revisions and LFS SHA-256"):
        replace(_record(b"pinned-gguf"), tokenizer_metadata_sha256="g" * 64)


def test_matching_evidence_binds_arch_runtime_source_qtypes_and_route(
    tmp_path, monkeypatch
) -> None:
    payload = b"pinned-gguf"
    source = tmp_path / "model.gguf"
    source.write_bytes(payload)
    record = _record(payload)
    monkeypatch.setattr(
        _runtime_evidence,
        "_RUNTIME_EVIDENCE",
        MappingProxyType({record.evidence_id: record}),
    )

    assert (
        matching_runtime_evidence(
            (record.evidence_id,),
            architecture="llama",
            runtime="onnx-genai",
            source_path=source,
            gguf_model=_model(),
            built_identity=gguf_artifact_identity(source, _model(), architecture="llama"),
            import_route=record.import_route,
            runtime_version="1.0.0",
            tokenizer_repository=record.tokenizer_repository,
            tokenizer_revision=record.tokenizer_revision,
        )
        is record
    )

    with pytest.raises(ValueError, match="No unique GGUF runtime evidence"):
        matching_runtime_evidence(
            (record.evidence_id,),
            architecture="llama",
            runtime="ort-genai",
            source_path=source,
            gguf_model=_model(),
            built_identity=gguf_artifact_identity(source, _model(), architecture="llama"),
            import_route=record.import_route,
            runtime_version="1.0.0",
            tokenizer_repository=record.tokenizer_repository,
            tokenizer_revision=record.tokenizer_revision,
        )
    with pytest.raises(ValueError, match="No unique GGUF runtime evidence"):
        matching_runtime_evidence(
            (record.evidence_id,),
            architecture="llama",
            runtime="onnx-genai",
            source_path=source,
            gguf_model=_model(),
            built_identity=gguf_artifact_identity(source, _model(), architecture="llama"),
            import_route=record.import_route,
            runtime_version="1.0.0",
            tokenizer_repository="attacker/replacement",
            tokenizer_revision=record.tokenizer_revision,
        )
    with pytest.raises(ValueError, match="No unique GGUF runtime evidence"):
        matching_runtime_evidence(
            (record.evidence_id,),
            architecture="llama",
            runtime="onnx-genai",
            source_path=source,
            gguf_model=_model(),
            built_identity=gguf_artifact_identity(source, _model(), architecture="llama"),
            import_route='{"route_schema":2}',
            runtime_version="1.0.0",
            tokenizer_repository=record.tokenizer_repository,
            tokenizer_revision=record.tokenizer_revision,
        )


def test_matching_evidence_rejects_source_replaced_after_build(tmp_path, monkeypatch) -> None:
    payload = b"pinned-gguf"
    source = tmp_path / "model.gguf"
    source.write_bytes(payload)
    record = _record(payload)
    built_identity = gguf_artifact_identity(source, _model(), architecture="llama")
    source.write_bytes(b"changed-gguf")
    monkeypatch.setattr(
        _runtime_evidence,
        "_RUNTIME_EVIDENCE",
        MappingProxyType({record.evidence_id: record}),
    )

    with pytest.raises(ValueError, match="no longer matches"):
        matching_runtime_evidence(
            (record.evidence_id,),
            architecture="llama",
            runtime="onnx-genai",
            source_path=source,
            gguf_model=_model(),
            built_identity=built_identity,
            import_route=record.import_route,
            runtime_version="1.0.0",
            tokenizer_repository=record.tokenizer_repository,
            tokenizer_revision=record.tokenizer_revision,
        )


def test_sharded_artifact_identity_frames_every_shard_and_tensor(tmp_path) -> None:
    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    tensors = [
        SimpleNamespace(tensor_type=SimpleNamespace(name="F32")),
        SimpleNamespace(tensor_type=SimpleNamespace(name="Q4_K")),
    ]
    model = SimpleNamespace(
        shard_paths=[first, second],
        reader_tensors=lambda: tensors,
    )

    identity = gguf_artifact_identity(
        second,
        model,
        architecture="llama",
        filename=first.name,
    )

    assert identity.filename == first.name
    assert identity.size == len(b"firstsecond")
    assert identity.tensor_count == 2
    assert identity.tensor_qtypes == (("F32", 1), ("Q4_K", 1))

    second.write_bytes(b"change")
    changed = gguf_artifact_identity(
        first,
        model,
        architecture="llama",
        filename=first.name,
    )
    assert changed.sha256 != identity.sha256


def test_sharded_artifact_identity_hashes_regular_aliases_for_snapshot_links(
    tmp_path,
) -> None:
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    first_blob = blobs / "first-blob"
    second_blob = blobs / "second-blob"
    first_blob.write_bytes(b"first")
    second_blob.write_bytes(b"second")
    first = snapshot / "model-00001-of-00002.gguf"
    second = snapshot / "model-00002-of-00002.gguf"
    first.symlink_to(first_blob)
    second.symlink_to(second_blob)
    tensors = [SimpleNamespace(tensor_type=SimpleNamespace(name="F32"))]
    model = SimpleNamespace(
        shard_paths=[first, second],
        identity_paths=[first_blob, second_blob],
        reader_tensors=lambda: tensors,
    )

    identity = gguf_artifact_identity(
        second,
        model,
        architecture="llama",
        filename=first.name,
    )

    assert identity.filename == first.name
    assert identity.size == len(b"firstsecond")


def test_evidence_id_cannot_cross_architectures(monkeypatch) -> None:
    record = _record(b"pinned-gguf")
    monkeypatch.setattr(
        _runtime_evidence,
        "_RUNTIME_EVIDENCE",
        MappingProxyType({record.evidence_id: record}),
    )
    with pytest.raises(ValueError, match="do not belong to 'qwen2'"):
        validate_runtime_evidence_ids("qwen2", (record.evidence_id,))


def test_graph_package_identity_frames_files_and_rejects_symlinks(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "a.onnx").write_bytes(b"ab")
    (package / "b.data").write_bytes(b"c")
    first = gguf_graph_package_identity(package)

    (package / "a.onnx").write_bytes(b"a")
    (package / "b.data").write_bytes(b"bc")
    second = gguf_graph_package_identity(package)
    assert first.files == second.files == ("a.onnx", "b.data")
    assert first.sha256 != second.sha256

    (package / "linked.data").symlink_to(package / "b.data")
    with pytest.raises(ValueError, match="must not contain symlinks"):
        gguf_graph_package_identity(package)
