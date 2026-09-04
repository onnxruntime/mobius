# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the fail-closed VibeVoice ASR BitNet native-GGUF verdict."""

from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from mobius.integrations.gguf._errors import VibeASRBitNetGGUFImportError
from mobius.integrations.gguf._header import (
    GGUFHeaderInfo,
    _gguf_header_info_from_header,
)
from mobius.integrations.gguf._reader import GGUFModel
from mobius.integrations.gguf._vibeasr_bitnet import (
    VIBEVOICE_ASR_BITNET_ARTIFACTS,
    VIBEVOICE_ASR_BITNET_REPOSITORY,
    VIBEVOICE_ASR_BITNET_REVISION,
    find_vibeasr_bitnet_gguf_artifact,
)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _string_entry(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _uint32_entry(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<II", 4, value)


def _tensor_entry(name: str, type_id: int) -> bytes:
    # The import verdict needs only type IDs, so zero-length tensor descriptors
    # exercise the bounded header path without requiring tensor payloads.
    return _string(name) + struct.pack("<IQI", 1, 0, type_id) + struct.pack("<Q", 0)


def _native_header(artifact) -> bytes:
    metadata = (
        _string_entry("general.architecture", artifact.architecture),
        _string_entry(
            "general.name",
            "models" if artifact.architecture == "qwen2" else "VibeASR VAE Encoder",
        ),
        _uint32_entry("general.file_type", artifact.file_type),
        _uint32_entry("general.quantization_version", 2),
    )
    tensors = [
        _tensor_entry(f"tensor-{index}", type_id)
        for index, type_id in enumerate(sorted(artifact.tensor_type_ids))
    ]
    return (
        b"GGUF"
        + struct.pack("<IQQ", 3, len(tensors), len(metadata))
        + b"".join(metadata)
        + b"".join(tensors)
    )


@pytest.mark.parametrize("artifact", VIBEVOICE_ASR_BITNET_ARTIFACTS)
def test_pinned_header_profile_has_exact_native_identity(artifact) -> None:
    header = _gguf_header_info_from_header(
        _native_header(artifact),
        source=artifact.filename,
        collect_tensor_type_ids=True,
    )

    assert header.architecture == artifact.architecture
    assert header.file_type == artifact.file_type
    assert header.quantization_version == 2
    assert header.tensor_type_ids == artifact.tensor_type_ids
    assert find_vibeasr_bitnet_gguf_artifact(header=header) == artifact


@pytest.mark.parametrize("artifact", VIBEVOICE_ASR_BITNET_ARTIFACTS)
def test_pinned_hub_artifact_fingerprint_is_exact(artifact) -> None:
    assert (
        find_vibeasr_bitnet_gguf_artifact(
            repository=VIBEVOICE_ASR_BITNET_REPOSITORY,
            revision=VIBEVOICE_ASR_BITNET_REVISION,
            filename=artifact.filename,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
        )
        == artifact
    )
    assert (
        find_vibeasr_bitnet_gguf_artifact(
            repository=VIBEVOICE_ASR_BITNET_REPOSITORY,
            revision=VIBEVOICE_ASR_BITNET_REVISION,
            filename=artifact.filename,
            size_bytes=artifact.size_bytes + 1,
            sha256=artifact.sha256,
        )
        is None
    )


@pytest.mark.parametrize("artifact", VIBEVOICE_ASR_BITNET_ARTIFACTS)
def test_local_reader_rejects_native_gguf_before_reader_payload_access(
    tmp_path: Path, artifact
) -> None:
    path = tmp_path / artifact.filename
    path.write_bytes(_native_header(artifact))

    with pytest.raises(
        VibeASRBitNetGGUFImportError,
        match=r"Direct GGUF import is unsupported.*No ONNX artifacts were emitted",
    ):
        GGUFModel(path)


@pytest.mark.parametrize("artifact", VIBEVOICE_ASR_BITNET_ARTIFACTS)
def test_hub_header_preflight_rejects_before_download(artifact) -> None:
    from mobius.integrations.gguf import _builder

    response = mock.MagicMock()
    response.iter_bytes.return_value = [_native_header(artifact)]
    response_context = mock.MagicMock()
    response_context.__enter__.return_value = response
    session = mock.MagicMock()
    session.stream.return_value = response_context

    with (
        mock.patch.object(
            _builder,
            "get_hf_file_metadata",
            return_value=SimpleNamespace(
                commit_hash=VIBEVOICE_ASR_BITNET_REVISION,
                location="https://cdn.example/native.gguf",
            ),
        ),
        mock.patch.object(_builder, "get_session", return_value=session),
        pytest.raises(VibeASRBitNetGGUFImportError, match=artifact.filename),
    ):
        _builder._preflight_hf_gguf_file(
            VIBEVOICE_ASR_BITNET_REPOSITORY,
            artifact.filename,
            revision=VIBEVOICE_ASR_BITNET_REVISION,
        )


@pytest.mark.parametrize("artifact", VIBEVOICE_ASR_BITNET_ARTIFACTS)
def test_builder_validation_rejects_native_profile_before_config_extraction(artifact) -> None:
    from mobius.integrations.gguf._builder import _validate_gguf_model

    class NativeGGUF:
        def __init__(self) -> None:
            self.architecture = artifact.architecture
            self.tensor_names = ["native-projection"]

        def get_metadata(self, key: str, default=None):
            return {
                "general.file_type": artifact.file_type,
                "general.quantization_version": 2,
            }.get(key, default)

        def reader_tensors(self):
            return [
                SimpleNamespace(tensor_type=SimpleNamespace(value=type_id))
                for type_id in artifact.tensor_type_ids
            ]

    with pytest.raises(VibeASRBitNetGGUFImportError, match=artifact.native_format):
        _validate_gguf_model(NativeGGUF(), source=artifact.filename)


def test_hub_preflight_reports_both_native_artifacts_without_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mobius.integrations.gguf._preflight import preflight_hf_gguf

    artifacts = {artifact.filename: artifact for artifact in VIBEVOICE_ASR_BITNET_ARTIFACTS}

    class MetadataOnlyApi:
        def list_repo_files(self, repo_id, revision=None, token=None):
            assert repo_id == VIBEVOICE_ASR_BITNET_REPOSITORY
            assert revision == VIBEVOICE_ASR_BITNET_REVISION
            return list(artifacts)

        def get_paths_info(self, repo_id, paths, revision=None, token=None, expand=False):
            assert repo_id == VIBEVOICE_ASR_BITNET_REPOSITORY
            assert revision == VIBEVOICE_ASR_BITNET_REVISION
            assert expand
            return [
                SimpleNamespace(
                    path=filename,
                    size=artifacts[filename].size_bytes,
                    lfs=SimpleNamespace(sha256=artifacts[filename].sha256),
                )
                for filename in paths
            ]

        def model_info(self, repo_id, revision=None, token=None, expand=None):
            assert repo_id == VIBEVOICE_ASR_BITNET_REPOSITORY
            assert revision == VIBEVOICE_ASR_BITNET_REVISION
            assert expand == ["gguf"]
            return SimpleNamespace(gguf={"architecture": "qwen2", "total": 1_777_088_000})

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *args, **kwargs: MetadataOnlyApi())
    report = preflight_hf_gguf(
        VIBEVOICE_ASR_BITNET_REPOSITORY,
        revision=VIBEVOICE_ASR_BITNET_REVISION,
    )

    assert report.total_tensors is None
    assert report.total_params == 1_777_088_000
    assert [(file.filename, file.size_bytes, file.sha256) for file in report.files] == [
        (artifact.filename, artifact.size_bytes, artifact.sha256)
        for artifact in VIBEVOICE_ASR_BITNET_ARTIFACTS
    ]
    assert len(report.blockers) == len(VIBEVOICE_ASR_BITNET_ARTIFACTS)
    assert all(
        artifact.filename in " ".join(report.blockers) for artifact in artifacts.values()
    )
    assert not report.exportable


def test_generic_qwen2_with_a_standard_q1_header_is_not_a_vibeasr_alias() -> None:
    header = GGUFHeaderInfo(
        architecture="qwen2",
        tensor_count=1,
        split_no=None,
        split_count=None,
        split_tensors_count=None,
        file_type=40,
        tensor_type_ids=frozenset({0, 41}),
    )

    assert find_vibeasr_bitnet_gguf_artifact(header=header) is None
