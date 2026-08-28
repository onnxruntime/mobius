# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for atomic, tokenizer-gated GGUF runtime package emission."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from mobius.integrations.gguf import _runtime_package, write_gguf_runtime_package
from mobius.integrations.gguf._spec import Support

_TOKENIZER_REPOSITORY = "owner/tokenizer"
_TOKENIZER_REVISION = "c" * 40
_BUILT_IDENTITY = SimpleNamespace(
    architecture="llama",
    filename="model.gguf",
    sha256="a" * 64,
)


class _FakePackage:
    def __init__(self):
        self.config = object()
        self.gguf_architecture = "llama"
        self.gguf_execution_provider = "cpu"
        self.gguf_import_route = '{"route_schema":1}'
        self.gguf_artifact_identity = _BUILT_IDENTITY
        self.gguf_tokenizer_verdict = _materialized()
        self.saved_to: str | None = None

    def save(self, path, **kwargs):
        self.saved_to = path
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.onnx").write_bytes(b"stub")

    def __iter__(self):
        return iter(("model",))


def _materialized():
    return SimpleNamespace(
        materialized=True,
        reason="exact embedded tokenizer",
        metadata_sha256="f" * 64,
    )


def _write_tokenizer(_source, output, **_kwargs):
    path = Path(output) / "tokenizer.json"
    path.write_text("{}", encoding="utf-8")
    return str(path)


def _write_config(_pkg, output, **_kwargs):
    path = Path(output) / "inference_metadata.yaml"
    path.write_text("model: {}", encoding="utf-8")
    return {"inference_metadata": str(path)}


def _write_draft(_manifest, output):
    path = Path(output) / "draft_manifest.json"
    path.write_text("{}", encoding="utf-8")
    return str(path)


@contextmanager
def _successful_runtime_dependencies(runtime: str = "onnx-genai"):
    config_target = (
        "mobius.integrations.ort_genai.write_ort_genai_config"
        if runtime == "ort-genai"
        else "mobius.integrations.onnx_genai.write_onnx_genai_config"
    )
    with (
        mock.patch(
            "mobius.integrations.gguf._runtime_package.open_gguf_model",
            return_value=SimpleNamespace(
                metadata={},
                architecture="llama",
                source_matches_path=lambda: True,
            ),
        ),
        mock.patch(
            "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
            return_value=_materialized(),
        ),
        mock.patch(
            "mobius.integrations.gguf._runtime_package.materialize_gguf_tokenizer",
            side_effect=_write_tokenizer,
        ),
        mock.patch(
            "mobius.integrations.gguf._runtime_package.write_gguf_tokenizer_json",
            side_effect=_write_tokenizer,
        ),
        mock.patch(config_target, side_effect=_write_config),
        mock.patch(
            "mobius.integrations.gguf._draft.write_draft_manifest",
            side_effect=_write_draft,
        ),
        mock.patch(
            "mobius.integrations.onnx_genai.inference_metadata.write_mtp_speculator_metadata",
            return_value=None,
        ),
    ):
        yield


def _write_runtime(pkg, source, output, **kwargs):
    return write_gguf_runtime_package(
        pkg,
        source,
        output,
        tokenizer_repository=_TOKENIZER_REPOSITORY,
        tokenizer_revision=_TOKENIZER_REVISION,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _runtime_supported():
    with (
        mock.patch(
            "mobius.integrations.gguf._runtime_package.get_arch_spec",
            return_value=SimpleNamespace(
                runtime=Support.SUPPORTED,
                reason=None,
                gguf_arch="llama",
                runtime_evidence_ids=("test-evidence",),
            ),
        ),
        mock.patch(
            "mobius.integrations.gguf._runtime_package.matching_runtime_evidence",
            return_value=SimpleNamespace(
                evidence_id="test-evidence",
                graph_files=("model.onnx",),
                graph_sha256="b" * 64,
                runtime_package_files=("recorded-package.onnx",),
                runtime_package_sha256="c" * 64,
                tokenizer_repository=_TOKENIZER_REPOSITORY,
                tokenizer_revision=_TOKENIZER_REVISION,
                tokenizer_metadata_sha256="f" * 64,
                tokenizer_assets=(("tokenizer.json", 2, "a" * 64),),
            ),
        ),
        mock.patch(
            "mobius.integrations.gguf._runtime_package.gguf_graph_package_identity",
            return_value=SimpleNamespace(files=("model.onnx",), sha256="b" * 64),
        ),
        mock.patch(
            "mobius.integrations.gguf._runtime_package.gguf_artifact_identity",
            return_value=_BUILT_IDENTITY,
        ),
    ):
        yield


class TestWriteGgufRuntimePackage:
    def test_atomic_publication_refuses_concurrent_destination(self, tmp_path):
        stage = tmp_path / "stage"
        stage.mkdir()
        (stage / "model.onnx").write_bytes(b"staged")
        output = tmp_path / "output"
        output.mkdir()

        with pytest.raises(FileExistsError):
            _runtime_package._publish_directory_no_replace(stage, output)

        assert (stage / "model.onnx").read_bytes() == b"staged"
        assert not list(output.iterdir())

    def test_deferred_architecture_exports_unvalidated_metadata(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.get_arch_spec",
                return_value=SimpleNamespace(
                    runtime=Support.DEFERRED,
                    reason="independent parity is missing",
                    gguf_arch="llama",
                    runtime_evidence_ids=(),
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.matching_runtime_evidence",
                side_effect=ValueError(
                    "No unique GGUF runtime evidence matches architecture='llama'"
                ),
            ),
            _successful_runtime_dependencies(),
        ):
            _write_runtime(pkg, tmp_path / "m.gguf", out)
        compatibility = json.loads((out / "runtime_compatibility.json").read_text())
        assert compatibility["runtime_validation_status"] == "unvalidated"
        assert "independent parity is missing" in compatibility["warnings"][0]

    def test_atomically_emits_graph_tokenizer_and_runtime_config(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model",
                return_value=SimpleNamespace(
                    metadata={},
                    architecture="llama",
                    source_matches_path=lambda: True,
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=_materialized(),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.materialize_gguf_tokenizer",
                side_effect=_write_tokenizer,
            ),
            mock.patch(
                "mobius.integrations.onnx_genai.write_onnx_genai_config",
                side_effect=_write_config,
            ),
        ):
            artifacts = _write_runtime(pkg, tmp_path / "m.gguf", out)

        assert (out / "model.onnx").read_bytes() == b"stub"
        assert Path(artifacts["tokenizer"]) == out / "tokenizer.json"
        assert Path(artifacts["inference_metadata"]) == out / "inference_metadata.yaml"
        compatibility = json.loads((out / "runtime_compatibility.json").read_text())
        assert compatibility["runtime_validation_status"] == "unvalidated"
        assert "completed runtime package" in compatibility["warnings"][-1]
        assert not list(tmp_path.glob(".out.*.tmp"))

    def test_missing_evidence_runtime_version_does_not_block_export(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.matching_runtime_evidence",
                side_effect=ValueError(
                    "Runtime packaging requires the exact runtime version covered by evidence."
                ),
            ),
            _successful_runtime_dependencies(),
        ):
            artifacts = _write_runtime(pkg, tmp_path / "m.gguf", out)

        assert Path(artifacts["inference_metadata"]).is_file()
        compatibility = json.loads((out / "runtime_compatibility.json").read_text())
        assert compatibility["runtime_validation_status"] == "unvalidated"

    def test_portable_graph_targets_ort_genai_cpu(self, tmp_path):
        pkg = _FakePackage()
        pkg.gguf_execution_provider = "default"
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model",
                return_value=SimpleNamespace(
                    metadata={},
                    architecture="llama",
                    source_matches_path=lambda: True,
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=_materialized(),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.materialize_gguf_tokenizer",
                side_effect=_write_tokenizer,
            ),
            mock.patch(
                "mobius.integrations.ort_genai.write_ort_genai_config",
                side_effect=_write_config,
            ) as write_config,
        ):
            _write_runtime(
                pkg,
                tmp_path / "m.gguf",
                out,
                runtime="ort-genai",
                runtime_version="0.15.2",
            )

        assert write_config.call_args.kwargs["ep"] == "cpu"

    def test_failure_after_graph_save_removes_staging_and_publishes_nothing(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model",
                return_value=SimpleNamespace(
                    metadata={},
                    architecture="llama",
                    source_matches_path=lambda: True,
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=_materialized(),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.materialize_gguf_tokenizer",
                side_effect=OSError("tokenizer asset write failed"),
            ),
            pytest.raises(OSError, match="tokenizer asset write failed"),
        ):
            _write_runtime(pkg, tmp_path / "m.gguf", out)

        assert pkg.saved_to is not None
        assert not out.exists()
        assert not list(tmp_path.glob(".out.*.tmp"))

    def test_tokenizer_source_options_must_be_paired(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model"
            ) as read_source,
            pytest.raises(ValueError, match="must be provided together"),
        ):
            write_gguf_runtime_package(
                pkg,
                tmp_path / "m.gguf",
                out,
                tokenizer_repository=_TOKENIZER_REPOSITORY,
            )
        read_source.assert_not_called()
        assert pkg.saved_to is None
        assert not out.exists()

    def test_replaced_source_tokenizer_rejects_before_save_or_output(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model",
                return_value=SimpleNamespace(
                    metadata={},
                    architecture="llama",
                    source_matches_path=lambda: True,
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=SimpleNamespace(
                    materialized=True,
                    reason="replacement",
                    metadata_sha256="different-tokenizer-metadata",
                ),
            ),
            pytest.raises(ValueError, match="replaced tokenizer source"),
        ):
            _write_runtime(pkg, tmp_path / "m.gguf", out)
        assert pkg.saved_to is None
        assert not out.exists()

    def test_replaced_same_architecture_source_identity_always_rejects(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        current_identity = SimpleNamespace(
            architecture="llama",
            filename="model.gguf",
            sha256="d" * 64,
        )
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model",
                return_value=SimpleNamespace(
                    metadata={},
                    architecture="llama",
                    source_matches_path=lambda: True,
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.gguf_artifact_identity",
                return_value=current_identity,
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.matching_runtime_evidence"
            ) as evidence_lookup,
            pytest.raises(ValueError, match="exact artifact identity"),
        ):
            write_gguf_runtime_package(pkg, tmp_path / "replacement.gguf", out)

        evidence_lookup.assert_not_called()
        assert pkg.saved_to is None
        assert not out.exists()

    def test_existing_output_is_never_replaced(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        out.mkdir()
        sentinel = out / "sentinel.bin"
        sentinel.write_bytes(b"unchanged")
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model"
            ) as read_source,
            pytest.raises(FileExistsError, match="non-atomic directory replacement"),
        ):
            _write_runtime(pkg, tmp_path / "m.gguf", out)
        read_source.assert_not_called()
        assert {path.name: path.read_bytes() for path in out.iterdir()} == {
            "sentinel.bin": b"unchanged"
        }

    def test_ort_genai_rejects_reused_gguf_weights(self, tmp_path):
        pkg = _FakePackage()
        pkg.gguf_reuse_plan = object()
        with pytest.raises(ValueError, match="no supported setting"):
            _write_runtime(
                pkg,
                tmp_path / "m.gguf",
                tmp_path / "out",
                runtime="ort-genai",
            )
        assert not (tmp_path / "out").exists()

    def test_unknown_execution_provider_is_preserved_as_advisory(self, tmp_path):
        pkg = _FakePackage()
        pkg.gguf_execution_provider = "future-accelerator"
        out = tmp_path / "out"

        with _successful_runtime_dependencies("ort-genai"):
            _write_runtime(pkg, tmp_path / "m.gguf", out, runtime="ort-genai")

        compatibility = json.loads((out / "runtime_compatibility.json").read_text())
        assert compatibility["execution_provider"] == "future-accelerator"
        assert compatibility["runtime_validation_status"] == "unvalidated"
        assert any("future-accelerator" in warning for warning in compatibility["warnings"])

    def test_target_coupled_draft_runtime_package_is_exported(self, tmp_path):
        pkg = _FakePackage()
        pkg.draft_manifest = {"architecture": "eagle3"}
        out = tmp_path / "out"

        with _successful_runtime_dependencies():
            _write_runtime(
                pkg,
                tmp_path / "eagle3.gguf",
                out,
            )
        assert (out / "draft_manifest.json").is_file()

    @pytest.mark.parametrize("runtime", ["onnx-genai", "ort-genai"])
    def test_unevidenced_mtp_does_not_block_runtime_export(self, tmp_path, runtime):
        pkg = _FakePackage()
        pkg.mtp_head = SimpleNamespace(config=object())
        out = tmp_path / "out"
        with _successful_runtime_dependencies(runtime):
            _write_runtime(pkg, tmp_path / "m.gguf", out, runtime=runtime)
        assert (out / "model.onnx").is_file()

    def test_unknown_runtime_rejects_before_source_read(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown runtime"):
            write_gguf_runtime_package(
                _FakePackage(), tmp_path / "m.gguf", tmp_path / "out", runtime="tflite"
            )

    def test_save_model_false_is_rejected_before_source_read(self, tmp_path):
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model"
            ) as read_source,
            pytest.raises(ValueError, match="save_model=False is not supported"),
        ):
            write_gguf_runtime_package(
                _FakePackage(), tmp_path / "m.gguf", tmp_path / "output", save_model=False
            )
        read_source.assert_not_called()
