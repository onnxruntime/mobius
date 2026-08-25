# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for atomic, tokenizer-gated GGUF runtime package emission."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from mobius.integrations.gguf import write_gguf_runtime_package
from mobius.integrations.gguf._spec import Support

_TOKENIZER_REPOSITORY = "owner/tokenizer"
_TOKENIZER_REVISION = "c" * 40


class _FakePackage:
    def __init__(self):
        self.config = object()
        self.gguf_architecture = "llama"
        self.gguf_execution_provider = "cpu"
        self.gguf_import_route = '{"route_schema":1}'
        self.gguf_artifact_identity = SimpleNamespace(
            architecture="llama", filename="model.gguf", sha256="a" * 64
        )
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
                graph_sha256=mock.ANY,
                runtime_package_files=("model.onnx",),
                runtime_package_sha256=mock.ANY,
                tokenizer_repository=_TOKENIZER_REPOSITORY,
                tokenizer_revision=_TOKENIZER_REVISION,
                tokenizer_metadata_sha256="f" * 64,
                tokenizer_assets=(("tokenizer.json", 2, "a" * 64),),
            ),
        ),
        mock.patch(
            "mobius.integrations.gguf._runtime_package.gguf_graph_package_identity",
            return_value=SimpleNamespace(files=("model.onnx",), sha256=mock.ANY),
        ),
    ):
        yield


class TestWriteGgufRuntimePackage:
    def test_deferred_architecture_rejects_before_source_read_or_output(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.get_arch_spec",
                return_value=SimpleNamespace(
                    runtime=Support.DEFERRED,
                    reason="independent parity is missing",
                ),
            ),
            mock.patch("mobius.integrations.gguf._runtime_package.GGUFModel") as read_source,
            pytest.raises(ValueError, match=r"runtime packaging.*deferred"),
        ):
            _write_runtime(pkg, tmp_path / "m.gguf", out)
        read_source.assert_not_called()
        assert not out.exists()

    def test_atomically_emits_graph_tokenizer_and_runtime_config(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.GGUFModel",
                return_value=SimpleNamespace(metadata={}, architecture="llama"),
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
        assert not list(tmp_path.glob(".out.*.tmp"))

    def test_missing_pinned_tokenizer_source_rejects_before_save_or_output(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch("mobius.integrations.gguf._runtime_package.GGUFModel") as read_source,
            pytest.raises(ValueError, match="explicit tokenizer_repository"),
        ):
            write_gguf_runtime_package(pkg, tmp_path / "m.gguf", out)
        read_source.assert_not_called()
        assert pkg.saved_to is None
        assert not out.exists()

    def test_replaced_source_tokenizer_rejects_before_save_or_output(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.GGUFModel",
                return_value=SimpleNamespace(metadata={}, architecture="llama"),
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

    def test_existing_output_is_never_replaced(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        out.mkdir()
        sentinel = out / "sentinel.bin"
        sentinel.write_bytes(b"unchanged")
        with (
            mock.patch("mobius.integrations.gguf._runtime_package.GGUFModel") as read_source,
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
            write_gguf_runtime_package(
                pkg,
                tmp_path / "m.gguf",
                tmp_path / "out",
                runtime="ort-genai",
            )
        assert not (tmp_path / "out").exists()

    def test_target_coupled_draft_runtime_package_is_rejected(self, tmp_path):
        pkg = _FakePackage()
        pkg.draft_manifest = {"architecture": "eagle3"}
        out = tmp_path / "out"
        out.mkdir()
        sentinel = out / "sentinel.bin"
        sentinel.write_bytes(b"unchanged")

        with pytest.raises(ValueError, match="target-coupled speculative draft"):
            write_gguf_runtime_package(
                pkg,
                tmp_path / "eagle3.gguf",
                out,
            )
        assert pkg.saved_to is None
        assert {path.name: path.read_bytes() for path in out.iterdir()} == {
            "sentinel.bin": b"unchanged"
        }

    @pytest.mark.parametrize("runtime", ["onnx-genai", "ort-genai"])
    def test_runtime_rejects_unevidenced_mtp_before_source_read(self, tmp_path, runtime):
        pkg = _FakePackage()
        pkg.mtp_head = SimpleNamespace(config=object())
        out = tmp_path / "out"
        with (
            mock.patch("mobius.integrations.gguf._runtime_package.GGUFModel") as read_source,
            pytest.raises(ValueError, match="runtime-evidenced GGUF MTP"),
        ):
            _write_runtime(pkg, tmp_path / "m.gguf", out, runtime=runtime)
        read_source.assert_not_called()
        assert not out.exists()

    def test_unknown_runtime_rejects_before_source_read(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown runtime"):
            write_gguf_runtime_package(
                _FakePackage(), tmp_path / "m.gguf", tmp_path / "out", runtime="tflite"
            )

    def test_save_model_false_is_rejected_before_source_read(self, tmp_path):
        with (
            mock.patch("mobius.integrations.gguf._runtime_package.GGUFModel") as read_source,
            pytest.raises(ValueError, match="save_model=False is not supported"),
        ):
            write_gguf_runtime_package(
                _FakePackage(), tmp_path / "m.gguf", tmp_path / "output", save_model=False
            )
        read_source.assert_not_called()

    def test_target_coupled_draft_rejects_before_source_read(self, tmp_path):
        pkg = _FakePackage()
        pkg.draft_manifest = {"architecture": "eagle3"}
        with pytest.raises(ValueError, match="target-coupled speculative draft"):
            _write_runtime(pkg, tmp_path / "m.gguf", tmp_path / "out")
