# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for atomic, tokenizer-gated GGUF runtime package emission."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from mobius.integrations.gguf import write_gguf_runtime_package


class _FakePackage:
    def __init__(self):
        self.config = object()
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
        metadata_sha256="tokenizer-metadata",
    )


def _write_tokenizer(_source, output, **_kwargs):
    path = Path(output) / "tokenizer.json"
    path.write_text("{}", encoding="utf-8")
    return str(path)


def _write_config(_pkg, output, **_kwargs):
    path = Path(output) / "inference_metadata.yaml"
    path.write_text("model: {}", encoding="utf-8")
    return {"inference_metadata": str(path)}


class TestWriteGgufRuntimePackage:
    def test_atomically_emits_graph_tokenizer_and_runtime_config(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.GGUFModel",
                return_value=SimpleNamespace(metadata={}),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=_materialized(),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.write_gguf_tokenizer_json",
                side_effect=_write_tokenizer,
            ),
            mock.patch(
                "mobius.integrations.onnx_genai.write_onnx_genai_config",
                side_effect=_write_config,
            ),
        ):
            artifacts = write_gguf_runtime_package(pkg, tmp_path / "m.gguf", out)

        assert (out / "model.onnx").read_bytes() == b"stub"
        assert Path(artifacts["tokenizer"]) == out / "tokenizer.json"
        assert Path(artifacts["inference_metadata"]) == out / "inference_metadata.yaml"
        assert not list(tmp_path.glob(".out.*.tmp"))

    def test_deferred_tokenizer_rejects_before_save_or_output(self, tmp_path):
        pkg = _FakePackage()
        pkg.gguf_tokenizer_verdict = SimpleNamespace(metadata_sha256="deferred")
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.GGUFModel",
                return_value=SimpleNamespace(metadata={}),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=SimpleNamespace(
                    materialized=False,
                    reason="pre is deferred",
                    metadata_sha256="deferred",
                ),
            ),
            pytest.raises(ValueError, match="will not claim a runnable package"),
        ):
            write_gguf_runtime_package(pkg, tmp_path / "m.gguf", out)
        assert pkg.saved_to is None
        assert not out.exists()

    def test_replaced_source_tokenizer_rejects_before_save_or_output(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.GGUFModel",
                return_value=SimpleNamespace(metadata={}),
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
            write_gguf_runtime_package(pkg, tmp_path / "m.gguf", out)
        assert pkg.saved_to is None
        assert not out.exists()

    def test_failed_config_write_leaves_existing_output_unchanged(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        out.mkdir()
        sentinel = out / "sentinel.bin"
        sentinel.write_bytes(b"unchanged")
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.GGUFModel",
                return_value=SimpleNamespace(metadata={}),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=_materialized(),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.write_gguf_tokenizer_json",
                side_effect=_write_tokenizer,
            ),
            mock.patch(
                "mobius.integrations.onnx_genai.write_onnx_genai_config",
                side_effect=RuntimeError("config failed"),
            ),
            pytest.raises(RuntimeError, match="config failed"),
        ):
            write_gguf_runtime_package(pkg, tmp_path / "m.gguf", out)
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

    def test_failed_mtp_metadata_write_leaves_existing_output_unchanged(self, tmp_path):
        pkg = _FakePackage()
        pkg.mtp_head = SimpleNamespace(config=object())
        out = tmp_path / "out"
        out.mkdir()
        sentinel = out / "sentinel.bin"
        sentinel.write_bytes(b"unchanged")
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.GGUFModel",
                return_value=SimpleNamespace(metadata={}),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=_materialized(),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.write_gguf_tokenizer_json",
                side_effect=_write_tokenizer,
            ),
            mock.patch(
                "mobius.integrations.onnx_genai.write_onnx_genai_config",
                side_effect=_write_config,
            ),
            mock.patch(
                "mobius.integrations.onnx_genai.inference_metadata."
                "write_mtp_speculator_metadata",
                side_effect=RuntimeError("mtp metadata failed"),
            ),
            pytest.raises(RuntimeError, match="mtp metadata failed"),
        ):
            write_gguf_runtime_package(pkg, tmp_path / "m.gguf", out)
        assert {path.name: path.read_bytes() for path in out.iterdir()} == {
            "sentinel.bin": b"unchanged"
        }

    def test_unknown_runtime_rejects_before_source_read(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown runtime"):
            write_gguf_runtime_package(
                _FakePackage(), tmp_path / "m.gguf", tmp_path / "out", runtime="tflite"
            )

    def test_save_model_false_requires_an_existing_graph(self, tmp_path):
        with pytest.raises(ValueError, match=r"existing package containing model\.onnx"):
            write_gguf_runtime_package(
                _FakePackage(),
                tmp_path / "m.gguf",
                tmp_path / "missing",
                save_model=False,
            )

    def test_save_model_false_does_not_accept_an_mtp_only_graph(self, tmp_path):
        output = tmp_path / "output"
        (output / "mtp").mkdir(parents=True)
        (output / "mtp" / "model.onnx").write_bytes(b"mtp")
        with pytest.raises(ValueError, match="primary package graph"):
            write_gguf_runtime_package(
                _FakePackage(),
                tmp_path / "m.gguf",
                output,
                save_model=False,
            )

    def test_ort_genai_rejects_undeclared_mtp_sidecar(self, tmp_path):
        pkg = _FakePackage()
        pkg.mtp_head = SimpleNamespace(config=object())
        with pytest.raises(ValueError, match="does not yet have a declared GGUF MTP"):
            write_gguf_runtime_package(
                pkg,
                tmp_path / "m.gguf",
                tmp_path / "output",
                runtime="ort-genai",
            )

    def test_target_coupled_draft_rejects_before_source_read(self, tmp_path):
        pkg = _FakePackage()
        pkg.draft_manifest = {"architecture": "eagle3"}
        with pytest.raises(ValueError, match="target-coupled speculative draft"):
            write_gguf_runtime_package(pkg, tmp_path / "m.gguf", tmp_path / "out")
