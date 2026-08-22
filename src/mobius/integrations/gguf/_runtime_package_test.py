# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the shared GGUF runtime-package emitter."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from mobius.integrations.gguf import write_gguf_runtime_package


class _FakePackage(dict):
    """Minimal stand-in for ModelPackage: records whether save() ran."""

    def __init__(self):
        super().__init__()
        self.config = object()
        self.saved_to: str | None = None

    def save(self, path, **kwargs):
        self.saved_to = path
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.onnx").write_bytes(b"stub")


class TestWriteGgufRuntimePackage:
    """A saved package must be loadable: graph + tokenizer + metadata."""

    def test_emits_graph_tokenizer_and_metadata(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        tok = str(out / "tokenizer.json")

        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.write_gguf_tokenizer_json",
                return_value=tok,
            ) as write_tok,
            mock.patch(
                "mobius.integrations.onnx_genai.write_onnx_genai_config",
                return_value={"inference_metadata": str(out / "inference_metadata.yaml")},
            ) as write_cfg,
        ):
            artifacts = write_gguf_runtime_package(pkg, tmp_path / "m.gguf", out)

        assert pkg.saved_to == str(out)
        assert artifacts["tokenizer"] == tok
        assert artifacts["inference_metadata"].endswith("inference_metadata.yaml")
        write_tok.assert_called_once()
        write_cfg.assert_called_once()

    def test_save_model_false_leaves_an_already_saved_graph_alone(self, tmp_path):
        """The CLI saves the graph itself, then asks only for runtime artifacts."""
        pkg = _FakePackage()
        out = tmp_path / "out"

        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.write_gguf_tokenizer_json",
                return_value=None,
            ),
            mock.patch(
                "mobius.integrations.onnx_genai.write_onnx_genai_config",
                return_value={"inference_metadata": "x.yaml"},
            ),
        ):
            artifacts = write_gguf_runtime_package(
                pkg, tmp_path / "m.gguf", out, save_model=False
            )

        assert pkg.saved_to is None
        # A GGUF with no tokenizer metadata yields no tokenizer key rather than
        # a None value the caller would have to special-case.
        assert "tokenizer" not in artifacts
        assert artifacts["inference_metadata"] == "x.yaml"
