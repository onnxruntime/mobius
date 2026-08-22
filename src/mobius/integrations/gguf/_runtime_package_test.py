# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the shared GGUF runtime-package emitter."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from mobius.integrations.gguf import write_gguf_runtime_package


class _FakePackage:
    """Minimal stand-in for ModelPackage: records whether save() ran.

    Deliberately not a ``dict`` subclass — the emitter only calls ``save()``
    and reads ``config``, so mapping behaviour would be unused state.
    """

    def __init__(self):
        self.config = object()
        self.saved_to: str | None = None

    def save(self, path, **kwargs):
        self.saved_to = path
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.onnx").write_bytes(b"stub")


class TestWriteGgufRuntimePackage:
    """A saved package must be loadable: graph + tokenizer + runtime contract."""

    def test_emits_graph_tokenizer_and_inference_metadata(self, tmp_path):
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
                return_value={
                    "inference_metadata": str(out / "inference_metadata.yaml")
                },
            ) as write_cfg,
        ):
            artifacts = write_gguf_runtime_package(pkg, tmp_path / "m.gguf", out)

        assert pkg.saved_to == str(out)
        assert artifacts["tokenizer"] == tok
        assert artifacts["inference_metadata"].endswith("inference_metadata.yaml")
        write_tok.assert_called_once()
        write_cfg.assert_called_once()

    def test_ort_genai_runtime_emits_genai_config(self, tmp_path):
        """Both runtimes are reachable from the one entry point.

        A GGUF checkpoint has no Hugging Face source directory, so the
        ort-genai writer is called without one; the tokenizer rebuilt from the
        GGUF is the one the package ships.
        """
        pkg = _FakePackage()
        out = tmp_path / "out"

        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.write_gguf_tokenizer_json",
                return_value=str(out / "tokenizer.json"),
            ),
            mock.patch(
                "mobius.integrations.ort_genai.write_ort_genai_config",
                return_value={"genai_config": str(out / "genai_config.json")},
            ) as write_cfg,
        ):
            artifacts = write_gguf_runtime_package(
                pkg, tmp_path / "m.gguf", out, runtime="ort-genai"
            )

        assert artifacts["genai_config"].endswith("genai_config.json")
        assert "inference_metadata" not in artifacts
        write_cfg.assert_called_once()

    def test_unknown_runtime_is_rejected(self, tmp_path):
        """Fail closed on an unrecognised runtime rather than silently picking one."""
        with pytest.raises(ValueError, match="Unknown runtime"):
            write_gguf_runtime_package(
                _FakePackage(),
                tmp_path / "m.gguf",
                tmp_path / "out",
                runtime="tflite",
            )

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
