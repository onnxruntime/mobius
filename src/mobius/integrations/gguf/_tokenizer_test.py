# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GGUF → onnx-genai runtime config emission (tokenizer + metadata)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


def _write_tokenizerless_gguf(path: Path) -> None:
    """Write a minimal weights-only llama GGUF with no tokenizer metadata."""
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), "llama")
    writer.add_context_length(64)
    writer.add_embedding_length(16)
    writer.add_feed_forward_length(32)
    writer.add_block_count(1)
    writer.add_head_count(2)
    writer.add_head_count_kv(2)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(32)
    writer.add_tensor("token_embd.weight", np.random.randn(32, 16).astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TestWriteGgufTokenizerJson:
    """Tests for ``write_gguf_tokenizer_json`` (best-effort tokenizer emission)."""

    def test_skips_gracefully_without_tokenizer_metadata(self, tmp_path: Path):
        """A GGUF with no ggml tokenizer metadata yields no tokenizer.json, no raise."""
        from mobius.integrations.gguf import write_gguf_tokenizer_json

        gguf_path = tmp_path / "tokenizerless.gguf"
        _write_tokenizerless_gguf(gguf_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = write_gguf_tokenizer_json(gguf_path, out_dir)

        assert result is None
        assert not (out_dir / "tokenizer.json").exists()

    def test_serializes_reconstructed_fast_tokenizer(self, tmp_path: Path):
        """When transformers reconstructs a fast tokenizer, its backend is saved."""
        from mobius.integrations.gguf import _tokenizer

        gguf_path = tmp_path / "model.gguf"
        gguf_path.write_bytes(b"GGUF")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        saved_to: dict[str, str] = {}

        class _FakeBackend:
            def save(self, path: str) -> None:
                saved_to["path"] = path
                Path(path).write_text("{}")

        fake_tokenizer = mock.Mock()
        fake_tokenizer.backend_tokenizer = _FakeBackend()

        fake_transformers = mock.Mock()
        fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tokenizer

        with mock.patch.dict("sys.modules", {"transformers": fake_transformers}):
            result = _tokenizer.write_gguf_tokenizer_json(gguf_path, out_dir)

        expected = os.path.join(str(out_dir), "tokenizer.json")
        assert result == expected
        assert saved_to["path"] == expected
        assert (out_dir / "tokenizer.json").exists()
        # Loaded from the GGUF file via its embedded metadata, not an HF repo.
        _, kwargs = fake_transformers.AutoTokenizer.from_pretrained.call_args
        assert kwargs["gguf_file"] == "model.gguf"


class TestGgufOnnxGenaiEmission:
    """The onnx-genai metadata is emitted for a GGUF-built decoder package."""

    def test_write_onnx_genai_config_emits_inference_metadata(self, tmp_path: Path):
        pytest.importorskip("onnx")
        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.onnx_genai import write_onnx_genai_config

        gguf_path = tmp_path / "model.gguf"
        _write_tokenizerless_gguf(gguf_path)
        pkg = build_from_gguf(gguf_path)
        out_dir = tmp_path / "onnx"
        out_dir.mkdir()

        write_onnx_genai_config(
            pkg, str(out_dir), config=getattr(pkg, "config", None), source=None
        )

        assert (out_dir / "inference_metadata.yaml").exists()
