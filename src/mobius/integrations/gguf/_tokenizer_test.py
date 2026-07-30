# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GGUF → onnx-genai runtime config emission (tokenizer + metadata)."""

from __future__ import annotations

import json
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
        _write_tokenizerless_gguf(gguf_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        fake_tokenizer = mock.Mock()
        fake_tokenizer.backend_tokenizer = mock.Mock()

        def _save_pretrained(path: str) -> None:
            (Path(path) / "tokenizer.json").write_text("{}")
            (Path(path) / "tokenizer_config.json").write_text("{}")

        fake_tokenizer.save_pretrained.side_effect = _save_pretrained

        fake_transformers = mock.Mock()
        fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tokenizer

        with mock.patch.dict("sys.modules", {"transformers": fake_transformers}):
            result = _tokenizer.write_gguf_tokenizer_json(gguf_path, out_dir)

        expected = os.path.join(str(out_dir), "tokenizer.json")
        assert result == expected
        assert (out_dir / "tokenizer.json").exists()
        assert (out_dir / "tokenizer_config.json").exists()
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


def _write_gguf_with_bpe_tokenizer(path):
    """Write a minimal GGUF carrying a byte-fallback BPE tokenizer."""
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), "llama")
    writer.add_context_length(8)
    writer.add_embedding_length(8)
    writer.add_block_count(1)
    writer.add_head_count(1)
    # tokens: specials + a few pieces that compose via merges (SentencePiece '▁').
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>", "▁", "h", "i", "▁h", "▁hi"]
    types = [3, 3, 3, 2, 1, 1, 1, 1, 1]  # 3=control, 2=unknown, 1=normal
    merges = ["▁ h", "▁h i"]  # ▁+h -> ▁h ; ▁h+i -> ▁hi
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(2)
    writer.add_eos_token_id(1)
    writer.add_unk_token_id(3)
    writer.add_add_bos_token(True)
    writer.add_chat_template(
        "{% for message in messages %}{{ message['role'] }}: "
        "{{ message['content'] }}{% endfor %}"
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TestReconstructTokenizerFromGgml:
    """Fallback reconstruction of tokenizer.json from GGUF ggml metadata."""

    def test_reconstructs_bpe_tokenizer_with_correct_ids(self, tmp_path):
        from tokenizers import Tokenizer

        from mobius.integrations.gguf._tokenizer import _reconstruct_tokenizer_from_ggml

        gguf_path = tmp_path / "tok.gguf"
        _write_gguf_with_bpe_tokenizer(gguf_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = _reconstruct_tokenizer_from_ggml(gguf_path, out_dir)

        assert result == str(out_dir / "tokenizer.json")
        tok = Tokenizer.from_file(result)
        assert tok.get_vocab_size() == 9
        # Token ids match the ggml ordering (no off-by-one from the reader).
        assert tok.token_to_id("<bos>") == 2
        assert tok.token_to_id("<eos>") == 1
        tokenizer_config = (out_dir / "tokenizer_config.json").read_text()
        assert '"tokenizer_class": "LlamaTokenizer"' in tokenizer_config
        assert (out_dir / "chat_template.jinja").read_text() == (
            "{% for message in messages %}{{ message['role'] }}: "
            "{{ message['content'] }}{% endfor %}"
        )
        # add_bos_token=True prepends <bos>; '▁hi' composes via the two merges.
        enc = tok.encode("hi")
        assert enc.ids[0] == 2  # <bos>
        assert tok.decode(enc.ids) == "hi"

    def test_gemma4_uses_ort_compatible_chat_template(self, tmp_path):
        from mobius.integrations.gguf._tokenizer import _write_chat_template
        from mobius.integrations.ort_genai.chat_template import (
            GEMMA4_ORT_CHAT_TEMPLATE,
        )

        tokenizer_config_path = tmp_path / "tokenizer_config.json"
        tokenizer_config_path.write_text("{}")

        path = _write_chat_template(
            {
                "general.architecture": "gemma4",
                "tokenizer.chat_template": "{{ raise_exception('unsupported') }}",
            },
            tmp_path,
        )

        assert path == str(tmp_path / "chat_template.jinja")
        assert (tmp_path / "chat_template.jinja").read_text() == GEMMA4_ORT_CHAT_TEMPLATE
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
        assert tokenizer_config["chat_template"] == GEMMA4_ORT_CHAT_TEMPLATE
