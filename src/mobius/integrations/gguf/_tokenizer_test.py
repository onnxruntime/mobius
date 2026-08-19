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

    def test_restores_bos_post_processor_when_primary_path_omits_it(self, tmp_path):
        """A reconstructed backend lacking BOS gets the GGUF's BOS post-processor.

        Regression: transformers' GGUF loader (e.g. for Gemma) can return a fast
        tokenizer whose post-processor does NOT prepend ``<bos>``. Gemma requires
        it — without BOS, greedy decode degenerates into token repetition. The
        emitter must restore the BOS post-processor from the GGUF metadata.
        """
        from tokenizers import Tokenizer
        from tokenizers.models import BPE

        from mobius.integrations.gguf import _tokenizer

        gguf_path = tmp_path / "model.gguf"
        _write_gguf_with_bpe_tokenizer(gguf_path)  # add_bos_token=True, bos_id=2
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # A backend with a valid vocab but NO BOS post-processor.
        vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "h": 4, "i": 5}
        backend = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
        assert backend.encode("hi").ids[0] != 2  # no BOS before the fix

        fake_tokenizer = mock.Mock()
        fake_tokenizer.backend_tokenizer = backend
        fake_transformers = mock.Mock()
        fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tokenizer

        with mock.patch.dict("sys.modules", {"transformers": fake_transformers}):
            result = _tokenizer.write_gguf_tokenizer_json(gguf_path, out_dir)

        saved = Tokenizer.from_file(result)
        assert saved.encode("hi").ids[0] == 2  # BOS now prepended


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
        # add_bos_token=True prepends <bos>; '▁hi' composes via the two merges.
        enc = tok.encode("hi")
        assert enc.ids[0] == 2  # <bos>
        assert tok.decode(enc.ids) == "hi"
