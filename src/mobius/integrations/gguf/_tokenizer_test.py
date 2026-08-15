# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GGUF → onnx-genai runtime config emission (tokenizer + metadata)."""

from __future__ import annotations

import hashlib
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


def _write_gguf_with_pixtral_gpt2_tokenizer(path: Path) -> None:
    """Write a tiny GGUF with the Nemotron/Pixtral ByteLevel BPE profile."""
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), "nemotron_h_moe")
    tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "Hello",
        "ĠcafÃ©",
        "ä½łå¥½",
        "x",
        "y",
        "xy",
    ]
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("pixtral")
    writer.add_token_list(tokens)
    writer.add_token_types([3, 3, 3, 3, 3, 4, 1, 1, 1, 1, 1, 1])
    writer.add_token_merges(["x y"])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TestPixtralGpt2Tokenizer:
    """Strict ByteLevel reconstruction for Nemotron's Pixtral/GPT-2 metadata."""

    def test_profile_selection(self):
        from mobius.integrations.gguf._tokenizer import _is_pixtral_gpt2_profile

        assert _is_pixtral_gpt2_profile(
            {
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.ggml.pre": "pixtral",
            }
        )
        assert not _is_pixtral_gpt2_profile({"tokenizer.ggml.model": "gpt2"})
        assert not _is_pixtral_gpt2_profile(
            {
                "tokenizer.ggml.model": "llama",
                "tokenizer.ggml.pre": "pixtral",
            }
        )

    def test_reconstructs_bytelevel_ids_and_contract(self, tmp_path: Path):
        from tokenizers import Tokenizer

        from mobius.integrations.gguf._tokenizer import _reconstruct_tokenizer_from_ggml

        gguf_path = tmp_path / "nemotron.gguf"
        _write_gguf_with_pixtral_gpt2_tokenizer(gguf_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = _reconstruct_tokenizer_from_ggml(gguf_path, out_dir)

        assert result == str(out_dir / "tokenizer.json")
        tokenizer = Tokenizer.from_file(result)
        assert tokenizer.encode("Hello café").ids == [6, 7]
        assert tokenizer.encode("你好").ids == [8]
        assert tokenizer.encode("<|im_end|>").ids == [4]
        assert tokenizer.encode("<think>").ids == [5]

        serialized = json.loads(Path(result).read_text(encoding="utf-8"))
        assert serialized["model"]["type"] == "BPE"
        assert serialized["model"]["ignore_merges"] is True
        assert serialized["pre_tokenizer"]["type"] == "Sequence"
        assert serialized["pre_tokenizer"]["pretokenizers"][1] == {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        }
        assert serialized["decoder"]["type"] == "ByteLevel"
        assert serialized["post_processor"] == {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": False,
            "use_regex": True,
        }
        added_tokens = {token["id"]: token for token in serialized["added_tokens"]}
        assert added_tokens[4]["special"] is True
        assert added_tokens[5]["special"] is False

    def test_public_writer_bypasses_autotokenizer(self, tmp_path: Path):
        from mobius.integrations.gguf import _tokenizer

        gguf_path = tmp_path / "nemotron.gguf"
        _write_gguf_with_pixtral_gpt2_tokenizer(gguf_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        fake_transformers = mock.Mock()
        fake_transformers.AutoTokenizer.from_pretrained.side_effect = AssertionError(
            "strict GPT-2 profile must not use AutoTokenizer"
        )
        with mock.patch.dict("sys.modules", {"transformers": fake_transformers}):
            result = _tokenizer.write_gguf_tokenizer_json(gguf_path, out_dir)

        assert result == str(out_dir / "tokenizer.json")
        fake_transformers.AutoTokenizer.from_pretrained.assert_not_called()

    def test_rejects_incomplete_strict_metadata(self, tmp_path: Path):
        from mobius.integrations.gguf._tokenizer import _reconstruct_pixtral_gpt2_tokenizer

        with pytest.raises(ValueError, match="token_type"):
            _reconstruct_pixtral_gpt2_tokenizer(
                {
                    "tokenizer.ggml.tokens": ["a"],
                    "tokenizer.ggml.merges": ["a b"],
                },
                tmp_path,
            )

    def test_private_fallback_preserves_strict_metadata_errors(self, tmp_path: Path):
        from mobius.integrations.gguf._tokenizer import _reconstruct_tokenizer_from_ggml

        metadata = {
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "pixtral",
            "tokenizer.ggml.tokens": ["a"],
            "tokenizer.ggml.merges": ["a b"],
        }
        with (
            mock.patch(
                "mobius.integrations.gguf._reader.GGUFModel",
                return_value=mock.Mock(metadata=metadata),
            ),
            pytest.raises(ValueError, match="token_type"),
        ):
            _reconstruct_tokenizer_from_ggml(tmp_path / "malformed.gguf", tmp_path)

    def test_pinned_artifact_when_explicitly_requested(self, tmp_path: Path):
        """Validate pinned artifact checksums without downloading it in normal tests."""
        artifact = os.environ.get("MOBIUS_NEMOTRON_GGUF_TOKENIZER_TEST_PATH")
        if artifact is None:
            pytest.skip(
                "Set MOBIUS_NEMOTRON_GGUF_TOKENIZER_TEST_PATH to run artifact validation."
            )

        from mobius.integrations.gguf._reader import GGUFModel
        from mobius.integrations.gguf._tokenizer import _reconstruct_tokenizer_from_ggml

        metadata = GGUFModel(artifact).metadata
        assert len(metadata["tokenizer.ggml.tokens"]) == 131072
        assert len(metadata["tokenizer.ggml.merges"]) == 269443
        assert (
            hashlib.sha256("\n".join(metadata["tokenizer.ggml.tokens"]).encode()).hexdigest()
            == "4999709474e3c967358c1f1199b6be65fb9055d3eb59e0cd387f9e7077fc40ed"
        )
        assert (
            hashlib.sha256("\n".join(metadata["tokenizer.ggml.merges"]).encode()).hexdigest()
            == "b1b0165185b1925118c2f7b1e978439b02010c3a420ebfec5c19a093a0d9b4cb"
        )

        result = _reconstruct_tokenizer_from_ggml(artifact, tmp_path)
        serialized = json.loads(Path(result).read_text(encoding="utf-8"))
        assert serialized["model"]["ignore_merges"] is True
        assert serialized["decoder"]["type"] == "ByteLevel"

    @pytest.mark.integration
    def test_pinned_artifact_matches_official_tokenizer(self, tmp_path: Path):
        artifact = os.environ.get("MOBIUS_NEMOTRON_GGUF_TOKENIZER_TEST_PATH")
        official_dir_value = os.environ.get("MOBIUS_NEMOTRON_OFFICIAL_TOKENIZER_DIR")
        if artifact is None or official_dir_value is None:
            pytest.skip(
                "Set MOBIUS_NEMOTRON_GGUF_TOKENIZER_TEST_PATH and "
                "MOBIUS_NEMOTRON_OFFICIAL_TOKENIZER_DIR for pinned parity."
            )

        from tokenizers import Tokenizer

        from mobius.integrations.gguf._reader import GGUFModel
        from mobius.integrations.gguf._tokenizer import _reconstruct_tokenizer_from_ggml

        official_dir = Path(official_dir_value)
        official_path = official_dir / "tokenizer.json"
        assert official_path.is_file()
        rebuilt_path = _reconstruct_tokenizer_from_ggml(Path(artifact), tmp_path)
        assert rebuilt_path is not None

        official = Tokenizer.from_file(str(official_path))
        rebuilt = Tokenizer.from_file(rebuilt_path)
        assert official.get_vocab(with_added_tokens=True) == rebuilt.get_vocab(
            with_added_tokens=True
        )
        samples = (
            "The capital of France is",
            " Paris.  \nThe capital of Germany",
            "Hello, world!",
            "café déjà vu — 你好 🌍",
            "  leading\tspaces\r\nnewlines  ",
            "<|im_start|>assistant\n<think>x</think><|im_end|>",
            bytes(range(1, 128)).decode("latin1"),
        )
        for sample in samples:
            official_encoding = official.encode(sample)
            rebuilt_encoding = rebuilt.encode(sample)
            assert rebuilt_encoding.ids == official_encoding.ids
            assert rebuilt.decode(
                rebuilt_encoding.ids, skip_special_tokens=False
            ) == official.decode(official_encoding.ids, skip_special_tokens=False)

        official_json = json.loads(official_path.read_text(encoding="utf-8"))
        rebuilt_json = json.loads(Path(rebuilt_path).read_text(encoding="utf-8"))
        official_added = {
            (token["id"], token["content"]): token["special"]
            for token in official_json["added_tokens"]
        }
        rebuilt_added = {
            (token["id"], token["content"]): token["special"]
            for token in rebuilt_json["added_tokens"]
        }
        assert rebuilt_added == official_added
        assert rebuilt_json["pre_tokenizer"] == official_json["pre_tokenizer"]
        assert rebuilt_json["decoder"] == official_json["decoder"]
        assert rebuilt_json["post_processor"] == official_json["post_processor"]

        metadata = GGUFModel(artifact).metadata
        gguf_template = metadata["tokenizer.chat_template"].replace("\r\n", "\n")
        official_template = (
            (official_dir / "chat_template.jinja")
            .read_text(encoding="utf-8")
            .replace("\r\n", "\n")
        )
        assert gguf_template != official_template
        assert hashlib.sha256(gguf_template.encode()).hexdigest() == (
            "cbb337473ffde036fd4b6e7e7763dcb97c7cd8b4a311cd52d361d2766b00eb7c"
        )
        assert hashlib.sha256(
            (official_dir / "chat_template.jinja").read_bytes()
        ).hexdigest() == ("58933db77d3099b4f78c55a38347a72e1ea05b97d6bd8f38775303dc0194e0a9")
