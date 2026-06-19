# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the NeMo ``.nemo`` archive reader.

These tests synthesise a tiny ``.nemo`` tar archive (config + checkpoint +
tokenizer stub) so they run fully offline, with no model download.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import torch
import yaml

from mobius.integrations.nemo._reader import NeMoArchive, _looks_like_hf_repo_id

_TINY_CONFIG = {
    "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
    "tokenizer": {
        "type": "bpe",
        "model_path": "nemo:abc_tokenizer.model",
        "vocab_path": "nemo:def_vocab.txt",
    },
    "encoder": {"d_model": 16, "n_layers": 1},
}


def _write_tiny_nemo(path: Path) -> dict[str, torch.Tensor]:
    state_dict = {
        "encoder.layers.0.self_attn.linear_q.weight": torch.randn(16, 16),
        "decoder.prediction.embed.weight": torch.randn(8, 4),
        "joint.enc.weight": torch.randn(4, 16),
    }
    ckpt_buf = io.BytesIO()
    torch.save(state_dict, ckpt_buf)

    def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    with tarfile.open(path, mode="w") as tar:
        _add_bytes(tar, "./model_config.yaml", yaml.safe_dump(_TINY_CONFIG).encode())
        _add_bytes(tar, "./model_weights.ckpt", ckpt_buf.getvalue())
        _add_bytes(tar, "./abc_tokenizer.model", b"fake-spm-model")
        _add_bytes(tar, "./def_vocab.txt", b"<unk>\na\nb\n")
    return state_dict


class TestNeMoArchive:
    def test_reads_config_and_target(self, tmp_path: Path):
        nemo = tmp_path / "tiny.nemo"
        _write_tiny_nemo(nemo)
        archive = NeMoArchive(nemo)
        assert archive.target.endswith("EncDecRNNTBPEModel")
        assert archive.config["encoder"]["d_model"] == 16

    def test_state_dict_roundtrip(self, tmp_path: Path):
        nemo = tmp_path / "tiny.nemo"
        expected = _write_tiny_nemo(nemo)
        archive = NeMoArchive(nemo)
        state_dict = archive.state_dict()
        assert set(state_dict) == set(expected)
        for key, value in expected.items():
            assert torch.allclose(state_dict[key], value)

    def test_extract_tokenizer(self, tmp_path: Path):
        nemo = tmp_path / "tiny.nemo"
        _write_tiny_nemo(nemo)
        archive = NeMoArchive(nemo)
        written = archive.extract_tokenizer(tmp_path / "tok")
        assert set(written) == {"model_path", "vocab_path"}
        assert Path(written["model_path"]).read_bytes() == b"fake-spm-model"
        assert Path(written["vocab_path"]).read_text().startswith("<unk>")

    def test_resolve_nemo_uri(self, tmp_path: Path):
        nemo = tmp_path / "tiny.nemo"
        _write_tiny_nemo(nemo)
        archive = NeMoArchive(nemo)
        assert archive.resolve_nemo_uri("nemo:foo.model") == "foo.model"
        assert archive.resolve_nemo_uri("bar.model") == "bar.model"
        assert archive.resolve_nemo_uri(None) is None

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            NeMoArchive(tmp_path / "does-not-exist.nemo")

    def test_archive_without_config_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.nemo"
        with tarfile.open(bad, mode="w") as tar:
            data = b"nope"
            info = tarfile.TarInfo(name="./random.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        with pytest.raises(ValueError, match=r"model_config\.yaml"):
            NeMoArchive(bad)


class TestLooksLikeHfRepoId:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("nvidia/nemotron-speech-streaming-en-0.6b", True),
            ("owner/repo", True),
            ("./local/path.nemo", False),
            ("/abs/path.nemo", False),
            ("single", False),
            ("a/b/c", False),
            ("owner/model.nemo", False),
        ],
    )
    def test_classification(self, value: str, expected: bool):
        assert _looks_like_hf_repo_id(value) is expected
