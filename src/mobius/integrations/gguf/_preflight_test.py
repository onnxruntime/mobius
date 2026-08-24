# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the metadata-only GGUF preflight (:mod:`._preflight`).

All assertions here use local synthetic split sets — no network, no tensor
downloads. The Hub path (:func:`preflight_hf_gguf`) is exercised via the shared
helpers with a fake ``HfApi`` so the metadata-only contract is enforced without
touching the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mobius.integrations.gguf._preflight import (
    _assess_sparse_moe,
    _detect_quantization,
    _select_shard_files,
    preflight_gguf,
    preflight_local_gguf,
)


def _write_sharded_gguf(
    directory: Path,
    *,
    architecture: str = "llama",
    stem: str = "tiny",
    split_max_tensors: int = 3,
    num_experts: int | None = None,
) -> list[Path]:
    from gguf import GGUFWriter

    directory.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(
        str(directory / stem), architecture, split_max_tensors=split_max_tensors
    )
    writer.add_context_length(128)
    writer.add_embedding_length(16)
    writer.add_feed_forward_length(32)
    writer.add_block_count(3)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_vocab_size(32)
    if num_experts is not None:
        writer.add_expert_count(num_experts)

    rng = np.random.default_rng(0)
    names = ["token_embd.weight"]
    for i in range(3):
        names.append(f"blk.{i}.attn_q.weight")
        names.append(f"blk.{i}.ffn_up.weight")
    names.append("output.weight")
    for name in names:
        writer.add_tensor(name, rng.standard_normal((8, 16)).astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return sorted(directory.glob(f"{stem}-*.gguf"))


# --------------------------------------------------------------------------- #
# Local preflight (metadata only)
# --------------------------------------------------------------------------- #


def test_local_preflight_reports_files_and_bytes(tmp_path):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=3)
    report = preflight_local_gguf(shards[0])
    assert report.location == "local"
    assert report.is_sharded
    assert report.split_count == len(shards)
    assert report.total_files == len(shards)
    assert report.total_bytes == sum(s.stat().st_size for s in shards)
    assert {f.filename for f in report.files} == {s.name for s in shards}
    # No quant/MoE => nothing blocks a (hypothetical) build.
    assert report.exportable
    assert report.sparse_moe_fusion_supported


def test_local_preflight_checksums_optional(tmp_path):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=3)
    no_sums = preflight_local_gguf(shards[0])
    assert all(f.sha256 is None for f in no_sums.files)
    with_sums = preflight_local_gguf(shards[0], verify_checksums=True)
    assert all(f.sha256 and len(f.sha256) == 64 for f in with_sums.files)


def test_local_preflight_single_file(tmp_path):
    from gguf import GGUFWriter

    path = tmp_path / "plain.gguf"
    writer = GGUFWriter(str(path), "llama")
    writer.add_context_length(128)
    writer.add_embedding_length(16)
    writer.add_block_count(1)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_vocab_size(32)
    writer.add_tensor("token_embd.weight", np.zeros((8, 16), np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    report = preflight_local_gguf(path)
    assert not report.is_sharded
    assert report.total_files == 1
    assert report.split_count == 1


def test_report_json_roundtrip_is_resumable(tmp_path):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=3)
    cache = tmp_path / "pf.json"
    first = preflight_gguf(shards[0], cache_path=cache)
    assert cache.exists()
    # Second call loads from cache (idempotent, no re-read needed).
    second = preflight_gguf(shards[0], cache_path=cache)
    assert first.as_dict() == second.as_dict()
    # The cache is valid JSON that deserialises back into a report.
    data = json.loads(cache.read_text())
    assert data["total_bytes"] == first.total_bytes


# --------------------------------------------------------------------------- #
# Sparse-MoE fusion assessment (the honesty blocker, metadata-level)
# --------------------------------------------------------------------------- #


def test_assess_sparse_moe_blocks_iq1_moe():
    ok, blockers = _assess_sparse_moe(
        model_type="glm_moe_dsa",
        num_experts=160,
        quantization="IQ1_S",
        source="unsloth/GLM-5.2-GGUF",
    )
    assert ok is False
    assert len(blockers) == 1
    assert "sparse" in blockers[0].lower()
    assert "dense-all-expert" in blockers[0]


def test_assess_sparse_moe_allows_non_moe():
    ok, blockers = _assess_sparse_moe(
        model_type="llama",
        num_experts=None,
        quantization="IQ1_S",
        source="x",
    )
    assert ok is True
    assert blockers == []


def test_assess_sparse_moe_allows_int4_moe():
    # int4 experts repack to MatMulNBits and fuse into QMoE.
    ok, blockers = _assess_sparse_moe(
        model_type="glm_moe_dsa",
        num_experts=160,
        quantization=None,  # no native-block quant detected
        source="x",
    )
    assert ok is True
    assert blockers == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("GLM-5.2-UD-IQ1_S-00001-of-00006.gguf", "IQ1_S"),
        ("model-UD-IQ1_M.gguf", "IQ1_M"),
        ("q4_k_m.gguf", None),
        ("model-IQ2_XXS.gguf", "IQ2_XXS"),
        ("model-MXFP4.gguf", "MXFP4"),
    ],
)
def test_detect_quantization(text, expected):
    assert _detect_quantization(text) == expected


# --------------------------------------------------------------------------- #
# Shard file selection (remote listing logic, no network)
# --------------------------------------------------------------------------- #


def test_select_shard_files_expands_split_set():
    files = [
        "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00001-of-00006.gguf",
        "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00002-of-00006.gguf",
        "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00003-of-00006.gguf",
        "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00004-of-00006.gguf",
        "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00005-of-00006.gguf",
        "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00006-of-00006.gguf",
        "UD-IQ1_M/GLM-5.2-UD-IQ1_M-00001-of-00006.gguf",
    ]
    selected = _select_shard_files(files, "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00001-of-00006.gguf")
    assert len(selected) == 6
    assert all("UD-IQ1_S" in f for f in selected)


def test_select_shard_files_single_plain_file():
    files = ["model-Q4_K_M.gguf"]
    selected = _select_shard_files(files, None)
    assert selected == ["model-Q4_K_M.gguf"]


# --------------------------------------------------------------------------- #
# Hub preflight with a fake API (metadata-only contract)
# --------------------------------------------------------------------------- #


class _FakeLfs:
    def __init__(self, sha256):
        self.sha256 = sha256


class _FakeRepoFile:
    def __init__(self, path, size, sha):
        self.path = path
        self.size = size
        self.lfs = _FakeLfs(sha)


class _FakeModelInfo:
    def __init__(self, gguf):
        self.gguf = gguf


class _FakeHfApi:
    """A stand-in that returns metadata and asserts no download is attempted."""

    def __init__(self, files, sizes, arch, experts=None):
        self._files = files
        self._sizes = sizes
        self._arch = arch
        self._experts = experts
        self.downloaded = []

    def list_repo_files(self, repo_id, revision=None, token=None):
        return self._files

    def get_paths_info(self, repo_id, paths, revision=None, token=None, expand=False):
        return [_FakeRepoFile(p, self._sizes[p], f"{i:064x}") for i, p in enumerate(paths)]

    def model_info(self, repo_id, revision=None, token=None, expand=None):
        gguf = {"architecture": self._arch, "total": 753_000_000_000}
        if self._experts is not None:
            gguf["expert_count"] = self._experts
        return _FakeModelInfo(gguf)

    def hf_hub_download(self, *a, **k):  # pragma: no cover - must never be called
        raise AssertionError("preflight must not download tensor payloads")


def test_hf_preflight_metadata_only(monkeypatch):
    from mobius.integrations.gguf import _preflight

    files = [f"UD-IQ1_S/GLM-5.2-UD-IQ1_S-{i:05d}-of-00006.gguf" for i in range(1, 7)]
    sizes = dict.fromkeys(files, 49000000000)
    sizes[files[0]] = 9_000_000
    fake = _FakeHfApi(files, sizes, arch="glm-dsa", experts=160)

    monkeypatch.setattr(_preflight, "HfApi", lambda *a, **k: fake, raising=False)
    # HfApi is imported inside the function; patch the huggingface_hub symbol.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: fake)

    report = _preflight.preflight_hf_gguf("unsloth/GLM-5.2-GGUF", filename=files[0])
    assert report.location == "hf"
    assert report.architecture == "glm-dsa"
    assert report.resolved_model_type == "glm_moe_dsa"
    assert report.quantization == "IQ1_S"
    assert report.total_files == 6
    assert report.total_bytes == sum(sizes.values())
    assert all(f.sha256 for f in report.files)
    # The IQ1 MoE honesty blocker must be present.
    assert not report.exportable
    assert any("sparse-MoE" in b for b in report.blockers)
    # And nothing was downloaded.
    assert fake.downloaded == []
