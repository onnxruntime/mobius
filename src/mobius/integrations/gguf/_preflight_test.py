# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the metadata-only GGUF preflight (:mod:`._preflight`).

All assertions here use local synthetic split sets — no network, no tensor
downloads. The Hub path (:func:`preflight_hf_gguf`) is exercised via the shared
helpers with a fake ``HfApi`` so the metadata-only contract is enforced without
touching the network.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pytest

from mobius.integrations.gguf._preflight import (
    _assess_sparse_moe,
    _classify_tensor_type,
    _detect_quantization,
    _matmulnbits_output_bytes,
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


def _raw_quant_rows(qtype, n_rows: int, k: int) -> np.ndarray:
    """Zeroed raw-block bytes shaped ``(n_rows, bytes_per_row)`` for *qtype*."""
    from gguf import GGML_QUANT_SIZES

    block_elems, block_bytes = GGML_QUANT_SIZES[qtype]
    bytes_per_row = (k // block_elems) * block_bytes
    return np.zeros((n_rows, bytes_per_row), dtype=np.uint8)


def _raw_quant_expert_rows(qtype, n_expert: int, n_rows: int, k: int) -> np.ndarray:
    """Zeroed raw-block bytes for a 3D routed-expert tensor.

    Shaped ``(n_expert, n_rows, bytes_per_row)`` — the layout of a real GGUF
    ``blk.i.ffn_*_exps.weight`` fused-expert tensor whose rows are *qtype*
    native blocks.
    """
    from gguf import GGML_QUANT_SIZES

    block_elems, block_bytes = GGML_QUANT_SIZES[qtype]
    bytes_per_row = (k // block_elems) * block_bytes
    return np.zeros((n_expert, n_rows, bytes_per_row), dtype=np.uint8)


def _write_quant_sharded_gguf(
    directory: Path,
    *,
    architecture: str = "llama",
    stem: str = "q",
    split_max_tensors: int = 2,
    include_unsupported: bool = False,
) -> list[Path]:
    """Write a shape-faithful multi-shard GGUF with mixed real quant types.

    F32 (passthrough), Q8_0 (repack), IQ1_S (native-preserve), and — when
    *include_unsupported* — Q5_K (no lossless path). Bodies are zeroed; only the
    headers (types/dims) matter to the metadata-only preflight. This dense
    (non-MoE) fixture is deliberately expert-free; sparse routed-expert coverage
    lives in :func:`_write_glm_moe_iq1_sharded_gguf`.
    """
    from gguf import GGMLQuantizationType, GGUFWriter

    directory.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(
        str(directory / stem), architecture, split_max_tensors=split_max_tensors
    )
    writer.add_context_length(128)
    writer.add_embedding_length(256)
    writer.add_block_count(2)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_vocab_size(32)

    k = 256
    writer.add_tensor("token_embd.weight", np.zeros((8, k), np.float32))
    writer.add_tensor(
        "blk.0.attn_q.weight",
        _raw_quant_rows(GGMLQuantizationType.Q8_0, 8, k),
        raw_dtype=GGMLQuantizationType.Q8_0,
    )
    writer.add_tensor(
        "blk.0.ffn_up.weight",
        _raw_quant_rows(GGMLQuantizationType.IQ1_S, 8, k),
        raw_dtype=GGMLQuantizationType.IQ1_S,
    )
    if include_unsupported:
        writer.add_tensor(
            "blk.1.ffn_down.weight",
            _raw_quant_rows(GGMLQuantizationType.Q5_K, 8, k),
            raw_dtype=GGMLQuantizationType.Q5_K,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return sorted(directory.glob(f"{stem}-*.gguf"))


def _write_glm_moe_iq1_sharded_gguf(
    directory: Path,
    *,
    real_prefix: str = "GLM-5.2-UD-IQ1_S",
    num_experts: int = 8,
    split_max_tensors: int = 2,
) -> list[Path]:
    """Write a sharded MoE GGUF faithful to the flagship GLM-5.2 UD-IQ1_S set.

    Independently mirrors the three signals production preflight actually
    consumes for the sparse-MoE honesty gate (see ``preflight_local_gguf`` /
    ``_detect_quantization`` / ``_assess_sparse_moe``):

    * the shard **filenames** carry the exact real
      ``GLM-5.2-UD-IQ1_S-000i-of-000N.gguf`` quant tag that
      ``_detect_quantization`` reads from ``path.name`` (candidate #2);
    * ``general.architecture='glm-dsa'`` resolves to the MoE model type
      ``glm_moe_dsa`` and ``glm-dsa.expert_count`` marks it as routed-expert
      MoE (``_is_moe`` via ``num_experts``);
    * the routed experts are **real per-projection**
      ``blk.i.ffn_{gate,up,down}_exps.weight`` tensors physically stored as
      ``IQ1_S`` native blocks (3D ``[n_expert, n_ff, k]``), so the detected
      quant tag is consistent with the actual tensor payload.

    ``gguf.GGUFWriter`` derives split filenames from the output stem and
    truncates a dotted stem (``GLM-5.2`` -> ``GLM-5``), so the shards are first
    written under a dot-free stem and then renamed to the exact real dotted
    filename. Only the on-disk names change — the embedded split metadata
    (``split.no``/``split.count``) is written by ``GGUFWriter`` and left
    untouched, and shard discovery is filename-prefix based
    (``discover_gguf_shards``), so the renamed set enumerates correctly.
    """
    from gguf import GGMLQuantizationType, GGUFWriter

    directory.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(
        str(directory / "iq1moe"), "glm-dsa", split_max_tensors=split_max_tensors
    )
    writer.add_context_length(128)
    writer.add_embedding_length(256)
    writer.add_block_count(1)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_vocab_size(32)
    writer.add_expert_count(num_experts)

    k = 256
    n_ff = 8
    # F32 embedding (passthrough) + one repackable attention projection (Q8_0).
    writer.add_tensor("token_embd.weight", np.zeros((8, k), np.float32))
    writer.add_tensor(
        "blk.0.attn_q.weight",
        _raw_quant_rows(GGMLQuantizationType.Q8_0, 8, k),
        raw_dtype=GGMLQuantizationType.Q8_0,
    )
    # Real per-projection routed-expert weights, physically IQ1_S native blocks.
    for projection in ("gate", "up", "down"):
        writer.add_tensor(
            f"blk.0.ffn_{projection}_exps.weight",
            _raw_quant_expert_rows(GGMLQuantizationType.IQ1_S, num_experts, n_ff, k),
            raw_dtype=GGMLQuantizationType.IQ1_S,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    renamed: list[Path] = []
    for shard in sorted(directory.glob("iq1moe-*.gguf")):
        match = re.match(r"^iq1moe-(\d{5})-of-(\d{5})\.gguf$", shard.name)
        assert match is not None, shard.name
        index, count = match.group(1), match.group(2)
        target = directory / f"{real_prefix}-{index}-of-{count}.gguf"
        shard.rename(target)
        renamed.append(target)
    return sorted(renamed)


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


@pytest.mark.parametrize("quantization", [None, "IQ2_XXS"])
def test_assess_sparse_moe_rejects_every_nemotron_h_quantized_route(quantization):
    ok, blockers = _assess_sparse_moe(
        architecture="nemotron_h_moe",
        model_type="nemotron_h",
        num_experts=None,
        quantization=quantization,
        source="pinned/nemotron-h.gguf",
    )
    assert ok is False
    assert len(blockers) == 1
    assert "correction-biased sigmoid" in blockers[0]
    assert "ReLU2" in blockers[0]
    assert "com.microsoft::MoE/QMoE cannot represent" in blockers[0]
    assert "keep_quantized=False" in blockers[0]


def test_assess_sparse_moe_does_not_misclassify_dense_nemotron_h():
    ok, blockers = _assess_sparse_moe(
        architecture="nemotron_h",
        model_type="nemotron_h",
        num_experts=None,
        quantization="F16",
        source="pinned/nemotron-h.gguf",
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


# --------------------------------------------------------------------------- #
# Block-preserving lossless classification + real-artifact budget (folded from
# the superseded standalone GGUF export preflight; validated vs. the real
# repacker, not a model-name allowlist).
# --------------------------------------------------------------------------- #


def test_type_classification_matches_the_real_repacker():
    from gguf import GGMLQuantizationType as T

    # Float -> passthrough; int4/int8 K-quants -> repack; IQ blocks -> preserve;
    # a type with no lossless path -> unsupported (zero output bytes).
    passthrough = _classify_tensor_type(T.F32.value, "F32", (8, 256), 8192)
    assert passthrough[0] == "passthrough"
    assert passthrough[1] == 8192

    repack = _classify_tensor_type(T.Q8_0.value, "Q8_0", (8, 256), 2176)
    assert repack[0] == "repack"
    assert repack[1] == _matmulnbits_output_bytes((8, 256), 8, 32) > 0

    native = _classify_tensor_type(T.IQ1_S.value, "IQ1_S", (8, 256), 400)
    assert native[0] == "native-preserve"
    assert native[1] == 400  # byte-for-byte preserved

    unsupported = _classify_tensor_type(T.Q5_K.value, "Q5_K", (8, 256), 1408)
    assert unsupported[0] == "unsupported"
    assert unsupported[1] == 0


def test_local_preflight_supported_sharded_set_passes_shard_and_arch_gates(tmp_path):
    # Regression for the removed hardcoded ``split.count > 1`` refusal: a
    # supported *sharded* set must clear the shard and arch gates. Direct
    # multi-shard import (GgufShardSet) means being sharded is never a blocker.
    shards = _write_quant_sharded_gguf(tmp_path, split_max_tensors=2)
    assert len(shards) > 1
    report = preflight_local_gguf(shards[0])

    assert report.is_sharded
    assert report.split_count == len(shards)
    # Architecture resolves dynamically (no per-model code).
    assert report.architecture == "llama"
    assert report.resolved_model_type == "llama"
    # No blocker mentions sharding / split / merge / double-copy.
    joined = " ".join(report.blockers).lower()
    for forbidden in ("shard", "split", "merge", "double-copy", "second"):
        assert forbidden not in joined, report.blockers
    # Every type is on a lossless path, so the set is exportable.
    assert report.unsupported_types == []
    assert report.exportable
    assert {s.disposition for s in report.type_stats} <= {
        "passthrough",
        "repack",
        "native-preserve",
    }
    assert report.output_bytes and report.output_bytes > 0


def test_local_preflight_flags_exact_unsupported_native_qtypes(tmp_path):
    shards = _write_quant_sharded_gguf(tmp_path, split_max_tensors=2, include_unsupported=True)
    report = preflight_local_gguf(shards[0])

    # Being sharded is still not a blocker; the ONLY blocker is the exact
    # unsupported qtype (Q5_K here), never arch/sharding.
    assert report.is_sharded
    assert report.unsupported_types == ["Q5_K"]
    assert len(report.blockers) == 1
    assert "cannot be preserved losslessly" in report.blockers[0]
    assert "Q5_K" in report.blockers[0]
    # Unsupported tensors contribute zero output bytes (no dequantize widening).
    q5k = next(s for s in report.type_stats if s.type_name == "Q5_K")
    assert q5k.disposition == "unsupported"
    assert q5k.output_bytes == 0


def test_budget_reflects_real_artifacts_only_no_second_copy(tmp_path):
    shards = _write_quant_sharded_gguf(tmp_path, split_max_tensors=2)
    report = preflight_local_gguf(shards[0])

    # No merged/second-copy disk budget exists anymore.
    field_names = set(report.as_dict())
    assert not any(
        "merge" in name or "second" in name or "double" in name for name in field_names
    )
    # Download budget == the shard set itself (read in place, random access).
    assert report.total_bytes == sum(s.stat().st_size for s in shards)
    # VRAM is derived from the export artifact, not the source dtype bytes.
    assert report.vram_weights_bytes == report.output_bytes
    assert report.output_bytes <= report.total_bytes  # block-preserving, not widened
    # The report (including repacked int-quant types) must be JSON-serialisable
    # for the resumable cache — no stray numpy scalars from the header reader.
    import json as _json

    assert _json.loads(report.to_json())["output_bytes"] == report.output_bytes


def test_sparse_iq1_moe_is_the_remaining_blocker_not_sharding(tmp_path):
    # Flagship GLM-5.2 UD-IQ1_S composition: a *sharded*, MoE, IQ1_S set whose
    # ONLY honesty blocker must be sparse-MoE fusion — never sharding or the
    # architecture. The fixture is faithful to the three signals production
    # actually consumes, so the gate is exercised end-to-end and the positive
    # assertions below fail if either filename quant detection or the
    # sparse-MoE gate is bypassed (a fixture that put IQ1_S into no
    # production-consumed candidate — the prior bug — leaves quantization=None,
    # fusion_supported=True and no blocker, failing this test).
    shards = _write_glm_moe_iq1_sharded_gguf(tmp_path, num_experts=8)

    # -- Preconditions, proved explicitly (not assumed) ----------------------
    assert len(shards) > 1
    # The exact real GLM-5.2 shard filename is what production reads for the
    # quant tag; prove filename detection independently before building.
    first_name = shards[0].name
    assert first_name.startswith("GLM-5.2-UD-IQ1_S-00001-of-000")
    assert first_name.endswith(".gguf")
    assert _detect_quantization(first_name) == "IQ1_S"

    report = preflight_local_gguf(shards[0])

    # Sharded set + architecture resolves => neither is (or causes) a blocker.
    assert report.is_sharded is True
    assert report.split_count == len(shards)
    assert report.architecture == "glm-dsa"
    assert report.resolved_model_type == "glm_moe_dsa"
    assert "architecture metadata missing" not in " ".join(report.warnings)
    # MoE + IQ1_S quant are detected from the real metadata/filename.
    assert report.num_experts == 8
    assert report.quantization == "IQ1_S"
    # The routed experts are physically IQ1_S and losslessly preserved, so the
    # only remaining concern is fusion — not a lossy/unsupported-qtype blocker.
    iq1_stats = [s for s in report.type_stats if s.type_name == "IQ1_S"]
    assert iq1_stats, report.type_stats
    assert all(s.disposition == "native-preserve" for s in iq1_stats)
    assert report.unsupported_types == []

    # -- The gate under test: sparse-MoE fusion is the remaining blocker ------
    assert report.sparse_moe_fusion_supported is False
    assert len(report.blockers) == 1
    blocker = report.blockers[0]
    assert "sparse-MoE fusion blocker" in blocker
    assert "dense-all-expert" in blocker
    assert "BlockQuantizedMoE" in blocker
    assert "glm_moe_dsa" in blocker
    assert "IQ1_S" in blocker

    # -- Sharding is explicitly NOT the blocker ------------------------------
    joined = " ".join(report.blockers).lower()
    for forbidden in ("shard", "split", "merge", "double-copy", "second"):
        assert forbidden not in joined, report.blockers


# --------------------------------------------------------------------------- #
# CLI de-duplication: exactly one ``preflight-gguf`` owner (parser must build on
# Python 3.12) and no parallel top-level module.
# --------------------------------------------------------------------------- #


def test_single_preflight_gguf_cli_registration():
    # build_parser() raises argparse.ArgumentError at construction time if a
    # subcommand is registered twice, so a clean build proves de-duplication.
    from mobius.__main__ import build_parser

    parser = build_parser()
    subparsers = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]
    assert subparsers, "no subparsers found"
    choices = subparsers[0].choices
    assert "preflight-gguf" in choices
    # A dict of choices can only hold one entry per key; assert the whole action
    # set registers the name exactly once.
    registered = [
        name
        for action in subparsers
        for name in getattr(action, "choices", {})
        if name == "preflight-gguf"
    ]
    assert registered == ["preflight-gguf"]


def test_no_parallel_preflight_gguf_module():
    # The superseded standalone module must not be reintroduced.
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("mobius.preflight_gguf")
