# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for direct multi-shard GGUF reading (:mod:`._shard_set`).

Fixtures are synthetic split sets built with ``gguf.GGUFWriter(split_max_tensors=N)``
— the same on-disk layout llama.cpp/Unsloth ship (``<name>-000i-of-000N.gguf``,
metadata only on the primary shard). No real weights are downloaded.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from mobius.integrations.gguf._reader import GGUFModel
from mobius.integrations.gguf._shard_set import (
    GgufShardError,
    GgufShardManifest,
    GgufShardSet,
    discover_gguf_shards,
    open_gguf_model,
    parse_shard_filename,
)


def _tensor_names(num_layers: int) -> list[str]:
    names = ["token_embd.weight"]
    for i in range(num_layers):
        names.append(f"blk.{i}.attn_q.weight")
        names.append(f"blk.{i}.ffn_up.weight")
    names.append("output_norm.weight")
    names.append("output.weight")
    return names


def _write_sharded_gguf(
    directory: Path,
    *,
    stem: str = "tiny",
    architecture: str = "llama",
    num_layers: int = 3,
    split_max_tensors: int = 3,
    rows: int = 8,
    cols: int = 16,
    name_override: str | None = None,
    seed: int = 0,
) -> list[Path]:
    """Write a synthetic split GGUF set and return its shard paths in order.

    Every tensor value is deterministic (seeded + derived from its name) so
    tests can assert the reader returns the exact bytes from the owning shard.
    """
    from gguf import GGUFWriter

    directory.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(
        str(directory / stem), architecture, split_max_tensors=split_max_tensors
    )
    writer.add_context_length(128)
    writer.add_embedding_length(cols)
    writer.add_feed_forward_length(cols * 2)
    writer.add_block_count(num_layers)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_vocab_size(32)
    if name_override is not None:
        writer.add_name(name_override)

    rng = np.random.default_rng(seed)
    for name in _tensor_names(num_layers):
        data = rng.standard_normal((rows, cols)).astype(np.float32)
        writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return sorted(directory.glob(f"{stem}-*.gguf"))


def _write_single_gguf(directory: Path, *, stem: str = "single") -> Path:
    """Write a plain (unsharded) GGUF file."""
    from gguf import GGUFWriter

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.gguf"
    writer = GGUFWriter(str(path), "llama")
    writer.add_context_length(128)
    writer.add_embedding_length(16)
    writer.add_feed_forward_length(32)
    writer.add_block_count(2)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_vocab_size(32)
    rng = np.random.default_rng(0)
    for name in _tensor_names(2):
        writer.add_tensor(name, rng.standard_normal((8, 16)).astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #


def test_parse_shard_filename_valid():
    assert parse_shard_filename("GLM-5.2-UD-IQ1_S-00001-of-00006.gguf") == (
        "GLM-5.2-UD-IQ1_S",
        1,
        6,
    )


@pytest.mark.parametrize(
    "name",
    [
        "model.gguf",
        "model-1-of-6.gguf",
        "model-00001-of-00006.bin",
        "model-00001-of-00006",
    ],
)
def test_parse_shard_filename_rejects_non_shard(name):
    assert parse_shard_filename(name) is None


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("split_max_tensors", [3, 4])
def test_happy_path_reads_all_tensors(tmp_path, split_max_tensors):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=split_max_tensors)
    assert len(shards) >= 2  # genuinely sharded

    model = open_gguf_model(shards[0])
    assert isinstance(model, GgufShardSet)
    assert model.architecture == "llama"
    assert set(model.tensor_names) == set(_tensor_names(3))
    assert model.num_tensors == len(_tensor_names(3))

    # Every tensor is readable and matches a direct single-shard read.
    single_reads = {}
    for shard in shards:
        gm = GGUFModel(shard)
        for name in gm.tensor_names:
            single_reads[name] = gm.get_tensor(name)
    for name in model.tensor_names:
        np.testing.assert_array_equal(model.get_tensor(name), single_reads[name])


def test_tensors_split_across_files(tmp_path):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=3)
    # Confirm the split actually places tensors in different shards.
    owners = {}
    for shard in shards:
        for name in GGUFModel(shard).tensor_names:
            owners[name] = shard.name
    distinct_files = set(owners.values())
    assert len(distinct_files) == len(shards)


def test_order_independence(tmp_path):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=3)
    # Opening via ANY shard (even a continuation shard) yields the same model.
    ref = open_gguf_model(shards[0])
    ref_names = sorted(ref.tensor_names)
    for shard in shards[1:]:
        other = open_gguf_model(shard)
        assert sorted(other.tensor_names) == ref_names
        assert other.manifest.split_count == ref.manifest.split_count


def test_open_directory(tmp_path):
    _write_sharded_gguf(tmp_path, split_max_tensors=3)
    model = open_gguf_model(tmp_path)  # a directory holding one split set
    assert isinstance(model, GgufShardSet)
    assert model.num_tensors == len(_tensor_names(3))


def test_discover_confined_to_directory(tmp_path):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=3)
    found = discover_gguf_shards(shards[2])
    assert sorted(p.name for p in found) == sorted(p.name for p in shards)
    # Discovery does not escape into a sibling directory.
    other = tmp_path / "other"
    _write_sharded_gguf(other, stem="zzz", split_max_tensors=3)
    found_again = discover_gguf_shards(shards[0])
    assert all(p.parent == tmp_path for p in found_again)


# --------------------------------------------------------------------------- #
# Single-file behaviour is unchanged
# --------------------------------------------------------------------------- #


def test_single_file_returns_plain_model(tmp_path):
    plain = _write_single_gguf(tmp_path)
    model = open_gguf_model(plain)
    assert isinstance(model, GGUFModel)
    assert not isinstance(model, GgufShardSet)
    assert model.architecture == "llama"


# --------------------------------------------------------------------------- #
# Fail-closed validation
# --------------------------------------------------------------------------- #


def test_missing_shard_rejected(tmp_path):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=3)
    shards[1].unlink()  # drop a middle shard
    with pytest.raises(GgufShardError, match=r"(?i)missing|count|contiguous|shard"):
        open_gguf_model(shards[0])


def test_duplicate_shard_index_rejected(tmp_path):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=3)
    # Copy shard 2's bytes over shard 3's filename => two files claim split.no==1.
    shards[2].write_bytes(shards[1].read_bytes())
    with pytest.raises(GgufShardError):
        open_gguf_model(shards[0])


def test_mixed_shard_set_rejected(tmp_path):
    # Shard 0 comes from a 3-layer export (7 tensors); a continuation shard is
    # swapped in from a structurally different 4-layer export (9 tensors). Every
    # shard records the export's total via ``split.tensors.count``, so the
    # divergence (7 vs 9) is detected — a mixed-revision / mixed-export set.
    set_a = _write_sharded_gguf(tmp_path, num_layers=3, split_max_tensors=3)
    other_dir = tmp_path / "b"
    set_b = _write_sharded_gguf(other_dir, num_layers=4, split_max_tensors=3, seed=1)
    set_a[1].write_bytes(set_b[1].read_bytes())
    with pytest.raises(GgufShardError, match=r"(?i)tensors|count|mismatch|conflict|identity"):
        open_gguf_model(set_a[0])


def test_duplicate_tensor_across_shards_rejected(tmp_path):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=3)
    # Overwrite shard 3 with a copy of shard 2 => a tensor name appears twice.
    shards[2].write_bytes(shards[1].read_bytes())
    with pytest.raises(GgufShardError):
        open_gguf_model(shards[0])


def test_corrupt_truncated_shard_rejected(tmp_path):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=3)
    # Truncate a shard mid-tensor: declared tensor extents now exceed file size.
    target = shards[1]
    data = target.read_bytes()
    target.write_bytes(data[: len(data) // 2])
    with pytest.raises((GgufShardError, ValueError)):
        open_gguf_model(shards[0])


def test_checksum_mismatch_rejected(tmp_path):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=3)
    manifest = open_gguf_model(shards[0], verify_checksums=True).manifest
    good = {s.path.name: s.sha256 for s in manifest.shards}
    bad = dict(good)
    # Flip an expected checksum => verification must fail closed.
    victim = shards[1].name
    bad[victim] = "0" * 64
    with pytest.raises(GgufShardError, match=r"(?i)sha|checksum|hash"):
        open_gguf_model(shards[0], verify_checksums=True, expected_sha256=bad)


def test_manifest_rejects_swapped_continuation_shard(tmp_path):
    """A same-model mixed-quant continuation-shard swap is caught by the manifest.

    Continuation shards carry no identity metadata, so without a manifest the
    swap passes structural validation (the documented residual gap in
    ``_IDENTITY_KEYS``). Supplying ``expected_sizes`` / ``expected_sha256`` (as
    :mod:`._preflight` produces from HF LFS metadata) fails closed — this is the
    byte-exact guard the task requires "when manifest data exists".
    """
    # A trusted set and a differently-quantized variant of the "same" model
    # whose continuation shards differ in byte length and content.
    good = _write_sharded_gguf(
        tmp_path / "good", stem="tiny", split_max_tensors=3, cols=16, seed=1
    )
    other = _write_sharded_gguf(
        tmp_path / "other", stem="tiny", split_max_tensors=3, cols=24, seed=2
    )

    manifest = open_gguf_model(good[0], verify_checksums=True).manifest
    expected_sizes = {s.path.name: s.size_bytes for s in manifest.shards}
    expected_sha256 = {s.path.name: s.sha256 for s in manifest.shards}

    # Assemble a working set: trusted primary + a continuation shard swapped in
    # from the other (mixed-quant) export.
    work = tmp_path / "work"
    work.mkdir()
    for p in good:
        (work / p.name).write_bytes(p.read_bytes())
    victim = other[1]  # continuation shard, same filename, different bytes/size
    (work / victim.name).write_bytes(victim.read_bytes())
    work_shards = sorted(work.glob("tiny-*.gguf"))

    # Documented residual: metadata alone cannot detect the swap, so this must
    # NOT raise (continuation shards carry no identity keys to compare).
    open_gguf_model(work_shards[0])

    # With the manifest it fails closed (the size guard fires before hashing).
    with pytest.raises(GgufShardError, match=r"(?i)size|sha|checksum"):
        open_gguf_model(
            work_shards[0],
            verify_checksums=True,
            expected_sha256=expected_sha256,
            expected_sizes=expected_sizes,
        )


# --------------------------------------------------------------------------- #
# Manifest / determinism
# --------------------------------------------------------------------------- #


def test_manifest_is_deterministic_and_serialisable(tmp_path):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=3)
    m1 = open_gguf_model(shards[0]).manifest
    m2 = open_gguf_model(shards[-1]).manifest  # opened via a different shard
    assert isinstance(m1, GgufShardManifest)
    assert m1.as_dict() == m2.as_dict()
    d = m1.as_dict()
    assert d["split_count"] == len(shards)
    assert d["total_tensors"] == len(_tensor_names(3))
    assert d["total_bytes"] == sum(s.stat().st_size for s in shards)
    # Shard order in the manifest follows split.no (0..n-1).
    assert [s["split_no"] for s in d["shards"]] == list(range(len(shards)))


def test_checksums_are_stable(tmp_path):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=3)
    a = open_gguf_model(shards[0], verify_checksums=True).manifest.as_dict()
    b = open_gguf_model(shards[0], verify_checksums=True).manifest.as_dict()
    assert a == b
    for shard in a["shards"]:
        assert shard["sha256"] and len(shard["sha256"]) == 64


# --------------------------------------------------------------------------- #
# Offset / alignment
# --------------------------------------------------------------------------- #


def test_offsets_within_file_and_aligned(tmp_path):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=3)
    model = open_gguf_model(shards[0])  # constructor runs offset/alignment checks
    # Independently confirm every tensor's byte-extent is inside its file.
    for shard in shards:
        gm = GGUFModel(shard)
        size = shard.stat().st_size
        for rt in gm.reader_tensors():
            assert rt.data_offset + rt.n_bytes <= size
    assert model.num_tensors == len(_tensor_names(3))


# --------------------------------------------------------------------------- #
# Bounded memory
# --------------------------------------------------------------------------- #


def _write_q8_sharded_gguf(
    directory: Path,
    *,
    stem: str = "q8",
    num_layers: int = 3,
    split_max_tensors: int = 3,
    rows: int = 512,
    cols: int = 512,
) -> list[Path]:
    """Write a split set whose tensors are native ``Q8_0`` blocks.

    Unlike F32 tensors (which ``get_tensor`` returns as a memmap-backed *view*),
    a quantized tensor is dequantized into a fresh float32 heap array, so
    ``tracemalloc`` actually observes the per-tensor payload and the
    bounded-memory assertion is meaningful. ``cols`` must be a multiple of the
    32-element ``Q8_0`` block; each block is 34 bytes (fp16 scale + 32 int8).
    """
    from gguf import GGMLQuantizationType, GGUFWriter

    assert cols % 32 == 0, "Q8_0 requires the last dim to be a multiple of 32"
    directory.mkdir(parents=True, exist_ok=True)
    block_bytes = cols // 32 * 34
    writer = GGUFWriter(str(directory / stem), "llama", split_max_tensors=split_max_tensors)
    writer.add_context_length(128)
    writer.add_embedding_length(cols)
    writer.add_block_count(num_layers)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_vocab_size(32)

    rng = np.random.default_rng(3)
    for name in _tensor_names(num_layers):
        raw = rng.integers(0, 256, size=(rows, block_bytes), dtype=np.uint8)
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return sorted(directory.glob(f"{stem}-*.gguf"))


def test_bounded_memory_random_access(tmp_path):
    # Native Q8_0 blocks so each get_tensor() dequantizes into a real float32
    # heap array (a memmap *view* would hide the payload from tracemalloc and
    # make this assertion vacuous).
    rows = cols = 512
    shards = _write_q8_sharded_gguf(tmp_path, split_max_tensors=3, rows=rows, cols=cols)
    model = open_gguf_model(shards[0])
    assert model.num_tensors == len(_tensor_names(3))
    one_tensor_bytes = rows * cols * 4  # dequantized float32 payload

    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        # Random Q8_0 bytes can encode non-finite fp16 scales; we only measure
        # memory here, so ignore the resulting numpy dequant warnings.
        with np.errstate(invalid="ignore", over="ignore"):
            for name in model.tensor_names:
                block = model.get_tensor(name)
                # Force materialisation of the dequantized payload.
                assert block.nbytes == one_tensor_bytes
                del block
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    # Peak stays within a small multiple of a single tensor — we never
    # materialise the whole (multi-tensor) checkpoint at once.
    assert peak - base < 3 * one_tensor_bytes


# --------------------------------------------------------------------------- #
# Native IQ1 block preservation
# --------------------------------------------------------------------------- #


def _write_iq1_sharded(directory: Path, *, quant: str = "iq1_s") -> list[Path]:
    """Write a split set whose ``blk.*`` weights are native IQ1 blocks."""
    from gguf import GGMLQuantizationType, GGUFWriter

    directory.mkdir(parents=True, exist_ok=True)
    qtype = GGMLQuantizationType.IQ1_S if quant == "iq1_s" else GGMLQuantizationType.IQ1_M
    block_bytes = 50 if quant == "iq1_s" else 56
    n_out, k = 8, 256  # one 256-element block per row
    writer = GGUFWriter(str(directory / "iq1"), "llama", split_max_tensors=2)
    writer.add_context_length(128)
    writer.add_embedding_length(k)
    writer.add_block_count(2)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_vocab_size(32)

    rng = np.random.default_rng(7)
    tensor_names = []
    for i in range(2):
        for proj in ("attn_q", "ffn_up"):
            name = f"blk.{i}.{proj}.weight"
            raw = rng.integers(0, 256, size=(n_out, block_bytes), dtype=np.uint8)
            writer.add_tensor(name, raw, raw_dtype=qtype)
            tensor_names.append(name)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return sorted(directory.glob("iq1-*.gguf"))


def test_iq1_native_blocks_preserved_byte_for_byte(tmp_path):
    shards = _write_iq1_sharded(tmp_path, quant="iq1_s")
    assert len(shards) >= 2
    model = open_gguf_model(shards[0])

    # Raw block bytes read through the shard set, keyed by tensor name.
    set_raw = {name: raw for name, raw, _qtype, _shape in model.tensor_items_raw()}

    for name in ("blk.0.attn_q.weight", "blk.1.ffn_up.weight"):
        owner = None
        for shard in shards:
            gm = GGUFModel(shard)
            if name in gm.tensor_names:
                owner = gm
                break
        assert owner is not None
        # Native IQ1_S type is reported unchanged (no dequant, no repack)...
        assert int(model.get_tensor_type(name)) == int(owner.get_tensor_type(name))
        # ...and the raw block bytes are identical to the owning shard's.
        owner_raw = {n: r for n, r, _q, _s in owner.tensor_items_raw()}[name]
        np.testing.assert_array_equal(set_raw[name], owner_raw)


def test_iq1_type_matches_single_file(tmp_path):
    shards = _write_iq1_sharded(tmp_path, quant="iq1_m")
    model = open_gguf_model(shards[0])
    from gguf import GGMLQuantizationType

    assert int(model.get_tensor_type("blk.0.attn_q.weight")) == int(GGMLQuantizationType.IQ1_M)


# --------------------------------------------------------------------------- #
# Identity-key conflict (defensive: mixed-quant / mixed-revision)
# --------------------------------------------------------------------------- #


def test_validate_identity_rejects_conflicting_metadata():
    from mobius.integrations.gguf import _shard_set as ss

    class _FakeInfo:
        def __init__(self, name):
            self.path = Path(name)

    class _FakeShard:
        def __init__(self, meta):
            self._meta = meta

        def get_metadata(self, key, default=None):
            return self._meta.get(key, default)

    infos = [_FakeInfo("a-00001-of-00002.gguf"), _FakeInfo("a-00002-of-00002.gguf")]
    shards = [
        _FakeShard({"general.architecture": "llama"}),
        _FakeShard({"general.architecture": "qwen2"}),
    ]
    with pytest.raises(GgufShardError, match=r"(?i)general.architecture|mixed"):
        ss._validate_identity(infos, shards)


def test_validate_identity_accepts_primary_only_metadata():
    from mobius.integrations.gguf import _shard_set as ss

    class _FakeInfo:
        def __init__(self, name):
            self.path = Path(name)

    class _FakeShard:
        def __init__(self, meta):
            self._meta = meta

        def get_metadata(self, key, default=None):
            return self._meta.get(key, default)

    # Real split sets carry identity keys only on the primary; that must pass.
    infos = [_FakeInfo("a-00001-of-00002.gguf"), _FakeInfo("a-00002-of-00002.gguf")]
    shards = [_FakeShard({"general.architecture": "glm-dsa"}), _FakeShard({})]
    ss._validate_identity(infos, shards)  # no raise
