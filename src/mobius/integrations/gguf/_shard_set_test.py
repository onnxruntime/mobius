# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for direct multi-shard GGUF reading (:mod:`._shard_set`).

Fixtures are synthetic split sets built with ``gguf.GGUFWriter(split_max_tensors=N)``
— the same on-disk layout llama.cpp/Unsloth ship (``<name>-000i-of-000N.gguf``,
metadata only on the primary shard). No real weights are downloaded.
"""

from __future__ import annotations

import hashlib
import tracemalloc
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from mobius.integrations.gguf._reader import GGUFModel
from mobius.integrations.gguf._shard_set import (
    GgufShardError,
    GgufShardManifest,
    GgufShardSet,
    _merge_metadata,
    discover_gguf_shards,
    open_gguf_model,
    parse_shard_filename,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    small_first_shard: bool = False,
) -> list[Path]:
    """Write a synthetic split GGUF set and return its shard paths in order.

    Every tensor value is deterministic (seeded + derived from its name) so
    tests can assert the reader returns the exact bytes from the owning shard.
    """
    from gguf import GGUFWriter

    directory.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(
        str(directory / stem),
        architecture,
        split_max_tensors=split_max_tensors,
        small_first_shard=small_first_shard,
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
    assert [name for name, _ in model.tensor_items()] == model.tensor_names


def test_metadata_only_first_shard_is_assembled(tmp_path):
    shards = _write_sharded_gguf(
        tmp_path,
        split_max_tensors=3,
        small_first_shard=True,
    )
    assert GGUFModel(shards[0]).num_tensors == 0

    model = open_gguf_model(shards[0])
    assert isinstance(model, GgufShardSet)
    assert model.num_tensors == len(_tensor_names(3))
    assert model.metadata["split.no"] == 0
    assert model.metadata["split.count"] == len(shards)
    assert model.metadata["split.tensors.count"] == len(_tensor_names(3))
    assert model.format_version == 3
    assert model.is_little_endian
    assert model.get_tensor_shape("token_embd.weight") == (8, 16)


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


def test_uppercase_gguf_suffix_discovers_complete_set_on_case_sensitive_filesystems(
    tmp_path,
):
    shards = _write_sharded_gguf(tmp_path, split_max_tensors=3)
    uppercase = []
    for shard in shards:
        renamed = shard.with_suffix(".GGUF")
        shard.rename(renamed)
        uppercase.append(renamed)

    model = open_gguf_model(uppercase[1])

    assert isinstance(model, GgufShardSet)
    assert model.num_tensors == len(_tensor_names(3))


def test_local_shard_count_is_bounded_before_sibling_allocation(tmp_path):
    path = tmp_path / "model-00001-of-01025.gguf"
    path.write_bytes(b"")

    with pytest.raises(GgufShardError, match=r"Invalid shard count 1025"):
        discover_gguf_shards(path)


def test_open_directory(tmp_path):
    _write_sharded_gguf(tmp_path, split_max_tensors=3)
    model = open_gguf_model(tmp_path)  # a directory holding one split set
    assert isinstance(model, GgufShardSet)
    assert model.num_tensors == len(_tensor_names(3))


def test_open_directory_rejects_split_set_mixed_with_standalone_file(tmp_path):
    _write_sharded_gguf(tmp_path, split_max_tensors=3)
    _write_single_gguf(tmp_path, stem="standalone")

    with pytest.raises(
        GgufShardError,
        match=r"both GGUF split sets and standalone files",
    ):
        open_gguf_model(tmp_path)


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


def _write_manual_split(
    directory: Path,
    *,
    split_counts: tuple[int, int] = (2, 2),
    tensor_names: tuple[str, str] = ("left", "right"),
) -> list[Path]:
    from gguf import GGUFWriter

    paths: list[Path] = []
    for split_no in range(2):
        path = directory / f"manual-{split_no + 1:05d}-of-00002.gguf"
        writer = GGUFWriter(str(path), "llama")
        writer.add_uint16("split.no", split_no)
        writer.add_uint16("split.count", split_counts[split_no])
        writer.add_uint64("split.tensors.count", 2)
        writer.add_tensor(
            tensor_names[split_no],
            np.full((2, 2), split_no + 1, dtype=np.float32),
        )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        paths.append(path)
    return paths


def test_split_count_metadata_mismatch_rejected(tmp_path):
    paths = _write_manual_split(tmp_path, split_counts=(2, 3))
    with pytest.raises(GgufShardError, match=r"split\.count"):
        open_gguf_model(paths[0])


def test_actual_duplicate_tensor_names_rejected(tmp_path):
    paths = _write_manual_split(tmp_path, tensor_names=("duplicate", "duplicate"))
    with pytest.raises(GgufShardError, match="Duplicate tensor"):
        open_gguf_model(paths[0])


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


def test_primary_shard_is_authoritative_for_semantic_metadata(tmp_path):
    infos = [
        SimpleNamespace(path=tmp_path / "primary.gguf", tensor_count=0),
        SimpleNamespace(path=tmp_path / "continuation.gguf", tensor_count=1),
    ]
    primary = SimpleNamespace(
        metadata={
            "general.architecture": "llama",
            "llama.context_length": 2048,
            "split.no": 0,
            "split.count": 2,
            "split.tensors.count": 1,
        }
    )
    continuation = SimpleNamespace(
        metadata={
            "split.no": 1,
            "split.count": 2,
            "split.tensors.count": 1,
            "llama.rope.freq_base": 500000.0,
        }
    )

    with pytest.raises(
        GgufShardError,
        match=r"continuation.*rope\.freq_base.*absent.*primary",
    ):
        _merge_metadata(infos, [primary, continuation])


def test_continuation_may_repeat_primary_semantics_but_cannot_override_them(tmp_path):
    infos = [
        SimpleNamespace(path=tmp_path / "primary.gguf", tensor_count=0),
        SimpleNamespace(path=tmp_path / "continuation.gguf", tensor_count=1),
    ]
    primary_metadata = {
        "general.architecture": "llama",
        "llama.context_length": 2048,
        "split.no": 0,
        "split.count": 2,
        "split.tensors.count": 1,
    }
    repeated = SimpleNamespace(
        metadata={
            **primary_metadata,
            "split.no": 1,
        }
    )
    merged = _merge_metadata(
        infos,
        [SimpleNamespace(metadata=primary_metadata), repeated],
    )
    assert merged["llama.context_length"] == 2048
    assert merged["split.no"] == 0

    repeated.metadata["llama.context_length"] = 4096
    with pytest.raises(GgufShardError, match=r"disagree.*context_length"):
        _merge_metadata(
            infos,
            [SimpleNamespace(metadata=primary_metadata), repeated],
        )


def test_open_validates_primary_metadata_authority_eagerly(tmp_path):
    from mobius.integrations.gguf import _shard_set

    shards = _write_sharded_gguf(
        tmp_path,
        split_max_tensors=3,
        small_first_shard=True,
    )
    with mock.patch.object(
        _shard_set,
        "_merge_metadata",
        wraps=_shard_set._merge_metadata,
    ) as merge:
        open_gguf_model(shards[0])

    merge.assert_called_once()


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
# Hugging Face resolution (metadata/download boundary only)
# --------------------------------------------------------------------------- #


def test_hub_shard_resolution_pins_and_downloads_complete_set(tmp_path):
    from mobius.integrations.gguf import _builder as builder

    shards = _write_sharded_gguf(
        tmp_path,
        split_max_tensors=3,
        small_first_shard=True,
    )
    remote_files = [f"weights/{path.name}" for path in shards]
    local_by_remote = dict(zip(remote_files, shards))
    api = mock.Mock()
    api.list_repo_files.side_effect = [remote_files, remote_files]
    api.get_paths_info.return_value = [
        SimpleNamespace(
            path=name,
            size=local_by_remote[name].stat().st_size,
            lfs=SimpleNamespace(sha256=_sha256(local_by_remote[name])),
        )
        for name in remote_files
    ]
    commit = "d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249"
    events: list[str] = []

    def disk_usage(_path):
        events.append("space")
        return SimpleNamespace(free=1 << 40)

    def download(*, repo_id, filename, revision):
        assert repo_id == "unsloth/tiny-sharded"
        assert revision == commit
        assert events == ["space"] or events[-1] == "download"
        events.append("download")
        return str(local_by_remote[filename])

    with (
        mock.patch.object(builder, "HfApi", return_value=api),
        mock.patch.object(
            builder,
            "_preflight_hf_gguf_file",
            return_value=commit,
        ) as preflight,
        mock.patch.object(builder.shutil, "disk_usage", side_effect=disk_usage),
        mock.patch.object(builder, "hf_hub_download", side_effect=download) as hub_download,
    ):
        resolved = builder._resolve_gguf_path(f"unsloth/tiny-sharded:{remote_files[1]}")

    assert resolved == str(shards[1])
    preflight.assert_called_once_with(
        "unsloth/tiny-sharded",
        remote_files[0],
        revision="main",
    )
    assert api.list_repo_files.call_args_list == [
        mock.call("unsloth/tiny-sharded", revision="main"),
        mock.call("unsloth/tiny-sharded", revision=commit),
    ]
    api.get_paths_info.assert_called_once_with(
        "unsloth/tiny-sharded",
        remote_files,
        revision=commit,
        expand=True,
    )
    assert hub_download.call_count == len(shards)
    assert resolved.expected_sizes == {path.name: path.stat().st_size for path in shards}
    assert resolved.expected_sha256 == {path.name: _sha256(path) for path in shards}
    assert open_gguf_model(resolved).num_tensors == len(_tensor_names(3))
    victim = shards[1].name
    resolved.expected_sha256[victim] = "0" * 64
    with pytest.raises(GgufShardError, match=r"SHA-256 mismatch"):
        open_gguf_model(resolved)


def test_build_rejects_same_size_corrupt_hub_shard_from_lfs_manifest(tmp_path):
    from mobius.integrations.gguf import _builder as builder

    good = _write_sharded_gguf(
        tmp_path / "good",
        stem="tiny",
        split_max_tensors=3,
        seed=1,
    )
    other = _write_sharded_gguf(
        tmp_path / "other",
        stem="tiny",
        split_max_tensors=3,
        seed=2,
    )
    assert [path.stat().st_size for path in good] == [path.stat().st_size for path in other]

    work = tmp_path / "work"
    work.mkdir()
    for path in good:
        (work / path.name).write_bytes(path.read_bytes())
    victim_index = 1
    (work / good[victim_index].name).write_bytes(other[victim_index].read_bytes())
    assert (work / good[victim_index].name).stat().st_size == good[victim_index].stat().st_size
    assert _sha256(work / good[victim_index].name) != _sha256(good[victim_index])

    remote_files = [f"weights/{path.name}" for path in good]
    work_by_remote = {name: work / path.name for name, path in zip(remote_files, good)}
    api = mock.Mock()
    api.list_repo_files.side_effect = [remote_files, remote_files]
    api.get_paths_info.return_value = [
        SimpleNamespace(
            path=name,
            size=path.stat().st_size,
            lfs=SimpleNamespace(sha256=_sha256(path)),
        )
        for name, path in zip(remote_files, good)
    ]
    commit = "c" * 40
    with (
        mock.patch.object(builder, "HfApi", return_value=api),
        mock.patch.object(
            builder,
            "_preflight_hf_gguf_file",
            return_value=commit,
        ),
        mock.patch.object(
            builder.shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=1 << 40),
        ),
        mock.patch.object(builder, "try_to_load_from_cache", return_value=None),
        mock.patch.object(
            builder,
            "hf_hub_download",
            side_effect=lambda *, repo_id, filename, revision: str(work_by_remote[filename]),
        ),
        pytest.raises(GgufShardError, match=r"SHA-256 mismatch.*corrupt shard"),
    ):
        builder.build_from_gguf(f"owner/repo:{remote_files[0]}")


def test_hub_incomplete_shard_set_rejected_before_preflight_or_download():
    from mobius.integrations.gguf import _builder as builder

    filename = "weights/model-00002-of-00003.gguf"
    api = mock.Mock()
    api.list_repo_files.return_value = [
        "weights/model-00001-of-00003.gguf",
        filename,
    ]
    with (
        mock.patch.object(builder, "HfApi", return_value=api),
        mock.patch.object(builder, "_preflight_hf_gguf_file") as preflight,
        mock.patch.object(builder, "hf_hub_download") as download,
        pytest.raises(ValueError, match=r"Incomplete.*missing indices.*00003"),
    ):
        builder._resolve_gguf_path(f"owner/repo:{filename}")

    preflight.assert_not_called()
    download.assert_not_called()


def test_hub_renamed_shard_headers_enumerate_complete_set_before_download(tmp_path):
    from mobius.integrations.gguf import _builder as builder

    source_shards = _write_sharded_gguf(
        tmp_path / "source",
        split_max_tensors=3,
        small_first_shard=True,
    )
    remote_files = [
        f"weights/{chr(ord('a') + index)}.gguf" for index in range(len(source_shards))
    ]
    mmproj_name = "weights/mmproj.gguf"
    local_dir = tmp_path / "downloaded"
    local_dir.mkdir()
    local_by_remote: dict[str, Path] = {}
    for remote_name, source in zip(remote_files, source_shards):
        target = local_dir / PurePosixPath(remote_name).name
        target.write_bytes(source.read_bytes())
        local_by_remote[remote_name] = target

    commit = "a" * 40
    api = mock.Mock()
    api.list_repo_files.return_value = [*remote_files, mmproj_name]
    api.get_paths_info.return_value = [
        SimpleNamespace(
            path=name,
            size=path.stat().st_size,
            lfs=SimpleNamespace(sha256=_sha256(path)),
        )
        for name, path in local_by_remote.items()
    ]

    def preflight(_repo_id, filename, *, revision, dispatch_architecture=True):
        if filename == mmproj_name:
            assert not dispatch_architecture
            return builder._GGUFPreflightRevision(
                commit,
                builder.GGUFHeaderInfo(
                    architecture="clip",
                    tensor_count=1,
                    split_no=None,
                    split_count=None,
                    split_tensors_count=None,
                ),
            )
        path = local_by_remote[filename]
        info = builder._gguf_header_info_from_header_prefix(
            path.read_bytes(),
            source=filename,
        )
        return builder._GGUFPreflightRevision(commit, info)

    with (
        mock.patch.object(builder, "HfApi", return_value=api),
        mock.patch.object(builder, "_preflight_hf_gguf_file", side_effect=preflight),
        mock.patch.object(
            builder.shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=1 << 40),
        ),
        mock.patch.object(builder, "try_to_load_from_cache", return_value=None),
        mock.patch.object(
            builder,
            "hf_hub_download",
            side_effect=lambda *, repo_id, filename, revision: str(local_by_remote[filename]),
        ) as download,
    ):
        resolved = builder._resolve_gguf_path(f"owner/renamed:{remote_files[1]}")

    assert download.call_count == len(source_shards)
    assert resolved == str(local_by_remote[remote_files[1]])
    assert resolved.shard_paths == [str(local_by_remote[name]) for name in remote_files]
    model = open_gguf_model(resolved)
    assert model.num_tensors == len(_tensor_names(3))


def test_hub_renamed_discovery_caps_header_candidates_before_probing():
    from mobius.integrations.gguf import _builder as builder

    selected = builder._GGUFPreflightRevision(
        "a" * 40,
        builder.GGUFHeaderInfo(
            architecture=None,
            tensor_count=1,
            split_no=1,
            split_count=2,
            split_tensors_count=2,
        ),
    )
    candidates = [f"weights/candidate-{index}.gguf" for index in range(9)]
    candidates[0] = "weights/selected.gguf"
    with (
        mock.patch.object(builder, "_preflight_hf_gguf_file") as preflight,
        pytest.raises(ValueError, match=r"bounded limit 8.*No additional headers"),
    ):
        builder._select_hf_gguf_set_from_split_headers(
            candidates,
            repo_id="owner/repo",
            selected_filename="weights/selected.gguf",
            revision="a" * 40,
            selected_preflight=selected,
        )

    preflight.assert_not_called()


def test_hub_split_count_is_bounded_before_candidate_allocation():
    from mobius.integrations.gguf import _builder as builder

    info = builder.GGUFHeaderInfo(
        architecture="llama",
        tensor_count=0,
        split_no=0,
        split_count=1025,
        split_tensors_count=1,
    )
    with pytest.raises(ValueError, match=r"count=1025, maximum=1024"):
        builder._validate_preflight_split_header(info, source="oversized.gguf")


def test_hub_renamed_incomplete_set_rejected_before_download(tmp_path):
    from mobius.integrations.gguf import _builder as builder

    source_shards = _write_sharded_gguf(
        tmp_path / "source",
        split_max_tensors=3,
        small_first_shard=True,
    )
    complete_remote_files = [
        f"weights/{chr(ord('a') + index)}.gguf" for index in range(len(source_shards))
    ]
    remote_files = complete_remote_files[:-1]
    local_by_remote = dict(zip(remote_files, source_shards[:-1]))
    commit = "b" * 40
    api = mock.Mock()
    api.list_repo_files.return_value = remote_files

    def preflight(_repo_id, filename, *, revision, dispatch_architecture=True):
        path = local_by_remote[filename]
        info = builder._gguf_header_info_from_header_prefix(
            path.read_bytes(),
            source=filename,
        )
        return builder._GGUFPreflightRevision(commit, info)

    with (
        mock.patch.object(builder, "HfApi", return_value=api),
        mock.patch.object(builder, "_preflight_hf_gguf_file", side_effect=preflight),
        mock.patch.object(builder, "hf_hub_download") as download,
        pytest.raises(ValueError, match=r"Incomplete.*split\.no"),
    ):
        builder._resolve_gguf_path(f"owner/renamed:{remote_files[1]}")

    api.get_paths_info.assert_not_called()
    download.assert_not_called()


def test_hub_renamed_header_fallback_rejects_potential_partial_download():
    from mobius.integrations.gguf import _builder as builder

    filename = "weights/selected.gguf"
    api = mock.Mock()
    api.list_repo_files.return_value = [filename, "weights/possible-sibling.gguf"]
    with (
        mock.patch.object(builder, "HfApi", return_value=api),
        mock.patch.object(
            builder,
            "_preflight_hf_gguf_file",
            return_value=builder._GGUFPreflightFallbackRevision("a" * 40),
        ),
        mock.patch.object(builder, "hf_hub_download") as download,
        pytest.raises(ValueError, match=r"potentially partial download"),
    ):
        builder._resolve_gguf_path(f"owner/repo:{filename}")

    download.assert_not_called()


def test_hub_free_space_preflight_is_actionable(tmp_path):
    from mobius.integrations.gguf import _builder as builder

    with (
        mock.patch.object(
            builder.shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=100),
        ),
        pytest.raises(OSError, match=r"requires 1,000 bytes.*only 100 bytes.*HF_HOME"),
    ):
        builder._preflight_hf_download_space(1000, cache_path=tmp_path)


def test_hub_download_space_counts_only_uncached_shards(tmp_path):
    from mobius.integrations.gguf import _builder as builder

    cached = tmp_path / "cached.gguf"
    missing = tmp_path / "missing.gguf"
    cached.write_bytes(b"cached")
    missing.write_bytes(b"new")
    names = ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf"]
    api = mock.Mock()
    api.get_paths_info.return_value = [
        SimpleNamespace(
            path=names[0],
            size=cached.stat().st_size,
            lfs=SimpleNamespace(sha256=_sha256(cached)),
        ),
        SimpleNamespace(
            path=names[1],
            size=missing.stat().st_size,
            lfs=SimpleNamespace(sha256=_sha256(missing)),
        ),
    ]
    with (
        mock.patch.object(
            builder,
            "try_to_load_from_cache",
            side_effect=[str(cached), None],
        ),
        mock.patch.object(builder, "_preflight_hf_download_space") as preflight,
        mock.patch.object(
            builder,
            "hf_hub_download",
            side_effect=[str(cached), str(missing)],
        ),
    ):
        resolved = builder._download_hf_gguf_shards(
            api,
            repo_id="owner/repo",
            selected_filename=names[0],
            shard_filenames=names,
            revision="a" * 40,
        )

    assert resolved == str(cached)
    preflight.assert_called_once()
    assert preflight.call_args.args == (missing.stat().st_size,)


def test_hub_cache_identity_paths_resolve_only_inside_cache(tmp_path):
    from mobius.integrations.gguf import _builder as builder

    cache = tmp_path / "hub"
    blobs = cache / "blobs"
    snapshot = cache / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / "abc"
    blob.write_bytes(b"GGUF")
    shard = snapshot / "model-00001-of-00002.gguf"
    shard.symlink_to(blob)

    with mock.patch("huggingface_hub.constants.HF_HUB_CACHE", str(cache)):
        assert builder._hub_cache_identity_paths([shard]) == [blob]
        assert builder._hub_cache_identity_paths([tmp_path / "outside.gguf"]) is None


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
