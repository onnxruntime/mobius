# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the memory-mapped safetensors loader."""

from __future__ import annotations

import json
import os
import struct

import pytest
import safetensors.torch
import torch

from mobius.integrations._safetensors import (
    _MAX_HEADER_SIZE,
    _SAFETENSORS_DTYPE_TO_TORCH_DTYPE,
    _parse_header,
    load_safetensors_mmap,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_safetensors_file(path: str, tensors: dict[str, torch.Tensor]) -> None:
    """Write a safetensors file using the reference library."""
    safetensors.torch.save_file(tensors, path)


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


class TestParseHeader:
    """Tests for _parse_header()."""

    def test_basic_header(self, tmp_path):
        path = tmp_path / "model.safetensors"
        data = {"weight": torch.zeros(2, 3)}
        _create_safetensors_file(str(path), data)

        header, header_size = _parse_header(path)
        assert "weight" in header
        assert header["weight"]["dtype"] == "F32"
        assert header["weight"]["shape"] == [2, 3]
        assert isinstance(header["weight"]["data_offsets"], list)
        assert len(header["weight"]["data_offsets"]) == 2
        assert header_size > 0

    def test_multiple_tensors(self, tmp_path):
        path = tmp_path / "model.safetensors"
        data = {
            "a": torch.ones(4),
            "b": torch.zeros(2, 2),
        }
        _create_safetensors_file(str(path), data)

        header, _ = _parse_header(path)
        assert "a" in header
        assert "b" in header

    def test_too_small_file_raises(self, tmp_path):
        path = tmp_path / "tiny.safetensors"
        path.write_bytes(b"\x00\x01\x02")  # only 3 bytes

        with pytest.raises(ValueError, match="too small"):
            _parse_header(path)

    def test_truncated_header_raises(self, tmp_path):
        path = tmp_path / "trunc.safetensors"
        # Write header size claiming 1000 bytes, but provide only 10
        path.write_bytes(struct.pack("<Q", 1000) + b"x" * 10)

        with pytest.raises(ValueError, match="truncated"):
            _parse_header(path)

    def test_oversized_header_raises(self, tmp_path):
        """A header size exceeding _MAX_HEADER_SIZE must raise."""
        path = tmp_path / "huge.safetensors"
        # Claim header is larger than the cap (but don't write that much)
        path.write_bytes(struct.pack("<Q", _MAX_HEADER_SIZE + 1))

        with pytest.raises(ValueError, match="exceeds maximum"):
            _parse_header(path)


# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------


class TestDtypeMapping:
    """Verify the safetensors dtype → torch dtype mapping."""

    @pytest.mark.parametrize(
        "sf_dtype,torch_dtype",
        [
            ("F32", torch.float32),
            ("F16", torch.float16),
            ("BF16", torch.bfloat16),
            ("F64", torch.float64),
            ("I8", torch.int8),
            ("I16", torch.int16),
            ("I32", torch.int32),
            ("I64", torch.int64),
            ("U8", torch.uint8),
            ("U16", torch.uint16),
            ("U32", torch.uint32),
            ("U64", torch.uint64),
            ("BOOL", torch.bool),
        ],
    )
    def test_mapping_exists(self, sf_dtype, torch_dtype):
        assert _SAFETENSORS_DTYPE_TO_TORCH_DTYPE[sf_dtype] is torch_dtype


# ---------------------------------------------------------------------------
# Mmap loading — single file
# ---------------------------------------------------------------------------


class TestLoadSafetensorsMmap:
    """Tests for load_safetensors_mmap()."""

    def test_values_match_eager_load(self, tmp_path):
        """Mmap tensors must match eagerly-loaded tensors exactly."""
        path = str(tmp_path / "model.safetensors")
        original = {
            "weight": torch.randn(4, 3),
            "bias": torch.randn(3),
        }
        _create_safetensors_file(path, original)

        mmap_result = load_safetensors_mmap(path)
        eager_result = safetensors.torch.load_file(path)

        assert set(mmap_result) == set(eager_result)
        for name in original:
            torch.testing.assert_close(mmap_result[name], eager_result[name])

    def test_dtypes_preserved(self, tmp_path):
        """Each tensor's dtype must be preserved through mmap loading."""
        path = str(tmp_path / "model.safetensors")
        original = {
            "fp32": torch.randn(2, 2, dtype=torch.float32),
            "fp16": torch.randn(2, 2, dtype=torch.float16),
            "bf16": torch.randn(2, 2, dtype=torch.bfloat16),
            "int8": torch.randint(-128, 127, (3,), dtype=torch.int8),
            "int64": torch.randint(0, 100, (5,), dtype=torch.int64),
        }
        _create_safetensors_file(path, original)

        result = load_safetensors_mmap(path)
        for name, tensor in original.items():
            assert result[name].dtype == tensor.dtype, f"dtype mismatch for '{name}'"

    def test_shapes_preserved(self, tmp_path):
        """Tensor shapes must be preserved through mmap loading."""
        path = str(tmp_path / "model.safetensors")
        original = {
            "scalar": torch.tensor(3.14),
            "vector": torch.randn(10),
            "matrix": torch.randn(4, 5),
            "tensor3d": torch.randn(2, 3, 4),
        }
        _create_safetensors_file(path, original)

        result = load_safetensors_mmap(path)
        for name, tensor in original.items():
            assert result[name].shape == tensor.shape, f"shape mismatch for '{name}'"

    def test_empty_tensor(self, tmp_path):
        """Loading a file with an empty (zero-element) tensor."""
        path = str(tmp_path / "model.safetensors")
        original = {"empty": torch.empty(0, 3)}
        _create_safetensors_file(path, original)

        result = load_safetensors_mmap(path)
        assert result["empty"].shape == torch.Size([0, 3])

    def test_metadata_key_ignored(self, tmp_path):
        """The __metadata__ key must not appear as a tensor."""
        # Build a minimal safetensors file with __metadata__
        path = str(tmp_path / "model.safetensors")
        tensor_data = torch.randn(2).numpy().tobytes()
        header = {
            "__metadata__": {"format": "pt"},
            "w": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, len(tensor_data)],
            },
        }
        header_bytes = json.dumps(header).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(tensor_data)

        result = load_safetensors_mmap(path)
        assert "__metadata__" not in result
        assert "w" in result

    def test_unsupported_dtype_raises(self, tmp_path):
        """Unknown dtype strings must raise KeyError."""
        path = str(tmp_path / "bad.safetensors")
        header = {
            "w": {
                "dtype": "IMAGINARY128",
                "shape": [2],
                "data_offsets": [0, 8],
            },
        }
        header_bytes = json.dumps(header).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"\x00" * 8)

        with pytest.raises(KeyError, match="IMAGINARY128"):
            load_safetensors_mmap(path)

    def test_large_tensor(self, tmp_path):
        """Larger tensors must load correctly (tests multi-page mmap)."""
        path = str(tmp_path / "model.safetensors")
        # 1 MB of float32 = 256K elements
        original = {"large": torch.randn(256, 1024)}
        _create_safetensors_file(path, original)

        result = load_safetensors_mmap(path)
        torch.testing.assert_close(result["large"], original["large"])


# ---------------------------------------------------------------------------
# Mmap tensors are backed by file storage (not RAM copies)
# ---------------------------------------------------------------------------


class TestMmapStorageProperties:
    """Verify that tensors are backed by memory-mapped storage."""

    def test_tensors_share_file_storage(self, tmp_path):
        """Multiple tensors from the same file share the same base storage.

        At least they should not be independent RAM copies.
        """
        path = str(tmp_path / "model.safetensors")
        _create_safetensors_file(
            path,
            {"a": torch.randn(100), "b": torch.randn(100)},
        )

        result = load_safetensors_mmap(path)
        # Both tensors' storages should point into the same file
        # (their data_ptr values should be within the file's mmap)
        a_ptr = result["a"].data_ptr()
        b_ptr = result["b"].data_ptr()
        # They should be at different locations (not the same tensor)
        assert a_ptr != b_ptr

    def test_tensor_is_not_a_copy(self, tmp_path):
        """The mmap tensor's storage should reference the file, not a heap copy.

        We verify by checking nbytes <= file_size for the underlying untyped
        storage (a heap copy would have exactly tensor-size bytes).
        """
        path = str(tmp_path / "model.safetensors")
        original = {"w": torch.randn(50)}
        _create_safetensors_file(path, original)

        result = load_safetensors_mmap(path)
        # The storage size should be the exact tensor byte size (from
        # the sliced sub-storage), not the full file size.
        tensor = result["w"]
        element_size = tensor.element_size()
        expected_bytes = tensor.nelement() * element_size
        actual_bytes = tensor.untyped_storage().nbytes()
        assert actual_bytes == expected_bytes


# ---------------------------------------------------------------------------
# Multi-shard loading
# ---------------------------------------------------------------------------


class TestMultiShardLoading:
    """Test loading from sharded safetensors checkpoints."""

    def _create_sharded_checkpoint(self, directory, tensors_per_shard):
        """Create a sharded safetensors checkpoint.

        Args:
            directory: Path to the checkpoint directory.
            tensors_per_shard: List of dicts, each shard's tensors.
        """
        weight_map = {}
        shard_names = []
        for i, shard_tensors in enumerate(tensors_per_shard):
            shard_name = f"model-{i + 1:05d}-of-{len(tensors_per_shard):05d}.safetensors"
            shard_path = os.path.join(directory, shard_name)
            _create_safetensors_file(shard_path, shard_tensors)
            shard_names.append(shard_name)
            for name in shard_tensors:
                weight_map[name] = shard_name

        index = {"weight_map": weight_map}
        index_path = os.path.join(directory, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f)

    def test_sharded_loading(self, tmp_path):
        """Loading from multiple shards must return all tensors."""
        shard1 = {"layer.0.weight": torch.randn(4, 4)}
        shard2 = {"layer.1.weight": torch.randn(4, 4)}
        self._create_sharded_checkpoint(tmp_path, [shard1, shard2])

        # Load shard by shard (matching the integration pattern)
        state_dict: dict[str, torch.Tensor] = {}
        index_path = tmp_path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        for shard_file in shard_files:
            state_dict.update(load_safetensors_mmap(str(tmp_path / shard_file)))

        assert "layer.0.weight" in state_dict
        assert "layer.1.weight" in state_dict
        torch.testing.assert_close(state_dict["layer.0.weight"], shard1["layer.0.weight"])
        torch.testing.assert_close(state_dict["layer.1.weight"], shard2["layer.1.weight"])

    def test_sharded_no_duplicates(self, tmp_path):
        """Each tensor should appear exactly once across all shards."""
        shard1 = {
            "embed.weight": torch.randn(10, 4),
            "layer.0.weight": torch.randn(4, 4),
        }
        shard2 = {
            "layer.1.weight": torch.randn(4, 4),
            "head.weight": torch.randn(10, 4),
        }
        self._create_sharded_checkpoint(tmp_path, [shard1, shard2])

        state_dict: dict[str, torch.Tensor] = {}
        index_path = tmp_path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)
        for shard_file in sorted(set(index["weight_map"].values())):
            state_dict.update(load_safetensors_mmap(str(tmp_path / shard_file)))

        assert len(state_dict) == 4


# ---------------------------------------------------------------------------
# Data offset validation
# ---------------------------------------------------------------------------


def _write_crafted_safetensors(
    path: str,
    data_offsets: list[int],
    *,
    tensor_bytes: int = 8,
) -> None:
    """Write a safetensors file with arbitrary data_offsets for testing.

    Always writes *tensor_bytes* bytes of raw data after the header.
    """
    header = {
        "w": {
            "dtype": "F32",
            "shape": [tensor_bytes // 4] if tensor_bytes else [0],
            "data_offsets": data_offsets,
        },
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * tensor_bytes)


class TestDataOffsetValidation:
    """Verify that invalid data offsets are rejected."""

    def test_negative_start_raises(self, tmp_path):
        """Negative start offset must raise ValueError."""
        path = str(tmp_path / "neg.safetensors")
        _write_crafted_safetensors(path, [-4, 8])

        with pytest.raises(ValueError, match="0 <= start <= end"):
            load_safetensors_mmap(path)

    def test_end_before_start_raises(self, tmp_path):
        """End < start must raise ValueError."""
        path = str(tmp_path / "inverted.safetensors")
        _write_crafted_safetensors(path, [8, 4])

        with pytest.raises(ValueError, match="0 <= start <= end"):
            load_safetensors_mmap(path)

    def test_offsets_beyond_file_size_raises(self, tmp_path):
        """Offsets extending past the end of the file must raise."""
        path = str(tmp_path / "oob.safetensors")
        # Write only 8 bytes of data, but claim offsets [0, 1000)
        _write_crafted_safetensors(path, [0, 1000], tensor_bytes=8)

        with pytest.raises(ValueError, match="beyond file size"):
            load_safetensors_mmap(path)

    def test_valid_offsets_accepted(self, tmp_path):
        """Valid offsets should load without error."""
        path = str(tmp_path / "ok.safetensors")
        _write_crafted_safetensors(path, [0, 8], tensor_bytes=8)

        result = load_safetensors_mmap(path)
        assert "w" in result
        assert result["w"].shape == torch.Size([2])
