# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the Tencent custom Q1_0 parser.

We synthesize Tencent-layout blocks in-memory and verify that the
repacked weights, scales, and zero-points dequantize via the
MatMulNBits ``bits=4, zp=3`` formula to the SEQ codebook
``{-3, -1, +1, +3} * scale``.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from mobius.integrations.gguf._tencent_q1_0 import (
    TENCENT_Q1_0_NATIVE_BLOCK_SIZE,
    TENCENT_Q1_0_ORT_BLOCK_SIZE,
    parse_tencent_q1_0_tensor,
)


def _pack_2bit_codes_lsb(codes: np.ndarray) -> bytes:
    """Pack 2-bit codes ``c ∈ {0..3}`` into bytes (4 codes per byte, LSB-first)."""
    assert codes.ndim == 1
    assert codes.size % 4 == 0
    out = bytearray(codes.size // 4)
    for i, c in enumerate(codes):
        out[i // 4] |= (int(c) & 0x3) << (2 * (i % 4))
    return bytes(out)


def _make_tencent_q1_0_block(scale: float, codes: list[int]) -> bytes:
    """Build a single 130-byte Tencent Q1_0 block."""
    assert len(codes) == TENCENT_Q1_0_NATIVE_BLOCK_SIZE
    scale_bytes = struct.pack("<e", scale)  # fp16
    code_bytes = _pack_2bit_codes_lsb(np.array(codes, dtype=np.uint8))
    return scale_bytes + code_bytes


class _FakeTensor:
    """Minimal stand-in for ``gguf.ReaderTensor`` used in tests."""

    def __init__(self, name: str, shape: tuple[int, int], offset: int):
        self.name = name
        # gguf stores shape as np.uint64 elements
        self.shape = np.array(shape, dtype=np.uint64)
        self.field = _FakeField(offset)


class _FakeField:
    def __init__(self, offset: int):
        # _tensor_data_offset reads parts[data[-1]][0]; we wire that path.
        self.parts = [np.array([offset], dtype=np.uint64)]
        self.data = [0]


def _round_trip(file_path: Path, ne0: int, ne1: int, codes: np.ndarray, scales: np.ndarray):
    """Write a synthetic Tencent-Q1_0 tensor to disk and parse it back."""
    n_native = ne0 // TENCENT_Q1_0_NATIVE_BLOCK_SIZE
    with open(file_path, "wb") as f:
        for n in range(ne1):
            for b in range(n_native):
                start = b * TENCENT_Q1_0_NATIVE_BLOCK_SIZE
                end = start + TENCENT_Q1_0_NATIVE_BLOCK_SIZE
                f.write(
                    _make_tencent_q1_0_block(float(scales[n, b]), codes[n, start:end].tolist())
                )
    tensor = _FakeTensor("w", (ne0, ne1), offset=0)
    return parse_tencent_q1_0_tensor(file_path, data_section_offset=0, tensor=tensor)


class TestTencentQ10:
    def test_block_size_constants(self):
        assert TENCENT_Q1_0_NATIVE_BLOCK_SIZE == 512
        assert TENCENT_Q1_0_ORT_BLOCK_SIZE == 128
        assert TENCENT_Q1_0_NATIVE_BLOCK_SIZE % TENCENT_Q1_0_ORT_BLOCK_SIZE == 0

    def test_single_block_shape(self, tmp_path: Path):
        ne0, ne1 = 512, 1  # one native block, one output row
        codes = np.zeros((ne1, ne0), dtype=np.uint8)
        scales = np.full((ne1, 1), 0.5, dtype=np.float16)
        result = _round_trip(tmp_path / "t.bin", ne0, ne1, codes, scales)
        # bits=4, block_size=128 → blob_size = 128*4/8 = 64 bytes per sub-block
        # 512 native = 4 sub-blocks
        assert result.bits == 4
        assert result.block_size == 128
        assert result.weight.shape == (1, 4, 64)
        # Scales replicated 4× across sub-blocks
        assert result.scales.shape == (1, 4)
        np.testing.assert_array_equal(result.scales, 0.5)
        # zero_points packed 0x33 per byte. 4 sub-blocks → 2 bytes
        assert result.zero_points.shape == (1, 2)
        np.testing.assert_array_equal(result.zero_points, 0x33)

    def test_all_code_0_packs_as_zeros(self, tmp_path: Path):
        """Code 0 → 4-bit slot 0; dequant scale·(0-3) = -3·scale."""
        ne0, ne1 = 512, 1
        codes = np.zeros((ne1, ne0), dtype=np.uint8)
        scales = np.ones((ne1, 1), dtype=np.float16)
        result = _round_trip(tmp_path / "t.bin", ne0, ne1, codes, scales)
        np.testing.assert_array_equal(result.weight, 0x00)

    def test_all_code_3_packs_as_0x66(self, tmp_path: Path):
        """Code 3 → 4-bit slot 6; byte = (6<<4) | 6 = 0x66."""
        ne0, ne1 = 512, 1
        codes = np.full((ne1, ne0), 3, dtype=np.uint8)
        scales = np.ones((ne1, 1), dtype=np.float16)
        result = _round_trip(tmp_path / "t.bin", ne0, ne1, codes, scales)
        np.testing.assert_array_equal(result.weight, 0x66)

    def test_dequant_round_trip_matches_seq_codebook(self, tmp_path: Path):
        """End-to-end: dequant via MatMulNBits formula gives ±{1,3}·scale."""
        ne0, ne1 = 512, 2
        rng = np.random.default_rng(123)
        codes = rng.integers(0, 4, size=(ne1, ne0)).astype(np.uint8)
        scales = np.array(
            [[0.25], [0.0625]], dtype=np.float16
        )  # one scale per row (only 1 native block)
        result = _round_trip(tmp_path / "t.bin", ne0, ne1, codes, scales)

        # Manually dequantize MatMulNBits-style: nibble - 3, then * scale
        N, n_blocks, blob = result.weight.shape  # (2, 4, 64)
        nibbles = np.empty((N, n_blocks, blob * 2), dtype=np.uint8)
        nibbles[:, :, 0::2] = result.weight & 0xF
        nibbles[:, :, 1::2] = (result.weight >> 4) & 0xF
        sc = result.scales.astype(np.float32)
        deq = (nibbles.astype(np.float32) - 3) * sc[:, :, None]
        deq = deq.reshape(N, n_blocks * blob * 2)

        # Reference dequant: scale * {-3,-1,+1,+3}[code]
        codebook = np.array([-3, -1, 1, 3], dtype=np.float32)
        ref = scales.astype(np.float32) * codebook[codes]

        np.testing.assert_allclose(deq, ref, rtol=0, atol=1e-3)

    def test_multi_native_block_replicates_scales(self, tmp_path: Path):
        """Each native scale should appear 4× consecutively in result.scales."""
        ne0, ne1 = 1024, 3  # 2 native blocks per row
        codes = np.zeros((ne1, ne0), dtype=np.uint8)
        scales = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float16)
        result = _round_trip(tmp_path / "t.bin", ne0, ne1, codes, scales)
        # 2 native blocks × 4 sub-blocks = 8 ORT blocks
        assert result.scales.shape == (3, 8)
        # Row 0: [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
        expected = np.repeat(scales, 4, axis=1)
        np.testing.assert_array_equal(result.scales, expected)

    def test_rejects_unaligned_k(self, tmp_path: Path):
        """K not divisible by 512 raises."""
        tensor = _FakeTensor("w", (256, 1), offset=0)
        with pytest.raises(ValueError, match="not divisible"):
            parse_tencent_q1_0_tensor(tmp_path / "t.bin", 0, tensor)
