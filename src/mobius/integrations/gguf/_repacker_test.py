# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import pytest
from gguf import quants

from mobius.integrations.gguf._repacker import (
    RepackedTensor,
    _unpack_q4_k_scales,
    can_repack,
    native_block_spec,
    preserve_native_blocks,
    repack_dequantized_tensor,
    repack_gguf_tensor,
)

_Q4_0 = 2
_Q4_1 = 3
_Q8_0 = 8
_Q4_K = 12
_Q1_0 = 41
_BLOCK_SIZE = 32


@pytest.mark.parametrize(
    ("qtype_name", "qtype_value", "format_name", "block_elements", "block_bytes"),
    [
        ("MXFP4", 39, "mxfp4", 32, 17),
        ("IQ4_NL", 20, "iq4_nl", 32, 18),
        ("IQ4_XS", 23, "iq4_xs", 256, 136),
        ("IQ3_S", 21, "iq3_s", 256, 110),
        ("IQ3_XXS", 18, "iq3_xxs", 256, 98),
        ("IQ2_XXS", 16, "iq2_xxs", 256, 66),
        ("IQ2_XS", 17, "iq2_xs", 256, 74),
        ("IQ2_S", 22, "iq2_s", 256, 82),
        ("IQ1_S", 19, "iq1_s", 256, 50),
        ("IQ1_M", 29, "iq1_m", 256, 56),
    ],
)
def test_native_block_specs_match_gguf_and_preserve_bytes(
    qtype_name: str,
    qtype_value: int,
    format_name: str,
    block_elements: int,
    block_bytes: int,
):
    from gguf import GGMLQuantizationType

    qtype = getattr(GGMLQuantizationType, qtype_name)
    assert qtype.value == qtype_value
    spec = native_block_spec(qtype.value)
    assert spec is not None
    assert (spec.format, spec.elements, spec.bytes) == (
        format_name,
        block_elements,
        block_bytes,
    )

    raw = np.arange(2 * block_bytes, dtype=np.uint8)
    packed = preserve_native_blocks(raw, qtype.value, (2, block_elements))
    assert packed.shape == (2, 1, block_bytes)
    np.testing.assert_array_equal(packed.reshape(-1), raw)


def test_native_block_size_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Native iq1_m data size mismatch"):
        preserve_native_blocks(np.zeros(55, dtype=np.uint8), 29, (1, 256))


def _make_q1_0_block(scale: float, bits: list[int]) -> np.ndarray:
    """Build a single Q1_0 block (18 bytes) from scale + 128 binary signs.

    GGUF packing per llama.cpp ``quantize_row_q1_0_ref``:
        bit ``j % 8`` of ``qs[j // 8]`` holds element j (LSB-first).
    Dequant: ``bit ? +scale : -scale``.
    """
    assert len(bits) == 128
    scale_bytes = np.array([scale], dtype=np.float16).view(np.uint8)
    packed = np.zeros(16, dtype=np.uint8)
    for j in range(128):
        if bits[j]:
            packed[j // 8] |= 1 << (j % 8)
    return np.concatenate([scale_bytes, packed])


def _make_q4_0_block(scale: float, nibbles: list[int]) -> np.ndarray:
    """Build a single Q4_0 block (18 bytes) from scale + 32 element values.

    GGUF packing: byte i = (element[i+16] << 4) | element[i]
    """
    assert len(nibbles) == 32
    scale_bytes = np.array([scale], dtype=np.float16).view(np.uint8)
    packed = np.zeros(16, dtype=np.uint8)
    for i in range(16):
        packed[i] = (nibbles[i + 16] << 4) | nibbles[i]
    return np.concatenate([scale_bytes, packed])


def _make_q4_1_block(scale: float, minimum: float, nibbles: list[int]) -> np.ndarray:
    """Build a single Q4_1 block (20 bytes).

    GGUF packing: byte i = (element[i+16] << 4) | element[i]
    """
    assert len(nibbles) == 32
    scale_bytes = np.array([scale], dtype=np.float16).view(np.uint8)
    min_bytes = np.array([minimum], dtype=np.float16).view(np.uint8)
    packed = np.zeros(16, dtype=np.uint8)
    for i in range(16):
        packed[i] = (nibbles[i + 16] << 4) | nibbles[i]
    return np.concatenate([scale_bytes, min_bytes, packed])


def _make_q8_0_block(scale: float, values: list[int]) -> np.ndarray:
    """Build a single Q8_0 block (34 bytes) from scale + 32 int8 values."""
    assert len(values) == 32
    scale_bytes = np.array([scale], dtype=np.float16).view(np.uint8)
    val_bytes = np.array(values, dtype=np.int8).view(np.uint8)
    return np.concatenate([scale_bytes, val_bytes])


class TestCanRepack:
    def test_supported_types(self):
        assert can_repack(_Q4_0) is True
        assert can_repack(_Q4_1) is True
        assert can_repack(_Q8_0) is True
        assert can_repack(_Q4_K) is True

    def test_unsupported_types(self):
        assert can_repack(0) is False  # F32
        assert can_repack(1) is False  # F16
        assert can_repack(6) is False  # Q5_0
        assert can_repack(99) is False


class TestRepackQ40:
    def test_single_block(self):
        """Repack a single Q4_0 block with known values."""
        nibbles = list(range(16)) + list(range(16))  # 0..15, 0..15
        block = _make_q4_0_block(scale=0.5, nibbles=nibbles)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_0, shape=(1, 32))

        assert isinstance(result, RepackedTensor)
        assert result.bits == 4
        assert result.block_size == 32
        assert result.weight.shape == (1, 1, 16)
        assert result.scales.shape == (1, 1)
        assert result.zero_points.shape == (1, 1)
        # Scale should match
        np.testing.assert_almost_equal(result.scales[0, 0], np.float16(0.5))
        # Zero point for Q4_0 is 8 (packed: 0x08 for single block)
        assert result.zero_points[0, 0] == 0x08

    def test_two_blocks_per_row(self):
        """Two blocks per row — zero points packed into one byte."""
        nibbles = [8] * 32
        block = _make_q4_0_block(scale=1.0, nibbles=nibbles)
        # 1 row, 2 blocks -> shape (1, 64)
        raw = np.concatenate([block, block])

        result = repack_gguf_tensor(raw, _Q4_0, shape=(1, 64))

        assert result.weight.shape == (1, 2, 16)
        assert result.scales.shape == (1, 2)
        # 2 blocks -> 1 ZP byte, both nibbles = 8 -> 0x88
        assert result.zero_points.shape == (1, 1)
        assert result.zero_points[0, 0] == 0x88

    def test_multiple_rows(self):
        """Multiple output features (N=3, K=32)."""
        nibbles = [5] * 32
        block = _make_q4_0_block(scale=2.0, nibbles=nibbles)
        raw = np.tile(block, 3)  # 3 rows x 1 block

        result = repack_gguf_tensor(raw, _Q4_0, shape=(3, 32))

        assert result.weight.shape == (3, 1, 16)
        assert result.scales.shape == (3, 1)
        assert result.zero_points.shape == (3, 1)

    def test_nibble_ordering_reordered(self):
        """Verify GGUF->ORT nibble reordering.

        GGUF: byte i has element[i] (low) and element[i+16] (high).
        ORT:  byte j has element[2j] (low) and element[2j+1] (high).

        Use elements 0-15 = [0]*16, elements 16-31 = [15]*16 to make
        the reordering visible.
        """
        nibbles = [0] * 16 + [15] * 16
        block = _make_q4_0_block(scale=1.0, nibbles=nibbles)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_0, shape=(1, 32))

        # ORT bytes 0-7: pairs from elements 0..15 (all 0)
        # byte j = (element[2j+1] << 4) | element[2j] = (0<<4)|0 = 0x00
        for j in range(8):
            assert result.weight[0, 0, j] == 0x00
        # ORT bytes 8-15: pairs from elements 16..31 (all 15)
        # byte j = (15 << 4) | 15 = 0xFF
        for j in range(8, 16):
            assert result.weight[0, 0, j] == 0xFF

    def test_round_trip_dequantize(self):
        """Verify repacked data dequantizes to same values as GGUF."""
        from gguf import quants

        scale = np.float16(0.25)
        nibbles = [
            3,
            7,
            0,
            15,
            8,
            10,
            1,
            14,
            5,
            9,
            2,
            12,
            6,
            11,
            4,
            13,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
        ]
        block = _make_q4_0_block(scale=float(scale), nibbles=nibbles)

        # GGUF dequantize
        gguf_deq = quants.dequantize(block.reshape(1, -1), quants.GGMLQuantizationType.Q4_0)

        # Repack and manually dequantize via MatMulNBits formula
        raw = block.reshape(-1)
        result = repack_gguf_tensor(raw, _Q4_0, shape=(1, 32))

        # Unpack nibbles from weight
        packed = result.weight[0, 0]  # (16,)
        low = (packed & 0x0F).astype(np.float32)
        high = ((packed >> 4) & 0x0F).astype(np.float32)
        elements = np.empty(32, dtype=np.float32)
        elements[0::2] = low
        elements[1::2] = high
        # MatMulNBits dequant: (element - zp) * scale
        zp = 8.0
        s = result.scales[0, 0].astype(np.float32)
        ort_deq = (elements - zp) * s

        np.testing.assert_allclose(ort_deq, gguf_deq.ravel(), atol=1e-3)


class TestRepackQ41:
    def test_single_block(self):
        """Repack a Q4_1 block with known scale and min."""
        nibbles = [0] * 32
        block = _make_q4_1_block(scale=0.5, minimum=-2.0, nibbles=nibbles)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_1, shape=(1, 32))

        assert result.bits == 4
        assert result.weight.shape == (1, 1, 16)
        assert result.scales.shape == (1, 1)
        np.testing.assert_almost_equal(result.scales[0, 0], np.float16(0.5))
        # zp = round(-min / scale) = round(2.0 / 0.5) = 4
        zp_low = result.zero_points[0, 0] & 0x0F
        assert zp_low == 4

    def test_zero_scale_gives_zero_zp(self):
        """When scale=0, zero_point should be 0 (no division by zero)."""
        nibbles = [7] * 32
        block = _make_q4_1_block(scale=0.0, minimum=-1.0, nibbles=nibbles)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_1, shape=(1, 32))

        zp = result.zero_points[0, 0] & 0x0F
        assert zp == 0

    def test_zp_clamped_to_15(self):
        """Zero point is clamped to [0, 15] for 4-bit."""
        # min = -100, scale = 1 -> zp = 100 -> clamp to 15
        nibbles = [0] * 32
        block = _make_q4_1_block(scale=1.0, minimum=-100.0, nibbles=nibbles)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_1, shape=(1, 32))

        zp = result.zero_points[0, 0] & 0x0F
        assert zp == 15

    def test_round_trip_dequantize(self):
        """Verify Q4_1 repacked values match GGUF dequantization."""
        from gguf import quants

        scale = np.float16(0.5)
        minimum = np.float16(-1.0)
        nibbles = [0, 5, 10, 15] * 8
        block = _make_q4_1_block(scale=float(scale), minimum=float(minimum), nibbles=nibbles)

        gguf_deq = quants.dequantize(
            block.reshape(1, -1),
            quants.GGMLQuantizationType.Q4_1,
        )

        raw = block.reshape(-1)
        result = repack_gguf_tensor(raw, _Q4_1, shape=(1, 32))

        # Unpack and dequantize via MatMulNBits formula
        packed = result.weight[0, 0]
        low = (packed & 0x0F).astype(np.float32)
        high = ((packed >> 4) & 0x0F).astype(np.float32)
        elements = np.empty(32, dtype=np.float32)
        elements[0::2] = low
        elements[1::2] = high

        zp = float(result.zero_points[0, 0] & 0x0F)
        s = result.scales[0, 0].astype(np.float32)
        ort_deq = (elements - zp) * s

        # Q4_1 -> MatMulNBits is lossy (zero_point quantization)
        # Allow larger tolerance
        np.testing.assert_allclose(ort_deq, gguf_deq.ravel(), atol=0.5)


class TestRepackQ80:
    def test_single_block(self):
        """Repack a Q8_0 block with known values."""
        values = list(range(-16, 16))  # 32 int8 values
        block = _make_q8_0_block(scale=0.1, values=values)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q8_0, shape=(1, 32))

        assert result.bits == 8
        assert result.block_size == 32
        assert result.weight.shape == (1, 1, 32)
        assert result.scales.shape == (1, 1)
        assert result.zero_points.shape == (1, 1)
        # Zero point for symmetric Q8_0 is 128
        assert result.zero_points[0, 0] == 128

    def test_int8_to_uint8_conversion(self):
        """Verify signed int8 -> unsigned uint8 + 128 offset."""
        values = [-128, -1, 0, 1, 127] + [0] * 27
        block = _make_q8_0_block(scale=1.0, values=values)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q8_0, shape=(1, 32))

        # -128 + 128 = 0, -1 + 128 = 127, 0 + 128 = 128,
        # 1 + 128 = 129, 127 + 128 = 255
        assert result.weight[0, 0, 0] == 0
        assert result.weight[0, 0, 1] == 127
        assert result.weight[0, 0, 2] == 128
        assert result.weight[0, 0, 3] == 129
        assert result.weight[0, 0, 4] == 255

    def test_round_trip_dequantize(self):
        """Verify Q8_0 repacked values match GGUF dequantization."""
        from gguf import quants

        values = [int(x) for x in np.random.randint(-128, 128, size=32)]
        scale = 0.05
        block = _make_q8_0_block(scale=scale, values=values)

        gguf_deq = quants.dequantize(
            block.reshape(1, -1),
            quants.GGMLQuantizationType.Q8_0,
        )

        raw = block.reshape(-1)
        result = repack_gguf_tensor(raw, _Q8_0, shape=(1, 32))

        # MatMulNBits dequant: (uint8 - 128) * scale
        elements = result.weight[0, 0].astype(np.float32)
        s = result.scales[0, 0].astype(np.float32)
        ort_deq = (elements - 128.0) * s

        np.testing.assert_allclose(ort_deq, gguf_deq.ravel(), atol=1e-3)


class TestEdgeCases:
    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported GGUF type"):
            repack_gguf_tensor(np.zeros(10, dtype=np.uint8), 99, (1, 32))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="Expected 2D shape"):
            repack_gguf_tensor(np.zeros(18, dtype=np.uint8), _Q4_0, (32,))

    def test_data_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="Data size mismatch"):
            repack_gguf_tensor(np.zeros(10, dtype=np.uint8), _Q4_0, (1, 32))

    def test_multi_block_multi_row(self):
        """N=4, K=128 -> 4 blocks per row."""
        n, k = 4, 128
        n_blocks = k // 32
        block_bytes = 18
        _total = n * n_blocks * block_bytes
        # Create uniform blocks with scale=1.0, all nibbles=8
        scale_bytes = np.array([1.0], dtype=np.float16).view(np.uint8)
        quant_bytes = np.full(16, 0x88, dtype=np.uint8)  # nibbles all 8
        single_block = np.concatenate([scale_bytes, quant_bytes])
        raw = np.tile(single_block, n * n_blocks)

        result = repack_gguf_tensor(raw, _Q4_0, shape=(n, k))

        assert result.weight.shape == (4, 4, 16)
        assert result.scales.shape == (4, 4)
        # 4 blocks -> 2 ZP bytes per row
        assert result.zero_points.shape == (4, 2)
        assert result.zero_points[0, 0] == 0x88
        assert result.zero_points[0, 1] == 0x88


# ---- Q4_K helpers ----


def _pack_6bit_scales(sub_scales: list[int], sub_mins: list[int]) -> np.ndarray:
    """Pack 8 sub_scales + 8 sub_mins into 12 bytes (inverse of unpack).

    This is the encoding side of the 6-bit packing used by Q4_K.
    """
    assert len(sub_scales) == 8 and len(sub_mins) == 8
    out = np.zeros(12, dtype=np.uint8)
    # Bytes 0-3 (d): low 6 bits of sc[0..3], bits 6-7 from sc[4..7]
    for i in range(4):
        out[i] = (sub_scales[i] & 0x3F) | ((sub_scales[i + 4] & 0x30) << 2)
    # Bytes 4-7 (m): low 6 bits of min[0..3], bits 6-7 from min[4..7]
    for i in range(4):
        out[4 + i] = (sub_mins[i] & 0x3F) | ((sub_mins[i + 4] & 0x30) << 2)
    # Bytes 8-11 (md): low 4 bits of sc[4..7], high 4 bits of min[4..7]
    for i in range(4):
        out[8 + i] = (sub_scales[i + 4] & 0x0F) | ((sub_mins[i + 4] & 0x0F) << 4)
    return out


def _pack_q4_k_quants(nibbles: list[int]) -> np.ndarray:
    """Pack 256 nibbles into 128 bytes in Q4_K format.

    Within each 32-byte group, byte[j] = (odd_sub_block[j] << 4) |
    even_sub_block[j].
    """
    assert len(nibbles) == 256
    nibs = np.array(nibbles, dtype=np.uint8).reshape(8, 32)
    packed = np.zeros(128, dtype=np.uint8)
    for g in range(4):
        even = nibs[2 * g]
        odd = nibs[2 * g + 1]
        packed[g * 32 : (g + 1) * 32] = (odd << 4) | even
    return packed


def _make_q4_k_block(
    d: float,
    dmin: float,
    sub_scales: list[int],
    sub_mins: list[int],
    nibbles: list[int],
) -> np.ndarray:
    """Build a single Q4_K super-block (144 bytes)."""
    d_bytes = np.array([d], dtype=np.float16).view(np.uint8)
    dmin_bytes = np.array([dmin], dtype=np.float16).view(np.uint8)
    scales_bytes = _pack_6bit_scales(sub_scales, sub_mins)
    qs_bytes = _pack_q4_k_quants(nibbles)
    return np.concatenate([d_bytes, dmin_bytes, scales_bytes, qs_bytes])


def _dequantize_repacked_q4(result: RepackedTensor, k_in: int) -> np.ndarray:
    """Dequantize a 4-bit RepackedTensor with MatMulNBits semantics."""
    n_out, n_blocks, _ = result.weight.shape
    packed = result.weight
    quants = np.empty((n_out, n_blocks, 32), dtype=np.float32)
    quants[:, :, 0::2] = packed & 0x0F
    quants[:, :, 1::2] = (packed >> 4) & 0x0F

    if result.zero_points is None:
        zero_points = np.full((n_out, n_blocks), 8.0, dtype=np.float32)
    else:
        zero_points = np.empty((n_out, n_blocks), dtype=np.float32)
        zero_points[:, 0::2] = result.zero_points & 0x0F
        zero_points[:, 1::2] = (result.zero_points >> 4)[:, : n_blocks // 2]

    values = (quants - zero_points[:, :, None]) * result.scales[:, :, None]
    return values.reshape(n_out, -1)[:, :k_in]


class TestUnpackQ4KScales:
    def test_simple_values(self):
        """Pack known 6-bit values and verify round-trip."""
        sc = [1, 2, 3, 4, 5, 6, 7, 8]
        mn = [10, 20, 30, 40, 50, 60, 11, 22]
        packed = _pack_6bit_scales(sc, mn)
        got_sc, got_mn = _unpack_q4_k_scales(packed.reshape(1, 12))
        np.testing.assert_array_equal(got_sc.ravel(), sc)
        np.testing.assert_array_equal(got_mn.ravel(), mn)

    def test_max_6bit_values(self):
        """All values at maximum (63)."""
        sc = [63] * 8
        mn = [63] * 8
        packed = _pack_6bit_scales(sc, mn)
        got_sc, got_mn = _unpack_q4_k_scales(packed.reshape(1, 12))
        np.testing.assert_array_equal(got_sc.ravel(), sc)
        np.testing.assert_array_equal(got_mn.ravel(), mn)

    def test_zero_values(self):
        """All zeros."""
        sc = [0] * 8
        mn = [0] * 8
        packed = _pack_6bit_scales(sc, mn)
        got_sc, got_mn = _unpack_q4_k_scales(packed.reshape(1, 12))
        np.testing.assert_array_equal(got_sc.ravel(), sc)
        np.testing.assert_array_equal(got_mn.ravel(), mn)

    def test_batch(self):
        """Multiple super-blocks at once."""
        sc1 = [10, 20, 30, 40, 50, 60, 15, 25]
        mn1 = [5, 15, 25, 35, 45, 55, 12, 22]
        sc2 = [1, 1, 1, 1, 1, 1, 1, 1]
        mn2 = [2, 2, 2, 2, 2, 2, 2, 2]
        packed = np.stack([_pack_6bit_scales(sc1, mn1), _pack_6bit_scales(sc2, mn2)])
        got_sc, got_mn = _unpack_q4_k_scales(packed)
        np.testing.assert_array_equal(got_sc[0], sc1)
        np.testing.assert_array_equal(got_mn[0], mn1)
        np.testing.assert_array_equal(got_sc[1], sc2)
        np.testing.assert_array_equal(got_mn[1], mn2)


class TestRepackQ4K:
    def test_single_super_block_shapes(self):
        """Single Q4_K super-block -> 8 MatMulNBits sub-blocks."""
        sc = [10] * 8
        mn = [5] * 8
        nibs = [7] * 256
        block = _make_q4_k_block(d=1.0, dmin=0.5, sub_scales=sc, sub_mins=mn, nibbles=nibs)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_K, shape=(1, 256))

        assert isinstance(result, RepackedTensor)
        assert result.bits == 4
        assert result.block_size == 32
        # 1 super-block -> 8 sub-blocks
        assert result.weight.shape == (1, 8, 16)
        assert result.scales.shape == (1, 8)
        assert result.zero_points.shape == (1, 4)  # 8 zps / 2

    def test_requantized_values_stay_within_half_scale(self):
        """Affine requantization bounds each value by half a block scale."""
        sc = [10, 20, 30, 40, 1, 2, 3, 4]
        mn = [0] * 8  # zero mins -> zp=0, no offset
        nibs = list(range(16)) * 16
        d_val = 0.5
        block = _make_q4_k_block(d=d_val, dmin=0.0, sub_scales=sc, sub_mins=mn, nibbles=nibs)
        gguf_deq = quants.dequantize(
            block.reshape(1, -1), quants.GGMLQuantizationType.Q4_K
        ).reshape(1, 256)
        result = repack_gguf_tensor(block, _Q4_K, shape=(1, 256))
        ort_deq = _dequantize_repacked_q4(result, 256)

        error = np.abs(ort_deq - gguf_deq).reshape(1, 8, 32)
        assert np.all(error.max(axis=-1) <= result.scales * 0.51 + 1e-6)

    def test_constant_negative_blocks_use_valid_zero_points(self):
        """A large Q4_K offset is requantized instead of clamping its source zp."""
        block = _make_q4_k_block(
            d=1.0,
            dmin=1.0,
            sub_scales=[0] * 8,
            sub_mins=[50] * 8,
            nibbles=[8] * 256,
        )
        result = repack_gguf_tensor(block, _Q4_K, shape=(1, 256))
        ort_deq = _dequantize_repacked_q4(result, 256)

        assert np.all(result.zero_points == 0xFF)
        np.testing.assert_allclose(ort_deq, -50.0, atol=float(result.scales.max()) * 0.51)

    def test_round_trip_dequantize(self):
        """Compare repacked MatMulNBits dequant against gguf native."""
        from gguf import quants

        d_val = np.float16(0.01)
        dmin_val = np.float16(0.005)
        sc = [10, 20, 30, 40, 15, 25, 35, 45]
        mn = [5, 10, 15, 20, 8, 12, 18, 22]
        rng = np.random.RandomState(42)
        nibs = rng.randint(0, 16, size=256).tolist()

        block = _make_q4_k_block(
            d=float(d_val),
            dmin=float(dmin_val),
            sub_scales=sc,
            sub_mins=mn,
            nibbles=nibs,
        )

        # GGUF native dequantization (reference)
        gguf_deq = quants.dequantize(
            block.reshape(1, -1),
            quants.GGMLQuantizationType.Q4_K,
        ).ravel()  # (256,)

        # Repack to MatMulNBits
        raw = block.reshape(-1)
        result = repack_gguf_tensor(raw, _Q4_K, shape=(1, 256))

        ort_deq = _dequantize_repacked_q4(result, 256).ravel()
        max_abs_diff = float(np.max(np.abs(ort_deq - gguf_deq)))
        tolerance = float(result.scales.max()) * 0.51 + 1e-6
        assert max_abs_diff <= tolerance

    def test_super_blocks_may_cross_logical_rows(self):
        """Flattened Q4_K blocks are reshaped only after reference dequant."""
        rng = np.random.RandomState(7)
        source_blocks = []
        for i in range(3):
            source_blocks.append(
                _make_q4_k_block(
                    d=0.01 * (i + 1),
                    dmin=0.004 * (i + 1),
                    sub_scales=[10 + i] * 8,
                    sub_mins=[3 + i] * 8,
                    nibbles=rng.randint(0, 16, size=256).tolist(),
                )
            )
        raw = np.concatenate(source_blocks)
        shape = (8, 96)  # 768 values: rows split across three 256-value blocks

        gguf_deq = quants.dequantize(
            raw.reshape(3, -1), quants.GGMLQuantizationType.Q4_K
        ).reshape(shape)
        result = repack_gguf_tensor(raw, _Q4_K, shape=shape)
        ort_deq = _dequantize_repacked_q4(result, shape[1])

        assert result.weight.shape == (8, 3, 16)
        max_abs_diff = float(np.max(np.abs(ort_deq - gguf_deq)))
        assert max_abs_diff <= float(result.scales.max()) * 0.51 + 1e-6

    def test_multiple_rows_and_super_blocks(self):
        """N=2 rows, K=512 -> 2 super-blocks per row."""
        sc = [10] * 8
        mn = [5] * 8
        nibs = [7] * 256
        block = _make_q4_k_block(d=0.1, dmin=0.05, sub_scales=sc, sub_mins=mn, nibbles=nibs)
        # 2 rows x 2 super-blocks = 4 blocks total
        raw = np.tile(block, 4)

        result = repack_gguf_tensor(raw, _Q4_K, shape=(2, 512))

        # 2 super-blocks/row x 8 sub-blocks/super = 16 sub-blocks/row
        assert result.weight.shape == (2, 16, 16)
        assert result.scales.shape == (2, 16)
        assert result.zero_points.shape == (2, 8)  # 16 zps / 2

    def test_d_zero_produces_zero_output(self):
        """When d=0, all effective scales and zero points are zero.

        This covers pruned layers where an entire super-block contributes
        nothing (all-zero weights).
        """
        sc = [10, 20, 30, 40, 15, 25, 35, 45]
        mn = [5, 10, 15, 20, 8, 12, 18, 22]
        nibs = [7] * 256
        block = _make_q4_k_block(d=0.0, dmin=0.0, sub_scales=sc, sub_mins=mn, nibbles=nibs)
        raw = block.reshape(-1)

        result = repack_gguf_tensor(raw, _Q4_K, shape=(1, 256))

        # All effective scales should be zero (d * sub_scale = 0)
        np.testing.assert_array_equal(result.scales, 0)
        # All zero points should be zero (guarded against div-by-zero)
        np.testing.assert_array_equal(result.zero_points, 0)

    def test_q4_k_in_can_repack(self):
        """Q4_K (type 12) is recognized as repackable."""
        assert can_repack(12) is True


class TestRepackQ6K:
    """Q6_K (type 14) -> MatMulNBits 4-bit repacking."""

    def test_in_can_repack(self):
        """Q6_K (type 14) is recognized as repackable."""
        assert can_repack(14) is True

    def test_dequantization_matches_gguf_reference_exactly(self):
        """Our unpack must equal ggml's ``dequantize_row_q6_K``, bit for bit.

        Q6_K interleaves each 256-element super-block across two halves, four
        emission groups, and a 16-entry scale table. Every plausible-looking
        wrong permutation still produces finite numbers of the right magnitude,
        so only an exact comparison against the reference implementation can
        distinguish a correct unpack from a subtly transposed one -- and the
        lossy requantization that follows would mask the difference, which is
        why this compares the dequantized floats directly.
        """
        gguf_quants = pytest.importorskip("gguf.quants")
        from gguf.constants import GGMLQuantizationType

        from mobius.integrations.gguf._repacker import _dequantize_q6_k

        rng = np.random.default_rng(0)
        n_super = 8
        blocks = rng.integers(0, 256, size=(n_super, 210), dtype=np.uint8)
        # Keep `d` finite and non-denormal so the comparison is about layout.
        blocks[:, 208:210] = np.frombuffer(
            np.float16([1.5] * n_super).tobytes(), dtype=np.uint8
        ).reshape(n_super, 2)

        expected = gguf_quants.dequantize(
            blocks.reshape(-1).copy(), GGMLQuantizationType.Q6_K
        ).astype(np.float32).ravel()
        got = _dequantize_q6_k(blocks)

        np.testing.assert_array_equal(got, expected)

    def test_repack_stays_within_half_a_block_scale(self):
        """The 4-bit requantization is lossy, but bounded by its own scale."""
        from mobius.integrations.gguf._repacker import _dequantize_q6_k, _repack_q6_k

        rng = np.random.default_rng(1)
        n_out, k_in = 4, 256
        n_super = n_out * k_in // 256
        blocks = rng.integers(0, 256, size=(n_super, 210), dtype=np.uint8)
        blocks[:, 208:210] = np.frombuffer(
            np.float16([1.5] * n_super).tobytes(), dtype=np.uint8
        ).reshape(n_super, 2)

        reference = _dequantize_q6_k(blocks)[: n_out * k_in].reshape(n_out, k_in)
        result = _repack_q6_k(blocks, n_out, k_in)
        got = _dequantize_repacked_q4(result, k_in)

        assert result.weight.shape[0] == n_out
        max_scale = float(result.scales.max())
        assert np.max(np.abs(got - reference)) <= max_scale * 0.51 + 1e-6

    def test_super_block_byte_and_element_counts(self):
        """Q6_K is 210 bytes per 256 elements; a wrong size mis-slices silently."""
        from mobius.integrations.gguf._repacker import (
            _BLOCK_BYTES,
            _GGUF_BLOCK_ELEMENTS,
            _GGUF_Q6_K,
        )

        assert _BLOCK_BYTES[_GGUF_Q6_K] == 210
        assert _GGUF_BLOCK_ELEMENTS[_GGUF_Q6_K] == 256


class TestRepackDequantizedTensor:
    def test_asymmetric_q4_round_trip_bound(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=(3, 70)).astype(np.float32)
        result = repack_dequantized_tensor(values)
        dequantized = _dequantize_repacked_q4(result, values.shape[1])

        max_abs_diff = float(np.max(np.abs(dequantized - values)))
        assert max_abs_diff <= float(result.scales.max()) * 0.51 + 1e-6
        assert result.weight.shape == (3, 3, 16)
        assert result.zero_points.shape == (3, 2)


# ---- Q1_0 tests ----


class TestRepackQ10:
    """Tests for Q1_0 (1-bit binary) -> MatMulNBits 2-bit repacking."""

    def test_in_can_repack(self):
        """Q1_0 (type 41) is recognized as repackable."""
        assert can_repack(_Q1_0) is True

    def test_single_block_shape(self):
        """One Q1_0 block (128 elt) -> one 32-byte MatMulNBits block, bits=2."""
        bits = [1] * 128
        block = _make_q1_0_block(scale=0.5, bits=bits)
        raw = block.reshape(-1)
        result = repack_gguf_tensor(raw, _Q1_0, shape=(1, 128))
        assert isinstance(result, RepackedTensor)
        assert result.bits == 2
        assert result.block_size == 128
        # 1 row, 1 block, 128*2/8 = 32 bytes blob
        assert result.weight.shape == (1, 1, 32)
        assert result.scales.shape == (1, 1)
        # zero_points: ceil(1 * 2 / 8) = 1 byte
        assert result.zero_points.shape == (1, 1)

    def test_zero_points_are_0x55(self):
        """All zero-points = 1 (packed as 0x55 = 01_01_01_01)."""
        bits = [0] * 128
        block = _make_q1_0_block(scale=1.0, bits=bits)
        raw = block.reshape(-1)
        result = repack_gguf_tensor(raw, _Q1_0, shape=(1, 128))
        assert result.zero_points is not None
        np.testing.assert_array_equal(result.zero_points, 0x55)

    def test_all_positive_bits_pack_to_code_2(self):
        """All bits = 1 -> every 2-bit code = 2; byte = 0xAA (10_10_10_10)."""
        bits = [1] * 128
        block = _make_q1_0_block(scale=1.0, bits=bits)
        raw = block.reshape(-1)
        result = repack_gguf_tensor(raw, _Q1_0, shape=(1, 128))
        # Each byte = (2<<6) | (2<<4) | (2<<2) | 2 = 0xAA
        np.testing.assert_array_equal(result.weight, 0xAA)

    def test_all_negative_bits_pack_to_code_0(self):
        """All bits = 0 -> every 2-bit code = 0; all bytes zero."""
        bits = [0] * 128
        block = _make_q1_0_block(scale=1.0, bits=bits)
        raw = block.reshape(-1)
        result = repack_gguf_tensor(raw, _Q1_0, shape=(1, 128))
        np.testing.assert_array_equal(result.weight, 0x00)

    def test_round_trip_dequantize(self):
        """Repacked W (with zp=1, scale=d) reproduces Q1_0 dequant exactly.

        Q1_0 dequant per llama.cpp: ``bit ? +d : -d``.
        MatMulNBits dequant: ``(B - zp) * scale``.
        With zp=1, scale=d, B = 2*bit: result = (2*bit - 1) * d = +/-d. ✓
        """
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, size=128).tolist()
        scale = 0.25
        block = _make_q1_0_block(scale=scale, bits=bits)
        raw = block.reshape(-1)
        result = repack_gguf_tensor(raw, _Q1_0, shape=(1, 128))

        # Unpack each ORT 2-bit code from the 32-byte blob
        blob = result.weight[0, 0]  # (32,) uint8
        codes = np.empty(128, dtype=np.uint8)
        for byte_i in range(32):
            for slot in range(4):
                codes[byte_i * 4 + slot] = (blob[byte_i] >> (2 * slot)) & 0x3

        # Each code should be 2 * original_bit
        expected_codes = 2 * np.array(bits, dtype=np.uint8)
        np.testing.assert_array_equal(codes, expected_codes)

        # Dequantize via MatMulNBits formula and compare to llama.cpp Q1_0
        scale_v = result.scales[0, 0].astype(np.float32)
        zp = 1  # encoded as 0x55 across the byte
        deq_ort = (codes.astype(np.float32) - zp) * scale_v
        expected_deq = np.where(np.array(bits) == 1, scale, -scale).astype(np.float32)
        np.testing.assert_allclose(deq_ort, expected_deq, rtol=1e-3)

    def test_multi_block_multi_row(self):
        """Multiple rows x multiple blocks pack into the expected shape."""
        n, k = 4, 256  # 4 rows, 2 blocks per row
        rng = np.random.default_rng(1)
        blocks = [
            _make_q1_0_block(
                scale=float(rng.uniform(0.1, 1.0)),
                bits=rng.integers(0, 2, size=128).tolist(),
            )
            for _ in range(n * 2)
        ]
        raw = np.concatenate(blocks)
        result = repack_gguf_tensor(raw, _Q1_0, shape=(n, k))
        # 4 rows x 2 blocks x 32 bytes
        assert result.weight.shape == (n, 2, 32)
        assert result.scales.shape == (n, 2)
        # zero_points: ceil(2*2/8) = 1 byte per row
        assert result.zero_points.shape == (n, 1)

    def test_bit_order_lsb_first(self):
        """Bit j of Q1_0 byte j//8 -> code in slot j%4 of ORT byte (j//4)%(blob/4).

        Sets only element 0 = +1 (bit 0 of Q1_0 byte 0) and verifies that
        ORT byte 0 has its low 2 bits = 2 and all other bits = 0.
        """
        bits = [0] * 128
        bits[0] = 1
        block = _make_q1_0_block(scale=1.0, bits=bits)
        raw = block.reshape(-1)
        result = repack_gguf_tensor(raw, _Q1_0, shape=(1, 128))
        blob = result.weight[0, 0]
        # Code 0 (low 2 bits of byte 0) should be 2; everything else = 0
        assert blob[0] == 0b00000010
        np.testing.assert_array_equal(blob[1:], 0)

        # Set element 7 (bit 7 of Q1_0 byte 0): this is slot 3 of ORT byte 1.
        bits = [0] * 128
        bits[7] = 1
        block = _make_q1_0_block(scale=1.0, bits=bits)
        raw = block.reshape(-1)
        result = repack_gguf_tensor(raw, _Q1_0, shape=(1, 128))
        blob = result.weight[0, 0]
        # ORT byte 1 covers codes 4..7; code 3 of that byte sits in bits 6..7
        assert blob[0] == 0
        assert blob[1] == 0b10000000  # code 3 = 2 (binary 10) in slot 3
        np.testing.assert_array_equal(blob[2:], 0)
