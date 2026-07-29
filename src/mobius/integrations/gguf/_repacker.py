# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Preserve or repack GGUF quantized blocks for ORT custom operators.

Converts raw GGUF block data for Q4_0, Q4_1, Q8_0, Q4_K, and Q1_0
quantization types into the (weight, scales, zero_points) tensors
expected by the ``com.microsoft.MatMulNBits`` operator.

The runtime-native IQ/MXFP4 formats are retained byte-for-byte for
``pkg.nxrt.BlockQuantizedMatMul``.

GGUF block layouts:
    Q4_0 (18 bytes, 32 elt):  [fp16 scale][16B packed nibbles]
    Q4_1 (20 bytes, 32 elt):  [fp16 scale][fp16 min][16B packed nibbles]
    Q8_0 (34 bytes, 32 elt):  [fp16 scale][32B int8 values]
    Q4_K (144 bytes, 256 elt): [fp16 d][fp16 dmin][12B sub-scales][128B nibbles]
    Q1_0 (18 bytes, 128 elt): [fp16 scale][16B packed bits, LSB-first]
        Dequant: ``bit ? +d : -d`` (1-bit binary).

MatMulNBits expects:
    weight:      [N, n_blocks, blob_size] uint8
    scales:      [N, n_blocks]            float16
    zero_points: [N, ceil(n_blocks*bits/8)] uint8 (bit-packed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_BLOCK_SIZE = 32

# GGUF quantization type IDs (from gguf.GGMLQuantizationType enum)
_GGUF_Q4_0 = 2
_GGUF_Q4_1 = 3
_GGUF_Q8_0 = 8
_GGUF_Q4_K = 12
_GGUF_IQ2_XXS = 16
_GGUF_IQ2_XS = 17
_GGUF_IQ3_XXS = 18
_GGUF_IQ1_S = 19
_GGUF_IQ4_NL = 20
_GGUF_IQ3_S = 21
_GGUF_IQ2_S = 22
_GGUF_IQ4_XS = 23
_GGUF_IQ1_M = 29
_GGUF_MXFP4 = 39
_GGUF_Q1_0 = 41

# Block byte sizes per GGUF type
_BLOCK_BYTES = {
    _GGUF_Q4_0: 18,  # 2B scale + 16B quants
    _GGUF_Q4_1: 20,  # 2B scale + 2B min + 16B quants
    _GGUF_Q8_0: 34,  # 2B scale + 32B int8 values
    _GGUF_Q4_K: 144,  # 2B d + 2B dmin + 12B scales + 128B quants
    _GGUF_Q1_0: 18,  # 2B scale + 16B packed bits (128 elements)
}

# Elements per GGUF block. Q4_K uses 256-element "super-blocks"
# that decompose into 8 sub-blocks of 32 for MatMulNBits. Q1_0 uses
# 128-element blocks (QK1_0 from llama.cpp).
_GGUF_BLOCK_ELEMENTS = {
    _GGUF_Q4_0: 32,
    _GGUF_Q4_1: 32,
    _GGUF_Q8_0: 32,
    _GGUF_Q4_K: 256,
    _GGUF_Q1_0: 128,
}

_SUPPORTED_TYPES = frozenset(_BLOCK_BYTES.keys())

# MatMulNBits representation produced for each supported GGUF type.
_REPACK_PARAMS = {
    _GGUF_Q4_0: (4, 32),
    _GGUF_Q4_1: (4, 32),
    _GGUF_Q8_0: (8, 32),
    _GGUF_Q4_K: (4, 32),
    _GGUF_Q1_0: (2, 128),
}


@dataclass(frozen=True)
class NativeBlockSpec:
    """Serialized GGUF block layout accepted directly by the runtime."""

    format: str
    elements: int
    bytes: int


_NATIVE_BLOCK_SPECS = {
    _GGUF_MXFP4: NativeBlockSpec("mxfp4", 32, 17),
    _GGUF_IQ4_NL: NativeBlockSpec("iq4_nl", 32, 18),
    _GGUF_IQ4_XS: NativeBlockSpec("iq4_xs", 256, 136),
    _GGUF_IQ3_S: NativeBlockSpec("iq3_s", 256, 110),
    _GGUF_IQ3_XXS: NativeBlockSpec("iq3_xxs", 256, 98),
    _GGUF_IQ2_XXS: NativeBlockSpec("iq2_xxs", 256, 66),
    _GGUF_IQ2_XS: NativeBlockSpec("iq2_xs", 256, 74),
    _GGUF_IQ2_S: NativeBlockSpec("iq2_s", 256, 82),
    _GGUF_IQ1_S: NativeBlockSpec("iq1_s", 256, 50),
    _GGUF_IQ1_M: NativeBlockSpec("iq1_m", 256, 56),
}
NATIVE_BLOCK_BYTE_SIZES = frozenset(spec.bytes for spec in _NATIVE_BLOCK_SPECS.values())


@dataclass
class RepackedTensor:
    """MatMulNBits-compatible representation of a quantized weight.

    Attributes:
        weight: Packed uint8 blob, shape ``[N, n_blocks, blob_size]``.
        scales: Per-block scale factors, float16, shape ``[N, n_blocks]``.
        zero_points: Per-block zero points, uint8, or ``None``.
            For 4-bit: nibble-packed, shape ``[N, ceil(n_blocks/2)]``.
            For 8-bit: shape ``[N, n_blocks]``.
        block_size: Elements per quantization block (always 32 for GGUF).
        bits: Quantization bit-width (4 or 8).
    """

    weight: np.ndarray
    scales: np.ndarray
    zero_points: np.ndarray | None
    block_size: int
    bits: int


def native_block_spec(gguf_type: int) -> NativeBlockSpec | None:
    """Return the runtime-native block layout for a GGUF type, if supported."""
    return _NATIVE_BLOCK_SPECS.get(gguf_type)


def preserve_native_blocks(
    raw_data: np.ndarray,
    gguf_type: int,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Validate and reshape raw GGUF blocks without changing any bytes."""
    spec = native_block_spec(gguf_type)
    if spec is None:
        raise ValueError(f"GGUF type {gguf_type} is not runtime-native")
    if len(shape) != 2:
        raise ValueError(f"Expected 2D shape (N, K), got {shape}")

    n_out, k_in = shape
    n_blocks = math.ceil(k_in / spec.elements)
    expected_bytes = n_out * n_blocks * spec.bytes
    packed = raw_data.ravel().view(np.uint8)
    if packed.size != expected_bytes:
        raise ValueError(
            f"Native {spec.format} data size mismatch: got {packed.size} bytes, "
            f"expected {expected_bytes} for shape {shape} "
            f"with {n_out * n_blocks} blocks x {spec.bytes} bytes"
        )
    return packed.reshape(n_out, n_blocks, spec.bytes)


def can_repack(gguf_type: int) -> bool:
    """Return True if the GGUF type can be repacked to MatMulNBits."""
    return repack_quant_params(gguf_type) is not None


def repack_quant_params(gguf_type: int) -> tuple[int, int] | None:
    """Return the ``(bits, block_size)`` produced for a GGUF type."""
    return _REPACK_PARAMS.get(gguf_type)


def repack_gguf_tensor(
    raw_data: np.ndarray,
    gguf_type: int,
    shape: tuple[int, ...],
) -> RepackedTensor:
    """Repack a GGUF quantized tensor into MatMulNBits format.

    Args:
        raw_data: Raw bytes as a uint8 numpy array (flat).
        gguf_type: GGUF quantization type ID (e.g. 2 for Q4_0).
        shape: Logical weight shape ``(N, K)`` where N = out_features,
            K = in_features. K-quant super-blocks are contiguous over the
            flattened tensor and may cross logical row boundaries.

    Returns:
        A ``RepackedTensor`` with MatMulNBits-compatible arrays.

    Raises:
        ValueError: If the GGUF type is unsupported or data size is wrong.
    """
    if gguf_type not in _SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported GGUF type {gguf_type}. Supported: {sorted(_SUPPORTED_TYPES)}"
        )

    if len(shape) != 2:
        raise ValueError(f"Expected 2D shape (N, K), got {shape}")

    n_out, k_in = shape
    block_bytes = _BLOCK_BYTES[gguf_type]
    gguf_block_elems = _GGUF_BLOCK_ELEMENTS[gguf_type]
    n_blocks_per_row = math.ceil(k_in / gguf_block_elems)
    if gguf_type == _GGUF_Q4_K:
        # K-quant super-blocks are laid out over the flattened tensor, not
        # independently padded at each logical row boundary. This matters for
        # dimensions such as Qwen2's K=896: rows end halfway through a
        # 256-element Q4_K super-block, while they still align perfectly to
        # MatMulNBits' 32-element blocks.
        total_blocks = math.ceil(n_out * k_in / gguf_block_elems)
    else:
        total_blocks = n_out * n_blocks_per_row
    expected_bytes = total_blocks * block_bytes

    if raw_data.size != expected_bytes:
        raise ValueError(
            f"Data size mismatch: got {raw_data.size} bytes, "
            f"expected {expected_bytes} for shape {shape} "
            f"with {total_blocks} blocks x {block_bytes} bytes"
        )

    # Reshape into (total_blocks, block_bytes) then dispatch
    blocks = raw_data.reshape(total_blocks, block_bytes)

    if gguf_type == _GGUF_Q4_0:
        return _repack_q4_0(blocks, n_out, n_blocks_per_row)
    elif gguf_type == _GGUF_Q4_1:
        return _repack_q4_1(blocks, n_out, n_blocks_per_row)
    elif gguf_type == _GGUF_Q4_K:
        return _repack_q4_k(blocks, n_out, k_in)
    elif gguf_type == _GGUF_Q1_0:
        return _repack_q1_0(blocks, n_out, n_blocks_per_row)
    else:
        return _repack_q8_0(blocks, n_out, n_blocks_per_row)


def _reorder_nibbles_gguf_to_ort(
    gguf_packed: np.ndarray,
) -> np.ndarray:
    """Convert GGUF nibble ordering to MatMulNBits ordering.

    GGUF packs 32 elements into 16 bytes as:
        byte i: low nibble = element[i], high nibble = element[i+16]

    MatMulNBits packs as:
        byte j: low nibble = element[2j], high nibble = element[2j+1]

    Args:
        gguf_packed: uint8 array with last dim = 16 (GGUF packed bytes).

    Returns:
        uint8 array with same shape, nibbles reordered for MatMulNBits.
    """
    low = gguf_packed & 0x0F  # Elements 0..15
    high = (gguf_packed >> 4) & 0x0F  # Elements 16..31

    # Group each set of 16 nibbles into 8 pairs, pack each pair
    shape = gguf_packed.shape[:-1]
    low_pairs = low.reshape(*shape, 8, 2)
    high_pairs = high.reshape(*shape, 8, 2)

    ort_low = (low_pairs[..., 1] << 4) | low_pairs[..., 0]  # 8 bytes
    ort_high = (high_pairs[..., 1] << 4) | high_pairs[..., 0]  # 8 bytes

    return np.concatenate([ort_low, ort_high], axis=-1)  # 16 bytes


def _repack_q4_0(
    blocks: np.ndarray,
    n_out: int,
    n_blocks_per_row: int,
) -> RepackedTensor:
    """Repack Q4_0 blocks.

    Q4_0 block (18 bytes): [fp16 scale (2B)][16B packed 4-bit quants]
    Dequant: (nibble - 8) * scale  ->  symmetric with zero_point = 8.

    GGUF nibble ordering differs from MatMulNBits — we reorder during
    repacking.  See ``_reorder_nibbles_gguf_to_ort`` for details.
    """
    # Split scale (first 2 bytes) from quants (remaining 16 bytes)
    raw_scales = blocks[:, :2].copy()
    raw_quants = blocks[:, 2:]  # (total_blocks, 16)

    # Scales: view as fp16 -> (total_blocks,) -> reshape to (N, n_blocks)
    scales = raw_scales.view(np.float16).reshape(n_out, n_blocks_per_row)

    # Reorder nibbles from GGUF order to MatMulNBits order
    ort_quants = _reorder_nibbles_gguf_to_ort(raw_quants)
    weight = ort_quants.reshape(n_out, n_blocks_per_row, 16)

    # Zero points: Q4_0 is symmetric around 8
    # For MatMulNBits 4-bit: two ZPs packed per byte (low=block_i, high=block_i+1)
    # All ZPs are 8, so each packed byte = (8 << 4) | 8 = 0x88
    zp_cols = math.ceil(n_blocks_per_row / 2)
    zero_points = np.full((n_out, zp_cols), 0x88, dtype=np.uint8)
    # If odd number of blocks, the high nibble of the last byte is padding
    # ORT ignores it, but set to 0 for cleanliness
    if n_blocks_per_row % 2 == 1:
        zero_points[:, -1] = 0x08  # low nibble = 8, high nibble = 0

    return RepackedTensor(
        weight=weight,
        scales=scales,
        zero_points=zero_points,
        block_size=_BLOCK_SIZE,
        bits=4,
    )


def _repack_q4_1(
    blocks: np.ndarray,
    n_out: int,
    n_blocks_per_row: int,
) -> RepackedTensor:
    """Repack Q4_1 blocks.

    Q4_1 block (20 bytes): [fp16 scale (2B)][fp16 min (2B)][16B quants]
    Dequant: nibble * scale + min  ->  asymmetric.

    MatMulNBits dequant: (nibble - zp) * scale
    So: zp = round(-min / scale), clamped to [0, 15].
    """
    raw_scales = blocks[:, :2].copy()
    raw_mins = blocks[:, 2:4].copy()
    raw_quants = blocks[:, 4:]  # (total_blocks, 16)

    scales_flat = raw_scales.view(np.float16).astype(np.float32).ravel()
    mins_flat = raw_mins.view(np.float16).astype(np.float32).ravel()

    # Compute zero points: zp = round(-min / scale), clamp to [0, 15]
    # Guard against division by zero: where scale == 0, zp = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        zp_float = np.where(
            scales_flat != 0,
            np.round(-mins_flat / scales_flat),
            0.0,
        )
    zp_uint4 = np.clip(zp_float, 0, 15).astype(np.uint8)

    # Reshape to (N, n_blocks)
    scales = scales_flat.astype(np.float16).reshape(n_out, n_blocks_per_row)
    # Reorder nibbles from GGUF order to MatMulNBits order
    ort_quants = _reorder_nibbles_gguf_to_ort(raw_quants)
    weight = ort_quants.reshape(n_out, n_blocks_per_row, 16)

    # Pack two 4-bit zero points per byte (vectorized, matching Q4_K)
    zp_2d = zp_uint4.reshape(n_out, n_blocks_per_row)
    zp_cols = math.ceil(n_blocks_per_row / 2)
    zp_padded = zp_2d
    if n_blocks_per_row % 2 == 1:
        zp_padded = np.zeros((n_out, n_blocks_per_row + 1), dtype=np.uint8)
        zp_padded[:, :n_blocks_per_row] = zp_2d
    zp_pairs = zp_padded.reshape(n_out, zp_cols, 2)
    zero_points = zp_pairs[:, :, 0] | (zp_pairs[:, :, 1] << 4)

    return RepackedTensor(
        weight=weight,
        scales=scales,
        zero_points=zero_points,
        block_size=_BLOCK_SIZE,
        bits=4,
    )


def _repack_q8_0(
    blocks: np.ndarray,
    n_out: int,
    n_blocks_per_row: int,
) -> RepackedTensor:
    """Repack Q8_0 blocks.

    Q8_0 block (34 bytes): [fp16 scale (2B)][32 x int8 values (32B)]
    Dequant: int8_val * scale  ->  symmetric around 0.

    MatMulNBits dequant: (uint8_val - zp) * scale
    Convert: uint8_val = int8_val + 128, zp = 128.
    """
    raw_scales = blocks[:, :2].copy()
    raw_quants = blocks[:, 2:]  # (total_blocks, 32) as uint8

    scales = raw_scales.view(np.float16).reshape(n_out, n_blocks_per_row)

    # Convert signed int8 -> unsigned uint8 by adding 128
    quants_int8 = raw_quants.view(np.int8).astype(np.int16)
    quants_uint8 = (quants_int8 + 128).astype(np.uint8)
    weight = quants_uint8.reshape(n_out, n_blocks_per_row, 32)

    # Zero points: 128 for all blocks
    zero_points = np.full((n_out, n_blocks_per_row), 128, dtype=np.uint8)

    return RepackedTensor(
        weight=weight,
        scales=scales,
        zero_points=zero_points,
        block_size=_BLOCK_SIZE,
        bits=8,
    )


def _unpack_q4_k_scales(
    scales_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack Q4_K 6-bit packed scales into sub-block scales and mins.

    Each Q4_K super-block stores 8 sub-block scales and 8 sub-block mins
    packed into 12 bytes using 6-bit encoding.  The packing layout (from
    llama.cpp) uses three 4-byte groups::

        Bytes 0-3 (d):   low 6 bits -> sub_scale[0..3],
                         high 2 bits -> sub_scale[4..7] bits 4-5
        Bytes 4-7 (m):   low 6 bits -> sub_min[0..3],
                         high 2 bits -> sub_min[4..7] bits 4-5
        Bytes 8-11 (md): low 4 bits -> sub_scale[4..7] bits 0-3,
                         high 4 bits -> sub_min[4..7] bits 0-3

    Args:
        scales_raw: uint8 array, shape ``(n_super_blocks, 12)``.

    Returns:
        Tuple of ``(sub_scales, sub_mins)``, each ``(n_super_blocks, 8)``
        as uint8 in range [0, 63].
    """
    n = scales_raw.shape[0]
    s = scales_raw.reshape(n, 3, 4)
    d = s[:, 0, :]  # (n, 4) — bytes 0-3
    m = s[:, 1, :]  # (n, 4) — bytes 4-7
    md = s[:, 2, :]  # (n, 4) — bytes 8-11

    # Sub-scales: lower 4 from d[0..3] bits 0-5, upper 4 from md+d
    sc = np.concatenate([d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    # Sub-mins: lower 4 from m[0..3] bits 0-5, upper 4 from md+m
    mn = np.concatenate([m & 0x3F, (md >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc.reshape(n, 8), mn.reshape(n, 8)


def _repack_q4_k(
    blocks: np.ndarray,
    n_out: int,
    k_in: int,
) -> RepackedTensor:
    """Dequantize Q4_K super-blocks and requantize to MatMulNBits.

    Q4_K uses 256-element super-blocks with a two-level scale hierarchy:
    ``value = d * sub_scale[i] * nibble - dmin * sub_min[i]``. Its
    fractional effective zero-points cannot generally be represented by
    MatMulNBits' packed uint4 zero-points. We therefore reference-dequantize
    the super-blocks, restore the logical row layout, and affine-requantize
    each 32-element MatMulNBits block.

    Requantization is lossy, but every source value is represented within
    half of the emitted block scale (apart from floating-point roundoff).
    Unlike directly rounding Q4_K's effective zero-point, this never clamps
    a large source offset to 15 while keeping an incompatible source scale.

    Args:
        blocks: uint8 array, shape ``(total_super_blocks, 144)``.
        n_out: Number of output rows (N dimension).
        k_in: Number of input columns (K dimension).

    Returns:
        A ``RepackedTensor`` with ``block_size=32`` and ``bits=4``.
    """
    total = blocks.shape[0]

    # Parse super-block fields (144 bytes each):
    # [d: 2B fp16][dmin: 2B fp16][scales: 12B][quants: 128B]
    d_raw = blocks[:, :2].copy()
    dmin_raw = blocks[:, 2:4].copy()
    scales_raw = blocks[:, 4:16]
    qs_raw = blocks[:, 16:]  # (total, 128)

    d = d_raw.view(np.float16).astype(np.float32).ravel()  # (total,)
    dmin = dmin_raw.view(np.float16).astype(np.float32).ravel()  # (total,)

    # Unpack 6-bit sub-block scales and mins
    sub_scales, sub_mins = _unpack_q4_k_scales(scales_raw)
    sub_scales_f = sub_scales.astype(np.float32)  # (total, 8)
    sub_mins_f = sub_mins.astype(np.float32)  # (total, 8)

    # Effective per-sub-block scale and minimum.
    eff_scales = d[:, None] * sub_scales_f  # (total, 8)
    eff_mins = dmin[:, None] * sub_mins_f  # (total, 8)

    # Unpack 4-bit quants from Q4_K layout.
    # 128 bytes = 4 groups of 32 bytes. Each group encodes two sub-blocks:
    #   byte[j] low nibble  -> even sub-block element j
    #   byte[j] high nibble -> odd sub-block element j
    qs = qs_raw.reshape(total, 4, 1, 32)
    shifts = np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    qs = (qs >> shifts) & np.uint8(0x0F)  # (total, 4, 2, 32)
    qs = qs.reshape(total, 8, 32)  # (total, 8, 32) — 8 sub-blocks

    dequantized = (
        eff_scales[:, :, None] * qs.astype(np.float32) - eff_mins[:, :, None]
    ).reshape(-1)
    logical_elements = n_out * k_in
    if dequantized.size < logical_elements:
        raise ValueError(
            f"Q4_K data has {dequantized.size} elements, "
            f"but shape ({n_out}, {k_in}) requires {logical_elements}"
        )

    values = dequantized[:logical_elements].reshape(n_out, k_in)
    return repack_dequantized_tensor(values, bits=4, block_size=_BLOCK_SIZE)


def repack_dequantized_tensor(
    values: np.ndarray,
    *,
    bits: int = 4,
    block_size: int = _BLOCK_SIZE,
    symmetric: bool = False,
) -> RepackedTensor:
    """Affine-quantize a float matrix into MatMulNBits layout.

    This is used for mixed GGUF presets such as Q4_K_M, where projection
    tensors may use Q4_K, Q5_0, Q6_K, and Q8_0 within one model while the
    ONNX graph must use one MatMulNBits configuration throughout.
    """
    if values.ndim != 2:
        raise ValueError(f"Expected 2D values (N, K), got shape {values.shape}")
    if bits not in (4, 8) or block_size != _BLOCK_SIZE:
        raise ValueError(
            "Float requantization currently supports only "
            f"bits=4/8, block_size={_BLOCK_SIZE}; got bits={bits}, "
            f"block_size={block_size}"
        )

    n_out, k_in = values.shape
    n_blocks = math.ceil(k_in / block_size)
    padded_k = n_blocks * block_size
    padded = np.zeros((n_out, padded_k), dtype=np.float32)
    padded[:, :k_in] = values.astype(np.float32, copy=False)
    blocks = padded.reshape(n_out, n_blocks, block_size)

    if bits == 8:
        block_min = np.minimum(blocks.min(axis=-1), 0.0)
        block_max = np.maximum(blocks.max(axis=-1), 0.0)
        if symmetric:
            scales = np.maximum(-block_min / 128.0, block_max / 127.0)
            zero_points_arr = np.full_like(scales, 128, dtype=np.uint8)
        else:
            scales = (block_max - block_min) / 255.0
            safe_scales = np.where(scales != 0, scales, 1.0)
            zero_points_arr = np.clip(
                np.rint(-block_min / safe_scales), 0, 255
            ).astype(np.uint8)
        safe_scales = np.where(scales != 0, scales, 1.0)
        quants = np.rint(blocks / safe_scales[:, :, None])
        quants += zero_points_arr[:, :, None]
        weight = np.clip(quants, 0, 255).astype(np.uint8)
        weight = np.where(scales[:, :, None] != 0, weight, 0).astype(np.uint8)
        return RepackedTensor(
            weight=weight,
            scales=scales.astype(np.float32),
            zero_points=None if symmetric else zero_points_arr,
            block_size=block_size,
            bits=bits,
        )

    if symmetric:
        # MatMulNBits' implicit uint4 zero-point is 8. Choose a scale that
        # covers the asymmetric signed code range [-8, 7].
        block_min = blocks.min(axis=-1)
        block_max = blocks.max(axis=-1)
        scales = np.maximum(-block_min / 8.0, block_max / 7.0)
        zero_points = np.full_like(scales, 8, dtype=np.uint8)
    else:
        # Include zero in the representable range so the rounded zero-point
        # always fits in uint4 without changing the selected scale.
        block_min = np.minimum(blocks.min(axis=-1), 0.0)
        block_max = np.maximum(blocks.max(axis=-1), 0.0)
        scales = (block_max - block_min) / 15.0
        safe_scales = np.where(scales != 0, scales, 1.0)
        zero_points = np.clip(np.rint(-block_min / safe_scales), 0, 15).astype(np.uint8)

    safe_scales = np.where(scales != 0, scales, 1.0)
    quants = np.rint(blocks / safe_scales[:, :, None])
    quants += zero_points[:, :, None]
    quants = np.clip(quants, 0, 15).astype(np.uint8)
    quants = np.where(scales[:, :, None] != 0, quants, 0).astype(np.uint8)

    pairs = quants.reshape(n_out, n_blocks, block_size // 2, 2)
    weight = (pairs[..., 1] << 4) | pairs[..., 0]

    if symmetric:
        packed_zero_points = None
    else:
        zp_cols = math.ceil(n_blocks / 2)
        zp_padded = np.zeros((n_out, zp_cols * 2), dtype=np.uint8)
        zp_padded[:, :n_blocks] = zero_points
        zp_pairs = zp_padded.reshape(n_out, zp_cols, 2)
        packed_zero_points = zp_pairs[:, :, 0] | (zp_pairs[:, :, 1] << 4)

    return RepackedTensor(
        weight=weight,
        scales=scales.astype(np.float32),
        zero_points=packed_zero_points,
        block_size=block_size,
        bits=bits,
    )


# Lookup table: 4-bit nibble (LSB-first) -> 8-bit ORT 2-bit packed byte.
# Each Q1_0 bit b in {0, 1} maps to MatMulNBits 2-bit code 2*b in {0, 2}
# (which under zp=1 dequantizes to {-1, +1}).
#
# Q1_0 bit packing: bit `j % 8` of `qs[j // 8]` holds element j.
# MatMulNBits 2-bit packing: code k of byte i = (byte >> (2*k)) & 0x3.
#
# nibble layout (LSB-first):    bit0 bit1 bit2 bit3
# packed 2-bit byte (LSB-first): c0   c1   c2   c3
#   where c_k = 2 * bit_k, giving byte = (bit3<<7)|(bit2<<5)|(bit1<<3)|(bit0<<1).
def _build_q1_0_expand_table() -> np.ndarray:
    table = np.zeros(16, dtype=np.uint8)
    for nibble in range(16):
        byte = 0
        for k in range(4):
            bit = (nibble >> k) & 1
            byte |= (bit * 2) << (2 * k)
        table[nibble] = byte
    return table


_Q1_0_EXPAND = _build_q1_0_expand_table()


def _repack_q1_0(
    blocks: np.ndarray,
    n_out: int,
    n_blocks_per_row: int,
) -> RepackedTensor:
    """Repack Q1_0 (1-bit binary) blocks into MatMulNBits 2-bit format.

    Q1_0 block (18 bytes, 128 elements):
        ``[fp16 scale (2B)][16B packed bits, LSB-first within each byte]``
    Dequant per llama.cpp: ``bit ? +scale : -scale``.

    MatMulNBits has no native 1-bit format, so we inflate each Q1_0 bit
    ``b in {0, 1}`` to a 2-bit code ``B in {0, 2}``. With zero-point=1 and
    scale=d, the op then dequantizes to ``(B - 1) * d = {-d, +d}`` — an
    exact equivalent of Q1_0 at 2x the on-disk weight bytes but bit-exact
    in values.

    Each 8-bit Q1_0 byte encodes 8 elements, producing 2 MatMulNBits 2-bit
    bytes (4 codes per byte).
    """
    raw_scales = blocks[:, :2].copy()
    raw_bits = blocks[:, 2:]  # (total_blocks, 16)

    scales = raw_scales.view(np.float16).reshape(n_out, n_blocks_per_row)

    # Expand each Q1_0 byte (8 bits) into 2 MatMulNBits 2-bit bytes
    # via the precomputed 4-bit nibble -> 8-bit lookup table.
    low_nibbles = raw_bits & 0x0F  # bits 0..3 of each Q1_0 byte
    high_nibbles = (raw_bits >> 4) & 0x0F  # bits 4..7

    ort_low = _Q1_0_EXPAND[low_nibbles]  # (total_blocks, 16) — even ORT bytes
    ort_high = _Q1_0_EXPAND[high_nibbles]  # (total_blocks, 16) — odd ORT bytes

    # Interleave: ORT bytes for elements 0..7 are (ort_low[0], ort_high[0]),
    # for elements 8..15 are (ort_low[1], ort_high[1]), etc.
    blob_size = 32  # 128 elements * 2 bits / 8 = 32 bytes per block
    interleaved = np.empty((raw_bits.shape[0], blob_size), dtype=np.uint8)
    interleaved[:, 0::2] = ort_low
    interleaved[:, 1::2] = ort_high
    weight = interleaved.reshape(n_out, n_blocks_per_row, blob_size)

    # Zero points: all blocks share zp=1 (so dequant maps {0,2} -> {-1,+1}).
    # Packed as 4 ZPs per byte (bits=2): byte = (1<<6) | (1<<4) | (1<<2) | 1 = 0x55.
    # Tail of the last byte (when n_blocks_per_row % 4 != 0) is unused by ORT;
    # we still fill it with 0x55 for cleanliness.
    zp_cols = math.ceil(n_blocks_per_row / 4)
    zero_points = np.full((n_out, zp_cols), 0x55, dtype=np.uint8)

    return RepackedTensor(
        weight=weight,
        scales=scales,
        zero_points=zero_points,
        block_size=128,
        bits=2,
    )
