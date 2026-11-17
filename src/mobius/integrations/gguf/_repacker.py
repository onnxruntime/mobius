# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Preserve or repack GGUF quantized blocks for ORT custom operators.

Converts raw GGUF block data for exact legacy Q4_0, Q8_0, and Q1_0
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

from mobius.integrations.gguf._quant_registry import iter_quant_specs, quant_spec_by_name
from mobius.integrations.gguf._spec import NativeBlockSpec

__all__ = [
    "NATIVE_BLOCK_BYTE_SIZES",
    "NativeBlockSpec",
    "RepackedTensor",
    "can_repack",
    "native_block_spec",
    "preserve_native_blocks",
    "repack_dequantized_tensor",
    "repack_gguf_tensor",
    "repack_quant_params",
]

_BLOCK_SIZE = 32


def _type_id(name: str) -> int:
    """Return the pinned ``ggml_type`` id for an upper-case type name."""
    spec = quant_spec_by_name(name)
    if spec is None:
        raise ValueError(f"GGML type {name!r} is not in the pinned llama.cpp census")
    return spec.ggml_type_id


# GGUF quantization type IDs, resolved from the pinned llama.cpp census rather
# than hand-typed, so they cannot drift from ``gguf.GGMLQuantizationType``.
_GGUF_Q4_0 = _type_id("Q4_0")
_GGUF_Q4_1 = _type_id("Q4_1")
_GGUF_Q8_0 = _type_id("Q8_0")
_GGUF_Q4_K = _type_id("Q4_K")
_GGUF_Q6_K = _type_id("Q6_K")
_GGUF_Q1_0 = _type_id("Q1_0")

# Block geometry, repack targets, and native layouts are all derived from
# ``_quant_registry``. They used to be four literal dicts here plus three more
# in ``_builder``, kept in sync by hand; a type present in one and missing from
# another raised ``KeyError`` partway through a build.
#
# Block byte sizes:
#   Q4_0  18 = 2B scale + 16B quants
#   Q4_1  20 = 2B scale + 2B min + 16B quants
#   Q8_0  34 = 2B scale + 32B int8 values
#   Q4_K 144 = 2B d + 2B dmin + 12B sub-scales + 128B quants
#   Q6_K 210 = 128B low nibbles + 64B high bits + 16B scales + 2B d
#   Q1_0  18 = 2B scale + 16B packed bits (128 elements)
_BLOCK_BYTES: dict[int, int] = {
    spec.ggml_type_id: spec.block_bytes
    for spec in iter_quant_specs()
    if spec.affine_repack is not None
}

# Elements per GGUF block. Q4_K/Q6_K use 256-element "super-blocks" that
# decompose into 8 sub-blocks of 32 for MatMulNBits; Q1_0 uses 128-element
# blocks (QK1_0 from llama.cpp).
_GGUF_BLOCK_ELEMENTS: dict[int, int] = {
    spec.ggml_type_id: spec.block_elements
    for spec in iter_quant_specs()
    if spec.affine_repack is not None
}

_SUPPORTED_TYPES = frozenset(_BLOCK_BYTES)

# MatMulNBits representation produced for each supported GGUF type.
_REPACK_PARAMS: dict[int, tuple[int, int]] = {
    spec.ggml_type_id: spec.affine_repack.as_params()
    for spec in iter_quant_specs()
    if spec.affine_repack is not None
}

_NATIVE_BLOCK_SPECS: dict[int, NativeBlockSpec] = {
    spec.ggml_type_id: spec.native_preserve
    for spec in iter_quant_specs()
    if spec.native_preserve is not None
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
    if gguf_type in {_GGUF_Q4_1, _GGUF_Q4_K, _GGUF_Q6_K}:
        raise ValueError(
            f"GGUF type {gguf_type} cannot be losslessly repacked to MatMulNBits. "
            "Enable MOBIUS_EXPERIMENTAL_OPSET29=1 and use --ep onnx-standard "
            "to preserve original bytes with DequantizePacked; lossy conversion was removed."
        )

    if len(shape) != 2:
        raise ValueError(f"Expected 2D shape (N, K), got {shape}")

    n_out, k_in = shape
    block_bytes = _BLOCK_BYTES[gguf_type]
    gguf_block_elems = _GGUF_BLOCK_ELEMENTS[gguf_type]
    n_blocks_per_row = math.ceil(k_in / gguf_block_elems)
    if gguf_type in (_GGUF_Q4_K, _GGUF_Q6_K):
        # K-quant super-blocks are laid out over the flattened tensor, not
        # independently padded at each logical row boundary. This matters for
        # dimensions such as Qwen2's K=896: rows end halfway through a
        # 256-element super-block, while they still align perfectly to
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


def repack_dequantized_tensor(
    values: np.ndarray,
    *,
    bits: int = 4,
    block_size: int = _BLOCK_SIZE,
    symmetric: bool = False,
) -> RepackedTensor:
    """Reject the removed lossy GGUF requantization path."""
    raise ValueError(
        "Lossy GGUF dequantize/requantize conversion was removed. Enable "
        "MOBIUS_EXPERIMENTAL_OPSET29=1 and use --ep onnx-standard for original "
        f"GGUF external weights (requested INT{bits}/block-{block_size}, "
        f"source shape={values.shape})."
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
