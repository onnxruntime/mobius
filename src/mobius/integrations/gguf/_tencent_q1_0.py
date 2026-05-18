# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reader for Tencent's custom Q1_0 layout used by Hy-MT1.5-1.8B-2bit.

The HuggingFace repo ``AngelSlim/Hy-MT1.5-1.8B-2bit-GGUF`` reuses
``GGML_TYPE_Q1_0`` (id 41) as a tag but stores tensors with an
**entirely different per-tensor layout** than llama.cpp mainline:

* Mainline Q1_0 block (18 bytes, 128 elements):
    ``[fp16 d][16B packed 1-bit signs]`` — dequant: ``bit ? +d : -d``.

* Tencent Q1_0 block (130 bytes, 512 elements):
    ``[fp16 scale][128B packed 2-bit codes, LSB-first slots]``
    Codes ``c ∈ {0,1,2,3}`` dequantize via the SEQ codebook
    ``{-1.5, -0.5, 0.5, 1.5}`` scaled by ``2·scale`` (equivalently
    ``{-3, -1, +1, +3}·scale``). Each output row of ``K`` elements
    splits into ``K/512`` blocks, each with its own scale.

Because the on-disk row stride differs (~4× larger than mainline),
mainline llama.cpp refuses to load these files — every tensor after
the first Q1_0 entry lands at the wrong offset. gguf-py likewise
under-reads tensor data. We bypass the size calculation entirely by
reading from the **explicit per-tensor file offset** stored in the
GGUF header.

The output of :func:`parse_tencent_q1_0_tensor` is a
:class:`~mobius.integrations.gguf._repacker.RepackedTensor` whose 2-bit
codes have been inflated to 4-bit slots (codes ∈ ``{0, 2, 4, 6}``) and
paired with integer zero-point ``3``. ORT's ``MatMulNBits`` then
computes ``scale · (B − 3)`` which equals ``{-3·s, -1·s, +1·s, +3·s}``,
i.e. exactly Tencent's dequantization.

Two implementation details forced by current ORT CPU kernel limits:

* We use 4-bit MatMulNBits (not 2-bit) because the CPU kernel does not
  yet support float-valued zero-points at ``bits=2``, which is what
  the half-integer SEQ offset ``1.5`` would otherwise require.
* The output ``block_size`` is ``128``, not Tencent's native ``512``,
  because the CPU kernel silently produces zeros for
  ``bits=4, block_size=512``. Each Tencent 512-element scale is
  replicated across 4 consecutive ORT blocks of 128 elements; the
  dequantized values are identical.
"""

from __future__ import annotations

__all__ = [
    "TENCENT_Q1_0_NATIVE_BLOCK_SIZE",
    "TENCENT_Q1_0_ORT_BLOCK_SIZE",
    "is_tencent_q1_0_layout",
    "parse_tencent_q1_0_tensor",
]

import math
import os

import numpy as np

from mobius.integrations.gguf._repacker import RepackedTensor

# Native Tencent block size (one fp16 scale governs this many weights).
TENCENT_Q1_0_NATIVE_BLOCK_SIZE = 512

# Block size we expose to ORT's MatMulNBits. The native 512 is rejected
# silently by the CPU kernel (returns zeros), so we replicate each
# native scale across 4 ORT sub-blocks of 128 elements.
TENCENT_Q1_0_ORT_BLOCK_SIZE = 128

# Per native block on-disk: 2 bytes (fp16 scale) + 512·2/8 = 128 bytes codes.
_TENCENT_Q1_0_BLOCK_BYTES = 2 + TENCENT_Q1_0_NATIVE_BLOCK_SIZE // 4  # 130

_SUBBLOCKS_PER_NATIVE = TENCENT_Q1_0_NATIVE_BLOCK_SIZE // TENCENT_Q1_0_ORT_BLOCK_SIZE  # 4


def is_tencent_q1_0_layout(gguf_model) -> bool:
    """Detect whether *gguf_model* uses Tencent's custom Q1_0 layout.

    Heuristic: any Q1_0 (type 41) tensor whose total on-disk size exceeds
    the mainline ``ggml_row_size`` value is using the Tencent layout. We
    compute the actual byte span from consecutive tensor offsets and
    compare against the mainline expectation.

    Returns ``True`` iff the file is recognisably Tencent-formatted.
    """
    from gguf import GGMLQuantizationType

    reader = gguf_model._reader
    tensors = sorted(
        reader.tensors,
        key=lambda t: _tensor_data_offset(t),
    )
    for i, t in enumerate(tensors):
        if t.tensor_type != GGMLQuantizationType.Q1_0:
            continue
        if i + 1 >= len(tensors):
            continue
        actual_span = _tensor_data_offset(tensors[i + 1]) - _tensor_data_offset(t)
        ne0 = int(t.shape[0])
        ne1 = int(t.shape[1]) if len(t.shape) > 1 else 1
        mainline_size = ne0 * ne1 * 18 // 128
        if actual_span >= mainline_size + 130:  # one Tencent block bigger
            return True
    return False


def _tensor_data_offset(tensor) -> int:
    """Return the file-relative data offset for a GGUF reader tensor."""
    # Each tensor's metadata ends with the data offset (relative to the
    # data section start). It is the last part in the field record.
    return int(tensor.field.parts[tensor.field.data[-1]][0])


def parse_tencent_q1_0_tensor(
    file_path: str | os.PathLike,
    data_section_offset: int,
    tensor,
) -> RepackedTensor:
    """Parse one Tencent-Q1_0 tensor into MatMulNBits-ready arrays.

    Args:
        file_path: Path to the source ``.gguf`` file.
        data_section_offset: Absolute byte offset where the GGUF data
            section begins (``GGUFReader.data_offset``).
        tensor: ``gguf.ReaderTensor`` for the target weight. Must have
            ``tensor_type == GGML_TYPE_Q1_0`` and a 2D shape.

    Returns:
        A :class:`RepackedTensor` with ``bits=4`` and
        ``block_size=512``. Codes are inflated from 2-bit ``c`` to 4-bit
        ``2·c``; zero-points are packed integer ``3``; scales are the
        per-block fp16 values read from disk (unchanged).
    """
    ne0 = int(tensor.shape[0])  # K (input features per output row)
    ne1 = int(tensor.shape[1])  # N (output features)
    if ne0 % TENCENT_Q1_0_NATIVE_BLOCK_SIZE != 0:
        raise ValueError(
            f"Tensor {tensor.name!r} has K={ne0} not divisible by "
            f"Tencent Q1_0 native block size {TENCENT_Q1_0_NATIVE_BLOCK_SIZE}"
        )

    n_native_blocks_per_row = ne0 // TENCENT_Q1_0_NATIVE_BLOCK_SIZE
    bytes_per_row = n_native_blocks_per_row * _TENCENT_Q1_0_BLOCK_BYTES
    total_bytes = ne1 * bytes_per_row

    abs_offset = data_section_offset + _tensor_data_offset(tensor)
    with open(file_path, "rb") as f:
        f.seek(abs_offset)
        blob = f.read(total_bytes)
    if len(blob) != total_bytes:
        raise IOError(
            f"Short read for {tensor.name!r}: got {len(blob)} bytes, expected {total_bytes}"
        )

    arr = np.frombuffer(blob, dtype=np.uint8).reshape(
        ne1, n_native_blocks_per_row, _TENCENT_Q1_0_BLOCK_BYTES
    )

    # Per native-block fp16 scales: shape (N, n_native_blocks_per_row).
    scales_raw = arr[:, :, :2].copy()
    native_scales = scales_raw.view(np.float16).reshape(ne1, n_native_blocks_per_row)

    # Unpack 2-bit codes (4 codes per byte, LSB-first within each byte).
    codes_packed = arr[:, :, 2:]  # (N, n_native_blocks, native_block_size/4 = 128)
    codes_2bit = np.empty(
        (ne1, n_native_blocks_per_row, TENCENT_Q1_0_NATIVE_BLOCK_SIZE),
        dtype=np.uint8,
    )
    for slot in range(4):
        codes_2bit[:, :, slot::4] = (codes_packed >> (2 * slot)) & np.uint8(0x3)

    # Inflate 2-bit codes to 4-bit slots so the result fits ORT's
    # MatMulNBits bits=4 kernel:
    #     2-bit code c ∈ {0,1,2,3} → 4-bit slot 2c ∈ {0,2,4,6}
    # Combined with zp=3, dequant gives scale·(2c - 3) ∈ {-3,-1,+1,+3}·scale
    # — exactly Tencent's SEQ-codebook dequantization.
    codes_4bit = (codes_2bit * 2).astype(np.uint8)  # ∈ {0,2,4,6}

    # Split each native 512-element block into _SUBBLOCKS_PER_NATIVE
    # sub-blocks of TENCENT_Q1_0_ORT_BLOCK_SIZE elements, then pack two
    # nibbles per byte for MatMulNBits:
    #     byte[j] = (code[2j+1] << 4) | code[2j]
    codes_sub = codes_4bit.reshape(
        ne1,
        n_native_blocks_per_row * _SUBBLOCKS_PER_NATIVE,
        TENCENT_Q1_0_ORT_BLOCK_SIZE,
    )
    pairs = codes_sub.reshape(ne1, codes_sub.shape[1], TENCENT_Q1_0_ORT_BLOCK_SIZE // 2, 2)
    weight = (pairs[..., 0] | (pairs[..., 1] << 4)).astype(np.uint8)
    # Shape: (N, n_ort_blocks, ort_block_size * bits / 8 = 64)

    n_ort_blocks_per_row = n_native_blocks_per_row * _SUBBLOCKS_PER_NATIVE

    # Replicate each native scale across its 4 sub-blocks so the
    # MatMulNBits per-block scales array matches the (N, n_ort_blocks)
    # shape with identical values within each native group.
    scales = np.repeat(native_scales, _SUBBLOCKS_PER_NATIVE, axis=1)

    # Zero-points: integer 3 packed at 4 bits each, two per byte.
    # byte = (3 << 4) | 3 = 0x33. Per output row: ceil(n_ort_blocks*4/8) bytes.
    zp_cols = math.ceil(n_ort_blocks_per_row * 4 / 8)
    zero_points = np.full((ne1, zp_cols), 0x33, dtype=np.uint8)
    # Odd block count: high nibble of the last byte is padding; clear it.
    if n_ort_blocks_per_row % 2 == 1:
        zero_points[:, -1] = 0x03

    return RepackedTensor(
        weight=weight,
        scales=scales,
        zero_points=zero_points,
        block_size=TENCENT_Q1_0_ORT_BLOCK_SIZE,
        bits=4,
    )
