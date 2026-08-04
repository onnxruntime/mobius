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
    Codes ``c in {0,1,2,3}`` dequantize via the SEQ codebook
    ``{-1.5, -0.5, 0.5, 1.5}`` scaled by ``2*scale`` (equivalently
    ``{-3, -1, +1, +3}*scale``). Each output row of ``K`` elements
    splits into ``K/512`` blocks, each with its own scale.

Because the on-disk row stride differs (~4x larger than mainline),
mainline llama.cpp refuses to load these files — every tensor after
the first Q1_0 entry lands at the wrong offset. gguf-py likewise
under-reads tensor data. We bypass the size calculation entirely by
reading from the **explicit per-tensor file offset** stored in the
GGUF header.

Two MatMulNBits representations are available, selected by the
``flags.tencent_q1_0_use_native_2bit`` flag:

* **Inflated bits=4** (default; ``flag=False``): each 2-bit code
  ``c in {0..3}`` is doubled to a 4-bit slot ``2c in {0,2,4,6}`` and
  paired with integer ``zero_point = 3``. Dequant: ``scale*(B - 3) =
  scale * {-3, -1, +1, +3}[c]``. Doubles on-disk weight bytes (4 bpw
  packed) but uses ORT's well-optimised bits=4 packed-uint8 kernel,
  giving ~20x higher CPU throughput than the native form.

* **Native bits=2** (``flag=True``): the 2-bit codes are copied
  through unchanged and paired with float ``zero_point = 1.5`` and
  ``effective_scale = 2*stored_scale``. Dequant:
  ``effective_scale*(B - 1.5) = stored_scale * {-3,-1,+1,+3}[c]``.
  Requires ORT >= 1.27 (`microsoft/onnxruntime#28354
  <https://github.com/microsoft/onnxruntime/pull/28354>`_), and the
  CPU path is currently a naive scalar fallback (`#28552
  <https://github.com/microsoft/onnxruntime/issues/28552>`_).
  Smaller and semantically faithful, but unusable for interactive
  decode until an MLAS fast path lands.

Block-size note: the native Tencent block size is 512 elements, but
the ORT CPU dequant kernel silently returns zeros for ``block_size``
larger than 256 (see `microsoft/onnxruntime#28551
<https://github.com/microsoft/onnxruntime/issues/28551>`_). Both
representations therefore expose ``block_size = 128`` to ORT by
replicating each native scale across 4 sub-blocks; dequantized
values are bit-identical to the native 512-element layout.
"""

from __future__ import annotations

__all__ = [
    "TENCENT_Q1_0_NATIVE_BLOCK_SIZE",
    "TENCENT_Q1_0_ORT_BLOCK_SIZE",
    "TENCENT_Q1_0_NATIVE_ZERO_POINT",
    "is_tencent_q1_0_layout",
    "parse_tencent_q1_0_tensor",
]

import math
import os

import numpy as np

from mobius._flags import flags
from mobius.integrations.gguf._repacker import RepackedTensor

# Native Tencent block size (one fp16 scale governs this many weights).
TENCENT_Q1_0_NATIVE_BLOCK_SIZE = 512

# Block size we expose to ORT's MatMulNBits. The native 512 is rejected
# silently by the CPU kernel (returns zeros), so we replicate each
# native scale across 4 ORT sub-blocks of 128 elements.
TENCENT_Q1_0_ORT_BLOCK_SIZE = 128

# Float zero point that produces the SEQ codebook centred between
# integer codes 0..3 in the native bits=2 representation:
#   stored_scale * {-3, -1, +1, +3}[c] = (2 * stored_scale) * (c - 1.5)
TENCENT_Q1_0_NATIVE_ZERO_POINT = 1.5

# Per native block on-disk: 2 bytes (fp16 scale) + 512*2/8 = 128 bytes codes.
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
        key=_tensor_data_offset,
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

    Dispatches between two MatMulNBits representations based on
    :attr:`flags.tencent_q1_0_use_native_2bit`:

    * ``False`` (default): :func:`_pack_inflated_4bit` — codes inflated
      to 4-bit slots, integer ``zp=3``. Fast on CPU EP but 2x the
      on-disk weight bytes.
    * ``True``: :func:`_pack_native_2bit` — codes passed through,
      float ``zp=1.5``. Native 2 bpw but slow on CPU EP today.

    Args:
        file_path: Path to the source ``.gguf`` file.
        data_section_offset: Absolute byte offset where the GGUF data
            section begins (``GGUFReader.data_offset``).
        tensor: ``gguf.ReaderTensor`` for the target weight. Must have
            ``tensor_type == GGML_TYPE_Q1_0`` and a 2D shape.

    Returns:
        A :class:`RepackedTensor` whose ``block_size`` is always 128.
        The ``bits`` field is 2 or 4 depending on the flag.
    """
    native_scales, codes_2bit, ne1, n_native = _read_tencent_blocks(
        file_path, data_section_offset, tensor
    )
    if flags.tencent_q1_0_use_native_2bit:
        return _pack_native_2bit(native_scales, codes_2bit, ne1, n_native)
    return _pack_inflated_4bit(native_scales, codes_2bit, ne1, n_native)


def _read_tencent_blocks(
    file_path: str | os.PathLike,
    data_section_offset: int,
    tensor,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Read raw Tencent Q1_0 blocks from disk and unpack to (scales, codes).

    Returns:
        ``(native_scales, codes_2bit, ne1, n_native_blocks_per_row)`` where
        ``native_scales`` is ``(N, n_native)`` float16 (the unmodified
        on-disk per-block fp16 scale) and ``codes_2bit`` is
        ``(N, n_native * 512 / 4)`` uint8 of the raw packed code bytes
        (4 codes/byte, LSB-first within each byte).
    """
    ne0 = int(tensor.shape[0])  # K
    ne1 = int(tensor.shape[1])  # N
    if ne0 % TENCENT_Q1_0_NATIVE_BLOCK_SIZE != 0:
        raise ValueError(
            f"Tensor {tensor.name!r} has K={ne0} not divisible by "
            f"Tencent Q1_0 native block size {TENCENT_Q1_0_NATIVE_BLOCK_SIZE}"
        )

    n_native = ne0 // TENCENT_Q1_0_NATIVE_BLOCK_SIZE
    bytes_per_row = n_native * _TENCENT_Q1_0_BLOCK_BYTES
    total_bytes = ne1 * bytes_per_row

    abs_offset = data_section_offset + _tensor_data_offset(tensor)
    with open(file_path, "rb") as f:
        f.seek(abs_offset)
        blob = f.read(total_bytes)
    if len(blob) != total_bytes:
        raise OSError(
            f"Short read for {tensor.name!r}: got {len(blob)} bytes, expected {total_bytes}"
        )

    arr = np.frombuffer(blob, dtype=np.uint8).reshape(ne1, n_native, _TENCENT_Q1_0_BLOCK_BYTES)

    # First 2 bytes of each block: fp16 stored_scale.
    scales_raw = arr[:, :, :2].copy()
    native_scales = scales_raw.view(np.float16).reshape(ne1, n_native)

    # Remaining bytes: 2-bit codes packed 4 per byte, LSB-first.
    codes_2bit = arr[:, :, 2:].reshape(ne1, n_native, TENCENT_Q1_0_NATIVE_BLOCK_SIZE // 4)
    return native_scales, codes_2bit, ne1, n_native


def _pack_native_2bit(
    native_scales: np.ndarray,
    codes_2bit: np.ndarray,
    ne1: int,
    n_native: int,
) -> RepackedTensor:
    """Pack Tencent codes as native ``MatMulNBits bits=2`` + float zp=1.5.

    The on-disk 2-bit codes match ORT's bit-packing exactly, so the
    bytes are copied through unchanged. We expose
    ``effective_scale = 2 * stored_scale`` so that the
    ``effective_scale * (B - 1.5)`` dequant formula produces
    ``stored_scale * {-3, -1, +1, +3}[B]``.

    Requires onnxruntime that includes the ``bits=2`` float zero-point
    CPU path (PR #28354, expected in 1.27+). See
    `microsoft/onnxruntime#28552
    <https://github.com/microsoft/onnxruntime/issues/28552>`_ for the
    CPU performance limitation that motivates the default-off
    ``flags.tencent_q1_0_use_native_2bit``.
    """
    bits = 2
    blob_size = TENCENT_Q1_0_ORT_BLOCK_SIZE * bits // 8  # 32
    n_ort_blocks_per_row = n_native * _SUBBLOCKS_PER_NATIVE

    # Bytes pass through; reshape splits each 512-elt native block into
    # 4 consecutive 128-elt sub-blocks of 32 bytes each.
    weight = codes_2bit.reshape(ne1, n_ort_blocks_per_row, blob_size).copy()

    # effective_scale = 2 * stored_scale, replicated across the 4 sub-blocks.
    effective_scales = (native_scales.astype(np.float32) * 2.0).astype(np.float16)
    scales = np.repeat(effective_scales, _SUBBLOCKS_PER_NATIVE, axis=1)

    # ORT MatMulNBits<float> registers T3 = {uint8, float}; use fp32.
    zero_points = np.full(
        (ne1, n_ort_blocks_per_row),
        TENCENT_Q1_0_NATIVE_ZERO_POINT,
        dtype=np.float32,
    )

    return RepackedTensor(
        weight=weight,
        scales=scales,
        zero_points=zero_points,
        block_size=TENCENT_Q1_0_ORT_BLOCK_SIZE,
        bits=bits,
    )


def _pack_inflated_4bit(
    native_scales: np.ndarray,
    codes_2bit: np.ndarray,
    ne1: int,
    n_native: int,
) -> RepackedTensor:
    """Pack Tencent codes as ``MatMulNBits bits=4`` + integer zp=3.

    Inflates each 2-bit code ``c in {0..3}`` to a 4-bit slot ``2c in
    {0,2,4,6}``. With integer ``zp=3``, the dequant formula
    ``scale*(B - 3)`` produces ``scale * {-3, -1, +1, +3}[c]`` — the
    same SEQ codebook as the native form. ``scales`` carry the
    unmodified ``stored_scale`` (no factor-of-2 fold-in: the codebook
    offset of ``3`` between slots already encodes the SEQ ``2x`` step).

    Doubles on-disk weight bytes vs the native form but exercises
    ORT's well-optimised bits=4 packed-uint8 path. This is the default
    until ORT's bits=2 + float-zp kernel is optimised (see
    `microsoft/onnxruntime#28552
    <https://github.com/microsoft/onnxruntime/issues/28552>`_).
    """
    bits = 4
    n_ort_blocks_per_row = n_native * _SUBBLOCKS_PER_NATIVE

    # Unpack 2-bit codes -> 4 codes per byte, LSB-first slot k holds code k.
    codes = np.empty(
        (ne1, n_native, TENCENT_Q1_0_NATIVE_BLOCK_SIZE),
        dtype=np.uint8,
    )
    for slot in range(4):
        codes[:, :, slot::4] = (codes_2bit >> (2 * slot)) & np.uint8(0x3)

    # Inflate c -> 2c so dequant scale*(2c - 3) = scale*{-3,-1,+1,+3}[c].
    codes_inflated = (codes * 2).astype(np.uint8)

    # Split each native 512-elt block into 4 sub-blocks of 128 elements,
    # then pack two nibbles per byte: byte[j] = (code[2j+1] << 4) | code[2j].
    sub = codes_inflated.reshape(ne1, n_ort_blocks_per_row, TENCENT_Q1_0_ORT_BLOCK_SIZE)
    pairs = sub.reshape(ne1, n_ort_blocks_per_row, TENCENT_Q1_0_ORT_BLOCK_SIZE // 2, 2)
    weight = (pairs[..., 0] | (pairs[..., 1] << 4)).astype(np.uint8)

    scales = np.repeat(native_scales, _SUBBLOCKS_PER_NATIVE, axis=1)

    # Integer zp=3 packed 2 per byte: (3 << 4) | 3 = 0x33.
    zp_cols = math.ceil(n_ort_blocks_per_row * bits / 8)
    zero_points = np.full((ne1, zp_cols), 0x33, dtype=np.uint8)
    if (
        n_ort_blocks_per_row % 2 == 1
    ):  # pragma: no cover — n_ort_blocks always even (4*n_native)
        zero_points[:, -1] = 0x03

    return RepackedTensor(
        weight=weight,
        scales=scales,
        zero_points=zero_points,
        block_size=TENCENT_Q1_0_ORT_BLOCK_SIZE,
        bits=bits,
    )
