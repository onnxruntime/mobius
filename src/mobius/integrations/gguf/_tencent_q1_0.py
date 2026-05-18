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
:class:`~mobius.integrations.gguf._repacker.RepackedTensor` that
preserves the native 2-bit codes and pairs them with a float
zero-point of ``1.5`` and an *effective* per-block scale of
``2·stored_scale``. The ORT ``MatMulNBits`` op then dequantizes via
``effective_scale · (B − 1.5) = stored_scale · {−3, −1, +1, +3}[c]``,
exactly Tencent's SEQ codebook.

ORT version requirement: this representation needs the
**float zero-point** path for ``bits=2`` that was added in
`microsoft/onnxruntime#28354
<https://github.com/microsoft/onnxruntime/pull/28354>`_ (merged
2026-05-13, expected in onnxruntime 1.27 and later). Older
versions throw
``Only 4b quantization is supported for unpacked compute using
non-MLAS de-quantization for now`` from
``ComputeBUnpacked`` at first inference.

Block-size note: the native Tencent block size is 512 elements, but
the ORT CPU dequant kernel silently returns zeros for ``block_size``
larger than 256 (see `microsoft/onnxruntime#28551
<https://github.com/microsoft/onnxruntime/issues/28551>`_). Each
Tencent 512-element scale is therefore replicated across 4
``MatMulNBits`` sub-blocks of 128 elements; the dequantized values
are bit-identical.
"""

from __future__ import annotations

__all__ = [
    "TENCENT_Q1_0_NATIVE_BLOCK_SIZE",
    "TENCENT_Q1_0_ORT_BLOCK_SIZE",
    "TENCENT_Q1_0_ORT_BITS",
    "TENCENT_Q1_0_ZERO_POINT",
    "is_tencent_q1_0_layout",
    "parse_tencent_q1_0_tensor",
]

import os

import numpy as np

from mobius.integrations.gguf._repacker import RepackedTensor

# Native Tencent block size (one fp16 scale governs this many weights).
TENCENT_Q1_0_NATIVE_BLOCK_SIZE = 512

# Block size we expose to ORT's MatMulNBits. The native 512 is rejected
# silently by the CPU kernel (returns zeros), so we replicate each
# native scale across 4 ORT sub-blocks of 128 elements.
TENCENT_Q1_0_ORT_BLOCK_SIZE = 128

# Bit width emitted to ORT. We keep Tencent's native 2 bits/elt rather
# than inflating to 4, halving the on-disk weight bytes. This relies on
# the bits=2 float-zero-point path landed in ORT PR #28354.
TENCENT_Q1_0_ORT_BITS = 2

# Float zero point that produces the SEQ codebook centred between
# integer codes 0..3:
#   stored_scale · {-3, -1, +1, +3}[c]
#     = (2 · stored_scale) · (c - 1.5)
TENCENT_Q1_0_ZERO_POINT = 1.5

# Per native block on-disk: 2 bytes (fp16 scale) + 512·2/8 = 128 bytes codes.
_TENCENT_Q1_0_BLOCK_BYTES = 2 + TENCENT_Q1_0_NATIVE_BLOCK_SIZE // 4  # 130

_SUBBLOCKS_PER_NATIVE = (
    TENCENT_Q1_0_NATIVE_BLOCK_SIZE // TENCENT_Q1_0_ORT_BLOCK_SIZE
)  # 4


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

    Args:
        file_path: Path to the source ``.gguf`` file.
        data_section_offset: Absolute byte offset where the GGUF data
            section begins (``GGUFReader.data_offset``).
        tensor: ``gguf.ReaderTensor`` for the target weight. Must have
            ``tensor_type == GGML_TYPE_Q1_0`` and a 2D shape.

    Returns:
        A :class:`RepackedTensor` with ``bits=2`` and
        ``block_size=128``. The 2-bit codes are passed through
        unchanged (ORT's packing matches Tencent's LSB-first 4-codes-
        per-byte layout); the per-block ``scales`` are
        ``2·stored_scale`` so that ``effective_scale · (B − 1.5)``
        matches the SEQ codebook ``stored_scale · {-3,-1,+1,+3}[B]``;
        and ``zero_points`` is a float16 tensor of all 1.5.

        Requires onnxruntime that includes the ``bits=2`` float
        zero-point CPU path (PR #28354, expected in 1.27+).
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

    # Per native-block fp16 scales. We expose 2·stored_scale to ORT so
    # that effective_scale·(B−1.5) = stored_scale·{-3,-1,+1,+3}[B]
    # matches Tencent's SEQ codebook exactly.
    scales_raw = arr[:, :, :2].copy()
    native_scales = scales_raw.view(np.float16).reshape(ne1, n_native_blocks_per_row)
    native_scales = (native_scales.astype(np.float32) * 2.0).astype(np.float16)

    # 2-bit codes are stored as 4 codes per byte, LSB-first within each
    # byte — ORT's MatMulNBits expects the same packing, so we copy the
    # bytes through unchanged. Reshape only: each 512-element native
    # block becomes _SUBBLOCKS_PER_NATIVE = 4 consecutive 128-element
    # ORT sub-blocks of 32 bytes each (128·2/8 = 32).
    codes_packed = arr[:, :, 2:]  # (N, n_native_blocks, 128)
    blob_size = TENCENT_Q1_0_ORT_BLOCK_SIZE * TENCENT_Q1_0_ORT_BITS // 8  # 32
    n_ort_blocks_per_row = n_native_blocks_per_row * _SUBBLOCKS_PER_NATIVE
    weight = codes_packed.reshape(ne1, n_ort_blocks_per_row, blob_size).copy()

    # Replicate each native scale across its 4 sub-blocks so the
    # per-block scales array matches (N, n_ort_blocks).
    scales = np.repeat(native_scales, _SUBBLOCKS_PER_NATIVE, axis=1)

    # Float zero point: one fp32 value per ORT block, all = 1.5.
    # ORT MatMulNBits<float> registers T3 = {uint8, float}, so float-zp
    # must be float32 for fp32 models. (For fp16 models we'd want fp16;
    # the GGUF builder currently uses fp32 weight application then casts,
    # so float32 is the right intermediate dtype.)
    zero_points = np.full(
        (ne1, n_ort_blocks_per_row),
        TENCENT_Q1_0_ZERO_POINT,
        dtype=np.float32,
    )

    return RepackedTensor(
        weight=weight,
        scales=scales,
        zero_points=zero_points,
        block_size=TENCENT_Q1_0_ORT_BLOCK_SIZE,
        bits=TENCENT_Q1_0_ORT_BITS,
    )
