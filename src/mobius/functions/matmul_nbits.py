# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standard-ONNX ir.Function body for ``com.microsoft::MatMulNBits`` (4-bit).

Provides a portable QDQ decomposition for the blockwise-INT4 ``MatMulNBits``
contrib op. When registered in ``model.functions``,
:class:`onnx_ir.passes.common.InlinePass` expands it for runtimes that lack a
native ``MatMulNBits`` kernel but can consume standard QDQ weights — notably
the Qualcomm Hexagon HTP (QNN EP), whose partitioner otherwise rejects every
``MatMulNBits`` node and forces the quantized MatMuls onto CPU.

**Op semantics (4-bit):** the weight ``B`` is stored packed as
``(N, n_blocks, block_size/2)`` ``uint8`` — two 4-bit values per byte, low
nibble first. ``scales`` is ``(N, n_blocks)`` and the optional ``zero_points``
is ``(N, ceil(n_blocks/2))`` ``uint8`` (also 4-bit packed along the block axis).
Element ``(n, k)`` dequantizes as ``(w[n,k] - zp[n, k//block]) * scale[n, k//block]``
and ``Y = A @ dequant(B)ᵀ``.

**Function body (all shape-generic; folds to a compact uint4 QDQ graph once the
weight initializer is loaded and ORT runs constant folding):**

.. code-block:: text

    lo   = BitwiseAnd(B, 0x0F)                 # (N, nb, blob)  low nibbles
    hi   = BitShift(B, 4, RIGHT)               # (N, nb, blob)  high nibbles
    w    = Reshape(Concat(lo[...,None], hi[...,None], axis=-1), (N, -1))  # (N, K)
    w4   = Cast(w, uint4)
    zp   = <unpack zero_points nibbles, slice to nb, Cast uint4>   # (N, nb)
    wf   = DequantizeLinear(w4, scales, zp4, axis=1, block_size=<forwarded>)
    Y    = MatMul(A, Transpose(wf))            # (..., N)

Only the 4-bit form is emitted by mobius's quantized builds (GGUF Q4_K, Olive
INT4), so this body targets ``bits=4`` with ``zero_points`` present. It is
verified numerically identical to the native ``MatMulNBits`` op.

Attributes:
    block_size (int): Elements per quantization block along K.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript._internal.builder import build_function

from mobius._constants import OPSET_VERSION

DOMAIN = "com.microsoft"


def matmul_nbits() -> ir.Function:
    """Build an ``ir.Function`` for the 4-bit ``com.microsoft::MatMulNBits`` op.

    Inputs:
        A:            (..., K) activation.
        B:            (N, n_blocks, block_size/2) uint8 packed 4-bit weight.
        scales:       (N, n_blocks) dequant scales.
        zero_points:  (N, ceil(n_blocks/2)) uint8 packed 4-bit zero points.

    Output:
        Y: (..., N)

    Attrs:
        block_size (int): forwarded to ``DequantizeLinear``.
    """

    def _unpack_nibbles(op, packed, interleave_axis):
        """Split each uint8 into its two 4-bit nibbles along a new last axis.

        ``packed`` -> low nibble then high nibble interleaved, so the returned
        tensor has one extra trailing element per input element (low, high).
        """
        c_low = op.Constant(value=ir.tensor(np.uint8(0x0F)))
        c_shift = op.Constant(value=ir.tensor(np.uint8(4)))
        lo = op.BitwiseAnd(packed, c_low)
        hi = op.BitShift(packed, c_shift, direction="RIGHT")
        axis_c = op.Constant(value=ir.tensor(np.array([interleave_axis], dtype=np.int64)))
        lo_e = op.Unsqueeze(lo, axis_c)
        hi_e = op.Unsqueeze(hi, axis_c)
        return op.Concat(lo_e, hi_e, axis=interleave_axis)

    def body(op, a_input, b_packed, scales_input, zero_points_input):
        # --- Unpack the packed 4-bit weight to uint4 (N, K) ---
        # B: (N, nb, blob) -> interleave nibbles on a new axis 3 -> (N, nb, blob, 2)
        w_inter = _unpack_nibbles(op, b_packed, interleave_axis=3)
        # (N, nb, blob, 2) -> (N, nb, block) -> (N, K)
        keep_n_nb = op.Constant(value=ir.tensor(np.array([0, 0, -1], dtype=np.int64)))
        keep_n = op.Constant(value=ir.tensor(np.array([0, -1], dtype=np.int64)))
        w_blocks = op.Reshape(w_inter, keep_n_nb)
        w_2d = op.Reshape(w_blocks, keep_n)
        w4 = op.Cast(w_2d, to=ir.DataType.UINT4)

        # --- Unpack the packed 4-bit zero points to uint4 (N, nb) ---
        # zero_points: (N, zpacked) -> interleave nibbles on axis 2 -> (N, zpacked, 2)
        zp_inter = _unpack_nibbles(op, zero_points_input, interleave_axis=2)
        zp_2d = op.Reshape(zp_inter, keep_n)  # (N, 2*zpacked) >= (N, nb)
        # Slice to exactly nb columns (nb = scales.shape[1]); handles odd nb.
        nb_end = op.Slice(
            op.Shape(scales_input),
            op.Constant(value=ir.tensor(np.array([1], dtype=np.int64))),
            op.Constant(value=ir.tensor(np.array([2], dtype=np.int64))),
        )
        zp_sliced = op.Slice(
            zp_2d,
            op.Constant(value=ir.tensor(np.array([0], dtype=np.int64))),
            nb_end,
            op.Constant(value=ir.tensor(np.array([1], dtype=np.int64))),
        )
        zp4 = op.Cast(zp_sliced, to=ir.DataType.UINT4)

        # --- Blocked dequantize + MatMul ---
        # block_size is forwarded from the MatMulNBits call site at inline time.
        block_size_attr = ir.Attr(
            "block_size",
            ir.AttributeType.INT,
            32,
            ref_attr_name="block_size",
        )
        weight_f = op.DequantizeLinear(
            w4, scales_input, zp4, axis=1, block_size=block_size_attr
        )  # (N, K) in scales dtype
        weight_t = op.Transpose(weight_f, perm=[1, 0])  # (K, N)
        y = op.MatMul(a_input, weight_t)  # (..., N)
        y.name = "Y"
        return y

    return build_function(
        body,
        [
            ir.Value(name="A"),
            ir.Value(name="B"),
            ir.Value(name="scales"),
            ir.Value(name="zero_points"),
        ],
        domain=DOMAIN,
        name="MatMulNBits",
        attributes=[
            ir.Attr("block_size", ir.AttributeType.INT, 32),
        ],
        opset_imports={"": OPSET_VERSION},
    )
