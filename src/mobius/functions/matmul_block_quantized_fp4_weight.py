# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standard-ONNX Function body for NVFP4 W4A16 weight-only MatMul.

The custom op ABI is pinned to the four required inputs used by Microsoft's
reference graph:

``MatMulBlockQuantizedFp4Weight(A, B, weight_scale, weight_scale_2)``

``B`` and ``weight_scale`` remain raw UINT8 payloads. The function unpacks
low-nibble-first E2M1 codes, decodes raw E4M3 scale bytes through a lookup
table, applies per-16 and global scales, then computes ``A @ W.T``. Bias stays
outside the function so the body remains faithful to the exact four-input ABI.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import onnx_ir as ir
from onnxscript._internal.builder import build_function

from mobius._constants import OPSET_VERSION

DOMAIN = "com.microsoft"
OP_TYPE = "MatMulBlockQuantizedFp4Weight"
BLOCK_SIZE = 16

_E2M1_TABLE = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)
_E4M3_TABLE = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)


def matmul_block_quantized_fp4_weight() -> ir.Function:
    """Build the exact four-input standard-ONNX NVFP4 fallback."""

    def body(op, a_input, b_packed, raw_scales, global_scale):
        low_mask = op.Constant(value=ir.tensor(np.uint8(0x0F)))
        shift = op.Constant(value=ir.tensor(np.uint8(4)))
        unpack_axis = op.Constant(value=ir.tensor(np.array([2], dtype=np.int64)))
        shape_2d = op.Constant(value=ir.tensor(np.array([0, -1], dtype=np.int64)))

        # Packed B is [N, K/2]. Interleave low then high nibbles to [N, K].
        low = op.BitwiseAnd(b_packed, low_mask)
        high = op.BitShift(b_packed, shift, direction="RIGHT")
        low = op.Unsqueeze(low, unpack_axis)
        high = op.Unsqueeze(high, unpack_axis)
        codes = op.Reshape(op.Concat(low, high, axis=2), shape_2d)
        code_indices = op.Cast(codes, to=ir.DataType.INT64)
        e2m1_table = op.Constant(value=ir.tensor(_E2M1_TABLE))
        fp4_values = op.Gather(e2m1_table, code_indices, axis=0)

        # Raw E4M3 bytes are decoded without numeric conversion or bit loss.
        scale_indices = op.Cast(raw_scales, to=ir.DataType.INT64)
        e4m3_table = op.Constant(value=ir.tensor(_E4M3_TABLE))
        block_scales = op.Gather(e4m3_table, scale_indices, axis=0)

        scale_axis = op.Constant(value=ir.tensor(np.array([2], dtype=np.int64)))
        block_scales = op.Unsqueeze(block_scales, scale_axis)
        scale_shape = op.Shape(block_scales)
        expanded_shape = op.Concat(
            op.Slice(
                scale_shape,
                op.Constant(value=ir.tensor(np.array([0], dtype=np.int64))),
                op.Constant(value=ir.tensor(np.array([2], dtype=np.int64))),
            ),
            op.Constant(value=ir.tensor(np.array([BLOCK_SIZE], dtype=np.int64))),
            axis=0,
        )
        block_scales = op.Expand(block_scales, expanded_shape)
        block_scales = op.Reshape(block_scales, shape_2d)

        weight = op.Mul(fp4_values, block_scales)
        weight = op.Mul(weight, global_scale)
        weight = op.CastLike(weight, a_input)
        result = op.MatMul(a_input, op.Transpose(weight, perm=[1, 0]))
        result.name = "Y"
        return result

    return build_function(
        body,
        [
            ir.Value(name="A"),
            ir.Value(name="B"),
            ir.Value(name="weight_scale"),
            ir.Value(name="weight_scale_2"),
        ],
        domain=DOMAIN,
        name=OP_TYPE,
        attributes=[ir.Attr("block_size", ir.AttributeType.INT, BLOCK_SIZE)],
        opset_imports={"": OPSET_VERSION},
    )


__all__ = [
    "BLOCK_SIZE",
    "DOMAIN",
    "OP_TYPE",
    "matmul_block_quantized_fp4_weight",
]
