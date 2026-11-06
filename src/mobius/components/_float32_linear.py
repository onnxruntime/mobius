# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dense floating-point projections with bias before the native output cast."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder

from mobius.components._common import Linear


class Float32Linear(Linear):
    """Compute ``x @ weight.T + bias`` in float32, then restore ``x``'s dtype.

    Unlike a native low-precision MatMul followed by Add, this preserves the
    float32 accumulator through the bias epilogue before rounding to float16 or
    bfloat16. Leading input dimensions and checkpoint parameter names are the
    same as :class:`Linear`. This is a dense floating-point component, not a
    replacement for a packed/quantized projection. Intermediate activation
    types may remain unresolved until graph-wide inference; CastLike preserves
    the original activation dtype once it is inferred.
    """

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        supported = (ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16)
        if (
            (x.dtype is not None and x.dtype not in supported)
            or self.weight.dtype not in supported
            or (self.bias is not None and self.bias.dtype not in supported)
        ):
            raise ValueError("Float32Linear requires float32, float16 or bfloat16 tensors")
        weight = op.Cast(self.weight, to=ir.DataType.FLOAT)
        result = op.MatMul(
            op.Cast(x, to=ir.DataType.FLOAT),
            op.Transpose(weight, perm=[1, 0]),
        )
        if self.bias is not None:
            result = op.Add(result, op.Cast(self.bias, to=ir.DataType.FLOAT))
        # A separate native MatMul output would round before adding the bias.
        return op.CastLike(result, x)
