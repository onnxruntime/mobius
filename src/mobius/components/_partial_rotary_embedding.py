# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Explicit FP32 partial rotary embedding with separate products and narrowing."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder


def apply_partial_rotary_pos_emb(
    op: OpBuilder,
    x: ir.Value,
    position_embeddings: tuple[ir.Value, ir.Value],
    *,
    num_heads: int,
    head_dim: int,
    rotary_embedding_dim: int,
    interleaved: bool = False,
) -> ir.Value:
    """Rotate a partial head with separate FP32 products, then narrow to ``x``.

    ``x`` is [B,T,H*D]; cosine/sine are FP32 [B,T,R/2], broadcast over
    heads. The unrotated tail is preserved. Unlike ``RotaryEmbedding``,
    this spells out the arithmetic boundaries; runtime fusion controls
    still determine whether a backend preserves those boundaries.
    """
    width = rotary_embedding_dim
    if num_heads < 1 or head_dim < 1 or not 0 < width <= head_dim or width % 2:
        raise ValueError("Partial RoPE requires positive heads and an even width <= head_dim")
    shaped = op.Reshape(x, [0, 0, num_heads, head_dim])
    wide = op.Cast(shaped, to=ir.DataType.FLOAT)
    cosine, sine = (op.Unsqueeze(value, [-2]) for value in position_embeddings)
    if interleaved:
        first = op.Slice(wide, [0], [width], [-1], [2])
        second = op.Slice(wide, [1], [width], [-1], [2])
    else:
        first = op.Slice(wide, [0], [width // 2], [-1])
        second = op.Slice(wide, [width // 2], [width], [-1])
    # Separate products are intentional: a fused multiply-add can round differently.
    rotated_first = op.Add(op.Mul(first, cosine), op.Mul(op.Neg(second), sine))
    rotated_second = op.Add(op.Mul(second, cosine), op.Mul(first, sine))
    if interleaved:
        pairs = op.Concat(
            op.Unsqueeze(rotated_first, [-1]),
            op.Unsqueeze(rotated_second, [-1]),
            axis=-1,
        )
        prefix = op.Reshape(pairs, [0, 0, num_heads, width])
    else:
        prefix = op.Concat(rotated_first, rotated_second, axis=-1)
    rounded = op.CastLike(prefix, x)
    tail = op.Slice(shaped, [width], [head_dim], [-1])
    output = op.Reshape(op.Concat(rounded, tail, axis=-1), op.Shape(x))
    output.type, output.shape = x.type, x.shape
    return output
