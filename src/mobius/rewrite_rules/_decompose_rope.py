# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Decompose the opset-24 ``RotaryEmbedding`` op into primitive ONNX ops.

The Qualcomm Hexagon HTP (QNN EP) has no kernel for the opset-24
``RotaryEmbedding`` op, so any decoder that emits it is forced onto the CPU EP.
This rule rewrites ``RotaryEmbedding`` into the rotate-half primitives HTP
*does* support (``Reshape``/``Slice``/``Mul``/``Sub``/``Add``/``Concat``), so
the RoPE nodes can be claimed by the QNN partitioner.

**Matched op** (as mobius emits it via ``apply_rotary_pos_emb``):

.. code-block:: text

    y = RotaryEmbedding(x, cos, sin, num_heads=N, rotary_embedding_dim=0,
                        interleaved=0)

``x`` is rank-3 ``(B, S, N*H)``; ``cos``/``sin`` are the pre-gathered
per-position embeddings ``(B, S, H/2)``.

**Replacement** (non-interleaved, full rotation — verified numerically identical
to the op):

.. code-block:: text

    xr   = Reshape(x, (B, S, N, H))
    half = Shape(cos)[-1]                       # == H/2
    x1   = xr[..., :half]                       # first half
    x2   = xr[..., half:]                       # second half
    cosb = Unsqueeze(cos, 2)                    # (B, S, 1, H/2)
    sinb = Unsqueeze(sin, 2)
    out1 = x1*cosb - x2*sinb
    out2 = x2*cosb + x1*sinb
    y    = Reshape(Concat(out1, out2, axis=-1), (B, S, N*H))

Only the ``interleaved=0`` / full-rotation form mobius emits for gemma4-family
decoders is decomposed; interleaved or partial-rotary nodes are left unchanged
(they stay on the CPU EP, but no mobius QNN decoder currently emits them).

Applied automatically by :func:`~mobius._optimizations.optimize_model` for EPs
that lack a ``RotaryEmbedding`` kernel (``supports_rotary_embedding=False``;
QNN HTP).
"""

from __future__ import annotations

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class DecomposeRotaryEmbedding(RewriteRuleClassBase):
    """Lower a non-interleaved, full-rotation ``RotaryEmbedding`` to rotate-half."""

    def pattern(self, op, x, cos, sin):
        return op.RotaryEmbedding(
            x,
            cos,
            sin,
            _allow_other_attributes=True,
            _outputs=["rope_out"],
        )

    def check(self, context, x, cos, sin, rope_out, **_):
        result = MatchResult()
        node = rope_out.producer()
        if node is None or node.op_type != "RotaryEmbedding":
            return result.fail("not a RotaryEmbedding op")
        # Only the non-interleaved, full-rotation form is decomposed here.
        if node.attributes.get_int("interleaved", 0) != 0:
            return result.fail("interleaved RoPE not supported by this rule")
        if node.attributes.get_int("rotary_embedding_dim", 0) != 0:
            return result.fail("partial rotary_embedding_dim not supported")
        if node.attributes.get_int("num_heads", 0) <= 0:
            return result.fail("num_heads must be a positive attribute")
        # x must be rank-3 (B, S, N*H) as mobius emits.
        if x.shape is None or len(x.shape) != 3:
            return result.fail("expected rank-3 x (B, S, N*H)")
        return result

    def rewrite(self, op, x, cos, sin, rope_out, **_):
        node = rope_out.producer()
        num_heads = node.attributes.get_int("num_heads")

        # (B, S, N*H) -> (B, S, N, H). ``-1`` infers H (may be symbolic).
        shape_4d = op.Constant(value_ints=[0, 0, num_heads, -1])
        xr = op.Reshape(x, shape_4d)

        # half == H/2 == cos's last dim.  Slice x into its two halves on axis -1.
        half = op.Shape(cos, start=2, end=3)  # (1,) int64 == [H/2]
        zero = op.Constant(value_ints=[0])
        axis_last = op.Constant(value_ints=[-1])
        int_max = op.Constant(value_ints=[9223372036854775807])
        x1 = op.Slice(xr, zero, half, axis_last)  # (B, S, N, H/2)
        x2 = op.Slice(xr, half, int_max, axis_last)  # (B, S, N, H/2)

        # Broadcast cos/sin over the head axis: (B, S, H/2) -> (B, S, 1, H/2).
        two = op.Constant(value_ints=[2])
        cos_b = op.Unsqueeze(cos, two)
        sin_b = op.Unsqueeze(sin, two)

        # out1 = x1*cos - x2*sin ; out2 = x2*cos + x1*sin
        out1 = op.Sub(op.Mul(x1, cos_b), op.Mul(x2, sin_b))
        out2 = op.Add(op.Mul(x2, cos_b), op.Mul(x1, sin_b))
        out_r = op.Concat(out1, out2, axis=-1)  # (B, S, N, H)

        # Back to (B, S, N*H).
        flatten = op.Constant(value_ints=[0, 0, -1])
        return op.Reshape(out_r, flatten)


def decompose_rope_rules() -> RewriteRuleSet:
    """Return a rule set that decomposes ``RotaryEmbedding`` into rotate-half.

    Used for EPs without a ``RotaryEmbedding`` kernel
    (``supports_rotary_embedding=False``; QNN HTP). Numerically identical to the
    fused op for the non-interleaved / full-rotation form.
    """
    return RewriteRuleSet([DecomposeRotaryEmbedding().rule()])
