# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite rule for eliminating Shape ops on attention masks for WebGPU.

WebGPU graph capture does not support the ONNX ``Shape`` operator for
dynamic dimension extraction at runtime.  The attention-mask sequence-length
extraction pattern ``Shape(attention_mask, start=1, end=2)`` can be replaced
with an equivalent data-driven computation:

.. code-block:: text

    # Original — not supported on WebGPU:
    total_seq_len = Shape(attention_mask, start=1, end=2)  # INT64[1]

    # Replacement — WebGPU safe:
    counts = ReduceSum(attention_mask, axes=[1], keepdims=False)  # INT64[batch]
    total_seq_len = ReduceMax(counts, axes=[0], keepdims=True)    # INT64[1]

This works because ``attention_mask`` is a 0/1 indicator tensor where each
row sums to the number of valid tokens in that batch item.  ``ReduceMax``
picks the maximum across the batch, which equals the total sequence length
for non-padded inputs (the typical case in generation).

The rule is intentionally conservative — it only fires when the input value:

1. Has dtype INT64 (attention masks are always INT64 in mobius)
2. Is a direct graph input, not an intermediate result
3. Is 2D (batch x sequence)
4. Has a name containing ``"mask"`` (e.g. ``attention_mask``,
   ``encoder_attention_mask``) to avoid replacing shape extraction on
   ``input_ids``, where ``ReduceSum`` would produce vocabulary-ID sums
   rather than sequence lengths.

This rule is applied as part of the WebGPU lowering pass in
``_builder._get_optimization_passes``.

Example usage::

    from mobius.rewrite_rules import eliminate_shape_rules
    from onnxscript.rewriter import rewrite

    model = build("Qwen/Qwen3-0.6B", execution_provider="webgpu")
    rewrite(model, pattern_rewrite_rules=eliminate_shape_rules())
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class ShapeToReduceSumMax(RewriteRuleClassBase):
    """Replace ``Shape(attention_mask, start=1, end=2)`` with ``ReduceMax(ReduceSum(…))``.

    **Matched pattern:**

    .. code-block:: text

        total_seq_len = Shape(attention_mask, start=1, end=2)

    **Replacement:**

    .. code-block:: text

        counts        = ReduceSum(attention_mask, axes=[1], keepdims=False)
        total_seq_len = ReduceMax(counts, axes=[0], keepdims=True)

    The output type is ``INT64[1]``, matching ``Shape``'s output exactly.
    """

    def pattern(self, op, x):
        return op.Shape(x, _allow_other_attributes=True, _outputs=["shape_out"])

    def check(self, context, x, shape_out, **_):
        result = MatchResult()

        shape_node = shape_out.producer()

        # Must be extracting dimension 1 (the sequence/token dimension)
        start = shape_node.attributes.get_int("start", 0)
        end = shape_node.attributes.get_int("end", -1)
        if start != 1 or end != 2:
            return result.fail(
                f"Shape start/end is {start}/{end}, expected 1/2 "
                "(only sequence-dimension extraction is replaced)"
            )

        # Must be a graph input, not an intermediate value
        if x.producer() is not None:
            return result.fail("Input is an intermediate value, not a graph input")

        # Must be INT64 (attention masks are INT64; embedding tables are not)
        if x.dtype != ir.DataType.INT64:
            return result.fail(f"Input dtype is {x.dtype}, expected INT64")

        # Must be 2D [batch, sequence]
        if x.shape is None or len(x.shape) != 2:
            return result.fail("Input is not 2D (expected [batch, sequence])")

        # Must have "mask" in the name to distinguish attention_mask from input_ids.
        # input_ids has the same shape and dtype but ReduceSum on vocab indices
        # gives the sum of token IDs, not the sequence length.
        if x.name is None or "mask" not in x.name:
            return result.fail(
                f"Input name {x.name!r} does not contain 'mask'; "
                "ReduceSum is only valid for 0/1 indicator tensors"
            )

        return result

    def rewrite(self, op, x, **_):
        # ReduceSum along axis 1: [batch, seq] → [batch]
        # Each entry is the number of valid tokens in that batch row.
        axis_1 = op.Constant(value_ints=[1])
        counts = op.ReduceSum(x, axis_1, keepdims=0)

        # ReduceMax with keepdims=True: [batch] → [1]
        # The maximum row sum equals the total sequence length for non-padded inputs.
        axis_0 = op.Constant(value_ints=[0])
        total_seq_len = op.ReduceMax(counts, axis_0, keepdims=1)

        return total_seq_len


class ShapeGatherToReduceSumMax(RewriteRuleClassBase):
    """Replace ``Gather(Shape(attention_mask), 1)`` with ``ReduceMax(ReduceSum(…))``.

    Some model variants extract the sequence-length dimension using the older
    two-step pattern ``Gather(Shape(x), indices=1, axis=0)`` rather than the
    direct ``Shape(x, start=1, end=2)`` form.  Both produce a scalar INT64
    value equal to the size of dimension 1.

    **Matched pattern:**

    .. code-block:: text

        shape_val     = Shape(attention_mask)
        total_seq_len = Gather(shape_val, indices=1, axis=0)

    **Replacement:**

    .. code-block:: text

        counts        = ReduceSum(attention_mask, axes=[1], keepdims=False)
        total_seq_len = ReduceMax(counts, axes=[0], keepdims=False)

    The output type is a scalar ``INT64`` value, matching ``Gather``'s output.
    """

    def pattern(self, op, x, indices):
        shape_val = op.Shape(x, _outputs=["shape_val"])
        return op.Gather(
            shape_val, indices, _allow_other_attributes=True, _outputs=["gather_out"]
        )

    def check(self, context, x, indices, shape_val, gather_out, **_):
        result = MatchResult()

        # The Gather must extract dimension index 1 (sequence dimension)
        if indices.const_value is None:
            return result.fail("Gather indices are not a constant")
        idx_val = int(indices.const_value.numpy().flat[0])
        if idx_val != 1:
            return result.fail(
                f"Gather index is {idx_val}, expected 1 (sequence-length dimension)"
            )

        # Gather axis must be 0 (indexing into the shape vector)
        gather_node = gather_out.producer()
        axis = gather_node.attributes.get_int("axis", 0)
        if axis != 0:
            return result.fail(f"Gather axis is {axis}, expected 0")

        # The Shape input must be a 2D INT64 graph input with "mask" in its name
        if x.producer() is not None:
            return result.fail("Shape input is not a graph input")
        if x.dtype != ir.DataType.INT64:
            return result.fail(f"Shape input dtype is {x.dtype}, expected INT64")
        if x.shape is not None and len(x.shape) != 2:
            return result.fail("Shape input is not 2D (expected [batch, sequence])")
        if x.name is None or "mask" not in x.name:
            return result.fail(
                f"Shape input name {x.name!r} does not contain 'mask'; "
                "ReduceSum is only valid for 0/1 indicator tensors"
            )

        return result

    def rewrite(self, op, x, **_):
        # ReduceSum along axis 1: [batch, seq] → [batch]
        axis_1 = op.Constant(value_ints=[1])
        counts = op.ReduceSum(x, axis_1, keepdims=0)
        # ReduceMax with keepdims=False: [batch] → scalar (matching Gather output shape)
        axis_0 = op.Constant(value_ints=[0])
        return op.ReduceMax(counts, axis_0, keepdims=0)


def eliminate_shape_rules() -> RewriteRuleSet:
    """Return rules that replace Shape ops on attention masks with ReduceSum+ReduceMax.

    On WebGPU, dynamic ``Shape`` ops are not supported for graph capture.
    Two patterns are handled:

    1. ``Shape(attention_mask, start=1, end=2)`` — direct dimension extraction
       (used by larger models with many layers).
    2. ``Gather(Shape(attention_mask), indices=1)`` — two-step extraction
       (used by smaller models with fewer layers).

    Both produce the total sequence length as an INT64 value, replaced by an
    equivalent ``ReduceSum`` + ``ReduceMax`` computation.

    Only fires on 2D INT64 graph inputs whose name contains ``"mask"``
    (e.g. ``attention_mask``).  Shape extraction on ``input_ids`` and other
    non-indicator tensors is intentionally excluded.

    Returns:
        :class:`RewriteRuleSet` containing both shape-elimination rules.
    """
    return RewriteRuleSet([ShapeToReduceSumMax().rule(), ShapeGatherToReduceSumMax().rule()])
