# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite rule for casting INT64 Gather indices to INT32 for WebGPU.

WebGPU shaders do not natively support 64-bit integer types.  ORT's WebGPU
execution provider requires that ``Gather`` index inputs be ``INT32`` rather
than ``INT64``.

This rule inserts ``Cast(indices, to=INT32)`` before any ``Gather`` node
whose index input has dtype ``INT64``.  The transformation is always safe
for embedding-table lookups because token IDs, position IDs, and vocabulary
indices are small non-negative integers that fit comfortably in ``INT32``
(max value ~2 billion vs. typical vocab sizes of 32K-256K).

.. code-block:: text

    # Original — Gather with INT64 indices:
    out = Gather(table, int64_indices, axis=0)

    # Replacement — INT64 indices cast to INT32:
    int32_indices = Cast(int64_indices, to=INT32)
    out = Gather(table, int32_indices, axis=0)

The ``axis`` attribute from the original ``Gather`` node is preserved.

This rule is applied as part of the WebGPU lowering pass in
``_builder._get_optimization_passes``.

Example usage::

    from mobius.rewrite_rules import cast_int64_to_int32_rules
    from onnxscript.rewriter import rewrite

    model = build("Qwen/Qwen3-0.6B", execution_provider="webgpu")
    rewrite(model, pattern_rewrite_rules=cast_int64_to_int32_rules())
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class CastGatherIndicesToInt32(RewriteRuleClassBase):
    """Insert ``Cast(to=INT32)`` before INT64 ``Gather`` index inputs.

    **Matched pattern:**

    .. code-block:: text

        out = Gather(table, indices, axis=N)   # indices has dtype INT64

    **Replacement:**

    .. code-block:: text

        indices_i32 = Cast(indices, to=INT32)
        out = Gather(table, indices_i32, axis=N)

    The data input and all attributes are preserved unchanged.
    """

    def pattern(self, op, data, indices):
        return op.Gather(data, indices, _allow_other_attributes=True, _outputs=["gather_out"])

    def check(self, context, data, indices, gather_out, **_):
        result = MatchResult()

        if indices.dtype != ir.DataType.INT64:
            return result.fail(
                f"Gather indices dtype is {indices.dtype}, not INT64; no cast needed"
            )

        return result

    def rewrite(self, op, data, indices, gather_out, **_):
        gather_node = gather_out.producer()
        axis = gather_node.attributes.get_int("axis", 0)

        # Cast INT64 indices → INT32. Safe for all typical index ranges
        # (vocab IDs, position IDs, sequence indices all fit in INT32).
        indices_i32 = op.Cast(indices, to=int(ir.DataType.INT32))

        return op.Gather(data, indices_i32, axis=axis)


def cast_int64_to_int32_rules() -> RewriteRuleSet:
    """Return rules that cast INT64 Gather indices to INT32 for WebGPU.

    ORT's WebGPU execution provider requires ``Gather`` index inputs to be
    ``INT32``.  This rule inserts ``Cast(to=INT32)`` before any ``Gather``
    node whose index input has dtype ``INT64``.

    The transformation is safe for all token-embedding, position-embedding,
    and vocabulary lookups because the relevant index values are small
    non-negative integers well within INT32 range.

    Returns:
        :class:`RewriteRuleSet` containing the INT64→INT32 cast rule for Gather.
    """
    return RewriteRuleSet([CastGatherIndicesToInt32().rule()])
