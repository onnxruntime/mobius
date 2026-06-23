# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Rewrite rule for lowering rank-4 RMSNormalization to rank-3.

Models with query/key normalization (e.g. Gemma-4, Qwen3) apply
``RMSNormalization`` over the head dimension of a rank-4 tensor shaped
``(batch, seq, heads, head_dim)`` — the norm reduces only the last axis.
The Qualcomm Hexagon HTP miscomputes RMSNormalization when the input is
rank-4, so the result is numerically wrong even though the graph finalizes;
3-D RMSNormalization (over the last axis of a rank-3 tensor) runs correctly.

This rule reshapes the rank-4 input down to rank-3
``(batch, seq*heads, head_dim)``, applies the identical RMSNormalization over
the last axis, and reshapes back. Both reshapes use constant shapes
(``[0, -1, head_dim]`` and ``[0, -1, heads, head_dim]``) so no ``Shape`` op is
introduced. The transform is numerically exact: the normalized axis
(``head_dim``) is unchanged.

These rules are applied automatically by
:func:`~mobius._optimizations.optimize_model` for EPs that miscompute rank-4
RMSNormalization (``supports_rank4_rmsnorm=False``; QNN). They can also be
applied manually::

    from mobius.rewrite_rules import reshape_rank4_rmsnorm_rules
    from onnxscript.rewriter import rewrite

    model = build("google/gemma-4-12B-it", execution_provider="qnn")
    rewrite(model, pattern_rewrite_rules=reshape_rank4_rmsnorm_rules())
"""

from __future__ import annotations

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class Rank4RMSNormToRank3(RewriteRuleClassBase):
    """Reshape a rank-4 RMSNormalization (norm over the last axis) to rank-3.

    **Matched pattern:**

    .. code-block:: text

        norm_out = RMSNormalization(x, weight)   # x rank-4, axis = -1

    **Replacement:**

    .. code-block:: text

        x3       = Reshape(x, [0, -1, head_dim])               # (B, S*H, Dh)
        normed   = RMSNormalization(x3, weight, axis=-1)
        norm_out = Reshape(normed, [0, -1, heads, head_dim])   # (B, S, H, Dh)

    Only fires when the input rank is 4, the norm reduces a single (last)
    axis, and the head/head_dim sizes are static. The two reshapes use
    constant shapes (no ``Shape`` op), so the rewritten graph stays static.
    The rank-3 replacement does not re-match (the check requires rank 4).
    """

    def pattern(self, op, x, weight):
        return op.RMSNormalization(
            x, weight, _allow_other_attributes=True, _outputs=["norm_out"]
        )

    def check(self, context, x, norm_out, **_):
        result = MatchResult()
        shape = x.shape
        if shape is None or len(shape) != 4:
            return result.fail("RMSNormalization input is not rank-4")
        heads, head_dim = shape[2], shape[3]
        if not isinstance(heads, int) or not isinstance(head_dim, int):
            return result.fail("rank-4 RMSNormalization head/head_dim are not static")
        # Reshaping heads into the sequence axis is only valid when the reduced
        # axis is head_dim alone, i.e. the last axis.
        axis = norm_out.producer().attributes.get_int("axis", -1)
        if axis not in (-1, 3):
            return result.fail("RMSNormalization reduces more than the last axis")
        if norm_out.producer().attributes.get_float("epsilon", None) is None:
            return result.fail("Missing epsilon attribute on RMSNormalization")
        return result

    def rewrite(self, op, x, weight, norm_out, **_):
        rmsnorm = norm_out.producer()
        heads, head_dim = x.shape[2], x.shape[3]
        attrs = {key: rmsnorm.attributes[key].value for key in rmsnorm.attributes}
        attrs["axis"] = -1
        x3 = op.Reshape(x, op.Constant(value_ints=[0, -1, head_dim]))
        normed = op.RMSNormalization(x3, weight, **attrs)
        return op.Reshape(normed, op.Constant(value_ints=[0, -1, heads, head_dim]))


def reshape_rank4_rmsnorm_rules() -> RewriteRuleSet:
    """Return rules that reshape rank-4 RMSNormalization to rank-3.

    Used for the QNN HTP, which miscomputes RMSNormalization over the last
    axis of a rank-4 tensor (query/key norm). The transform is numerically
    exact.

    Returns:
        :class:`RewriteRuleSet` containing the :class:`Rank4RMSNormToRank3` rule.
    """
    return RewriteRuleSet([Rank4RMSNormToRank3().rule()])
