# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite rules for decomposing SimplifiedLayerNormalization into primitives.

TRT-RTX does not support ``com.microsoft::SimplifiedLayerNormalization``.
This rule decomposes it into standard ONNX primitive ops that implement
RMSNorm: ``Pow → ReduceMean → Add(eps) → Sqrt → Div → Mul(weight)``.

``SimplifiedLayerNormalization`` is Microsoft's custom op equivalent of
ONNX standard ``RMSNormalization``.  In practice, mobius emits the
standard ``RMSNormalization`` op, so this rule is primarily useful when
processing externally-produced graphs that contain the Microsoft variant.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class DecomposeSimplifiedLayerNorm(RewriteRuleClassBase):
    """Decompose SimplifiedLayerNormalization into primitive ONNX ops.

    **Matched pattern (com.microsoft domain):**

    .. code-block:: text

        out = SimplifiedLayerNormalization(x, weight, epsilon=eps)

    **Replacement (RMSNorm via primitives):**

    .. code-block:: text

        x_sq      = Pow(x, 2)
        mean_sq   = ReduceMean(x_sq, axes=[-1], keepdims=1)
        rms_sq    = Add(mean_sq, epsilon)
        rms       = Sqrt(rms_sq)
        x_norm    = Div(x, rms)
        out       = Mul(x_norm, weight)
    """

    def pattern(self, op, x, weight):
        return op.SimplifiedLayerNormalization(
            x,
            weight,
            _domain="com.microsoft",
            _allow_other_attributes=True,
            _outputs=["out"],
        )

    def check(self, context, out, **_):
        result = MatchResult()
        node = out.producer()
        if node is None:
            return result.fail("out has no producer")
        if node.attributes.get_float("epsilon", None) is None:
            return result.fail("Missing epsilon attribute")
        return result

    def rewrite(self, op, x, weight, out, **_):
        node = out.producer()
        epsilon = node.attributes.get_float("epsilon")

        # RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
        two = op.Constant(value=ir.Tensor(np.array(2.0, dtype=np.float32)))
        x_sq = op.Pow(x, two)
        axes = op.Constant(
            value=ir.Tensor(np.array([-1], dtype=np.int64))
        )
        mean_sq = op.ReduceMean(x_sq, axes, keepdims=True)
        eps_const = op.Constant(
            value=ir.Tensor(np.array(epsilon, dtype=np.float32))
        )
        rms_sq = op.Add(mean_sq, eps_const)
        rms = op.Sqrt(rms_sq)
        x_norm = op.Div(x, rms)
        return op.Mul(x_norm, weight)


def decompose_simplified_layer_norm_rules() -> RewriteRuleSet:
    """Return rules that decompose SimplifiedLayerNormalization to primitives.

    Decomposes the ``com.microsoft::SimplifiedLayerNormalization`` custom
    op into standard ONNX ops (Pow, ReduceMean, Add, Sqrt, Div, Mul).
    Used as a TRT-RTX lowering pass.

    Returns:
        :class:`RewriteRuleSet` containing the decomposition rule.
    """
    return RewriteRuleSet(
        [DecomposeSimplifiedLayerNorm().rule()]
    )
