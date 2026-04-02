# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite rules for decomposing SkipLayerNormalization into Add + LayerNorm.

TRT-RTX does not support ``com.microsoft::SkipLayerNormalization``.
This rule decomposes it back into its constituent standard ONNX ops:
``Add(skip_a, skip_b)`` followed by ``LayerNormalization(add_out, weight,
bias?, epsilon)``.

This is the inverse of the fusion in ``skip_layer_norm_rules()``.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


def _zero_scalar(op):
    """Create a zero scalar constant as a placeholder for unused outputs."""
    return op.Constant(value=ir.Tensor(np.array(0.0, dtype=np.float32)))


class DecomposeSkipLayerNorm(RewriteRuleClassBase):
    """Decompose SkipLayerNormalization → Add + LayerNormalization.

    **Matched pattern (com.microsoft domain):**

    .. code-block:: text

        norm_out, mean, inv_std, skip_out = SkipLayerNormalization(
            skip_a, skip_b, gamma, beta, epsilon=eps,
        )

    **Replacement:**

    .. code-block:: text

        add_out = Add(skip_a, skip_b)
        norm_out = LayerNormalization(add_out, gamma, beta, epsilon=eps)

    ``skip_out`` users are redirected to ``add_out``.
    """

    def pattern(self, op, skip_a, skip_b, gamma, beta):
        return op.SkipLayerNormalization(
            skip_a,
            skip_b,
            gamma,
            beta,
            _domain="com.microsoft",
            _allow_other_attributes=True,
            _outputs=["norm_out", "mean_out", "inv_std_out", "skip_out"],
        )

    def check(self, context, norm_out, **_):
        result = MatchResult()
        node = norm_out.producer()
        if node is None:
            return result.fail("norm_out has no producer")
        if node.attributes.get_float("epsilon", None) is None:
            return result.fail("Missing epsilon attribute")
        return result

    def rewrite(self, op, skip_a, skip_b, gamma, beta, norm_out, skip_out, **_):
        node = norm_out.producer()
        epsilon = node.attributes.get_float("epsilon")

        add_out = op.Add(skip_a, skip_b)
        new_norm = op.LayerNormalization(
            add_out, gamma, beta, epsilon=epsilon, axis=-1
        )

        # Return 4 outputs: norm, mean(unused), inv_std(unused), skip
        return new_norm, _zero_scalar(op), _zero_scalar(op), add_out


class DecomposeSkipLayerNormNoBias(RewriteRuleClassBase):
    """Decompose bias-free SkipLayerNormalization → Add + LayerNormalization.

    Same as :class:`DecomposeSkipLayerNorm` but for the 3-input variant
    (skip_a, skip_b, gamma) without the beta (bias) parameter.
    """

    def pattern(self, op, skip_a, skip_b, gamma):
        return op.SkipLayerNormalization(
            skip_a,
            skip_b,
            gamma,
            _domain="com.microsoft",
            _allow_other_attributes=True,
            _outputs=["norm_out", "mean_out", "inv_std_out", "skip_out"],
        )

    def check(self, context, norm_out, **_):
        result = MatchResult()
        node = norm_out.producer()
        if node is None:
            return result.fail("norm_out has no producer")
        if node.attributes.get_float("epsilon", None) is None:
            return result.fail("Missing epsilon attribute")
        # Ensure this is truly bias-free (3 inputs, not 4)
        if len(node.inputs) > 3:
            return result.fail("Has bias — use the 4-input rule")
        return result

    def rewrite(self, op, skip_a, skip_b, gamma, norm_out, skip_out, **_):
        node = norm_out.producer()
        epsilon = node.attributes.get_float("epsilon")

        add_out = op.Add(skip_a, skip_b)
        new_norm = op.LayerNormalization(
            add_out, gamma, epsilon=epsilon, axis=-1
        )

        # Return 4 outputs: norm, mean(unused), inv_std(unused), skip
        return new_norm, _zero_scalar(op), _zero_scalar(op), add_out


class DecomposeSkipSimplifiedLayerNorm(RewriteRuleClassBase):
    """Decompose SkipSimplifiedLayerNormalization → Add + RMSNormalization.

    TRT-RTX does not support ``com.microsoft::SkipSimplifiedLayerNormalization``
    either. Decompose to ``Add(skip_a, skip_b)`` + ``RMSNormalization(add_out,
    weight, epsilon)``.
    """

    def pattern(self, op, skip_a, skip_b, weight):
        return op.SkipSimplifiedLayerNormalization(
            skip_a,
            skip_b,
            weight,
            _domain="com.microsoft",
            _allow_other_attributes=True,
            _outputs=["norm_out", "dummy1", "dummy2", "skip_out"],
        )

    def check(self, context, norm_out, **_):
        result = MatchResult()
        node = norm_out.producer()
        if node is None:
            return result.fail("norm_out has no producer")
        if node.attributes.get_float("epsilon", None) is None:
            return result.fail("Missing epsilon attribute")
        return result

    def rewrite(self, op, skip_a, skip_b, weight, norm_out, skip_out, **_):
        node = norm_out.producer()
        epsilon = node.attributes.get_float("epsilon")

        add_out = op.Add(skip_a, skip_b)
        new_norm = op.RMSNormalization(
            add_out, weight, epsilon=epsilon, axis=-1
        )

        # Return 4 outputs: norm, dummy(unused), dummy(unused), skip
        return new_norm, _zero_scalar(op), _zero_scalar(op), add_out


def decompose_skip_layer_norm_rules() -> RewriteRuleSet:
    """Return rules that decompose skip-norm fusions into standard ops.

    Decomposes ``SkipLayerNormalization`` → ``Add + LayerNormalization``
    and ``SkipSimplifiedLayerNormalization`` → ``Add + RMSNormalization``.
    Used as a TRT-RTX lowering pass.

    Returns:
        :class:`RewriteRuleSet` containing the decomposition rules.
    """
    return RewriteRuleSet(
        [
            DecomposeSkipLayerNorm().rule(),
            DecomposeSkipLayerNormNoBias().rule(),
            DecomposeSkipSimplifiedLayerNorm().rule(),
        ]
    )
