# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite rules for fusing Transpose + MatMul into FusedMatMul.

:class:`~mobius.components._common.Linear` emits
``Transpose(weight, perm=[1, 0]) → MatMul(x, weight_t)`` so that
:class:`~mobius.passes.FoldTransposedInitializerPass` can pre-compute the
transpose after weights are loaded.  For EPs that report
``supports_fused_matmul=True``, :func:`~mobius._optimizations.optimize_model`
applies these rules **before** registering function bodies, converting the
``Transpose + MatMul`` pattern into
``com.microsoft::FusedMatMul(transB=1)``.  For all other EPs the
``FoldTransposedInitializerPass`` handles the pattern instead.

These rules are therefore applied to **every mobius-built model** whose EP
supports FusedMatMul.  They can also be applied to **external models** (e.g.
models from other exporters) that have the same ``Transpose(weight) → MatMul``
pattern.

.. warning::
   Do not apply these rules to a model that has a ``com.microsoft::FusedMatMul``
   function body registered (e.g. the default body from
   :func:`~mobius.functions.register_function_bodies`).  The rule matches
   the ``Transpose + MatMul`` pattern inside the function body itself and
   would replace it with a self-recursive call, creating a cyclic dependency
   that :class:`~onnx_ir.passes.common.InlinePass` will reject.

   :func:`~mobius._optimizations.optimize_model` avoids this by applying the
   rules **before** registering function bodies — see the inline comments
   there.  If you apply these rules manually, ensure no FusedMatMul function
   body is registered first, or remove it beforehand:

   .. code-block:: python

       m.functions.pop(("com.microsoft", "FusedMatMul", ""), None)
       rewrite(m, pattern_rewrite_rules=fused_matmul_rules())
"""

from __future__ import annotations

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import (
    RewriteRuleClassBase,
    RewriteRuleSet,
)


class TransposeMatMulToFusedMatMul(RewriteRuleClassBase):
    """Replace Transpose(perm=[1,0]) + MatMul with FusedMatMul(transB=1).

    **Matched pattern:**

    .. code-block:: text

        w_t = Transpose(weight, perm=[1, 0])
        result = MatMul(x, w_t)

    Where ``w_t`` has exactly one consumer (the MatMul node).

    **Replacement:**

    .. code-block:: text

        result = FusedMatMul(x, weight, transB=1)

    This eliminates the Transpose node entirely.  The ``FusedMatMul``
    op computes ``x @ weight.T`` in a single fused kernel.
    """

    def pattern(self, op, x, w_t):
        return op.MatMul(x, w_t, _outputs=["matmul_out"])

    def check(self, context, w_t, matmul_out, **_):
        result = MatchResult()

        # w_t must come from a Transpose node
        producer = w_t.producer()
        if producer is None or producer.op_type != "Transpose":
            return result.fail("MatMul input B is not from Transpose")

        # Must be a 2-D weight transpose: perm=[1, 0]
        perm = producer.attributes.get("perm", None)
        if perm is None:
            return result.fail("Transpose has no perm attribute")
        perm_val = list(perm.value)
        if perm_val != [1, 0]:
            return result.fail(f"Transpose perm={perm_val}, expected [1, 0]")

        # The Transpose output must have only 1 consumer (the MatMul)
        # so we can safely remove it
        uses = list(w_t.uses())
        if len(uses) != 1:
            return result.fail(
                f"Transpose output has {len(uses)} consumers, "
                "expected exactly 1 (the MatMul node)"
            )

        return result

    def rewrite(self, op, x, w_t, matmul_out, **_):
        transpose_node = w_t.producer()
        weight = transpose_node.inputs[0]

        return op.op(
            "FusedMatMul",
            inputs=[x, weight],
            domain="com.microsoft",
            attributes={"transB": 1},
        )


def fused_matmul_rules() -> RewriteRuleSet:
    """Return rules that fuse Transpose + MatMul into FusedMatMul.

    These rules match the ``Transpose(weight, [1,0]) → MatMul(x, w_t)``
    pattern emitted by every ``Linear`` layer and replace it with the
    Microsoft ``FusedMatMul`` custom op with ``transB=1``.

    Returns:
        :class:`RewriteRuleSet` containing the Transpose+MatMul
        fusion rule.
    """
    return RewriteRuleSet([TransposeMatMulToFusedMatMul().rule()])
