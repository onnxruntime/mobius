# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Lower bfloat16 ``Clip`` to ``Min``/``Max`` primitives.

ONNX Runtime has no ``Clip`` kernel for ``BFLOAT16`` on any execution provider.
Because ``Clip`` carries an ONNX function body, ORT silently expands it into
comparison primitives (``Less``/``Where``) that also lack bfloat16 kernels, so
the expanded nodes cannot be assigned to any provider and **session creation
fails outright**::

    FAIL : Exception during initialization: transformer_memcpy.cc:253
    Provider type for Less node with name '' is not set.

``Min`` and ``Max`` do have bfloat16 kernels, and ``Clip(x, lo, hi)`` is exactly
``Min(Max(x, lo), hi)`` — including for NaN inputs, since both lowerings inherit
the same propagation behaviour from the underlying comparisons. Rewriting the op
is therefore numerically exact, not an approximation.

The rule only fires for bfloat16 inputs; integer clamps (index clamping) and
float32/float16 clamps keep the compact ``Clip`` op and its native kernel.

These rules are applied automatically by
:func:`~mobius._optimizations.optimize_model` for bfloat16 models. They can also
be applied manually::

    from mobius.rewrite_rules import clip_to_min_max_rules
    from onnxscript.rewriter import rewrite

    rewrite(model, pattern_rewrite_rules=clip_to_min_max_rules())
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class _ClipToMinMaxBase(RewriteRuleClassBase):
    """Shared bfloat16 check for the ``Clip`` lowering variants."""

    def check(self, context, x, **_):
        result = MatchResult()
        if x.dtype != ir.DataType.BFLOAT16:
            return result.fail("Clip input is not bfloat16")
        return result


class ClipBothToMinMax(_ClipToMinMaxBase):
    """Rewrite ``Clip(x, lo, hi)`` → ``Min(Max(x, lo), hi)`` for bfloat16."""

    def pattern(self, op, x, lo, hi):
        return op.Clip(x, lo, hi)

    def rewrite(self, op, x, lo, hi):
        return op.Min(op.Max(x, lo), hi)


class ClipMinToMax(_ClipToMinMaxBase):
    """Rewrite the lower-bound-only ``Clip(x, lo)`` → ``Max(x, lo)`` for bfloat16."""

    def pattern(self, op, x, lo):
        return op.Clip(x, lo)

    def rewrite(self, op, x, lo):
        return op.Max(x, lo)


def clip_to_min_max_rules() -> RewriteRuleSet:
    """Return rules lowering bfloat16 ``Clip`` to ``Min``/``Max``.

    ONNX Runtime cannot execute ``Clip`` in bfloat16 on any execution provider;
    the ONNX function expansion produces ``Less``/``Where`` nodes with no
    bfloat16 kernel, which aborts session creation. ``Min``/``Max`` have
    bfloat16 kernels and are numerically identical.

    Returns:
        :class:`RewriteRuleSet` with the two-bound and lower-bound-only rules.
    """
    return RewriteRuleSet([ClipBothToMinMax().rule(), ClipMinToMax().rule()])
