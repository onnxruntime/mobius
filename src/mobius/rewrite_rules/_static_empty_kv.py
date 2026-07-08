# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Rewrite rule to replace dynamic empty-KV construction with a static Constant.

When WebGPU graph capture is enabled, the ``Shape`` and ``ConstantOfShape``
ops used to build the empty ``[batch, 0, kv_hidden]`` KV tensor for shared-KV
layers (e.g. Gemma4 layers 15-34) are problematic:

- ``Shape`` outputs to CPU, breaking the GPU graph capture boundary (affects
  all GPU EPs including WebGPU and CUDA).
- ``ConstantOfShape`` is unsupported by the WebGPU EP (not an issue for CUDA).

Two pattern variants are matched (both are emitted by onnxscript depending on
the optimization pass that runs first):

**Pattern A — pre-cleanup (CastLike):**

.. code-block:: text

    batch_dim   = Shape(query_states, start=0, end=1)
    empty_shape = Concat(batch_dim, Constant(value_ints=[0, kv_hidden]), axis=0)
    cos         = ConstantOfShape(empty_shape)
    empty_kv    = CastLike(cos, query_states)

**Pattern B — post-cleanup (Cast):**

.. code-block:: text

    batch_dim   = Shape(query_states, start=0, end=1)
    empty_shape = Concat(batch_dim, Constant(value_ints=[0, kv_hidden]), axis=0)
    cos         = ConstantOfShape(empty_shape)
    empty_kv    = Cast(cos, to=<dtype>)

**Replacement (both patterns):**

.. code-block:: text

    empty_kv = Constant(value=zeros([1, 0, kv_hidden], dtype=<model_dtype>))

The batch dimension is fixed to 1 (graph capture requires static shapes).
The dtype matches the model's activation dtype (e.g. float16) so the GQA op
receives a type-consistent input. The content is never read because
kv_sequence_length=0.

This rule is applied automatically by
:func:`~mobius._optimizations.optimize_model` for EPs with
``enable_graph_capture=True``.  It can also be applied manually::

    from mobius.rewrite_rules import static_empty_kv_rules
    from onnxscript.rewriter import rewrite

    model = build("google/gemma-4-E2B-it", execution_provider="webgpu")
    rewrite(model, pattern_rewrite_rules=static_empty_kv_rules())
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


def _check_shape_tail(shape_tail) -> int | None:
    """Return kv_hidden if shape_tail is Constant(value_ints=[0, kv_hidden]), else None."""
    tail_node = shape_tail.producer()
    if tail_node is None or tail_node.op_type != "Constant":
        return None
    value_ints = tail_node.attributes.get("value_ints")
    if value_ints is None:
        return None
    ints = value_ints.value
    if len(ints) != 2 or ints[0] != 0:
        return None
    return int(ints[1])


def _static_zero_constant(op, kv_hidden: int, ir_dtype: ir.DataType):
    """Return a static [1, 0, kv_hidden] zero tensor in the target dtype.

    Uses ``ir_dtype.numpy()`` to obtain the numpy dtype, which handles
    bfloat16 via ``ml_dtypes``.  The content is never read because
    kv_sequence_length=0.
    """
    return op.Constant(value=ir.tensor(np.zeros((1, 0, kv_hidden), dtype=ir_dtype.numpy())))


class _DynamicEmptyKVCastLike(RewriteRuleClassBase):
    """Replace ``Shape → Concat → ConstantOfShape → CastLike`` with a static zero tensor.

    Matches the pre-cleanup pattern emitted directly by onnxscript.
    """

    def pattern(self, op, query_states, shape_tail):
        batch_dim = op.Shape(query_states, start=0, end=1)
        empty_shape = op.Concat(batch_dim, shape_tail, axis=0)
        cos = op.ConstantOfShape(empty_shape)
        return op.CastLike(cos, query_states)

    def check(self, context, query_states, shape_tail, **_):
        result = MatchResult()
        if _check_shape_tail(shape_tail) is None:
            return result.fail("shape_tail is not Constant(value_ints=[0, kv_hidden])")
        return result

    def rewrite(self, op, query_states, shape_tail, **_):
        kv_hidden = _check_shape_tail(shape_tail)
        return _static_zero_constant(op, kv_hidden, query_states.dtype or ir.DataType.FLOAT)


class _DynamicEmptyKVCast(RewriteRuleClassBase):
    """Replace ``Shape → Concat → ConstantOfShape → Cast`` with a static zero tensor.

    Matches the post-cleanup pattern where CastLike has been materialized to
    Cast with a concrete dtype (e.g. after quantization or ONNX cleanup passes).
    """

    def pattern(self, op, query_states, shape_tail):
        batch_dim = op.Shape(query_states, start=0, end=1)
        empty_shape = op.Concat(batch_dim, shape_tail, axis=0)
        cos = op.ConstantOfShape(empty_shape)
        return op.Cast(cos, _allow_other_attributes=True)

    def check(self, context, query_states, shape_tail, **_):
        result = MatchResult()
        if _check_shape_tail(shape_tail) is None:
            return result.fail("shape_tail is not Constant(value_ints=[0, kv_hidden])")
        return result

    def rewrite(self, op, query_states, shape_tail, **_):
        kv_hidden = _check_shape_tail(shape_tail)
        # Walk Concat → ConstantOfShape → Cast to read the target dtype.
        # shape_tail feeds Concat; Concat output feeds ConstantOfShape; its output feeds Cast.
        ir_dtype = ir.DataType.FLOAT
        for use in shape_tail.uses():
            concat_node = use.node
            if concat_node.op_type != "Concat":
                continue
            for use2 in concat_node.outputs[0].uses():
                cos_node = use2.node
                if cos_node.op_type != "ConstantOfShape":
                    continue
                for use3 in cos_node.outputs[0].uses():
                    cast_node = use3.node
                    if cast_node.op_type == "Cast":
                        to_attr = cast_node.attributes.get("to")
                        if to_attr is not None:
                            ir_dtype = ir.DataType(to_attr.value)
                        break
                break
            break
        return _static_zero_constant(op, kv_hidden, ir_dtype)


def static_empty_kv_rules() -> RewriteRuleSet:
    """Return a rule set that replaces dynamic empty-KV construction with a static Constant.

    Applied for EPs with ``enable_graph_capture=True`` where ``Shape``
    (and for WebGPU, ``ConstantOfShape``) are incompatible with graph capture.

    Two variants are handled: the ``CastLike`` form (pre-cleanup) and the
    ``Cast`` form (post-cleanup, after quantization or ONNX simplification).

    Returns:
        :class:`RewriteRuleSet` containing both :class:`_DynamicEmptyKVCastLike`
        and :class:`_DynamicEmptyKVCast`.
    """
    return RewriteRuleSet(
        [
            _DynamicEmptyKVCastLike().rule(),
            _DynamicEmptyKVCast().rule(),
        ]
    )
