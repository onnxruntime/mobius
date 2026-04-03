# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard-ONNX ir.Function bodies for com.microsoft custom ops.

When these functions are registered in ``model.functions``,
:class:`onnx_ir.passes.common.InlinePass` can expand them for EPs that do
not support the custom op natively — without any pattern-matching rewrite rules.

The approach:
  1. After fusion, call :func:`register_function_bodies` to add these
     function definitions to the model.
  2. Call ``InlinePass(criteria=...)`` with a predicate that returns ``True``
     for ops the target EP cannot execute.  InlinePass replaces each matching
     call-node with the standard-ONNX function body.

This is the inverse of the fusion rules:

  * ``skip_layer_norm_rules``   fuses Add + LayerNorm → SkipLayerNormalization
  * ``register_function_bodies`` registers the reverse expansion
  * ``InlinePass``               expands back when the EP doesn't support it

Adding a new custom op with a portable fallback:
  1. Define a ``_build_<name>_function()`` helper below.
  2. Add it to :data:`_FUNCTION_BUILDERS`.
  3. Done — no rewrite rule needed.
"""

from __future__ import annotations

import functools

import onnx
import onnx.helper as helper
import onnx_ir as ir
import onnx_ir.serde as serde
from onnx import AttributeProto


def _attr_ref(name: str, attr_type: int) -> AttributeProto:
    """Return an attribute reference proto for forwarding a function-level attribute."""
    ref = AttributeProto()
    ref.name = name
    ref.type = attr_type
    ref.ref_attr_name = name
    return ref


def _build_skip_layer_norm_function() -> ir.Function:
    """ir.Function for ``com.microsoft::SkipLayerNormalization``.

    Standard-ONNX body::

        add_out  = Add(input, skip)
        norm_out, mean_out, inv_std_out = LayerNormalization(
            add_out, weight, bias, epsilon=epsilon, axis=-1
        )
        # add_out is also exposed as skip_out (the unnormalized sum)

    Inputs:  ``[input, skip, weight, bias]``  (``bias`` is optional — may be
             ``None`` at the call site; LayerNormalization accepts optional B).
    Outputs: ``[norm_out, mean_out, inv_std_out, add_out]``
    Attr:    ``epsilon`` (float)
    """
    add = helper.make_node("Add", inputs=["input", "skip"], outputs=["add_out"])
    ln = helper.make_node(
        "LayerNormalization",
        inputs=["add_out", "weight", "bias"],
        outputs=["norm_out", "mean_out", "inv_std_out"],
    )
    ln.attribute.extend([
        helper.make_attribute("axis", -1),
        _attr_ref("epsilon", AttributeProto.FLOAT),
    ])
    proto = helper.make_function(
        domain="com.microsoft",
        fname="SkipLayerNormalization",
        inputs=["input", "skip", "weight", "bias"],
        outputs=["norm_out", "mean_out", "inv_std_out", "add_out"],
        nodes=[add, ln],
        opset_imports=[helper.make_opsetid("", 17)],
        attributes=["epsilon"],
    )
    return serde.deserialize_function(proto)


def _build_skip_simplified_layer_norm_function() -> ir.Function:
    """ir.Function for ``com.microsoft::SkipSimplifiedLayerNormalization``.

    Standard-ONNX body::

        add_out  = Add(input, skip)
        norm_out = RMSNormalization(add_out, weight, epsilon=epsilon)

    Inputs:  ``[input, skip, weight]``
    Outputs: ``[norm_out, add_out]``
    Attr:    ``epsilon`` (float)
    """
    add = helper.make_node("Add", inputs=["input", "skip"], outputs=["add_out"])
    rms = helper.make_node(
        "RMSNormalization",
        inputs=["add_out", "weight"],
        outputs=["norm_out"],
    )
    rms.attribute.append(_attr_ref("epsilon", AttributeProto.FLOAT))
    proto = helper.make_function(
        domain="com.microsoft",
        fname="SkipSimplifiedLayerNormalization",
        inputs=["input", "skip", "weight"],
        outputs=["norm_out", "add_out"],
        nodes=[add, rms],
        opset_imports=[helper.make_opsetid("", 23)],
        attributes=["epsilon"],
    )
    return serde.deserialize_function(proto)


def _build_simplified_layer_norm_function() -> ir.Function:
    """ir.Function for ``com.microsoft::SimplifiedLayerNormalization``.

    This op appears in externally-produced (ORT-optimized) graphs.
    Mobius itself emits the standard ``RMSNormalization`` op.

    Standard-ONNX body::

        out = RMSNormalization(x, weight, epsilon=epsilon)

    Inputs:  ``[x, weight]``
    Outputs: ``[out]``
    Attr:    ``epsilon`` (float)
    """
    rms = helper.make_node(
        "RMSNormalization",
        inputs=["x", "weight"],
        outputs=["out"],
    )
    rms.attribute.append(_attr_ref("epsilon", AttributeProto.FLOAT))
    proto = helper.make_function(
        domain="com.microsoft",
        fname="SimplifiedLayerNormalization",
        inputs=["x", "weight"],
        outputs=["out"],
        nodes=[rms],
        opset_imports=[helper.make_opsetid("", 23)],
        attributes=["epsilon"],
    )
    return serde.deserialize_function(proto)


# Lazy-built singletons keyed by (domain, name, overload).
_FUNCTION_BUILDERS: dict[ir.OperatorIdentifier, object] = {
    ("com.microsoft", "SkipLayerNormalization", ""): _build_skip_layer_norm_function,
    ("com.microsoft", "SkipSimplifiedLayerNormalization", ""): _build_skip_simplified_layer_norm_function,
    ("com.microsoft", "SimplifiedLayerNormalization", ""): _build_simplified_layer_norm_function,
}

_cache: dict[ir.OperatorIdentifier, ir.Function] = {}


def get_function(op_id: ir.OperatorIdentifier) -> ir.Function | None:
    """Return the cached ir.Function for *op_id*, or ``None`` if unknown."""
    if op_id not in _FUNCTION_BUILDERS:
        return None
    if op_id not in _cache:
        _cache[op_id] = _FUNCTION_BUILDERS[op_id]()  # type: ignore[operator]
    return _cache[op_id]


def register_function_bodies(model: ir.Model) -> None:
    """Add standard-ONNX function bodies to *model* for all known custom ops.

    After calling this, :class:`onnx_ir.passes.common.InlinePass` can expand
    any of these ops by passing a suitable ``criteria`` predicate.

    Only registers functions for ops that are not already defined in the model
    (to avoid overwriting user-provided function bodies).
    """
    for op_id in _FUNCTION_BUILDERS:
        if op_id in model.functions:
            continue  # preserve any existing function body
        fn = get_function(op_id)
        if fn is not None:
            model.functions[op_id] = fn
