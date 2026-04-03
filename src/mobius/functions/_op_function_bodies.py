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

All function bodies are built using onnxscript.ir APIs directly (ir.Graph,
ir.Node, ir.Function) — never through onnx.helper / protobuf helpers.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript._internal import builder


def _build_skip_layer_norm_function() -> ir.Function:
    """ir.Function for ``com.microsoft::SkipLayerNormalization``.

    Standard-ONNX body::

        add_out  = Add(input, skip)
        norm_out, mean_out, inv_std_out = LayerNormalization(
            add_out, weight, bias, axis=-1, epsilon=<forwarded>
        )
        # add_out is also exposed as the unnormalized residual sum

    Inputs:  ``[input, skip, weight, bias]``  (``bias`` may be absent at the
             call site; LayerNormalization accepts optional B).
    Outputs: ``[norm_out, mean_out, inv_std_out, add_out]``
    Attr:    ``epsilon`` (float)
    """
    v_input = ir.Value(name="input")
    v_skip = ir.Value(name="skip")
    v_weight = ir.Value(name="weight")
    v_bias = ir.Value(name="bias")

    graph = ir.Graph(
        inputs=[v_input, v_skip, v_weight, v_bias],
        outputs=[],
        nodes=[],
        name="SkipLayerNormalization_body",
        opset_imports={"": 17},
    )
    gb = builder.GraphBuilder(graph)
    op = gb.op

    # add_out = Add(input, skip)  — the unnormalized residual sum
    add_out = op.Add(v_input, v_skip)

    # LayerNormalization(add_out, weight, bias, axis=-1, epsilon=<from caller>)
    # Multi-output node — construct ir.Node directly so we can forward epsilon.
    # LayerNormalization has 3 outputs: Y (required), Mean (optional), InvStdDev (optional).
    ln_node = ir.Node(
        "",
        "LayerNormalization",
        inputs=[add_out, v_weight, v_bias],
        attributes=[
            ir.Attr("axis", ir.AttributeType.INT, -1),
            # ref_attr_name="epsilon" means InlinePass substitutes the
            # caller's epsilon value when expanding this function body.
            ir.Attr("epsilon", ir.AttributeType.FLOAT, 1e-5, ref_attr_name="epsilon"),
        ],
        num_outputs=3,
    )
    graph.append(ln_node)
    norm_out, mean_out, inv_std_out = ln_node.outputs

    graph.outputs.extend([norm_out, mean_out, inv_std_out, add_out])

    return ir.Function(
        domain="com.microsoft",
        name="SkipLayerNormalization",
        graph=graph,
        attributes={"epsilon": ir.Attr("epsilon", ir.AttributeType.FLOAT, 1e-5)},
    )


def _build_skip_simplified_layer_norm_function() -> ir.Function:
    """ir.Function for ``com.microsoft::SkipSimplifiedLayerNormalization``.

    Standard-ONNX body::

        add_out  = Add(input, skip)
        norm_out = RMSNormalization(add_out, weight, epsilon=<forwarded>)

    Inputs:  ``[input, skip, weight]``
    Outputs: ``[norm_out, add_out]``
    Attr:    ``epsilon`` (float)
    """
    v_input = ir.Value(name="input")
    v_skip = ir.Value(name="skip")
    v_weight = ir.Value(name="weight")

    graph = ir.Graph(
        inputs=[v_input, v_skip, v_weight],
        outputs=[],
        nodes=[],
        name="SkipSimplifiedLayerNormalization_body",
        opset_imports={"": 23},
    )
    gb = builder.GraphBuilder(graph)
    op = gb.op

    # add_out = Add(input, skip)
    add_out = op.Add(v_input, v_skip)

    # RMSNormalization(add_out, weight, epsilon=<from caller>)
    # Schema at opset 23: single output Y only (no InvStdDev).
    rms_node = ir.Node(
        "",
        "RMSNormalization",
        inputs=[add_out, v_weight],
        attributes=[
            ir.Attr("epsilon", ir.AttributeType.FLOAT, 1e-5, ref_attr_name="epsilon"),
        ],
        num_outputs=1,
    )
    graph.append(rms_node)
    norm_out = rms_node.outputs[0]

    graph.outputs.extend([norm_out, add_out])

    return ir.Function(
        domain="com.microsoft",
        name="SkipSimplifiedLayerNormalization",
        graph=graph,
        attributes={"epsilon": ir.Attr("epsilon", ir.AttributeType.FLOAT, 1e-5)},
    )


def _build_simplified_layer_norm_function() -> ir.Function:
    """ir.Function for ``com.microsoft::SimplifiedLayerNormalization``.

    This op appears in externally-produced (ORT-optimized) graphs.
    Mobius itself emits the standard ``RMSNormalization`` op.

    Standard-ONNX body::

        out = RMSNormalization(x, weight, epsilon=<forwarded>)

    Inputs:  ``[x, weight]``
    Outputs: ``[out]``
    Attr:    ``epsilon`` (float)
    """
    v_x = ir.Value(name="x")
    v_weight = ir.Value(name="weight")

    graph = ir.Graph(
        inputs=[v_x, v_weight],
        outputs=[],
        nodes=[],
        name="SimplifiedLayerNormalization_body",
        opset_imports={"": 23},
    )

    # RMSNormalization(x, weight, epsilon=<from caller>)
    # Schema at opset 23: single output Y only.
    rms_node = ir.Node(
        "",
        "RMSNormalization",
        inputs=[v_x, v_weight],
        attributes=[
            ir.Attr("epsilon", ir.AttributeType.FLOAT, 1e-5, ref_attr_name="epsilon"),
        ],
        num_outputs=1,
    )
    graph.append(rms_node)
    out = rms_node.outputs[0]

    graph.outputs.extend([out])

    return ir.Function(
        domain="com.microsoft",
        name="SimplifiedLayerNormalization",
        graph=graph,
        attributes={"epsilon": ir.Attr("epsilon", ir.AttributeType.FLOAT, 1e-5)},
    )


# Lazy-built singletons keyed by (domain, name, overload).
_FUNCTION_BUILDERS: dict[ir.OperatorIdentifier, object] = {
    ("com.microsoft", "SkipLayerNormalization", ""): _build_skip_layer_norm_function,
    (
        "com.microsoft",
        "SkipSimplifiedLayerNormalization",
        "",
    ): _build_skip_simplified_layer_norm_function,
    (
        "com.microsoft",
        "SimplifiedLayerNormalization",
        "",
    ): _build_simplified_layer_norm_function,
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
