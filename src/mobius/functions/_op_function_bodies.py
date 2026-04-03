# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard-ONNX ir.Function bodies for com.microsoft custom ops.

When these functions are registered in ``model.functions``,
:class:`onnx_ir.passes.common.InlinePass` can expand them for EPs that do
not support the custom op natively — without any pattern-matching rewrite
rules.

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

1. Write a public factory function below (snake_case, returns ``ir.Function``).
2. Add a PascalCase alias for discoverability.
3. Add it to :data:`_FUNCTION_BUILDERS`.

All function bodies are built using ``onnx_ir`` APIs directly
(``ir.Graph``, ``ir.Node``, ``ir.Function``) — never through
``onnx.helper`` / protobuf helpers.

Naming convention:
    Python factory functions are snake_case
    (e.g. ``skip_layer_normalization``) while the ``ir.Function`` op type
    strings are PascalCase (``"SkipLayerNormalization"``).  PascalCase
    aliases (``SkipLayerNormalization``) are provided for discoverability.
    This matches the convention in :mod:`~mobius.functions.causal_conv` and
    :mod:`~mobius.functions.linear_attention`.

.. note::

   These function bodies use raw ``ir.Node`` construction for
   ``LayerNormalization`` and ``RMSNormalization`` nodes because
   ``ref_attr_name`` (which tells InlinePass to forward the caller's
   ``epsilon`` value) is not supported by the ``OpBuilder`` API.
   All other ops use the standard ``OpBuilder`` (``op.Add``, etc.).
"""

from __future__ import annotations

from collections.abc import Callable

import onnx_ir as ir
from onnxscript._internal import builder

from mobius._constants import OPSET_VERSION

DOMAIN = "com.microsoft"


def skip_layer_normalization() -> ir.Function:
    """Build an ``ir.Function`` for ``SkipLayerNormalization``.

    Standard-ONNX body::

        add_out  = Add(input, skip)
        norm_out, mean_out, inv_std_out = LayerNormalization(
            add_out, weight, bias, axis=-1, epsilon=<forwarded>
        )

    Inputs:  ``[input, skip, weight, bias]``  (``bias`` may be absent at
             the call site; ``LayerNormalization`` accepts optional B).
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
        opset_imports={"": OPSET_VERSION},
    )
    gb = builder.GraphBuilder(graph)
    op = gb.op

    add_out = op.Add(v_input, v_skip)

    # ir.Node is required here: ref_attr_name forwards the caller's
    # epsilon value at InlinePass expand time (OpBuilder doesn't
    # support ref_attr_name).
    ln_node = ir.Node(
        "",
        "LayerNormalization",
        inputs=[add_out, v_weight, v_bias],
        attributes=[
            ir.Attr("axis", ir.AttributeType.INT, -1),
            ir.Attr(
                "epsilon",
                ir.AttributeType.FLOAT,
                1e-5,
                ref_attr_name="epsilon",
            ),
        ],
        num_outputs=3,
    )
    graph.append(ln_node)
    norm_out, mean_out, inv_std_out = ln_node.outputs

    graph.outputs.extend([norm_out, mean_out, inv_std_out, add_out])

    return ir.Function(
        domain=DOMAIN,
        name="SkipLayerNormalization",
        graph=graph,
        attributes={
            "epsilon": ir.Attr("epsilon", ir.AttributeType.FLOAT, 1e-5),
        },
    )


# PascalCase alias — matches the ONNX op type name for discoverability.
SkipLayerNormalization = skip_layer_normalization


def skip_simplified_layer_normalization() -> ir.Function:
    """Build an ``ir.Function`` for ``SkipSimplifiedLayerNormalization``.

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
        opset_imports={"": OPSET_VERSION},
    )
    gb = builder.GraphBuilder(graph)
    op = gb.op

    add_out = op.Add(v_input, v_skip)

    # ir.Node required: ref_attr_name forwards caller's epsilon.
    rms_node = ir.Node(
        "",
        "RMSNormalization",
        inputs=[add_out, v_weight],
        attributes=[
            ir.Attr(
                "epsilon",
                ir.AttributeType.FLOAT,
                1e-5,
                ref_attr_name="epsilon",
            ),
        ],
        num_outputs=1,
    )
    graph.append(rms_node)
    norm_out = rms_node.outputs[0]

    graph.outputs.extend([norm_out, add_out])

    return ir.Function(
        domain=DOMAIN,
        name="SkipSimplifiedLayerNormalization",
        graph=graph,
        attributes={
            "epsilon": ir.Attr("epsilon", ir.AttributeType.FLOAT, 1e-5),
        },
    )


# PascalCase alias — matches the ONNX op type name for discoverability.
SkipSimplifiedLayerNormalization = skip_simplified_layer_normalization


def simplified_layer_normalization() -> ir.Function:
    """Build an ``ir.Function`` for ``SimplifiedLayerNormalization``.

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
        opset_imports={"": OPSET_VERSION},
    )
    gb = builder.GraphBuilder(graph)
    # gb.op not needed — the only node uses ir.Node for ref_attr_name.
    _ = gb

    # ir.Node required: ref_attr_name forwards caller's epsilon.
    rms_node = ir.Node(
        "",
        "RMSNormalization",
        inputs=[v_x, v_weight],
        attributes=[
            ir.Attr(
                "epsilon",
                ir.AttributeType.FLOAT,
                1e-5,
                ref_attr_name="epsilon",
            ),
        ],
        num_outputs=1,
    )
    graph.append(rms_node)
    out = rms_node.outputs[0]

    graph.outputs.extend([out])

    return ir.Function(
        domain=DOMAIN,
        name="SimplifiedLayerNormalization",
        graph=graph,
        attributes={
            "epsilon": ir.Attr("epsilon", ir.AttributeType.FLOAT, 1e-5),
        },
    )


# PascalCase alias — matches the ONNX op type name for discoverability.
SimplifiedLayerNormalization = simplified_layer_normalization


# ---------------------------------------------------------------------------
# Registry and caching
# ---------------------------------------------------------------------------

# Lazy-built singletons keyed by (domain, name, overload).
_FUNCTION_BUILDERS: dict[ir.OperatorIdentifier, Callable[[], ir.Function]] = {
    (DOMAIN, "SkipLayerNormalization", ""): skip_layer_normalization,
    (
        DOMAIN,
        "SkipSimplifiedLayerNormalization",
        "",
    ): skip_simplified_layer_normalization,
    (
        DOMAIN,
        "SimplifiedLayerNormalization",
        "",
    ): simplified_layer_normalization,
}

_cache: dict[ir.OperatorIdentifier, ir.Function] = {}


def get_function(op_id: ir.OperatorIdentifier) -> ir.Function | None:
    """Return the cached ``ir.Function`` for *op_id*, or ``None``."""
    if op_id not in _FUNCTION_BUILDERS:
        return None
    if op_id not in _cache:
        _cache[op_id] = _FUNCTION_BUILDERS[op_id]()
    return _cache[op_id]


def register_function_bodies(model: ir.Model) -> None:
    """Add standard-ONNX function bodies to *model* for all known ops.

    After calling this, :class:`onnx_ir.passes.common.InlinePass` can
    expand any of these ops by passing a suitable ``criteria`` predicate.

    Only registers functions for ops that are not already defined in the
    model (to avoid overwriting user-provided function bodies).
    """
    for op_id in _FUNCTION_BUILDERS:
        if op_id in model.functions:
            continue
        fn = get_function(op_id)
        if fn is not None:
            model.functions[op_id] = fn
