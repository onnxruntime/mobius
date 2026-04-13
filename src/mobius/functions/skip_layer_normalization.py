# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standard-ONNX ir.Function bodies for SkipLayerNormalization operators.

Provides portable fallback implementations for the ``com.microsoft``
``SkipLayerNormalization`` and ``SkipSimplifiedLayerNormalization`` custom ops.
When registered in ``model.functions``, :class:`onnx_ir.passes.common.InlinePass`
can expand them for EPs that do not support the custom ops natively.

Naming convention:
    Python factory functions are snake_case (e.g. ``skip_layer_normalization``)
    while the ``ir.Function`` op type strings are PascalCase
    (``"SkipLayerNormalization"``). PascalCase aliases are provided for
    discoverability, matching the convention in
    :mod:`~mobius.functions.causal_conv` and
    :mod:`~mobius.functions.linear_attention`.

.. note::

   These function bodies use raw ``ir.Node`` construction for
   ``LayerNormalization`` and ``RMSNormalization`` nodes because
   ``ref_attr_name`` (which tells InlinePass to forward the caller's
   ``epsilon`` value) is not supported by the ``OpBuilder`` API.
   All other ops use the standard ``OpBuilder`` (``op.Add``, etc.).
"""

from __future__ import annotations

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
