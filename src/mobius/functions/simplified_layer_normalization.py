# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard-ONNX ir.Function body for the SimplifiedLayerNormalization operator.

Provides a portable fallback implementation for the ``com.microsoft``
``SimplifiedLayerNormalization`` custom op. When registered in
``model.functions``, :class:`onnx_ir.passes.common.InlinePass` can expand
it for EPs that do not support the custom op natively.

Naming convention:
    Python factory function is snake_case (``simplified_layer_normalization``)
    while the ``ir.Function`` op type string is PascalCase
    (``"SimplifiedLayerNormalization"``). A PascalCase alias is provided for
    discoverability, matching the convention in
    :mod:`~mobius.functions.causal_conv` and
    :mod:`~mobius.functions.linear_attention`.

.. note::

   This function body uses raw ``ir.Node`` construction for the
   ``RMSNormalization`` node because ``ref_attr_name`` (which tells
   InlinePass to forward the caller's ``epsilon`` value) is not supported
   by the ``OpBuilder`` API.
"""

from __future__ import annotations

import onnx_ir as ir

from mobius._constants import OPSET_VERSION

DOMAIN = "com.microsoft"


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
