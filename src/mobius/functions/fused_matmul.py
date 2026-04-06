# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard-ONNX ir.Function body for the FusedMatMul operator.

Provides a portable fallback decomposition for ``com.microsoft::FusedMatMul``.
When registered in ``model.functions``, :class:`onnx_ir.passes.common.InlinePass`
can expand the op for EPs that do not support the custom kernel natively.

The function body implements the ``transB=1`` case used by every
:class:`~mobius.components._common.Linear` layer::

    B_t    = Transpose(B, perm=[1, 0])
    result = MatMul(A, B_t)
    alpha  = Constant(value_float=<forwarded from caller>)
    out    = Mul(result, alpha)

``alpha`` defaults to ``1.0``.  When the caller passes ``alpha=1.0``, a
constant-fold pass will simplify ``Mul(result, 1.0)`` to ``result``.

The ``transA`` attribute is present for schema compatibility but is not used
in the function body (``Linear`` always passes ``transA=0``).  If your model
requires ``transA=1`` and you want InlinePass to handle it, override the
function body by registering a custom ``ir.Function`` with the same op
identifier.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript._internal import builder

from mobius._constants import OPSET_VERSION

DOMAIN = "com.microsoft"


def fused_matmul() -> ir.Function:
    """Build an ``ir.Function`` for ``FusedMatMul`` (transB=1, any alpha).

    Standard-ONNX body::

        B_t    = Transpose(B, perm=[1, 0])
        result = MatMul(A, B_t)
        alpha  = Constant(value_float=<alpha forwarded from caller>)
        out    = Mul(result, alpha)

    Inputs:  ``[A, B]``
    Output:  ``[out]``  (``A @ B.T * alpha``)
    Attrs:   ``transA`` (int, default 0, unused in body),
             ``transB`` (int, default 1, hardcoded in body as Transpose),
             ``alpha``  (float, default 1.0, forwarded via ref_attr_name)
    """
    v_a = ir.Value(name="A")
    v_b = ir.Value(name="B")

    graph = ir.Graph(
        inputs=[v_a, v_b],
        outputs=[],
        nodes=[],
        name="FusedMatMul_body",
        opset_imports={"": OPSET_VERSION},
    )
    gb = builder.GraphBuilder(graph)
    op = gb.op

    # transB=1: transpose B so MatMul computes A @ B.T
    b_t = op.Transpose(v_b, perm=[1, 0])
    matmul_out = op.MatMul(v_a, b_t)

    # alpha scaling: Constant with ref_attr_name forwards the caller's alpha
    # value at InlinePass expand time.  When alpha=1.0 a constant-fold pass
    # simplifies Mul(result, 1.0) → result.
    alpha_node = ir.Node(
        "",
        "Constant",
        inputs=[],
        attributes=[
            ir.Attr(
                "value_float",
                ir.AttributeType.FLOAT,
                1.0,
                ref_attr_name="alpha",
            ),
        ],
        num_outputs=1,
    )
    graph.append(alpha_node)
    alpha_value = alpha_node.outputs[0]

    out = op.Mul(matmul_out, alpha_value)
    graph.outputs.append(out)

    return ir.Function(
        domain=DOMAIN,
        name="FusedMatMul",
        graph=graph,
        attributes={
            "transA": ir.Attr("transA", ir.AttributeType.INT, 0),
            "transB": ir.Attr("transB", ir.AttributeType.INT, 1),
            "alpha": ir.Attr("alpha", ir.AttributeType.FLOAT, 1.0),
        },
    )
