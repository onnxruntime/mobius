# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test-copy GroupedMatMul and SwiGLU expansions for execution on opset-24 ORT."""

from __future__ import annotations

import onnx_ir as ir
from onnx_ir.passes.common import InlinePass, RemoveUnusedFunctionsPass
from onnxscript._internal.builder import build_function


def expand_grouped_matmul(model: ir.Model) -> ir.Model:
    """Expand GroupedMatMul and SwiGLU on an executable test copy.

    GroupedMatMul source (transB=1/no-bias FLOAT): onnx/defs/math/defs.cc at
    82721c0908fa965dd84e674c006147791313f3e1.
    This is not a production lowering or a general opset converter.
    """

    def body(op, x, weights, indices):
        m = op.Shape(x, start=0, end=1)
        width = op.Shape(x, start=1, end=2)
        n = op.Shape(weights, start=1, end=2)
        selections = op.Shape(indices, start=1, end=2)
        one = op.Constant(value_ints=[1])
        ms = op.Mul(m, selections)
        flat_indices = op.Reshape(indices, ms)
        chosen = op.Transpose(op.Gather(weights, flat_indices, axis=0), perm=[0, 2, 1])
        expanded = op.Expand(op.Unsqueeze(x, one), op.Concat(m, selections, width, axis=0))
        lhs = op.Reshape(expanded, op.Concat(ms, one, width, axis=0))
        return op.Reshape(op.MatMul(lhs, chosen), op.Concat(m, selections, n, axis=0))

    fn = build_function(
        body,
        [ir.Value(name=n) for n in ("x", "weights", "indices")],
        domain="test.qmoe",
        name="GroupedMatMul",
        opset_imports={"": 29},
    )
    model.functions[fn.identifier()] = fn
    model.graph.opset_imports["test.qmoe"] = 1
    for node in model.graph:
        if node.op_type == "GroupedMatMul":
            assert node.attributes.get_int("transB") == 1
            node.domain = "test.qmoe"
            node.attributes.clear()
    InlinePass()(model)
    RemoveUnusedFunctionsPass()(model)
    for node in list(model.graph):
        if node.domain == "" and node.op_type == "SwiGLU":
            # SwiGLU-28(A,B,alpha) = Swish-24(A,alpha) * B. Never merely
            # relabel SwiGLU as opset 24: no such schema exists.
            swish = ir.Node(
                "",
                "Swish",
                [node.inputs[0]],
                attributes=[ir.AttrFloat32("alpha", node.attributes.get_float("alpha", 1.0))],
                num_outputs=1,
            )
            product = ir.Node("", "Mul", [swish.outputs[0], node.inputs[1]], num_outputs=1)
            ir.convenience.replace_nodes_and_values(
                model.graph,
                insertion_point=node,
                old_nodes=[node],
                new_nodes=[swish, product],
                old_values=node.outputs,
                new_values=product.outputs,
            )
    # Remaining operators have the same used semantics at 24. Production
    # graphs retain their standard operators and opset-29 imports.
    model.graph.opset_imports[""] = 24
    return model
