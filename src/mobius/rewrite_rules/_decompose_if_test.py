# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import onnx_ir as ir

from mobius.rewrite_rules._decompose_if import DecomposeIfPass, decompose_if_pass
from mobius.rewrite_rules._testing_utils import count_ops


def _build_if_model() -> ir.Model:
    """Build a minimal ONNX model with a single If node.

    Graph:
        x: float32 scalar
        cond: bool scalar

        then_branch: result = Abs(x)
        else_branch: result = Neg(x)

        result = If(cond, then_branch, else_branch)
    """
    opset = {"": 23}
    x = ir.Value(name="x")
    cond = ir.Value(name="cond")

    # then_branch: result = Abs(x)
    then_abs = ir.Node(domain="", op_type="Abs", inputs=[x], num_outputs=1)
    then_result = then_abs.outputs[0]
    then_result.name = "then_result"
    then_graph = ir.Graph(
        inputs=[],
        outputs=[then_result],
        nodes=[then_abs],
        name="then_branch",
        opset_imports=opset,
    )

    # else_branch: result = Neg(x)
    else_neg = ir.Node(domain="", op_type="Neg", inputs=[x], num_outputs=1)
    else_result = else_neg.outputs[0]
    else_result.name = "else_result"
    else_graph = ir.Graph(
        inputs=[],
        outputs=[else_result],
        nodes=[else_neg],
        name="else_branch",
        opset_imports=opset,
    )

    if_node = ir.Node(
        domain="",
        op_type="If",
        inputs=[cond],
        num_outputs=1,
        attributes=[
            ir.Attr("then_branch", ir.AttributeType.GRAPH, then_graph),
            ir.Attr("else_branch", ir.AttributeType.GRAPH, else_graph),
        ],
    )
    if_out = if_node.outputs[0]
    if_out.name = "result"

    main_graph = ir.Graph(
        inputs=[x, cond],
        outputs=[if_out],
        nodes=[if_node],
        name="main",
        opset_imports=opset,
    )
    return ir.Model(main_graph, ir_version=10)


def _build_multi_output_if_model() -> ir.Model:
    """Build a model with a 2-output If node.

    then_branch: (Abs(x), Relu(x))
    else_branch: (Neg(x), Identity(x))
    """
    opset = {"": 23}
    x = ir.Value(name="x")
    cond = ir.Value(name="cond")

    then_abs = ir.Node(domain="", op_type="Abs", inputs=[x], num_outputs=1)
    then_relu = ir.Node(domain="", op_type="Relu", inputs=[x], num_outputs=1)
    then_abs.outputs[0].name = "then_abs"
    then_relu.outputs[0].name = "then_relu"
    then_graph = ir.Graph(
        inputs=[],
        outputs=[then_abs.outputs[0], then_relu.outputs[0]],
        nodes=[then_abs, then_relu],
        name="then_branch",
        opset_imports=opset,
    )

    else_neg = ir.Node(domain="", op_type="Neg", inputs=[x], num_outputs=1)
    else_id = ir.Node(domain="", op_type="Identity", inputs=[x], num_outputs=1)
    else_neg.outputs[0].name = "else_neg"
    else_id.outputs[0].name = "else_id"
    else_graph = ir.Graph(
        inputs=[],
        outputs=[else_neg.outputs[0], else_id.outputs[0]],
        nodes=[else_neg, else_id],
        name="else_branch",
        opset_imports=opset,
    )

    if_node = ir.Node(
        domain="",
        op_type="If",
        inputs=[cond],
        num_outputs=2,
        attributes=[
            ir.Attr("then_branch", ir.AttributeType.GRAPH, then_graph),
            ir.Attr("else_branch", ir.AttributeType.GRAPH, else_graph),
        ],
    )
    if_out0 = if_node.outputs[0]
    if_out1 = if_node.outputs[1]
    if_out0.name = "result_0"
    if_out1.name = "result_1"

    main_graph = ir.Graph(
        inputs=[x, cond],
        outputs=[if_out0, if_out1],
        nodes=[if_node],
        name="main",
        opset_imports=opset,
    )
    return ir.Model(main_graph, ir_version=10)


class TestDecomposeIfPass:
    def test_factory_returns_pass(self):
        p = decompose_if_pass()
        assert isinstance(p, DecomposeIfPass)

    def test_decomposes_single_output_if(self):
        """A single-output If is replaced by inlined branches + Where."""
        model = _build_if_model()
        counts_before = count_ops(model)
        assert counts_before["If"] == 1
        assert counts_before.get("Where", 0) == 0

        decompose_if_pass()(model)

        counts_after = count_ops(model)
        assert counts_after.get("If", 0) == 0, "If node should be removed"
        assert counts_after["Where"] == 1, "Where node should replace If"

    def test_inlines_both_branch_nodes(self):
        """Branch nodes (Abs, Neg) should appear in the main graph after decompose."""
        model = _build_if_model()
        decompose_if_pass()(model)

        ops = count_ops(model)
        assert ops.get("Abs", 0) == 1
        assert ops.get("Neg", 0) == 1

    def test_decomposes_multi_output_if(self):
        """Multi-output If produces one Where per output."""
        model = _build_multi_output_if_model()
        decompose_if_pass()(model)

        counts = count_ops(model)
        assert counts.get("If", 0) == 0
        assert counts["Where"] == 2

    def test_no_change_when_no_if_nodes(self):
        """Pass reports no modification when the model has no If nodes."""
        model = _build_if_model()
        decompose_if_pass()(model)  # first application removes the If
        counts_before = dict(count_ops(model))

        result = decompose_if_pass()(model)  # second application is a no-op
        assert not result.modified
        assert dict(count_ops(model)) == counts_before

    def test_where_uses_correct_condition(self):
        """The Where node uses the original If condition as its first input."""
        model = _build_if_model()
        cond_value = next(v for v in model.graph.inputs if v.name == "cond")

        decompose_if_pass()(model)

        where_node = next(n for n in model.graph if n.op_type == "Where")
        assert where_node.inputs[0] is cond_value

    def test_graph_outputs_updated(self):
        """The model's graph outputs point to Where results, not If outputs."""
        model = _build_if_model()
        decompose_if_pass()(model)

        for out in model.graph.outputs:
            producer = out.producer()
            assert producer is not None
            assert producer.op_type == "Where", (
                f"Graph output should come from Where, got {producer.op_type}"
            )
