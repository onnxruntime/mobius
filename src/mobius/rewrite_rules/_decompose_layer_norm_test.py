# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for decompose_simplified_layer_norm_rules().

Verifies that the com.microsoft::SimplifiedLayerNormalization custom op
is correctly decomposed into primitive ONNX ops (Pow, ReduceMean, Add,
Sqrt, Div, Mul).
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius.rewrite_rules._decompose_layer_norm import (
    decompose_simplified_layer_norm_rules,
)
from mobius.rewrite_rules._testing_utils import count_ops


def _make_model_with_simplified_layer_norm(
    hidden_size: int = 64,
    epsilon: float = 1e-6,
) -> ir.Model:
    """Build a minimal model containing SimplifiedLayerNormalization."""
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT))
    x.shape = ir.Shape([1, 3, hidden_size])

    weight = ir.Value(name="weight", type=ir.TensorType(ir.DataType.FLOAT))
    weight.shape = ir.Shape([hidden_size])

    sln_node = ir.Node(
        domain="com.microsoft",
        op_type="SimplifiedLayerNormalization",
        inputs=[x, weight],
        attributes=[ir.Attr("epsilon", ir.AttributeType.FLOAT, epsilon)],
        num_outputs=1,
        name="sln_0",
    )
    sln_node.outputs[0].name = "output"
    sln_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    sln_node.outputs[0].shape = ir.Shape([1, 3, hidden_size])

    graph = ir.Graph(
        [x, weight],
        [sln_node.outputs[0]],
        nodes=[sln_node],
        name="test_graph",
        opset_imports={"": 23, "com.microsoft": 1},
    )

    return ir.Model(graph, ir_version=10)


class TestDecomposeSimplifiedLayerNormRules:
    def test_rules_returns_rule_set(self):
        rules = decompose_simplified_layer_norm_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_decomposes_to_primitive_ops(self):
        """SimplifiedLayerNormalization decomposes into Pow, ReduceMean, etc."""
        model = _make_model_with_simplified_layer_norm()
        counts_before = count_ops(model)
        assert counts_before.get("SimplifiedLayerNormalization", 0) == 1

        rewrite(model, pattern_rewrite_rules=decompose_simplified_layer_norm_rules())

        counts_after = count_ops(model)
        assert counts_after.get("SimplifiedLayerNormalization", 0) == 0
        # Should have the primitive RMSNorm ops
        assert counts_after.get("Pow", 0) >= 1
        assert counts_after.get("ReduceMean", 0) >= 1
        assert counts_after.get("Sqrt", 0) >= 1
        assert counts_after.get("Div", 0) >= 1
        assert counts_after.get("Mul", 0) >= 1

    def test_no_op_when_no_simplified_ln(self):
        """Rule is a no-op when model has no SimplifiedLayerNormalization."""
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT))
        x.shape = ir.Shape([1, 3, 64])

        weight = ir.Value(name="weight", type=ir.TensorType(ir.DataType.FLOAT))
        weight.shape = ir.Shape([64])

        ln_node = ir.Node(
            domain="",
            op_type="LayerNormalization",
            inputs=[x, weight],
            attributes=[ir.Attr("epsilon", ir.AttributeType.FLOAT, 1e-6)],
            num_outputs=1,
            name="ln_0",
        )
        ln_node.outputs[0].name = "output"
        ln_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
        ln_node.outputs[0].shape = ir.Shape([1, 3, 64])

        graph = ir.Graph(
            [x, weight],
            [ln_node.outputs[0]],
            nodes=[ln_node],
            name="test_graph",
            opset_imports={"": 23},
        )
        model = ir.Model(graph, ir_version=10)

        counts_before = count_ops(model)
        rewrite(model, pattern_rewrite_rules=decompose_simplified_layer_norm_rules())
        counts_after = count_ops(model)

        # No change — LayerNormalization should not be touched
        assert counts_after["LayerNormalization"] == counts_before["LayerNormalization"]
