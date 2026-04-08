# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for FoldTransposedInitializerPass."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

from mobius.passes._fold_transpose import FoldTransposedInitializerPass


def _make_model_with_transpose(
    weight_data: np.ndarray,
    perm: list[int],
) -> tuple[ir.Model, ir.Value, ir.Node]:
    """Helper: build a small model with Transpose(weight, perm=perm) → MatMul(x, w_t)."""
    weight_val = ir.Value(name="weight")
    weight_val.shape = ir.Shape(list(weight_data.shape))
    weight_val.dtype = ir.DataType.FLOAT
    weight_val.const_value = ir.tensor(weight_data)

    x = ir.Value(name="x")
    x.shape = ir.Shape([2, weight_data.shape[0]])
    x.dtype = ir.DataType.FLOAT

    transpose_node = ir.Node(
        "",
        "Transpose",
        inputs=[weight_val],
        attributes=[ir.Attr("perm", ir.AttributeType.INTS, perm)],
        num_outputs=1,
    )
    w_t = transpose_node.outputs[0]
    w_t.shape = ir.Shape([weight_data.shape[1], weight_data.shape[0]])
    w_t.dtype = ir.DataType.FLOAT

    matmul_node = ir.Node("", "MatMul", inputs=[x, w_t], num_outputs=1)
    out = matmul_node.outputs[0]

    graph = ir.Graph(
        inputs=[x],
        outputs=[out],
        nodes=[transpose_node, matmul_node],
        name="test_graph",
        opset_imports={"": 20},
    )
    graph.register_initializer(weight_val)

    model = ir.Model(graph, ir_version=10)
    return model, weight_val, transpose_node


class TestFoldTransposedInitializerPass:
    def test_folds_2d_transpose(self):
        """Transpose(weight, perm=[1,0]) is replaced by a pre-transposed initializer."""
        data = np.arange(12, dtype=np.float32).reshape(4, 3)
        model, weight_val, transpose_node = _make_model_with_transpose(data, [1, 0])

        result = FoldTransposedInitializerPass()(model)
        assert result.modified

        # Transpose node must be gone
        node_types = [n.op_type for n in model.graph.all_nodes()]
        assert "Transpose" not in node_types

        # New initializer registered
        assert "weight_t" in model.graph.initializers

        # New initializer has correct transposed shape
        t_val = model.graph.initializers["weight_t"]
        assert t_val.shape == ir.Shape([3, 4])

        # Lazy data matches numpy transpose
        np.testing.assert_array_equal(t_val.const_value.numpy(), data.T)

    def test_matmul_now_uses_precomputed_weight(self):
        """After folding, MatMul input is the pre-transposed initializer, not Transpose output."""
        data = np.ones((4, 3), dtype=np.float32)
        model, _, _ = _make_model_with_transpose(data, [1, 0])

        FoldTransposedInitializerPass()(model)

        matmul_nodes = [n for n in model.graph.all_nodes() if n.op_type == "MatMul"]
        assert len(matmul_nodes) == 1
        b_input = matmul_nodes[0].inputs[1]
        assert b_input is not None
        assert b_input.name == "weight_t"

    def test_non_10_perm_not_folded(self):
        """Transpose with perm != [1, 0] is NOT folded."""
        data = np.zeros((2, 3, 4), dtype=np.float32)
        weight_val = ir.Value(name="weight3d")
        weight_val.shape = ir.Shape([2, 3, 4])
        weight_val.dtype = ir.DataType.FLOAT
        weight_val.const_value = ir.tensor(data)

        x = ir.Value(name="x")
        transpose_node = ir.Node(
            "",
            "Transpose",
            inputs=[weight_val],
            attributes=[ir.Attr("perm", ir.AttributeType.INTS, [0, 2, 1])],
            num_outputs=1,
        )
        w_t = transpose_node.outputs[0]
        graph = ir.Graph(
            inputs=[x],
            outputs=[w_t],
            nodes=[transpose_node],
            name="test_graph",
            opset_imports={"": 20},
        )
        graph.register_initializer(weight_val)
        model = ir.Model(graph, ir_version=10)

        result = FoldTransposedInitializerPass()(model)
        assert not result.modified
        assert "weight3d_t" not in model.graph.initializers

    def test_non_initializer_not_folded(self):
        """Transpose whose input is a graph input (not initializer) is NOT folded."""
        x = ir.Value(name="x")
        x.shape = ir.Shape([4, 3])
        x.dtype = ir.DataType.FLOAT

        transpose_node = ir.Node(
            "",
            "Transpose",
            inputs=[x],
            attributes=[ir.Attr("perm", ir.AttributeType.INTS, [1, 0])],
            num_outputs=1,
        )
        out = transpose_node.outputs[0]
        graph = ir.Graph(
            inputs=[x],
            outputs=[out],
            nodes=[transpose_node],
            name="test_graph",
            opset_imports={"": 20},
        )
        model = ir.Model(graph, ir_version=10)

        result = FoldTransposedInitializerPass()(model)
        assert not result.modified

    def test_shared_initializer_creates_one_precomputed(self):
        """Two Transpose nodes on the same initializer share a single folded result."""
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        weight_val = ir.Value(name="shared_weight")
        weight_val.shape = ir.Shape([2, 3])
        weight_val.dtype = ir.DataType.FLOAT
        weight_val.const_value = ir.tensor(data)

        x = ir.Value(name="x")

        t1 = ir.Node(
            "",
            "Transpose",
            inputs=[weight_val],
            attributes=[ir.Attr("perm", ir.AttributeType.INTS, [1, 0])],
            num_outputs=1,
        )
        t1.outputs[0].shape = ir.Shape([3, 2])
        t1.outputs[0].dtype = ir.DataType.FLOAT

        t2 = ir.Node(
            "",
            "Transpose",
            inputs=[weight_val],
            attributes=[ir.Attr("perm", ir.AttributeType.INTS, [1, 0])],
            num_outputs=1,
        )
        t2.outputs[0].shape = ir.Shape([3, 2])
        t2.outputs[0].dtype = ir.DataType.FLOAT

        graph = ir.Graph(
            inputs=[x],
            outputs=[t1.outputs[0], t2.outputs[0]],
            nodes=[t1, t2],
            name="test_graph",
            opset_imports={"": 20},
        )
        graph.register_initializer(weight_val)
        model = ir.Model(graph, ir_version=10)

        FoldTransposedInitializerPass()(model)

        # Only one new initializer, not two
        t_inits = [k for k in model.graph.initializers if k.endswith("_t")]
        assert len(t_inits) == 1

    def test_missing_const_value_raises_on_access(self):
        """LazyTensor raises AssertionError when accessed before weights are loaded."""
        data = np.zeros((4, 3), dtype=np.float32)
        model, weight_val, _ = _make_model_with_transpose(data, [1, 0])

        # Clear const_value to simulate pre-weight-load state
        weight_val.const_value = None

        FoldTransposedInitializerPass()(model)

        t_val = model.graph.initializers.get("weight_t")
        assert t_val is not None
        with pytest.raises(AssertionError, match="no const_value"):
            _ = t_val.const_value.numpy()
