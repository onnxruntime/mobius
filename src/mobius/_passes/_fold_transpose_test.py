# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for FoldTransposedInitializerPass."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from mobius._passes._fold_transpose import FoldTransposedInitializerPass


def _make_model_with_transpose(
    weight_data: np.ndarray,
    perm: list[int],
) -> tuple[ir.Model, ir.Value, ir.Node]:
    """Helper: build a small model with Transpose(weight, perm=perm) → MatMul(x, w_t)."""
    weight_val = ir.Value(name="weight")
    weight_val.shape = ir.Shape(list(weight_data.shape))
    weight_val.dtype = ir.DataType.FLOAT
    weight_val.const_value = ir.tensor(weight_data)

    # MatMul contract: x[batch, K] @ w_t[K, N] → out[batch, N].
    # weight_data has shape (K, N); w_t = weight_data.T has shape (N, K) — wait,
    # actually perm=[1,0] swaps axes: w_t.shape = [weight_data.shape[1], weight_data.shape[0]].
    # For MatMul(x, w_t) to be valid x's last dim must equal w_t.shape[0] = weight_data.shape[1].
    x = ir.Value(name="x")
    x.shape = ir.Shape([2, weight_data.shape[1]])
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
        model, _, _ = _make_model_with_transpose(data, [1, 0])

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

    def test_shared_initializer_not_folded_when_two_transposes(self):
        """Initializer shared by two Transpose nodes is skipped by the pass.

        Each Transpose sees len(inp.uses()) == 2, so both are conservatively
        skipped to avoid corrupting the other consumer.
        """
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

        result = FoldTransposedInitializerPass()(model)

        # Both Transposes are skipped — initializer has 2 users.
        assert not result.modified
        t_inits = [k for k in model.graph.initializers if k.endswith("_t")]
        assert len(t_inits) == 0

    def test_idempotent(self):
        """Running the pass twice does not create duplicate initializers."""
        data = np.arange(12, dtype=np.float32).reshape(4, 3)
        model, _, _ = _make_model_with_transpose(data, perm=[1, 0])

        FoldTransposedInitializerPass()(model)
        result2 = FoldTransposedInitializerPass()(model)
        # Second run: no Transpose nodes left → not modified
        assert not result2.modified
        t_inits = [k for k in model.graph.initializers if k.endswith("_t")]
        assert len(t_inits) == 1, "Should have exactly one folded initializer after two runs"

    def test_non_none_value_uses_lazy_tensor(self):
        """The folded initializer wraps the source in a LazyTensor.

        LazyTensor defers the transpose computation until serialization, avoiding
        holding a second full copy of the weight in memory.
        """
        data = np.arange(12, dtype=np.float32).reshape(4, 3)
        model, _, _ = _make_model_with_transpose(data, perm=[1, 0])

        FoldTransposedInitializerPass()(model)

        t_val = model.graph.initializers["weight_t"]
        assert isinstance(t_val.const_value, ir.LazyTensor), (
            "Folded initializer should use LazyTensor for deferred computation, "
            f"got {type(t_val.const_value)}"
        )
        # Value must still be numerically correct when materialised.
        np.testing.assert_array_equal(t_val.const_value.numpy(), data.T)

    def test_identity_perm_not_folded(self):
        """Transpose with perm=[0,1] (identity) on a 2D weight is NOT folded.

        Only perm=[1,0] (the true matrix transpose) is eligible for folding.
        """
        data = np.zeros((4, 3), dtype=np.float32)
        weight_val = ir.Value(name="weight_id")
        weight_val.shape = ir.Shape([4, 3])
        weight_val.dtype = ir.DataType.FLOAT
        weight_val.const_value = ir.tensor(data)

        x = ir.Value(name="x")
        transpose_node = ir.Node(
            "",
            "Transpose",
            inputs=[weight_val],
            attributes=[ir.Attr("perm", ir.AttributeType.INTS, [0, 1])],
            num_outputs=1,
        )
        out = transpose_node.outputs[0]
        out.shape = ir.Shape([4, 3])
        out.dtype = ir.DataType.FLOAT

        graph = ir.Graph(
            inputs=[x],
            outputs=[out],
            nodes=[transpose_node],
            name="test_graph",
            opset_imports={"": 20},
        )
        graph.register_initializer(weight_val)
        model = ir.Model(graph, ir_version=10)

        result = FoldTransposedInitializerPass()(model)
        assert not result.modified
        assert "weight_id_t" not in model.graph.initializers
        node_types = [n.op_type for n in model.graph.all_nodes()]
        assert "Transpose" in node_types, "Identity Transpose should not be removed"

    def test_transpose_with_zero_inputs_skipped(self):
        """A Transpose node with no inputs does not cause an error and is not folded."""
        # Build a Transpose node with no inputs (malformed but should be skipped gracefully).
        x = ir.Value(name="x")
        transpose_node = ir.Node("", "Transpose", inputs=[], num_outputs=1)
        matmul_node = ir.Node(
            "", "MatMul", inputs=[x, transpose_node.outputs[0]], num_outputs=1
        )
        graph = ir.Graph(
            inputs=[x],
            outputs=[matmul_node.outputs[0]],
            nodes=[transpose_node, matmul_node],
            name="test_graph",
            opset_imports={"": 20},
        )
        model = ir.Model(graph, ir_version=10)
        result = FoldTransposedInitializerPass()(model)
        # Nothing folded — the node has no inputs so it is skipped.
        assert not result.modified

    def test_transpose_with_none_input_skipped(self):
        """A Transpose node whose first input is None is skipped gracefully."""
        # Construct a Transpose node that has an input slot but the slot is None
        # (can happen with optional inputs in some graph constructions).
        x = ir.Value(name="x")
        # Use a value with no name to trigger the inp.name is None guard.
        nameless = ir.Value()  # name defaults to None
        transpose_node = ir.Node(
            "",
            "Transpose",
            inputs=[nameless],
            attributes=[ir.Attr("perm", ir.AttributeType.INTS, [1, 0])],
            num_outputs=1,
        )
        matmul_node = ir.Node(
            "", "MatMul", inputs=[x, transpose_node.outputs[0]], num_outputs=1
        )
        graph = ir.Graph(
            inputs=[x],
            outputs=[matmul_node.outputs[0]],
            nodes=[transpose_node, matmul_node],
            name="test_graph",
            opset_imports={"": 20},
        )
        model = ir.Model(graph, ir_version=10)
        result = FoldTransposedInitializerPass()(model)
        assert not result.modified

    def test_shape_derived_from_input_when_output_shape_missing(self):
        """When the Transpose output has no shape, shape is derived from the input."""
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        weight_val = ir.Value(name="weight")
        weight_val.shape = ir.Shape([2, 3])
        weight_val.dtype = ir.DataType.FLOAT
        weight_val.const_value = ir.tensor(data)

        x = ir.Value(name="x")
        transpose_node = ir.Node(
            "",
            "Transpose",
            inputs=[weight_val],
            attributes=[ir.Attr("perm", ir.AttributeType.INTS, [1, 0])],
            num_outputs=1,
        )
        # Deliberately leave output shape as None (no shape inference).
        w_t = transpose_node.outputs[0]
        w_t.dtype = ir.DataType.FLOAT
        # w_t.shape is not set → None

        matmul_node = ir.Node("", "MatMul", inputs=[x, w_t], num_outputs=1)
        graph = ir.Graph(
            inputs=[x],
            outputs=[matmul_node.outputs[0]],
            nodes=[transpose_node, matmul_node],
            name="test_graph",
            opset_imports={"": 20},
        )
        graph.register_initializer(weight_val)
        model = ir.Model(graph, ir_version=10)

        result = FoldTransposedInitializerPass()(model)

        assert result.modified
        assert "weight_t" in model.graph.initializers
        # Shape should be derived from input shape via perm=[1,0]: (2,3) → (3,2).
        w_t_init = model.graph.initializers["weight_t"]
        assert list(w_t_init.shape) == [3, 2]


def test_shared_initializer_not_folded():
    """Transpose(initializer) is skipped when the initializer has multiple consumers.

    Folding would rename/remove the shared initializer, silently breaking the
    other consumers. The pass must leave the graph unchanged in this case.
    """
    weight_data = np.ones((4, 8), dtype=np.float32)
    weight_val = ir.Value(name="weight")
    weight_val.shape = ir.Shape([4, 8])
    weight_val.dtype = ir.DataType.FLOAT
    weight_val.const_value = ir.tensor(weight_data)

    x = ir.Value(name="x")
    x.shape = ir.Shape([2, 4])
    x.dtype = ir.DataType.FLOAT

    # Consumer 1: Transpose(weight) → MatMul(x, w_t)
    transpose_node = ir.Node(
        "",
        "Transpose",
        inputs=[weight_val],
        attributes=[ir.Attr("perm", ir.AttributeType.INTS, [1, 0])],
        num_outputs=1,
    )
    w_t = transpose_node.outputs[0]
    w_t.shape = ir.Shape([8, 4])
    w_t.dtype = ir.DataType.FLOAT
    matmul_node = ir.Node("", "MatMul", inputs=[x, w_t], num_outputs=1)

    # Consumer 2: another MatMul that uses the raw (untransposed) weight directly,
    # so the initializer has two uses: (transpose_node, 0) and (matmul2_node, 1).
    x2 = ir.Value(name="x2")
    x2.shape = ir.Shape([2, 8])
    x2.dtype = ir.DataType.FLOAT
    matmul2_node = ir.Node("", "MatMul", inputs=[x2, weight_val], num_outputs=1)

    graph = ir.Graph(
        inputs=[x, x2],
        outputs=[matmul_node.outputs[0], matmul2_node.outputs[0]],
        nodes=[transpose_node, matmul_node, matmul2_node],
        name="test_graph",
        opset_imports={"": 20},
    )
    graph.initializers["weight"] = weight_val

    model = ir.Model(graph, ir_version=10)

    # The initializer has 2 uses: Transpose and MatMul2 — folding must be skipped.
    assert len(weight_val.uses()) == 2

    result = FoldTransposedInitializerPass()(model)

    # Pass must report no changes and the Transpose node must remain.
    assert not result.modified
    assert "weight_t" not in model.graph.initializers
    transpose_nodes = [n for n in model.graph.all_nodes() if n.op_type == "Transpose"]
    assert len(transpose_nodes) == 1
