# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

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

    def test_missing_const_value_leaves_none_on_folded(self):
        """When source const_value is None, folded initializer const_value is also None.

        The fold is still structural (Transpose node removed, weight_t registered)
        but value computation is deferred to fold_initializers_after_weights().
        """
        data = np.zeros((4, 3), dtype=np.float32)
        model, weight_val, _ = _make_model_with_transpose(data, [1, 0])

        # Clear const_value to simulate pre-weight-load state
        weight_val.const_value = None

        FoldTransposedInitializerPass()(model)

        t_val = model.graph.initializers.get("weight_t")
        assert t_val is not None, "weight_t should be registered even without data"
        # No data yet — const_value stays None until weights are applied
        assert t_val.const_value is None
        # Source name recorded for later materialisation
        assert t_val.metadata_props.get("_fold_source") == "weight"


class TestFoldTransposeStructural:
    """Structural fold works even when const_value is None (no weights loaded).

    onnxscript's nn.Module registers initializers without const_value (bypassing
    register_initializer's const_value check). The fold passes must handle this
    gracefully: fold the graph structure, set const_value=None, and record source
    info in metadata_props so fold_initializers_after_weights() can materialise.
    """

    def test_folds_before_weights_loaded(self):
        """FoldTransposedInitializerPass folds Transpose even with const_value=None."""
        weight = ir.Value(name="weight")
        weight.shape = ir.Shape([4, 8])
        weight.dtype = ir.DataType.FLOAT
        # No const_value — simulates how onnxscript registers build-time parameters.

        transpose_node = ir.Node(
            "",
            "Transpose",
            inputs=[weight],
            attributes=[ir.Attr("perm", ir.AttributeType.INTS, [1, 0])],
            num_outputs=1,
        )
        transpose_node.outputs[0].shape = ir.Shape([8, 4])
        transpose_node.outputs[0].dtype = ir.DataType.FLOAT

        matmul_node = ir.Node(
            "",
            "MatMul",
            inputs=[ir.Value(name="x"), transpose_node.outputs[0]],
            num_outputs=1,
        )

        graph = ir.Graph(
            inputs=[ir.Value(name="x")],
            outputs=[matmul_node.outputs[0]],
            nodes=[transpose_node, matmul_node],
            name="test",
            opset_imports={"": 20},
        )
        # Use dict assignment (not register_initializer) to bypass const_value check,
        # matching how onnxscript registers build-time parameters.
        graph.initializers["weight"] = weight
        model = ir.Model(graph, ir_version=10)

        result = FoldTransposedInitializerPass()(model)
        assert result.modified

        node_types = [n.op_type for n in model.graph.all_nodes()]
        assert "Transpose" not in node_types, "Transpose should be removed structurally"
        assert "weight_t" in model.graph.initializers, (
            "weight_t initializer should be registered"
        )

        t_val = model.graph.initializers["weight_t"]
        # No data yet — const_value is None until weights are loaded.
        assert t_val.const_value is None
        # Source name recorded for materialisation.
        assert t_val.metadata_props.get("_fold_source") == "weight"

        # The original initializer must stay registered so apply_weights can populate it.
        assert "weight" in model.graph.initializers, (
            "original initializer must stay registered so apply_weights can find it"
        )

    def test_materialises_after_weights_set(self):
        """After weights are set on the source, _materialize_deferred_initializers fills in weight_t."""
        from mobius._optimizations import _materialize_deferred_initializers

        weight = ir.Value(name="weight")
        weight.shape = ir.Shape([4, 8])
        weight.dtype = ir.DataType.FLOAT

        transpose_node = ir.Node(
            "",
            "Transpose",
            inputs=[weight],
            attributes=[ir.Attr("perm", ir.AttributeType.INTS, [1, 0])],
            num_outputs=1,
        )
        transpose_node.outputs[0].shape = ir.Shape([8, 4])
        transpose_node.outputs[0].dtype = ir.DataType.FLOAT

        matmul_node = ir.Node(
            "",
            "MatMul",
            inputs=[ir.Value(name="x"), transpose_node.outputs[0]],
            num_outputs=1,
        )

        graph = ir.Graph(
            inputs=[ir.Value(name="x")],
            outputs=[matmul_node.outputs[0]],
            nodes=[transpose_node, matmul_node],
            name="test",
            opset_imports={"": 20},
        )
        graph.initializers["weight"] = weight
        model = ir.Model(graph, ir_version=10)

        FoldTransposedInitializerPass()(model)

        # Simulate weight loading: set const_value on the original.
        data = np.arange(32, dtype=np.float32).reshape(4, 8)
        model.graph.initializers["weight"].const_value = ir.tensor(data)

        # Materialise deferred values.
        _materialize_deferred_initializers(model)

        np.testing.assert_array_equal(
            model.graph.initializers["weight_t"].const_value.numpy(), data.T
        )
