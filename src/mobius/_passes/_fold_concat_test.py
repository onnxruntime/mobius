# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for FoldConcatInitializersPass."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from mobius._passes._fold_concat import FoldConcatInitializersPass


def _make_concat_model(
    arrays: list[np.ndarray],
    axis: int = 0,
    add_dynamic_input: bool = False,
) -> tuple[ir.Model, list[ir.Value]]:
    """Build a model with a Concat node over initializers."""
    init_vals: list[ir.Value] = []
    for i, arr in enumerate(arrays):
        v = ir.Value(name=f"init_{i}")
        v.shape = ir.Shape(list(arr.shape))
        v.dtype = ir.DataType.FLOAT
        v.const_value = ir.tensor(arr)
        init_vals.append(v)

    inputs_to_concat: list[ir.Value] = list(init_vals)
    if add_dynamic_input:
        dyn = ir.Value(name="dynamic")
        dyn.shape = ir.Shape(list(arrays[0].shape))
        dyn.dtype = ir.DataType.FLOAT
        inputs_to_concat.append(dyn)

    concat_node = ir.Node(
        "",
        "Concat",
        inputs=inputs_to_concat,
        attributes=[ir.Attr("axis", ir.AttributeType.INT, axis)],
        num_outputs=1,
    )
    out = concat_node.outputs[0]
    cat_shape = list(arrays[0].shape)
    cat_shape[axis] = sum(a.shape[axis] for a in arrays)
    if not add_dynamic_input:
        out.shape = ir.Shape(cat_shape)
    out.dtype = ir.DataType.FLOAT

    graph_inputs = [ir.Value(name="x")]
    if add_dynamic_input:
        graph_inputs.append(inputs_to_concat[-1])  # type: ignore[arg-type]

    graph = ir.Graph(
        inputs=graph_inputs,
        outputs=[out],
        nodes=[concat_node],
        name="test_graph",
        opset_imports={"": 20},
    )
    for v in init_vals:
        graph.register_initializer(v)

    model = ir.Model(graph, ir_version=10)
    return model, init_vals


class TestFoldConcatInitializersPass:
    def test_folds_two_initializers_axis0(self):
        """Concat of two 2-D initializers along axis=0 is folded."""
        a = np.ones((4, 8), dtype=np.float32)
        b = np.ones((4, 8), dtype=np.float32) * 2
        model, _ = _make_concat_model([a, b], axis=0)

        result = FoldConcatInitializersPass()(model)
        assert result.modified

        node_types = [n.op_type for n in model.graph.all_nodes()]
        assert "Concat" not in node_types

        packed_name = "init_0__init_1__axis_0__concat"
        assert packed_name in model.graph.initializers

        packed = model.graph.initializers[packed_name]
        assert packed.shape == ir.Shape([8, 8])
        expected = np.concatenate([a, b], axis=0)
        np.testing.assert_array_equal(packed.const_value.numpy(), expected)

    def test_folds_three_qkv_initializers(self):
        """Concat of three weight matrices (Q, K, V) is packed into one."""
        q = np.arange(12, dtype=np.float32).reshape(4, 3)
        k = np.arange(12, dtype=np.float32).reshape(4, 3) + 12
        v = np.arange(12, dtype=np.float32).reshape(4, 3) + 24
        model, _ = _make_concat_model([q, k, v], axis=0)

        FoldConcatInitializersPass()(model)

        packed_name = "init_0__init_1__init_2__axis_0__concat"
        assert packed_name in model.graph.initializers
        packed = model.graph.initializers[packed_name]
        assert packed.shape == ir.Shape([12, 3])
        np.testing.assert_array_equal(
            packed.const_value.numpy(), np.concatenate([q, k, v], axis=0)
        )

    def test_folds_along_axis1(self):
        """Concat along axis=1 is correctly folded."""
        a = np.ones((4, 3), dtype=np.float32)
        b = np.ones((4, 5), dtype=np.float32)
        model, _ = _make_concat_model([a, b], axis=1)

        FoldConcatInitializersPass()(model)

        packed_name = "init_0__init_1__axis_1__concat"
        packed = model.graph.initializers[packed_name]
        np.testing.assert_array_equal(
            packed.const_value.numpy(), np.concatenate([a, b], axis=1)
        )

    def test_dynamic_input_not_folded(self):
        """Concat with a dynamic (non-initializer) input is NOT folded."""
        a = np.ones((4, 3), dtype=np.float32)
        model, _ = _make_concat_model([a], axis=0, add_dynamic_input=True)

        result = FoldConcatInitializersPass()(model)
        assert not result.modified

    def test_folded_inputs_detached_so_dce_strips_dead_weights(self):
        """After folding, the source initializers must be detached and DCE-able.

        FoldConcat removes the Concat node; if it does not detach the node from
        its inputs (``safe=True``), the source q/k/v initializers keep a stale
        use pointing at the removed node, so RemoveUnusedNodesPass cannot strip
        them. For fp16 GQA that leaves ~1.8 GB of dead pre-pack weights in the
        exported model.
        """
        from onnx_ir.passes import common as common_passes

        a = np.ones((4, 8), dtype=np.float16)
        b = np.ones((4, 8), dtype=np.float16) * 2
        model, init_vals = _make_concat_model([a, b], axis=0)
        # The Concat output is the only graph output; route it through a MatMul
        # so the packed initializer stays live while the sources become dead.
        packed_out = model.graph.outputs[0]
        hidden = ir.Value(
            name="hidden",
            shape=ir.Shape(["N", 8]),
            type=ir.TensorType(ir.DataType.FLOAT16),
        )
        matmul = ir.Node("", "MatMul", inputs=[hidden, packed_out], num_outputs=1)
        mm_out = matmul.outputs[0]
        mm_out.shape = ir.Shape(["N", 8])
        mm_out.dtype = ir.DataType.FLOAT16
        model.graph.append(matmul)
        model.graph.inputs.append(hidden)
        model.graph.outputs[0] = mm_out

        FoldConcatInitializersPass()(model)

        # Sources must be detached (zero uses) so DCE can remove them.
        for v in init_vals:
            assert len(list(v.uses())) == 0, (
                f"{v.name} still has uses after fold — node not detached"
            )

        common_passes.RemoveUnusedNodesPass()(model)
        remaining = set(model.graph.initializers)
        assert "init_0" not in remaining and "init_1" not in remaining, (
            f"Dead pre-pack weights survived DCE: {remaining}"
        )
        assert "init_0__init_1__axis_0__concat" in remaining, (
            "the live packed result must NOT be stripped by DCE: "
            f"{remaining}"
        )

    def test_uses_lazy_tensor(self):
        """The packed initializer wraps sources in a LazyTensor.

        LazyTensor defers np.concatenate until serialization, so the individual
        source weights and the packed weight are not both in memory simultaneously.
        """
        a = np.ones((4, 8), dtype=np.float32)
        b = np.ones((4, 8), dtype=np.float32) * 2
        model, _ = _make_concat_model([a, b], axis=0)

        FoldConcatInitializersPass()(model)

        packed = model.graph.initializers["init_0__init_1__axis_0__concat"]
        assert isinstance(packed.const_value, ir.LazyTensor), (
            "Packed initializer should use LazyTensor for deferred computation, "
            f"got {type(packed.const_value)}"
        )
        # Value must still be numerically correct when materialised.
        np.testing.assert_array_equal(
            packed.const_value.numpy(), np.concatenate([a, b], axis=0)
        )

    def test_idempotent(self):
        """Running the pass twice does not create duplicate initializers."""
        a = np.ones((4, 3), dtype=np.float32)
        b = np.ones((4, 3), dtype=np.float32)
        model, _ = _make_concat_model([a, b], axis=0)

        FoldConcatInitializersPass()(model)
        result2 = FoldConcatInitializersPass()(model)
        # Second run: no Concat nodes left → not modified
        assert not result2.modified

    def test_mixed_dtype_not_folded(self):
        """Concat of initializers with different dtypes is skipped with a warning."""
        a = np.ones((4, 3), dtype=np.float32)
        b = np.ones((4, 3), dtype=np.float16)
        model, init_vals = _make_concat_model([a, b], axis=0)
        # Assign mismatched dtypes explicitly so the pass sees them.
        init_vals[0].dtype = ir.DataType.FLOAT
        init_vals[1].dtype = ir.DataType.FLOAT16

        result = FoldConcatInitializersPass()(model)
        assert not result.modified

        node_types = [n.op_type for n in model.graph.all_nodes()]
        assert "Concat" in node_types, "Mixed-dtype Concat should not be removed"

    def test_same_inputs_different_axis_both_folded(self):
        """Same inputs concatenated along two different axes produce distinct packed names."""
        a = np.ones((4, 3), dtype=np.float32)
        b = np.ones((4, 3), dtype=np.float32)

        # Build a model with TWO Concat nodes over the same initializers along axis=0
        # and axis=1 respectively.  They must produce different packed names.
        init_0 = ir.Value(name="init_0")
        init_0.shape = ir.Shape([4, 3])
        init_0.dtype = ir.DataType.FLOAT
        init_0.const_value = ir.tensor(a)

        init_1 = ir.Value(name="init_1")
        init_1.shape = ir.Shape([4, 3])
        init_1.dtype = ir.DataType.FLOAT
        init_1.const_value = ir.tensor(b)

        concat0 = ir.Node(
            "",
            "Concat",
            inputs=[init_0, init_1],
            attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
            num_outputs=1,
        )
        concat0.outputs[0].shape = ir.Shape([8, 3])
        concat0.outputs[0].dtype = ir.DataType.FLOAT

        concat1 = ir.Node(
            "",
            "Concat",
            inputs=[init_0, init_1],
            attributes=[ir.Attr("axis", ir.AttributeType.INT, 1)],
            num_outputs=1,
        )
        concat1.outputs[0].shape = ir.Shape([4, 6])
        concat1.outputs[0].dtype = ir.DataType.FLOAT

        graph = ir.Graph(
            inputs=[ir.Value(name="x")],
            outputs=[concat0.outputs[0], concat1.outputs[0]],
            nodes=[concat0, concat1],
            name="test_graph",
            opset_imports={"": 20},
        )
        graph.register_initializer(init_0)
        graph.register_initializer(init_1)
        model = ir.Model(graph, ir_version=10)

        result = FoldConcatInitializersPass()(model)
        assert result.modified

        assert "init_0__init_1__axis_0__concat" in model.graph.initializers
        assert "init_0__init_1__axis_1__concat" in model.graph.initializers
        node_types = [n.op_type for n in model.graph.all_nodes()]
        assert "Concat" not in node_types

    def test_duplicate_concat_nodes_reuse_packed_initializer(self):
        """Two identical Concat nodes reuse the same packed initializer."""
        a = np.ones((4, 3), dtype=np.float32)
        b = np.ones((4, 3), dtype=np.float32)

        init_0 = ir.Value(name="init_0")
        init_0.shape = ir.Shape([4, 3])
        init_0.dtype = ir.DataType.FLOAT
        init_0.const_value = ir.tensor(a)

        init_1 = ir.Value(name="init_1")
        init_1.shape = ir.Shape([4, 3])
        init_1.dtype = ir.DataType.FLOAT
        init_1.const_value = ir.tensor(b)

        concat0 = ir.Node(
            "",
            "Concat",
            inputs=[init_0, init_1],
            attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
            num_outputs=1,
        )
        concat0.outputs[0].shape = ir.Shape([8, 3])
        concat0.outputs[0].dtype = ir.DataType.FLOAT

        concat1 = ir.Node(
            "",
            "Concat",
            inputs=[init_0, init_1],
            attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
            num_outputs=1,
        )
        concat1.outputs[0].shape = ir.Shape([8, 3])
        concat1.outputs[0].dtype = ir.DataType.FLOAT

        graph = ir.Graph(
            inputs=[ir.Value(name="x")],
            outputs=[concat0.outputs[0], concat1.outputs[0]],
            nodes=[concat0, concat1],
            name="test_graph",
            opset_imports={"": 20},
        )
        graph.register_initializer(init_0)
        graph.register_initializer(init_1)
        model = ir.Model(graph, ir_version=10)

        result = FoldConcatInitializersPass()(model)
        assert result.modified

        # Only one packed initializer should be registered.
        packed_keys = [k for k in model.graph.initializers if "__concat" in k]
        assert len(packed_keys) == 1
        node_types = [n.op_type for n in model.graph.all_nodes()]
        assert "Concat" not in node_types


class TestConcatThenTransposeFolding:
    """Verify FoldConcat + FoldTranspose eliminates the Concat+Transpose pattern from PackQKV."""

    def test_concat_then_transpose_fully_folded(self):
        """FoldConcat followed by FoldTranspose eliminates Concat AND Transpose.

        PackQKV emits: Concat(W_q, W_k, W_v) → Transpose(concat_out).
        Running FoldConcat first folds Concat → packed_init.
        Running FoldTranspose second folds Transpose(packed_init) → packed_t_init.
        No runtime Concat or Transpose should remain.
        """
        from mobius._passes._fold_transpose import FoldTransposedInitializerPass

        w_q = np.random.randn(4, 8).astype(np.float32)
        w_k = np.random.randn(4, 8).astype(np.float32)
        w_v = np.random.randn(4, 8).astype(np.float32)

        q_init = ir.Value(name="q_weight")
        q_init.shape = ir.Shape([4, 8])
        q_init.dtype = ir.DataType.FLOAT
        q_init.const_value = ir.tensor(w_q)

        k_init = ir.Value(name="k_weight")
        k_init.shape = ir.Shape([4, 8])
        k_init.dtype = ir.DataType.FLOAT
        k_init.const_value = ir.tensor(w_k)

        v_init = ir.Value(name="v_weight")
        v_init.shape = ir.Shape([4, 8])
        v_init.dtype = ir.DataType.FLOAT
        v_init.const_value = ir.tensor(w_v)

        concat_node = ir.Node(
            "",
            "Concat",
            inputs=[q_init, k_init, v_init],
            attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
            num_outputs=1,
        )
        concat_node.outputs[0].shape = ir.Shape([12, 8])
        concat_node.outputs[0].dtype = ir.DataType.FLOAT

        transpose_node = ir.Node(
            "",
            "Transpose",
            inputs=[concat_node.outputs[0]],
            attributes=[ir.Attr("perm", ir.AttributeType.INTS, [1, 0])],
            num_outputs=1,
        )
        transpose_node.outputs[0].shape = ir.Shape([8, 12])
        transpose_node.outputs[0].dtype = ir.DataType.FLOAT

        graph = ir.Graph(
            inputs=[ir.Value(name="x")],
            outputs=[transpose_node.outputs[0]],
            nodes=[concat_node, transpose_node],
            name="qkv_graph",
            opset_imports={"": 20},
        )
        graph.register_initializer(q_init)
        graph.register_initializer(k_init)
        graph.register_initializer(v_init)
        model = ir.Model(graph, ir_version=10)

        # Run FoldConcat first, then FoldTranspose (correct order).
        FoldConcatInitializersPass()(model)
        FoldTransposedInitializerPass()(model)

        node_types = [n.op_type for n in model.graph.all_nodes()]
        assert "Concat" not in node_types, "Concat should be folded"
        assert "Transpose" not in node_types, "Transpose should be folded"

        # Exactly one packed+transposed initializer should remain.
        packed_t = [k for k in model.graph.initializers if "_t" in k and "__concat" in k]
        assert len(packed_t) == 1, f"Expected 1 packed+transposed init, got: {packed_t}"

        # Verify the actual numeric content is correct.
        expected = np.concatenate([w_q, w_k, w_v], axis=0).T  # shape (8, 12)
        np.testing.assert_array_equal(
            model.graph.initializers[packed_t[0]].const_value.numpy(), expected
        )


class TestLiftConstantsThenFold:
    """Verify that Constant ops are promoted before fold passes run.

    fold_initializers_after_weights runs LiftConstantsToInitializersPass
    first so that any Constant nodes present after weight loading are visible
    to FoldConcatInitializersPass and FoldTransposedInitializerPass.
    """

    def test_constant_op_is_promoted_before_concat_fold(self):
        """A Constant op feeding a Concat should be promoted and then folded."""
        from onnx_ir.passes import common as common_passes

        w_k = np.random.randn(4, 8).astype(np.float32)
        w_v = np.random.randn(4, 8).astype(np.float32)

        # W_q is delivered as a Constant op node (not an initializer yet).
        constant_node = ir.Node(
            "",
            "Constant",
            inputs=[],
            attributes=[
                ir.Attr(
                    "value",
                    ir.AttributeType.TENSOR,
                    ir.tensor(np.random.randn(4, 8).astype(np.float32)),
                )
            ],
            num_outputs=1,
        )
        constant_node.outputs[0].shape = ir.Shape([4, 8])
        constant_node.outputs[0].dtype = ir.DataType.FLOAT

        k_init = ir.Value(name="k_weight")
        k_init.shape = ir.Shape([4, 8])
        k_init.dtype = ir.DataType.FLOAT
        k_init.const_value = ir.tensor(w_k)

        v_init = ir.Value(name="v_weight")
        v_init.shape = ir.Shape([4, 8])
        v_init.dtype = ir.DataType.FLOAT
        v_init.const_value = ir.tensor(w_v)

        concat_node = ir.Node(
            "",
            "Concat",
            inputs=[constant_node.outputs[0], k_init, v_init],
            attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
            num_outputs=1,
        )
        concat_node.outputs[0].shape = ir.Shape([12, 8])
        concat_node.outputs[0].dtype = ir.DataType.FLOAT

        graph = ir.Graph(
            inputs=[ir.Value(name="x")],
            outputs=[concat_node.outputs[0]],
            nodes=[constant_node, concat_node],
            name="test",
            opset_imports={"": 20},
        )
        graph.register_initializer(k_init)
        graph.register_initializer(v_init)
        model = ir.Model(graph, ir_version=10)

        # Run in the same order as fold_initializers_after_weights.
        ir.passes.PassManager(
            [
                common_passes.LiftConstantsToInitializersPass(),
                FoldConcatInitializersPass(),
            ]
        )(model)

        node_types = [n.op_type for n in model.graph.all_nodes()]
        assert "Constant" not in node_types, "Constant op should be promoted"
        assert "Concat" not in node_types, "Concat should be folded after Lift"


class TestFoldConcatEdgeCases:
    """Cover edge-case guard lines in FoldConcatInitializersPass."""

    def test_concat_with_no_inputs_skipped(self):
        """A Concat node with an empty input list is skipped without error."""
        concat_node = ir.Node(
            "",
            "Concat",
            inputs=[],
            attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
            num_outputs=1,
        )
        x = ir.Value(name="x")
        graph = ir.Graph(
            inputs=[x],
            outputs=[concat_node.outputs[0]],
            nodes=[concat_node],
            name="test_graph",
            opset_imports={"": 20},
        )
        model = ir.Model(graph, ir_version=10)
        result = FoldConcatInitializersPass()(model)
        assert not result.modified

    def test_packed_dtype_follows_const_value_when_declared_dtype_missing(self):
        """Regression: fp16 weights with an unset declared dtype must fold to fp16.

        After ``_cast_module_dtype`` casts parameters to fp16, the resulting
        initializer Values keep an fp16 ``const_value`` but lose their declared
        ``.dtype`` (it becomes ``None``).  The pass must resolve the packed dtype
        from ``const_value`` rather than defaulting to FLOAT, otherwise GQA's
        packed-QKV weight is serialized as fp32 and ORT rejects the model with a
        fp16/fp32 MatMul type-mismatch error.
        """
        q = ir.Value(name="q_weight")
        q.const_value = ir.tensor(np.ones((4, 3), dtype=np.float16))
        k = ir.Value(name="k_weight")
        k.const_value = ir.tensor(np.ones((4, 3), dtype=np.float16))
        v = ir.Value(name="v_weight")
        v.const_value = ir.tensor(np.ones((4, 3), dtype=np.float16))
        # Critically: do NOT set .dtype — declared dtype stays None.
        assert q.dtype is None and k.dtype is None and v.dtype is None

        concat_node = ir.Node(
            "",
            "Concat",
            inputs=[q, k, v],
            attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
            num_outputs=1,
        )
        graph = ir.Graph(
            inputs=[ir.Value(name="x")],
            outputs=[concat_node.outputs[0]],
            nodes=[concat_node],
            name="test_graph",
            opset_imports={"": 20},
        )
        graph.register_initializer(q)
        graph.register_initializer(k)
        graph.register_initializer(v)
        model = ir.Model(graph, ir_version=10)

        result = FoldConcatInitializersPass()(model)
        assert result.modified

        packed = model.graph.initializers["q_weight__k_weight__v_weight__axis_0__concat"]
        assert packed.dtype == ir.DataType.FLOAT16, (
            f"Packed initializer declared dtype should be FLOAT16, got {packed.dtype}"
        )
        assert packed.const_value.dtype == ir.DataType.FLOAT16, (
            "Packed LazyTensor dtype should be FLOAT16, got "
            f"{packed.const_value.dtype}"
        )
        assert packed.const_value.numpy().dtype == np.float16

    def test_shape_computed_from_inputs_when_output_shape_missing(self):
        """When the Concat output has no shape, shape is inferred from input shapes."""
        a = np.ones((4, 3), dtype=np.float32)
        b = np.ones((4, 3), dtype=np.float32)

        init_0 = ir.Value(name="init_0")
        init_0.shape = ir.Shape([4, 3])
        init_0.dtype = ir.DataType.FLOAT
        init_0.const_value = ir.tensor(a)

        init_1 = ir.Value(name="init_1")
        init_1.shape = ir.Shape([4, 3])
        init_1.dtype = ir.DataType.FLOAT
        init_1.const_value = ir.tensor(b)

        concat_node = ir.Node(
            "",
            "Concat",
            inputs=[init_0, init_1],
            attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
            num_outputs=1,
        )
        out = concat_node.outputs[0]
        # Deliberately leave out.shape as None — shape inference did not run.
        out.dtype = ir.DataType.FLOAT

        graph = ir.Graph(
            inputs=[ir.Value(name="x")],
            outputs=[out],
            nodes=[concat_node],
            name="test_graph",
            opset_imports={"": 20},
        )
        graph.register_initializer(init_0)
        graph.register_initializer(init_1)
        model = ir.Model(graph, ir_version=10)

        result = FoldConcatInitializersPass()(model)

        assert result.modified
        packed_name = "init_0__init_1__axis_0__concat"
        assert packed_name in model.graph.initializers
        # Shape should be computed from inputs: (4,3) + (4,3) along axis 0 → (8,3).
        packed = model.graph.initializers[packed_name]
        assert list(packed.shape) == [8, 3]


class TestFoldConcatOrtLoad:
    """End-to-end regression: a folded fp16 GQA-style graph must load in ORT.

    Reproduces the original failure where the packed-QKV initializer was
    serialized as fp32 while the rest of the graph was fp16, causing ORT to
    reject the model with::

        Type parameter (T) of Optype (MatMul) bound to different types
        (tensor(float16) and tensor(float))

    The failure occurs on the CPU EP too, so no GPU is needed.
    """

    def test_folded_fp16_concat_matmul_loads_in_ort(self, tmp_path):
        import onnx
        import onnxruntime as ort

        # Three fp16 weights with UNSET declared dtype (as after _cast_module_dtype),
        # but with shape retained so they serialize before they are pruned.
        q = ir.Value(name="q_weight", shape=ir.Shape([8, 4]))
        q.const_value = ir.tensor(np.random.randn(8, 4).astype(np.float16))
        k = ir.Value(name="k_weight", shape=ir.Shape([8, 4]))
        k.const_value = ir.tensor(np.random.randn(8, 4).astype(np.float16))
        v = ir.Value(name="v_weight", shape=ir.Shape([8, 4]))
        v.const_value = ir.tensor(np.random.randn(8, 4).astype(np.float16))

        concat_node = ir.Node(
            "",
            "Concat",
            inputs=[q, k, v],
            attributes=[ir.Attr("axis", ir.AttributeType.INT, 1)],
            num_outputs=1,
        )
        packed_out = concat_node.outputs[0]  # shape (8, 12), packed QKV weight
        packed_out.shape = ir.Shape([8, 12])

        # hidden @ packed_qkv : (N, 8) @ (8, 12) -> (N, 12), all fp16.
        hidden = ir.Value(
            name="hidden", shape=ir.Shape(["N", 8]), type=ir.TensorType(ir.DataType.FLOAT16)
        )
        matmul_node = ir.Node(
            "", "MatMul", inputs=[hidden, packed_out], num_outputs=1
        )
        out = matmul_node.outputs[0]
        out.shape = ir.Shape(["N", 12])
        out.dtype = ir.DataType.FLOAT16

        graph = ir.Graph(
            inputs=[hidden],
            outputs=[out],
            nodes=[concat_node, matmul_node],
            name="qkv_matmul",
            opset_imports={"": 20},
        )
        graph.register_initializer(q)
        graph.register_initializer(k)
        graph.register_initializer(v)
        model = ir.Model(graph, ir_version=10)

        FoldConcatInitializersPass()(model)

        # Drop the now-unused source initializers (a DCE step runs after folding
        # in the real export pipeline). Without this they linger as dead weights.
        for dead in ("q_weight", "k_weight", "v_weight"):
            del model.graph.initializers[dead]

        model_path = tmp_path / "fp16_qkv.onnx"
        onnx.save(ir.to_proto(model), str(model_path))

        # Before the fix this raises a fp16/fp32 MatMul type-mismatch at load.
        sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        feed = {"hidden": np.random.randn(2, 8).astype(np.float16)}
        (result,) = sess.run(None, feed)
        assert result.shape == (2, 12)
        assert result.dtype == np.float16
