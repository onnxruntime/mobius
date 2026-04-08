# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for FoldConcatInitializersPass."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

from mobius.passes._fold_concat import FoldConcatInitializersPass


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

        packed_name = "init_0__init_1__concat"
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

        packed_name = "init_0__init_1__init_2__concat"
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

        packed_name = "init_0__init_1__concat"
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

    def test_missing_const_value_raises_on_access(self):
        """LazyTensor raises AssertionError when weights not loaded."""
        a = np.ones((4, 3), dtype=np.float32)
        b = np.ones((4, 3), dtype=np.float32)
        model, init_vals = _make_concat_model([a, b], axis=0)

        # Clear const_value to simulate pre-weight-load state
        init_vals[0].const_value = None

        FoldConcatInitializersPass()(model)

        packed = model.graph.initializers.get("init_0__init_1__concat")
        assert packed is not None
        with pytest.raises(AssertionError, match="no const_value"):
            _ = packed.const_value.numpy()

    def test_idempotent(self):
        """Running the pass twice does not create duplicate initializers."""
        a = np.ones((4, 3), dtype=np.float32)
        b = np.ones((4, 3), dtype=np.float32)
        model, _ = _make_concat_model([a, b], axis=0)

        FoldConcatInitializersPass()(model)
        result2 = FoldConcatInitializersPass()(model)
        # Second run: no Concat nodes left → not modified
        assert not result2.modified
