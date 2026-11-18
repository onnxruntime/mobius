# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for TransposedWeightMatMulPass."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

from mobius._passes._transposed_matmul import TransposedWeightMatMulPass


def _value(name: str, shape: list[int]) -> ir.Value:
    value = ir.Value(name=name, shape=ir.Shape(shape), type=ir.TensorType(ir.DataType.FLOAT))
    return value


def _initializer(name: str, data: np.ndarray) -> ir.Value:
    value = _value(name, list(data.shape))
    value.const_value = ir.tensor(data)
    return value


def _transpose(source: ir.Value, perm: list[int]) -> ir.Node:
    return ir.Node(
        "",
        "Transpose",
        inputs=[source],
        attributes=[ir.Attr("perm", ir.AttributeType.INTS, perm)],
        num_outputs=1,
    )


def _model(
    nodes: list[ir.Node],
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    initializers: list[ir.Value],
) -> ir.Model:
    graph = ir.Graph(
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        initializers=initializers,
        opset_imports={"": 21},
        name="g",
    )
    return ir.Model(graph, ir_version=10)


def _projection(weight: np.ndarray, perm: list[int] | None = None) -> ir.Model:
    """``logits = MatMul(x, Transpose(weight))`` with ``x: [1, 3, in]``."""
    out_features, in_features = weight.shape
    x = _value("x", [1, 3, in_features])
    w = _initializer("weight", weight)
    transpose = _transpose(w, perm or [1, 0])
    matmul = ir.Node("", "MatMul", inputs=[x, transpose.outputs[0]], num_outputs=1)
    matmul.outputs[0].name = "logits"
    matmul.outputs[0].shape = ir.Shape([1, 3, out_features])
    matmul.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    return _model([transpose, matmul], [x], [matmul.outputs[0]], [w])


def _run(model: ir.Model, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    import onnxruntime as ort

    proto = ir.serde.serialize_model(model)
    session = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return session.run(None, feeds)


class TestTransposedWeightMatMulPass:
    def test_rewrites_projection_and_keeps_output_name(self):
        weight = np.random.default_rng(0).standard_normal((6, 4)).astype(np.float32)
        model = _projection(weight)

        pass_ = TransposedWeightMatMulPass()
        assert pass_(model).modified
        assert pass_.converted == 1

        [node] = list(model.graph)
        assert (node.domain, node.op_type) == ("com.microsoft", "FusedMatMul")
        assert node.attributes["transB"].value == 1
        assert node.inputs[1].name == "weight"
        assert model.graph.outputs[0].name == "logits"
        assert model.opset_imports["com.microsoft"] == 1

        x = np.random.default_rng(1).standard_normal((1, 3, 4)).astype(np.float32)
        [logits] = _run(model, {"x": x})
        np.testing.assert_allclose(logits, x @ weight.T, rtol=1e-5, atol=1e-5)

    def test_rewrites_packed_qkv_concat(self):
        rng = np.random.default_rng(0)
        parts = [rng.standard_normal((n, 4)).astype(np.float32) for n in (4, 2, 2)]
        x = _value("x", [1, 3, 4])
        weights = [_initializer(name, part) for name, part in zip("qkv", parts)]
        concat = ir.Node(
            "",
            "Concat",
            inputs=weights,
            attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
            num_outputs=1,
        )
        transpose = _transpose(concat.outputs[0], [1, 0])
        matmul = ir.Node("", "MatMul", inputs=[x, transpose.outputs[0]], num_outputs=1)
        matmul.outputs[0].name = "qkv"
        model = _model([concat, transpose, matmul], [x], [matmul.outputs[0]], weights)

        TransposedWeightMatMulPass()(model)

        assert [n.op_type for n in model.graph] == ["Concat", "FusedMatMul"]
        x_data = rng.standard_normal((1, 3, 4)).astype(np.float32)
        [qkv] = _run(model, {"x": x_data})
        np.testing.assert_allclose(qkv, x_data @ np.concatenate(parts).T, rtol=1e-5, atol=1e-5)

    def test_leaves_transposed_activation(self):
        """``Q @ K^T`` style products transpose an activation, not a weight."""
        a = _value("a", [3, 4])
        b = _value("b", [5, 4])
        transpose = _transpose(b, [1, 0])
        matmul = ir.Node("", "MatMul", inputs=[a, transpose.outputs[0]], num_outputs=1)
        model = _model([transpose, matmul], [a, b], [matmul.outputs[0]], [])

        assert not TransposedWeightMatMulPass()(model).modified
        assert [n.op_type for n in model.graph] == ["Transpose", "MatMul"]

    def test_leaves_shared_transpose(self):
        """A transposed weight with another consumer keeps its Transpose."""
        x = _value("x", [1, 3, 4])
        w = _initializer("weight", np.ones((6, 4), np.float32))
        transpose = _transpose(w, [1, 0])
        first = ir.Node("", "MatMul", inputs=[x, transpose.outputs[0]], num_outputs=1)
        second = ir.Node("", "MatMul", inputs=[x, transpose.outputs[0]], num_outputs=1)
        model = _model(
            [transpose, first, second], [x], [first.outputs[0], second.outputs[0]], [w]
        )

        assert not TransposedWeightMatMulPass()(model).modified

    def test_leaves_transpose_that_is_a_graph_output(self):
        x = _value("x", [1, 3, 4])
        w = _initializer("weight", np.ones((6, 4), np.float32))
        transpose = _transpose(w, [1, 0])
        matmul = ir.Node("", "MatMul", inputs=[x, transpose.outputs[0]], num_outputs=1)
        model = _model(
            [transpose, matmul], [x], [matmul.outputs[0], transpose.outputs[0]], [w]
        )

        assert not TransposedWeightMatMulPass()(model).modified

    @pytest.mark.parametrize("perm", [[0, 1]])
    def test_leaves_other_permutations(self, perm):
        model = _projection(np.ones((4, 4), np.float32), perm=perm)
        assert not TransposedWeightMatMulPass()(model).modified
