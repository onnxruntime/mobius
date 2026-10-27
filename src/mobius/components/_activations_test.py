# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for activation functions."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import onnx_ir as ir
import pytest
import torch
from onnxscript import GraphBuilder

from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius._testing.ort_inference import OnnxModelSession
from mobius.components import Float32SiLU
from mobius.components._activations import (
    ACT2FN,
    gelu,
    get_activation,
    linear,
    quick_gelu,
    relu,
    silu,
    tanh,
)


class TestActivations:
    @pytest.mark.parametrize(
        ("dtype", "array_dtype", "torch_dtype"),
        [
            (ir.DataType.FLOAT16, np.float16, torch.float16),
            (ir.DataType.BFLOAT16, ml_dtypes.bfloat16, torch.bfloat16),
        ],
    )
    def test_float32_silu_rounds_only_at_output(self, dtype, array_dtype, torch_dtype):
        graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
        builder = GraphBuilder(graph)
        x = builder.input("x", dtype, [10])
        output = Float32SiLU()(builder.op, x)
        output.type, output.shape = x.type, x.shape
        builder.add_output(output, "output")
        division = next(node for node in graph if node.op_type == "Div")
        assert division.inputs[0].producer().attributes["to"].as_int() == ir.DataType.FLOAT
        values = np.array([-1000, -8, -3, -0.75, -0.1, 0, 0.1, 0.75, 3, 8], np.float32)
        inputs = values.astype(array_dtype)
        reference = torch.nn.functional.silu(torch.tensor(values).to(torch_dtype))
        session = OnnxModelSession(ir.Model(graph, ir_version=12))
        try:
            actual = session.run({"x": inputs})["output"]
        finally:
            session.close()
        np.testing.assert_array_equal(actual.astype(np.float32), reference.float().numpy())

    def _run_activation(self, act_fn):
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        result = act_fn(op, x)
        assert result is not None

    def test_silu(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        result = silu(op, x)
        assert result is not None
        assert count_op_type(graph, "Swish") == 1
        assert count_op_type(graph, "Sigmoid") == 0
        assert count_op_type(graph, "Mul") == 0

    def test_relu(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        result = relu(op, x)
        assert result is not None
        assert count_op_type(graph, "Relu") == 1

    def test_tanh(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        result = tanh(op, x)
        assert result is not None
        assert count_op_type(graph, "Tanh") == 1

    def test_gelu(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        result = gelu(op, x)
        assert result is not None
        assert graph.num_nodes() >= 1

    def test_quick_gelu(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        result = quick_gelu(op, x)
        assert result is not None
        assert count_op_type(graph, "Sigmoid") >= 1

    def test_linear_activation(self):
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        result = linear(op, x)
        assert result is x  # Identity

    def test_get_activation_valid(self):
        for name in ACT2FN:
            fn = get_activation(name)
            assert callable(fn)

    def test_get_activation_invalid(self):
        with pytest.raises(KeyError):
            get_activation("nonexistent_activation")

    def test_get_activation_none_raises_value_error(self):
        with pytest.raises(ValueError, match="hidden_act is None"):
            get_activation(None)

    @pytest.mark.parametrize("act_name", list(ACT2FN.keys()))
    def test_all_activations_produce_output(self, act_name):
        act_fn = get_activation(act_name)
        self._run_activation(act_fn)
