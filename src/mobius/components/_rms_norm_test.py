# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for RMSNorm component."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnxscript import nn

from mobius._builder import _cast_module_dtype
from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius.components._rms_norm import OffsetRMSNorm, RMSNorm, apply_rms_norm


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@pytest.mark.parametrize("unknown_input_type", [False, True])
def test_offset_rms_norm_learned_scale_precision(tmp_path, dtype, unknown_input_type):
    """Keep the learned offset and scale multiplication in FP32 until output cast."""
    values = np.asarray([[1.0, 3.0, 6.0, 10.0]], dtype=dtype.numpy())
    weight = np.asarray([0.0004, -0.0004, 0.0003, -0.0003], dtype=dtype.numpy())
    norm = OffsetRMSNorm(4, eps=1e-6)
    _cast_module_dtype(norm, dtype)
    norm.weight = nn.Parameter([4], data=ir.Tensor(weight))
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1, 4], dtype)
    hidden = op.Identity(x) if unknown_input_type else x
    if unknown_input_type:
        hidden.type = None
    result = norm(op, hidden)
    result.type = ir.TensorType(dtype)
    graph.outputs.append(result)
    if dtype == ir.DataType.FLOAT and not unknown_input_type:
        assert count_op_type(graph, "Cast") == 0
        assert count_op_type(graph, "CastLike") == 0
    else:
        assert count_op_type(graph, "Cast") >= 1
        assert count_op_type(graph, "CastLike") == 1
    path = tmp_path / "offset-norm.onnx"
    ir.save(ir.Model(graph, ir_version=11), path)
    actual = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        None, {"x": values}
    )[0]
    values_f32 = values.astype(np.float32)
    expected = (
        values_f32
        / np.sqrt(np.mean(values_f32**2, axis=-1, keepdims=True) + 1e-6)
        * (1.0 + weight.astype(np.float32))
    ).astype(dtype.numpy())
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7, strict=True)


class TestRMSNorm:
    def test_rms_norm_creates_parameters(self):
        norm = RMSNorm(64, eps=1e-6)
        params = list(norm.parameters())
        # Should have weight only (eps is now a float attribute, not a parameter)
        assert len(params) == 1

    def test_rms_norm_weight_shape(self):
        norm = RMSNorm(128)
        assert list(norm.weight.shape) == [128]

    def test_rms_norm_forward_builds_graph(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        norm = RMSNorm(64, eps=1e-5)
        result = norm(op, x)
        assert result is not None
        # Should use the ONNX RMSNormalization op
        assert count_op_type(graph, "RMSNormalization") >= 1

    def test_rms_norm_different_eps(self):
        norm1 = RMSNorm(64, eps=1e-5)
        norm2 = RMSNorm(64, eps=1e-6)
        assert norm1.variance_epsilon != norm2.variance_epsilon

    def test_apply_rms_norm_function(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        weight = nn.Parameter([64], name="test_weight")
        weight._realize(builder)

        result = apply_rms_norm(op, x, weight, 1e-6)
        assert result is not None
        assert count_op_type(graph, "RMSNormalization") >= 1

    def test_multiple_rms_norms_in_same_graph(self):
        """Ensure two RMSNorm modules can coexist in the same graph."""
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])

        norm1 = RMSNorm(64, eps=1e-6)
        norm2 = RMSNorm(64, eps=1e-6)

        # Manually set names to avoid collision
        builder.push_module("norm1")
        for p in norm1._parameters.values():
            p._realize(builder)
        r1 = norm1.forward(op, x)
        builder.pop_module()

        builder.push_module("norm2")
        for p in norm2._parameters.values():
            p._realize(builder)
        r2 = norm2.forward(op, x)
        builder.pop_module()

        assert r1 is not None
        assert r2 is not None
        assert count_op_type(graph, "RMSNormalization") == 2
