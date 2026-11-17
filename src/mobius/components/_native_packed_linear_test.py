# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native packed ABIs and scale values survive model-dtype specialization."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from mobius._builder import _cast_module_dtype
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import _numpy_to_ort_value, _ort_value_to_numpy
from mobius.components import (
    GGMLIQ2XXSLinear,
    GGMLQ8Linear,
    Linear,
    NVFP4A16Linear,
    NVFP4QuantizedLinear,
)


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
@pytest.mark.parametrize(
    ("linear_type", "operator", "block_size", "payload"),
    [
        (GGMLIQ2XXSLinear, "MatMulGGMLIQ2XXS", 256, 66),
        (GGMLQ8Linear, "MatMulGGMLQ8_0", 32, 34),
    ],
)
def test_ggml_exact_ort_cuda_abi(dtype, linear_type, operator, block_size, payload):
    linear = linear_type(2 * block_size, 7, bias=True)
    _cast_module_dtype(linear, dtype)
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [2, 3, 2 * block_size], dtype)
    result = linear(op, x)
    native = next(node for node in graph if node.op_type == operator)
    assert native.domain == "com.microsoft"
    assert not native.attributes
    assert native.inputs[1].dtype == ir.DataType.UINT8
    assert native.inputs[1].shape == [7, 2, payload]
    assert native.inputs[2].dtype == dtype
    assert result.dtype == dtype
    assert result.shape == [2, 3, 7]
    assert graph.opset_imports["com.microsoft"] == 1
    assert not any(node.domain == "pkg.nxrt" for node in graph)


@pytest.mark.parametrize(("k", "n"), [(255, 3), (-256, 3), (256, -1)])
def test_iq2_rejects_invalid_storage_geometry(k, n):
    with pytest.raises(ValueError, match="divisible by 256"):
        GGMLIQ2XXSLinear(k, n)


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
@pytest.mark.parametrize("bias", [False, True])
def test_iq2_after_dense_projection_has_declared_activation_dtype(dtype, bias):
    dense = Linear(32, 256, bias=bias)
    _cast_module_dtype(dense, dtype)
    builder, op, _graph = create_test_builder()
    x = create_test_input(builder, "x", [1, 32], dtype)
    projected = dense(op, x)
    assert projected.dtype == dtype
    assert GGMLIQ2XXSLinear(256, 7)(op, projected).dtype == dtype


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
@pytest.mark.parametrize(
    ("linear_type", "operator"),
    [
        (NVFP4QuantizedLinear, "MatMulBlockQuantizedFp4Weight"),
        (NVFP4A16Linear, "MatMulNVFP4A16"),
    ],
)
def test_nvfp4_dtype_and_global_scale_boundary(dtype, linear_type, operator):
    linear = linear_type(64, 32, bias=True)
    _cast_module_dtype(linear, dtype)
    assert linear.weight_scale_2.dtype == ir.DataType.FLOAT
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [2, 64], dtype)
    result = linear(op, x)
    native = next(node for node in graph if node.op_type == operator)
    assert native.inputs[0].dtype == dtype
    assert native.inputs[2].dtype == ir.DataType.UINT8
    assert native.inputs[3].dtype == ir.DataType.FLOAT
    assert len(native.inputs) == 4
    assert len([node for node in graph if node.op_type == "Add"]) == 1
    if linear_type is NVFP4A16Linear:
        assert not native.attributes
    assert result.dtype == dtype
    assert result.shape == [2, 32]


@pytest.mark.integration
def test_bf16_q8_rounds_complete_weights_not_scales(tmp_path):
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider is required")
    linear = GGMLQ8Linear(64, 4, bias=True)
    _cast_module_dtype(linear, ir.DataType.BFLOAT16)
    # Complete-weight BF16 rounding yields 1; early scale rounding yields 0,
    # while retaining FP32 weights until the output boundary yields 0.875.
    scales = np.tile(np.array([1.00390625, 1.0], dtype=np.float32), (4, 1))
    np.testing.assert_array_equal(scales.astype(np.float16).astype(np.float32), scales)
    np.testing.assert_array_equal(
        scales.astype(ir.DataType.BFLOAT16.numpy()).astype(np.float32), np.float32(1.0)
    )
    packed = np.empty((4, 2, 34), dtype=np.uint8)
    packed[..., :2] = scales.astype(np.float16).view(np.uint8).reshape(4, 2, 2)
    packed[:, 0, 2:] = 7
    packed[:, 1, 2:] = 249
    linear.weight.const_value = ir.tensor(packed)
    linear.bias.const_value = ir.tensor(np.full(4, -0.5, dtype=ir.DataType.BFLOAT16.numpy()))
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1, 64], ir.DataType.BFLOAT16)
    result = linear(op, x)
    result.name = "output"
    graph.outputs.append(result)
    native = next(node for node in graph if node.op_type == "MatMulGGMLQ8_0")
    assert native.inputs[0].dtype == ir.DataType.BFLOAT16
    assert native.inputs[1].shape == [4, 2, 34]
    assert native.inputs[2].dtype == ir.DataType.BFLOAT16
    path = tmp_path / "q8.onnx"
    ir.save(ir.Model(graph, ir_version=11), path)
    session = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider"])
    outputs = session.run_with_ort_values(
        None,
        {"x": _numpy_to_ort_value(np.ones((1, 64), dtype=ir.DataType.BFLOAT16.numpy()))},
    )
    np.testing.assert_array_equal(
        _ort_value_to_numpy(outputs[0]),
        np.full((1, 4), 0.5, dtype=ir.DataType.BFLOAT16.numpy()),
    )
