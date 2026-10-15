# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CUDA regressions for fused-Mamba grouped normalization precision."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from onnx_ir import tensor_adapters
from onnxscript import nn

from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import _numpy_to_ort_value, _ort_value_to_numpy
from mobius.components import Float32GatedRMSNorm


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
@pytest.mark.parametrize(("hidden_size", "group_size"), [(32, 8), (8192, 1024)])
def test_float32_gated_rms_norm_cuda(tmp_path, dtype, hidden_size, group_size):
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider is required")
    generator = torch.Generator().manual_seed(123)
    torch_dtype = tensor_adapters.to_torch_dtype(dtype)
    x = torch.randn((2, 3, hidden_size), generator=generator).to(torch_dtype)
    z = (torch.randn(x.shape, generator=generator) * 3).to(torch_dtype)
    gamma = (torch.randn((hidden_size,), generator=generator) * 0.2 + 1).to(torch_dtype)
    norm = Float32GatedRMSNorm(hidden_size, eps=1e-5, group_size=group_size)
    norm.weight = nn.Parameter([hidden_size], data=tensor_adapters.TorchTensor(gamma))
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [2, 3, hidden_size], dtype)
    gate = create_test_input(builder, "gate", [2, 3, hidden_size], dtype)
    result = norm(op, hidden, gate)
    result.name = "output"
    result.type = ir.TensorType(dtype)
    result.shape = ir.Shape([2, 3, hidden_size])
    graph.outputs.append(result)
    # No grouped 2D RMSNormalization scale or early activation cast is emitted.
    assert all(node.op_type != "RMSNormalization" for node in graph)
    output_cast = result.producer()
    assert output_cast.op_type == "CastLike"
    assert output_cast.inputs[0].producer().op_type == "Mul"

    path = tmp_path / "norm.onnx"
    optimized_path = tmp_path / "optimized.onnx"
    ir.save(ir.Model(graph, ir_version=11), path)
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.optimized_model_filepath = str(optimized_path)
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CUDAExecutionProvider"]
    )
    feeds = {
        name: _numpy_to_ort_value(
            value.contiguous()
            .view(torch.uint8)
            .numpy()
            .view(dtype.numpy())
            .reshape(value.shape)
        )
        for name, value in (("hidden", x), ("gate", z))
    }
    actual = _ort_value_to_numpy(session.run_with_ort_values(None, feeds)[0])
    # Independent source equation: gamma is FP32, then one final output cast.
    gated = x.float() * torch.nn.functional.silu(z.float())
    grouped = gated.reshape(2, 3, hidden_size // group_size, group_size)
    normalized = grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + 1e-5)
    expected = (normalized.reshape(x.shape) * gamma.float()).to(torch_dtype).float().numpy()
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected,
        rtol=2e-6 if dtype == ir.DataType.FLOAT else 0.008,
        atol=2e-6 if dtype == ir.DataType.FLOAT else 0.0001,
    )
    optimized = ir.load(optimized_path)
    for node in optimized.graph:
        if node.op_type == "RMSNormalization":
            assert len(node.inputs[1].shape) == 1


@pytest.mark.parametrize("group_size", [0, -1, 3])
def test_float32_gated_rms_norm_rejects_invalid_groups(group_size):
    with pytest.raises(ValueError, match="group_size"):
        Float32GatedRMSNorm(32, group_size=group_size)
