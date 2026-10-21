# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Weight-only FP8 layout, graph, and native CUDA kernel tests."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import torch
from onnx_ir.tensor_adapters import TorchTensor

from mobius._builder import _cast_module_dtype
from mobius._testing import create_test_builder, create_test_input, make_config
from mobius.components._fp8 import Fp8Linear, Fp8QwenMoEBlock, quantize_fp8


@pytest.mark.parametrize("square_blocks", [False, True])
def test_fp8_quantization_handles_partial_blocks_and_zero_weights(square_blocks):
    weight = torch.randn(131, 259)
    weight[:128, :128] = 0
    quantized, scales = quantize_fp8(weight, square_blocks=square_blocks)
    assert quantized.shape == weight.shape
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float32
    assert scales.shape == ((2, 3) if square_blocks else (131, 3))
    assert torch.isfinite(scales).all() and torch.all(scales > 0)
    expanded = scales.repeat_interleave(128, -1)[:, :259]
    if square_blocks:
        expanded = expanded.repeat_interleave(128, 0)[:131]
    reconstructed = quantized.float() * expanded
    assert (
        torch.linalg.vector_norm(reconstructed - weight) / torch.linalg.vector_norm(weight)
        < 0.04
    )
    assert torch.count_nonzero(reconstructed[:128, :128]) == 0


@pytest.mark.parametrize(
    "weight",
    [
        torch.ones(4),
        torch.ones(0, 4),
        torch.full((4, 4), float("nan")),
        torch.ones(4, 4, dtype=torch.int32),
    ],
)
def test_fp8_quantization_rejects_invalid_weights(weight):
    with pytest.raises(ValueError, match="FP8"):
        quantize_fp8(weight)


def test_fp8_linear_preserves_fp32_scales_and_emits_weight_only_op():
    layer = Fp8Linear(256, 64, bias=True)
    _cast_module_dtype(layer, ir.DataType.FLOAT16)
    assert layer.weight.dtype == ir.DataType.FLOAT8E4M3FN
    assert layer.scales.dtype == ir.DataType.FLOAT
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 2, 256], ir.DataType.FLOAT16)
    result = layer(op, hidden)
    builder._adapt_outputs([result], "")
    node = next(node for node in graph if node.op_type == "MatMulBlockQuantizedFp8Weight")
    assert node.domain == "com.microsoft"
    assert node.inputs[3] is None
    assert node.inputs[4] is layer.bias


def test_fp8_moe_uses_one_sparse_qmoe_and_float_router():
    config = make_config(
        hidden_size=128,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
    )
    layer = Fp8QwenMoEBlock(config)
    _cast_module_dtype(layer, ir.DataType.FLOAT16)
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 2, 128], ir.DataType.FLOAT16)
    result = layer(op, hidden)
    builder._adapt_outputs([result], "")
    qmoe = next(node for node in graph if node.op_type == "QMoE")
    assert qmoe.attributes["quant_type"].value == "fp8"
    assert qmoe.attributes["swiglu_fusion"].value == 2
    assert qmoe.attributes["block_size"].value == 128
    assert sum(node.op_type == "QMoE" for node in graph) == 1
    assert sum(node.op_type == "MatMulBlockQuantizedFp8Weight" for node in graph) == 3
    assert layer.fc1_scales.dtype == ir.DataType.FLOAT
    assert layer.gate.weight.dtype == ir.DataType.FLOAT16


@pytest.mark.integration
def test_fp8_linear_cuda_matches_dequantized_reference():
    import onnxruntime as ort

    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    ort.preload_dlls()
    torch.manual_seed(4)
    weight = torch.randn(64, 256) * 0.1
    quantized, scales = quantize_fp8(weight)
    layer = Fp8Linear(256, 64)
    layer.weight.const_value = TorchTensor(quantized)
    layer.scales.const_value = TorchTensor(scales)
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 2, 256], ir.DataType.FLOAT16)
    result = layer(op, hidden)
    result.type, result.shape = ir.TensorType(ir.DataType.FLOAT16), ir.Shape([1, 2, 64])
    builder.add_output(result, "output")
    graph.opset_imports["com.microsoft"] = 1
    model = ir.Model(graph, ir_version=10)
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        ir.serde.serialize_model(model).SerializeToString(),
        sess_options=options,
        providers=["CUDAExecutionProvider"],
    )
    x = torch.randn(1, 2, 256).half()
    actual = session.run(None, {"hidden": x.numpy()})[0]
    reconstructed = quantized.float() * scales.repeat_interleave(128, -1)
    expected = torch.nn.functional.linear(x.float(), reconstructed).half().numpy()
    np.testing.assert_allclose(actual, expected, rtol=0.005, atol=0.005)
