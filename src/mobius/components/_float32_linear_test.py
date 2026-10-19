# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Independent dense epilogue oracle, including native-rounding sentinels."""

from __future__ import annotations

import pathlib
import uuid

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from torch.nn import functional as functional

from mobius._builder import _cast_module_dtype
from mobius._optimizations import SymbolicShapeInferencePass
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import _numpy_to_ort_value, _ort_value_to_numpy
from mobius.components import Float32Linear, Linear, RadioV4VisionModel
from mobius.components._radio_vision import RadioVisionModel


@pytest.mark.parametrize("provider", ["CPUExecutionProvider", "CUDAExecutionProvider"])
@pytest.mark.parametrize(
    "dtype,torch_dtype",
    [
        (ir.DataType.FLOAT, torch.float32),
        (ir.DataType.FLOAT16, torch.float16),
        (ir.DataType.BFLOAT16, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_dense_epilogue(provider, dtype, torch_dtype, bias, rank, monkeypatch):
    if provider == "CUDAExecutionProvider" and (
        provider not in ort.get_available_providers() or not torch.cuda.is_available()
    ):
        pytest.skip("CUDA PyTorch and ONNX Runtime are required")
    # The first product loses its low bits when rounded before the cancelling
    # bias. Other rows/columns distinguish shape/order errors from that loss.
    a = 1.0009765625 if torch_dtype == torch.float16 else 1.0078125
    x = torch.tensor([[a, a], [-a, 0.5], [0.25, -a]], dtype=torch_dtype)
    weight = torch.tensor([[a, 0], [0, a], [a, -a]], dtype=torch_dtype)
    rounded_product = torch.tensor(a * a, dtype=torch_dtype).item()
    beta = (
        torch.tensor([-rounded_product, rounded_product, -0.25], dtype=torch_dtype)
        if bias
        else None
    )
    reference = functional.linear(
        x.double(), weight.double(), beta.double() if beta is not None else None
    ).to(torch_dtype)
    if bias and torch_dtype != torch.float32:
        premature = functional.linear(x.float(), weight.float()).to(torch_dtype) + beta
        assert premature[0, 0] == 0
        assert reference[0, 0] > 0

    component = Float32Linear(2, 3, bias)
    _cast_module_dtype(component, dtype)
    for name, parameter in component.named_parameters():
        tensor = weight if name == "weight" else beta
        array = (
            tensor.view(torch.uint16).numpy().view(ir.DataType.BFLOAT16.numpy())
            if dtype == ir.DataType.BFLOAT16
            else tensor.numpy()
        )
        parameter.const_value = ir.tensor(array)
    builder, op, graph = create_test_builder()
    dimensions = {1: [2], 2: ["sequence", 2], 3: ["batch", "sequence", 2]}[rank]
    value = create_test_input(builder, "x", dimensions, dtype)
    result = component(op, value)
    result.name = "y"
    result.type = ir.TensorType(dtype)
    result.shape = ir.Shape([*dimensions[:-1], 3])
    graph.outputs.append(result)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    if provider == "CUDAExecutionProvider":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    path = pathlib.Path.cwd() / f".float32-linear-{uuid.uuid4().hex}.onnx"
    try:
        ir.save(ir.Model(graph, ir_version=11), path)
        session = ort.InferenceSession(
            str(path), options, providers=[(provider, {"use_tf32": "0"})]
        )
    finally:
        path.unlink(missing_ok=True)
    device = "cuda" if provider == "CUDAExecutionProvider" else "cpu"
    with monkeypatch.context() as precision, torch.inference_mode():
        if provider == "CUDAExecutionProvider" and torch_dtype == torch.float16:
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                # Native FP16 accumulation can erase the cancelling residual;
                # this component's source comparison requires FP32 accumulation.
                precision.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
        source = functional.linear(
            x.to(device), weight.to(device), beta.to(device) if beta is not None else None
        ).cpu()
    torch.testing.assert_close(source, reference, rtol=0, atol=0)
    samples = {
        1: [(x[0], reference[0]), (x[2], reference[2])],
        2: [(x, reference), (x.repeat(2, 1), reference.repeat(2, 1))],
        3: [
            (x.repeat(batch, sequence // 3, 1), reference.repeat(batch, sequence // 3, 1))
            for batch, sequence in [(1, 3), (2, 6)]
        ],
    }
    for pixels, expected in samples[rank]:
        pixels = pixels.contiguous()
        array = (
            pixels.view(torch.uint16).numpy().view(ir.DataType.BFLOAT16.numpy())
            if dtype == ir.DataType.BFLOAT16
            else pixels.numpy()
        )
        output = _ort_value_to_numpy(
            session.run_with_ort_values(None, {"x": _numpy_to_ort_value(array)})[0]
        )
        expected = expected.contiguous()
        expected_array = (
            expected.view(torch.uint16).numpy().view(ir.DataType.BFLOAT16.numpy())
            if dtype == ir.DataType.BFLOAT16
            else expected.numpy()
        )
        np.testing.assert_array_equal(output, expected_array, strict=True)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
def test_deferred_activation_dtype_preserves_cast_like(dtype, bias):
    component = Float32Linear(2, 3, bias)
    _cast_module_dtype(component, dtype)
    builder, op, graph = create_test_builder()
    value = create_test_input(builder, "x", [1, 2], dtype)
    intermediate = op.Identity(value)
    # Custom quantized operators can defer downstream types until graph inference.
    intermediate.type = None

    result = component(op, intermediate)

    assert intermediate.dtype is None
    cast = result.producer()
    assert cast is not None
    assert cast.op_type == "CastLike"
    assert cast.inputs[1] is intermediate
    epilogue = cast.inputs[0]
    assert epilogue is not None
    graph.outputs.append(result)
    SymbolicShapeInferencePass()(ir.Model(graph, ir_version=11))
    assert intermediate.dtype == dtype
    assert epilogue.dtype == ir.DataType.FLOAT
    assert result.dtype == dtype


def test_parameter_names_and_dense_only_validation():
    component = Float32Linear(2, 3)
    assert list(dict(component.named_parameters())) == ["weight", "bias"]
    builder, op, _ = create_test_builder()
    for dtype in (ir.DataType.INT8, ir.DataType.DOUBLE):
        value = create_test_input(builder, f"x_{dtype}", [1, 2], dtype)
        with pytest.raises(ValueError, match="requires float32"):
            component(op, value)
    value = create_test_input(builder, "floating_x", [1, 2])
    component.weight.type = ir.TensorType(ir.DataType.UINT8)
    with pytest.raises(ValueError, match="requires float32"):
        component(op, value)
    component.weight.type = ir.TensorType(ir.DataType.FLOAT)
    component.bias.type = ir.TensorType(ir.DataType.UINT8)
    with pytest.raises(ValueError, match="requires float32"):
        component(op, value)


def test_radio_v4_only_uses_float32_biased_projections():
    v4 = RadioV4VisionModel(
        patch_size=2,
        max_grid_size=4,
        hidden_size=32,
        intermediate_size=64,
        num_layers=1,
        num_heads=4,
        video_temporal_patch_size=2,
    )
    block = v4.blocks[0]
    assert all(
        isinstance(linear, Float32Linear)
        for linear in (block.attn.qkv, block.attn.proj, block.mlp.fc1, block.mlp.fc2)
    )
    assert type(v4.patch_generator.embedder) is Linear
    assert type(v4.patch_generator.video_embedder) is Linear
    v2 = RadioVisionModel(
        image_height=8,
        image_width=4,
        patch_size=2,
        max_grid_size=4,
        hidden_size=32,
        intermediate_size=64,
        num_layers=1,
        num_heads=4,
    )
    assert type(v2.blocks[0].attn.qkv) is Linear
    assert type(v2.blocks[0].attn.proj) is Linear
    assert type(v2.blocks[0].mlp.fc1) is Linear
    assert type(v2.blocks[0].mlp.fc2) is Linear
