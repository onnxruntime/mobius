# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for RMSNorm component."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from onnxscript import nn

from mobius._builder import _cast_module_dtype
from mobius._flags import override_flags
from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius._testing.ort_inference import OnnxModelSession
from mobius.components._rms_norm import (
    GatedRMSNorm,
    GroupedGatedRMSNorm,
    RMSNorm,
    apply_rms_norm,
)


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


def _grouped_gated_norm_model(shape, group_size, dtype):
    norm = GroupedGatedRMSNorm(shape[-1], group_size, eps=1e-5)
    _cast_module_dtype(norm, dtype)
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", shape, dtype)
    gate = create_test_input(builder, "gate", shape, dtype)
    builder.add_output(norm(op, hidden, gate), "out")
    return ir.Model(graph, ir_version=11)


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
@pytest.mark.parametrize("workaround", [False, True])
def test_grouped_gated_norm_precision_graph(dtype, workaround):
    with override_flags(ort_cuda_grouped_rmsnorm_workaround=workaround):
        model = _grouped_gated_norm_model([2, 3, 24], 8, dtype)
    graph = model.graph
    assert [
        name for name, value in graph.initializers.items() if value.const_value is None
    ] == ["weight"]
    assert graph.initializers["weight"].dtype == dtype
    assert not any(node.op_type == "RMSNormalization" for node in graph)
    variance = next(node for node in graph if node.op_type == "ReduceMean")
    square = variance.inputs[0].producer()
    assert square.op_type == "Mul"
    grouped = square.inputs[0].producer()
    assert grouped.op_type == "Reshape"
    gated = grouped.inputs[0].producer()
    assert gated.op_type == "Mul"
    assert gated.inputs[0].producer().attributes["to"].as_int() == ir.DataType.FLOAT
    swish = gated.inputs[1].producer()
    assert swish.op_type == "Swish"
    assert swish.inputs[0].producer().attributes["to"].as_int() == ir.DataType.FLOAT
    gamma_mul = graph.outputs[0].producer()
    assert gamma_mul.op_type == "Mul"
    # The rounding boundary is after normalization but before the native gamma multiply.
    for value in gamma_mul.inputs:
        cast = value.producer()
        assert cast.op_type == "CastLike"
        assert cast.inputs[1] is graph.inputs[0]
    assert gamma_mul.inputs[0].producer().inputs[0].producer().op_type == "Reshape"


@pytest.mark.parametrize(
    "shape,group_size",
    [
        ([2, 24], 8),
        ([2, 3, 24], 8),
        ([2, 3, 24], 24),
        ([2, 3, 24], 1),
        ([2, 3, 4096], 512),
    ],
)
@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
def test_grouped_gated_norm_source_precision_parity(tmp_path, shape, group_size, dtype):
    if (
        dtype == ir.DataType.BFLOAT16
        and "CUDAExecutionProvider" not in ort.get_available_providers()
    ):
        pytest.skip("BF16 grouped norm arithmetic requires CUDA execution for this probe")
    model = _grouped_gated_norm_model(shape, group_size, dtype)
    np_dtype, torch_dtype = {
        ir.DataType.FLOAT: (np.float32, torch.float32),
        ir.DataType.FLOAT16: (np.float16, torch.float16),
        ir.DataType.BFLOAT16: (ml_dtypes.bfloat16, torch.bfloat16),
    }[dtype]
    rng = np.random.default_rng(761)
    # Distinct magnitudes and nonuniform gamma expose cross-group scale indexing errors.
    magnitudes = np.geomspace(0.01, 20, shape[-1] // group_size).repeat(group_size)
    hidden_np = (rng.normal(size=shape) * magnitudes).astype(np_dtype)
    gate_np = rng.normal(size=shape).astype(np_dtype)
    weight_np = rng.uniform(0.1, 2, shape[-1]).astype(np_dtype)
    model.graph.initializers["weight"].const_value = ir.tensor(weight_np)
    hidden = torch.from_numpy(hidden_np.astype(np.float32)).to(torch_dtype)
    gate = torch.from_numpy(gate_np.astype(np.float32)).to(torch_dtype)
    weight = torch.from_numpy(weight_np.astype(np.float32)).to(torch_dtype)
    # Pinned HF Zamba2RMSNormGated: FP32 SiLU/variance, cast normalized x, then gamma.
    gated = hidden.float() * torch.nn.functional.silu(gate.float())
    grouped = gated.reshape(*shape[:-1], -1, group_size)
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-5)
    expected = weight * normalized.reshape(shape).to(torch_dtype)
    device = "cuda" if dtype == ir.DataType.BFLOAT16 else "cpu"
    optimized_path = tmp_path / "optimized.onnx"
    session = OnnxModelSession(
        model, device=device, optimized_model_filepath=str(optimized_path)
    )
    try:
        actual = session.run({"hidden": hidden_np, "gate": gate_np})["out"]
        optimized = ir.load(optimized_path)
        assert all(
            node.inputs[1].shape.rank() == 1
            for node in optimized.graph
            if node.op_type == "RMSNormalization"
        )
    finally:
        session.close()
    tolerance = {
        ir.DataType.FLOAT: 2e-6,
        ir.DataType.FLOAT16: 1e-3,
        ir.DataType.BFLOAT16: 1e-2,
    }[dtype]
    np.testing.assert_allclose(
        actual.astype(np.float32), expected.float().numpy(), rtol=tolerance, atol=tolerance
    )
    if dtype != ir.DataType.FLOAT:
        wrong_early_cast = gated.to(torch_dtype).float().reshape_as(grouped)
        wrong_early_cast *= torch.rsqrt(
            wrong_early_cast.square().mean(-1, keepdim=True) + 1e-5
        )
        wrong_early_cast = weight * wrong_early_cast.reshape(shape).to(torch_dtype)
        wrong_late_cast = (normalized.reshape(shape) * weight.float()).to(torch_dtype)
        assert torch.any(expected != wrong_early_cast)
        assert torch.any(expected != wrong_late_cast)


@pytest.mark.parametrize("hidden,group", [(0, 4), (8, 0), (8, -1), (8, 3)])
def test_grouped_gated_norm_rejects_invalid_groups(hidden, group):
    with pytest.raises(ValueError, match="group_size"):
        GroupedGatedRMSNorm(hidden, group)


def test_default_gated_norm_retains_existing_grouped_operator():
    with override_flags(ort_cuda_grouped_rmsnorm_workaround=False):
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [2, 3, 24])
        gate = create_test_input(builder, "gate", [2, 3, 24])
        GatedRMSNorm(24, group_size=8)(op, hidden, gate)
    assert count_op_type(graph, "RMSNormalization") == 1
