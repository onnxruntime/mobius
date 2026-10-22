# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Sparse IQ2 ABI and portable numerical router tests; no checkpoint download."""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from onnx_ir import tensor_adapters

from mobius._build_context import build_context
from mobius._builder import _cast_module_dtype
from mobius._configs import NemotronHConfig
from mobius._execution_providers import ep_registry
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import _numpy_to_ort_value, _ort_value_to_numpy
from mobius.components import GGMLIQ2XXSLinear, GGMLIQ2XXSMoE
from mobius.models.nemotron_h import NemotronHMoEBlock, NemotronHMoEGate
from mobius.models.nemotron_h_omni import _SparseIQ2MoEBlock
from mobius.models.nemotron_h_puzzle_test import _reference_moe


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
@pytest.mark.parametrize("leading", [[5], [2, 3], ["batch", "sequence"], [0], [2, 0]])
def test_sparse_iq2_exact_abi_and_native_output(dtype, leading):
    module = GGMLIQ2XXSMoE(256, 512, 3, 2)
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [*leading, 256], dtype)
    ids = create_test_input(builder, "ids", [*leading, 2], ir.DataType.INT64)
    weights = create_test_input(builder, "weights", [*leading, 2], ir.DataType.FLOAT)
    with build_context(ep_registry.require("cuda"), dtype):
        output = module(op, x, weights, ids)
    node = output.producer()
    assert (node.domain, node.op_type) == ("com.microsoft", "QMoE")
    assert len(node.inputs) == 23
    assert {index for index, value in enumerate(node.inputs) if value is not None} == {
        0,
        2,
        5,
        21,
        22,
    }
    assert node.inputs[0] is x
    assert list(node.inputs[2].shape) == [3, 512, 1, 66]
    assert list(node.inputs[5].shape) == [3, 256, 2, 66]
    assert node.inputs[21].dtype == ir.DataType.INT64
    assert node.inputs[22].dtype == ir.DataType.FLOAT
    assert node.attributes["quant_type"].as_string() == "iq2_xxs"
    assert node.attributes["activation_type"].as_string() == "relu2"
    for name, expected in {
        "k": 2,
        "expert_weight_bits": 2,
        "block_size": 256,
        "normalize_routing_weights": 0,
        "use_sparse_mixer": 0,
        "swiglu_fusion": 0,
        "weights_prepacked": 0,
    }.items():
        assert node.attributes[name].as_int() == expected
    assert output.dtype == dtype
    assert output.shape == x.shape
    assert graph.opset_imports["com.microsoft"] == 1
    assert not any(node.op_type.startswith("MatMul") for node in graph)


@pytest.mark.parametrize(
    "dimensions",
    [
        (0, 256, 2, 1),
        (256, 0, 2, 1),
        (255, 256, 2, 1),
        (256, 257, 2, 1),
        (256, 256, 0, 1),
        (256, 256, 2, 0),
        (256, 256, 2, 3),
        (256, 256, 40, 33),
    ],
)
def test_sparse_iq2_rejects_unsupported_topology(dimensions):
    with pytest.raises(ValueError, match="multiples of 256"):
        GGMLIQ2XXSMoE(*dimensions)


@pytest.mark.parametrize("provider", ["default", "cpu", "dml", "webgpu", "trt-rtx"])
def test_sparse_iq2_rejects_unsupported_provider(provider):
    module = GGMLIQ2XXSMoE(256, 256, 2, 1)
    builder, op, _ = create_test_builder()
    x = create_test_input(builder, "x", [2, 256])
    ids = create_test_input(builder, "ids", [2, 1], ir.DataType.INT64)
    weights = create_test_input(builder, "weights", [2, 1])
    with (
        build_context(ep_registry.require(provider)),
        pytest.raises(ValueError, match="companion CUDA QMoE"),
    ):
        module(op, x, weights, ids)


def test_sparse_iq2_requires_dtype_capability_not_just_fused_moe():
    caps = dataclasses.replace(ep_registry.require("cuda"), iq2_moe_dtypes=frozenset())
    module = GGMLIQ2XXSMoE(256, 256, 2, 1)
    builder, op, _ = create_test_builder()
    x = create_test_input(builder, "x", [2, 256])
    ids = create_test_input(builder, "ids", [2, 1], ir.DataType.INT64)
    weights = create_test_input(builder, "weights", [2, 1])
    assert caps.supports_fused_moe
    with build_context(caps), pytest.raises(ValueError, match="unsupported"):
        module(op, x, weights, ids)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16])
def test_router_uses_biased_selection_but_unbiased_fp32_weights(
    tmp_path, normalize, input_dtype
):
    gate = NemotronHMoEGate(2, 3, 2, norm_topk_prob=normalize, routed_scaling_factor=5.0)
    matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    bias = torch.tensor([-2.0, 2.0, 0.1])
    gate.weight.const_value = tensor_adapters.TorchTensor(matrix)
    gate.e_score_correction_bias.const_value = tensor_adapters.TorchTensor(bias)
    builder, op, graph = create_test_builder()
    dtype = ir.DataType.FLOAT if input_dtype == torch.float32 else ir.DataType.FLOAT16
    x = create_test_input(builder, "x", [3, 2], dtype)
    weights, ids = gate(op, x)
    for value, name, value_dtype in (
        (weights, "weights", ir.DataType.FLOAT),
        (ids, "ids", ir.DataType.INT64),
    ):
        value.name = name
        value.dtype = value_dtype
        value.shape = ir.Shape([3, 2])
        graph.outputs.append(value)
    path = tmp_path / "gate.onnx"
    # This standalone router uses no opset-24 operations; exercise older CPU ORT too.
    graph.opset_imports[""] = 23
    ir.save(ir.Model(graph, ir_version=11), path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    inputs = torch.tensor([[1.0, -2.0], [-3.0, 4.0], [-1000.0, -1000.0]]).to(input_dtype)
    actual_weights, actual_ids = session.run(None, {"x": inputs.numpy()})
    probs = torch.sigmoid(inputs.float() @ matrix.T)
    selected = torch.topk(probs + bias, 2, dim=-1).indices
    expected = probs.gather(-1, selected)
    if normalize:
        expected = expected / (expected.sum(-1, keepdim=True) + 1e-20)
    expected = expected * 5.0
    np.testing.assert_array_equal(actual_ids, selected.numpy())
    np.testing.assert_allclose(actual_weights, expected.numpy(), rtol=1e-6, atol=1e-7)
    assert actual_weights.dtype == np.float32
    assert np.isfinite(actual_weights).all()


def test_dense_diagnostic_dispatch_still_matches_independent_cpu_oracle(tmp_path):
    config = NemotronHConfig(
        hidden_size=16,
        moe_latent_size=8,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=24,
        num_local_experts=3,
        num_experts_per_tok=2,
        hidden_act="relu2",
        mlp_bias=False,
        routed_scaling_factor=5.0,
    )
    module = NemotronHMoEBlock(config)
    generator = torch.Generator().manual_seed(81)
    parameters = {}
    for name, parameter in module.named_parameters():
        value = torch.randn(tuple(parameter.shape), generator=generator) * 0.1
        parameter.const_value = tensor_adapters.TorchTensor(value)
        parameters[name] = value
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [2, 3, 16])
    output = module(op, x)
    output.name = "output"
    output.dtype = ir.DataType.FLOAT
    output.shape = x.shape
    graph.outputs.append(output)
    graph.opset_imports[""] = 23
    path = tmp_path / "dense.onnx"
    ir.save(ir.Model(graph, ir_version=11), path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    inputs = torch.randn((2, 3, 16), generator=generator)
    expected = _reference_moe(inputs, parameters, config).numpy()
    actual = session.run(None, {"x": inputs.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(
    os.environ.get("MOBIUS_TEST_SPARSE_IQ2_CUDA") != "1",
    reason="Requires opt-in and the companion CUDA QMoE IQ2 runtime; not stock ORT",
)
@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
@pytest.mark.parametrize("leading", [[2, 3], [1], [0], [2, 0]])
def test_sparse_iq2_real_cuda_matches_native_dense_diagnostic(tmp_path, dtype, leading):
    # Opt-in is a hard gate: missing CUDA/kernel support must fail, not be skipped.
    assert "CUDAExecutionProvider" in ort.get_available_providers()
    config = NemotronHConfig(
        hidden_size=256,
        moe_latent_size=256,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=32,
        num_local_experts=3,
        num_experts_per_tok=2,
        hidden_act="relu2",
        mlp_bias=False,
        routed_scaling_factor=5.0,
        dtype=dtype,
    )
    dense = NemotronHMoEBlock(config)
    for expert in dense.experts:
        expert.up_proj = GGMLIQ2XXSLinear(256, 256)
        expert.down_proj = GGMLIQ2XXSLinear(256, 256)
    _cast_module_dtype(dense, dtype)
    generator = torch.Generator().manual_seed(29)
    parameters = {}
    for name, parameter in dense.named_parameters():
        shape = tuple(parameter.shape)
        if parameter.dtype == ir.DataType.UINT8:
            value = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
            value[..., :2] = torch.tensor([0.01], dtype=torch.float16).view(torch.uint8)
        else:
            value = (torch.randn(shape, generator=generator) * 0.1).to(
                tensor_adapters.to_torch_dtype(parameter.dtype)
            )
        parameter.const_value = tensor_adapters.TorchTensor(value)
        parameters[name] = value
    sparse = _SparseIQ2MoEBlock(dense, config)
    for projection in ("up_proj", "down_proj"):
        bank = torch.stack(
            [parameters[f"experts.{index}.{projection}.weight"] for index in range(3)]
        )
        getattr(sparse.packed_experts, projection).const_value = tensor_adapters.TorchTensor(
            bank
        )
    inputs = (torch.randn((*leading, 256), generator=generator) * 0.1).to(
        tensor_adapters.to_torch_dtype(dtype)
    )
    raw_inputs = inputs.contiguous().view(torch.uint8).numpy().view(dtype.numpy())
    results = []
    for name, module in (("dense", dense), ("sparse", sparse)):
        builder, op, graph = create_test_builder()
        shape = ["tokens", 256] if len(leading) == 1 else ["batch", "sequence", 256]
        x = create_test_input(builder, "x", shape, dtype)
        with build_context(ep_registry.require("cuda"), dtype):
            output = module(op, x)
        output.name = "output"
        output.dtype = dtype
        output.shape = x.shape
        graph.outputs.append(output)
        path = tmp_path / f"{name}.onnx"
        ir.save(ir.Model(graph, ir_version=11), path)
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=[("CUDAExecutionProvider", {"use_tf32": "0", "enable_cuda_graph": "0"})],
        )
        result = session.run_with_ort_values(
            None, {"x": _numpy_to_ort_value(raw_inputs.reshape(inputs.shape))}
        )[0]
        results.append(_ort_value_to_numpy(result).astype(np.float32))
    assert results[0].shape == results[1].shape == tuple(inputs.shape)
    np.testing.assert_allclose(
        results[1],
        results[0],
        atol=1e-4 if dtype == ir.DataType.FLOAT else 0.003,
        rtol=1e-4 if dtype == ir.DataType.FLOAT else 0.03,
    )
