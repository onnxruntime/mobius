# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Independent tiny MoE oracle for the heterogeneous Nemotron-H decoder."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from onnx_ir import tensor_adapters

from mobius._builder import _cast_module_dtype
from mobius._configs import NemotronHConfig
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import _numpy_to_ort_value, _ort_value_to_numpy
from mobius.models.nemotron_h import NemotronHMoEBlock


def _reference_moe(x, parameters, config):
    def linear(value, prefix):
        return torch.nn.functional.linear(value, parameters[f"{prefix}.weight"])

    scores = torch.sigmoid(
        torch.nn.functional.linear(x.float(), parameters["gate.weight"].float())
    )
    choices = scores + parameters["gate.e_score_correction_bias"]
    selected = torch.topk(choices, config.num_experts_per_tok, dim=-1).indices
    weights = torch.gather(scores, -1, selected)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * config.routed_scaling_factor
    latent = linear(x, "fc1_latent_proj")
    accumulated = torch.zeros_like(latent, dtype=torch.float32)
    for index in range(config.num_local_experts):
        up = linear(latent, f"experts.{index}.up_proj")
        expert = linear(torch.relu(up).square(), f"experts.{index}.down_proj")
        contribution = (weights * (selected == index)).sum(dim=-1, keepdim=True)
        accumulated += expert.float() * contribution
    routed = linear(accumulated.to(x.dtype), "fc2_latent_proj")
    shared = linear(
        torch.relu(linear(x, "shared_experts.up_proj")).square(),
        "shared_experts.down_proj",
    )
    return routed + shared


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
@pytest.mark.parametrize("topology", [(2, 1, 8), (3, 2, 12)])
def test_nemotron_h_moe_cuda_precision(tmp_path, dtype, topology):
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider is required")
    num_experts, top_k, intermediate = topology
    config = NemotronHConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        layer_types=["moe"],
        hidden_act="relu2",
        num_local_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=intermediate,
        moe_latent_size=8,
        shared_expert_intermediate_size=24,
        routed_scaling_factor=5.0,
    )
    module = NemotronHMoEBlock(config)
    _cast_module_dtype(module, dtype)
    generator = torch.Generator().manual_seed(42)
    parameters = {}
    for name, parameter in module.named_parameters():
        parameter_dtype = tensor_adapters.to_torch_dtype(parameter.dtype)
        value = (torch.randn(list(parameter.shape), generator=generator) * 0.15).to(
            parameter_dtype
        )
        parameter.const_value = tensor_adapters.TorchTensor(value)
        parameters[name] = value
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "hidden_states", [2, 3, 16], dtype)
    result = module(op, x)
    result.type = ir.TensorType(dtype)
    result.shape = ir.Shape([2, 3, 16])
    result.name = "output"
    graph.outputs.append(result)
    model = ir.Model(graph, ir_version=11)
    inputs = (torch.randn((2, 3, 16), generator=generator) * 2).to(
        tensor_adapters.to_torch_dtype(dtype)
    )
    expected = _reference_moe(inputs, parameters, config).float().numpy()
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "moe-cuda-profile")
    path = tmp_path / "moe.onnx"
    ir.save(model, path)
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"use_tf32": "0"})],
    )
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    raw_inputs = inputs.contiguous().view(torch.uint8).numpy().view(dtype.numpy())
    outputs = session.run_with_ort_values(
        None,
        {"hidden_states": _numpy_to_ort_value(raw_inputs.reshape(inputs.shape))},
    )
    actual = _ort_value_to_numpy(outputs[0])
    session.end_profiling()
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected,
        atol=1e-4 if dtype == ir.DataType.FLOAT else 0.003,
        rtol=1e-4 if dtype == ir.DataType.FLOAT else 0.03,
    )
