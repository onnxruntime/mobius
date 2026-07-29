# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for MoE components."""

from __future__ import annotations

import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import QuantizationConfig
from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
    make_config,
)
from mobius.components._moe import (
    FusedQuantizedMoE,
    MoELayer,
    SparseMixerGate,
    TopKGate,
    pack_fused_quantized_moe_weights,
)
from mobius.models.deepseek import DeepSeekMoEGate, DeepSeekV3CausalLMModel


class TestTopKGate:
    def test_gate_has_weight_parameter(self):
        gate = TopKGate(hidden_size=64, num_experts=4, top_k=2)
        param_names = [n for n, _ in gate.named_parameters()]
        assert "weight" in param_names

    def test_gate_weight_shape(self):
        gate = TopKGate(hidden_size=64, num_experts=4, top_k=2)
        assert gate.weight.shape == ir.Shape([4, 64])

    def test_gate_forward_produces_outputs(self):
        gate = TopKGate(hidden_size=64, num_experts=4, top_k=2)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        routing_weights, selected_experts = gate(op, hidden)
        builder._adapt_outputs([routing_weights, selected_experts], "")
        assert graph.num_nodes() > 0


class TestSparseMixerGate:
    def test_gate_has_weight_parameter(self):
        gate = SparseMixerGate(hidden_size=64, num_experts=4, top_k=2)
        param_names = [n for n, _ in gate.named_parameters()]
        assert "weight" in param_names

    def test_gate_weight_shape(self):
        gate = SparseMixerGate(hidden_size=64, num_experts=4, top_k=2)
        assert gate.weight.shape == ir.Shape([4, 64])

    def test_gate_forward_produces_outputs(self):
        gate = SparseMixerGate(hidden_size=64, num_experts=4, top_k=2)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        routing_weights, selected_experts = gate(op, hidden)
        builder._adapt_outputs([routing_weights, selected_experts], "")
        assert graph.num_nodes() > 0

    def test_gate_custom_jitter_eps(self):
        gate = SparseMixerGate(hidden_size=64, num_experts=4, top_k=2, jitter_eps=0.05)
        assert gate.jitter_eps == pytest.approx(0.05)

    def test_gate_top_k_1(self):
        gate = SparseMixerGate(hidden_size=64, num_experts=4, top_k=1)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        routing_weights, selected_experts = gate(op, hidden)
        builder._adapt_outputs([routing_weights, selected_experts], "")
        assert graph.num_nodes() > 0


@pytest.mark.parametrize("route_for_qmoe", [False, True])
def test_deepseek_gate_casts_both_router_matmul_inputs_to_float32(route_for_qmoe):
    config = make_config(
        dtype=ir.DataType.FLOAT16,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    gate = DeepSeekMoEGate(config)
    builder, op, graph = create_test_builder()
    hidden = create_test_input(
        builder,
        "hidden",
        [1, 2, config.hidden_size],
        dtype=ir.DataType.FLOAT16,
    )

    outputs = gate.route_for_qmoe(op, hidden) if route_for_qmoe else gate(op, hidden)
    builder._adapt_outputs(list(outputs), "")

    router_matmul = next(node for node in graph if node.op_type == "MatMul")
    assert [value.dtype for value in router_matmul.inputs] == [
        ir.DataType.FLOAT,
        ir.DataType.FLOAT,
    ]


class TestMoELayer:
    def test_moe_layer_has_gate(self):
        config = make_config(num_local_experts=4, num_experts_per_tok=2)
        layer = MoELayer(config)
        param_names = [n for n, _ in layer.named_parameters()]
        assert any("gate" in n for n in param_names)

    def test_moe_layer_has_experts(self):
        config = make_config(num_local_experts=4, num_experts_per_tok=2)
        layer = MoELayer(config)
        param_names = [n for n, _ in layer.named_parameters()]
        assert any("experts.0" in n for n in param_names)
        assert any("experts.3" in n for n in param_names)

    def test_moe_layer_num_experts(self):
        config = make_config(num_local_experts=8, num_experts_per_tok=2)
        layer = MoELayer(config)
        assert len(layer.experts) == 8

    def test_moe_layer_forward(self):
        config = make_config(num_local_experts=4, num_experts_per_tok=2)
        layer = MoELayer(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        result = layer(op, hidden)
        builder._adapt_outputs([result], "")
        assert graph.num_nodes() > 0

    def test_moe_layer_requires_expert_config(self):
        config = make_config()  # No MoE config
        with pytest.raises(AssertionError):
            MoELayer(config)

    def test_moe_layer_with_custom_gate(self):
        config = make_config(num_local_experts=4, num_experts_per_tok=2)
        gate = SparseMixerGate(
            config.hidden_size, config.num_local_experts, config.num_experts_per_tok
        )
        layer = MoELayer(config, gate=gate)
        assert isinstance(layer.gate, SparseMixerGate)

    def test_moe_layer_forward_with_sparse_mixer_gate(self):
        config = make_config(num_local_experts=4, num_experts_per_tok=2)
        gate = SparseMixerGate(
            config.hidden_size, config.num_local_experts, config.num_experts_per_tok
        )
        layer = MoELayer(config, gate=gate)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        result = layer(op, hidden)
        builder._adapt_outputs([result], "")
        assert graph.num_nodes() > 0


def test_fused_qmoe_wires_asymmetric_zero_points():
    config = make_config(
        hidden_size=32,
        intermediate_size=16,
        moe_intermediate_size=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        quantization=QuantizationConfig(
            bits=4,
            group_size=16,
            quant_method="gptq",
            sym=False,
        ),
    )
    layer = FusedQuantizedMoE(config, DeepSeekMoEGate(config))
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 2, 32])
    builder._adapt_outputs([layer(op, hidden)], "")

    qmoe = next(node for node in graph if node.op_type == "QMoE")
    assert qmoe.inputs[11] is not None
    assert qmoe.inputs[12] is not None
    assert qmoe.inputs[3] is not None
    assert qmoe.inputs[3].producer().op_type == "Cast"
    assert qmoe.inputs[3].dtype == ir.DataType.FLOAT
    assert layer.fc1_experts_zero_points.shape == ir.Shape([4, 32, 1])
    assert layer.fc2_experts_zero_points.shape == ir.Shape([4, 32, 1])


def test_fused_qmoe_keeps_activation_input_model_dtype():
    config = make_config(
        hidden_size=32,
        intermediate_size=16,
        moe_intermediate_size=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        quantization=QuantizationConfig(
            bits=4,
            group_size=16,
            quant_method="gptq",
            sym=True,
        ),
    )
    layer = FusedQuantizedMoE(config, DeepSeekMoEGate(config))
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 2, 32], dtype=ir.DataType.FLOAT16)
    builder._adapt_outputs([layer(op, hidden)], "")

    qmoe = next(node for node in graph if node.op_type == "QMoE")
    assert qmoe.inputs[0].dtype == ir.DataType.FLOAT16
    assert qmoe.inputs[1].dtype == ir.DataType.FLOAT


def test_deepseek_fused_qmoe_graph_has_one_node_per_moe_layer():
    config = make_config(
        hidden_size=32,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        first_k_dense_replace=1,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        fused_quantized_moe=True,
        quantization=QuantizationConfig(
            bits=4,
            group_size=16,
            quant_method="gptq",
            sym=True,
        ),
    )
    graph = build_from_module(DeepSeekV3CausalLMModel(config), config)["model"].graph

    assert count_op_type(graph, "QMoE") == 1
    assert not any(
        value is not None
        and value.name is not None
        and ".moe.experts." in value.name
        for node in graph
        for value in node.inputs
    )
    shared_prefix = "model.layers.1.mlp.shared_experts."
    assert (
        sum(
            node.op_type == "MatMulNBits"
            and any(
                value is not None
                and value.name is not None
                and shared_prefix in value.name
                for value in node.inputs
            )
            for node in graph
        )
        == 3
    )


def test_deepseek_fused_qmoe_rejects_cuda():
    config = make_config(
        hidden_size=32,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        first_k_dense_replace=1,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        fused_quantized_moe=True,
        quantization=QuantizationConfig(
            bits=4,
            group_size=16,
            quant_method="gguf",
            sym=True,
        ),
    )
    with pytest.raises(ValueError, match="CUDA.*ignores router_weights"):
        build_from_module(
            DeepSeekV3CausalLMModel(config),
            config,
            execution_provider="cuda",
        )


def test_expert_major_packing_matches_static_64_expert_top6_reference():
    torch.manual_seed(0)
    experts, top_k = 64, 6
    hidden_size, intermediate_size, block_size = 32, 16, 16
    gate_codes = torch.randint(0, 16, (experts, intermediate_size, hidden_size))
    up_codes = torch.randint(0, 16, (experts, intermediate_size, hidden_size))
    down_codes = torch.randint(0, 16, (experts, hidden_size, intermediate_size))
    gate_scales = torch.rand(experts, intermediate_size, hidden_size // block_size)
    up_scales = torch.rand(experts, intermediate_size, hidden_size // block_size)
    down_scales = torch.rand(experts, hidden_size, intermediate_size // block_size)
    state_dict = {}
    for expert in range(experts):
        prefix = f"model.layers.0.mlp.experts.{expert}"
        for name, codes, scales in (
            ("gate_proj", gate_codes, gate_scales),
            ("up_proj", up_codes, up_scales),
            ("down_proj", down_codes, down_scales),
        ):
            state_dict[f"{prefix}.{name}.weight"] = _pack_matmul_nbits(
                codes[expert], block_size
            )
            state_dict[f"{prefix}.{name}.scales"] = scales[expert]

    config = make_config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_intermediate_size=intermediate_size,
        num_local_experts=experts,
        num_experts_per_tok=top_k,
        fused_quantized_moe=True,
        quantization=QuantizationConfig(
            bits=4,
            group_size=block_size,
            quant_method="gguf",
            sym=True,
        ),
    )
    packed = DeepSeekV3CausalLMModel(config).preprocess_weights(state_dict)
    fc1_weight = packed["model.layers.0.mlp.moe.fc1_experts_weights"]
    fc1_scales = packed["model.layers.0.mlp.moe.fc1_scales"]
    fc2_weight = packed["model.layers.0.mlp.moe.fc2_experts_weights"]
    fc2_scales = packed["model.layers.0.mlp.moe.fc2_scales"]
    assert fc1_weight.shape == (experts, 2 * intermediate_size, hidden_size // 2)
    assert fc2_weight.shape == (experts, hidden_size, intermediate_size // 2)

    packed_fc1 = _dequant(fc1_weight, fc1_scales, block_size)
    packed_fc2 = _dequant(fc2_weight, fc2_scales, block_size)
    static_fc1 = torch.stack(
        (
            _dequant_codes(gate_codes, gate_scales, block_size),
            _dequant_codes(up_codes, up_scales, block_size),
        ),
        dim=2,
    ).flatten(1, 2)
    static_fc2 = _dequant_codes(down_codes, down_scales, block_size)

    hidden = torch.randn(3, hidden_size)
    router_probs = torch.softmax(torch.randn(3, experts), dim=-1)
    weights, selected = router_probs.topk(top_k, dim=-1)
    weights /= weights.sum(dim=-1, keepdim=True)
    expected = _static_moe(
        hidden, weights, selected, static_fc1, static_fc2, intermediate_size
    )
    actual = _static_moe(
        hidden, weights, selected, packed_fc1, packed_fc2, intermediate_size
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shared_g_idx", [False, True])
def test_fused_gptq_checkpoint_tensors_pack_to_qmoe_layout(caplog, shared_g_idx):
    experts, hidden_size, intermediate_size, block_size = 4, 32, 16, 16
    gate_up_codes = torch.randint(
        0, 16, (experts, 2 * intermediate_size, hidden_size)
    )
    down_codes = torch.randint(0, 16, (experts, hidden_size, intermediate_size))
    gate_up_scales = torch.rand(
        experts, 2 * intermediate_size, hidden_size // block_size
    )
    down_scales = torch.rand(
        experts, hidden_size, intermediate_size // block_size
    )
    gate_up_g_idx = torch.arange(hidden_size, dtype=torch.int32) // block_size
    gate_up_g_idx[0] = 1
    down_g_idx = torch.arange(intermediate_size, dtype=torch.int32) // block_size
    if not shared_g_idx:
        gate_up_g_idx = gate_up_g_idx.repeat(experts, 1)
        down_g_idx = down_g_idx.repeat(experts, 1)
    state_dict = {
        "model.layers.0.mlp.experts.gate_up_proj.qweight": _to_gptq_qweight(
            gate_up_codes
        ),
        "model.layers.0.mlp.experts.gate_up_proj.scales": gate_up_scales.transpose(
            -1, -2
        ),
        "model.layers.0.mlp.experts.gate_up_proj.g_idx": gate_up_g_idx,
        "model.layers.0.mlp.experts.down_proj.qweight": _to_gptq_qweight(
            down_codes
        ),
        "model.layers.0.mlp.experts.down_proj.scales": down_scales.transpose(-1, -2),
        "model.layers.0.mlp.experts.down_proj.g_idx": down_g_idx,
    }
    config = make_config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_intermediate_size=intermediate_size,
        num_local_experts=experts,
        num_experts_per_tok=2,
        fused_quantized_moe=True,
        quantization=QuantizationConfig(
            bits=4,
            group_size=block_size,
            quant_method="gptq",
            sym=True,
        ),
    )

    packed = pack_fused_quantized_moe_weights(state_dict, config)
    fc1 = packed["model.layers.0.mlp.moe.fc1_experts_weights"]
    scales = packed["model.layers.0.mlp.moe.fc1_scales"]
    expected_codes = gate_up_codes.reshape(
        experts, 2, intermediate_size, hidden_size
    ).transpose(1, 2).flatten(1, 2)
    expected_scales = gate_up_scales.reshape(
        experts, 2, intermediate_size, hidden_size // block_size
    ).transpose(1, 2).flatten(1, 2)

    assert fc1.shape == (experts, 2 * intermediate_size, hidden_size // 2)
    torch.testing.assert_close(
        _dequant(fc1, scales, block_size),
        _dequant_codes(expected_codes, expected_scales, block_size),
    )
    assert "desc_act models with non-trivial g_idx" in caplog.text
    assert not any(key.endswith(".g_idx") for key in packed)


def _pack_matmul_nbits(codes: torch.Tensor, block_size: int) -> torch.Tensor:
    low = codes[..., 0::2].to(torch.uint8)
    high = codes[..., 1::2].to(torch.uint8)
    return (low | (high << 4)).reshape(
        codes.shape[0], codes.shape[1] // block_size, block_size // 2
    )


def _to_gptq_qweight(codes: torch.Tensor) -> torch.Tensor:

    shifts = torch.arange(8, dtype=torch.int64) * 4
    packed = torch.sum(
        codes.to(torch.int64).reshape(*codes.shape[:-1], -1, 8) << shifts,
        dim=-1,
    ).to(torch.int32)
    return packed.transpose(-1, -2).contiguous()


def _dequant_codes(
    codes: torch.Tensor, scales: torch.Tensor, block_size: int
) -> torch.Tensor:

    blocks = codes.shape[-1] // block_size
    values = codes.reshape(*codes.shape[:-1], blocks, block_size).float() - 8.0
    return (values * scales.unsqueeze(-1)).flatten(-2)


def _dequant(
    packed: torch.Tensor, scales: torch.Tensor, block_size: int
) -> torch.Tensor:
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)

    return _dequant_codes(codes, scales, block_size)


def _static_moe(
    hidden: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    fc1: torch.Tensor,
    fc2: torch.Tensor,
    intermediate_size: int,
) -> torch.Tensor:
    result = torch.zeros_like(hidden)
    for expert in range(fc1.shape[0]):
        projected = hidden @ fc1[expert].T
        activated = torch.nn.functional.silu(projected[:, 0::2])
        activated *= projected[:, 1::2]
        assert activated.shape[-1] == intermediate_size
        expert_output = activated @ fc2[expert].T
        weight = (
            routing_weights
            * (selected_experts == expert).to(routing_weights.dtype)

        ).sum(dim=-1, keepdim=True)
        result += expert_output * weight
    return result
