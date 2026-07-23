# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for MoE components."""

from __future__ import annotations

import onnx_ir as ir
import pytest
import torch

from mobius._builder import _cast_module_dtype
from mobius._configs import QuantizationConfig
from mobius._testing import (
    create_test_builder,
    create_test_input,
    make_config,
)
from mobius._weight_utils import pack_qmoe_expert_weights, preprocess_gptq_weights
from mobius.components._moe import MoELayer, SparseMixerGate, TopKGate


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

    def test_int4_moe_emits_expert_major_qmoe(self):
        config = make_config(
            hidden_size=64,
            intermediate_size=32,
            moe_intermediate_size=32,
            num_local_experts=64,
            num_experts_per_tok=6,
            quantization=QuantizationConfig(
                bits=4,
                group_size=32,
                quant_method="gptq",
                sym=False,
            ),
        )
        layer = MoELayer(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        result = layer(op, hidden)
        builder._adapt_outputs([result], "")

        qmoe_nodes = [node for node in graph if node.op_type == "QMoE"]
        assert len(qmoe_nodes) == 1
        qmoe = qmoe_nodes[0]
        assert qmoe.attributes["k"].value == 6
        assert qmoe.attributes["block_size"].value == 32
        assert qmoe.attributes["swiglu_fusion"].value == 2
        assert layer.fc1_experts_weights.shape == ir.Shape([64, 64, 32])
        assert layer.fc1_scales.shape == ir.Shape([64, 64, 2])
        assert layer.fc1_experts_zero_points.shape == ir.Shape([64, 64, 1])
        assert layer.fc2_experts_weights.shape == ir.Shape([64, 64, 16])
        assert layer.fc2_scales.shape == ir.Shape([64, 64, 1])
        assert layer.fc2_experts_zero_points.shape == ir.Shape([64, 64, 1])
        assert sum(node.op_type == "MatMul" for node in graph) == 1  # router only

        _cast_module_dtype(layer, ir.DataType.FLOAT16)
        assert layer.gate.weight.dtype == ir.DataType.FLOAT16
        assert layer.fc1_scales.dtype == ir.DataType.FLOAT
        assert layer.fc2_scales.dtype == ir.DataType.FLOAT

    def test_expert_major_packing_matches_static_64_expert_top6_reference(self):
        """Packed QMoE math matches the existing loop-over-experts semantics."""
        torch.manual_seed(0)
        num_experts, top_k = 64, 6
        hidden_size, intermediate_size, block_size = 32, 16, 16
        fc1_codes = torch.randint(
            0, 16, (num_experts, 2 * intermediate_size, hidden_size)
        )
        fc2_codes = torch.randint(
            0, 16, (num_experts, hidden_size, intermediate_size)
        )
        fc1_scales = torch.rand(
            num_experts, 2 * intermediate_size, hidden_size // block_size
        )
        fc2_scales = torch.rand(
            num_experts, hidden_size, intermediate_size // block_size
        )
        raw = {
            "model.layers.1.mlp.experts.gate_up_proj.qweight": _to_gptq_qweight(
                fc1_codes
            ),
            "model.layers.1.mlp.experts.gate_up_proj.scales": fc1_scales.transpose(
                -1, -2
            ),
            "model.layers.1.mlp.experts.down_proj.qweight": _to_gptq_qweight(
                fc2_codes
            ),
            "model.layers.1.mlp.experts.down_proj.scales": fc2_scales.transpose(
                -1, -2
            ),
        }
        packed = pack_qmoe_expert_weights(
            preprocess_gptq_weights(raw, bits=4, group_size=block_size)
        )
        fc1_packed = packed["model.layers.1.mlp.moe.fc1_experts_weights"]
        fc2_packed = packed["model.layers.1.mlp.moe.fc2_experts_weights"]
        assert fc1_packed.shape == (num_experts, 2 * intermediate_size, hidden_size // 2)
        assert fc2_packed.shape == (num_experts, hidden_size, intermediate_size // 2)

        packed_fc1 = _dequant_qmoe(fc1_packed, fc1_scales, block_size)
        packed_fc2 = _dequant_qmoe(fc2_packed, fc2_scales, block_size)
        static_fc1 = _dequant_codes(fc1_codes, fc1_scales, block_size)
        static_fc2 = _dequant_codes(fc2_codes, fc2_scales, block_size)

        hidden = torch.randn(3, hidden_size)
        router_probs = torch.softmax(torch.randn(3, num_experts), dim=-1)
        selected_weights, selected_experts = router_probs.topk(top_k, dim=-1)
        selected_weights /= selected_weights.sum(dim=-1, keepdim=True)
        static = _static_moe(
            hidden,
            selected_weights,
            selected_experts,
            static_fc1,
            static_fc2,
            intermediate_size,
        )
        fused_reference = _static_moe(
            hidden,
            selected_weights,
            selected_experts,
            packed_fc1,
            packed_fc2,
            intermediate_size,
        )
        torch.testing.assert_close(fused_reference, static, atol=1e-5, rtol=1e-5)


def _to_gptq_qweight(codes: torch.Tensor) -> torch.Tensor:
    """Pack output-major int4 codes into GPTQ's [E, K/8, N] int32 layout."""
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


def _dequant_qmoe(
    packed: torch.Tensor, scales: torch.Tensor, block_size: int
) -> torch.Tensor:
    low = packed & 0x0F
    high = packed >> 4
    codes = torch.stack((low, high), dim=-1).flatten(-2)
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
    for expert_idx in range(fc1.shape[0]):
        projected = hidden @ fc1[expert_idx].T
        activated = torch.nn.functional.silu(projected[:, :intermediate_size])
        activated *= projected[:, intermediate_size:]
        expert_output = activated @ fc2[expert_idx].T
        weight = (
            routing_weights
            * (selected_experts == expert_idx).to(routing_weights.dtype)
        ).sum(dim=-1, keepdim=True)
        result += expert_output * weight
    return result
