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
from mobius._weight_utils import (
    pack_qmoe_expert_weights,
    preprocess_gptq_weights,
    preprocess_olive_weights,
)
from mobius.components._moe import (
    MoELayer,
    SparseMixerGate,
    TopKGate,
    _supported_qmoe_quantization,
)


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

    def test_int4_olive_moe_emits_expert_major_qmoe(self):
        """Olive (blk32 int4) MoE emits a fused QMoE node, not per-expert MLPs."""
        config = make_config(
            hidden_size=64,
            intermediate_size=32,
            moe_intermediate_size=32,
            num_local_experts=64,
            num_experts_per_tok=6,
            quantization=QuantizationConfig(
                bits=4,
                group_size=32,
                quant_method="olive",
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
        assert qmoe.attributes["expert_weight_bits"].value == 4
        assert layer.experts is None
        assert layer.fc1_experts_weights.shape == ir.Shape([64, 64, 32])
        assert layer.fc2_experts_weights.shape == ir.Shape([64, 64, 16])
        # No per-expert MatMulNBits expert storm: only the router MatMul remains.
        assert sum(node.op_type == "MatMul" for node in graph) == 1
        assert sum(node.op_type == "MatMulNBits" for node in graph) == 0

    def test_olive_group_size_not_power_of_two_falls_back_to_dense(self):
        """Non-pow2 block_size is unrunnable by CUDA QMoE -> dense fallback."""
        quant = QuantizationConfig(bits=4, group_size=48, quant_method="olive", sym=False)
        assert _supported_qmoe_quantization(quant) is None
        config = make_config(
            hidden_size=96,
            intermediate_size=48,
            moe_intermediate_size=48,
            num_local_experts=4,
            num_experts_per_tok=2,
            quantization=quant,
        )
        layer = MoELayer(config)
        assert layer.experts is not None
        assert len(layer.experts) == 4

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
        fc1_codes = torch.randint(0, 16, (num_experts, 2 * intermediate_size, hidden_size))
        fc2_codes = torch.randint(0, 16, (num_experts, hidden_size, intermediate_size))
        fc1_scales = torch.rand(num_experts, 2 * intermediate_size, hidden_size // block_size)
        fc2_scales = torch.rand(num_experts, hidden_size, intermediate_size // block_size)
        raw = {
            "model.layers.1.mlp.experts.gate_up_proj.qweight": _to_gptq_qweight(fc1_codes),
            "model.layers.1.mlp.experts.gate_up_proj.scales": fc1_scales.transpose(-1, -2),
            "model.layers.1.mlp.experts.down_proj.qweight": _to_gptq_qweight(fc2_codes),
            "model.layers.1.mlp.experts.down_proj.scales": fc2_scales.transpose(-1, -2),
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

    def test_olive_fused_expert_packing_is_byte_exact_vs_dense(self):
        """Olive fused expert-major weights repack to QMoE byte-exactly.

        Feeds synthetic Olive ``.qweight``/``.scales``/``.qzeros`` fused expert
        tensors (asymmetric int4, blk32) through ``preprocess_olive_weights`` +
        ``pack_qmoe_expert_weights`` and asserts the dequantized QMoE weights --
        and a full loop-over-experts forward -- match the dense per-expert
        reference to fp32 rounding. This is a pure layout transform, so the two
        paths must agree exactly.
        """
        torch.manual_seed(0)
        num_experts, top_k = 16, 4
        hidden_size, intermediate_size, block_size = 32, 16, 16
        fc1_out = 2 * intermediate_size
        fc1_blocks = hidden_size // block_size
        fc2_blocks = intermediate_size // block_size

        fc1_codes = torch.randint(0, 16, (num_experts, fc1_out, hidden_size))
        fc2_codes = torch.randint(0, 16, (num_experts, hidden_size, intermediate_size))
        fc1_scales = torch.rand(num_experts, fc1_out, fc1_blocks)
        fc2_scales = torch.rand(num_experts, hidden_size, fc2_blocks)
        fc1_zp = torch.randint(0, 16, (num_experts, fc1_out, fc1_blocks))
        fc2_zp = torch.randint(0, 16, (num_experts, hidden_size, fc2_blocks))

        raw = {
            "model.layers.1.mlp.experts.gate_up_proj.qweight": _to_olive_qweight(fc1_codes),
            "model.layers.1.mlp.experts.gate_up_proj.scales": fc1_scales,
            "model.layers.1.mlp.experts.gate_up_proj.qzeros": _to_olive_qzeros(fc1_zp),
            "model.layers.1.mlp.experts.down_proj.qweight": _to_olive_qweight(fc2_codes),
            "model.layers.1.mlp.experts.down_proj.scales": fc2_scales,
            "model.layers.1.mlp.experts.down_proj.qzeros": _to_olive_qzeros(fc2_zp),
        }
        packed = pack_qmoe_expert_weights(
            preprocess_olive_weights(raw, bits=4, group_size=block_size),
            target_moe_path=".mlp",
        )
        fc1_packed = packed["model.layers.1.mlp.fc1_experts_weights"]
        fc2_packed = packed["model.layers.1.mlp.fc2_experts_weights"]
        fc1_zp_packed = packed["model.layers.1.mlp.fc1_experts_zero_points"]
        fc2_zp_packed = packed["model.layers.1.mlp.fc2_experts_zero_points"]
        assert fc1_packed.shape == (num_experts, fc1_out, hidden_size // 2)
        assert fc2_packed.shape == (num_experts, hidden_size, intermediate_size // 2)

        qmoe_fc1 = _dequant_qmoe_asym(fc1_packed, fc1_scales, fc1_zp_packed, block_size)
        qmoe_fc2 = _dequant_qmoe_asym(fc2_packed, fc2_scales, fc2_zp_packed, block_size)
        dense_fc1 = _dequant_codes_asym(fc1_codes, fc1_scales, fc1_zp, block_size)
        dense_fc2 = _dequant_codes_asym(fc2_codes, fc2_scales, fc2_zp, block_size)
        torch.testing.assert_close(qmoe_fc1, dense_fc1, atol=0, rtol=0)
        torch.testing.assert_close(qmoe_fc2, dense_fc2, atol=0, rtol=0)

        hidden = torch.randn(5, hidden_size)
        router_probs = torch.softmax(torch.randn(5, num_experts), dim=-1)
        selected_weights, selected_experts = router_probs.topk(top_k, dim=-1)
        selected_weights /= selected_weights.sum(dim=-1, keepdim=True)
        dense = _static_moe(
            hidden, selected_weights, selected_experts, dense_fc1, dense_fc2, intermediate_size
        )
        fused = _static_moe(
            hidden, selected_weights, selected_experts, qmoe_fc1, qmoe_fc2, intermediate_size
        )
        torch.testing.assert_close(fused, dense, atol=1e-5, rtol=1e-5)


def _to_olive_qweight(codes: torch.Tensor) -> torch.Tensor:
    """Pack output-major int4 codes into Olive's [..., N, K/2] uint8 layout.

    Low nibble holds element ``2j``, high nibble element ``2j+1`` -- the same
    little-endian nibble order ORT MatMulNBits and QMoE decode.
    """
    low = codes[..., 0::2]
    high = codes[..., 1::2]
    return (low | (high << 4)).to(torch.uint8).contiguous()


def _to_olive_qzeros(zp_codes: torch.Tensor) -> torch.Tensor:
    """Pack per-block int4 zero-points into [..., N, ceil(n_blocks/2)] uint8."""
    n_blocks = zp_codes.shape[-1]
    if n_blocks % 2:
        pad = torch.zeros(*zp_codes.shape[:-1], 1, dtype=zp_codes.dtype)
        zp_codes = torch.cat((zp_codes, pad), dim=-1)
    low = zp_codes[..., 0::2]
    high = zp_codes[..., 1::2]
    return (low | (high << 4)).to(torch.uint8).contiguous()


def _unpack_zp(packed_zp: torch.Tensor, n_blocks: int) -> torch.Tensor:
    """Unpack [..., ceil(n_blocks/2)] uint8 zero-points to [..., n_blocks]."""
    low = packed_zp & 0x0F
    high = packed_zp >> 4
    codes = torch.stack((low, high), dim=-1).flatten(-2)
    return codes[..., :n_blocks].to(torch.float32)


def _dequant_codes_asym(codes, scales, zp_codes, block_size):
    blocks = codes.shape[-1] // block_size
    values = codes.reshape(*codes.shape[:-1], blocks, block_size).float()
    values = values - zp_codes.float().unsqueeze(-1)
    return (values * scales.unsqueeze(-1)).flatten(-2)


def _dequant_qmoe_asym(packed, scales, packed_zp, block_size):
    low = packed & 0x0F
    high = packed >> 4
    codes = torch.stack((low, high), dim=-1).flatten(-2)
    blocks = codes.shape[-1] // block_size
    zp = _unpack_zp(packed_zp, blocks)
    return _dequant_codes_asym(codes, scales, zp, block_size)


def _to_gptq_qweight(codes: torch.Tensor) -> torch.Tensor:
    """Pack output-major int4 codes into GPTQ's [E, K/8, N] int32 layout."""
    shifts = torch.arange(8, dtype=torch.int64) * 4
    packed = torch.sum(
        codes.to(torch.int64).reshape(*codes.shape[:-1], -1, 8) << shifts,
        dim=-1,
    ).to(torch.int32)
    return packed.transpose(-1, -2).contiguous()


def _dequant_codes(codes: torch.Tensor, scales: torch.Tensor, block_size: int) -> torch.Tensor:
    blocks = codes.shape[-1] // block_size
    values = codes.reshape(*codes.shape[:-1], blocks, block_size).float() - 8.0
    return (values * scales.unsqueeze(-1)).flatten(-2)


def _dequant_qmoe(packed: torch.Tensor, scales: torch.Tensor, block_size: int) -> torch.Tensor:
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
            routing_weights * (selected_experts == expert_idx).to(routing_weights.dtype)
        ).sum(dim=-1, keepdim=True)
        result += expert_output * weight
    return result
