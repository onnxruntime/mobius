# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CUDA weight-only FP8 projections and routed/shared Qwen experts."""

from __future__ import annotations

import dataclasses
import math

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components._common import Linear
from mobius.components._mlp import MLP
from mobius.components._moe import (
    TopKGate,
    _flatten_to_2d,
    _realize_gate_and_get_qmoe_routing,
)


def quantize_fp8(
    weight: torch.Tensor, block_size: int = 128, *, square_blocks: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [N,K] weights to E4M3 with FP32 row-block or square-block scales."""
    if weight.ndim != 2 or block_size <= 0:
        raise ValueError("FP8 weights must be rank 2 with a positive block_size")
    if not weight.is_floating_point() or not torch.isfinite(weight).all():
        raise ValueError("FP8 source weights must be finite floating-point values")
    n, k = weight.shape
    if n == 0 or k == 0:
        raise ValueError("FP8 weights must have non-empty dimensions")
    row_block = block_size if square_blocks else 1
    rows, columns = math.ceil(n / row_block), math.ceil(k / block_size)
    padded = torch.nn.functional.pad(
        weight.float(), (0, columns * block_size - k, 0, rows * row_block - n)
    )
    blocks = padded.reshape(rows, row_block, columns, block_size)
    scales = blocks.abs().amax(dim=(1, 3)) / 448.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = (blocks / scales[:, None, :, None]).clamp(-448, 448)
    return (
        quantized.reshape(padded.shape)[:n, :k].to(torch.float8_e4m3fn).contiguous(),
        scales.contiguous(),
    )


class Fp8Linear(nn.Module):
    """Native W8A16 MatMulBlockQuantizedFp8Weight (no activation quantization)."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, block_size: int = 128
    ):
        super().__init__()
        if min(in_features, out_features, block_size) <= 0:
            raise ValueError("FP8 projection dimensions and block_size must be positive")
        self.block_size = block_size
        self.weight = nn.Parameter([out_features, in_features], dtype=ir.DataType.FLOAT8E4M3FN)
        self.scales = nn.Parameter(
            [out_features, math.ceil(in_features / block_size)], dtype=ir.DataType.FLOAT
        )
        self.scales._keep_float32 = True
        self.bias = nn.Parameter([out_features]) if bias else None

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return op.MatMulBlockQuantizedFp8Weight(
            x,
            self.weight,
            self.scales,
            None,
            self.bias,
            block_size=self.block_size,
            _domain="com.microsoft",
        )


class Fp8QwenMoEBlock(nn.Module):
    """Block-FP8 routed experts plus a weight-only FP8 shared expert."""

    def __init__(self, config: ArchitectureConfig, block_size: int = 128):
        super().__init__()
        if not config.num_local_experts or not config.moe_intermediate_size:
            raise ValueError("FP8 Qwen MoE requires expert count and intermediate size")
        if not config.num_experts_per_tok or not config.shared_expert_intermediate_size:
            raise ValueError("FP8 Qwen MoE requires routing and shared-expert dimensions")
        if block_size != 128:
            raise ValueError("Block-FP8 QMoE currently requires block_size=128")
        self.block_size = block_size
        self.top_k = config.num_experts_per_tok
        e, h, inter = (
            config.num_local_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        )
        self.gate = TopKGate(h, e, self.top_k)
        self.fc1_experts_weights = nn.Parameter(
            [e, 2 * inter, h], dtype=ir.DataType.FLOAT8E4M3FN
        )
        self.fc2_experts_weights = nn.Parameter([e, h, inter], dtype=ir.DataType.FLOAT8E4M3FN)
        self.fc1_scales = nn.Parameter(
            [e, math.ceil(2 * inter / block_size), math.ceil(h / block_size)]
        )
        self.fc2_scales = nn.Parameter(
            [e, math.ceil(h / block_size), math.ceil(inter / block_size)]
        )
        self.fc1_scales._keep_float32 = True
        self.fc2_scales._keep_float32 = True
        shared_config = dataclasses.replace(
            config, intermediate_size=config.shared_expert_intermediate_size
        )
        self.shared_expert = MLP(shared_config, linear_class=Fp8Linear)
        self.shared_expert_gate = Linear(h, 1, bias=False)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        logits, _, _, _ = _realize_gate_and_get_qmoe_routing(op, self.gate, hidden_states)
        router = _flatten_to_2d(op, logits)
        routed = op.QMoE(
            hidden_states,
            router,
            self.fc1_experts_weights,
            self.fc1_scales,
            None,
            self.fc2_experts_weights,
            self.fc2_scales,
            activation_type="swiglu",
            k=self.top_k,
            normalize_routing_weights=1,
            expert_weight_bits=8,
            block_size=self.block_size,
            swiglu_fusion=2,
            quant_type="fp8",
            weights_prepacked=0,
            _domain="com.microsoft",
        )
        shared = self.shared_expert(op, hidden_states)
        gate = op.Sigmoid(self.shared_expert_gate(op, hidden_states))
        return op.Add(routed, op.Mul(shared, gate))
