# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact explicit-float decoder for ``general.architecture=minimax-m2`` GGUF."""

from __future__ import annotations

import math

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import RMSNorm
from mobius.models.base import CausalLMModel, TextModel
from mobius.models.moe import MoEDecoderLayer

_F16_NORMAL_MIN = 6.103515625e-5


class MiniMaxM2Gate(nn.Module):
    """F32 sigmoid router with selection-only correction bias."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        if config.num_local_experts is None or config.num_experts_per_tok is None:
            raise ValueError("MiniMax-M2 requires expert count and top-k")
        if config.scoring_func != "sigmoid":
            raise ValueError("MiniMax-M2 requires sigmoid expert routing")
        if not config.norm_topk_prob:
            raise ValueError("MiniMax-M2 requires normalized selected-expert weights")
        floor = config.routing_weight_normalization_floor
        if floor is None or not math.isclose(floor, _F16_NORMAL_MIN):
            raise ValueError("MiniMax-M2 requires the llama.cpp F16-normal routing floor")

        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.normalization_floor = floor
        self.output_scale = config.routed_scaling_factor
        self.weight = nn.Parameter(
            [self.num_experts, config.hidden_size],
            dtype=ir.DataType.FLOAT,
        )
        self.e_score_correction_bias = nn.Parameter(
            [self.num_experts],
            dtype=ir.DataType.FLOAT,
        )
        self.weight._keep_float32 = True  # type: ignore[attr-defined]
        self.e_score_correction_bias._keep_float32 = True  # type: ignore[attr-defined]

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        router_input = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        logits = op.MatMul(router_input, op.Transpose(self.weight, perm=[1, 0]))
        probabilities = op.Sigmoid(logits)
        selection_scores = op.Add(probabilities, self.e_score_correction_bias)
        _, selected_experts = op.TopK(
            selection_scores,
            op.Constant(value_ints=[self.top_k]),
            axis=-1,
            largest=1,
            sorted=0,
            _outputs=2,
        )
        routing_weights = op.GatherElements(probabilities, selected_experts, axis=-1)
        denominator = op.ReduceSum(routing_weights, [-1], keepdims=True)
        denominator = op.Max(denominator, float(self.normalization_floor))
        routing_weights = op.Div(routing_weights, denominator)
        if not math.isclose(self.output_scale, 1.0):
            routing_weights = op.Mul(routing_weights, float(self.output_scale))
        return op.CastLike(routing_weights, hidden_states), selected_experts


class MiniMaxM2DecoderLayer(MoEDecoderLayer):
    """Standard pre-norm attention block with MiniMax-M2 routed experts."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config, gate=MiniMaxM2Gate(config), norm_class=RMSNorm)


class MiniMaxM2TextModel(TextModel):
    """MiniMax-M2 backbone with full-vector Q/K norms and partial NeoX RoPE."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MiniMaxM2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )


class MiniMaxM2GGUFCausalLMModel(CausalLMModel):
    """MiniMax-M2 GGUF decoder with full attention and all-layer routed MoE."""

    default_task: str = "text-generation"
    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self._replace_text_model(MiniMaxM2TextModel(config))

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return super().preprocess_weights(state_dict)
