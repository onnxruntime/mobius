# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact llama.cpp SmallThinker GGUF decoder.

The implementation mirrors ``llama_model_smallthinker``: router logits are
computed from the unnormalized layer input, attention and experts have separate
RMSNorm inputs, and each layer independently selects full/sliding attention and
global/local NeoX RoPE.
"""

from __future__ import annotations

import dataclasses
import math

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import MLP, Attention, RMSNorm, create_attention_bias, initialize_rope
from mobius.models.base import CausalLMModel, embedding_for_config, linear_class_for_config


class SmallThinkerGate(nn.Module):
    """SmallThinker's metadata-selected softmax/sigmoid top-k router."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        if config.num_local_experts is None or config.num_experts_per_tok is None:
            raise ValueError("SmallThinker requires routed-expert counts")
        if config.scoring_func not in {"softmax", "sigmoid"}:
            raise ValueError(
                "SmallThinker routing supports only softmax and sigmoid scoring, "
                f"got {config.scoring_func!r}"
            )
        if not config.norm_topk_prob:
            raise ValueError("SmallThinker requires normalized top-k routing weights")
        floor = config.routing_weight_normalization_floor
        if floor is None or not math.isclose(floor, 6.103515625e-5):
            raise ValueError("SmallThinker requires the llama.cpp F16-min routing floor")

        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.scoring_func = config.scoring_func
        self.output_scale = config.routed_scaling_factor
        self.normalization_floor = floor
        self.weight = nn.Parameter([self.num_experts, config.hidden_size])

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        # llama.cpp evaluates the router in float32 before selecting experts.
        logits = op.MatMul(
            op.Cast(hidden_states, to=ir.DataType.FLOAT),
            op.Cast(op.Transpose(self.weight, perm=[1, 0]), to=ir.DataType.FLOAT),
        )
        scores = (
            op.Sigmoid(logits)
            if self.scoring_func == "sigmoid"
            else op.Softmax(logits, axis=-1)
        )
        _, selected_experts = op.TopK(
            scores,
            op.Constant(value_ints=[self.top_k]),
            axis=-1,
            _outputs=2,
        )
        selected_weights = op.GatherElements(scores, selected_experts, axis=-1)
        weight_sum = op.ReduceSum(selected_weights, [-1], keepdims=True)
        denominator = op.Max(weight_sum, float(self.normalization_floor))
        selected_weights = op.Div(selected_weights, denominator)
        if not math.isclose(self.output_scale, 1.0):
            selected_weights = op.Mul(selected_weights, float(self.output_scale))
        return selected_weights, selected_experts


class SmallThinkerMoELayer(nn.Module):
    """ReGLU experts with distinct expert and pre-norm router inputs."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        if config.num_local_experts is None or config.moe_intermediate_size is None:
            raise ValueError("SmallThinker requires expert count and expert width")
        expert_config = dataclasses.replace(
            config,
            hidden_act="relu",
            intermediate_size=config.moe_intermediate_size,
            mlp_bias=False,
        )
        linear_class = linear_class_for_config(config)
        self.gate = SmallThinkerGate(config)
        self.experts = nn.ModuleList(
            [
                MLP(expert_config, linear_class=linear_class)
                for _ in range(config.num_local_experts)
            ]
        )

    def forward(
        self,
        op: OpBuilder,
        expert_hidden_states: ir.Value,
        router_hidden_states: ir.Value,
    ):
        routing_weights, selected_experts = self.gate(op, router_hidden_states)
        result = None
        for expert_index, expert in enumerate(self.experts):
            expert_output = expert(op, expert_hidden_states)
            matches = op.Equal(selected_experts, op.Constant(value_int=expert_index))
            selected = op.Mul(routing_weights, op.CastLike(matches, routing_weights))
            weight = op.CastLike(
                op.ReduceSum(selected, [-1], keepdims=True),
                expert_output,
            )
            contribution = op.Mul(expert_output, weight)
            result = contribution if result is None else op.Add(result, contribution)
        assert result is not None
        return result


class SmallThinkerDecoderLayer(nn.Module):
    """Attention plus routed experts in SmallThinker's exact residual order."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        linear_class = linear_class_for_config(config)
        self.self_attn = Attention(config, linear_class=linear_class)
        self.mlp = SmallThinkerMoELayer(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple | None,
        past_key_value: tuple | None = None,
    ):
        # The router consumes inpL, before either layer norm or residual update.
        router_hidden_states = hidden_states
        attention_input = self.input_layernorm(op, hidden_states)
        attention_output, present_key_value = self.self_attn(
            op,
            hidden_states=attention_input,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        ffn_input = op.Add(hidden_states, attention_output)
        expert_input = self.post_attention_layernorm(op, ffn_input)
        expert_output = self.mlp(op, expert_input, router_hidden_states)
        return op.Add(ffn_input, expert_output), present_key_value


class SmallThinkerTextModel(nn.Module):
    """SmallThinker backbone with per-layer SWA and RoPE selection."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = embedding_for_config(config)
        self.layers = nn.ModuleList(
            [SmallThinkerDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rotary_emb = initialize_rope(config)
        if rotary_emb is None:
            raise ValueError("SmallThinker requires default NeoX RoPE metadata")
        self.rotary_emb = rotary_emb
        rope_theta = config.rope_theta
        if rope_theta is None:
            raise ValueError("SmallThinker requires a RoPE base frequency")
        local_base = config.rope_local_base_freq or rope_theta
        self.local_rotary_emb: nn.Module | None = None
        if not math.isclose(local_base, rope_theta):
            local_rotary_emb = initialize_rope(
                dataclasses.replace(config, rope_theta=local_base)
            )
            if local_rotary_emb is None:
                raise ValueError("SmallThinker local RoPE initialization failed")
            self.local_rotary_emb = local_rotary_emb
        self.layer_types = list(config.layer_types or ())
        self.use_rope_layers = list(config.no_rope_layers or ())
        self.sliding_window = config.sliding_window
        if len(self.layer_types) != config.num_hidden_layers:
            raise ValueError("SmallThinker requires one attention type per layer")
        if len(self.use_rope_layers) != config.num_hidden_layers:
            raise ValueError("SmallThinker requires one RoPE decision per layer")

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        if attention_mask is None:
            raise ValueError("SmallThinker supports only the dynamic attention-mask cache ABI")
        hidden_states = self.embed_tokens(op, input_ids)
        global_position_embeddings = self.rotary_emb(op, position_ids)
        local_position_embeddings = (
            global_position_embeddings
            if self.local_rotary_emb is None
            else self.local_rotary_emb(op, position_ids)
        )
        full_attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )
        sliding_attention_bias = (
            create_attention_bias(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                sliding_window=self.sliding_window,
                dtype=self._dtype,
            )
            if self.sliding_window is not None
            else full_attention_bias
        )

        present_key_values = []
        past_values = past_key_values or [None] * len(self.layers)
        for index, (layer, past_key_value) in enumerate(zip(self.layers, past_values)):
            is_sliding = self.layer_types[index] == "sliding_attention"
            attention_bias = sliding_attention_bias if is_sliding else full_attention_bias
            if not self.use_rope_layers[index]:
                position_embeddings = None
            elif is_sliding:
                position_embeddings = local_position_embeddings
            else:
                position_embeddings = global_position_embeddings
            hidden_states, present_key_value = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
            )
            present_key_values.append(present_key_value)

        return self.norm(op, hidden_states), present_key_values


class SmallThinkerGGUFCausalLMModel(CausalLMModel):
    """Exact float-import model for ``general.architecture=smallthinker`` GGUF."""

    default_task = "smallthinker-gguf-text-generation"

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self._replace_text_model(SmallThinkerTextModel(config))

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return super().preprocess_weights(state_dict)
