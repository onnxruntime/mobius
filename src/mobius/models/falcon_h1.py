# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import FalconH1Config
from mobius.components import (
    Attention,
    Embedding,
    Linear,
    Mamba2Block,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)

class FalconH1MLP(nn.Module):
    """Falcon-H1 SwiGLU feed-forward network with MuP gate and output scaling."""

    def __init__(self, config: FalconH1Config):
        super().__init__()
        self.gate_proj = Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.up_proj = Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.down_proj = Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
        )
        self.gate_multiplier, self.down_multiplier = config.mlp_multipliers

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        gate = self.gate_proj(op, hidden_states)
        if self.gate_multiplier != 1.0:
            gate = op.Mul(gate, self.gate_multiplier)
        gate = op.Mul(gate, op.Sigmoid(gate))
        hidden_states = op.Mul(self.up_proj(op, hidden_states), gate)
        hidden_states = self.down_proj(op, hidden_states)
        if self.down_multiplier != 1.0:
            hidden_states = op.Mul(hidden_states, self.down_multiplier)
        return hidden_states


class FalconH1DecoderLayer(nn.Module):
    """One parallel Attention + Mamba2 Falcon-H1 decoder layer."""

    def __init__(self, config: FalconH1Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mamba = Mamba2Block(
            config.hidden_size,
            d_inner=config.mamba_d_ssm,
            d_head=config.mamba_d_head,
            d_state=config.mamba_d_state,
            num_heads=config.mamba_n_heads,
            n_groups=config.mamba_n_groups,
            chunk_size=config.mamba_chunk_size,
            conv_kernel=config.mamba_d_conv,
            eps=config.rms_norm_eps,
            conv_bias=config.mamba_conv_bias,
            norm_group_size=config.mamba_d_ssm // config.mamba_n_groups,
            time_step_min=config.time_step_limit[0],
            time_step_max=(
                config.time_step_limit[1]
                if config.time_step_limit[1] != float("inf")
                else None
            ),
            in_proj_bias=config.mamba_proj_bias,
            out_proj_bias=config.projectors_bias,
            use_norm=config.mamba_rms_norm,
            norm_before_gate=config.mamba_norm_before_gate,
            input_multiplier=config.ssm_in_multiplier,
            projection_multipliers=config.ssm_multipliers,
        )
        self.self_attn = Attention(config)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = FalconH1MLP(config)
        self.attention_in_multiplier = config.attention_in_multiplier
        self.attention_out_multiplier = config.attention_out_multiplier
        self.ssm_out_multiplier = config.ssm_out_multiplier

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple[ir.Value, ...],
        attention_bias: ir.Value,
        mamba_attention_mask: ir.Value,
        past_key: ir.Value,
        past_value: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value, ir.Value]:
        residual = hidden_states
        normalized = self.input_layernorm(op, hidden_states)

        # Both branches consume the same normalized residual stream.
        mamba_input = op.Mul(
            normalized,
            op.CastLike(mamba_attention_mask, normalized),
        )
        mamba_output, present_conv, present_ssm = self.mamba(
            op,
            mamba_input,
            conv_state,
            ssm_state,
            mamba_attention_mask,
        )
        if self.ssm_out_multiplier != 1.0:
            mamba_output = op.Mul(mamba_output, self.ssm_out_multiplier)

        attention_input = normalized
        if self.attention_in_multiplier != 1.0:
            attention_input = op.Mul(attention_input, self.attention_in_multiplier)
        attention_output, present_key_value = self.self_attn(
            op,
            hidden_states=attention_input,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=(past_key, past_value),
        )
        present_key, present_value = present_key_value
        if self.attention_out_multiplier != 1.0:
            attention_output = op.Mul(
                attention_output,
                self.attention_out_multiplier,
            )

        hidden_states = op.Add(residual, op.Add(mamba_output, attention_output))
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(op, hidden_states)
        hidden_states = op.Add(residual, self.feed_forward(op, hidden_states))
        return hidden_states, present_key, present_value, present_conv, present_ssm


class FalconH1Model(nn.Module):
    """Falcon-H1 decoder matching HuggingFace ``FalconH1Model``."""

    def __init__(self, config: FalconH1Config):
        super().__init__()
        self.config = config
        self._dtype = config.dtype
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [FalconH1DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        self.embedding_multiplier = config.embedding_multiplier

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        position_ids: ir.Value,
        attention_mask: ir.Value,
        mamba_attention_mask: ir.Value,
        past_key_values: tuple[ir.Value, ...],
    ) -> tuple[ir.Value, tuple[ir.Value, ...]]:
        hidden_states = self.embed_tokens(op, input_ids)
        if self.embedding_multiplier != 1.0:
            hidden_states = op.Mul(hidden_states, self.embedding_multiplier)

        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )
        present_key_values: list[ir.Value] = []
        for layer_idx, layer in enumerate(self.layers):
            offset = layer_idx * 4
            (
                hidden_states,
                present_key,
                present_value,
                present_conv,
                present_ssm,
            ) = layer(
                op,
                hidden_states,
                position_embeddings,
                attention_bias,
                mamba_attention_mask,
                past_key_values[offset],
                past_key_values[offset + 1],
                past_key_values[offset + 2],
                past_key_values[offset + 3],
            )
            present_key_values.extend(
                [present_key, present_value, present_conv, present_ssm]
            )
        return self.final_layernorm(op, hidden_states), tuple(present_key_values)


class FalconH1ForCausalLM(nn.Module):
    """Dedicated Falcon-H1 causal LM with four recurrent states per decoder layer."""

    default_task: str = "falcon-h1-text-generation"
    category: str = "Hybrid SSM+Attention"
    config_class: type = FalconH1Config

    def __init__(self, config: FalconH1Config):
        super().__init__()
        self.config = config
        self.model = FalconH1Model(config)
        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.lm_head_multiplier = config.lm_head_multiplier

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        position_ids: ir.Value,
        attention_mask: ir.Value,
        mamba_attention_mask: ir.Value,
        past_key_values: tuple[ir.Value, ...],
    ) -> tuple[ir.Value, tuple[ir.Value, ...]]:
        hidden_states, present_key_values = self.model(
            op,
            input_ids,
            position_ids,
            attention_mask,
            mamba_attention_mask,
            past_key_values,
        )
        logits = self.lm_head(op, hidden_states)
        if self.lm_head_multiplier != 1.0:
            logits = op.Mul(logits, self.lm_head_multiplier)
        return logits, present_key_values

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.config.tie_word_embeddings:
            if "model.embed_tokens.weight" not in state_dict:
                state_dict["model.embed_tokens.weight"] = state_dict["lm_head.weight"]
            state_dict.pop("lm_head.weight", None)
        return state_dict
