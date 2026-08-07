# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LiquidAI LFM2 hybrid short-convolution and GQA causal language model."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Attention,
    Embedding,
    GatedShortConv,
    MLP,
    RMSNorm,
    create_padding_mask,
    initialize_rope,
)
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import onnx_ir as ir


class Lfm2DecoderLayer(nn.Module):
    """LFM2 pre-norm decoder layer with either short convolution or full GQA."""

    def __init__(self, config: ArchitectureConfig, layer_idx: int):
        super().__init__()
        layer_types = config.layer_types or []
        self.layer_type = (
            layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
        )
        if self.layer_type == "conv":
            self.conv = GatedShortConv(
                config.hidden_size,
                config.short_conv_kernel,
                bias=config.short_conv_bias,
            )
        else:
            self.self_attn = Attention(config)

        self.feed_forward = MLP(config)
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
        past_key_value: tuple[ir.Value, ...],
    ) -> tuple[ir.Value, tuple[ir.Value, ...]]:
        residual = hidden_states
        operator_input = self.operator_norm(op, hidden_states)

        if self.layer_type == "conv":
            (conv_state,) = past_key_value
            operator_output, present_state = self.conv(
                op,
                operator_input,
                conv_state,
                attention_mask,
            )
            present_key_value = (present_state,)
        else:
            operator_output, present_key_value = self.self_attn(
                op,
                hidden_states=operator_input,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
            )

        # Both operators are pre-normalized and feed separate residual branches.
        hidden_states = op.Add(residual, operator_output)  # (B, T, H)
        feed_forward = self.feed_forward(op, self.ffn_norm(op, hidden_states))
        hidden_states = op.Add(hidden_states, feed_forward)  # (B, T, H)
        return hidden_states, present_key_value


class Lfm2TextModel(nn.Module):
    """LFM2 decoder backbone with mixed convolution and full-attention layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [Lfm2DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list[tuple[ir.Value, ...]] | None = None,
    ) -> tuple[ir.Value, list[tuple[ir.Value, ...]]]:
        hidden_states = self.embed_tokens(op, input_ids)  # (B, T) -> (B, T, H)
        position_embeddings = self.rotary_emb(op, position_ids)
        # ONNX Attention applies causality internally; this mask carries padding only.
        attention_bias = create_padding_mask(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.embedding_norm(op, hidden_states)  # (B, T, H)
        return hidden_states, present_key_values


class Lfm2CausalLMModel(CausalLMModel):
    """LiquidAI LFM2 causal LM with double-gated short convolutions and QK-norm GQA."""

    default_task: str = "hybrid-text-generation"
    category: str = "Hybrid Convolution+Attention"

    def __init__(self, config: ArchitectureConfig):
        # LFM2 hardcodes per-head Q/K RMSNorm and SiLU-gated feed-forward blocks.
        config = dataclasses.replace(
            config,
            attn_qk_norm=True,
            hidden_act=config.hidden_act or "silu",
        )
        super().__init__(config)
        self.model = Lfm2TextModel(config)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Map upstream LFM2 projection names to shared mobius components."""
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = (
                key.replace(".self_attn.out_proj.", ".self_attn.o_proj.")
                .replace(".self_attn.q_layernorm.", ".self_attn.q_norm.")
                .replace(".self_attn.k_layernorm.", ".self_attn.k_norm.")
                .replace(".feed_forward.w1.", ".feed_forward.gate_proj.")
                .replace(".feed_forward.w3.", ".feed_forward.up_proj.")
                .replace(".feed_forward.w2.", ".feed_forward.down_proj.")
            )
            renamed[new_key] = value
        return super().preprocess_weights(renamed)
