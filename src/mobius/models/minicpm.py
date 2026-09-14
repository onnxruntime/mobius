# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact text graphs for pinned MiniCPM and MiniCPM3 GGUF checkpoints."""

from __future__ import annotations

import math

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import MLP, RMSNorm, create_attention_bias, initialize_rope
from mobius.components._deepseek_mla import DeepSeekMLA
from mobius.models.base import (
    CausalLMModel,
    embedding_for_config,
    linear_class_for_config,
)


def _scale_like(op: OpBuilder, value: ir.Value, scale: float) -> ir.Value:
    return op.Mul(value, op.CastLike(op.Constant(value_float=scale), value))


class _MiniCPMTextModel(nn.Module):
    """MiniCPM dense decoder with model-owned embedding, residual, and head scales."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        from mobius.components import DecoderLayer

        linear_class = linear_class_for_config(config)
        self._dtype = config.dtype
        self._embedding_scale = float(config.embedding_multiplier)
        self._logit_divisor = float(config.logits_scaling)
        self.embed_tokens = embedding_for_config(config)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    config,
                    residual_multiplier=config.residual_multiplier,
                    linear_class=linear_class,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rotary_emb = initialize_rope(config)
        if rotary_emb is None:
            raise ValueError("MiniCPM requires rotary position embeddings")
        self.rotary_emb = rotary_emb

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value | None,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("MiniCPM requires input_ids or inputs_embeds")
            hidden_states = _scale_like(
                op, self.embed_tokens(op, input_ids), self._embedding_scale
            )
        else:
            hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids if input_ids is not None else inputs_embeds,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        for layer, past_kv in zip(self.layers, past_key_values or [None] * len(self.layers)):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        # MiniCPM scales the normalized hidden state before the bias-free LM head.
        hidden_states = self.norm(op, hidden_states)
        hidden_states = _scale_like(op, hidden_states, 1.0 / self._logit_divisor)
        return hidden_states, present_key_values


class MiniCPMCausalLMModel(CausalLMModel):
    """Dense MiniCPM graph used by exact GGUF import."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self._replace_text_model(_MiniCPMTextModel(config))


class _MiniCPM3DecoderLayer(nn.Module):
    """MiniCPM3 MLA block with expanded K/V cache and scaled residual branches."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        linear_class = linear_class_for_config(config)
        self.self_attn = DeepSeekMLA(config, linear_class=linear_class)
        self.mlp = MLP(config, linear_class=linear_class)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._residual_scale = float(config.residual_multiplier)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        residual = hidden_states
        attention_output, present_key_value = self.self_attn(
            op,
            hidden_states=self.input_layernorm(op, hidden_states),
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(
            residual, _scale_like(op, attention_output, self._residual_scale)
        )

        residual = hidden_states
        mlp_output = self.mlp(op, self.post_attention_layernorm(op, hidden_states))
        hidden_states = op.Add(residual, _scale_like(op, mlp_output, self._residual_scale))
        return hidden_states, present_key_value


class _MiniCPM3TextModel(nn.Module):
    """MiniCPM3 Q/KV-LoRA MLA decoder with llama.cpp's expanded cache geometry."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        if config.export_paged_attention:
            raise ValueError(
                "MiniCPM3 GGUF supports only expanded standard K/V cache, not latent "
                "paged-attention state"
            )
        if config.q_lora_rank is None or config.q_lora_rank <= 0:
            raise ValueError("MiniCPM3 requires a positive q_lora_rank")
        if config.kv_lora_rank is None or config.kv_lora_rank <= 0:
            raise ValueError("MiniCPM3 requires a positive kv_lora_rank")

        self._dtype = config.dtype
        self._embedding_scale = float(config.embedding_multiplier)
        self._logit_divisor = float(config.logits_scaling)
        self.embed_tokens = embedding_for_config(config)
        self.layers = nn.ModuleList(
            [_MiniCPM3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rotary_emb = initialize_rope(config)
        if rotary_emb is None:
            raise ValueError("MiniCPM3 requires rotary position embeddings")
        self.rotary_emb = rotary_emb

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = _scale_like(
            op, self.embed_tokens(op, input_ids), self._embedding_scale
        )
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        for layer, past_kv in zip(self.layers, past_key_values or [None] * len(self.layers)):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        hidden_states = _scale_like(op, hidden_states, 1.0 / self._logit_divisor)
        return hidden_states, present_key_values


class MiniCPM3CausalLMModel(CausalLMModel):
    """MiniCPM3 Q/KV-LoRA MLA graph used by exact GGUF import."""

    def __init__(self, config: ArchitectureConfig):
        if not math.isclose(
            config.residual_multiplier,
            1.4 / math.sqrt(config.num_hidden_layers),
            rel_tol=1e-6,
        ):
            raise ValueError("MiniCPM3 residual scaling does not match the pinned graph")
        super().__init__(config)
        self._replace_text_model(_MiniCPM3TextModel(config))
