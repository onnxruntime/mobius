# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact PLM decoder matching ``PLMForCausalLM`` and llama.cpp's PLM graph."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    FCMLP,
    DeepSeekMLA,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.models.base import (
    CausalLMModel,
    embedding_for_config,
    linear_class_for_config,
)

if TYPE_CHECKING:
    import onnx_ir as ir


def _validate_plm_config(config: ArchitectureConfig) -> None:
    """Reject geometries that cannot represent the pinned PLM architecture."""
    nope_dim = config.qk_nope_head_dim
    rope_dim = config.qk_rope_head_dim
    value_dim = config.v_head_dim
    if not nope_dim or not rope_dim or not value_dim or not config.kv_lora_rank:
        raise ValueError(
            "PLM requires positive qk_nope_head_dim, qk_rope_head_dim, "
            "v_head_dim, and kv_lora_rank"
        )
    if rope_dim % 2:
        raise ValueError("PLM qk_rope_head_dim must be even")
    if config.num_key_value_heads != config.num_attention_heads:
        raise ValueError("PLM caches expanded K/V for every attention head")
    if config.q_lora_rank is not None:
        raise ValueError("PLM uses a direct q_proj and does not support q_lora_rank")
    if config.hidden_act != "relu2":
        raise ValueError("PLM requires hidden_act='relu2'")
    if config.attn_qkv_bias or config.mlp_bias:
        raise ValueError("PLM does not support attention or MLP projection biases")
    if config.export_paged_attention:
        raise ValueError("PLM requires expanded K/V caches, not latent paged attention")


class PLMAttention(DeepSeekMLA):
    """PLM attention with a normalized latent projection and expanded K/V cache.

    The inherited primitive implements the exact PLM data flow: direct Q projection,
    per-head ``[NoPE, RoPE]`` query channels, normalized KV-A latent, fused per-head
    ``[K-NoPE, V]`` KV-B expansion, and a shared RoPE key replicated to every head.
    Unlike latent-cache MLA paths, its cache stores the resulting full-width K/V.
    """

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        _validate_plm_config(config)
        super().__init__(config, linear_class=linear_class)


class PLMDecoderLayer(nn.Module):
    """Pre-norm PLM layer with exact attention and non-gated ReLU-squared FFN."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        linear_class = linear_class_for_config(config)
        self.self_attn = PLMAttention(config, linear_class=linear_class)
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation="relu2",
            bias=False,
            linear_class=linear_class,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        return op.Add(residual, hidden_states), present_kv


class PLMTextModel(nn.Module):
    """PLM embedding, exact expanded-cache decoder layers, and final RMSNorm."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        _validate_plm_config(config)
        self.embed_tokens = embedding_for_config(config)
        self.layers = nn.ModuleList(
            [PLMDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._dtype = config.dtype

        # PLM rotates only the trailing RoPE slice, whose raw channels use
        # adjacent real/imaginary pairs in both the HF and llama.cpp implementations.
        rope_config = dataclasses.replace(
            config,
            head_dim=config.qk_rope_head_dim,
            partial_rotary_factor=1.0,
            rope_interleave=True,
        )
        self.rotary_emb = initialize_rope(rope_config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.embed_tokens(op, input_ids)
        )
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=hidden_states if input_ids is None else input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        return self.norm(op, hidden_states), present_key_values


class PLMCausalLMModel(CausalLMModel):
    """PLM causal LM with one embedding-owned, tied output allocation."""

    def __init__(self, config: ArchitectureConfig):
        _validate_plm_config(config)
        if not config.tie_word_embeddings:
            raise ValueError("PLM requires tied token embedding and output weights")
        super().__init__(config)
        self._replace_text_model(PLMTextModel(config))
