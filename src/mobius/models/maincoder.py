# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Maincoder decoder matching llama.cpp's post-RoPE Q/K normalization order."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    MLP,
    Embedding,
    Linear,
    RMSNorm,
    StaticCacheState,
    apply_rotary_pos_emb,
    create_padding_mask,
    initialize_rope,
)
from mobius.components._attention import _apply_attention
from mobius.models.base import CausalLMModel


class MaincoderAttention(nn.Module):
    """Maincoder GQA with learned per-head Q/K RMSNorm applied after RoPE."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        kv_hidden_size = self.num_key_value_heads * self.head_dim
        self.q_proj = Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = Linear(config.hidden_size, kv_hidden_size, bias=False)
        self.v_proj = Linear(config.hidden_size, kv_hidden_size, bias=False)
        self.o_proj = Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
        static_cache: StaticCacheState | None = None,
    ):
        query_states = self.q_proj(op, hidden_states)
        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)

        # Maincoder rotates adjacent real/imaginary pairs before learned per-head
        # normalization. Keeping these operations external makes the order explicit.
        query_states = apply_rotary_pos_emb(
            op,
            query_states,
            position_embeddings,
            num_heads=self.num_attention_heads,
            interleaved=True,
        )
        key_states = apply_rotary_pos_emb(
            op,
            key_states,
            position_embeddings,
            num_heads=self.num_key_value_heads,
            interleaved=True,
        )
        query_states = op.Reshape(
            query_states, [0, 0, self.num_attention_heads, self.head_dim]
        )
        key_states = op.Reshape(key_states, [0, 0, self.num_key_value_heads, self.head_dim])
        query_states = self.q_norm(op, query_states)
        key_states = self.k_norm(op, key_states)
        query_states = op.Reshape(query_states, [0, 0, -1])
        key_states = op.Reshape(key_states, [0, 0, -1])

        attn_output, present_key, present_value = _apply_attention(
            op,
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key_value[0] if past_key_value is not None else None,
            past_key_value[1] if past_key_value is not None else None,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale=self.head_dim**-0.5,
            static_cache=static_cache,
        )
        return self.o_proj(op, attn_output), (present_key, present_value)


class MaincoderDecoderLayer(nn.Module):
    """Sequential pre-norm Maincoder attention and SwiGLU residual block."""

    _supports_static_cache = True

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.self_attn = MaincoderAttention(config)
        self.mlp = MLP(config, linear_class=Linear)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value,
    ):
        if isinstance(past_key_value, StaticCacheState):
            static_cache = past_key_value
            past_key_value = None
        else:
            static_cache = None

        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present_key_value = self.self_attn(
            op,
            hidden_states,
            attention_bias,
            position_embeddings,
            past_key_value,
            static_cache,
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        return op.Add(residual, hidden_states), present_key_value


class MaincoderTextModel(nn.Module):
    """Maincoder embedding, post-RoPE-QK-norm decoder, and final RMSNorm."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MaincoderDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rotary_emb = initialize_rope(config)
        if rotary_emb is None:
            raise ValueError("Maincoder requires rotary position embeddings")
        self.rotary_emb = rotary_emb

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = (
            create_padding_mask(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if attention_mask is not None
            else None
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_key_value in zip(self.layers, past_kvs):
            hidden_states, present_key_value = layer(
                op,
                hidden_states,
                attention_bias,
                position_embeddings,
                past_key_value,
            )
            present_key_values.append(present_key_value)
        return self.norm(op, hidden_states), present_key_values


class MaincoderCausalLMModel(CausalLMModel):
    """Exact tied-output Maincoder causal LM used by the GGUF importer."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self._replace_text_model(MaincoderTextModel(config))

    def kv_cache_specs(self) -> list[tuple[int, int]]:
        return [
            (self.config.num_key_value_heads, self.config.head_dim)
            for _ in range(self.config.num_hidden_layers)
        ]
