# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Talkie causal LM with scaleless normalization and embedding skip connections.

Replicates the pinned llama.cpp Talkie graph: every RMSNorm is weight-free,
Q/K normalization follows NeoX RoPE, the learned query gain is per head, and
each block adds a scaled copy of the normalized token embedding.
"""

from __future__ import annotations

import math

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    MLP,
    Embedding,
    Linear,
    StaticCacheState,
    apply_rotary_pos_emb,
    create_padding_mask,
    initialize_rope,
)
from mobius.components._attention import _apply_attention
from mobius.models.base import CausalLMModel, linear_class_for_config


class TalkieRMSNorm(nn.Module):
    """RMSNorm with a constant all-ones scale and no checkpoint parameter."""

    def __init__(self, size: int, eps: float):
        super().__init__()
        self._scale = [1.0] * size
        self._eps = eps

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        scale = op.CastLike(op.Constant(value_floats=self._scale), hidden_states)
        return op.RMSNormalization(
            hidden_states,
            scale,
            axis=-1,
            epsilon=self._eps,
            stash_type=1,
        )


class TalkieAttention(nn.Module):
    """Causal attention with flipped-sine RoPE followed by scaleless Q/K norm."""

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__()
        if linear_class is None:
            linear_class = Linear
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self._rope_interleave = config.rope_interleave
        self.q_proj = linear_class(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = linear_class(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = linear_class(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = linear_class(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = TalkieRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = TalkieRMSNorm(self.head_dim, config.rms_norm_eps)
        self.q_gain = nn.Parameter([self.num_attention_heads, 1])

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

        # Talkie's reference rotates first, then normalizes each head.
        query_states = apply_rotary_pos_emb(
            op,
            query_states,
            position_embeddings,
            num_heads=self.num_attention_heads,
            interleaved=self._rope_interleave,
        )
        key_states = apply_rotary_pos_emb(
            op,
            key_states,
            position_embeddings,
            num_heads=self.num_key_value_heads,
            interleaved=self._rope_interleave,
        )
        query_states = op.Reshape(
            query_states, [0, 0, self.num_attention_heads, self.head_dim]
        )
        key_states = op.Reshape(key_states, [0, 0, self.num_key_value_heads, self.head_dim])
        query_states = self.q_norm(op, query_states)
        key_states = self.k_norm(op, key_states)
        # The gain is indexed by query head and broadcasts over batch/token/head_dim.
        q_gain = op.Reshape(self.q_gain, [1, 1, self.num_attention_heads, 1])
        query_states = op.Mul(query_states, q_gain)
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
            scale=self.scaling,
            static_cache=static_cache,
        )
        return self.o_proj(op, attn_output), (present_key, present_value)


class TalkieDecoderLayer(nn.Module):
    """Talkie block with scaleless pre-norms and normalized-embedding skip."""

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__()
        self.self_attn = TalkieAttention(config, linear_class)
        self.mlp = MLP(config, linear_class=linear_class)
        self.input_layernorm = TalkieRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = TalkieRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.embed_skip = nn.Parameter([1])

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        embedding_skip: ir.Value,
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
        hidden_states = op.Add(residual, hidden_states)
        hidden_states = op.Add(hidden_states, op.Mul(embedding_skip, self.embed_skip))
        return hidden_states, present_key_value


class TalkieTextModel(nn.Module):
    """Talkie decoder backbone with dynamic/static causal KV cache support."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        linear_class = linear_class_for_config(config)
        self.layers = nn.ModuleList(
            [TalkieDecoderLayer(config, linear_class) for _ in range(config.num_hidden_layers)]
        )
        self.norm = TalkieRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        hidden_states = self.norm(op, hidden_states)
        embedding_skip = hidden_states
        cos, sin = self.rotary_emb(op, position_ids)
        # Talkie stores ordinary NeoX frequencies but applies the inverse rotation.
        position_embeddings = (cos, op.Neg(sin))
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
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states,
                embedding_skip,
                attention_bias,
                position_embeddings,
                past_kv,
            )
            present_key_values.append(present_kv)
        return self.norm(op, hidden_states), present_key_values


class TalkieForCausalLM(CausalLMModel):
    """Exact float Talkie causal LM with post-RoPE Q/K norm and embedding skips."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self._replace_text_model(TalkieTextModel(config))

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
        )
        logits = self.lm_head(op, hidden_states)
        if not math.isclose(self.config.logit_scale, 1.0):
            logits = op.Mul(logits, self.config.logit_scale)
        return logits, present_key_values
