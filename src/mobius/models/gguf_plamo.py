# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact source-level PLaMo decoder used by validated llama.cpp GGUF imports."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    MLP,
    Embedding,
    Linear,
    RMSNorm,
    apply_rotary_pos_emb,
    create_attention_bias,
    initialize_rope,
)


class _PlamoAttention(nn.Module):
    """PLaMo attention with cyclic KV expansion before RoPE and caching."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        if config.num_key_value_heads <= 0:
            raise ValueError("PLaMo requires a positive num_key_value_heads")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("PLaMo requires query heads divisible by KV heads")
        if config.hidden_size != config.num_attention_heads * config.head_dim:
            raise ValueError("PLaMo requires hidden_size == num_attention_heads * head_dim")

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.kv_repeat = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = Linear(config.hidden_size, self.q_dim, bias=False)
        self.k_proj = Linear(config.hidden_size, self.kv_dim, bias=False)
        self.v_proj = Linear(config.hidden_size, self.kv_dim, bias=False)
        self.o_proj = Linear(self.q_dim, config.hidden_size, bias=False)

    def _expand_kv(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        # PLaMo repeats the complete KV-head sequence:
        # [k0, ..., k4] -> [k0, ..., k4, k0, ..., k4, ...].
        value = op.Reshape(value, [0, 0, self.num_kv_heads, self.head_dim])
        value = op.Tile(value, [1, 1, self.kv_repeat, 1])
        return op.Reshape(value, [0, 0, self.q_dim])

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple[ir.Value, ...],
        past_key_value: tuple[ir.Value, ir.Value] | None,
    ) -> tuple[ir.Value, tuple[ir.Value, ir.Value]]:
        query = self.q_proj(op, hidden_states)
        key = self._expand_kv(op, self.k_proj(op, hidden_states))
        value = self._expand_kv(op, self.v_proj(op, hidden_states))

        # The expanded 40-head K is rotated and cached exactly as in the
        # reference implementation; this is not a conventional 5-head GQA cache.
        query = apply_rotary_pos_emb(
            op,
            query,
            position_embeddings,
            num_heads=self.num_heads,
            rotary_embedding_dim=0,
            interleaved=False,
        )
        key = apply_rotary_pos_emb(
            op,
            key,
            position_embeddings,
            num_heads=self.num_heads,
            rotary_embedding_dim=0,
            interleaved=False,
        )
        output, present_key, present_value = op.Attention(
            query,
            key,
            value,
            attention_bias,
            past_key_value[0] if past_key_value is not None else None,
            past_key_value[1] if past_key_value is not None else None,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            scale=self.scale,
            is_causal=0,
            _outputs=3,
        )
        return self.o_proj(op, output), (present_key, present_value)


class _PlamoDecoderLayer(nn.Module):
    """Shared-pre-norm parallel attention and SwiGLU residual block."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = _PlamoAttention(config)
        self.mlp = MLP(config)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple[ir.Value, ...],
        past_key_value: tuple[ir.Value, ir.Value] | None,
    ) -> tuple[ir.Value, tuple[ir.Value, ir.Value]]:
        residual = hidden_states
        normalized = self.input_layernorm(op, hidden_states)
        attention_output, present_key_value = self.self_attn(
            op,
            normalized,
            attention_bias,
            position_embeddings,
            past_key_value,
        )
        # Both parallel branches consume the same RMS-normalized activation.
        hidden_states = op.Add(
            residual,
            op.Add(attention_output, self.mlp(op, normalized)),
        )
        return hidden_states, present_key_value


class _PlamoTextModel(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [_PlamoDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list[tuple[ir.Value, ir.Value]] | None,
    ) -> tuple[ir.Value, list[tuple[ir.Value, ir.Value]]]:
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )
        past_key_values = past_key_values or [None] * len(self.layers)
        present_key_values = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present_key_value = layer(
                op,
                hidden_states,
                attention_bias,
                position_embeddings,
                past_key_value,
            )
            present_key_values.append(present_key_value)
        return self.norm(op, hidden_states), present_key_values


class PlamoGGUFCausalLMModel(nn.Module):
    """PLaMo causal LM for the exact validated ``plamo`` GGUF contract."""

    default_task = "plamo-text-generation"
    category = "Text Generation"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        architecture = getattr(config, "_gguf_arch", None)
        if architecture not in {None, "plamo"}:
            raise ValueError("PlamoGGUFCausalLMModel requires a PLaMo GGUF config")
        if config.tie_word_embeddings:
            raise ValueError("PLaMo requires an untied output projection")
        self.config = config
        self.model = _PlamoTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def kv_cache_specs(self) -> list[tuple[int, int]]:
        return [
            (self.config.num_attention_heads, self.config.head_dim)
            for _ in range(self.config.num_hidden_layers)
        ]

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values=None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
        )
        return self.lm_head(op, hidden_states), present_key_values
