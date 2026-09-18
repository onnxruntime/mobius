# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stateless embedding models for the canonical GGUF embedding architectures."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Embedding,
    Linear,
    RMSNorm,
    apply_rotary_pos_emb,
    create_padding_mask,
    get_activation,
    initialize_rope,
)

if TYPE_CHECKING:
    import onnx_ir as ir


def _positions(op: OpBuilder, input_ids: ir.Value, rotary_emb):
    seq_len = op.Squeeze(op.Shape(input_ids, start=1, end=2))
    position_ids = op.Range(op.Constant(value_int=0), seq_len, op.Constant(value_int=1))
    return rotary_emb(op, op.Unsqueeze(position_ids, [0]))


def _symmetric_attention_mask(
    op: OpBuilder,
    input_ids: ir.Value,
    attention_mask: ir.Value,
    window: int | None,
):
    padding = create_padding_mask(op, input_ids, attention_mask)
    if window is None:
        return padding
    seq_len = op.Squeeze(op.Shape(input_ids, start=1, end=2))
    positions = op.Range(op.Constant(value_int=0), seq_len, op.Constant(value_int=1))
    distance = op.Abs(op.Sub(op.Unsqueeze(positions, [1]), op.Unsqueeze(positions, [0])))
    local = op.LessOrEqual(distance, op.Constant(value_int=window // 2))
    return op.And(padding, op.Unsqueeze(local, [0, 1]))


def _pool(op: OpBuilder, hidden_states, attention_mask, pooling_type: int):
    if pooling_type == 0:
        return hidden_states
    if pooling_type == 2:
        index = op.ArgMax(attention_mask, axis=1, keepdims=0, select_last_index=0)
    elif pooling_type == 3:
        index = op.ArgMax(attention_mask, axis=1, keepdims=0, select_last_index=1)
    else:
        mask = op.Unsqueeze(op.CastLike(attention_mask, hidden_states), [2])
        total = op.ReduceSum(op.Mul(hidden_states, mask), axes=[1], keepdims=0)
        count = op.ReduceSum(mask, axes=[1], keepdims=0)
        return op.Div(total, op.Max(count, op.CastLike(1.0, count)))
    seq_len = op.Squeeze(op.Shape(hidden_states, start=1, end=2))
    one_hot = op.OneHot(index, seq_len, [0, 1], axis=-1)
    selector = op.Unsqueeze(op.CastLike(one_hot, hidden_states), [2])
    return op.ReduceSum(op.Mul(hidden_states, selector), axes=[1], keepdims=0)


class _EmbeddingAttention(nn.Module):
    """Split-QKV GQA with optional per-head Q/K RMS normalization."""

    def __init__(self, config: ArchitectureConfig, *, qk_norm: bool):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self._qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.scale = float(
            config.attention_multiplier
            if config.attention_multiplier is not None
            else self.head_dim**-0.5
        )

    def forward(self, op, hidden_states, attention_mask, position_embeddings):
        query = self.q_proj(op, hidden_states)
        key = self.k_proj(op, hidden_states)
        value = self.v_proj(op, hidden_states)
        if self._qk_norm:
            query = op.Reshape(query, [0, 0, self.num_heads, self.head_dim])
            key = op.Reshape(key, [0, 0, self.num_kv_heads, self.head_dim])
            query = self.q_norm(op, query)
            key = self.k_norm(op, key)
            query = op.Reshape(query, [0, 0, -1])
            key = op.Reshape(key, [0, 0, -1])
        query = apply_rotary_pos_emb(op, query, position_embeddings, num_heads=self.num_heads)
        key = apply_rotary_pos_emb(op, key, position_embeddings, num_heads=self.num_kv_heads)
        attended = op.Attention(
            query,
            key,
            value,
            attention_mask,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_kv_heads,
            scale=self.scale,
        )
        return self.o_proj(op, attended)


class _EmbeddingMLP(nn.Module):
    def __init__(self, config: ArchitectureConfig, activation: str):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = get_activation(activation)

    def forward(self, op, hidden_states):
        return self.down_proj(
            op,
            op.Mul(
                self.activation(op, self.gate_proj(op, hidden_states)),
                self.up_proj(op, hidden_states),
            ),
        )


class _LlamaEmbeddingLayer(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = _EmbeddingAttention(config, qk_norm=False)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = _EmbeddingMLP(config, "silu")

    def forward(self, op, hidden_states, attention_mask, position_embeddings):
        residual = hidden_states
        hidden_states = self.self_attn(
            op, self.input_layernorm(op, hidden_states), attention_mask, position_embeddings
        )
        hidden_states = op.Add(residual, hidden_states)
        return op.Add(
            hidden_states, self.mlp(op, self.post_attention_layernorm(op, hidden_states))
        )


class LlamaEmbedGGUFModel(nn.Module):
    """Exact dense, unbiased, default-RoPE subset of llama.cpp ``llama-embed``."""

    default_task = "gguf-embedding-feature-extraction"
    category = "encoder"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._pooling_type = config.pooling_type
        self.token_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [_LlamaEmbeddingLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.output_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(self, op, input_ids, attention_mask):
        hidden_states = self.token_embeddings(op, input_ids)
        positions = _positions(op, input_ids, self.rotary_emb)
        mask = _symmetric_attention_mask(op, input_ids, attention_mask, None)
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, mask, positions)
        return _pool(
            op, self.output_norm(op, hidden_states), attention_mask, self._pooling_type
        )

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]):
        return state_dict


class _GemmaEmbeddingLayer(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = _EmbeddingAttention(config, qk_norm=True)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = _EmbeddingMLP(config, "gelu_pytorch_tanh")

    def forward(self, op, hidden_states, attention_mask, position_embeddings):
        residual = hidden_states
        attention = self.self_attn(
            op, self.input_layernorm(op, hidden_states), attention_mask, position_embeddings
        )
        hidden_states = op.Add(residual, self.post_attention_layernorm(op, attention))
        residual = hidden_states
        feed_forward = self.mlp(op, self.pre_feedforward_layernorm(op, hidden_states))
        return op.Add(residual, self.post_feedforward_layernorm(op, feed_forward))


class GemmaEmbeddingGGUFModel(nn.Module):
    """EmbeddingGemma's bidirectional, alternating symmetric-window GGUF graph."""

    default_task = "gguf-embedding-feature-extraction"
    category = "encoder"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._pooling_type = config.pooling_type
        self._sliding_window = config.sliding_window
        self._layer_types = config.layer_types or []
        self._embed_scale = math.sqrt(config.hidden_size)
        self.token_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [_GemmaEmbeddingLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.output_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.global_rotary_emb = initialize_rope(config)
        local_config = dataclasses.replace(config, rope_theta=config.rope_local_base_freq)
        self.local_rotary_emb = initialize_rope(local_config)
        if config.embedding_dense_2_out is not None:
            self.dense_2 = Linear(config.hidden_size, config.embedding_dense_2_out, bias=False)
        if config.embedding_dense_3_in is not None:
            self.dense_3 = Linear(config.embedding_dense_3_in, config.hidden_size, bias=False)

    def forward(self, op, input_ids, attention_mask):
        hidden_states = op.Mul(self.token_embeddings(op, input_ids), self._embed_scale)
        global_positions = _positions(op, input_ids, self.global_rotary_emb)
        local_positions = _positions(op, input_ids, self.local_rotary_emb)
        full_mask = _symmetric_attention_mask(op, input_ids, attention_mask, None)
        local_mask = _symmetric_attention_mask(
            op, input_ids, attention_mask, self._sliding_window
        )
        for layer_index, layer in enumerate(self.layers):
            local = self._layer_types[layer_index] == "sliding_attention"
            hidden_states = layer(
                op,
                hidden_states,
                local_mask if local else full_mask,
                local_positions if local else global_positions,
            )
        hidden_states = _pool(
            op,
            self.output_norm(op, hidden_states),
            attention_mask,
            self._pooling_type,
        )
        if hasattr(self, "dense_2"):
            hidden_states = self.dense_2(op, hidden_states)
        if hasattr(self, "dense_3"):
            hidden_states = self.dense_3(op, hidden_states)
        return hidden_states

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]):
        return state_dict
