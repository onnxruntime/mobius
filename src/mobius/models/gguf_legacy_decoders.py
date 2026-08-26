# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact narrowed graphs for conventional llama.cpp GGUF decoder families.

These modules implement the executable subsets selected by GGUF validation for
GPT-NeoX, JAIS, MPT, Refact, dense ERNIE 4.5, and OpenELM. They deliberately
share no HuggingFace registry aliases: the GGUF loader contracts, including
ALiBi scaling and per-layer OpenELM geometry, are architecture-owned.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import TYPE_CHECKING

import numpy as np
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import split_fused_qkv
from mobius.components import (
    FCMLP,
    MLP,
    Attention,
    Embedding,
    LayerNorm,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)

if TYPE_CHECKING:
    import onnx_ir as ir


def _alibi_slopes(num_heads: int, max_bias: float) -> list[float]:
    """Return llama.cpp's power-of-two ALiBi slope schedule."""
    closest_power_of_2 = 1 << math.floor(math.log2(num_heads))
    first = 2.0 ** (-max_bias / closest_power_of_2)
    second = 2.0 ** (-(max_bias / 2.0) / closest_power_of_2)
    return [
        first ** (head + 1)
        if head < closest_power_of_2
        else second ** (2 * (head - closest_power_of_2) + 1)
        for head in range(num_heads)
    ]


def _create_alibi_bias(
    op: OpBuilder,
    input_ids: ir.Value,
    attention_mask: ir.Value,
    *,
    num_heads: int,
    max_bias: float,
    dtype,
):
    """Combine causal/padding masking with llama.cpp's signed ALiBi distance."""
    causal_bias = create_attention_bias(
        op,
        input_ids=input_ids,
        attention_mask=attention_mask,
        dtype=dtype,
    )
    positions = op.CumSum(attention_mask, 1)
    query_length = op.Shape(input_ids, start=1, end=2)
    total_length = op.Shape(attention_mask, start=1, end=2)
    query_start = op.Sub(total_length, query_length)
    query_positions = op.Slice(positions, query_start, total_length, [1])
    distance = op.Sub(
        op.Unsqueeze(op.Cast(query_positions, to=1), [1, 3]),
        op.Unsqueeze(op.Cast(positions, to=1), [1, 2]),
    )
    slopes = op.Constant(
        value_floats=np.asarray(_alibi_slopes(num_heads, max_bias), dtype=np.float32).tolist()
    )
    alibi = op.Mul(op.Unsqueeze(slopes, [0, 2, 3]), op.Neg(op.Abs(distance)))
    return op.Add(causal_bias, op.Cast(alibi, to=dtype))


class _LegacyGGUFDecoderLayer(nn.Module):
    """Pre-norm decoder layer parameterized only by validated GGUF semantics."""

    def __init__(
        self,
        config: ArchitectureConfig,
        *,
        layer_norm: bool,
        gated_mlp: bool,
        parallel_residual: bool,
    ):
        super().__init__()
        norm_class = LayerNorm if layer_norm else RMSNorm
        self.input_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config, scale=config.attention_scale)
        self.mlp = (
            MLP(config)
            if gated_mlp
            else FCMLP(
                config.hidden_size,
                config.intermediate_size,
                activation=config.hidden_act or "gelu",
                bias=config.mlp_bias,
            )
        )
        self._parallel_residual = parallel_residual

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings,
        past_key_value=None,
    ):
        residual = hidden_states
        attention_input = self.input_layernorm(op, hidden_states)
        attention_output, present_key_value = self.self_attn(
            op,
            attention_input,
            attention_bias,
            position_embeddings,
            past_key_value,
        )
        if self._parallel_residual:
            feed_forward_input = self.post_attention_layernorm(op, residual)
            hidden_states = op.Add(
                residual,
                op.Add(attention_output, self.mlp(op, feed_forward_input)),
            )
        else:
            hidden_states = op.Add(residual, attention_output)
            residual = hidden_states
            hidden_states = op.Add(
                residual,
                self.mlp(op, self.post_attention_layernorm(op, hidden_states)),
            )
        return hidden_states, present_key_value


class _LegacyGGUFTextModel(nn.Module):
    """Validated conventional decoder body with RoPE or causal ALiBi."""

    def __init__(self, config: ArchitectureConfig, architecture: str):
        super().__init__()
        self._architecture = architecture
        self._dtype = config.dtype
        self._alibi_max_bias = config.alibi_max_bias
        self._num_heads = config.num_attention_heads
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        layer_norm = architecture in {"gptneox", "jais", "mpt"}
        gated_mlp = architecture in {"jais", "refact", "ernie4_5"}
        self.layers = nn.ModuleList(
            [
                _LegacyGGUFDecoderLayer(
                    config,
                    layer_norm=layer_norm,
                    gated_mlp=gated_mlp,
                    parallel_residual=config.use_parallel_residual,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        norm_class = LayerNorm if layer_norm else RMSNorm
        self.norm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        if self._alibi_max_bias is None:
            self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values=None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        if self._alibi_max_bias is None:
            position_embeddings = self.rotary_emb(op, position_ids)
            attention_bias = create_attention_bias(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                dtype=self._dtype,
            )
        else:
            position_embeddings = None
            attention_bias = _create_alibi_bias(
                op,
                input_ids,
                attention_mask,
                num_heads=self._num_heads,
                max_bias=self._alibi_max_bias,
                dtype=self._dtype,
            )

        present_key_values = []
        past_key_values = past_key_values or [None] * len(self.layers)
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


class _OpenELMTextModel(nn.Module):
    """OpenELM body with exact per-layer Q/KV heads and feed-forward widths."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList()
        for heads, kv_heads, intermediate in zip(
            config.layer_attention_head_counts,
            config.layer_attention_kv_head_counts,
            config.layer_intermediate_sizes,
        ):
            layer_config = dataclasses.replace(
                config,
                num_attention_heads=heads,
                num_key_value_heads=kv_heads,
                intermediate_size=intermediate,
                attn_qk_norm=True,
                attn_qk_norm_full=False,
            )
            self.layers.append(
                _LegacyGGUFDecoderLayer(
                    layer_config,
                    layer_norm=False,
                    gated_mlp=True,
                    parallel_residual=False,
                )
            )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values=None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )
        present_key_values = []
        past_key_values = past_key_values or [None] * len(self.layers)
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


class ExactLegacyGGUFCausalLMModel(nn.Module):
    """Exact fail-closed conventional decoder selected by GGUF architecture metadata."""

    default_task = "text-generation"
    category = "Text Generation"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        architecture = getattr(config, "_gguf_arch", None)
        if architecture not in {"gptneox", "jais", "mpt", "refact", "ernie4_5", "openelm"}:
            raise ValueError(
                "ExactLegacyGGUFCausalLMModel requires a validated conventional GGUF config"
            )
        self.config = config
        self.model = (
            _OpenELMTextModel(config)
            if architecture == "openelm"
            else _LegacyGGUFTextModel(config, architecture)
        )
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def kv_cache_specs(self) -> list[tuple[int, int]]:
        if self.config.layer_attention_kv_head_counts:
            return [
                (heads, self.config.head_dim)
                for heads in self.config.layer_attention_kv_head_counts
            ]
        return [
            (self.config.num_key_value_heads, self.config.head_dim)
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

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = dict(state_dict)
        for name in tuple(state_dict):
            if ".self_attn.qkv_proj." not in name:
                continue
            layer_match = re.search(r"\.layers\.(\d+)\.", name)
            layer = int(layer_match.group(1)) if layer_match is not None else 0
            num_heads = (
                self.config.layer_attention_head_counts[layer]
                if self.config.layer_attention_head_counts
                else self.config.num_attention_heads
            )
            num_kv_heads = (
                self.config.layer_attention_kv_head_counts[layer]
                if self.config.layer_attention_kv_head_counts
                else self.config.num_key_value_heads
            )
            q, k, v = split_fused_qkv(
                state_dict.pop(name),
                num_heads,
                num_kv_heads,
                self.config.head_dim,
            )
            state_dict[name.replace("qkv_proj", "q_proj")] = q
            state_dict[name.replace("qkv_proj", "k_proj")] = k
            state_dict[name.replace("qkv_proj", "v_proj")] = v

        if getattr(self.config, "_gguf_arch", None) == "mpt" and not self.config.mlp_bias:
            hidden = self.config.hidden_size
            for layer in range(self.config.num_hidden_layers):
                prefix = f"model.layers.{layer}"
                state_dict.setdefault(f"{prefix}.input_layernorm.bias", torch.zeros(hidden))
                state_dict.setdefault(
                    f"{prefix}.post_attention_layernorm.bias", torch.zeros(hidden)
                )
            state_dict.setdefault("model.norm.bias", torch.zeros(hidden))
        if self.config.tie_word_embeddings:
            state_dict.pop("lm_head.weight", None)
        return state_dict
