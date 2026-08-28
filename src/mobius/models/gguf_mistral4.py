# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact explicit-float Mistral4 GGUF decoder with a latent K-only cache."""

from __future__ import annotations

import dataclasses
import math

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    MLP,
    Embedding,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.models.base import CausalLMModel
from mobius.models.deepseek import DeepSeekMoEGate, DeepSeekV3CausalLMModel, _DeepSeekMoEFFN


class Mistral4LatentAttention(nn.Module):
    """DeepSeek-V2 MLA with the graph-visible ``[latent | RoPE]`` K cache."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        if (
            config.q_lora_rank is None
            or config.q_lora_rank <= 0
            or config.kv_lora_rank is None
            or config.kv_lora_rank <= 0
            or config.qk_nope_head_dim is None
            or config.qk_nope_head_dim <= 0
            or config.qk_rope_head_dim is None
            or config.qk_rope_head_dim <= 0
            or config.v_head_dim is None
            or config.v_head_dim <= 0
        ):
            raise ValueError("Mistral4 requires positive Q/KV-LoRA and MLA head dimensions")

        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self._rope_interleave = config.rope_interleave
        self.scaling = _mistral4_attention_scale(config, self.qk_head_dim)

        self.q_a_proj = Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = Linear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
        )
        self.kv_a_proj_with_mqa = Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.k_b_proj = Linear(
            self.kv_lora_rank,
            self.num_heads * self.qk_nope_head_dim,
            bias=False,
        )
        self.v_b_proj = Linear(
            self.kv_lora_rank,
            self.num_heads * self.v_head_dim,
            bias=False,
        )
        self.o_proj = Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
        past_key_value: ir.Value | None = None,
    ):
        # Q: (B, S, H*(D_nope + D_rope)) -> (B, S, H, D_qk).
        query = self.q_b_proj(
            op,
            self.q_a_layernorm(op, self.q_a_proj(op, hidden_states)),
        )
        query = op.Reshape(query, [0, 0, self.num_heads, self.qk_head_dim])
        query_nope, query_rope = op.Split(
            query,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            axis=-1,
            _outputs=2,
        )
        query_rope = apply_rotary_pos_emb(
            op,
            op.Reshape(query_rope, [0, 0, -1]),
            position_embeddings,
            num_heads=self.num_heads,
            rotary_embedding_dim=0,
            interleaved=self._rope_interleave,
        )
        query = op.Concat(
            query_nope,
            op.Reshape(
                query_rope,
                [0, 0, self.num_heads, self.qk_rope_head_dim],
            ),
            axis=-1,
        )

        # Cache exactly the normalized latent content followed by the rotated
        # single-head RoPE key: (B, 1, T, L + D_rope).
        compressed = self.kv_a_proj_with_mqa(op, hidden_states)
        latent, key_rope = op.Split(
            compressed,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            axis=-1,
            _outputs=2,
        )
        latent = self.kv_a_layernorm(op, latent)
        key_rope = apply_rotary_pos_emb(
            op,
            key_rope,
            position_embeddings,
            num_heads=1,
            rotary_embedding_dim=0,
            interleaved=self._rope_interleave,
        )
        current = op.Unsqueeze(op.Concat(latent, key_rope, axis=-1), [1])
        present = (
            current if past_key_value is None else op.Concat(past_key_value, current, axis=2)
        )

        # Re-expand the graph-visible latent cache for portable ONNX Attention.
        # This is algebraically equivalent to llama.cpp's absorbed Q/value path.
        cached = op.Squeeze(present, [1])
        cached_latent, cached_rope = op.Split(
            cached,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            axis=-1,
            _outputs=2,
        )
        key_nope = self.k_b_proj(op, cached_latent)
        key_nope = op.Reshape(
            key_nope,
            [0, 0, self.num_heads, self.qk_nope_head_dim],
        )
        cached_rope = op.Reshape(
            cached_rope,
            [0, 0, 1, self.qk_rope_head_dim],
        )
        cached_rope = op.Expand(cached_rope, [1, 1, self.num_heads, 1])
        key = op.Concat(key_nope, cached_rope, axis=-1)
        value = self.v_b_proj(op, cached_latent)

        output = op.Attention(
            op.Reshape(query, [0, 0, -1]),
            op.Reshape(key, [0, 0, -1]),
            value,
            attention_bias,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            scale=self.scaling,
        )
        return self.o_proj(op, output), present


class Mistral4DecoderLayer(nn.Module):
    """One pre-norm latent-attention block with dense or shared/routed MoE FFN."""

    def __init__(self, config: ArchitectureConfig, *, is_moe: bool):
        super().__init__()
        self.self_attn = Mistral4LatentAttention(config)
        self.mlp = _DeepSeekMoEFFN(config, DeepSeekMoEGate(config)) if is_moe else MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
        past_key_value: ir.Value | None = None,
    ):
        residual = hidden_states
        attention_output, present = self.self_attn(
            op,
            self.input_layernorm(op, hidden_states),
            attention_bias,
            position_embeddings,
            past_key_value,
        )
        hidden_states = op.Add(residual, attention_output)

        residual = hidden_states
        hidden_states = self.mlp(
            op,
            self.post_attention_layernorm(op, hidden_states),
        )
        return op.Add(residual, hidden_states), present


class Mistral4TextModel(nn.Module):
    """Mistral4 trunk with a per-layer compressed latent K cache."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        if config.first_k_dense_replace >= config.num_hidden_layers:
            raise ValueError("Mistral4 requires at least one routed-expert layer")
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [
                Mistral4DecoderLayer(
                    config,
                    is_moe=layer >= config.first_k_dense_replace,
                )
                for layer in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        assert config.qk_rope_head_dim is not None
        rope_config = dataclasses.replace(
            config,
            head_dim=config.qk_rope_head_dim,
            partial_rotary_factor=1.0,
        )
        rotary_emb = initialize_rope(rope_config)
        if rotary_emb is None:
            raise ValueError("Mistral4 requires RoPE metadata")
        self.rotary_emb = rotary_emb

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
        past_values = past_key_values or [None] * len(self.layers)
        for layer, past_key_value in zip(self.layers, past_values):
            hidden_states, present = layer(
                op,
                hidden_states,
                attention_bias,
                position_embeddings,
                past_key_value,
            )
            present_key_values.append(present)
        return self.norm(op, hidden_states), present_key_values


class Mistral4GGUFCausalLMModel(DeepSeekV3CausalLMModel):
    """Mistral4 GGUF decoder with MLA, mandatory shared/routed MoE, and latent cache."""

    default_task: str = "mistral4-gguf-text-generation"
    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        base_config = (
            config
            if config.partial_rotary_factor is not None
            else dataclasses.replace(config, partial_rotary_factor=1.0)
        )
        CausalLMModel.__init__(self, base_config)
        self.config = config
        self._replace_text_model(Mistral4TextModel(config))

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return DeepSeekV3CausalLMModel.preprocess_weights(self, state_dict)

    def latent_cache_width(self) -> int:
        """Return the serialized ``[latent | RoPE]`` width for every layer."""
        assert self.config.kv_lora_rank is not None
        assert self.config.qk_rope_head_dim is not None
        return self.config.kv_lora_rank + self.config.qk_rope_head_dim


def _mistral4_attention_scale(
    config: ArchitectureConfig,
    qk_head_dim: int,
) -> float:
    """Return pinned llama.cpp's Mistral4 MLA softmax scale."""
    scale = qk_head_dim**-0.5
    if config.rope_type != "yarn" or not config.rope_scaling:
        return scale
    factor = float(config.rope_scaling.get("factor", 1.0))
    if factor <= 1.0:
        return scale
    mscale = 1.0 + 0.1 * math.log(factor)
    return scale * mscale * mscale
