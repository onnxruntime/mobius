# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DFlash speculative-decoding draft model components.

Cross-attention drafter described in the DFlash paper (z-lab/dflash).  Each
draft layer attends from "noise" queries (the mask/draft tokens) over keys
and values built by concatenating projected target hidden states and the
same noise tokens, then applying RoPE.  Attention is non-causal across the
full ``[context ‖ noise]`` sequence — every draft token sees every context
token and every other draft token in its block.

The drafter has no embedding table or LM head of its own; those are
borrowed from the target at inference time (the target's ``embed_tokens``
provides the mask-token embedding fed in as ``noise_embedding``, and the
target's ``lm_head`` decodes the drafter's output hidden states).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components._attention import _apply_attention
from mobius.components._common import Linear
from mobius.components._mlp import MLP
from mobius.components._rms_norm import RMSNorm
from mobius.components._rotary_embedding import apply_rotary_pos_emb

if TYPE_CHECKING:
    import onnx_ir as ir


class DFlashAttention(nn.Module):
    """Non-causal cross-attention used by every DFlash draft layer.

    Q comes from the noise stream (``hidden_states``) only.  K and V are
    built from the concatenation of projected target hidden states
    (``target_hidden``) and the noise stream, then per-head RMSNorm'd
    (Qwen3 style) and RoPE-rotated.  Past K/V from previous draft steps
    are concatenated by ``op.Attention`` internally.

    The forward signature mirrors :class:`mobius.components.Attention` but
    adds the ``target_hidden`` argument and accepts two separate
    ``position_embeddings`` tuples (one for Q over the noise positions,
    one for K over the full ``[context ‖ noise]`` positions) so the graph
    avoids any symbolic slicing of the rotary tables.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        prf = config.partial_rotary_factor if config.partial_rotary_factor is not None else 1.0
        self.rotary_embedding_dim = 0 if math.isclose(prf, 1.0) else int(self.head_dim * prf)
        self._rope_interleave = config.rope_interleave

        self.q_proj = Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.k_proj = Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.v_proj = Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.o_proj = Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attn_o_bias,
        )
        # Qwen3-family per-head Q/K RMSNorm.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        target_hidden: ir.Value,
        q_position_embeddings: tuple,
        k_position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        # --- Q from noise only ---
        q = self.q_proj(op, hidden_states)
        q = op.Reshape(q, [0, 0, -1, self.head_dim])
        q = self.q_norm(op, q)
        q = op.Reshape(q, [0, 0, -1])
        q = apply_rotary_pos_emb(
            op,
            x=q,
            position_embeddings=q_position_embeddings,
            num_heads=self.num_attention_heads,
            rotary_embedding_dim=self.rotary_embedding_dim,
            interleaved=self._rope_interleave,
        )

        # --- K, V from [target_hidden ‖ noise] ---
        k_ctx = self.k_proj(op, target_hidden)
        k_noise = self.k_proj(op, hidden_states)
        k = op.Concat(k_ctx, k_noise, axis=1)
        v_ctx = self.v_proj(op, target_hidden)
        v_noise = self.v_proj(op, hidden_states)
        v = op.Concat(v_ctx, v_noise, axis=1)

        k = op.Reshape(k, [0, 0, -1, self.head_dim])
        k = self.k_norm(op, k)
        k = op.Reshape(k, [0, 0, -1])
        k = apply_rotary_pos_emb(
            op,
            x=k,
            position_embeddings=k_position_embeddings,
            num_heads=self.num_key_value_heads,
            rotary_embedding_dim=self.rotary_embedding_dim,
            interleaved=self._rope_interleave,
        )

        # --- Non-causal attention.  op.Attention concatenates past_key /
        # past_value with the supplied K / V internally, giving the final
        # K = [past_K ‖ k_ctx ‖ k_noise] — exactly what DFlash needs. ---
        attn_output, present_key, present_value = _apply_attention(
            op,
            q,
            k,
            v,
            attn_mask=None,
            past_key=past_key_value[0] if past_key_value is not None else None,
            past_value=past_key_value[1] if past_key_value is not None else None,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale=self.scaling,
            is_causal=0,
        )
        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)


class DFlashDecoderLayer(nn.Module):
    """One DFlash draft transformer block.

    Pre-norm pattern matching ``Qwen3DFlashDecoderLayer`` in
    ``z-lab/dflash:dflash/model.py`` — input_layernorm → cross-attention
    → residual → post_attention_layernorm → SwiGLU MLP → residual.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.self_attn = DFlashAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        target_hidden: ir.Value,
        q_position_embeddings: tuple,
        k_position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_output, present_key_value = self.self_attn(
            op,
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            q_position_embeddings=q_position_embeddings,
            k_position_embeddings=k_position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states, present_key_value
