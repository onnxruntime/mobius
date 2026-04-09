# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma4 text and multimodal model.

Gemma4 extends Gemma3 with several architectural changes:
- Standard RMSNorm (not OffsetRMSNorm) throughout
- Parameterless V normalization in attention (per-head RMS, no learnable scale)
- Attention scale fixed at 1.0 (not 1/sqrt(head_dim))
- Dual head_dim: sliding-window layers use head_dim=256, full-attention layers
  use global_head_dim=512 with partial rotary (partial_rotary_factor=0.25)
- Per-layer input gating: each decoder layer receives a per-layer embedding
  derived from input_ids, gated and projected into the hidden space
- Optional final logit soft-capping (like Gemma2)
- Optional MoE blocks (not yet implemented; guarded by assert)
- Embedding scale uses float() not float16() rounding
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import Gemma4Config
from mobius.components import (
    MLP,
    Embedding,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._activations import get_activation
from mobius.components._attention import _apply_attention, apply_rotary_pos_emb
from mobius.models.base import CausalLMModel
from mobius.models.gemma3 import Gemma3MultiModalModel

if TYPE_CHECKING:
    import onnx_ir as ir


class Gemma4ScaledWordEmbedding(Embedding):
    """Embedding table with sqrt(hidden_size) scaling.

    Unlike Gemma3TextScaledWordEmbedding, uses float() not float16()
    to match Gemma4's embedding scale computation.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, op: builder.OpBuilder, input_ids: "ir.Value"):
        embeddings = super().forward(op, input_ids)
        return op.Mul(embeddings, self.embed_scale)


class Gemma4Attention(nn.Module):
    """Gemma4 multi-head attention with per-head QKV normalization.

    Key differences from standard Attention:
    - Attention scale is fixed at 1.0 (HF Gemma4TextAttention hardcodes this)
    - Q and K are normalized per-head with learnable RMSNorm (not OffsetRMSNorm)
    - V is normalized per-head with a *parameterless* RMS (no learnable scale)
    - head_dim and rotary_embedding_dim are passed explicitly (not from config)
      because they differ between sliding-window and full-attention layers

    Args:
        config: Gemma4Config with architecture hyperparameters.
        head_dim: Head dimension for Q/K/V projections (256 for sliding layers,
            ``config.global_head_dim`` for full-attention layers).
        rotary_embedding_dim: Number of dimensions to rotate (0 = full head_dim).
    """

    def __init__(self, config: Gemma4Config, head_dim: int, rotary_embedding_dim: int):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = head_dim
        self.scaling = 1.0
        self._v_norm_eps = config.rms_norm_eps
        self.rotary_embedding_dim = rotary_embedding_dim
        self._rope_interleave = config.rope_interleave

        self.q_proj = Linear(
            config.hidden_size, config.num_attention_heads * head_dim, bias=False
        )
        self.k_proj = Linear(
            config.hidden_size, config.num_key_value_heads * head_dim, bias=False
        )
        self.v_proj = Linear(
            config.hidden_size, config.num_key_value_heads * head_dim, bias=False
        )
        self.o_proj = Linear(
            config.num_attention_heads * head_dim, config.hidden_size, bias=False
        )

        # Per-head Q/K norms (standard RMSNorm with learnable weight)
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: "ir.Value",
        attention_bias: "ir.Value",
        position_embeddings: tuple | None = None,
        past_key_value: tuple | None = None,
    ):
        query_states = self.q_proj(op, hidden_states)
        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)

        # Per-head Q/K normalization: reshape to (B, T, num_heads, head_dim), norm, flatten
        query_states = op.Reshape(query_states, [0, 0, -1, self.head_dim])
        key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
        query_states = self.q_norm(op, query_states)
        key_states = self.k_norm(op, key_states)
        query_states = op.Reshape(query_states, [0, 0, -1])
        key_states = op.Reshape(key_states, [0, 0, -1])

        # V normalization: parameterless per-head RMS (no learnable scale)
        # Reshape to (B, T, num_kv_heads, head_dim), normalize over last dim, flatten
        value_states = op.Reshape(
            value_states,
            op.Constant(value_ints=[0, 0, self.num_key_value_heads, self.head_dim]),
        )
        sq = op.Mul(value_states, value_states)
        mean_sq = op.ReduceMean(sq, [-1], keepdims=1)
        rms = op.Sqrt(op.Add(mean_sq, self._v_norm_eps))
        value_states = op.Div(value_states, rms)
        value_states = op.Reshape(value_states, [0, 0, -1])

        # Apply RoPE to Q and K
        if position_embeddings is not None:
            query_states = apply_rotary_pos_emb(
                op,
                x=query_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_attention_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )
            key_states = apply_rotary_pos_emb(
                op,
                x=key_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_key_value_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )

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
        )

        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)


class Gemma4DecoderLayer(nn.Module):
    """Gemma4 decoder layer with four norms, layer_scalar, and optional per-layer input.

    Architecture:
        residual + post_attn_norm(attn(input_layernorm(x)))
        residual + post_ff_norm(mlp(pre_ff_norm(x)))
        x = x * layer_scalar
        if per_layer_input:
            x += post_per_layer_norm(project(act(gate(x)) * per_layer_input))

    Unlike Gemma3, Gemma4 uses standard RMSNorm (not OffsetRMSNorm).

    Args:
        config: Gemma4Config.
        layer_type: ``"sliding_attention"`` or ``"full_attention"``.
    """

    def __init__(self, config: Gemma4Config, layer_type: str):
        super().__init__()
        is_full = layer_type == "full_attention"
        head_dim = (config.global_head_dim or config.head_dim) if is_full else config.head_dim
        # Sliding layers: rotate all head_dim dims (rotary_dim=0 = full rotation)
        # Full-attention layers: partially rotate (partial_rotary_factor * global_head_dim)
        if is_full:
            rotary_dim = int(head_dim * config.global_partial_rotary_factor)
        else:
            rotary_dim = 0  # full rotation for sliding attention

        self.self_attn = Gemma4Attention(config, head_dim=head_dim, rotary_embedding_dim=rotary_dim)
        self.mlp = MLP(config)

        # Four norms (standard RMSNorm, not OffsetRMSNorm)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Scalar applied at the end of each layer (initialized to 1, learned or fixed)
        self.layer_scalar = nn.Parameter([1])

        # Per-layer input gating (disabled when hidden_size_per_layer_input == 0)
        self._per_layer_dim = config.hidden_size_per_layer_input
        if self._per_layer_dim > 0:
            self.per_layer_input_gate = Linear(
                config.hidden_size, self._per_layer_dim, bias=False
            )
            self.per_layer_projection = Linear(
                self._per_layer_dim, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.act_fn = get_activation(config.hidden_act)

        if config.enable_moe_block:
            raise NotImplementedError(
                "Gemma4 MoE block (enable_moe_block=True) is not yet implemented. "
                "Set enable_moe_block=False to use the dense-only path."
            )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: "ir.Value",
        attention_bias: "ir.Value",
        position_embeddings: tuple,
        per_layer_input: "ir.Value | None",
        past_key_value: tuple | None,
    ):
        # Pre-attention norm + attention
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_output, present_key_value = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        # Post-attention norm applied to attention output, then residual
        hidden_states = self.post_attention_layernorm(op, attn_output)
        hidden_states = op.Add(residual, hidden_states)

        # Pre-feedforward norm + MLP + post-feedforward norm + residual
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = self.post_feedforward_layernorm(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        # Layer scalar (initialized to 1; applied at end of layer body)
        hidden_states = op.Mul(hidden_states, self.layer_scalar)

        # Per-layer input gating (skip when disabled)
        if self._per_layer_dim > 0 and per_layer_input is not None:
            # Gate: act(W_gate @ hidden) -> [B, T, per_layer_dim]
            gated = self.per_layer_input_gate(op, hidden_states)
            gated = self.act_fn(op, gated)
            # Hadamard with the per-layer embedding for this layer
            gated = op.Mul(gated, per_layer_input)
            # Project back to hidden_size and normalize
            projected = self.per_layer_projection(op, gated)
            projected = self.post_per_layer_input_norm(op, projected)
            hidden_states = op.Add(hidden_states, projected)

        return hidden_states, present_key_value


class Gemma4TextModel(nn.Module):
    """Gemma4 text model with hybrid local/global attention and dual RoPE.

    Uses standard RMSNorm (not OffsetRMSNorm) and float() embedding scaling.
    Optionally maintains per-layer input embeddings when
    ``config.hidden_size_per_layer_input > 0``.
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self._dtype = config.dtype

        # Embedding scale: float() not float16() (Gemma4 differs from Gemma3 here)
        embed_scale = float(config.hidden_size**0.5)
        self.embed_tokens = Gemma4ScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=embed_scale,
        )

        self.layer_types = config.layer_types or ["sliding_attention"] * config.num_hidden_layers
        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, lt) for lt in self.layer_types]
        )
        self.sliding_window = config.sliding_window
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Sliding RoPE: use base config (rope_theta=10000, full rotation)
        self.rotary_emb = initialize_rope(config)

        # Full-attention RoPE: different theta, head_dim, and partial_rotary_factor
        global_head_dim = config.global_head_dim or config.head_dim
        global_rope_config = dataclasses.replace(
            config,
            rope_theta=config.global_rope_theta,
            head_dim=global_head_dim,
            partial_rotary_factor=config.global_partial_rotary_factor,
            rope_scaling=None,
        )
        self.rotary_emb_global = initialize_rope(global_rope_config)

        # Per-layer input components (disabled when hidden_size_per_layer_input == 0)
        self._per_layer_dim = config.hidden_size_per_layer_input
        self._num_layers = config.num_hidden_layers
        if self._per_layer_dim > 0:
            # Plain embedding (no scale), vocab = vocab_size_per_layer_input
            self.embed_tokens_per_layer = Embedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * self._per_layer_dim,
                config.pad_token_id,
            )
            # Projects hidden_states to [num_layers * per_layer_dim]
            self.per_layer_model_projection = Linear(
                config.hidden_size,
                config.num_hidden_layers * self._per_layer_dim,
                bias=False,
            )
            self.per_layer_projection_norm = RMSNorm(
                self._per_layer_dim, eps=config.rms_norm_eps
            )
            self._per_layer_projection_scale = float(config.hidden_size**-0.5)
            self._per_layer_input_scale = float(0.5**0.5)  # 1/sqrt(2)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: "ir.Value",
        attention_mask: "ir.Value",
        position_ids: "ir.Value",
        past_key_values: list | None = None,
        inputs_embeds: "ir.Value | None" = None,
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)

        # Compute per-layer inputs from input_ids (skip when disabled or no input_ids)
        per_layer_inputs = None
        if self._per_layer_dim > 0 and input_ids is not None:
            per_layer_inputs = self._compute_per_layer_inputs(op, input_ids, hidden_states)

        # Build per-type position embeddings and attention biases
        query_input = input_ids if input_ids is not None else hidden_states
        position_embeddings_dict = {
            "full_attention": self.rotary_emb_global(op, position_ids),
            "sliding_attention": self.rotary_emb(op, position_ids),
        }
        attention_bias_dict = {
            "full_attention": create_attention_bias(
                op,
                input_ids=query_input,
                attention_mask=attention_mask,
                dtype=self._dtype,
            ),
            "sliding_attention": create_attention_bias(
                op,
                input_ids=query_input,
                attention_mask=attention_mask,
                sliding_window=self.sliding_window,
                dtype=self._dtype,
            ),
        }

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for i, (layer, layer_type, past_kv) in enumerate(
            zip(self.layers, self.layer_types, past_kvs)
        ):
            per_layer_input = (
                self._get_per_layer_input(op, per_layer_inputs, i)
                if per_layer_inputs is not None
                else None
            )
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias_dict[layer_type],
                position_embeddings=position_embeddings_dict[layer_type],
                per_layer_input=per_layer_input,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values

    def _compute_per_layer_inputs(
        self,
        op: builder.OpBuilder,
        input_ids: "ir.Value",
        inputs_embeds: "ir.Value",
    ) -> "ir.Value":
        """Build per-layer inputs from input_ids and projected hidden states.

        Returns a tensor of shape [B, T, num_layers, per_layer_dim] that is
        sliced per-layer inside the decoder layer loop.
        """
        # Embed input_ids with the per-layer vocabulary
        # per_layer_embed: [B, T, num_layers * per_layer_dim]
        per_layer_embed = self.embed_tokens_per_layer(op, input_ids)
        # Reshape to [B, T, num_layers, per_layer_dim]
        per_layer_embed = op.Reshape(
            per_layer_embed,
            op.Constant(value_ints=[0, 0, self._num_layers, self._per_layer_dim]),
        )

        # Project hidden states and scale by hidden_size**-0.5
        proj = self.per_layer_model_projection(op, inputs_embeds)
        proj = op.Mul(proj, self._per_layer_projection_scale)
        # Reshape to [B, T, num_layers, per_layer_dim] and normalize
        proj = op.Reshape(
            proj,
            op.Constant(value_ints=[0, 0, self._num_layers, self._per_layer_dim]),
        )
        proj = self.per_layer_projection_norm(op, proj)

        # Combine: (embed + proj) * (1/sqrt(2))
        per_layer_inputs = op.Add(per_layer_embed, proj)
        per_layer_inputs = op.Mul(per_layer_inputs, self._per_layer_input_scale)
        return per_layer_inputs

    def _get_per_layer_input(
        self, op: builder.OpBuilder, per_layer_inputs: "ir.Value", layer_idx: int
    ) -> "ir.Value":
        """Extract the per-layer input slice for layer_idx.

        per_layer_inputs: [B, T, num_layers, per_layer_dim]
        Returns: [B, T, per_layer_dim]
        """
        idx = op.Constant(value_ints=[layer_idx])
        result = op.Gather(per_layer_inputs, idx, axis=2)
        return op.Squeeze(result, [2])


class Gemma4CausalLMModel(CausalLMModel):
    """Gemma4 causal LM with hybrid local/global attention and optional MoE.

    Implements the Gemma4 text-decoder architecture:
    - Standard RMSNorm (not OffsetRMSNorm)
    - Dual head_dim: 256 for sliding layers, 512 for full-attention layers
    - Dual RoPE: theta=10000 (sliding), theta=1e6 with partial rotation (full)
    - Per-head Q/K RMSNorm, parameterless V normalization, scale=1.0
    - Per-layer input gating (when ``config.hidden_size_per_layer_input > 0``)
    - Optional final logit soft-capping (tanh scaled by ``final_logit_softcapping``)
    """

    config_class: type = Gemma4Config

    def __init__(self, config: Gemma4Config):
        super().__init__(config)
        self.model = Gemma4TextModel(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: "ir.Value",
        attention_mask: "ir.Value | None",
        position_ids: "ir.Value",
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(op, hidden_states)

        # Optional final logit soft-capping (tanh scaled): logit_cap * tanh(x / logit_cap)
        if self.config.final_logit_softcapping:
            cap = float(self.config.final_logit_softcapping)
            logits = op.Mul(op.Tanh(op.Div(logits, cap)), cap)

        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Strip multimodal prefixes and remove non-text weights."""
        for key in list(state_dict.keys()):
            if "language_model." in key:
                new_key = key.replace("language_model.", "")
                state_dict[new_key] = state_dict.pop(key)
            elif "vision_tower" in key or "multi_modal_projector" in key:
                state_dict.pop(key)
            elif "audio_tower" in key:
                state_dict.pop(key)
        return super().preprocess_weights(state_dict)


class Gemma4MultiModalModel(Gemma3MultiModalModel):
    """Placeholder for the Gemma4 multimodal model (vision + audio + text).

    TODO: Implement the full Gemma4 multimodal architecture:
    - Vision encoder: ViT-like SigLIP, 16 layers, patch_size=16, 280 soft tokens/image,
      use_clipped_linears=True
    - Audio encoder: Conformer, 12 layers, hidden_size=1024,
      subsampling_conv_channels=[128, 32], causal chunked attention,
      output projected to hidden_size=1536
    - Text decoder: Gemma4CausalLMModel with dual RoPE and optional MoE
    """

    default_task: str = "gemma4-multimodal"
