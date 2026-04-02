# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Aria multimodal model (Rhymes AI).

Aria combines a SigLIP vision encoder with a Mixture-of-Experts text decoder,
connected via a perceiver-style cross-attention projector.  The 3-model split
matches the onnxruntime-genai pipeline convention.

Architecture:
    pixel_values (B, 3, 980, 980)
        → SigLIP ViT encoder (idefics3_vision, 27 layers, dim=1152)
        → AriaProjector (perceiver cross-attn resampler)
            - learnable query tokens (max 256 queries at full resolution)
            - cross-attention: queries ← visual patches (SigLIP output)
            - layer norm + MLP (GELU-new)
        → visual_tokens (batch * num_queries, 2560)

    token IDs + visual tokens
        → embedding fusion (CumSum-gather scatter at image-token positions)
        → AriaText MoE decoder
            - 28 layers: RoPE self-attn + MoE FFN
            - MoE: TopK router (top-6 of 64 experts) + 2 always-active shared experts
        → logits (batch, seq_len, vocab_size)

Three exported ONNX models:
    "vision"    — SigLIP + projector:  pixel_values → image_features (2D flat)
    "embedding" — token embed + fusion: input_ids + image_features → inputs_embeds
    "decoder"   — AriaText MoE LM:     inputs_embeds → logits + KV cache

HuggingFace reference: ``AriaForConditionalGeneration``
(model_type="aria", text_config.model_type="aria_text").

Weight naming:
    Vision model:
        vision_tower.vision_model.*       — SigLIP ViT (via VisionModel component)
        projector.query                   — learnable query tensor
        projector.cross_attn.*            — perceiver cross-attn (QKV weights fused)
        projector.layer_norm.*            — post cross-attn layer norm
        projector.feed_forward.*          — 2-layer GELU MLP projecting to text dim

    Decoder model (after stripping ``language_model.`` prefix):
        model.embed_tokens.*
        model.layers.{i}.input_layernorm.*
        model.layers.{i}.self_attn.*           — q/k/v/o_proj standard attention
        model.layers.{i}.post_attention_layernorm.*
        model.layers.{i}.mlp.gate.weight       — router (from router.weight)
        model.layers.{i}.mlp.experts.{j}.*    — per-expert SwiGLU (from fc1/fc2)
        model.layers.{i}.mlp.shared_experts.* — always-active shared experts
        model.norm.*
        lm_head.*

    Embedding model:
        embed_tokens.*
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import vlm_decoder_weights, vlm_embedding_weights
from mobius.components import (
    Embedding,
    Linear,
    RMSNorm,
    VisionModel,
    create_padding_mask,
    initialize_rope,
)
from mobius.components._attention import Attention
from mobius.components._common import LayerNorm
from mobius.components._moe import TopKGate

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# Perceiver cross-attention projector
# ---------------------------------------------------------------------------


class _AriaProjectorMLP(nn.Module):
    """Post-attention MLP in the Aria projector: vision_hidden → text_hidden.

    Two-layer MLP with GELU-new activation (tanh approximation).

    Weight names match HF ``projector.feed_forward.{linear_in,linear_out}.*``.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear_in = Linear(in_dim, hidden_dim, bias=False)
        self.linear_out = Linear(hidden_dim, out_dim, bias=False)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        h = self.linear_in(op, x)
        # gelu_new(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
        sqrt_2_over_pi = op.CastLike(op.Constant(value_float=0.7978845608028654), h)
        coeff = op.CastLike(op.Constant(value_float=0.044715), h)
        inner = op.Mul(h, op.Mul(h, h))  # x^3
        inner = op.Mul(sqrt_2_over_pi, op.Add(h, op.Mul(coeff, inner)))
        cdf = op.Mul(
            op.CastLike(op.Constant(value_float=0.5), h),
            op.Add(op.CastLike(op.Constant(value_float=1.0), h), op.Tanh(inner)),
        )
        return self.linear_out(op, op.Mul(h, cdf))


class _AriaPerceiverCrossAttn(nn.Module):
    """Perceiver-style cross-attention for the Aria projector.

    HF applies two sequential QKV linear layers (q_proj → mha.in_proj_weight).
    ``preprocess_weights`` fuses them into a single linear per Q/K/V:

        fused_q_weight = mha.in_proj_weight[0:H] @ q_proj.weight  (H x H)
        fused_k_weight = mha.in_proj_weight[H:2H] @ k_proj.weight (H x H)
        fused_v_weight = mha.in_proj_weight[2H:3H] @ v_proj.weight(H x H)
        q/k/v biases  = mha.in_proj_bias slices                   (H,)
        out_proj       = mha.out_proj.weight / bias
        linear         = cross_attn.linear.weight / bias
        layer_norm     = cross_attn.layer_norm.weight / bias
        layer_norm_kv  = cross_attn.layer_norm_kv.weight / bias
    """

    def __init__(self, vision_hidden: int, num_heads: int) -> None:
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = vision_hidden // num_heads

        self.layer_norm = LayerNorm(vision_hidden)
        self.layer_norm_kv = LayerNorm(vision_hidden)
        # Fused Q/K/V projections (weights set in preprocess_weights)
        self.q_proj = Linear(vision_hidden, vision_hidden, bias=True)
        self.k_proj = Linear(vision_hidden, vision_hidden, bias=True)
        self.v_proj = Linear(vision_hidden, vision_hidden, bias=True)
        # Output projection + extra linear (matching HF mha.out_proj + linear)
        self.out_proj = Linear(vision_hidden, vision_hidden, bias=True)
        self.linear = Linear(vision_hidden, vision_hidden, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,  # learnable queries (B, Nq, H_vis)
        key_value_states: ir.Value,  # vision patches   (B, Np, H_vis)
    ) -> ir.Value:
        """Cross-attend query tokens over vision patches.

        Returns:
            (B, Nq, H_vis) — updated query representations.
        """
        query = self.q_proj(op, self.layer_norm(op, hidden_states))  # (B, Nq, H)
        kv_norm = self.layer_norm_kv(op, key_value_states)
        key = self.k_proj(op, kv_norm)  # (B, Np, H)
        value = self.v_proj(op, kv_norm)  # (B, Np, H)

        # Non-causal cross-attention; different Q/KV sequence lengths supported
        attn_out = op.Attention(
            query,
            key,
            value,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            scale=float(self._head_dim**-0.5),
            is_causal=0,
        )  # (B, Nq, H)

        # HF applies out_proj then linear sequentially
        return self.linear(op, self.out_proj(op, attn_out))  # (B, Nq, H)


class _AriaProjector(nn.Module):
    """Aria perceiver projector: visual patches → query tokens in text space.

    Uses a fixed ``max_query_tokens`` (256 for full-resolution 4900-patch images).
    Smaller images use fewer patches but the same query count for simplicity.

    Parameters match HF ``AriaProjector``:
        query:        (max_query_tokens, vision_hidden) — learnable
        cross_attn:   AriaCrossAttention (QKV weights fused)
        layer_norm:   LayerNorm(vision_hidden)
        feed_forward: _AriaProjectorMLP(vision_hidden → hidden_size)
    """

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        v_dim = config.vision.hidden_size
        t_dim = config.hidden_size
        num_heads = config.vision.num_attention_heads
        max_q = getattr(config, "max_query_tokens", 256)

        # Learnable query embedding — shape: (max_query_tokens, vision_hidden)
        self.query = nn.Parameter([max_q, v_dim])
        self.cross_attn = _AriaPerceiverCrossAttn(v_dim, num_heads)
        self.layer_norm = LayerNorm(v_dim)
        self.feed_forward = _AriaProjectorMLP(v_dim, t_dim, t_dim)

    def forward(
        self,
        op: builder.OpBuilder,
        vision_features: ir.Value,  # (B, num_patches, vision_hidden)
    ) -> ir.Value:
        """Project visual patches to text-aligned query tokens.

        Args:
            vision_features: (batch, num_patches, vision_hidden) from SigLIP encoder.

        Returns:
            (batch, max_query_tokens, hidden_size) projected visual tokens.
        """
        batch = op.Shape(vision_features, start=0, end=1)  # (1,)
        nq = op.Shape(self.query, start=0, end=1)  # (1,)
        h = op.Shape(self.query, start=1, end=2)  # (1,)
        # Expand learnable queries: (1, Nq, H) → (B, Nq, H)
        q_shape = op.Concat(batch, nq, h, axis=0)
        queries = op.Expand(op.Unsqueeze(self.query, [0]), q_shape)  # (B, Nq, H_vis)

        # Cross-attend and project
        queries = self.cross_attn(op, queries, vision_features)  # (B, Nq, H_vis)
        queries = self.layer_norm(op, queries)  # (B, Nq, H_vis)
        return self.feed_forward(op, queries)  # (B, Nq, T_dim)


# ---------------------------------------------------------------------------
# AriaText MoE FFN
# ---------------------------------------------------------------------------


class _AriaExpertMLP(nn.Module):
    """Single SwiGLU MLP expert.

    Weight names (after per-expert splitting in preprocess_weights):
        gate_proj.weight: (intermediate_size, hidden_size)
        up_proj.weight:   (intermediate_size, hidden_size)
        down_proj.weight: (hidden_size, intermediate_size)
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        gate = self.gate_proj(op, x)
        silu = op.Mul(gate, op.Sigmoid(gate))  # SiLU activation
        return self.down_proj(op, op.Mul(silu, self.up_proj(op, x)))


class _AriaSharedMLP(nn.Module):
    """Always-active shared expert SwiGLU MLP.

    Combined capacity of all shared experts in a single MLP:
    intermediate_size = per_expert_size * moe_num_shared_experts

    Weight names match HF ``mlp.shared_experts.*``.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        gate = self.gate_proj(op, x)
        silu = op.Mul(gate, op.Sigmoid(gate))  # SiLU
        return self.down_proj(op, op.Mul(silu, self.up_proj(op, x)))


class _AriaMoEFFN(nn.Module):
    """Aria MoE feed-forward layer.

    Combines top-k routed experts with always-active shared experts:
        output = routed_output + shared_output

    Uses loop-over-experts dispatch (same pattern as MoELayer component).

    Weight names after preprocess_weights (inside ``mlp.*`` of each layer):
        gate.weight:                   (num_experts, hidden_size)  ← router.weight
        experts.{i}.gate_proj.weight:  (inter, hidden)  ← split from fc1
        experts.{i}.up_proj.weight:    (inter, hidden)  ← split from fc1
        experts.{i}.down_proj.weight:  (hidden, inter)  ← split from fc2
        shared_experts.{gate,up,down}_proj.weight:  pass-through
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        num_shared_experts: int,
    ) -> None:
        super().__init__()
        self.gate = TopKGate(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList(
            [_AriaExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        )
        # Combined shared-expert MLP: capacity = per_expert * num_shared
        self.shared_experts = _AriaSharedMLP(
            hidden_size, intermediate_size * num_shared_experts
        )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        """Apply routed experts + shared experts.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            (batch, seq_len, hidden_size)
        """
        routing_weights, selected_experts = self.gate(op, hidden_states)
        # routing_weights:  (B, T, top_k)  softmax-normalized
        # selected_experts: (B, T, top_k)  expert indices

        # Loop-over-experts dispatch: each expert processes all tokens,
        # then contributions are masked by whether that expert was selected.
        routed_result = None
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert(op, hidden_states)  # (B, T, H)
            expert_id = op.Constant(value_int=expert_idx)
            match = op.Equal(selected_experts, expert_id)  # (B, T, top_k) bool
            match_float = op.Cast(match, to=1)  # FLOAT
            weighted = op.Mul(routing_weights, match_float)  # (B, T, top_k)
            weight = op.ReduceSum(weighted, [-1], keepdims=True)  # (B, T, 1)
            contribution = op.Mul(expert_output, weight)  # (B, T, H)
            if routed_result is None:
                routed_result = contribution
            else:
                routed_result = op.Add(routed_result, contribution)

        # Always-active shared experts (additive)
        shared_output = self.shared_experts(op, hidden_states)
        return op.Add(routed_result, shared_output)


# ---------------------------------------------------------------------------
# AriaText decoder layer using MoE FFN
# ---------------------------------------------------------------------------


class _AriaTextBlock(nn.Module):
    """Single AriaText decoder layer: RoPE attention + MoE FFN (pre-norm).

    Follows the same calling convention as :class:`DecoderLayer` —
    takes ``(op, hidden_states, attention_bias, position_embeddings,
    past_key_value)`` — so it can be used directly by :class:`_AriaTextModel`.

    Structure (pre-LayerNorm, same as LLaMA):
        hidden → input_layernorm → self_attn → +residual
               → post_attention_layernorm → mlp (MoE) → +residual

    Weight names match HF ``model.layers.{i}.*``.
    """

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        moe_num_experts = getattr(config, "moe_num_experts", 64)
        moe_topk = getattr(config, "moe_topk", 6)
        moe_num_shared = getattr(config, "moe_num_shared_experts", 2)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Named 'mlp' to match HF weight paths
        self.mlp = _AriaMoEFFN(
            config.hidden_size,
            config.intermediate_size,
            moe_num_experts,
            moe_topk,
            moe_num_shared,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,  # (cos, sin)
        past_key_value: tuple | None,
    ) -> tuple[ir.Value, tuple]:
        # Pre-LN self-attention with residual
        residual = hidden_states
        attn_out, kv = self.self_attn(
            op,
            self.input_layernorm(op, hidden_states),
            attention_bias,
            position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, attn_out)

        # Pre-LN MoE FFN with residual
        residual = hidden_states
        ffn_out = self.mlp(op, self.post_attention_layernorm(op, hidden_states))
        hidden_states = op.Add(residual, ffn_out)

        return hidden_states, kv


class _AriaTextModel(nn.Module):
    """AriaText decoder body: embedding + decoder layers + final norm.

    Identical to :class:`TextModel` in structure but uses ``_AriaTextBlock``
    (with MoE FFN) instead of ``DecoderLayer``.
    """

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [_AriaTextBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,  # (B, T, H)
        attention_mask: ir.Value | None,  # (B, past+T) INT64
        position_ids: ir.Value,  # (B, T) INT64
        past_key_values: list | None,
    ) -> tuple[ir.Value, list]:
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(op, position_ids)  # (cos, sin)

        # Build causal attention bias from INT64 padding mask
        if attention_mask is not None:
            attention_bias = create_padding_mask(
                op,
                input_ids=inputs_embeds,  # used only for seq_len extraction
                attention_mask=attention_mask,
            )
        else:
            attention_bias = None

        past_kvs = past_key_values or [None] * len(self.layers)
        present_kvs: list = []
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, kv = layer(
                op, hidden_states, attention_bias, position_embeddings, past_kv
            )
            present_kvs.append(kv)

        return self.norm(op, hidden_states), present_kvs


# ---------------------------------------------------------------------------
# Three-model split sub-modules
# ---------------------------------------------------------------------------


class _AriaVisionModel(nn.Module):
    """Vision tower (SigLIP ViT) + Aria perceiver projector.

    Exported as the ``"vision"`` ONNX model in the 3-model split.

    Output is 2D ``(batch * max_query_tokens, hidden_size)`` to match the
    flat ``num_image_tokens`` dimension expected by the embedding model.

    Weight paths:
        vision_tower.*   — SigLIP encoder (VisionModel component)
        projector.*      — AriaProjector (learnable queries + cross-attn + MLP)
    """

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.vision_tower = VisionModel(config)
        self.projector = _AriaProjector(config)

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value,  # (B, 3, H, W)
    ) -> ir.Value:
        """Encode images to flat projected visual tokens.

        Returns:
            (B * max_query_tokens, hidden_size) — flat for embedding model.
        """
        visual_features = self.vision_tower(op, pixel_values)  # (B, Np, H_vis)
        projected = self.projector(op, visual_features)  # (B, Nq, T_dim)

        # Flatten batch and query dimensions: (B, Nq, H) → (B*Nq, H)
        h_dim = op.Shape(projected, start=2, end=3)  # (1,)
        neg_one = op.Constant(value_ints=[-1])
        flat_shape = op.Concat(neg_one, h_dim, axis=0)  # [-1, H]
        return op.Reshape(projected, flat_shape)  # (B*Nq, H)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Keep vision + projector weights and fuse double QKV projection."""
        result = {
            k: v
            for k, v in state_dict.items()
            if k.startswith(("vision_tower.", "projector."))
        }
        return _fuse_projector_cross_attn(result)


class _AriaDecoderModel(nn.Module):
    """AriaText MoE causal LM decoder.

    Exported as the ``"decoder"`` ONNX model in the 3-model split.
    Accepts ``inputs_embeds`` (fused text + image) rather than input IDs.

    Weight paths (after stripping ``language_model.`` in preprocess_weights):
        model.layers.{i}.self_attn.*
        model.layers.{i}.mlp.*          — MoE FFN (router renamed, experts split)
        model.norm.*
        lm_head.*
    """

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.model = _AriaTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ) -> tuple[ir.Value, list]:
        hidden_states, present_kvs = self.model(
            op, inputs_embeds, attention_mask, position_ids, past_key_values
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_kvs

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Strip language_model. prefix and split batched expert weights."""
        stripped = vlm_decoder_weights(state_dict, tie=False)
        return _split_expert_weights(stripped)


class _AriaEmbeddingModel(nn.Module):
    """Token embedding + image feature fusion for Aria.

    Exported as the ``"embedding"`` ONNX model in the 3-model split.

    Fuses projected visual tokens into the text embedding sequence at
    positions marked by ``image_token_id`` (default 9).

    Uses CumSum-based index gathering: for each position in the sequence,
    if it is an image token, gather the corresponding row from
    ``image_features`` (using cumulative count as the row index).
    """

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self._image_token_id = config.image_token_id or 9
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,  # (B, T) INT64
        image_features: ir.Value,  # (num_image_tokens, hidden_size) FLOAT — 2D flat
    ) -> ir.Value:
        """Replace image token positions with projected visual features.

        Args:
            input_ids:      (batch, seq_len) INT64.
            image_features: (num_image_tokens, hidden_size) FLOAT.

        Returns:
            inputs_embeds: (batch, seq_len, hidden_size)
        """
        text_embeds = self.embed_tokens(op, input_ids)  # (B, T, H)

        # Boolean mask for image token positions: True where input_ids == image_token_id
        image_mask = op.Equal(
            input_ids,
            op.Constant(value_int=self._image_token_id),
        )  # (B, T)

        # CumSum trick: convert running count to 0-based index into image_features
        mask_int = op.Cast(image_mask, to=7)  # INT64
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))  # (B, T) 1-indexed
        indices = op.Sub(cumsum, op.Constant(value_int=1))  # (B, T) 0-indexed
        indices = op.Clip(indices, op.Constant(value_int=0))  # clip negatives

        # Gather image features at each position (axis=0: rows of image_features)
        gathered = op.Gather(image_features, indices, axis=0)  # (B, T, H)

        # Use image features only where the mask is true
        image_mask_3d = op.Unsqueeze(image_mask, [-1])  # (B, T, 1) bool
        return op.Where(image_mask_3d, gathered, text_embeds)  # (B, T, H)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return vlm_embedding_weights(state_dict)


# ---------------------------------------------------------------------------
# Top-level model (3-model split)
# ---------------------------------------------------------------------------


class AriaForConditionalGeneration(nn.Module):
    """Aria multimodal model for onnxruntime-genai (3-model split).

    Sub-modules match the VisionLanguageTask contract:
        vision_encoder: pixel_values → image_features (flat 2D)
        decoder:        inputs_embeds → logits + KV cache
        embedding:      input_ids + image_features → inputs_embeds
    """

    default_task = "vision-language"
    category = "multimodal"

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.vision_encoder = _AriaVisionModel(config)
        self.decoder = _AriaDecoderModel(config)
        self.embedding = _AriaEmbeddingModel(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return state_dict


# ---------------------------------------------------------------------------
# Weight preprocessing helpers
# ---------------------------------------------------------------------------


def _fuse_projector_cross_attn(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Fuse the double QKV projection in AriaCrossAttention.

    HF applies two sequential linear layers for Q/K/V:
        q_out = mha.in_proj_weight[0:H] @ q_proj(q_in)   + mha.in_proj_bias[0:H]
        k_out = mha.in_proj_weight[H:2H] @ k_proj(k_in)  + mha.in_proj_bias[H:2H]
        v_out = mha.in_proj_weight[2H:3H] @ v_proj(v_in) + mha.in_proj_bias[2H:3H]

    We fuse into a single (out, in) projection each:
        fused_q.weight = mha.in_proj_weight[0:H] @ q_proj.weight   (H, H)
        fused_q.bias   = mha.in_proj_bias[0:H]                     (H,)
    """
    result: dict[str, torch.Tensor] = {}
    prefix = "projector.cross_attn."

    q_proj_w = state_dict.get(f"{prefix}q_proj.weight")
    k_proj_w = state_dict.get(f"{prefix}k_proj.weight")
    v_proj_w = state_dict.get(f"{prefix}v_proj.weight")
    in_proj_w = state_dict.get(f"{prefix}multihead_attn.in_proj_weight")
    in_proj_b = state_dict.get(f"{prefix}multihead_attn.in_proj_bias")
    out_proj_w = state_dict.get(f"{prefix}multihead_attn.out_proj.weight")
    out_proj_b = state_dict.get(f"{prefix}multihead_attn.out_proj.bias")
    linear_w = state_dict.get(f"{prefix}linear.weight")
    linear_b = state_dict.get(f"{prefix}linear.bias")
    ln_w = state_dict.get(f"{prefix}layer_norm.weight")
    ln_b = state_dict.get(f"{prefix}layer_norm.bias")
    ln_kv_w = state_dict.get(f"{prefix}layer_norm_kv.weight")
    ln_kv_b = state_dict.get(f"{prefix}layer_norm_kv.bias")

    # Pass through all weights that don't need transformation
    skip = {
        f"{prefix}q_proj.weight",
        f"{prefix}k_proj.weight",
        f"{prefix}v_proj.weight",
        f"{prefix}multihead_attn.in_proj_weight",
        f"{prefix}multihead_attn.in_proj_bias",
        f"{prefix}multihead_attn.out_proj.weight",
        f"{prefix}multihead_attn.out_proj.bias",
        f"{prefix}linear.weight",
        f"{prefix}linear.bias",
        f"{prefix}layer_norm.weight",
        f"{prefix}layer_norm.bias",
        f"{prefix}layer_norm_kv.weight",
        f"{prefix}layer_norm_kv.bias",
    }
    result = {k: v for k, v in state_dict.items() if k not in skip}

    # Re-add layer norms and extra linear under correct names
    if ln_w is not None:
        result[f"{prefix}layer_norm.weight"] = ln_w
    if ln_b is not None:
        result[f"{prefix}layer_norm.bias"] = ln_b
    if ln_kv_w is not None:
        result[f"{prefix}layer_norm_kv.weight"] = ln_kv_w
    if ln_kv_b is not None:
        result[f"{prefix}layer_norm_kv.bias"] = ln_kv_b
    if linear_w is not None:
        result[f"{prefix}linear.weight"] = linear_w
    if linear_b is not None:
        result[f"{prefix}linear.bias"] = linear_b

    # Fuse QKV: fused_weight = mha_slice @ proj_weight
    if q_proj_w is not None and in_proj_w is not None:
        head_dim = q_proj_w.shape[0]
        result[f"{prefix}q_proj.weight"] = in_proj_w[0:head_dim] @ q_proj_w
        result[f"{prefix}k_proj.weight"] = in_proj_w[head_dim : 2 * head_dim] @ k_proj_w
        result[f"{prefix}v_proj.weight"] = in_proj_w[2 * head_dim : 3 * head_dim] @ v_proj_w
        if in_proj_b is not None:
            result[f"{prefix}q_proj.bias"] = in_proj_b[0:head_dim]
            result[f"{prefix}k_proj.bias"] = in_proj_b[head_dim : 2 * head_dim]
            result[f"{prefix}v_proj.bias"] = in_proj_b[2 * head_dim : 3 * head_dim]

    if out_proj_w is not None:
        result[f"{prefix}out_proj.weight"] = out_proj_w
    if out_proj_b is not None:
        result[f"{prefix}out_proj.bias"] = out_proj_b

    return result


def _split_expert_weights(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Split AriaExperts batched weight tensors into per-expert Linear weights.

    HF stores expert weights as batched tensors (no individual expert modules):
        model.layers.{i}.mlp.experts.fc1.weight: (num_experts, hidden, inter*2)
        model.layers.{i}.mlp.experts.fc2.weight: (num_experts, inter, hidden)

    fc1 encodes gate_proj + up_proj as a combined SwiGLU weight; we split it.
    We also rename router → gate to match TopKGate naming.

    After splitting:
        model.layers.{i}.mlp.gate.weight:              (num_experts, hidden)
        model.layers.{i}.mlp.experts.{j}.gate_proj.weight: (inter, hidden)
        model.layers.{i}.mlp.experts.{j}.up_proj.weight:   (inter, hidden)
        model.layers.{i}.mlp.experts.{j}.down_proj.weight: (hidden, inter)
    """
    result: dict[str, torch.Tensor] = {}
    fc1_by_layer: dict[str, torch.Tensor] = {}
    fc2_by_layer: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if ".mlp.router.weight" in key:
            # Rename router → gate to match TopKGate weight attribute name
            result[key.replace(".mlp.router.weight", ".mlp.gate.weight")] = value
        elif ".mlp.experts.fc1.weight" in key:
            fc1_by_layer[key] = value
        elif ".mlp.experts.fc2.weight" in key:
            fc2_by_layer[key] = value
        else:
            result[key] = value

    # Split per-expert from batched fc1/fc2
    for fc1_key, fc1 in fc1_by_layer.items():
        # fc1 shape: (num_experts, hidden_size, inter * 2)
        # Linear weight convention: (out, in) → we need to transpose
        base = fc1_key.replace(".fc1.weight", "")
        fc2_key = fc1_key.replace(".fc1.weight", ".fc2.weight")
        fc2 = fc2_by_layer.get(fc2_key)

        num_experts = fc1.shape[0]
        inter_size = fc1.shape[2] // 2  # gate + up are concatenated

        for j in range(num_experts):
            # fc1[j]: (hidden, inter*2) — combined gate+up in SwiGLU
            # Transpose to (inter*2, hidden) then split → gate: (inter, hidden)
            expert_fc1 = fc1[j].T  # (inter*2, hidden)
            result[f"{base}.{j}.gate_proj.weight"] = expert_fc1[:inter_size]
            result[f"{base}.{j}.up_proj.weight"] = expert_fc1[inter_size:]

            if fc2 is not None:
                # fc2[j]: (inter, hidden) — transpose to (hidden, inter)
                result[f"{base}.{j}.down_proj.weight"] = fc2[j].T

    return result
