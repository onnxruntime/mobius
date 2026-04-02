# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""LFM2-MoE hybrid ShortConv+Attention model with Mixture-of-Experts FFN.

LFM2-MoE extends LFM2 by replacing the feed-forward MLP in most layers
with a Mixture-of-Experts layer (MoELayer + TopK gate). The first
``num_dense_layers`` layers retain a standard MLP.

Layer type selection:
    ``"conv"`` + index < num_dense_layers  → Lfm2ConvDecoderLayer (dense MLP)
    ``"conv"`` + index >= num_dense_layers → _Lfm2MoeConvDecoderLayer (MoE FFN)
    ``"full_attention"`` + dense → Lfm2AttentionDecoderLayer
    ``"full_attention"`` + moe  → _Lfm2MoeAttentionDecoderLayer

Gate: ``_Lfm2MoeGate`` — TopK routing with optional per-expert bias
(``use_expert_bias=True``) and scaling (``routed_scaling_factor``).

HuggingFace weight renames (on top of base LFM2 renames):
    feed_forward.router.weight                  → feed_forward.gate.weight
    feed_forward.router.e_score_correction_bias → feed_forward.gate.e_score_correction_bias
    feed_forward.experts.N.w1.weight            → feed_forward.experts.N.gate_proj.weight
    feed_forward.experts.N.w3.weight            → feed_forward.experts.N.up_proj.weight
    feed_forward.experts.N.w2.weight            → feed_forward.experts.N.down_proj.weight

HuggingFace reference: ``Lfm2MoeForCausalLM``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import Lfm2MoeConfig
from mobius._weight_utils import tie_word_embeddings
from mobius.components import (
    Attention,
    Embedding,
    Linear,
    RMSNorm,
    ShortConv,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._moe import MoELayer
from mobius.models.lfm2 import (
    Lfm2AttentionDecoderLayer,
    Lfm2ConvDecoderLayer,
)

if TYPE_CHECKING:
    import onnx_ir as ir


class _Lfm2MoeGate(nn.Module):
    """LFM2-MoE routing gate: TopK with optional per-expert bias.

    Computes routing logits via MatMul, optionally adds per-expert bias,
    selects top-k experts, and normalizes weights with softmax.
    """

    def __init__(self, config: Lfm2MoeConfig):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.use_expert_bias = config.use_expert_bias
        self.weight = nn.Parameter([config.num_local_experts, config.hidden_size])
        if config.use_expert_bias:
            self.e_score_correction_bias = nn.Parameter([config.num_local_experts])

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # hidden_states: (batch, seq, hidden_size)
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)  # (batch, seq, num_experts)
        if self.use_expert_bias:
            # Add per-expert bias to routing logits before selection
            router_logits = op.Add(router_logits, self.e_score_correction_bias)
        k = op.Constant(value_ints=[self.top_k])
        routing_weights, selected_experts = op.TopK(router_logits, k, axis=-1, _outputs=2)
        # Normalize selected weights: softmax → sum to 1
        routing_weights = op.Softmax(routing_weights, axis=-1)
        if self.routed_scaling_factor != 1.0:  # noqa: RUF069
            scale = op.CastLike(
                op.Constant(value_float=self.routed_scaling_factor), routing_weights
            )
            routing_weights = op.Mul(routing_weights, scale)
        return routing_weights, selected_experts


class _Lfm2MoeConvDecoderLayer(nn.Module):
    """LFM2 ShortConv layer with MoE FFN.

    Same structure as Lfm2ConvDecoderLayer but replaces the dense MLP
    with a MoELayer (TopK-gated expert ensemble).
    """

    def __init__(self, config: Lfm2MoeConfig):
        super().__init__()
        self.conv = ShortConv(
            hidden_size=config.hidden_size,
            kernel_size=config.short_conv_kernel,
            bias=config.short_conv_bias,
        )
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = MoELayer(config, gate=_Lfm2MoeGate(config))

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (conv_state,))."""
        del attention_bias, position_embeddings  # unused by conv layers

        # Pre-norm → ShortConv → residual
        residual = hidden_states
        hidden_states = self.operator_norm(op, hidden_states)

        conv_state = past_key_value[0] if past_key_value is not None else None
        conv_out, new_conv_state = self.conv(op, hidden_states, conv_state)
        hidden_states = op.Add(residual, conv_out)

        # MoE FFN with pre-norm
        residual = hidden_states
        hidden_states = self.ffn_norm(op, hidden_states)
        hidden_states = self.feed_forward(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, (new_conv_state,)


class _Lfm2MoeAttentionDecoderLayer(nn.Module):
    """LFM2 attention layer with MoE FFN.

    Same structure as Lfm2AttentionDecoderLayer but replaces the dense
    MLP with a MoELayer (TopK-gated expert ensemble).
    """

    def __init__(self, config: Lfm2MoeConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = MoELayer(config, gate=_Lfm2MoeGate(config))

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (key_cache, value_cache))."""
        # Pre-norm → Attention → residual
        residual = hidden_states
        hidden_states = self.operator_norm(op, hidden_states)

        hidden_states, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        # MoE FFN with pre-norm
        residual = hidden_states
        hidden_states = self.ffn_norm(op, hidden_states)
        hidden_states = self.feed_forward(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


class _Lfm2MoeTextModel(nn.Module):
    """LFM2-MoE text backbone: embedding -> N x (ShortConv|Attention with MoE FFN) -> norm.

    Layers 0..num_dense_layers-1 use standard MLP; layers num_dense_layers..N use MoELayer.
    """

    def __init__(self, config: Lfm2MoeConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        layer_types = config.layer_types or []
        num_dense = config.num_dense_layers
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            ltype = layer_types[i] if i < len(layer_types) else "full_attention"
            use_moe = i >= num_dense
            if ltype == "conv":
                layer = (
                    _Lfm2MoeConvDecoderLayer(config)
                    if use_moe
                    else Lfm2ConvDecoderLayer(config)
                )
            else:
                layer = (
                    _Lfm2MoeAttentionDecoderLayer(config)
                    if use_moe
                    else Lfm2AttentionDecoderLayer(config)
                )
            self.layers.append(layer)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value = None,
        attention_mask: ir.Value = None,
        position_ids: ir.Value = None,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value = None,
    ):
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.embed_tokens(op, input_ids)
        )
        position_embeddings = self.rotary_emb(op, position_ids)

        # When inputs_embeds is provided (VL case), use it for sequence length extraction
        seq_tensor = input_ids if input_ids is not None else inputs_embeds
        attention_bias = create_attention_bias(
            op,
            input_ids=seq_tensor,
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

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class Lfm2MoeCausalLMModel(nn.Module):
    """LFM2-MoE hybrid ShortConv+Attention causal language model with MoE FFN.

    Extends LFM2 by replacing the feed-forward MLP in most layers with a
    Mixture-of-Experts layer. Uses ``hybrid-text-generation`` task (same
    mixed conv+attention cache as base LFM2).

    HuggingFace reference: ``Lfm2MoeForCausalLM``.
    """

    default_task: str = "hybrid-text-generation"
    category: str = "Hybrid Conv+Attention"
    config_class: type = Lfm2MoeConfig

    def __init__(self, config: Lfm2MoeConfig):
        super().__init__()
        self.config = config
        self.model = _Lfm2MoeTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value = None,
        attention_mask: ir.Value = None,
        position_ids: ir.Value = None,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace Lfm2MoeForCausalLM weights to ONNX parameters.

        Extends base LFM2 renames with MoE-specific mappings:
        - feed_forward.router.* → feed_forward.gate.*
        - feed_forward.experts.N.w1 → .gate_proj
        - feed_forward.experts.N.w3 → .up_proj
        - feed_forward.experts.N.w2 → .down_proj
        """
        if self.config.tie_word_embeddings:
            tie_word_embeddings(state_dict)

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _rename_lfm2_moe_weight(key)
            new_state_dict[new_key] = value
        return new_state_dict


# Regex for layer-level weight keys
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def _rename_lfm2_moe_weight(key: str) -> str:
    """Rename HF weight keys for LFM2-MoE, extending base LFM2 renames.

    Per-layer renames (on top of base LFM2):
        feed_forward.router.*                → feed_forward.gate.*
        feed_forward.experts.N.w1.*          → feed_forward.experts.N.gate_proj.*
        feed_forward.experts.N.w3.*          → feed_forward.experts.N.up_proj.*
        feed_forward.experts.N.w2.*          → feed_forward.experts.N.down_proj.*
    """
    m = _LAYER_RE.match(key)
    if m is None:
        return key

    idx = m.group(1)
    rest = m.group(2)

    # Apply base LFM2 renames first (conv, MLP, attention, QK norm)
    rest = rest.replace("conv.conv.weight", "conv.conv_weight")
    rest = rest.replace("conv.conv.bias", "conv.conv_bias")
    rest = rest.replace("self_attn.out_proj.", "self_attn.o_proj.")
    rest = rest.replace("self_attn.q_layernorm.", "self_attn.q_norm.")
    rest = rest.replace("self_attn.k_layernorm.", "self_attn.k_norm.")

    # Dense MLP renames (layers 0..num_dense_layers-1)
    rest = rest.replace("feed_forward.w1.", "feed_forward.gate_proj.")
    rest = rest.replace("feed_forward.w3.", "feed_forward.up_proj.")
    rest = rest.replace("feed_forward.w2.", "feed_forward.down_proj.")

    # MoE gate rename: router → gate
    rest = rest.replace("feed_forward.router.", "feed_forward.gate.")

    # MoE expert MLP renames: w1→gate_proj, w3→up_proj, w2→down_proj
    rest = re.sub(
        r"feed_forward\.experts\.(\d+)\.w1\.",
        r"feed_forward.experts.\1.gate_proj.",
        rest,
    )
    rest = re.sub(
        r"feed_forward\.experts\.(\d+)\.w3\.",
        r"feed_forward.experts.\1.up_proj.",
        rest,
    )
    rest = re.sub(
        r"feed_forward\.experts\.(\d+)\.w2\.",
        r"feed_forward.experts.\1.down_proj.",
        rest,
    )

    return f"model.layers.{idx}.{rest}"
