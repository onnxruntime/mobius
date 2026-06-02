# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LFM2 hybrid ShortConv + Attention causal language model.

LFM2 interleaves ShortConv (gated causal depthwise Conv1d) layers with
Transformer attention layers. Both layer types include a SiLU-gated MLP
(SwiGLU).

Layer type selection via ``layer_types`` in config:
    ``"conv"`` → ShortConv: in_proj → B*x gating → causal conv → C gating → out_proj
    ``"full_attention"`` → standard GQA with QK norm and RoPE

Architecture per layer:
    conv layers: operator_norm → ShortConv → residual → ffn_norm → MLP → residual
    attn layers: operator_norm → Attention → residual → ffn_norm → MLP → residual

State per layer:
    Conv: conv_state (batch, hidden_size, kernel_size-1)
    Attention: standard KV cache (key + value)

HuggingFace weight names (conv layer)::

    model.layers.N.conv.conv.weight      → layers.N.conv.conv_weight
    model.layers.N.conv.in_proj.weight   → layers.N.conv.in_proj.weight
    model.layers.N.conv.out_proj.weight  → layers.N.conv.out_proj.weight
    model.layers.N.operator_norm.weight  → layers.N.operator_norm.weight
    model.layers.N.ffn_norm.weight       → layers.N.ffn_norm.weight
    model.layers.N.feed_forward.w1.weight → layers.N.feed_forward.gate_proj.weight
    model.layers.N.feed_forward.w3.weight → layers.N.feed_forward.up_proj.weight
    model.layers.N.feed_forward.w2.weight → layers.N.feed_forward.down_proj.weight

HuggingFace weight names (attention layer)::

    model.layers.N.self_attn.q_proj.weight     → layers.N.self_attn.q_proj.weight
    model.layers.N.self_attn.k_proj.weight     → layers.N.self_attn.k_proj.weight
    model.layers.N.self_attn.v_proj.weight     → layers.N.self_attn.v_proj.weight
    model.layers.N.self_attn.out_proj.weight   → layers.N.self_attn.o_proj.weight
    model.layers.N.self_attn.q_layernorm.weight → layers.N.self_attn.q_norm.weight
    model.layers.N.self_attn.k_layernorm.weight → layers.N.self_attn.k_norm.weight

HuggingFace reference: ``Lfm2ForCausalLM``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import Lfm2Config
from mobius._weight_utils import tie_word_embeddings
from mobius.components import (
    MLP,
    Attention,
    Embedding,
    Linear,
    RMSNorm,
    ShortConv,
    create_attention_bias,
    initialize_rope,
)

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


class Lfm2ConvDecoderLayer(nn.Module):
    """LFM2 ShortConv layer: operator_norm → ShortConv → residual → ffn_norm → MLP.

    Args:
        config: LFM2 architecture config.
    """

    def __init__(self, config: Lfm2Config):
        super().__init__()
        self.conv = ShortConv(
            hidden_size=config.hidden_size,
            kernel_size=config.short_conv_kernel,
            bias=config.short_conv_bias,
        )
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = MLP(config)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (conv_state,)).

        attention_bias and position_embeddings are unused by conv layers
        but accepted for uniform interface with attention layers.
        """
        del attention_bias, position_embeddings  # unused

        # Pre-norm → ShortConv → residual
        residual = hidden_states
        hidden_states = self.operator_norm(op, hidden_states)

        conv_state = past_key_value[0] if past_key_value is not None else None
        conv_out, new_conv_state = self.conv(op, hidden_states, conv_state)
        hidden_states = op.Add(residual, conv_out)

        # MLP path with pre-norm
        residual = hidden_states
        hidden_states = self.ffn_norm(op, hidden_states)
        hidden_states = self.feed_forward(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, (new_conv_state,)


class Lfm2AttentionDecoderLayer(nn.Module):
    """LFM2 attention layer: operator_norm → Attention → residual → ffn_norm → MLP.

    Uses GQA with per-head QK norm and RoPE.

    Args:
        config: LFM2 architecture config.
    """

    def __init__(self, config: Lfm2Config):
        super().__init__()
        self.self_attn = Attention(config)
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = MLP(config)

    def forward(
        self,
        op: OpBuilder,
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

        # MLP path with pre-norm
        residual = hidden_states
        hidden_states = self.ffn_norm(op, hidden_states)
        hidden_states = self.feed_forward(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class _Lfm2TextModel(nn.Module):
    """LFM2 text backbone: embedding -> N x (ShortConv|Attention) -> norm.

    Layer types are selected based on ``config.layer_types``:
        ``"conv"`` → Lfm2ConvDecoderLayer
        ``"full_attention"`` → Lfm2AttentionDecoderLayer
    """

    def __init__(self, config: Lfm2Config):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        layer_types = config.layer_types or []
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            ltype = layer_types[i] if i < len(layer_types) else "full_attention"
            if ltype == "conv":
                self.layers.append(Lfm2ConvDecoderLayer(config))
            else:
                self.layers.append(Lfm2AttentionDecoderLayer(config))

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
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


class Lfm2CausalLMModel(nn.Module):
    """LFM2 hybrid ShortConv+Attention causal language model.

    Uses ``HybridCausalLMTask`` with mixed ``"conv"`` and
    ``"full_attention"`` layer types for the cache.

    HuggingFace reference: ``Lfm2ForCausalLM``.
    """

    default_task: str = "hybrid-text-generation"
    category: str = "Hybrid Conv+Attention"
    config_class: type = Lfm2Config

    def __init__(self, config: Lfm2Config):
        super().__init__()
        self.config = config
        self.model = _Lfm2TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
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
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace Lfm2ForCausalLM weights to ONNX parameters.

        Handles:
        1. Weight tying (embed_tokens ↔ lm_head)
        2. MLP w1/w3/w2 → gate_proj/up_proj/down_proj rename
        3. Attention out_proj → o_proj rename
        4. QK norm: q_layernorm/k_layernorm → q_norm/k_norm
        5. Conv weight nesting: conv.conv.weight → conv.conv_weight
        """
        if self.config.tie_word_embeddings:
            tie_word_embeddings(state_dict)

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _rename_lfm2_weight(key)
            new_state_dict[new_key] = value

        return new_state_dict


# Regex for layer-level weight keys
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def _rename_lfm2_weight(key: str) -> str:
    """Rename a single HF weight key to match ONNX module structure.

    Global renames:
        model.embed_tokens.weight → model.embed_tokens.weight  (no change)
        model.norm.weight → model.norm.weight  (no change)

    Per-layer renames (within model.layers.N):
        conv.conv.weight → conv.conv_weight  (nested module → parameter)
        feed_forward.w1 → feed_forward.gate_proj
        feed_forward.w3 → feed_forward.up_proj
        feed_forward.w2 → feed_forward.down_proj
        self_attn.out_proj → self_attn.o_proj
        self_attn.q_layernorm → self_attn.q_norm
        self_attn.k_layernorm → self_attn.k_norm
    """
    m = _LAYER_RE.match(key)
    if m is None:
        return key

    idx = m.group(1)
    rest = m.group(2)

    # Conv weight: conv.conv.weight → conv.conv_weight
    rest = rest.replace("conv.conv.weight", "conv.conv_weight")
    rest = rest.replace("conv.conv.bias", "conv.conv_bias")

    # MLP: w1 → gate_proj, w3 → up_proj, w2 → down_proj
    rest = rest.replace("feed_forward.w1.", "feed_forward.gate_proj.")
    rest = rest.replace("feed_forward.w3.", "feed_forward.up_proj.")
    rest = rest.replace("feed_forward.w2.", "feed_forward.down_proj.")

    # Attention output projection: out_proj → o_proj
    rest = rest.replace("self_attn.out_proj.", "self_attn.o_proj.")

    # QK norm: q_layernorm → q_norm, k_layernorm → k_norm
    rest = rest.replace("self_attn.q_layernorm.", "self_attn.q_norm.")
    rest = rest.replace("self_attn.k_layernorm.", "self_attn.k_norm.")

    return f"model.layers.{idx}.{rest}"
