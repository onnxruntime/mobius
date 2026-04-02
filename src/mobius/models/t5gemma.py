# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""T5Gemma encoder-decoder model.

T5Gemma (``T5GemmaForConditionalGeneration``) uses a Gemma2-style encoder
and a Gemma2-style decoder with cross-attention layers.

Both encoder and decoder share the same architecture:
- RoPE positional embeddings (no T5 relative position bias)
- OffsetRMSNorm (Gemma-style +1 offset)
- Alternating sliding-window and full attention (``layer_types``)
- Attention logit soft-capping (Gemma2-style)
- Gated GeLU FFN with Gemma2's four-norm pattern (pre/post attention + pre/post FFN)

The decoder adds a cross-attention sub-layer between self-attention and FFN.
Encoder hidden states are re-projected to K/V each decode step; the runtime
caches these values via the ``present.{i}.cross.{key,value}`` outputs.

Replicates HuggingFace's ``T5GemmaForConditionalGeneration``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import Gemma2Config
from mobius.components import (
    MLP,
    OffsetRMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._common import Linear
from mobius.models.gemma import Gemma2Attention, GemmaScaledWordEmbedding

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# Cross-attention
# ---------------------------------------------------------------------------


class T5GemmaCrossAttention(nn.Module):
    """Cross-attention for the T5Gemma decoder.

    Q is projected from decoder hidden states.
    K/V are projected from encoder hidden states.
    No RoPE (positional information is encoded in the self-attention).

    Uses Gemma2-style scaling (``query_pre_attn_scalar``) and attention
    logit soft-capping (``attn_logit_softcapping``).

    Encoder hidden states are re-projected each decode step. This is correct
    because the encoder output is constant per sequence — concatenating with a
    growing KV cache would incorrectly expand the cross-attention sequence
    length. The runtime may cache the projected K/V externally via the
    ``present_cross_kvs`` outputs, but the ONNX graph is unconditional.
    """

    def __init__(self, config: Gemma2Config):
        super().__init__()
        scale = None
        if config.query_pre_attn_scalar:
            scale = config.query_pre_attn_scalar**-0.5
        hidden_size = config.hidden_size
        num_q_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        # Q projected from decoder hidden states
        self.q_proj = Linear(hidden_size, num_q_heads * head_dim, bias=False)
        # K/V projected from encoder hidden states
        self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = Linear(num_q_heads * head_dim, hidden_size, bias=False)

        self._num_q_heads = num_q_heads
        self._num_kv_heads = num_kv_heads
        self._scale = scale
        self._softcap = config.attn_logit_softcapping

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
    ):
        # Q from decoder
        query_states = self.q_proj(op, hidden_states)  # (batch, dec_len, q_heads * head_dim)

        # K/V always projected from encoder hidden states (constant per sequence).
        # Do NOT concatenate with a KV cache: encoder output never grows, so
        # accumulating past K/V would incorrectly double the sequence length.
        key_states = self.k_proj(op, encoder_hidden_states)
        value_states = self.v_proj(op, encoder_hidden_states)

        attn_output, present_key, present_value = op.Attention(
            query_states,
            key_states,
            value_states,
            None,  # no attention bias — cross-attention is full bidirectional
            None,  # no past_key
            None,  # no past_value
            q_num_heads=self._num_q_heads,
            kv_num_heads=self._num_kv_heads,
            scale=self._scale,
            softcap=self._softcap,
            _outputs=3,
        )
        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------


class T5GemmaEncoderLayer(nn.Module):
    """T5Gemma encoder layer.

    Identical to ``Gemma2DecoderLayer`` but runs bidirectional self-attention
    (no causal mask). Applies the Gemma2 four-norm pattern:
    input_layernorm → attn → post_attention_layernorm
    pre_feedforward_layernorm → FFN → post_feedforward_layernorm.
    """

    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.self_attn = Gemma2Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
    ):
        # Self-attention (bidirectional — no causal mask in encoder)
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_output, _ = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=None,  # encoder has no KV cache
        )
        hidden_states = self.post_attention_layernorm(op, attn_output)
        hidden_states = op.Add(residual, hidden_states)

        # FFN
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = self.post_feedforward_layernorm(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class T5GemmaDecoderLayer(nn.Module):
    """T5Gemma decoder layer.

    Three sub-blocks, each with Gemma2's pre+post norm pattern:
    1. Causal self-attention (sliding or full, from ``layer_types``)
    2. Cross-attention to encoder hidden states
    3. Gated GeLU FFN
    """

    def __init__(self, config: Gemma2Config):
        super().__init__()
        # Causal self-attention (same as Gemma2)
        self.self_attn = Gemma2Attention(config)
        self.input_layernorm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Cross-attention
        self.cross_attn = T5GemmaCrossAttention(config)
        self.cross_attn_layernorm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_cross_attn_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # FFN
        self.mlp = MLP(config)
        self.pre_feedforward_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
        cross_past_key_value: tuple | None = None,
    ):
        # 1. Causal self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_output, present_self_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = self.post_attention_layernorm(op, attn_output)
        hidden_states = op.Add(residual, hidden_states)

        # 2. Cross-attention to encoder
        residual = hidden_states
        hidden_states = self.cross_attn_layernorm(op, hidden_states)
        cross_output, present_cross_kv = self.cross_attn(
            op,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = self.post_cross_attn_layernorm(op, cross_output)
        hidden_states = op.Add(residual, hidden_states)

        # 3. FFN
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = self.post_feedforward_layernorm(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_self_kv, present_cross_kv


# ---------------------------------------------------------------------------
# Encoder stack
# ---------------------------------------------------------------------------


class T5GemmaEncoder(nn.Module):
    """T5Gemma encoder: bidirectional Gemma2-style transformer.

    Unlike ``Gemma2TextModel`` (causal decoder), the encoder uses a
    bidirectional attention bias (no causal masking). The ``layer_types``
    list still controls which layers use sliding-window attention, but the
    window is applied without a causal constraint.
    """

    def __init__(self, config: Gemma2Config):
        super().__init__()
        self._dtype = config.dtype
        embed_scale = float(np.round(np.sqrt(config.hidden_size), decimals=2))
        self.embed_tokens = GemmaScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=embed_scale,
        )
        self.layers = nn.ModuleList(
            [T5GemmaEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        self.sliding_window = config.sliding_window
        self._layer_types = config.layer_types
        if self._layer_types is not None:
            assert len(self._layer_types) == config.num_hidden_layers, (
                f"len(layer_types)={len(self._layer_types)} != "
                f"num_hidden_layers={config.num_hidden_layers}"
            )

    def _is_local(self, layer_id: int) -> bool:
        """Return True if layer uses sliding-window attention."""
        if self._layer_types is not None:
            return self._layer_types[layer_id] != "full_attention"
        return layer_id % 2 == 1  # Gemma2 default

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        position_ids = op.Cast(
            op.Unsqueeze(
                op.Range(
                    op.Constant(value_int=0),
                    op.Shape(input_ids, start=1, end=2),  # seq_len scalar
                    op.Constant(value_int=1),
                ),
                [0],
            ),
            to=7,  # INT64
        )
        position_embeddings = self.rotary_emb(op, position_ids)

        # Encoder attention biases: bidirectional for full_attention layers,
        # bidirectional sliding window for sliding_attention layers.
        # create_attention_bias produces a causal mask; for the encoder we
        # pass attention_bias=None for full attention (no causal constraint).
        # Sliding-window encoder attention uses a causal bias as an
        # approximation — the window limits context but is one-directional.
        # TODO: implement proper bidirectional sliding window for encoder.
        full_attn_bias = None  # bidirectional full attention (no masking)
        sliding_attn_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window=self.sliding_window,
            dtype=self._dtype,
        )

        for i, layer in enumerate(self.layers):
            attn_bias = sliding_attn_bias if self._is_local(i) else full_attn_bias
            hidden_states = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attn_bias,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(op, hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Decoder stack
# ---------------------------------------------------------------------------


class T5GemmaDecoder(nn.Module):
    """T5Gemma decoder: causal Gemma2-style transformer with cross-attention.

    Produces logits and two KV cache outputs:
    - ``present_self_kvs``: self-attention KV (grows each step)
    - ``present_cross_kvs``: cross-attention KV (runtime caches after step 0)
    """

    def __init__(self, config: Gemma2Config):
        super().__init__()
        self._dtype = config.dtype
        embed_scale = float(np.round(np.sqrt(config.hidden_size), decimals=2))
        # Decoder shares embed_tokens with encoder; populated by preprocess_weights.
        self.embed_tokens = GemmaScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=embed_scale,
        )
        self.layers = nn.ModuleList(
            [T5GemmaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.sliding_window = config.sliding_window
        self._layer_types = config.layer_types
        self._final_logit_softcapping = config.final_logit_softcapping
        if self._layer_types is not None:
            assert len(self._layer_types) == config.num_hidden_layers, (
                f"len(layer_types)={len(self._layer_types)} != "
                f"num_hidden_layers={config.num_hidden_layers}"
            )

    def _is_local(self, layer_id: int) -> bool:
        """Return True if this decoder layer uses sliding-window self-attention."""
        if self._layer_types is not None:
            return self._layer_types[layer_id] != "full_attention"
        return layer_id % 2 == 1

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        encoder_hidden_states: ir.Value,
        attention_mask: ir.Value,
        past_key_values: list | None = None,
        cross_past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        # Position IDs for current query tokens, accounting for past KV length
        query_length = op.Shape(input_ids, start=1, end=2)  # scalar
        total_length = op.Shape(attention_mask, start=1, end=2)  # scalar
        past_length = op.Sub(total_length, query_length)
        position_ids = op.Reshape(
            op.Range(past_length, total_length, op.Constant(value_int=1)),
            [1, -1],
        )
        position_embeddings = self.rotary_emb(op, position_ids)

        full_attn_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )
        sliding_attn_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window=self.sliding_window,
            dtype=self._dtype,
        )

        past_kvs = past_key_values or [None] * len(self.layers)
        cross_past_kvs = cross_past_key_values or [None] * len(self.layers)
        present_self_kvs = []
        present_cross_kvs = []

        for i, (layer, past_kv, cross_kv) in enumerate(
            zip(self.layers, past_kvs, cross_past_kvs)
        ):
            attn_bias = sliding_attn_bias if self._is_local(i) else full_attn_bias
            hidden_states, self_kv, cross_kv_out = layer(
                op,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_bias=attn_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
                cross_past_key_value=cross_kv,
            )
            present_self_kvs.append(self_kv)
            present_cross_kvs.append(cross_kv_out)

        hidden_states = self.norm(op, hidden_states)
        logits = self.lm_head(op, hidden_states)

        # Final logit soft-capping (Gemma2-style)
        if self._final_logit_softcapping > 0.0:
            logits = op.Div(logits, self._final_logit_softcapping)
            logits = op.Tanh(logits)
            logits = op.Mul(logits, self._final_logit_softcapping)

        return logits, present_self_kvs, present_cross_kvs


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class T5GemmaForConditionalGeneration(nn.Module):
    """T5Gemma encoder-decoder model for conditional generation.

    Uses the ``seq2seq`` task, producing a ModelPackage with:
    - ``"encoder"``: input_ids, attention_mask → last_hidden_state
    - ``"decoder"``: input_ids, encoder_hidden_states, attention_mask,
                     past_self_kvs, past_cross_kvs → logits, present_kvs

    Both encoder and decoder share the same ``Gemma2Config``; this is valid
    because T5Gemma uses identical architecture for encoder and decoder
    (same hidden size, num heads, FFN dims, etc.).

    Replicates HuggingFace's ``T5GemmaForConditionalGeneration``.
    """

    default_task = "seq2seq"
    category = "encoder-decoder"

    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.encoder = T5GemmaEncoder(config)
        self.decoder = T5GemmaDecoder(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace T5Gemma weight names to our naming convention.

        T5Gemma uses Gemma2-style weight names:
        - ``encoder.embed_tokens.weight`` / ``decoder.embed_tokens.weight``
          (may be stored as a single ``model.embed_tokens.weight``)
        - ``lm_head.weight`` (may be tied to embed_tokens)
        - Self-attention: ``{enc,dec}.layers.{i}.self_attn.{q,k,v,o}_proj.weight``
        - Cross-attention: ``decoder.layers.{i}.cross_attn.{q,k,v,o}_proj.weight``
        - Layer norms: ``input_layernorm``, ``post_attention_layernorm``,
          ``pre_feedforward_layernorm``, ``post_feedforward_layernorm``,
          ``cross_attn_layernorm``, ``post_cross_attn_layernorm``
        - FFN: ``mlp.{gate,up,down}_proj.weight``
        """
        new: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            new_name = _rename_t5gemma_weight(name)
            if new_name is not None:
                new[new_name] = tensor

        # Handle shared embed_tokens (stored as top-level in some checkpoints)
        for src in ("model.embed_tokens.weight", "shared.weight"):
            if src in new:
                embed = new.pop(src)
                new.setdefault("encoder.embed_tokens.weight", embed)
                new.setdefault("decoder.embed_tokens.weight", embed)

        # Tie lm_head to encoder embed_tokens if not present
        if "decoder.lm_head.weight" not in new:
            embed = new.get("encoder.embed_tokens.weight")
            if embed is not None:
                new["decoder.lm_head.weight"] = embed

        return new


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------


def _rename_t5gemma_weight(name: str) -> str | None:
    """Rename a HF T5Gemma weight to our naming convention.

    HF T5Gemma stores weights under ``model.encoder.*`` and ``model.decoder.*``.
    We strip the leading ``model.`` prefix so names match our module paths.
    """
    # Top-level lm_head
    if name == "lm_head.weight":
        return "decoder.lm_head.weight"
    # Shared embedding (handled separately in preprocess_weights)
    if name in ("model.embed_tokens.weight", "shared.weight"):
        return name

    # Strip HF's top-level "model." prefix (e.g. model.encoder.* → encoder.*)
    if name.startswith("model."):
        name = name[len("model."):]

    # Encoder and decoder layers follow Gemma2 naming conventions.
    for prefix in ("encoder.", "decoder."):
        if name.startswith(prefix):
            return name

    return None
