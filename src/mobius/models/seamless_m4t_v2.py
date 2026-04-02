# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""SeamlessM4T v2 text-to-text encoder-decoder model.

Architecture:
    encoder: _SeamlessM4Tv2TextEncoder
        Scaled word embedding (x sqrt(hidden_size)) + sinusoidal positional
        embeddings + N x EncoderBlock (pre-norm, ReLU FFN) + LayerNorm
    decoder: _SeamlessM4Tv2TextDecoder
        Scaled word embedding + sinusoidal positional embeddings + N x
        DecoderBlock (self-attn + cross-attn + FFN, all pre-norm) +
        LayerNorm + lm_head (tied to encoder embed_tokens)

Differs from BART in:
  - Embeddings are scaled by sqrt(hidden_size) before adding positions
  - No layernorm_embedding before the transformer blocks
  - A final layer_norm is applied after all blocks
  - Positional embeddings are sinusoidal (pre-computed buffer), not learned
  - Separate FFN dims per encoder/decoder (encoder_ffn_dim, decoder_ffn_dim)
  - ReLU activation (BART defaults to GELU)

HuggingFace reference: SeamlessM4Tv2ForTextToText (model_type='seamless_m4t_v2')

Weight prefixes:
    model.shared.weight                          → shared embedding
    model.text_encoder.*                         → encoder.*
    model.text_decoder.*                         → decoder.*
    model.text_encoder.embed_positions.weights   → encoder.embed_positions.weight
    model.text_decoder.embed_positions.weights   → decoder.embed_positions.weight
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import SeamlessM4Tv2Config
from mobius.components._activations import ACT2FN
from mobius.components._common import Embedding, LayerNorm, Linear
from mobius.components._encoder_decoder_attention import EncoderDecoderAttention

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# Encoder and Decoder Blocks
# ---------------------------------------------------------------------------


class _SeamlessM4Tv2EncoderBlock(nn.Module):
    """Pre-norm encoder block using encoder_ffn_dim for the feed-forward layer.

    Matches HuggingFace SeamlessM4Tv2EncoderLayer: LayerNorm is applied BEFORE
    attention/FFN (pre-norm), and the residual is added AFTER.  Weight names
    follow HF exactly: self_attn_layer_norm, ffn_layer_norm.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.self_attn = EncoderDecoderAttention(config)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc1 = Linear(config.hidden_size, config.encoder_ffn_dim)
        self.fc2 = Linear(config.encoder_ffn_dim, config.hidden_size)
        self.ffn_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._act_fn = ACT2FN[config.hidden_act]

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # Pre-norm self-attention: norm → attn → add residual
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(op, hidden_states)
        hidden_states, _ = self.self_attn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm FFN: norm → ffn → add residual
        residual = hidden_states
        hidden_states = self.ffn_layer_norm(op, hidden_states)
        hidden_states = self.fc1(op, hidden_states)
        hidden_states = self._act_fn(op, hidden_states)
        hidden_states = self.fc2(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states


class _SeamlessM4Tv2DecoderBlock(nn.Module):
    """Pre-norm decoder block (self-attn + cross-attn + FFN) using decoder_ffn_dim.

    Matches HuggingFace SeamlessM4Tv2DecoderLayer: LayerNorm applied BEFORE each
    sub-layer (pre-norm).  Weight names follow HF exactly: self_attn_layer_norm,
    cross_attention, cross_attention_layer_norm, ffn_layer_norm.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.self_attn = EncoderDecoderAttention(config, is_causal=True)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attention = EncoderDecoderAttention(config)
        self.cross_attention_layer_norm = LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.fc1 = Linear(config.hidden_size, config.decoder_ffn_dim)
        self.fc2 = Linear(config.decoder_ffn_dim, config.hidden_size)
        self.ffn_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        past_key_value: tuple | None = None,
        cross_past_key_value: ir.Value | None = None,
    ):
        # Pre-norm causal self-attention (with KV cache)
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(op, hidden_states)
        hidden_states, self_kv = self.self_attn(
            op, hidden_states, past_key_value=past_key_value
        )
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm cross-attention to encoder output
        residual = hidden_states
        hidden_states = self.cross_attention_layer_norm(op, hidden_states)
        hidden_states, cross_kv = self.cross_attention(
            op,
            hidden_states,
            key_value_states=encoder_hidden_states,
            past_key_value=cross_past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.ffn_layer_norm(op, hidden_states)
        hidden_states = self.fc1(op, hidden_states)
        hidden_states = self._act_fn(op, hidden_states)
        hidden_states = self.fc2(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, self_kv, cross_kv


# ---------------------------------------------------------------------------
# Text Encoder and Decoder
# ---------------------------------------------------------------------------


class _SeamlessM4Tv2TextEncoder(nn.Module):
    """SeamlessM4T v2 text encoder.

    Applies scaled token embeddings + sinusoidal positional embeddings, passes
    through N encoder blocks, then applies a final LayerNorm.  Unlike BART,
    there is no layernorm_embedding before the blocks; instead the layer_norm
    at the end matches HF's text_encoder.layer_norm.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        # Sinusoidal positional embeddings: size is (max_position_embeddings + 2)
        # to accommodate the HF offset of 2 (positions start at index 2).
        self.embed_positions = Embedding(
            config.max_position_embeddings + 2, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [_SeamlessM4Tv2EncoderBlock(config) for _ in range(config.num_hidden_layers)]
        )
        # Applied after all transformer blocks (not before, unlike BART)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Pre-compute scale: sqrt(hidden_size) when scale_embedding=True
        self._embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None = None,
    ) -> ir.Value:
        # (batch, seq_len, hidden_size) — scaled by sqrt(hidden_size)
        inputs_embeds = self.embed_tokens(op, input_ids)
        inputs_embeds = op.Mul(inputs_embeds, self._embed_scale)

        # Sinusoidal position IDs with offset 2: [2, 3, ..., seq_len + 1]
        seq_len = op.Shape(input_ids, start=1, end=2)
        position_ids = op.Range(
            op.Constant(value_int=2),
            op.Add(seq_len, op.Constant(value_int=2)),
            op.Constant(value_int=1),
        )
        position_ids = op.Cast(position_ids, to=7)  # INT64
        position_ids = op.Unsqueeze(position_ids, [0])
        position_embeds = self.embed_positions(op, position_ids)  # (1, seq_len, hidden)

        # (batch, seq_len, hidden_size)
        hidden_states = op.Add(inputs_embeds, position_embeds)

        for layer in self.layers:
            hidden_states = layer(op, hidden_states)

        # Final layer norm (matches HF text_encoder.layer_norm)
        hidden_states = self.layer_norm(op, hidden_states)
        return hidden_states


class _SeamlessM4Tv2TextDecoder(nn.Module):
    """SeamlessM4T v2 text decoder.

    Scaled token embeddings + sinusoidal positional embeddings + N decoder
    blocks (self-attn + cross-attn + FFN) + final LayerNorm + lm_head.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        num_decoder_layers = config.num_decoder_layers or config.num_hidden_layers
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = Embedding(
            config.max_position_embeddings + 2, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [_SeamlessM4Tv2DecoderBlock(config) for _ in range(num_decoder_layers)]
        )
        # Applied after all decoder blocks (matches HF text_decoder.layer_norm)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self._embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        encoder_hidden_states: ir.Value,
        position_ids: ir.Value | None = None,
        attention_mask: ir.Value | None = None,
        past_key_values: list | None = None,
        cross_past_key_values: ir.Value | None = None,
    ):
        # (batch, seq_len, hidden_size) — scaled
        inputs_embeds = self.embed_tokens(op, input_ids)
        inputs_embeds = op.Mul(inputs_embeds, self._embed_scale)

        # Position IDs with offset 2, accounting for past KV cache length
        if position_ids is None:
            seq_len = op.Shape(input_ids, start=1, end=2)
            if past_key_values is not None:
                # past_key shape: (batch, num_heads, past_seq_len, head_dim)
                past_len = op.Shape(past_key_values[0][0], start=2, end=3)
            else:
                past_len = op.Constant(value_int=0)
            start = op.Add(past_len, op.Constant(value_int=2))
            end = op.Add(start, seq_len)
            position_ids = op.Range(start, end, op.Constant(value_int=1))
            position_ids = op.Cast(position_ids, to=7)  # INT64
            position_ids = op.Unsqueeze(position_ids, [0])

        position_embeds = self.embed_positions(op, position_ids)
        hidden_states = op.Add(inputs_embeds, position_embeds)

        past_kvs = past_key_values or [None] * len(self.layers)
        cross_past_kvs = cross_past_key_values or [None] * len(self.layers)
        present_self_kvs = []
        present_cross_kvs = []

        for layer, past_kv, cross_kv in zip(self.layers, past_kvs, cross_past_kvs):
            hidden_states, self_kv, cross_kv_out = layer(
                op,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                past_key_value=past_kv,
                cross_past_key_value=cross_kv,
            )
            present_self_kvs.append(self_kv)
            present_cross_kvs.append(cross_kv_out)

        # Final layer norm + lm_head
        hidden_states = self.layer_norm(op, hidden_states)
        logits = self.lm_head(op, hidden_states)
        return logits, present_self_kvs, present_cross_kvs


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class SeamlessM4Tv2Model(nn.Module):
    """SeamlessM4T v2 text-to-text encoder-decoder for multilingual translation.

    Produces a ModelPackage with separate encoder and decoder ONNX graphs via
    Seq2SeqTask.  Uses a BART-like architecture but with scaled embeddings,
    sinusoidal positional embeddings, ReLU FFN activations, and separate
    encoder/decoder FFN dimensions.

    HuggingFace: SeamlessM4Tv2ForTextToText (model_type='seamless_m4t_v2')
    """

    default_task = "seq2seq"
    category = "encoder-decoder"

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.config = config
        self.encoder = _SeamlessM4Tv2TextEncoder(config)
        self.decoder = _SeamlessM4Tv2TextDecoder(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_dict: dict[str, torch.Tensor] = {}
        shared: torch.Tensor | None = None

        for name, tensor in state_dict.items():
            # Strip "model." prefix present in SeamlessM4Tv2ForTextToText state dict
            if name.startswith("model."):
                name = name[len("model.") :]

            if name == "shared.weight":
                shared = tensor
                continue

            # Remap text_encoder.* → encoder.* and text_decoder.* → decoder.*
            if name.startswith("text_encoder."):
                name = "encoder." + name[len("text_encoder.") :]
            elif name.startswith("text_decoder."):
                name = "decoder." + name[len("text_decoder.") :]

            # Sinusoidal positional embedding buffer is named "weights" in HF
            # (SeamlessM4Tv2SinusoidalPositionalEmbedding registers a buffer called
            # "weights"), but our Embedding stores it as "weight" (nn.Parameter).
            name = name.replace(".embed_positions.weights", ".embed_positions.weight")

            # lm_head is tied to shared.weight in HF; we handle it below
            if name == "lm_head.weight":
                continue

            new_dict[name] = tensor

        # Shared embedding → encoder and decoder embed_tokens
        if shared is not None:
            new_dict.setdefault("encoder.embed_tokens.weight", shared)
            new_dict.setdefault("decoder.embed_tokens.weight", shared)

        # Tie lm_head to encoder embed_tokens (weight tying)
        embed = new_dict.get("encoder.embed_tokens.weight")
        if embed is not None:
            new_dict.setdefault("decoder.lm_head.weight", embed)

        return new_dict
