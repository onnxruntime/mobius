# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Moonshine lightweight ASR encoder-decoder model.

Implements the Moonshine speech-to-text architecture
(UsefulSensors/moonshine-tiny) as two separate traceable modules:

- ``MoonshineEncoder``: raw waveform → encoder hidden states
  (Conv1d frontend + RoPE transformer encoder)
- ``MoonshineDecoder``: decoder input IDs + encoder output → logits + KV cache
  (RoPE self-attention + cross-attention transformer decoder)

``MoonshineForConditionalGeneration`` holds both and provides
``preprocess_weights()`` for HuggingFace weight name mapping.

Replicates ``MoonshineForConditionalGeneration`` from HuggingFace transformers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import MoonshineConfig
from mobius.components._activations import get_activation
from mobius.components._common import Embedding, GroupNorm, LayerNormNoBias, Linear
from mobius.components._rotary_embedding import apply_rotary_pos_emb, initialize_rope
from mobius.components._whisper import Conv1d

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# Conv1d without bias (used for the first conv layer in the encoder)
# ---------------------------------------------------------------------------


class _Conv1dNoBias(nn.Module):
    """1D convolution layer without bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        self.weight = nn.Parameter([out_channels, in_channels, kernel_size])
        self._kernel_shape = [kernel_size]
        self._strides = [stride]

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: (batch, in_channels, seq_len)
        return op.Conv(
            x,
            self.weight,
            kernel_shape=self._kernel_shape,
            strides=self._strides,
        )


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class _MoonshineSelfAttention(nn.Module):
    """Self-attention with RoPE for Moonshine encoder and decoder.

    Applies rotary position embeddings to Q and K before the ONNX Attention
    op.  Supports optional KV cache for autoregressive decoding.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        rotary_dim: Number of dimensions to apply RoPE to (partial RoPE).
        is_causal: Whether to use causal masking (decoder self-attention).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        rotary_dim: int,
        is_causal: bool = False,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._rotary_dim = rotary_dim
        self._scale = float(head_dim) ** -0.5
        self._is_causal = is_causal
        self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        # Project Q, K, V: (B, S, H*D)
        q = self.q_proj(op, hidden_states)
        k = self.k_proj(op, hidden_states)
        v = self.v_proj(op, hidden_states)

        # Apply RoPE to Q and K (operates on 3D: B, S, H*D)
        q = apply_rotary_pos_emb(
            op,
            q,
            position_embeddings,
            self._num_heads,
            rotary_embedding_dim=self._rotary_dim,
        )
        k = apply_rotary_pos_emb(
            op,
            k,
            position_embeddings,
            self._num_heads,
            rotary_embedding_dim=self._rotary_dim,
        )

        # KV cache handling
        if past_key_value is not None:
            past_k = past_key_value[0]
            past_v = past_key_value[1]
        else:
            past_k = None
            past_v = None

        # ONNX Attention op with built-in KV cache and causal mask
        attn_output, present_key, present_value = op.Attention(
            q,
            k,
            v,
            None,
            past_k,
            past_v,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            scale=self._scale,
            is_causal=1 if self._is_causal else 0,
            _outputs=3,
        )
        return self.o_proj(op, attn_output), (present_key, present_value)


class _MoonshineCrossAttention(nn.Module):
    """Cross-attention for the Moonshine decoder.

    Projects Q from decoder hidden states and K/V from encoder output.
    No RoPE, no KV cache (encoder output is re-projected each step).
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._scale = float(head_dim) ** -0.5
        self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
    ):
        # Q from decoder, K/V from encoder
        q = self.q_proj(op, hidden_states)  # (B, S_dec, H*D)
        k = self.k_proj(op, encoder_hidden_states)  # (B, S_enc, H*D)
        v = self.v_proj(op, encoder_hidden_states)  # (B, S_enc, H*D)

        # Cross-attention: not causal, no KV cache
        attn_output, _, _ = op.Attention(
            q,
            k,
            v,
            None,
            None,
            None,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            scale=self._scale,
            is_causal=0,
            _outputs=3,
        )
        return self.o_proj(op, attn_output)


# ---------------------------------------------------------------------------
# MLP modules
# ---------------------------------------------------------------------------


class _MoonshineEncoderMLP(nn.Module):
    """Two-layer FC MLP for encoder: fc1 → GELU → fc2.

    Matches HF weight names ``mlp.fc1`` / ``mlp.fc2``.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = self.fc1(op, x)
        x = op.Gelu(x)
        return self.fc2(op, x)


class _MoonshineDecoderMLP(nn.Module):
    """Gated SiLU MLP for decoder: fc1 → split → silu(gate)*value → fc2.

    ``fc1`` projects to ``2 * intermediate_size``, which is split in half.
    The first half is gated with SiLU, element-wise multiplied with the
    second half, then projected back by ``fc2``.

    Matches HF weight names ``mlp.fc1`` / ``mlp.fc2``.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = Linear(hidden_size, 2 * intermediate_size, bias=True)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=True)
        self._act_fn = get_activation("silu")

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = self.fc1(op, x)  # (B, S, 2 * intermediate)
        # Split into gate and value halves
        gate, value = op.Split(x, num_outputs=2, axis=-1, _outputs=2)
        gate = self._act_fn(op, gate)  # SiLU activation on gate
        return self.fc2(op, op.Mul(gate, value))  # (B, S, hidden)


# ---------------------------------------------------------------------------
# Encoder / decoder layers
# ---------------------------------------------------------------------------


class _MoonshineEncoderLayer(nn.Module):
    """Pre-norm Moonshine encoder layer.

    Structure: input_layernorm → self_attn (RoPE) → residual
               → post_attention_layernorm → MLP (GELU) → residual
    """

    def __init__(self, config: MoonshineConfig):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = config.head_dim
        rotary_dim = int(head_dim * config.partial_rotary_factor)
        intermediate_size = config.intermediate_size
        eps = config.rms_norm_eps

        self.self_attn = _MoonshineSelfAttention(
            hidden_size,
            num_heads,
            head_dim,
            rotary_dim,
            is_causal=False,
        )
        self.input_layernorm = LayerNormNoBias(hidden_size, eps=eps)
        self.post_attention_layernorm = LayerNormNoBias(hidden_size, eps=eps)
        self.mlp = _MoonshineEncoderMLP(hidden_size, intermediate_size)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple,
    ):
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, _ = self.self_attn(
            op,
            hidden_states,
            position_embeddings,
        )
        hidden_states = op.Add(residual, hidden_states)

        # FFN with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states


class _MoonshineDecoderLayer(nn.Module):
    """Pre-norm Moonshine decoder layer with three layer norms.

    Structure: input_layernorm → self_attn (RoPE, causal, KV cache) → residual
               → post_attention_layernorm → cross_attn (no RoPE) → residual
               → final_layernorm → gated MLP (SiLU) → residual
    """

    def __init__(self, config: MoonshineConfig):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = config.head_dim
        rotary_dim = int(head_dim * config.partial_rotary_factor)
        intermediate_size = config.intermediate_size
        eps = config.rms_norm_eps

        self.self_attn = _MoonshineSelfAttention(
            hidden_size,
            num_heads,
            head_dim,
            rotary_dim,
            is_causal=True,
        )
        self.encoder_attn = _MoonshineCrossAttention(
            hidden_size,
            num_heads,
            head_dim,
        )
        self.input_layernorm = LayerNormNoBias(hidden_size, eps=eps)
        self.post_attention_layernorm = LayerNormNoBias(hidden_size, eps=eps)
        self.final_layernorm = LayerNormNoBias(hidden_size, eps=eps)
        self.mlp = _MoonshineDecoderMLP(hidden_size, intermediate_size)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        # Self-attention with RoPE and KV cache
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present_kv = self.self_attn(
            op,
            hidden_states,
            position_embeddings,
            past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        # Cross-attention to encoder output (no RoPE, no KV cache)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.encoder_attn(
            op,
            hidden_states,
            encoder_hidden_states,
        )
        hidden_states = op.Add(residual, hidden_states)

        # Gated MLP
        residual = hidden_states
        hidden_states = self.final_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


# ---------------------------------------------------------------------------
# Encoder / decoder top-level modules
# ---------------------------------------------------------------------------


class MoonshineEncoder(nn.Module):
    """Moonshine encoder: raw waveform → encoder hidden states.

    Architecture:
        Conv1d frontend (conv1 → tanh → groupnorm → conv2 → gelu → conv3 → gelu)
        → transpose → RoPE transformer encoder layers → LayerNorm

    Input:  ``(batch, audio_length)`` raw waveform
    Output: ``(batch, T_enc, hidden_size)`` encoder hidden states
    """

    def __init__(self, config: MoonshineConfig):
        super().__init__()
        hidden_size = config.hidden_size
        encoder_layers = config.encoder_layers or config.num_hidden_layers
        eps = config.rms_norm_eps

        # Conv frontend
        self.conv1 = _Conv1dNoBias(1, hidden_size, kernel_size=127, stride=64)
        self.groupnorm = GroupNorm(1, hidden_size)
        self.conv2 = Conv1d(hidden_size, 2 * hidden_size, kernel_size=7, stride=3)
        self.conv3 = Conv1d(2 * hidden_size, hidden_size, kernel_size=3, stride=2)

        # Transformer encoder
        self.layers = nn.ModuleList(
            [_MoonshineEncoderLayer(config) for _ in range(encoder_layers)]
        )
        self.layer_norm = LayerNormNoBias(hidden_size, eps=eps)

        # RoPE
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_values: ir.Value,
    ):
        # Conv frontend: (B, audio_length) → (B, 1, audio_length) → convs
        x = op.Unsqueeze(input_values, [1])  # (B, 1, T)
        x = op.Tanh(self.conv1(op, x))  # (B, hidden, T1)
        x = self.groupnorm(op, x)  # (B, hidden, T1)
        x = op.Gelu(self.conv2(op, x))  # (B, 2*hidden, T2)
        x = op.Gelu(self.conv3(op, x))  # (B, hidden, T3)

        # Transpose to sequence-first: (B, hidden, T_enc) → (B, T_enc, hidden)
        x = op.Transpose(x, perm=[0, 2, 1])

        # Compute position_ids from encoder sequence length
        seq_len = op.Shape(x, start=1, end=2)
        position_ids = op.Range(
            op.Constant(value_int=0),
            seq_len,
            op.Constant(value_int=1),
        )
        position_ids = op.Cast(position_ids, to=7)  # INT64
        position_ids = op.Unsqueeze(position_ids, [0])  # (1, T_enc)
        position_embeddings = self.rotary_emb(op, position_ids)

        # Encoder layers
        for layer in self.layers:
            x = layer(op, x, position_embeddings)

        return self.layer_norm(op, x)


class MoonshineDecoder(nn.Module):
    """Moonshine decoder: token IDs + encoder output → logits + KV cache.

    Architecture:
        token embed → RoPE self-attention + cross-attention decoder layers
        → LayerNorm → proj_out

    The decoder uses causal self-attention with KV caching and
    cross-attention to encoder hidden states (re-projected each step).
    """

    def __init__(self, config: MoonshineConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.embed_tokens = Embedding(
            config.vocab_size,
            hidden_size,
            config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [_MoonshineDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LayerNormNoBias(hidden_size, eps=config.rms_norm_eps)

        # RoPE (decoder has its own RoPE instance)
        self.rotary_emb = initialize_rope(config)

        # Output projection (shared reference set by parent)
        self.proj_out = Linear(hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        decoder_input_ids: ir.Value,
        encoder_hidden_states: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        # Token embeddings
        x = self.embed_tokens(op, decoder_input_ids)  # (B, S, hidden)

        # RoPE position embeddings
        position_embeddings = self.rotary_emb(op, position_ids)

        # Decoder layers
        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            x, present_kv = layer(
                op,
                x,
                encoder_hidden_states,
                position_embeddings,
                past_kv,
            )
            present_key_values.append(present_kv)

        x = self.norm(op, x)
        logits = self.proj_out(op, x)
        return logits, present_key_values


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class _MoonshineModel(nn.Module):
    """Inner model holding encoder + decoder.

    Matches HF ``model.encoder`` / ``model.decoder`` naming structure.
    """

    def __init__(self, config: MoonshineConfig):
        super().__init__()
        self.encoder = MoonshineEncoder(config)
        self.decoder = MoonshineDecoder(config)


class MoonshineForConditionalGeneration(nn.Module):
    """Moonshine encoder-decoder model for automatic speech recognition.

    This class holds both ``MoonshineEncoder`` and ``MoonshineDecoder``
    as sub-modules via ``_MoonshineModel``.  Use
    ``MoonshineSpeechToTextTask.build()`` to trace them into separate
    ONNX models.

    Inputs (encoder):
        ``input_values``: Raw audio waveform ``(batch, audio_length)``
    Outputs (encoder):
        ``encoder_hidden_states``: ``(batch, T_enc, hidden_size)``

    Inputs (decoder):
        ``decoder_input_ids``: ``(batch, seq_len)``
        ``encoder_hidden_states``: ``(batch, T_enc, hidden_size)``
        ``position_ids``: ``(batch, seq_len)``
        ``past_key_values``: list of KV cache tuples
    Outputs (decoder):
        ``logits``: ``(batch, seq_len, vocab_size)``
        ``present_key_values``: updated KV cache tuples

    Replicates ``MoonshineForConditionalGeneration`` from HuggingFace.
    """

    default_task: str = "moonshine-speech-to-text"
    category: str = "Speech-to-Text"
    config_class: type = MoonshineConfig

    def __init__(self, config: MoonshineConfig):
        super().__init__()
        self.config = config
        self.model = _MoonshineModel(config)
        # Shared reference to decoder's output projection
        self.proj_out = self.model.decoder.proj_out

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace weight names to our module structure.

        HF weights use ``model.encoder.X`` / ``model.decoder.X``.
        Our ONNX graphs use ``encoder.X`` / ``decoder.X`` (no ``model.``
        prefix) because the task traces ``module.model.encoder`` directly.

        Also strips rotary embedding inverse frequency weights which are
        recomputed from config.
        """
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if "rotary_emb" in key:
                continue  # skip inv_freq — computed from config
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[len("model.") :]
            remapped[new_key] = value
        return remapped
