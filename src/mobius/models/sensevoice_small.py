# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SenseVoiceSmall: CTC-based speech recognition with language control.

Architecture:
  - Query embeddings: 4 prepended tokens (language, event, emo, textnorm)
    looked up from a learned embedding table (16 entries x input_dim).
  - Audio encoder: SenseVoiceEncoderSmall — 3 stacks of SANM layers:
    encoders0 (1 layer, input_dim→hidden), encoders (N-1 layers),
    tp_encoders (M refinement layers). No temporal pooling.
  - CTC head: Linear(hidden_dim, vocab_size) → LogSoftmax

The model takes LFR-processed fbank features (560-dim) and a language ID
integer, prepends query tokens, encodes, and produces CTC log-probabilities.

Reference: https://github.com/FunAudioLLM/SenseVoice
HuggingFace class: SenseVoiceSmall
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components._common import (
    Embedding,
    LayerNorm,
    Linear,
)
from mobius.components._sanm_attention import SANMEncoderLayer


def _sinusoidal_position_embedding(max_positions: int, d_model: int) -> np.ndarray:
    """Compute sinusoidal positional embeddings (1-indexed, like FunASR).

    Uses log-timescale increments. Layout: [sin, cos] per position.
    Positions are 1-indexed: arange(1, max_positions+1).
    """
    channels = d_model
    log_timescale_increment = np.log(10000.0) / (channels // 2 - 1)
    inv_timescales = np.exp(
        -log_timescale_increment * np.arange(channels // 2, dtype=np.float32)
    )
    positions = np.arange(1, max_positions + 1, dtype=np.float32)
    scaled_time = positions[:, np.newaxis] * inv_timescales[np.newaxis, :]
    pe = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return pe.astype(np.float32)


class SenseVoiceSmallModel(nn.Module):
    """SenseVoiceSmall: CTC encoder-only ASR with language control.

    The model prepends 4 query tokens (language, event, emo, textnorm)
    to the input features before encoding. The encoder is the same
    SenseVoiceEncoderSmall used in Fun-ASR-Nano: 3 stacks of SANM layers
    with no temporal pooling.

    Inputs:
        input_features: ``(batch, time, input_dim)`` LFR fbank features
        language_id: ``(batch, 1)`` integer language ID (0=auto, 3=zh, ...)

    Output:
        logits: ``(batch, time + 4, vocab_size)`` CTC log-probabilities

    Weight prefixes (HuggingFace → ONNX)::

        encoder.encoders0.N.*  → encoder.encoders0.N.*
        encoder.encoders.N.*   → encoder.encoders.N.*
        encoder.tp_encoders.N.* → encoder.tp_encoders.N.*
        encoder.after_norm.*   → encoder.after_norm.*
        encoder.tp_norm.*      → encoder.tp_norm.*
        ctc.ctc_lo.*           → ctc_head.*
        embed.weight           → query_embed.weight
    """

    default_task: str = "audio-ctc"
    category: str = "Speech-to-Text"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        audio = config.audio
        assert audio is not None

        input_size = audio.input_size or 560
        hidden_size = audio.attention_dim or 512
        n_heads = audio.attention_heads or 4
        ffn_dim = audio.linear_units or 2048
        kernel_size = audio.kernel_size or 11
        num_blocks = audio.num_blocks or 50
        tp_blocks = audio.tp_num_blocks or 20
        vocab_size = config.vocab_size

        self._hidden_size = hidden_size
        self._input_size = input_size
        self.config = config

        # Query embedding: language(7) + lid_dict(6) + textnorm_dict(2) = 16
        # Indices: 0=auto, 3=zh, 4=en, 7=yue, 11=ja, 12=ko, 13=nospeech,
        #          14=withitn, 15=woitn; 1=event_query, 2=emo_query
        num_query_tokens = 16
        self.query_embed = Embedding(num_query_tokens, input_size)

        # Sinusoidal positional encoding (precomputed, input_dim)
        max_positions = 6000
        pe_data = _sinusoidal_position_embedding(max_positions, input_size)
        self.positional_embedding = nn.Parameter(
            [max_positions, input_size],
            name="positional_embedding",
            data=ir.tensor(pe_data),
        )

        # Encoder stacks (same as FunASRAudioEncoder, without adaptor)
        # Stack 1: 1 layer projecting input_size → hidden_size
        self.encoder = _SenseVoiceEncoder(
            input_size,
            hidden_size,
            n_heads,
            ffn_dim,
            kernel_size,
            num_blocks,
            tp_blocks,
        )

        # CTC head: Linear(hidden_size, vocab_size)
        self.ctc_head = Linear(hidden_size, vocab_size, bias=True)

    def forward(
        self,
        op: OpBuilder,
        input_features: ir.Value,
        language_id: ir.Value,
    ) -> ir.Value:
        """Encode audio features with language query and produce CTC logits.

        Args:
            input_features: ``(batch, time, input_dim)`` LFR fbank
            language_id: ``(batch, 1)`` language ID integer

        Returns:
            logits: ``(batch, time + 4, vocab_size)`` CTC log-probabilities
        """
        # Prepend 4 query tokens: [language, event, emo, textnorm]
        # language_id: (batch, 1) → embed → (batch, 1, input_dim)
        language_query = self.query_embed(op, language_id)

        # Fixed query indices for event(1), emo(2), textnorm(15=woitn)
        event_emo_ids = op.Constant(value_ints=[1, 2])
        event_emo_query = self.query_embed(op, event_emo_ids)  # (2, input_dim)
        # Expand to (batch, 2, input_dim)
        batch_size = op.Shape(input_features, start=0, end=1)
        target_shape_2 = op.Concat(
            batch_size, op.Constant(value_ints=[2, self._input_size]), axis=0
        )
        event_emo_query = op.Expand(op.Unsqueeze(event_emo_query, [0]), target_shape_2)

        # textnorm query (woitn=15 by default)
        textnorm_ids = op.Constant(value_ints=[15])
        textnorm_query = self.query_embed(op, textnorm_ids)  # (1, input_dim)
        target_shape_1 = op.Concat(
            batch_size, op.Constant(value_ints=[1, self._input_size]), axis=0
        )
        textnorm_query = op.Expand(op.Unsqueeze(textnorm_query, [0]), target_shape_1)

        # Prepend: [textnorm, language, event, emo] + input_features
        # FunASR order: textnorm prepended first, then [language, event, emo]
        hidden_states = op.Concat(textnorm_query, input_features, axis=1)
        hidden_states = op.Concat(language_query, event_emo_query, hidden_states, axis=1)

        # Scale by sqrt(hidden_size)
        scale = float(self._hidden_size**0.5)
        hidden_states = op.Mul(
            hidden_states,
            op.CastLike(op.Constant(value_float=scale), hidden_states),
        )

        # Add sinusoidal positional encoding
        seq_len = op.Shape(hidden_states, start=1, end=2)
        pe_slice = op.Slice(
            self.positional_embedding,
            op.Constant(value_ints=[0]),
            seq_len,
            op.Constant(value_ints=[0]),
        )
        pe_slice = op.CastLike(pe_slice, hidden_states)
        hidden_states = op.Add(hidden_states, pe_slice)

        # Encode
        hidden_states = self.encoder(op, hidden_states)

        # CTC head: logits + log_softmax
        logits = self.ctc_head(op, hidden_states)
        logits = op.LogSoftmax(logits, axis=-1)

        return logits

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace weight names to ONNX module structure.

        HF SenseVoiceSmall weight name layout::

            encoder.encoders0.N.*   → encoder.encoders0.N.*
            encoder.encoders.N.*    → encoder.encoders.N.*
            encoder.tp_encoders.N.* → encoder.tp_encoders.N.*
            encoder.after_norm.*    → encoder.after_norm.*
            encoder.tp_norm.*       → encoder.tp_norm.*
            ctc.ctc_lo.*            → ctc_head.*
            embed.weight            → query_embed.weight
        """
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # CTC head: ctc.ctc_lo.* → ctc_head.*
            if key.startswith("ctc.ctc_lo."):
                inner = key[len("ctc.ctc_lo.") :]
                cleaned[f"ctc_head.{inner}"] = value
                continue

            # Query embedding: embed.weight → query_embed.weight
            if key == "embed.weight":
                cleaned["query_embed.weight"] = value
                continue

            # Encoder weights pass through unchanged
            if key.startswith("encoder."):
                cleaned[key] = value
                continue

            # Pass through unknown keys
            cleaned[key] = value

        return cleaned


class _SenseVoiceEncoder(nn.Module):
    """SenseVoiceEncoderSmall: 3-stack SANM encoder.

    Shared between SenseVoiceSmall (CTC) and Fun-ASR-Nano (LLM decoder).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_heads: int,
        ffn_dim: int,
        kernel_size: int,
        num_blocks: int,
        tp_blocks: int,
    ):
        super().__init__()
        # Stack 1: 1 layer projecting input_size → hidden_size
        self.encoders0 = nn.ModuleList(
            [SANMEncoderLayer(input_size, hidden_size, n_heads, ffn_dim, kernel_size)]
        )
        # Stack 2: (num_blocks - 1) layers at hidden_size
        self.encoders = nn.ModuleList(
            [
                SANMEncoderLayer(
                    hidden_size,
                    hidden_size,
                    n_heads,
                    ffn_dim,
                    kernel_size,
                )
                for _ in range(num_blocks - 1)
            ]
        )
        self.after_norm = LayerNorm(hidden_size)

        # Stack 3: tp_blocks refinement layers (no temporal pooling)
        self.tp_encoders = nn.ModuleList(
            [
                SANMEncoderLayer(
                    hidden_size,
                    hidden_size,
                    n_heads,
                    ffn_dim,
                    kernel_size,
                )
                for _ in range(tp_blocks)
            ]
        )
        self.tp_norm = LayerNorm(hidden_size)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        """Encode through all 3 stacks.

        Args:
            hidden_states: ``(batch, seq_len, dim)``

        Returns:
            ``(batch, seq_len, hidden_size)``
        """
        for layer in self.encoders0:
            hidden_states = layer(op, hidden_states)

        for layer in self.encoders:
            hidden_states = layer(op, hidden_states)

        hidden_states = self.after_norm(op, hidden_states)

        for layer in self.tp_encoders:
            hidden_states = layer(op, hidden_states)

        hidden_states = self.tp_norm(op, hidden_states)
        return hidden_states
