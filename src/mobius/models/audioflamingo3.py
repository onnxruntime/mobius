# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""AudioFlamingo-3: Whisper audio encoder + Qwen2 language decoder.

Architecture:
  - Audio encoder: Whisper-Large encoder (Conv1d x2 → sinusoidal PE → 32
    encoder layers → LayerNorm) + 2-layer MLP projector (1280 → 3584)
  - Text decoder: Qwen2-7B (28 layers, GQA 28h/4kv, RMSNorm, standard 1D RoPE)
  - Fusion: Audio features replace ``audio_token_id`` positions in text embeddings

Three ONNX models are exported via :class:`~mobius.tasks.AudioLanguageTask`:

1. ``audio_encoder`` — mel (batch, 128, 3000) → audio_features (batch, 1500, 3584)
2. ``embedding``     — input_ids + audio_features → inputs_embeds
3. ``decoder``       — inputs_embeds + position_ids → logits + KV cache

Reference: https://huggingface.co/nvidia/audio-flamingo-3-hf
HuggingFace class: AudioFlamingo3ForConditionalGeneration
"""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._common import (
    Embedding,
    LayerNorm,
    Linear,
    create_attention_bias,
)
from mobius.components._decoder import DecoderLayer
from mobius.components._rms_norm import RMSNorm
from mobius.components._rotary_embedding import initialize_rope
from mobius.components._whisper import Conv1d, WhisperEncoderLayer


def _sinusoidal_positional_embedding(max_positions: int, d_model: int) -> np.ndarray:
    """Whisper-style sinusoidal positional embeddings.

    Returns ``[max_positions, d_model]`` float32 array matching
    ``WhisperPositionalEmbedding`` in HuggingFace transformers.
    """
    position = np.arange(max_positions, dtype=np.float32)[:, np.newaxis]
    half_dim = d_model // 2
    div_term = np.exp(np.arange(half_dim, dtype=np.float32) * -(math.log(10000.0) / half_dim))
    pe = np.zeros((max_positions, d_model), dtype=np.float32)
    pe[:, :half_dim] = np.sin(position * div_term)
    pe[:, half_dim:] = np.cos(position * div_term)
    return pe


class _AudioFlamingo3Projector(nn.Module):
    """2-layer MLP projector for AudioFlamingo-3.

    Maps Whisper encoder output to text hidden size:
        Linear(d_model → text_hidden, bias=True) + GELU + Linear(text_hidden → text_hidden, bias=True)

    HF weight names: ``multi_modal_projector.linear_1.*``, ``multi_modal_projector.linear_2.*``
    (remapped to ``projector.linear_1.*``, ``projector.linear_2.*`` by ``preprocess_weights``).
    """

    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.linear_1 = Linear(d_model, output_dim, bias=True)
        self.linear_2 = Linear(output_dim, output_dim, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        # (batch, seq_len, d_model) → (batch, seq_len, output_dim)
        x = self.linear_1(op, x)
        x = op.Gelu(x)
        x = self.linear_2(op, x)
        return x


class AudioFlamingo3Encoder(nn.Module):
    """AudioFlamingo-3 audio encoder: mel spectrogram → projected features.

    Architecture mirrors Whisper-Large (identical dims) followed by a
    2-layer MLP projector that maps audio hidden states to text hidden size.

    Inputs:
        input_features: ``(batch, num_mel_bins, audio_seq_len)`` mel spectrogram

    Output:
        audio_features: ``(batch * audio_seq_len // 2, text_hidden_size)``
        Flattened to 2D to match SpeechLanguageTask embedding contract.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        audio = config.audio
        assert audio is not None, "AudioConfig required for AudioFlamingo3Encoder"

        d_model = audio.d_model or 1280
        num_mel_bins = audio.num_mel_bins or 128
        encoder_layers = audio.encoder_layers or 32
        encoder_heads = audio.encoder_attention_heads or 20
        encoder_ffn = audio.encoder_ffn_dim or 5120
        max_source_positions = audio.max_source_positions or 1500

        # eps=1e-5 matches Whisper's LayerNorm epsilon
        eps = 1e-5

        # Conv1d downsampling (stride=2 on second conv halves the sequence)
        self.conv1 = Conv1d(num_mel_bins, d_model, kernel_size=3, padding=1)
        self.conv2 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        # Frozen sinusoidal positional embeddings (not learned, stored as a constant)
        pe_data = _sinusoidal_positional_embedding(max_source_positions, d_model)
        self.embed_positions = nn.Parameter(
            [max_source_positions, d_model],
            name="embed_positions.weight",
            data=ir.tensor(pe_data),
        )

        # 32 bidirectional encoder layers (Whisper-Large dims)
        self.layers = nn.ModuleList(
            [
                WhisperEncoderLayer(d_model, encoder_heads, encoder_ffn, eps=eps)
                for _ in range(encoder_layers)
            ]
        )
        self.layer_norm = LayerNorm(d_model, eps=eps)

        # 2-layer MLP projector: audio hidden → text hidden
        self.projector = _AudioFlamingo3Projector(d_model, config.hidden_size)

    def forward(self, op: builder.OpBuilder, input_features: ir.Value) -> ir.Value:
        # input_features: (batch, num_mel_bins, audio_seq_len)

        # Conv1d pair + GELU: (batch, d_model, audio_seq_len // 2)
        hidden_states = op.Gelu(self.conv1(op, input_features))
        hidden_states = op.Gelu(self.conv2(op, hidden_states))

        # Transpose: (batch, audio_seq_len // 2, d_model)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])

        # Slice positional embeddings to actual sequence length
        # (PE is max_source_positions long; audio may be shorter in practice)
        seq_len = op.Shape(hidden_states, start=1, end=2)
        pe_slice = op.Slice(
            self.embed_positions,
            op.Constant(value_ints=[0]),
            seq_len,
            op.Constant(value_ints=[0]),
        )
        hidden_states = op.Add(hidden_states, pe_slice)

        # 32 bidirectional encoder layers
        for layer in self.layers:
            hidden_states = layer(op, hidden_states)

        # Final LayerNorm: (batch, audio_seq_len // 2, d_model)
        hidden_states = self.layer_norm(op, hidden_states)

        # 2-layer MLP projector: (batch, audio_seq_len // 2, text_hidden)
        hidden_states = self.projector(op, hidden_states)

        # Flatten to 2D (batch * audio_seq_len // 2, text_hidden) to match
        # SpeechLanguageTask._build_embedding which expects (num_audio_tokens, hidden)
        hidden_dim = op.Shape(hidden_states, start=2, end=3)
        hidden_states = op.Reshape(
            hidden_states, op.Concat(op.Constant(value_ints=[-1]), hidden_dim, axis=0)
        )

        return hidden_states


class AudioFlamingo3EmbeddingModel(nn.Module):
    """AudioFlamingo-3 embedding model: fuses text and audio embeddings.

    Replaces ``audio_token_id`` positions in the text embedding with
    audio features from the audio encoder (using Gather + CumSum + Where).

    Inputs:
        input_ids:      ``(batch, seq_len)`` token IDs
        audio_features: ``(num_audio_tokens, text_hidden_size)`` projected features (2D)

    Output:
        inputs_embeds: ``(batch, seq_len, hidden_size)``
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        # 151669 is the default audio_token_id in Qwen2's tokenizer for AudioFlamingo3
        self._audio_token_id = config.audio.audio_token_id if config.audio else 151669

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        audio_features: ir.Value,
    ) -> ir.Value:
        """Fuse text embeddings with audio features.

        Audio features are inserted at positions where ``input_ids == audio_token_id``.
        Uses cumulative sum on the audio mask to index into the flattened
        ``audio_features`` tensor.

        ``audio_features`` is expected to be 2D: ``(total_audio_tokens, hidden_size)``.
        """
        # Text embeddings: (batch, seq_len, hidden_size)
        inputs_embeds = self.embed_tokens(op, input_ids)

        # audio_features: (total_audio_tokens, hidden_size)
        # Get feature dim for zero-padding row construction
        feature_dim = op.Shape(audio_features, start=1, end=2)  # hidden_size

        # Create mask: True where input_ids == audio_token_id
        audio_token = op.Constant(value_int=self._audio_token_id)
        is_audio = op.Equal(input_ids, audio_token)  # (batch, seq_len)
        is_audio_3d = op.Unsqueeze(is_audio, [-1])  # (batch, seq_len, 1)

        # Prepend a zero row so gather index 0 maps to zero padding
        zero_row = op.ConstantOfShape(
            op.Concat(op.Constant(value_ints=[1]), feature_dim, axis=0),
            value=ir.tensor(np.zeros(1, dtype=np.float32)),
        )
        # padded: (total_audio_tokens + 1, hidden_size)
        padded_features = op.Concat(zero_row, audio_features, axis=0)

        # CumSum of the flattened boolean mask gives 1-based indices
        # (0 = zero-padding row for non-audio positions)
        is_audio_int = op.Cast(is_audio, to=7)  # INT64
        flat_mask = op.Reshape(is_audio_int, op.Constant(value_ints=[-1]))
        flat_indices = op.CumSum(flat_mask, op.Constant(value_int=0))
        # Zero out non-audio positions so they gather from the padding row
        flat_indices = op.Mul(flat_indices, flat_mask)
        indices = op.Reshape(flat_indices, op.Shape(input_ids))  # (batch, seq_len)

        # Gather audio features at computed indices
        gathered = op.Gather(padded_features, indices, axis=0)

        # Where: replace audio positions with gathered features
        inputs_embeds = op.Where(is_audio_3d, gathered, inputs_embeds)
        return inputs_embeds


class AudioFlamingo3DecoderModel(nn.Module):
    """AudioFlamingo-3 text decoder: inputs_embeds → logits + KV cache.

    Standard Qwen2-7B decoder with GQA (28h/4kv), RMSNorm, and
    standard 1D RoPE. Takes ``inputs_embeds`` (fused text+audio).

    Inputs:
        inputs_embeds:   ``(batch, seq_len, hidden_size)``
        attention_mask:  ``(batch, past_seq_len + seq_len)``
        position_ids:    ``(batch, seq_len)`` standard 1D RoPE
        past_key_values: list of ``(key, value)`` tensors per layer

    Outputs:
        logits:           ``(batch, seq_len, vocab_size)``
        present_key_values: list of updated ``(key, value)`` tensors
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ) -> tuple[ir.Value, list]:
        hidden_states = inputs_embeds
        # position_ids: (batch, seq_len) → standard 1D RoPE embeddings
        position_embeddings = self.rotary_emb(op, position_ids)

        attention_bias = create_attention_bias(
            op,
            input_ids=inputs_embeds,
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

        # Post-norm and LM head: (batch, seq_len, vocab_size)
        hidden_states = self.norm(op, hidden_states)
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values


class AudioFlamingo3ForConditionalGeneration(nn.Module):
    """AudioFlamingo-3 composite model for audio understanding.

    Exports three ONNX models via :class:`~mobius.tasks.AudioLanguageTask`:

    - ``audio_encoder``: Whisper-Large encoder + 2-layer MLP projector
    - ``embedding``: text embedding + audio token fusion
    - ``decoder``: Qwen2-7B decoder with KV cache

    HF weight layout:
        ``audio_tower.*``              → ``audio_tower.*`` (encoder, direct match)
        ``multi_modal_projector.*``    → ``audio_tower.projector.*``
        ``language_model.model.embed_tokens.*`` → ``embedding.embed_tokens.*``
        ``language_model.model.layers.*`` → ``decoder.layers.*``
        ``language_model.model.norm.*``   → ``decoder.norm.*``
        ``language_model.lm_head.*``      → ``decoder.lm_head.*``

    Reference: https://huggingface.co/nvidia/audio-flamingo-3-hf
    HuggingFace class: AudioFlamingo3ForConditionalGeneration
    """

    default_task: str = "audio-language"
    category: str = "Audio"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.audio_tower = AudioFlamingo3Encoder(config)
        self.embedding = AudioFlamingo3EmbeddingModel(config)
        self.decoder = AudioFlamingo3DecoderModel(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ) -> tuple[ir.Value, list]:
        """Text-generation forward (audio features fused externally)."""
        inputs_embeds = self.embedding.embed_tokens(op, input_ids)
        return self.decoder(
            op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace weight names to ONNX module structure.

        Remapping rules:
        - ``audio_tower.*``                           → ``audio_tower.*``  (direct, encoder matches)
        - ``multi_modal_projector.linear_1.*``        → ``audio_tower.projector.linear_1.*``
        - ``multi_modal_projector.linear_2.*``        → ``audio_tower.projector.linear_2.*``
        - ``language_model.model.embed_tokens.*``     → ``embedding.embed_tokens.*``
        - ``language_model.model.{layers,norm,rotary_emb}.*`` → ``decoder.{layers,norm,rotary_emb}.*``
        - ``language_model.lm_head.*``                → ``decoder.lm_head.*``
        """
        cleaned: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            # audio_tower.* → audio_tower.*  (direct, weight names align)
            if key.startswith("audio_tower."):
                cleaned[key] = value
                continue

            # multi_modal_projector.* → audio_tower.projector.*
            if key.startswith("multi_modal_projector."):
                inner = key[len("multi_modal_projector.") :]
                cleaned[f"audio_tower.projector.{inner}"] = value
                continue

            # language_model.lm_head.* → decoder.lm_head.*
            if key.startswith("language_model.lm_head."):
                inner = key[len("language_model.lm_head.") :]
                cleaned[f"decoder.lm_head.{inner}"] = value
                continue

            # language_model.model.* → route to embedding or decoder
            if key.startswith("language_model.model."):
                inner = key[len("language_model.model.") :]

                if inner.startswith("embed_tokens."):
                    cleaned[f"embedding.{inner}"] = value
                    continue

                if inner.startswith(("layers.", "norm.", "rotary_emb.")):
                    cleaned[f"decoder.{inner}"] = value
                    continue

            # Pass through any other keys unchanged
            cleaned[key] = value

        # Weight tying: lm_head shares embed_tokens weights in Qwen2
        embed_key = "embedding.embed_tokens.weight"
        lm_key = "decoder.lm_head.weight"
        if self.config.tie_word_embeddings:
            if embed_key in cleaned and lm_key not in cleaned:
                cleaned[lm_key] = cleaned[embed_key]
            elif lm_key in cleaned and embed_key not in cleaned:
                cleaned[embed_key] = cleaned[lm_key]

        return cleaned
