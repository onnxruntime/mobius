# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fun-ASR-Nano: Audio speech recognition with Qwen3-0.6B text decoder.

Architecture:
  - Audio encoder (SenseVoiceEncoderSmall + adaptor): 3 stacks of SANM layers
    followed by a 2-layer MLP + 2 transformer blocks projecting 512→1024
    (LLM hidden dimension). Sequence length is preserved (no temporal pooling).
  - Text decoder: Qwen3-0.6B (reused from existing Qwen3 decoder code).
  - Fusion: Audio features (already LLM-dim) replace audio_token_id positions
    in text embeddings.

The encoder outputs 512-dim features which the built-in adaptor projects to
1024 (the LLM hidden_size). The embedding model receives LLM-dimension
features and performs simple token scatter without further projection.

Reference: https://huggingface.co/FunAudioLLM/Fun-ASR-Nano
HuggingFace class: FunASRForConditionalGeneration
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

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
from mobius.components._sanm_attention import (
    SANMFFN,
    SANMEncoderLayer,
)


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
    # 1-indexed positions (FunASR convention)
    positions = np.arange(1, max_positions + 1, dtype=np.float32)
    scaled_time = positions[:, np.newaxis] * inv_timescales[np.newaxis, :]
    pe = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return pe.astype(np.float32)


# ── Audio Encoder ──────────────────────────────────────────────────────


class FunASRAudioEncoder(nn.Module):
    """Fun-ASR Nano audio encoder: SenseVoiceEncoderSmall + adaptor.

    Three stacks of SANM (Self-Attention with Normalization and Memory)
    encoder layers (sequence length is preserved throughout), followed by
    an adaptor that projects to the LLM hidden dimension:

    1. ``encoders0``: 1 SANM layer projecting input_dim (560) → hidden_dim (512)
    2. ``encoders``: N-1 SANM layers at hidden_dim (512)
    3. ``tp_encoders``: M SANM layers at hidden_dim (512) — refinement, no pooling
    4. ``adaptor``: MLP + transformer blocks projecting 512 → LLM hidden (1024)

    Input: ``(batch, seq_len, input_dim)`` — LFR-processed fbank features
    Output: ``(batch, seq_len, llm_hidden_size)`` — LLM-dimension features
    """

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

        self._hidden_size = hidden_size
        self._input_size = input_size

        # Sinusoidal positional encoding (precomputed, input_dim = input_size)
        max_positions = 6000  # max LFR frames for ~6 min audio
        pe_data = _sinusoidal_position_embedding(max_positions, input_size)
        self.positional_embedding = nn.Parameter(
            [max_positions, input_size],
            name="positional_embedding",
            data=ir.tensor(pe_data),
        )

        # Stack 1: 1 layer projecting input_size → hidden_size
        self.encoders0 = nn.ModuleList(
            [SANMEncoderLayer(input_size, hidden_size, n_heads, ffn_dim, kernel_size)]
        )
        # Stack 2: (num_blocks - 1) layers at hidden_size
        self.encoders = nn.ModuleList(
            [
                SANMEncoderLayer(hidden_size, hidden_size, n_heads, ffn_dim, kernel_size)
                for _ in range(num_blocks - 1)
            ]
        )
        self.after_norm = LayerNorm(hidden_size)

        # Stack 3: tp_blocks layers at hidden_size (refinement, no temporal pooling)
        self.tp_encoders = nn.ModuleList(
            [
                SANMEncoderLayer(hidden_size, hidden_size, n_heads, ffn_dim, kernel_size)
                for _ in range(tp_blocks)
            ]
        )
        self.tp_norm = LayerNorm(hidden_size)

        # Adaptor: projects encoder hidden_dim (512) → LLM hidden_size (1024)
        self.adaptor = FunASRAudioAdaptor(config)

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        """Encode LFR-processed fbank features to LLM-dimension audio embeddings.

        Args:
            input_features: ``(batch, seq_len, input_dim)`` fbank features.

        Returns:
            audio_features: ``(batch, seq_len, llm_hidden_size)``
        """
        # input_features: (batch, seq_len, input_dim)

        # Step 1: Scale input by sqrt(hidden_size) (FunASR convention)
        scale = float(self._hidden_size**0.5)
        hidden_states = op.Mul(
            input_features,
            op.CastLike(op.Constant(value_float=scale), input_features),
        )

        # Step 2: Add sinusoidal positional encoding (input_dim, before projection)
        seq_len = op.Shape(hidden_states, start=1, end=2)
        pe_slice = op.Slice(
            self.positional_embedding,
            op.Constant(value_ints=[0]),
            seq_len,
            op.Constant(value_ints=[0]),
        )
        pe_slice = op.CastLike(pe_slice, hidden_states)
        hidden_states = op.Add(hidden_states, pe_slice)

        # Stack 1: input projection (560 → 512)
        for layer in self.encoders0:
            hidden_states = layer(op, hidden_states)

        # Stack 2: main encoder
        for layer in self.encoders:
            hidden_states = layer(op, hidden_states)

        hidden_states = self.after_norm(op, hidden_states)

        # Stack 3: tp_encoders (additional refinement layers — no temporal pooling)
        for layer in self.tp_encoders:
            hidden_states = layer(op, hidden_states)

        hidden_states = self.tp_norm(op, hidden_states)
        # Project encoder features (512) to LLM dimension (1024) via adaptor
        hidden_states = self.adaptor(op, hidden_states)
        return hidden_states  # (batch, T, llm_hidden_size)


# ── Adaptor Attention ──────────────────────────────────────────────────


class AdaptorAttention(nn.Module):
    """Standard multi-head attention with separate Q/K/V projections.

    Unlike SANMAttention, this uses separate linear projections for Q, K, V
    (not fused) and has no FSMN memory block. Used in the audio adaptor
    transformer blocks.

    Weight names match HuggingFace:
    ``audio_adaptor.blocks.N.self_attn.linear_{q,k,v,out}``
    """

    def __init__(self, hidden_size: int, n_heads: int):
        super().__init__()
        self.linear_q = Linear(hidden_size, hidden_size, bias=True)
        self.linear_k = Linear(hidden_size, hidden_size, bias=True)
        self.linear_v = Linear(hidden_size, hidden_size, bias=True)
        self.linear_out = Linear(hidden_size, hidden_size, bias=True)
        self._n_heads = n_heads
        self._head_dim = hidden_size // n_heads

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        """Multi-head self-attention.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)``

        Returns:
            output: ``(batch, seq_len, hidden_size)``
        """
        q = self.linear_q(op, hidden_states)  # (B, T, H)
        k = self.linear_k(op, hidden_states)  # (B, T, H)
        v = self.linear_v(op, hidden_states)  # (B, T, H)

        scale = self._head_dim**-0.5
        attn_output, _, _ = op.Attention(
            q,
            k,
            v,
            q_num_heads=self._n_heads,
            kv_num_heads=self._n_heads,
            scale=scale,
            _outputs=3,
        )  # (B, T, H)

        return self.linear_out(op, attn_output)  # (B, T, H)


# ── Adaptor Block ──────────────────────────────────────────────────────


class FunASRAdaptorBlock(nn.Module):
    """Single transformer block in the audio adaptor.

    Architecture: pre-norm attention + pre-norm FFN with residuals::

        LayerNorm → AdaptorAttention → +residual
        → LayerNorm → SANMFFN → +residual

    Weight names match HuggingFace:
    ``audio_adaptor.blocks.N.{norm1,self_attn,norm2,feed_forward}``
    """

    def __init__(self, hidden_size: int, ffn_dim: int, n_heads: int):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size)
        self.self_attn = AdaptorAttention(hidden_size, n_heads)
        self.norm2 = LayerNorm(hidden_size)
        self.feed_forward = SANMFFN(hidden_size, ffn_dim)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # Pre-norm attention with residual
        residual = hidden_states
        hidden_states = self.norm1(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states)
        hidden_states = op.Add(hidden_states, residual)

        # Pre-norm FFN with residual
        residual = hidden_states
        hidden_states = self.norm2(op, hidden_states)
        hidden_states = self.feed_forward(op, hidden_states)
        hidden_states = op.Add(hidden_states, residual)

        return hidden_states


# ── Audio Adaptor ──────────────────────────────────────────────────────


class FunASRAudioAdaptor(nn.Module):
    """Audio adaptor: projects encoder output to LLM hidden dimension.

    Pipeline:
        1. ``linear1``: upproject (encoder_dim → proj_dim, e.g. 512 → 2048)
        2. ReLU activation
        3. ``linear2``: downproject (proj_dim → llm_hidden, e.g. 2048 → 1024)
        4. N transformer blocks refining the projected features

    Weight names match HuggingFace:
    ``audio_adaptor.{linear1,linear2,blocks.N.*}``
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        audio = config.audio
        assert audio is not None

        encoder_dim = audio.attention_dim or 512
        proj_dim = audio.adaptor_proj_dim or 2048
        llm_hidden = config.hidden_size
        num_blocks = audio.adaptor_num_blocks or 2
        ffn_dim = audio.adaptor_ffn_dim or 256
        n_heads = audio.adaptor_num_heads or 16

        self.linear1 = Linear(encoder_dim, proj_dim, bias=True)
        self.linear2 = Linear(proj_dim, llm_hidden, bias=True)
        self.blocks = nn.ModuleList(
            [FunASRAdaptorBlock(llm_hidden, ffn_dim, n_heads) for _ in range(num_blocks)]
        )

    def forward(self, op: OpBuilder, audio_features: ir.Value) -> ir.Value:
        """Project audio features from encoder dim to LLM hidden dim.

        Args:
            audio_features: ``(batch, seq_len, encoder_dim)`` — 3D features

        Returns:
            projected: ``(batch, seq_len, llm_hidden_size)``
        """
        # MLP projection: encoder_dim → proj_dim → llm_hidden
        hidden = self.linear1(op, audio_features)  # (B, T, proj_dim)
        hidden = op.Relu(hidden)
        hidden = self.linear2(op, hidden)  # (B, T, llm_hidden)

        # Transformer blocks refine the projected features
        for block in self.blocks:
            hidden = block(op, hidden)

        return hidden  # (B, T, llm_hidden)


# ── Embedding Model ────────────────────────────────────────────────────


class FunASREmbeddingModel(nn.Module):
    """Fun-ASR embedding model: text/audio embedding fusion.

    Audio features arrive already projected to LLM dimension (from the
    audio encoder's built-in adaptor). This module replaces audio_token_id
    positions in the text embedding with the provided audio features.

    Inputs:
        input_ids: ``(batch, seq_len)`` token IDs
        audio_features: ``(num_audio_tokens, llm_hidden_size)`` from audio encoder

    Output:
        inputs_embeds: ``(batch, seq_len, hidden_size)``

    Note:
        This model only supports ``batch=1``. The CumSum-based audio scatter
        uses a flat ``audio_features`` table shared across the batch, so with
        ``batch > 1`` each row's cumsum restarts at 1 and indexes the same
        prefix of ``audio_features``.

        When ``audio_token_id`` is 0, generated token 0 during autoregressive
        decoding would collide with audio placeholders. Callers should bypass
        this model for decode steps and use the embed_tokens weight table
        directly (see ``examples/fun_asr.py``).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        audio = config.audio
        audio_token_id = (
            audio.audio_token_id if audio and audio.audio_token_id is not None else 151676
        )
        self._audio_token_id = audio_token_id
        self._llm_hidden_size = config.hidden_size

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        audio_features: ir.Value,
    ) -> ir.Value:
        """Fuse text embeddings with LLM-dimension audio features.

        Audio features are already projected to LLM hidden size by the
        audio encoder's adaptor, so we just scatter them at audio_token_id
        positions via Gather + Where.

        Note: When ``audio_token_id`` is 0, any generated token 0 during
        autoregressive decoding would collide with audio placeholders.
        Callers should bypass this model for decode steps and use the
        embed_tokens weight table directly (see examples/fun_asr.py).
        """
        # Text embeddings: (batch, seq_len, hidden_size)
        inputs_embeds = self.embed_tokens(op, input_ids)

        # Create mask: True where input_ids == audio_token_id
        audio_token = op.Constant(value_int=self._audio_token_id)
        is_audio = op.Equal(input_ids, audio_token)
        # Unsqueeze for broadcasting: (batch, seq_len, 1)
        is_audio_3d = op.Unsqueeze(is_audio, [-1])

        # Pad projected features with a zero row at index 0 so the
        # Gather on non-audio positions returns zeros.
        zero_row = op.Unsqueeze(
            op.CastLike(
                op.Constant(value_floats=[0.0] * self._llm_hidden_size),
                audio_features,
            ),
            [0],
        )
        # Prepend zero row: (num_audio_tokens + 1, llm_hidden)
        padded_features = op.Concat(zero_row, audio_features, axis=0)

        # Cumulative sum of audio mask → 1-based indices into padded_features
        is_audio_int = op.Cast(is_audio, to=7)  # INT64
        cumsum = op.CumSum(is_audio_int, op.Constant(value_int=1))  # axis=1 (seq dim)
        indices = op.Mul(cumsum, is_audio_int)  # zero out non-audio positions

        # Gather audio features at computed indices
        gathered = op.Gather(padded_features, indices, axis=0)

        # Where: replace audio positions with gathered features
        inputs_embeds = op.Where(is_audio_3d, gathered, inputs_embeds)

        return inputs_embeds


# ── Decoder Model ──────────────────────────────────────────────────────


class FunASRDecoderModel(nn.Module):
    """Fun-ASR text decoder: inputs_embeds → logits + KV cache.

    Standard Qwen3 decoder with QK norm. Takes inputs_embeds (fused
    text+audio from the embedding model) instead of raw input_ids.
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
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        """Decode inputs_embeds to logits with KV cache.

        Args:
            inputs_embeds: ``(batch, seq_len, hidden_size)``
            attention_mask: ``(batch, past_seq + seq_len)``
            position_ids: ``(batch, seq_len)``
            past_key_values: list of (key, value) tuples per layer

        Returns:
            logits: ``(batch, seq_len, vocab_size)``
            present_key_values: list of (key, value) tuples per layer
        """
        hidden_states = inputs_embeds
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

        hidden_states = self.norm(op, hidden_states)
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values


# ── Top-Level Model ────────────────────────────────────────────────────


class FunASRForConditionalGeneration(nn.Module):
    """Fun-ASR-Nano composite model for speech recognition.

    Contains:
    - ``audio_tower``: Audio encoder + adaptor (fbank → LLM-dim features)
    - ``embedding``: Text/audio embedding fusion (no adaptor)
    - ``decoder``: Text decoder with KV cache (Qwen3-based)

    The 3-model split for ONNX export is handled by the
    ``fun-asr-speech-language`` task.

    HuggingFace class: ``FunASRForConditionalGeneration``
    """

    default_task: str = "fun-asr-speech-language"
    category: str = "Speech-to-Text"
    config_class: type = ArchitectureConfig

    # HF module sub-trees per ONNX component, read by inspect_components without
    # instantiating the model (mirrors the prefixes routed in preprocess_weights).
    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "audio_encoder": ("audio_encoder", "audio_adaptor"),
        "embedding": ("llm.model.embed_tokens",),
        "decoder": ("llm.model.layers", "llm.model.norm", "llm.lm_head"),
    }

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config

        self.audio_tower = FunASRAudioEncoder(config)
        self.embedding = FunASREmbeddingModel(config)
        self.decoder = FunASRDecoderModel(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        """Forward pass for text-generation task (no audio fusion).

        Embeds input_ids using the text embedding (no audio fusion in
        this path — audio features are fused externally), then runs
        the decoder to produce logits and KV cache.
        """
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

        HF Fun-ASR weight name layout::

            audio_encoder.encoders0.0.*     → audio_tower.encoders0.0.*
            audio_encoder.encoders.N.*      → audio_tower.encoders.N.*
            audio_encoder.tp_encoders.N.*   → audio_tower.tp_encoders.N.*
            audio_encoder.after_norm.*      → audio_tower.after_norm.*
            audio_encoder.tp_norm.*         → audio_tower.tp_norm.*
            audio_adaptor.*                 → audio_tower.adaptor.*
            llm.model.embed_tokens.*        → embedding.embed_tokens.*
            llm.model.layers.N.*            → decoder.layers.N.*
            llm.model.norm.*                → decoder.norm.*
            llm.lm_head.*                   → decoder.lm_head.*
        """
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Route audio_encoder weights → audio_tower
            if key.startswith("audio_encoder."):
                inner = key[len("audio_encoder.") :]
                cleaned[f"audio_tower.{inner}"] = value
                continue

            # Route audio_adaptor weights → audio_tower.adaptor
            if key.startswith("audio_adaptor."):
                inner = key[len("audio_adaptor.") :]
                cleaned[f"audio_tower.adaptor.{inner}"] = value
                continue

            # Route llm.lm_head to decoder.lm_head
            if key.startswith("llm.lm_head."):
                inner = key[len("llm.") :]
                cleaned[f"decoder.{inner}"] = value
                continue

            # Route llm.model.* to appropriate sub-module
            if key.startswith("llm.model."):
                inner = key[len("llm.model.") :]

                # embed_tokens → embedding module
                if inner.startswith("embed_tokens."):
                    cleaned[f"embedding.{inner}"] = value
                    continue

                # layers.N.* and norm.* → decoder module
                if inner.startswith(("layers.", "norm.")):
                    cleaned[f"decoder.{inner}"] = value
                    continue

                # rotary_emb → decoder module
                if inner.startswith("rotary_emb."):
                    cleaned[f"decoder.{inner}"] = value
                    continue

            cleaned[key] = value

        # Transpose FSMN conv weights if needed: (C, K, 1) → (C, 1, K)
        for key in list(cleaned.keys()):
            if "fsmn_block.weight" in key:
                w = cleaned[key]
                if w.ndim == 3 and w.shape[2] == 1 and w.shape[1] > 1:
                    cleaned[key] = w.transpose(1, 2)

        # Weight tying: embed_tokens → lm_head
        embed_key = "embedding.embed_tokens.weight"
        lm_key = "decoder.lm_head.weight"
        if self.config.tie_word_embeddings:
            if embed_key in cleaned and lm_key not in cleaned:
                cleaned[lm_key] = cleaned[embed_key]
            elif lm_key in cleaned and embed_key not in cleaned:
                cleaned[embed_key] = cleaned[lm_key]

        return cleaned
