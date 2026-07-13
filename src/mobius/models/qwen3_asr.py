# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen3-ASR: Audio speech recognition with Qwen3 text decoder.

Architecture:
  - Audio encoder: 3x Conv2d downsampling → sinusoidal PE → N bidirectional
    encoder layers → LayerNorm → proj1 → GELU → proj2
  - Text decoder: Qwen3 with QK norm + interleaved MRoPE
  - Fusion: Audio features replace audio_token_id positions in text embeddings

Also supports Qwen3-ForcedAligner (same architecture, classification head).

Reference: https://huggingface.co/Qwen/Qwen3-ASR-0.6B
HuggingFace class: Qwen3ASRForConditionalGeneration
"""

from __future__ import annotations

import dataclasses
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
from mobius.components._conv import Conv2d
from mobius.components._decoder import DecoderLayer
from mobius.components._qwen3_asr_audio import (
    Qwen3ASRAudioEncoderLayer,
)
from mobius.components._rms_norm import RMSNorm
from mobius.components._rotary_embedding import initialize_rope


def _sinusoidal_position_embedding(max_positions: int, d_model: int) -> np.ndarray:
    """Compute sinusoidal positional embeddings matching Qwen3-ASR.

    Uses log-timescale increments (different from Whisper which uses
    alternating sin/cos layout). Layout: [sin_0..sin_n, cos_0..cos_n].
    """
    channels = d_model
    log_timescale_increment = np.log(10000.0) / (channels // 2 - 1)
    inv_timescales = np.exp(
        -log_timescale_increment * np.arange(channels // 2, dtype=np.float32)
    )
    scaled_time = (
        np.arange(max_positions, dtype=np.float32)[:, np.newaxis]
        * inv_timescales[np.newaxis, :]
    )
    # Layout: [sin, cos] matching HF SinusoidsPositionEmbedding
    pe = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1).astype(np.float32)
    return pe


class Qwen3ASRAudioEncoder(nn.Module):
    """Qwen3-ASR audio encoder.

    Converts mel spectrogram to audio feature embeddings:
      mel (batch, num_mel_bins, seq_len)
      → 3x Conv2d with GELU downsampling
      → linear projection (conv_out)
      → sinusoidal positional embeddings
      → N bidirectional encoder layers
      → LayerNorm (ln_post)
      → proj1 → GELU → proj2

    Output: (batch, out_seq_len, output_dim)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        audio = config.audio
        assert audio is not None

        d_model = audio.d_model or 896
        num_mel_bins = audio.num_mel_bins or 128
        encoder_layers = audio.encoder_layers or 18
        encoder_heads = audio.encoder_attention_heads or 14
        encoder_ffn = audio.encoder_ffn_dim or 3584
        max_source_positions = audio.max_source_positions or 1500
        downsample_hidden_size = audio.downsample_hidden_size or 480
        output_dim = audio.output_dim or 1024
        # Qwen3-ASR chunked conv constants. Defaults match both
        # 0.6B and 1.7B (n_window=50, n_window_infer=800).
        n_window = audio.n_window or 50
        n_window_infer = audio.n_window_infer or 800

        # Chunk geometry: HF runs the conv over fixed-size chunks of
        # ``2 * n_window`` mel frames, each producing
        # ``tokens_per_chunk`` post-conv tokens. Encoder self-attention
        # is then block-diagonal with windows of ``block_size`` post-
        # conv tokens. All three numbers are derived from config.
        self._chunk_size_mel = 2 * n_window  # mel frames per conv chunk (100)
        # tokens_per_chunk = 3x ceil-div-2 on chunk_size_mel
        # = ceil(ceil(ceil(100/2)/2)/2) = 13 for the default
        t = self._chunk_size_mel
        for _ in range(3):
            t = (t + 1) // 2
        self._tokens_per_chunk = t  # post-conv tokens per chunk (13)
        # block_size in post-conv tokens. Each attention block covers
        # ``n_window_infer`` mel frames = (n_window_infer / chunk_size)
        # full chunks = that many * tokens_per_chunk post-conv tokens.
        # Requires n_window_infer to be a multiple of chunk_size_mel —
        # both shipped Qwen3-ASR variants satisfy this (800 % 100 = 0).
        if n_window_infer % self._chunk_size_mel != 0:
            raise ValueError(
                f"n_window_infer ({n_window_infer}) must be a multiple of "
                f"2 * n_window ({self._chunk_size_mel})"
            )
        self._block_size = self._tokens_per_chunk * (n_window_infer // self._chunk_size_mel)

        # 3x Conv2d downsampling: (1, mel, seq) → (dhs, mel//8, seq//8)
        self.conv2d1 = Conv2d(
            1,
            downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2d2 = Conv2d(
            downsample_hidden_size,
            downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2d3 = Conv2d(
            downsample_hidden_size,
            downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Linear projection from flattened conv features to d_model.
        # After 3 strides of 2: freq_dim = (((mel+1)//2+1)//2+1)//2.
        freq_after_conv = (((num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        conv_out_dim = downsample_hidden_size * freq_after_conv
        self._conv_out_dim = conv_out_dim
        self._d_model = d_model
        self._num_mel_bins = num_mel_bins
        self.conv_out = Linear(conv_out_dim, d_model, bias=False)

        # Sinusoidal positional embeddings (frozen). HF uses these
        # PER-CHUNK (each chunk reuses PE[0:tokens_per_chunk]) — see
        # forward() for the slicing logic.
        pe_data = _sinusoidal_position_embedding(max_source_positions, d_model)
        self.positional_embedding = nn.Parameter(
            [max_source_positions, d_model],
            name="positional_embedding.positional_embedding",
            data=ir.tensor(pe_data),
        )

        # Encoder transformer layers
        self.layers = nn.ModuleList(
            [
                Qwen3ASRAudioEncoderLayer(d_model, encoder_heads, encoder_ffn)
                for _ in range(encoder_layers)
            ]
        )

        # Post-encoder normalization
        self.ln_post = LayerNorm(d_model)

        # Output projection: d_model → output_dim
        self.proj1 = Linear(d_model, d_model, bias=True)
        self.proj2 = Linear(d_model, output_dim, bias=True)

    def forward(
        self,
        op: OpBuilder,
        input_features: ir.Value,
        feature_attention_mask: ir.Value,
    ):
        """Encode mel spectrogram to audio features (HF-faithful chunked conv).

        Reproduces the HF Qwen3-ASR audio encoder architecture:
        conv runs on fixed ``2 * n_window``-frame mel chunks (one
        chunk's conv outputs are independent of its neighbours),
        each chunk's post-conv tokens get the same per-chunk
        positional embedding, and self-attention is block-diagonal
        with ``block_size`` post-conv tokens per block. Without
        chunking, the conv arithmetic produces ``ceil(mel_seq / 8)``
        tokens per 30s instead of HF's ``30 * tokens_per_chunk``,
        which misaligns the LLM and triggers repetition loops on
        long inputs.

        Args:
            input_features: (batch, num_mel_bins, mel_seq_len) mel
                spectrogram. ``mel_seq_len`` MUST be a multiple of
                ``2 * n_window`` (= 100 for both 0.6B and 1.7B).
                The HF processor pads to 3000 frames by default,
                which satisfies this.
            feature_attention_mask: (batch, mel_seq_len) int64 mask.
                ``1`` = real audio frame, ``0`` = right-padding from
                the processor. Required.

        Returns:
            audio_features: (batch, mel_seq_len // chunk_size_mel *
                tokens_per_chunk, output_dim). For 30s @ 100-frame
                chunks: 390 tokens. Includes ``audio_feature_lengths``
                padding-derived tokens at the tail; callers MUST crop
                via ``audio_feature_lengths`` before feeding into the
                embedding model.
            audio_feature_lengths: (batch,) int64 — number of valid
                audio tokens per batch item, computed via the same
                formula HF uses
                (``_get_feat_extract_output_lengths``).
        """
        chunk_size_mel = self._chunk_size_mel
        tokens_per_chunk = self._tokens_per_chunk
        block_size = self._block_size
        d_model = self._d_model
        num_mel_bins = self._num_mel_bins

        # ----------------------------------------------------------
        # 1. Chunk the mel input.
        #    (B, num_mel_bins, mel_seq) → (B, num_mel_bins,
        #    num_chunks, chunk_size_mel) → (B, num_chunks,
        #    num_mel_bins, chunk_size_mel) → (B*num_chunks, 1,
        #    num_mel_bins, chunk_size_mel) for batched 2D conv.
        #
        #    Requires mel_seq divisible by chunk_size_mel. The
        #    WhisperFeatureExtractor always pads to 3000 frames
        #    (30s * 100 fps), which is divisible by the default
        #    chunk_size_mel (2 * n_window = 100).
        # ----------------------------------------------------------
        chunked = op.Reshape(
            input_features,
            op.Constant(value_ints=[0, 0, -1, chunk_size_mel]),
        )  # (B, num_mel_bins, num_chunks, chunk_size_mel)
        chunked = op.Transpose(chunked, perm=[0, 2, 1, 3])
        # (B, num_chunks, num_mel_bins, chunk_size_mel)
        flat_chunks = op.Reshape(
            chunked,
            op.Constant(value_ints=[-1, 1, num_mel_bins, chunk_size_mel]),
        )  # (B*num_chunks, 1, num_mel_bins, chunk_size_mel)

        # ----------------------------------------------------------
        # 2. Apply the 3 strided convolutions independently to each
        #    chunk. Same conv weights as HF (single shared set).
        # ----------------------------------------------------------
        conv_out = op.Gelu(self.conv2d1(op, flat_chunks))
        conv_out = op.Gelu(self.conv2d2(op, conv_out))
        conv_out = op.Gelu(self.conv2d3(op, conv_out))
        # (B*num_chunks, dhs, freq_after_conv, tokens_per_chunk)

        # ----------------------------------------------------------
        # 3. Permute and flatten: (B*nc, dhs, freq, t) →
        #    (B*nc, t, dhs, freq) → (B*nc, t, dhs*freq).
        # ----------------------------------------------------------
        conv_out = op.Transpose(conv_out, perm=[0, 3, 1, 2])
        conv_out = op.Reshape(conv_out, op.Constant(value_ints=[0, 0, -1]))
        # (B*num_chunks, tokens_per_chunk, conv_out_dim)

        # ----------------------------------------------------------
        # 4. Linear projection to d_model.
        # ----------------------------------------------------------
        chunked_features = self.conv_out(op, conv_out)
        # (B*num_chunks, tokens_per_chunk, d_model)

        # ----------------------------------------------------------
        # 5. Add per-chunk positional embedding. HF reuses
        #    PE[0:tokens_per_chunk] for EVERY chunk independently —
        #    cross-chunk ordering comes from the attention mask, not
        #    from PE.
        # ----------------------------------------------------------
        pe_slice = op.Slice(
            self.positional_embedding,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[tokens_per_chunk]),
            op.Constant(value_ints=[0]),
        )  # (tokens_per_chunk, d_model)
        pe_slice = op.CastLike(pe_slice, chunked_features)
        pe_slice = op.Unsqueeze(pe_slice, [0])  # (1, tokens_per_chunk, d_model)
        chunked_features = op.Add(chunked_features, pe_slice)

        # ----------------------------------------------------------
        # 6. Reshape back to per-batch flat sequence:
        #    (B*nc, t, d) → (B, nc*t, d).
        # ----------------------------------------------------------
        batch_size = op.Shape(input_features, start=0, end=1)  # (1,)
        target_shape = op.Concat(
            batch_size,
            op.Constant(value_ints=[-1, d_model]),
            axis=0,
        )
        hidden_states = op.Reshape(chunked_features, target_shape)
        # (B, total_post_conv, d_model)

        # ----------------------------------------------------------
        # 7. Compute audio_feature_lengths via HF's
        #    _get_feat_extract_output_lengths. For valid_mel ≥ 0
        #    (always true since the mask has 0/1 values):
        #      num_full_chunks = valid_mel // chunk_size_mel
        #      remainder       = valid_mel % chunk_size_mel
        #      tail_tokens     = ceil(ceil(ceil(rem/2)/2)/2)
        #                      = (((rem+1)//2 + 1)//2 + 1)//2
        #      length = num_full_chunks * tokens_per_chunk + tail_tokens
        #    ONNX int Div truncates toward zero; identical to floor
        #    div for non-negative operands.
        # ----------------------------------------------------------
        chunk_size_mel_const = op.Constant(value_ints=[chunk_size_mel])
        tokens_per_chunk_const = op.Constant(value_ints=[tokens_per_chunk])
        one_const = op.Constant(value_ints=[1])
        two_const = op.Constant(value_ints=[2])

        valid_mel = op.ReduceSum(
            feature_attention_mask,
            op.Constant(value_ints=[1]),
            keepdims=0,
        )  # (B,) int64
        num_full_chunks = op.Div(valid_mel, chunk_size_mel_const)
        full_contrib = op.Mul(num_full_chunks, tokens_per_chunk_const)
        remainder = op.Sub(valid_mel, op.Mul(num_full_chunks, chunk_size_mel_const))
        s1 = op.Div(op.Add(remainder, one_const), two_const)
        s2 = op.Div(op.Add(s1, one_const), two_const)
        s3 = op.Div(op.Add(s2, one_const), two_const)
        audio_feature_lengths = op.Add(full_contrib, s3)  # (B,) int64

        # ----------------------------------------------------------
        # 8. Build the block-diagonal attention mask.
        #
        #    HF's reference uses a varlen FlashAttention path with
        #    cu_seqlens to enforce that token q only attends to
        #    keys in the same ``block_size``-token window. With
        #    fixed-shape ONNX we instead build a (B, S, S) bool
        #    mask:
        #        same_block(q, k) = (q // block_size) == (k // block_size)
        #        valid(b, i)      = i < audio_feature_lengths[b]
        #        mask(b, q, k)    = same_block(q, k)
        #                           AND valid(b, q) AND valid(b, k)
        #
        #    BUT: rows where the query is past audio_feature_lengths
        #    would have all-False keys, which produces NaN on
        #    softmax. To keep ORT's Attention numerically safe, we
        #    OR in the diagonal so every query has at least one
        #    allowed key (itself). Padding queries then "attend to
        #    themselves only", producing some finite (junk) value
        #    that the caller throws away by cropping via
        #    audio_feature_lengths.
        # ----------------------------------------------------------
        seq_dim = op.Shape(hidden_states, start=1, end=2)  # (1,)
        seq_scalar = op.Squeeze(seq_dim, op.Constant(value_ints=[0]))
        zero_scalar = op.Constant(value_int=0)
        one_scalar = op.Constant(value_int=1)
        block_size_scalar = op.Constant(value_int=block_size)
        position = op.Range(zero_scalar, seq_scalar, one_scalar)  # (S,) int64

        block_id = op.Div(position, block_size_scalar)  # (S,) int64
        block_id_q = op.Unsqueeze(block_id, [1])  # (S, 1)
        block_id_k = op.Unsqueeze(block_id, [0])  # (1, S)
        same_block = op.Equal(block_id_q, block_id_k)  # (S, S) bool

        pos_2d = op.Unsqueeze(position, [0])  # (1, S)
        afl_2d = op.Unsqueeze(audio_feature_lengths, [1])  # (B, 1)
        valid = op.Less(pos_2d, afl_2d)  # (B, S) bool

        valid_q = op.Unsqueeze(valid, [2])  # (B, S, 1)
        valid_k = op.Unsqueeze(valid, [1])  # (B, 1, S)
        same_block_3d = op.Unsqueeze(same_block, [0])  # (1, S, S)
        block_mask = op.And(op.And(same_block_3d, valid_q), valid_k)
        # (B, S, S) bool

        # Diagonal fallback so padded queries always have ≥1 key.
        diag_eq = op.Equal(
            op.Unsqueeze(position, [1]),  # (S, 1)
            op.Unsqueeze(position, [0]),  # (1, S)
        )  # (S, S) bool — True only on the diagonal
        diag_3d = op.Unsqueeze(diag_eq, [0])  # (1, S, S)
        attention_mask = op.Or(block_mask, diag_3d)  # (B, S, S) bool

        # ----------------------------------------------------------
        # 9. Encoder layers + post-encoder norm + output projection.
        # ----------------------------------------------------------
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, attention_mask)

        hidden_states = self.ln_post(op, hidden_states)
        hidden_states = self.proj1(op, hidden_states)
        hidden_states = op.Gelu(hidden_states)
        hidden_states = self.proj2(op, hidden_states)

        return hidden_states, audio_feature_lengths


class Qwen3ASREmbeddingModel(nn.Module):
    """Qwen3-ASR embedding model: fuses text and audio embeddings.

    Replaces audio_token_id positions in the text embedding with
    audio features from the audio encoder.

    Inputs:
        input_ids: (batch, seq_len) token IDs
        audio_features: (num_audio_tokens, output_dim) from audio encoder

    Output:
        inputs_embeds: (batch, seq_len, hidden_size)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        audio_token_id = config.audio.audio_token_id if config.audio else 151676
        self._audio_token_id = audio_token_id
        self._audio_output_dim = (config.audio.output_dim or 1024) if config.audio else 1024

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        audio_features: ir.Value,
    ):
        """Fuse text embeddings with audio features.

        Audio features replace positions where input_ids == audio_token_id.
        Uses Gather + Where pattern (equivalent to masked_scatter).
        """
        # Text embeddings: (batch, seq_len, hidden_size)
        inputs_embeds = self.embed_tokens(op, input_ids)

        # Create mask: True where input_ids == audio_token_id
        audio_token = op.Constant(value_int=self._audio_token_id)
        is_audio = op.Equal(input_ids, audio_token)
        # Unsqueeze for broadcasting: (batch, seq_len, 1)
        is_audio_3d = op.Unsqueeze(is_audio, [-1])

        # Pad audio_features with a zero row at index 0 to handle
        # the case where no audio tokens exist in the sequence.
        # Use static Constant (not ConstantOfShape) to preserve shape inference.
        zero_row = op.Unsqueeze(
            op.CastLike(
                op.Constant(value_floats=[0.0] * self._audio_output_dim),
                audio_features,
            ),
            [0],
        )
        # Prepend zero row: (num_audio_tokens + 1, output_dim)
        padded_features = op.Concat(zero_row, audio_features, axis=0)

        # Compute gather indices: cumulative sum of audio mask gives
        # 1-based indices into padded_features (0 = zero padding row)
        is_audio_int = op.Cast(is_audio, to=7)  # INT64
        cumsum = op.CumSum(is_audio_int, op.Constant(value_int=1))  # axis=1 (seq dim)
        indices = op.Mul(cumsum, is_audio_int)  # zero out non-audio positions

        # Gather audio features using computed indices
        gathered = op.Gather(padded_features, indices, axis=0)

        # Where: replace audio positions with gathered features
        inputs_embeds = op.Where(is_audio_3d, gathered, inputs_embeds)

        return inputs_embeds


class Qwen3ASRDecoderModel(nn.Module):
    """Qwen3-ASR text decoder: inputs_embeds → logits + KV cache.

    Standard Qwen3 decoder with QK norm and interleaved MRoPE.
    Takes inputs_embeds (fused text+audio) instead of input_ids.
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


class Qwen3ASRForConditionalGeneration(nn.Module):
    """Qwen3-ASR composite model for speech recognition.

    Contains:
    - ``audio_tower``: Audio encoder (mel → audio features)
    - ``embedding``: Text+audio embedding fusion
    - ``decoder``: Text decoder with KV cache

    Also supports Qwen3-ForcedAligner when ``classify_num`` is set
    in the audio config (uses classification head instead of LM head).

    HuggingFace class: ``Qwen3ASRForConditionalGeneration``
    """

    default_task: str = "speech-language"
    category: str = "Speech-to-Text"
    config_class: type = ArchitectureConfig

    # HF module sub-trees per ONNX component, read by inspect_components without
    # instantiating the model (mirrors preprocess_weights; optional outer ``thinker.``).
    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "audio_encoder": ("thinker.audio_tower",),
        "embedding": ("thinker.model.embed_tokens",),
        "decoder": ("thinker.model.layers", "thinker.model.norm", "thinker.lm_head"),
    }

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config

        # Determine output vocab: classify_num for ForcedAligner,
        # else vocab_size
        output_vocab = config.vocab_size
        if config.audio and config.audio.classify_num:
            output_vocab = config.audio.classify_num

        self.audio_tower = Qwen3ASRAudioEncoder(config)

        self.embedding = Qwen3ASREmbeddingModel(config)
        self.decoder = Qwen3ASRDecoderModel(
            config
            if output_vocab == config.vocab_size
            else dataclasses.replace(config, vocab_size=output_vocab)
        )

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        """Forward pass for text-generation task.

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

        HF weights have ``thinker.`` prefix:
        - ``thinker.audio_tower.*`` → ``audio_tower.*``
        - ``thinker.model.*`` → text decoder layers
        - ``thinker.lm_head.*`` → ``decoder.lm_head.*``

        For embedding:
            ``thinker.model.embed_tokens.weight``
            → ``embedding.embed_tokens.weight``
        For decoder layers:
            ``thinker.model.layers.N.*`` → ``decoder.layers.N.*``
        For decoder norm:
            ``thinker.model.norm.*`` → ``decoder.norm.*``
        """
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Strip thinker. prefix
            if key.startswith("thinker."):
                key = key[len("thinker.") :]

            # Route audio_tower weights
            if key.startswith("audio_tower."):
                cleaned[key] = value
                continue

            # Route lm_head to decoder
            if key.startswith("lm_head."):
                cleaned[f"decoder.{key}"] = value
                continue

            # Route model.* to appropriate sub-module
            if key.startswith("model."):
                inner = key[len("model.") :]

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

        # Weight tying: embed_tokens → lm_head
        embed_key = "embedding.embed_tokens.weight"
        lm_key = "decoder.lm_head.weight"
        if self.config.tie_word_embeddings:
            if embed_key in cleaned and lm_key not in cleaned:
                cleaned[lm_key] = cleaned[embed_key]
            elif lm_key in cleaned and embed_key not in cleaned:
                cleaned[embed_key] = cleaned[lm_key]

        return cleaned
