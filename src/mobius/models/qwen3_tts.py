# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen3-TTS: Text-to-speech with Qwen3-like talker and code predictor.

Architecture (4-model split for full TTS):
  1. **Talker**: Qwen3 decoder with MRoPE 3D, QK norm. Outputs logits
     (first code group) and last_hidden_state (for code predictor).
  2. **Code Predictor**: Small 5-layer decoder with 1D RoPE.
     Per-step lm_heads and embeddings selected by step_index.
  3. **Embedding**: text_embedding → text_projection + codec_embedding.
     Additive fusion of text and codec embeddings.
  4. **Speaker Encoder**: ECAPA-TDNN extracting speaker embedding from mel.

Config comparison (0.6B vs 1.7B):
  - talker.hidden_size: 1024 vs 2048
  - talker.text_hidden_size: 2048 (both)
  - code_predictor.hidden_size: 1024 (both)
  - speaker_encoder.enc_dim: 1024 vs 2048

Reference: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
HuggingFace class: Qwen3TTSForConditionalGeneration
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    DecoderLayer,
    Embedding,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._ecapa_tdnn import SpeakerEncoder

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Talker decoder model (model 1 of 4)
# ---------------------------------------------------------------------------


class Qwen3TTSTalkerDecoderModel(nn.Module):
    """Talker decoder: inputs_embeds → logits + last_hidden_state + KV cache.

    Standard Qwen3 decoder with QK norm and interleaved MRoPE.
    Takes ``inputs_embeds`` (from the embedding model) rather than input_ids.
    Outputs both logits (via codec_head) and the pre-norm hidden state from
    the last position, which feeds into the code predictor.

    HuggingFace class: ``Qwen3TTSTalkerForConditionalGeneration``
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        # codec_head maps hidden → codec vocab (3072)
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

        # Apply norm (matches HuggingFace's model output: after layers + norm)
        hidden_states = self.norm(op, hidden_states)

        # Extract last position hidden state AFTER norm for code predictor.
        # HF's past_hidden = outputs.last_hidden_state[:, -1:, :] which is
        # post-norm. The code predictor was trained with normed inputs.
        seq_len = op.Shape(hidden_states, start=1, end=2)
        last_idx = op.Sub(seq_len, op.Constant(value_ints=[1]))
        # last_hidden_state: (batch, 1, hidden_size)
        last_hidden_state = op.Gather(hidden_states, last_idx, axis=1)

        # Compute logits from normed hidden states
        logits = self.lm_head(op, hidden_states)

        return logits, last_hidden_state, present_key_values


# ---------------------------------------------------------------------------
# Code predictor model (model 2 of 4)
# ---------------------------------------------------------------------------


class Qwen3TTSCodePredictorModel(nn.Module):
    """Code predictor: generates remaining code groups from talker hidden.

    A small 5-layer Qwen3-like decoder with 1D RoPE (NOT MRoPE).
    Has per-step lm_heads selected by ``step_index``, and per-step
    codec embeddings exposed as weights for external lookup.

    The generation loop constructs ``inputs_embeds`` externally:
      - Step 0 (prefill): concat(projected_hidden, talker_codec_embed(code_0))
        → 2 tokens, matching HF's code predictor prefill.
      - Steps 1-14 (generation): CP_codec_embed[step-1](code_i)
        → 1 token, matching HF's autoregressive step.

    This model only handles: optional projection → transformer → lm_head.
    Codec embedding lookups are done in the generation loop.

    Inputs:
      - ``inputs_embeds``: (batch, seq, cp_hidden_size) — pre-embedded
        input sequence. seq=2 for prefill (step 0), seq=1 for generation.
      - ``step_index``: scalar int64 — selects which lm_head to use
        (0 to num_code_groups-2).

    Output: logits (batch, seq, cp_vocab_size), KV cache.

    HuggingFace class: ``Qwen3TTSTalkerCodePredictorModelForConditionalGeneration``
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        tts = config.tts
        cp = tts.code_predictor if tts else None

        # Talker hidden → code predictor hidden projection (only when dims differ)
        talker_hidden = config.hidden_size  # from parent talker config
        cp_hidden = cp.hidden_size if cp else 1024
        self._needs_projection = talker_hidden != cp_hidden
        if self._needs_projection:
            self.small_to_mtp_projection = Linear(talker_hidden, cp_hidden, bias=True)

        # Build a sub-config for the code predictor decoder layers
        cp_config = self._make_cp_config(config)
        self.layers = nn.ModuleList(
            [DecoderLayer(cp_config) for _ in range(cp_config.num_hidden_layers)]
        )
        self.norm = RMSNorm(cp_config.hidden_size, eps=cp_config.rms_norm_eps)
        self.rotary_emb = initialize_rope(cp_config)

        # Per-step codec embeddings stacked into one parameter:
        # (num_code_groups-1, cp_vocab_size, talker_hidden_size)
        # Stored in talker_hidden space (same as HF's embedding_dim).
        # Not used in forward — exposed for the generation loop to
        # extract and do numpy lookups for codec_sum computation.
        num_extra_groups = (tts.num_code_groups if tts else 16) - 1
        cp_vocab = cp.vocab_size if cp else 2048
        self.stacked_codec_embedding = nn.Parameter(
            [num_extra_groups, cp_vocab, talker_hidden]
        )

        # Per-step lm_heads stacked into one parameter:
        # (num_code_groups-1, cp_vocab_size, cp_hidden_size)
        self.stacked_lm_head = nn.Parameter([num_extra_groups, cp_vocab, cp_hidden])

    @staticmethod
    def _make_cp_config(config: ArchitectureConfig) -> ArchitectureConfig:
        """Create an ArchitectureConfig for the code predictor decoder."""
        tts = config.tts
        cp = tts.code_predictor if tts else None
        return dataclasses.replace(
            config,
            hidden_size=cp.hidden_size if cp else 1024,
            intermediate_size=cp.intermediate_size if cp else 3072,
            num_hidden_layers=cp.num_hidden_layers if cp else 5,
            num_attention_heads=cp.num_attention_heads if cp else 16,
            num_key_value_heads=cp.num_key_value_heads if cp else 8,
            head_dim=cp.head_dim if cp else 128,
            vocab_size=cp.vocab_size if cp else 2048,
            rms_norm_eps=cp.rms_norm_eps if cp else 1e-6,
            rope_theta=cp.rope_theta if cp else 1_000_000.0,
            hidden_act=cp.hidden_act if cp else "silu",
            layer_types=cp.layer_types if cp else None,
            # Code predictor uses standard 1D RoPE, no MRoPE
            rope_scaling=None,
            mrope_section=None,
            mrope_interleaved=False,
            attn_qk_norm=True,
        )

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        step_index: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        """Code predictor step: project → decode → lm_head.

        The generation loop constructs ``inputs_embeds`` externally in
        **talker_hidden** space (e.g. 2048 for 1.7B, 1024 for 0.6B):
          - Step 0 (prefill): concat(talker_hidden, talker_embed(code_0)),
            shape (batch, 2, talker_hidden). Matches HF's 2-token prefill.
          - Steps 1-14: CP_embed[step-1](code_i), shape (batch, 1, talker_hidden).
            CP codec embeddings are stored in talker_hidden space.

        The model projects inputs to cp_hidden via ``small_to_mtp_projection``
        (Identity when dims match), then runs the transformer decoder.

        Args:
            inputs_embeds: (batch, seq, talker_hidden_size) pre-embedded input.
            step_index: scalar int64. Selects which lm_head (0..14).
            attention_mask: (batch, past_seq_len + seq).
            position_ids: (batch, seq) for 1D RoPE.
            past_key_values: list of (key, value) tuples or None.

        Returns:
            logits: (batch, seq, cp_vocab_size).
            present_key_values: updated KV cache.
        """
        # 1. Project from talker_hidden to cp_hidden space.
        # When dims match (e.g. 0.6B: both 1024), this is a no-op.
        # When they differ (e.g. 1.7B: 2048→1024), Linear projection.
        if self._needs_projection:
            inputs_embeds = self.small_to_mtp_projection(op, inputs_embeds)

        # 2. Decoder layers with 1D RoPE
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

        # 3. Select and apply step-specific lm_head
        # stacked_lm_head: (num_groups-1, cp_vocab, cp_hidden)
        lm_head_weight = op.Gather(
            self.stacked_lm_head, step_index, axis=0
        )  # (cp_vocab, cp_hidden)
        # logits = hidden @ lm_head.T → (batch, seq, cp_vocab)
        lm_head_t = op.Transpose(lm_head_weight, perm=[1, 0])
        logits = op.MatMul(hidden_states, lm_head_t)

        # 4. Return stacked codec embeddings so the generation loop can
        # extract them for codec_sum computation (numpy lookup).
        return logits, present_key_values, op.Identity(self.stacked_codec_embedding)


# ---------------------------------------------------------------------------
# Embedding model (model 3 of 4)
# ---------------------------------------------------------------------------


class Qwen3TTSEmbeddingModel(nn.Module):
    """TTS embedding: text_ids → projected text_embeds, codec_ids → codec_embeds.

    Contains:
      - text_embedding: text_vocab_size → text_hidden_size
      - text_projection (ResizeMLP): text_hidden_size → hidden_size
      - codec_embedding: codec_vocab_size → hidden_size

    The additive fusion (text_embeds + codec_embeds) is done externally
    in the inference loop.

    HuggingFace classes:
      - ``talker.model.text_embedding``
      - ``talker.text_projection`` (ResizeMLP with bias)
      - ``talker.model.codec_embedding`` (aka ``model.embed_tokens``)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        tts = config.tts
        text_hidden = tts.text_hidden_size if tts else 2048
        text_vocab = tts.text_vocab_size if tts else 151936
        hidden = config.hidden_size

        # Text embedding + projection
        self.text_embedding = Embedding(text_vocab, text_hidden)
        # ResizeMLP: text_hidden → SiLU → hidden, with bias
        # fc1: text_hidden → text_hidden (pre-activation)
        # fc2: text_hidden → hidden (final projection)
        self.text_projection_fc1 = Linear(text_hidden, text_hidden, bias=True)
        self.text_projection_fc2 = Linear(text_hidden, hidden, bias=True)

        # Codec embedding (audio codes → hidden)
        self.codec_embedding = Embedding(config.vocab_size, hidden)

    def forward(
        self,
        op: OpBuilder,
        text_ids: ir.Value,
        codec_ids: ir.Value,
    ):
        """Produce text_embeds and codec_embeds for additive fusion.

        Args:
            text_ids: (batch, text_seq_len) text token IDs.
            codec_ids: (batch, codec_seq_len) codec token IDs.

        Returns:
            text_embeds: (batch, text_seq_len, hidden_size) projected text.
            codec_embeds: (batch, codec_seq_len, hidden_size) codec embeddings.
        """
        # Text: embed → project (fc1 → SiLU → fc2)
        text_embeds = self.text_embedding(op, text_ids)
        text_embeds = self.text_projection_fc1(op, text_embeds)
        text_embeds = op.Mul(text_embeds, op.Sigmoid(text_embeds))  # SiLU
        text_embeds = self.text_projection_fc2(op, text_embeds)

        # Codec: simple embedding lookup
        codec_embeds = self.codec_embedding(op, codec_ids)

        return text_embeds, codec_embeds


# ---------------------------------------------------------------------------
# Talker step embedder (pre-embedding component)
# ---------------------------------------------------------------------------


class Qwen3TTSTalkerStepEmbedder(nn.Module):
    r"""Materializes the per-step talker ``codec_sum`` construction in ONNX.

    In the reference generation loop the talker's per-step ``inputs_embeds``
    is built in Python as::

        codec_sum = talker.codec_embed(code_0)
                    + Σ_{i=0..14} cp_codec_weights[i][codes[i + 1]]
        inputs_embeds = codec_sum + text_embed

    where ``talker.codec_embed`` is the talker codec-embedding table (the
    embedding model's ``codec_embedding.weight``) and ``cp_codec_weights`` is
    the code predictor's stacked codec-embedding table
    (``[num_code_groups - 1, cp_vocab, hidden]``).

    This module folds that construction into a graph so a generic runtime loop
    can drive the talker by feeding the previous frame's raw integer codes
    (plus the already-embedded trailing-text vector) instead of a pre-built
    ``inputs_embeds``. The text projection stays on the embedding model; this
    component only holds the two codec-embedding tables (shared weights, not
    re-quantized) and does the Gather + Add.

    Inputs:
      - ``frame_codes``: (batch, num_code_groups) int64 — the previous frame's
        codes (``code_0`` followed by ``codes[1..num_code_groups - 1]``).
      - ``text_embed``: (batch, 1, hidden) float — the trailing-text (or pad)
        embedding for this step, produced by the embedding model.

    Output:
      - ``inputs_embeds``: (batch, 1, hidden) float — exactly
        ``codec_sum + text_embed``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        tts = config.tts
        hidden = config.hidden_size
        # Talker codec-embedding table (audio codes → hidden). Same weights as
        # the embedding model's ``codec_embedding``; shared, not re-quantized.
        self.codec_embedding = nn.Parameter([config.vocab_size, hidden])

        # Code predictor stacked codec-embedding table, stored in talker_hidden
        # space: (num_code_groups - 1, cp_vocab, hidden). Same weights as the
        # code predictor's ``stacked_codec_embedding`` (exposed as its
        # ``codec_embeddings`` graph output).
        num_extra_groups = (tts.num_code_groups if tts else 16) - 1
        cp = tts.code_predictor if tts else None
        cp_vocab = cp.vocab_size if cp else 2048
        self.stacked_codec_embedding = nn.Parameter([num_extra_groups, cp_vocab, hidden])
        self._num_extra_groups = num_extra_groups

    def forward(
        self,
        op: OpBuilder,
        frame_codes: ir.Value,
        text_embed: ir.Value,
    ):
        """Compute ``codec_sum + text_embed`` from raw frame codes.

        Args:
            frame_codes: (batch, num_code_groups) int64 previous-frame codes.
            text_embed: (batch, 1, hidden) float trailing-text/pad embedding.

        Returns:
            inputs_embeds: (batch, 1, hidden) float.
        """
        # code_0 → talker codec embedding. Keep the length-1 group axis so the
        # result is (batch, 1, hidden) and broadcasts cleanly against text.
        code_0 = op.Gather(frame_codes, [0], axis=1)  # (batch, 1)
        codec_sum = op.Gather(self.codec_embedding, code_0, axis=0)  # (batch, 1, hidden)

        # codes[1..num_extra_groups] → per-group code predictor codec embedding.
        for i in range(self._num_extra_groups):
            code_i = op.Gather(frame_codes, [i + 1], axis=1)  # (batch, 1)
            table_i = op.Gather(self.stacked_codec_embedding, [i], axis=0)
            table_i = op.Squeeze(table_i, [0])  # (cp_vocab, hidden)
            emb_i = op.Gather(table_i, code_i, axis=0)  # (batch, 1, hidden)
            codec_sum = op.Add(codec_sum, emb_i)

        return op.Add(codec_sum, text_embed)


# ---------------------------------------------------------------------------
# Talker prefill embedder (pre-embedding component)
# ---------------------------------------------------------------------------

# Text-side special token IDs (shared text vocab).
_TTS_BOS_ID = 151672
_TTS_EOS_ID = 151673
_TTS_PAD_ID = 151671
# Codec prefill token IDs for the DEFAULT path (language="Auto", no speaker,
# no instruct). N = 5: [nothink, think_bos, think_eos, pad, bos].
_CODEC_NOTHINK_ID = 2155
_CODEC_THINK_BOS_ID = 2156
_CODEC_THINK_EOS_ID = 2157
_CODEC_PAD_ID = 2148
_CODEC_BOS_ID = 2149
_AUTO_CODEC_PREFILL_IDS = [
    _CODEC_NOTHINK_ID,
    _CODEC_THINK_BOS_ID,
    _CODEC_THINK_EOS_ID,
    _CODEC_PAD_ID,
    _CODEC_BOS_ID,
]


class Qwen3TTSTalkerPrefillEmbedder(nn.Module):
    r"""Materializes the talker PREFILL + trailing-text construction in ONNX.

    In the reference generation loop (see
    ``examples/qwen3_tts.py::generate_codes``) two embedding sequences are built
    once, up front, from the already-tokenized prompt ids. This component folds
    that construction into a single graph so a generic runtime loop can drive
    the talker without any Qwen3-TTS-specific slicing/interleaving logic.

    Scope: the DEFAULT path only — ``language="Auto"`` (codec prefill ids
    ``[nothink, think_bos, think_eos, pad, bos]``, so ``N = 5``), NO instruct
    and NO speaker embedding. Speaker/language/instruct branches are a
    documented follow-up and are intentionally not implemented here.

    The prompt is tokenized (by the runtime) as::

        <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n

    giving ``text_ids`` with layout: ``[0:3]`` role prefix, ``[3]`` first text
    token, ``[4:-5]`` remaining text, ``[-5:]`` end role (discarded).

    Two sequences are produced (matching ``generate_codes`` exactly):

    - ``prefill_embeds`` = ``role(3) + codec_text_pairs(N-1) + first_text_codec(1)``
      where ``codec_text_pairs = (tts_pad*(N-2) ++ tts_bos) + codec_prefill[:-1]``
      and ``first_text_codec = text_embeds[:, 3:4] + codec_prefill[:, -1:]``.
      Length is constant: ``3 + (N-1) + 1 = 3 + 4 + 1 = 8``.
    - ``trailing_text_embeds`` = ``text_embeds[:, 4:-5] ++ tts_eos``.
      Length is ``(text_len - 9) + 1 = text_len - 8``.

    Weights are the SAME tables as the ``embedding`` model (text embedding +
    projection, codec embedding); they are shared, not re-quantized. Only the
    Slice/Concat/Add/Tile interleaving lives here; the codec prefill ids and the
    tts special ids are graph constants.

    Inputs:
      - ``text_ids``: (batch, text_len) int64 — already-tokenized prompt ids.

    Outputs:
      - ``prefill_embeds``: (batch, 8, hidden) float.
      - ``trailing_text_embeds``: (batch, text_len - 8, hidden) float.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        tts = config.tts
        text_hidden = tts.text_hidden_size if tts else 2048
        text_vocab = tts.text_vocab_size if tts else 151936
        hidden = config.hidden_size

        # Same tables as the embedding model (shared weights, routed in
        # preprocess_weights). Text embedding + ResizeMLP projection.
        self.text_embedding = Embedding(text_vocab, text_hidden)
        self.text_projection_fc1 = Linear(text_hidden, text_hidden, bias=True)
        self.text_projection_fc2 = Linear(text_hidden, hidden, bias=True)
        # Codec embedding (audio codes → hidden).
        self.codec_embedding = Embedding(config.vocab_size, hidden)

    def _text_path(self, op: OpBuilder, ids: ir.Value) -> ir.Value:
        """Embed *ids* through the text embedding + projection (fc1 → SiLU → fc2)."""
        embeds = self.text_embedding(op, ids)
        embeds = self.text_projection_fc1(op, embeds)
        embeds = op.Mul(embeds, op.Sigmoid(embeds))  # SiLU
        return self.text_projection_fc2(op, embeds)

    def forward(self, op: OpBuilder, text_ids: ir.Value):
        """Build ``prefill_embeds`` and ``trailing_text_embeds`` from ``text_ids``.

        Args:
            text_ids: (batch, text_len) int64 already-tokenized prompt ids.

        Returns:
            Tuple ``(prefill_embeds, trailing_text_embeds)``, both
            (batch, seq, hidden) float.
        """
        num_prefill = len(_AUTO_CODEC_PREFILL_IDS)  # N = 5

        # Projected text embeds for every prompt token.
        all_text_embeds = self._text_path(op, text_ids)  # (B, L, H)

        # TTS special token embeds (bos, eos, pad) through the text path.
        special_ids = op.Unsqueeze(
            op.Constant(value_ints=[_TTS_BOS_ID, _TTS_EOS_ID, _TTS_PAD_ID]), [0]
        )  # (1, 3)
        special_embeds = self._text_path(op, special_ids)  # (1, 3, H)
        tts_bos = op.Slice(
            special_embeds, [0], [1], [1]
        )  # (1, 1, H)
        tts_eos = op.Slice(special_embeds, [1], [2], [1])
        tts_pad = op.Slice(special_embeds, [2], [3], [1])

        # Codec prefill embeds over the constant Auto-path ids.
        codec_ids = op.Unsqueeze(
            op.Constant(value_ints=_AUTO_CODEC_PREFILL_IDS), [0]
        )  # (1, N)
        codec_prefill = self.codec_embedding(op, codec_ids)  # (1, N, H)

        # Role prefix: first 3 text tokens (pure text, no codec overlay).
        role = op.Slice(all_text_embeds, [0], [3], [1])  # (B, 3, H)

        # Text side of codec prefill: tts_pad * (N-2) ++ tts_bos.
        text_pad_rep = op.Tile(
            tts_pad, op.Constant(value_ints=[1, num_prefill - 2, 1])
        )  # (1, N-2, H)
        text_side = op.Concat(text_pad_rep, tts_bos, axis=1)  # (1, N-1, H)
        codec_side = op.Slice(codec_prefill, [0], [num_prefill - 1], [1])  # (1, N-1, H)
        codec_text_pairs = op.Add(text_side, codec_side)  # (1, N-1, H)

        # First text token + last codec token (bos).
        first_text = op.Slice(all_text_embeds, [3], [4], [1])  # (B, 1, H)
        codec_bos = op.Slice(
            codec_prefill, [num_prefill - 1], [num_prefill], [1]
        )  # (1, 1, H)
        first_text_codec = op.Add(first_text, codec_bos)  # (B, 1, H)

        prefill_embeds = op.Concat(
            role, codec_text_pairs, first_text_codec, axis=1
        )  # (B, 8, H)

        # Trailing text: remaining tokens [4:-5] ++ tts_eos.
        remaining_text = op.Slice(all_text_embeds, [4], [-5], [1])  # (B, L-9, H)
        trailing_text_embeds = op.Concat(remaining_text, tts_eos, axis=1)  # (B, L-8, H)

        return prefill_embeds, trailing_text_embeds


# ---------------------------------------------------------------------------
# Speaker encoder model (model 4 of 4)
# ---------------------------------------------------------------------------


class Qwen3TTSSpeakerEncoderModel(nn.Module):
    """ECAPA-TDNN speaker encoder for voice cloning.

    Input: mel spectrogram (batch, time, mel_dim=128)
    Output: speaker embedding (batch, enc_dim)

    HuggingFace class: ``Qwen3TTSSpeakerEncoder``
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        tts = config.tts
        se = tts.speaker_encoder if tts else None
        self.encoder = SpeakerEncoder(
            config,
            mel_dim=se.mel_dim if se else 128,
            enc_dim=se.enc_dim if se else 1024,
            enc_channels=se.enc_channels if se else None,
            enc_kernel_sizes=se.enc_kernel_sizes if se else None,
            enc_dilations=se.enc_dilations if se else None,
            enc_attention_channels=se.enc_attention_channels if se else 128,
            enc_res2net_scale=se.enc_res2net_scale if se else 8,
            enc_se_channels=se.enc_se_channels if se else 128,
        )

    def forward(self, op: OpBuilder, mel_input: ir.Value):
        """Extract speaker embedding from mel spectrogram.

        Args:
            mel_input: (batch, time, mel_dim) mel spectrogram.

        Returns:
            speaker_embedding: (batch, enc_dim).
        """
        return self.encoder(op, mel_input)


# ---------------------------------------------------------------------------
# Composite model (wires sub-models together)
# ---------------------------------------------------------------------------


class Qwen3TTSForConditionalGeneration(nn.Module):
    """Qwen3-TTS composite model for full 4-model TTS pipeline.

    Contains:
      - ``talker``: Talker decoder (inputs_embeds → logits + hidden)
      - ``code_predictor``: Code predictor (hidden → 15 code groups)
      - ``embedding``: Text + codec embedding model
      - ``speaker_encoder``: ECAPA-TDNN speaker encoder

    HuggingFace class: ``Qwen3TTSForConditionalGeneration``
    """

    default_task: str = "tts"
    category: str = "Audio"
    config_class: type = ArchitectureConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.talker = Qwen3TTSTalkerDecoderModel(config)
        self.code_predictor = Qwen3TTSCodePredictorModel(config)
        self.embedding = Qwen3TTSEmbeddingModel(config)
        self.talker_step_embedder = Qwen3TTSTalkerStepEmbedder(config)
        self.talker_prefill_embedder = Qwen3TTSTalkerPrefillEmbedder(config)
        # Speaker encoder is optional — not all TTS models include it
        tts = config.tts
        if tts and tts.speaker_encoder:
            self.speaker_encoder = Qwen3TTSSpeakerEncoderModel(config)
        else:
            self.speaker_encoder = None

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        """Not used directly — TTSTask builds each sub-model separately."""
        raise NotImplementedError("Use TTSTask for 4-model export")

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route HuggingFace weights to the 4 sub-models.

        HF weight structure:
          - ``talker.model.layers.*`` → ``talker.layers.*``
          - ``talker.model.norm.*`` → ``talker.norm.*``
          - ``talker.codec_head.*`` → ``talker.lm_head.*``
          - ``talker.model.text_embedding.*`` → ``embedding.text_embedding.*``
          - ``talker.text_projection.*`` → ``embedding.text_projection_fc*``
          - ``talker.model.codec_embedding.*`` → ``embedding.codec_embedding.*``
          - ``talker.code_predictor.*`` → ``code_predictor.*``
          - ``speaker_encoder.*`` → ``speaker_encoder.encoder.*``
        """
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Speaker encoder weights
            if key.startswith("speaker_encoder."):
                cleaned[f"speaker_encoder.encoder.{key[len('speaker_encoder.') :]}"] = value
                continue

            # Strip talker. prefix
            if not key.startswith("talker."):
                continue
            inner = key[len("talker.") :]

            # Code predictor weights
            if inner.startswith("code_predictor."):
                cp_key = inner[len("code_predictor.") :]
                self._route_code_predictor_weight(cleaned, cp_key, value)
                continue

            # Text embedding → embedding model (+ shared with prefill embedder)
            if inner.startswith("model.text_embedding."):
                emb_key = inner[len("model.") :]  # text_embedding.*
                cleaned[f"embedding.{emb_key}"] = value
                cleaned[f"talker_prefill_embedder.{emb_key}"] = value
                continue

            # Text projection → embedding model (+ shared with prefill embedder)
            if inner.startswith("text_projection."):
                proj_key = inner[len("text_projection.") :]
                # linear_fc1.weight → text_projection_fc1.weight
                proj_key = proj_key.replace("linear_fc1.", "text_projection_fc1.")
                proj_key = proj_key.replace("linear_fc2.", "text_projection_fc2.")
                cleaned[f"embedding.{proj_key}"] = value
                cleaned[f"talker_prefill_embedder.{proj_key}"] = value
                continue

            # Codec embedding: talker.model.codec_embedding.weight
            if inner.startswith("model.codec_embedding."):
                emb_key = inner.replace("model.codec_embedding.", "codec_embedding.")
                cleaned[f"embedding.{emb_key}"] = value
                # Share the same table with the talker step embedder so it can
                # materialize codec_sum in-graph (talker.codec_embed(code_0)).
                if emb_key == "codec_embedding.weight":
                    cleaned["talker_step_embedder.codec_embedding"] = value
                    # Also share with the prefill embedder's codec table.
                    cleaned[f"talker_prefill_embedder.{emb_key}"] = value
                continue

            # Codec head → talker lm_head
            if inner.startswith("codec_head."):
                lm_key = inner.replace("codec_head.", "lm_head.")
                cleaned[f"talker.{lm_key}"] = value
                continue

            # Decoder layers and norm → talker
            if inner.startswith("model.layers."):
                cleaned[f"talker.{inner[len('model.') :]}"] = value
                continue
            if inner.startswith("model.norm."):
                cleaned[f"talker.{inner[len('model.') :]}"] = value
                continue
            if inner.startswith("model.rotary_emb."):
                cleaned[f"talker.{inner[len('model.') :]}"] = value
                continue

        return self.finalize_stacked_weights(cleaned)

    def _route_code_predictor_weight(
        self,
        cleaned: dict[str, torch.Tensor],
        cp_key: str,
        value: torch.Tensor,
    ) -> None:
        """Route a single code predictor weight to the correct parameter.

        HF code predictor has:
          - ``model.codec_embedding.{0..14}.weight`` → stack into
            ``stacked_codec_embedding``
          - ``lm_head.{0..14}.weight`` → stack into ``stacked_lm_head``
          - ``small_to_mtp_projection.weight/bias``
          - ``model.layers.{i}.*``, ``model.norm.*``
        """
        # Projection
        if cp_key.startswith("small_to_mtp_projection."):
            cleaned[f"code_predictor.{cp_key}"] = value
            return

        # Decoder layers and norm
        if cp_key.startswith("model.layers."):
            cleaned[f"code_predictor.{cp_key[len('model.') :]}"] = value
            return
        if cp_key.startswith("model.norm."):
            cleaned[f"code_predictor.{cp_key[len('model.') :]}"] = value
            return
        if cp_key.startswith("model.rotary_emb."):
            cleaned[f"code_predictor.{cp_key[len('model.') :]}"] = value
            return

        # Per-step codec embeddings: model.codec_embedding.{i}.weight
        # These will be collected and stacked in _finalize_code_predictor_weights
        if cp_key.startswith("model.codec_embedding."):
            cleaned[f"code_predictor._codec_emb.{cp_key[len('model.codec_embedding.') :]}"] = (
                value
            )
            return

        # Per-step lm_heads: lm_head.{i}.weight
        if cp_key.startswith("lm_head."):
            cleaned[f"code_predictor._lm_head.{cp_key[len('lm_head.') :]}"] = value
            return

    @staticmethod
    def finalize_stacked_weights(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Stack per-step code predictor embeddings and lm_heads.

        Called by preprocess_weights to merge individual per-step
        weights into stacked tensors. Also initializes missing
        small_to_mtp_projection as identity when dims match.
        """
        # Collect per-step codec embeddings
        codec_emb_weights: dict[int, torch.Tensor] = {}
        lm_head_weights: dict[int, torch.Tensor] = {}
        remaining: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            if key.startswith("code_predictor._codec_emb."):
                # code_predictor._codec_emb.{i}.weight → index i
                suffix = key[len("code_predictor._codec_emb.") :]
                parts = suffix.split(".")
                idx = int(parts[0])
                codec_emb_weights[idx] = value
            elif key.startswith("code_predictor._lm_head."):
                # code_predictor._lm_head.{i}.weight → index i
                suffix = key[len("code_predictor._lm_head.") :]
                parts = suffix.split(".")
                idx = int(parts[0])
                lm_head_weights[idx] = value
            else:
                remaining[key] = value

        # Stack codec embeddings: (num_groups-1, cp_vocab, talker_hidden).
        # The code predictor exposes this table via `op.Identity(stacked_codec_embedding)`
        # as its `codec_embeddings` graph output, and onnx-ir folds the Identity so
        # the sole initializer takes the OUTPUT name `codec_embeddings` (unprefixed) —
        # unlike every other code_predictor initializer. Key the stacked weight to that
        # exact initializer name so it is applied (apply_weights routes an unprefixed
        # name to the only component that declares it).
        if codec_emb_weights:
            num_groups = max(codec_emb_weights.keys()) + 1
            stacked = torch.stack([codec_emb_weights[i] for i in range(num_groups)])
            remaining["codec_embeddings"] = stacked
            # Share the same stacked table with the talker step embedder so it
            # can materialize codec_sum in-graph (Σ cp_codec_weights[i][code]).
            remaining["talker_step_embedder.stacked_codec_embedding"] = stacked

        # Stack lm_heads: (num_groups-1, cp_vocab, cp_hidden)
        if lm_head_weights:
            num_groups = max(lm_head_weights.keys()) + 1
            stacked = torch.stack([lm_head_weights[i] for i in range(num_groups)])
            remaining["code_predictor.stacked_lm_head"] = stacked

        return remaining
