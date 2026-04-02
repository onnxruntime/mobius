# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""VibeVoice ASR model implementation.

Architecture: microsoft/VibeVoice-ASR-HF

A speech recognition model combining two causal 1D CNN audio encoders
(acoustic + semantic tokenizers) with a Qwen2 language model backbone:

1. ``acoustic_tokenizer_encoder``: causal ConvNeXt CNN → latents (B, T, 64)
2. ``semantic_tokenizer_encoder``: causal ConvNeXt CNN → latents (B, T, 128)
3. ``multi_modal_projector``: 2-path MLP, outputs summed → (N_tokens, hidden)
4. ``language_model``: Qwen2 decoder (hidden_size=3584, 28 layers)

The three sub-models produced are:

- ``audio_tower``: raw waveform (B, 1, T) → audio features (N_tokens, lm_hidden)
- ``embedding``: input_ids + audio_features → inputs_embeds
- ``decoder``: inputs_embeds → logits + KV cache
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript import nn

from onnxscript._internal import builder

from mobius._configs import AudioTokenizerEncoderConfig, VibeVoiceAsrConfig
from mobius.components import Embedding, Linear, RMSNorm
from mobius.components._common import create_attention_bias
from mobius.components._decoder import DecoderLayer
from mobius.components._rotary_embedding import initialize_rope


class _CausalConv1d(nn.Module):
    """Causal 1D convolution with left-only padding.

    Pads ``left_pad = (kernel_size - 1) * dilation - (stride - 1)`` samples
    on the left so the output is causal (no look-ahead).

    Matches HF ``VibeVoiceAcousticTokenizerCausalConv1d``.

    Weight names: ``conv.weight``, ``conv.bias``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        # Left causal padding: eliminates look-ahead
        self._left_pad = (kernel_size - 1) * dilation - (stride - 1)
        self._stride = stride
        self._dilation = dilation
        self._groups = groups
        self._kernel_size = kernel_size
        # Weight: (out_channels, in_channels // groups, kernel_size)
        self.conv = nn.Parameter([out_channels, in_channels // groups, kernel_size])
        self.bias = nn.Parameter([out_channels])

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        """Apply causal padding then Conv1d.

        Args:
            x: (batch, channels, seq_len)

        Returns:
            (batch, out_channels, out_seq_len)
        """
        if self._left_pad > 0:
            # ONNX Pad pads format for 3D input (N,C,L): [N_beg,C_beg,L_beg, N_end,C_end,L_end]
            pads = op.Constant(value_ints=[0, 0, self._left_pad, 0, 0, 0])
            x = op.Pad(x, pads)

        # Conv1d: kernel_shape=[k], strides=[s], dilations=[d], group=g
        return op.Conv(
            x,
            self.conv,
            self.bias,
            kernel_shape=[self._kernel_size],
            strides=[self._stride],
            dilations=[self._dilation],
            group=self._groups,
            pads=[0, 0],
        )


class _FeedForward(nn.Module):
    """2-layer FFN: linear1 → GELU → linear2.

    Matches HF ``VibeVoiceAcousticTokenizerFeedForward``.
    Weight names: linear1.weight, linear1.bias, linear2.weight, linear2.bias
    """

    def __init__(self, hidden_size: int, ffn_hidden: int):
        super().__init__()
        self.linear1 = Linear(hidden_size, ffn_hidden, bias=True)
        self.linear2 = Linear(ffn_hidden, hidden_size, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        """Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            (batch, seq_len, hidden_size)
        """
        x = self.linear1(op, x)
        x = op.Gelu(x)
        return self.linear2(op, x)


class _ConvNeXt1DBlock(nn.Module):
    """ConvNeXt-style 1D block with depthwise mixer and FFN.

    Structure (inputs in (batch, channels, seq_len) format):
      mixer:
        x → transpose → RMSNorm → transpose → depthwise_CausalConv1d
          → scale by gamma (channel-wise) → residual_add
      ffn:
        x → transpose → RMSNorm → linear1 → GELU → linear2 → transpose
          → scale by ffn_gamma (channel-wise) → residual_add

    Matches HF ``VibeVoiceAcousticTokenizerConvNext1dLayer``.

    Weight names:
      norm.weight, mixer.conv.weight, mixer.bias,
      gamma, ffn_norm.weight, ffn.linear1.weight, ffn.linear1.bias,
      ffn.linear2.weight, ffn.linear2.bias, ffn_gamma
    """

    def __init__(self, enc_config: AudioTokenizerEncoderConfig, hidden_size: int):
        super().__init__()
        self._hidden_size = hidden_size

        self.norm = RMSNorm(hidden_size, eps=enc_config.rms_norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, eps=enc_config.rms_norm_eps)

        # Depthwise causal conv: groups=hidden_size (one filter per channel)
        self.mixer = _CausalConv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=enc_config.kernel_size,
            groups=hidden_size,
        )

        # Layer scale parameters: trainable scalar per channel, initialized to layer_scale_init_value
        self.gamma = nn.Parameter(
            [hidden_size],
            data=ir.Tensor(
                np.full((hidden_size,), enc_config.layer_scale_init_value, dtype=np.float32)
            ),
        )
        self.ffn_gamma = nn.Parameter(
            [hidden_size],
            data=ir.Tensor(
                np.full((hidden_size,), enc_config.layer_scale_init_value, dtype=np.float32)
            ),
        )

        # FFN: linear1 → GELU → linear2
        ffn_hidden = enc_config.ffn_expansion * hidden_size
        self.ffn = _FeedForward(hidden_size, ffn_hidden)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        """Forward pass for one ConvNeXt block.

        Args:
            x: (batch, channels, seq_len)

        Returns:
            (batch, channels, seq_len) — same shape
        """
        residual = x

        # --- Mixer branch ---
        # Transpose: (B, C, L) → (B, L, C) for RMSNorm (normalizes last dim)
        xt = op.Transpose(x, perm=[0, 2, 1])
        xt = self.norm(op, xt)  # (B, L, C)
        # Transpose back: (B, L, C) → (B, C, L)
        xt = op.Transpose(xt, perm=[0, 2, 1])
        xt = self.mixer(op, xt)  # depthwise CausalConv1d: (B, C, L)

        # Channel-wise scaling: gamma shape (C,) → (1, C, 1) for broadcasting
        gamma = op.Reshape(self.gamma, op.Constant(value_ints=[1, self._hidden_size, 1]))
        xt = op.Mul(xt, gamma)
        x = op.Add(residual, xt)

        # --- FFN branch ---
        residual = x
        # Transpose: (B, C, L) → (B, L, C) for FFN (operates on last dim)
        xf = op.Transpose(x, perm=[0, 2, 1])
        xf = self.ffn_norm(op, xf)  # (B, L, C)
        xf = self.ffn(op, xf)  # (B, L, C) → (B, L, C)
        # Transpose back: (B, L, C) → (B, C, L)
        xf = op.Transpose(xf, perm=[0, 2, 1])

        # Channel-wise scaling by ffn_gamma
        ffn_gamma = op.Reshape(
            self.ffn_gamma, op.Constant(value_ints=[1, self._hidden_size, 1])
        )
        xf = op.Mul(xf, ffn_gamma)
        return op.Add(residual, xf)


class _AudioEncoderStem(nn.Module):
    """Encoder stem: CausalConv1d(channels→num_filters) + N ConvNeXt blocks.

    Matches HF ``VibeVoiceAcousticTokenizerEncoderStem``.
    Weight names:
      conv.conv.weight, conv.bias
      stage.0.{norm,mixer,gamma,ffn_norm,ffn,ffn_gamma}
    """

    def __init__(self, enc_config: AudioTokenizerEncoderConfig):
        super().__init__()
        # Stem conv: input channels → num_filters, no stride, kernel=kernel_size
        self.conv = _CausalConv1d(
            in_channels=enc_config.channels,
            out_channels=enc_config.num_filters,
            kernel_size=enc_config.kernel_size,
        )
        # ConvNeXt blocks in stem, all at num_filters channels
        self.stage = nn.ModuleList(
            [
                _ConvNeXt1DBlock(enc_config, enc_config.num_filters)
                for _ in range(enc_config.depths[0])
            ]
        )

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        """Args:
            x: (batch, channels=1, num_samples)
        Returns:
            (batch, num_filters, num_samples)
        """
        x = self.conv(op, x)  # (B, num_filters, L)
        for block in self.stage:
            x = block(op, x)
        return x


class _AudioEncoderLayer(nn.Module):
    """One encoder stage: strided CausalConv1d (downsampling) + N ConvNeXt blocks.

    stage_idx is 0-based; output channels = num_filters * 2^(stage_idx+1).

    Matches HF ``VibeVoiceAcousticTokenizerEncoderLayer``.
    Weight names:
      conv.conv.weight, conv.bias
      stage.0.{…}, stage.1.{…}, …
    """

    def __init__(self, enc_config: AudioTokenizerEncoderConfig, stage_idx: int):
        super().__init__()
        stride = enc_config.downsampling_ratios[stage_idx]
        # Input channels: num_filters * 2^stage_idx (output of previous stage)
        # Output channels: num_filters * 2^(stage_idx+1)
        in_ch = enc_config.num_filters * (2**stage_idx)
        out_ch = enc_config.num_filters * (2 ** (stage_idx + 1))

        # Strided conv for downsampling: kernel = stride * 2 (per HF source)
        self.conv = _CausalConv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=stride * 2,
            stride=stride,
        )
        # ConvNeXt blocks at output channel count
        # depths[stage_idx+1]: first depth entry is for stem
        self.stage = nn.ModuleList(
            [
                _ConvNeXt1DBlock(enc_config, out_ch)
                for _ in range(enc_config.depths[stage_idx + 1])
            ]
        )

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        """Args:
            x: (batch, in_channels, seq_len)
        Returns:
            (batch, out_channels, seq_len // stride)
        """
        x = self.conv(op, x)  # Downsampling strided CausalConv1d
        for block in self.stage:
            x = block(op, x)
        return x


class AudioTokenizerEncoder(nn.Module):
    """Full audio tokenizer encoder: raw waveform → latent frame embeddings.

    Architecture:
      stem → 6 encoder layers (strided) → head CausalConv1d → permute

    Input:  ``(batch, 1, num_samples)`` raw audio at 24 kHz
    Output: ``(batch, num_frames, hidden_size)``
            where num_frames = num_samples // hop_length

    Matches HF ``VibeVoiceAcousticTokenizerEncoderModel``.

    Weight names:
      stem.conv.conv.weight, stem.conv.bias
      stem.stage.{0..N-1}.*
      conv_layers.{0..5}.conv.conv.weight, conv_layers.{0..5}.conv.bias
      conv_layers.{0..5}.stage.{0..M-1}.*
      head.conv.weight, head.bias
    """

    def __init__(self, enc_config: AudioTokenizerEncoderConfig):
        super().__init__()
        n_stages = len(enc_config.downsampling_ratios)

        self.stem = _AudioEncoderStem(enc_config)
        self.conv_layers = nn.ModuleList(
            [_AudioEncoderLayer(enc_config, i) for i in range(n_stages)]
        )
        # Head: final channels → hidden_size, kernel_size=config.kernel_size
        final_channels = enc_config.num_filters * (2**n_stages)
        self.head = _CausalConv1d(
            in_channels=final_channels,
            out_channels=enc_config.hidden_size,
            kernel_size=enc_config.kernel_size,
        )

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        """Encode raw waveform to latent frames.

        Args:
            x: (batch, 1, num_samples)

        Returns:
            (batch, num_frames, hidden_size)
        """
        x = self.stem(op, x)  # (B, num_filters, T)
        for layer in self.conv_layers:
            x = layer(op, x)  # Progressive downsampling
        x = self.head(op, x)  # (B, hidden_size, T//hop_total)
        # Permute: (B, C, T) → (B, T, C) to match LM convention
        return op.Transpose(x, perm=[0, 2, 1])  # (B, num_frames, hidden_size)


class VibeVoiceAsrMultiModalProjector(nn.Module):
    """Two-path MLP projector: acoustic + semantic → single LM-dimension tensor.

    Each path: linear_1 → RMSNorm → linear_2
    Output: acoustic_out + semantic_out

    Matches HF ``VibeVoiceAsrMultiModalProjector``.

    Weight names:
      acoustic_linear_1.{weight,bias}, acoustic_norm.weight,
      acoustic_linear_2.{weight,bias},
      semantic_linear_1.{weight,bias}, semantic_norm.weight,
      semantic_linear_2.{weight,bias}
    """

    def __init__(self, config: VibeVoiceAsrConfig):
        super().__init__()
        lm_hidden = config.hidden_size
        acoustic_dim = config.acoustic_encoder.hidden_size
        semantic_dim = config.semantic_encoder.hidden_size

        self.acoustic_linear_1 = Linear(acoustic_dim, lm_hidden, bias=True)
        self.acoustic_norm = RMSNorm(lm_hidden, eps=1e-6)
        self.acoustic_linear_2 = Linear(lm_hidden, lm_hidden, bias=True)

        self.semantic_linear_1 = Linear(semantic_dim, lm_hidden, bias=True)
        self.semantic_norm = RMSNorm(lm_hidden, eps=1e-6)
        self.semantic_linear_2 = Linear(lm_hidden, lm_hidden, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        acoustic_latents: ir.Value,
        semantic_latents: ir.Value,
    ) -> ir.Value:
        """Project audio latents to LM hidden dimension.

        Args:
            acoustic_latents: (batch, num_frames, acoustic_hidden)
            semantic_latents: (batch, num_frames, semantic_hidden)

        Returns:
            (batch, num_frames, lm_hidden) — sum of both paths
        """
        # Acoustic path: linear_1 → norm → linear_2
        a = self.acoustic_linear_1(op, acoustic_latents)
        a = self.acoustic_norm(op, a)
        a = self.acoustic_linear_2(op, a)

        # Semantic path: linear_1 → norm → linear_2
        s = self.semantic_linear_1(op, semantic_latents)
        s = self.semantic_norm(op, s)
        s = self.semantic_linear_2(op, s)

        return op.Add(a, s)


class VibeVoiceAsrAudioTower(nn.Module):
    """Combined audio encoding: two CNN encoders + projector.

    Takes raw waveform input and produces projected audio features
    suitable for the language model.

    Sub-modules (named to match HF weight key prefixes):
      - ``acoustic_tokenizer_encoder``: causal CNN (hidden_size=64)
      - ``semantic_tokenizer_encoder``: causal CNN (hidden_size=128)
      - ``multi_modal_projector``: 2-path MLP → summed output
    """

    def __init__(self, config: VibeVoiceAsrConfig):
        super().__init__()
        self.acoustic_tokenizer_encoder = AudioTokenizerEncoder(config.acoustic_encoder)
        self.semantic_tokenizer_encoder = AudioTokenizerEncoder(config.semantic_encoder)
        self.multi_modal_projector = VibeVoiceAsrMultiModalProjector(config)

    def forward(self, op: builder.OpBuilder, input_values: ir.Value) -> ir.Value:
        """Encode audio waveform to projected features.

        Args:
            input_values: (batch, 1, num_samples) raw audio at 24 kHz

        Returns:
            (num_audio_tokens, lm_hidden) — flattened batch×time
        """
        # Each encoder: (B, 1, T) → (B, num_frames, hidden_size)
        acoustic_latents = self.acoustic_tokenizer_encoder(op, input_values)
        semantic_latents = self.semantic_tokenizer_encoder(op, input_values)

        # Project to LM dimension: (B, num_frames, lm_hidden)
        features = self.multi_modal_projector(op, acoustic_latents, semantic_latents)

        # Flatten batch × time → (N_audio_tokens, lm_hidden) for embedding injection
        batch_time = op.Shape(features, start=0, end=2)
        feat_dim = op.Shape(features, start=2, end=3)
        # keepdims=1: result shape (1,) so Concat with feat_dim works on axis=0
        n_tokens = op.ReduceProd(batch_time, keepdims=1)
        flat_shape = op.Concat(n_tokens, feat_dim, axis=0)
        return op.Reshape(features, flat_shape)


class VibeVoiceAsrEmbeddingModel(nn.Module):
    """Text embedding with audio feature injection.

    Replaces audio token placeholders in the text embedding sequence
    with the corresponding projected audio features using a CumSum-based
    gather (same pattern as Qwen3ASREmbeddingModel).

    Weight names:
      embed_tokens.weight
    """

    def __init__(self, config: VibeVoiceAsrConfig):
        super().__init__()
        self._audio_token_id = config.audio_token_id
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        audio_features: ir.Value,
    ) -> ir.Value:
        """Embed input_ids and scatter in audio features at audio token positions.

        Args:
            input_ids: (batch, seq_len) INT64
            audio_features: (num_audio_tokens, lm_hidden) float

        Returns:
            (batch, seq_len, lm_hidden) inputs_embeds
        """
        # Text embeddings: (batch, seq_len, hidden_size)
        inputs_embeds = self.embed_tokens(op, input_ids)

        # Boolean mask where input_ids == audio_token_id
        audio_token = op.Constant(value_int=self._audio_token_id)
        is_audio = op.Equal(input_ids, audio_token)  # (batch, seq_len) bool
        # Expand to (batch, seq_len, 1) for broadcasting with embeddings
        is_audio_3d = op.Unsqueeze(is_audio, [-1])

        # Prepend a zero row to audio_features so index 0 → "no audio"
        feature_dim = op.Shape(audio_features, start=1, end=2)
        zero_row_shape = op.Concat(op.Constant(value_ints=[1]), feature_dim, axis=0)
        zero_row = op.ConstantOfShape(
            zero_row_shape,
            value=ir.tensor(np.zeros(1, dtype=np.float32)),
        )
        # padded_features: (num_audio_tokens + 1, lm_hidden)
        padded_features = op.Concat(zero_row, audio_features, axis=0)

        # CumSum gives 1-based indices into padded_features at audio positions;
        # non-audio positions are masked back to 0 (the zero padding row)
        is_audio_int = op.Cast(is_audio, to=ir.DataType.INT64)
        flat_mask = op.Reshape(is_audio_int, op.Constant(value_ints=[-1]))
        flat_indices = op.CumSum(flat_mask, op.Constant(value_int=0))
        flat_indices = op.Mul(flat_indices, flat_mask)
        # Reshape back to (batch, seq_len)
        indices = op.Reshape(flat_indices, op.Shape(input_ids))

        # Gather audio features at computed indices: (batch, seq_len, lm_hidden)
        gathered = op.Gather(padded_features, indices, axis=0)

        # Replace audio token positions with projected audio features
        return op.Where(is_audio_3d, gathered, inputs_embeds)


class VibeVoiceAsrDecoderModel(nn.Module):
    """Qwen2 text decoder for VibeVoice ASR.

    Standard decoder with 2D position_ids (not MRoPE 3D).
    Takes inputs_embeds (fused text+audio) instead of input_ids.

    Weight names (after preprocess_weights strips language_model.model./lm_head. prefixes):
      layers.N.{self_attn, mlp, input_layernorm, post_attention_layernorm}.*
      norm.weight
      lm_head.weight
    """

    def __init__(self, config: VibeVoiceAsrConfig):
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
        """Decode embeddings to logits with KV cache.

        Args:
            inputs_embeds: (batch, seq_len, hidden_size)
            attention_mask: (batch, past_seq_len + seq_len) INT64
            position_ids: (batch, seq_len) INT64 — standard 2D, not MRoPE

        Returns:
            (logits, present_key_values)
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


class VibeVoiceAsrModel(nn.Module):
    """VibeVoice ASR composite model for speech recognition.

    Contains three sub-models for the three-model ONNX split:

    - ``audio_tower``: two causal CNN encoders + projector
      (raw waveform → audio features)
    - ``embedding``: text embedding + audio feature injection
    - ``decoder``: Qwen2 text decoder with KV cache

    HuggingFace class: ``VibeVoiceAsrForConditionalGeneration``
    (microsoft/VibeVoice-ASR-HF)
    """

    default_task: str = "vibevoice-asr"
    category: str = "Speech-to-Text"
    config_class: type = VibeVoiceAsrConfig

    def __init__(self, config: VibeVoiceAsrConfig):
        super().__init__()
        self.config = config

        self.audio_tower = VibeVoiceAsrAudioTower(config)
        self.embedding = VibeVoiceAsrEmbeddingModel(config)
        self.decoder = VibeVoiceAsrDecoderModel(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        """Text-only forward: embeds input_ids and decodes (no audio injection).

        Audio injection happens via the separate embedding ONNX model.
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
        self, state_dict: dict[str, object]
    ) -> dict[str, object]:
        """Map HuggingFace weight names to ONNX module structure.

        HF layout:
          acoustic_tokenizer_encoder.*   → audio_tower.acoustic_tokenizer_encoder.*
          semantic_tokenizer_encoder.*   → audio_tower.semantic_tokenizer_encoder.*
          multi_modal_projector.*        → audio_tower.multi_modal_projector.*
          language_model.model.embed_tokens.* → embedding.embed_tokens.*
          language_model.model.layers.*  → decoder.layers.*
          language_model.model.norm.*    → decoder.norm.*
          language_model.model.rotary_emb.* → decoder.rotary_emb.*
          language_model.lm_head.*       → decoder.lm_head.*
        """
        cleaned: dict[str, object] = {}
        for key, value in state_dict.items():
            # Audio tower sub-modules
            if key.startswith((
                "acoustic_tokenizer_encoder.",
                "semantic_tokenizer_encoder.",
                "multi_modal_projector.",
            )):
                cleaned[f"audio_tower.{key}"] = value
                continue

            # Language model → decoder / embedding split
            if key.startswith("language_model."):
                rest = key[len("language_model."):]

                if rest.startswith("lm_head."):
                    cleaned[f"decoder.{rest}"] = value
                    continue

                if rest.startswith("model."):
                    inner = rest[len("model."):]
                    if inner.startswith("embed_tokens."):
                        cleaned[f"embedding.{inner}"] = value
                    elif inner.startswith(("layers.", "norm.", "rotary_emb.")):
                        cleaned[f"decoder.{inner}"] = value
                    else:
                        cleaned[key] = value
                    continue

            cleaned[key] = value

        # Weight tying: embed_tokens ↔ lm_head
        if self.config.tie_word_embeddings:
            embed_key = "embedding.embed_tokens.weight"
            lm_key = "decoder.lm_head.weight"
            if embed_key in cleaned and lm_key not in cleaned:
                cleaned[lm_key] = cleaned[embed_key]
            elif lm_key in cleaned and embed_key not in cleaned:
                cleaned[embed_key] = cleaned[lm_key]

        return cleaned
