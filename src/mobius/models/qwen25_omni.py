"""Qwen2.5-Omni: Multimodal model with audio + vision + text.

Architecture (Thinker only):
  - Audio encoder: Conv1d x2 → sinusoidal PE → 32 encoder layers → AvgPool → proj
  - Vision encoder: Conv3d patch embed → 32 ViT blocks → patch merger
  - Fusion: Audio/vision features replace placeholder token positions
  - Text decoder: Qwen2 (no QK norm) + MRoPE

Reference: https://huggingface.co/Qwen/Qwen2.5-Omni-7B
HuggingFace class: Qwen2_5OmniForConditionalGeneration
"""

from __future__ import annotations

import dataclasses

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
from mobius.components._conv import Conv1d
from mobius.components._qwen25_omni_audio import Qwen25OmniAudioEncoderLayer
from mobius.components import Qwen25VLVisionModel



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


class Qwen25OmniAudioEncoder(nn.Module):
    """Qwen25-Omni audio encoder
    Converts mel spectrogram to audio feature embeddings:
      mel (batch, num_mel_bins, seq_len)
      -> 2x Conv1d with GELU
      -> sinusoidal position embeddings
      -> N bidirectional encoder layers
      -> AvgPool1d (2x downsample)
      -> LayerNorm (ln_post)
      -> Linear proj (d_model -> output_dim)

    Output: (batch, out_seq_len, output_dim)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        audio = config.audio
        assert audio is not None

        d_model = audio.d_model or 1280
        num_mel_bin = audio.num_mel_bins or 128
        encoder_layers = audio.encoder_layers or 32
        encoder_heads = audio.encoder_attention_heads or 20
        encoder_ffn = audio.encoder_ffn_dim or 3584
        max_source_positions = audio.max_source_positions or 1500
        n_window = audio.n_window or 100
        output_dim = audio.output_dim or 3584

        # 2x Conv1d: mel -> d_model with GELU between them
        self.conv1 = Conv1d(
            num_mel_bin,
            d_model,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Sinusoidal positional embeddings (frozen)
        pe_data = _sinusoidal_position_embedding(max_source_positions, d_model)
        self.positional_embedding = nn.Parameter(
            [max_source_positions, d_model],
            name="positional_embedding.positional_embedding",
            data=ir.tensor(pe_data),
        )


        # Encoder transformer layers
        self.layers = nn.ModuleList(
            [
                Qwen25OmniAudioEncoderLayer(d_model, encoder_heads, encoder_ffn)
                for _ in range(encoder_layers)
            ]
        )

        # Post-encoder normalization
        self.ln_post = LayerNorm(d_model)

        # Output projection: d_model -> output_dim
        self.proj = Linear(d_model, output_dim)

    def forward(self, op: builder.OpBuilder, input_features: ir.Value):
        """Encode mel spectrogram to audio features.

        Args:
            input_features: (batch, num_mel_bins, seq_len) mel spectrogram

        Returns:
            audio_features: (batch, out_seq_len, output_dim)
        """

        # 2X Conv1d with GELU: (batch, mel, seq) -> (batch, d_model, seq//2)
        hidden_states = op.Gelu(self.conv1(op, input_features))
        hidden_states = op.Gelu(self.conv2(op, hidden_states))

        # Transpose to (batch, seq//2, d_model) for transformer layers
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])

        # Add sinusoidal positional embeddings
        seq_len = op.Shape(hidden_states, start=1, end=2)
        pe_slice = op.Slice(
            self.positional_embedding,
            op.Constant(value_ints=[0]),
            seq_len,
            op.Constant(value_ints=[0])
        )

        hidden_states = op.Add(hidden_states, pe_slice)

        # Encoder layer
        for layer in self.layers:
            hidden_states = layer(op, hidden_states)

        # AvgPool1d(kernel=2, stride=2): halves sequence length
        # Transpose to (batch, d_model, seq) for pooling, then back
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = op.AveragePool(hidden_states, kernel_shape=[2], strides=[2])
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])

        # ln_post once, then proj
        hidden_states = self.ln_post(op, hidden_states)
        hidden_states = self.proj(op, hidden_states)

        return hidden_states


class Qwen25OmniVisionEncoder(nn.Module):
    """
    Qwen2.5-Omni vision encoder - reuses Qwen2.5-VL ViT
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None

        self.visual = Qwen25VLVisionModel(
            depth=vc.num_hidden_layers or 32,
            hidden_size = vc.hidden_size or 1280,
            intermediate_size = vc.intermediate_size or 3420,
            num_heads = vc.num_attention_heads or 16,
            patch_size = vc.patch_size or 14,
            temporal_patch_size = vc.temporal_patch_size or 2,
            in_channels = vc.in_channels or 3,
            out_hidden_size = vc.out_hidden_size or 3584,
            spatial_merge_size = vc.spatial_merge_size or 2,
            fullatt_block_indexes = vc.fullatt_block_indexes or (7, 15, 23, 31),
            window_size = vc.window_size or 112
        )

    def forward(self, op, pixel_values, image_grid_thw):
        return self.visual(op, pixel_values, image_grid_thw)

class Qwen25OmniEmbeddingModel(nn.Module):
    """Fuses text embedding with audio and image feature
    
    Replaces audio_token_id positions with audio features,
    and image_token_id positions with image features.

    Inputs:
        input_ids: (batch, seq_len)
        audio_features: (num_audio_tokens, hidden_size)
        image_features: (num_image_tokens, hidden_size)

    Output:
        inputs_embeds: (batch, seq_len, hidden_size)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        # Token IDs from Qwen2.5-Omni config
        audio = config.audio
        self._audio_token_id = audio.audio_token_id if audio else 151646
        self._image_token_id = audio.image_token_id or 151655

    
    def forward(self, op, inputs_ids, audio_features, image_features):
        inputs_embeds = self.embed_tokens(op, inputs_ids)

        # Fuse audio features at audio token positions
        inputs_embeds = self._replace_tokens(
            op, inputs_embeds, inputs_ids, audio_features, self._audio_token_id
        )

        # Fuse image features at image token position
        inputs_embeds = self._replace_tokens(
            op, inputs_embeds, inputs_ids, image_features, self._image_token_ids
        )

        return input_embeds

    def _replace_tokens(self, op, inputs_embeds, input_ids, features, token_id):
        """Replace token positions with encoder features (masked_scatter equivalent)."""
        mask = op.Equal(input_ids, op.Constant(value_int=token_id))
        mask_3d = op.Unsqueeze(mask, [-1])

        # Pad with zero row for safety (text-only case: no features)
        feature_dim = op.Shape(features, start=1, end=2)
        zero_shape = op.Concat(op.Constant(value_ints=[1]), feature_dim, axis=0)
        zero_row = op.ConstantOfShape(zero_shape, value=ir.tensor(np.zeros(1, dtype=np.float32)))
        padded = op.Concat(zero_row, features, axis=0)

        # CumSum to get gather indices
        mask_int = op.Cast(mask, to=7)
        flat = op.Reshape(mask_int, op.Constant(value_ints=[-1]))
        indices = op.CumSum(flat, op.Constant(value_int=0))
        indices = op.Mul(indices, flat)
        indices = op.Reshape(indices, op.Shape(input_ids))

        gathered = op.Gather(padded, indices, axis=0)
        return op.Where(mask_3d, gathered, inputs_embeds)


class Qwen25OmniDecoderModel(nn.Module):
    """Qwen2.5-Omni text decoder: inputs_embeds → logits + KV cache.

    Standard Qwen2 decoder with MRoPE (3D position_ids).
    No QK norm (unlike Qwen3-ASR which uses attn_qk_norm=True).
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

    def forward(self, op, inputs_embeds, attention_mask, position_ids, past_key_values=None):
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
        )

class Qwen25OmniThinkerForConditionalGeneration(nn.Module):
    pass


