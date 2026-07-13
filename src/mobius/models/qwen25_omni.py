# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen2.5-Omni Thinker: audio + vision + text.

Architecture (Thinker only):
  - Audio encoder: Conv1d x2 → sinusoidal PE → 32 encoder layers → AvgPool → proj
  - Vision encoder: Conv3d patch embed → 32 ViT blocks → patch merger
  - Fusion: Audio/vision features replace placeholder token positions
  - Text decoder: Qwen2 (no QK norm) + MRoPE

Reference: https://huggingface.co/Qwen/Qwen2.5-Omni-7B
HuggingFace class: Qwen2_5OmniForConditionalGeneration
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities
from mobius._configs import ArchitectureConfig
from mobius.components import (
    GatedMLP,
    Qwen25OmniAudioEncoderLayer,
    Qwen25VLPatchMerger,
    Qwen25VLVisionAttention,
    Qwen25VLVisionBlock,
    Qwen25VLVisionModel,
)
from mobius.components._common import (
    Embedding,
    LayerNorm,
    Linear,
    create_attention_bias,
)
from mobius.components._conv import Conv1d
from mobius.components._decoder import DecoderLayer
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


class Qwen25OmniAudioEncoder(nn.Module):
    """Qwen25-Omni audio encoder.

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
        self._d_model = d_model
        num_mel_bin = audio.num_mel_bins or 128
        encoder_layers = audio.encoder_layers or 32
        encoder_heads = audio.encoder_attention_heads or 20
        encoder_ffn = audio.encoder_ffn_dim or 5120
        max_source_positions = audio.max_source_positions or 1500
        output_dim = audio.output_dim or 3584

        # 2x Conv1d: mel -> d_model with GELU between them
        self.conv1 = Conv1d(num_mel_bin, d_model, kernel_size=3, padding=1)
        self.conv2 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

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

    def forward(
        self,
        op: OpBuilder,
        input_features: ir.Value,
        chunk_lengths: ir.Value,
        pool_indices: ir.Value,
    ):
        """Encode pre-chunked mel spectrograms to packed audio features.

        Args:
            input_features: (num_chunks, num_mel_bins, max_chunk_len)
            chunk_lengths: Valid mel-frame count for each chunk.
            pool_indices: Indices of the first token in each stride-2 pooling pair.

        Returns:
            audio_features: (num_audio_tokens, output_dim)
        """
        input_features = op.CastLike(input_features, self.conv1.weight)

        # Match HF's chunk padding mask before the stride-2 convolution.
        chunk_seq_len = op.Shape(input_features, start=2, end=3)
        chunk_positions = op.Range(0, op.Squeeze(chunk_seq_len, [0]), 1)
        chunk_mask = op.Less(
            op.Unsqueeze(chunk_positions, [0]),
            op.Unsqueeze(chunk_lengths, [1]),
        )
        chunk_mask = op.Unsqueeze(op.CastLike(chunk_mask, input_features), [1])

        # (num_chunks, mel, time) -> (num_chunks, d_model, ceil(time / 2))
        hidden_states = op.Mul(op.Gelu(self.conv1(op, input_features)), chunk_mask)
        hidden_states = op.Gelu(self.conv2(op, hidden_states))

        # (num_chunks, d_model, time) -> (num_chunks, time, d_model)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])

        # Add sinusoidal positional embeddings
        seq_len = op.Shape(hidden_states, start=1, end=2)
        pe_slice = op.Slice(
            self.positional_embedding,
            op.Constant(value_ints=[0]),
            seq_len,
            op.Constant(value_ints=[0]),
        )
        hidden_states = op.Add(hidden_states, pe_slice)

        # Remove per-chunk padding and derive packed-attention boundaries.
        after_conv_lengths = op.Add(op.Div(op.Sub(chunk_lengths, 1), 2), 1)
        valid_mask = op.Less(
            op.Unsqueeze(op.Range(0, op.Squeeze(seq_len, [0]), 1), [0]),
            op.Unsqueeze(after_conv_lengths, [1]),
        )
        valid_indices = op.Squeeze(op.NonZero(op.Reshape(valid_mask, [-1])), [0])
        hidden_states = op.Gather(
            op.Reshape(hidden_states, [-1, self._d_model]),
            valid_indices,
            axis=0,
        )
        cu_seqlens = op.Concat(
            op.Constant(value_ints=[0]),
            op.CumSum(after_conv_lengths, op.Constant(value_int=0)),
            axis=0,
        )

        for layer in self.layers:
            hidden_states = layer(op, hidden_states, cu_seqlens)

        # HF pools adjacent valid tokens using indices computed per original audio.
        pooled_first = op.Gather(hidden_states, pool_indices, axis=0)
        pooled_second = op.Gather(hidden_states, op.Add(pool_indices, 1), axis=0)
        hidden_states = op.Mul(
            op.Add(pooled_first, pooled_second),
            op.CastLike(0.5, hidden_states),
        )
        hidden_states = self.ln_post(op, hidden_states)
        hidden_states = self.proj(op, hidden_states)
        return hidden_states


class Qwen25OmniVisionAttention(Qwen25VLVisionAttention):
    """Qwen2.5-Omni vision attention with separate Q/K/V checkpoint weights."""

    def __init__(self, hidden_size: int, num_heads: int):
        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = Linear(hidden_size, hidden_size, bias=True)
        self.k = Linear(hidden_size, hidden_size, bias=True)
        self.v = Linear(hidden_size, hidden_size, bias=True)
        self.proj = Linear(hidden_size, hidden_size, bias=True)

    def forward(self, op, hidden_states, cu_seqlens, cos, sin):
        seq_len = op.Shape(hidden_states, start=0, end=1)
        head_shape = op.Concat(seq_len, [self.num_heads, self.head_dim], axis=0)
        q = self._apply_rotary(op, op.Reshape(self.q(op, hidden_states), head_shape), cos, sin)
        k = self._apply_rotary(op, op.Reshape(self.k(op, hidden_states), head_shape), cos, sin)
        v = op.Reshape(self.v(op, hidden_states), head_shape)

        if ep_capabilities().supports_packed_multi_head_attention:
            output = self._emit_packed_mha(op, q, k, v, cu_seqlens, seq_len)
        else:
            output = self._emit_standard_attention(op, q, k, v, cu_seqlens, seq_len)
        return self.proj(op, output)


class Qwen25OmniVisionBlock(Qwen25VLVisionBlock):
    """Qwen2.5-Omni vision block with separate attention projections."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int):
        super().__init__(hidden_size, intermediate_size, num_heads)
        self.attn = Qwen25OmniVisionAttention(hidden_size, num_heads)
        self.mlp = GatedMLP(
            hidden_size,
            intermediate_size,
            activation="silu",
            bias=True,
        )


class Qwen25OmniVisionModel(Qwen25VLVisionModel):
    """Qwen2.5-Omni vision tower using the Omni checkpoint parameter layout."""

    def __init__(
        self,
        depth: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        **kwargs,
    ):
        super().__init__(
            depth,
            hidden_size,
            intermediate_size,
            num_heads,
            **kwargs,
        )
        self.blocks = nn.ModuleList(
            [
                Qwen25OmniVisionBlock(hidden_size, intermediate_size, num_heads)
                for _ in range(depth)
            ]
        )
        self.merger = Qwen25VLPatchMerger(
            out_hidden_size=kwargs.get("out_hidden_size") or hidden_size,
            hidden_size=hidden_size,
            spatial_merge_size=kwargs.get("spatial_merge_size", 2),
        )


class Qwen25OmniVisionEncoder(nn.Module):
    """Qwen2.5-Omni vision encoder."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None

        self.visual = Qwen25OmniVisionModel(
            depth=vc.num_hidden_layers or 32,
            hidden_size=vc.hidden_size or 1280,
            intermediate_size=vc.intermediate_size or 3420,
            num_heads=vc.num_attention_heads or 16,
            patch_size=vc.patch_size or 14,
            temporal_patch_size=vc.temporal_patch_size or 2,
            in_channels=vc.in_channels or 3,
            out_hidden_size=vc.out_hidden_size or 3584,
            spatial_merge_size=vc.spatial_merge_size or 2,
            fullatt_block_indexes=vc.fullatt_block_indexes or (7, 15, 23, 31),
            window_size=vc.window_size or 112,
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value, image_grid_thw: ir.Value):
        return self.visual(op, pixel_values, image_grid_thw)


class Qwen25OmniEmbeddingModel(nn.Module):
    """Fuses text embedding with audio and image features.

    Replaces audio, image, and video placeholder tokens with encoder features.

    Inputs:
        input_ids: (batch, seq_len)
        audio_features: (num_audio_tokens, hidden_size)
        image_features: (num_image_tokens, hidden_size)
        video_features: (num_video_tokens, hidden_size)

    Output:
        inputs_embeds: (batch, seq_len, hidden_size)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )

        # Token IDs default to the values used by Qwen2.5-Omni-7B.
        audio = config.audio
        vision = config.vision
        self._audio_token_id = (audio.audio_token_id if audio else None) or 151646
        self._image_token_id = (vision.image_token_id if vision else None) or 151655
        self._video_token_id = config.video_token_id or 151656

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        audio_features: ir.Value,
        image_features: ir.Value,
        video_features: ir.Value,
    ):
        inputs_embeds = self.embed_tokens(op, input_ids)

        # Fuse audio features at audio token positions.
        inputs_embeds = self._replace_tokens(
            op, inputs_embeds, input_ids, audio_features, self._audio_token_id
        )

        # Fuse image features at image token positions.
        inputs_embeds = self._replace_tokens(
            op, inputs_embeds, input_ids, image_features, self._image_token_id
        )
        inputs_embeds = self._replace_tokens(
            op, inputs_embeds, input_ids, video_features, self._video_token_id
        )

        return inputs_embeds

    def _replace_tokens(self, op, inputs_embeds, input_ids, features, token_id):
        """Replace token positions with encoder features (masked_scatter equivalent)."""
        mask = op.Equal(input_ids, op.Constant(value_int=token_id))
        mask_3d = op.Unsqueeze(mask, [-1])

        # Pad with a zero row for safety (text-only case: no features).
        feature_dim = op.Shape(features, start=1, end=2)
        zero_shape = op.Concat(op.Constant(value_ints=[1]), feature_dim, axis=0)
        zero_row = op.Expand(op.CastLike(0.0, features), zero_shape)
        padded = op.Concat(zero_row, features, axis=0)

        # CumSum-based per-position gather index. Mask positions get the
        # next feature row in order; non-mask positions get the zero row.
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

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values=None,
    ):
        hidden_states = inputs_embeds
        position_embeddings = (
            self.rotary_emb(op, position_ids) if self.rotary_emb is not None else None
        )

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


class Qwen25OmniThinkerForConditionalGeneration(nn.Module):
    """Qwen2.5-Omni Thinker: composite audio + vision + text model.

    Builds four separate ONNX models:

    - ``decoder``: Qwen2.5 text decoder taking ``inputs_embeds``
    - ``vision_encoder``: Qwen2.5-VL ViT (pixel_values + grid_thw → image features)
    - ``audio_tower``: 2x Conv1d + transformer audio tower (mel → audio features)
    - ``embedding``: word embedding + multimodal feature fusion

    HuggingFace class: ``Qwen2_5OmniForConditionalGeneration`` (Thinker only —
    the Talker / streaming code generation head is out of scope for now).
    """

    default_task: str = "qwen25-omni"
    category: str = "Multimodal"
    config_class: type = ArchitectureConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = Qwen25OmniDecoderModel(config)
        self.embedding = Qwen25OmniEmbeddingModel(config)
        self.vision_encoder: Qwen25OmniVisionEncoder | None = (
            Qwen25OmniVisionEncoder(config) if config.vision is not None else None
        )
        self.audio_encoder: Qwen25OmniAudioEncoder | None = (
            Qwen25OmniAudioEncoder(config) if config.audio is not None else None
        )

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "Qwen25OmniThinkerForConditionalGeneration is a multi-model split; the corresponding "
            "Qwen25OmniTask builds each sub-module (decoder, embedding, vision_encoder, "
            "audio_encoder) "
            "separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace weight names to ONNX module structure.

        HF Qwen2.5-Omni checkpoints prefix every Thinker key with ``thinker.``:

        - ``thinker.audio_tower.*`` → ``audio_encoder.*``
        - ``thinker.visual.*`` → ``vision_encoder.visual.*``
        - ``thinker.model.embed_tokens.*`` → ``embedding.embed_tokens.*``
        - ``thinker.model.layers.N.*`` and ``model.norm.*`` → ``decoder.*``
        - ``thinker.lm_head.*`` → ``decoder.lm_head.*``
        - ``thinker.model.rotary_emb.*`` → ``decoder.rotary_emb.*``

        The Talker sub-tree (``talker.*``) and the audio-output codec head
        are not consumed by this model and are silently dropped.
        """
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Strip the thinker. prefix if present.
            if key.startswith("thinker."):
                key = key[len("thinker.") :]

            # Drop talker.* and any codec output keys — not part of Thinker.
            if key.startswith(("talker.", "token2wav.", "code_predictor.")):
                continue

            if key.startswith("audio_tower."):
                if ".audio_bos_eos_token." not in key:
                    cleaned["audio_encoder." + key[len("audio_tower.") :]] = value
                continue

            if key.startswith("visual."):
                new_key = "vision_encoder." + key
                new_key = new_key.replace(".merger.mlp.0.", ".merger.mlp_0.")
                new_key = new_key.replace(".merger.mlp.2.", ".merger.mlp_2.")
                cleaned[new_key] = value
                continue

            if key.startswith("lm_head."):
                cleaned["decoder." + key] = value
                continue

            if key.startswith("model."):
                inner = key[len("model.") :]
                if inner.startswith("embed_tokens."):
                    cleaned["embedding." + inner] = value
                    continue
                if inner.startswith(("layers.", "norm.", "rotary_emb.")):
                    cleaned["decoder." + inner] = value
                    continue

            cleaned[key] = value

        # Weight tying: ``embedding.embed_tokens.weight`` ↔ ``decoder.lm_head.weight``.
        embed_key = "embedding.embed_tokens.weight"
        lm_key = "decoder.lm_head.weight"
        if getattr(self.config, "tie_word_embeddings", False):
            if embed_key in cleaned and lm_key not in cleaned:
                cleaned[lm_key] = cleaned[embed_key]
            elif lm_key in cleaned and embed_key not in cleaned:
                cleaned[embed_key] = cleaned[lm_key]

        return cleaned
