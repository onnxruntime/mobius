# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GLM-ASR speech recognition with a partial-RoPE audio encoder and Llama decoder.

Replicates Hugging Face ``GlmAsrForConditionalGeneration`` as three ONNX models:
an audio encoder with its temporal projector, an audio/text embedding mixer, and
a cached Llama-like decoder.
"""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, GlmAsrConfig
from mobius.components import (
    Conv1d,
    DecoderLayer,
    Embedding,
    LayerNorm,
    Linear,
    RMSNorm,
    apply_rotary_pos_emb,
    create_attention_bias,
    get_activation,
    initialize_rope,
)


def _audio_architecture_config(config: ArchitectureConfig) -> ArchitectureConfig:
    audio = config.audio
    if audio is None:
        raise ValueError("GLM-ASR requires an audio_config")

    hidden_size = audio.d_model or 1280
    num_heads = audio.encoder_attention_heads or 20
    head_dim = audio.encoder_head_dim or hidden_size // num_heads
    return dataclasses.replace(
        config,
        model_type="glmasr_encoder",
        hidden_size=hidden_size,
        intermediate_size=audio.encoder_ffn_dim or 5120,
        num_hidden_layers=audio.encoder_layers or 32,
        num_attention_heads=num_heads,
        num_key_value_heads=audio.encoder_num_key_value_heads or num_heads,
        head_dim=head_dim,
        hidden_act=audio.activation_function,
        max_position_embeddings=audio.max_source_positions or 1500,
        rms_norm_eps=audio.encoder_layer_norm_eps or 1e-5,
        rope_type="default",
        rope_theta=audio.encoder_rope_theta or 10_000.0,
        rope_scaling=None,
        partial_rotary_factor=audio.encoder_partial_rotary_factor or 0.5,
        rope_interleave=False,
        attn_qk_norm=False,
    )


class GlmAsrAudioAttention(nn.Module):
    """Bidirectional audio attention with GLM-ASR's asymmetric projection biases."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._num_attention_heads = config.num_attention_heads
        self._num_key_value_heads = config.num_key_value_heads
        self._head_dim = config.head_dim
        self._rotary_dim = int(config.head_dim * (config.partial_rotary_factor or 1.0))
        self._scale = config.head_dim**-0.5

        q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        self.q_proj = Linear(config.hidden_size, q_size, bias=True)
        self.k_proj = Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = Linear(config.hidden_size, kv_size, bias=True)
        self.o_proj = Linear(q_size, config.hidden_size, bias=True)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
    ) -> ir.Value:
        query_states = self.q_proj(op, hidden_states)
        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)

        # Only the first half of each 64-wide audio head is rotated; the
        # remaining channels pass through unchanged, matching HF partial RoPE.
        query_states = apply_rotary_pos_emb(
            op,
            query_states,
            position_embeddings,
            num_heads=self._num_attention_heads,
            rotary_embedding_dim=self._rotary_dim,
        )
        key_states = apply_rotary_pos_emb(
            op,
            key_states,
            position_embeddings,
            num_heads=self._num_key_value_heads,
            rotary_embedding_dim=self._rotary_dim,
        )

        attention_output = op.Attention(
            query_states,
            key_states,
            value_states,
            None,
            None,
            None,
            q_num_heads=self._num_attention_heads,
            kv_num_heads=self._num_key_value_heads,
            scale=self._scale,
            is_causal=0,
        )
        return self.o_proj(op, attention_output)


class GlmAsrAudioMLP(nn.Module):
    """Bias-enabled feed-forward network used by the audio encoder."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self._activation = get_activation(config.hidden_act)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return self.fc2(op, self._activation(op, self.fc1(op, hidden_states)))


class GlmAsrAudioEncoderLayer(nn.Module):
    """Pre-norm bidirectional transformer layer from ``GlmAsrEncoder``."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.self_attn = GlmAsrAudioAttention(config)
        self.mlp = GlmAsrAudioMLP(config)
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
    ) -> ir.Value:
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states, position_embeddings)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        return op.Add(residual, hidden_states)


class GlmAsrAudioTower(nn.Module):
    """GLM-ASR audio tower: log-mel frames to 1280-wide encoder states."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        audio_config = _audio_architecture_config(config)
        audio = config.audio
        assert audio is not None

        self.conv1 = Conv1d(
            audio.num_mel_bins or 128,
            audio_config.hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = Conv1d(
            audio_config.hidden_size,
            audio_config.hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.layers = nn.ModuleList(
            [
                GlmAsrAudioEncoderLayer(audio_config)
                for _ in range(audio_config.num_hidden_layers)
            ]
        )
        self.norm = LayerNorm(audio_config.hidden_size, eps=audio_config.rms_norm_eps)
        self.rotary_emb = initialize_rope(audio_config)

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        hidden_states = op.Gelu(self.conv1(op, input_features))
        hidden_states = op.Gelu(self.conv2(op, hidden_states))
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        # (audio_batch, ceil(mel_sequence_len / 2), audio_hidden_size)

        sequence_len = op.Squeeze(op.Shape(hidden_states, start=1, end=2), [0])
        position_ids = op.Unsqueeze(
            op.Range(
                op.Constant(value_int=0),
                sequence_len,
                op.Constant(value_int=1),
            ),
            [0],
        )
        if self.rotary_emb is None:
            raise ValueError("GLM-ASR audio encoder requires rotary embeddings")
        position_embeddings = self.rotary_emb(op, position_ids)

        for layer in self.layers:
            hidden_states = layer(op, hidden_states, position_embeddings)
        return self.norm(op, hidden_states)


class GlmAsrMultiModalProjector(nn.Module):
    """Four-frame merge followed by the checkpoint's 5120→4096→2048 MLP."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        audio = config.audio
        if audio is None:
            raise ValueError("GLM-ASR requires an audio_config")
        audio_hidden_size = audio.d_model or 1280
        projector_input_size = audio.encoder_ffn_dim or 5120
        if projector_input_size % audio_hidden_size != 0:
            raise ValueError(
                "GLM-ASR projector input size must be divisible by audio hidden size"
            )
        self._merge_factor = projector_input_size // audio_hidden_size
        self._projector_input_size = projector_input_size
        projector_hidden_size = config.hidden_size * 2
        self.linear_1 = Linear(projector_input_size, projector_hidden_size, bias=True)
        self.linear_2 = Linear(projector_hidden_size, config.hidden_size, bias=True)
        self._activation = get_activation(config.projector_hidden_act)

    @property
    def merge_factor(self) -> int:
        return self._merge_factor

    def forward(self, op: OpBuilder, audio_hidden_states: ir.Value) -> ir.Value:
        # Concatenate each group of four consecutive encoder frames:
        # (B, T, 1280) -> (B, T / 4, 5120).
        merged = op.Reshape(
            audio_hidden_states,
            op.Constant(value_ints=[0, -1, self._projector_input_size]),
        )
        return self.linear_2(op, self._activation(op, self.linear_1(op, merged)))


class GlmAsrAudioEncoder(nn.Module):
    """Audio encoder plus projector exported as the ``audio_encoder`` model."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.audio_tower = GlmAsrAudioTower(config)
        self.multi_modal_projector = GlmAsrMultiModalProjector(config)

    def forward(
        self,
        op: OpBuilder,
        input_features: ir.Value,
        input_features_mask: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        audio_hidden_states = self.audio_tower(op, input_features)
        audio_features = self.multi_modal_projector(op, audio_hidden_states)

        # Processor mask is over mel frames. Conv1 preserves length, Conv2
        # ceil-divides by two, then the projector consumes non-overlapping
        # groups of four encoder frames.
        valid_mel_frames = op.ReduceSum(
            input_features_mask,
            op.Constant(value_ints=[1]),
            keepdims=0,
        )
        one = op.Constant(value_ints=[1])
        two = op.Constant(value_ints=[2])
        conv_length = op.Div(op.Add(valid_mel_frames, one), two)
        merge = op.Constant(value_ints=[self.multi_modal_projector.merge_factor])
        # HF's ``(conv_length - merge) // merge + 1`` equals floor division
        # by ``merge`` for nonnegative lengths. This form also avoids ONNX's
        # truncation-toward-zero difference when ``conv_length < merge``.
        audio_feature_lengths = op.Div(conv_length, merge)

        # Match HF's ``pooler_output`` contract by stripping padding-derived
        # projector rows and flattening valid features across the batch:
        # (B, projected_T, text_hidden) -> (sum(valid_T), text_hidden).
        projected_length = op.Squeeze(
            op.Shape(audio_features, start=1, end=2),
            op.Constant(value_ints=[0]),
        )
        positions = op.Unsqueeze(
            op.Range(
                op.Constant(value_int=0),
                projected_length,
                op.Constant(value_int=1),
            ),
            [0],
        )
        valid_mask = op.Less(positions, op.Unsqueeze(audio_feature_lengths, [1]))
        valid_indices = op.Transpose(op.NonZero(valid_mask), perm=[1, 0])
        valid_audio_features = op.GatherND(audio_features, valid_indices)
        return valid_audio_features, audio_feature_lengths


class GlmAsrEmbeddingModel(nn.Module):
    """Replace GLM audio placeholder tokens with flattened projected features."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        if config.audio_token_id is None:
            raise ValueError("GLM-ASR config is missing audio_token_id")
        self._audio_token_id = config.audio_token_id
        self._hidden_size = config.hidden_size

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        audio_features: ir.Value,
    ) -> ir.Value:
        inputs_embeds = self.embed_tokens(op, input_ids)
        is_audio = op.Equal(input_ids, op.Constant(value_int=self._audio_token_id))
        flat_audio_mask = op.Reshape(is_audio, [-1])
        flat_audio_indices = op.CumSum(
            op.Cast(flat_audio_mask, to=ir.DataType.INT64),
            op.Constant(value_int=0),
        )
        flat_audio_indices = op.Mul(
            flat_audio_indices,
            op.Cast(flat_audio_mask, to=ir.DataType.INT64),
        )
        indices = op.Reshape(flat_audio_indices, op.Shape(input_ids))

        zero_row = op.Unsqueeze(
            op.CastLike(
                op.Constant(value_floats=[0.0] * self._hidden_size),
                audio_features,
            ),
            [0],
        )
        padded_features = op.Concat(zero_row, audio_features, axis=0)
        gathered = op.Gather(padded_features, indices, axis=0)
        return op.Where(op.Unsqueeze(is_audio, [-1]), gathered, inputs_embeds)


class GlmAsrDecoderModel(nn.Module):
    """Llama-like decoder with full RoPE and reusable KV cache."""

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
        if self.rotary_emb is None:
            raise ValueError("GLM-ASR text decoder requires rotary embeddings")
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=inputs_embeds,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        hidden_states = inputs_embeds
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
        return self.lm_head(op, hidden_states), present_key_values


class GlmAsrForConditionalGeneration(nn.Module):
    """GLM-ASR-Nano composite speech recognition model."""

    default_task: str = "glmasr-speech-language"
    category: str = "Speech-to-Text"
    config_class: type = GlmAsrConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.audio_encoder = GlmAsrAudioEncoder(config)
        self.embedding = GlmAsrEmbeddingModel(config)
        self.decoder = GlmAsrDecoderModel(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route checkpoint weights into the three standardized ONNX models."""
        routed: dict[str, torch.Tensor] = {}
        for source_key, value in state_dict.items():
            key = source_key.removeprefix("model.")
            if key.startswith(("audio_encoder.", "embedding.", "decoder.")):
                routed[key] = value
            elif key.startswith("audio_tower."):
                routed[f"audio_encoder.{key}"] = value
            elif key.startswith("multi_modal_projector."):
                routed[f"audio_encoder.{key}"] = value
            elif key.startswith("language_model.model.embed_tokens."):
                suffix = key.removeprefix("language_model.model.")
                routed[f"embedding.{suffix}"] = value
            elif key.startswith("language_model.model."):
                suffix = key.removeprefix("language_model.model.")
                routed[f"decoder.{suffix}"] = value
            elif key.startswith("language_model.embed_tokens."):
                suffix = key.removeprefix("language_model.")
                routed[f"embedding.{suffix}"] = value
            elif key.startswith(("language_model.layers.", "language_model.norm.")):
                suffix = key.removeprefix("language_model.")
                routed[f"decoder.{suffix}"] = value
            elif key.startswith("language_model.lm_head."):
                suffix = key.removeprefix("language_model.")
                routed[f"decoder.{suffix}"] = value
            elif key.startswith("lm_head."):
                routed[f"decoder.{key}"] = value
            else:
                routed[key] = value

        embed_key = "embedding.embed_tokens.weight"
        lm_head_key = "decoder.lm_head.weight"
        if self.config.tie_word_embeddings:
            if embed_key in routed and lm_head_key not in routed:
                routed[lm_head_key] = routed[embed_key]
            elif lm_head_key in routed and embed_key not in routed:
                routed[embed_key] = routed[lm_head_key]
        return routed
