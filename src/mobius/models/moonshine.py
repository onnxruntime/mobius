# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Moonshine raw-waveform encoder-decoder model for speech recognition.

Replicates Hugging Face ``MoonshineForConditionalGeneration`` as separate
encoder and cached decoder ONNX graphs.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import MoonshineConfig
from mobius.components import (
    Conv1d,
    Embedding,
    LayerNormNoBias,
    Linear,
    apply_rotary_pos_emb,
)

_INT64_MAX = 9223372036854775807


def _silu(op: OpBuilder, x: ir.Value) -> ir.Value:
    return op.Mul(x, op.Sigmoid(x))


def _padding_attention_bias(
    op: OpBuilder,
    attention_mask: ir.Value,
    query_states: ir.Value,
    dtype: ir.DataType,
) -> ir.Value:
    """Expand ``[B, K]`` padding mask to additive ``[B, 1, Q, K]`` bias."""
    valid = op.Cast(attention_mask, to=ir.DataType.BOOL)
    valid = op.Unsqueeze(valid, [1, 2])
    target_shape = op.Concat(
        op.Shape(attention_mask, start=0, end=1),
        [1],
        op.Shape(query_states, start=1, end=2),
        op.Shape(attention_mask, start=1, end=2),
        axis=0,
    )
    valid = op.Expand(valid, target_shape)
    bias = op.Where(valid, 0.0, float(dtype.min))
    return op.Cast(bias, to=dtype)


class MoonshineGroupNorm(nn.Module):
    """Single-group normalization without the CPU-only ONNX GroupNormalization op."""

    def __init__(self, channels: int, eps: float, dtype: ir.DataType):
        super().__init__()
        self.weight = nn.Parameter([channels])
        self.bias = nn.Parameter([channels])
        self._eps = eps
        self._dtype = dtype

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # GroupNorm(num_groups=1) normalizes each sample jointly across channels
        # and time. Upcast the reduction to fp32 like PyTorch's mixed-precision
        # kernel, then restore the model dtype after the affine transform.
        compute = hidden_states
        weight = self.weight
        bias = self.bias
        if self._dtype != ir.DataType.FLOAT:
            compute = op.Cast(compute, to=ir.DataType.FLOAT)
            weight = op.Cast(weight, to=ir.DataType.FLOAT)
            bias = op.Cast(bias, to=ir.DataType.FLOAT)
        mean = op.ReduceMean(compute, [1, 2], keepdims=1)
        centered = op.Sub(compute, mean)
        variance = op.ReduceMean(op.Mul(centered, centered), [1, 2], keepdims=1)
        normalized = op.Div(centered, op.Sqrt(op.Add(variance, self._eps)))
        scale = op.Reshape(weight, [1, -1, 1])
        offset = op.Reshape(bias, [1, -1, 1])
        normalized = op.Add(op.Mul(normalized, scale), offset)
        if self._dtype != ir.DataType.FLOAT:
            normalized = op.Cast(normalized, to=self._dtype)
        return normalized


class MoonshineRotaryEmbedding(nn.Module):
    """Dynamic interleaved partial RoPE used by Moonshine encoder and decoder."""

    def __init__(self, config: MoonshineConfig):
        super().__init__()
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        self._inv_freq = 1.0 / (
            config.rope_theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
        )
        self._dtype = config.dtype

    def forward(self, op: OpBuilder, position_ids: ir.Value) -> tuple[ir.Value, ir.Value]:
        # [B, S] x [D/2] -> frequency table [B, S, D/2].
        positions = op.Unsqueeze(op.Cast(position_ids, to=ir.DataType.FLOAT), [2])
        inv_freq = op.Constant(value=ir.tensor(self._inv_freq))
        angles = op.Mul(positions, inv_freq)
        cos = op.Cos(angles)
        sin = op.Sin(angles)
        if self._dtype != ir.DataType.FLOAT:
            cos = op.Cast(cos, to=self._dtype)
            sin = op.Cast(sin, to=self._dtype)
        return cos, sin


class MoonshineAttention(nn.Module):
    """Bias-free self/cross attention with optional partial interleaved RoPE."""

    def __init__(
        self,
        config: MoonshineConfig,
        *,
        num_heads: int,
        num_key_value_heads: int,
        is_causal: bool = False,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._num_key_value_heads = num_key_value_heads
        self._head_dim = config.hidden_size // num_heads
        self._rotary_dim = int(self._head_dim * config.partial_rotary_factor)
        self._scale = self._head_dim**-0.5
        self._is_causal = is_causal
        self.q_proj = Linear(config.hidden_size, num_heads * self._head_dim, bias=False)
        self.k_proj = Linear(
            config.hidden_size, num_key_value_heads * self._head_dim, bias=False
        )
        self.v_proj = Linear(
            config.hidden_size, num_key_value_heads * self._head_dim, bias=False
        )
        self.o_proj = Linear(num_heads * self._head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        *,
        attention_bias: ir.Value | None = None,
        position_embeddings: tuple[ir.Value, ir.Value] | None = None,
        key_value_states: ir.Value | None = None,
        past_key_value: tuple[ir.Value, ir.Value] | None = None,
    ) -> tuple[ir.Value, tuple[ir.Value, ir.Value]]:
        query_states = self.q_proj(op, hidden_states)  # [B, Q, Hq * Dh]
        kv_source = hidden_states if key_value_states is None else key_value_states
        key_states = self.k_proj(op, kv_source)  # [B, K, Hkv * Dh]
        value_states = self.v_proj(op, kv_source)  # [B, K, Hkv * Dh]

        # Moonshine rotates encoder and decoder self-attention only. Cross-attention
        # projects the encoder sequence without positional rotation.
        if position_embeddings is not None:
            query_states = apply_rotary_pos_emb(
                op,
                query_states,
                position_embeddings,
                self._num_heads,
                rotary_embedding_dim=self._rotary_dim,
                interleaved=True,
            )
            key_states = apply_rotary_pos_emb(
                op,
                key_states,
                position_embeddings,
                self._num_key_value_heads,
                rotary_embedding_dim=self._rotary_dim,
                interleaved=True,
            )

        # Cross-attention K/V are constant and intentionally not appended to the
        # decoder self-attention cache exposed by the package.
        use_past = key_value_states is None and past_key_value is not None
        past_key = past_key_value[0] if use_past else None
        past_value = past_key_value[1] if use_past else None
        output, present_key, present_value = op.Attention(
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key,
            past_value,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_key_value_heads,
            scale=self._scale,
            is_causal=1 if self._is_causal and attention_bias is None else 0,
            _outputs=3,
        )
        return self.o_proj(op, output), (present_key, present_value)


class MoonshineEncoderMLP(nn.Module):
    """GELU feed-forward network used by Moonshine encoder layers."""

    def __init__(self, config: MoonshineConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return self.fc2(op, op.Gelu(self.fc1(op, hidden_states)))


class MoonshineDecoderMLP(nn.Module):
    """Fused up/gate SiLU feed-forward network used by the decoder."""

    def __init__(self, config: MoonshineConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, 2 * config.intermediate_size, bias=True)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self._intermediate_size = config.intermediate_size

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # HF chunks fc1 as (data, gate), not the gate-first ordering used by Llama.
        data, gate = op.Split(
            self.fc1(op, hidden_states),
            split=[self._intermediate_size, self._intermediate_size],
            axis=-1,
            _outputs=2,
        )
        return self.fc2(op, op.Mul(data, _silu(op, gate)))


class MoonshineEncoderLayer(nn.Module):
    """Pre-norm bidirectional transformer layer used by the audio encoder."""

    def __init__(self, config: MoonshineConfig):
        super().__init__()
        self.input_layernorm = LayerNormNoBias(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = MoonshineAttention(
            config,
            num_heads=config.encoder_num_attention_heads,
            num_key_value_heads=config.encoder_num_key_value_heads,
        )
        self.post_attention_layernorm = LayerNormNoBias(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = MoonshineEncoderMLP(config)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
    ) -> ir.Value:
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, _ = self.self_attn(
            op,
            hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        return op.Add(residual, hidden_states)


class MoonshineDecoderLayer(nn.Module):
    """Pre-norm decoder layer with causal self-attention and encoder attention."""

    def __init__(self, config: MoonshineConfig):
        super().__init__()
        self.input_layernorm = LayerNormNoBias(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = MoonshineAttention(
            config,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            is_causal=True,
        )
        self.post_attention_layernorm = LayerNormNoBias(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.encoder_attn = MoonshineAttention(
            config,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
        )
        self.final_layernorm = LayerNormNoBias(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MoonshineDecoderMLP(config)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        encoder_attention_bias: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
        past_key_value: tuple[ir.Value, ir.Value] | None,
    ) -> tuple[ir.Value, tuple[ir.Value, ir.Value]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present_key_value = self.self_attn(
            op,
            hidden_states,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states, _ = self.encoder_attn(
            op,
            hidden_states,
            attention_bias=encoder_attention_bias,
            key_value_states=encoder_hidden_states,
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.final_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        return op.Add(residual, hidden_states), present_key_value


class MoonshineEncoderModel(nn.Module):
    """Raw waveform encoder: strided convolution stem followed by RoPE layers."""

    def __init__(self, config: MoonshineConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.conv1 = Conv1d(1, hidden_size, kernel_size=127, stride=64, bias=False)
        self.conv2 = Conv1d(hidden_size, 2 * hidden_size, kernel_size=7, stride=3)
        self.conv3 = Conv1d(2 * hidden_size, hidden_size, kernel_size=3, stride=2)
        self.groupnorm = MoonshineGroupNorm(
            hidden_size, eps=config.layer_norm_eps, dtype=config.dtype
        )
        self.layers = nn.ModuleList(
            [MoonshineEncoderLayer(config) for _ in range(config.encoder_num_hidden_layers)]
        )
        self.layer_norm = LayerNormNoBias(hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = MoonshineRotaryEmbedding(config)
        self._dtype = config.dtype

    def forward(
        self,
        op: OpBuilder,
        input_values: ir.Value,
        attention_mask: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        # Raw audio [B, L] -> [B, 1, L]. The unpadded 127/7/3 kernels reduce
        # temporal length by the exact HF formula with aggregate stride 384.
        hidden_states = op.Unsqueeze(input_values, [1])
        hidden_states = op.Tanh(self.conv1(op, hidden_states))
        hidden_states = self.groupnorm(op, hidden_states)
        hidden_states = op.Gelu(self.conv2(op, hidden_states))
        hidden_states = op.Gelu(self.conv3(op, hidden_states))
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])  # [B, Tenc, D]

        # HF downsamples the sample mask by strided slicing, then truncates it to
        # the actual convolution output length.
        encoder_attention_mask = op.Slice(
            attention_mask,
            [0],
            [_INT64_MAX],
            [1],
            [384],
        )
        encoder_length = op.Shape(hidden_states, start=1, end=2)
        encoder_attention_mask = op.Slice(encoder_attention_mask, [0], encoder_length, [1])
        attention_bias = _padding_attention_bias(
            op, encoder_attention_mask, hidden_states, self._dtype
        )

        sequence_length = op.Squeeze(encoder_length, [0])
        position_ids = op.Unsqueeze(op.Range(0, sequence_length, 1), [0])
        position_embeddings = self.rotary_emb(op, position_ids)
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, attention_bias, position_embeddings)
        return self.layer_norm(op, hidden_states), encoder_attention_mask


class MoonshineDecoderModel(nn.Module):
    """Cached autoregressive decoder with cross-attention to encoded speech."""

    def __init__(self, config: MoonshineConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [MoonshineDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LayerNormNoBias(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = MoonshineRotaryEmbedding(config)
        self.proj_out = Linear(config.hidden_size, config.vocab_size, bias=False)
        self._dtype = config.dtype

    def forward(
        self,
        op: OpBuilder,
        decoder_input_ids: ir.Value,
        encoder_hidden_states: ir.Value,
        encoder_attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list[tuple[ir.Value, ir.Value]] | None = None,
    ) -> tuple[ir.Value, list[tuple[ir.Value, ir.Value]]]:
        hidden_states = self.embed_tokens(op, decoder_input_ids)  # [B, Tdec, D]
        position_embeddings = self.rotary_emb(op, position_ids)
        encoder_attention_bias = _padding_attention_bias(
            op, encoder_attention_mask, hidden_states, self._dtype
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_key_value in zip(self.layers, past_kvs):
            hidden_states, present_key_value = layer(
                op,
                hidden_states,
                encoder_hidden_states,
                encoder_attention_bias,
                position_embeddings,
                past_key_value,
            )
            present_key_values.append(present_key_value)

        hidden_states = self.norm(op, hidden_states)
        return self.proj_out(op, hidden_states), present_key_values


class _MoonshineModel(nn.Module):
    def __init__(self, config: MoonshineConfig):
        super().__init__()
        self.encoder = MoonshineEncoderModel(config)
        self.decoder = MoonshineDecoderModel(config)


class MoonshineForConditionalGeneration(nn.Module):
    """Moonshine RoPE encoder-decoder model for raw-waveform speech recognition."""

    default_task: str = "speech-to-text"
    category: str = "Speech-to-Text"
    config_class: type = MoonshineConfig

    def __init__(self, config: MoonshineConfig):
        super().__init__()
        self.config = config
        self.model = _MoonshineModel(config)
        self.proj_out = self.model.decoder.proj_out

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Tie the output projection and strip the HF ``model.`` prefix."""
        embed_key = "model.decoder.embed_tokens.weight"
        proj_key = "proj_out.weight"
        if (
            self.config.tie_word_embeddings
            and proj_key not in state_dict
            and embed_key in state_dict
        ):
            state_dict[proj_key] = state_dict[embed_key]
        if proj_key in state_dict:
            state_dict[f"model.decoder.{proj_key}"] = state_dict.pop(proj_key)

        return {
            key[len("model.") :] if key.startswith("model.") else key: value
            for key, value in state_dict.items()
        }
