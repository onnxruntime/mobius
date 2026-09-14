# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""VibeVoice text-to-speech stages for the Transformers-native HF checkpoint.

The implementation mirrors ``VibeVoiceForConditionalGeneration`` as eight
ONNX graphs. Host code owns the deterministic DPM-Solver loop and the
positive/negative decoder-cache orchestration.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import (
    VibeVoiceConfig,
    VibeVoiceDiffusionConfig,
    VibeVoiceTokenizerConfig,
)
from mobius.components import (
    DecoderLayer,
    Embedding,
    GatedMLP,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


VIBEVOICE_MODEL_ID = "vibevoice/VibeVoice-1.5B-hf"
VIBEVOICE_REVISION = "edc39f80f5cae656da37baf8faa8f5502bf7081f"
VIBEVOICE_MICROSOFT_PROVENANCE_REVISION = "c00898d257e6b46004e3e2866a47534085fb685a"


class _CacheAllocator:
    """Assign stable explicit state slots while the convolution stack is built."""

    def __init__(self) -> None:
        self.specs: list[tuple[int, int]] = []

    def add(self, channels: int, left_pad: int) -> int:
        index = len(self.specs)
        self.specs.append((channels, left_pad))
        return index


class _ConvState:
    """Thread explicit streaming padding state through ONNX convolution calls."""

    def __init__(self, past: Sequence[ir.Value]):
        self._past = list(past)
        self._present: list[ir.Value | None] = [None] * len(self._past)

    def prepend(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        *,
        index: int,
        left_pad: int,
    ) -> ir.Value:
        past = self._past[index]
        padded = op.Concat(past, hidden_states, axis=2)
        self._present[index] = op.Slice(
            padded,
            op.Constant(value_ints=[-left_pad]),
            op.Constant(value_ints=[2**63 - 1]),
            op.Constant(value_ints=[2]),
        )
        return padded

    def outputs(self) -> list[ir.Value]:
        if any(value is None for value in self._present):
            raise RuntimeError("Every VibeVoice streaming convolution must update its cache")
        return [value for value in self._present if value is not None]


class _RawConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.weight = nn.Parameter([out_channels, in_channels // groups, kernel_size])
        self.bias = nn.Parameter([out_channels])
        self._kernel_size = kernel_size
        self._stride = stride
        self._dilation = dilation
        self._groups = groups

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        return op.Conv(
            hidden_states,
            self.weight,
            self.bias,
            kernel_shape=[self._kernel_size],
            strides=[self._stride],
            dilations=[self._dilation],
            group=self._groups,
        )


class _RawConvTranspose1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.weight = nn.Parameter([in_channels, out_channels, kernel_size])
        self.bias = nn.Parameter([out_channels])
        self._kernel_size = kernel_size
        self._stride = stride

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        return op.ConvTranspose(
            hidden_states,
            self.weight,
            self.bias,
            kernel_shape=[self._kernel_size],
            strides=[self._stride],
        )


class _CausalConv1d(nn.Module):
    """Causal Conv1d with explicit state matching the HF padding-cache update."""

    def __init__(
        self,
        allocator: _CacheAllocator,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = _RawConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self._left_pad = (kernel_size - 1) * dilation - (stride - 1)
        if self._left_pad < 0:
            raise ValueError("VibeVoice causal convolution padding must be non-negative")
        self._cache_index = allocator.add(in_channels, self._left_pad)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        state: _ConvState | None = None,
    ):
        if state is None:
            hidden_states = op.Pad(
                hidden_states,
                op.Constant(value_ints=[0, 0, self._left_pad, 0, 0, 0]),
            )
        else:
            hidden_states = state.prepend(
                op,
                hidden_states,
                index=self._cache_index,
                left_pad=self._left_pad,
            )
        return self.conv(op, hidden_states)


class _CausalConvTranspose1d(nn.Module):
    """Causal ConvTranspose1d with HF-equivalent right trimming and state."""

    def __init__(
        self,
        allocator: _CacheAllocator,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.convtr = _RawConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        self._stride = stride
        self._padding_total = kernel_size - stride
        self._left_pad = kernel_size - 1
        self._cache_index = allocator.add(in_channels, self._left_pad)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        state: _ConvState | None = None,
    ):
        input_length = op.Shape(hidden_states, start=2, end=3)
        if state is not None:
            hidden_states = state.prepend(
                op,
                hidden_states,
                index=self._cache_index,
                left_pad=self._left_pad,
            )
        hidden_states = self.convtr(op, hidden_states)
        if self._padding_total:
            hidden_states = op.Slice(
                hidden_states,
                op.Constant(value_ints=[0]),
                op.Constant(value_ints=[-self._padding_total]),
                op.Constant(value_ints=[2]),
            )
        if state is not None:
            # Streaming returns exactly the samples produced by the current input chunk.
            new_length = op.Mul(input_length, op.Constant(value_int=self._stride))
            hidden_states = op.Slice(
                hidden_states,
                op.Neg(new_length),
                op.Constant(value_ints=[2**63 - 1]),
                op.Constant(value_ints=[2]),
            )
        return hidden_states


class _TokenizerFeedForward(nn.Module):
    def __init__(self, config: VibeVoiceTokenizerConfig, hidden_size: int):
        super().__init__()
        self.linear1 = Linear(hidden_size, config.ffn_expansion * hidden_size)
        self.linear2 = Linear(config.ffn_expansion * hidden_size, hidden_size)
        self._activation = config.hidden_act

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        hidden_states = self.linear1(op, hidden_states)
        if self._activation != "gelu":
            raise ValueError(f"Unsupported VibeVoice tokenizer activation: {self._activation}")
        hidden_states = op.Gelu(hidden_states)
        return self.linear2(op, hidden_states)


class _ConvNext1dLayer(nn.Module):
    """ConvNeXt residual mixer and FFN in channels-first layout."""

    def __init__(
        self,
        config: VibeVoiceTokenizerConfig,
        allocator: _CacheAllocator,
        hidden_size: int,
    ):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.ffn = _TokenizerFeedForward(config, hidden_size)
        self.gamma = nn.Parameter([hidden_size])
        self.ffn_gamma = nn.Parameter([hidden_size])
        self.mixer = _CausalConv1d(
            allocator,
            hidden_size,
            hidden_size,
            config.kernel_size,
            groups=hidden_size,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        state: _ConvState | None = None,
    ):
        residual = hidden_states
        mixed = op.Transpose(hidden_states, perm=[0, 2, 1])
        mixed = self.norm(op, mixed)
        mixed = op.Transpose(mixed, perm=[0, 2, 1])
        mixed = self.mixer(op, mixed, state)
        mixed = op.Mul(mixed, op.Unsqueeze(self.gamma, [-1]))
        hidden_states = op.Add(residual, mixed)  # (batch, channels, frames)

        residual = hidden_states
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = self.ffn_norm(op, hidden_states)
        hidden_states = self.ffn(op, hidden_states)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = op.Mul(hidden_states, op.Unsqueeze(self.ffn_gamma, [-1]))
        return op.Add(residual, hidden_states)


class _EncoderStem(nn.Module):
    def __init__(self, config: VibeVoiceTokenizerConfig, allocator: _CacheAllocator):
        super().__init__()
        self.conv = _CausalConv1d(
            allocator,
            config.channels,
            config.num_filters,
            config.kernel_size,
        )
        self.stage = nn.ModuleList(
            [
                _ConvNext1dLayer(config, allocator, config.num_filters)
                for _ in range(config.depths[0])
            ]
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value, state: _ConvState | None):
        hidden_states = self.conv(op, hidden_states, state)
        for block in self.stage:
            hidden_states = block(op, hidden_states, state)
        return hidden_states


class _EncoderLayer(nn.Module):
    def __init__(
        self,
        config: VibeVoiceTokenizerConfig,
        allocator: _CacheAllocator,
        stage_index: int,
    ):
        super().__init__()
        output_channels = config.num_filters * 2 ** (stage_index + 1)
        self.conv = _CausalConv1d(
            allocator,
            config.num_filters * 2**stage_index,
            output_channels,
            config.downsampling_ratios[stage_index] * 2,
            stride=config.downsampling_ratios[stage_index],
        )
        self.stage = nn.ModuleList(
            [
                _ConvNext1dLayer(config, allocator, output_channels)
                for _ in range(config.depths[stage_index + 1])
            ]
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value, state: _ConvState | None):
        hidden_states = self.conv(op, hidden_states, state)
        for block in self.stage:
            hidden_states = block(op, hidden_states, state)
        return hidden_states


class VibeVoiceTokenizerEncoder(nn.Module):
    """Causal 3200x waveform encoder shared by acoustic and semantic towers."""

    def __init__(self, config: VibeVoiceTokenizerConfig):
        super().__init__()
        allocator = _CacheAllocator()
        self.stem = _EncoderStem(config, allocator)
        self.conv_layers = nn.ModuleList(
            [
                _EncoderLayer(config, allocator, index)
                for index in range(len(config.downsampling_ratios))
            ]
        )
        self.head = _CausalConv1d(
            allocator,
            config.num_filters * 2 ** len(config.downsampling_ratios),
            config.hidden_size,
            config.kernel_size,
        )
        self.cache_specs = tuple(allocator.specs)

    def forward(
        self,
        op: OpBuilder,
        input_values: ir.Value,
        past_conv_states: Sequence[ir.Value] | None = None,
    ):
        state = _ConvState(past_conv_states) if past_conv_states is not None else None
        hidden_states = self.stem(op, input_values, state)
        for layer in self.conv_layers:
            hidden_states = layer(op, hidden_states, state)
        hidden_states = self.head(op, hidden_states, state)
        latents = op.Transpose(hidden_states, perm=[0, 2, 1])  # (batch, frames, latent)
        return latents, state.outputs() if state is not None else []


class _DecoderStem(nn.Module):
    def __init__(self, config: VibeVoiceTokenizerConfig, allocator: _CacheAllocator):
        super().__init__()
        channels = config.num_filters * 2 ** (len(config.depths) - 1)
        self.conv = _CausalConv1d(
            allocator,
            config.hidden_size,
            channels,
            config.kernel_size,
        )
        self.stage = nn.ModuleList(
            [_ConvNext1dLayer(config, allocator, channels) for _ in range(config.depths[0])]
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value, state: _ConvState | None):
        hidden_states = self.conv(op, hidden_states, state)
        for block in self.stage:
            hidden_states = block(op, hidden_states, state)
        return hidden_states


class _DecoderLayer(nn.Module):
    def __init__(
        self,
        config: VibeVoiceTokenizerConfig,
        allocator: _CacheAllocator,
        stage_index: int,
        upsampling_ratios: list[int],
    ):
        super().__init__()
        input_channels = config.num_filters * 2 ** (len(config.depths) - 1 - stage_index)
        output_channels = config.num_filters * 2 ** (len(config.depths) - 2 - stage_index)
        ratio = upsampling_ratios[stage_index]
        self.convtr = _CausalConvTranspose1d(
            allocator,
            input_channels,
            output_channels,
            ratio * 2,
            ratio,
        )
        self.stage = nn.ModuleList(
            [
                _ConvNext1dLayer(config, allocator, output_channels)
                for _ in range(config.depths[stage_index + 1])
            ]
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value, state: _ConvState | None):
        hidden_states = self.convtr(op, hidden_states, state)
        for block in self.stage:
            hidden_states = block(op, hidden_states, state)
        return hidden_states


class VibeVoiceTokenizerDecoder(nn.Module):
    """Streaming continuous-latent decoder producing 3200 waveform samples per frame."""

    def __init__(
        self,
        encoder_config: VibeVoiceTokenizerConfig,
        *,
        reverse_ratios: bool = True,
    ):
        super().__init__()
        config = VibeVoiceTokenizerConfig(
            **{
                **encoder_config.__dict__,
                "depths": list(reversed(encoder_config.depths)),
                "downsampling_ratios": (
                    list(reversed(encoder_config.downsampling_ratios))
                    if reverse_ratios
                    else list(encoder_config.downsampling_ratios)
                ),
            }
        )
        allocator = _CacheAllocator()
        upsampling_ratios = config.downsampling_ratios
        self.stem = _DecoderStem(config, allocator)
        self.conv_layers = nn.ModuleList(
            [
                _DecoderLayer(config, allocator, index, upsampling_ratios)
                for index in range(len(upsampling_ratios))
            ]
        )
        self.head = _CausalConv1d(
            allocator,
            config.num_filters,
            config.channels,
            config.kernel_size,
        )
        self.cache_specs = tuple(allocator.specs)

    def forward(
        self,
        op: OpBuilder,
        latents: ir.Value,
        past_conv_states: Sequence[ir.Value],
    ):
        state = _ConvState(past_conv_states)
        hidden_states = op.Transpose(latents, perm=[0, 2, 1])  # (batch, latent, frames)
        hidden_states = self.stem(op, hidden_states, state)
        for layer in self.conv_layers:
            hidden_states = layer(op, hidden_states, state)
        waveform = self.head(op, hidden_states, state)  # (batch, 1, frames * 3200)
        return waveform, state.outputs()


class VibeVoiceReferenceAudioEncoder(nn.Module):
    """Reference waveform encoder with explicit sigma-VAE noise inputs."""

    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.encoder = VibeVoiceTokenizerEncoder(config.acoustic_tokenizer)
        self._vae_std = config.acoustic_tokenizer.vae_std
        self._dtype = config.dtype
        self._hop_length = config.acoustic_tokenizer.hop_length

    def forward(
        self,
        op: OpBuilder,
        input_values: ir.Value,
        padding_mask: ir.Value,
        sample_noise: ir.Value,
        latent_noise: ir.Value,
    ):
        input_values = op.Cast(input_values, to=self._dtype)
        latents, _ = self.encoder(op, input_values)
        noise_scale = op.Mul(sample_noise, self._vae_std)
        latents = op.Add(latents, op.Mul(op.Unsqueeze(noise_scale, [1, 2]), latent_noise))

        # Keep only processor-declared valid 3200-sample frames across all voice prompts.
        valid_samples = op.ReduceSum(
            op.Cast(padding_mask, to=ir.DataType.INT64),
            op.Constant(value_ints=[1]),
            keepdims=0,
        )
        valid_frames = op.Div(
            op.Add(valid_samples, self._hop_length - 1),
            self._hop_length,
        )
        frame_count = op.Squeeze(op.Shape(latents, start=1, end=2), [0])
        frame_positions = op.Range(
            op.Constant(value_int=0),
            frame_count,
            op.Constant(value_int=1),
        )
        valid_mask = op.Less(
            op.Unsqueeze(frame_positions, [0]),
            op.Unsqueeze(valid_frames, [1]),
        )
        valid_indices = op.Transpose(op.NonZero(valid_mask), perm=[1, 0])
        return op.GatherND(latents, valid_indices)  # (valid_audio_frames, latent_size)


class VibeVoiceMultiModalProjector(nn.Module):
    """Linear -> RMSNorm -> Linear connector used by both continuous tokenizers."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear_1 = Linear(input_dim, output_dim)
        self.act = RMSNorm(output_dim, eps=1e-6)
        self.linear_2 = Linear(output_dim, output_dim)

    def forward(self, op: OpBuilder, audio_features: ir.Value):
        hidden_states = self.linear_1(op, audio_features)
        hidden_states = self.act(op, hidden_states)
        return self.linear_2(op, hidden_states)


class VibeVoiceAcousticProjector(nn.Module):
    """Scale reference latents or accept generated scaled latents, then project."""

    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.multi_modal_projector = VibeVoiceMultiModalProjector(
            config.acoustic_tokenizer.hidden_size,
            config.hidden_size,
        )
        self.latent_scaling_factor = nn.Parameter([])
        self.latent_bias_factor = nn.Parameter([])

    def forward(
        self,
        op: OpBuilder,
        latents: ir.Value,
        latents_are_scaled: ir.Value,
    ):
        scaled_reference = op.Mul(
            op.Add(latents, self.latent_bias_factor),
            self.latent_scaling_factor,
        )
        scaled_latents = op.Where(latents_are_scaled, latents, scaled_reference)
        return scaled_latents, self.multi_modal_projector(op, scaled_latents)


class VibeVoiceEmbeddingModel(nn.Module):
    """Qwen2 token embedding with optional reference-audio placeholder replacement."""

    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self._audio_token_id = config.audio_token_id

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        audio_embeds: ir.Value,
        replace_audio_tokens: ir.Value,
    ):
        inputs_embeds = self.embed_tokens(op, input_ids)
        audio_mask = op.And(
            op.Equal(input_ids, self._audio_token_id),
            replace_audio_tokens,
        )
        audio_positions = op.Transpose(
            op.NonZero(audio_mask),
            perm=[1, 0],
        )
        return op.ScatterND(inputs_embeds, audio_positions, audio_embeds)


class VibeVoiceDecoderModel(nn.Module):
    """Qwen2 decoder returning vocabulary logits and post-norm hidden states."""

    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = initialize_rope(config)
        self._dtype = config.dtype

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
            hidden_states, present = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present)
        hidden_states = self.norm(op, hidden_states)
        return self.lm_head(op, hidden_states), hidden_states, present_key_values


class _DiffusionTimestepEmbedding(nn.Module):
    def __init__(self, config: VibeVoiceDiffusionConfig):
        super().__init__()
        half = config.frequency_embedding_size // 2
        self._frequency = ir.tensor(
            np.exp(
                -math.log(config.diffusion_max_period)
                * np.arange(half, dtype=np.float32)
                / half
            ).astype(np.float32)
        )

    def forward(self, op: OpBuilder, timesteps: ir.Value):
        timesteps_f32 = op.Cast(timesteps, to=ir.DataType.FLOAT)
        angles = op.Mul(op.Unsqueeze(timesteps_f32, [1]), op.Constant(value=self._frequency))
        embedding = op.Concat(op.Cos(angles), op.Sin(angles), axis=-1)
        return op.CastLike(embedding, timesteps)


class _DiffusionTimestepMLP(nn.Module):
    def __init__(self, config: VibeVoiceDiffusionConfig):
        super().__init__()
        self.fc1 = Linear(config.frequency_embedding_size, config.hidden_size, bias=False)
        self.fc2 = Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        return self.fc2(op, op.Swish(self.fc1(op, hidden_states)))


class _DiffusionAdaLayerNorm(nn.Module):
    def __init__(self, config: VibeVoiceDiffusionConfig):
        super().__init__()
        self.ffn = GatedMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act,
            bias=config.mlp_bias,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.linear = Linear(config.hidden_size, 3 * config.hidden_size, bias=False)

    def forward(self, op: OpBuilder, hidden_states: ir.Value, condition: ir.Value):
        shift, scale, gate = op.Split(
            self.linear(op, op.Swish(condition)),
            num_outputs=3,
            axis=-1,
            _outputs=3,
        )
        modulated = op.Add(
            op.Mul(self.norm(op, hidden_states), op.Add(scale, 1.0)),
            shift,
        )
        return op.Add(hidden_states, op.Mul(gate, self.ffn(op, modulated)))


class _DiffusionFinalLayer(nn.Module):
    def __init__(self, config: VibeVoiceDiffusionConfig):
        super().__init__()
        self.linear_1 = Linear(config.hidden_size, 2 * config.hidden_size, bias=False)
        self.linear_2 = Linear(config.hidden_size, config.latent_size, bias=False)
        self._hidden_size = config.hidden_size
        self._eps = config.rms_norm_eps

    def forward(self, op: OpBuilder, hidden_states: ir.Value, condition: ir.Value):
        shift, scale = op.Split(
            self.linear_1(op, op.Swish(condition)),
            num_outputs=2,
            axis=-1,
            _outputs=2,
        )
        unit_scale = op.CastLike(
            op.Constant(value=ir.tensor(np.ones(self._hidden_size, dtype=np.float32))),
            hidden_states,
        )
        normed = op.RMSNormalization(
            hidden_states,
            unit_scale,
            axis=-1,
            epsilon=self._eps,
        )
        hidden_states = op.Add(op.Mul(normed, op.Add(scale, 1.0)), shift)
        return self.linear_2(op, hidden_states)


class VibeVoiceDiffusionHead(nn.Module):
    """Four-layer token-level AdaLN diffusion velocity predictor."""

    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        diffusion = config.diffusion_head
        self.noisy_images_proj = Linear(
            diffusion.latent_size,
            diffusion.hidden_size,
            bias=False,
        )
        self.cond_proj = Linear(diffusion.hidden_size, diffusion.hidden_size, bias=False)
        self.timestep_embedding = _DiffusionTimestepEmbedding(diffusion)
        self.timestep_proj = _DiffusionTimestepMLP(diffusion)
        self.layers = nn.ModuleList(
            [_DiffusionAdaLayerNorm(diffusion) for _ in range(diffusion.num_hidden_layers)]
        )
        self.final_layer = _DiffusionFinalLayer(diffusion)

    def forward(
        self,
        op: OpBuilder,
        noisy_images: ir.Value,
        timesteps: ir.Value,
        condition: ir.Value,
    ):
        hidden_states = self.noisy_images_proj(op, noisy_images)
        condition = op.Add(
            self.cond_proj(op, condition),
            self.timestep_proj(op, self.timestep_embedding(op, timesteps)),
        )
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, condition)
        return self.final_layer(op, hidden_states, condition)


class VibeVoiceAudioDecoder(nn.Module):
    """Streaming acoustic decoder with in-graph inverse latent scaling."""

    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.decoder = VibeVoiceTokenizerDecoder(config.acoustic_tokenizer)
        self.latent_scaling_factor = nn.Parameter([])
        self.latent_bias_factor = nn.Parameter([])
        self.cache_specs = self.decoder.cache_specs

    def forward(
        self,
        op: OpBuilder,
        scaled_latents: ir.Value,
        past_conv_states: Sequence[ir.Value],
    ):
        latents = op.Sub(
            op.Div(scaled_latents, self.latent_scaling_factor),
            self.latent_bias_factor,
        )
        return self.decoder(op, latents, past_conv_states)


class VibeVoiceSemanticEncoder(nn.Module):
    """Streaming semantic tokenizer used to feed generated audio back to the LM."""

    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.encoder = VibeVoiceTokenizerEncoder(config.semantic_tokenizer)
        self.cache_specs = self.encoder.cache_specs

    def forward(
        self,
        op: OpBuilder,
        waveform: ir.Value,
        past_conv_states: Sequence[ir.Value],
    ):
        return self.encoder(op, waveform, past_conv_states)


class VibeVoiceSemanticProjector(nn.Module):
    """Project streaming semantic latents into the Qwen2 embedding space."""

    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.semantic_connector = VibeVoiceMultiModalProjector(
            config.semantic_tokenizer.hidden_size,
            config.hidden_size,
        )

    def forward(self, op: OpBuilder, semantic_latents: ir.Value):
        return self.semantic_connector(op, semantic_latents)


class VibeVoiceForConditionalGeneration(nn.Module):
    """Eight-stage VibeVoice 1.5B continuous-token TTS pipeline.

    ```{mermaid}
    flowchart LR
        Ref[Reference audio] --> AE[Audio encoder]
        AE --> AP[Acoustic projector]
        Text[Text tokens] --> Emb[Embedding mixer]
        AP --> Emb
        Emb --> Qwen[Qwen2 decoder]
        Qwen --> PosCond[Positive decoder condition]
        Qwen --> NegCond[Negative decoder condition]
        PosCond --> Diff[DPM-Solver diffusion head]
        NegCond --> Diff
        Qwen --> PosKV[Positive KV cache]
        Qwen --> NegKV[Negative CFG KV cache]
        PosKV --> Qwen
        NegKV --> Qwen
        Diff --> Latent[64-D acoustic latent]
        Latent --> AD[Streaming audio decoder]
        AD --> Wave[3200-sample waveform chunk]
        Wave --> SE[Semantic encoder]
        SE --> SP[Semantic projector]
        SP --> Qwen
    ```

    The exported package contains the audio encoder, acoustic projector,
    embedding mixer, Qwen2 decoder, diffusion head, audio decoder, semantic
    encoder, and semantic projector. For every generated audio token, the host
    runs the positive and negative CFG decoder branches, samples one 64-D
    acoustic latent with DPM-Solver, emits a 3200-sample waveform chunk, and
    feeds its semantic embedding into the next decoder step.

    The decoder owns dual Qwen KV caches. The streaming acoustic decoder and
    semantic encoder each expose their convolution histories, for 34 explicit
    cache slots per tokenizer stack. The negative CFG branch resets to a valid
    suffix after audio BOS; because ``GroupQueryAttention`` only represents
    prefix-valid lengths through ``seqlens_k``, this decoder declares an
    arbitrary-mask contract and retains standard ONNX ``Attention``. The host,
    rather than ONNX Runtime GenAI, owns this multi-stage orchestration; the
    package is therefore exportable but not an OGA runnable model.
    """

    default_task = "vibevoice-tts"
    category = "Audio"
    config_class = VibeVoiceConfig

    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "audio_encoder": ("model.audio_tower.encoder",),
        "audio_projection": (
            "model.multi_modal_projector",
            "model.latent_scaling_factor",
            "model.latent_bias_factor",
        ),
        "embedding": ("model.language_model.embed_tokens",),
        "decoder": (
            "model.language_model.layers",
            "model.language_model.norm",
            "lm_head",
        ),
        "diffusion_head": ("model.diffusion_head",),
        "audio_decoder": (
            "model.audio_tower.decoder",
            "model.latent_scaling_factor",
            "model.latent_bias_factor",
        ),
        "semantic_encoder": ("model.semantic_tokenizer_encoder",),
        "semantic_projection": ("model.semantic_connector",),
    }

    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.config = config
        self.audio_encoder = VibeVoiceReferenceAudioEncoder(config)
        self.audio_projection = VibeVoiceAcousticProjector(config)
        self.embedding = VibeVoiceEmbeddingModel(config)
        self.decoder = VibeVoiceDecoderModel(config)
        self.diffusion_head = VibeVoiceDiffusionHead(config)
        self.audio_decoder = VibeVoiceAudioDecoder(config)
        self.semantic_encoder = VibeVoiceSemanticEncoder(config)
        self.semantic_projection = VibeVoiceSemanticProjector(config)

    def forward(self, op: OpBuilder, *args, **kwargs):
        raise NotImplementedError("VibeVoiceTask exports each TTS stage independently")

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Route the native HF composite checkpoint to standardized package stages."""
        routed: dict[str, torch.Tensor] = {}
        stage_prefixes = tuple(f"{name}." for name in self.HF_COMPONENT_SOURCES)
        for key, value in state_dict.items():
            if key.startswith(stage_prefixes):
                routed[key] = value
            elif key.startswith("model.audio_tower.encoder."):
                suffix = key.removeprefix("model.audio_tower.encoder.")
                routed[f"audio_encoder.encoder.{suffix}"] = value
            elif key.startswith("model.audio_tower.decoder."):
                suffix = key.removeprefix("model.audio_tower.decoder.")
                routed[f"audio_decoder.decoder.{suffix}"] = value
            elif key.startswith("model.semantic_tokenizer_encoder."):
                suffix = key.removeprefix("model.semantic_tokenizer_encoder.")
                routed[f"semantic_encoder.encoder.{suffix}"] = value
            elif key.startswith("model.multi_modal_projector."):
                suffix = key.removeprefix("model.multi_modal_projector.")
                routed[f"audio_projection.multi_modal_projector.{suffix}"] = value
            elif key.startswith("model.semantic_connector."):
                suffix = key.removeprefix("model.semantic_connector.")
                routed[f"semantic_projection.semantic_connector.{suffix}"] = value
            elif key.startswith("model.language_model.embed_tokens."):
                suffix = key.removeprefix("model.language_model.embed_tokens.")
                routed[f"embedding.embed_tokens.{suffix}"] = value
                if suffix == "weight":
                    routed["decoder.lm_head.weight"] = value
            elif key.startswith("model.language_model.layers."):
                suffix = key.removeprefix("model.language_model.")
                routed[f"decoder.{suffix}"] = value
            elif key.startswith("model.language_model.norm."):
                suffix = key.removeprefix("model.language_model.")
                routed[f"decoder.{suffix}"] = value
            elif key.startswith("model.diffusion_head."):
                suffix = key.removeprefix("model.diffusion_head.")
                routed[f"diffusion_head.{suffix}"] = value
            elif key in {"model.latent_scaling_factor", "model.latent_bias_factor"}:
                suffix = key.removeprefix("model.")
                routed[f"audio_projection.{suffix}"] = value
                routed[f"audio_decoder.{suffix}"] = value
            elif key == "lm_head.weight":
                routed["decoder.lm_head.weight"] = value
        return routed
