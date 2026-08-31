# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph contracts for the multi-stage VibeVoice text-to-speech pipeline."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import VibeVoiceConfig
from mobius._model_package import ModelPackage
from mobius._pipeline_contract import (
    declare_arbitrary_attention_mask,
    declare_component_presence,
    declare_optional_input,
)
from mobius.tasks._base import ComponentSpec, ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import (
    _make_kv_cache_inputs,
    _register_kv_cache_outputs,
)


def _make_conv_cache_inputs(
    builder,
    specs: tuple[tuple[int, int], ...],
    batch: ir.SymbolicDim,
    dtype: ir.DataType,
) -> list[ir.Value]:
    return [
        builder.input(
            f"past_conv.{index}",
            dtype=dtype,
            shape=[batch, channels, left_pad],
        )
        for index, (channels, left_pad) in enumerate(specs)
    ]


def _register_conv_cache_outputs(builder, values: list[ir.Value]) -> None:
    for index, value in enumerate(values):
        builder.add_output(value, f"present_conv.{index}")


class VibeVoiceTask(ModelTask):
    """Build all neural stages needed by VibeVoice's continuous-token TTS loop."""

    model_roles: ClassVar[dict[str, str]] = {
        "audio_encoder": "encoder",
        "audio_projection": "embedding",
        "embedding": "embedding",
        "decoder": "decoder",
        "diffusion_head": "denoiser",
        "audio_decoder": "encoder",
        "semantic_encoder": "encoder",
        "semantic_projection": "embedding",
    }
    components: ClassVar[ComponentSpec] = ComponentSpec(
        audio_encoder="audio_encoder",
        audio_projection="audio_projection",
        embedding="embedding",
        decoder="decoder",
        diffusion_head="diffusion_head",
        audio_decoder="audio_decoder",
        semantic_encoder="semantic_encoder",
        semantic_projection="semantic_projection",
    )

    def build(self, module: nn.Module, config: VibeVoiceConfig) -> ModelPackage:
        self._validate_components(module)
        models = {
            "audio_encoder": self._build_audio_encoder(module.audio_encoder, config),
            "audio_projection": self._build_audio_projection(
                module.audio_projection, config
            ),
            "embedding": self._build_embedding(module.embedding, config),
            "decoder": self._build_decoder(module.decoder, config),
            "diffusion_head": self._build_diffusion_head(module.diffusion_head, config),
            "audio_decoder": self._build_audio_decoder(module.audio_decoder, config),
            "semantic_encoder": self._build_semantic_encoder(
                module.semantic_encoder, config
            ),
            "semantic_projection": self._build_semantic_projection(
                module.semantic_projection, config
            ),
        }
        return ModelPackage(models, config=config)

    def _build_audio_encoder(
        self,
        module: nn.Module,
        config: VibeVoiceConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("audio_batch")
        samples = ir.SymbolicDim("audio_samples")
        frames = ir.SymbolicDim("audio_frames")
        latent_size = config.acoustic_tokenizer.hidden_size
        graph, builder = _make_graph(name="vibevoice_audio_encoder")
        input_values = builder.input(
            "input_values",
            dtype=ir.DataType.FLOAT,
            shape=[batch, config.acoustic_tokenizer.channels, samples],
        )
        padding_mask = builder.input(
            "padding_mask",
            dtype=ir.DataType.BOOL,
            shape=[batch, samples],
        )
        sample_noise = builder.input(
            "sample_noise",
            dtype=config.dtype,
            shape=[batch],
        )
        latent_noise = builder.input(
            "latent_noise",
            dtype=config.dtype,
            shape=[batch, frames, latent_size],
        )
        latents = module(
            builder.op,
            input_values,
            padding_mask,
            sample_noise,
            latent_noise,
        )
        latents.shape = ir.Shape(["valid_audio_frames", latent_size])
        builder.add_output(latents, "audio_latents")
        declare_component_presence(graph, "audio")
        return _make_model(graph)

    def _build_audio_projection(
        self,
        module: nn.Module,
        config: VibeVoiceConfig,
    ) -> ir.Model:
        graph, builder = _make_graph(name="vibevoice_audio_projection")
        latents = builder.input(
            "audio_latents",
            dtype=config.dtype,
            shape=["audio_frames", config.acoustic_tokenizer.hidden_size],
        )
        latents_are_scaled = builder.input(
            "latents_are_scaled",
            dtype=ir.DataType.BOOL,
            shape=[],
        )
        scaled, embeds = module(builder.op, latents, latents_are_scaled)
        builder.add_output(scaled, "scaled_audio_latents")
        builder.add_output(embeds, "audio_embeds")
        return _make_model(graph)

    def _build_embedding(
        self,
        module: nn.Module,
        config: VibeVoiceConfig,
    ) -> ir.Model:
        graph, builder = _make_graph(name="vibevoice_embedding")
        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=["batch", "sequence_length"],
        )
        audio_embeds = builder.input(
            "audio_embeds",
            dtype=config.dtype,
            shape=["audio_frames", config.hidden_size],
        )
        declare_optional_input(
            audio_embeds,
            presence="audio",
            absent_shape=[0, config.hidden_size],
        )
        replace_audio_tokens = builder.input(
            "replace_audio_tokens",
            dtype=ir.DataType.BOOL,
            shape=[],
        )
        inputs_embeds = module(
            builder.op,
            input_ids,
            audio_embeds,
            replace_audio_tokens,
        )
        builder.add_output(inputs_embeds, "inputs_embeds")
        return _make_model(graph)

    def _build_decoder(
        self,
        module: nn.Module,
        config: VibeVoiceConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        sequence = ir.SymbolicDim("sequence_length")
        past_sequence = ir.SymbolicDim("past_sequence_length")
        graph, builder = _make_graph(name="vibevoice_decoder")
        # The CFG negative branch resets to a valid suffix after every audio
        # BOS. GQA's seqlens_k ABI only represents valid prefixes.
        declare_arbitrary_attention_mask(graph)
        inputs_embeds = builder.input(
            "inputs_embeds",
            dtype=config.dtype,
            shape=[batch, sequence, config.hidden_size],
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_sequence_length + sequence_length"],
        )
        position_ids = builder.input(
            "position_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, sequence],
        )
        past = _make_kv_cache_inputs(
            builder,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_sequence,
        )
        logits, hidden_states, present = module(
            builder.op,
            inputs_embeds,
            attention_mask,
            position_ids,
            past,
        )
        builder.add_output(logits, "logits")
        builder.add_output(hidden_states, "last_hidden_state")
        _register_kv_cache_outputs(builder, present)
        return _make_model(graph)

    def _build_diffusion_head(
        self,
        module: nn.Module,
        config: VibeVoiceConfig,
    ) -> ir.Model:
        diffusion = config.diffusion_head
        graph, builder = _make_graph(name="vibevoice_diffusion_head")
        noisy = builder.input(
            "noisy_audio_latents",
            dtype=config.dtype,
            shape=["guidance_batch", diffusion.latent_size],
        )
        timesteps = builder.input(
            "timesteps",
            dtype=config.dtype,
            shape=["guidance_batch"],
        )
        condition = builder.input(
            "condition",
            dtype=config.dtype,
            shape=["guidance_batch", diffusion.hidden_size],
        )
        velocity = module(builder.op, noisy, timesteps, condition)
        builder.add_output(velocity, "velocity")
        return _make_model(graph)

    def _build_audio_decoder(
        self,
        module: nn.Module,
        config: VibeVoiceConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        graph, builder = _make_graph(name="vibevoice_audio_decoder")
        scaled_latents = builder.input(
            "scaled_audio_latents",
            dtype=config.dtype,
            shape=[batch, "audio_frames", config.acoustic_tokenizer.hidden_size],
        )
        past = _make_conv_cache_inputs(
            builder,
            module.cache_specs,
            batch,
            config.dtype,
        )
        waveform, present = module(builder.op, scaled_latents, past)
        waveform.shape = ir.Shape([batch, config.acoustic_tokenizer.channels, "audio_samples"])
        builder.add_output(waveform, "waveform")
        _register_conv_cache_outputs(builder, present)
        return _make_model(graph)

    def _build_semantic_encoder(
        self,
        module: nn.Module,
        config: VibeVoiceConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        graph, builder = _make_graph(name="vibevoice_semantic_encoder")
        waveform = builder.input(
            "waveform",
            dtype=config.dtype,
            shape=[batch, config.semantic_tokenizer.channels, "audio_samples"],
        )
        past = _make_conv_cache_inputs(
            builder,
            module.cache_specs,
            batch,
            config.dtype,
        )
        latents, present = module(builder.op, waveform, past)
        builder.add_output(latents, "semantic_latents")
        _register_conv_cache_outputs(builder, present)
        return _make_model(graph)

    def _build_semantic_projection(
        self,
        module: nn.Module,
        config: VibeVoiceConfig,
    ) -> ir.Model:
        graph, builder = _make_graph(name="vibevoice_semantic_projection")
        latents = builder.input(
            "semantic_latents",
            dtype=config.dtype,
            shape=["batch", "audio_frames", config.semantic_tokenizer.hidden_size],
        )
        embeds = module(builder.op, latents)
        builder.add_output(embeds, "semantic_embeds")
        return _make_model(graph)
