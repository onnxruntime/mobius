# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph contracts for source-faithful VibeVoice Realtime TTS stages."""

from __future__ import annotations

import json
from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import VibeVoiceStreamingConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ComponentSpec, ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import (
    _make_conv_cache_inputs,
    _make_kv_cache_inputs,
    _register_conv_cache_outputs,
    _register_kv_cache_outputs,
)

_HOST_PROTOCOL_METADATA = "mobius.vibevoice_streaming.host_protocol"


class VibeVoiceStreamingTask(ModelTask):
    """Build the six neural stages used by VibeVoice Realtime's host loop."""

    model_roles: ClassVar[dict[str, str]] = {
        "embedding": "embedding",
        "lm_backbone": "decoder",
        "tts_backbone": "decoder",
        "speech_connector": "embedding",
        "diffusion_head": "denoiser",
        "audio_decoder": "encoder",
    }
    components: ClassVar[ComponentSpec] = ComponentSpec(
        embedding="embedding",
        lm_backbone="lm_backbone",
        tts_backbone="tts_backbone",
        speech_connector="speech_connector",
        diffusion_head="diffusion_head",
        audio_decoder="audio_decoder",
    )

    def build(self, module: nn.Module, config: VibeVoiceStreamingConfig) -> ModelPackage:
        self._validate_components(module)
        models = {
            "embedding": self._build_embedding(module.embedding, config),
            "lm_backbone": self._build_lm_backbone(module.lm_backbone, config),
            "tts_backbone": self._build_tts_backbone(module.tts_backbone, config),
            "speech_connector": self._build_speech_connector(module.speech_connector, config),
            "diffusion_head": self._build_diffusion_head(module.diffusion_head, config),
            "audio_decoder": self._build_audio_decoder(module.audio_decoder, config),
        }
        protocol = json.dumps(
            {
                "owner": "host",
                "voice_prompt": "prefilled_cache_only",
                "text_window_size": config.text_window_size,
                "speech_window_size": config.speech_window_size,
                "cfg_cache_families": ["lm", "tts_lm", "neg_lm", "neg_tts_lm"],
                "diffusion_scheduler": "DPMSolverMultistepScheduler",
                "onnxruntime_genai": "unsupported",
            },
            sort_keys=True,
        )
        for model in models.values():
            model.graph.metadata_props[_HOST_PROTOCOL_METADATA] = protocol
        return ModelPackage(models, config=config)

    def _build_embedding(
        self,
        module: nn.Module,
        config: VibeVoiceStreamingConfig,
    ) -> ir.Model:
        graph, builder = _make_graph(name="vibevoice_streaming_embedding")
        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=["batch", "sequence_length"],
        )
        inputs_embeds = module(builder.op, input_ids)
        builder.add_output(inputs_embeds, "inputs_embeds")
        return _make_model(graph)

    def _build_lm_backbone(
        self,
        module: nn.Module,
        config: VibeVoiceStreamingConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        sequence = ir.SymbolicDim("sequence_length")
        past_sequence = ir.SymbolicDim("past_sequence_length")
        graph, builder = _make_graph(name="vibevoice_streaming_lm_backbone")
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
            config.lm_backbone_num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_sequence,
        )
        hidden_states, present = module(
            builder.op,
            inputs_embeds,
            attention_mask,
            position_ids,
            past,
        )
        builder.add_output(hidden_states, "last_hidden_state")
        _register_kv_cache_outputs(builder, present)
        return _make_model(graph)

    def _build_tts_backbone(
        self,
        module: nn.Module,
        config: VibeVoiceStreamingConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        sequence = ir.SymbolicDim("sequence_length")
        replacement_sequence = ir.SymbolicDim("replacement_sequence_length")
        past_sequence = ir.SymbolicDim("past_sequence_length")
        graph, builder = _make_graph(name="vibevoice_streaming_tts_backbone")
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
        lm_last_hidden_state = builder.input(
            "lm_last_hidden_state",
            dtype=config.dtype,
            shape=[batch, replacement_sequence, config.hidden_size],
        )
        # The source provides a [B, 1] text/speech type and broadcasts it over
        # each current window, which keeps text prefill and speech decode identical.
        tts_text_masks = builder.input(
            "tts_text_masks",
            dtype=ir.DataType.BOOL,
            shape=[batch, 1],
        )
        past = _make_kv_cache_inputs(
            builder,
            config.tts_backbone_num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_sequence,
        )
        eos_logits, hidden_states, present = module(
            builder.op,
            inputs_embeds,
            attention_mask,
            position_ids,
            past,
            lm_last_hidden_state=lm_last_hidden_state,
            tts_text_masks=tts_text_masks,
        )
        builder.add_output(eos_logits, "eos_logits")
        builder.add_output(hidden_states, "last_hidden_state")
        _register_kv_cache_outputs(builder, present)
        return _make_model(graph)

    def _build_speech_connector(
        self,
        module: nn.Module,
        config: VibeVoiceStreamingConfig,
    ) -> ir.Model:
        graph, builder = _make_graph(name="vibevoice_streaming_speech_connector")
        speech_latents = builder.input(
            "speech_latents",
            dtype=config.dtype,
            shape=["batch", "acoustic_frames", config.acoustic_tokenizer.vae_dim],
        )
        speech_embeds = module(builder.op, speech_latents)
        builder.add_output(speech_embeds, "speech_embeds")
        return _make_model(graph)

    def _build_diffusion_head(
        self,
        module: nn.Module,
        config: VibeVoiceStreamingConfig,
    ) -> ir.Model:
        diffusion = config.diffusion_head
        graph, builder = _make_graph(name="vibevoice_streaming_diffusion_head")
        noisy_speech_latents = builder.input(
            "noisy_speech_latents",
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
        velocity = module(builder.op, noisy_speech_latents, timesteps, condition)
        builder.add_output(velocity, "velocity")
        return _make_model(graph)

    def _build_audio_decoder(
        self,
        module: nn.Module,
        config: VibeVoiceStreamingConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        graph, builder = _make_graph(name="vibevoice_streaming_audio_decoder")
        speech_latents = builder.input(
            "speech_latents",
            dtype=config.dtype,
            shape=[batch, "acoustic_frames", config.acoustic_tokenizer.vae_dim],
        )
        past = _make_conv_cache_inputs(builder, module.cache_specs, batch, config.dtype)
        waveform, present = module(builder.op, speech_latents, past)
        waveform.shape = ir.Shape([batch, config.acoustic_tokenizer.channels, "audio_samples"])
        builder.add_output(waveform, "waveform")
        _register_conv_cache_outputs(builder, present)
        return _make_model(graph)
