# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph contracts for VibeVoice streaming ASR's staged inference pipeline."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import VibeVoiceASRConfig
from mobius._model_package import ModelPackage
from mobius._pipeline_contract import (
    declare_arbitrary_attention_mask,
    declare_component_presence,
    declare_optional_input,
)
from mobius.tasks._base import ComponentSpec, ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import _make_kv_cache_inputs, _register_kv_cache_outputs
from mobius.tasks._streaming_convolution import (
    make_conv_cache_inputs,
    register_conv_cache_outputs,
)


class VibeVoiceASRStreamingTask(ModelTask):
    """Build the executable ASR stages while retaining host-owned stream orchestration."""

    model_roles: ClassVar[dict[str, str]] = {
        "audio_encoder": "encoder",
        "embedding": "embedding",
        "decoder": "decoder",
    }
    components: ClassVar[ComponentSpec] = ComponentSpec(
        audio_encoder="audio_encoder",
        embedding="embedding",
        decoder="decoder",
    )

    def build(self, module: nn.Module, config: VibeVoiceASRConfig) -> ModelPackage:
        self._validate_components(module)
        return ModelPackage(
            {
                "audio_encoder": self._build_audio_encoder(module.audio_encoder, config),
                "embedding": self._build_embedding(module.embedding, config),
                "decoder": self._build_decoder(module.decoder, config),
            },
            config=config,
        )

    def _build_audio_encoder(self, module: nn.Module, config: VibeVoiceASRConfig) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        samples = ir.SymbolicDim("audio_samples")
        frames = ir.SymbolicDim("audio_frames")
        graph, builder = _make_graph(name="vibevoice_asr_audio_encoder")
        speech_tensors = builder.input(
            "speech_tensors",
            dtype=ir.DataType.FLOAT,
            shape=[batch, samples],
        )
        speech_masks = builder.input(
            "speech_masks",
            dtype=ir.DataType.BOOL,
            shape=[batch, frames],
        )
        acoustic_sample_noise = builder.input(
            "acoustic_sample_noise",
            dtype=config.dtype,
            shape=[batch],
        )
        acoustic_latent_noise = builder.input(
            "acoustic_latent_noise",
            dtype=config.dtype,
            shape=[batch, frames, config.acoustic_tokenizer.hidden_size],
        )
        is_final_chunk = builder.input("is_final_chunk", dtype=ir.DataType.BOOL, shape=[])
        acoustic_past = make_conv_cache_inputs(
            builder,
            module.acoustic_cache_specs,
            batch,
            config.dtype,
            prefix="past_acoustic_conv",
        )
        semantic_past = make_conv_cache_inputs(
            builder,
            module.semantic_cache_specs,
            batch,
            config.dtype,
            prefix="past_semantic_conv",
        )
        speech_embeds, acoustic_present, semantic_present = module(
            builder.op,
            speech_tensors,
            speech_masks,
            acoustic_sample_noise,
            acoustic_latent_noise,
            acoustic_past,
            semantic_past,
            is_final_chunk,
        )
        speech_embeds.shape = ir.Shape(["valid_speech_frames", config.hidden_size])
        builder.add_output(speech_embeds, "speech_embeds")
        register_conv_cache_outputs(
            builder,
            acoustic_present,
            prefix="present_acoustic_conv",
        )
        register_conv_cache_outputs(
            builder,
            semantic_present,
            prefix="present_semantic_conv",
        )
        declare_component_presence(graph, "audio")
        return _make_model(graph)

    def _build_embedding(self, module: nn.Module, config: VibeVoiceASRConfig) -> ir.Model:
        graph, builder = _make_graph(name="vibevoice_asr_embedding")
        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=["batch", "sequence_length"],
        )
        speech_embeds = builder.input(
            "speech_embeds",
            dtype=config.dtype,
            shape=["valid_speech_frames", config.hidden_size],
        )
        declare_optional_input(
            speech_embeds,
            presence="audio",
            absent_shape=[0, config.hidden_size],
        )
        acoustic_input_mask = builder.input(
            "acoustic_input_mask",
            dtype=ir.DataType.BOOL,
            shape=["batch", "sequence_length"],
        )
        inputs_embeds = module(
            builder.op,
            input_ids,
            speech_embeds,
            acoustic_input_mask,
        )
        builder.add_output(inputs_embeds, "inputs_embeds")
        return _make_model(graph)

    def _build_decoder(self, module: nn.Module, config: VibeVoiceASRConfig) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        sequence = ir.SymbolicDim("sequence_length")
        past_sequence = ir.SymbolicDim("past_sequence_length")
        graph, builder = _make_graph(name="vibevoice_asr_decoder")
        # The processor left-pads batches, making valid tokens a suffix. The
        # generic GQA ABI only represents valid prefixes, so it must stay unfused.
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
