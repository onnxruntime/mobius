# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph contracts for the staged, offline VibeVoice ASR pipeline."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import VibeVoiceASRConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ComponentSpec,
    ModelTask,
    _make_graph,
    _make_model,
    build_decoder_from_embeds,
    build_embedding_from_features,
)
from mobius.tasks._vibevoice import _make_conv_cache_inputs, _register_conv_cache_outputs


class VibeVoiceASRTask(ModelTask):
    """Build cached audio encoders, deterministic connectors, embedding, and decoder."""

    model_roles: ClassVar[dict[str, str]] = {
        "acoustic_encoder": "encoder",
        "semantic_encoder": "encoder",
        "connectors": "embedding",
        "embedding": "embedding",
        # ASR receives ordinary left-padded prefix masks. Its causal decoder can
        # use GQA fusion; the TTS arbitrary-mask exception does not apply.
        "decoder": "decoder",
    }
    components = ComponentSpec(
        acoustic_encoder="acoustic_encoder",
        semantic_encoder="semantic_encoder",
        connectors="connectors",
        embedding="embedding",
        decoder="decoder",
    )

    def build(self, module: nn.Module, config: VibeVoiceASRConfig) -> ModelPackage:
        self._validate_components(module)
        return ModelPackage(
            {
                "acoustic_encoder": self._build_encoder(
                    "vibevoice_asr_acoustic_encoder",
                    module.acoustic_encoder,
                    config,
                    config.acoustic_tokenizer.hidden_size,
                ),
                "semantic_encoder": self._build_encoder(
                    "vibevoice_asr_semantic_encoder",
                    module.semantic_encoder,
                    config,
                    config.semantic_tokenizer.hidden_size,
                ),
                "connectors": self._build_connectors(module.connectors, config),
                "embedding": build_embedding_from_features(
                    module.embedding,
                    config,
                    feature_name="audio_features",
                    feature_dim=config.hidden_size,
                ),
                "decoder": build_decoder_from_embeds(module.decoder, config),
            },
            config=config,
        )

    @staticmethod
    def _build_encoder(
        name: str,
        module: nn.Module,
        config: VibeVoiceASRConfig,
        latent_size: int,
    ) -> ir.Model:
        batch = ir.SymbolicDim("audio_batch")
        samples = ir.SymbolicDim("audio_samples")
        frames = ir.SymbolicDim("audio_frames")
        graph, builder = _make_graph(name=name)
        input_values = builder.input(
            "input_values",
            dtype=ir.DataType.FLOAT,
            shape=[batch, 1, samples],
        )
        past = _make_conv_cache_inputs(builder, module.cache_specs, batch, config.dtype)
        is_final_chunk = builder.input(
            "is_final_chunk",
            dtype=ir.DataType.BOOL,
            shape=[1],
        )
        latents, present = module(builder.op, input_values, past, is_final_chunk)
        latents.shape = ir.Shape([batch, frames, latent_size])
        builder.add_output(latents, "audio_latents")
        _register_conv_cache_outputs(builder, present)
        return _make_model(graph)

    @staticmethod
    def _build_connectors(
        module: nn.Module,
        config: VibeVoiceASRConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("audio_batch")
        samples = ir.SymbolicDim("audio_samples")
        frames = ir.SymbolicDim("audio_frames")
        graph, builder = _make_graph(name="vibevoice_asr_connectors")
        acoustic_latents = builder.input(
            "acoustic_latents",
            dtype=config.dtype,
            shape=[batch, frames, config.acoustic_tokenizer.hidden_size],
        )
        semantic_latents = builder.input(
            "semantic_latents",
            dtype=config.dtype,
            shape=[batch, frames, config.semantic_tokenizer.hidden_size],
        )
        padding_mask = builder.input(
            "padding_mask",
            dtype=ir.DataType.BOOL,
            shape=[batch, samples],
        )
        acoustic_noise_scale = builder.input(
            "acoustic_noise_scale",
            dtype=config.dtype,
            shape=[batch],
        )
        acoustic_latent_noise = builder.input(
            "acoustic_latent_noise",
            dtype=config.dtype,
            shape=[batch, frames, config.acoustic_tokenizer.hidden_size],
        )
        audio_features, audio_feature_lengths = module(
            builder.op,
            acoustic_latents,
            semantic_latents,
            padding_mask,
            acoustic_noise_scale,
            acoustic_latent_noise,
        )
        audio_features.shape = ir.Shape(["valid_audio_frames", config.hidden_size])
        builder.add_output(audio_features, "audio_features")
        builder.add_output(audio_feature_lengths, "audio_feature_lengths")
        return _make_model(graph)
