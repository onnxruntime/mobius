# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen2.5-Omni Thinker four-model split task."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ComponentSpec,
    _make_graph,
    _make_model,
    build_decoder_from_embeds,
)
from mobius.tasks._vision_language_3model import QwenVLTask


class Qwen25OmniTask(QwenVLTask):
    """Build the Thinker's audio, vision, embedding, and decoder ONNX models."""

    model_roles: ClassVar[dict[str, str]] = {
        "audio_encoder": "encoder",
        "vision_encoder": "encoder",
        "embedding": "embedding",
        "decoder": "decoder",
    }
    components = ComponentSpec(
        audio_encoder="audio_encoder",
        vision_encoder="vision_encoder",
        embedding="embedding",
        decoder="decoder",
    )

    def build(self, module: nn.Module, config: ArchitectureConfig) -> ModelPackage:
        self._validate_components(module)
        models = {
            "audio_encoder": self._build_audio(module.audio_encoder, config),
            "vision_encoder": self._build_vision(module.vision_encoder, config),
            "embedding": self._build_embedding(module.embedding, config),
            "decoder": build_decoder_from_embeds(module.decoder, config, mrope=True),
        }
        return ModelPackage(models, config=config)

    def _build_audio(self, audio_encoder: nn.Module, config: ArchitectureConfig) -> ir.Model:
        """Build packed audio chunks into packed LLM audio tokens."""
        num_chunks = ir.SymbolicDim("num_audio_chunks")
        chunk_len = ir.SymbolicDim("audio_chunk_len")
        num_audio_tokens = ir.SymbolicDim("num_audio_tokens")
        n_mels = (config.audio.num_mel_bins if config.audio else None) or 128

        graph, builder = _make_graph(name="audio_encoder")
        input_features = builder.input(
            "input_features",
            dtype=ir.DataType.FLOAT,
            shape=[num_chunks, n_mels, chunk_len],
        )
        chunk_lengths = builder.input(
            "chunk_lengths",
            dtype=ir.DataType.INT64,
            shape=[num_chunks],
        )
        pool_indices = builder.input(
            "pool_indices",
            dtype=ir.DataType.INT64,
            shape=[num_audio_tokens],
        )
        audio_features = audio_encoder(
            builder.op,
            input_features,
            chunk_lengths,
            pool_indices,
        )
        builder.add_output(audio_features, "audio_features")
        return _make_model(graph)

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build text embedding and three-modality feature replacement."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        num_audio_tokens = ir.SymbolicDim("num_audio_tokens")
        num_image_tokens = ir.SymbolicDim("num_image_tokens")
        num_video_tokens = ir.SymbolicDim("num_video_tokens")

        graph, builder = _make_graph(name="embedding")
        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )
        audio_features = builder.input(
            "audio_features",
            dtype=config.dtype,
            shape=[num_audio_tokens, config.hidden_size],
        )
        image_features = builder.input(
            "image_features",
            dtype=config.dtype,
            shape=[num_image_tokens, config.hidden_size],
        )
        video_features = builder.input(
            "video_features",
            dtype=config.dtype,
            shape=[num_video_tokens, config.hidden_size],
        )
        inputs_embeds = embedding(
            builder.op,
            input_ids,
            audio_features,
            image_features,
            video_features,
        )
        builder.add_output(inputs_embeds, "inputs_embeds")
        return _make_model(graph)
