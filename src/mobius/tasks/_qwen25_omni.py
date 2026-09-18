# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen2.5-Omni Thinker and Talker split task."""

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
from mobius.tasks._cache_utils import (
    _make_kv_cache_inputs,
    _register_kv_cache_outputs,
)
from mobius.tasks._vision_language_3model import QwenVLTask


class Qwen25OmniTask(QwenVLTask):
    """Build the Thinker models and optional Talker ONNX models."""

    model_roles: ClassVar[dict[str, str]] = {
        "audio_encoder": "encoder",
        "vision_encoder": "encoder",
        "embedding": "embedding",
        "decoder": "decoder",
        "talker_embedding": "embedding",
        "talker": "decoder",
    }
    components = ComponentSpec(
        audio_encoder="audio_encoder",
        vision_encoder="vision_encoder",
        embedding="embedding",
        decoder="decoder",
    )

    def build(self, module: nn.Module, config: ArchitectureConfig) -> ModelPackage:
        self._validate_components(module)
        models: dict[str, ir.Model] = {
            "audio_encoder": self._build_audio(module.audio_encoder, config),
            "vision_encoder": self._build_vision(module.vision_encoder, config),
            "embedding": self._build_embedding(module.embedding, config),
            "decoder": build_decoder_from_embeds(module.decoder, config, mrope=True),
        }
        if module.talker is not None and config.talker is not None:
            models["talker_embedding"] = self._build_talker_embedding(
                module.talker.model.embed_tokens, config.talker
            )
            models["talker"] = self._build_talker(module.talker, config.talker)
        return ModelPackage(models, config=config)

    def _build_talker_embedding(
        self,
        embedding: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build codec-token embeddings in the shared Thinker-width space."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        graph, builder = _make_graph(name="talker_embedding")
        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )
        codec_embeds = embedding(builder.op, input_ids)
        builder.add_output(codec_embeds, "codec_embeds")
        return _make_model(graph)

    def _build_talker(
        self,
        talker: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build speech-token logits from preconstructed shared-space embeddings."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")
        embedding_size = config.embedding_size or config.hidden_size

        graph, builder = _make_graph(name="talker")
        inputs_embeds = builder.input(
            "inputs_embeds",
            dtype=config.dtype,
            shape=[batch, seq_len, embedding_size],
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_seq_len + seq_len"],
        )
        position_ids = builder.input(
            "position_ids",
            dtype=ir.DataType.INT64,
            shape=[3, batch, seq_len],
        )
        past_key_values = _make_kv_cache_inputs(
            builder,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )
        logits, present_key_values = talker(
            builder.op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        builder.add_output(logits, "logits")
        _register_kv_cache_outputs(builder, present_key_values)
        return _make_model(graph)

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
