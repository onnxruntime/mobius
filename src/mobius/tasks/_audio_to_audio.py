# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Audio-to-audio task for end-to-end speech models.

Builds separate ONNX models for audio-to-audio pipelines like
LFM2-Audio and Moshi/PersonaPlex. These models take audio in and
produce audio out, with an intermediate language model backbone.

Typical model split:
1. **audio_encoder**: mel/waveform → audio features (Conformer/encoder)
2. **decoder**: inputs_embeds → logits + KV cache (hybrid conv+attention LM)
3. **embedding**: text + audio token fusion → inputs_embeds
4. **audio_decoder**: codec codes → waveform (optional, for Mimi/codec output)
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_kv_cache_inputs,
    _make_model,
    _register_kv_cache_outputs,
)


class AudioToAudioTask(ModelTask):
    """Multi-model split for audio-to-audio models.

    The module must provide sub-modules as attributes:

    - ``audio_encoder``: audio encoder taking mel/waveform input
    - ``embedding``: embedding model fusing text + audio features
    - ``decoder``: language model backbone with KV cache
    - ``audio_decoder`` (optional): codec decoder for waveform synthesis

    Each sub-module is wired into its own ONNX graph.
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        models: dict[str, ir.Model] = {}

        models["audio_encoder"] = self._build_audio_encoder(module.audio_encoder, config)
        models["embedding"] = self._build_embedding(module.embedding, config)
        models["decoder"] = self._build_decoder(module.decoder, config)

        if hasattr(module, "audio_decoder"):
            models["audio_decoder"] = self._build_audio_decoder(module.audio_decoder, config)

        return ModelPackage(models, config=config)

    def _build_audio_encoder(
        self,
        audio_encoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build audio encoder: mel (batch, n_mels, time) → audio features."""
        batch = ir.SymbolicDim("batch")
        mel_seq = ir.SymbolicDim("mel_sequence_len")
        n_mels = config.audio.num_mel_bins or 128 if config.audio else 128

        input_features = ir.Value(
            name="input_features",
            shape=ir.Shape([batch, n_mels, mel_seq]),
            type=ir.TensorType(config.dtype),
        )

        graph, builder = _make_graph([input_features], name="audio_encoder")
        audio_features = audio_encoder(builder.op, input_features)

        audio_features.name = "audio_features"
        graph.outputs.append(audio_features)
        return _make_model(graph)

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build embedding: text_ids + audio_features → inputs_embeds."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        num_audio_tokens = ir.SymbolicDim("num_audio_tokens")
        output_dim = (
            config.audio.output_dim or config.hidden_size
            if config.audio
            else config.hidden_size
        )

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        audio_features = ir.Value(
            name="audio_features",
            shape=ir.Shape([num_audio_tokens, output_dim]),
            type=ir.TensorType(config.dtype),
        )

        graph, builder = _make_graph([input_ids, audio_features], name="embedding")
        inputs_embeds = embedding(
            builder.op,
            input_ids=input_ids,
            audio_features=audio_features,
        )

        inputs_embeds.name = "inputs_embeds"
        graph.outputs.append(inputs_embeds)
        return _make_model(graph)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build decoder: inputs_embeds → logits + KV cache.

        Uses 1D position_ids by default. Subclasses can override for
        MRoPE 3D or other position embedding schemes.
        """
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        inputs_embeds = ir.Value(
            name="inputs_embeds",
            shape=ir.Shape([batch, seq_len, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [inputs_embeds, attention_mask, position_ids]

        kv_inputs, past_key_values = _make_kv_cache_inputs(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )
        graph_inputs.extend(kv_inputs)

        graph, builder = _make_graph(graph_inputs, name="decoder")
        logits, present_key_values = decoder(
            builder.op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        _register_kv_cache_outputs(graph, present_key_values)
        return _make_model(graph)

    def _build_audio_decoder(
        self,
        audio_decoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build audio decoder: codec codes → waveform.

        Takes discrete codec codes and synthesizes a waveform. This is
        the output stage for models using Mimi or similar audio codecs.
        """
        batch = ir.SymbolicDim("batch")
        codec_seq = ir.SymbolicDim("codec_sequence_len")
        # Number of codebooks (e.g. 8 for Mimi, 16 for Qwen3-TTS)
        num_codebooks = 8  # TODO: get from config

        codes = ir.Value(
            name="codes",
            shape=ir.Shape([batch, num_codebooks, codec_seq]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph, builder = _make_graph([codes], name="audio_decoder")
        waveform = audio_decoder(builder.op, codes)

        waveform.name = "waveform"
        graph.outputs.append(waveform)
        return _make_model(graph)
