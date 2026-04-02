# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""VibeVoice ASR 3-model split task.

Produces three ONNX models:
1. **audio_encoder**: raw waveform (batch, 1, num_samples) → audio_features
2. **embedding**: input_ids + audio_features → inputs_embeds
3. **decoder**: inputs_embeds + position_ids + KV cache → logits + KV cache

The key differences from the generic SpeechLanguageTask:
- Audio input is raw waveform (B, 1, T), NOT a mel spectrogram
- Decoder uses standard 2D position_ids, NOT MRoPE 3D
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


class VibeVoiceAsrTask(ModelTask):
    """3-model split for VibeVoice ASR.

    The module must provide three sub-modules as attributes:

    - ``audio_tower``: audio encoder taking ``input_values`` (raw waveform)
    - ``embedding``: embedding model fusing text + audio features
    - ``decoder``: text decoder taking ``inputs_embeds`` with KV cache

    Each sub-module is wired into its own ONNX graph.
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        models: dict[str, ir.Model] = {}

        models["audio_encoder"] = self._build_audio_encoder(module.audio_tower, config)
        models["embedding"] = self._build_embedding(module.embedding, config)
        models["decoder"] = self._build_decoder(module.decoder, config)

        return ModelPackage(models, config=config)

    def _build_audio_encoder(
        self,
        audio_tower: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build audio encoder: raw waveform (B, 1, T) → audio features (N, hidden)."""
        batch = ir.SymbolicDim("batch")
        num_samples = ir.SymbolicDim("num_samples")

        # Raw waveform input: (batch, 1, num_samples)
        input_values = ir.Value(
            name="input_values",
            shape=ir.Shape([batch, 1, num_samples]),
            type=ir.TensorType(config.dtype),
        )

        graph, builder = _make_graph([input_values], name="audio_encoder")
        audio_features = audio_tower(builder.op, input_values)

        audio_features.name = "audio_features"
        graph.outputs.append(audio_features)
        return _make_model(graph)

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build embedding: input_ids + audio_features → inputs_embeds."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        num_audio_tokens = ir.SymbolicDim("num_audio_tokens")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        audio_features = ir.Value(
            name="audio_features",
            shape=ir.Shape([num_audio_tokens, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )

        graph, builder = _make_graph([input_ids, audio_features], name="embedding")
        inputs_embeds = embedding(builder.op, input_ids, audio_features)

        inputs_embeds.name = "inputs_embeds"
        graph.outputs.append(inputs_embeds)
        return _make_model(graph)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build decoder with standard 2D position_ids (not MRoPE 3D)."""
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
        # Standard 2D position_ids (batch, seq_len) — not MRoPE's (3, batch, seq_len)
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
