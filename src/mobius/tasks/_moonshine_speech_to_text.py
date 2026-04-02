# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Speech-to-text task for Moonshine encoder-decoder models.

Moonshine differs from Whisper in that the encoder input is a raw audio
waveform ``(batch, audio_length)`` instead of pre-computed mel features.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig, MoonshineConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_kv_cache_inputs,
    _make_model,
    _register_kv_cache_outputs,
)


class MoonshineSpeechToTextTask(ModelTask):
    """Encoder-decoder speech-to-text task for Moonshine.

    This task builds **two** separate ONNX models via :meth:`build`:

    - **encoder**: ``input_values`` (raw waveform) → ``encoder_hidden_states``
    - **decoder**: ``decoder_input_ids``, ``encoder_hidden_states``,
      ``position_ids``, ``past_key_values``
      → ``logits``, ``present_key_values``

    The module must expose ``model.encoder`` and ``model.decoder`` sub-modules
    (matching the ``MoonshineForConditionalGeneration`` layout).
    """

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        if not isinstance(config, MoonshineConfig):
            raise TypeError(
                f"MoonshineSpeechToTextTask requires MoonshineConfig, "
                f"got {type(config).__name__}"
            )

        encoder_model = self._build_encoder(module.model.encoder, config)
        decoder_model = self._build_decoder(module.model.decoder, config)
        return ModelPackage(
            {"encoder": encoder_model, "decoder": decoder_model},
            config=config,
        )

    def _build_encoder(
        self,
        encoder: nn.Module,
        config: MoonshineConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        audio_length = ir.SymbolicDim("audio_length")

        # Raw waveform input: (batch, audio_length)
        input_values = ir.Value(
            name="input_values",
            shape=ir.Shape([batch, audio_length]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )

        graph, graph_builder = _make_graph([input_values], name="encoder")
        op = graph_builder.op

        encoder_hidden_states = encoder(op, input_values=input_values)

        encoder_hidden_states.name = "encoder_hidden_states"
        graph.outputs.append(encoder_hidden_states)

        return _make_model(graph)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: MoonshineConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")
        encoder_seq_len = ir.SymbolicDim("encoder_sequence_len")

        decoder_input_ids = ir.Value(
            name="decoder_input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        encoder_hidden_states = ir.Value(
            name="encoder_hidden_states",
            shape=ir.Shape([batch, encoder_seq_len, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [decoder_input_ids, encoder_hidden_states, position_ids]

        kv_inputs, past_key_values = _make_kv_cache_inputs(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )
        graph_inputs.extend(kv_inputs)

        graph, graph_builder = _make_graph(graph_inputs, name="decoder")
        op = graph_builder.op

        logits, present_key_values = decoder(
            op,
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        _register_kv_cache_outputs(graph, present_key_values)

        return _make_model(graph)
