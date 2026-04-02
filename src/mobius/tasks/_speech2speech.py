# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Speech-to-speech task for SeamlessM4T v2 and similar full-pipeline models.

Produces a ModelPackage with five ONNX models:

- **speech_encoder**: ``input_features`` → ``last_hidden_state``
- **decoder**: ``input_ids``, ``encoder_hidden_states``, ``attention_mask``,
  ``past_key_values`` → ``logits``, ``present_key_values``
- **t2u**: ``input_ids`` → ``logits``
- **vocoder_dur**: ``unit_ids`` → ``log_dur``
- **vocoder_hifigan**: ``unit_ids``, ``speaker_id``, ``lang_id`` → ``waveform``
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import SeamlessM4Tv2Config
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_kv_cache_inputs,
    _make_model,
)


class Speech2SpeechTask(ModelTask):
    """Full speech-to-speech pipeline task for SeamlessM4T v2.

    Builds five separate ONNX models from a
    :class:`~mobius.models.seamless_m4t_v2.SeamlessM4Tv2SpeechToSpeechModel`
    wrapper:

    1. ``speech_encoder`` — Conformer encoder: fbank → encoder hidden states
    2. ``decoder``        — Autoregressive text decoder with KV cache
    3. ``t2u``            — Text-to-unit converter: text tokens → unit logits
    4. ``vocoder_dur``    — Duration predictor: unit tokens → log-duration
    5. ``vocoder_hifigan``— HiFi-GAN synthesiser: expanded units + conditioning → waveform
    """

    def build(
        self,
        module: nn.Module,
        config: SeamlessM4Tv2Config,
    ) -> ModelPackage:
        speech_enc = self._build_speech_encoder(module, config)
        decoder = self._build_decoder(module, config)
        t2u = self._build_t2u(module, config)
        vocoder_dur = self._build_vocoder_dur(module, config)
        vocoder_hifigan = self._build_vocoder_hifigan(module, config)
        return ModelPackage(
            {
                "speech_encoder": speech_enc,
                "decoder": decoder,
                "t2u": t2u,
                "vocoder_dur": vocoder_dur,
                "vocoder_hifigan": vocoder_hifigan,
            },
            config=config,
        )

    # ------------------------------------------------------------------
    # Sub-graph builders
    # ------------------------------------------------------------------

    def _build_speech_encoder(
        self,
        module: nn.Module,
        config: SeamlessM4Tv2Config,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        audio_len = ir.SymbolicDim("audio_seq_len")

        input_features = ir.Value(
            name="input_features",
            shape=ir.Shape([batch, audio_len, config.feature_projection_input_dim]),
            type=ir.TensorType(config.dtype),
        )
        graph, builder = _make_graph([input_features], name="speech_encoder")
        op = builder.op

        hidden = module.speech_encoder(op, input_features)
        hidden.name = "last_hidden_state"
        graph.outputs.append(hidden)

        return _make_model(graph)

    def _build_decoder(
        self,
        module: nn.Module,
        config: SeamlessM4Tv2Config,
    ) -> ir.Model:
        """Autoregressive text decoder with self + cross KV caches."""
        batch = ir.SymbolicDim("batch")
        dec_seq_len = ir.SymbolicDim("decoder_sequence_len")
        enc_seq_len = ir.SymbolicDim("encoder_sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, dec_seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        encoder_hidden_states = ir.Value(
            name="encoder_hidden_states",
            shape=ir.Shape([batch, enc_seq_len, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + dec_seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        graph_inputs = [input_ids, encoder_hidden_states, attention_mask]

        num_heads = config.num_attention_heads
        head_dim = config.head_dim
        num_decoder_layers = config.num_decoder_layers

        # Self-attention KV cache
        self_kv_inputs, past_self_kvs = _make_kv_cache_inputs(
            num_decoder_layers,
            num_heads,
            head_dim,
            config.dtype,
            batch,
            past_seq_len,
            prefix="past_key_values",
        )
        for v in self_kv_inputs:
            idx = v.name.split(".")[1]
            kv_type = v.name.rsplit(".", 1)[-1]
            v.name = f"past_key_values.{idx}.self.{kv_type}"
        graph_inputs.extend(self_kv_inputs)

        # Cross-attention KV cache
        cross_kv_inputs, cross_past_kvs = _make_kv_cache_inputs(
            num_decoder_layers,
            num_heads,
            head_dim,
            config.dtype,
            batch,
            enc_seq_len,
            prefix="past_key_values",
        )
        for v in cross_kv_inputs:
            idx = v.name.split(".")[1]
            kv_type = v.name.rsplit(".", 1)[-1]
            v.name = f"past_key_values.{idx}.cross.{kv_type}"
        graph_inputs.extend(cross_kv_inputs)

        graph, builder = _make_graph(graph_inputs, name="decoder")
        op = builder.op

        logits, present_self_kvs, present_cross_kvs = module.decoder(
            op,
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_self_kvs,
            cross_past_key_values=cross_past_kvs,
        )
        logits.name = "logits"
        graph.outputs.append(logits)

        for i, (k, v) in enumerate(present_self_kvs):
            k.name = f"present.{i}.self.key"
            v.name = f"present.{i}.self.value"
            graph.outputs.extend([k, v])
        for i, (k, v) in enumerate(present_cross_kvs):
            k.name = f"present.{i}.cross.key"
            v.name = f"present.{i}.cross.value"
            graph.outputs.extend([k, v])

        return _make_model(graph)

    def _build_t2u(
        self,
        module: nn.Module,
        config: SeamlessM4Tv2Config,
    ) -> ir.Model:
        """Text-to-unit converter: text token IDs → unit logits."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        graph, builder = _make_graph([input_ids], name="t2u")
        op = builder.op

        logits = module.t2u(op, input_ids)
        logits.name = "logits"
        graph.outputs.append(logits)

        return _make_model(graph)

    def _build_vocoder_dur(
        self,
        module: nn.Module,
        config: SeamlessM4Tv2Config,
    ) -> ir.Model:
        """Duration predictor: unit_ids (B, T) → log_dur (B, T)."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("unit_seq_len")

        unit_ids = ir.Value(
            name="unit_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        graph, builder = _make_graph([unit_ids], name="vocoder_dur")
        op = builder.op

        log_dur = module.vocoder.dur_head(op, unit_ids)
        log_dur.name = "log_dur"
        graph.outputs.append(log_dur)

        return _make_model(graph)

    def _build_vocoder_hifigan(
        self,
        module: nn.Module,
        config: SeamlessM4Tv2Config,
    ) -> ir.Model:
        """HiFi-GAN synthesiser: expanded unit_ids + conditioning → waveform."""
        batch = ir.SymbolicDim("batch")
        expanded_len = ir.SymbolicDim("expanded_unit_len")

        unit_ids = ir.Value(
            name="unit_ids",
            shape=ir.Shape([batch, expanded_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        speaker_id = ir.Value(
            name="speaker_id",
            shape=ir.Shape([batch, 1]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        lang_id = ir.Value(
            name="lang_id",
            shape=ir.Shape([batch, 1]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        graph, builder = _make_graph([unit_ids, speaker_id, lang_id], name="vocoder_hifigan")
        op = builder.op

        waveform = module.vocoder.hifigan_head(op, unit_ids, speaker_id, lang_id)
        waveform.name = "waveform"
        graph.outputs.append(waveform)

        return _make_model(graph)
