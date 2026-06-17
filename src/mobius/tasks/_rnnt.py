# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""RNN-T (transducer) task — builds three ONNX graphs for FastConformer-RNNT.

Produces three models:
  - ``"encoder"``: mel features ``(B, feat_in, T)`` + ``length`` ``(B,)`` →
                   encoded ``(B, d_model, T')`` + ``encoder_length`` ``(B,)``
  - ``"decoder"``: token ids ``(B, U)`` + LSTM state → prediction ``(B, d_pred, U)``
                   + next LSTM state
  - ``"joint"``:   encoder + prediction → logits ``(B, T', U, vocab+1)``

The transducer greedy-decode loop (alternating decoder/joint steps with the
encoder run once) is performed by the runtime, not inside any single graph.

Contract / current limitations:
  - **Offline full-context only.** The encoder consumes the full mel-feature
    sequence and uses causal convolutions with zero left-padding; it does not
    expose NeMo's cache-aware streaming state, so chunk-by-chunk streaming is
    not supported by this export.
  - **Ragged batches are supported** via the ``length`` input: padded frames
    are masked out of attention and zeroed in the output, and ``encoder_length``
    reports the subsampled valid length per sample.
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ComponentSpec, ModelTask, _make_graph, _make_model


class RNNTTask(ModelTask):
    """Build encoder/decoder/joint ONNX graphs for a FastConformer-RNNT model."""

    name = "fastconformer-rnnt"
    components: ClassVar[ComponentSpec] = ComponentSpec(
        encoder="encoder", decoder="prediction", joint="joint"
    )
    model_roles: ClassVar[dict[str, str]] = {
        "encoder": "encoder",
        "decoder": "encoder",
        "joint": "encoder",
    }

    def build(self, module, config: ArchitectureConfig) -> ModelPackage:
        self._validate_components(module)
        return ModelPackage(
            {
                "encoder": self._build_encoder(module.encoder, config),
                "decoder": self._build_decoder(module.prediction, config),
                "joint": self._build_joint(module.joint, config),
            },
            config=config,
        )

    def _build_encoder(self, encoder, config: ArchitectureConfig) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        time = ir.SymbolicDim("time")
        graph, builder = _make_graph(name="encoder")
        audio_signal = builder.input(
            "audio_signal",
            dtype=config.dtype,
            shape=[batch, config.fastconformer_feat_in, time],
        )
        # Per-sample valid feature-frame counts; drives padding-aware attention
        # masking and the returned subsampled lengths.
        length = builder.input("length", dtype=ir.DataType.INT64, shape=[batch])
        encoder_output, encoder_length = encoder(
            builder.op, audio_signal=audio_signal, length=length
        )
        builder.add_output(encoder_output, "encoder_output")
        builder.add_output(encoder_length, "encoder_length")
        return _make_model(graph)

    def _build_decoder(self, prediction, config: ArchitectureConfig) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        labels = ir.SymbolicDim("labels")
        layers = config.rnnt_pred_rnn_layers
        hidden = config.rnnt_pred_hidden
        graph, builder = _make_graph(name="decoder")
        targets = builder.input("targets", dtype=ir.DataType.INT64, shape=[batch, labels])
        state_h = builder.input("state_h", dtype=config.dtype, shape=[layers, batch, hidden])
        state_c = builder.input("state_c", dtype=config.dtype, shape=[layers, batch, hidden])
        g, new_h, new_c = prediction(
            builder.op, targets=targets, state_h=state_h, state_c=state_c
        )
        builder.add_output(g, "decoder_output")
        builder.add_output(new_h, "state_h_out")
        builder.add_output(new_c, "state_c_out")
        return _make_model(graph)

    def _build_joint(self, joint, config: ArchitectureConfig) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        time = ir.SymbolicDim("time")
        labels = ir.SymbolicDim("labels")
        graph, builder = _make_graph(name="joint")
        encoder_outputs = builder.input(
            "encoder_outputs",
            dtype=config.dtype,
            shape=[batch, config.hidden_size, time],
        )
        decoder_outputs = builder.input(
            "decoder_outputs",
            dtype=config.dtype,
            shape=[batch, config.rnnt_pred_hidden, labels],
        )
        logits = joint(
            builder.op,
            encoder_outputs=encoder_outputs,
            decoder_outputs=decoder_outputs,
        )
        builder.add_output(logits, "logits")
        return _make_model(graph)
