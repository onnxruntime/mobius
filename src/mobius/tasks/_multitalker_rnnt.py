# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Multi-talker streaming RNN-T task.

Builds the three ONNX graphs used by cache-aware streaming RNN-T ASR models
(NVIDIA NeMo ``EncDecMultiTalkerRNNTBPEModel`` / Parakeet multi-talker),
matching the ``nemotron-speech-streaming`` deployment contract:

1. **encoder** — FastConformer with speaker-kernel injection and streaming
   channel/time caches.
2. **decoder** — RNN-T LSTM prediction network.
3. **joint** — RNN-T joint network producing per-(time, target) logits.
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ComponentSpec,
    ModelTask,
    _make_graph,
    _make_model,
)


class MultiTalkerRNNTTask(ModelTask):
    """Build the encoder/decoder/joint graphs for streaming multi-talker RNN-T.

    The module must provide three sub-modules as attributes:

    - ``encoder``: streaming FastConformer with speaker kernels.
    - ``decoder``: RNN-T LSTM prediction network.
    - ``joint``: RNN-T joint network.
    """

    model_roles: ClassVar[dict[str, str]] = {
        "encoder": "encoder",
        "decoder": "decoder",
        "joint": "joint",
    }
    components = ComponentSpec(
        encoder="encoder",
        decoder="decoder",
        joint="joint",
    )

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        models: dict[str, ir.Model] = {
            "encoder": self._build_encoder(module.encoder, config),
            "decoder": self._build_decoder(module.decoder, config),
            "joint": self._build_joint(module.joint, config),
        }
        return ModelPackage(models, config=config)

    def _build_encoder(self, encoder: nn.Module, config: BaseModelConfig) -> ir.Model:
        """encoder: audio + streaming caches -> encoder outputs + updated caches."""
        graph, builder = _make_graph(name="encoder")

        d_model = config.d_model
        num_layers = config.num_layers
        cache_channel = config.last_channel_cache_size
        cache_time = config.conv_cache_size

        # audio_signal: [batch, mel_time, feat]
        audio_signal = builder.input(
            "audio_signal",
            dtype=config.dtype,
            shape=["batch", "mel_time", config.feat_in],
        )
        length = builder.input("length", dtype=ir.DataType.INT64, shape=["batch"])
        cache_last_channel = builder.input(
            "cache_last_channel",
            dtype=config.dtype,
            shape=["batch", num_layers, cache_channel, d_model],
        )
        cache_last_time = builder.input(
            "cache_last_time",
            dtype=config.dtype,
            shape=["batch", num_layers, d_model, cache_time],
        )
        cache_last_channel_len = builder.input(
            "cache_last_channel_len", dtype=ir.DataType.INT64, shape=["batch"]
        )
        # Multi-talker speaker/background masks over the encoder time axis.
        spk_mask = builder.input("spk_mask", dtype=config.dtype, shape=["batch", "enc_time"])
        bg_mask = builder.input("bg_mask", dtype=config.dtype, shape=["batch", "enc_time"])

        outputs, enc_len, cc_next, ct_next, ccl_next = encoder(
            builder.op,
            audio_signal=audio_signal,
            length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            spk_mask=spk_mask,
            bg_mask=bg_mask,
        )

        builder.add_output(outputs, "outputs")
        builder.add_output(enc_len, "encoded_lengths")
        builder.add_output(cc_next, "cache_last_channel_next")
        builder.add_output(ct_next, "cache_last_time_next")
        builder.add_output(ccl_next, "cache_last_channel_len_next")
        return _make_model(graph)

    def _build_decoder(self, decoder: nn.Module, config: BaseModelConfig) -> ir.Model:
        """decoder: targets + LSTM state -> decoder output [B, H, U] + new state."""
        graph, builder = _make_graph(name="decoder")
        op = builder.op

        hidden = config.pred_hidden
        num_layers = config.pred_rnn_layers

        targets = builder.input(
            "targets", dtype=ir.DataType.INT64, shape=["batch", "target_len"]
        )
        h_in = builder.input("h_in", dtype=config.dtype, shape=[num_layers, "batch", hidden])
        c_in = builder.input("c_in", dtype=config.dtype, shape=[num_layers, "batch", hidden])

        g, h_out, c_out = decoder(op, targets=targets, h_in=h_in, c_in=c_in)
        # NeMo exports decoder_output as [B, H, U]; internal g is [B, U, H].
        g = op.Transpose(g, perm=[0, 2, 1])

        builder.add_output(g, "decoder_output")
        builder.add_output(h_out, "h_out")
        builder.add_output(c_out, "c_out")
        return _make_model(graph)

    def _build_joint(self, joint: nn.Module, config: BaseModelConfig) -> ir.Model:
        """joint: encoder + decoder outputs -> logits [B, T, U, V]."""
        graph, builder = _make_graph(name="joint")

        encoder_output = builder.input(
            "encoder_output",
            dtype=config.dtype,
            shape=["batch", "time", config.d_model],
        )
        # decoder_output fed as [B, U, H] (host transposes NeMo's [B, H, U]).
        decoder_output = builder.input(
            "decoder_output",
            dtype=config.dtype,
            shape=["batch", "target_len", config.pred_hidden],
        )

        logits = joint(
            builder.op,
            encoder_outputs=encoder_output,
            decoder_outputs=decoder_output,
        )

        builder.add_output(logits, "joint_output")
        return _make_model(graph)
