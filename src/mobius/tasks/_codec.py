# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Codec tokenizer 2-model task for Qwen3-TTS-Tokenizer-12Hz.

Builds two ONNX models:
1. **decoder**: codes (B, 16, T) → waveform (B, 1, T*1920)
2. **encoder**: waveform (B, 1, samples) → codes (B, 16, T)
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ComponentSpec,
    ModelTask,
    _make_graph,
    _make_model,
)


class CodecTask(ModelTask):
    """2-model split for Qwen3-TTS codec tokenizer.

    The module must provide two sub-modules:
    - ``decoder``: codes → waveform
    - ``encoder``: waveform → codes

    Each is wired into its own ONNX graph with no KV cache.
    """

    model_roles: ClassVar[dict[str, str]] = {"encoder": "encoder", "decoder": "decoder"}
    components: ClassVar[ComponentSpec] = ComponentSpec(encoder="encoder", decoder="decoder")

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        models: dict[str, ir.Model] = {}
        models["decoder"] = self._build_decoder(module.decoder, config)
        models["encoder"] = self._build_encoder(module.encoder, config)
        return ModelPackage(models, config=config)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build decoder: codes → waveform.

        Inputs:
            codes: (B, num_quantizers, T) int64
        Outputs:
            waveform: (B, 1, T * upsample_factor) float32
        """
        batch = ir.SymbolicDim("batch")
        num_q = ir.SymbolicDim("num_quantizers")
        seq_len = ir.SymbolicDim("sequence_len")

        graph, builder = _make_graph()
        codes = builder.input("codes", dtype=ir.DataType.INT64, shape=[batch, num_q, seq_len])

        waveform = decoder(builder.op, codes)

        builder.add_output(waveform, "waveform")
        return _make_model(graph)

    def _build_encoder(
        self,
        encoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build encoder: waveform → codes.

        Inputs:
            waveform: (B, audio_channels, audio_samples) float32
        Outputs:
            codes: (B, num_quantizers, T) int64
        """
        batch = ir.SymbolicDim("batch")
        audio_len = ir.SymbolicDim("audio_length")
        # The first conv is sized from codec_encoder.audio_channels, so the
        # graph input must declare the same channel count.
        audio_channels = config.codec_encoder.audio_channels if config.codec_encoder else 1

        graph, builder = _make_graph(name="encoder")
        waveform = builder.input(
            "waveform",
            dtype=ir.DataType.FLOAT,
            shape=[batch, audio_channels, audio_len],
        )

        codes = encoder(builder.op, waveform)

        builder.add_output(codes, "codes")
        return _make_model(graph)
