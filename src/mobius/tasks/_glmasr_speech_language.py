# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GLM-ASR three-model speech-language export task."""

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
    build_decoder_from_embeds,
    build_embedding_from_features,
)


class GlmAsrSpeechLanguageTask(ModelTask):
    """Build GLM-ASR audio encoder, embedding mixer, and cached decoder graphs."""

    model_roles: ClassVar[dict[str, str]] = {
        "audio_encoder": "encoder",
        "embedding": "embedding",
        # Keep standard Attention for this checkpoint. CUDA GQA fusion changes
        # the FP16 greedy transcript, while the portable graph matches HF exactly.
        "decoder": "decoder_no_gqa",
    }
    components = ComponentSpec(
        audio_encoder="audio_encoder",
        embedding="embedding",
        decoder="decoder",
    )

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        models = {
            "audio_encoder": self._build_audio_encoder(module.audio_encoder, config),
            "embedding": build_embedding_from_features(
                module.embedding,
                config,
                feature_name="audio_features",
                feature_dim=config.hidden_size,
            ),
            "decoder": build_decoder_from_embeds(module.decoder, config, mrope=False),
        }
        return ModelPackage(models, config=config)

    def _build_audio_encoder(
        self,
        audio_encoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        audio = config.audio
        if audio is None:
            raise ValueError("GLM-ASR requires an audio_config")

        audio_batch = ir.SymbolicDim("audio_batch")
        mel_sequence_len = ir.SymbolicDim("mel_sequence_len")
        graph, builder = _make_graph(name="audio_encoder")

        # WhisperFeatureExtractor always emits float32, even for reduced-
        # precision checkpoints. Cast once at the graph boundary.
        input_features = builder.input(
            "input_features",
            dtype=ir.DataType.FLOAT,
            shape=[audio_batch, audio.num_mel_bins or 128, mel_sequence_len],
        )
        input_features_mask = builder.input(
            "input_features_mask",
            dtype=ir.DataType.INT64,
            shape=[audio_batch, mel_sequence_len],
        )
        model_features = builder.op.Cast(input_features, to=config.dtype)
        audio_features, audio_feature_lengths = audio_encoder(
            builder.op,
            model_features,
            input_features_mask,
        )

        builder.add_output(audio_features, "audio_features")
        builder.add_output(audio_feature_lengths, "audio_feature_lengths")
        return _make_model(graph)
