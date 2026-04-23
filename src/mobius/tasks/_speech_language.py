# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Speech-language 3-model split task for ASR / forced alignment.

Builds three separate ONNX models:
1. **audio**: input_features (mel spectrogram) → audio_features
2. **embedding**: input_ids + audio_features → inputs_embeds
3. **decoder**: inputs_embeds → logits + KV cache (MRoPE 3D position_ids)

Used by Qwen3-ASR and Qwen3-ForcedAligner.
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
    build_decoder_from_embeds,
    build_embedding_from_features,
)


class SpeechLanguageTask(ModelTask):
    """3-model split for speech-language models (ASR / forced alignment).

    The module must provide three sub-modules as attributes:

    - ``audio_tower``: audio encoder taking ``input_features`` (mel)
    - ``embedding``: embedding model fusing text + audio features
    - ``decoder``: text decoder taking ``inputs_embeds`` with KV cache

    Each sub-module is wired into its own ONNX graph.
    """

    model_roles: ClassVar[dict[str, str]] = {
        "audio": "encoder",
        "embedding": "embedding",
        "decoder": "decoder",
    }
    components = ComponentSpec(
        audio="audio_tower",
        embedding="embedding",
        decoder="decoder",
    )

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        models: dict[str, ir.Model] = {}
        models["audio"] = self._build_audio_encoder(module.audio_tower, config)
        output_dim = (config.audio.output_dim if config.audio else None) or config.hidden_size
        models["embedding"] = build_embedding_from_features(
            module.embedding,
            config,
            feature_name="audio_features",
            feature_dim=output_dim,
        )
        # MRoPE 3D position_ids (temporal, height, width)
        models["decoder"] = build_decoder_from_embeds(module.decoder, config, mrope=True)
        return ModelPackage(models, config=config)

    def _build_audio_encoder(
        self,
        audio_encoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build audio encoder: mel (batch, n_mels, time) → audio features."""
        batch = ir.SymbolicDim("batch")
        mel_seq = ir.SymbolicDim("mel_sequence_len")
        n_mels = (config.audio.num_mel_bins if config.audio else None) or 128

        input_features = ir.Value(
            name="input_features",
            shape=ir.Shape([batch, n_mels, mel_seq]),
            type=ir.TensorType(config.dtype),
        )

        graph, builder = _make_graph([input_features], name="audio")
        audio_features = audio_encoder(builder.op, input_features)

        audio_features.name = "audio_features"
        graph.outputs.append(audio_features)
        return _make_model(graph)
