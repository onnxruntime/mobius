# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fun-ASR speech-language 3-model split task.

Builds three separate ONNX models:
1. **audio_encoder**: input_features (batch, seq_len, input_dim) → audio_features (LLM dim)
2. **embedding**: input_ids + audio_features (LLM dim) → inputs_embeds (token scatter)
3. **decoder**: inputs_embeds → logits + KV cache

Unlike the base :class:`SpeechLanguageTask`, the audio encoder accepts
LFR-processed fbank features with shape ``(batch, seq_len, input_dim)``
rather than mel spectrograms ``(batch, n_mels, mel_seq)``. The audio
encoder includes the adaptor, so its output is already in LLM dimension.
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


class FunASRSpeechLanguageTask(ModelTask):
    """3-model split for Fun-ASR speech-language models.

    The module must provide three sub-modules as attributes:

    - ``audio_tower``: audio encoder taking LFR fbank features
    - ``embedding``: embedding model with adaptor + text/audio fusion
    - ``decoder``: text decoder taking ``inputs_embeds`` with KV cache

    Each sub-module is wired into its own ONNX graph.
    """

    model_roles: ClassVar[dict[str, str]] = {
        "audio_encoder": "encoder",
        "embedding": "embedding",
        "decoder": "decoder",
    }
    components = ComponentSpec(
        audio_encoder="audio_tower",
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

        models["audio_encoder"] = self._build_audio_encoder(module.audio_tower, config)

        # The audio encoder includes the adaptor, so output is LLM hidden size.
        llm_hidden = config.hidden_size
        models["embedding"] = build_embedding_from_features(
            module.embedding,
            config,
            feature_name="audio_features",
            feature_dim=llm_hidden,
        )

        # Standard decoder (no MRoPE — Fun-ASR uses standard RoPE)
        models["decoder"] = build_decoder_from_embeds(module.decoder, config, mrope=False)

        return ModelPackage(models, config=config)

    def _build_audio_encoder(
        self,
        audio_encoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build audio encoder: fbank (batch, seq_len, input_dim) → audio features."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("audio_sequence_len")
        input_dim = (config.audio.input_size if config.audio else None) or 560

        graph, builder = _make_graph(name="audio_encoder")

        input_features = builder.input(
            "input_features",
            dtype=config.dtype,
            shape=[batch, seq_len, input_dim],
        )

        audio_features = audio_encoder(builder.op, input_features)

        builder.add_output(audio_features, "audio_features")
        return _make_model(graph)
