# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Speech-language 3-model split task for ASR / forced alignment.

Builds three separate ONNX models:
1. **audio_encoder**: input_features (mel spectrogram) → audio_features
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
    _cast_encoder_input,
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

        graph, builder = _make_graph(name="audio_encoder")
        op = builder.op

        input_features = builder.input(
            "input_features",
            dtype=ir.DataType.FLOAT,
            shape=[batch, n_mels, mel_seq],
        )
        input_features = _cast_encoder_input(op, input_features, config)
        # Mask is required: without it the encoder consumes padded mel
        # frames as if they were real audio, which causes the
        # downstream LLM to emit degenerate loops on any input padded
        # to 30s by the standard HF processor.
        feature_attention_mask = builder.input(
            "feature_attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, mel_seq],
        )

        audio_features, audio_feature_lengths = audio_encoder(
            op, input_features, feature_attention_mask
        )

        # Crop padding-derived tokens so the output contains only valid
        # audio features.  The encoder emits a fixed-length output equal
        # to ceil(padded_mel_length / 8), but only the first
        # ``audio_feature_lengths`` tokens per batch item are real.
        # Cropping inside the graph avoids callers needing to handle
        # the extra output and manual slicing.
        #
        # NOTE: This Slice uses a single ``ends`` value for the token
        # axis, so it is only correct for batch_size=1 (the standard
        # GenAI inference mode).  For batch>1 each item may have a
        # different valid length; supporting that would require a
        # per-batch crop (e.g. loop or ScatterND).
        # audio_features: [batch, max_tokens, hidden_size]
        # audio_feature_lengths: [batch] → reshape to [1] for Slice ends
        audio_features = builder.op.Slice(
            audio_features,
            [0],  # starts
            builder.op.Reshape(audio_feature_lengths, [-1]),  # ends
            [1],  # axes (token dim)
        )

        builder.add_output(audio_features, "audio_features")
        return _make_model(graph)
