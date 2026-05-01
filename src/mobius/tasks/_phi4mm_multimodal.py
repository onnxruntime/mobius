# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Phi4MM four-model split task (vision, speech, embedding, decoder).

Builds four separate ONNX models:

1. **vision**: pixel_values + image_sizes → image_features (SigLIP + projection)
2. **speech**: audio_features + audio_sizes + audio_projection_mode →
   audio_features (Conformer + mode-selected projection)
3. **embedding**: input_ids + image_features + audio_features → inputs_embeds
4. **decoder**: inputs_embeds + KV cache → logits (LoRA text decoder)

This is the Phi-4-multimodal-specific task. Unlike the generic
``MultiModalTask`` (single unified model), this splits each component
into its own ONNX graph for independent optimization and runtime
flexibility.
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
)


class Phi4MMMultiModalTask(ModelTask):
    """Four-model split for Phi4MM: vision, speech, embedding, decoder.

    The module must provide four sub-modules as attributes:

    - ``vision_encoder``: SigLIP encoder + projection MLP + HD params
    - ``speech_encoder``: Conformer encoder + projection MLP(s)
    - ``embedding``: token embedding + InputMixer fusion
    - ``decoder``: LoRA text decoder + RMSNorm + lm_head

    Each sub-module is wired into its own ONNX graph.
    """

    model_roles: ClassVar[dict[str, str]] = {
        "decoder": "decoder",
        "vision_encoder": "encoder",
        "audio_encoder": "encoder",
        "embedding": "embedding",
    }
    components = ComponentSpec(
        vision_encoder="vision_encoder",
        audio_encoder="speech_encoder",
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
        models["vision_encoder"] = self._build_vision(module.vision_encoder, config)
        models["audio_encoder"] = self._build_speech(module.speech_encoder, config)
        models["embedding"] = self._build_embedding(module.embedding, config)
        models["decoder"] = build_decoder_from_embeds(module.decoder, config)
        return ModelPackage(models, config=config)

    def _build_vision(
        self,
        vision: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build vision encoder: pixel_values + image_sizes → image_features."""
        batch = ir.SymbolicDim("batch")
        num_images = ir.SymbolicDim("num_images")
        image_size = (config.vision.image_size if config.vision else None) or 448

        graph, builder = _make_graph(name="vision_encoder")

        pixel_values = builder.input(
            "pixel_values",
            dtype=config.dtype,
            shape=[batch, 3, image_size, image_size],
        )
        image_sizes = builder.input(
            "image_sizes",
            dtype=ir.DataType.INT64,
            shape=[num_images, 2],
        )

        image_features = vision(builder.op, pixel_values, image_sizes=image_sizes)

        builder.add_output(image_features, "image_features")
        return _make_model(graph)

    def _build_speech(
        self,
        speech: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build speech encoder: audio_features + metadata → audio_features.

        The ``audio_projection_mode`` input selects the projection branch:
        0 = speech-only mode, 1 = combined vision+audio mode.
        """
        batch = ir.SymbolicDim("batch")
        audio_seq_len = ir.SymbolicDim("audio_seq_len")
        num_audio_clips = ir.SymbolicDim("num_audio_clips")
        input_size = (config.audio.input_size if config.audio else None) or 80

        graph, builder = _make_graph(name="audio_encoder")

        audio_embeds = builder.input(
            "audio_embeds",
            dtype=config.dtype,
            shape=[batch, audio_seq_len, input_size],
        )
        audio_sizes = builder.input(
            "audio_sizes",
            dtype=ir.DataType.INT64,
            shape=[num_audio_clips],
        )
        audio_projection_mode = builder.input(
            "audio_projection_mode",
            dtype=ir.DataType.INT64,
            shape=[],
        )

        speech_out = speech(
            builder.op,
            audio_embeds,
            audio_sizes=audio_sizes,
            audio_projection_mode=audio_projection_mode,
        )

        builder.add_output(speech_out, "audio_features")
        return _make_model(graph)

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build embedding: input_ids + features → inputs_embeds."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        num_image_tokens = ir.SymbolicDim("num_image_tokens")
        num_speech_tokens = ir.SymbolicDim("num_speech_tokens")

        graph, builder = _make_graph(name="embedding")

        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )
        image_features = builder.input(
            "image_features",
            dtype=config.dtype,
            shape=[num_image_tokens, config.hidden_size],
        )
        audio_features = builder.input(
            "audio_features",
            dtype=config.dtype,
            shape=[num_speech_tokens, config.hidden_size],
        )

        inputs_embeds = embedding(
            builder.op,
            input_ids=input_ids,
            image_features=image_features,
            audio_features=audio_features,
        )

        builder.add_output(inputs_embeds, "inputs_embeds")
        return _make_model(graph)
