# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standalone graph tasks for GGUF multimodal-projector sidecars."""

from __future__ import annotations

from typing import ClassVar

from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius._pipeline_contract import declare_component_presence
from mobius.tasks._base import ComponentSpec, ModelTask, _make_graph, _make_model


class GGUFVisionProjectorModel(nn.Module):
    """Container exposing one exact sidecar vision encoder/projector."""

    def __init__(self, vision_encoder: nn.Module) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder

    def forward(self, op, **kwargs):
        del op, kwargs
        raise NotImplementedError("GGUFVisionProjectorTask builds the vision component")


class GGUFVisionProjectorTask(ModelTask):
    """Build one processor-native rank-2 vision feature graph."""

    model_roles: ClassVar[dict[str, str]] = {"vision_encoder": "vision"}
    components = ComponentSpec(vision_encoder="vision_encoder")

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        vision_encoder = module.vision_encoder  # type: ignore[attr-defined]
        input_schema = getattr(vision_encoder, "input_schema", None)
        if not isinstance(input_schema, tuple) or not input_schema:
            raise TypeError("GGUF vision encoder must declare a non-empty input_schema")

        graph, builder = _make_graph(name="vision_encoder")
        inputs = {
            name: builder.input(name, dtype=dtype, shape=list(shape))
            for name, dtype, shape in input_schema
        }
        image_features = vision_encoder(builder.op, **inputs)
        builder.add_output(image_features, "image_features")
        declare_component_presence(graph, "image")
        return ModelPackage({"vision_encoder": _make_model(graph)}, config=config)


class GGUFSpeakerProjectorModel(nn.Module):
    """Container exposing one speaker-conditioning encoder sidecar."""

    def __init__(self, speaker_encoder: nn.Module) -> None:
        super().__init__()
        self.speaker_encoder = speaker_encoder

    def forward(self, op, **kwargs):
        del op, kwargs
        raise NotImplementedError("GGUFSpeakerProjectorTask builds the speaker component")


class GGUFSpeakerProjectorTask(ModelTask):
    """Build a speaker embedding graph without claiming audio-token output."""

    model_roles: ClassVar[dict[str, str]] = {"speaker_encoder": "encoder"}
    components = ComponentSpec(speaker_encoder="speaker_encoder")

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        speaker_encoder = module.speaker_encoder  # type: ignore[attr-defined]
        input_schema = getattr(speaker_encoder, "input_schema", None)
        if not isinstance(input_schema, tuple) or not input_schema:
            raise TypeError("GGUF speaker encoder must declare a non-empty input_schema")

        graph, builder = _make_graph(name="speaker_encoder")
        inputs = {
            name: builder.input(name, dtype=dtype, shape=list(shape))
            for name, dtype, shape in input_schema
        }
        speaker_embedding = speaker_encoder(builder.op, **inputs)
        builder.add_output(speaker_embedding, "speaker_embedding")
        return ModelPackage({"speaker_encoder": _make_model(graph)}, config=config)
