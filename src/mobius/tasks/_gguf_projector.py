# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standalone graph tasks for GGUF multimodal-projector sidecars."""

from __future__ import annotations

from typing import ClassVar, Protocol, cast

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius._pipeline_contract import declare_component_presence
from mobius.tasks._base import ComponentSpec, ModelTask, _make_graph, _make_model


class _VisionProjectorModule(Protocol):
    vision_encoder: nn.Module


class _AudioProjectorModule(Protocol):
    audio_encoder: nn.Module


class GGUFAudioProjectorModel(nn.Module):
    """Container exposing one exact sidecar audio encoder/projector."""

    def __init__(self, audio_encoder: nn.Module) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder

    def forward(self, op, **kwargs):
        del op, kwargs
        raise NotImplementedError("GGUFAudioProjectorTask builds the audio component")


class GGUFAudioProjectorTask(ModelTask):
    """Build one processor-native rank-2 audio feature graph."""

    model_roles: ClassVar[dict[str, str]] = {"audio_encoder": "encoder"}
    components = ComponentSpec(audio_encoder="audio_encoder")

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        if not isinstance(module, GGUFAudioProjectorModel):
            raise TypeError("GGUFAudioProjectorTask requires GGUFAudioProjectorModel")
        audio_encoder = module.audio_encoder
        input_schema = getattr(audio_encoder, "input_schema", None)
        if not isinstance(input_schema, tuple) or not input_schema:
            raise TypeError("GGUF audio encoder must declare a non-empty input_schema")

        graph, builder = _make_graph(name="audio_encoder")
        inputs = {
            name: builder.input(name, dtype=dtype, shape=list(shape))
            for name, dtype, shape in input_schema
        }
        outputs = audio_encoder(builder.op, **inputs)
        output_names = getattr(audio_encoder, "output_names", ("audio_features",))
        if not isinstance(output_names, tuple) or not output_names:
            raise TypeError("GGUF audio encoder output_names must be a non-empty tuple")
        output_values = outputs if isinstance(outputs, tuple) else (outputs,)
        if len(output_values) != len(output_names):
            raise ValueError("GGUF audio encoder output count does not match output_names")
        for value, name in zip(output_values, output_names):
            builder.add_output(value, name)
        declare_component_presence(graph, "audio")
        return ModelPackage({"audio_encoder": _make_model(graph)}, config=config)


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
        if not isinstance(module, GGUFVisionProjectorModel):
            raise TypeError("GGUFVisionProjectorTask requires GGUFVisionProjectorModel")
        vision_encoder = module.vision_encoder
        input_schema = getattr(vision_encoder, "input_schema", None)
        if not isinstance(input_schema, tuple) or not input_schema:
            raise TypeError("GGUF vision encoder must declare a non-empty input_schema")

        graph, builder = _make_graph(name="vision_encoder")
        inputs = {
            name: builder.input(name, dtype=dtype, shape=list(shape))
            for name, dtype, shape in input_schema
        }
        image_features = vision_encoder(builder.op, **inputs)
        if bool(getattr(vision_encoder, "squeeze_batch_dim", False)):
            image_features = builder.op.Squeeze(image_features, [0])
        builder.add_output(image_features, "image_features")
        declare_component_presence(graph, "image")
        return ModelPackage({"vision_encoder": _make_model(graph)}, config=config)


class GGUFVisionAudioProjectorModel(nn.Module):
    """Container exposing co-resident vision and audio sidecar encoders."""

    def __init__(self, vision_encoder: nn.Module, audio_encoder: nn.Module) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder

    def forward(self, op, **kwargs):
        del op, kwargs
        raise NotImplementedError(
            "GGUFVisionAudioProjectorTask builds the vision and audio components"
        )


class GGUFVisionAudioProjectorTask(ModelTask):
    """Build both executable roles from one mixed-modality sidecar."""

    model_roles: ClassVar[dict[str, str]] = {
        "vision_encoder": "vision",
        "audio_encoder": "encoder",
    }
    components = ComponentSpec(
        vision_encoder="vision_encoder",
        audio_encoder="audio_encoder",
    )

    @staticmethod
    def _build_component(
        component: nn.Module,
        *,
        graph_name: str,
        output_name: str,
    ) -> ir.Model:
        input_schema = getattr(component, "input_schema", None)
        if not isinstance(input_schema, tuple) or not input_schema:
            raise TypeError(f"GGUF {graph_name} must declare a non-empty input_schema")
        graph, builder = _make_graph(name=graph_name)
        inputs = {
            name: builder.input(name, dtype=dtype, shape=list(shape))
            for name, dtype, shape in input_schema
        }
        output = component(builder.op, **inputs)
        builder.add_output(output, output_name)
        return _make_model(graph)

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        return ModelPackage(
            {
                "vision_encoder": self._build_component(
                    cast(_VisionProjectorModule, module).vision_encoder,
                    graph_name="vision_encoder",
                    output_name="image_features",
                ),
                "audio_encoder": self._build_component(
                    cast(_AudioProjectorModule, module).audio_encoder,
                    graph_name="audio_encoder",
                    output_name="audio_features",
                ),
            },
            config=config,
        )


class GGUFSpeakerProjectorModel(nn.Module):
    """Container exposing one speaker-conditioning encoder sidecar."""

    def __init__(
        self,
        speaker_encoder: nn.Module,
        *,
        output_name: str = "speaker_embedding",
    ) -> None:
        super().__init__()
        self.speaker_encoder = speaker_encoder
        self.output_name = output_name

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
        if not isinstance(module, GGUFSpeakerProjectorModel):
            raise TypeError("GGUFSpeakerProjectorTask requires GGUFSpeakerProjectorModel")
        speaker_encoder = module.speaker_encoder
        input_schema = getattr(speaker_encoder, "input_schema", None)
        if not isinstance(input_schema, tuple) or not input_schema:
            raise TypeError("GGUF speaker encoder must declare a non-empty input_schema")

        graph, builder = _make_graph(name="speaker_encoder")
        inputs = {
            name: builder.input(name, dtype=dtype, shape=list(shape))
            for name, dtype, shape in input_schema
        }
        speaker_embedding = speaker_encoder(builder.op, **inputs)
        builder.add_output(speaker_embedding, module.output_name)
        return ModelPackage({"speaker_encoder": _make_model(graph)}, config=config)
