# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standalone task for GGUF audio encoder/projector sidecars."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ComponentSpec, ModelTask, _make_graph, _make_model


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
        audio_encoder = module.audio_encoder
        input_schema = getattr(audio_encoder, "input_schema", None)
        if not isinstance(input_schema, tuple) or not input_schema:
            raise TypeError("GGUF audio encoder must declare a non-empty input_schema")

        graph, builder = _make_graph(name="audio_encoder")
        inputs = {
            name: builder.input(name, dtype=dtype, shape=list(shape))
            for name, dtype, shape in input_schema
        }
        audio_features = audio_encoder(builder.op, **inputs)
        builder.add_output(audio_features, "audio_features")
        return ModelPackage({"audio_encoder": _make_model(graph)}, config=config)
