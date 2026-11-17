# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standalone role graphs for architecture-specific GGUF projector sidecars."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._model_package import ModelPackage
from mobius._pipeline_contract import declare_component_presence
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class CoreVLMProjectorTask(ModelTask):
    """Build only the encoder role explicitly owned by one projector string."""

    model_roles: ClassVar[dict[str, str]] = {
        "vision_encoder": "vision",
        "audio_encoder": "encoder",
    }

    def __init__(self, projector_type: str):
        self._projector_type = projector_type

    def build(self, module: nn.Module, config) -> ModelPackage:
        projector_type = self._projector_type
        if projector_type not in {"gemma3na", "gemma4a", "gemma4ua"}:
            raise ValueError(f"{projector_type} is not an audio projector route")
        model = self._build_audio(
            module.audio_encoder,  # type: ignore[attr-defined]
            config,
        )
        return ModelPackage({"audio_encoder": model}, config=config)

    def _build_audio(self, audio: nn.Module, config) -> ir.Model:
        projector_type = self._projector_type
        graph, builder = _make_graph(name="audio_encoder")
        op = builder.op
        time = "time"
        if projector_type == "gemma3na":
            input_size = int(config.audio.input_feat_size)
        elif projector_type == "gemma4ua":
            input_size = int(config.audio.hidden_size)
        else:
            input_size = int(config.audio.input_size)
        input_features = builder.input(
            "input_features",
            dtype=ir.DataType.FLOAT,
            shape=[1, time, input_size],
        )
        input_features_mask = builder.input(
            "input_features_mask",
            dtype=ir.DataType.BOOL,
            shape=[1, time],
        )
        result = audio(
            op,
            op.Cast(input_features, to=config.dtype),
            input_features_mask=input_features_mask,
        )
        downsampled_mask = None
        if projector_type == "gemma3na":
            output = result
        else:
            output, downsampled_mask = result
            output = op.Reshape(output, [-1, config.hidden_size])
            if downsampled_mask is not None:
                output = op.CastLike(
                    op.Compress(
                        op.Cast(output, to=ir.DataType.FLOAT),
                        op.Reshape(downsampled_mask, [-1]),
                        axis=0,
                    ),
                    output,
                )
        builder.add_output(output, "audio_features")
        if projector_type != "gemma3na" and downsampled_mask is not None:
            builder.add_output(downsampled_mask, "audio_features_mask")
        declare_component_presence(graph, "audio")
        return _make_model(graph)
