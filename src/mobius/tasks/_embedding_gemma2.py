# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stateless EmbeddingGemma 2 export, including masked mean pooling."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.models.embedding_gemma2 import EmbeddingGemma2Config, EmbeddingGemma2Model
from mobius.tasks._base import (
    ComponentConfig,
    ComponentSpec,
    ModelTask,
    _make_graph,
    _make_model,
)
from mobius.tasks._gemma4 import Gemma4Task


class EmbeddingGemma2Task(ModelTask):
    """Encoder plus optional media towers; no generation inputs or outputs.

    ``image_features`` and ``video_features`` are separate flat row streams
    from the shared vision tower. Audio rows are stripped using the encoder
    mask. The caller supplies exactly one feature row per matching placeholder
    in flattened batch order. Absent media use empty [0, hidden_size] arrays.
    """

    model_roles: ClassVar[dict[str, str]] = {
        "model": "encoder",
        "vision_encoder": "encoder",
        "audio_encoder": "encoder",
    }
    components = ComponentSpec(
        model=ComponentConfig("encoder", "encoder"),
        vision_encoder=ComponentConfig("vision_encoder", "encoder"),
        audio_encoder=ComponentConfig("audio_encoder", "encoder"),
    )

    def build(self, module: nn.Module, config: BaseModelConfig) -> ModelPackage:
        if not isinstance(module, EmbeddingGemma2Model) or not isinstance(
            config, EmbeddingGemma2Config
        ):
            raise TypeError(
                "EmbeddingGemma2Task requires EmbeddingGemma2Model and its config."
            )
        self._validate_components(module)
        graph, builder = _make_graph("embedding_encoder")
        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=["batch", "seq"])
        mask = builder.input("attention_mask", dtype=ir.DataType.INT64, shape=["batch", "seq"])
        positions = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=["batch", "seq"]
        )
        features = {}
        if config.vision is not None:
            for name in ("image_features", "video_features"):
                features[name] = builder.input(
                    name, dtype=config.dtype, shape=[f"num_{name}", config.hidden_size]
                )
        if config.audio is not None:
            features["audio_features"] = builder.input(
                "audio_features",
                dtype=config.dtype,
                shape=["num_audio_features", config.hidden_size],
            )
        tokens, sentence = module.encoder(builder.op, input_ids, mask, positions, **features)
        builder.add_output(tokens, "last_hidden_state")
        builder.add_output(sentence, "sentence_embedding")
        models = {"model": _make_model(graph)}
        media_task = Gemma4Task()
        if module.vision_encoder is not None:
            models["vision_encoder"] = media_task._build_vision(module.vision_encoder, config)
        if module.audio_encoder is not None:
            models["audio_encoder"] = media_task._build_audio(module.audio_encoder, config)
        return ModelPackage(models, config=config)
