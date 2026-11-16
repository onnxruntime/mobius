# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native-grid TIPS image and text feature extraction."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._model_package import ModelPackage
from mobius.models.tips import TipsConfig, TipsModel
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class TipsFeatureExtractionTask(ModelTask):
    """Export independent image/text batches, with 1=padding text masks."""

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(self, module: TipsModel, config: TipsConfig) -> ModelPackage:
        config.validate()
        graph, builder = _make_graph()
        pixels = builder.input(
            "pixel_values",
            dtype=ir.DataType.FLOAT,
            shape=["image_batch", 3, config.image_size, config.image_size],
        )
        ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=["text_batch", config.max_position_embeddings],
        )
        padding = builder.input(
            "padding_mask",
            dtype=ir.DataType.INT64,
            shape=["text_batch", config.max_position_embeddings],
        )
        outputs = module(builder.op, pixels, ids, padding)
        for name, output in zip(
            ("cls_token", "register_tokens", "patch_tokens", "text_embeds"), outputs
        ):
            builder.add_output(output, name)
        graph.metadata_props["tips.temperature"] = str(config.temperature)
        graph.metadata_props["tips.pixel_range"] = "[0,1]"
        graph.metadata_props["tips.padding_mask"] = "1=padding,0=valid"
        return ModelPackage({"model": _make_model(graph)}, config=config)
