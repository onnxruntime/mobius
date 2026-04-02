# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Grounding DINO open-set object detection task.

Unlike the standard :class:`ObjectDetectionTask` (image-only input),
this task provides both ``pixel_values`` and ``input_ids`` so the model
can perform text-guided open-vocabulary object detection.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class GroundingDinoDetectionTask(ModelTask):
    """Text-guided open-set object detection.

    Inputs:
        pixel_values : [batch, channels, height, width]  FLOAT
        input_ids    : [batch, seq_len]                   INT64

    Outputs:
        logits     : [batch, num_queries, text_seq_len]  FLOAT
        pred_boxes : [batch, num_queries, 4]             FLOAT
    """

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")

        image_size = getattr(config, "image_size", 384)
        num_channels = getattr(config, "num_channels", 3)
        text_config = getattr(config, "text_config", {})
        text_seq_len = text_config.get("max_position_embeddings", 32)

        pixel_values = ir.Value(
            name="pixel_values",
            shape=ir.Shape([batch, num_channels, image_size, image_size]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, text_seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph, builder = _make_graph([pixel_values, input_ids])
        op = builder.op

        logits, pred_boxes = module(op, pixel_values=pixel_values, input_ids=input_ids)

        logits.name = "logits"
        pred_boxes.name = "pred_boxes"
        graph.outputs.append(logits)
        graph.outputs.append(pred_boxes)

        return ModelPackage({"model": _make_model(graph)}, config=config)
