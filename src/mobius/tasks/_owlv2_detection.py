# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""OWLv2 open-vocabulary object detection task.

Unlike the standard :class:`ObjectDetectionTask` (image-only input),
this task provides both ``pixel_values`` and ``input_ids`` so the model
can match text queries against image patches.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class Owlv2ObjectDetectionTask(ModelTask):
    """Open-vocabulary object detection with image + text query inputs.

    Inputs:
        pixel_values : [batch, channels, height, width]  FLOAT
        input_ids    : [batch, num_queries, seq_len]      INT64

    Outputs:
        logits             : [batch, num_patches, num_queries]  FLOAT
        pred_boxes         : [batch, num_patches, 4]            FLOAT
        objectness_logits  : [batch, num_patches]               FLOAT
    """

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        num_queries = ir.SymbolicDim("num_queries")

        image_size = getattr(config, "image_size", 960)
        num_channels = getattr(config, "num_channels", 3)
        seq_len = getattr(config, "text_max_position_embeddings", 16)

        pixel_values = ir.Value(
            name="pixel_values",
            shape=ir.Shape([batch, num_channels, image_size, image_size]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, num_queries, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph, builder = _make_graph([pixel_values, input_ids])
        op = builder.op

        logits, pred_boxes, objectness_logits = module(
            op, pixel_values=pixel_values, input_ids=input_ids
        )

        logits.name = "logits"
        pred_boxes.name = "pred_boxes"
        objectness_logits.name = "objectness_logits"
        graph.outputs.append(logits)
        graph.outputs.append(pred_boxes)
        graph.outputs.append(objectness_logits)

        return ModelPackage({"model": _make_model(graph)}, config=config)
