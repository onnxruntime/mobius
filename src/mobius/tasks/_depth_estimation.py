# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Depth estimation task for dense-prediction models (DPT, ZoeDepth, DepthAnything)."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class DepthEstimationTask(ModelTask):
    """Monocular depth estimation with pixel_values input.

    Inputs:
        - pixel_values: [batch, channels, height, width] FLOAT

    Outputs:
        - predicted_depth: [batch, height, width] FLOAT
    """

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")

        image_size = getattr(config, "image_size", 224)
        num_channels = getattr(config, "num_channels", 3)

        pixel_values = ir.Value(
            name="pixel_values",
            shape=ir.Shape([batch, num_channels, image_size, image_size]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )

        graph, builder = _make_graph([pixel_values])
        op = builder.op

        predicted_depth = module(op, pixel_values=pixel_values)

        predicted_depth.name = "predicted_depth"
        graph.outputs.append(predicted_depth)

        return ModelPackage({"model": _make_model(graph)}, config=config)
