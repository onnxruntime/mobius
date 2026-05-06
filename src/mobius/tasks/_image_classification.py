# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Image classification task for ViT-like models."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class ImageClassificationTask(ModelTask):
    """Image classification with pixel_values input.

    Inputs:
        - pixel_values: [batch, channels, height, width] FLOAT

    Outputs:
        - last_hidden_state: [batch, sequence_len, hidden_size] FLOAT
    """

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")

        image_size = getattr(config, "image_size", 224)
        num_channels = getattr(config, "num_channels", 3)

        graph, builder = _make_graph()
        op = builder.op

        pixel_values = builder.input(
            "pixel_values",
            dtype=ir.DataType.FLOAT,
            shape=[batch, num_channels, image_size, image_size],
        )

        last_hidden_state = module(op, pixel_values=pixel_values)

        builder.add_output(last_hidden_state, "last_hidden_state")

        return ModelPackage({"model": _make_model(graph)}, config=config)
