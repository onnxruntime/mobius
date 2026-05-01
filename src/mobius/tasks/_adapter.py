# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Adapter task for T2I-Adapter and IP-Adapter models."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class AdapterTask(ModelTask):
    """Build ONNX graph for conditioning adapters (T2I, IP-Adapter)."""

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(
        self,
        module,
        config,
    ) -> ModelPackage:
        graph, builder = _make_graph()
        op = builder.op

        # Determine input shape based on adapter type
        if hasattr(config, "in_channels"):
            # T2I-Adapter: conditioning image input
            condition = builder.input(
                "condition",
                dtype=ir.DataType.FLOAT,
                shape=["batch", config.in_channels, "height", "width"],
            )
        else:
            # IP-Adapter: image embedding input
            condition = builder.input(
                "image_embeds",
                dtype=ir.DataType.FLOAT,
                shape=["batch", config.image_embed_dim],
            )

        outputs = module(op, condition)

        if isinstance(outputs, list):
            for i, out in enumerate(outputs):
                builder.add_output(out, f"feature_{i}")
        else:
            builder.add_output(outputs, "adapter_output")

        return ModelPackage(
            {"model": _make_model(graph, builder.functions.values())}, config=config
        )
