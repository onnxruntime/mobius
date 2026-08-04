# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ControlNet task: produces residuals for UNet conditioning."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._model_package import ModelPackage

# ControlNetConfig lives in the model file because diffusion models use
# their own config types (from_diffusers) rather than ArchitectureConfig.
# Tasks depend on models, so this import direction is correct.
from mobius.models.controlnet import ControlNetConfig
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class ControlNetTask(ModelTask):
    """Build ONNX graph for ControlNet residual generation."""

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(
        self,
        module,
        config: ControlNetConfig,
    ) -> ModelPackage:
        graph, builder = _make_graph()
        op = builder.op

        sample = builder.input(
            "sample",
            dtype=ir.DataType.FLOAT,
            shape=["batch", config.in_channels, "height", "width"],
        )
        timestep = builder.input("timestep", dtype=ir.DataType.INT64, shape=["batch"])
        encoder_hidden_states = builder.input(
            "encoder_hidden_states",
            dtype=ir.DataType.FLOAT,
            shape=["batch", "sequence_length", config.cross_attention_dim],
        )
        controlnet_cond = builder.input(
            "controlnet_cond",
            dtype=ir.DataType.FLOAT,
            shape=["batch", config.conditioning_channels, "cond_height", "cond_width"],
        )

        down_outputs, mid_output = module(
            op,
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
        )

        # Register outputs
        for i, out in enumerate(down_outputs):
            builder.add_output(out, f"down_block_res_{i}")
        builder.add_output(mid_output, "mid_block_res")

        return ModelPackage({"model": _make_model(graph)}, config=config)
