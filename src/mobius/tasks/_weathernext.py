# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""WeatherNext forecast task wiring."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import WeatherNextConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class WeatherNextForecastTask(ModelTask):
    """Build a one-step WeatherNext forecast graph.

    Inputs:
        - input_state: ``[batch, lat, lon, input_variables]``
        - forcings: ``[batch, lat, lon, forcing_variables]``
        - sample_noise: ``[batch, lat, lon, noise_channels]``

    Outputs:
        - next_state: ``[batch, lat, lon, output_variables]``
    """

    input_names: ClassVar[tuple[str, ...]] = ("input_state", "forcings", "sample_noise")
    output_names: ClassVar[tuple[str, ...]] = ("next_state",)
    model_roles: ClassVar[dict[str, str]] = {"model": "forecast"}

    def build(
        self,
        module: nn.Module,
        config: WeatherNextConfig,
    ) -> ModelPackage:
        config.validate()
        batch = ir.SymbolicDim("batch")
        graph, builder = _make_graph(name="weathernext_one_step_forecast")

        input_state = builder.input(
            self.input_names[0],
            dtype=config.dtype,
            shape=[batch, config.lat, config.lon, config.input_variables],
        )
        forcings = builder.input(
            self.input_names[1],
            dtype=config.dtype,
            shape=[batch, config.lat, config.lon, config.forcing_variables],
        )
        sample_noise = builder.input(
            self.input_names[2],
            dtype=config.dtype,
            shape=[batch, config.lat, config.lon, config.noise_channels],
        )

        next_state = module(builder.op, input_state, forcings, sample_noise)
        builder.add_output(next_state, self.output_names[0])
        return ModelPackage({"model": _make_model(graph)}, config=config)
