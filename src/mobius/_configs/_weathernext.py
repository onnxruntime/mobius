# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration for WeatherNext-style one-step forecast graphs."""

from __future__ import annotations

import dataclasses

from mobius._configs._base import BaseModelConfig


@dataclasses.dataclass
class WeatherNextConfig(BaseModelConfig):
    """Configuration for grid→mesh→grid WeatherNext forecast modules.

    Shapes exclude the leading batch dimension. The task exposes a one-step
    forecast contract:
    ``input_state + forcings + sample_noise -> next_state``.
    """

    lat: int = 4
    lon: int = 8
    mesh_nodes: int = 6
    input_variables: int = 5
    forcing_variables: int = 2
    noise_channels: int = 2
    output_variables: int = 5
    hidden_size: int = 16
    intermediate_size: int = 64
    num_hidden_layers: int = 1
    hidden_act: str | None = "silu"

    @property
    def grid_points(self) -> int:
        """Number of lat/lon grid cells."""
        return self.lat * self.lon

    @property
    def encoder_channels(self) -> int:
        """Per-grid-cell input channels after concatenating all inputs."""
        return self.input_variables + self.forcing_variables + self.noise_channels

    def validate(self) -> None:
        """Validate dimensions required by the WeatherNext forecast task."""
        for name in (
            "lat",
            "lon",
            "mesh_nodes",
            "input_variables",
            "forcing_variables",
            "noise_channels",
            "output_variables",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_act is None:
            raise ValueError("hidden_act must be set")
