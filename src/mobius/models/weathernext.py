# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""WeatherNext-style grid→mesh→grid forecast model.

This module provides Mobius-native ONNX graph construction for the one-step
forecast contract used by WeatherNext-family models. It does not trace JAX;
instead, each graph stage is declared directly with ONNX ops.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import WeatherNextConfig
from mobius.components import Linear, get_activation


def projection_matrix(rows: int, cols: int) -> np.ndarray:
    """Create a normalized deterministic projection between grid and mesh points."""
    row_positions = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    col_positions = np.linspace(0.0, 1.0, cols, dtype=np.float32)[None, :]
    distance = np.abs(row_positions - col_positions)
    weights = np.maximum(1.0 - 2.0 * distance, 0.0)
    weights += 1e-3
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32)


class WeatherNextModel(nn.Module):
    """One-step WeatherNext-style grid→mesh→grid forecast module.

    Inputs:
        - input_state: ``[batch, lat, lon, input_variables]``
        - forcings: ``[batch, lat, lon, forcing_variables]``
        - sample_noise: ``[batch, lat, lon, noise_channels]``

    Output:
        - next_state: ``[batch, lat, lon, output_variables]``
    """

    default_task = "weathernext-forecast"
    config_class = WeatherNextConfig
    category = "Weather"

    def __init__(self, config: WeatherNextConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.grid_encoder = Linear(config.encoder_channels, config.hidden_size)
        self.mesh_update_in = nn.ModuleList(
            [
                Linear(config.hidden_size, config.intermediate_size)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.mesh_update_out = nn.ModuleList(
            [
                Linear(config.intermediate_size, config.hidden_size)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.grid_decoder = Linear(config.hidden_size, config.output_variables)
        self._activation = get_activation(config.hidden_act)

        # Fixed topology projections model the grid↔mesh connectivity. A real
        # checkpoint may override these names with learned/sparse projection data.
        self.grid_to_mesh = nn.Parameter(
            [config.mesh_nodes, config.grid_points],
            data=ir.tensor(projection_matrix(config.mesh_nodes, config.grid_points)),
        )
        self.mesh_to_grid = nn.Parameter(
            [config.grid_points, config.mesh_nodes],
            data=ir.tensor(projection_matrix(config.grid_points, config.mesh_nodes)),
        )

    def forward(
        self,
        op: OpBuilder,
        input_state: ir.Value,
        forcings: ir.Value,
        sample_noise: ir.Value,
    ) -> ir.Value:
        config = self.config

        # Concatenate per-cell weather variables, future forcings, and stochastic
        # noise: [B, lat, lon, input+forcing+noise].
        grid_features = op.Concat(input_state, forcings, sample_noise, axis=-1)

        # Encode each lat/lon cell independently, then flatten the spatial grid:
        # [B, lat, lon, hidden] -> [B, grid_points, hidden].
        grid_latent = self._activation(op, self.grid_encoder(op, grid_features))
        batch_dim = op.Shape(grid_latent, start=0, end=1)
        flat_grid_shape = op.Concat(
            batch_dim,
            op.Constant(value_ints=[config.grid_points, config.hidden_size]),
            axis=0,
        )
        grid_points = op.Reshape(grid_latent, flat_grid_shape)

        # Aggregate encoded grid cells onto mesh nodes:
        # [mesh_nodes, grid_points] @ [B, grid_points, hidden]
        # -> [B, mesh_nodes, hidden].
        mesh_latent = op.MatMul(self.grid_to_mesh, grid_points)

        # Apply one or more residual mesh-update MLP blocks.
        for update_in, update_out in zip(
            self.mesh_update_in, self.mesh_update_out, strict=True
        ):
            mesh_delta = self._activation(op, update_in(op, mesh_latent))
            mesh_latent = op.Add(mesh_latent, update_out(op, mesh_delta))

        # Decode mesh latents back to grid cells and retain the encoded-grid
        # residual: [grid_points, mesh_nodes] @ [B, mesh_nodes, hidden]
        # -> [B, grid_points, hidden].
        grid_delta = op.MatMul(self.mesh_to_grid, mesh_latent)
        grid_points = op.Add(grid_points, grid_delta)

        # Return one forecast step on the original grid:
        # [B, grid_points, output_variables] -> [B, lat, lon, output_variables].
        forecast_points = self.grid_decoder(op, grid_points)
        forecast_shape = op.Concat(
            batch_dim,
            op.Constant(value_ints=[config.lat, config.lon, config.output_variables]),
            axis=0,
        )
        return op.Reshape(forecast_points, forecast_shape)

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return WeatherNext weights unchanged after external integration mapping."""
        return state_dict
