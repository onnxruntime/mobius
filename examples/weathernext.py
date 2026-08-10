#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""WeatherNext-style ONNX conversion demo.

WeatherNext 2 is a JAX/Haiku + xarray model rather than a HuggingFace
``transformers`` model, so it does not fit the normal ``mobius build`` path.
This example demonstrates the ONNX workflow for that family of models by
defining the same high-level one-step forecast contract used by WeatherNext:

``input weather grid + forcings + stochastic noise → next weather grid``.

The tiny model below uses WeatherNext-like graph data flow:

1. encode lat/lon grid variables at each grid cell,
2. aggregate grid cells onto a mesh,
3. update the mesh latent state,
4. project mesh latents back to the grid, and
5. decode per-grid-cell forecast variables.

It intentionally uses small deterministic weights so the demo can be run
without downloading WeatherNext checkpoints.  The graph I/O and component
boundaries are the pieces to keep when replacing the toy modules with a full
translation of google-deepmind/weathernext's Haiku modules and ``.npz``
checkpoints.

Usage::

    python examples/weathernext.py output/weathernext-mini --validate

    python examples/weathernext.py output/weathernext-mini \
        --lat 8 --lon 16 --mesh-nodes 12 --hidden-size 32
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius import ArchitectureConfig, ModelPackage, build_from_module
from mobius.tasks import ModelTask
from mobius.tasks._base import _make_graph, _make_model


@dataclass(frozen=True)
class WeatherNextDemoShape:
    """Concrete shape for the one-step WeatherNext forecast demo."""

    lat: int
    lon: int
    mesh_nodes: int
    input_variables: int
    forcing_variables: int
    noise_channels: int
    output_variables: int
    hidden_size: int

    @property
    def grid_points(self) -> int:
        return self.lat * self.lon

    @property
    def encoder_channels(self) -> int:
        return self.input_variables + self.forcing_variables + self.noise_channels


def _make_parameter(rng: np.random.Generator, shape: tuple[int, ...]) -> nn.Parameter:
    values = rng.standard_normal(shape).astype(np.float32) * 0.05
    return nn.Parameter(list(shape), data=ir.tensor(values))


class DemoLinear(nn.Module):
    """Small deterministic ``Linear`` layer for a self-contained runnable demo."""

    def __init__(self, rng: np.random.Generator, in_features: int, out_features: int):
        super().__init__()
        self.weight = _make_parameter(rng, (out_features, in_features))
        self.bias = _make_parameter(rng, (out_features,))

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # Project the trailing feature dimension: [..., in_features] -> [..., out_features].
        x = op.MatMul(x, op.Transpose(self.weight, perm=[1, 0]))
        return op.Add(x, self.bias)


class WeatherNextGridMeshBlock(nn.Module):
    """One grid→mesh→grid block mirroring WeatherNext's graph-forecast data flow."""

    def __init__(self, rng: np.random.Generator, shape: WeatherNextDemoShape):
        super().__init__()
        self._shape = shape
        self.grid_encoder = DemoLinear(rng, shape.encoder_channels, shape.hidden_size)
        self.mesh_update_in = DemoLinear(rng, shape.hidden_size, 4 * shape.hidden_size)
        self.mesh_update_out = DemoLinear(rng, 4 * shape.hidden_size, shape.hidden_size)
        self.grid_decoder = DemoLinear(rng, shape.hidden_size, shape.output_variables)

        grid_to_mesh = _projection_matrix(shape.mesh_nodes, shape.grid_points)
        mesh_to_grid = _projection_matrix(shape.grid_points, shape.mesh_nodes)
        self.grid_to_mesh = nn.Parameter(
            [shape.mesh_nodes, shape.grid_points], data=ir.tensor(grid_to_mesh)
        )
        self.mesh_to_grid = nn.Parameter(
            [shape.grid_points, shape.mesh_nodes], data=ir.tensor(mesh_to_grid)
        )

    def forward(
        self,
        op: OpBuilder,
        input_state: ir.Value,
        forcings: ir.Value,
        sample_noise: ir.Value,
    ) -> ir.Value:
        s = self._shape

        # Concatenate per-cell weather variables, known future forcings, and FGN noise:
        # [B, lat, lon, input+forcing+noise].
        grid_features = op.Concat(input_state, forcings, sample_noise, axis=-1)

        # Encode each lat/lon cell independently, then flatten the grid to points:
        # [B, lat, lon, hidden] -> [B, grid_points, hidden].
        grid_latent = op.Tanh(self.grid_encoder(op, grid_features))
        grid_points = op.Reshape(grid_latent, [0, s.grid_points, s.hidden_size])

        # Aggregate grid points onto the mesh with a fixed sparse-style projection:
        # [mesh_points, grid_points] @ [B, grid_points, hidden] -> [B, mesh_points, hidden].
        mesh_latent = op.MatMul(self.grid_to_mesh, grid_points)

        # A compact MLP stands in for WeatherNext's mesh GNN/update blocks.
        mesh_delta = op.Tanh(self.mesh_update_in(op, mesh_latent))
        mesh_latent = op.Add(mesh_latent, self.mesh_update_out(op, mesh_delta))

        # Decode mesh latents back onto the lat/lon grid and add the encoded-grid residual:
        # [grid_points, mesh_points] @ [B, mesh_points, hidden] -> [B, grid_points, hidden].
        grid_delta = op.MatMul(self.mesh_to_grid, mesh_latent)
        grid_points = op.Add(grid_points, grid_delta)

        # Return a one-step forecast grid: [B, lat, lon, output_variables].
        forecast_points = self.grid_decoder(op, grid_points)
        return op.Reshape(forecast_points, [0, s.lat, s.lon, s.output_variables])


class WeatherNextDemoTask(ModelTask):
    """Task wiring for a one-step WeatherNext-style forecast graph."""

    model_roles = {"model": "encoder"}

    def __init__(self, shape: WeatherNextDemoShape):
        self._shape = shape

    def build(self, module: nn.Module, config: ArchitectureConfig) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        s = self._shape

        graph, builder = _make_graph("weathernext_one_step_forecast")
        op = builder.op

        input_state = builder.input(
            "input_state",
            dtype=config.dtype,
            shape=[batch, s.lat, s.lon, s.input_variables],
        )
        forcings = builder.input(
            "forcings",
            dtype=config.dtype,
            shape=[batch, s.lat, s.lon, s.forcing_variables],
        )
        sample_noise = builder.input(
            "sample_noise",
            dtype=config.dtype,
            shape=[batch, s.lat, s.lon, s.noise_channels],
        )

        next_state = module(op, input_state, forcings, sample_noise)
        builder.add_output(next_state, "next_state")

        return ModelPackage({"model": _make_model(graph)}, config=config)


def _projection_matrix(rows: int, cols: int) -> np.ndarray:
    """Create deterministic normalized projections between grid and mesh points."""

    row_positions = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    col_positions = np.linspace(0.0, 1.0, cols, dtype=np.float32)[None, :]
    distance = np.abs(row_positions - col_positions)
    weights = np.maximum(1.0 - 2.0 * distance, 0.0)
    weights += 1e-3
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32)


def build_weathernext_demo_package(shape: WeatherNextDemoShape, dtype: ir.DataType) -> ModelPackage:
    """Build the demo WeatherNext-style ONNX package."""

    rng = np.random.default_rng(20260810)
    module = WeatherNextGridMeshBlock(rng, shape)
    config = ArchitectureConfig(
        vocab_size=0,
        hidden_size=shape.hidden_size,
        intermediate_size=4 * shape.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=shape.hidden_size,
        dtype=dtype,
    )
    return build_from_module(module, config, task=WeatherNextDemoTask(shape))


def _resolve_dtype(name: str) -> ir.DataType:
    if name == "f32":
        return ir.DataType.FLOAT
    if name == "f16":
        return ir.DataType.FLOAT16
    raise ValueError(f"Unsupported dtype: {name}")


def _validate_with_ort(output_dir: str, shape: WeatherNextDemoShape) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed; skipping validation.", file=sys.stderr)
        return

    model_path = os.path.join(output_dir, "model.onnx")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    feeds = {
        "input_state": rng.standard_normal(
            (1, shape.lat, shape.lon, shape.input_variables), dtype=np.float32
        ),
        "forcings": rng.standard_normal(
            (1, shape.lat, shape.lon, shape.forcing_variables), dtype=np.float32
        ),
        "sample_noise": rng.standard_normal(
            (1, shape.lat, shape.lon, shape.noise_channels), dtype=np.float32
        ),
    }
    (next_state,) = sess.run(None, feeds)
    print(f"Validation output next_state shape: {next_state.shape}")
    print(f"Validation output range: [{next_state.min():.6f}, {next_state.max():.6f}]")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a runnable WeatherNext-style grid→mesh→grid ONNX demo.",
    )
    parser.add_argument("output_dir", help="Directory to save model.onnx and model.onnx.data.")
    parser.add_argument("--lat", type=int, default=4, help="Number of latitude points.")
    parser.add_argument("--lon", type=int, default=8, help="Number of longitude points.")
    parser.add_argument("--mesh-nodes", type=int, default=6, help="Number of demo mesh nodes.")
    parser.add_argument("--input-variables", type=int, default=5, help="Input weather channels.")
    parser.add_argument("--forcing-variables", type=int, default=2, help="Known forcing channels.")
    parser.add_argument("--noise-channels", type=int, default=2, help="FGN stochastic noise channels.")
    parser.add_argument("--output-variables", type=int, default=5, help="Forecast weather channels.")
    parser.add_argument("--hidden-size", type=int, default=16, help="Latent feature size.")
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f32", help="ONNX weight dtype.")
    parser.add_argument("--validate", action="store_true", help="Run one ONNX Runtime inference.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    shape = WeatherNextDemoShape(
        lat=args.lat,
        lon=args.lon,
        mesh_nodes=args.mesh_nodes,
        input_variables=args.input_variables,
        forcing_variables=args.forcing_variables,
        noise_channels=args.noise_channels,
        output_variables=args.output_variables,
        hidden_size=args.hidden_size,
    )

    if min(
        shape.lat,
        shape.lon,
        shape.mesh_nodes,
        shape.input_variables,
        shape.forcing_variables,
        shape.noise_channels,
        shape.output_variables,
        shape.hidden_size,
    ) <= 0:
        raise ValueError("All shape arguments must be positive.")

    print("Building WeatherNext-style one-step forecast ONNX graph...")
    print(f"  grid: {shape.lat} x {shape.lon} ({shape.grid_points} cells)")
    print(f"  mesh nodes: {shape.mesh_nodes}")
    print(f"  channels: input={shape.input_variables}, forcing={shape.forcing_variables}, "
          f"noise={shape.noise_channels}, output={shape.output_variables}")

    pkg = build_weathernext_demo_package(shape, dtype=_resolve_dtype(args.dtype))
    model = pkg["model"]
    num_nodes = model.graph.num_nodes
    if callable(num_nodes):
        num_nodes = num_nodes()
    print(f"Built model with {num_nodes} ONNX nodes.")

    pkg.save(args.output_dir, check_weights=True, progress_bar=False)
    print(f"Saved WeatherNext demo package to {args.output_dir!r}.")

    if args.validate:
        _validate_with_ort(args.output_dir, shape)


if __name__ == "__main__":
    main()
