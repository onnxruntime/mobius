#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build and run a WeatherNext-style one-step forecast ONNX model.

The formal Mobius support lives in ``mobius.models.WeatherNextModel``,
``mobius.tasks.WeatherNextForecastTask``, and ``mobius.integrations.weathernext``.
This script demonstrates that path and can run the exported model on either:

* an ``.npz`` file with ``input_state``, ``forcings``, and ``sample_noise`` arrays, or
* a local xarray NetCDF/Zarr weather dataset plus selected variable names.

If no data or checkpoint is provided, the script uses deterministic demo weights
and synthetic inputs so the ONNX workflow remains runnable in a fresh checkout.

Usage::

    PYTHONPATH=src python examples/weathernext.py output/weathernext-mini --validate

    PYTHONPATH=src python examples/weathernext.py output/weathernext-era5 \
        --input-data era5_sample.npz --weights converted_weathernext_weights.npz --run

    PYTHONPATH=src python examples/weathernext.py output/weathernext-xarray \
        --input-data weatherbench_sample.zarr \
        --input-variable-names 2m_temperature mean_sea_level_pressure \
        --forcing-variable-names toa_incident_solar_radiation --run
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnx_ir as ir

from mobius import WeatherNextConfig
from mobius.integrations.weathernext import (
    build_weathernext_package,
    infer_config_from_feeds,
    load_npz_forecast_inputs,
    load_npz_weights,
    load_xarray_forecast_inputs,
)


def _resolve_dtype(name: str) -> ir.DataType:
    if name == "f32":
        return ir.DataType.FLOAT
    if name == "f16":
        return ir.DataType.FLOAT16
    raise ValueError(f"Unsupported dtype: {name}")


def _load_real_data(args: argparse.Namespace) -> dict[str, np.ndarray] | None:
    if args.input_data is None:
        return None
    if args.input_data.endswith(".npz"):
        return load_npz_forecast_inputs(args.input_data)
    if not args.input_variable_names or not args.forcing_variable_names:
        raise ValueError(
            "--input-variable-names and --forcing-variable-names are required for xarray data"
        )
    return load_xarray_forecast_inputs(
        args.input_data,
        input_variables=args.input_variable_names,
        forcing_variables=args.forcing_variable_names,
        noise_channels=args.noise_channels,
        batch_index=args.batch_index,
        sample_noise_seed=args.sample_noise_seed,
    )


def _synthetic_feeds(config: WeatherNextConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "input_state": rng.standard_normal(
            (1, config.lat, config.lon, config.input_variables)
        ).astype(np.float32),
        "forcings": rng.standard_normal(
            (1, config.lat, config.lon, config.forcing_variables)
        ).astype(np.float32),
        "sample_noise": rng.standard_normal(
            (1, config.lat, config.lon, config.noise_channels)
        ).astype(np.float32),
    }


def _numpy_dtype(dtype: ir.DataType) -> type[np.float32 | np.float16]:
    if dtype == ir.DataType.FLOAT:
        return np.float32
    if dtype == ir.DataType.FLOAT16:
        return np.float16
    raise ValueError(f"Unsupported WeatherNext feed dtype: {dtype}")


def _cast_feeds_to_dtype(
    feeds: dict[str, np.ndarray], dtype: ir.DataType
) -> dict[str, np.ndarray]:
    feed_dtype = _numpy_dtype(dtype)
    return {name: np.asarray(value, dtype=feed_dtype) for name, value in feeds.items()}


def _config_from_args(
    args: argparse.Namespace, feeds: dict[str, np.ndarray] | None
) -> WeatherNextConfig:
    dtype = _resolve_dtype(args.dtype)
    if feeds is not None:
        return infer_config_from_feeds(
            feeds,
            mesh_nodes=args.mesh_nodes,
            hidden_size=args.hidden_size,
            output_variables=args.output_variables,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.num_hidden_layers,
            dtype=dtype,
        )
    return WeatherNextConfig(
        lat=args.lat,
        lon=args.lon,
        mesh_nodes=args.mesh_nodes,
        input_variables=args.input_variables,
        forcing_variables=args.forcing_variables,
        noise_channels=args.noise_channels,
        output_variables=args.input_variables
        if args.output_variables is None
        else args.output_variables,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size or 4 * args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        dtype=dtype,
    )


def _run_with_ort(output_dir: str, feeds: dict[str, np.ndarray], dtype: ir.DataType) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed; skipping inference.", file=sys.stderr)
        return

    model_path = os.path.join(output_dir, "model.onnx")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    (next_state,) = sess.run(["next_state"], _cast_feeds_to_dtype(feeds, dtype))
    print(f"Inference output next_state shape: {next_state.shape}")
    print(f"Inference output range: [{next_state.min():.6f}, {next_state.max():.6f}]")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Mobius WeatherNext one-step forecast ONNX graph.",
    )
    parser.add_argument("output_dir", help="Directory to save model.onnx and model.onnx.data.")
    parser.add_argument("--lat", type=int, default=4, help="Synthetic-data latitude points.")
    parser.add_argument("--lon", type=int, default=8, help="Synthetic-data longitude points.")
    parser.add_argument("--mesh-nodes", type=int, default=6, help="Forecast mesh nodes.")
    parser.add_argument(
        "--input-variables", type=int, default=5, help="Synthetic input channels."
    )
    parser.add_argument(
        "--forcing-variables", type=int, default=2, help="Synthetic forcing channels."
    )
    parser.add_argument(
        "--noise-channels", type=int, default=2, help="Stochastic noise channels."
    )
    parser.add_argument(
        "--output-variables",
        type=int,
        help="Output channels. Defaults to input-variable count for synthetic and real data.",
    )
    parser.add_argument("--hidden-size", type=int, default=16, help="Latent feature size.")
    parser.add_argument("--intermediate-size", type=int, help="Mesh MLP intermediate size.")
    parser.add_argument("--num-hidden-layers", type=int, default=1, help="Mesh update blocks.")
    parser.add_argument(
        "--dtype", choices=["f32", "f16"], default="f32", help="ONNX weight dtype."
    )
    parser.add_argument("--weights", help="Optional Mobius-aligned WeatherNext weights .npz.")
    parser.add_argument(
        "--input-data",
        help="Optional .npz, NetCDF, or Zarr weather sample used for real-data inference.",
    )
    parser.add_argument(
        "--input-variable-names",
        nargs="+",
        help="xarray variables stacked into input_state channels.",
    )
    parser.add_argument(
        "--forcing-variable-names",
        nargs="+",
        help="xarray variables stacked into forcings channels.",
    )
    parser.add_argument(
        "--batch-index", type=int, default=0, help="Time/batch index for xarray."
    )
    parser.add_argument(
        "--sample-noise-seed", type=int, default=0, help="Generated noise seed."
    )
    parser.add_argument("--run", action="store_true", help="Run one ONNX Runtime inference.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Alias for --run, kept for the original self-contained demo workflow.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    feeds = _load_real_data(args)
    config = _config_from_args(args, feeds)
    config.validate()

    print("Building WeatherNext one-step forecast ONNX graph...")
    print(f"  grid: {config.lat} x {config.lon} ({config.grid_points} cells)")
    print(f"  mesh nodes: {config.mesh_nodes}")
    print(
        "  channels: "
        f"input={config.input_variables}, forcing={config.forcing_variables}, "
        f"noise={config.noise_channels}, output={config.output_variables}"
    )

    weights = load_npz_weights(args.weights) if args.weights else None
    package = build_weathernext_package(config, weights=weights)
    model = package["model"]
    print(f"Built model with {model.graph.num_nodes()} ONNX nodes.")

    package.save(args.output_dir, check_weights=True, progress_bar=False)
    print(f"Saved WeatherNext package to {args.output_dir!r}.")

    if args.run or args.validate:
        _run_with_ort(
            args.output_dir,
            feeds if feeds is not None else _synthetic_feeds(config),
            config.dtype,
        )


if __name__ == "__main__":
    main()
