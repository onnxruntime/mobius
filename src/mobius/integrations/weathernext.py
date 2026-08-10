# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helpers for building and running WeatherNext-style Mobius packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx_ir as ir
import torch

from mobius._builder import build_from_module
from mobius._configs import WeatherNextConfig
from mobius._model_package import ModelPackage
from mobius.models import WeatherNextModel

_INPUT_NAMES = ("input_state", "forcings", "sample_noise")


def load_npz_weights(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a Mobius-aligned WeatherNext state dict from an ``.npz`` file."""
    with np.load(path) as data:
        return {name: torch.from_numpy(np.asarray(data[name])) for name in data.files}


def create_demo_state_dict(
    config: WeatherNextConfig,
    *,
    seed: int = 20260810,
) -> dict[str, torch.Tensor]:
    """Create deterministic weights for examples and tests.

    These weights are intentionally small and are not a trained WeatherNext
    checkpoint. Pass ``weights=load_npz_weights(...)`` to
    :func:`build_weathernext_package` for converted checkpoint weights.
    """
    rng = np.random.default_rng(seed)

    def parameter(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32) * 0.05)

    state: dict[str, torch.Tensor] = {
        "grid_encoder.weight": parameter((config.hidden_size, config.encoder_channels)),
        "grid_encoder.bias": parameter((config.hidden_size,)),
        "grid_decoder.weight": parameter((config.output_variables, config.hidden_size)),
        "grid_decoder.bias": parameter((config.output_variables,)),
    }
    for layer_idx in range(config.num_hidden_layers):
        state[f"mesh_update_in.{layer_idx}.weight"] = parameter(
            (config.intermediate_size, config.hidden_size)
        )
        state[f"mesh_update_in.{layer_idx}.bias"] = parameter((config.intermediate_size,))
        state[f"mesh_update_out.{layer_idx}.weight"] = parameter(
            (config.hidden_size, config.intermediate_size)
        )
        state[f"mesh_update_out.{layer_idx}.bias"] = parameter((config.hidden_size,))
    return state


def build_weathernext_package(
    config: WeatherNextConfig,
    *,
    weights: dict[str, torch.Tensor] | None = None,
    execution_provider: str = "default",
) -> ModelPackage:
    """Build a WeatherNext one-step forecast package and apply weights."""
    package = build_from_module(
        WeatherNextModel(config),
        config,
        task="weathernext-forecast",
        execution_provider=execution_provider,
    )
    package.apply_weights(weights if weights is not None else create_demo_state_dict(config))
    return package


def load_npz_forecast_inputs(path: str | Path) -> dict[str, np.ndarray]:
    """Load ``input_state``, ``forcings``, and ``sample_noise`` arrays from ``.npz``."""
    with np.load(path) as data:
        missing = [name for name in _INPUT_NAMES if name not in data]
        if missing:
            expected = ", ".join(_INPUT_NAMES)
            missing_names = ", ".join(missing)
            raise ValueError(
                f"WeatherNext input file is missing {missing_names}; expected keys: {expected}"
            )
        feeds = {name: np.asarray(data[name], dtype=np.float32) for name in _INPUT_NAMES}
    return feeds


def load_xarray_forecast_inputs(
    path: str | Path,
    *,
    input_variables: list[str],
    forcing_variables: list[str],
    noise_channels: int,
    batch_index: int = 0,
    sample_noise_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Load WeatherNext inputs from a local xarray NetCDF or Zarr dataset.

    The selected variables must share latitude/longitude dimensions. Optional
    time dimensions are indexed by ``batch_index`` and each variable is stacked
    into the trailing channel dimension expected by the ONNX graph. The default
    noise seed is deterministic so examples are reproducible; pass a different
    seed when sampling stochastic forecast noise for real workflows.
    """
    try:
        import xarray as xr
    except ImportError as e:  # pragma: no cover - exercised only without optional xarray
        raise ImportError("xarray is required to read NetCDF/Zarr WeatherNext inputs") from e

    path = Path(path)
    dataset = (
        xr.open_zarr(path)
        if path.is_dir() or path.suffix == ".zarr"
        else xr.open_dataset(path)
    )
    try:
        input_state = _stack_xarray_variables(
            dataset,
            input_variables,
            batch_index=batch_index,
        )
        forcings = _stack_xarray_variables(
            dataset,
            forcing_variables,
            batch_index=batch_index,
        )
        rng = np.random.default_rng(sample_noise_seed)
        sample_noise = rng.standard_normal(
            (input_state.shape[0], input_state.shape[1], input_state.shape[2], noise_channels)
        ).astype(np.float32)
        return {
            "input_state": input_state,
            "forcings": forcings,
            "sample_noise": sample_noise,
        }
    finally:
        dataset.close()


def infer_config_from_feeds(
    feeds: dict[str, np.ndarray],
    *,
    mesh_nodes: int,
    hidden_size: int,
    output_variables: int | None = None,
    intermediate_size: int | None = None,
    num_hidden_layers: int = 1,
    dtype: ir.DataType = ir.DataType.FLOAT,
) -> WeatherNextConfig:
    """Infer a WeatherNext config from loaded forecast input arrays."""
    input_state = feeds["input_state"]
    forcings = feeds["forcings"]
    sample_noise = feeds["sample_noise"]
    if input_state.ndim != 4 or forcings.ndim != 4 or sample_noise.ndim != 4:
        raise ValueError(
            "WeatherNext inputs must be rank-4 [batch, lat, lon, channels] arrays"
        )
    if (
        input_state.shape[:3] != forcings.shape[:3]
        or input_state.shape[:3] != sample_noise.shape[:3]
    ):
        raise ValueError("WeatherNext inputs must have matching batch/lat/lon dimensions")
    return WeatherNextConfig(
        lat=int(input_state.shape[1]),
        lon=int(input_state.shape[2]),
        mesh_nodes=mesh_nodes,
        input_variables=int(input_state.shape[3]),
        forcing_variables=int(forcings.shape[3]),
        noise_channels=int(sample_noise.shape[3]),
        output_variables=int(
            input_state.shape[3] if output_variables is None else output_variables
        ),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size or 4 * hidden_size,
        num_hidden_layers=num_hidden_layers,
        dtype=dtype,
    )


def _stack_xarray_variables(
    dataset: Any,
    names: list[str],
    *,
    batch_index: int,
) -> np.ndarray:
    if not names:
        raise ValueError("At least one xarray variable name is required")
    arrays = []
    for name in names:
        if name not in dataset:
            raise KeyError(f"Variable {name!r} not found in dataset")
        value = dataset[name]
        for dim in value.dims:
            if dim.lower() in {"time", "batch"}:
                value = value.isel({dim: batch_index})
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(
                f"Variable {name!r} must resolve to a 2-D lat/lon array after "
                f"time/batch selection, got shape {array.shape}"
            )
        arrays.append(array)
    stacked = np.stack(arrays, axis=-1)
    return stacked[None, ...]
