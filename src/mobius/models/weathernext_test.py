# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

from mobius import (
    WeatherNextConfig,
    WeatherNextForecastTask,
    WeatherNextModel,
    build_from_module,
)
from mobius.integrations.weathernext import (
    build_weathernext_package,
    create_demo_state_dict,
    infer_config_from_feeds,
    load_npz_forecast_inputs,
)
from mobius.tasks import TASK_REGISTRY, get_task


def _config() -> WeatherNextConfig:
    return WeatherNextConfig(
        lat=3,
        lon=4,
        mesh_nodes=5,
        input_variables=2,
        forcing_variables=1,
        noise_channels=1,
        output_variables=2,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
    )


class TestWeatherNextConfig:
    def test_derived_sizes(self):
        config = _config()
        assert config.grid_points == 12
        assert config.encoder_channels == 4

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("lat", 0),
            ("lon", True),
            ("mesh_nodes", -1),
            ("input_variables", 0),
            ("forcing_variables", 0),
            ("noise_channels", 0),
            ("output_variables", 0),
            ("hidden_size", 0),
            ("intermediate_size", 0),
            ("num_hidden_layers", 0),
            ("hidden_act", None),
        ],
    )
    def test_invalid_config_raises(self, field, value):
        config = _config()
        setattr(config, field, value)
        with pytest.raises(ValueError):
            config.validate()


class TestWeatherNextForecastTask:
    def test_registered(self):
        assert TASK_REGISTRY["weathernext-forecast"] is WeatherNextForecastTask
        assert isinstance(get_task("weathernext-forecast"), WeatherNextForecastTask)

    def test_graph_contract(self, tmp_path):
        config = _config()
        package = build_weathernext_package(config)
        package.save(tmp_path, check_weights=True, progress_bar=False)

        model = ir.load(tmp_path / "model.onnx")
        assert [value.name for value in model.graph.inputs] == list(
            WeatherNextForecastTask.input_names
        )
        assert [value.name for value in model.graph.outputs] == list(
            WeatherNextForecastTask.output_names
        )
        assert model.graph.outputs[0].shape == ir.Shape(["batch", 3, 4, 2])
        assert model.graph.name == "weathernext_one_step_forecast"


def test_weathernext_model_requires_weights_for_standard_build(tmp_path):
    config = _config()
    package = build_from_module(WeatherNextModel(config), config, task="weathernext-forecast")
    package.apply_weights(create_demo_state_dict(config))
    package.save(tmp_path, check_weights=True, progress_bar=False)


def test_npz_inputs_infer_config(tmp_path):
    input_path = tmp_path / "sample.npz"
    np.savez(
        input_path,
        input_state=np.zeros((2, 5, 6, 3), dtype=np.float32),
        forcings=np.zeros((2, 5, 6, 2), dtype=np.float32),
        sample_noise=np.zeros((2, 5, 6, 1), dtype=np.float32),
    )

    feeds = load_npz_forecast_inputs(input_path)
    config = infer_config_from_feeds(
        feeds,
        mesh_nodes=7,
        hidden_size=8,
        output_variables=4,
    )

    assert config.lat == 5
    assert config.lon == 6
    assert config.mesh_nodes == 7
    assert config.input_variables == 3
    assert config.forcing_variables == 2
    assert config.noise_channels == 1
    assert config.output_variables == 4


def test_npz_inputs_report_missing_keys(tmp_path):
    input_path = tmp_path / "missing.npz"
    np.savez(
        input_path,
        input_state=np.zeros((1, 2, 3, 1), dtype=np.float32),
        forcings=np.zeros((1, 2, 3, 1), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="sample_noise"):
        load_npz_forecast_inputs(input_path)
