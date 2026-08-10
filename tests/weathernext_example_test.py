# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import onnx_ir as ir


def _load_weathernext_example():
    path = Path(__file__).parents[1] / "examples" / "weathernext.py"
    spec = importlib.util.spec_from_file_location("weathernext_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_weathernext_demo_exports_one_step_forecast_model(tmp_path):
    example = _load_weathernext_example()
    shape = example.WeatherNextDemoShape(
        lat=3,
        lon=4,
        mesh_nodes=5,
        input_variables=2,
        forcing_variables=1,
        noise_channels=1,
        output_variables=2,
        hidden_size=8,
    )

    pkg = example.build_weathernext_demo_package(shape, dtype=ir.DataType.FLOAT)
    pkg.save(tmp_path, check_weights=True, progress_bar=False)

    model = ir.load(tmp_path / "model.onnx")
    assert [value.name for value in model.graph.inputs] == [
        "input_state",
        "forcings",
        "sample_noise",
    ]
    assert [value.name for value in model.graph.outputs] == ["next_state"]
    assert model.graph.outputs[0].shape == ir.Shape(["batch", 3, 4, 2])
