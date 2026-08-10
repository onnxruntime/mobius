# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_weathernext_example():
    path = Path(__file__).parents[1] / "examples" / "weathernext.py"
    spec = importlib.util.spec_from_file_location("weathernext_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_weathernext_example_infers_config_from_real_npz_inputs(tmp_path):
    example = _load_weathernext_example()
    input_path = tmp_path / "sample.npz"
    np.savez(
        input_path,
        input_state=np.zeros((1, 3, 4, 2), dtype=np.float32),
        forcings=np.zeros((1, 3, 4, 1), dtype=np.float32),
        sample_noise=np.zeros((1, 3, 4, 1), dtype=np.float32),
    )

    args = argparse.Namespace(
        input_data=str(input_path),
        input_variable_names=None,
        forcing_variable_names=None,
        noise_channels=1,
        batch_index=0,
        sample_noise_seed=0,
        mesh_nodes=5,
        hidden_size=8,
        intermediate_size=None,
        num_hidden_layers=1,
        dtype="f32",
    )
    feeds = example._load_real_data(args)
    config = example._config_from_args(args, feeds)

    assert config.lat == 3
    assert config.lon == 4
    assert config.input_variables == 2
    assert config.forcing_variables == 1
    assert config.noise_channels == 1
    assert config.output_variables == 2
