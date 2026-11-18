# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest


def test_weathernext_example_builds_from_real_npz_inputs(tmp_path):
    repo_root = Path(__file__).parents[1]
    input_path = tmp_path / "sample.npz"
    output_dir = tmp_path / "weathernext"
    np.savez(
        input_path,
        input_state=np.zeros((1, 3, 4, 2), dtype=np.float32),
        forcings=np.zeros((1, 3, 4, 1), dtype=np.float32),
        sample_noise=np.zeros((1, 3, 4, 1), dtype=np.float32),
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples" / "weathernext.py"),
            str(output_dir),
            "--input-data",
            str(input_path),
            "--mesh-nodes",
            "5",
            "--hidden-size",
            "8",
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )

    model = ir.load(output_dir / "model.onnx")
    assert [value.name for value in model.graph.inputs] == [
        "input_state",
        "forcings",
        "sample_noise",
    ]
    assert [value.name for value in model.graph.outputs] == ["next_state"]
    assert model.graph.outputs[0].shape == ir.Shape(["batch", 3, 4, 2])


def test_weathernext_example_runs_f16_real_npz_inputs(tmp_path):
    pytest.importorskip("onnxruntime")

    repo_root = Path(__file__).parents[1]
    input_path = tmp_path / "sample.npz"
    output_dir = tmp_path / "weathernext-f16"
    np.savez(
        input_path,
        input_state=np.zeros((1, 3, 4, 2), dtype=np.float32),
        forcings=np.zeros((1, 3, 4, 1), dtype=np.float32),
        sample_noise=np.zeros((1, 3, 4, 1), dtype=np.float32),
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples" / "weathernext.py"),
            str(output_dir),
            "--input-data",
            str(input_path),
            "--mesh-nodes",
            "5",
            "--hidden-size",
            "8",
            "--dtype",
            "f16",
            "--run",
            "--validate",
        ],
        check=True,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "Inference output next_state shape: (1, 3, 4, 2)" in result.stdout
