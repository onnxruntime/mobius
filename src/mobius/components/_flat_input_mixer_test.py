# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Flat media features preserve batch-major order and exact token accounting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnxruntime.capi.onnxruntime_pybind11_state import Fail

from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import _numpy_to_ort_value, _ort_value_to_numpy
from mobius.components import FlatInputMixer


def _session(tmp_path, dtype, provider):
    if provider not in ort.get_available_providers():
        pytest.skip(f"{provider} is required")
    builder, op, graph = create_test_builder()
    text = create_test_input(builder, "text", [2, "T", 4], dtype)
    features = create_test_input(builder, "features", ["N", 4], dtype)
    ids = create_test_input(builder, "ids", [2, "T"], ir.DataType.INT64)
    result = FlatInputMixer(18)(op, text, features, ids)
    result.name = "output"
    result.dtype = dtype
    result.shape = ir.Shape([2, "T", 4])
    graph.outputs.append(result)
    path = tmp_path / "mixer.onnx"
    ir.save(ir.Model(graph, ir_version=11), path)
    options = ort.SessionOptions()
    if provider == "CUDAExecutionProvider":
        options.enable_profiling = True
        options.profile_file_prefix = str(tmp_path / "placement")
    return ort.InferenceSession(str(path), sess_options=options, providers=[provider])


@pytest.mark.parametrize(
    ("dtype", "provider"),
    [
        (ir.DataType.FLOAT, "CPUExecutionProvider"),
        (ir.DataType.FLOAT, "CUDAExecutionProvider"),
        (ir.DataType.FLOAT16, "CUDAExecutionProvider"),
        (ir.DataType.BFLOAT16, "CUDAExecutionProvider"),
    ],
)
@pytest.mark.parametrize("media_case", ["features", "none", "cached_placeholder"])
def test_flat_input_mixer_batch_order_and_empty_decode(tmp_path, dtype, provider, media_case):
    session = _session(tmp_path, dtype, provider)
    ids = np.array([[2, 18, 18, 3], [18, 4, 5, 18]], dtype=np.int64)
    if media_case == "none":
        ids[:] = 2
    text = np.arange(32, dtype=np.float32).reshape(2, 4, 4).astype(dtype.numpy())
    count = int((ids == 18).sum()) if media_case == "features" else 0
    features = (np.arange(count * 4).reshape(count, 4) + 100).astype(dtype.numpy())
    expected = text.copy()
    if media_case == "features":
        expected[ids == 18] = features
    outputs = session.run_with_ort_values(
        None,
        {
            key: _numpy_to_ort_value(value)
            for key, value in {
                "text": text,
                "features": features,
                "ids": ids,
            }.items()
        },
    )
    np.testing.assert_array_equal(_ort_value_to_numpy(outputs[0]), expected)
    if provider == "CUDAExecutionProvider":
        events = json.loads(Path(session.end_profiling()).read_text())
        kernels = [event["args"] for event in events if event.get("cat") == "Node"]
        assert any(event.get("provider") == provider for event in kernels)
        for event in kernels:
            if event.get("provider") != "CPUExecutionProvider":
                continue
            # Token-count validation may run as host shape arithmetic. Floating
            # embedding/feature data must never fall back to CPU computation.
            if event.get("op_name") == "Shape":
                continue
            assert not any(
                key in {"float", "float16", "bfloat16", "double"}
                for item in event.get("input_type_shape", [])
                for key in item
            ), event


@pytest.mark.parametrize("feature_count", [1, 3])
def test_flat_input_mixer_rejects_feature_count_mismatch(tmp_path, feature_count):
    session = _session(tmp_path, ir.DataType.FLOAT, "CPUExecutionProvider")
    with pytest.raises(Fail, match="reshape"):
        session.run(
            None,
            {
                "text": np.ones((2, 1, 4), dtype=np.float32),
                "features": np.ones((feature_count, 4), dtype=np.float32),
                "ids": np.full((2, 1), 18, dtype=np.int64),
            },
        )
