# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Network-free checks for the full-checkpoint validation driver."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest
import validate_drama3
from validate_drama3 import (
    GOLDEN,
    REVISION,
    SOURCE_REVISION,
    empty_cache_ports,
    event,
    validate_reference_golden,
)


def test_logging_accepts_a_metric_name(capsys):
    event("comparison", name="decoder.logits")
    assert json.loads(capsys.readouterr().out) == {
        "event": "comparison",
        "name": "decoder.logits",
    }


def test_empty_cache_uses_the_abi_not_symbolic_dimension_names():
    ports = [
        SimpleNamespace(
            name="past_key_values.3.key",
            shape=["component.decoder.batch", 2, "component.decoder.past_sequence_len", 256],
        ),
        SimpleNamespace(
            name="past_key_values.0.conv_state",
            shape=["component.decoder.batch", 8192, 3],
        ),
        SimpleNamespace(
            name="past_key_values.0.recurrent_state",
            shape=["component.decoder.batch", 32, 128, 128],
        ),
    ]
    cache = empty_cache_ports(ports)
    assert cache["past_key_values.3.key"].shape == (1, 2, 0, 256)
    assert cache["past_key_values.0.conv_state"].shape == (1, 8192, 3)
    assert cache["past_key_values.0.recurrent_state"].shape == (1, 32, 128, 128)


def test_unknown_nonbatch_cache_dimension_is_an_error():
    with pytest.raises(AssertionError):
        empty_cache_ports(
            [
                SimpleNamespace(
                    name="past_key_values.0.conv_state", shape=["batch", "unknown", 3]
                )
            ]
        )


def test_empty_cache_preserves_bfloat16_and_float32_state_types():
    cache = empty_cache_ports(
        [
            SimpleNamespace(
                name="past_key_values.0.key",
                shape=["batch", 2, "past", 8],
                type="tensor(bfloat16)",
            ),
            SimpleNamespace(
                name="past_key_values.0.recurrent_state",
                shape=["batch", 2, 8, 8],
                type="tensor(float)",
            ),
        ]
    )
    assert cache["past_key_values.0.key"].dtype == ml_dtypes.bfloat16
    assert cache["past_key_values.0.recurrent_state"].dtype == np.float32


def test_unknown_cache_dtype_is_an_error():
    with pytest.raises(ValueError, match="Unsupported cache dtype"):
        empty_cache_ports(
            [
                SimpleNamespace(
                    name="past_key_values.0.conv_state",
                    shape=["batch", 8, 3],
                    type="tensor(double)",
                )
            ]
        )


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_runtime_bridge_preserves_types_and_output_names(tmp_path, dtype):
    import onnx_ir as ir
    import onnxruntime as ort
    import onnxruntime_easy as ort_easy
    from onnxscript import GraphBuilder

    element_type = ir.DataType.FLOAT if dtype == "float32" else ir.DataType.BFLOAT16
    array_type = np.float32 if dtype == "float32" else ml_dtypes.bfloat16
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    builder = GraphBuilder(graph)
    x = builder.input("x", element_type, [2])
    result = builder.op.Identity(x)
    result.type, result.shape = x.type, x.shape
    builder.add_output(result, "result")
    path = tmp_path / "bridge.onnx"
    ir.save(ir.Model(graph, ir_version=12), path)
    session = ort_easy.EasySession(
        str(path), sess_options=ort.SessionOptions(), providers=["CPUExecutionProvider"]
    )
    values = np.array([0.75, -1.5], dtype=array_type)
    output = validate_drama3.run_session(session, {"x": values})
    assert set(output) == {"result"}
    assert output["result"].dtype == values.dtype
    np.testing.assert_array_equal(output["result"], values)


def test_runtime_bridge_selects_the_session_cuda_device(monkeypatch):
    import torch

    active = []
    events = []

    @contextmanager
    def device(index):
        events.append(("enter", index))
        active.append(index)
        try:
            yield
        finally:
            assert active.pop() == index
            events.append(("exit", index))

    class Session:
        def get_provider_options(self):
            return {"CUDAExecutionProvider": {"device_id": "3"}}

        def __call__(self, **feeds):
            assert active == [3]
            return [feeds["x"]]

        def get_outputs(self):
            return [SimpleNamespace(name="result")]

    monkeypatch.setattr(torch.cuda, "device", device)
    values = np.array([0.75, -1.5], dtype=np.float32)
    result = validate_drama3.run_session(Session(), {"x": values})
    np.testing.assert_array_equal(result["result"], values)
    assert events == [("enter", 3), ("exit", 3)]


def test_committed_token_golden_is_pinned_and_complete():
    golden = json.loads(GOLDEN.read_text())
    validate_reference_golden(golden)
    assert golden["revision"] == REVISION
    assert golden["source_revision"] == SOURCE_REVISION
    assert golden["max_new_tokens"] == len(golden["generated_frames"]) == 20
    for frame in golden["generated_frames"]:
        assert len(frame) == 11
        assert 248099 <= frame[0] <= 252194
        assert frame[1] == frame[0] - 248099
        assert all(0 <= code < 4096 for code in frame[1:])


@pytest.mark.parametrize("change", ["revision", "length", "frame"])
def test_reference_mismatch_fails_before_model_loading(change):
    golden = json.loads(GOLDEN.read_text())
    if change == "revision":
        golden["revision"] = "not-the-pinned-checkpoint"
    elif change == "length":
        golden["generated_frames"].pop()
    else:
        golden["generated_frames"][0][2] += 1
    with pytest.raises(AssertionError):
        validate_reference_golden(golden)


def test_stale_checkpoint_path_is_not_relabelled_as_the_pinned_revision(tmp_path, monkeypatch):
    wrong = tmp_path / "other-revision"
    wrong.mkdir()
    (tmp_path / "checkpoint-path.txt").write_text(str(wrong))
    monkeypatch.setattr(validate_drama3, "ROOT", tmp_path, raising=False)
    with pytest.raises(ValueError, match="immutable checkpoint snapshot"):
        validate_drama3.checkpoint_path()


def test_pinned_checkpoint_path_accepts_trailing_newline(tmp_path, monkeypatch):
    snapshot = tmp_path / REVISION
    snapshot.mkdir()
    (tmp_path / "checkpoint-path.txt").write_text(f"{snapshot}\n")
    monkeypatch.setattr(validate_drama3, "ROOT", tmp_path, raising=False)
    assert validate_drama3.checkpoint_path() == snapshot
