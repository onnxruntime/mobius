# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Network-free checks for the full-checkpoint validation driver."""

from __future__ import annotations

import copy
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


@pytest.mark.parametrize("generate", [False, True])
def test_tower_bridge_selects_teacher_or_single_pass_without_rng(monkeypatch, generate):
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="generate")],
    )
    output = {"valid": np.array([True]), "next_counter": np.zeros(1, np.int64)}
    hidden = np.zeros((1, 1, 8), np.float32)

    def run(actual, feeds):
        assert actual is session
        assert feeds["hidden_states"] is hidden
        assert feeds["generate"].shape == ()
        assert bool(feeds["generate"]) is generate
        assert not bool(feeds["do_sample"])
        np.testing.assert_array_equal(feeds["counter"], [0])
        return output

    monkeypatch.setattr(validate_drama3, "run_session", run)
    assert (
        validate_drama3.run_tower_session(
            session, {"hidden_states": hidden}, generate=generate
        )
        is output
    )


@pytest.mark.parametrize("invalid", ["valid", "counter"])
def test_tower_bridge_rejects_invalid_generation_or_greedy_rng_advance(monkeypatch, invalid):
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="generate")])
    monkeypatch.setattr(
        validate_drama3,
        "run_session",
        lambda *args: {
            "valid": np.array([invalid != "valid"]),
            "next_counter": np.array([int(invalid == "counter")], np.int64),
        },
    )
    with pytest.raises(
        FloatingPointError if invalid == "valid" else AssertionError,
        match=r"rejected|advanced",
    ):
        validate_drama3.run_tower_session(session, {}, generate=True)


def test_tower_bridge_preserves_legacy_teacher_feeds(monkeypatch):
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="hidden_states")])
    feeds = {"hidden_states": np.zeros((1, 3, 8), np.float32)}
    output = {"codebook_logits": np.zeros((1, 3, 2, 16), np.float32)}

    def run(actual, actual_feeds):
        assert actual is session and actual_feeds is feeds
        return output

    monkeypatch.setattr(validate_drama3, "run_session", run)
    assert validate_drama3.run_tower_session(session, feeds) is output


def test_full_export_forwards_explicit_task_and_execution_provider(tmp_path, monkeypatch):
    import mobius

    calls = {}

    class Package:
        def __init__(self):
            self.weight_loading_report = {}

        def save(self, path, *, max_workers):
            calls["save"] = (path, max_workers)

    def build(model, **kwargs):
        calls["build"] = (model, kwargs)
        return Package()

    monkeypatch.setattr(mobius, "build", build)
    monkeypatch.setattr(validate_drama3, "ROOT", tmp_path, raising=False)
    monkeypatch.setattr(validate_drama3, "DTYPE", "f32")
    validate_drama3.export(
        tmp_path / "checkpoint", task="fish-omni-generation", execution_provider="cuda"
    )
    model, options = calls["build"]
    assert model == str(tmp_path / "checkpoint")
    assert options["revision"] == REVISION
    assert options["task"] == "fish-omni-generation"
    assert options["execution_provider"] == "cuda"
    assert options["keep_quantized"] is False
    assert calls["save"] == (str(tmp_path / "onnx"), 1)


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


def test_native_fp8_reference_cannot_be_compared_as_dense():
    golden = json.loads(GOLDEN.read_text())
    golden["expert_execution"] = "dynamic_fp8"
    with pytest.raises(AssertionError, match="expert_execution"):
        validate_reference_golden(golden)


@pytest.mark.parametrize("version", ["2.8.0", "2.8.0+cu126", "2.8.0+cu128"])
def test_reference_accepts_the_pinned_torch_release(version):
    validate_drama3.validate_reference_torch_version(version)


@pytest.mark.parametrize("version", ["2.7.1", "2.8.0a0", "2.10.0+cu128"])
def test_reference_rejects_a_different_torch_release(version):
    with pytest.raises(RuntimeError, match=r"requires torch==2\.8\.0"):
        validate_drama3.validate_reference_torch_version(version)


def test_reference_environment_changes_are_not_silently_compared(tmp_path, monkeypatch):
    committed = json.loads(GOLDEN.read_text())
    committed["reference_environment"] = {"torch": "2.8.0+cu126"}
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(committed))
    monkeypatch.setattr(validate_drama3, "GOLDEN", path)
    actual = copy.deepcopy(committed)
    actual["reference_environment"]["torch"] = "2.10.0+cu128"
    with pytest.raises(AssertionError, match="reference_environment"):
        validate_reference_golden(actual)


@pytest.mark.parametrize(
    ("record_committed", "record_actual"), [(True, True), (True, False), (False, True)]
)
def test_reference_environment_preserves_matching_and_legacy_goldens(
    tmp_path, monkeypatch, record_committed, record_actual
):
    committed = json.loads(GOLDEN.read_text())
    actual = copy.deepcopy(committed)
    environment = {"torch": "2.8.0+cu126", "torch_cuda": "12.6", "triton": "3.4.0"}
    if record_committed:
        committed["reference_environment"] = environment
    if record_actual:
        actual["reference_environment"] = environment
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(committed))
    monkeypatch.setattr(validate_drama3, "GOLDEN", path)
    validate_reference_golden(actual)


@pytest.mark.parametrize("native_fp8", [False, True])
def test_export_preserves_the_requested_expert_execution(tmp_path, monkeypatch, native_fp8):
    import mobius

    calls = {}
    package = SimpleNamespace(
        weight_loading_report={},
        save=lambda *args, **kwargs: None,
    )

    def build(*args, **kwargs):
        calls.update(kwargs)
        return package

    monkeypatch.setattr(mobius, "build", build)
    monkeypatch.setattr(validate_drama3, "ROOT", tmp_path, raising=False)
    monkeypatch.setattr(validate_drama3, "DTYPE", "f32")
    validate_drama3.export(tmp_path, native_fp8=native_fp8)
    assert calls["keep_quantized"] is native_fp8
    assert calls["execution_provider"] == ("cuda" if native_fp8 else "default")
    report = json.loads((tmp_path / "export-result.json").read_text())
    assert report["expert_execution"] == ("dynamic_fp8" if native_fp8 else "dense")


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
