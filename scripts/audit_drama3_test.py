# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only consumer contracts, including fresh-process framework isolation."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from audit_drama3 import (
    CHECKPOINT,
    SOURCE,
    ReferenceAudit,
    checked_file,
    digest,
    load_inputs,
    performance_result,
    validate_ownership,
    validate_profile,
    verify_artifacts,
)
from benchmark_drama3 import greedy_model
from drama3_runtime import (
    SOURCE_CONTEXT_CONTRACT,
    SOURCE_CONTEXT_METADATA,
    BoundSession,
    numpy_to_ort,
    ort_to_numpy,
)
from onnxscript import GraphBuilder


def descriptor(path):
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def save_graph(path, inputs, build, *, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    builder = GraphBuilder(graph)
    values = {name: builder.input(name, dtype, shape) for name, dtype, shape in inputs}
    for name, value, dtype, shape in build(builder.op, values):
        value.type, value.shape = ir.TensorType(dtype), ir.Shape(shape)
        builder.add_output(value, name)
    model = ir.Model(graph, ir_version=12)
    if metadata:
        model.graph.metadata_props.update(metadata)
        model.metadata_props.update(metadata)
    ir.save(model, path)


@pytest.fixture
def consumer(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    config = {
        "text_config": {"vocab_size": 8, "max_position_embeddings": 256},
        "audio_decoder_config": {"num_codebooks": 2, "vocab_size": 16},
        "semantic_start_token_id": 2,
        "semantic_end_token_id": 4,
        "eos_token_id": 7,
    }
    config_path = assets / "config.json"
    config_path.write_text(json.dumps(config))
    f32, i64, boolean = ir.DataType.FLOAT, ir.DataType.INT64, ir.DataType.BOOL
    t = "time"
    paths = {
        name: assets / name / "model.onnx"
        for name in ("embedding", "decoder", "audio_generator", "main_policy", "codec")
    }

    def embedding(op, x):
        value = op.Cast(x["frames"], to=f32)
        return [
            ("inputs_embeds", value, f32, [1, t, 3]),
            ("shared_input", op.Identity(value), f32, [1, t, 3]),
        ]

    save_graph(paths["embedding"], [("frames", i64, [1, t, 3])], embedding)
    cache_inputs = [
        ("past_key_values.0.key", f32, [1, 1, "past", 1]),
        ("past_key_values.0.value", f32, [1, 1, "past", 1]),
        ("past_key_values.1.conv_state", f32, [1, 3, 2]),
        ("past_key_values.1.recurrent_state", f32, [1, 1, 1, 1]),
    ]

    def caches(op, x, hidden):
        current = op.Unsqueeze(op.ReduceSum(hidden, [-1], keepdims=1), [1])
        return [
            (
                "present." + name.removeprefix("past_key_values."),
                op.Concat(value, current, axis=2)
                if name.endswith((".key", ".value"))
                else op.Add(value, 1.0),
                f32,
                [1, 1, "total", 1]
                if name.endswith((".key", ".value"))
                else ([1, 3, 2] if name.endswith("conv_state") else [1, 1, 1, 1]),
            )
            for name, value in x.items()
            if name.startswith("past_key_values.")
        ]

    def decoder(op, x):
        carry = op.Add(
            op.ReduceSum(x["past_key_values.1.recurrent_state"], keepdims=0),
            op.ReduceSum(x["past_key_values.1.conv_state"], keepdims=0),
        )
        hidden = op.Add(op.Add(x["inputs_embeds"], 1.0), carry)
        logits = op.Add(
            op.Pad(hidden, [0, 0, 0, 0, 0, 5]),
            op.Constant(value_floats=[0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        return [
            ("hidden_states", hidden, f32, [1, t, 3]),
            ("logits", logits, f32, [1, t, 8]),
            *caches(op, x, hidden),
        ]

    save_graph(
        paths["decoder"],
        [
            ("inputs_embeds", f32, [1, t, 3]),
            ("position_ids", i64, [1, t]),
            ("attention_mask", i64, [1, "total"]),
            *cache_inputs,
        ],
        decoder,
        metadata={SOURCE_CONTEXT_METADATA: SOURCE_CONTEXT_CONTRACT},
    )

    def tower(op, x):
        hidden = op.Add(x["hidden_states"], x["shared_input"])
        code = op.Mod(op.Cast(op.ReduceSum(hidden, [-1], keepdims=1), to=i64), 16)
        codes = op.Concat(op.Slice(x["codes"], [0], [1], [2]), code, axis=-1)
        return [
            ("codebook_logits", op.Unsqueeze(hidden, [2]), f32, [1, t, 1, 3]),
            ("frame_codes", codes, i64, [1, t, 2]),
            ("valid", op.Constant(value_ints=[1]), i64, [1]),
            ("next_counter", op.Identity(x["counter"]), i64, [1]),
            *caches(op, x, hidden),
        ]

    save_graph(
        paths["audio_generator"],
        [
            ("hidden_states", f32, [1, t, 3]),
            ("shared_input", f32, [1, t, 3]),
            ("codes", i64, [1, t, 2]),
            ("semantic_mask", boolean, [1, t]),
            ("position_ids", i64, [1, t]),
            ("attention_mask", i64, [1, "total"]),
            ("generate", boolean, []),
            ("do_sample", boolean, []),
            ("temperature", f32, [1]),
            ("top_k", i64, [1]),
            ("top_p", f32, [1]),
            ("seed", i64, [1]),
            ("counter", i64, [1]),
            *cache_inputs,
        ],
        tower,
    )
    paths["main_policy"].parent.mkdir()
    ir.save(greedy_model(config, f32), paths["main_policy"])

    def codec(op, x):
        audio = op.Cast(op.ReduceSum(x["codes"], [1], keepdims=1), to=f32)
        samples = op.Tile(op.Unsqueeze(audio, [-1]), [1, 1, 1, 2048])
        return [("waveform", op.Reshape(samples, [1, 1, -1]), f32, [1, 1, "samples"])]

    save_graph(
        paths["codec"],
        [("codes", i64, [1, 2, t])],
        codec,
        metadata={"sample_rate": "44100", "samples_per_frame": "2048"},
    )
    # Independent NumPy expectations: no output/cache from a target session is
    # used to build the fixture. The second prompt/frame differs from the first.
    prompt = np.array([[1, 0, 1], [2, 1, 2]], np.int64)
    current = prompt
    states, frames = [], []
    for step in range(3):
        embedded = current.astype(np.float32)[None]
        hidden = embedded + 1 + 7 * step
        logits = np.pad(hidden, ((0, 0), (0, 0), (0, 5)))
        logits[..., 2] += 100
        tower_logits = (hidden[:, -1:] + embedded[:, -1:])[:, :, None]
        code = int(tower_logits.sum()) % 16
        frame = np.array([[2, 0, code]], np.int64)
        state = {
            "input_frames": current,
            "inputs_embeds": embedded,
            "shared_input": embedded.copy(),
            "hidden_states": hidden,
            "logits": logits,
            "position_ids": np.arange(2, dtype=np.int64)[None]
            if not step
            else np.array([[1 + step]], np.int64),
            "tower_logits": tower_logits,
            "sampled_frame": frame,
        }
        if not step:
            state["prefill_tower_logits"] = (hidden[:, :-1] + embedded[:, :-1])[:, :, None]
        path = assets / f"state-{step:02}.npz"
        np.savez(path, **state)
        states.append({"index": step, **descriptor(path), "fields": sorted(state)})
        frames.append(frame[0].tolist())
        current = frame
    golden = {
        "model_id": "fishaudio/drama-3",
        "revision": CHECKPOINT,
        "source_revision": SOURCE,
        "dtype": "float32",
        "expert_execution": "dense",
        "do_sample": False,
        "reference_environment": {"fixture": "independent synthetic NumPy"},
        "attempted_frames": 3,
        "max_new_tokens": 3,
        "prompt_rows": 2,
        "generated_frames": frames,
    }
    golden_path = assets / "golden.json"
    golden_path.write_text(json.dumps(golden))
    import hashlib

    request_data = {
        **{
            key: golden[key]
            for key in (
                "do_sample",
                "reference_environment",
                "attempted_frames",
                "max_new_tokens",
                "prompt_rows",
            )
        },
        "golden_path": str(golden_path),
        "golden_sha256": digest(golden_path),
        "states": states,
        "initial_input_frames_shape": list(prompt.shape),
        "initial_input_frames_sha256": hashlib.sha256(prompt.tobytes()).hexdigest(),
        "expected_generated_frames_shape": [3, 3],
        "expected_generated_frames_sha256": hashlib.sha256(
            np.asarray(frames, np.int64).tobytes()
        ).hexdigest(),
    }
    inputs = {
        "schema_version": 1,
        "checkpoint_revision": CHECKPOINT,
        "source_revision": SOURCE,
        "requests": {"f32": request_data},
    }
    inputs_path = assets / "inputs.json"
    inputs_path.write_text(json.dumps(inputs))
    import onnxruntime.capi.onnxruntime_pybind11_state as native

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(paths["decoder"]), options, providers=["CPUExecutionProvider"]
    )
    profile = {
        "schema_version": 1,
        "status": "ready",
        "dtype": "f32",
        "expert_execution": "dense",
        "checkpoint_revision": CHECKPOINT,
        "source_revision": SOURCE,
        "inputs_sha256": digest(inputs_path),
        "max_total_length": 256,
        "device": "cpu",
        "device_id": 0,
        "optimization": "ORT_ENABLE_ALL",
        "providers": ["CPUExecutionProvider"],
        "actual_providers": session.get_providers(),
        "actual_provider_options": session.get_provider_options(),
        "disabled_optimizers": [],
        "session_config": {},
        "policy": "saved-greedy-main-v1",
        "config": descriptor(config_path),
        "runtime": {
            "ort_version": ort.__version__,
            "ort_build": ort.get_build_info(),
            "files": [descriptor(Path(native.__file__))],
        },
        "components": {
            name: {
                "graph": descriptor(path),
                "external_data": [],
                "required_runtime_features": "",
                "dtype": "int64" if name == "main_policy" else "f32",
            }
            for name, path in paths.items()
        },
    }
    selection = assets / "selection.json"
    selection.write_text(json.dumps(profile))
    return SimpleNamespace(
        assets=assets,
        inputs=inputs_path,
        selection=selection,
        profile=profile,
        data=inputs,
        request=request_data,
        golden=golden,
        paths=paths,
        prompt=prompt,
    )


def subprocess_run(code, *arguments):
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "-1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(Path(__file__).parent),
    }
    block_frameworks = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mobius', 'onnxscript', 'onnx_ir'}:
            raise ImportError('blocked target dependency: ' + fullname)
sys.meta_path.insert(0, Block())
"""
    return subprocess.run(
        [sys.executable, "-c", block_frameworks + code, *map(str, arguments)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def test_fresh_process_import_and_cli_isolation():
    completed = subprocess_run("""
import audit_drama3
import drama3_runtime
assert not any(k.split('.')[0] in {'torch', 'mobius'} for k in sys.modules)
audit_drama3.parser().parse_args(['--help'])
""")
    assert completed.returncode == 0, completed.stderr
    assert "--selection-sha256" in completed.stdout


def execute_consumer(consumer, output, *extra):
    return subprocess_run(
        """
import audit_drama3
args = audit_drama3.parser().parse_args(sys.argv[1:])
result = audit_drama3.run(args)
assert not any(k.split('.')[0] in {'torch', 'mobius'} for k in sys.modules)
assert result['native_cuda13_acceptance'] is False
""",
        "--inputs",
        consumer.inputs,
        "--inputs-sha256",
        digest(consumer.inputs),
        "--selection",
        consumer.selection,
        "--selection-sha256",
        digest(consumer.selection),
        "--output",
        output,
        "--raw-bits",
        *extra,
    )


def test_actual_cpu_chain_in_fresh_framework_blocked_process(consumer, tmp_path):
    completed = execute_consumer(
        consumer, tmp_path / "results", "--benchmark", "--warmup", "1"
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "results" / "report.json").read_text())
    assert report["status"] == "passed"
    assert report["audit"]["source_fields"] == 25
    assert report["audit"]["float_tensors"] == 16
    assert report["audit"]["f32_patterns"] == 80
    assert report["performance"]["measurement_qualified"] is False
    assert report["performance"]["meets_end_to_end_rtf_target"] is None
    assert len(report["performance"]["frame_seconds"]) == 3
    np.testing.assert_array_equal(
        np.load(tmp_path / "results" / "generated-frames.npy"),
        consumer.golden["generated_frames"],
    )
    expected_samples = np.asarray(consumer.golden["generated_frames"])[:, 1:].sum(-1)
    np.testing.assert_array_equal(
        np.load(tmp_path / "results" / "waveform.npy")[0, 0],
        np.repeat(expected_samples.astype(np.float32), 2048),
    )


def test_diagnostics_record_actual_optimized_graphs_and_executed_opcodes(consumer, tmp_path):
    completed = execute_consumer(
        consumer, tmp_path / "diagnostics", "--diagnostics", "--benchmark", "--warmup", "1"
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "diagnostics" / "report.json").read_text())
    assert report["selection"]["disabled_optimizers"] == []
    assert report["selection"]["optimization"] == "ORT_ENABLE_ALL"
    assert report["performance"]["instrumented"] is True
    assert report["performance"]["measurement_qualified"] is False
    for evidence in report["diagnostics"].values():
        assert (
            digest(evidence["optimized_graph"]["path"])
            == evidence["optimized_graph"]["sha256"]
        )
        assert digest(evidence["profile"]["path"]) == evidence["profile"]["sha256"]
        assert evidence["executed_operators"]
        for item in evidence["executed_operators"]:
            assert item["provider"] == "CPUExecutionProvider"
            assert item["calls"] > 0
        assert all(
            Path(item["path"]).is_relative_to(tmp_path / "diagnostics")
            for item in evidence["files"]
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float16, ml_dtypes.bfloat16, np.bool_])
@pytest.mark.parametrize("shape", [(), (0, 2), (1, 0, 3), (2, 2)])
def test_transport_preserves_rank_empty_dtype_and_bits(dtype, shape):
    array = np.zeros(shape, dtype=dtype)
    if array.size and dtype != np.bool_:
        array.flat[0] = -0.0
    value = numpy_to_ort(array)
    actual = ort_to_numpy(value)
    assert actual.shape == shape and actual.dtype == array.dtype
    assert actual.tobytes() == array.tobytes()


def test_bf16_gpu_pointer_is_never_dereferenced():
    value = SimpleNamespace(data_type=lambda: "tensor(bfloat16)", device_name=lambda: "cuda")
    with pytest.raises(ValueError, match="CPU-bound"):
        ort_to_numpy(value)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_bound_identity_runs_low_dtype_without_framework(consumer, tmp_path, dtype):
    path = consumer.assets / "identity.onnx"
    save_graph(
        path,
        [("x", dtype, [2])],
        lambda op, x: [("y", op.Identity(x["x"]), dtype, [2])],
    )
    completed = subprocess_run(
        """
import numpy as np
import ml_dtypes
from drama3_runtime import BoundSession, ort_to_numpy
dtype = np.float16 if sys.argv[2] == '10' else ml_dtypes.bfloat16
x = np.array([-0.0, 0.75], dtype=dtype)
y = ort_to_numpy(BoundSession(__import__('pathlib').Path(sys.argv[1]),
    device='cpu').run({'x': x})['y'])
assert x.tobytes() == y.tobytes()
""",
        path,
        int(dtype),
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("mutation", ["hash", "fields", "index", "source", "frames"])
def test_input_manifest_rejects_mutations(consumer, mutation):
    inputs = copy.deepcopy(consumer.data)
    request_data = inputs["requests"]["f32"]
    if mutation == "hash":
        request_data["states"][1]["sha256"] = "0" * 64
    elif mutation == "fields":
        request_data["states"][0]["fields"].remove("logits")
    elif mutation == "index":
        request_data["states"][1]["index"] = 0
    elif mutation == "source":
        inputs["source_revision"] = "0" * 40
    else:
        request_data["expected_generated_frames_sha256"] = "0" * 64
    consumer.inputs.write_text(json.dumps(inputs))
    with pytest.raises(ValueError):
        load_inputs(consumer.inputs, digest(consumer.inputs), "f32")


@pytest.mark.parametrize("field", ["inputs_embeds", "hidden_states", "tower_logits"])
def test_matching_frame_hash_cannot_bypass_full_vector_gate(consumer, tmp_path, field):
    state = consumer.assets / "state-01.npz"
    with np.load(state) as saved:
        values = {name: saved[name] for name in saved.files}
    values[field] = values[field] + 1
    np.savez(state, **values)
    consumer.data["requests"]["f32"]["states"][1].update(descriptor(state))
    consumer.inputs.write_text(json.dumps(consumer.data))
    consumer.profile["inputs_sha256"] = digest(consumer.inputs)
    consumer.selection.write_text(json.dumps(consumer.profile))
    completed = execute_consumer(consumer, tmp_path / "failed")
    assert completed.returncode != 0
    report = json.loads((tmp_path / "failed" / "report.json").read_text())
    assert report["status"] == "failed"
    assert field in report["error"]
    assert "generated_frames_sha256" not in report


def test_raw_bit_gate_preserves_signed_zero(consumer):
    audit = ReferenceAudit(consumer.request, "f32", raw_bits=True)
    audit.state = {"inputs_embeds": np.array([-0.0], np.float32)}
    with pytest.raises(AssertionError):
        audit.check(0, "inputs_embeds", np.array([0.0], np.float32))


@pytest.mark.parametrize("mutation", ["notready", "extended", "capture", "tf32", "cuda12"])
def test_selected_profile_controls_are_not_silently_changed(consumer, mutation):
    profile = copy.deepcopy(consumer.profile)
    profile.update(
        device="cuda",
        gpu_uuid="GPU-owned",
        providers=[
            [
                "CUDAExecutionProvider",
                {"device_id": "0", "use_tf32": "0", "enable_cuda_graph": "0"},
            ],
            "CPUExecutionProvider",
        ],
    )
    profile["runtime"].update(cuda_toolkit="13.0.48", cublas="13.0.0.19", cudnn="9.14.0.64")
    if mutation == "notready":
        profile["status"] = "NOT READY / NOT QUALIFIED"
    elif mutation == "extended":
        profile["optimization"] = "ORT_ENABLE_EXTENDED"
    elif mutation == "capture":
        profile["providers"][0][1]["enable_cuda_graph"] = "1"
    elif mutation == "tf32":
        profile["providers"][0][1]["use_tf32"] = "1"
    else:
        profile["runtime"]["cuda_toolkit"] = "12.8"
    with pytest.raises(ValueError):
        validate_profile(profile)


@pytest.mark.parametrize("drift", ["provider", "options"])
def test_selected_session_forwards_all_controls_and_rejects_provider_drift(
    consumer, monkeypatch, drift
):
    seen = []

    def fake_session(path, options, *, providers, disabled_optimizers):
        seen.append((options, providers, disabled_optimizers))
        return SimpleNamespace(
            get_providers=lambda: (
                ["UnexpectedExecutionProvider"]
                if drift == "provider"
                else consumer.profile["actual_providers"]
            ),
            get_provider_options=lambda: {"CPUExecutionProvider": {"unselected": "1"}},
        )

    monkeypatch.setattr(ort, "InferenceSession", fake_session)
    profile = copy.deepcopy(consumer.profile)
    profile["disabled_optimizers"] = ["RotaryEmbeddingFusion"]
    profile["session_config"] = {"session.disable_prepacking": "1"}
    with pytest.raises(RuntimeError, match="provider"):
        BoundSession(consumer.paths["decoder"], device="cpu", selection=profile)
    options, providers, disabled = seen[0]
    assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    assert options.get_session_config_entry("session.disable_prepacking") == "1"
    assert disabled == {"RotaryEmbeddingFusion"}
    assert providers == ["CPUExecutionProvider"]


@pytest.mark.parametrize("phase", [0, 1, 2])
def test_foreign_process_at_any_phase_invalidates_measurement(phase):
    snapshots = [{"uuid": "GPU-owned", "foreign_pids": []} for _ in range(3)]
    snapshots[phase]["foreign_pids"] = [1234]
    with pytest.raises(RuntimeError, match="foreign process"):
        validate_ownership(snapshots, "GPU-owned")


def test_uuid_drift_invalidates_measurement():
    with pytest.raises(RuntimeError, match="UUID"):
        validate_ownership([{"uuid": "GPU-other", "foreign_pids": []}], "GPU-owned")


def test_raw_first_cached_outlier_is_retained_and_rtf_gate_is_not_trimmed():
    timing = {
        "frame_seconds": [0.01, 4.0, *([0.01] * 98)],
        "token_pipeline_seconds": 5.2,
        "prefill_seconds": 0.2,
        "first_frame_seconds": 0.21,
    }
    profile = {"device": "cuda", "gpu_uuid": "GPU-owned"}
    snapshots = [{"uuid": "GPU-owned", "foreign_pids": []} for _ in range(3)]
    result = performance_result(
        timing,
        0.1,
        np.zeros((100, 3), np.int64),
        snapshots,
        profile,
        {"raw_bits_required": True},
    )
    assert result["frame_seconds"][1] == timing["frame_seconds"][1]
    assert result["measurement_qualified"]
    assert result["meets_end_to_end_rtf_target"] is False


def test_instrumented_cuda_timing_cannot_qualify_even_below_rtf_gate():
    timing = {
        "frame_seconds": [0.001] * 100,
        "token_pipeline_seconds": 0.11,
        "prefill_seconds": 0.01,
        "first_frame_seconds": 0.011,
    }
    snapshots = [{"uuid": "GPU-owned", "foreign_pids": []} for _ in range(3)]
    result = performance_result(
        timing,
        0.01,
        np.zeros((100, 3), np.int64),
        snapshots,
        {"device": "cuda", "gpu_uuid": "GPU-owned"},
        {"raw_bits_required": True},
        instrumented=True,
    )
    assert result["end_to_end_rtf"] < 1.0
    assert result["measurement_qualified"] is False
    assert result["meets_end_to_end_rtf_target"] is None


def test_missing_weights_fail_explicitly(consumer):
    path = consumer.assets / "weights.bin"
    path.write_bytes(b"weights")
    item = descriptor(path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        checked_file(item)


def test_unlisted_external_data_is_not_a_complete_weight_inventory(consumer):
    (consumer.paths["decoder"].parent / "weights.bin").write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="external-data inventory"):
        verify_artifacts(consumer.profile)


def test_positions_are_expectations_not_execution_controls(consumer, tmp_path):
    path = consumer.assets / "state-01.npz"
    with np.load(path) as arrays:
        values = {name: arrays[name] for name in arrays.files}
    values["position_ids"] += 1
    np.savez(path, **values)
    consumer.data["requests"]["f32"]["states"][1].update(descriptor(path))
    consumer.inputs.write_text(json.dumps(consumer.data))
    consumer.profile["inputs_sha256"] = digest(consumer.inputs)
    consumer.selection.write_text(json.dumps(consumer.profile))
    completed = execute_consumer(consumer, tmp_path / "position-failure")
    assert completed.returncode != 0
    report = json.loads((tmp_path / "position-failure" / "report.json").read_text())
    assert "position_ids" in report["error"]


@pytest.mark.parametrize("mutation", ["runtime", "request", "dtype"])
def test_provenance_drift_fails_before_target_execution(consumer, tmp_path, mutation):
    if mutation == "runtime":
        consumer.profile["runtime"]["ort_build"] = "a different binary"
    elif mutation == "request":
        consumer.profile["inputs_sha256"] = "0" * 64
    else:
        consumer.profile["components"]["embedding"]["dtype"] = "bf16"
    consumer.selection.write_text(json.dumps(consumer.profile))
    completed = execute_consumer(consumer, tmp_path / "provenance-failure")
    assert completed.returncode != 0
    assert "generated-frames.npy" not in {
        item.name for item in (tmp_path / "provenance-failure").glob("*")
    }


def test_output_root_is_new_and_never_rewrites_frozen_inputs(consumer):
    completed = execute_consumer(consumer, consumer.assets / "forbidden-output")
    assert completed.returncode != 0
    assert "read-only input" in completed.stderr
    assert not (consumer.assets / "forbidden-output").exists()


@pytest.mark.parametrize(
    "mode", ["ok", "foreign-cuda", "selected-cuda12", "torch", "unhashed-provider"]
)
def test_loaded_library_attestation_without_loading_cuda(consumer, mode):
    profile = copy.deepcopy(consumer.profile)
    profile["device"] = "cuda"
    paths = {}
    for name in (
        "libcudart.so.13",
        "libcublas.so.13",
        "libonnxruntime_providers_cuda.so",
        "libcudart.so.12",
        "libtorch_cuda.so",
    ):
        path = consumer.assets / name
        path.write_bytes(b"synthetic mapping, not a loaded library")
        paths["/mock/" + name] = str(path)
    profile["runtime"]["wheel"] = descriptor(consumer.paths["decoder"])
    profile["runtime"]["libraries"] = [
        descriptor(Path(path)) for key, path in paths.items() if key.endswith(".13")
    ]
    if mode == "selected-cuda12":
        profile["runtime"]["libraries"].append(
            descriptor(Path(paths["/mock/libcudart.so.12"]))
        )
    if mode != "unhashed-provider":
        profile["runtime"]["files"].append(
            descriptor(Path(paths["/mock/libonnxruntime_providers_cuda.so"]))
        )
    consumer.selection.write_text(json.dumps(profile))
    completed = subprocess_run(
        """
import json
from pathlib import Path
import audit_drama3
profile = json.loads(Path(sys.argv[1]).read_text())
paths = json.loads(sys.argv[2])
mode = sys.argv[3]
names = ['libcudart.so.13', 'libcublas.so.13', 'libonnxruntime_providers_cuda.so']
if mode in ('foreign-cuda', 'selected-cuda12'):
    names.append('libcudart.so.12')
if mode == 'torch':
    names.append('libtorch_cuda.so')
class Maps:
    def is_file(self):
        return True
    def read_text(self):
        return '\\n'.join('1-2 r-xp 0 0 0 /mock/' + name for name in names)
def mapped_path(value):
    if str(value) == '/proc/self/maps':
        return Maps()
    return Path(paths.get(str(value), value))
audit_drama3.Path = mapped_path
audit_drama3.runtime_identity(profile)
""",
        consumer.selection,
        json.dumps(paths),
        mode,
    )
    if mode == "ok":
        assert completed.returncode == 0, completed.stderr
    else:
        assert completed.returncode != 0
        assert "RuntimeError" in completed.stderr
