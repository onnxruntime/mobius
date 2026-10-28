# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Consume pinned Drama 3 references and a separately selected saved-graph profile.

This is not an exporter, source reference generator, or runtime installer. Both
audit and benchmark use the same selected ORT_ENABLE_ALL sessions. CUDA admission
requires exact 13.0 runtime identities and an explicit lease UUID. CPU runs are
diagnostic only. See docs/drama3-runtime-harness.md for the manifest contract.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from drama3_runtime import (
    SOURCE_CONTEXT_CONTRACT,
    SOURCE_CONTEXT_METADATA,
    BoundSession,
    gpu_snapshot,
    next_cache,
    ort_to_numpy,
    require_idle,
    sequence_view,
)

CHECKPOINT = "b2b5ee998034b2975720906c815e7fb0e9f007bd"
SOURCE = "541566932abbaebcb0a40e562a1f846738383f38"
COMPONENTS = {"embedding", "decoder", "audio_generator", "main_policy", "codec"}
SMALL_OUTPUTS = {"token", "frame_codes", "valid", "semantic_code", "next_counter"}
FLOAT_FIELDS = {"inputs_embeds", "shared_input", "hidden_states", "logits", "tower_logits"}


def digest(path):
    """Hash large weights with bounded reads, never parsing protobuf."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def checked_file(descriptor):
    path = Path(descriptor["path"]).resolve(strict=True)
    if not path.is_file() or digest(path) != descriptor["sha256"]:
        raise ValueError(f"File SHA256 mismatch: {path}")
    if "bytes" in descriptor and path.stat().st_size != descriptor["bytes"]:
        raise ValueError(f"File byte count mismatch: {path}")
    return path


def checked_json(path, sha256):
    path = checked_file({"path": str(path), "sha256": sha256})
    return json.loads(path.read_text())


def validate_profile(profile):
    if profile["schema_version"] != 1 or profile["status"] != "ready":
        raise ValueError("Selected consumer graph/profile is NOT READY")
    if profile["optimization"] != "ORT_ENABLE_ALL":
        raise ValueError("Full-vector audit and benchmark require ORT_ENABLE_ALL")
    if profile["dtype"] not in ("f32", "f16", "bf16"):
        raise ValueError("Unknown selected tensor dtype")
    if profile["expert_execution"] not in ("dense", "dynamic_fp8"):
        raise ValueError("Unknown expert execution contract")
    if profile["checkpoint_revision"] != CHECKPOINT or profile["source_revision"] != SOURCE:
        raise ValueError("Selected consumer checkpoint/source revisions differ")
    if profile["device"] not in ("cpu", "cuda") or profile["device_id"] < 0:
        raise ValueError("Invalid execution device")
    if profile["policy"] != "saved-greedy-main-v1":
        raise ValueError("Frozen greedy references require the saved ONNX greedy policy")
    if set(profile["components"]) != COMPONENTS:
        raise ValueError("Select embedding, decoder, audio_generator, main_policy and codec")
    if not isinstance(profile["disabled_optimizers"], list) or not all(
        isinstance(name, str) and name for name in profile["disabled_optimizers"]
    ):
        raise ValueError("disabled_optimizers must be an explicit list of names")
    if len(set(profile["disabled_optimizers"])) != len(profile["disabled_optimizers"]):
        raise ValueError("Duplicate disabled optimizers")
    if profile["device"] == "cuda":
        runtime = profile["runtime"]
        if (
            runtime["cuda_toolkit"] != "13.0.48"
            or runtime["cublas"] != "13.0.0.19"
            or runtime["cudnn"] != "9.14.0.64"
        ):
            raise ValueError("Active acceptance target is exact CUDA 13.0, not legacy/13.3")
        providers = profile["providers"]
        if (
            len(providers) != 2
            or providers[0][0] != "CUDAExecutionProvider"
            or providers[1] != "CPUExecutionProvider"
        ):
            raise ValueError("Select explicit CUDA+CPU providers")
        options = providers[0][1]
        if (
            options.get("use_tf32") != "0"
            or options.get("enable_cuda_graph") != "0"
            or str(options.get("device_id")) != str(profile["device_id"])
        ):
            raise ValueError("Selected provider must disable TF32/capture and pin device")
        if not profile["gpu_uuid"].startswith("GPU-"):
            raise ValueError("CUDA qualification requires a lease GPU UUID")
        if not 1 <= profile["max_total_length"] <= 256:
            raise ValueError("Native whole-model profile is limited to total length <= 256")
    elif profile["providers"] != ["CPUExecutionProvider"]:
        raise ValueError("CPU diagnostics cannot select CUDA providers")
    for component in profile["components"].values():
        if component["dtype"] not in ("f32", "f16", "bf16", "int64"):
            raise ValueError("Invalid component dtype")
    if profile["components"]["codec"]["dtype"] != "f32":
        raise ValueError("Waveform codec must be the saved FP32 graph")
    if any(
        profile["components"][name]["dtype"] != profile["dtype"]
        for name in ("embedding", "decoder", "audio_generator")
    ):
        raise ValueError("Selected model components have inconsistent dtypes")


def load_inputs(path, sha256, dtype):
    manifest = checked_json(path, sha256)
    if manifest["schema_version"] != 1:
        raise ValueError("Unknown input manifest schema")
    if manifest["checkpoint_revision"] != CHECKPOINT or manifest["source_revision"] != SOURCE:
        raise ValueError("Unpinned checkpoint/source revision")
    request = manifest["requests"][dtype]
    golden = checked_json(request["golden_path"], request["golden_sha256"])
    if (
        golden["model_id"] != "fishaudio/drama-3"
        or golden["revision"] != CHECKPOINT
        or golden["source_revision"] != SOURCE
        or golden["dtype"] != {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}[dtype]
        or golden["do_sample"] is not False
        or request["do_sample"] is not False
        or golden["reference_environment"] != request["reference_environment"]
        or golden["attempted_frames"] != request["attempted_frames"]
        or len(request["states"]) != golden["attempted_frames"]
        or golden["max_new_tokens"] != request["max_new_tokens"]
    ):
        raise ValueError("Golden and immutable request contract differ")
    for index, state in enumerate(request["states"]):
        if state["index"] != index:
            raise ValueError("Reference state indices are not contiguous")
        checked_file(state)
        with np.load(state["path"], allow_pickle=False) as arrays:
            required = FLOAT_FIELDS | {"input_frames", "position_ids", "sampled_frame"}
            if not index:
                required = required | {"prefill_tower_logits"}
            if set(arrays.files) != set(state["fields"]) or set(arrays.files) != required:
                raise ValueError(f"Incomplete/unknown source fields at state {index}")
    with np.load(request["states"][0]["path"], allow_pickle=False) as arrays:
        prompt = arrays["input_frames"].copy()
    if (
        prompt.dtype != np.int64
        or list(prompt.shape) != request["initial_input_frames_shape"]
        or hashlib.sha256(prompt.tobytes()).hexdigest()
        != request["initial_input_frames_sha256"]
        or len(prompt) != golden["prompt_rows"]
        or len(prompt) != request["prompt_rows"]
    ):
        raise ValueError("Initial request frames differ from the frozen input")
    frames = np.asarray(golden["generated_frames"], dtype=np.int64)
    if (
        list(frames.shape) != request["expected_generated_frames_shape"]
        or hashlib.sha256(frames.tobytes()).hexdigest()
        != request["expected_generated_frames_sha256"]
    ):
        raise ValueError("Golden generated-frame inventory differs")
    return manifest, request, golden, prompt


def runtime_identity(profile, *, check_loaded=True):
    """Verify installed ORT plus all selected native binaries and loaded CUDA DSOs."""
    runtime = profile["runtime"]
    if (
        ort.__version__ != runtime["ort_version"]
        or ort.get_build_info() != runtime["ort_build"]
    ):
        raise RuntimeError("Installed ORT version/build differs from selected runtime")
    if not runtime["files"]:
        raise ValueError("Runtime binary inventory is required")
    files = [checked_file(item) for item in runtime["files"]]
    import onnxruntime.capi.onnxruntime_pybind11_state as native

    if Path(native.__file__).resolve() not in files:
        raise ValueError("Runtime inventory must pin the loaded ORT native binding")
    if any(name.split(".")[0] in ("torch", "mobius") for name in sys.modules):
        raise RuntimeError("Target process imported Torch or the eager Mobius builder")
    if profile["device"] == "cuda":
        checked_file(runtime["wheel"])
        selected = {checked_file(item) for item in runtime["libraries"]}
        if not selected:
            raise ValueError("CUDA library inventory is required")
    if profile["device"] == "cuda" and check_loaded:
        maps_path = Path("/proc/self/maps")
        if not maps_path.is_file():
            raise RuntimeError(
                "CUDA loaded-library attestation currently requires Linux procfs"
            )
        loaded = set()
        loaded_ort = set()
        for line in maps_path.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/"):
                library = Path(fields[5]).resolve()
                name = library.name.lower()
                if "onnxruntime" in name:
                    loaded_ort.add(library)
                if any(
                    marker in name
                    for marker in ("libcuda", "libcublas", "libcudnn", "libnvrtc", "libnvjit")
                ):
                    loaded.add(library)
                    if (
                        name.startswith(
                            (
                                "libcudart.",
                                "libcublas.",
                                "libcublaslt.",
                                "libnvrtc",
                                "libnvjitlink.",
                            )
                        )
                        and ".so.13" not in name
                    ):
                        raise RuntimeError(f"Pre-CUDA13 library loaded: {library}")
                if "libtorch" in name or name.startswith("libc10"):
                    raise RuntimeError("Torch native library loaded in the target process")
        if loaded != selected or not any("libcudart" in item.name for item in loaded):
            raise RuntimeError("Loaded CUDA library inventory differs from selected runtime")
        if not loaded_ort <= set(files):
            raise RuntimeError("Loaded ORT provider/runtime binary is not fingerprinted")
    return {"ort_version": ort.__version__, "ort_build": ort.get_build_info()}


def load_sessions(profile, *, evidence_root=None):
    sessions = {}
    for name, component in profile["components"].items():
        graph = checked_file(component["graph"])
        for weights in component["external_data"]:
            checked_file(weights)
        # An empty string means no native features, not that every optional runtime
        # operator is mandatory for a portable graph.
        if not isinstance(component["required_runtime_features"], str):
            raise TypeError("Runtime feature inventory must be explicit")
        session = BoundSession(
            graph,
            device=profile["device"],
            device_id=profile["device_id"],
            selection=profile,
            evidence_directory=evidence_root / name if evidence_root is not None else None,
        )
        metadata = session.session.get_modelmeta().custom_metadata_map
        if (
            metadata.get("mobius.required_runtime_features", "")
            != component["required_runtime_features"]
        ):
            raise ValueError(f"Actual runtime features differ for {name}")
        if name == "decoder" and profile["dtype"] != "f32":
            if metadata.get(SOURCE_CONTEXT_METADATA) != SOURCE_CONTEXT_CONTRACT:
                raise ValueError("Reduced-precision decoder lacks source context contract")
        if name == "codec":
            if (
                metadata.get("sample_rate") != "44100"
                or metadata.get("samples_per_frame") != "2048"
            ):
                raise ValueError("Codec must declare 44100Hz/2048 samples per frame")
            if any(
                port.type not in ("tensor(float)", "tensor(int64)")
                for port in session.session.get_inputs() + session.session.get_outputs()
            ):
                raise ValueError("Selected codec does not have an FP32/int64 I/O contract")
        sessions[name] = session
    expected = {"f32": "tensor(float)", "f16": "tensor(float16)", "bf16": "tensor(bfloat16)"}
    if sessions["decoder"].session.get_outputs()[0].type != expected[profile["dtype"]]:
        raise ValueError("Actual decoder dtype differs from selected graph")
    features = (
        sessions["decoder"]
        .session.get_modelmeta()
        .custom_metadata_map.get("mobius.required_runtime_features", "")
    )
    experts = "dynamic_fp8" if "activation_quantization=dynamic_fp8" in features else "dense"
    if experts != profile["expert_execution"]:
        raise ValueError("Actual decoder expert execution differs from selected profile")
    if {port.name for port in sessions["main_policy"].session.get_inputs()} != {"logits"}:
        raise ValueError("Greedy main policy must be the existing saved logits-only graph")
    return sessions


def collect_evidence(sessions, root):
    """Hash ORT-serialized graphs and summarize actual execution profile opcodes."""
    result = {}
    for name, session in sessions.items():
        profile_path = Path(session.session.end_profiling()).resolve(strict=True)
        directory = (root / name).resolve(strict=True)
        if profile_path.parent != directory:
            raise ValueError("ORT profile escaped the explicit evidence directory")
        counts = collections.Counter()
        for entry in json.loads(profile_path.read_text()):
            details = entry.get("args", {})
            if details.get("provider") and details.get("op_name"):
                counts[(details["provider"], details["op_name"])] += 1
        if not counts:
            raise ValueError(f"No runtime operator evidence for {name}")
        optimized = directory / "optimized.onnx"
        result[name] = {
            "optimized_graph": {"path": str(optimized), "sha256": digest(optimized)},
            "profile": {"path": str(profile_path), "sha256": digest(profile_path)},
            "executed_operators": [
                {"provider": provider, "op_name": opcode, "calls": count}
                for (provider, opcode), count in sorted(counts.items())
            ],
            "files": [
                {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
                for path in sorted(directory.rglob("*"))
                if path.is_file()
            ],
        }
    return result


def verify_artifacts(profile):
    """Verify every file in each isolated graph directory, including external data."""
    for component in profile["components"].values():
        graph = checked_file(component["graph"])
        files = {graph, *(checked_file(item) for item in component["external_data"])}
        actual = set()
        for path in graph.parent.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Graph directories must not contain symlinks: {path}")
            if path.is_file():
                actual.add(path.resolve())
        if actual != files:
            raise ValueError(f"Incomplete graph/external-data inventory: {graph.parent}")


def request(sessions, config, prompt, limit, *, observer=None):
    """Execute actual embedding -> decoder -> ONNX policy -> single-pass tower.

    The only fixture operand is the initial prompt. All subsequent frames,
    intermediate activations and returned caches come from the target sessions.
    """
    count = config["audio_decoder_config"]["num_codebooks"]
    if (
        prompt.ndim != 2
        or prompt.shape[1] != count + 1
        or len(prompt) < 2
        or prompt.dtype != np.int64
        or limit < 1
        or len(prompt) + limit > config["text_config"]["max_position_embeddings"]
    ):
        raise ValueError("Invalid batch-one prompt or request exceeds position capacity")
    origin = time.perf_counter()
    device = sessions["decoder"].device
    decoder_cache = sessions["decoder"].empty_cache()
    tower_cache = sessions["audio_generator"].empty_cache()
    controls = {
        "generate": np.array(False),
        "do_sample": np.array(False),
        "temperature": np.array([0.7], np.float32),
        "top_k": np.array([30], np.int64),
        "top_p": np.array([0.95], np.float32),
        "seed": np.array([42], np.int64),
        "counter": np.zeros(1, np.int64),
    }

    def run(name, feeds, step, stage):
        host = SMALL_OUTPUTS
        if observer is not None:
            host = host | {
                output
                for output in sessions[name].outputs
                if not output.startswith("present.")
            }
        result = sessions[name].run(feeds, cpu_outputs=host)
        if "valid" in result and not ort_to_numpy(result["valid"]).all():
            raise FloatingPointError(f"{name} rejected inputs at step {step}")
        if "next_counter" in result and ort_to_numpy(result["next_counter"]).any():
            raise RuntimeError("Greedy ONNX tower unexpectedly advanced RNG counter")
        if observer is not None:
            observer(step, stage, feeds, result)
        return result

    positions = np.arange(len(prompt), dtype=np.int64)[None]
    mask = np.ones_like(positions)
    embedded = run("embedding", {"frames": prompt[None]}, 0, "embedding")
    decoded = run(
        "decoder",
        {
            **decoder_cache,
            "inputs_embeds": embedded["inputs_embeds"],
            "position_ids": positions,
            "attention_mask": mask,
        },
        0,
        "decoder",
    )
    decoder_cache = next_cache(decoded, device)
    semantic = (prompt[None, 1:, 0] >= config["semantic_start_token_id"]) & (
        prompt[None, 1:, 0] <= config["semantic_end_token_id"]
    )
    prefilled = run(
        "audio_generator",
        {
            **tower_cache,
            **controls,
            "hidden_states": sequence_view(decoded["hidden_states"], 0, len(prompt) - 1),
            "shared_input": sequence_view(embedded["shared_input"], 0, len(prompt) - 1),
            "codes": prompt[None, 1:, 1:],
            "semantic_mask": semantic,
            "position_ids": positions[:, :-1],
            "attention_mask": mask[:, :-1],
        },
        0,
        "prefill",
    )
    tower_cache = next_cache(prefilled, device)
    prefill_seconds = time.perf_counter() - origin
    frames, durations = [], []
    first_frame_seconds = None
    frame = prompt[None, -1:]
    for step in range(limit):
        tick = time.perf_counter()
        if step:
            position = len(prompt) - 1 + step
            positions = np.array([[position]], np.int64)
            mask = np.ones((1, position + 1), np.int64)
            embedded = run("embedding", {"frames": frame}, step, "embedding")
            decoded = run(
                "decoder",
                {
                    **decoder_cache,
                    "inputs_embeds": embedded["inputs_embeds"],
                    "position_ids": positions,
                    "attention_mask": mask,
                },
                step,
                "decoder",
            )
            decoder_cache = next_cache(decoded, device)
        selected = run(
            "main_policy",
            {
                "logits": sequence_view(
                    decoded["logits"], decoded["logits"].shape()[1] - 1, 1, squeeze_time=True
                )
            },
            step,
            "policy",
        )
        token = ort_to_numpy(selected["token"])
        if token.dtype != np.int64 or token.shape != (1,):
            raise ValueError("Main ONNX policy returned an invalid token contract")
        main = int(token[0])
        is_semantic = (
            config["semantic_start_token_id"] <= main <= config["semantic_end_token_id"]
        )
        if not is_semantic and main != config["eos_token_id"]:
            raise ValueError("Main ONNX policy selected a noncandidate token")
        if main == config["eos_token_id"] and observer is None:
            break
        codes = np.zeros((1, 1, count), np.int64)
        codes[0, 0, 0] = main - config["semantic_start_token_id"] if is_semantic else 0
        output = run(
            "audio_generator",
            {
                **tower_cache,
                **controls,
                "generate": np.array(True),
                "hidden_states": sequence_view(
                    decoded["hidden_states"], decoded["hidden_states"].shape()[1] - 1, 1
                ),
                "shared_input": sequence_view(
                    embedded["shared_input"], embedded["shared_input"].shape()[1] - 1, 1
                ),
                "codes": codes,
                "semantic_mask": np.array([[is_semantic]]),
                "position_ids": positions[:, -1:],
                "attention_mask": mask,
            },
            step,
            "tower",
        )
        codes = ort_to_numpy(output["frame_codes"])
        if codes.dtype != np.int64 or codes.shape != (1, 1, count):
            raise ValueError("Tower returned an invalid generated frame contract")
        if (codes < 0).any() or (codes >= config["audio_decoder_config"]["vocab_size"]).any():
            raise ValueError("Tower returned out-of-vocabulary codes")
        frame = np.concatenate((np.array([[[main]]], np.int64), codes), axis=-1)
        if observer is not None:
            observer(step, "frame", {"frame": frame[0]}, {})
        tower_cache = next_cache(output, device)
        if main == config["eos_token_id"]:
            break
        frames.append(frame[0, 0])
        durations.append(time.perf_counter() - tick)
        if first_frame_seconds is None:
            first_frame_seconds = time.perf_counter() - origin
    elapsed = time.perf_counter() - origin
    if not frames:
        raise RuntimeError("No generated audio frames; cannot audit/measure speech")
    return np.stack(frames), {
        "frame_seconds": durations,
        "prefill_seconds": prefill_seconds,
        "first_frame_seconds": first_frame_seconds,
        "token_pipeline_seconds": elapsed,
    }


class ReferenceAudit:
    """Observe expectations without ever supplying a source activation to execution."""

    def __init__(self, request_data, dtype, *, raw_bits):
        self.request_data = request_data
        self.dtype = dtype
        self.raw_bits = raw_bits
        self.metrics = {}
        self.fields = set()
        self.float_patterns = 0
        self._step = None
        self.state = {}

    def check(self, step, name, actual):
        expected = self.state[name]
        key = f"{step}.{name}"
        if key in self.fields:
            raise ValueError(f"Duplicate source field audit: {key}")
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError(f"Invalid shape/nonfinite target tensor: {key}")
        if np.issubdtype(expected.dtype, np.integer):
            if actual.dtype != expected.dtype:
                raise ValueError(f"Integer dtype mismatch: {key}")
            np.testing.assert_array_equal(actual, expected, err_msg=key)
        else:
            dtype_name = {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}[self.dtype]
            if str(actual.dtype) != dtype_name or expected.dtype != np.float32:
                raise ValueError(f"Source F32 transport/target dtype mismatch: {key}")
            if not np.isfinite(expected).all():
                raise ValueError(f"Nonfinite source tensor: {key}")
            transported = actual.astype(np.float32)
            difference = np.abs(transported - expected)
            self.metrics[key] = {
                "shape": list(actual.shape),
                "target_dtype": str(actual.dtype),
                "max_abs_diff": float(difference.max()) if difference.size else 0.0,
                "raw_f32_mismatches": int(
                    np.count_nonzero(transported.view(np.uint32) != expected.view(np.uint32))
                ),
            }
            tolerance = 1e-3 if self.dtype == "f32" else 1e-2
            np.testing.assert_allclose(
                transported, expected, atol=tolerance, rtol=tolerance, err_msg=key
            )
            if self.raw_bits:
                np.testing.assert_array_equal(
                    transported.view(np.uint32), expected.view(np.uint32), err_msg=key
                )
            self.float_patterns += expected.size
        self.fields.add(key)

    def __call__(self, step, stage, feeds, outputs):
        if self._step != step:
            if self._step is not None:
                self.finish_step()
            with np.load(
                self.request_data["states"][step]["path"], allow_pickle=False
            ) as data:
                self.state = {name: data[name] for name in data.files}
            self._step = step
        if stage == "embedding":
            self.check(step, "input_frames", feeds["frames"][0])
            for name in ("inputs_embeds", "shared_input"):
                self.check(step, name, ort_to_numpy(outputs[name]))
        elif stage == "decoder":
            self.check(step, "position_ids", feeds["position_ids"])
            for name in ("hidden_states", "logits"):
                self.check(step, name, ort_to_numpy(outputs[name]))
        elif stage in ("prefill", "tower"):
            name = "prefill_tower_logits" if stage == "prefill" else "tower_logits"
            self.check(step, name, ort_to_numpy(outputs["codebook_logits"]))
        elif stage == "frame":
            self.check(step, "sampled_frame", feeds["frame"])

    def finish_step(self):
        expected = {f"{self._step}.{name}" for name in self.state}
        if not expected <= self.fields:
            raise ValueError(f"Unaudited source fields: {expected - self.fields}")

    def finish(self):
        self.finish_step()
        count = sum(len(state["fields"]) for state in self.request_data["states"])
        if len(self.fields) != count:
            raise ValueError("Not every source state/field was audited")
        return {
            "source_fields": len(self.fields),
            "float_tensors": len(self.metrics),
            "f32_patterns": int(self.float_patterns),
            "raw_bits_required": self.raw_bits,
            "metrics": self.metrics,
        }


def decode_waveform(session, frames):
    tick = time.perf_counter()
    outputs = session.run(
        {"codes": frames[None, :, 1:].transpose(0, 2, 1)}, cpu_outputs={"waveform"}
    )
    waveform = ort_to_numpy(outputs["waveform"])
    elapsed = time.perf_counter() - tick
    if (
        waveform.dtype != np.float32
        or waveform.shape != (1, 1, len(frames) * 2048)
        or not np.isfinite(waveform).all()
    ):
        raise ValueError("Invalid actual-frame FP32 waveform output")
    return waveform, elapsed


def validate_ownership(snapshots, uuid):
    if any(item["uuid"] != uuid or item["foreign_pids"] for item in snapshots):
        raise RuntimeError(
            "GPU UUID changed or foreign process appeared; timing is unqualified"
        )


def performance_result(
    timing, waveform_seconds, frames, snapshots, profile, audit, *, instrumented=False
):
    durations = timing["frame_seconds"]
    if len(durations) != len(frames) or not all(
        np.isfinite(value) and value > 0
        for value in [
            *durations,
            waveform_seconds,
            timing["token_pipeline_seconds"],
            timing["prefill_seconds"],
            timing["first_frame_seconds"],
        ]
    ):
        raise ValueError("Invalid/incomplete raw latency samples")
    if profile["device"] == "cuda":
        if len(snapshots) != 3:
            raise ValueError("Need GPU snapshots before, after warmup and after measurement")
        validate_ownership(snapshots, profile["gpu_uuid"])
    audio_seconds = len(frames) * 2048 / 44100
    rtf = (timing["token_pipeline_seconds"] + waveform_seconds) / audio_seconds
    qualified = (
        profile["device"] == "cuda"
        and len(frames) >= 50
        and audit["raw_bits_required"]
        and not instrumented
    )
    return {
        **timing,
        "waveform_seconds": waveform_seconds,
        "audio_seconds": audio_seconds,
        "end_to_end_rtf": rtf,
        "end_to_end_rtf_target": 1.0,
        "measurement_qualified": qualified,
        "instrumented": instrumented,
        "meets_end_to_end_rtf_target": rtf <= 1.0 if qualified else None,
        "gpu_snapshots": snapshots,
    }


def run(args):
    profile = checked_json(args.selection, args.selection_sha256)
    validate_profile(profile)
    if profile["inputs_sha256"] != args.inputs_sha256:
        raise ValueError("Selected profile belongs to a different request manifest")
    manifest, request_data, golden, prompt = load_inputs(
        args.inputs, args.inputs_sha256, profile["dtype"]
    )
    if golden["expert_execution"] != profile["expert_execution"]:
        raise ValueError("Source and selected consumer expert execution differ")
    config_path = checked_file(profile["config"])
    config = json.loads(config_path.read_text())
    if len(prompt) + golden["attempted_frames"] - 1 > profile["max_total_length"]:
        raise ValueError("Request exceeds the selected whole-model position bound")
    if args.warmup < 1:
        raise ValueError("At least one complete warmup request is required")
    verify_artifacts(profile)
    # Refuse overwrites, including a report path beneath any read-only input root.
    output = args.output.resolve()
    inputs = [
        Path(args.inputs).resolve(),
        Path(args.selection).resolve(),
        config_path,
        Path(request_data["golden_path"]).resolve(),
        *(Path(item["path"]).resolve() for item in request_data["states"]),
    ]
    for component in profile["components"].values():
        inputs.extend(
            Path(item["path"]).resolve()
            for item in [component["graph"], *component["external_data"]]
        )
    if any(output == path or path.parent in output.parents for path in inputs):
        raise ValueError("Output root must be separate from read-only input directories")
    if profile["device"] == "cuda" and args.lease_uuid != profile["gpu_uuid"]:
        raise ValueError("Explicit lease UUID must match the selected GPU")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "failed",
        "inputs_sha256": args.inputs_sha256,
        "selection_sha256": args.selection_sha256,
        "checkpoint_revision": manifest["checkpoint_revision"],
        "source_revision": manifest["source_revision"],
        "selection": profile,
        "reference_environment": golden["reference_environment"],
        "native_cuda13_acceptance": False,
        "codec_numerical_quality": {
            "status": "UNVERIFIED",
            "independent_waveform_reference": None,
            "reason": (
                "The immutable neural DATA manifest contains no waveform oracle. "
                "Codec execution, finite output, duration and hash do not establish "
                "independent FP32 waveform numerical parity."
            ),
        },
    }
    try:
        snapshots = []
        if profile["device"] == "cuda":
            snapshots.append(gpu_snapshot(profile["device_id"]))
            validate_ownership(snapshots, profile["gpu_uuid"])
            require_idle(snapshots[0], False)
        runtime_identity(profile, check_loaded=False)
        evidence_root = output / "evidence" if args.diagnostics else None
        sessions = load_sessions(profile, evidence_root=evidence_root)
        report["runtime"] = runtime_identity(profile, check_loaded=False)
        audit = ReferenceAudit(request_data, profile["dtype"], raw_bits=args.raw_bits)
        report["audit_progress"] = audit.metrics
        frames, _ = request(
            sessions, config, prompt, golden["attempted_frames"], observer=audit
        )
        report["audit"] = audit.finish()
        del report["audit_progress"]
        np.testing.assert_array_equal(frames, golden["generated_frames"])
        waveform, _ = decode_waveform(sessions["codec"], frames)
        report["runtime_after_audit"] = runtime_identity(profile)
        report["generated_frames_sha256"] = hashlib.sha256(frames.tobytes()).hexdigest()
        report["waveform_sha256"] = hashlib.sha256(waveform.tobytes()).hexdigest()
        np.save(output / "generated-frames.npy", frames, allow_pickle=False)
        np.save(output / "waveform.npy", waveform, allow_pickle=False)
        if args.benchmark:
            for _ in range(args.warmup):
                warm_frames, _ = request(sessions, config, prompt, golden["attempted_frames"])
                decode_waveform(sessions["codec"], warm_frames)
            if profile["device"] == "cuda":
                snapshots.append(gpu_snapshot(profile["device_id"]))
                validate_ownership(snapshots, profile["gpu_uuid"])
            measured, timing = request(sessions, config, prompt, golden["attempted_frames"])
            _, codec_seconds = decode_waveform(sessions["codec"], measured)
            if profile["device"] == "cuda":
                snapshots.append(gpu_snapshot(profile["device_id"]))
            np.testing.assert_array_equal(measured, frames)
            report["performance"] = performance_result(
                timing,
                codec_seconds,
                measured,
                snapshots,
                profile,
                report["audit"],
                instrumented=args.diagnostics,
            )
            report["runtime_after_measurement"] = runtime_identity(profile)
            if report["performance"]["meets_end_to_end_rtf_target"] is False:
                fail_rtf()
        verify_artifacts(profile)
        if args.diagnostics:
            report["diagnostics"] = collect_evidence(sessions, evidence_root)
        report["status"] = "passed"
        finish_acceptance(report, benchmark=args.benchmark)
    except (ValueError, RuntimeError, AssertionError, OSError, FloatingPointError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    return report


def finish_acceptance(report, *, benchmark):
    """Keep passing neural/timing checks separate from the missing codec quality gate."""
    report["native_cuda13_neural_and_performance_gates_passed"] = (
        benchmark
        and report["performance"]["measurement_qualified"]
        and report["performance"]["meets_end_to_end_rtf_target"] is True
    )
    # This consumer has no independent waveform-reference comparison contract.
    # Neither a selected graph hash nor successful codec execution closes it.
    report["native_cuda13_acceptance"] = False
    report["blocked_acceptance_gates"] = ["independent_fp32_codec_waveform_numerical_quality"]


def fail_rtf():
    raise RuntimeError("Unchanged end-to-end RTF <= 1.0 gate failed")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--inputs", type=Path, required=True)
    result.add_argument("--inputs-sha256", required=True)
    result.add_argument("--selection", type=Path, required=True)
    result.add_argument("--selection-sha256", required=True)
    result.add_argument("--output", type=Path, required=True, help="New, separate output root")
    result.add_argument("--raw-bits", action="store_true")
    result.add_argument("--benchmark", action="store_true")
    result.add_argument("--warmup", type=int, default=2)
    result.add_argument("--lease-uuid")
    result.add_argument(
        "--diagnostics",
        action="store_true",
        help="Save optimized graphs and opcode profiles; timing is unqualified",
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
