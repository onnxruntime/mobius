# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Repository-local Drama 3 execution helpers without importing the model builder.

Only NumPy, ml_dtypes and ORT are needed in the target process. Source export and
policy construction remain separate actions in the existing tools.
"""

from __future__ import annotations

import csv
import ctypes
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np
import onnxruntime as ort

# These scripts run from a source checkout. Loading this pure validator directly
# avoids executing mobius.__init__ (and its builder/framework dependencies).
_context_path = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mobius"
    / "generation"
    / "_fish_omni_context.py"
)
_spec = importlib.util.spec_from_file_location("_drama3_context", _context_path)
assert _spec is not None and _spec.loader is not None
_context = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_context)
SOURCE_CONTEXT_CONTRACT = _context.SOURCE_CONTEXT_CONTRACT
SOURCE_CONTEXT_METADATA = _context.SOURCE_CONTEXT_METADATA
validate_source_context = _context.validate_source_context


def numpy_to_ort(value):
    """Preserve raw BF16 storage, scalar rank, empty dimensions and ownership."""
    value = np.ascontiguousarray(value) if value.ndim else value
    if value.dtype == np.dtype(ml_dtypes.bfloat16):
        return ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
            value.view(np.uint16), onnx_element_type=16
        )
    return ort.OrtValue.ortvalue_from_numpy(value)


def ort_to_numpy(value):
    """Copy CPU BF16 storage without Torch/DLPack or dereferencing a GPU pointer."""
    if value.data_type() != "tensor(bfloat16)":
        return value.numpy()
    if value.device_name() != "cpu":
        raise ValueError("BF16 inspection requires an explicitly CPU-bound output")
    shape = tuple(value.shape())
    size = int(np.prod(shape, dtype=np.int64))
    if not size:
        return np.empty(shape, dtype=ml_dtypes.bfloat16)
    if not value.data_ptr():
        raise RuntimeError("ORT returned a null pointer for nonempty BF16 storage")
    raw = ctypes.string_at(value.data_ptr(), size * 2)
    return np.frombuffer(raw, dtype=ml_dtypes.bfloat16).copy().reshape(shape)


def empty_cache_ports(ports):
    caches = {}
    for port in ports:
        if not port.name.startswith("past_key_values."):
            continue
        shape = list(port.shape)
        shape[0] = 1
        if port.name.endswith((".key", ".value")):
            if len(shape) != 4:
                raise ValueError(f"Expected rank-four KV cache: {port.name}")
            shape[2] = 0
        if not all(isinstance(dim, int) and dim >= 0 for dim in shape):
            raise AssertionError(f"Unresolved cache dimensions: {port.name}, {shape}")
        dtypes = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(bfloat16)": ml_dtypes.bfloat16,
        }
        type_name = getattr(port, "type", "tensor(float)")
        if type_name not in dtypes:
            raise ValueError(f"Unsupported cache dtype {type_name}: {port.name}")
        caches[port.name] = np.zeros(shape, dtype=dtypes[type_name])
    return caches


@dataclass(frozen=True)
class TensorView:
    """Keep the owning OrtValue alive while binding a zero-copy contiguous slice."""

    owner: ort.OrtValue
    shape: tuple[int, ...]
    offset: int = 0


def sequence_view(value, start, length, *, squeeze_time=False):
    shape = value.shape()
    if len(shape) != 3 or shape[0] != 1 or not 0 <= start < start + length <= shape[1]:
        raise ValueError(f"Invalid batch-one contiguous sequence slice: {shape}")
    itemsize = {1: 4, 10: 2, 16: 2}[value.element_type()]
    target = (1, shape[2]) if squeeze_time else (1, length, shape[2])
    if squeeze_time and length != 1:
        raise ValueError("Only a length-one sequence can lose its time axis")
    return TensorView(value, target, start * shape[2] * itemsize)


def codebook_view(value, index):
    shape = value.shape()
    if len(shape) != 4 or shape[:2] != [1, 1] or not 0 <= index < shape[2]:
        raise ValueError(f"Invalid single-frame codebook view: {shape}, index={index}")
    itemsize = {1: 4, 10: 2, 16: 2}[value.element_type()]
    return TensorView(value, (1, shape[3]), index * shape[3] * itemsize)


class BoundSession:
    """Bind caches on-device and explicitly select CPU outputs for inspection."""

    def __init__(
        self,
        path,
        *,
        device,
        device_id=0,
        profile=False,
        profile_directory=None,
        cudnn_conv1d_pad_to_nc1d=False,
        cudnn_conv_algo_search="EXHAUSTIVE",
        selection=None,
        evidence_directory=None,
    ):
        if cudnn_conv_algo_search not in ("EXHAUSTIVE", "HEURISTIC", "DEFAULT"):
            raise ValueError(f"Unknown cuDNN convolution search: {cudnn_conv_algo_search!r}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_mem_pattern = False
        options.enable_profiling = profile
        if evidence_directory is not None:
            evidence_directory.mkdir(parents=True, exist_ok=False)
            options.optimized_model_filepath = str(evidence_directory / "optimized.onnx")
            options.enable_profiling = True
            options.profile_file_prefix = str(evidence_directory / "execution")
        if profile:
            profile_directory = profile_directory or Path(tempfile.gettempdir())
            profile_directory.mkdir(parents=True, exist_ok=True)
            suffix = hashlib.sha256(str(path).encode()).hexdigest()[:12]
            options.profile_file_prefix = str(
                profile_directory / f"{path.parent.name}-{suffix}"
            )
        providers = (
            [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": device_id,
                        "use_tf32": "0",
                        "arena_extend_strategy": "kSameAsRequested",
                        "cudnn_conv1d_pad_to_nc1d": str(int(cudnn_conv1d_pad_to_nc1d)),
                        "cudnn_conv_algo_search": cudnn_conv_algo_search,
                    },
                ),
                "CPUExecutionProvider",
            ]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        disabled = set()
        if selection is not None:
            if profile:
                raise ValueError(
                    "Selected qualification profile cannot enable instrumentation"
                )
            providers = [
                (item[0], item[1]) if isinstance(item, list) else item
                for item in selection["providers"]
            ]
            disabled = set(selection["disabled_optimizers"])
            for key, value in selection["session_config"].items():
                options.add_session_config_entry(key, value)
        self.session = ort.InferenceSession(
            str(path),
            options,
            providers=providers,
            **({"disabled_optimizers": disabled} if selection is not None else {}),
        )
        if selection is not None:
            if self.session.get_providers() != selection["actual_providers"]:
                raise RuntimeError("ORT provider fallback differs from selected profile")
            if self.session.get_provider_options() != selection["actual_provider_options"]:
                raise RuntimeError("Actual provider options differ from selected profile")
        self.device, self.device_id = device, device_id if device == "cuda" else 0
        self.outputs = [port.name for port in self.session.get_outputs()]
        self.context_contract = self.session.get_modelmeta().custom_metadata_map.get(
            SOURCE_CONTEXT_METADATA
        )
        if self.context_contract not in (None, SOURCE_CONTEXT_CONTRACT):
            raise ValueError(f"Unknown Fish context contract: {self.context_contract}")

    def run(self, feeds, *, fetches=None, cpu_outputs=()):
        context_input = (
            "inputs_embeds"
            if "inputs_embeds" in feeds
            else "hidden_states"
            if "hidden_states" in feeds
            else None
        )
        # Candidate package metadata also appears on the stateless embedding
        # graph. Only decoder/tower calls consume the position/cache contract.
        if self.context_contract == SOURCE_CONTEXT_CONTRACT and context_input is not None:

            def shape(value):
                if isinstance(value, TensorView):
                    return value.shape
                return value.shape() if isinstance(value, ort.OrtValue) else value.shape

            if not all(
                isinstance(feeds[name], np.ndarray)
                for name in ("position_ids", "attention_mask")
            ):
                raise ValueError("Source Fish context controls must be CPU NumPy arrays")
            validate_source_context(
                feeds["position_ids"],
                feeds["attention_mask"],
                shape(feeds[context_input]),
                {
                    name: shape(value)
                    for name, value in feeds.items()
                    if name.startswith("past_key_values.")
                },
            )
        names = self.outputs if fetches is None else list(fetches)
        if not set(names) <= set(self.outputs):
            raise ValueError(f"Unknown output names: {set(names) - set(self.outputs)}")
        binding = self.session.io_binding()
        owners = {}
        for name, value in feeds.items():
            if isinstance(value, TensorView):
                owners[name] = value.owner
                owner_device = value.owner.device_name()
                binding.bind_input(
                    name,
                    device_type=owner_device,
                    device_id=self.device_id if owner_device == "cuda" else 0,
                    element_type=value.owner.element_type(),
                    shape=value.shape,
                    buffer_ptr=value.owner.data_ptr() + value.offset,
                )
            else:
                owners[name] = (
                    value if isinstance(value, ort.OrtValue) else numpy_to_ort(value)
                )
                binding.bind_ortvalue_input(name, owners[name])
        for name in names:
            target = "cpu" if name in cpu_outputs else self.device
            binding.bind_output(
                name, device_type=target, device_id=self.device_id if target == "cuda" else 0
            )
        binding.synchronize_inputs()
        self.session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        values = binding.get_outputs()
        if len(values) != len(names):
            raise RuntimeError("ORT returned an unexpected bound-output count")
        return dict(zip(names, values))

    def empty_cache(self):
        return empty_cache_ports(self.session.get_inputs())


def next_cache(outputs, device):
    result = {
        name.replace("present.", "past_key_values.", 1): value
        for name, value in outputs.items()
        if name.startswith("present.")
    }
    if not result or any(value.device_name() != device for value in result.values()):
        raise RuntimeError("Recurrent/KV state did not remain on the execution device")
    return result


class GpuSnapshotError(RuntimeError):
    """Unknown GPU health with raw query and bounded owned-process cleanup evidence."""

    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def _wait_gpu_query(process, command, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(command, 20)
    process.wait(timeout=remaining)


def _gpu_query(command):
    evidence = {
        "command": command,
        "status": "UNKNOWN",
        "query_deadline_seconds": 20,
        "cleanup_grace_seconds": 2,
        "owned_pid": None,
        "returncode": None,
        "reaped": False,
        "timed_out": False,
        "stdout": "",
        "stderr": "",
    }
    # Files avoid unbounded pipe draining or reader threads when an NVML child
    # hangs. Inspect only bounded raw output after the bounded wait/cleanup.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        started = time.monotonic()
        deadline = started + 20
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        except OSError as error:
            evidence["error"] = f"{type(error).__name__}: {error}"
            evidence["elapsed_seconds"] = time.monotonic() - started
            raise GpuSnapshotError(
                "GPU query could not start; health UNKNOWN", evidence
            ) from error
        evidence["owned_pid"] = process.pid
        failure = None
        try:
            _wait_gpu_query(process, command, deadline)
            evidence["reaped"] = True
        except (subprocess.TimeoutExpired, OSError) as error:
            failure = error
            evidence["timed_out"] = isinstance(error, subprocess.TimeoutExpired)
            evidence["error"] = f"{type(error).__name__}: {error}"
            cleanup_started = time.monotonic()
            evidence["kill_attempted"] = True
            try:
                # Only this query's Popen child is killed; no global/name kill.
                process.kill()
            except OSError as cleanup_error:
                evidence["kill_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
            try:
                process.wait(timeout=max(0.0, 2 - (time.monotonic() - cleanup_started)))
                evidence["reaped"] = True
            except (subprocess.TimeoutExpired, OSError) as cleanup_error:
                evidence["reap_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
        evidence["returncode"] = process.returncode
        evidence["elapsed_seconds"] = time.monotonic() - started
        for name, stream in (("stdout", stdout), ("stderr", stderr)):
            stream.seek(0)
            raw = stream.read(1024 * 1024 + 1)
            evidence[name] = raw[: 1024 * 1024].decode("utf-8", errors="replace")
            evidence[name + "_truncated"] = len(raw) > 1024 * 1024
        if failure is not None or process.returncode != 0:
            raise GpuSnapshotError("GPU query failed; health UNKNOWN", evidence) from failure
        if evidence["stdout_truncated"] or evidence["stderr_truncated"]:
            raise GpuSnapshotError("GPU query output exceeded limit; health UNKNOWN", evidence)
    evidence["status"] = "OBSERVED"
    return evidence


def _snapshot_rows(text, columns):
    rows = list(csv.reader(io.StringIO(text), strict=True, skipinitialspace=True))
    if any(len(row) != columns or any(not item.strip() for item in row) for row in rows):
        raise ValueError("Malformed GPU query CSV")
    return [[item.strip() for item in row] for row in rows]


def _collect_gpu_snapshot(device_id, queries):
    if not isinstance(device_id, int) or device_id < 0:
        raise ValueError("Invalid CUDA device ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected = visible.split(",")[device_id].strip() if visible else str(device_id)
    if not selected or (not selected.isdigit() and not selected.startswith("GPU-")):
        raise ValueError("Invalid CUDA visible-device selector")
    queries.append(
        _gpu_query(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
    )
    queries.append(
        _gpu_query(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ]
        )
    )
    rows = _snapshot_rows(queries[0]["stdout"], 5)
    processes = _snapshot_rows(queries[1]["stdout"], 2)
    matches = []
    seen_indices, seen_uuids = set(), set()
    for row in rows:
        index, _name, uuid, memory, utilization = row
        if (
            not index.isdigit()
            or not memory.isdigit()
            or not utilization.isdigit()
            or int(utilization) > 100
            or not uuid.startswith("GPU-")
            or int(index) in seen_indices
            or uuid in seen_uuids
        ):
            raise ValueError("Invalid GPU identity or health data")
        seen_indices.add(int(index))
        seen_uuids.add(uuid)
        if (selected.isdigit() and int(index) == int(selected)) or (
            not selected.isdigit() and uuid.startswith(selected)
        ):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError("Selected GPU missing or UUID selector ambiguous")
    if any(
        not uuid.startswith("GPU-") or not pid.isdigit() or int(pid) <= 0
        for uuid, pid in processes
    ):
        raise ValueError("Invalid GPU process ownership data")
    row = matches[0]
    return {
        "status": "OBSERVED",
        "device_id": device_id,
        "physical_index": int(row[0]),
        "uuid": row[2],
        "name": row[1],
        "memory_used_mib": int(row[3]),
        "utilization_percent": int(row[4]),
        "foreign_pids": [
            int(pid) for uuid, pid in processes if uuid == row[2] and int(pid) != os.getpid()
        ],
        "queries": queries,
    }


def gpu_snapshot(device_id):
    queries = []
    try:
        return _collect_gpu_snapshot(device_id, queries)
    except (GpuSnapshotError, ValueError, IndexError, csv.Error, OSError) as error:
        evidence = {
            "status": "UNKNOWN",
            "device_id": device_id,
            "queries": queries,
            "error": f"{type(error).__name__}: {error}",
        }
        if isinstance(error, GpuSnapshotError):
            evidence["failed_query"] = error.evidence
        print(json.dumps({"event": "gpu_snapshot_error", **evidence}), flush=True)
        raise GpuSnapshotError(
            "GPU health/ownership UNKNOWN; refusing measurement", evidence
        ) from error


def require_idle(snapshot, allow_contended):
    if snapshot.get("status") == "UNKNOWN":
        raise GpuSnapshotError("GPU health UNKNOWN; refusing measurement", snapshot)
    busy = (
        snapshot["utilization_percent"] > 5
        or snapshot["memory_used_mib"] > 1024
        or bool(snapshot.get("foreign_pids"))
    )
    if busy and not allow_contended:
        raise RuntimeError(
            "GPU is occupied; refusing misleading performance measurements. "
            "Use an idle device, or --allow-contended for an explicitly unqualified diagnostic."
        )
    return not busy
