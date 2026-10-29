# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic files and maps only: no ORT import, sessions, CUDA or hardware queries."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import drama3_library_identity as identity
import pytest


def binary(root, name, data=b"CPU fixture, not an ELF"):
    path = root / name
    path.write_bytes(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def rows(*files):
    return "\n".join(f"1-2 r-xp 00000000 00:00 0 {proc_path(item['path'])}" for item in files)


def proc_path(value):
    value = str(value)
    return value if value.startswith("/") else "/" + value


@pytest.fixture(autouse=True)
def synthetic_linux_paths_on_windows(monkeypatch):
    if sys.platform == "win32":
        original = identity._resolve
        monkeypatch.setattr(
            identity,
            "_resolve",
            lambda value: original(str(value)[1:] if str(value).startswith("/C:") else value),
        )


@pytest.fixture
def inventory(tmp_path):
    return (
        binary(tmp_path, "libcudart.so.13"),
        binary(tmp_path, "onnxruntime_pybind11_state.so"),
    )


def test_known_maps_observe_hashes_and_all_segments(inventory):
    cuda, ort = inventory
    result = identity.observe_loaded_libraries(
        [cuda],
        [ort],
        maps_text=rows(cuda, ort, ort),
    )
    assert result["status"] == "PASSED"
    assert len(result["observed_mappings"]) == 2
    observed = result["observed_mappings"][1]
    assert observed["observed_sha256"] == ort["sha256"]
    assert len(observed["map_lines"]) == 2
    assert observed["match_decision"] == "MATCH"
    assert observed["hash_scope"] == "resolved_file_bytes_not_resident_memory"
    assert result["expected_sources"]["ort"][0]["source_field"] == "runtime.files"


@pytest.mark.parametrize(
    "name",
    [
        "onnxruntime_unexpected_binding.so",
        "libonnxruntime_providers_unexpected.so",
        "libonnxruntime.so.1.32.0",
        "libcublas_unknown.so.13",
        "libtorch_cuda.so",
    ],
)
def test_unknown_dso_retains_exact_offender_and_later_maps(tmp_path, inventory, name):
    cuda, ort = inventory
    unknown = binary(tmp_path, name)
    with pytest.raises(identity.LibraryIdentityError) as caught:
        identity.observe_loaded_libraries(
            [cuda],
            [ort],
            maps_text=rows(unknown, cuda, ort),
        )
    result = caught.value.evidence
    assert result["status"] == "FAILED"
    assert len(result["observed_mappings"]) == 3
    offender = result["observed_mappings"][0]
    assert offender["mapped_path"] == proc_path(unknown["path"])
    assert offender["resolved_path"] == str(Path(unknown["path"]).resolve())
    assert offender["library_basename"] == name
    assert offender["map_lines"] == [rows(unknown)]
    assert offender["observed_sha256"] == unknown["sha256"]
    assert offender["match_decision"] == "REJECT"
    assert result["observed_mappings"][-1]["match_decision"] == "MATCH"
    assert result["expected_sources"]["ort"][0]["declared_path"] == ort["path"]
    assert unknown["path"] in str(caught.value)
    json.dumps(result)


def test_same_hash_different_unapproved_path_still_fails(tmp_path, inventory):
    cuda, ort = inventory
    alias = binary(tmp_path, "onnxruntime_other_path.so")
    assert alias["sha256"] == ort["sha256"]
    with pytest.raises(identity.LibraryIdentityError) as caught:
        identity.observe_loaded_libraries([cuda], [ort], maps_text=rows(cuda, alias))
    offender = caught.value.evidence["observed_mappings"][1]
    assert offender["reasons"] == ["UNAPPROVED_RESOLVED_PATH"]
    assert offender["same_hash_expected_other_paths"] == [str(Path(ort["path"]).resolve())]


def test_canonical_alias_keeps_raw_path_and_existing_resolution_policy(
    monkeypatch,
    tmp_path,
    inventory,
):
    cuda, ort = inventory
    alias = tmp_path / "neutral_alias.so"
    real_resolve = identity._resolve
    monkeypatch.setattr(
        identity,
        "_resolve",
        lambda value: real_resolve(ort["path"] if str(value) == proc_path(alias) else value),
    )
    result = identity.observe_loaded_libraries(
        [cuda],
        [ort],
        maps_text=rows(cuda, {"path": str(alias)}),
    )
    record = result["observed_mappings"][1]
    assert record["mapped_path"] == proc_path(alias)
    assert record["resolved_path"] == str(real_resolve(ort["path"]))
    assert record["path_alias_resolved"] is True
    assert record["resolved_library_basename"] == Path(ort["path"]).name


def test_hash_mismatch_retains_actual_not_expected(inventory):
    cuda, ort = inventory
    wrong = {**ort, "sha256": "0" * 64}
    with pytest.raises(identity.LibraryIdentityError) as caught:
        identity.observe_loaded_libraries([cuda], [wrong], maps_text=rows(cuda, ort))
    record = caught.value.evidence["observed_mappings"][1]
    assert record["observed_sha256"] == ort["sha256"]
    assert record["expected_sha256"] == "0" * 64
    assert record["reasons"] == ["OBSERVED_HASH_MISMATCH"]


@pytest.mark.parametrize("failure", ["missing", "deleted", "hash-read-error", "resolve-error"])
def test_unavailable_observation_has_no_synthesized_hash(
    tmp_path,
    monkeypatch,
    inventory,
    failure,
):
    cuda, ort = inventory
    unknown = binary(tmp_path, "onnxruntime_unavailable.so")
    if failure == "missing":
        Path(unknown["path"]).unlink()
    elif failure == "deleted":
        unknown["path"] += " (deleted)"
    elif failure == "hash-read-error":
        original = identity._digest

        def fail(path):
            if str(path) == unknown["path"]:
                raise PermissionError("CPU fixture denied read")
            return original(path)

        monkeypatch.setattr(identity, "_digest", fail)
    else:
        original = identity._resolve

        def fail(path):
            if str(path) == proc_path(unknown["path"]):
                raise RuntimeError("CPU fixture symlink loop")
            return original(path)

        monkeypatch.setattr(identity, "_resolve", fail)
    with pytest.raises(identity.LibraryIdentityError) as caught:
        identity.observe_loaded_libraries(
            [cuda],
            [ort],
            maps_text=rows(unknown, cuda, ort),
        )
    records = caught.value.evidence["observed_mappings"]
    assert "observed_sha256" not in records[0]
    assert records[-1]["match_decision"] == "MATCH"


def test_maps_getter_failure_preserves_expected_sources(inventory):
    cuda, ort = inventory

    class FailedMaps:
        def read_text(self):
            raise PermissionError("CPU mock procfs unavailable")

    with pytest.raises(identity.LibraryIdentityError) as caught:
        identity.observe_loaded_libraries([cuda], [ort], maps_path=FailedMaps())
    result = caught.value.evidence
    assert result["observed_mappings"] == []
    assert result["expected_sources"]["ort"][0]["expected_sha256"] == ort["sha256"]
    assert result["violations"][0]["reason"] == "MAPS_UNAVAILABLE"


def test_fresh_process_import_and_mock_mapping_with_frameworks_blocked(tmp_path):
    script = r"""
import importlib.abc, pathlib, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'mobius', 'onnxruntime', 'onnxscript', 'onnx_ir', 'numpy'
        }:
            raise ImportError('Blocked CPU isolation fixture: ' + fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
from drama3_library_identity import observe_loaded_libraries
import drama3_library_identity as identity
if sys.platform == 'win32':
    original = identity._resolve
    identity._resolve = lambda value: original(str(value)[1:] if str(value).startswith('/C:') else value)
import hashlib
root = pathlib.Path(sys.argv[2])
items = []
for name in ['libcudart.so.13', 'onnxruntime_pybind11_state.so']:
    path = root / name
    path.write_bytes(b'CPU fixture only')
    items.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
maps = '\n'.join('1-2 r-xp 0 0 0 ' + ('' if item['path'].startswith('/') else '/') + item['path'] for item in items)
assert observe_loaded_libraries(items[:1], items[1:], maps_text=maps)['status'] == 'PASSED'
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            script,
            str(Path(identity.__file__).parent),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_actual_consumer_error_writer_retains_maps_and_all_five_profiles_without_nn(
    tmp_path,
    inventory,
):
    """Execute the published run AST with CPU doubles, without importing its runtime."""
    cuda, ort = inventory
    unknown = binary(tmp_path, "onnxruntime_foreign.so")
    with pytest.raises(identity.LibraryIdentityError) as caught:
        identity.observe_loaded_libraries(
            [cuda],
            [ort],
            maps_text=rows(unknown, cuda, ort),
        )
    failure = caught.value
    root = tmp_path / "readonly"
    root.mkdir()
    config = root / "config.json"
    config.write_text("{}")
    names = {"embedding", "decoder", "audio_generator", "main_policy", "codec"}
    actual = {
        "actual_provider_options": {"CPU-MOCK": {"full": "retained"}},
        "actual_session_options": {"CPU-MOCK": "retained"},
    }
    profile = {
        "inputs_sha256": "CPU-MOCK",
        "dtype": "bf16",
        "expert_execution": "dense",
        "config": {},
        "max_total_length": 100,
        "device": "cuda",
        "device_id": 0,
        "gpu_uuid": "GPU-SYNTHETIC-NOT-A-GRANT",
        "components": {},
    }
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("NN")
        raise AssertionError("Identity failure must precede every neural/codec call")

    def attest(sessions, profile, *, diagnostics, evidence):
        evidence.update({name: {"status": "ATTESTED", "actual": actual} for name in names})

    def runtime(profile, *, check_loaded=True):
        if check_loaded:
            raise failure
        return {}

    class FakeGpuError(RuntimeError):
        pass

    scope = {
        "Path": Path,
        "json": json,
        "time": time,
        "checked_json": lambda *args: profile,
        "validate_profile": lambda *args: None,
        "checked_file": lambda *args: config,
        "verify_artifacts": lambda *args: None,
        "load_waveform_reference": lambda *args: None,
        "load_inputs": lambda *args: (
            {"checkpoint_revision": "CPU-MOCK", "source_revision": "CPU-MOCK"},
            {"golden_path": str(root / "golden.json"), "states": []},
            {"expert_execution": "dense", "attempted_frames": 1, "reference_environment": {}},
            [30],
        ),
        "gpu_snapshot": lambda *args: {},
        "validate_ownership": lambda *args: None,
        "require_idle": lambda *args: None,
        "runtime_identity": runtime,
        "load_sessions": lambda *args, **kwargs: dict.fromkeys(names),
        "attest_loaded_sessions": attest,
        "request": forbidden,
        "decode_waveform": forbidden,
        "ReferenceAudit": forbidden,
        "GpuSnapshotError": FakeGpuError,
        "LibraryIdentityError": identity.LibraryIdentityError,
    }
    source = ast.parse((Path(identity.__file__).parent / "audit_drama3.py").read_text())
    node = next(
        item
        for item in source.body
        if isinstance(item, ast.FunctionDef) and item.name == "run"
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "actual-audit-run-AST", "exec"),
        scope,
    )
    args = SimpleNamespace(
        selection=root / "selection.json",
        selection_sha256="CPU-MOCK",
        inputs=root / "inputs.json",
        inputs_sha256="CPU-MOCK",
        warmup=1,
        output=tmp_path / "result",
        lease_uuid=profile["gpu_uuid"],
        diagnostics=False,
    )
    with pytest.raises(identity.LibraryIdentityError):
        scope["run"](args)
    report = json.loads((args.output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["native_cuda13_acceptance"] is False
    assert report["loaded_library_identity_error"] == failure.evidence
    assert set(report["actual_model_session_profiles_before_first_nn"]) == names
    assert report["actual_model_session_profiles_before_first_nn"]["codec"]["actual"] == actual
    assert calls == []


@pytest.mark.parametrize("unknown", [False, True])
def test_actual_runtime_identity_ast_uses_shared_observations_without_ort_import(
    tmp_path,
    monkeypatch,
    inventory,
    unknown,
):
    cuda, native_pin = inventory
    files = [cuda, native_pin]
    if unknown:
        files.insert(0, binary(tmp_path, "libonnxruntime_unpinned_provider.so"))
    text = rows(*files)
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.__path__ = []
    fake_ort.__version__ = "CPU-MOCK"
    fake_ort.get_build_info = lambda: "CPU-MOCK"
    capi = types.ModuleType("onnxruntime.capi")
    capi.__path__ = []
    native = types.ModuleType("onnxruntime.capi.onnxruntime_pybind11_state")
    native.__file__ = native_pin["path"]
    fake_ort.capi = capi
    capi.onnxruntime_pybind11_state = native
    for name, value in {
        "onnxruntime": fake_ort,
        "onnxruntime.capi": capi,
        "onnxruntime.capi.onnxruntime_pybind11_state": native,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    profile = {
        "device": "cuda",
        "runtime": {
            "ort_version": "CPU-MOCK",
            "ort_build": "CPU-MOCK",
            "files": [native_pin],
            "libraries": [cuda],
            "wheel": binary(tmp_path, "CPU-MOCK-wheel.whl"),
        },
    }

    def path(value):
        if str(value) == "/proc/self/maps":
            return SimpleNamespace(read_text=lambda: text)
        return Path(value)

    scope = {
        "Path": path,
        "ort": fake_ort,
        "sys": sys,
        "digest": identity._digest,
        "observe_loaded_libraries": identity.observe_loaded_libraries,
    }
    source = ast.parse((Path(identity.__file__).parent / "audit_drama3.py").read_text())
    nodes = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"checked_file", "runtime_identity"}
    ]
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), "actual-runtime-identity-AST", "exec"
        ),
        scope,
    )
    if unknown:
        with pytest.raises(identity.LibraryIdentityError) as caught:
            scope["runtime_identity"](profile)
        records = caught.value.evidence["observed_mappings"]
        assert records[0]["mapped_path"] == proc_path(files[0]["path"])
        assert records[-1]["match_decision"] == "MATCH"
    else:
        actual = scope["runtime_identity"](profile)
        assert actual["loaded_library_evidence"]["status"] == "PASSED"
        assert actual["actual_loaded_cuda_libraries"] == [
            {"path": str(Path(cuda["path"]).resolve()), "sha256": cuda["sha256"]}
        ]
