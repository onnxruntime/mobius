# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic files and maps only: no ORT import, sessions, CUDA or hardware queries."""

from __future__ import annotations

import ast
import hashlib
import json
import struct
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import drama3_library_identity as identity
import pytest


def elf_header():
    data = bytearray(64)
    data[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<H", data, 16, 3)
    struct.pack_into("<H", data, 18, 62)
    struct.pack_into("<I", data, 20, 1)
    struct.pack_into("<H", data, 52, 64)
    return bytes(data)


def binary(root, name, data=None):
    if data is None:
        data = elf_header()
    path = root / name
    path.write_bytes(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def rows(*files, permissions="r-xp", inode=0):
    return "\n".join(
        f"1-2 {permissions} 00000000 00:00 {inode} {proc_path(item['path'])}" for item in files
    )


def proc_path(value):
    value = str(value)
    return value if value.startswith("/") else "/" + value


def _synthetic_linux_paths_on_windows(monkeypatch):
    if sys.platform == "win32":
        original = identity._resolve
        monkeypatch.setattr(
            identity,
            "_resolve",
            lambda value: original(str(value)[1:] if str(value).startswith("/C:") else value),
        )


@pytest.fixture(autouse=True)
def synthetic_linux_paths_on_windows(monkeypatch):
    _synthetic_linux_paths_on_windows(monkeypatch)


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


@pytest.mark.parametrize(
    "name", ["onnxruntime.db-shm", "onnxruntime_other_data", "neutral_data"]
)
def test_readable_nonexec_nonelf_data_excluded_by_type_not_a_cache_path_rule(
    tmp_path,
    inventory,
    name,
):
    cuda, ort = inventory
    data = binary(tmp_path, name, b"CPU synthetic mapped data".ljust(128, b"\0"))
    result = identity.observe_loaded_libraries(
        [cuda],
        [ort],
        maps_text=rows(cuda, ort) + "\n" + rows(data, permissions="rw-s"),
    )
    record = result["observed_mappings"][-1]
    assert record["match_decision"] == "EXCLUDED_NON_NATIVE_DATA"
    assert record["reasons"] == ["EXCLUDED_NON_NATIVE_DATA"]
    assert record["observed_header"]["file_type"] == "NON_ELF"
    assert record["observed_header"]["bytes_read"] == 64
    assert record["object_has_executable_segment"] is False
    assert record["native_candidate"] is False
    assert record["observed_sha256"] == data["sha256"]
    assert result["expected_sources"]["ort"][0]["expected_sha256"] == ort["sha256"]


@pytest.mark.parametrize("execute_first", [False, True])
def test_segment_grouping_never_excludes_nonexec_row_before_exec_row(
    tmp_path,
    inventory,
    execute_first,
):
    cuda, ort = inventory
    data = binary(tmp_path, "onnxruntime_disguised_data", b"CPU data".ljust(128, b"\0"))
    nonexec = rows(data, permissions="rw-s")
    execute = rows(data, permissions="r-xp")
    segments = [execute, nonexec] if execute_first else [nonexec, execute]
    with pytest.raises(identity.LibraryIdentityError) as caught:
        identity.observe_loaded_libraries(
            [cuda],
            [ort],
            maps_text=rows(cuda, ort) + "\n" + "\n".join(segments),
        )
    record = caught.value.evidence["observed_mappings"][-1]
    assert record["object_has_executable_segment"] is True
    assert len(record["map_lines"]) == 2
    assert record["match_decision"] == "REJECT"
    assert "NON_ELF_NATIVE_CANDIDATE" in record["reasons"]


def test_device_inode_groups_differently_named_segments_before_classification(
    tmp_path,
    inventory,
):
    cuda, ort = inventory
    first = binary(tmp_path, "onnxruntime_data", b"CPU data".ljust(128, b"\0"))
    second = binary(tmp_path, "neutral_alias", b"CPU data".ljust(128, b"\0"))
    with pytest.raises(identity.LibraryIdentityError) as caught:
        identity.observe_loaded_libraries(
            [cuda],
            [ort],
            maps_text=rows(cuda, ort)
            + "\n"
            + rows(first, permissions="rw-s", inode=987)
            + "\n"
            + rows(second, permissions="r-xp", inode=987),
        )
    first_record, second_record = caught.value.evidence["observed_mappings"][-2:]
    assert first_record["object_id"] == second_record["object_id"]
    assert first_record["object_has_executable_segment"] is True
    assert first_record["match_decision"] == second_record["match_decision"] == "REJECT"


@pytest.mark.parametrize(
    "name,contents,permissions",
    [
        ("neutral_extension.cache", None, "r--p"),
        ("onnxruntime.db-shm", None, "rw-s"),
        ("onnxruntime_candidate.so.42", b"CPU not ELF".ljust(128, b"\0"), "r--p"),
    ],
)
def test_disguised_elf_and_versioned_shared_library_candidates_still_reject(
    tmp_path,
    inventory,
    name,
    contents,
    permissions,
):
    cuda, ort = inventory
    unknown = binary(tmp_path, name, contents)
    with pytest.raises(identity.LibraryIdentityError) as caught:
        identity.observe_loaded_libraries(
            [cuda],
            [ort],
            maps_text=rows(cuda, ort) + "\n" + rows(unknown, permissions=permissions),
        )
    record = caught.value.evidence["observed_mappings"][-1]
    assert record["native_candidate"] is True
    assert record["match_decision"] == "REJECT"
    assert "UNAPPROVED_RESOLVED_PATH" in record["reasons"]


@pytest.mark.parametrize("failure", ["unreadable", "deleted", "short", "malformed-elf"])
def test_ambiguous_relevant_data_fails_instead_of_being_excluded(
    tmp_path,
    inventory,
    monkeypatch,
    failure,
):
    cuda, ort = inventory
    data = binary(tmp_path, "onnxruntime.data", b"CPU data".ljust(128, b"\0"))
    if failure == "unreadable":
        original = identity._header

        def denied(path):
            if str(path) == data["path"]:
                raise PermissionError("CPU mock unreadable header")
            return original(path)

        monkeypatch.setattr(identity, "_header", denied)
    elif failure == "deleted":
        data["path"] += " (deleted)"
    elif failure == "short":
        Path(data["path"]).write_bytes(b"short")
    else:
        Path(data["path"]).write_bytes(b"\x7fELF\xff".ljust(64, b"\0"))
    with pytest.raises(identity.LibraryIdentityError) as caught:
        identity.observe_loaded_libraries(
            [cuda],
            [ort],
            maps_text=rows(cuda, ort) + "\n" + rows(data, permissions="rw-s"),
        )
    record = caught.value.evidence["observed_mappings"][-1]
    assert record["match_decision"] == "REJECT"
    assert not any(reason == "EXCLUDED_NON_NATIVE_DATA" for reason in record["reasons"])
    if failure in ("deleted", "unreadable"):
        assert "observed_header" not in record


def test_unrelated_native_objects_are_outside_scope_not_validated_or_non_native(
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "unrelated"
    allowed.mkdir()
    outside.mkdir()
    cuda = binary(allowed, "libcudart.so.13")
    ort = binary(allowed, "onnxruntime_pybind11_state.so")
    unrelated = binary(outside, "libpython.so")
    original = identity._header

    def scoped_header(path):
        assert path != Path(unrelated["path"]), "Must not invent full-process closure"
        return original(path)

    monkeypatch.setattr(identity, "_header", scoped_header)
    result = identity.observe_loaded_libraries(
        [cuda],
        [ort],
        maps_text=rows(cuda, ort, unrelated),
    )
    assert len(result["observed_mappings"]) == 2
    record = result["outside_attested_scope"][0]
    assert record["match_decision"] == "OUTSIDE_ATTESTED_SCOPE"
    assert "observed_sha256" not in record and "observed_header" not in record


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
import hashlib, struct
root = pathlib.Path(sys.argv[2])
items = []
for name in ['libcudart.so.13', 'onnxruntime_pybind11_state.so']:
    path = root / name
    header = bytearray(64)
    header[:7] = b'\x7fELF\x02\x01\x01'
    struct.pack_into('<H', header, 16, 3)
    struct.pack_into('<H', header, 18, 62)
    struct.pack_into('<I', header, 20, 1)
    struct.pack_into('<H', header, 52, 64)
    path.write_bytes(header)
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


def _check_actual_runtime_identity_ast(
    tmp_path,
    monkeypatch,
    inventory,
    unknown,
    forbidden_module=None,
):
    cuda, native_pin = inventory
    files = [cuda, native_pin]
    if unknown is True:
        files.insert(0, binary(tmp_path, "libonnxruntime_unpinned_provider.so"))
    text = rows(*files)
    if unknown == "data":
        data = binary(tmp_path, "onnxruntime.db-shm", b"CPU data".ljust(128, b"\0"))
        text += "\n" + rows(data, permissions="rw-s")
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
    if forbidden_module is not None:
        monkeypatch.setitem(sys.modules, forbidden_module, types.ModuleType(forbidden_module))
        with pytest.raises(RuntimeError, match="Target process imported Torch"):
            scope["runtime_identity"](profile)
    elif unknown is True:
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
        if unknown == "data":
            assert str(Path(data["path"]).resolve()) not in actual["actual_loaded_ort_files"]
            assert (
                actual["loaded_library_evidence"]["observed_mappings"][-1]["match_decision"]
                == "EXCLUDED_NON_NATIVE_DATA"
            )


def _run_runtime_identity_ast(tmp_path, inventory, unknown, forbidden_module=None):
    script = r"""
import importlib.abc, json, pathlib, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'mobius', 'onnxruntime', 'onnxscript', 'onnx_ir', 'numpy'
        }:
            raise ImportError('Blocked identity fixture dependency: ' + fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
import drama3_library_identity_test as tests
assert not any(name.split('.')[0] in {'torch', 'mobius', 'onnxruntime'} for name in sys.modules)
with tests.pytest.MonkeyPatch.context() as monkeypatch:
    tests._synthetic_linux_paths_on_windows(monkeypatch)
    tests._check_actual_runtime_identity_ast(
        pathlib.Path(sys.argv[2]), monkeypatch, json.loads(sys.argv[3]),
        json.loads(sys.argv[4]), json.loads(sys.argv[5]),
    )
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            script,
            str(Path(__file__).parent),
            str(tmp_path),
            json.dumps(inventory),
            json.dumps(unknown),
            json.dumps(forbidden_module),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("unknown", [False, True, "data"])
def test_actual_runtime_identity_ast_uses_shared_observations_without_ort_import(
    tmp_path, inventory, unknown
):
    _run_runtime_identity_ast(tmp_path, inventory, unknown)


@pytest.mark.parametrize("forbidden_module", ["torch", "mobius._builder"])
def test_actual_runtime_identity_ast_rejects_framework_contamination(
    tmp_path, inventory, forbidden_module
):
    _run_runtime_identity_ast(tmp_path, inventory, False, forbidden_module)


def test_runtime_identity_cases_survive_framework_import_during_pytest_collection(tmp_path):
    collected = tmp_path / "builder_collection_test.py"
    collected.write_text(
        "import sys\n"
        "import torch\n"
        "import mobius\n"
        "assert 'torch' in sys.modules and 'mobius' in sys.modules\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "pytest",
            "-c",
            str(Path(__file__).resolve().parents[1] / "pyproject.toml"),
            str(collected),
            str(Path(__file__).resolve()),
            "-q",
            "--tb=short",
            "-k",
            "test_actual_runtime_identity_ast_uses_shared_observations_without_ort_import",
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "3 passed" in completed.stdout
