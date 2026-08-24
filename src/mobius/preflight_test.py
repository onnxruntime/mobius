# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the resumable export preflight / dry-run.

Cover the three things that make a preflight worth running: it derives an exact
budget from metadata alone, it *refuses* (never silently passes) when a budget
is not met or metadata is unavailable, and it is resumable with identity-drift
protection.
"""

from __future__ import annotations

import json
import pathlib
import types

import pytest
import safetensors.torch
import torch

from mobius import preflight
from mobius.preflight import (
    ExportMode,
    LoaderMode,
    ShardMeta,
    estimate_budget,
    estimate_output_bytes,
    resolve_source,
    run_preflight,
)

# ---------------------------------------------------------------------------
# Local checkpoint fixtures
# ---------------------------------------------------------------------------


def _write_sharded(tmp_path: pathlib.Path, n_shards: int = 2) -> None:
    weight_map = {}
    for i in range(n_shards):
        fn = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
        safetensors.torch.save_file(
            {f"w{i}": torch.zeros(1024, 1024, dtype=torch.bfloat16)},
            str(tmp_path / fn),
        )
        weight_map[f"w{i}"] = fn
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": n_shards * 1024 * 1024 * 2}, "weight_map": weight_map})
    )


# ---------------------------------------------------------------------------
# Fake Hub
# ---------------------------------------------------------------------------


def _sibling(name, size, sha=None):
    lfs = types.SimpleNamespace(sha256=sha) if sha else None
    return types.SimpleNamespace(rfilename=name, size=size, lfs=lfs)


class _FakeApi:
    def __init__(self, info):
        self._info = info
        self.calls = 0

    def model_info(self, repo_id, revision=None, files_metadata=False):
        self.calls += 1
        return self._info


def _fake_hub(tmp_path, monkeypatch, weight_map):
    """Monkeypatch hf_hub_download to serve an index.json from tmp_path."""
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))

    def _dl(repo_id, filename, revision=None):
        return str(index_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _dl)


# ===========================================================================
# resolve_source
# ===========================================================================


class TestResolveSource:
    def test_local_sharded(self, tmp_path):
        _write_sharded(tmp_path, n_shards=3)
        commit, shards, _index = resolve_source(str(tmp_path))
        assert commit is None
        assert len(shards) == 3
        assert all(s.present_local and s.size for s in shards)

    def test_local_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_source(str(tmp_path))

    def test_hub_resolves_shards_and_sizes(self, tmp_path, monkeypatch):
        weight_map = {"w0": "model-00001-of-00002.safetensors", "w1": "model-00002-of-00002.safetensors"}
        _fake_hub(tmp_path, monkeypatch, weight_map)
        info = types.SimpleNamespace(
            sha="deadbeef",
            siblings=[
                _sibling("model-00001-of-00002.safetensors", 100, "aa"),
                _sibling("model-00002-of-00002.safetensors", 200, "bb"),
                _sibling("model.safetensors.index.json", 50),
            ],
            safetensors=types.SimpleNamespace(parameters={"BF16": 150}, total=150),
        )
        commit, shards, _index = resolve_source("org/model", hf_api=_FakeApi(info))
        assert commit == "deadbeef"
        assert {s.filename for s in shards} == set(weight_map.values())
        assert {s.size for s in shards} == {100, 200}
        assert {s.sha256 for s in shards} == {"aa", "bb"}

    def test_hub_index_references_absent_file_raises(self, tmp_path, monkeypatch):
        weight_map = {"w0": "missing.safetensors"}
        _fake_hub(tmp_path, monkeypatch, weight_map)
        info = types.SimpleNamespace(
            sha="c0ffee", siblings=[_sibling("other.safetensors", 100)], safetensors=None
        )
        with pytest.raises(FileNotFoundError, match="absent from the repository"):
            resolve_source("org/model", hf_api=_FakeApi(info))


# ===========================================================================
# Budget math
# ===========================================================================


class TestBudget:
    def test_output_bytes_modes(self):
        params = 1_000_000
        dtype_bytes = {"BF16": params * 2}
        assert estimate_output_bytes(params, dtype_bytes, ExportMode.PASSTHROUGH) == params * 2
        assert estimate_output_bytes(params, dtype_bytes, ExportMode.FP16) == params * 2
        int4 = estimate_output_bytes(params, dtype_bytes, ExportMode.INT4_QMOE, group_size=32)
        # 0.5 + 2/32 + 0.5/32 = 0.578125 bytes/param
        assert int4 == int(params * 0.578125)

    def test_passthrough_fp8_output_is_dequantized_size(self):
        params = 1000
        dtype_bytes = {"F8_E4M3": params}  # fp8 stores 1 byte/param
        # A passthrough export dequantizes fp8 -> bf16, so the output external
        # data is 2 bytes/param, not the 1 byte/param it was stored at.
        assert estimate_output_bytes(params, dtype_bytes, ExportMode.PASSTHROUGH) == params * 2

    def test_eager_peak_exceeds_stream_peak(self):
        shards = [ShardMeta(f"s{i}", size=5 * 1000**3) for i in range(10)]
        index = {"metadata": {"total_size": 50 * 1000**3}}
        meta = {"parameters": {"BF16": 25 * 1000**3}, "total": 25 * 1000**3}
        eager = estimate_budget(shards, index, meta, loader=LoaderMode.EAGER)
        stream = estimate_budget(shards, index, meta, loader=LoaderMode.STREAM)
        assert eager.peak_ram_bytes == eager.source_bytes  # whole checkpoint resident
        assert stream.peak_ram_bytes <= 2 * stream.largest_shard_bytes
        assert stream.peak_ram_bytes < eager.peak_ram_bytes

    def test_fp8_source_adds_upconvert_to_eager_peak(self):
        shards = [ShardMeta("s0", size=100)]
        meta = {"parameters": {"F8_E4M3": 100}, "total": 100}
        eager = estimate_budget(shards, {}, meta, loader=LoaderMode.EAGER)
        # source (100 * 1 byte) + fp8->bf16 upconvert (extra 100)
        assert eager.peak_ram_eager_bytes == 200


# ===========================================================================
# run_preflight — refusal semantics
# ===========================================================================


class TestRunPreflight:
    def test_metadata_unavailable_is_a_blocker_not_a_pass(self, tmp_path):
        # A local dir with no safetensors -> resolve fails -> refused.
        result = run_preflight(str(tmp_path), output_dir=str(tmp_path / "out"))
        assert result.ok is False
        assert any("metadata unavailable" in b for b in result.blockers)

    def test_hub_access_denied_refuses(self, tmp_path):
        class _DenyApi:
            def model_info(self, *a, **k):
                raise PermissionError("401 Client Error: Invalid username or password")

        result = run_preflight(
            "org/private", output_dir=str(tmp_path / "out"), hf_api=_DenyApi()
        )
        assert result.ok is False
        assert any("metadata unavailable" in b for b in result.blockers)

    def test_ready_when_space_sufficient(self, tmp_path, monkeypatch):
        _write_sharded(tmp_path, n_shards=2)
        monkeypatch.setattr(preflight, "_free_bytes", lambda p: 10**15)
        monkeypatch.setattr(preflight, "_available_ram_bytes", lambda: 10**15)
        result = run_preflight(
            str(tmp_path), output_dir=str(tmp_path / "out"), loader=LoaderMode.STREAM
        )
        assert result.ok is True
        assert result.budget is not None
        assert not result.blockers

    def test_refuses_when_disk_insufficient(self, tmp_path, monkeypatch):
        _write_sharded(tmp_path, n_shards=2)
        monkeypatch.setattr(preflight, "_free_bytes", lambda p: 1)  # ~no disk
        monkeypatch.setattr(preflight, "_available_ram_bytes", lambda: 10**15)
        result = run_preflight(str(tmp_path), output_dir=str(tmp_path / "out"))
        assert result.ok is False
        assert any("disk" in b for b in result.blockers)

    def test_refuses_when_ram_insufficient_eager(self, tmp_path, monkeypatch):
        _write_sharded(tmp_path, n_shards=2)
        monkeypatch.setattr(preflight, "_free_bytes", lambda p: 10**15)
        monkeypatch.setattr(preflight, "_available_ram_bytes", lambda: 1)
        result = run_preflight(
            str(tmp_path), output_dir=str(tmp_path / "out"), loader=LoaderMode.EAGER
        )
        assert result.ok is False
        assert any("ram" in b for b in result.blockers)

    def test_vram_single_device_refusal(self, tmp_path, monkeypatch):
        _write_sharded(tmp_path, n_shards=2)
        monkeypatch.setattr(preflight, "_free_bytes", lambda p: 10**15)
        monkeypatch.setattr(preflight, "_available_ram_bytes", lambda: 10**15)
        result = run_preflight(
            str(tmp_path),
            output_dir=str(tmp_path / "out"),
            gpu_total_bytes=1,  # absurdly small device
        )
        assert result.ok is False
        assert any("VRAM" in b for b in result.blockers)

    def test_same_device_disk_check_sums_download_and_output(self, tmp_path, monkeypatch):
        # Hub source (not present locally) so a real download is required.
        weight_map = {
            "w0": "model-00001-of-00002.safetensors",
            "w1": "model-00002-of-00002.safetensors",
        }
        _fake_hub(tmp_path, monkeypatch, weight_map)
        info = types.SimpleNamespace(
            sha="deadbeef",
            siblings=[
                _sibling("model-00001-of-00002.safetensors", 100, "aa"),
                _sibling("model-00002-of-00002.safetensors", 100, "bb"),
            ],
            safetensors=types.SimpleNamespace(parameters={"BF16": 100}, total=100),
        )
        monkeypatch.setattr(preflight, "_available_ram_bytes", lambda: 10**15)
        # Free space (240) fits the 200B download or the 200B output alone, but
        # not both (400B combined) on one filesystem. Independent checks would
        # each pass; the combined check must refuse.
        monkeypatch.setattr(preflight, "_free_bytes", lambda p: 240)
        result = run_preflight(
            "org/model",
            output_dir=str(tmp_path / "out"),
            download_dir=str(tmp_path),  # same filesystem as the output dir
            hf_api=_FakeApi(info),
        )
        assert result.ok is False
        assert any("download+output" in b for b in result.blockers)

    def test_ram_unverifiable_refuses(self, tmp_path, monkeypatch):
        _write_sharded(tmp_path, n_shards=2)
        monkeypatch.setattr(preflight, "_free_bytes", lambda p: 10**15)
        # MemAvailable can't be read -> the RAM budget is unverifiable and must
        # not produce a success-shaped pass.
        monkeypatch.setattr(preflight, "_available_ram_bytes", lambda: None)
        result = run_preflight(str(tmp_path), output_dir=str(tmp_path / "out"))
        assert result.ok is False
        assert any("RAM budget could not be verified" in b for b in result.blockers)


# ===========================================================================
# Resumability + identity drift
# ===========================================================================


class TestResumability:
    def test_state_written_and_reused(self, tmp_path, monkeypatch):
        _write_sharded(tmp_path, n_shards=2)
        monkeypatch.setattr(preflight, "_free_bytes", lambda p: 10**15)
        monkeypatch.setattr(preflight, "_available_ram_bytes", lambda: 10**15)
        state_file = tmp_path / "state.json"

        r1 = run_preflight(
            str(tmp_path), output_dir=str(tmp_path / "out"), state_path=str(state_file)
        )
        assert r1.ok
        assert state_file.is_file()
        saved = json.loads(state_file.read_text())
        assert len(saved["validated_shards"]) == 2

        # Second run reuses state (marks shards validated).
        r2 = run_preflight(
            str(tmp_path), output_dir=str(tmp_path / "out"), state_path=str(state_file)
        )
        assert r2.ok
        assert all(s.validated for s in r2.shards)

    def test_identity_drift_refuses(self, tmp_path, monkeypatch):
        weight_map = {"w0": "model-00001-of-00001.safetensors"}
        _fake_hub(tmp_path, monkeypatch, weight_map)
        monkeypatch.setattr(preflight, "_free_bytes", lambda p: 10**15)
        monkeypatch.setattr(preflight, "_available_ram_bytes", lambda: 10**15)
        state_file = tmp_path / "state.json"
        state_file.write_text(
            json.dumps({"commit_sha": "OLDSHA", "validated_shards": ["model-00001-of-00001.safetensors"]})
        )
        info = types.SimpleNamespace(
            sha="NEWSHA",
            siblings=[_sibling("model-00001-of-00001.safetensors", 100, "aa")],
            safetensors=types.SimpleNamespace(parameters={"BF16": 50}, total=50),
        )
        result = run_preflight(
            "org/model",
            output_dir=str(tmp_path / "out"),
            state_path=str(state_file),
            hf_api=_FakeApi(info),
        )
        assert any("identity drift" in b for b in result.blockers)
        assert result.ok is False
