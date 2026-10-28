# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bounded NVML process/CSV contracts; no physical GPU queries."""

from __future__ import annotations

import json
import os
import subprocess

import drama3_runtime as runtime
import pytest


class QueryProcess:
    def __init__(self, *, output="", error="", waits=(0,), kill_error=None):
        self.output = output.encode()
        self.error = error.encode()
        self.waits = list(waits)
        self.kill_error = kill_error
        self.pid = 123456
        self.returncode = None
        self.timeouts = []
        self.kills = 0

    def start(self, command, *, stdout, stderr):
        stdout.write(self.output)
        stderr.write(self.error)
        return self

    def wait(self, *, timeout):
        self.timeouts.append(timeout)
        result = self.waits.pop(0)
        if isinstance(result, Exception):
            raise result
        self.returncode = result
        return result

    def kill(self):
        self.kills += 1
        if self.kill_error:
            raise self.kill_error


def install_queries(monkeypatch, *queries):
    remaining = list(queries)
    launched = []

    def start(command, **kwargs):
        launched.append(command)
        assert remaining, "No automatic query retry is allowed"
        return remaining.pop(0).start(command, **kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", start)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return launched


@pytest.mark.parametrize("cleanup", ["reaped", "kill-error", "reap-timeout", "reap-error"])
def test_query_timeout_kills_only_owned_child_and_bounded_reaps(monkeypatch, cleanup, capsys):
    timeout = subprocess.TimeoutExpired(["nvidia-smi"], 20)
    second = {
        "reaped": -9,
        "kill-error": -9,
        "reap-timeout": subprocess.TimeoutExpired(["nvidia-smi"], 2),
        "reap-error": OSError("owned child wait failed"),
    }[cleanup]
    query = QueryProcess(
        output="partial GPU row",
        error="NVML stalled",
        waits=(timeout, second),
        kill_error=OSError("owned kill failed") if cleanup == "kill-error" else None,
    )
    launched = install_queries(monkeypatch, query)
    with pytest.raises(runtime.GpuSnapshotError, match="UNKNOWN") as caught:
        runtime.gpu_snapshot(0)
    evidence = caught.value.evidence["failed_query"]
    assert evidence["status"] == "UNKNOWN"
    assert evidence["owned_pid"] == query.pid
    assert evidence["stdout"] == "partial GPU row"
    assert evidence["stderr"] == "NVML stalled"
    assert evidence["kill_attempted"] is True
    assert evidence["reaped"] is (cleanup in ("reaped", "kill-error"))
    assert query.timeouts[0] == 20
    assert 0 <= query.timeouts[1] <= 2
    assert query.kills == 1 and len(launched) == 1
    assert ("kill_error" in evidence) is (cleanup == "kill-error")
    assert ("reap_error" in evidence) is cleanup.startswith("reap-")
    event = json.loads(capsys.readouterr().out)
    assert event["event"] == "gpu_snapshot_error"
    assert event["status"] == "UNKNOWN"
    assert event["failed_query"] == evidence


def test_cleanup_consumed_grace_never_gets_a_second_full_wait(monkeypatch):
    query = QueryProcess(
        waits=(
            subprocess.TimeoutExpired("nvidia-smi", 20),
            subprocess.TimeoutExpired("nvidia-smi", 0),
        )
    )
    install_queries(monkeypatch, query)
    ticks = iter([0.0, 20.0, 22.5, 22.5])
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(ticks))
    with pytest.raises(runtime.GpuSnapshotError):
        runtime.gpu_snapshot(0)
    assert query.timeouts == [20, 0]


def test_query_nonzero_exit_is_not_empty_healthy_data(monkeypatch):
    query = QueryProcess(output="", error="NVML init failed", waits=(7,))
    launched = install_queries(monkeypatch, query)
    with pytest.raises(runtime.GpuSnapshotError) as caught:
        runtime.gpu_snapshot(0)
    evidence = caught.value.evidence["failed_query"]
    assert evidence["returncode"] == 7 and evidence["reaped"]
    assert evidence["stderr"] == "NVML init failed"
    assert not query.kills and len(launched) == 1


def test_query_spawn_failure_is_unknown_without_retry(monkeypatch):
    def fail(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not installed")

    monkeypatch.setattr(runtime.subprocess, "Popen", fail)
    with pytest.raises(runtime.GpuSnapshotError) as caught:
        runtime.gpu_snapshot(0)
    assert caught.value.evidence["failed_query"]["owned_pid"] is None
    assert "FileNotFoundError" in caught.value.evidence["failed_query"]["error"]


def test_app_query_timeout_retains_successful_gpu_query_evidence(monkeypatch):
    gpu = QueryProcess(output="0, H100, GPU-zero, 0, 0\n")
    apps = QueryProcess(
        output="GPU-zero,", waits=(subprocess.TimeoutExpired("nvidia-smi", 20), -9)
    )
    launched = install_queries(monkeypatch, gpu, apps)
    with pytest.raises(runtime.GpuSnapshotError) as caught:
        runtime.gpu_snapshot(0)
    evidence = caught.value.evidence
    assert len(evidence["queries"]) == 1
    assert evidence["queries"][0]["status"] == "OBSERVED"
    assert evidence["failed_query"]["stdout"] == "GPU-zero,"
    assert len(launched) == 2 and apps.kills == 1


@pytest.mark.parametrize(
    "gpu",
    [
        "",
        "0, H100, GPU-zero, 0\n",
        "0, H100, GPU-zero, N/A, 0\n",
        "0, H100, GPU-zero, -1, 0\n",
        "0, H100, GPU-zero, 0, 101\n",
        "0, H100, not-a-uuid, 0, 0\n",
        "1, H100, GPU-one, 0, 0\n",
        "0, H100, GPU-zero, 0, 0\n0, H100, GPU-zero, 0, 0\n",
        '0, "unfinished, GPU-zero, 0, 0\n',
    ],
)
def test_malformed_missing_or_duplicate_gpu_data_is_unknown(monkeypatch, gpu):
    install_queries(monkeypatch, QueryProcess(output=gpu), QueryProcess())
    with pytest.raises(runtime.GpuSnapshotError) as caught:
        runtime.gpu_snapshot(0)
    assert caught.value.evidence["status"] == "UNKNOWN"
    assert len(caught.value.evidence["queries"]) == 2


@pytest.mark.parametrize(
    "apps",
    [
        "GPU-zero\n",
        "GPU-zero, not-a-pid\n",
        "GPU-zero, 0\n",
        "bogus, 123\n",
        "\n",
        "GPU-zero, 123, extra\n",
    ],
)
def test_malformed_process_ownership_never_means_idle(monkeypatch, apps):
    install_queries(
        monkeypatch,
        QueryProcess(output="0, H100, GPU-zero, 0, 0\n"),
        QueryProcess(output=apps),
    )
    with pytest.raises(runtime.GpuSnapshotError):
        runtime.gpu_snapshot(0)


@pytest.mark.parametrize("foreign", [False, True])
def test_healthy_busy_exact_visible_uuid_and_raw_query_evidence(monkeypatch, foreign):
    text = "0, H100, GPU-zero, 0, 0\n2, H100, GPU-two, 4, 0\n"
    apps = f"GPU-two, {os.getpid()}\nGPU-zero, 99999\n"
    if foreign:
        apps += "GPU-two, 88888\n"
    launched = install_queries(
        monkeypatch, QueryProcess(output=text), QueryProcess(output=apps)
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-two")
    snapshot = runtime.gpu_snapshot(0)
    assert snapshot["uuid"] == "GPU-two" and snapshot["physical_index"] == 2
    assert snapshot["foreign_pids"] == ([88888] if foreign else [])
    assert snapshot["queries"][0]["stdout"] == text
    assert all(item["owned_pid"] == 123456 for item in snapshot["queries"])
    assert len(launched) == 2
    if foreign:
        with pytest.raises(RuntimeError, match="occupied"):
            runtime.require_idle(snapshot, False)
    else:
        assert runtime.require_idle(snapshot, False)


def test_empty_successful_app_query_is_valid_idle_not_failure_fallback(monkeypatch):
    install_queries(
        monkeypatch, QueryProcess(output="0, H100, GPU-zero, 0, 0\n"), QueryProcess(output="")
    )
    assert runtime.require_idle(runtime.gpu_snapshot(0), False)


def test_ambiguous_uuid_prefix_is_unknown(monkeypatch):
    install_queries(
        monkeypatch,
        QueryProcess(output="0, H100, GPU-zero-1, 0, 0\n1, H100, GPU-zero-2, 0, 0\n"),
        QueryProcess(),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-zero")
    with pytest.raises(runtime.GpuSnapshotError):
        runtime.gpu_snapshot(0)


def test_unknown_snapshot_cannot_be_accepted_as_contended_diagnostic():
    with pytest.raises(runtime.GpuSnapshotError):
        runtime.require_idle({"status": "UNKNOWN"}, True)


def test_output_limit_is_explicit_unknown_not_partial_csv(monkeypatch):
    install_queries(monkeypatch, QueryProcess(output="x" * (1024 * 1024 + 1)))
    with pytest.raises(runtime.GpuSnapshotError) as caught:
        runtime.gpu_snapshot(0)
    assert caught.value.evidence["failed_query"]["stdout_truncated"] is True
    assert len(caught.value.evidence["failed_query"]["stdout"]) == 1024 * 1024
