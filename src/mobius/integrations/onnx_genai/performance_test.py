# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import copy

import pytest

from mobius.integrations.onnx_genai.performance import compare_performance


def _record(*, workflow: bool) -> dict:
    return {
        "identity": {
            "scenario": "decoder-min-p",
            "workload": "generation",
            "model": "tiny-decoder",
            "model_hash": "sha256:fixture",
            "runtime_commit": "b2157a2",
            "runtime_build": "release+cuda",
            "execution_provider": "cuda",
            "provider_options": {"enable_cuda_graph": True},
            "device": "cuda:0",
            "precision": "float16",
            "batch_size": 1,
            "input_shapes": {"prompt_tokens": [1, 8]},
            "work_units": {"prompt_tokens": 8, "generated_tokens": 32},
            "sampling_algorithm": "seeded_min_p",
            "sampling_parameters": {"temperature": 0.8, "min_p": 0.1},
            "rng_seed": [7],
            "rng_offset": [0],
            "kv_mode": "paged",
            "kv_capacity": 128,
            "graph_capture": True,
            "graph_capture_shape": {"batch": 1, "decode_sequence": 1},
            "warmup_count": 3,
            "measured_iterations": 20,
            "synchronization_timing": "before_and_after_sample",
            "required_islands": [
                {
                    "device": "cuda",
                    "components": [
                        "decoder",
                        "last_token_logits",
                        "token_sampler",
                        "termination",
                    ],
                }
            ],
        },
        "metrics": {
            "throughput_unit": "tokens_per_second",
            "throughput": 102.0 if workflow else 100.0,
            "ttft_ms": 9.8 if workflow else 10.0,
            "peak_memory_bytes": 1000,
            "device_sync_count": 0,
            "host_to_device_copy_count": 0,
            "host_to_device_bytes": 0,
            "device_to_host_copy_count": 0,
            "device_to_host_bytes": 0,
            "session_boundary_count": 1,
            "kernel_launch_count": 12,
            "device_resident_intermediate_count": 8,
            "intermediate_value_count": 8,
        },
        "islands": [
            {
                "components": [
                    "decoder",
                    "last_token_logits",
                    "token_sampler",
                    "termination",
                ],
                "device": "cuda",
                "capture_eligible": True,
                "captures": 1,
                "replays": 10,
                "fallback_reason": None,
            }
        ],
    }


def test_equivalent_captured_workflow_meets_performance_bar():
    result = compare_performance(_record(workflow=True), _record(workflow=False))
    assert result.passed
    assert "throughput ratio=1.0200 tokens_per_second" in result.observations


def test_reports_throughput_sync_memory_and_capture_gaps():
    workflow = _record(workflow=True)
    native = _record(workflow=False)
    workflow["metrics"].update(
        {
            "throughput": 80.0,
            "ttft_ms": 12.0,
            "peak_memory_bytes": 1200,
            "device_sync_count": 2,
            "session_boundary_count": 3,
            "device_resident_intermediate_count": 6,
        }
    )
    workflow["islands"][0].update(
        {"captures": 0, "replays": 0, "fallback_reason": "dynamic allocation"}
    )

    result = compare_performance(workflow, native)

    assert not result.passed
    assert any("throughput regressed by 20.00%" in failure for failure in result.failures)
    assert any("ttft_ms regressed by 20.00%" in failure for failure in result.failures)
    assert any(
        "peak_memory_bytes regressed by 20.00%" in failure for failure in result.failures
    )
    assert any("device_sync_count increased" in failure for failure in result.failures)
    assert any("session_boundary_count increased" in failure for failure in result.failures)
    assert any("device-resident intermediate ratio" in failure for failure in result.failures)
    assert any("dynamic allocation" in failure for failure in result.failures)


def test_rejects_nonidentical_benchmark_conditions():
    workflow = _record(workflow=True)
    native = copy.deepcopy(_record(workflow=False))
    native["identity"]["kv_mode"] = "dense"

    with pytest.raises(ValueError, match="kv_mode"):
        compare_performance(workflow, native)


def test_requires_memory_metrics_and_island_device():
    workflow = _record(workflow=True)
    native = _record(workflow=False)
    del workflow["metrics"]["peak_memory_bytes"]
    with pytest.raises(ValueError, match="peak_memory_bytes"):
        compare_performance(workflow, native)

    workflow = _record(workflow=True)
    workflow["islands"][0]["device"] = "cpu"
    result = compare_performance(workflow, native)
    assert any("expected 'cuda'" in failure for failure in result.failures)
