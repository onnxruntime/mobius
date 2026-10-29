# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only profile discovery/readback contracts; no CUDA sessions or inference."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import onnxruntime as ort
import pytest
from drama3_session_profile import SessionProfileError, read_session_profile, session_options


@pytest.fixture
def selected():
    cuda = {
        "device_id": "0",
        "use_tf32": "0",
        "enable_cuda_graph": "0",
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
    }
    return {
        "device": "cuda",
        "device_id": 0,
        "optimization": "ORT_ENABLE_ALL",
        "providers": [["CUDAExecutionProvider", cuda], "CPUExecutionProvider"],
        "actual_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "actual_provider_options": {
            "CUDAExecutionProvider": {**cuda, "full_actual_extra_option": "retained"},
            "CPUExecutionProvider": {},
        },
        "disabled_optimizers": [],
        "session_config": {"session.disable_prepacking": "1"},
    }


def actual_session(selected, options=None):
    return SimpleNamespace(
        get_providers=lambda: selected["actual_providers"],
        get_provider_options=lambda: copy.deepcopy(selected["actual_provider_options"]),
        get_session_options=lambda: options or session_options(selected),
        run=lambda *_: pytest.fail("No inference allowed for profile attestation"),
    )


def test_full_actual_map_and_options_are_observed_not_reconstructed(selected):
    observed = read_session_profile(actual_session(selected), selected)
    assert observed["actual_provider_options"] == selected["actual_provider_options"]
    assert (
        observed["actual_provider_options"]["CUDAExecutionProvider"][
            "full_actual_extra_option"
        ]
        == "retained"
    )
    assert observed["actual_session_options"]["graph_optimization_level"] == "ORT_ENABLE_ALL"
    assert observed["actual_session_config"] == {"session.disable_prepacking": "1"}
    assert observed["disabled_optimizers_requested_not_runtime_readback"] == []


def test_discovery_cannot_accept_contradictory_semantics_without_an_expected_map(selected):
    expected = copy.deepcopy(selected)
    actual = copy.deepcopy(selected)
    actual["actual_provider_options"]["CUDAExecutionProvider"]["use_tf32"] = "1"
    with pytest.raises(RuntimeError, match="required selected semantics"):
        read_session_profile(actual_session(actual), expected, compare_observed_map=False)


def test_discovery_captures_full_map_without_inventing_expected_map(selected):
    expected = copy.deepcopy(selected)
    del expected["actual_provider_options"]
    del expected["actual_providers"]
    observed = read_session_profile(
        actual_session(selected), expected, compare_observed_map=False
    )
    assert observed["actual_provider_options"] == selected["actual_provider_options"]


@pytest.mark.parametrize(
    "option,value",
    [
        ("graph_optimization_level", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
        ("intra_op_num_threads", 1),
        ("inter_op_num_threads", 8),
        ("enable_mem_pattern", True),
        ("enable_profiling", True),
    ],
)
def test_actual_session_options_mismatch_fails(selected, option, value):
    actual = session_options(selected)
    setattr(actual, option, value)
    with pytest.raises(RuntimeError, match="Actual SessionOptions"):
        read_session_profile(actual_session(selected, actual), selected)


def test_session_config_readback_drift_fails(selected):
    actual = session_options(selected)
    actual.add_session_config_entry("session.disable_prepacking", "0")
    with pytest.raises(RuntimeError, match="Actual session config"):
        read_session_profile(actual_session(selected, actual), selected)


@pytest.mark.parametrize("failure", ["options", "config"])
def test_getter_exception_preserves_only_available_observations(selected, failure):
    actual = actual_session(selected)
    if failure == "options":

        def fail_options():
            raise RuntimeError("actual SessionOptions getter unavailable")

        actual.get_session_options = fail_options
    else:
        options = session_options(selected)
        actual.get_session_options = lambda: SimpleNamespace(
            graph_optimization_level=options.graph_optimization_level,
            intra_op_num_threads=8,
            inter_op_num_threads=1,
            enable_mem_pattern=False,
            enable_profiling=False,
            get_session_config_entry=lambda _key: (_ for _ in ()).throw(
                RuntimeError("actual config key missing")
            ),
        )
    with pytest.raises(SessionProfileError, match="getter failed") as caught:
        read_session_profile(actual, selected)
    readback = caught.value.readback
    assert readback["actual_provider_options"] == selected["actual_provider_options"]
    assert readback["actual_providers"] == selected["actual_providers"]
    assert readback["getter_errors"][0]["error_type"] == "RuntimeError"
    if failure == "options":
        assert "actual_session_options" not in readback
        assert "actual_session_config" not in readback
    else:
        assert readback["actual_session_options"]["intra_op_num_threads"] == 8
        assert readback["actual_session_config"] == {}
        assert "session.disable_prepacking" not in readback["actual_session_config"]


def test_declared_diagnostics_profiling_is_explicit_not_a_performance_profile(selected):
    actual = session_options(selected, profiling=True)
    observed = read_session_profile(actual_session(selected, actual), selected, profiling=True)
    assert observed["actual_session_options"]["enable_profiling"] is True


def test_full_map_difference_rejected_even_when_required_subset_matches(selected):
    actual = copy.deepcopy(selected)
    actual["actual_provider_options"]["CUDAExecutionProvider"]["extra"] = "new-value"
    with pytest.raises(RuntimeError, match="Actual provider options"):
        read_session_profile(actual_session(actual), selected)


def test_declared_provider_placement_cannot_fallback(selected):
    actual = actual_session(selected)
    actual.get_providers = lambda: ["CPUExecutionProvider"]
    with pytest.raises(RuntimeError, match="fallback"):
        read_session_profile(actual, selected)


@pytest.mark.parametrize(
    "key", ["use_tf32", "enable_cuda_graph", "arena_extend_strategy", "cudnn_conv_algo_search"]
)
def test_requested_control_omission_is_not_repaired(selected, key):
    del selected["providers"][0][1][key]
    with pytest.raises(ValueError, match="explicitly preserve"):
        session_options(selected)
