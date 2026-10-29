# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ORT-only session controls and actual readback; no builder/framework imports."""

from __future__ import annotations

import onnxruntime as ort

_REQUIRED_CUDA = {
    "use_tf32": "0",
    "enable_cuda_graph": "0",
    "arena_extend_strategy": "kSameAsRequested",
    "cudnn_conv_algo_search": "EXHAUSTIVE",
}


class SessionProfileError(RuntimeError):
    """Actual readback is preserved even when required semantics fail."""

    def __init__(self, message, readback):
        super().__init__(message)
        self.readback = readback


def validate_requested_profile(selection):
    """Keep the qualification profile explicit; never repair or invent options."""
    if selection["optimization"] != "ORT_ENABLE_ALL":
        raise ValueError("Selected session profile requires ORT_ENABLE_ALL")
    if selection["device"] == "cuda":
        provider, options = selection["providers"][0]
        if provider != "CUDAExecutionProvider" or any(
            options.get(key) != value for key, value in _REQUIRED_CUDA.items()
        ):
            raise ValueError(
                "Selected CUDA profile must explicitly preserve required controls"
            )
        if str(options.get("device_id")) != str(selection["device_id"]):
            raise ValueError("Selected CUDA device placement differs")
        if selection["session_config"].get("session.disable_prepacking") != "1":
            raise ValueError("Selected CUDA profile requires explicit disabled prepacking")


def session_options(selection=None, *, profiling=False):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_mem_pattern = False
    options.enable_profiling = profiling
    if selection is not None:
        validate_requested_profile(selection)
        for key, value in selection["session_config"].items():
            options.add_session_config_entry(key, value)
    return options


def read_session_profile(session, selection, *, profiling=False, compare_observed_map=True):
    """Read actual providers/options/SessionOptions before any neural execution.

    ORT exposes no disabled-optimizer readback API. Record that explicit request
    separately, rather than mislabelling it an observed runtime attribute.
    """
    validate_requested_profile(selection)
    actual_providers = session.get_providers()
    actual_map = session.get_provider_options()
    options = session.get_session_options()
    observed = {
        "graph_optimization_level": str(options.graph_optimization_level).split(".")[-1],
        "intra_op_num_threads": options.intra_op_num_threads,
        "inter_op_num_threads": options.inter_op_num_threads,
        "enable_mem_pattern": options.enable_mem_pattern,
        "enable_profiling": options.enable_profiling,
    }
    config = {
        key: options.get_session_config_entry(key) for key in selection["session_config"]
    }
    readback = {
        "actual_providers": actual_providers,
        "registered_provider_order_not_node_placement": actual_providers,
        "runtime_node_placement_attested": False,
        "actual_provider_options": actual_map,
        "actual_session_options": observed,
        "actual_session_config": config,
        "disabled_optimizers_requested_not_runtime_readback": selection["disabled_optimizers"],
        "requested_device": selection["device"],
        "requested_device_id": selection["device_id"],
        "requested_providers": selection["providers"],
    }
    expected_names = [
        item[0] if isinstance(item, (list, tuple)) else item for item in selection["providers"]
    ]
    if actual_providers != expected_names:
        raise SessionProfileError(
            "ORT provider fallback differs from selected profile", readback
        )
    if compare_observed_map and actual_providers != selection["actual_providers"]:
        raise SessionProfileError(
            "ORT provider identity differs from observed profile", readback
        )
    if compare_observed_map and actual_map != selection["actual_provider_options"]:
        raise SessionProfileError(
            "Actual provider options differ from selected profile", readback
        )
    if selection["device"] == "cuda":
        required = {**_REQUIRED_CUDA, "device_id": str(selection["device_id"])}
        if any(
            actual_map.get("CUDAExecutionProvider", {}).get(key) != value
            for key, value in required.items()
        ):
            raise SessionProfileError(
                "Actual CUDA options contradict required selected semantics", readback
            )
    expected = {
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "intra_op_num_threads": 8,
        "inter_op_num_threads": 1,
        "enable_mem_pattern": False,
        "enable_profiling": profiling,
    }
    if observed != expected:
        raise SessionProfileError(
            f"Actual SessionOptions differ from selected profile: {observed}", readback
        )
    if config != selection["session_config"]:
        raise SessionProfileError(
            "Actual session config differs from selected profile", readback
        )
    return readback
