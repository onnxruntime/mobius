# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Performance-conformance records for metadata-driven ONNX GenAI workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PerformanceComparison:
    """Result of comparing one workflow run with its native control."""

    failures: tuple[str, ...]
    observations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


_IDENTITY_FIELDS = (
    "scenario",
    "workload",
    "model",
    "model_hash",
    "runtime_commit",
    "runtime_build",
    "execution_provider",
    "provider_options",
    "device",
    "precision",
    "batch_size",
    "input_shapes",
    "work_units",
    "sampling_algorithm",
    "sampling_parameters",
    "rng_seed",
    "rng_offset",
    "kv_mode",
    "kv_capacity",
    "graph_capture",
    "graph_capture_shape",
    "warmup_count",
    "measured_iterations",
    "synchronization_timing",
    "required_islands",
)


def _number(record: dict[str, Any], name: str) -> float:
    value = record.get(name)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"metric {name!r} must be numeric")
    return float(value)


def _validate_identity(workflow: dict[str, Any], native: dict[str, Any]) -> None:
    workflow_identity = workflow.get("identity")
    native_identity = native.get("identity")
    if not isinstance(workflow_identity, dict) or not isinstance(native_identity, dict):
        raise TypeError("both records require an identity object")
    missing = [
        name
        for name in _IDENTITY_FIELDS
        if name not in workflow_identity or name not in native_identity
    ]
    if missing:
        raise ValueError(
            "performance records omit required identity fields: " + ", ".join(missing)
        )
    mismatches = [
        name for name in _IDENTITY_FIELDS if workflow_identity[name] != native_identity[name]
    ]
    extra_mismatches = [
        name
        for name in workflow_identity.keys() | native_identity.keys()
        if workflow_identity.get(name) != native_identity.get(name)
    ]
    mismatches.extend(name for name in extra_mismatches if name not in mismatches)
    if mismatches:
        raise ValueError(
            "performance records are not comparable; identity differs for "
            + ", ".join(mismatches)
        )


def _required_island_failures(record: dict[str, Any]) -> list[str]:
    identity = record["identity"]
    if not identity.get("graph_capture"):
        return []
    required = identity.get("required_islands", [])
    diagnostics = record.get("islands", [])
    failures: list[str] = []
    for expected in required:
        components = expected["components"]
        match = next(
            (island for island in diagnostics if island.get("components") == components),
            None,
        )
        label = " -> ".join(components)
        if match is None:
            failures.append(f"required execution island was not formed: {label}")
        elif match.get("device") != expected["device"]:
            failures.append(
                f"required execution island used {match.get('device')!r}, "
                f"expected {expected['device']!r}: {label}"
            )
        elif _number(match, "captures") < 1 or _number(match, "replays") < 1:
            reason = match.get("fallback_reason") or "capture/replay was not observed"
            failures.append(f"required execution island was not replayed: {label}: {reason}")
    return failures


def compare_performance(
    workflow: dict[str, Any],
    native: dict[str, Any],
    *,
    max_regression_fraction: float = 0.05,
) -> PerformanceComparison:
    """Compare equivalent workflow/native measurements and enforce the release bar."""
    if not 0 <= max_regression_fraction < 1:
        raise ValueError("max_regression_fraction must be in [0, 1)")
    _validate_identity(workflow, native)
    workflow_metrics = workflow.get("metrics")
    native_metrics = native.get("metrics")
    if not isinstance(workflow_metrics, dict) or not isinstance(native_metrics, dict):
        raise TypeError("both records require a metrics object")
    required_metrics = {
        "throughput_unit",
        "throughput",
        "peak_memory_bytes",
        "device_sync_count",
        "host_to_device_copy_count",
        "host_to_device_bytes",
        "device_to_host_copy_count",
        "device_to_host_bytes",
        "session_boundary_count",
        "kernel_launch_count",
        "device_resident_intermediate_count",
        "intermediate_value_count",
    }
    if workflow["identity"]["workload"] == "generation":
        required_metrics.add("ttft_ms")
    missing_metrics = [
        name
        for name in sorted(required_metrics)
        if name not in workflow_metrics or name not in native_metrics
    ]
    if missing_metrics:
        raise ValueError(
            "performance records omit required metrics: " + ", ".join(missing_metrics)
        )

    failures = _required_island_failures(workflow)
    observations: list[str] = []
    lower_is_better = ("ttft_ms", "peak_memory_bytes")
    exact_or_better = (
        "device_sync_count",
        "host_to_device_copy_count",
        "host_to_device_bytes",
        "device_to_host_copy_count",
        "device_to_host_bytes",
        "session_boundary_count",
        "kernel_launch_count",
    )

    throughput_unit = workflow_metrics.get("throughput_unit")
    if throughput_unit != native_metrics.get("throughput_unit"):
        raise ValueError("throughput units differ")
    workflow_throughput = _number(workflow_metrics, "throughput")
    native_throughput = _number(native_metrics, "throughput")
    throughput_ratio = workflow_throughput / native_throughput
    observations.append(f"throughput ratio={throughput_ratio:.4f} {throughput_unit}")
    if throughput_ratio < 1 - max_regression_fraction:
        failures.append(
            f"throughput regressed by {(1 - throughput_ratio) * 100:.2f}% "
            f"({workflow_throughput:g} vs {native_throughput:g} {throughput_unit})"
        )

    for name in lower_is_better:
        if name not in workflow_metrics or name not in native_metrics:
            continue
        workflow_value = _number(workflow_metrics, name)
        native_value = _number(native_metrics, name)
        ratio = workflow_value / native_value if native_value else 1.0
        observations.append(f"{name} ratio={ratio:.4f}")
        if workflow_value > native_value * (1 + max_regression_fraction):
            failures.append(
                f"{name} regressed by {(ratio - 1) * 100:.2f}% "
                f"({workflow_value:g} vs {native_value:g})"
            )

    for name in exact_or_better:
        workflow_value = _number(workflow_metrics, name)
        native_value = _number(native_metrics, name)
        observations.append(f"{name}={workflow_value:g} vs {native_value:g}")
        if workflow_value > native_value:
            failures.append(f"{name} increased ({workflow_value:g} vs {native_value:g})")

    workflow_intermediates = _number(workflow_metrics, "intermediate_value_count")
    native_intermediates = _number(native_metrics, "intermediate_value_count")
    if workflow_intermediates <= 0 or native_intermediates <= 0:
        raise ValueError("intermediate_value_count must be positive")
    workflow_residency = (
        _number(workflow_metrics, "device_resident_intermediate_count")
        / workflow_intermediates
    )
    native_residency = (
        _number(native_metrics, "device_resident_intermediate_count") / native_intermediates
    )
    observations.append(f"device residency={workflow_residency:.4f} vs {native_residency:.4f}")
    if workflow_residency < native_residency:
        failures.append(
            "device-resident intermediate ratio decreased "
            f"({workflow_residency:.4f} vs {native_residency:.4f})"
        )

    return PerformanceComparison(tuple(failures), tuple(observations))
