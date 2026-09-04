# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fail-closed Lightning checkpoint loading for MatterGen score-core exports."""

from __future__ import annotations

__all__ = [
    "MATTERGEN_MODEL_STATE_PREFIX",
    "apply_mattergen_checkpoint",
    "load_mattergen_state_dict",
]

import json
from collections.abc import Mapping
from pathlib import Path

import torch

from mobius._model_package import ModelPackage

MATTERGEN_MODEL_STATE_PREFIX = "diffusion_module.model."


def load_mattergen_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Read precisely the inference model state from a MatterGen Lightning checkpoint.

    The official archives contain optimizer, callback, and Hydra metadata in
    addition to the state dict.  ``weights_only=True`` rejects executable
    pickle payloads, and the strict prefix check ensures that none of that
    training state can be confused with an exported model tensor.
    """
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        raise TypeError("MatterGen checkpoint must deserialize to a mapping.")
    state_dict = loaded.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("MatterGen checkpoint has no mapping-valued 'state_dict'.")

    routed: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    for source_name, tensor in state_dict.items():
        if not isinstance(source_name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("MatterGen state_dict must contain string tensor entries only.")
        if not source_name.startswith(MATTERGEN_MODEL_STATE_PREFIX):
            unexpected.append(source_name)
            continue
        target_name = source_name.removeprefix(MATTERGEN_MODEL_STATE_PREFIX)
        if not target_name:
            raise ValueError("MatterGen state_dict contains an empty inference tensor name.")
        if target_name in routed:
            raise ValueError(f"MatterGen state_dict maps multiple tensors to {target_name!r}.")
        routed[target_name] = tensor
    if unexpected:
        raise ValueError(
            "MatterGen checkpoint has state_dict tensors outside "
            f"{MATTERGEN_MODEL_STATE_PREFIX!r}: {sorted(unexpected)[:5]}"
        )
    if not routed:
        raise ValueError("MatterGen checkpoint has no inference model tensors.")
    return routed


def _assert_exact_tensor_routing(
    package: ModelPackage, state_dict: Mapping[str, torch.Tensor]
) -> None:
    """Reject missing, unknown, or shape-mismatched tensors before mutation."""
    if set(package) != {"model"}:
        raise ValueError(
            "MatterGen weight routing requires exactly one score-core component named 'model'."
        )
    initializers = package["model"].graph.initializers
    # GraphBuilder materializes scalar ONNX constants as initializers too. They
    # are graph literals, not checkpoint parameters, and already have values.
    expected = {
        name for name, initializer in initializers.items() if initializer.const_value is None
    }
    actual = set(state_dict)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {len(missing)} graph tensor(s): {missing[:5]}")
        if unexpected:
            details.append(f"unrouted {len(unexpected)} checkpoint tensor(s): {unexpected[:5]}")
        raise ValueError("MatterGen checkpoint routing is incomplete; " + "; ".join(details))

    shape_mismatches = []
    for name in sorted(expected):
        initializer = initializers[name]
        if initializer.shape is None:
            raise ValueError(f"MatterGen initializer {name!r} has no concrete shape.")
        if not all(isinstance(dimension, int) for dimension in initializer.shape):
            raise ValueError(f"MatterGen initializer {name!r} has a symbolic shape.")
        expected_shape = tuple(initializer.shape)
        actual_shape = tuple(state_dict[name].shape)
        if expected_shape != actual_shape:
            shape_mismatches.append((name, expected_shape, actual_shape))
    if shape_mismatches:
        name, expected_shape, actual_shape = shape_mismatches[0]
        raise ValueError(
            f"MatterGen tensor shape mismatch for {name!r}: graph expects "
            f"{expected_shape}, checkpoint provides {actual_shape} "
            f"({len(shape_mismatches)} mismatch(es) total)."
        )


def apply_mattergen_checkpoint(
    package: ModelPackage,
    module,
    checkpoint_path: str | Path,
) -> None:
    """Load all official MatterGen inference tensors into *package* exactly once.

    ``module.preprocess_weights`` is responsible only for architecture-specific
    key normalization.  The complete post-normalization key set must match the
    score graph's initializers exactly; unlike the generic loader, no unknown
    checkpoint weights are merely logged and skipped.
    """
    state_dict = load_mattergen_state_dict(checkpoint_path)
    preprocess_weights = getattr(module, "preprocess_weights", None)
    if not callable(preprocess_weights):
        raise TypeError(f"{type(module).__name__} must define preprocess_weights().")
    normalized = preprocess_weights(state_dict)
    if not isinstance(normalized, Mapping) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in normalized.items()
    ):
        raise TypeError("MatterGen preprocess_weights() must return a string-to-tensor mapping.")
    normalized_tensors = dict(normalized)
    _assert_exact_tensor_routing(package, normalized_tensors)
    package.apply_weights(normalized_tensors)

    report = {
        "format": "mobius.weight-loading-report.v1",
        "source": str(checkpoint_path),
        "source_state_prefix": MATTERGEN_MODEL_STATE_PREFIX,
        "output_weight_format": "dense",
        "native_fp8": False,
        "source_tensors": len(state_dict),
        "assigned_tensors": len(normalized_tensors),
        "canonicalized_alias_tensors": len(state_dict) - len(normalized_tensors),
        "ignored_tensors": 0,
        "routing": "exact-post-preprocess-with-validated-outputblock-aliases",
    }
    package.weight_loading_report = report
    package["model"].metadata_props["mobius.weight_loading"] = json.dumps(report, sort_keys=True)
