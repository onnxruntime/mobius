# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Load standard adapter sources into the model-agnostic Mobius representation."""

from __future__ import annotations

__all__ = ["adapter_source_from_onnx_adapter", "load_peft_adapter"]

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import onnx_ir as ir
from safetensors.numpy import load_file

from mobius.adapters import (
    AdapterArtifact,
    AdapterSource,
    AdapterTarget,
    AdapterWeights,
)

_PEFT_CONFIG = "adapter_config.json"
_PEFT_WEIGHTS = "adapter_model.safetensors"
_FACTOR_PATTERN = re.compile(r"^(?P<module>.+)\.lora_(?P<factor>[AB])(?:\.[^.]+)?\.weight$")


def _source_checksum(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _pattern_value(patterns: Mapping[str, object], module_key: str, default: object) -> object:
    matches = [(key, value) for key, value in patterns.items() if module_key.endswith(key)]
    if not matches:
        return default
    return max(matches, key=lambda item: len(item[0]))[1]


def _resolve_target(module_key: str, targets: Mapping[str, AdapterTarget]) -> AdapterTarget:
    if module_key in targets:
        return targets[module_key]
    matches = [(name, target) for name, target in targets.items() if module_key.endswith(name)]
    if len(matches) != 1:
        raise ValueError(
            f"PEFT module {module_key!r} resolves to {len(matches)} producer targets; "
            "provide one exact or unique suffix binding"
        )
    return matches[0][1]


def load_peft_adapter(
    directory: str | Path,
    *,
    target_bindings: Mapping[str, AdapterTarget],
    base_fingerprint: str,
    name: str | None = None,
) -> AdapterArtifact:
    """Load PEFT config/safetensors using producer-declared, model-agnostic bindings."""
    directory = Path(directory)
    config_path = directory / _PEFT_CONFIG
    weights_path = directory / _PEFT_WEIGHTS
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError(
            f"PEFT adapter directory {directory} must contain "
            f"{_PEFT_CONFIG} and {_PEFT_WEIGHTS}"
        )
    config = json.loads(config_path.read_text())
    tensors = load_file(weights_path)
    target_modules = tuple(config.get("target_modules", ()))
    if not target_modules:
        raise ValueError("PEFT adapter target_modules must not be empty")
    default_rank = int(config.get("r", 0))
    default_alpha = float(config.get("lora_alpha", default_rank))
    rank_pattern = config.get("rank_pattern", {})
    alpha_pattern = config.get("alpha_pattern", {})

    pending: dict[str, dict[str, np.ndarray]] = {}
    for key, tensor in tensors.items():
        match = _FACTOR_PATTERN.match(key)
        if match is None:
            continue
        module_key = match.group("module")
        if not any(module_key.endswith(target) for target in target_modules):
            raise ValueError(f"PEFT tensor {key!r} is not covered by target_modules")
        pending.setdefault(module_key, {})[match.group("factor")] = tensor
    if not pending:
        raise ValueError("PEFT adapter contains no LoRA A/B tensors")

    loaded_weights: list[AdapterWeights] = []
    for module_key, factors in sorted(pending.items()):
        if set(factors) != {"A", "B"}:
            raise ValueError(f"PEFT module {module_key!r} must contain paired A/B factors")
        a = np.ascontiguousarray(factors["A"])
        b = np.ascontiguousarray(factors["B"])
        rank = int(_pattern_value(rank_pattern, module_key, default_rank))
        alpha = float(_pattern_value(alpha_pattern, module_key, default_alpha))
        if rank <= 0 or not math.isfinite(alpha):
            raise ValueError(
                f"PEFT module {module_key!r} has invalid rank/alpha {rank}/{alpha}"
            )
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(
                f"PEFT module {module_key!r} factors must have shapes "
                f"[rank,K]/[N,rank] for rank {rank}, got {a.shape}/{b.shape}"
            )
        loaded_weights.append(
            AdapterWeights(
                _resolve_target(module_key, target_bindings),
                ir.tensor(a),
                ir.tensor(b),
                alpha,
            )
        )

    source = AdapterSource(
        "peft_safetensors",
        path=str(directory),
        checksum=_source_checksum([config_path, weights_path]),
        base_model=config.get("base_model_name_or_path"),
        revision=config.get("revision"),
    )
    return AdapterArtifact(
        name=name or directory.name,
        base_fingerprint=base_fingerprint,
        weights=tuple(loaded_weights),
        source=source,
    )


def adapter_source_from_onnx_adapter(
    path: str | Path,
    *,
    native_parameters: Mapping[AdapterTarget, tuple[str, str]] | None = None,
) -> AdapterSource:
    """Declare an ORT FlatBuffers adapter source without making it mandatory."""
    path = Path(path)
    payload = path.read_bytes()
    if len(payload) < 8 or payload[4:8] != b"TORT":
        raise ValueError(f"ONNX adapter {path} does not contain the TORT identifier")
    return AdapterSource(
        "onnx_adapter",
        path=str(path),
        checksum=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        native_parameters=tuple(
            (target, names[0], names[1])
            for target, names in sorted(
                (native_parameters or {}).items(),
                key=lambda item: (item[0].component, item[0].parameter),
            )
        ),
    )
