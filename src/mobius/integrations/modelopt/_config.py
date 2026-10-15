# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Strict mixed FP8/NVFP4 weight-only reconstruction plans for ModelOpt exports."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Literal


@dataclasses.dataclass(frozen=True)
class ModelOptConfig:
    """Exact source-module formats, not an affine INT4 quantization config.

    Only explicit module targets with scalar-scaled E4M3 or block-16 E2M1
    weights are accepted. Input FP8 scales are validated and accounted for,
    but dense reconstruction does not preserve activation quantization.
    """

    modules: Mapping[str, Literal["modelopt_fp8", "modelopt_nvfp4"]]
    activation_modules: frozenset[str]
    kv_cache_fp8: bool = False

    @classmethod
    def parse(cls, value: object) -> ModelOptConfig:
        """Reject unknown layouts instead of inferring semantics from bit width."""
        if not isinstance(value, Mapping) or value.get("quant_method") != "modelopt":
            raise ValueError("ModelOpt reconstruction requires quant_method='modelopt'")
        unknown = set(value) - {
            "quant_method",
            "config_groups",
            "ignore",
            "quant_algo",
            "quantized_layers",
            "kv_cache_scheme",
            "producer",
        }
        if unknown:
            raise NotImplementedError(
                f"Unsupported ModelOpt configuration fields: {sorted(unknown)}"
            )
        groups = value.get("config_groups")
        if not isinstance(groups, Mapping) or not groups:
            raise ValueError("ModelOpt reconstruction requires explicit config_groups")
        modules: dict[str, Literal["modelopt_fp8", "modelopt_nvfp4"]] = {}
        activation_modules: set[str] = set()
        for group_name, group in groups.items():
            if not isinstance(group, Mapping):
                raise TypeError(f"Malformed ModelOpt group {group_name!r}")
            unknown = set(group) - {
                "weights",
                "input_activations",
                "output_activations",
                "targets",
            }
            if unknown:
                raise NotImplementedError(
                    f"Unsupported ModelOpt group fields: {sorted(unknown)}"
                )
            weights = group.get("weights")
            if not isinstance(weights, Mapping) or weights.get("dynamic") is not False:
                raise ValueError(f"ModelOpt weights in {group_name!r} must be static")
            if weights == {"dynamic": False, "num_bits": 8, "type": "float"}:
                mode: Literal["modelopt_fp8", "modelopt_nvfp4"] = "modelopt_fp8"
            elif weights == {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": 16,
            }:
                mode = "modelopt_nvfp4"
            else:
                raise NotImplementedError(
                    f"Unsupported ModelOpt weight layout in {group_name!r}: {weights!r}"
                )
            activation = group.get("input_activations")
            if activation is not None and (
                mode != "modelopt_fp8"
                or activation != {"dynamic": False, "num_bits": 8, "type": "float"}
                or not isinstance(activation, Mapping)
                or activation.get("dynamic") is not False
            ):
                raise NotImplementedError(
                    f"Unsupported ModelOpt activation layout in {group_name!r}"
                )
            if group.get("output_activations") is not None:
                raise NotImplementedError(
                    "ModelOpt output activation quantization unsupported"
                )
            targets = group.get("targets")
            if not isinstance(targets, list) or not targets:
                raise ValueError(f"ModelOpt group {group_name!r} needs explicit targets")
            for target in targets:
                if (
                    not isinstance(target, str)
                    or not target
                    or any(
                        not (part.isidentifier() or part.isdecimal())
                        for part in target.split(".")
                    )
                ):
                    raise ValueError(
                        f"ModelOpt target must be an exact module path: {target!r}"
                    )
                if target in modules:
                    raise ValueError(f"Duplicate ModelOpt target {target!r}")
                modules[target] = mode
                if activation is not None:
                    activation_modules.add(target)
        ignored = value.get("ignore", [])
        if not isinstance(ignored, list) or any(not isinstance(x, str) for x in ignored):
            raise ValueError("ModelOpt ignore must contain module paths")
        if set(ignored) & modules.keys():
            raise ValueError("ModelOpt targets must not also be ignored")
        algo = value.get("quant_algo")
        if algo is not None and (
            algo != "MIXED_PRECISION"
            or set(modules.values()) != {"modelopt_fp8", "modelopt_nvfp4"}
        ):
            raise NotImplementedError(f"Unsupported ModelOpt quant_algo {algo!r}")
        legacy = value.get("quantized_layers")
        if legacy is not None:
            expected = {
                name: (
                    {"quant_algo": "FP8"}
                    if mode == "modelopt_fp8"
                    else {"quant_algo": "W4A16_NVFP4", "group_size": 16}
                )
                for name, mode in modules.items()
            }
            if legacy != expected:
                raise ValueError("ModelOpt quantized_layers and config_groups disagree")
        cache_scheme = value.get("kv_cache_scheme")
        if cache_scheme is not None and (
            not isinstance(cache_scheme, Mapping)
            or cache_scheme != {"dynamic": False, "num_bits": 8, "type": "float"}
            or cache_scheme.get("dynamic") is not False
        ):
            raise NotImplementedError(
                "Only static E4M3 ModelOpt KV-cache scales are supported"
            )
        producer = value.get("producer")
        if producer is not None and (
            not isinstance(producer, Mapping)
            or set(producer) != {"name", "version"}
            or producer["name"] != "modelopt"
            or not isinstance(producer["version"], str)
            or not producer["version"]
        ):
            raise ValueError("Malformed ModelOpt producer metadata")
        return cls(modules, frozenset(activation_modules), cache_scheme is not None)

    @property
    def source_format(self) -> str:
        """Describe the declared numeric formats rather than guessing from model name."""
        modes = set(self.modules.values())
        if modes == {"modelopt_fp8", "modelopt_nvfp4"}:
            return "modelopt-mixed-fp8-nvfp4"
        if modes == {"modelopt_fp8"}:
            return "modelopt-fp8"
        return "modelopt-nvfp4"
