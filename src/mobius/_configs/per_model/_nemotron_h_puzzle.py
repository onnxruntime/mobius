# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Per-block Nemotron-H Puzzle configuration, matching NemotronHPuzzleConfig."""

from __future__ import annotations

import copy
import dataclasses
import math

import onnx_ir as ir

from mobius._configs._base import (
    NemotronHConfig,
    _as_attribute_config,
    _resolve_dtype_value,
    _shallow_fields,
)


@dataclasses.dataclass(frozen=True)
class NemotronHMoEBlockConfig:
    """Resolved routed-expert topology for one Puzzle block."""

    num_experts: int
    top_k: int
    intermediate_size: int


@dataclasses.dataclass
class NemotronHPuzzleConfig(NemotronHConfig):
    """Heterogeneous Nemotron-H decoder configuration.

    ``block_configs`` is authoritative for the layer schedule. Each MoE field
    must be specified for every MoE block or inherited globally for every block,
    as in the pinned HuggingFace ``NemotronHPuzzleConfig``.
    """

    mamba_time_step_min: float = 0.0
    mamba_ssm_dtype: ir.DataType | None = ir.DataType.FLOAT
    moe_block_configs: tuple[NemotronHMoEBlockConfig | None, ...] = ()

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> NemotronHPuzzleConfig:
        config = _as_attribute_config(config)
        blocks = getattr(config, "block_configs", None)
        if not blocks:
            raise ValueError("Nemotron-H Puzzle requires nonempty block_configs")
        blocks = [_as_attribute_config(block) for block in blocks]
        kinds = [getattr(block, "block_type", None) for block in blocks]
        if any(kind not in {"mamba", "attention", "moe"} for kind in kinds):
            raise ValueError("Nemotron-H Puzzle has an unsupported block_type")

        field_aliases = {
            "n_routed_experts": "num_local_experts",
            "num_experts_per_tok": "num_experts_per_tok",
            "moe_intermediate_size": "moe_intermediate_size",
        }
        for name in field_aliases:
            explicit = {
                getattr(block, name, None) is not None
                for block, kind in zip(blocks, kinds)
                if kind == "moe"
            }
            if len(explicit) > 1:
                raise ValueError(
                    f"Puzzle field {name!r} must be set for all MoE blocks or none"
                )

        # HF removes global blockwise members after resolving the block list.
        # Supply only the authoritative schedule to the common config reader.
        common = copy.copy(config)
        common.layers_block_type = kinds
        base = NemotronHConfig.from_transformers(common, parent_config)
        limits = getattr(config, "time_step_limit", None)
        if limits is not None:
            lower, upper = limits
            upper = _as_attribute_config(upper)
            if getattr(upper, "__float__", None) == "Infinity":
                upper = math.inf
            if lower != 0 or upper != math.inf:
                raise NotImplementedError(
                    "Puzzle non-default time_step_limit requires separate prefill/decode semantics"
                )
        fields = _shallow_fields(base)
        # time_step_min initializes dt_bias; it is not a runtime lower bound.
        fields["mamba_time_step_min"] = 0.0
        ssm_dtype = _resolve_dtype_value(getattr(config, "mamba_ssm_cache_dtype", "float32"))
        if ssm_dtype not in {ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16}:
            raise ValueError("Puzzle mamba_ssm_cache_dtype must be a floating-point dtype")
        fields["mamba_ssm_dtype"] = ssm_dtype
        resolved: list[NemotronHMoEBlockConfig | None] = []
        for block, kind in zip(blocks, kinds):
            if kind != "moe":
                resolved.append(None)
                continue
            values = []
            for source_name, target_name in field_aliases.items():
                value = getattr(block, source_name, None)
                if value is None:
                    value = getattr(config, source_name, None)
                if value is None:
                    value = getattr(base, target_name)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"Puzzle {source_name} must be a positive integer")
                values.append(value)
            topology = NemotronHMoEBlockConfig(*values)
            if topology.top_k > topology.num_experts:
                raise ValueError("Puzzle num_experts_per_tok exceeds n_routed_experts")
            resolved.append(topology)
        return cls(**fields, moe_block_configs=tuple(resolved))

    def for_layer(self, layer_index: int) -> NemotronHConfig:
        """Resolve one layer without overwriting global or neighboring topology."""
        if len(self.moe_block_configs) != self.num_hidden_layers:
            raise ValueError("Puzzle moe_block_configs must align with num_hidden_layers")
        topology = self.moe_block_configs[layer_index]
        if topology is None:
            return self
        return dataclasses.replace(
            self,
            num_local_experts=topology.num_experts,
            num_experts_per_tok=topology.top_k,
            moe_intermediate_size=topology.intermediate_size,
        )
