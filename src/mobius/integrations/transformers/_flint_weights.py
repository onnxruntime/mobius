# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bounded, fail-closed loading of Flint's individually stored expert matrices."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import onnx_ir as ir

from mobius._configs import FlintConfig
from mobius.integrations._weight_loading import (
    StreamingExpertBankSource,
    StreamingWeightPlan,
    StreamingWeightSource,
)


def build_flint_streaming_plan(
    config: FlintConfig,
    key_index: Mapping[str, tuple[str, list[int], str]],
    initializers: Mapping[str, ir.Value],
) -> StreamingWeightPlan:
    """Classify every checkpoint tensor before binding any lazy initializer.

    Expert banks are assembled one destination at a time. Direct bindings also
    preserve the graph's target dtype, including the publisher's FP32 LM head.
    """
    config.validate()
    assert config.num_local_experts is not None
    assert config.moe_intermediate_size is not None
    targets: dict[str, StreamingWeightSource | StreamingExpertBankSource] = {}
    consumed: set[str] = set()

    def source(name: str, shape: list[int]) -> StreamingWeightSource:
        metadata = key_index.get(name)
        if metadata is None:
            raise ValueError(f"Missing Flint checkpoint tensor {name!r}")
        if metadata[1] != shape:
            raise ValueError(
                f"Malformed Flint checkpoint tensor {name!r}: "
                f"expected {shape}, got {metadata[1]}"
            )
        if metadata[2] not in {"BF16", "F16", "F32"}:
            raise ValueError(f"Unsupported Flint checkpoint dtype {metadata[2]} for {name!r}")
        if name in consumed:
            raise ValueError(f"Duplicate Flint checkpoint binding {name!r}")
        consumed.add(name)
        return StreamingWeightSource(name)

    for name, initializer in initializers.items():
        if initializer.const_value is not None:
            continue
        if name.endswith((".fc1_experts_weights", ".fc2_experts_weights")):
            prefix, bank = name.rsplit(".", 1)
            projections = (
                ("gate_proj", "up_proj") if bank == "fc1_experts_weights" else ("down_proj",)
            )
            targets[name] = StreamingExpertBankSource(
                tuple(
                    tuple(
                        source(
                            f"{prefix}.experts.{expert}.{projection}.weight",
                            (
                                [config.hidden_size, config.moe_intermediate_size]
                                if projection == "down_proj"
                                else [config.moe_intermediate_size, config.hidden_size]
                            ),
                        )
                        for projection in projections
                    )
                    for expert in range(config.num_local_experts)
                )
            )
        else:
            if initializer.shape is None or any(
                not isinstance(dimension, int) for dimension in initializer.shape
            ):
                raise ValueError(f"Flint parameter {name!r} must have a concrete shape")
            targets[name] = source(name, cast(list[int], list(initializer.shape)))
    unmatched = set(key_index) - consumed
    if unmatched:
        raise ValueError(f"Unmapped Flint checkpoint tensor(s): {sorted(unmatched)[:5]}")
    return StreamingWeightPlan(
        targets,
        report={
            "source_weight_format": "floating",
            "storage_policy": "streamed-expert-banks",
            "accounted_source_tensor_count": len(consumed),
            "expert_count": config.num_local_experts,
            "experts_per_token": config.num_experts_per_tok,
            "lm_head_dtype": "float32",
            "quantization_preserved": False,
        },
    )
