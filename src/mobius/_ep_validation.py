# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""EP validation: reject unsupported model + execution provider combinations.

Some model architectures are incompatible with certain execution providers.
For example, Mixture-of-Experts (MoE) models rely on control flow (If/Loop)
or dynamic routing that DML cannot handle. This module provides a registry-level
check that fails fast before graph construction begins.
"""

from __future__ import annotations

# Known incompatible (model_type, ep) pairs with explanation.
# This is intentionally a simple allow-list/deny-list rather than
# encoding all constraints inside ModelRegistration, because:
#   1. The exclusion set is small and grows slowly.
#   2. New EPs are added rarely.
#   3. A flat table is easier for AI agents and humans to audit.
_UNSUPPORTED_COMBOS: dict[tuple[str, str], str] = {
    # MoE models use dynamic Top-K routing + Scatter that DML cannot lower.
    ("mixtral", "dml"): "Mixtral (MoE) uses dynamic expert routing unsupported by DML",
    ("phimoe", "dml"): "PhiMoE uses dynamic expert routing unsupported by DML",
    ("qwen2_moe", "dml"): "Qwen2MoE uses dynamic expert routing unsupported by DML",
    ("qwen3_moe", "dml"): "Qwen3MoE uses dynamic expert routing unsupported by DML",
    ("dbrx", "dml"): "DBRX (MoE) uses dynamic expert routing unsupported by DML",
    ("jamba", "dml"): "Jamba (MoE + Mamba) is unsupported by DML",
    ("jetmoe", "dml"): "JetMoE uses dynamic expert routing unsupported by DML",
    ("olmoe", "dml"): "OLMoE uses dynamic expert routing unsupported by DML",
    ("deepseek_v3", "dml"): "DeepSeek-V3 (MoE) uses dynamic expert routing unsupported by DML",
    # MoE models on WebGPU (same issue — control flow + dynamic routing)
    ("mixtral", "webgpu"): "Mixtral (MoE) uses dynamic expert routing unsupported by WebGPU",
    ("phimoe", "webgpu"): "PhiMoE uses dynamic expert routing unsupported by WebGPU",
    ("qwen2_moe", "webgpu"): "Qwen2MoE uses dynamic expert routing unsupported by WebGPU",
    ("qwen3_moe", "webgpu"): "Qwen3MoE uses dynamic expert routing unsupported by WebGPU",
    ("dbrx", "webgpu"): "DBRX (MoE) uses dynamic expert routing unsupported by WebGPU",
    ("jamba", "webgpu"): "Jamba (MoE + Mamba) is unsupported by WebGPU",
    ("jetmoe", "webgpu"): "JetMoE uses dynamic expert routing unsupported by WebGPU",
    ("olmoe", "webgpu"): "OLMoE uses dynamic expert routing unsupported by WebGPU",
    (
        "deepseek_v3",
        "webgpu",
    ): "DeepSeek-V3 (MoE) uses dynamic expert routing unsupported by WebGPU",
    # SSM/hybrid models on WebGPU (Mamba layers need Scan/Loop)
    ("mamba", "webgpu"): "Mamba SSM layers use Scan op unsupported by WebGPU",
    ("mamba2", "webgpu"): "Mamba2 SSM layers use Scan op unsupported by WebGPU",
    ("jamba", "trt-rtx"): "Jamba (MoE + Mamba) hybrid is unsupported by TRT-RTX",
}

# Canonical set of known EPs for input validation
KNOWN_EPS: frozenset[str] = frozenset({"cpu", "cuda", "dml", "webgpu", "trt-rtx"})


def validate_ep_support(model_type: str, ep: str) -> None:
    """Validate that *model_type* is compatible with *ep*.

    Raises:
        ValueError: If the ``(model_type, ep)`` combination is known
            to be unsupported, with an explanation of why.
        ValueError: If *ep* is not a recognised execution provider.
    """
    if ep not in KNOWN_EPS:
        raise ValueError(f"Unknown execution provider {ep!r}. Supported: {sorted(KNOWN_EPS)}")

    key = (model_type, ep)
    if key in _UNSUPPORTED_COMBOS:
        reason = _UNSUPPORTED_COMBOS[key]
        raise ValueError(
            f"Model type {model_type!r} is not compatible with "
            f"execution provider {ep!r}: {reason}"
        )
