# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""ONNX Function definitions for proposed linear attention operators.

These functions define reference implementations using standard ONNX ops
for the operators proposed in https://github.com/onnx/onnx/issues/7689.
They serve as:

1. **Semantic specifications** — precise mathematical definitions of each op
2. **Fallback implementations** — backends that don't have native kernels
   can expand the function body and execute via standard ops

Each function returns an ``ir.Function`` that can be attached to an
``ir.Model`` or used as a rewrite target.

Naming convention:
    Python factory functions are snake_case (e.g. ``causal_conv_nd_with_state``,
    ``linear_attention``) while the ir.Function op type strings are PascalCase
    (``"CausalConvWithState"``, ``"LinearAttention"``).  PascalCase aliases
    (``CausalConvWithState``) are provided for discoverability.
"""

from __future__ import annotations

from collections.abc import Callable

import onnx_ir as ir

from mobius.functions.causal_conv import (
    causal_conv1d_with_state,
    causal_conv_nd_with_state,
)
from mobius.functions.fused_matmul import (
    fused_matmul,
)
from mobius.functions.linear_attention import (
    linear_attention,
)
from mobius.functions.packed_multi_head_attention import (
    packed_multi_head_attention,
)
from mobius.functions.skip_layer_normalization import (
    skip_layer_normalization,
    skip_simplified_layer_normalization,
)

_DOMAIN = "com.microsoft"

# Registry mapping (domain, name, overload) → zero-arg factory function.
#
# This registry is for **config-independent** function bodies — ops whose
# standard-ONNX fallback doesn't depend on model shape or head counts.
# These are attached to every model by ``register_function_bodies()``.
#
# ``CausalConvWithState`` and ``LinearAttention`` are **parametric** factories
# (kernel_size, channels, num_heads are baked into the function body) and are
# registered per-model by ``tasks._base._register_linear_attention_functions``.
_FUNCTION_BUILDERS: dict[ir.OperatorIdentifier, Callable[[], ir.Function]] = {
    (_DOMAIN, "FusedMatMul", ""): fused_matmul,
    (_DOMAIN, "PackedMultiHeadAttention", ""): packed_multi_head_attention,
    (_DOMAIN, "SkipLayerNormalization", ""): skip_layer_normalization,
    (_DOMAIN, "SkipSimplifiedLayerNormalization", ""): skip_simplified_layer_normalization,
}

_cache: dict[ir.OperatorIdentifier, ir.Function] = {}


def get_function(op_id: ir.OperatorIdentifier) -> ir.Function | None:
    """Return the cached ``ir.Function`` for *op_id*, or ``None``."""
    if op_id not in _FUNCTION_BUILDERS:
        return None
    if op_id not in _cache:
        _cache[op_id] = _FUNCTION_BUILDERS[op_id]()
    return _cache[op_id]


def register_function_bodies(model: ir.Model) -> None:
    """Add standard-ONNX function bodies to *model* for all known ops.

    After calling this, :class:`onnx_ir.passes.common.InlinePass` can
    expand any of these ops by passing a suitable ``criteria`` predicate.

    Only registers functions for ops that are not already defined in the
    model (to avoid overwriting user-provided function bodies).
    """
    for op_id in _FUNCTION_BUILDERS:
        if op_id in model.functions:
            continue
        fn = get_function(op_id)
        if fn is not None:
            model.functions[op_id] = fn


__all__ = [
    "causal_conv1d_with_state",
    "causal_conv_nd_with_state",
    "fused_matmul",
    "get_function",
    "linear_attention",
    "packed_multi_head_attention",
    "register_function_bodies",
    "skip_layer_normalization",
    "skip_simplified_layer_normalization",
]
