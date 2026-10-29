# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
    (``"CausalConvWithState"``, ``"LinearAttention"``).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import onnx_ir as ir

from mobius.functions.causal_conv import (
    causal_conv1d_with_state,
    causal_conv_nd_with_state,
)
from mobius.functions.linear_attention import (
    linear_attention,
)
from mobius.functions.matmul_block_quantized_fp4_weight import (
    matmul_block_quantized_fp4_weight,
)
from mobius.functions.matmul_nbits import (
    matmul_nbits,
)
from mobius.functions.packed_multi_head_attention import (
    packed_multi_head_attention,
)
from mobius.functions.skip_layer_normalization import (
    skip_layer_normalization,
    skip_simplified_layer_normalization,
)

_DOMAIN = "com.microsoft"
_FUNCTION_BODY_MARKER = "mobius.function_body"
_MATMUL_NBITS_ID = (_DOMAIN, "MatMulNBits", "")
_MATMUL_NBITS_DEFAULT_ZERO_POINTS_ID = (_DOMAIN, "MatMulNBits", "default_zero_points")

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
    (
        _DOMAIN,
        "MatMulBlockQuantizedFp4Weight",
        "",
    ): matmul_block_quantized_fp4_weight,
    _MATMUL_NBITS_ID: matmul_nbits,
    _MATMUL_NBITS_DEFAULT_ZERO_POINTS_ID: partial(matmul_nbits, has_zero_points=False),
    (_DOMAIN, "PackedMultiHeadAttention", ""): packed_multi_head_attention,
    (_DOMAIN, "SkipLayerNormalization", ""): skip_layer_normalization,
    (_DOMAIN, "SkipSimplifiedLayerNormalization", ""): skip_simplified_layer_normalization,
}


def get_function(op_id: ir.OperatorIdentifier) -> ir.Function | None:
    """Return a fresh ``ir.Function`` for *op_id*, or ``None``.

    Each call returns a **new** function object so that models cannot
    accidentally share mutable function bodies.  Rewrite passes (e.g.
    onnxscript's ``RewriteRuleSet.apply_to_model``) process function bodies
    in-place; sharing a single instance across models would corrupt it when
    one model's rewrite mutates the body.
    """
    builder = _FUNCTION_BUILDERS.get(op_id)
    if builder is None:
        return None
    function = builder()
    function.metadata_props[_FUNCTION_BODY_MARKER] = "1"
    return function


def _specialize_matmul_nbits_calls(model: ir.Model) -> None:
    existing = model.functions.get(_MATMUL_NBITS_ID)
    if existing is not None and existing.metadata_props.get(_FUNCTION_BODY_MARKER) != "1":
        return

    graphs = [model.graph, *(function.graph for function in model.functions.values())]
    for graph in graphs:
        for node in graph.all_nodes():
            if (
                node.op_identifier() != _MATMUL_NBITS_ID
                or node.attributes.get_int("bits", 4) != 4
            ):
                continue
            if len(node.inputs) < 4 or node.inputs[3] is None:
                node.overload = _MATMUL_NBITS_DEFAULT_ZERO_POINTS_ID[2]


def register_function_bodies(model: ir.Model) -> None:
    """Add standard-ONNX function bodies to *model* for all known ops.

    After calling this, :class:`onnx_ir.passes.common.InlinePass` can
    expand any of these ops by passing a suitable ``criteria`` predicate.

    Only registers functions for ops that are not already defined in the
    model (to avoid overwriting user-provided function bodies).

    Each call creates **fresh** ``ir.Function`` objects so that concurrent
    builds in different threads (or sequential builds in the same process)
    cannot corrupt each other's function bodies.

    MatMulNBits calls without zero points select an overload implementing the
    operator's default. Native op inputs and packed weights are left unchanged,
    and caller-provided MatMulNBits function definitions are preserved.
    """
    _specialize_matmul_nbits_calls(model)
    for op_id in _FUNCTION_BUILDERS:
        if op_id in model.functions:
            continue
        # The NVFP4 body contains a 256-entry E4M3 decode table and its native
        # node is introduced only while compressed weights are loaded, after
        # ordinary graph optimization. Avoid attaching it to unrelated models.
        if op_id == (_DOMAIN, "MatMulBlockQuantizedFp4Weight", "") and not any(
            node.domain == op_id[0] and node.op_type == op_id[1] for node in model.graph
        ):
            continue
        fn = get_function(op_id)
        if fn is not None:
            model.functions[op_id] = fn


__all__ = [
    "causal_conv1d_with_state",
    "causal_conv_nd_with_state",
    "get_function",
    "linear_attention",
    "matmul_block_quantized_fp4_weight",
    "matmul_nbits",
    "packed_multi_head_attention",
    "register_function_bodies",
    "skip_layer_normalization",
    "skip_simplified_layer_normalization",
]
