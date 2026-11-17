# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standard-ONNX function bodies for contrib operators.

These factories define reference implementations for attention, convolution,
normalization, quantized matrix multiplication, and mixture-of-experts operators.
They use standard ONNX ops and serve as:

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

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping
from functools import partial

import onnx_ir as ir

from mobius._constants import OPSET_VERSION
from mobius._flags import flags
from mobius.functions._moe_utils import float32_parameters
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
    FUNCTION_BODY_METADATA_KEY,
    FUNCTION_BODY_METADATA_VALUE,
    matmul_nbits,
)
from mobius.functions.moe import MoEFunctionConfig, moe
from mobius.functions.moe import validate_call as validate_moe_call
from mobius.functions.packed_multi_head_attention import (
    packed_multi_head_attention,
)
from mobius.functions.qmoe import QMoEFunctionConfig, qmoe
from mobius.functions.qmoe import validate_call as validate_qmoe_call
from mobius.functions.skip_layer_normalization import (
    skip_layer_normalization,
    skip_simplified_layer_normalization,
)

_DOMAIN = "com.microsoft"
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
_FUNCTION_BUILDERS: dict[ir.OperatorIdentifier, Callable[..., ir.Function]] = {
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


def get_function(
    op_id: ir.OperatorIdentifier, *, opset_version: int = OPSET_VERSION
) -> ir.Function | None:
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
    function = builder(opset_version=opset_version)
    function.metadata_props[FUNCTION_BODY_METADATA_KEY] = FUNCTION_BODY_METADATA_VALUE
    return function


def _specialize_matmul_nbits_calls(model: ir.Model) -> None:
    existing = model.functions.get(_MATMUL_NBITS_ID)
    if (
        existing is not None
        and existing.metadata_props.get(FUNCTION_BODY_METADATA_KEY)
        != FUNCTION_BODY_METADATA_VALUE
    ):
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


_MOE_CONFIG_METADATA = "mobius.function_body.config"


def _core_opset_version(opset_imports: Mapping[str, int]) -> int | None:
    """Resolve the standard-domain aliases without changing their imports."""
    version = opset_imports.get("")
    alias_version = opset_imports.get("ai.onnx")
    if version is not None and alias_version is not None and version != alias_version:
        raise ValueError(
            "Conflicting core-domain opset imports: "
            f"'' imports {version}, 'ai.onnx' imports {alias_version}"
        )
    return version if version is not None else alias_version


def _config_identity(config: MoEFunctionConfig | QMoEFunctionConfig) -> str:
    # Encode floats as exact strings, including +inf (the native default limit).
    # RFC8785 and strict JSON numbers intentionally cannot represent infinity.
    fields = dataclasses.asdict(config)
    fields.update(
        zip(
            ("alpha", "beta", "limit"),
            float32_parameters(config.alpha, config.beta, config.limit),
            strict=True,
        )
    )
    return json.dumps(
        {
            key: value.hex() if isinstance(value, float) else value
            for key, value in fields.items()
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _register_moe_bodies(model: ir.Model) -> None:
    if not flags.experimental_opset29:
        raise ValueError("MoE fallback requires experimental_opset29=True")
    if _core_opset_version(model.graph.opset_imports) != 29:
        raise ValueError("MoE fallback requires opset 29; rebuild under experimental opset29")
    specializations: dict[
        ir.OperatorIdentifier, tuple[MoEFunctionConfig | QMoEFunctionConfig, str]
    ] = {}
    calls: list[tuple[ir.Node, str]] = []
    # Validate every target before any function construction.
    roots = [model.graph, *(fn.graph for fn in model.functions.values())]
    for root in roots:
        for node in root.all_nodes():
            if node.domain != _DOMAIN or node.op_type not in {"MoE", "QMoE"}:
                continue
            source = model.functions.get(node.op_identifier())
            if (
                source is not None
                and source.metadata_props.get(FUNCTION_BODY_METADATA_KEY)
                != FUNCTION_BODY_METADATA_VALUE
            ):
                raise ValueError(f"{node.op_type} fallback: caller-owned source function")
            # Unknown user overloads are not native operator calls.
            if node.overload and source is None:
                continue
            if _core_opset_version(root.opset_imports) != 29:
                raise ValueError(
                    f"{node.op_type} fallback: containing function must use opset 29; "
                    "rebuild under experimental opset29"
                )
            config = (
                validate_moe_call(node) if node.op_type == "MoE" else validate_qmoe_call(node)
            )
            identity = _config_identity(config)
            overload = "mobius_" + hashlib.sha256(identity.encode()).hexdigest()
            identifier = (_DOMAIN, node.op_type, overload)
            destination = model.functions.get(identifier)
            if destination is not None and (
                destination.metadata_props.get(FUNCTION_BODY_METADATA_KEY)
                != FUNCTION_BODY_METADATA_VALUE
                or destination.metadata_props.get(_MOE_CONFIG_METADATA) != identity
                or _core_opset_version(destination.opset_imports) != 29
            ):
                raise ValueError(
                    f"{node.op_type} fallback: caller-owned or conflicting destination"
                )
            if (
                source is not None
                and node.overload
                and (
                    source.metadata_props.get(_MOE_CONFIG_METADATA) != identity
                    or _core_opset_version(source.opset_imports) != 29
                )
            ):
                raise ValueError(f"{node.op_type} fallback: specialization config mismatch")
            if identifier in specializations and specializations[identifier][1] != identity:
                raise ValueError(
                    f"{node.op_type} fallback: conflicting destination specialization"
                )
            if destination is None:
                specializations.setdefault(identifier, (config, identity))
            calls.append((node, overload))

    # Build all unique bodies without changing the model, so build failures are atomic.
    additions: dict[ir.OperatorIdentifier, ir.Function] = {}
    for identifier, (config, identity) in specializations.items():
        function = moe(config) if isinstance(config, MoEFunctionConfig) else qmoe(config)
        function.overload = identifier[2]
        function.metadata_props[FUNCTION_BODY_METADATA_KEY] = FUNCTION_BODY_METADATA_VALUE
        function.metadata_props[_MOE_CONFIG_METADATA] = identity
        additions[identifier] = function

    # Commit only after every target and body has succeeded.
    model.functions.update(additions)
    for node, overload in calls:
        node.overload = overload
        output, x = node.outputs[0], node.inputs[0]
        assert x is not None
        if output.type is None:
            output.type = x.type
        if output.shape is None:
            output.shape = x.shape.copy() if x.shape is not None else None


def register_function_bodies(model: ir.Model, *, moe_fallback: bool = False) -> None:
    """Add standard-ONNX function bodies to *model* for all known ops.

    After calling this, :class:`onnx_ir.passes.common.InlinePass` can
    expand any of these ops by passing a suitable ``criteria`` predicate.

    Ordinary bodies are registered only when their identifiers are not already
    defined, preserving user-provided bodies. The optional MoE path additionally
    specializes calls as described below.

    Newly registered bodies are **fresh** ``ir.Function`` objects so that
    concurrent builds in different threads (or sequential builds in the same
    process) cannot corrupt each other's function bodies.

    MatMulNBits calls without zero points select an overload implementing the
    operator's default. Native op inputs and packed weights are left unchanged,
    and caller-provided MatMulNBits function definitions are preserved.

    With ``moe_fallback=True``, ``flags.experimental_opset29`` must be enabled
    and the model root must already use opset 29, even without native targets;
    rebuild existing opset-24 models rather than converting their imports.
    Native MoE/QMoE calls receive
    configuration-specialized overloads and matching function bodies, without
    changing their inputs or attributes. Caller-owned or conflicting source
    and destination functions are rejected rather than overwritten.

    All MoE/QMoE targets are validated and all needed bodies are constructed
    before changing their call overloads, output annotations, or registered
    functions. This target-specific guarantee does not cover subsequent failures
    in the ordinary function registry.
    """
    if moe_fallback:
        _register_moe_bodies(model)
    opset_version = _core_opset_version(model.graph.opset_imports)
    if opset_version is None:
        opset_version = OPSET_VERSION
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
        fn = get_function(op_id, opset_version=opset_version)
        if fn is not None:
            model.functions[op_id] = fn


__all__ = [
    "causal_conv1d_with_state",
    "causal_conv_nd_with_state",
    "get_function",
    "linear_attention",
    "MoEFunctionConfig",
    "QMoEFunctionConfig",
    "moe",
    "qmoe",
    "matmul_block_quantized_fp4_weight",
    "matmul_nbits",
    "packed_multi_head_attention",
    "register_function_bodies",
    "skip_layer_normalization",
    "skip_simplified_layer_normalization",
]
