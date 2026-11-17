# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Parametric standard bodies for source-level component operators."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable

import onnx_ir as ir
from onnxscript._internal.builder import build_function

from mobius._constants import OPSET_VERSION
from mobius.functions.matmul_nbits import (
    FUNCTION_BODY_METADATA_KEY,
    FUNCTION_BODY_METADATA_VALUE,
)
from mobius.functions.normalization import offset_rms_norm, post_gated_rms_norm


def residual_stream_read(
    *, streams: int, hidden_size: int, opset_version: int = OPSET_VERSION
) -> ir.Function:
    def body(op, read, normalized):
        read = op.Reshape(op.Sigmoid(read), [0, 0, streams, hidden_size])
        values = op.Reshape(normalized, [0, 0, streams, hidden_size])
        weighted = op.Mul(read, values)
        return op.CastLike(
            op.ReduceMean(op.Cast(weighted, to=ir.DataType.FLOAT), [-2], keepdims=False),
            weighted,
        )

    return build_function(
        body,
        [ir.Value(name="read"), ir.Value(name="normalized")],
        domain="com.microsoft",
        name="ResidualStreamRead",
        opset_imports={"": opset_version},
    )


def full_cache_depthwise_conv1d(
    *, channels: int, kernel_size: int, activation: str, opset_version: int = OPSET_VERSION
) -> ir.Function:
    from mobius.components._activations import get_activation

    activate = get_activation(activation)

    def body(op, input_val, conv_state, weight):
        # Unlike CausalConvWithState, this ABI keeps K (not K-1) cached samples.
        history = op.Concat(conv_state, input_val, axis=2)
        conv_input = op.Slice(history, [1], [2**63 - 1], [2])
        output = op.Conv(conv_input, weight, group=channels, kernel_shape=[kernel_size])
        present = op.Slice(history, [-kernel_size], [2**63 - 1], [2])
        return activate(op, output), present

    return build_function(
        body,
        [ir.Value(name="input"), ir.Value(name="conv_state"), ir.Value(name="weight")],
        domain="com.microsoft",
        name="FullCacheDepthwiseConv1d",
        opset_imports={"": opset_version},
    )


def grouped_swiglu_experts(
    *,
    top_k: int,
    normalize: int,
    intermediate_size: int,
    hidden_size: int,
    opset_version: int = OPSET_VERSION,
) -> ir.Function:
    if opset_version != 29:
        raise ValueError(
            "GroupedSwiGLUExperts requires opset 29; rebuild with experimental_opset29"
        )

    def body(op, hidden, probabilities, gate_up_proj, down_proj):
        weights, selected = op.TopK(
            op.Cast(probabilities, to=ir.DataType.FLOAT), [top_k], axis=-1, _outputs=2
        )
        if normalize:
            weights = op.Div(weights, op.ReduceSum(weights, [-1], keepdims=1))
        weights = op.CastLike(weights, hidden)
        projected = op.GroupedMatMul(hidden, gate_up_proj, selected, transB=1)
        gate, up = op.Split(
            projected, [intermediate_size, intermediate_size], axis=-1, _outputs=2
        )
        gate_f32 = op.Cast(gate, to=ir.DataType.FLOAT)
        silu_f32 = op.Div(
            gate_f32, op.Add(op.Constant(value_float=1.0), op.Exp(op.Neg(gate_f32)))
        )
        activated = op.Mul(op.CastLike(silu_f32, hidden), up)
        down = op.GroupedMatMul(
            op.Reshape(activated, [-1, intermediate_size]),
            down_proj,
            op.Reshape(selected, [-1, 1]),
            transB=1,
        )
        down = op.Reshape(down, [-1, top_k, hidden_size])
        return op.ReduceSum(op.Mul(down, op.Unsqueeze(weights, [2])), [1], keepdims=0)

    return build_function(
        body,
        [
            ir.Value(name="hidden"),
            ir.Value(name="probabilities"),
            ir.Value(name="gate_up_proj"),
            ir.Value(name="down_proj"),
        ],
        domain="com.microsoft",
        name="GroupedSwiGLUExperts",
        opset_imports={"": opset_version},
    )


def packed_vision_rotary_attention(
    *, num_heads: int, head_dim: int, opset_version: int = OPSET_VERSION
) -> ir.Function:
    def body(op, qkv, cos, sin, cu_seqlens, hidden):
        q, k, v = op.Split(qkv, axis=-1, num_outputs=3, _outputs=3)
        q = op.Reshape(q, [0, num_heads, head_dim])
        k = op.Reshape(k, [0, num_heads, head_dim])
        cos = op.Unsqueeze(op.CastLike(cos, q), [1])
        sin = op.Unsqueeze(op.CastLike(sin, q), [1])
        half = head_dim // 2
        q1, q2 = op.Split(q, [half, half], axis=-1, _outputs=2)
        k1, k2 = op.Split(k, [half, half], axis=-1, _outputs=2)
        q_rot = op.Concat(
            op.Sub(op.Mul(cos, q1), op.Mul(sin, q2)),
            op.Add(op.Mul(sin, q1), op.Mul(cos, q2)),
            axis=-1,
        )
        k_rot = op.Concat(
            op.Sub(op.Mul(cos, k1), op.Mul(sin, k2)),
            op.Add(op.Mul(sin, k1), op.Mul(cos, k2)),
            axis=-1,
        )
        query = op.Reshape(q_rot, [0, -1])
        key = op.Reshape(k_rot, [0, -1])
        total_seq = op.Squeeze(op.Shape(hidden, start=0, end=1))
        positions = op.Range(op.Constant(value_int=0), total_seq, op.Constant(value_int=1))
        segments = op.Sub(
            op.ReduceSum(
                op.Cast(
                    op.GreaterOrEqual(
                        op.Unsqueeze(positions, [1]), op.Unsqueeze(cu_seqlens, [0])
                    ),
                    to=ir.DataType.INT64,
                ),
                [1],
                keepdims=False,
            ),
            op.Constant(value_int=1),
        )
        same = op.Equal(op.Unsqueeze(segments, [1]), op.Unsqueeze(segments, [0]))
        bias = op.Unsqueeze(
            op.Where(same, op.CastLike(0.0, query), op.CastLike(-10000.0, query)), [0, 1]
        )
        output = op.Attention(
            op.Unsqueeze(query, [0]),
            op.Unsqueeze(key, [0]),
            op.Unsqueeze(v, [0]),
            bias,
            kv_num_heads=num_heads,
            q_num_heads=num_heads,
            scale=head_dim**-0.5,
            _outputs=1,
        )
        return op.Squeeze(output, [0])

    return build_function(
        body,
        [ir.Value(name=name) for name in ("qkv", "cos", "sin", "cu_seqlens", "hidden")],
        domain="com.microsoft",
        name="PackedVisionRotaryAttention",
        opset_imports={"": opset_version},
    )


COMPONENT_FUNCTION_BUILDERS: dict[str, Callable[..., ir.Function]] = {
    "OffsetRMSNorm": offset_rms_norm,
    "GroupedOffsetRMSNorm": offset_rms_norm,
    "PostGatedRMSNorm": post_gated_rms_norm,
    "ResidualStreamRead": residual_stream_read,
    "FullCacheDepthwiseConv1d": full_cache_depthwise_conv1d,
    "GroupedSwiGLUExperts": grouped_swiglu_experts,
    "PackedVisionRotaryAttention": packed_vision_rotary_attention,
}


def is_component_function(function: ir.Function) -> bool:
    return (
        function.domain == "com.microsoft"
        and function.name in COMPONENT_FUNCTION_BUILDERS
        and function.metadata_props.get(FUNCTION_BODY_METADATA_KEY)
        == FUNCTION_BODY_METADATA_VALUE
    )


def register_component_bodies(model: ir.Model) -> None:
    """Specialize component calls atomically; never overwrite caller definitions."""
    additions: dict[ir.OperatorIdentifier, ir.Function] = {}
    calls: list[tuple[ir.Node, str]] = []
    roots = [model.graph, *(fn.graph for fn in model.functions.values())]
    for root in roots:
        for node in root.all_nodes():
            factory = COMPONENT_FUNCTION_BUILDERS.get(node.op_type)
            if node.domain != "com.microsoft" or factory is None:
                continue
            source = model.functions.get(node.op_identifier())
            if source is not None and not is_component_function(source):
                continue
            if node.overload and source is None:
                continue
            version = root.opset_imports.get("", root.opset_imports.get("ai.onnx"))
            if version is None:
                raise ValueError(
                    f"{node.op_type} body requires a containing standard opset import; "
                    "restore the model/function imports before registration."
                )
            if root.opset_imports.get(node.domain) != 1:
                raise ValueError(
                    f"{node.op_type} body requires com.microsoft opset 1; "
                    "restore the function-domain import before registration."
                )
            # ONNX FLOAT attributes serialize as float32; identities must remain
            # stable when a saved model is registered again.
            config = {
                name: struct.unpack("f", struct.pack("f", attr.as_float()))[0]
                if attr.type == ir.AttributeType.FLOAT
                else attr.value
                for name, attr in node.attributes.items()
            }
            identity = json.dumps([version, config], sort_keys=True, allow_nan=False)
            overload = "mobius_" + hashlib.sha256(identity.encode()).hexdigest()
            identifier = (node.domain, node.op_type, overload)
            existing = model.functions.get(identifier)
            if existing is not None and (
                not is_component_function(existing)
                or existing.metadata_props.get("mobius.function_body.config") != identity
            ):
                raise ValueError(
                    f"{node.op_type} body conflicts with caller-owned definition "
                    f"{identifier!r}; use a distinct identifier for the caller function."
                )
            if existing is None and identifier not in additions:
                function = factory(**config, opset_version=version)
                function.attributes.update(node.attributes)
                function.overload = overload
                function.metadata_props[FUNCTION_BODY_METADATA_KEY] = (
                    FUNCTION_BODY_METADATA_VALUE
                )
                function.metadata_props["mobius.function_body.config"] = identity
                additions[identifier] = function
            calls.append((node, overload))
    model.functions.update(additions)
    for node, overload in calls:
        node.overload = overload
