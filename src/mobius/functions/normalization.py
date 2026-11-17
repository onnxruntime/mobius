# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standard bodies preserving offset and post-gated RMSNorm rounding order."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder

from mobius._function_builder import emit_standard_function


def offset_rms_norm(
    op: OpBuilder,
    hidden_states: ir.Value,
    weight: ir.Value,
    *,
    epsilon: float,
    groups: int = 1,
    group_size: int = 0,
) -> ir.Value:
    """Normalize/scale in float32, then round to the activation dtype."""

    def body(op, hidden_states, weight):
        hidden_f32 = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        if group_size:
            original_shape = op.Shape(hidden_states)
            hidden_f32 = op.Reshape(hidden_f32, [-1, groups, group_size])
        variance = op.ReduceMean(op.Mul(hidden_f32, hidden_f32), [-1], keepdims=True)
        normalized = op.Mul(hidden_f32, op.Reciprocal(op.Sqrt(op.Add(variance, epsilon))))
        scale = op.Add(op.Cast(weight, to=ir.DataType.FLOAT), op.Constant(value_float=1.0))
        if group_size:
            scale = op.Reshape(scale, [groups, group_size])
        result = op.Mul(normalized, scale)
        if group_size:
            result = op.Reshape(result, original_shape)
        return op.CastLike(result, hidden_states)

    return emit_standard_function(
        op,
        "GroupedOffsetRMSNorm" if group_size else "OffsetRMSNorm",
        body,
        [hidden_states, weight],
        specialization=(epsilon, groups, group_size),
    )


def post_gated_rms_norm(
    op: OpBuilder,
    hidden_states: ir.Value,
    weight: ir.Value,
    gate: ir.Value,
    *,
    epsilon: float,
    activation: str,
) -> ir.Value:
    """Round normalized activations before native-dtype scaling and FP32 gating."""

    def body(op, hidden_states, weight, gate):
        hidden_f32 = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        variance = op.ReduceMean(op.Mul(hidden_f32, hidden_f32), [-1], keepdims=True)
        normalized_f32 = op.Mul(hidden_f32, op.Reciprocal(op.Sqrt(op.Add(variance, epsilon))))
        normalized = op.CastLike(normalized_f32, hidden_states)
        weighted = op.Mul(normalized, weight)
        gate_f32 = op.Cast(gate, to=ir.DataType.FLOAT)
        activated = op.Sigmoid(gate_f32) if activation == "sigmoid" else op.Swish(gate_f32)
        return op.CastLike(
            op.Mul(op.Cast(weighted, to=ir.DataType.FLOAT), activated), hidden_states
        )

    return emit_standard_function(
        op,
        "PostGatedRMSNorm",
        body,
        [hidden_states, weight, gate],
        specialization=(epsilon, activation),
    )
