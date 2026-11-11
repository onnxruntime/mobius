# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native CPU MoE reference function over modern row-major expert projections."""

from __future__ import annotations

import math
from dataclasses import dataclass

import onnx_ir as ir
from onnxscript import OpBuilder
from onnxscript._internal.builder import build_function

from mobius.functions._moe_utils import validate_output_metadata


@dataclass(frozen=True)
class MoEFunctionConfig:
    experts: int
    hidden: int
    intermediate: int
    k: int
    activation: str
    normalize: bool
    alpha: float
    beta: float
    limit: float
    dtype: ir.DataType = ir.DataType.FLOAT
    bias1: bool = False
    bias2: bool = False


def _error(message: str) -> ValueError:
    return ValueError(f"MoE fallback: {message}")


def _input(node: ir.Node, index: int) -> ir.Value | None:
    return node.inputs[index] if index < len(node.inputs) else None


def _shape(value: ir.Value, name: str) -> tuple[int | ir.SymbolicDim, ...]:
    if not isinstance(value.type, ir.TensorType):
        raise _error(f"{name} requires tensor type metadata")
    if value.shape is None:
        raise _error(f"{name} requires shape metadata")
    shape = tuple(value.shape)
    if any(isinstance(d, int) and d < 0 for d in shape):
        raise _error(f"{name} has a negative dimension")
    return shape


def validate_call(node: ir.Node) -> MoEFunctionConfig:
    """Check the bounded native CPU ABI without changing the model."""
    allowed = {
        "k": ir.AttributeType.INT,
        "activation_type": ir.AttributeType.STRING,
        "normalize_routing_weights": ir.AttributeType.INT,
        "use_sparse_mixer": ir.AttributeType.INT,
        "swiglu_fusion": ir.AttributeType.INT,
        "swiglu_limit_after_activation": ir.AttributeType.INT,
        "swiglu_limit": ir.AttributeType.FLOAT,
        "activation_alpha": ir.AttributeType.FLOAT,
        "activation_beta": ir.AttributeType.FLOAT,
    }
    if unknown := set(node.attributes) - set(allowed):
        raise _error(f"unsupported MoE attributes: {sorted(unknown)}")
    for name, attr in node.attributes.items():
        if attr.is_ref() or attr.type != allowed[name]:
            raise _error(f"attribute {name} must be a literal {allowed[name].name}")
    attrs = node.attributes
    activation = attrs.get_string("activation_type", "")
    if activation not in {"relu", "gelu", "silu", "identity", "swiglu"}:
        raise _error("activation_type must be relu, gelu, silu, identity, or swiglu")
    fusion = attrs.get_int("swiglu_fusion", 0)
    if fusion != (1 if activation == "swiglu" else 0):
        raise _error("swiglu_fusion must be 1 for swiglu and 0 otherwise")
    for name in ("use_sparse_mixer", "swiglu_limit_after_activation"):
        if attrs.get_int(name, 0) != 0:
            raise _error(f"{name} is not supported")
    normalize = attrs.get_int("normalize_routing_weights", 0)
    if normalize not in (0, 1):
        raise _error("normalize_routing_weights must be 0 or 1")
    alpha = attrs.get_float("activation_alpha", 1.0)
    beta = attrs.get_float("activation_beta", 0.0)
    limit = attrs.get_float("swiglu_limit", math.inf)
    if not math.isfinite(alpha) or not math.isfinite(beta) or math.isnan(limit) or limit < 0:
        raise _error("activation alpha/beta must be finite and swiglu_limit nonnegative")
    if len(node.outputs) != 1:
        raise _error("MoE must have exactly one output")
    if len(node.inputs) > 8:
        raise _error("extended MoE input slots are not supported")
    for index in (0, 1, 2, 4):
        if _input(node, index) is None:
            raise _error(f"required MoE input {index} is absent")
    for index in (6, 7):
        if _input(node, index) is not None:
            raise _error(f"FC3 input {index} is not supported")
    x, router, w1, _, w2 = node.inputs[:5]
    assert x is not None and router is not None and w1 is not None and w2 is not None
    xs = _shape(x, "input")
    if len(xs) not in (2, 3) or not isinstance(xs[-1], int) or xs[-1] <= 0:
        raise _error("input must be rank 2 or 3 with a positive static hidden dimension")
    hidden = xs[-1]
    if x.dtype not in (ir.DataType.FLOAT, ir.DataType.FLOAT16):
        raise _error("only FLOAT and FLOAT16 are supported")
    for index, value in enumerate(node.inputs):
        if value is not None:
            _shape(value, f"input {index}")
            if value.dtype != x.dtype:
                raise _error(f"input {index} must have the same FLOAT/FLOAT16 dtype as input")
    s1, s2 = _shape(w1, "fc1 weights"), _shape(w2, "fc2 weights")
    if (
        len(s1) != 3
        or len(s2) != 3
        or any(not isinstance(d, int) or d <= 0 for d in (*s1, *s2))
    ):
        raise _error("weights require positive static rank-3 shapes")
    experts, width, _ = s1
    intermediate = s2[2]
    assert isinstance(experts, int) and isinstance(width, int)
    assert isinstance(intermediate, int)
    # Pinned CheckInputs rejects legacy shapes. Square non-SwiGLU projections
    # are modern row-major, not an ambiguous ABI requiring special rejection.
    legacy = (hidden != intermediate and s2[1] == intermediate) or (
        hidden == intermediate and activation == "swiglu" and s1[1] == hidden
    )
    expected_width = intermediate * (2 if activation == "swiglu" else 1)
    if (
        legacy
        or s1 != (experts, expected_width, hidden)
        or s2 != (experts, hidden, intermediate)
    ):
        raise _error("only modern [E,N,H] and [E,H,I] weights are supported (not legacy)")
    for index, shape in ((3, (experts, width)), (5, (experts, hidden))):
        bias = _input(node, index)
        if bias is not None and _shape(bias, f"bias {index}") != shape:
            raise _error(f"bias {index} must have shape {shape}")
    k = attrs.get_int("k", 0)
    if not 1 <= k <= experts:
        raise _error(f"k must be between 1 and num_experts ({experts})")
    rs = _shape(router, "router_probs")
    if len(rs) != 2 or rs[1] != experts:
        raise _error("router_probs must have shape [tokens, experts]")
    if all(isinstance(d, int) for d in xs[:-1]) and isinstance(rs[0], int):
        if math.prod(xs[:-1]) != rs[0]:
            raise _error("router_probs token dimension does not match input")
    validate_output_metadata(node, x, caller="MoE fallback")
    return MoEFunctionConfig(
        experts,
        hidden,
        intermediate,
        k,
        activation,
        bool(normalize),
        alpha,
        beta,
        limit,
        x.dtype,
        _input(node, 3) is not None,
        _input(node, 5) is not None,
    )


def moe(config: MoEFunctionConfig) -> ir.Function:
    """Create a fresh opset-29 reference body with all eight native input slots."""
    c = config
    if c.dtype not in (ir.DataType.FLOAT, ir.DataType.FLOAT16):
        raise _error("only FLOAT and FLOAT16 are supported")

    def body(op: OpBuilder, x, router, w1, bias1, w2, bias2, w3, bias3):
        half = c.dtype == ir.DataType.FLOAT16

        def scalar(value: float):
            return op.Constant(value_float=float(value))

        def ints(values: list[int]):
            return op.Constant(value_ints=values)

        def compute(value):
            return op.Cast(value, to=ir.DataType.FLOAT) if half else value

        def round_half(value):
            # Native CPU stores GEMMs, biases, and activation separately in T.
            return compute(op.Cast(value, to=ir.DataType.FLOAT16)) if half else value

        # Full-E FLOAT softmax precedes TopK. Reverse to select higher IDs on ties.
        probabilities = op.Softmax(compute(router), axis=1)
        reversed_probs = op.Gather(
            probabilities, ints(list(range(c.experts - 1, -1, -1))), axis=1
        )
        routing, indices = op.TopK(reversed_probs, ints([c.k]), axis=1, sorted=1, _outputs=2)
        selected = op.Sub(ints([c.experts - 1]), indices)
        if c.normalize:
            routing = op.Mul(
                routing, op.Reciprocal(op.ReduceSum(routing, ints([1]), keepdims=1))
            )
        active = op.Greater(routing, scalar(0))
        flat = op.Reshape(compute(x), ints([-1, c.hidden]))
        first = round_half(op.GroupedMatMul(flat, compute(w1), selected, transB=1))
        if c.bias1:
            first = round_half(op.Add(first, op.Gather(compute(bias1), selected, axis=0)))
        if c.activation == "swiglu":
            gate = op.Gather(first, ints(list(range(0, 2 * c.intermediate, 2))), axis=2)
            up = op.Gather(first, ints(list(range(1, 2 * c.intermediate, 2))), axis=2)
            if math.isfinite(c.limit):
                gate = op.Min(gate, scalar(c.limit))
                up = op.Clip(up, scalar(-c.limit), scalar(c.limit))
            activated = op.Mul(
                op.Mul(gate, op.Sigmoid(op.Mul(gate, scalar(c.alpha)))),
                op.Add(up, scalar(c.beta)),
            )
        elif c.activation == "silu":
            activated = op.Mul(first, op.Sigmoid(first))
        elif c.activation == "gelu":
            activated = op.Gelu(first, approximate="tanh")
        elif c.activation == "relu":
            activated = op.Relu(first)
        else:
            activated = first
        activated = round_half(activated)  # [T,k,I]
        # Every selected route has its own intermediate: flatten T*k, not T.
        down = round_half(
            op.GroupedMatMul(
                op.Reshape(activated, ints([-1, c.intermediate])),
                compute(w2),
                op.Reshape(selected, ints([-1, 1])),
                transB=1,
            )
        )
        down = op.Reshape(down, ints([-1, c.k, c.hidden]))
        if c.bias2:
            down = round_half(op.Add(down, op.Gather(compute(bias2), selected, axis=0)))
        weighted = op.Mul(down, op.Unsqueeze(routing, ints([2])))
        # CPU never enqueues zero routes; Where also masks their NaNs.
        weighted = op.Where(op.Unsqueeze(active, ints([2])), weighted, scalar(0))
        result = op.ReduceSum(weighted, ints([1]), keepdims=0)
        result = op.Reshape(result, op.Shape(x), allowzero=1)
        return op.Cast(result, to=ir.DataType.FLOAT16) if half else result

    return build_function(
        body,
        [
            ir.Value(name=name)
            for name in (
                "input",
                "router_probs",
                "fc1_experts_weights",
                "fc1_experts_bias",
                "fc2_experts_weights",
                "fc2_experts_bias",
                "fc3_experts_weights",
                "fc3_experts_bias",
            )
        ],
        domain="com.microsoft",
        name="MoE",
        opset_imports={"": 29},
    )
