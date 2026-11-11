# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Explicit native float CPU MoE lowering; see ``docs/moe-lowering.md``.

Consumes hidden states, router logits, and modern row-major expert projections.
This is not generic component dispatch or an automatically registered function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import onnx_ir as ir

from mobius.functions._lowering import Emitter, lower_nodes


@dataclass(frozen=True)
class _Config:
    experts: int
    hidden: int
    intermediate: int
    k: int
    activation: str
    normalize: bool
    alpha: float
    beta: float
    limit: float


def _error(message: str) -> ValueError:
    return ValueError(f"lower_moe: {message}")


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


def _validate(node: ir.Node) -> _Config:
    """Check the bounded native CPU ABI without changing the model."""
    if node.overload:
        raise _error("MoE overloads are not supported")
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
    output = node.outputs[0]
    # Native custom-op intermediates may be untyped/unshaped (a following
    # CastLike restores the public output type). Infer absent metadata from X,
    # but never overwrite known contradictions supplied by the caller.
    if output.type is not None and (
        not isinstance(output.type, ir.TensorType) or output.dtype != x.dtype
    ):
        raise _error("output metadata must match input shape and dtype")
    if output.shape is not None:
        ys = tuple(output.shape)
        if len(ys) != len(xs) or any(
            isinstance(b, int) and (b < 0 or (isinstance(a, int) and a != b))
            for a, b in zip(xs, ys, strict=False)
        ):
            raise _error("output metadata must match input shape and dtype")
    return _Config(
        experts, hidden, intermediate, k, activation, bool(normalize), alpha, beta, limit
    )


def _lower(node: ir.Node, c: _Config, prefix: str) -> tuple[list[ir.Node], ir.Value]:
    e = Emitter(prefix)
    x, router, w1, _, w2 = node.inputs[:5]
    assert x is not None and router is not None and w1 is not None and w2 is not None
    half = x.dtype == ir.DataType.FLOAT16

    # Arithmetic and scalar constants are FLOAT; half storage is modeled explicitly.
    def compute(value: ir.Value) -> ir.Value:
        return e.op("Cast", value, to=ir.DataType.FLOAT) if half else value

    def round_half(value: ir.Value) -> ir.Value:
        # Native CPU stores GEMM results, biased results, and activation in T.
        return compute(e.op("Cast", value, to=ir.DataType.FLOAT16)) if half else value

    # Native MoE softmaxes ALL E logits before selecting probabilities. Reversing
    # that probability axis reproduces descending (probability, expert_id) ties.
    probabilities = e.op("Softmax", compute(router), axis=1)
    reversed_probs = e.op(
        "Gather", probabilities, e.ints(list(range(c.experts - 1, -1, -1))), axis=1
    )
    top = ir.Node(
        "",
        "TopK",
        inputs=[reversed_probs, e.ints([c.k])],
        num_outputs=2,
        attributes=ir.convenience.convert_attributes({"axis": 1, "sorted": 1}),
    )
    for index, value in enumerate(top.outputs):
        value.name = f"{prefix}_top_{index}"
    e.nodes.append(top)
    routing = top.outputs[0]  # [T,k], FLOAT even for half inputs
    selected = e.op("Sub", e.ints([c.experts - 1]), top.outputs[1])
    if c.normalize:
        routing = e.op(
            "Mul",
            routing,
            e.op("Reciprocal", e.op("ReduceSum", routing, e.ints([1]), keepdims=1)),
        )
    active = e.op("Greater", routing, e.scalar(0))
    flat = e.op("Reshape", compute(x), e.ints([-1, c.hidden]))
    first = round_half(e.op("GroupedMatMul", flat, compute(w1), selected, transB=1))
    bias1 = _input(node, 3)
    if bias1 is not None:
        first = round_half(
            e.op("Add", first, e.op("Gather", compute(bias1), selected, axis=0))
        )
    if c.activation == "swiglu":
        gate = e.op("Gather", first, e.ints(list(range(0, 2 * c.intermediate, 2))), axis=2)
        up = e.op("Gather", first, e.ints(list(range(1, 2 * c.intermediate, 2))), axis=2)
        if math.isfinite(c.limit):
            gate = e.op("Min", gate, e.scalar(c.limit))
            up = e.op("Clip", up, e.scalar(-c.limit), e.scalar(c.limit))
        activated = e.op(
            "Mul",
            e.op("Mul", gate, e.op("Sigmoid", e.op("Mul", gate, e.scalar(c.alpha)))),
            e.op("Add", up, e.scalar(c.beta)),
        )
    elif c.activation == "silu":
        activated = e.op("Mul", first, e.op("Sigmoid", first))
    elif c.activation == "gelu":
        activated = e.op("Gelu", first, approximate="tanh")
    elif c.activation == "relu":
        activated = e.op("Relu", first)
    else:
        activated = first
    activated = round_half(activated)  # [T,k,I]
    # Every route has its own intermediate: flatten T*k, not just T.
    down = round_half(
        e.op(
            "GroupedMatMul",
            e.op("Reshape", activated, e.ints([-1, c.intermediate])),
            compute(w2),
            e.op("Reshape", selected, e.ints([-1, 1])),
            transB=1,
        )
    )
    down = e.op("Reshape", down, e.ints([-1, c.k, c.hidden]))
    bias2 = _input(node, 5)
    if bias2 is not None:
        down = round_half(e.op("Add", down, e.op("Gather", compute(bias2), selected, axis=0)))
    weighted = e.op("Mul", down, e.op("Unsqueeze", routing, e.ints([2])))
    # CPU never enqueues zero-weight routes; Where also masks their NaNs.
    weighted = e.op("Where", e.op("Unsqueeze", active, e.ints([2])), weighted, e.scalar(0))
    result = e.op("ReduceSum", weighted, e.ints([1]), keepdims=0)
    result = e.op("Reshape", result, e.op("Shape", x), allowzero=1)
    if half:
        result = e.op("Cast", result, to=ir.DataType.FLOAT16)
    result.type, result.shape = x.type, x.shape
    return e.nodes, result


def lower_moe(model: ir.Model) -> ir.Model:
    """Replace supported native ``com.microsoft::MoE`` calls in-place, returning model.

    Requires the default-off ``experimental_opset29`` permission. No-MoE models
    are untouched regardless of the flag. All calls and existing scopes are
    audited before mutation; unsupported ABIs raise ``ValueError`` atomically.
    Targets pinned CPU FLOAT/FLOAT16 semantics, not provider-universal parity.
    Requires GroupedMatMul-29 runtime support; enabling the flag does not invoke
    this pass, change native export defaults, or register a MoE function body.
    See ``docs/moe-lowering.md`` for shapes, arithmetic, routing, and exclusions.
    """
    return lower_nodes(model, op_type="MoE", validate=_validate, generate=_lower)
