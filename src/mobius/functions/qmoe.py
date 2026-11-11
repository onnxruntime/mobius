# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Opset-29 reference function for the raw, blockwise INT4 CPU QMoE ABI."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder
from onnxscript._internal.builder import build_function

from mobius.functions._moe_utils import validate_output_metadata


@dataclass(frozen=True)
class QMoEFunctionConfig:
    experts: int
    hidden: int
    intermediate: int
    block: int
    k: int
    normalize: bool
    alpha: float
    beta: float
    limit: float
    dtype: ir.DataType = ir.DataType.FLOAT
    zero_points1: bool = False
    zero_points2: bool = False
    router_weights: bool = False


def _input(node: ir.Node, index: int) -> ir.Value | None:
    return node.inputs[index] if index < len(node.inputs) else None


def _error(message: str) -> ValueError:
    return ValueError(f"QMoE fallback: {message}")


def _shape(value: ir.Value, name: str) -> tuple[int | ir.SymbolicDim, ...]:
    if value.shape is None:
        raise _error(f"{name} requires shape metadata")
    return tuple(value.shape)


def validate_call(node: ir.Node) -> QMoEFunctionConfig:
    """Validate the entire supported ABI without creating or changing any nodes."""
    allowed = {
        "k",
        "activation_type",
        "normalize_routing_weights",
        "use_sparse_mixer",
        "swiglu_fusion",
        "swiglu_limit",
        "activation_alpha",
        "activation_beta",
        "swiglu_limit_after_activation",
        "expert_weight_bits",
        "block_size",
        "quant_type",
        "weights_prepacked",
        "fc1_expert_weight_bits",
        "fc2_expert_weight_bits",
        "fc3_expert_weight_bits",
        "accuracy_level",
    }
    unknown = set(node.attributes) - allowed
    if unknown:
        raise _error(f"unsupported QMoE attributes: {sorted(unknown)}")
    for name, attr in node.attributes.items():
        expected = ir.AttributeType.INT
        if name in {"quant_type", "activation_type"}:
            expected = ir.AttributeType.STRING
        elif name in {"activation_alpha", "activation_beta", "swiglu_limit"}:
            expected = ir.AttributeType.FLOAT
        if attr.is_ref() or attr.type != expected:
            raise _error(f"attribute {name} must be a literal {expected.name}")
    attrs = node.attributes
    if attrs.get_string("quant_type", "int") != "int":
        raise _error("only quant_type='int' is supported; FP4 is not supported")
    if attrs.get_int("weights_prepacked", -1) != 0:
        raise _error("weights_prepacked must be 0")
    if attrs.get_int("accuracy_level", 0) not in (0, 1):
        raise _error("accuracy_level must be 0 or 1 (FLOAT GEMM)")
    bits = attrs.get_int("expert_weight_bits", 4)
    if bits != 4 or any(
        attrs.get_int(f"fc{i}_expert_weight_bits", bits) != 4 for i in (1, 2, 3)
    ):
        raise _error("only uniform INT4 expert_weight_bits are supported")
    if attrs.get_string("activation_type", "") != "swiglu":
        raise _error("activation_type must be 'swiglu'")
    if attrs.get_int("swiglu_fusion", 0) != 1:
        raise _error("only interleaved swiglu_fusion=1 is supported")
    if attrs.get_int("use_sparse_mixer", 0) != 0:
        raise _error("use_sparse_mixer is not supported")
    if attrs.get_int("swiglu_limit_after_activation", 0) != 0:
        raise _error("swiglu_limit_after_activation is not supported")
    normalize = attrs.get_int("normalize_routing_weights", 0)
    if normalize not in (0, 1):
        raise _error("normalize_routing_weights must be 0 or 1")
    alpha = attrs.get_float("activation_alpha", 1.0)
    beta = attrs.get_float("activation_beta", 0.0)
    limit = attrs.get_float("swiglu_limit", math.inf)
    if not math.isfinite(alpha) or not math.isfinite(beta) or math.isnan(limit) or limit < 0:
        raise _error("activation alpha/beta must be finite and swiglu_limit nonnegative")
    if len(node.outputs) != 1:
        raise _error("QMoE must have exactly one output")
    if len(node.inputs) > 15:
        raise _error("unsupported QMoE input 15 (extended ABI)")
    for index in (0, 1, 2, 3, 5, 6):
        if _input(node, index) is None:
            raise _error(f"required QMoE input {index} is absent")
    for index in (4, 7, 8, 9, 10, 13):
        if _input(node, index) is not None:
            raise _error(f"unsupported QMoE input {index} (bias, fc3, or extended ABI)")

    x, router = node.inputs[:2]
    assert x is not None and router is not None
    xs = _shape(x, "input")
    if len(xs) not in (2, 3) or not isinstance(xs[-1], int) or xs[-1] <= 0:
        raise _error("input must be rank 2 or 3 with a positive static hidden dimension")
    hidden = xs[-1]
    # The raw CPU kernel lifts half inputs/scales to FLOAT and keeps both GEMMs,
    # SwiGLU, and routing accumulation in FLOAT, casting only the final output.
    if x.dtype not in (ir.DataType.FLOAT, ir.DataType.FLOAT16):
        raise _error("only FLOAT and FLOAT16 hidden inputs are supported")
    for index in (0, 1, 3, 6, 14):
        value = _input(node, index)
        if value is not None and value.dtype != x.dtype:
            raise _error(
                f"input {index} must have the same FLOAT/FLOAT16 dtype as hidden input"
            )
    w1, w2 = node.inputs[2], node.inputs[5]
    assert w1 is not None and w2 is not None
    s1, s2 = _shape(w1, "fc1 weights"), _shape(w2, "fc2 weights")
    if (
        len(s1) != 3
        or len(s2) != 3
        or any(not isinstance(d, int) or d <= 0 for d in (*s1, *s2))
    ):
        raise _error("packed weights require positive static rank-3 shapes")
    experts, twice_intermediate, _ = s1
    assert isinstance(experts, int) and isinstance(twice_intermediate, int)
    intermediate = twice_intermediate // 2
    block = attrs.get_int("block_size", 0)
    if block < 16 or block & (block - 1):
        raise _error("block_size must be a power of two >= 16")
    if hidden % block or intermediate % block:
        raise _error("hidden and intermediate dimensions must be divisible by block_size")
    if s1 != (experts, 2 * intermediate, hidden // 2) or s2 != (
        experts,
        hidden,
        intermediate // 2,
    ):
        raise _error("only raw [E,2I,H/2] and [E,H,I/2] packed weights are supported")
    k = attrs.get_int("k", 0)
    if not 1 <= k <= experts:
        raise _error(f"k must be between 1 and num_experts ({experts})")
    rs = _shape(router, "router_probs")
    if len(rs) != 2 or rs[1] != experts:
        raise _error("router_probs must have shape [tokens, experts]")
    if all(isinstance(d, int) for d in xs[:-1]) and isinstance(rs[0], int):
        if math.prod(xs[:-1]) != rs[0]:
            raise _error("router_probs token dimension does not match input")
    rw = _input(node, 14)
    if rw is not None:
        rws = _shape(rw, "router_weights")
        # Symbol names are annotations, not proof of unequal runtime counts.
        # Feeds must have equal token counts even when metadata cannot prove it.
        if (
            len(rws) != 2
            or rws[1] != experts
            or (isinstance(rws[0], int) and isinstance(rs[0], int) and rws[0] != rs[0])
            or (
                isinstance(rws[0], int)
                and all(isinstance(d, int) for d in xs[:-1])
                and math.prod(xs[:-1]) != rws[0]
            )
        ):
            raise _error(
                "router_weights must be rank 2 with E experts and compatible token counts"
            )
    for weight_index, scale_index, zp_index, n, width in (
        (2, 3, 11, 2 * intermediate, hidden),
        (5, 6, 12, hidden, intermediate),
    ):
        weight, scale = node.inputs[weight_index], node.inputs[scale_index]
        assert weight is not None and scale is not None
        if weight.dtype != ir.DataType.UINT8:
            raise _error(f"input {weight_index} must be packed UINT8")
        blocks = width // block
        if _shape(scale, f"input {scale_index} scales") != (experts, n, blocks):
            raise _error(f"input {scale_index} scales must have shape [E,N,K/block_size]")
        zp = _input(node, zp_index)
        if zp is not None and (
            zp.dtype != ir.DataType.UINT8
            or _shape(zp, f"input {zp_index} zero points") != (experts, n, (blocks + 1) // 2)
        ):
            raise _error(f"input {zp_index} zero points must be UINT8 [E,N,ceil(nblocks/2)]")
    validate_output_metadata(node, x, caller="QMoE fallback")
    return QMoEFunctionConfig(
        experts,
        hidden,
        intermediate,
        block,
        k,
        bool(normalize),
        alpha,
        beta,
        limit,
        x.dtype,
        _input(node, 11) is not None,
        _input(node, 12) is not None,
        rw is not None,
    )


def qmoe(config: QMoEFunctionConfig) -> ir.Function:
    """Create a fresh opset-29 body retaining all fifteen native positional slots."""
    c = config
    if c.dtype not in (ir.DataType.FLOAT, ir.DataType.FLOAT16):
        raise _error("only FLOAT and FLOAT16 are supported")

    def body(
        op: OpBuilder,
        x,
        router,
        w1,
        scale1,
        bias1,
        w2,
        scale2,
        bias2,
        w3,
        scale3,
        bias3,
        zp1,
        zp2,
        zp3,
        rw,
    ):
        half = c.dtype == ir.DataType.FLOAT16

        def ints(values: list[int]):
            return op.Constant(value_ints=values)

        def scalar(value: float):
            return op.Constant(value_float=float(value))

        def compute(value):
            return op.Cast(value, to=ir.DataType.FLOAT) if half else value

        def unpack(packed, shape: list[int]):
            mask = op.Constant(value=ir.tensor(np.uint8(15)))
            shift = op.Constant(value=ir.tensor(np.uint8(4)))
            low = op.BitwiseAnd(packed, mask)
            high = op.BitShift(packed, shift, direction="RIGHT")
            pairs = op.Concat(
                op.Unsqueeze(low, ints([-1])), op.Unsqueeze(high, ints([-1])), axis=-1
            )
            return op.Reshape(pairs, ints(shape))

        def dequantize(packed, scale, zp, has_zp: bool, n: int, width: int):
            weights = op.Cast(unpack(packed, [c.experts, n, width]), to=ir.DataType.UINT4)
            blocks = width // c.block
            if has_zp:
                zeros = unpack(zp, [c.experts, n, -1])
                zeros = op.Slice(zeros, ints([0]), ints([blocks]), ints([2]))
            else:
                zeros = op.ConstantOfShape(
                    op.Shape(scale), value=ir.tensor(np.array([8], dtype=np.uint8))
                )
            zeros = op.Cast(zeros, to=ir.DataType.UINT4)
            return op.DequantizeLinear(weights, scale, zeros, axis=2, block_size=c.block)

        router, scale1, scale2 = compute(router), compute(scale1), compute(scale2)
        # QMoE selects logits first, with descending expert-ID ties.
        reversed_router = op.Gather(router, ints(list(range(c.experts - 1, -1, -1))), axis=1)
        logits, indices = op.TopK(reversed_router, ints([c.k]), axis=1, sorted=1, _outputs=2)
        selected = op.Sub(ints([c.experts - 1]), indices)
        if c.router_weights:
            routing = op.GatherElements(compute(rw), selected, axis=1)
            if c.normalize:
                total = op.ReduceSum(routing, ints([1]), keepdims=1)
                zero = op.Equal(total, scalar(0))
                safe_total = op.Where(zero, scalar(1), total)
                routing = op.Where(zero, scalar(0), op.Div(routing, safe_total))
        else:
            routing = op.Softmax(logits, axis=1)
        # CPU does not enqueue routes <=1e-8, including negative scales.
        active = op.Greater(routing, scalar(1e-8))
        routing = op.Where(active, routing, scalar(0))
        first_weights = dequantize(
            w1, scale1, zp1, c.zero_points1, 2 * c.intermediate, c.hidden
        )
        down_weights = dequantize(w2, scale2, zp2, c.zero_points2, c.hidden, c.intermediate)
        flat = op.Reshape(compute(x), ints([-1, c.hidden]))
        first = op.GroupedMatMul(flat, first_weights, selected, transB=1)  # [T,k,2I]
        gate = op.Gather(first, ints(list(range(0, 2 * c.intermediate, 2))), axis=2)
        up = op.Gather(first, ints(list(range(1, 2 * c.intermediate, 2))), axis=2)
        if math.isfinite(c.limit):
            gate = op.Min(gate, scalar(c.limit))
            up = op.Clip(up, scalar(-c.limit), scalar(c.limit))
        sigmoid = op.Sigmoid(op.Mul(gate, scalar(c.alpha)))
        activated = op.Mul(op.Mul(gate, sigmoid), op.Add(up, scalar(c.beta)))
        down = op.GroupedMatMul(
            op.Reshape(activated, ints([-1, c.intermediate])),
            down_weights,
            op.Reshape(selected, ints([-1, 1])),
            transB=1,
        )
        down = op.Reshape(down, ints([-1, c.k, c.hidden]))
        weighted = op.Mul(down, op.Unsqueeze(routing, ints([2])))
        # Where also masks NaNs from inactive expert computation.
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
                "fc1_scales",
                "fc1_bias",
                "fc2_experts_weights",
                "fc2_scales",
                "fc2_bias",
                "fc3_experts_weights",
                "fc3_scales",
                "fc3_bias",
                "fc1_zero_points",
                "fc2_zero_points",
                "fc3_zero_points",
                "router_weights",
            )
        ],
        domain="com.microsoft",
        name="QMoE",
        opset_imports={"": 29},
    )
