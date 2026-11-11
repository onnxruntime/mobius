# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Explicit opset-29 lowering of the raw, blockwise INT4 CPU QMoE ABI.

See ``docs/qmoe-lowering.md`` for the supported contract and reference sources.
This is deliberately not part of the automatic function-body registry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import onnx_ir as ir

from mobius.functions._qmoe_opset import plan_opset29


@dataclass(frozen=True)
class _Config:
    experts: int
    hidden: int
    intermediate: int
    block: int
    k: int
    normalize: bool
    alpha: float
    beta: float
    limit: float


def _input(node: ir.Node, index: int) -> ir.Value | None:
    return node.inputs[index] if index < len(node.inputs) else None


def _error(message: str) -> ValueError:
    return ValueError(f"lower_qmoe: {message}")


def _shape(value: ir.Value, name: str) -> tuple[int | ir.SymbolicDim, ...]:
    if value.shape is None:
        raise _error(f"{name} requires shape metadata")
    return tuple(value.shape)


def _validate(node: ir.Node) -> _Config:
    """Validate the entire supported ABI without creating or changing any nodes."""
    if node.overload:
        raise _error("QMoE overloads are not supported")
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
    for index in (0, 1, 2, 3, 5, 6):
        if _input(node, index) is None:
            raise _error(f"required QMoE input {index} is absent")
    for index in (4, 7, 8, 9, 10, 13, *range(15, len(node.inputs))):
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
    if rw is not None and _shape(rw, "router_weights") != rs:
        raise _error("router_weights must have the same shape as router_probs")
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
    return _Config(
        experts, hidden, intermediate, block, k, bool(normalize), alpha, beta, limit
    )


class _Emitter:
    """Build detached nodes, using graph-local unique names supplied by the pass."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.nodes: list[ir.Node] = []

    def op(self, op_type: str, *inputs: ir.Value, **attributes) -> ir.Value:
        value = ir.Value(name=f"{self.prefix}_{len(self.nodes)}")
        self.nodes.append(
            ir.Node(
                "",
                op_type,
                inputs=inputs,
                outputs=[value],
                attributes=ir.convenience.convert_attributes(attributes),
            )
        )
        return value

    def ints(self, values: list[int]) -> ir.Value:
        return self.op("Constant", value=ir.tensor(np.array(values, dtype=np.int64)))

    def scalar(self, value: float) -> ir.Value:
        return self.op("Constant", value=ir.tensor(np.float32(value)))

    def unpack(self, packed: ir.Value, shape: list[int]) -> ir.Value:
        mask = self.op("Constant", value=ir.tensor(np.uint8(15)))
        shift = self.op("Constant", value=ir.tensor(np.uint8(4)))
        low = self.op("BitwiseAnd", packed, mask)
        high = self.op("BitShift", packed, shift, direction="RIGHT")
        axis = self.ints([-1])
        pairs = self.op(
            "Concat",
            self.op("Unsqueeze", low, axis),
            self.op("Unsqueeze", high, axis),
            axis=-1,
        )
        return self.op("Reshape", pairs, self.ints(shape))

    def dequantize(
        self,
        packed: ir.Value,
        scale: ir.Value,
        zp: ir.Value | None,
        c: _Config,
        n: int,
        width: int,
    ) -> ir.Value:
        weights = self.op(
            "Cast", self.unpack(packed, [c.experts, n, width]), to=ir.DataType.UINT4
        )
        blocks = width // c.block
        if zp is None:
            zeros = self.op(
                "ConstantOfShape",
                self.op("Shape", scale),
                value=ir.tensor(np.array([8], dtype=np.uint8)),
            )
        else:
            zeros = self.unpack(zp, [c.experts, n, -1])
            zeros = self.op(
                "Slice", zeros, self.ints([0]), self.ints([blocks]), self.ints([2])
            )
        zeros = self.op("Cast", zeros, to=ir.DataType.UINT4)
        return self.op("DequantizeLinear", weights, scale, zeros, axis=2, block_size=c.block)


def _lower(node: ir.Node, c: _Config, prefix: str) -> tuple[list[ir.Node], ir.Value]:
    e = _Emitter(prefix)
    x, router, w1, scale1, _, w2, scale2 = node.inputs[:7]
    assert x is not None and router is not None
    assert w1 is not None and scale1 is not None
    assert w2 is not None and scale2 is not None
    rw = _input(node, 14)
    if x.dtype == ir.DataType.FLOAT16:
        router = e.op("Cast", router, to=ir.DataType.FLOAT)
        scale1 = e.op("Cast", scale1, to=ir.DataType.FLOAT)
        scale2 = e.op("Cast", scale2, to=ir.DataType.FLOAT)
        if rw is not None:
            rw = e.op("Cast", rw, to=ir.DataType.FLOAT)
    # CPU sorts (logit, expert_id) descending, whereas ONNX TopK breaks ties
    # toward the lower index. Reverse the expert axis before selecting.
    reversed_router = e.op(
        "Gather", router, e.ints(list(range(c.experts - 1, -1, -1))), axis=1
    )
    top = ir.Node(
        "",
        "TopK",
        inputs=[reversed_router, e.ints([c.k])],
        num_outputs=2,
        attributes=ir.convenience.convert_attributes({"axis": 1, "sorted": 1}),
    )
    for i, output in enumerate(top.outputs):
        output.name = f"{prefix}_top_{i}"
    e.nodes.append(top)
    selected = e.op("Sub", e.ints([c.experts - 1]), top.outputs[1])
    if rw is None:
        routing = e.op("Softmax", top.outputs[0], axis=1)
    else:
        routing = e.op("GatherElements", rw, selected, axis=1)
        if c.normalize:
            total = e.op("ReduceSum", routing, e.ints([1]), keepdims=1)
            zero = e.op("Equal", total, e.scalar(0))
            safe_total = e.op("Where", zero, e.scalar(1), total)
            routing = e.op("Where", zero, e.scalar(0), e.op("Div", routing, safe_total))
    # CPU does not enqueue routes with scale <= 1e-8, including negative scales.
    active = e.op("Greater", routing, e.scalar(1e-8))
    routing = e.op("Where", active, routing, e.scalar(0))
    first_weights = e.dequantize(w1, scale1, _input(node, 11), c, 2 * c.intermediate, c.hidden)
    down_weights = e.dequantize(w2, scale2, _input(node, 12), c, c.hidden, c.intermediate)
    compute_x = e.op("Cast", x, to=ir.DataType.FLOAT) if x.dtype == ir.DataType.FLOAT16 else x
    flat = e.op("Reshape", compute_x, e.ints([-1, c.hidden]))
    first = e.op("GroupedMatMul", flat, first_weights, selected, transB=1)  # [T,k,2I]
    gate = e.op("Gather", first, e.ints(list(range(0, 2 * c.intermediate, 2))), axis=2)
    up = e.op("Gather", first, e.ints(list(range(1, 2 * c.intermediate, 2))), axis=2)
    if math.isfinite(c.limit):
        gate = e.op("Min", gate, e.scalar(c.limit))
        up = e.op("Clip", up, e.scalar(-c.limit), e.scalar(c.limit))
    sigmoid = e.op("Sigmoid", e.op("Mul", gate, e.scalar(c.alpha)))
    activated = e.op("Mul", e.op("Mul", gate, sigmoid), e.op("Add", up, e.scalar(c.beta)))
    # Down projection must operate on each selected expert's own intermediate.
    down = e.op(
        "GroupedMatMul",
        e.op("Reshape", activated, e.ints([-1, c.intermediate])),
        down_weights,
        e.op("Reshape", selected, e.ints([-1, 1])),
        transB=1,
    )
    down = e.op("Reshape", down, e.ints([-1, c.k, c.hidden]))
    weighted = e.op("Mul", down, e.op("Unsqueeze", routing, e.ints([2])))
    # Where, rather than only multiplying by zero, masks inactive expert NaNs.
    weighted = e.op("Where", e.op("Unsqueeze", active, e.ints([2])), weighted, e.scalar(0))
    result = e.op("ReduceSum", weighted, e.ints([1]), keepdims=0)
    result = e.op("Reshape", result, e.op("Shape", x))
    if x.dtype == ir.DataType.FLOAT16:
        result = e.op("Cast", result, to=ir.DataType.FLOAT16)
    result.type, result.shape = x.type, x.shape
    return e.nodes, result


def lower_qmoe(model: ir.Model) -> ir.Model:
    """Lower supported QMoE nodes in-place and return *model*.

    Opts the model and compatible existing functions into standard opset 29.
    No-QMoE models are untouched. All calls and import changes are validated
    before mutation; unsupported ABIs or unaudited schema changes raise
    ``ValueError``. This targets the pinned **CPU**
    reference semantics, not CUDA's differing input-14/normalization behavior.

    Supports FLOAT/FLOAT16 inputs with CPU-reference FLOAT compute, raw UINT8
    INT4 weights, blocked scales, interleaved SwiGLU, and rank-2/3 hidden input.
    See ``docs/qmoe-lowering.md``.
    Requires a runtime supporting GroupedMatMul-29 (or function expansion).
    """
    roots = [model.graph, *(function.graph for function in model.functions.values())]
    plans = []
    for root in roots:
        for node in root.all_nodes():
            if node.domain == "com.microsoft" and node.op_type == "QMoE":
                if node.op_identifier() in model.functions:
                    raise _error("caller-defined QMoE functions cannot be overridden")
                plans.append((node, _validate(node)))
    if not plans:
        return model
    aligned_graphs, aligned_nodes = plan_opset29(roots)
    # Generate every detached replacement before inserting any of them.
    names = {
        value.name
        for root in roots
        for graph in (root, *root.subgraphs())
        for value in (
            *graph.inputs,
            *graph.initializers.values(),
            *(out for node in graph for out in node.outputs),
        )
    }
    replacements = []
    for index, (node, config) in enumerate(plans):
        prefix = f"qmoe29_{index}"
        while any(name and name.startswith(prefix) for name in names):
            prefix += "_"
        nodes, result = _lower(node, config, prefix)
        replacements.append((node, nodes, result))
    for node, nodes, result in replacements:
        graph = node.graph
        assert graph is not None
        graph.insert_before(node, nodes)
        output = node.outputs[0]
        if output.type is not None:
            result.type = output.type
        if output.shape is not None:
            result.shape = output.shape
        result.doc_string = output.doc_string
        result.metadata_props.update(output.metadata_props)
        result.meta.update(output.meta)
        output.replace_all_uses_with(result, replace_graph_outputs=True)
        graph.remove(node, safe=True)
        result.name = output.name
    for graph in aligned_graphs:
        graph.opset_imports.pop("ai.onnx", None)
        graph.opset_imports[""] = 29
    for node in aligned_nodes:
        if node.version is not None:
            node.version = 29
    return model
