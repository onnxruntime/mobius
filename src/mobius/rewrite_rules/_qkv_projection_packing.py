# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pack projections, then split before normalization, RoPE, attention, or caches.

Supports distinct Q/K/V or Q/K with raw K reused for V. Parameter Concats defer
packing until checkpoint binding. No weights are dequantized or requantized.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable

import numpy as np
import onnx_ir as ir

from mobius._constants import QKV_PROJECTION_SOURCES_METADATA_KEY
from mobius.functions.matmul_nbits import (
    FUNCTION_BODY_METADATA_KEY,
    FUNCTION_BODY_METADATA_VALUE,
)

_Group = tuple[ir.Node, ...]
_FLOAT_DTYPES = frozenset({ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16})
_NBITS_ATTRIBUTES = frozenset({"K", "N", "bits", "block_size", "accuracy_level"})
_DATA_FIRST = frozenset(
    {
        "Identity",
        "Reshape",
        "Transpose",
        "Cast",
        "CastLike",
        "Squeeze",
        "Unsqueeze",
        "Slice",
        "Gather",
        "RotaryEmbedding",
        "RMSNormalization",
        "SimplifiedLayerNormalization",
        "ReduceMean",
        "Sqrt",
        "Reciprocal",
        "Pow",
        "Neg",
    }
)
_DATA_ALL = frozenset({"Add", "Sub", "Mul", "Div", "Concat"})


def _standard(node: ir.Node, op_type: str) -> bool:
    return node.domain in ("", "ai.onnx") and node.op_type == op_type and not node.overload


def _parameter(
    graph: ir.Graph, value: ir.Value | None, dtype: ir.DataType, shape: list[int]
) -> bool:
    if (
        value is None
        or value.name is None
        or value.producer() is not None
        or graph.initializers.get(value.name) is not value
        or value in graph.inputs
        or value.dtype != dtype
        or value.shape is None
        or list(value.shape) != shape
    ):
        return False
    tensor = value.const_value
    return tensor is None or (tensor.dtype == dtype and list(tensor.shape) == shape)


def _projection(graph: ir.Graph, value: ir.Value | None) -> ir.Node | None:
    pending = [value]
    visited: set[ir.Node] = set()
    projections: set[ir.Node] = set()
    while pending:
        current = pending.pop()
        if current is None or current.producer() is None:
            continue
        node = current.producer()
        assert node is not None
        if node in visited:
            continue
        if node.graph is not graph or len(visited) >= 128:
            return None
        visited.add(node)
        if _standard(node, "MatMul") or (
            node.domain == "com.microsoft" and node.op_type == "MatMulNBits"
        ):
            projections.add(node)
        elif _standard(node, "Constant"):
            continue
        elif node.domain in ("", "ai.onnx", "com.microsoft"):
            if node.op_type in _DATA_FIRST:
                pending.extend(node.inputs[:1])
            elif node.domain != "com.microsoft" and node.op_type in _DATA_ALL:
                pending.extend(node.inputs)
            else:
                return None
        else:
            return None
    return next(iter(projections)) if len(projections) == 1 else None


def _shared_input(group: _Group) -> ir.Value | None:
    first = group[0]
    if not first.inputs or first.inputs[0] is None:
        return None
    value = first.inputs[0]
    if (
        value.dtype not in _FLOAT_DTYPES
        or value.shape is None
        or len(value.shape) != 3
        or not isinstance(value.shape[-1], int)
        or value.shape[-1] <= 0
        or any(node.inputs[0] is not value or len(node.outputs) != 1 for node in group)
    ):
        return None
    return value


def _output_matches(node: ir.Node, value: ir.Value, width: int) -> bool:
    assert value.shape is not None
    output = node.outputs[0]
    if output.dtype not in (None, value.dtype):
        return False
    if output.shape is None:
        return True
    expected = [*value.shape[:-1], width]
    return len(output.shape) == 3 and all(
        (isinstance(dim, ir.SymbolicDim) and dim.value is None) or dim == wanted
        for dim, wanted in zip(output.shape, expected, strict=True)
    )


def _concat(
    nodes: list[ir.Node], values: list[ir.Value], axis: int, shape: list[int | ir.SymbolicDim]
) -> ir.Value:
    output = ir.Value(shape=ir.Shape(shape), type=values[0].type)
    nodes.append(
        ir.Node("", "Concat", values, {"axis": ir.AttrInt64("axis", axis)}, outputs=[output])
    )
    return output


def _replace(
    graph: ir.Graph,
    group: _Group,
    value: ir.Value,
    widths: list[int],
    nodes: list[ir.Node],
    packed: ir.Value,
) -> ir.Node:
    assert value.shape is not None and value.dtype is not None
    tensor = ir.tensor(np.asarray(widths, dtype=np.int64))
    sizes = ir.Value(
        shape=ir.Shape([len(widths)]),
        type=ir.TensorType(ir.DataType.INT64),
        const_value=tensor,
    )
    nodes.append(
        ir.Node("", "Constant", [], {"value": ir.AttrTensor("value", tensor)}, outputs=[sizes])
    )
    outputs = []
    for node, width in zip(group, widths, strict=True):
        original = node.outputs[0]
        output = ir.Value(
            name=original.name,
            doc_string=original.doc_string,
            shape=ir.Shape([*value.shape[:-1], width]),
            type=ir.TensorType(value.dtype),
        )
        output.metadata_props.update(original.metadata_props)
        output.meta.update(original.meta)
        outputs.append(output)
    split = ir.Node(
        "", "Split", [packed, sizes], {"axis": ir.AttrInt64("axis", -1)}, outputs=outputs
    )
    nodes.append(split)
    ir.convenience.replace_nodes_and_values(
        graph,
        insertion_point=next(node for node in graph if node in group),
        old_nodes=list(group),
        new_nodes=nodes,
        old_values=[node.outputs[0] for node in group],
        new_values=outputs,
    )
    for output, width in zip(outputs, widths, strict=True):
        output.shape = ir.Shape([*value.shape[:-1], width])
        output.type = ir.TensorType(value.dtype)
    return split


def _pack_float(model: ir.Model, group: _Group) -> bool:
    graph = model.graph
    if any(
        not _standard(node, "MatMul") or node.attributes or len(node.inputs) != 2
        for node in group
    ):
        return False
    value = _shared_input(group)
    if value is None:
        return False
    assert value.shape is not None and value.dtype is not None
    k = value.shape[-1]
    assert isinstance(k, int)
    weights: list[tuple[ir.Value, bool]] = []
    widths = []
    for node in group:
        weight = node.inputs[1]
        if weight is None:
            return False
        producer = weight.producer()
        transposed = producer is not None and _standard(producer, "Transpose")
        if transposed:
            assert producer is not None
            if (
                producer.graph is not graph
                or len(producer.inputs) != 1
                or set(producer.attributes) - {"perm"}
                or tuple(producer.attributes.get_ints("perm", [1, 0])) != (1, 0)
            ):
                return False
            weight = producer.inputs[0]
        if weight is None or weight.shape is None or len(weight.shape) != 2:
            return False
        width = weight.shape[0 if transposed else 1]
        if not isinstance(width, int) or width <= 0:
            return False
        if not _parameter(
            graph, weight, value.dtype, [width, k] if transposed else [k, width]
        ) or not _output_matches(node, value, width):
            return False
        # Restoring PEFT sources must remain unambiguous. A source initializer
        # or its logical Transpose view shared outside this group cannot safely
        # be hidden in independently packed groups. Decline before mutating IR.
        allowed_consumers = set(group)
        if transposed:
            assert producer is not None
            allowed_consumers.add(producer)
            if producer.outputs[0] in graph.outputs or any(
                use.node not in group for use in producer.outputs[0].uses()
            ):
                return False
        if weight in graph.outputs or any(
            use.node not in allowed_consumers for use in weight.uses()
        ):
            return False
        weights.append((weight, transposed))
        widths.append(width)
    # Distinct projection nodes can still read the same source (including
    # separate Transpose views). Duplicate names cannot be restored as
    # independent PEFT targets. Refuse before creating any replacement nodes.
    if len({weight.name for weight, _ in weights}) != len(weights):
        return False
    nodes: list[ir.Node] = []
    all_transposed = all(transposed for _, transposed in weights)
    logical_weights = []
    for weight, transposed in weights:
        if transposed and not all_transposed:
            assert weight.shape is not None
            logical = ir.Value(shape=ir.Shape([k, weight.shape[0]]), type=weight.type)
            nodes.append(
                ir.Node(
                    "",
                    "Transpose",
                    [weight],
                    {"perm": ir.AttrInt64s("perm", [1, 0])},
                    outputs=[logical],
                )
            )
            logical_weights.append(logical)
        else:
            logical_weights.append(weight)
    packed_weight = _concat(
        nodes,
        logical_weights,
        0 if all_transposed else 1,
        [sum(widths), k] if all_transposed else [k, sum(widths)],
    )
    if all_transposed:
        logical = ir.Value(shape=ir.Shape([k, sum(widths)]), type=packed_weight.type)
        nodes.append(
            ir.Node(
                "",
                "Transpose",
                [packed_weight],
                {"perm": ir.AttrInt64s("perm", [1, 0])},
                outputs=[logical],
            )
        )
        packed_weight = logical
    packed = ir.Value(shape=ir.Shape([*value.shape[:-1], sum(widths)]), type=value.type)
    nodes.append(ir.Node("", "MatMul", [value, packed_weight], outputs=[packed]))
    split = _replace(graph, group, value, widths, nodes, packed)
    split.metadata_props[QKV_PROJECTION_SOURCES_METADATA_KEY] = json.dumps(
        {
            "version": 1,
            "weights": [weight.name for weight, _ in weights],
            "widths": widths,
        }
    )
    return True


def _pack_nbits(model: ir.Model, group: _Group) -> bool:
    graph = model.graph
    if graph.opset_imports.get("com.microsoft") != 1 or any(
        node.domain != "com.microsoft" or node.op_type != "MatMulNBits" or len(node.inputs) < 3
        for node in group
    ):
        return False
    value = _shared_input(group)
    if value is None:
        return False
    assert value.shape is not None and value.dtype is not None
    first = group[0]
    if any(
        attr.type != ir.AttributeType.INT
        for node in group
        for attr in node.attributes.values()
    ):
        return False
    k = first.attributes.get_int("K", 0)
    block = first.attributes.get_int("block_size", 0)
    if k != value.shape[-1] or block < 16 or block & (block - 1):
        return False
    blocks = (k + block - 1) // block
    common = {key: attr for key, attr in first.attributes.items() if key != "N"}
    has_zp = len(first.inputs) > 3 and first.inputs[3] is not None
    layouts: list[tuple[int, ...]] = []
    widths = []
    for node in group:
        if node.overload != first.overload or node.overload not in ("", "default_zero_points"):
            return False
        function = model.functions.get(node.op_identifier())
        if (
            function is not None
            and function.metadata_props.get(FUNCTION_BODY_METADATA_KEY)
            != FUNCTION_BODY_METADATA_VALUE
        ):
            return False
        if node.overload == "default_zero_points" and has_zp:
            return False
        if set(node.attributes) - _NBITS_ATTRIBUTES:
            return False
        if {key: attr for key, attr in node.attributes.items() if key != "N"} != common:
            return False
        if (
            node.attributes.get_int("bits", 0) != 4
            or node.attributes.get_int("accuracy_level", 0) != 0
        ):
            return False
        n = node.attributes.get_int("N", 0)
        if n <= 0 or any(item is not None for item in node.inputs[4:]):
            return False
        if not _parameter(graph, node.inputs[1], ir.DataType.UINT8, [n, blocks, block // 2]):
            return False
        scale = node.inputs[2]
        if scale is None or scale.shape is None:
            return False
        scale_shape = [n, blocks] if len(scale.shape) == 2 else [n * blocks]
        if not _parameter(graph, scale, value.dtype, scale_shape):
            return False
        zp = node.inputs[3] if len(node.inputs) > 3 else None
        if (zp is not None) != has_zp:
            return False
        layout: tuple[int, ...] = (len(scale_shape),)
        if zp is not None:
            if zp.shape is None:
                return False
            zp_shape = (
                [n, (blocks + 1) // 2] if len(zp.shape) == 2 else [n * ((blocks + 1) // 2)]
            )
            if not _parameter(graph, zp, ir.DataType.UINT8, zp_shape):
                return False
            layout += (len(zp_shape),)
        if layouts and layout != layouts[0]:
            return False
        if not _output_matches(node, value, n):
            return False
        layouts.append(layout)
        widths.append(n)
    parameters = [item for node in group for item in node.inputs[1:4] if item is not None]
    if len({item.name for item in parameters}) != len(parameters) or any(
        item in graph.outputs or any(use.node not in group for use in item.uses())
        for item in parameters
    ):
        return False
    nodes: list[ir.Node] = []
    inputs: list[ir.Value | None] = [value]
    for slot in range(1, 4 if has_zp else 3):
        originals = [node.inputs[slot] for node in group]
        assert all(item is not None and item.shape is not None for item in originals)
        parameters = [item for item in originals if item is not None]
        shape = parameters[0].shape
        assert shape is not None
        extent = 0
        for parameter in parameters:
            assert parameter.shape is not None
            dimension = parameter.shape[0]
            assert isinstance(dimension, int)
            extent += dimension
        inputs.append(_concat(nodes, parameters, 0, [extent, *shape[1:]]))
    packed = ir.Value(shape=ir.Shape([*value.shape[:-1], sum(widths)]), type=value.type)
    attributes = dict(first.attributes)
    attributes["N"] = ir.AttrInt64("N", sum(widths))
    nodes.append(
        ir.Node(
            "com.microsoft",
            "MatMulNBits",
            inputs,
            attributes,
            overload=first.overload,
            outputs=[packed],
        )
    )
    _replace(graph, group, value, widths, nodes, packed)
    return True


def _run(model: ir.Model, pack: Callable[[ir.Model, _Group], bool]) -> ir.passes.PassResult:
    graph = model.graph
    if graph.opset_imports.get("", 0) < 13:
        return ir.passes.PassResult(model, modified=False)
    candidates: set[_Group] = set()
    for node in graph:
        if (
            not (
                _standard(node, "Attention")
                or (node.domain == "com.microsoft" and node.op_type == "GroupQueryAttention")
            )
            or len(node.inputs) < 3
            or any(item is None for item in node.inputs[:3])
        ):
            continue
        projections = [_projection(graph, item) for item in node.inputs[:3]]
        if any(item is None for item in projections):
            continue
        # Only K=V reuse is supported. Q=K or Q=V is not an independent
        # projection group and must not produce ambiguous source metadata.
        if projections[0] in projections[1:]:
            continue
        group = tuple(dict.fromkeys(item for item in projections if item is not None))
        if len(group) in (2, 3):
            candidates.add(group)
    counts = Counter(node for group in candidates for node in group)
    order = {node: index for index, node in enumerate(graph)}
    modified = False
    for group in sorted(candidates, key=lambda group: min(order[node] for node in group)):
        if all(counts[node] == 1 for node in group):
            modified = pack(model, group) or modified
    return ir.passes.PassResult(model, modified=modified)


class PackQKVProjectionsPass(ir.passes.InPlacePass):
    """Pack compatible float projections, retaining all downstream branches."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return _run(model, _pack_float)


class PackMatMulNBitsQKVPass(ir.passes.InPlacePass):
    """Pack homogeneous INT4 projections without dequantizing or requantizing."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return _run(model, _pack_nbits)


def pack_qkv_projections_pass() -> PackQKVProjectionsPass:
    return PackQKVProjectionsPass()


def pack_matmul_nbits_qkv_pass() -> PackMatMulNBitsQKVPass:
    return PackMatMulNBitsQKVPass()
