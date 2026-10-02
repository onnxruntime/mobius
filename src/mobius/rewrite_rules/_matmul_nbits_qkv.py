# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pack compatible INT4 projections anchored at an attention operator.

Only the projections change: bias, per-head normalization, rotary embeddings,
attention masks, and cache consumers retain their original inputs and outputs.
Parameter Concats defer packing until the original checkpoint names are bound.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import onnx_ir as ir

_ATTRIBUTES = frozenset({"K", "N", "bits", "block_size", "accuracy_level"})
_Triple = tuple[ir.Node, ir.Node, ir.Node]


def _standard(node: ir.Node | None, op_type: str) -> bool:
    return node is not None and node.domain == "" and node.op_type == op_type


def _constant_reshape(node: ir.Node | None) -> bool:
    return (
        _standard(node, "Reshape")
        and node is not None
        and len(node.inputs) == 2
        and node.inputs[1] is not None
        and node.inputs[1].const_value is not None
    )


def _initializer(graph: ir.Graph, value: ir.Value | None) -> bool:
    return (
        value is not None
        and value.name is not None
        and value.producer() is None
        and graph.initializers.get(value.name) is value
        and value not in graph.inputs
    )


def _trace(graph: ir.Graph, value: ir.Value | None, *, normalized: bool) -> ir.Node | None:
    if value is None:
        return None
    node = value.producer()
    if normalized:
        if _standard(node, "RotaryEmbedding"):
            assert node is not None
            value = node.inputs[0]
            if value is None:
                return None
            node = value.producer()
        enclosing = _constant_reshape(node)
        if enclosing:
            assert node is not None
            value = node.inputs[0]
            if value is None:
                return None
            node = value.producer()
        if _standard(node, "RMSNormalization"):
            assert node is not None
            value = node.inputs[0]
            if value is None:
                return None
            node = value.producer()
            if enclosing:
                if not _constant_reshape(node):
                    return None
                assert node is not None
                value = node.inputs[0]
                if value is None:
                    return None
                node = value.producer()
        elif enclosing:
            return None
    bias: ir.Value | None = None
    if _standard(node, "Add"):
        assert node is not None
        if len(node.inputs) != 2:
            return None
        left, right = node.inputs
        if _initializer(graph, left):
            bias, value = left, right
        elif _initializer(graph, right):
            bias, value = right, left
        else:
            return None
        if value is None:
            return None
        node = value.producer()
    if node is None or node.graph is not graph:
        return None
    if node.domain != "com.microsoft" or node.op_type != "MatMulNBits":
        return None
    if bias is not None:
        width = node.attributes.get("N")
        if width is None or width.type != ir.AttributeType.INT:
            return None
        if bias.shape is None or list(bias.shape) != [width.as_int()]:
            return None
    return node


def _parameter(
    graph: ir.Graph, value: ir.Value | None, dtype: ir.DataType, shape: list[int]
) -> bool:
    if not _initializer(graph, value):
        return False
    assert value is not None
    if value.dtype != dtype or value.shape is None or list(value.shape) != shape:
        return False
    tensor = value.const_value
    return tensor is None or (tensor.dtype == dtype and list(tensor.shape) == shape)


def _compatible(model: ir.Model, triple: _Triple) -> bool:
    graph = model.graph
    if graph.opset_imports.get("com.microsoft") != 1 or len(set(triple)) != 3:
        return False
    first = triple[0]
    if any(
        attr.type != ir.AttributeType.INT
        for node in triple
        for attr in node.attributes.values()
    ):
        return False
    if not first.inputs or first.inputs[0] is None:
        return False
    a = first.inputs[0]
    k = first.attributes.get_int("K", 0)
    block = first.attributes.get_int("block_size", 0)
    if k <= 0 or block < 16 or block & (block - 1):
        return False
    if a.dtype != ir.DataType.FLOAT16 or a.shape is None or len(a.shape) != 3:
        return False
    if a.shape[-1] != k:
        return False
    blocks = (k + block - 1) // block
    common_attrs = {name: attr for name, attr in first.attributes.items() if name != "N"}
    has_zp = len(first.inputs) > 3 and first.inputs[3] is not None
    for node in triple:
        if len(node.outputs) != 1 or len(node.inputs) < 3:
            return False
        if node.inputs[0] is not a or node.overload != first.overload:
            return False
        if node.overload not in ("", "default_zero_points"):
            return False
        function = model.functions.get(node.op_identifier())
        if function is not None and function.metadata_props.get("mobius.function_body") != "1":
            return False
        if node.overload == "default_zero_points" and has_zp:
            return False
        if set(node.attributes) - _ATTRIBUTES:
            return False
        attrs = {name: attr for name, attr in node.attributes.items() if name != "N"}
        if attrs != common_attrs:
            return False
        if node.attributes.get_int("bits", 0) != 4:
            return False
        if node.attributes.get_int("accuracy_level", 0) != 0:
            return False
        n = node.attributes.get_int("N", 0)
        if n <= 0 or any(value is not None for value in node.inputs[4:]):
            return False
        if not _parameter(graph, node.inputs[1], ir.DataType.UINT8, [n, blocks, block // 2]):
            return False
        if not _parameter(graph, node.inputs[2], ir.DataType.FLOAT16, [n, blocks]):
            return False
        zp = node.inputs[3] if len(node.inputs) > 3 else None
        if (zp is not None) != has_zp:
            return False
        if has_zp and not _parameter(graph, zp, ir.DataType.UINT8, [n, (blocks + 1) // 2]):
            return False
        output = node.outputs[0]
        expected = [*a.shape[:-1], n]
        if output.dtype not in (None, ir.DataType.FLOAT16):
            return False
        if output.shape is not None:
            if len(output.shape) != len(expected):
                return False
            if any(
                not (isinstance(dim, ir.SymbolicDim) and dim.value is None) and dim != wanted
                for dim, wanted in zip(output.shape, expected, strict=True)
            ):
                return False
    return True


def _pack(graph: ir.Graph, triple: _Triple) -> None:
    first = triple[0]
    a = first.inputs[0]
    assert a is not None and a.shape is not None
    widths = [node.attributes.get_int("N") for node in triple]
    nodes: list[ir.Node] = []
    inputs: list[ir.Value | None] = [a]
    slots = 4 if len(first.inputs) > 3 and first.inputs[3] is not None else 3
    for slot in range(1, slots):
        original = first.inputs[slot]
        assert original is not None and original.shape is not None
        packed = ir.Value(
            shape=ir.Shape([sum(widths), *original.shape[1:]]), type=original.type
        )
        nodes.append(
            ir.Node(
                "",
                "Concat",
                [node.inputs[slot] for node in triple],
                attributes=ir.convenience.convert_attributes({"axis": 0}),
                outputs=[packed],
            )
        )
        inputs.append(packed)
    packed_output = ir.Value(
        shape=ir.Shape([*a.shape[:-1], sum(widths)]), type=ir.TensorType(ir.DataType.FLOAT16)
    )
    attributes = dict(first.attributes)
    attributes["N"] = ir.AttrInt64("N", sum(widths))
    nodes.append(
        ir.Node(
            "com.microsoft",
            "MatMulNBits",
            inputs,
            attributes=attributes,
            overload=first.overload,
            outputs=[packed_output],
        )
    )
    sizes_tensor = ir.tensor(np.asarray(widths, dtype=np.int64))
    sizes = ir.Value(
        shape=ir.Shape([3]),
        type=ir.TensorType(ir.DataType.INT64),
        const_value=sizes_tensor,
    )
    nodes.append(
        ir.Node(
            "",
            "Constant",
            [],
            attributes={"value": ir.AttrTensor("value", sizes_tensor)},
            outputs=[sizes],
        )
    )
    outputs = []
    for node, width in zip(triple, widths, strict=True):
        original_output = node.outputs[0]
        output = ir.Value(
            name=original_output.name,
            doc_string=original_output.doc_string,
            shape=ir.Shape([*a.shape[:-1], width]),
            type=ir.TensorType(ir.DataType.FLOAT16),
        )
        output.metadata_props.update(original_output.metadata_props)
        output.meta.update(original_output.meta)
        outputs.append(output)
    nodes.append(
        ir.Node(
            "",
            "Split",
            [packed_output, sizes],
            attributes=ir.convenience.convert_attributes({"axis": -1}),
            outputs=outputs,
        )
    )
    insertion_point = next(node for node in graph if node in triple)
    ir.convenience.replace_nodes_and_values(
        graph,
        insertion_point=insertion_point,
        old_nodes=list(triple),
        new_nodes=nodes,
        old_values=[node.outputs[0] for node in triple],
        new_values=outputs,
    )
    # Replacement copies old shape metadata, which may contain unknown dimensions.
    for output, width in zip(outputs, widths, strict=True):
        output.shape = ir.Shape([*a.shape[:-1], width])
        output.type = ir.TensorType(ir.DataType.FLOAT16)


class PackMatMulNBitsQKVPass(ir.passes.InPlacePass):
    """Pack homogeneous FP16/INT4 attention projections without requantization."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        candidates: set[_Triple] = set()
        for node in graph:
            if not (
                _standard(node, "Attention")
                or (node.domain == "com.microsoft" and node.op_type == "GroupQueryAttention")
            ):
                continue
            if len(node.inputs) < 3 or any(value is None for value in node.inputs[:3]):
                continue
            q, k, v = node.inputs[:3]
            assert q is not None and k is not None and v is not None
            projections = (
                _trace(graph, q, normalized=True),
                _trace(graph, k, normalized=True),
                _trace(graph, v, normalized=False),
            )
            if any(projection is None for projection in projections):
                continue
            q_proj, k_proj, v_proj = projections
            assert q_proj is not None and k_proj is not None and v_proj is not None
            candidates.add((q_proj, k_proj, v_proj))
        counts = Counter(node for triple in candidates for node in set(triple))
        targets = [
            triple
            for triple in candidates
            if all(counts[node] == 1 for node in triple) and _compatible(model, triple)
        ]
        # Graph order makes insertion deterministic even with multiple anchors.
        order = {node: index for index, node in enumerate(graph)}
        targets.sort(key=lambda triple: order[triple[0]])
        for triple in targets:
            _pack(graph, triple)
        return ir.passes.PassResult(model, modified=bool(targets))


def pack_matmul_nbits_qkv_pass() -> PackMatMulNBitsQKVPass:
    """Return a projection-only, attention-anchored INT4 packing pass."""
    return PackMatMulNBitsQKVPass()
