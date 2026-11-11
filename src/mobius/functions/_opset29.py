# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bounded, non-mutating opset-29 compatibility audit for explicit MoE passes.

Audited against ONNX 82721c0908fa965dd84e674c006147791313f3e1. This is not a
general version converter: unchanged schemas are safe, and the explicit deltas
below preserve the old valid type/attribute contract. Everything else fails.
No protobuf schema defaults or model-conversion APIs are used.
See ``docs/qmoe-lowering.md`` under "Import alignment" for the audited changes.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnx_ir as ir

# These exact schema transitions only extend type constraints. Existing valid
# operands, attributes (including defaults), and numerical semantics are unchanged.
_TYPE_EXTENSIONS = {
    ("Cast", 24, 28),
    ("CastLike", 24, 25),
    ("Constant", 24, 25),
    ("ConstantOfShape", 24, 25),
    ("DequantizeLinear", 24, 28),
    ("Identity", 24, 25),
    ("If", 24, 25),
    ("Reshape", 24, 25),
    ("Shape", 24, 25),
    ("Squeeze", 24, 25),
    ("Transpose", 24, 25),
    ("Unsqueeze", 24, 25),
}


def _fail(node: ir.Node, message: str, caller: str) -> ValueError:
    return ValueError(f"{caller}: cannot align {node.op_type} to opset 29: {message}")


def _check_old_types(node: ir.Node, schema: onnx.defs.OpSchema, caller: str) -> None:
    """Reject known types outside the old contract, including type attributes."""
    constraints = {
        constraint.type_param_str: set(constraint.allowed_type_strs)
        for constraint in schema.type_constraints
    }

    def check(dtype: ir.DataType, parameter: str) -> None:
        allowed = constraints.get(parameter, {parameter})
        if f"tensor({dtype.name.lower()})" not in allowed:
            raise _fail(
                node, f"{dtype.name} is outside the original schema's {parameter}", caller
            )

    for formals, values in ((schema.inputs, node.inputs), (schema.outputs, node.outputs)):
        if values and not formals:
            raise _fail(node, "operands outside the original schema", caller)
        for index, value in enumerate(values):
            formal = formals[min(index, len(formals) - 1)]
            if value is not None and value.dtype is not None:
                check(value.dtype, formal.type_str)
    if node.op_type == "Cast":
        attr = node.attributes.get("to")
        if attr is not None and not attr.is_ref():
            check(ir.DataType(attr.as_int()), "T2")
    elif node.op_type == "DequantizeLinear":
        attr = node.attributes.get("output_dtype")
        if attr is not None and not attr.is_ref() and attr.as_int() != 0:
            check(ir.DataType(attr.as_int()), "T3")
    elif node.op_type in {"Constant", "ConstantOfShape"}:
        attr = node.attributes.get("value")
        if attr is not None and not attr.is_ref():
            check(attr.as_tensor().dtype, "T" if node.op_type == "Constant" else "T2")


def _check_shift(node: ir.Node, caller: str) -> None:
    # BitShift-28 additionally defines signed and out-of-range shifts. The old
    # schema only accepted unsigned integers (at least 8 bits). Prove the shift
    # is in [0,7], where the old/new semantics coincide for every old valid T.
    shift = node.inputs[1] if len(node.inputs) > 1 else None
    tensor = None if shift is None else shift.const_value
    if tensor is None and shift is not None:
        producer = shift.producer()
        if producer is not None and producer.domain == "" and producer.op_type == "Constant":
            attr = producer.attributes.get("value")
            if attr is not None and not attr.is_ref():
                tensor = attr.as_tensor()
    if tensor is None or tensor.dtype not in {
        ir.DataType.UINT8,
        ir.DataType.UINT16,
        ir.DataType.UINT32,
        ir.DataType.UINT64,
    }:
        raise _fail(node, "BitShift requires a constant unsigned shift in [0,7]", caller)
    shifts = tensor.numpy()
    if not np.all(shifts < 8):
        raise _fail(node, "BitShift requires a constant unsigned shift in [0,7]", caller)


def _check_node(node: ir.Node, version: int, caller: str) -> None:
    try:
        old = onnx.defs.get_schema(node.op_type, version)
        new = onnx.defs.get_schema(node.op_type, 29)
        if not (
            old.min_input <= len(node.inputs) <= old.max_input
            and old.min_output <= len(node.outputs) <= old.max_output
        ):
            raise _fail(node, "input/output arity outside the original schema", caller)
        if node.version is not None:
            explicit = onnx.defs.get_schema(node.op_type, node.version)
            if explicit.since_version != old.since_version:
                raise _fail(
                    node, "explicit node version conflicts with containing import", caller
                )
    except onnx.defs.SchemaError as error:
        raise _fail(node, f"schema unavailable at original opset {version}", caller) from error
    if old.since_version == new.since_version:
        return
    delta = (node.op_type, old.since_version, new.since_version)
    if delta not in _TYPE_EXTENSIONS and delta not in {
        ("Attention", 24, 25),
        ("Range", 11, 27),
        ("BitShift", 11, 28),
    }:
        raise _fail(
            node,
            f"unaudited schema transition {old.since_version}→{new.since_version}",
            caller,
        )
    if set(node.attributes) - set(old.attributes):
        raise _fail(node, "attributes outside the original schema", caller)
    _check_old_types(node, old, caller)
    if node.op_type == "BitShift":
        _check_shift(node, caller)
    # Attention-25 adds left/right windows, both defaulting to -1 (unbounded).
    # The old schema has neither attribute, as enforced above. Range-27 only
    # adds FLOAT16/BFLOAT16 with stash_type; all old types are unchanged.


def plan_opset29(
    roots: list[ir.Graph], *, caller: str
) -> tuple[list[ir.Graph], list[ir.Node]]:
    """Verify all existing scopes, returning an import/node-version commit plan."""
    if not onnx.defs.has("GroupedMatMul", 29):
        raise ValueError(f"{caller}: ONNX opset-29 schemas are required for import alignment")
    graphs: list[ir.Graph] = []
    nodes: list[ir.Node] = []
    for root in roots:
        # GraphProto has no imports: every descendant serializes under its
        # enclosing ModelProto/FunctionProto root, regardless of IR-local hints.
        version = root.opset_imports.get("", root.opset_imports.get("ai.onnx"))
        if version is not None and not 1 <= version <= 29:
            raise ValueError(f"{caller}: cannot align default-domain opset {version} to 29")
        if (
            "ai.onnx" in root.opset_imports
            and root.opset_imports.get("", version) != root.opset_imports["ai.onnx"]
        ):
            raise ValueError(f"{caller}: conflicting default-domain opset imports")
        for graph in (root, *root.subgraphs()):
            for node in graph:
                if node.domain:
                    continue
                if version is None:
                    raise _fail(node, "containing scope has no default-domain import", caller)
                _check_node(node, version, caller)
                nodes.append(node)
            graphs.append(graph)
    return graphs, nodes
