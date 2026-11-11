# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Detached emission and atomic, metadata-preserving explicit lowering mechanics."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
import onnx_ir as ir

from mobius._flags import flags
from mobius.functions._opset29 import plan_opset29

_Config = TypeVar("_Config")


class Emitter:
    """Build detached nodes with graph-local unique names supplied by the pass."""

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


def validate_output_metadata(node: ir.Node, x: ir.Value, *, caller: str) -> None:
    """Allow absent/refined output annotations, rejecting known contradictions."""
    output = node.outputs[0]
    message = f"{caller}: output metadata must match input shape and dtype"
    # Native custom-op intermediates can be untyped/unshaped; generation infers
    # absent metadata from X, while replacement preserves existing annotations.
    if output.type is not None and (
        not isinstance(output.type, ir.TensorType) or output.dtype != x.dtype
    ):
        raise ValueError(message)
    if output.shape is not None:
        assert x.shape is not None  # The caller has already validated X.
        xs, ys = tuple(x.shape), tuple(output.shape)
        if len(ys) != len(xs) or any(
            isinstance(b, int) and (b < 0 or (isinstance(a, int) and a != b))
            for a, b in zip(xs, ys, strict=False)
        ):
            raise ValueError(message)


def lower_nodes(
    model: ir.Model,
    *,
    op_type: str,
    validate: Callable[[ir.Node], _Config],
    generate: Callable[[ir.Node, _Config, str], tuple[list[ir.Node], ir.Value]],
) -> ir.Model:
    """Validate all calls and imports, emit detached plans, then commit atomically."""
    caller = f"lower_{op_type.lower()}"
    roots = [model.graph, *(function.graph for function in model.functions.values())]
    plans = []
    for root in roots:
        for node in root.all_nodes():
            if node.domain != "com.microsoft" or node.op_type != op_type:
                continue
            if not flags.experimental_opset29:
                raise ValueError(
                    f"{caller}: opset 29 is experimental; set MOBIUS_EXPERIMENTAL_OPSET29=1 "
                    "before importing mobius or use "
                    "override_flags(experimental_opset29=True). "
                    "This opt-in does not imply runtime kernel support."
                )
            if node.op_identifier() in model.functions:
                raise ValueError(
                    f"{caller}: caller-defined {op_type} functions cannot be overridden"
                )
            plans.append((node, validate(node)))
    if not plans:
        return model
    aligned_graphs, aligned_nodes = plan_opset29(roots, caller=caller)
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
        prefix = f"{op_type.lower()}29_{index}"
        while any(name and name.startswith(prefix) for name in names):
            prefix += "_"
        nodes, result = generate(node, config, prefix)
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
