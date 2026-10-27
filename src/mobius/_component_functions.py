# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Preserve standard computations as versioned, model-local component functions.

Weights remain initializers in their owning graph and are passed explicitly to
functions. Bodies reuse emitted ONNX operations or existing stateful definitions;
there is no second implementation of component mathematics.
"""

from __future__ import annotations

import hashlib

import onnx
import onnx_ir as ir
from onnx_ir.passes import common as common_passes

COMPONENT_DOMAIN = "mobius.components"
COMPONENT_VERSION = 1

_COMPONENT_OPERATORS = {
    "Attention": "Attention",
    "RMSNormalization": "RMSNorm",
    "LayerNormalization": "LayerNorm",
    "RotaryEmbedding": "RotaryEmbedding",
}
_STATEFUL_OPERATORS = frozenset({"CausalConvWithState", "LinearAttention"})


def preserve_stateful_functions(model: ir.Model) -> None:
    """Keep existing standard stateful bodies without vendor-domain call sites."""
    renamed: dict[ir.OperatorIdentifier, ir.OperatorIdentifier] = {}
    for identifier, function in tuple(model.functions.items()):
        if function.domain != "com.microsoft" or function.name not in _STATEFUL_OPERATORS:
            continue
        del model.functions[identifier]
        function.domain = COMPONENT_DOMAIN
        function.metadata_props["mobius.component.version"] = str(COMPONENT_VERSION)
        model.functions[function.identifier()] = function
        renamed[identifier] = function.identifier()

    for graph in [model.graph, *(function.graph for function in model.functions.values())]:
        for node in graph.all_nodes():
            if node.op_identifier() in renamed:
                node.domain, node.op_type, node.overload = renamed[node.op_identifier()]
                graph.opset_imports[COMPONENT_DOMAIN] = COMPONENT_VERSION


def _component_function(node: ir.Node, version: int) -> ir.Function:
    schema = onnx.defs.get_schema(node.op_type, version)
    inputs = [
        ir.Value(name=schema.inputs[index].name) if value is not None else None
        for index, value in enumerate(node.inputs)
    ]
    outputs = [ir.Value(name=schema.outputs[index].name) for index in range(len(node.outputs))]
    attributes = [
        ir.Attr(name, attr.type, None, ref_attr_name=name)
        for name, attr in node.attributes.items()
    ]
    body = ir.Node("", node.op_type, inputs, attributes=attributes, outputs=outputs)
    graph = ir.Graph(
        [value for value in inputs if value is not None],
        outputs,
        nodes=[body],
        opset_imports={"": version},
        name=f"{node.op_type}_body",
    )
    # Optional operands and attribute declarations define the signature, not
    # layer-specific attribute values or tensor dimensions.
    signature = (
        tuple(value is not None for value in node.inputs),
        len(node.outputs),
        tuple(sorted((name, attr.type.value) for name, attr in node.attributes.items())),
    )
    digest = hashlib.sha256(repr(signature).encode("ascii")).hexdigest()[:16]
    function = ir.Function(
        COMPONENT_DOMAIN,
        _COMPONENT_OPERATORS[node.op_type],
        overload=f"opset{version}_{digest}",
        graph=graph,
        attributes=[ir.Attr(name, attr.type, None) for name, attr in node.attributes.items()],
    )
    function.metadata_props["mobius.component.version"] = str(COMPONENT_VERSION)
    function.metadata_props["mobius.component.operator"] = f"ai.onnx::{node.op_type}"
    return function


def preserve_component_functions(model: ir.Model) -> None:
    """Lift standard attention, norm, and RoPE operations without changing values."""
    common_passes.RemoveUnusedFunctionsPass()(model)
    graphs = [
        model.graph,
        *(
            function.graph
            for function in model.functions.values()
            if function.domain != COMPONENT_DOMAIN
        ),
    ]
    for graph in graphs:
        version = graph.opset_imports.get("", graph.opset_imports.get("ai.onnx"))
        if version is None:
            continue
        for node in graph.all_nodes():
            if (
                node.domain not in {"", "ai.onnx"}
                or node.op_type not in _COMPONENT_OPERATORS
                or node.overload
                or node.op_identifier() in model.functions
            ):
                continue
            function = _component_function(node, version)
            model.functions.setdefault(function.identifier(), function)
            node.domain, node.op_type, node.overload = function.identifier()
            inputs = [value for value in node.inputs if value is not None]
            node.resize_inputs(len(inputs))
            for index, value in enumerate(inputs):
                node.replace_input_with(index, value)
            graph.opset_imports[COMPONENT_DOMAIN] = COMPONENT_VERSION
    common_passes.RemoveUnusedFunctionsPass()(model)
    common_passes.RemoveUnusedOpsetsPass()(model)


def inline_component_functions(model: ir.Model) -> None:
    """Expand component functions for flat consumers, preserving graph I/O."""
    common_passes.InlinePass(criteria=lambda function: function.domain == COMPONENT_DOMAIN)(
        model
    )
    common_passes.RemoveUnusedFunctionsPass()(model)
    common_passes.RemoveUnusedOpsetsPass()(model)
