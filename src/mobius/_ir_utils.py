# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared helpers for inspecting ONNX IR tensor data types."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain

import onnx_ir as ir


def _iter_type_dtypes(
    type_proto: ir.TypeProtocol | ir.MapTypeProtocol | ir.DataType | None,
) -> Iterator[ir.DataType]:
    """Yield leaf tensor dtypes from a possibly nested ONNX type."""
    if type_proto is None:
        return
    if isinstance(type_proto, ir.DataType):
        yield type_proto
        return
    if isinstance(type_proto, ir.TypeProtocol):
        yield from _iter_type_dtypes(type_proto.elem_type)
        return
    if isinstance(type_proto, ir.MapTypeProtocol):
        yield from _iter_type_dtypes(type_proto.value_type)


def _iter_sparse_tensor_dtypes(
    sparse_tensor: ir.SparseTensorProtocol,
) -> Iterator[ir.DataType]:
    """Yield dtypes stored by a sparse tensor attribute."""
    yield sparse_tensor.values.dtype
    yield sparse_tensor.indices.dtype


def iter_graph_tensor_dtypes(graph: ir.Graph) -> Iterator[ir.DataType]:
    """Yield tensor dtypes declared or stored anywhere in *graph*."""
    for current_graph in chain((graph,), graph.subgraphs()):
        values = chain(
            current_graph.inputs,
            current_graph.outputs,
            current_graph.initializers.values(),
            chain.from_iterable(chain(node.inputs, node.outputs) for node in current_graph),
        )
        for value in values:
            if value is None:
                continue
            if value.dtype is not None:
                yield value.dtype
            if value.const_value is not None:
                yield value.const_value.dtype

        for node in current_graph:
            for attribute in node.attributes.values():
                if attribute.type == ir.AttributeType.TENSOR:
                    yield attribute.as_tensor().dtype
                elif attribute.type == ir.AttributeType.TENSORS:
                    yield from (tensor.dtype for tensor in attribute.as_tensors())
                elif attribute.type == ir.AttributeType.SPARSE_TENSOR:
                    yield from _iter_sparse_tensor_dtypes(attribute.value)
                elif attribute.type == ir.AttributeType.SPARSE_TENSORS:
                    for sparse_tensor in attribute.value:
                        yield from _iter_sparse_tensor_dtypes(sparse_tensor)
                elif attribute.type == ir.AttributeType.TYPE_PROTO:
                    yield from _iter_type_dtypes(attribute.value.type)
                elif attribute.type == ir.AttributeType.TYPE_PROTOS:
                    for type_proto in attribute.value:
                        yield from _iter_type_dtypes(type_proto.type)


def minimum_ir_version(graph: ir.Graph, *, baseline: int = 11) -> int:
    """Return the minimum ONNX IR version required by tensor types in *graph*."""
    dtype_floors = {
        ir.DataType.FLOAT4E2M1: 11,
        ir.DataType.FLOAT8E8M0: 12,
    }
    return max(
        chain(
            (baseline,),
            (dtype_floors.get(dtype, baseline) for dtype in iter_graph_tensor_dtypes(graph)),
        )
    )
