# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Lazily fold static weight Reshape/Transpose chains after checkpoint loading."""

from __future__ import annotations

import logging
import math

import onnx_ir as ir

from mobius._constants import LAZY_SOURCE_FILES_META_KEY
from mobius._passes._dtype_utils import initializer_dtype

logger = logging.getLogger(__name__)


def _reshape_shape(node: ir.Node, source_shape: tuple[int, ...]) -> tuple[int, ...] | None:
    """Resolve a static shape, skip dynamic shapes, and reject invalid reshapes."""
    shape_value = node.inputs[1]
    if shape_value is None or shape_value.is_graph_input():
        return None
    if shape_value.is_initializer():
        shape_tensor = shape_value.const_value
    else:
        constant = shape_value.producer()
        if (
            constant is None
            or constant.domain not in ("", "ai.onnx")
            or constant.op_type != "Constant"
            or len(constant.attributes) != 1
        ):
            return None
        if "value" in constant.attributes:
            shape_tensor = constant.attributes["value"].as_tensor()
        elif "value_ints" in constant.attributes:
            shape_tensor = ir.tensor(
                constant.attributes["value_ints"].as_ints(), dtype=ir.DataType.INT64
            )
        else:
            return None
    if shape_tensor is None:
        return None
    if shape_tensor.dtype != ir.DataType.INT64 or len(shape_tensor.shape) != 1:
        raise ValueError(f"Reshape {node.name!r} requires a one-dimensional INT64 shape")
    requested = tuple(int(dim) for dim in shape_tensor.numpy())
    allowzero = node.attributes.get_int("allowzero", 0)
    if allowzero not in (0, 1) or requested.count(-1) > 1 or any(d < -1 for d in requested):
        raise ValueError(f"Invalid Reshape attributes or shape at {node.name!r}: {requested}")
    resolved = list(requested)
    if not allowzero:
        for index, dim in enumerate(resolved):
            if dim == 0:
                if index >= len(source_shape):
                    raise ValueError(f"Reshape {node.name!r} copies a nonexistent dimension")
                resolved[index] = source_shape[index]
    size = math.prod(source_shape)
    known_size = math.prod(dim for dim in resolved if dim != -1)
    if -1 in resolved:
        if known_size == 0 or size % known_size:
            raise ValueError(f"Reshape {node.name!r} cannot infer dimension in {requested}")
        resolved[resolved.index(-1)] = size // known_size
    elif known_size != size:
        raise ValueError(
            f"Reshape {node.name!r} changes element count: {source_shape} to {requested}"
        )
    return tuple(resolved)


class FoldWeightLayoutPass(ir.passes.InPlacePass):
    """Fold named static Reshape/Transpose outputs into lazy initializers.

    Graphs must be topologically ordered. A single traversal folds layout chains
    without reading weight payloads: only constant Reshape shape vectors are read.
    Materialization preserves the source dtype and uses no arithmetic or casts.
    Overridable initializers, shared sources, and dynamic shapes remain untouched.
    Dead folded inputs are removed in their owning graph; run dead-code elimination
    afterward to remove unused Constant shape nodes.
    Malformed static layout nodes raise ValueError rather than being skipped.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        folded = 0
        graphs = tuple(model.graphs())

        # Registry identity is authoritative, including parent-owned captures.
        # Do not infer ownership from the graph containing the consumer node.
        def owners(value: ir.Value) -> list[ir.Graph]:
            return [
                scope
                for scope in graphs
                if value.name is not None and scope.initializers.get(value.name) is value
            ]

        for graph in graphs:
            for node in list(graph):
                if (
                    node.overload
                    or node.domain not in ("", "ai.onnx")
                    or node.op_type
                    not in (
                        "Reshape",
                        "Transpose",
                    )
                ):
                    continue
                expected_inputs = 2 if node.op_type == "Reshape" else 1
                if len(node.inputs) != expected_inputs or len(node.outputs) != 1:
                    raise ValueError(f"Malformed weight layout node {node.name!r}")
                source = node.inputs[0]
                if source is None or not source.is_initializer() or source.is_graph_input():
                    continue
                # Keep tied tables shared instead of serializing a second full layout.
                if (
                    len(source.uses()) != 1
                    or len(owners(source)) != 1
                    or any(
                        source in scope.inputs or source in scope.outputs for scope in graphs
                    )
                ):
                    continue
                tensor = source.const_value
                if tensor is None:
                    logger.warning(
                        "FoldWeightLayoutPass: skipping %r; initializer %r has no data",
                        node.name,
                        source.name,
                    )
                    continue
                output = node.outputs[0]
                if not output.name or any(not isinstance(dim, int) for dim in tensor.shape):
                    logger.debug(
                        "FoldWeightLayoutPass: skipping unnamed or symbolic %r", node.name
                    )
                    continue
                source_shape = tuple(tensor.shape.numpy())
                dtype = initializer_dtype(source)
                if dtype is None:
                    raise ValueError(f"Weight layout source {source.name!r} has no dtype")
                is_transpose = node.op_type == "Transpose"
                if is_transpose:
                    layout = tuple(
                        node.attributes.get_ints(
                            "perm", tuple(reversed(range(len(source_shape))))
                        )
                    )
                    if sorted(layout) != list(range(len(source_shape))):
                        raise ValueError(
                            f"Invalid Transpose permutation at {node.name!r}: {layout}"
                        )
                    shape = tuple(source_shape[index] for index in layout)
                else:
                    resolved = _reshape_shape(node, source_shape)
                    if resolved is None:
                        logger.debug(
                            "FoldWeightLayoutPass: skipping dynamic shape at %r", node.name
                        )
                        continue
                    layout = shape = resolved

                # Bind this iteration's values before the lazy callback runs later.
                def materialize(
                    data: ir.TensorProtocol = tensor,
                    dimensions: tuple[int, ...] = layout,
                    transpose: bool = is_transpose,
                    data_type: ir.DataType = dtype,
                ) -> ir.TensorProtocol:
                    array = data.numpy()
                    result = (
                        array.transpose(dimensions) if transpose else array.reshape(dimensions)
                    )
                    return ir.tensor(result, dtype=data_type)

                output_shape = ir.Shape(shape)
                replacement = ir.Value(
                    name=output.name,
                    shape=output_shape,
                    type=ir.TensorType(dtype),
                    doc_string=output.doc_string,
                    metadata_props=dict(output.metadata_props),
                )
                replacement.const_value = ir.LazyTensor(
                    materialize, dtype=dtype, shape=output_shape, name=output.name
                )
                source_files = (
                    (tensor.path,)
                    if isinstance(tensor, ir.ExternalTensor)
                    else tensor.meta.get(LAZY_SOURCE_FILES_META_KEY, ())
                )
                if source_files:
                    replacement.const_value.meta[LAZY_SOURCE_FILES_META_KEY] = tuple(
                        source_files
                    )
                    model.meta[LAZY_SOURCE_FILES_META_KEY] = frozenset(
                        (*model.meta.get(LAZY_SOURCE_FILES_META_KEY, ()), *source_files)
                    )
                output.replace_all_uses_with(replacement, replace_graph_outputs=True)
                inputs = tuple(node.inputs)
                graph.remove(node, safe=True)
                # Downstream layout nodes can now capture this lazy initializer.
                graph.register_initializer(replacement)
                for value in inputs:
                    if (
                        value is not None
                        and value.name is not None
                        and not value.uses()
                        and not any(
                            value in scope.inputs or value in scope.outputs for scope in graphs
                        )
                    ):
                        registries = owners(value)
                        if len(registries) == 1:
                            del registries[0].initializers[value.name]
                folded += 1
        logger.debug("FoldWeightLayoutPass: folded %d weight layout nodes", folded)
        return ir.passes.PassResult(model, modified=folded > 0)
