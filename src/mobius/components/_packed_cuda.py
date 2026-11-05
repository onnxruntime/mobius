# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Small-row packed CUDA projections, with bounded tiles for longer inputs.

The ABI is supplied by ORT's MatMulPacked* contrib kernels (SM80+, BF16).
These kernels have a different reduction schedule from the general GGML ops;
selecting them is an explicit backend choice, not a bitwise-equivalent fusion.
"""

from __future__ import annotations

from collections.abc import Callable

import onnx_ir as ir
from onnxscript import OpBuilder

from mobius.components._scan_utils import rename_subgraph_values


def packed_row_tiles(
    op: OpBuilder,
    x: ir.Value,
    *,
    input_width: int,
    output_width: int,
    tile_rows: int,
    project: Callable[[OpBuilder, ir.Value], ir.Value],
) -> ir.Value:
    """Apply a row-independent kernel, using Scan only beyond its row limit."""
    if x.dtype is None:
        raise ValueError("Packed row tiling requires a known activation dtype")
    dtype = x.dtype
    op.builder.root.graph.opset_imports.setdefault("com.microsoft", 1)
    flat = op.Reshape(x, [-1, input_width])
    rows = op.Shape(flat, start=0, end=1)

    def short(branch_op: OpBuilder):
        # Bound even the unselected branch's static shape for contrib inference.
        return project(branch_op, branch_op.Slice(flat, [0], [tile_rows], [0]))

    def tiled(branch_op: OpBuilder):
        # Repeat only the last real row; CUDA Pad has no BF16 kernel.
        padding = branch_op.Mod(branch_op.Neg(rows), [tile_rows], fmod=0)
        indices = branch_op.Min(
            branch_op.Range(0, branch_op.Squeeze(branch_op.Add(rows, padding), [0]), 1),
            branch_op.Sub(rows, [1]),
        )
        padded = branch_op.Gather(flat, indices, axis=0)
        chunks = branch_op.Reshape(padded, [-1, tile_rows, input_width])
        chunk = ir.Value(
            name="packed_chunk",
            type=ir.TensorType(dtype),
            shape=ir.Shape([tile_rows, input_width]),
        )
        body = branch_op.builder.subgraph(
            project,
            inputs=[chunk],
            outputs=[ir.Value(name="packed_chunk_output")],
            name="packed_rows",
        )
        rename_subgraph_values(body, "packed_tile_body_")
        result = branch_op.Scan(chunks, body=body, num_scan_inputs=1)
        result.dtype = x.dtype
        result = branch_op.Reshape(result, [-1, output_width])
        return branch_op.Slice(result, [0], rows, [0])

    branches = [
        op.builder.subgraph(
            callback, inputs=[], outputs=[ir.Value(name=name + "_output")], name=name
        )
        for callback, name in ((short, "packed_short"), (tiled, "packed_tiled"))
    ]
    for branch in branches:
        rename_subgraph_values(branch, f"{branch.name}_")
    result = op.If(
        op.Squeeze(op.LessOrEqual(rows, [tile_rows]), [0]),
        then_branch=branches[0],
        else_branch=branches[1],
    )
    shape = op.Concat(op.Shape(x, end=-1), op.Constant(value_ints=[output_width]), axis=0)
    result = op.Reshape(result, shape)
    result.dtype = x.dtype
    if x.shape is not None:
        result.shape = ir.Shape([*x.shape[:-1], output_width])
    return result
