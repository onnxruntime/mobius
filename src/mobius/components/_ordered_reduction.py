# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Portable FP32 row reductions with an explicit vector/group/thread/tree order."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder


def ordered_row_reduce(
    op: OpBuilder, values: ir.Value, width: int, *, normalize: bool = False
) -> ir.Value:
    """Reduce the last axis in FP32, retaining that axis with length one.

    ``width`` must be a power of two in [1, 4096]. Leading dimensions are
    flattened to choose the reduction schedule dynamically: row counts below
    2/4/8/16 use at most 512/256/128/64/32 threads, respectively. Widths above
    128 use four adjacent values per thread; smaller widths use one.
    Each vector lane accumulates strided groups in order, then lanes add left
    to right. Thread reduction halves across warps before an adjacent-pair tree
    within the final warp. ``normalize`` multiplies the sum by 1/width.

    This spells out the finite-value association of the Torch 2.8 CUDA
    contiguous-row source reduction without relying on a provider's ReduceSum
    tree or a custom runtime op. It does not promise that arbitrary source
    versions/devices or exceptional-value handling have identical semantics.
    Sub-warp widths also support tiny model graphs.
    """
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not (1 <= width <= 4096 and width & (width - 1) == 0)
    ):
        raise ValueError("width must be a power of two in [1, 4096]")
    if values.dtype is not None and values.dtype not in (
        ir.DataType.FLOAT,
        ir.DataType.FLOAT16,
        ir.DataType.BFLOAT16,
    ):
        raise ValueError("ordered_row_reduce requires FLOAT, FLOAT16 or BFLOAT16 values")
    if values.shape is not None:
        if len(values.shape) == 0:
            raise ValueError("ordered_row_reduce requires a positive-rank input")
        last = values.shape[-1]
        if isinstance(last, int) and last != width:
            raise ValueError("last dimension must match width")

    wide = op.Cast(values, to=ir.DataType.FLOAT)
    vector = 4 if width > 128 else 1
    max_threads = min(width // vector, 512)
    max_groups = width // (min(width // vector, 32) * vector)
    rows = op.Div(op.Size(wide), width)
    threads = op.Constant(value_int=max_threads)
    for cutoff in (2, 4, 8, 16):
        threads = op.Where(
            op.GreaterOrEqual(rows, cutoff), min(width // vector, 512 // cutoff), threads
        )
    groups = op.Div(width, op.Mul(threads, vector))
    shape = [
        op.Unsqueeze(rows, [0]),
        op.Unsqueeze(groups, [0]),
        op.Unsqueeze(threads, [0]),
        op.Constant(value_ints=[vector]),
    ]
    grouped = op.Reshape(wide, op.Concat(*shape, axis=0), allowzero=1)
    # Pad groups/threads to fixed maxima without changing the source association.
    padding = op.Concat(
        op.Constant(value_ints=[0, 0, 0, 0, 0]),
        op.Unsqueeze(op.Sub(max_groups, groups), [0]),
        op.Constant(value_ints=[0, 0]),
        axis=0,
    )
    grouped = op.Pad(grouped, padding)  # (rows, max_groups, threads, vector)
    # Source accumulators start at +0. A shaped zero keeps providers from
    # treating scalar Add(0, x) as an identity and losing signed-zero semantics.
    first = op.Gather(grouped, 0, axis=1)
    total = op.Add(op.Expand(0.0, op.Shape(first)), first)
    for group in range(1, max_groups):
        total = op.Add(total, op.Gather(grouped, group, axis=1))
    result = op.Gather(total, 0, axis=-1)
    for lane in range(1, vector):
        result = op.Add(result, op.Gather(total, lane, axis=-1))
    padding = op.Concat(
        op.Constant(value_ints=[0, 0, 0]),
        op.Unsqueeze(op.Sub(max_threads, threads), [0]),
        axis=0,
    )
    result = op.Pad(result, padding)
    count = max_threads
    while count > 32:
        half = count // 2
        result = op.Add(
            op.Slice(result, [0], [half], [1]),
            op.Slice(result, [half], [count], [1]),
        )
        count = half
    while count > 1:
        shape = op.Concat(
            op.Unsqueeze(rows, [0]), op.Constant(value_ints=[count // 2, 2]), axis=0
        )
        pairs = op.Reshape(result, shape, allowzero=1)
        result = op.Add(op.Gather(pairs, 0, axis=-1), op.Gather(pairs, 1, axis=-1))
        count //= 2
    if normalize:
        result = op.Mul(result, 1.0 / width)
    restored = op.Reshape(
        result, op.Concat(op.Shape(values, end=-1), [1], axis=0), allowzero=1
    )
    restored.type = ir.TensorType(ir.DataType.FLOAT)
    if values.shape is not None:
        restored.shape = ir.Shape([*values.shape[:-1], 1])
    return restored
