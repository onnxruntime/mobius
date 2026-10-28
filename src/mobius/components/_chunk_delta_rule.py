# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Portable FP32 chunk and recurrent gated delta rules.

Inputs are expanded-head Q/K/V [B,T,H,K/V], decay/beta [B,T,H], and
state [B,H,K,V]. These helpers preserve source normalization/layout and
64-token chunk boundaries. The standard-ONNX triangular substitution is
mathematically equivalent to a unit-lower solve, not bit-identical to cuBLAS.
No custom runtime, static-cache, or CUDA graph capture capability is implied.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder

from mobius.components._ordered_reduction import ordered_row_reduce
from mobius.components._scan_utils import create_body_graph, rename_subgraph_values


def ordered_matrix_reduce_128(op: OpBuilder, value: ir.Value) -> ir.Value:
    """Sum axis -2 of FP32 [...,128,128] using the source vec4-output order.

    This is not a contiguous-row reduction after transposing: each of four
    lanes first accumulates eight strided groups, then four input warps merge.
    """
    if value.dtype is not None and value.dtype != ir.DataType.FLOAT:
        raise ValueError("ordered_matrix_reduce_128 requires FLOAT")
    if value.shape is not None and (
        len(value.shape) < 2 or any(isinstance(d, int) and d != 128 for d in value.shape[-2:])
    ):
        raise ValueError("ordered_matrix_reduce_128 requires [...,128,128]")
    grouped = op.Reshape(value, op.Concat(op.Shape(value, end=-2), [8, 4, 4, 128], axis=0))
    first = op.Gather(grouped, 0, axis=-4)
    total = op.Add(op.Expand(0.0, op.Shape(first)), first)
    for index in range(1, 8):
        total = op.Add(total, op.Gather(grouped, index, axis=-4))
    result = op.Gather(total, 0, axis=-3)
    for index in range(1, 4):
        result = op.Add(result, op.Gather(total, index, axis=-3))
    for count in (4, 2):
        half = count // 2
        result = op.Add(
            op.Slice(result, [0], [half], [-2]),
            op.Slice(result, [half], [count], [-2]),
        )
    result = op.Squeeze(result, [-2])
    result.type = ir.TensorType(ir.DataType.FLOAT)
    if value.shape is not None:
        result.shape = ir.Shape([*value.shape[:-2], 128])
    return result


def _normalized(op, value, width):
    wide = op.Cast(op.Transpose(value, perm=[0, 2, 1, 3]), to=ir.DataType.FLOAT)
    squared = ordered_row_reduce(op, op.Mul(wide, wide), width)
    return op.Mul(wide, op.Pow(op.Add(squared, 1e-6), -0.5))


def _prefix(op, value, width):
    columns = np.arange(width, dtype=np.int64)
    stride = 1
    while stride < width:
        boundary = (columns // (2 * stride)) * (2 * stride) + stride - 1
        value = op.Where(
            op.Constant(value=ir.tensor((columns % (2 * stride)) >= stride)),
            op.Add(value, op.Gather(value, op.Constant(value=ir.tensor(boundary)), axis=-1)),
            value,
        )
        stride *= 2
    return value


def _chunk_prefix(op, decay):
    full = _prefix(op, decay, 64)
    left = _prefix(op, op.Slice(decay, [0], [32], [-1]), 32)
    # Source switches to 32-element tiles above 32 rows. The carry enters
    # the next tile's FIRST element before its scan, not after its prefix.
    head = op.Add(op.Slice(decay, [32], [33], [-1]), op.Slice(left, [31], [32], [-1]))
    right = _prefix(op, op.Concat(head, op.Slice(decay, [33], [64], [-1]), axis=-1), 32)
    return op.Where(
        op.Greater(op.Div(op.Size(decay), 64), 32),
        op.Concat(left, right, axis=-1),
        full,
    )


def _unit_lower_solve(op, system, rhs, heads, width, name):
    """Forward substitution across 64 rows, with dynamic batch/chunk prefixes."""
    dtype = ir.TensorType(ir.DataType.FLOAT)
    carry = ir.Value(
        name="solution", type=dtype, shape=ir.Shape(["batch", heads, "chunks", 64, width])
    )
    row_a = ir.Value(
        name="system_row", type=dtype, shape=ir.Shape(["batch", heads, "chunks", 64])
    )
    row_b = ir.Value(
        name="rhs_row", type=dtype, shape=ir.Shape(["batch", heads, "chunks", width])
    )
    index = ir.Value(
        name="row_index", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
    )
    body, builder = create_body_graph([carry], [row_a, row_b, index], name)
    bop = builder.op
    columns = bop.Constant(value=ir.tensor(np.arange(64, dtype=np.int64)))
    lower = bop.Where(bop.Less(columns, index), row_a, 0.0)
    remembered = bop.ReduceSum(bop.Mul(bop.Unsqueeze(lower, [-1]), carry), [-2], keepdims=0)
    row = bop.Sub(row_b, remembered)
    updates = bop.Unsqueeze(row, [-2])
    next_carry = bop.ScatterElements(
        carry, bop.Expand(index, bop.Shape(updates)), updates, axis=-2
    )
    next_carry.type, next_carry.shape = dtype, carry.shape
    row.type, row.shape = dtype, row_b.shape
    next_carry.name, row.name = "next_solution", "solved_row"
    body.outputs.extend([next_carry, row])
    rename_subgraph_values(body, name + "_")
    result, _ = op.Scan(
        op.Expand(0.0, op.Shape(rhs)),
        system,
        rhs,
        op.Constant(value=ir.tensor(np.arange(64, dtype=np.int64))),
        body=body,
        num_scan_inputs=3,
        scan_input_axes=[3, 3, 0],
        scan_output_axes=[3],
        _outputs=2,
    )
    return result


def chunk_gated_delta_rule(
    op: OpBuilder,
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    decay: ir.Value,
    beta: ir.Value,
    state: ir.Value,
    *,
    heads: int,
    key_width: int,
    value_width: int,
    unit_lower_solver: Callable[[OpBuilder, ir.Value, ir.Value], ir.Value] | None = None,
) -> tuple[ir.Value, ir.Value]:
    """Run source-structured 64-token chunks, including a partial final chunk.

    Q/K normalization happens after head expansion and transposition. FP32
    output [B,T,H,V] and state [B,H,K,V] are returned; callers own narrowing
    and resetting incoming state at the beginning of a new sequence.
    An explicit ``unit_lower_solver`` may replace only the two FLOAT unit-lower
    systems with matching [B,H,chunks,64] prefixes; the default stays portable.
    """
    batch = op.Shape(query, start=0, end=1)
    length = op.Shape(query, start=1, end=2)
    padding = op.Mod(op.Sub(64, op.Mod(length, [64])), [64])
    count = op.Div(op.Add(length, padding), [64])
    q_shape = op.Concat(batch, [heads], count, [64, key_width], axis=0)
    v_shape = op.Concat(batch, [heads], count, [64, value_width], axis=0)
    d_shape = op.Concat(batch, [heads], count, [64], axis=0)
    pads4 = op.Concat([0, 0, 0, 0, 0, 0], padding, [0], axis=0)
    pads3 = op.Concat([0, 0, 0, 0, 0], padding, axis=0)
    query = op.Mul(_normalized(op, query, key_width), key_width**-0.5)
    key = _normalized(op, key, key_width)
    query, key, value = [
        op.Pad(x, pads4)
        for x in (
            query,
            key,
            op.Cast(op.Transpose(value, perm=[0, 2, 1, 3]), to=ir.DataType.FLOAT),
        )
    ]
    beta, decay = [
        op.Pad(op.Cast(op.Transpose(x, perm=[0, 2, 1]), to=ir.DataType.FLOAT), pads3)
        for x in (beta, decay)
    ]
    values = op.Reshape(op.Mul(value, op.Unsqueeze(beta, [-1])), v_shape)
    keys_beta = op.Reshape(op.Mul(key, op.Unsqueeze(beta, [-1])), q_shape)
    query, key = [op.Reshape(x, q_shape) for x in (query, key)]
    cumulative = _chunk_prefix(op, op.Reshape(decay, d_shape))
    upper = op.Constant(value=ir.tensor(np.triu(np.ones((64, 64), bool), 1)))
    pairwise = op.Exp(
        op.Where(
            upper,
            op.Constant(value_float=float("-inf")),
            op.Sub(op.Unsqueeze(cumulative, [-1]), op.Unsqueeze(cumulative, [-2])),
        )
    )
    key_t = op.Transpose(key, perm=[0, 1, 2, 4, 3])
    system = op.Mul(op.MatMul(keys_beta, key_t), pairwise)
    intra = op.Mul(op.MatMul(query, key_t), pairwise)
    exp_cumulative = op.Exp(cumulative)
    decayed_keys = op.Mul(keys_beta, op.Unsqueeze(exp_cumulative, [-1]))
    if unit_lower_solver is None:
        solved_values = _unit_lower_solve(
            op, system, values, heads, value_width, "value_solve"
        )
        solved_keys = _unit_lower_solve(
            op, system, decayed_keys, heads, key_width, "key_solve"
        )
    else:
        # Both RHS share the exact dynamic batch/head/chunk prefix; no broadcasting.
        for tensor, width in ((system, 64), (values, value_width), (decayed_keys, key_width)):
            tensor.type = ir.TensorType(ir.DataType.FLOAT)
            tensor.shape = ir.Shape(["batch", heads, "chunks", 64, width])
        solved_values = unit_lower_solver(op, system, values)
        solved_keys = unit_lower_solver(op, system, decayed_keys)
    weighted_query = op.Mul(query, op.Unsqueeze(exp_cumulative, [-1]))
    last = op.Slice(cumulative, [-1], [np.iinfo(np.int64).max], [-1])
    weighted_key = op.Mul(key, op.Unsqueeze(op.Exp(op.Sub(last, cumulative)), [-1]))
    chunk_decay = op.Unsqueeze(op.Exp(last), [-1])

    dtype = ir.TensorType(ir.DataType.FLOAT)
    state_input = ir.Value(
        name="state", type=dtype, shape=ir.Shape(["batch", heads, key_width, value_width])
    )
    scan_inputs = [
        ir.Value(name=name, type=dtype, shape=ir.Shape(["batch", heads, *shape]))
        for name, shape in (
            ("query", [64, key_width]),
            ("key", [64, key_width]),
            ("values", [64, value_width]),
            ("keys_solved", [64, key_width]),
            ("intra", [64, 64]),
            ("decay", [1, 1]),
        )
    ]
    body, builder = create_body_graph([state_input], scan_inputs, "delta_chunks")
    bop = builder.op
    q, k, v, ks, local, factor = scan_inputs
    corrected = bop.Sub(v, bop.MatMul(ks, state_input))
    output = bop.Add(bop.MatMul(q, state_input), bop.MatMul(local, corrected))
    next_state = bop.Add(
        bop.Mul(state_input, factor),
        bop.MatMul(bop.Transpose(k, perm=[0, 1, 3, 2]), corrected),
    )
    next_state.type = output.type = dtype
    next_state.shape = state_input.shape
    output.shape = ir.Shape(["batch", heads, 64, value_width])
    next_state.name, output.name = "next_state", "output"
    body.outputs.extend([next_state, output])
    rename_subgraph_values(body, "delta_chunks_")
    final, output = op.Scan(
        state,
        weighted_query,
        weighted_key,
        solved_values,
        solved_keys,
        intra,
        chunk_decay,
        body=body,
        num_scan_inputs=6,
        scan_input_axes=[2] * 6,
        scan_output_axes=[2],
        _outputs=2,
    )
    flat = op.Reshape(output, op.Concat(batch, [heads, -1, value_width], axis=0))
    flat = op.Slice(flat, [0], length, [2])
    return op.Transpose(flat, perm=[0, 2, 1, 3]), final


def recurrent_gated_delta_rule(
    op: OpBuilder,
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    decay: ir.Value,
    beta: ir.Value,
    state: ir.Value,
    *,
    key_width: int,
    value_width: int,
) -> tuple[ir.Value, ir.Value]:
    """Run one continuing token; query/key/value time dimension must equal one."""
    query = op.Squeeze(_normalized(op, query, key_width), [2])
    query = op.Mul(query, key_width**-0.5)
    key = op.Squeeze(_normalized(op, key, key_width), [2])
    value = op.Cast(op.Squeeze(value, [1]), to=ir.DataType.FLOAT)
    factor = op.Unsqueeze(op.Exp(op.Squeeze(decay, [1])), [-1, -2])
    decayed = op.Mul(state, factor)

    def reduce(matrix):
        if key_width == value_width == 128:
            return ordered_matrix_reduce_128(op, matrix)
        return op.ReduceSum(matrix, [-2], keepdims=0)

    remembered = reduce(op.Mul(decayed, op.Unsqueeze(key, [-1])))
    delta = op.Mul(op.Sub(value, remembered), op.Unsqueeze(op.Squeeze(beta, [1]), [-1]))
    updated = op.Add(decayed, op.Mul(op.Unsqueeze(key, [-1]), op.Unsqueeze(delta, [-2])))
    output = reduce(op.Mul(updated, op.Unsqueeze(query, [-1])))
    return op.Unsqueeze(output, [1]), updated
