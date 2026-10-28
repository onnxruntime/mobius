# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU association tests, not Torch CPU or full-model CUDA parity goldens.

The schedule derives from the frozen source_norm28.py dynamic helper and its
source_reductions.cu proof (SHA256
6f56843f97b35bc40312208e0161384bb8dfc82449163ee266be21f9ac4a3bbd).
These tests independently simulate active lanes and shuffle-down instructions,
rather than reproducing the padded ONNX tree or using numpy/torch.sum.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from onnxscript import GraphBuilder

from mobius._testing.ort_inference import OnnxModelSession
from mobius.components import ordered_row_reduce


def _reference_row_reduce(values, *, normalize=False):
    width = values.shape[-1]
    flat = values.astype(np.float32).reshape(-1, width)
    rows = len(flat)
    vector = 4 if width > 128 else 1
    budget = 512
    for cutoff in (2, 4, 8, 16):
        if rows >= cutoff:
            budget = 512 // cutoff
    threads = min(width // vector, budget)
    lanes = []
    for lane in range(vector):
        columns = np.arange(threads) * vector + lane
        accumulators = np.add(np.float32(0), flat[:, columns])
        for offset in range(threads * vector, width, threads * vector):
            accumulators = np.add(accumulators, flat[:, columns + offset])
        lanes.append(accumulators)
    partial = lanes[0]
    for lane in lanes[1:]:
        partial = np.add(partial, lane)
    count = threads
    while count > 32:
        count //= 2
        partial = np.add(partial[:, :count], partial[:, count : count * 2])
    # Each source lane reads the old lane+stride value, not an already-updated one.
    stride = 1
    while stride < count:
        old = partial.copy()
        partial[:, : count - stride] = np.add(old[:, : count - stride], old[:, stride:count])
        stride *= 2
    result = partial[:, :1]
    if normalize:
        result = np.multiply(result, np.float32(1.0 / width))
    return result.reshape(*values.shape[:-1], 1)


@pytest.mark.parametrize("width", [32, 64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
def test_ordered_row_reduce_dynamic_schedule(width, normalize, dtype):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    builder = GraphBuilder(graph)
    x = builder.input("x", dtype, ["rows", width])
    y = ordered_row_reduce(builder.op, x, width, normalize=normalize)
    assert y.dtype == ir.DataType.FLOAT
    assert list(y.shape) == [ir.SymbolicDim("rows"), 1]
    builder.add_output(y, "y")
    assert all(node.domain == "" for node in graph)
    assert not any(node.op_type.startswith("Reduce") for node in graph)
    array_dtype = dtype.numpy()
    rng = np.random.default_rng(42)
    session = OnnxModelSession(ir.Model(graph, ir_version=11), intra_op_num_threads=1)
    try:
        for rows in (0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 30, 480, 960):
            values = (
                rng.standard_normal((rows, width))
                * np.exp2(rng.integers(-8, 8, (rows, width)))
            ).astype(array_dtype)
            actual = session.run({"x": values})["y"]
            expected = _reference_row_reduce(values, normalize=normalize)
            assert actual.dtype == np.float32
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    finally:
        session.close()


@pytest.mark.parametrize("shape", [(32,), (1, 32), (2, 8, 32), (2, 3, 4, 32), (2, 0, 32)])
def test_ordered_row_reduce_restores_rank_and_leading_dimensions(shape):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    builder = GraphBuilder(graph)
    x = builder.input("x", ir.DataType.FLOAT, [*[f"d{i}" for i in range(len(shape) - 1)], 32])
    y = ordered_row_reduce(builder.op, x, 32)
    builder.add_output(y, "y")
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    session = OnnxModelSession(ir.Model(graph, ir_version=11), intra_op_num_threads=1)
    try:
        actual = session.run({"x": values})["y"]
    finally:
        session.close()
    assert actual.shape == (*shape[:-1], 1)
    np.testing.assert_array_equal(actual, _reference_row_reduce(values))


@pytest.mark.parametrize("width", [1, 2, 4, 8, 16])
def test_ordered_row_reduce_subwarp_tiny_widths(width):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    builder = GraphBuilder(graph)
    x = builder.input("x", ir.DataType.FLOAT, ["rows", width])
    builder.add_output(ordered_row_reduce(builder.op, x, width), "y")
    session = OnnxModelSession(ir.Model(graph, ir_version=11), intra_op_num_threads=1)
    try:
        for rows in (0, 1, 2, 16):
            values = np.arange(rows * width, dtype=np.float32).reshape(rows, width)
            np.testing.assert_array_equal(
                session.run({"x": values})["y"], _reference_row_reduce(values)
            )
    finally:
        session.close()


@pytest.mark.parametrize("width", [0, -32, 3, 96, 8192, 32.0, True])
def test_ordered_row_reduce_rejects_unsupported_width(width):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    builder = GraphBuilder(graph)
    x = builder.input("x", ir.DataType.FLOAT, [1, 32])
    with pytest.raises(ValueError, match="power of two"):
        ordered_row_reduce(builder.op, x, width)


@pytest.mark.parametrize(
    "dtype,shape,message",
    [
        (ir.DataType.INT32, [2, 32], "requires FLOAT"),
        (ir.DataType.DOUBLE, [2, 32], "requires FLOAT"),
        (ir.DataType.FLOAT, [], "positive-rank"),
        (ir.DataType.FLOAT, [2, 64], "last dimension"),
    ],
)
def test_ordered_row_reduce_rejects_invalid_input(dtype, shape, message):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    builder = GraphBuilder(graph)
    x = builder.input("x", dtype, shape)
    with pytest.raises(ValueError, match=message):
        ordered_row_reduce(builder.op, x, 32)


@pytest.mark.parametrize("width,expected", [(128, 1.0), (256, 0.0)])
def test_ordered_row_reduce_preserves_vector_association(width, expected):
    values = np.zeros((1, width), dtype=np.float32)
    values[0, :4] = [2**24, 1, 1, -(2**24)]
    # Width 128 uses adjacent thread pairs; 256 first adds vec4 lanes left-to-right.
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    builder = GraphBuilder(graph)
    x = builder.input("x", ir.DataType.FLOAT, ["rows", width])
    builder.add_output(ordered_row_reduce(builder.op, x, width), "y")
    session = OnnxModelSession(ir.Model(graph, ir_version=11), intra_op_num_threads=1)
    try:
        assert session.run({"x": values})["y"].item() == expected
    finally:
        session.close()
    assert _reference_row_reduce(values).item() == expected


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_ordered_row_reduce_has_fp32_accumulation(dtype):
    values = np.full((1, 4096), 256, dtype=dtype.numpy())
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    builder = GraphBuilder(graph)
    x = builder.input("x", dtype, [1, 4096])
    builder.add_output(ordered_row_reduce(builder.op, x, 4096), "y")
    session = OnnxModelSession(ir.Model(graph, ir_version=11), intra_op_num_threads=1)
    try:
        actual = session.run({"x": values})["y"]
    finally:
        session.close()
    assert actual.dtype == np.float32
    assert actual.item() == 1_048_576


@pytest.mark.parametrize("width", [1, 32, 128, 256, 4096])
@pytest.mark.parametrize("normalize", [False, True])
def test_ordered_row_reduce_signed_zero_and_subnormal(width, normalize):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    builder = GraphBuilder(graph)
    x = builder.input("x", ir.DataType.FLOAT, ["rows", width])
    builder.add_output(ordered_row_reduce(builder.op, x, width, normalize=normalize), "y")
    session = OnnxModelSession(ir.Model(graph, ir_version=11), intra_op_num_threads=1)
    try:
        for rows in (1, 2, 4, 8, 16):
            for value in (np.float32(-0.0), np.float32(2**-140), np.float32(-(2**-140))):
                values = np.full((rows, width), value, dtype=np.float32)
                actual = session.run({"x": values})["y"]
                expected = _reference_row_reduce(values, normalize=normalize)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                if value == 0:
                    assert not np.signbit(actual).any()
    finally:
        session.close()
