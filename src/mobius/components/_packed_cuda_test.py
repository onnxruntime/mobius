# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Execute both short and tiled paths without requiring contrib kernels."""

from __future__ import annotations

import os

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnxscript import GraphBuilder

from mobius._constants import OPSET_VERSION
from mobius.components import packed_row_tiles


@pytest.mark.parametrize("shape", [(1, 3), (8, 3), (9, 3), (2, 9, 3), (3,)])
def test_packed_row_tiles_preserves_order_tail_and_rank(tmp_path, shape):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": OPSET_VERSION})
    builder = GraphBuilder(graph)
    x = builder.input("x", dtype=ir.DataType.FLOAT, shape=[None] * len(shape))

    def project(op, rows):
        return op.Add(rows, 7.0)

    result = packed_row_tiles(
        builder.op, x, input_width=3, output_width=3, tile_rows=8, project=project
    )
    builder.add_output(result, "y")
    path = tmp_path / "tiles.onnx"
    ir.save(ir.Model(graph, ir_version=10), path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(session.run(None, {"x": data})[0], data + 7.0)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("MOBIUS_TEST_PACKED_CUDA") != "1",
    reason="Requires an explicitly selected CUDA ORT build containing the chosen packed ABI",
)
@pytest.mark.parametrize("rows", [1, 8, 9, 17])
@pytest.mark.parametrize("native_blocks", [False, True])
def test_packed_row_tiles_executes_cuda_projection(tmp_path, rows, native_blocks):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": OPSET_VERSION, "com.microsoft": 1})
    builder = GraphBuilder(graph)
    n, k = 19, 256
    data = np.zeros((rows, k), np.float32)
    positions = np.arange(rows) % 8 * 32 + 3
    data[np.arange(rows), positions] = 1
    codes = np.arange(n * k, dtype=np.uint8).reshape(n, k)
    scales = np.full((n, k // 32), 0.001003, np.float16)
    storage = (
        ("payload", (codes ^ np.uint8(128)).reshape(n, k // 32, 32)),
        ("scales", scales),
    )
    if native_blocks:
        native = np.concatenate(
            (scales.view(np.uint8).reshape(n, k // 32, 2), codes.reshape(n, k // 32, 32)),
            axis=-1,
        ).reshape(n, -1)
        storage = (("native", native),)
    initializers = []
    for name, array in storage:
        value = ir.Value(
            name=name,
            type=ir.TensorType(ir.DataType.from_numpy(array.dtype)),
            shape=ir.Shape(array.shape),
            const_value=ir.tensor(array),
        )
        graph.initializers[name] = value
        initializers.append(value)
    x = builder.input("x", dtype=ir.DataType.FLOAT, shape=[None, k])
    x = builder.op.Cast(x, to=ir.DataType.BFLOAT16)

    def project(op, tile):
        if native_blocks:
            result = op.MatMulPacked(
                tile, *initializers, K=k, N=n, quant_type="q8_0", _domain="com.microsoft"
            )
        else:
            result = op.MatMulPackedQ8(tile, *initializers, _domain="com.microsoft")
        result.dtype = ir.DataType.BFLOAT16
        result.shape = ir.Shape([None, n])
        return result

    result = packed_row_tiles(
        builder.op, x, input_width=k, output_width=n, tile_rows=8, project=project
    )
    builder.add_output(builder.op.Cast(result, to=ir.DataType.FLOAT), "y")
    path = tmp_path / "cuda_tiles.onnx"
    ir.save(ir.Model(graph, ir_version=10), path)
    session = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider"])
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    expected = (
        (codes.view(np.int8)[:, positions].T.astype(np.float32) * np.float32(scales[0, 0]))
        .astype(ml_dtypes.bfloat16)
        .astype(np.float32)
    )
    np.testing.assert_array_equal(session.run(None, {"x": data})[0], expected)
