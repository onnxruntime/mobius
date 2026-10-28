# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._testing.ort_inference import OnnxModelSession
from mobius.components._chunk_delta_rule import (
    _chunk_prefix,
    chunk_gated_delta_rule,
    ordered_matrix_reduce_128,
    recurrent_gated_delta_rule,
)
from mobius.tasks._base import _make_graph


def _session(rule, heads, key_width=128, value_width=128):
    graph, builder = _make_graph("delta_rule")
    values = [
        builder.input(name, ir.DataType.FLOAT, ["batch", "time", heads, width])
        for name, width in (("q", key_width), ("k", key_width), ("v", value_width))
    ]
    values += [
        builder.input(name, ir.DataType.FLOAT, ["batch", "time", heads])
        for name in ("decay", "beta")
    ]
    values.append(
        builder.input("state", ir.DataType.FLOAT, ["batch", heads, key_width, value_width])
    )
    kwargs = dict(key_width=key_width, value_width=value_width)
    if rule is chunk_gated_delta_rule:
        kwargs["heads"] = heads
    output, state = rule(builder.op, *values, **kwargs)
    for name, value, shape in (
        ("output", output, ["batch", "time", heads, value_width]),
        ("present", state, ["batch", heads, key_width, value_width]),
    ):
        value.type, value.shape = ir.TensorType(ir.DataType.FLOAT), ir.Shape(shape)
        builder.add_output(value, name)
    assert all(node.domain == "" for node in graph)
    return OnnxModelSession(ir.Model(graph, ir_version=11), intra_op_num_threads=1)


def _sequential_reference(feeds):
    # Independent mathematical recurrence, not a second implementation of chunks.
    q, k, v, decay, beta, state = [
        torch.from_numpy(feeds[name]) for name in ("q", "k", "v", "decay", "beta", "state")
    ]
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * q.shape[-1] ** -0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    outputs = []
    for index in range(q.shape[1]):
        state = state * decay[:, index].exp()[..., None, None]
        key = k[:, index]
        remembered = (state * key[..., None]).sum(-2)
        delta = (v[:, index] - remembered) * beta[:, index, :, None]
        state = state + key[..., None] * delta[..., None, :]
        outputs.append((state * q[:, index, :, :, None]).sum(-2))
    return torch.stack(outputs, dim=1).numpy(), state.numpy()


@pytest.mark.parametrize("heads", [2, 32])
def test_chunks_match_independent_recurrence_across_dynamic_batches_and_boundaries(heads):
    rng = np.random.default_rng(42)
    session = _session(chunk_gated_delta_rule, heads)
    try:
        for batch in (1, 2):
            for length in (1, 2, 30, 64, 65, 128, 129, 257):
                shape = (batch, length, heads, 128)
                feeds = {
                    name: rng.normal(0, 0.2, shape).astype(np.float32)
                    for name in ("q", "k", "v")
                }
                feeds.update(
                    decay=-rng.uniform(0.01, 0.2, shape[:-1]).astype(np.float32),
                    beta=rng.uniform(0.1, 0.9, shape[:-1]).astype(np.float32),
                    state=rng.normal(0, 0.02, (batch, heads, 128, 128)).astype(np.float32),
                )
                expected, state = _sequential_reference(feeds)
                actual = session.run(feeds)
                np.testing.assert_allclose(actual["output"], expected, atol=1e-4, rtol=1e-4)
                np.testing.assert_allclose(actual["present"], state, atol=1e-4, rtol=1e-4)
    finally:
        session.close()


def test_recurrent_128_matrix_reduction_and_non_square_tiny_dimensions():
    rng = np.random.default_rng(7)
    for key_width, value_width in ((128, 128), (16, 8)):
        session = _session(recurrent_gated_delta_rule, 2, key_width, value_width)
        try:
            feeds = {
                name: rng.normal(0, 0.2, (2, 1, 2, width)).astype(np.float32)
                for name, width in (("q", key_width), ("k", key_width), ("v", value_width))
            }
            feeds.update(
                decay=np.full((2, 1, 2), -0.02, np.float32),
                beta=np.full((2, 1, 2), 0.7, np.float32),
                state=rng.normal(0, 0.02, (2, 2, key_width, value_width)).astype(np.float32),
            )
            expected, state = _sequential_reference(feeds)
            actual = session.run(feeds)
            np.testing.assert_allclose(actual["output"], expected, atol=1e-4, rtol=1e-4)
            np.testing.assert_allclose(actual["present"], state, atol=1e-4, rtol=1e-4)
        finally:
            session.close()


@pytest.mark.parametrize("rows", [1, 32, 33, 64, 128])
def test_source_prefix_tile_schedule_and_carry_association(rows):
    graph, builder = _make_graph("prefix")
    x = builder.input("x", ir.DataType.FLOAT, ["rows", 64])
    output = _chunk_prefix(builder.op, x)
    output.type, output.shape = x.type, x.shape
    builder.add_output(output, "output")
    rng = np.random.default_rng(76)
    values = rng.uniform(-5, -0.0001, (rows, 64)).astype(np.float32)

    def prefix(array):
        array = array.copy()
        width = array.shape[-1]
        columns = np.arange(width)
        stride = 1
        while stride < width:
            boundary = (columns // (2 * stride)) * (2 * stride) + stride - 1
            array = np.where(
                columns % (2 * stride) >= stride, array + array[..., boundary], array
            )
            stride *= 2
        return array

    expected = prefix(values)
    if rows > 32:
        left = prefix(values[:, :32])
        right = values[:, 32:].copy()
        right[:, 0] += left[:, -1]
        expected = np.concatenate((left, prefix(right)), axis=-1)
        post_add = prefix(values[:, 32:]) + left[:, -1:]
        assert np.any(post_add.view(np.uint32) != expected[:, 32:].view(np.uint32))
    session = OnnxModelSession(ir.Model(graph, ir_version=11), intra_op_num_threads=1)
    try:
        actual = session.run({"x": values})["output"]
    finally:
        session.close()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_ordered_matrix_reduction_layout_is_not_transposed_row_sum():
    rng = np.random.default_rng(48)
    values = rng.normal(0, 100, (2, 3, 128, 128)).astype(np.float32)
    grouped = values.reshape(2, 3, 8, 4, 4, 128)
    total = np.zeros_like(grouped[:, :, 0])
    for group in range(8):
        total += grouped[:, :, group]
    expected = total[:, :, 0].copy()
    for lane in range(1, 4):
        expected += total[:, :, lane]
    expected = expected[:, :, :2] + expected[:, :, 2:]
    expected = expected[:, :, 0] + expected[:, :, 1]
    assert np.any(expected != values.sum(-2))
    graph, builder = _make_graph("matrix_reduction")
    x = builder.input("x", ir.DataType.FLOAT, ["batch", 3, 128, 128])
    builder.add_output(ordered_matrix_reduce_128(builder.op, x), "output")
    session = OnnxModelSession(ir.Model(graph, ir_version=11), intra_op_num_threads=1)
    try:
        actual = session.run({"x": values})["output"]
    finally:
        session.close()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
