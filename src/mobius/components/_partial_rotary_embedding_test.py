# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Numerical tests for explicit FP32 partial rotations, independent of frequencies."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._testing.ort_inference import OnnxModelSession
from mobius.components import apply_partial_rotary_pos_emb
from mobius.tasks._base import _make_graph


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
@pytest.mark.parametrize("heads,head_dim,width", [(3, 12, 8), (2, 8, 8), (16, 256, 64)])
def test_partial_rotation_rounds_once_and_preserves_tail(
    dtype, interleaved, heads, head_dim, width
):
    graph, builder = _make_graph("partial_rotation")
    x = builder.input("x", dtype, ["batch", "time", heads * head_dim])
    cos = builder.input("cos", ir.DataType.FLOAT, ["batch", "time", width // 2])
    sin = builder.input("sin", ir.DataType.FLOAT, ["batch", "time", width // 2])
    output = apply_partial_rotary_pos_emb(
        builder.op,
        x,
        (cos, sin),
        num_heads=heads,
        head_dim=head_dim,
        rotary_embedding_dim=width,
        interleaved=interleaved,
    )
    assert output.shape == x.shape
    builder.add_output(output, "y")
    assert all(not node.domain and node.op_type != "RotaryEmbedding" for node in graph)
    assert sum(node.op_type == "Mul" for node in graph) == 4
    assert sum(node.op_type == "Add" for node in graph) == 2
    session = OnnxModelSession(
        ir.Model(graph, ir_version=11),
        intra_op_num_threads=1,
        graph_optimization_level="disabled",
    )
    rng = np.random.default_rng(108)
    try:
        for batch, length in ((1, 1), (2, 7), (1, 33)):
            values = rng.standard_normal((batch, length, heads, head_dim)).astype(
                dtype.numpy()
            )
            # Include signed zeros in the tail: preserving its values is not enough.
            if width < head_dim:
                values[..., width] = -0.0
            angles = rng.uniform(-20, 20, (batch, length, 1, width // 2)).astype(np.float32)
            cosine, sine = torch.from_numpy(np.cos(angles)), torch.from_numpy(np.sin(angles))
            wide = torch.from_numpy(values.astype(np.float32))
            prefix = wide[..., :width]
            if interleaved:
                even, odd = prefix[..., ::2], prefix[..., 1::2]
                expected_prefix = torch.stack(
                    (even * cosine - odd * sine, odd * cosine + even * sine), -1
                ).flatten(-2)
            else:
                duplicated_cos = torch.cat((cosine, cosine), -1)
                duplicated_sin = torch.cat((sine, sine), -1)
                first, second = prefix.chunk(2, -1)
                rotated = torch.cat((-second, first), -1)
                expected_prefix = prefix * duplicated_cos + rotated * duplicated_sin
            expected = np.concatenate(
                (expected_prefix.numpy().astype(dtype.numpy()), values[..., width:]), axis=-1
            ).reshape(batch, length, -1)
            actual = session.run(
                {
                    "x": values.reshape(batch, length, -1),
                    "cos": cosine.numpy().squeeze(-2),
                    "sin": sine.numpy().squeeze(-2),
                }
            )["y"]
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
            unsigned = np.uint32 if dtype == ir.DataType.FLOAT else np.uint16
            np.testing.assert_array_equal(
                actual.reshape(values.shape)[..., width:].view(unsigned),
                values[..., width:].view(unsigned),
            )
    finally:
        session.close()


@pytest.mark.parametrize(
    "heads,head_dim,width", [(0, 8, 4), (2, 0, 4), (2, 8, 0), (2, 8, 3), (2, 8, 10)]
)
def test_partial_rotation_rejects_invalid_geometry(heads, head_dim, width):
    _, builder = _make_graph("invalid_rotation")
    with pytest.raises(ValueError, match="even width"):
        apply_partial_rotary_pos_emb(
            builder.op,
            ir.Value(),
            (ir.Value(), ir.Value()),
            num_heads=heads,
            head_dim=head_dim,
            rotary_embedding_dim=width,
        )
