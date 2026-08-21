# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the bfloat16 ``Clip`` → ``Min``/``Max`` lowering rule."""

from __future__ import annotations

from collections import Counter

import ml_dtypes
import numpy as np
import onnx_ir as ir
from onnxscript.rewriter import rewrite

from mobius._constants import OPSET_VERSION
from mobius.rewrite_rules import clip_to_min_max_rules

_NUMPY_DTYPE = {
    ir.DataType.BFLOAT16: ml_dtypes.bfloat16,
    ir.DataType.FLOAT: np.float32,
}


def _clip_model(dtype: ir.DataType, *, both_bounds: bool = True) -> ir.Model:
    """Build a single-node ``Clip`` graph of the given dtype."""
    np_dtype = _NUMPY_DTYPE[dtype]
    x = ir.Value(name="x", type=ir.TensorType(dtype), shape=ir.Shape(["batch", 4]))
    lo = ir.Value(
        name="lo",
        const_value=ir.tensor(np.array(-1.0, dtype=np_dtype)),
        type=ir.TensorType(dtype),
    )
    inputs = [x, lo]
    initializers = [lo]
    if both_bounds:
        hi = ir.Value(
            name="hi",
            const_value=ir.tensor(np.array(1.0, dtype=np_dtype)),
            type=ir.TensorType(dtype),
        )
        inputs.append(hi)
        initializers.append(hi)
    y = ir.Value(name="y", type=ir.TensorType(dtype), shape=ir.Shape(["batch", 4]))
    node = ir.Node("", "Clip", inputs=inputs, outputs=[y], name="clip")
    graph = ir.Graph(
        inputs=[x],
        outputs=[y],
        nodes=[node],
        initializers=initializers,
        opset_imports={"": OPSET_VERSION},
        name="clip_graph",
    )
    return ir.Model(graph, ir_version=10)


def _op_counts(model: ir.Model) -> Counter:
    return Counter(node.op_type for node in ir.traversal.RecursiveGraphIterator(model.graph))


def test_bfloat16_clip_lowers_to_min_max() -> None:
    model = _clip_model(ir.DataType.BFLOAT16)
    rewrite(model, pattern_rewrite_rules=clip_to_min_max_rules())
    counts = _op_counts(model)
    assert counts["Clip"] == 0
    assert counts["Min"] == 1
    assert counts["Max"] == 1


def test_bfloat16_single_bound_clip_lowers_to_max() -> None:
    model = _clip_model(ir.DataType.BFLOAT16, both_bounds=False)
    rewrite(model, pattern_rewrite_rules=clip_to_min_max_rules())
    counts = _op_counts(model)
    assert counts["Clip"] == 0
    assert counts["Max"] == 1
    assert counts["Min"] == 0


def test_float32_clip_is_untouched() -> None:
    model = _clip_model(ir.DataType.FLOAT)
    rewrite(model, pattern_rewrite_rules=clip_to_min_max_rules())
    counts = _op_counts(model)
    assert counts["Clip"] == 1
    assert counts["Min"] == 0
    assert counts["Max"] == 0


def test_lowering_is_numerically_exact() -> None:
    """``Min(Max(x, lo), hi)`` reproduces ``Clip(x, lo, hi)`` elementwise."""
    x = np.array([-3.0, -1.0, -0.25, 0.0, 0.5, 1.0, 7.0], dtype=np.float32)
    np.testing.assert_array_equal(np.minimum(np.maximum(x, -1.0), 1.0), np.clip(x, -1.0, 1.0))
