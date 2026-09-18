# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for :func:`tensor_scatter_to_scatternd_rules`.

The rule rewrites the static-cache ``TensorScatter`` in-place KV write into
``ScatterND`` for EPs without a ``TensorScatter`` kernel (QNN HTP). Invariants:

1. **Numerical parity** — for batch=1, the ScatterND graph produces the same
   updated cache as ``TensorScatter`` for both prefill (S_q>1) and decode
   (S_q=1) with any ``write_indices`` offset.
2. **Op replacement** — no ``TensorScatter`` remains; a ``ScatterND`` appears.
3. **Safety** — only axis=1 scatters over a statically-sized cache are rewritten.
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
from onnxscript.rewriter import rewrite

from mobius.rewrite_rules import tensor_scatter_to_scatternd_rules


def _run(model: ir.Model, feeds: dict) -> np.ndarray:
    path = os.path.join(tempfile.mkdtemp(), "s.onnx")
    ir.save(model, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)[0]


def _build_scatter_model(max_len: int, dim: int, *, batch: int = 1) -> ir.Model:
    cache = ir.Value(
        name="cache",
        shape=ir.Shape([batch, max_len, dim]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    update = ir.Value(
        name="update",
        shape=ir.Shape([batch, "sq", dim]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    widx = ir.Value(name="widx", shape=ir.Shape([1]), type=ir.TensorType(ir.DataType.INT64))
    y = ir.Value(name="y")
    node = ir.Node(
        "",
        "TensorScatter",
        inputs=[cache, update, widx],
        outputs=[y],
        attributes=ir.convenience.convert_attributes({"axis": 1}),
    )
    graph = ir.Graph(
        inputs=[cache, update, widx],
        outputs=[y],
        nodes=[node],
        opset_imports={"": 24},
        name="ts",
    )
    return ir.Model(graph, ir_version=10)


def _count(model: ir.Model) -> Counter:
    return Counter(n.op_type for n in model.graph)


class TestTensorScatterToScatterND:
    def test_batch_greater_than_one_is_left_unchanged(self):
        model = _build_scatter_model(16, 4, batch=2)
        rewrite(model, pattern_rewrite_rules=tensor_scatter_to_scatternd_rules())
        assert _count(model).get("TensorScatter", 0) == 1
        assert _count(model).get("ScatterND", 0) == 0

    def _feeds(self, max_len, dim, sq, start):
        rng = np.random.default_rng(sq * 10 + start)
        return {
            "cache": rng.standard_normal((1, max_len, dim)).astype(np.float32),
            "update": rng.standard_normal((1, sq, dim)).astype(np.float32),
            "widx": np.array([start], dtype=np.int64),
        }

    def test_prefill_parity(self):
        max_len, dim = 16, 4
        feeds = self._feeds(max_len, dim, sq=5, start=0)
        ref = _run(_build_scatter_model(max_len, dim), feeds)
        model = _build_scatter_model(max_len, dim)
        rewrite(model, pattern_rewrite_rules=tensor_scatter_to_scatternd_rules())
        ops = _count(model)
        assert ops.get("TensorScatter", 0) == 0
        assert ops.get("ScatterND", 0) == 1
        np.testing.assert_allclose(_run(model, feeds), ref, rtol=0, atol=0)

    def test_decode_parity_with_offset(self):
        max_len, dim = 16, 4
        feeds = self._feeds(max_len, dim, sq=1, start=7)  # decode step at slot 7
        ref = _run(_build_scatter_model(max_len, dim), feeds)
        model = _build_scatter_model(max_len, dim)
        rewrite(model, pattern_rewrite_rules=tensor_scatter_to_scatternd_rules())
        np.testing.assert_allclose(_run(model, feeds), ref, rtol=0, atol=0)

    def test_rules_returns_rule_set(self):
        from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

        assert isinstance(tensor_scatter_to_scatternd_rules(), RewriteRuleSet)
