# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for :func:`decompose_rope_rules`.

The rule rewrites the opset-24 ``RotaryEmbedding`` op into rotate-half
primitives for EPs without a ``RotaryEmbedding`` kernel (QNN HTP). Invariants:

1. **Numerical parity** — the decomposed graph is identical to the fused op for
   the non-interleaved / full-rotation form mobius emits.
2. **Op replacement** — no ``RotaryEmbedding`` remains; rotate-half primitives
   (Slice/Mul/Sub/Add/Concat) appear.
3. **Safety** — interleaved / partial-rotary nodes are left unchanged.
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
from onnxscript.rewriter import rewrite

from mobius.rewrite_rules import decompose_rope_rules


def _run(model: ir.Model, feeds: dict) -> np.ndarray:
    path = os.path.join(tempfile.mkdtemp(), "r.onnx")
    ir.save(model, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)[0]


def _build_rope_model(
    batch: int, seq: int, num_heads: int, head_dim: int, *, interleaved: int = 0
) -> ir.Model:
    hidden = num_heads * head_dim
    half = head_dim // 2
    x = ir.Value(
        name="x", shape=ir.Shape([batch, seq, hidden]), type=ir.TensorType(ir.DataType.FLOAT)
    )
    cos = ir.Value(
        name="cos", shape=ir.Shape([batch, seq, half]), type=ir.TensorType(ir.DataType.FLOAT)
    )
    sin = ir.Value(
        name="sin", shape=ir.Shape([batch, seq, half]), type=ir.TensorType(ir.DataType.FLOAT)
    )
    y = ir.Value(name="y")
    node = ir.Node(
        "",
        "RotaryEmbedding",
        inputs=[x, cos, sin],
        outputs=[y],
        attributes=ir.convenience.convert_attributes(
            {"num_heads": num_heads, "rotary_embedding_dim": 0, "interleaved": interleaved}
        ),
    )
    graph = ir.Graph(
        inputs=[x, cos, sin], outputs=[y], nodes=[node], opset_imports={"": 24}, name="rope"
    )
    return ir.Model(graph, ir_version=10)


def _count(model: ir.Model) -> Counter:
    return Counter(n.op_type for n in model.graph)


class TestDecomposeRope:
    def test_rank2_cos_or_sin_is_left_unchanged(self):
        for table_index in (1, 2):
            model = _build_rope_model(1, 4, 2, 8)
            table = next(iter(model.graph)).inputs[table_index]
            table.shape = ir.Shape([4, 4])
            rewrite(model, pattern_rewrite_rules=decompose_rope_rules())
            assert _count(model).get("RotaryEmbedding", 0) == 1

    def test_parity_and_replacement(self):
        b, s, n, h = 1, 4, 3, 8
        rng = np.random.default_rng(0)
        feeds = {
            "x": rng.standard_normal((b, s, n * h)).astype(np.float32),
            "cos": rng.standard_normal((b, s, h // 2)).astype(np.float32),
            "sin": rng.standard_normal((b, s, h // 2)).astype(np.float32),
        }
        ref = _run(_build_rope_model(b, s, n, h), feeds)

        model = _build_rope_model(b, s, n, h)
        rewrite(model, pattern_rewrite_rules=decompose_rope_rules())
        ops = _count(model)
        assert ops.get("RotaryEmbedding", 0) == 0
        assert ops.get("Concat", 0) >= 1 and ops.get("Slice", 0) >= 2
        got = _run(model, feeds)
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5)

    def test_gqa_head_count(self):
        # num_heads != kv path: q uses many heads, still decomposes exactly.
        b, s, n, h = 2, 3, 4, 16
        rng = np.random.default_rng(1)
        feeds = {
            "x": rng.standard_normal((b, s, n * h)).astype(np.float32),
            "cos": rng.standard_normal((b, s, h // 2)).astype(np.float32),
            "sin": rng.standard_normal((b, s, h // 2)).astype(np.float32),
        }
        ref = _run(_build_rope_model(b, s, n, h), feeds)
        model = _build_rope_model(b, s, n, h)
        rewrite(model, pattern_rewrite_rules=decompose_rope_rules())
        assert _count(model).get("RotaryEmbedding", 0) == 0
        np.testing.assert_allclose(_run(model, feeds), ref, rtol=0, atol=1e-5)

    def test_interleaved_left_unchanged(self):
        model = _build_rope_model(1, 4, 2, 8, interleaved=1)
        rewrite(model, pattern_rewrite_rules=decompose_rope_rules())
        assert _count(model).get("RotaryEmbedding", 0) == 1  # not decomposed

    def test_rules_returns_rule_set(self):
        from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

        assert isinstance(decompose_rope_rules(), RewriteRuleSet)
