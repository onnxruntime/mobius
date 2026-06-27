# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
from onnxscript import GraphBuilder
from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius._constants import OPSET_VERSION
from mobius.rewrite_rules import htp_rank4_rmsnorm_rules
from mobius.rewrite_rules._testing_utils import count_ops


def _rmsnorm_model(shape: list[int]) -> ir.Model:
    """A single ai.onnx::RMSNormalization (norm over the last axis) of *shape*."""
    rng = np.random.default_rng(0)
    x = ir.Value(name="x", shape=ir.Shape(shape), type=ir.TensorType(ir.DataType.FLOAT))
    graph = ir.Graph(
        inputs=[x],
        outputs=[],
        nodes=[],
        name="rmsnorm",
        opset_imports={"": OPSET_VERSION},
    )
    op = GraphBuilder(graph).op
    weight = op.Constant(value=ir.tensor(rng.standard_normal(shape[-1]).astype(np.float32)))
    y = op.RMSNormalization(x, weight, epsilon=1e-6)
    y.name = "y"
    graph.outputs.append(y)
    return ir.Model(graph, ir_version=11)


def _run(model: ir.Model, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(
        ir.serde.serialize_model(model).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    return sess.run(["y"], {"x": x})[0]


class TestHtpRank4RMSNorm:
    def test_rules_returns_rule_set(self):
        assert isinstance(htp_rank4_rmsnorm_rules(), RewriteRuleSet)

    def test_reshapes_rank4_and_is_numerically_exact(self):
        # (batch, seq, heads, head_dim) — the query/key-norm shape.
        model = _rmsnorm_model([1, 3, 4, 8])
        x = np.random.default_rng(1).standard_normal((1, 3, 4, 8)).astype(np.float32)
        before = _run(model, x)

        rewrite(model, pattern_rewrite_rules=htp_rank4_rmsnorm_rules())

        counts = count_ops(model)
        assert counts["RMSNormalization"] == 1  # still one norm, now over a rank-3 tensor
        assert counts.get("Reshape", 0) == 2  # reshaped down then back

        after = _run(model, x)
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)

    def test_rank3_rmsnorm_untouched(self):
        # A hidden-state RMSNorm is rank-3 and must not be rewritten.
        model = _rmsnorm_model([1, 3, 8])
        rewrite(model, pattern_rewrite_rules=htp_rank4_rmsnorm_rules())
        counts = count_ops(model)
        assert counts.get("Reshape", 0) == 0
        assert counts["RMSNormalization"] == 1
