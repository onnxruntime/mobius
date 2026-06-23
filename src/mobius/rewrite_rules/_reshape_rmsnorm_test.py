# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
from onnx import TensorProto, helper
from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius.rewrite_rules import reshape_rank4_rmsnorm_rules
from mobius.rewrite_rules._testing_utils import count_ops


def _rmsnorm_proto(shape: list[int]):
    """A single ai.onnx::RMSNormalization (norm over the last axis) of *shape*."""
    rng = np.random.default_rng(0)
    weight = helper.make_tensor(
        "weight", TensorProto.FLOAT, [shape[-1]],
        rng.standard_normal(shape[-1]).astype(np.float32),
    )
    node = helper.make_node("RMSNormalization", ["x", "weight"], ["y"], epsilon=1e-6)
    graph = helper.make_graph(
        [node], "rmsnorm",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    model.ir_version = 10
    return model


def _run(proto, x):
    sess = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(["y"], {"x": x})[0]


class TestReshapeRank4RMSNorm:
    def test_rules_returns_rule_set(self):
        assert isinstance(reshape_rank4_rmsnorm_rules(), RewriteRuleSet)

    def test_reshapes_rank4_and_is_numerically_exact(self):
        # (batch, seq, heads, head_dim) — the query/key-norm shape.
        proto = _rmsnorm_proto([1, 3, 4, 8])
        x = np.random.default_rng(1).standard_normal((1, 3, 4, 8)).astype(np.float32)
        before = _run(proto, x)

        model = ir.from_proto(proto)
        rewrite(model, pattern_rewrite_rules=reshape_rank4_rmsnorm_rules())

        counts = count_ops(model)
        assert counts["RMSNormalization"] == 1  # still one norm, now over a rank-3 tensor
        assert counts.get("Reshape", 0) == 2  # reshaped down then back

        after = _run(ir.to_proto(model), x)
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)

    def test_rank3_rmsnorm_untouched(self):
        # A hidden-state RMSNorm is rank-3 and must not be rewritten.
        model = ir.from_proto(_rmsnorm_proto([1, 3, 8]))
        rewrite(model, pattern_rewrite_rules=reshape_rank4_rmsnorm_rules())
        counts = count_ops(model)
        assert counts.get("Reshape", 0) == 0
        assert counts["RMSNormalization"] == 1
