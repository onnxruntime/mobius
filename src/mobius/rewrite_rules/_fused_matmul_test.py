# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.rewrite_rules import fused_matmul_rules
from mobius.rewrite_rules._testing_utils import (
    count_ops,
    fill_random_weights,
    make_prefill_feeds,
)

_TINY_CONFIG = ArchitectureConfig(
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    num_hidden_layers=2,
    vocab_size=256,
    max_position_embeddings=128,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    rope_type="default",
    rope_theta=10000.0,
    pad_token_id=0,
)


class TestFusedMatMulRules:
    def test_rules_returns_rule_set(self):
        rules = fused_matmul_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_linear_emits_transpose_matmul_before_optimization(self):
        """Linear.forward() emits Transpose(weight,[1,0]) + MatMul(x, w_t).

        Before optimization, the raw graph produced by build_from_module
        (pre-optimization) has Transpose+MatMul from every Linear layer and
        no FusedMatMul nodes.
        """
        module = registry.get("qwen2")(_TINY_CONFIG)
        # Build without optimization to inspect the raw graph
        from mobius.tasks import get_task

        task = get_task("text-generation")
        pkg = task.build(module, _TINY_CONFIG)
        model = pkg["model"]

        counts = count_ops(model)
        # Every Linear emits Transpose+MatMul; no FusedMatMul before optimization
        assert counts.get("MatMul", 0) > 0
        assert counts.get("Transpose", 0) > 0
        assert counts.get("FusedMatMul", 0) == 0

    def test_fused_matmul_rules_convert_transpose_matmul(self):
        """fused_matmul_rules() converts Transpose(w,[1,0])+MatMul → FusedMatMul.

        Applies to mobius models (pre-optimization raw graph) and external models alike.
        """
        module = registry.get("qwen2")(_TINY_CONFIG)
        from mobius.tasks import get_task

        task = get_task("text-generation")
        pkg = task.build(module, _TINY_CONFIG)
        m = pkg["model"]

        counts_before = count_ops(m)
        assert counts_before.get("MatMul", 0) > 0
        assert counts_before.get("FusedMatMul", 0) == 0

        rewrite(m, pattern_rewrite_rules=fused_matmul_rules())

        counts_after = count_ops(m)
        assert counts_after.get("FusedMatMul", 0) > 0
        assert counts_after.get("MatMul", 0) == 0

    def test_build_from_module_has_fused_matmul_for_default_ep(self):
        """After build_from_module (default EP), model has FusedMatMul nodes.

        build_from_module applies optimize_model which runs fused_matmul_rules
        when caps.supports_fused_matmul is True (default EP).
        """
        module = registry.get("qwen2")(_TINY_CONFIG)
        pkg = build_from_module(module, _TINY_CONFIG)
        m = pkg["model"]

        counts = count_ops(m)
        # Default EP supports FusedMatMul → fused_matmul_rules applied
        assert counts.get("FusedMatMul", 0) > 0
        assert counts.get("MatMul", 0) == 0

    def test_fused_matmul_model_runs_with_ort(self):
        """Model with FusedMatMul (from default EP optimization) runs with ORT."""
        module = registry.get("qwen2")(_TINY_CONFIG)
        pkg = build_from_module(module, _TINY_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        counts = count_ops(m)
        assert counts.get("FusedMatMul", 0) > 0

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()

    def test_combined_with_skip_norm(self):
        """SkipNorm + FusedMatMul model runs correctly with ORT."""
        from mobius.rewrite_rules import skip_norm_rules

        module = registry.get("qwen2")(_TINY_CONFIG)
        pkg = build_from_module(module, _TINY_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        rewrite(m, pattern_rewrite_rules=skip_norm_rules())

        counts = count_ops(m)
        assert counts.get("SkipSimplifiedLayerNormalization", 0) > 0
        assert counts.get("FusedMatMul", 0) > 0
        assert counts.get("MatMul", 0) == 0

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()
