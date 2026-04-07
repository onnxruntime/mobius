# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from onnx_ir.passes.common import InlinePass
from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius import build
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

    def test_llm_emits_fused_matmul_directly(self):
        """Linear layers emit FusedMatMul directly — 197 for Qwen3-0.6B.

        Linear.forward() now emits com.microsoft::FusedMatMul(x, weight, transB=1)
        rather than Transpose(weight) + MatMul(x, weight_t).  The 197 FusedMatMul
        nodes correspond to all Q/K/V/O projections, gate and up projections,
        and the lm_head Linear.
        """
        pkg = build("Qwen/Qwen3-0.6B", load_weights=False)
        model = pkg["model"]
        counts = count_ops(model)
        # Every Linear is now a FusedMatMul node; no raw Transpose+MatMul from Linear
        assert counts["FusedMatMul"] == 197
        assert counts.get("MatMul", 0) == 0

    def test_rule_converts_external_transpose_matmul(self):
        """fused_matmul_rules() fuses Transpose+MatMul for external (non-mobius) models.

        External models may have Transpose(weight,[1,0]) + MatMul(x, weight_t) patterns.
        We simulate this by inlining FusedMatMul back to its Transpose+MatMul fallback
        body, then verifying the rule fuses them back.
        """
        model = registry.get("qwen2")(_TINY_CONFIG)
        pkg = build_from_module(model, _TINY_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        # Inline FusedMatMul → Transpose+MatMul to simulate an external model
        inline_pass = InlinePass(criteria=lambda fn: fn.name == "FusedMatMul")
        inline_pass(m)
        # Remove the function body so rewrite() won't process it and self-loop
        fn_key = ("com.microsoft", "FusedMatMul", "")
        m.functions.pop(fn_key, None)

        counts_before = count_ops(m)
        assert counts_before["Transpose"] > 0
        assert counts_before["MatMul"] > 0
        assert counts_before.get("FusedMatMul", 0) == 0

        rewrite(m, pattern_rewrite_rules=fused_matmul_rules())

        counts_after = count_ops(m)
        assert counts_after["FusedMatMul"] > 0
        assert counts_after.get("MatMul", 0) == 0

    def test_fused_matmul_model_runs_with_ort(self):
        """FusedMatMul model (direct emission from Linear) runs correctly with ORT."""
        model = registry.get("qwen2")(_TINY_CONFIG)
        pkg = build_from_module(model, _TINY_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        counts = count_ops(m)
        assert counts["FusedMatMul"] > 0
        assert counts.get("MatMul", 0) == 0

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()

    def test_combined_with_skip_norm(self):
        """SkipNorm + FusedMatMul model runs correctly with ORT."""
        from mobius.rewrite_rules import skip_norm_rules

        model = registry.get("qwen2")(_TINY_CONFIG)
        pkg = build_from_module(model, _TINY_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        rewrite(m, pattern_rewrite_rules=skip_norm_rules())

        counts = count_ops(m)
        assert counts["SkipSimplifiedLayerNormalization"] > 0
        assert counts["FusedMatMul"] > 0
        assert counts.get("MatMul", 0) == 0

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()
