# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius import build
from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.rewrite_rules import skip_norm_rules
from mobius.rewrite_rules._testing_utils import (
    count_ops,
    fill_random_weights,
    make_prefill_feeds,
)


class TestSkipNormRules:
    def test_rules_returns_rule_set(self):
        rules = skip_norm_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_fuses_add_rmsnorm(self):
        """The default EP optimization pipeline fuses all 56 Add+RMSNorm pairs in Qwen3-0.6B.

        Previously only 55 were fused (the last layer's single-consumer Add was skipped).
        After the off-by-1 fix, all 56 are fused including the last layer's Add → final norm.
        The remaining 57 RMSNorms are QK norms (cannot be fused — no preceding Add).
        """
        pkg = build("Qwen/Qwen3-0.6B", load_weights=False)
        model = pkg["model"]
        counts_before = count_ops(model)
        # All 56 Add+RMSNorm pairs already fused by the build pipeline
        assert counts_before["RMSNormalization"] == 57
        assert counts_before.get("Add", 0) == 0

        rewrite(model, pattern_rewrite_rules=skip_norm_rules())

        counts_after = count_ops(model)
        # No-op: all fusible pairs already handled by the build pipeline
        assert counts_after["SkipSimplifiedLayerNormalization"] == 56
        assert counts_after.get("Add", 0) == 0
        assert counts_after["RMSNormalization"] == 57

    def test_fuses_last_layer_single_consumer_add(self):
        """The last decoder layer's Add → final norm is fused even though Add has 1 consumer.

        This was an off-by-1 bug: the >= 2 consumer guard wrongly prevented fusion of the
        last layer's residual Add (which feeds only the final model norm).
        """
        pkg = build("Qwen/Qwen3-0.6B", load_weights=False)
        model = pkg["model"]
        counts = count_ops(model)
        # All Add+RMSNorm pairs fused — including the last layer's single-consumer Add
        assert counts.get("Add", 0) == 0
        assert counts["SkipSimplifiedLayerNormalization"] == 56

    def test_rewritten_model_runs_with_ort(self):
        """SkipNorm-rewritten model can be serialized and run with ORT."""
        config = ArchitectureConfig(
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
        model = registry.get("qwen2")(config)
        pkg = build_from_module(model, config)
        m = pkg["model"]
        fill_random_weights(m)

        rewrite(m, pattern_rewrite_rules=skip_norm_rules())
        assert count_ops(m)["SkipSimplifiedLayerNormalization"] == 4

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()
