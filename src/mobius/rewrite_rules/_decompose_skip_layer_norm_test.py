# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for decompose_skip_layer_norm_rules().

Verifies that SkipLayerNormalization and SkipSimplifiedLayerNormalization
are correctly decomposed into their constituent Add + LayerNorm / RMSNorm
standard ONNX ops (for TRT-RTX lowering).
"""

from __future__ import annotations

from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.rewrite_rules import skip_layer_norm_rules, skip_norm_rules
from mobius.rewrite_rules._decompose_skip_layer_norm import (
    decompose_skip_layer_norm_rules,
)
from mobius.rewrite_rules._testing_utils import (
    count_ops,
    fill_random_weights,
    make_prefill_feeds,
)


class TestDecomposeSkipLayerNormRules:
    def test_rules_returns_rule_set(self):
        rules = decompose_skip_layer_norm_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_decomposes_skip_layer_norm(self):
        """Fuse Add+LN → SkipLayerNorm, then decompose back to Add+LN."""
        # GPT-2 uses LayerNorm. Fuse first, then decompose.
        pkg = build_from_module(
            registry.get("gpt2")(
                ArchitectureConfig(
                    hidden_size=64,
                    intermediate_size=128,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=16,
                    num_hidden_layers=2,
                    vocab_size=256,
                    max_position_embeddings=128,
                    hidden_act="gelu_new",
                    rms_norm_eps=1e-6,
                    rope_type="default",
                    rope_theta=10000.0,
                    pad_token_id=0,
                    tie_word_embeddings=True,
                )
            ),
            ArchitectureConfig(
                hidden_size=64,
                intermediate_size=128,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                num_hidden_layers=2,
                vocab_size=256,
                max_position_embeddings=128,
                hidden_act="gelu_new",
                rms_norm_eps=1e-6,
                rope_type="default",
                rope_theta=10000.0,
                pad_token_id=0,
                tie_word_embeddings=True,
            ),
        )
        model = pkg["model"]

        # Step 1: Fuse → SkipLayerNormalization
        rewrite(model, pattern_rewrite_rules=skip_layer_norm_rules())
        counts_fused = count_ops(model)
        assert counts_fused.get("SkipLayerNormalization", 0) > 0

        num_fused = counts_fused["SkipLayerNormalization"]

        # Step 2: Decompose → back to Add + LayerNormalization
        rewrite(model, pattern_rewrite_rules=decompose_skip_layer_norm_rules())
        counts_decomposed = count_ops(model)
        assert counts_decomposed.get("SkipLayerNormalization", 0) == 0
        # Each decomposed SkipLayerNorm produces one Add + one LayerNorm
        assert counts_decomposed.get("Add", 0) >= num_fused

    def test_decomposes_skip_simplified_layer_norm(self):
        """Fuse Add+RMSNorm → SkipSimplifiedLN, then decompose back."""
        # Qwen3 uses RMSNorm. Fuse first, then decompose.
        pkg = build_from_module(
            registry.get("qwen3")(
                ArchitectureConfig(
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
            ),
            ArchitectureConfig(
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
            ),
        )
        model = pkg["model"]

        # Step 1: Fuse → SkipSimplifiedLayerNormalization
        rewrite(model, pattern_rewrite_rules=skip_norm_rules())
        counts_fused = count_ops(model)
        assert counts_fused.get("SkipSimplifiedLayerNormalization", 0) > 0

        num_fused = counts_fused["SkipSimplifiedLayerNormalization"]

        # Step 2: Decompose → Add + RMSNormalization
        rewrite(model, pattern_rewrite_rules=decompose_skip_layer_norm_rules())
        counts_decomposed = count_ops(model)
        assert counts_decomposed.get("SkipSimplifiedLayerNormalization", 0) == 0
        assert counts_decomposed.get("Add", 0) >= num_fused
        assert counts_decomposed.get("RMSNormalization", 0) >= num_fused

    def test_roundtrip_runs_with_ort(self):
        """Fuse → decompose roundtrip produces a valid, runnable model."""
        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="gelu_new",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
        model = registry.get("gpt2")(config)
        pkg = build_from_module(model, config)
        m = pkg["model"]
        fill_random_weights(m)

        # Fuse then decompose
        rewrite(m, pattern_rewrite_rules=skip_layer_norm_rules())
        assert count_ops(m).get("SkipLayerNormalization", 0) > 0
        rewrite(m, pattern_rewrite_rules=decompose_skip_layer_norm_rules())
        assert count_ops(m).get("SkipLayerNormalization", 0) == 0

        # Must still run with ORT
        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()

    def test_no_op_when_no_fused_ops(self):
        """Decompose rules are no-ops on models without fused skip norms."""
        pkg = build_from_module(
            registry.get("llama")(
                ArchitectureConfig(
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
            ),
            ArchitectureConfig(
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
            ),
        )
        model = pkg["model"]

        rewrite(model, pattern_rewrite_rules=decompose_skip_layer_norm_rules())

        counts_after = count_ops(model)
        # No SkipLayerNorm or SkipSimplifiedLayerNorm to decompose
        assert counts_after.get("SkipLayerNormalization", 0) == 0
        assert counts_after.get("SkipSimplifiedLayerNormalization", 0) == 0
