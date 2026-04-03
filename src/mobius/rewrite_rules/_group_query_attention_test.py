# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius import build
from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig, Gemma2Config
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.rewrite_rules import group_query_attention_rules
from mobius.rewrite_rules._testing_utils import (
    count_ops,
    fill_random_weights,
    make_prefill_feeds,
)

# Tiny llama config: no QK norm, weights are packable
_LLAMA_CONFIG = ArchitectureConfig(
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

# Tiny qwen3 config: has QK norm, weights NOT packable
_QWEN3_CONFIG = ArchitectureConfig(
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
    attn_qk_norm=True,
)

# Tiny GLM4 config: interleaved RoPE (rope_interleave=True)
_GLM4_CONFIG = ArchitectureConfig(
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
    attn_qkv_bias=True,
    rope_interleave=True,
)

# Tiny Gemma2 config: attn_logit_softcapping=50.0 (must survive GQA fusion)
_GEMMA2_CONFIG = Gemma2Config(
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    num_hidden_layers=2,
    vocab_size=256,
    max_position_embeddings=128,
    hidden_act="gelu_pytorch_tanh",
    rms_norm_eps=1e-6,
    query_pre_attn_scalar=16,
    attn_logit_softcapping=50.0,
    final_logit_softcapping=30.0,
)


class TestGroupQueryAttentionRules:
    def test_rules_returns_rule_set(self):
        rules = group_query_attention_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_replaces_attention_with_gqa(self):
        pkg = build("Qwen/Qwen3-0.6B", load_weights=False)
        model = pkg["model"]
        counts_before = count_ops(model)
        assert counts_before["Attention"] == 28

        rewrite(model, pattern_rewrite_rules=group_query_attention_rules())

        counts_after = count_ops(model)
        assert counts_after.get("Attention", 0) == 0
        assert counts_after["GroupQueryAttention"] == 28

    def test_absorbs_rotary_embedding(self):
        """RotaryEmbedding ops are absorbed into GQA with do_rotary=1."""
        pkg = build("Qwen/Qwen3-0.6B", load_weights=False)
        model = pkg["model"]
        counts_before = count_ops(model)
        assert counts_before["RotaryEmbedding"] == 56

        rewrite(model, pattern_rewrite_rules=group_query_attention_rules())

        counts_after = count_ops(model)
        assert counts_after.get("RotaryEmbedding", 0) == 0
        assert counts_after["GroupQueryAttention"] == 28

    def test_absorbs_rotary_without_qk_norm(self):
        """Models without QK norm also get rotary absorbed."""
        pkg = build("HuggingFaceTB/SmolLM2-135M-Instruct", load_weights=False)
        model = pkg["model"]
        counts_before = count_ops(model)
        assert counts_before["RotaryEmbedding"] == 60

        rewrite(model, pattern_rewrite_rules=group_query_attention_rules())

        counts_after = count_ops(model)
        assert counts_after.get("RotaryEmbedding", 0) == 0
        assert counts_after["GroupQueryAttention"] == 30

    def test_preserves_non_matching_model(self):
        """Vision encoder attention (no KV cache) is not replaced."""
        pkg = build(
            "Qwen/Qwen3-VL-2B-Instruct",
            load_weights=False,
        )
        # Vision model: no KV cache, Attention should remain untouched
        vision = pkg["vision"]
        vision_attn_before = count_ops(vision).get("Attention", 0)
        assert vision_attn_before == 24

        rewrite(
            vision,
            pattern_rewrite_rules=group_query_attention_rules(),
        )
        vision_counts = count_ops(vision)
        assert vision_counts.get("Attention", 0) == vision_attn_before
        assert vision_counts.get("GroupQueryAttention", 0) == 0

    def test_rewritten_model_runs_with_ort(self):
        """GQA-rewritten model can be serialized and run with ORT."""
        model = registry.get("qwen3")(_QWEN3_CONFIG)
        pkg = build_from_module(model, _QWEN3_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())
        assert count_ops(m)["GroupQueryAttention"] == 2

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()

    def test_combined_gqa_and_skip_norm_runs_with_ort(self):
        """Applying GQA + SkipNorm together produces a valid ORT model."""
        from mobius.rewrite_rules import skip_norm_rules

        model = registry.get("qwen3")(_QWEN3_CONFIG)
        pkg = build_from_module(model, _QWEN3_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())
        rewrite(m, pattern_rewrite_rules=skip_norm_rules())

        counts = count_ops(m)
        assert counts["GroupQueryAttention"] == 2
        assert counts["SkipSimplifiedLayerNormalization"] > 0
        assert counts.get("Attention", 0) == 0

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()

    # ---- Packed QKV tests ----

    def test_packed_qkv_reduces_matmul_count(self):
        """Packing Q/K/V into one MatMul removes 2 MatMuls per layer."""
        model = registry.get("llama")(_LLAMA_CONFIG)
        pkg = build_from_module(model, _LLAMA_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        matmul_before = count_ops(m)["MatMul"]

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())

        matmul_after = count_ops(m)["MatMul"]
        num_layers = _LLAMA_CONFIG.num_hidden_layers
        # 3 separate Q/K/V MatMuls -> 1 packed MatMul per layer = -2 per layer
        assert matmul_after == matmul_before - 2 * num_layers

    def test_packed_weight_shape_is_correct(self):
        """Packed W_qkv has shape (q_dim + 2*kv_dim, hidden_size)."""
        model = registry.get("llama")(_LLAMA_CONFIG)
        pkg = build_from_module(model, _LLAMA_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())

        q_dim = _LLAMA_CONFIG.num_attention_heads * _LLAMA_CONFIG.head_dim
        kv_dim = _LLAMA_CONFIG.num_key_value_heads * _LLAMA_CONFIG.head_dim
        hidden = _LLAMA_CONFIG.hidden_size
        expected_shape = (q_dim + 2 * kv_dim, hidden)

        # Packed weights are stored as graph initializers
        found = False
        for init in m.graph.initializers.values():
            if init.const_value is None:
                continue
            if tuple(init.const_value.shape) == expected_shape:
                arr = init.const_value.numpy()
                assert arr.dtype == np.float32
                assert not np.any(np.isnan(arr))
                found = True
                break
        assert found, f"No packed weight initializer with shape {expected_shape}"

    def test_falls_back_to_separate_qkv_with_qk_norm(self):
        """Qwen3 (QK norm) falls back; MatMul count unchanged."""
        model = registry.get("qwen3")(_QWEN3_CONFIG)
        pkg = build_from_module(model, _QWEN3_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        matmul_before = count_ops(m)["MatMul"]

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())

        # GQA should still be applied
        assert count_ops(m)["GroupQueryAttention"] == 2
        # But MatMul count should not decrease (no packing)
        assert count_ops(m)["MatMul"] == matmul_before

    def test_packed_model_runs_with_ort(self):
        """Packed-QKV GQA model runs correctly with ORT."""
        model = registry.get("llama")(_LLAMA_CONFIG)
        pkg = build_from_module(model, _LLAMA_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())
        assert count_ops(m)["GroupQueryAttention"] == 2

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()

    def test_combined_packed_gqa_and_skip_norm_runs_with_ort(self):
        """Packed GQA + SkipNorm produces a valid ORT model."""
        from mobius.rewrite_rules import skip_norm_rules

        model = registry.get("llama")(_LLAMA_CONFIG)
        pkg = build_from_module(model, _LLAMA_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())
        rewrite(m, pattern_rewrite_rules=skip_norm_rules())

        counts = count_ops(m)
        assert counts["GroupQueryAttention"] == 2
        assert counts["SkipSimplifiedLayerNormalization"] > 0
        assert counts.get("Attention", 0) == 0

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()

    def test_packed_gqa_then_fused_matmul_runs_with_ort(self):
        """Packing runs before fused_matmul (mirrors --optimize=all order)."""
        from mobius.rewrite_rules import fused_matmul_rules

        model = registry.get("llama")(_LLAMA_CONFIG)
        pkg = build_from_module(model, _LLAMA_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        matmul_before = count_ops(m)["MatMul"]

        # GQA (with packing) runs first — sees plain MatMul nodes
        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())
        counts_after_gqa = count_ops(m)
        assert counts_after_gqa["GroupQueryAttention"] == 2
        num_layers = _LLAMA_CONFIG.num_hidden_layers
        assert counts_after_gqa["MatMul"] == matmul_before - 2 * num_layers

        # Then fused_matmul converts remaining Transpose+MatMul
        rewrite(m, pattern_rewrite_rules=fused_matmul_rules())
        counts_final = count_ops(m)
        assert counts_final.get("FusedMatMul", 0) > 0

        session = OnnxModelSession(m)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()

    def test_rotary_interleaved_attribute_propagated(self):
        """GQA fusion reads interleaved from RotaryEmbedding, not hardcoded 0.

        GLM4/ChatGLM use interleaved=1. Hardcoding 0 would silently produce
        incorrect RoPE inside the fused GQA kernel.
        """
        model = registry.get("glm4")(_GLM4_CONFIG)
        pkg = build_from_module(model, _GLM4_CONFIG)
        m = pkg["model"]

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())

        gqa_nodes = [n for n in m.graph if n.op_type == "GroupQueryAttention"]
        assert len(gqa_nodes) > 0, "Expected at least one GQA node after fusion"

        for node in gqa_nodes:
            val = node.attributes.get("rotary_interleaved")
            assert val is not None, "rotary_interleaved attribute missing on GQA node"
            assert val.value == 1, (
                f"Expected rotary_interleaved=1 for GLM4 (interleaved RoPE), got {val.value}"
            )

    def test_rotary_interleaved_default_is_zero(self):
        """Non-interleaved models (Llama, Qwen) get rotary_interleaved=0."""
        model = registry.get("qwen3")(_QWEN3_CONFIG)
        pkg = build_from_module(model, _QWEN3_CONFIG)
        m = pkg["model"]

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())

        gqa_nodes = [n for n in m.graph if n.op_type == "GroupQueryAttention"]
        assert len(gqa_nodes) > 0

        for node in gqa_nodes:
            val = node.attributes.get("rotary_interleaved")
            assert val is not None
            assert val.value == 0, (
                f"Expected rotary_interleaved=0 for Qwen3 (half-split RoPE), got {val.value}"
            )

    def test_softcap_preserved_on_gqa_fusion(self):
        """Gemma2 softcap attribute on Attention must be forwarded to GQA.

        Without this, the GQA kernel uses its default softcap=0.0 (disabled),
        silently producing incorrect attention weights for Gemma2 models.
        """
        model = registry.get("gemma2")(_GEMMA2_CONFIG)
        pkg = build_from_module(model, _GEMMA2_CONFIG)
        m = pkg["model"]

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())

        gqa_nodes = [n for n in m.graph if n.op_type == "GroupQueryAttention"]
        assert len(gqa_nodes) > 0, "Expected GQA nodes after fusion"

        for node in gqa_nodes:
            sc = node.attributes.get("softcap")
            assert sc is not None, (
                "softcap attribute missing from GQA node — "
                "Gemma2 attention logit capping will be silently disabled"
            )
            assert sc.value == pytest.approx(50.0), (
                f"Expected softcap=50.0 (from _GEMMA2_CONFIG), got {sc.value}"
            )

    def test_softcap_absent_for_non_softcap_models(self):
        """Llama/Qwen (no softcap) must not get a spurious softcap=0 on GQA."""
        model = registry.get("llama")(_LLAMA_CONFIG)
        pkg = build_from_module(model, _LLAMA_CONFIG)
        m = pkg["model"]

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())

        gqa_nodes = [n for n in m.graph if n.op_type == "GroupQueryAttention"]
        assert len(gqa_nodes) > 0

        for node in gqa_nodes:
            sc = node.attributes.get("softcap")
            assert sc is None, f"Unexpected softcap attribute on GQA node for Llama: {sc}"
