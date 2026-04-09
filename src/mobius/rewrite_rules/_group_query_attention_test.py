# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius import build
from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig, Gemma2Config
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.rewrite_rules import group_query_attention_rules, pack_qkv_for_gqa_rules
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

# Tiny Qwen2-style config: standard RoPE + QKV bias (no QK norm)
# Mirrors Qwen2.5: attn_qkv_bias=True, no QK norm, half-split RoPE.
_QWEN2_BIAS_CONFIG = ArchitectureConfig(
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

    def test_fallback_attention_to_gqa_no_rope(self):
        """AttentionToGQA fallback fires when applied in isolation (do_rotary=0).

        When applied alone (not after RotaryAttentionToGQA), ``AttentionToGQA``
        should convert any decoder ``Attention`` node to GQA with do_rotary=0.
        In normal usage it only fires for models like Qwen3.5 VL whose Q/K
        don't come from standard ``RotaryEmbedding`` ops, because
        ``RotaryAttentionToGQA`` always takes priority in the combined rule set.
        """
        from mobius.rewrite_rules._group_query_attention import AttentionToGQA

        qwen35_model_type = "qwen3_5_text"
        if qwen35_model_type not in registry:
            pytest.skip(f"{qwen35_model_type!r} not in registry")

        # Minimal Qwen3.5 config: 4 layers, 1 full_attention layer at index 3.
        # GatedDeltaNet needs explicit linear_* fields.
        cfg = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=4,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        )
        model = registry.get(qwen35_model_type)(cfg)
        pkg = build_from_module(model, cfg, execution_provider="default")
        m = pkg["model"]
        fill_random_weights(m)

        assert count_ops(m).get("Attention", 0) == 1, (
            "Expected 1 Attention (full_attention layer)"
        )

        # Apply ONLY the fallback rule (not combined set).  In practice, the
        # fallback fires for models whose Q/K come from Where-based 3D mRoPE
        # (e.g. Qwen3.5 VL text decoder) instead of RotaryEmbedding.
        fallback_only = RewriteRuleSet([AttentionToGQA().rule()])
        rewrite(m, pattern_rewrite_rules=fallback_only)

        counts = count_ops(m)
        assert counts.get("Attention", 0) == 0
        assert counts.get("GroupQueryAttention", 0) == 1
        gqa_node = next(n for n in m.graph if n.op_type == "GroupQueryAttention")
        assert gqa_node.attributes.get_int("do_rotary", -1) == 0, (
            "AttentionToGQA fallback must set do_rotary=0 (RoPE applied externally)"
        )

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
        """Packing Q/K/V into one MatMul removes 2 MatMuls per layer.

        The CPU EP build runs GQA fusion + PackQKV in stage 2, so the MatMul
        count reduction is visible by comparing a default EP build (no packing)
        with a CPU EP build.  The structural fold (Transpose/Concat over
        initializers) runs inside apply_weights, after weights are loaded.
        """
        num_layers = _LLAMA_CONFIG.num_hidden_layers

        m_default = build_from_module(registry.get("llama")(_LLAMA_CONFIG), _LLAMA_CONFIG)[
            "model"
        ]
        matmul_default = count_ops(m_default)["MatMul"]

        m_cpu = build_from_module(
            registry.get("llama")(_LLAMA_CONFIG),
            _LLAMA_CONFIG,
            execution_provider="cpu",
        )["model"]
        matmul_packed = count_ops(m_cpu)["MatMul"]

        # 3 separate Q/K/V MatMuls → 1 packed MatMul per layer = -2 per layer
        assert matmul_packed == matmul_default - 2 * num_layers

    def test_packed_weight_uses_concat_node(self):
        """CPU EP build produces packed GQA with Concat+Transpose weight form.

        After the CPU EP pipeline (stage 2: GQA fusion + PackQKV), the packed
        weight is ``MatMul(hidden, Transpose(Concat(W_q, W_k, W_v)))``.
        Fold passes run later (in fold_initializers_after_weights after weights
        are loaded), so Concat and Transpose nodes are still present here.
        """
        m = build_from_module(
            registry.get("llama")(_LLAMA_CONFIG),
            _LLAMA_CONFIG,
            execution_provider="cpu",
        )["model"]

        gqa_nodes = [n for n in m.graph if n.op_type == "GroupQueryAttention"]
        assert len(gqa_nodes) == _LLAMA_CONFIG.num_hidden_layers

        for gqa in gqa_nodes:
            assert gqa.inputs[1] is None, "GQA should be in packed mode (k=None)"
            assert gqa.inputs[2] is None, "GQA should be in packed mode (v=None)"
            packed_proj = gqa.inputs[0]
            mm = packed_proj.producer()
            assert mm is not None and mm.op_type == "MatMul"

            # The weight goes through Transpose(Concat(W_q, W_k, W_v)).
            w_input = mm.inputs[1]
            assert w_input is not None
            transpose = w_input.producer()
            assert transpose is not None and transpose.op_type == "Transpose", (
                "Packed weight should go through a Transpose node before fold passes run"
            )
            concat = transpose.inputs[0].producer()
            assert concat is not None and concat.op_type == "Concat", (
                "Transpose input should be a Concat of W_q, W_k, W_v"
            )
            assert len(concat.inputs) == 3, "Concat should have exactly 3 inputs (Q, K, V)"

    def test_falls_back_to_separate_qkv_with_qk_norm(self):
        """Qwen3 (QK norm) falls back; MatMul count unchanged after packing attempt."""
        model = registry.get("qwen3")(_QWEN3_CONFIG)
        pkg = build_from_module(model, _QWEN3_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        matmul_before = count_ops(m)["MatMul"]

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())
        rewrite(m, pattern_rewrite_rules=pack_qkv_for_gqa_rules())

        # GQA should still be applied
        assert count_ops(m)["GroupQueryAttention"] == 2
        # But MatMul count should not decrease (no packing due to QK norm)
        assert count_ops(m)["MatMul"] == matmul_before

    def test_packed_model_runs_with_ort(self):
        """Packed-QKV GQA model runs correctly with ORT."""
        model = registry.get("llama")(_LLAMA_CONFIG)
        pkg = build_from_module(model, _LLAMA_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())
        rewrite(m, pattern_rewrite_rules=pack_qkv_for_gqa_rules())
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
        rewrite(m, pattern_rewrite_rules=pack_qkv_for_gqa_rules())
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

    def test_packed_gqa_count_and_runs_with_ort(self):
        """Packing reduces MatMul count and the model still runs with ORT.

        Build with CPU EP so that stage 2 applies GQA fusion + PackQKV.
        The structural fold (Transpose/Concat over initializers) runs inside
        apply_weights after weights are loaded.  The final model has fewer
        MatMuls (3 separate Q/K/V → 1 packed per layer) and still produces
        correct output.
        """
        num_layers = _LLAMA_CONFIG.num_hidden_layers

        m_default = build_from_module(registry.get("llama")(_LLAMA_CONFIG), _LLAMA_CONFIG)[
            "model"
        ]
        matmul_default = count_ops(m_default)["MatMul"]

        m_cpu = build_from_module(
            registry.get("llama")(_LLAMA_CONFIG),
            _LLAMA_CONFIG,
            execution_provider="cpu",
        )["model"]
        fill_random_weights(m_cpu)
        counts_after = count_ops(m_cpu)
        assert counts_after["GroupQueryAttention"] == num_layers
        assert counts_after["MatMul"] == matmul_default - 2 * num_layers

        session = OnnxModelSession(m_cpu)
        feeds = make_prefill_feeds(session)
        result = session.run(feeds)
        assert "logits" in result
        assert result["logits"].shape == (1, 3, 256)
        session.close()

    # ---- Packed QKV with bias tests (Qwen2.5 / Phi3/4 style) ----

    def test_packed_qkv_with_bias_reduces_matmul_count(self):
        """PackQKVWithBiasForGQA packs Q/K/V MatMuls on a biased model (Qwen2-style).

        Build with CPU EP so stage 2 fires PackQKVWithBiasForGQA.  The
        structural fold runs inside apply_weights after weights are loaded.
        Compare against a default EP build.
        """
        num_layers = _QWEN2_BIAS_CONFIG.num_hidden_layers

        m_default = build_from_module(
            registry.get("qwen2")(_QWEN2_BIAS_CONFIG), _QWEN2_BIAS_CONFIG
        )["model"]
        matmul_default = count_ops(m_default)["MatMul"]

        m_cpu = build_from_module(
            registry.get("qwen2")(_QWEN2_BIAS_CONFIG),
            _QWEN2_BIAS_CONFIG,
            execution_provider="cpu",
        )["model"]
        matmul_packed = count_ops(m_cpu)["MatMul"]

        # 3 separate Q/K/V MatMuls → 1 packed MatMul per layer = -2 per layer
        assert matmul_packed == matmul_default - 2 * num_layers

    def test_packed_qkv_with_bias_uses_concat_nodes(self):
        """Biased packing produces packed GQA with Concat+Transpose weight and bias forms.

        After the CPU EP pipeline (stage 2: PackQKVWithBiasForGQA), the packed
        weight is ``MatMul(hidden, Transpose(Concat(W_q, W_k, W_v)))`` and the
        bias is ``Concat(bias_q, bias_k, bias_v)``.  Fold passes run after
        weights are loaded, so Concat and Transpose nodes are still present here.
        """
        m = build_from_module(
            registry.get("qwen2")(_QWEN2_BIAS_CONFIG),
            _QWEN2_BIAS_CONFIG,
            execution_provider="cpu",
        )["model"]

        gqa_nodes = [n for n in m.graph if n.op_type == "GroupQueryAttention"]
        assert len(gqa_nodes) == _QWEN2_BIAS_CONFIG.num_hidden_layers

        for gqa in gqa_nodes:
            assert gqa.inputs[1] is None, "GQA should be in packed mode (k=None)"
            assert gqa.inputs[2] is None, "GQA should be in packed mode (v=None)"
            packed_proj = gqa.inputs[0]

            # Structure: Add(MatMul(hidden, Transpose(Concat(W_q,W_k,W_v))), Concat(b_q,b_k,b_v))
            add_node = packed_proj.producer()
            assert add_node is not None and add_node.op_type == "Add", (
                "Packed QKV with bias should be wrapped in Add"
            )

            # Find MatMul and the bias value among Add inputs.
            add_ins = [i for i in add_node.inputs if i is not None]
            matmul = next(
                (
                    i.producer()
                    for i in add_ins
                    if i.producer() and i.producer().op_type == "MatMul"
                ),
                None,
            )
            bias_val = next(
                (i for i in add_ins if i.producer() and i.producer().op_type == "Concat"),
                None,
            )
            assert matmul is not None, "Add should contain MatMul"
            assert bias_val is not None, "Add should contain Concat bias"

            # The bias is a Concat of individual biases.
            bias_concat = bias_val.producer()
            assert bias_concat is not None and bias_concat.op_type == "Concat"
            assert len(bias_concat.inputs) == 3, "Bias Concat should have 3 inputs"

            # The weight goes through Transpose(Concat(W_q, W_k, W_v)).
            w_input = matmul.inputs[1]
            assert w_input is not None
            transpose = w_input.producer()
            assert transpose is not None and transpose.op_type == "Transpose", (
                "Weight should go through Transpose before fold passes run"
            )
            concat = transpose.inputs[0].producer()
            assert concat is not None and concat.op_type == "Concat", (
                "Transpose input should be Concat of W_q, W_k, W_v"
            )

    def test_packed_qkv_with_bias_runs_with_ort(self):
        """Biased packed-QKV GQA model runs correctly with ORT."""
        model = registry.get("qwen2")(_QWEN2_BIAS_CONFIG)
        pkg = build_from_module(model, _QWEN2_BIAS_CONFIG)
        m = pkg["model"]
        fill_random_weights(m)

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())
        rewrite(m, pattern_rewrite_rules=pack_qkv_for_gqa_rules())
        assert count_ops(m)["GroupQueryAttention"] == _QWEN2_BIAS_CONFIG.num_hidden_layers

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
