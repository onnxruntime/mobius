# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import pytest
from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius import build
from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig, Gemma2Config, QuantizationConfig
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

# Tiny DeepSeek MLA config for structural K/V head-dimension coverage.
_DEEPSEEK_MLA_CONFIG = ArchitectureConfig(
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=16,
    num_hidden_layers=1,
    vocab_size=64,
    max_position_embeddings=32,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    rope_type="default",
    rope_theta=10000.0,
    dtype=ir.DataType.FLOAT16,
    q_lora_rank=16,
    kv_lora_rank=16,
    qk_nope_head_dim=8,
    qk_rope_head_dim=8,
    v_head_dim=8,
)

# Synthetic DeepSeek-V2-Lite-shaped config for the int4 QMoE regression guard.
# Mirrors the real architecture (MLA attention + first_k_dense_replace, softmax
# routing, shared experts) at tiny dimensions so the test runs fully offline with
# no HuggingFace download. hidden_size and moe_intermediate_size are multiples of
# the quantization group_size (128) because QMoE requires divisibility, while the
# small MLA/dense dimensions are fine for MatMulNBits (which ceil-pads the K axis).
# Layer 0 is a dense MLP; layers 1..2 are routed MoE layers -> 2 QMoE, 3 Attention.
_DEEPSEEK_V2_LITE_INT4_CONFIG = ArchitectureConfig(
    hidden_size=128,
    intermediate_size=256,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=16,
    num_hidden_layers=3,
    vocab_size=128,
    max_position_embeddings=32,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    rope_type="default",
    rope_theta=10000.0,
    dtype=ir.DataType.FLOAT16,
    q_lora_rank=16,
    kv_lora_rank=16,
    qk_nope_head_dim=8,
    qk_rope_head_dim=8,
    v_head_dim=8,
    num_local_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=128,
    n_shared_experts=1,
    first_k_dense_replace=1,
    scoring_func="softmax",
    norm_topk_prob=False,
    routed_scaling_factor=1.0,
    quantization=QuantizationConfig(
        bits=4,
        group_size=128,
        quant_method="gptq",
        sym=True,
    ),
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
    # ArchitectureConfig defaults ``rope_type`` to None (NoPE) since
    # April 2026; enable RoPE explicitly for this Gemma2 test config.
    rope_type="default",
)


def _build_tiny_mla_graph(v_head_dim: int):
    """Build a CPU-only MLA graph with configurable K/V head dimensions."""
    config = ArchitectureConfig(
        model_type="deepseek_v2",
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        num_hidden_layers=1,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10000.0,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=v_head_dim,
    )
    return build_from_module(registry.get("deepseek_v2")(config), config)["model"]


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

    def test_fuses_large_head_dim(self):
        """GQA fusion applies uniformly, including head_dim > 256.

        There is no head-dim cap: every decoder ``Attention`` node is fused to
        ``GroupQueryAttention`` regardless of ``head_dim``.  This covers
        Gemma4-style global-attention layers (head_dim=512), which the CUDA GQA
        kernel handles via its FP32-QK-accumulation unfused fallback.
        """
        cfg = ArchitectureConfig(
            hidden_size=1024,
            intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=512,
            num_hidden_layers=2,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
        )
        model = registry.get("llama")(cfg)
        pkg = build_from_module(model, cfg, execution_provider="default")
        m = pkg["model"]
        assert count_ops(m).get("Attention", 0) == 2

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())

        counts_after = count_ops(m)
        assert counts_after.get("Attention", 0) == 0
        assert counts_after.get("GroupQueryAttention", 0) == 2

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
        vision = pkg["vision_encoder"]
        vision_attn_before = count_ops(vision).get("Attention", 0)
        assert vision_attn_before == 24

        rewrite(
            vision,
            pattern_rewrite_rules=group_query_attention_rules(),
        )
        vision_counts = count_ops(vision)
        assert vision_counts.get("Attention", 0) == vision_attn_before
        assert vision_counts.get("GroupQueryAttention", 0) == 0

    @pytest.mark.parametrize(
        ("v_head_dim", "expected_attention", "expected_gqa"),
        [(8, 1, 0), (16, 0, 1)],
    )
    def test_mla_gqa_fusion_requires_equal_kv_head_dimensions(
        self, v_head_dim, expected_attention, expected_gqa
    ):
        """GQA fusion declines unequal MLA K/V dimensions and accepts equal ones."""
        config = dataclasses.replace(_DEEPSEEK_MLA_CONFIG, v_head_dim=v_head_dim)
        model = build_from_module(registry.get("deepseek_v3")(config), config)["model"]
        attention = next(node for node in model.graph if node.op_type == "Attention")
        assert attention.inputs[4] is not None
        assert attention.inputs[4].shape[-1] == 16
        assert attention.inputs[5] is not None
        assert attention.inputs[5].shape[-1] == v_head_dim

        rewrite(model, pattern_rewrite_rules=group_query_attention_rules())
        counts = count_ops(model)
        assert counts.get("Attention", 0) == expected_attention
        assert counts.get("GroupQueryAttention", 0) == expected_gqa

    def test_deepseek_v2_lite_int4_uses_one_qmoe_per_moe_layer(self):
        """The int4 graph keeps shared experts dense and routed experts fused.

        Self-contained regression guard: builds a tiny DeepSeek-V2-Lite-shaped
        model from an inline config (no HuggingFace download) so it runs in the
        offline main CI. Layer 0 is a dense MLP and layers 1..2 are routed MoE
        layers, so exactly one com.microsoft::QMoE is emitted per routed MoE
        layer while the shared experts remain dense MatMulNBits.
        """
        config = _DEEPSEEK_V2_LITE_INT4_CONFIG
        num_layers = config.num_hidden_layers
        num_moe_layers = num_layers - config.first_k_dense_replace
        with pytest.warns(UserWarning, match="GQA fusion expected"):
            model = build_from_module(
                registry.get("deepseek_v3")(config),
                config,
                execution_provider="cuda",
            )["model"]
        counts = count_ops(model)

        assert counts.get("QMoE", 0) == num_moe_layers
        assert counts.get("GroupQueryAttention", 0) == 0
        assert counts.get("Attention", 0) == num_layers
        for node in (node for node in model.graph if node.op_type == "QMoE"):
            assert node.inputs[1] is not None
            assert node.inputs[1].dtype == ir.DataType.FLOAT
            assert node.inputs[3] is not None
            assert node.inputs[3].dtype == ir.DataType.FLOAT
            assert node.inputs[6] is not None
            assert node.inputs[6].dtype == ir.DataType.FLOAT
            assert node.inputs[14] is not None
            assert node.inputs[14].dtype == ir.DataType.FLOAT

        input_names = [
            value.name
            for node in model.graph
            for value in node.inputs
            if value is not None and value.name is not None
        ]
        assert not any(".moe.experts." in name for name in input_names)
        for layer_idx in range(config.first_k_dense_replace, num_layers):
            prefix = f"model.layers.{layer_idx}.mlp.shared_experts."
            shared_matmuls = [
                node
                for node in model.graph
                if node.op_type == "MatMulNBits"
                and any(
                    value is not None and value.name is not None and prefix in value.name
                    for value in node.inputs
                )
            ]
            assert len(shared_matmuls) == 3

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

    @pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
    def test_packed_weight_intermediates_declare_weight_dtype(self, dtype):
        """Concat/Transpose intermediates carry the projection weight dtype.

        The replacement builder leaves new values untyped. Without an explicit
        stamp, folding ``Transpose(Concat(W_q, W_k, W_v))`` into an initializer
        has no declared type to inherit and can widen fp16 weights to fp32.
        """
        config = dataclasses.replace(_LLAMA_CONFIG, dtype=dtype)
        m = build_from_module(registry.get("llama")(config), config)["model"]

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())
        rewrite(m, pattern_rewrite_rules=pack_qkv_for_gqa_rules())

        gqa_nodes = [n for n in m.graph if n.op_type == "GroupQueryAttention"]
        assert len(gqa_nodes) == config.num_hidden_layers

        for gqa in gqa_nodes:
            transpose = gqa.inputs[0].producer().inputs[1].producer()
            assert transpose.op_type == "Transpose"
            concat = transpose.inputs[0].producer()
            assert concat.op_type == "Concat"
            assert concat.outputs[0].dtype == dtype
            assert transpose.outputs[0].dtype == dtype

    @pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
    def test_packed_bias_intermediate_declares_bias_dtype(self, dtype):
        """The packed-bias Concat intermediate carries the bias dtype."""
        config = dataclasses.replace(_QWEN2_BIAS_CONFIG, dtype=dtype)
        m = build_from_module(registry.get("qwen2")(config), config)["model"]

        rewrite(m, pattern_rewrite_rules=group_query_attention_rules())
        rewrite(m, pattern_rewrite_rules=pack_qkv_for_gqa_rules())

        gqa_nodes = [n for n in m.graph if n.op_type == "GroupQueryAttention"]
        assert len(gqa_nodes) == config.num_hidden_layers

        for gqa in gqa_nodes:
            add = gqa.inputs[0].producer()
            assert add.op_type == "Add"
            bias_concat = add.inputs[1].producer()
            assert bias_concat.op_type == "Concat"
            assert bias_concat.outputs[0].dtype == dtype

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
