# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import torch

from mobius._configs import KimiLinearConfig
from mobius.models.kimi_linear import KimiLinearCausalLMModel, _KimiMoEGate


def _config() -> KimiLinearConfig:
    return KimiLinearConfig(
        model_type="kimi_linear",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=12,
        hidden_act="silu",
        layer_types=["kimi_linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=6,
        kv_lora_rank=8,
        num_local_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        n_shared_experts=1,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        disable_qmoe=True,
    )


def test_preprocess_splits_fused_mla_kv_b_in_head_major_order() -> None:
    model = KimiLinearCausalLMModel(_config())
    fused = torch.arange(2 * (8 + 6) * 8, dtype=torch.float32).reshape(28, 8)

    result = model.preprocess_weights(
        {"model.layers.1.self_attn.kv_b_proj.weight": fused}
    )

    expected = fused.reshape(2, 14, 8)
    torch.testing.assert_close(
        result["model.layers.1.self_attn.k_b_proj.weight"],
        expected[:, :8].reshape(16, 8),
    )
    torch.testing.assert_close(
        result["model.layers.1.self_attn.v_b_proj.weight"],
        expected[:, 8:].reshape(12, 8),
    )


def test_preprocess_aligns_routed_expert_names() -> None:
    model = KimiLinearCausalLMModel(_config())
    weight = torch.ones(8, 16)

    result = model.preprocess_weights(
        {"model.layers.1.block_sparse_moe.experts.0.w1.weight": weight}
    )

    assert (
        "model.layers.1.block_sparse_moe.moe.experts.0.gate_proj.weight"
        in result
    )


def test_top_one_router_keeps_the_unbiased_sigmoid_weight() -> None:
    gate = _KimiMoEGate(_config())
    assert gate.top_k == 1
    assert gate.norm_topk_prob is False
