# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import types

import pytest
import torch

from mobius._configs import KimiK3Config
from mobius.models.kimi_k3 import KimiK3CausalLMModel
from mobius.tasks import KimiK3CausalLMTask


def _text_config(**overrides):
    values = {
        "model_type": "kimi_linear",
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "hidden_act": "situ",
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
        "linear_attn_config": {
            "kda_layers": [1, 3],
            "full_attn_layers": [2],
            "num_heads": 2,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
        },
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "q_lora_rank": 12,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "attn_res_block_size": 2,
        "first_k_dense_replace": 1,
        "moe_intermediate_size": 12,
        "moe_layer_freq": 1,
        "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid",
        "num_experts": 2,
        "num_experts_per_token": 1,
        "num_expert_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "num_shared_experts": 2,
        "num_nextn_predict_layers": 0,
        "routed_expert_hidden_size": 16,
        "latent_moe_use_norm": True,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _config() -> KimiK3Config:
    parent = types.SimpleNamespace(
        model_type="kimi_k3",
        text_config=_text_config(),
        tie_word_embeddings=False,
        pad_token_id=0,
    )
    return KimiK3Config.from_transformers(parent)


def test_config_extracts_nested_k3_semantics() -> None:
    config = _config()

    assert config.model_type == "kimi_k3"
    assert config.layer_types == [
        "kimi_k3_attention",
        "full_attention",
        "kimi_k3_attention",
    ]
    assert config.linear_gate_lower_bound == pytest.approx(5.0)
    assert config.linear_use_full_rank_gate
    assert config.mla_use_output_gate
    assert config.routed_expert_hidden_size == 16
    assert config.activation_situ_beta == pytest.approx(4.0)
    assert not config.tie_word_embeddings


def test_config_rejects_non_k3_gate_profile() -> None:
    text = _text_config()
    text.linear_attn_config["use_full_rank_gate"] = False
    parent = types.SimpleNamespace(model_type="kimi_k3", text_config=text)

    with pytest.raises(ValueError, match="full-rank"):
        KimiK3Config.from_transformers(parent)


def test_config_rejects_selective_compressed_tensors_checkpoint() -> None:
    text = _text_config(quantization_config={"quant_method": "compressed-tensors"})
    parent = types.SimpleNamespace(model_type="kimi_k3", text_config=text)

    with pytest.raises(NotImplementedError, match="selective MXFP4"):
        KimiK3Config.from_transformers(parent)


def test_model_has_attention_residual_latent_moe_and_untied_head() -> None:
    model = KimiK3CausalLMModel(_config())

    assert model.model.layers[0].mlp is not None
    moe = model.model.layers[1].block_sparse_moe
    assert moe is not None
    assert len(moe.moe.experts) == 2
    assert moe.routed_down_proj is not None
    assert model.model.layers[0].attn_res_score is not None
    assert model.lm_head.weight is not model.model.embed_tokens.weight


def test_preprocess_strips_multimodal_prefix_and_aligns_experts() -> None:
    model = KimiK3CausalLMModel(_config())
    weight = torch.ones(12, 16)

    result = model.preprocess_weights(
        {
            "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight": weight,
            "vision_tower.blocks.0.weight": torch.ones(1),
        }
    )

    assert "model.layers.1.block_sparse_moe.moe.experts.0.gate_proj.weight" in result
    assert not any(key.startswith("vision_tower.") for key in result)


def test_task_rejects_static_cache() -> None:
    with pytest.raises(ValueError, match="does not support static cache"):
        KimiK3CausalLMTask(static_cache=True)


def test_graph_exposes_k3_state_and_architecture_contracts() -> None:
    config = _config()
    package = KimiK3CausalLMTask().build(KimiK3CausalLMModel(config), config)
    graph = package["model"].graph
    initializer_names = set(graph.initializers)
    output_names = {output.name for output in graph.outputs}
    op_types = {node.op_type for node in graph}

    assert "model.layers.0.self_attn.g_proj.weight" in initializer_names
    assert "model.layers.1.self_attn.q_a_proj.weight" in initializer_names
    assert "model.layers.1.self_attn.k_b_proj.weight" in initializer_names
    assert "model.layers.1.self_attn.v_b_proj.weight" in initializer_names
    assert "model.layers.1.self_attn.g_proj.weight" in initializer_names
    assert "model.layers.1.block_sparse_moe.moe.gate.weight" in initializer_names
    assert (
        "model.layers.1.block_sparse_moe.moe.experts.0.gate_proj.weight" in initializer_names
    )
    assert "model.layers.1.block_sparse_moe.routed_norm.weight" in initializer_names
    assert "model.layers.1.attn_res_score.weight" in initializer_names
    assert "model.layers.0.ffn_res_score.weight" in initializer_names
    assert "model.output_res_score.weight" in initializer_names
    assert "Tanh" in op_types
    assert not any("rotary_emb" in name for name in initializer_names)
    assert "present.0.recurrent_state" in output_names
    assert "present.1.key" in output_names
    assert "expanded-semantic-cache" in package["model"].metadata_props["mobius.cache_abi"]


def test_preprocess_splits_mla_and_folds_attention_residual_weights() -> None:
    model = KimiK3CausalLMModel(_config())
    fused = torch.arange(2 * (8 + 8) * 16, dtype=torch.float32).reshape(32, 16)
    norm = torch.arange(1, 33, dtype=torch.float32)
    projection = torch.full((1, 32), 2.0)

    result = model.preprocess_weights(
        {
            "language_model.model.layers.1.self_attn.kv_b_proj.weight": fused,
            "language_model.model.layers.1.self_attention_res_norm.weight": norm,
            "language_model.model.layers.1.self_attention_res_proj.weight": projection,
        }
    )

    expected = fused.reshape(2, 16, 16)
    torch.testing.assert_close(
        result["model.layers.1.self_attn.k_b_proj.weight"],
        expected[:, :8].reshape(16, 16),
    )
    torch.testing.assert_close(
        result["model.layers.1.self_attn.v_b_proj.weight"],
        expected[:, 8:].reshape(16, 16),
    )
    torch.testing.assert_close(
        result["model.layers.1.attn_res_score.weight"],
        projection.float() * norm.float().unsqueeze(0),
    )


def test_preprocess_folds_attention_residual_weights_in_float32() -> None:
    model = KimiK3CausalLMModel(_config())
    norm = torch.linspace(0.9, 1.1, 32, dtype=torch.bfloat16)
    projection = torch.linspace(-0.2, 0.2, 32, dtype=torch.bfloat16).unsqueeze(0)

    result = model.preprocess_weights(
        {
            "model.output_attn_res_norm.weight": norm,
            "model.output_attn_res_proj.weight": projection,
        }
    )

    folded = result["model.output_res_score.weight"]
    assert folded.dtype == torch.float32
    torch.testing.assert_close(folded, projection.float() * norm.float().unsqueeze(0))
