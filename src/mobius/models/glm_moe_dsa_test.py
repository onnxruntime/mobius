# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig, QuantizationConfig
from mobius._registry import registry
from mobius._testing import count_op_type, make_config
from mobius.models.glm_moe_dsa import GlmMoeDsaCausalLMModel


def _tiny_glm_moe_dsa_config(**overrides):
    defaults = dict(
        model_type="glm_moe_dsa",
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=8,
        head_dim=4,
        rope_interleave=True,
        num_local_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        n_shared_experts=1,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        index_topk=8,
        index_head_dim=8,
        index_n_heads=2,
        indexer_types=["full", "shared", "shared", "shared"],
        num_nextn_predict_layers=1,
    )
    defaults.update(overrides)
    return make_config(**defaults)


class TestGlmMoeDsaExport:
    def test_hf_config_extraction(self):
        hf_config = GlmMoeDsaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            q_lora_rank=16,
            kv_lora_rank=8,
            qk_nope_head_dim=12,
            qk_rope_head_dim=4,
            v_head_dim=8,
            n_routed_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=32,
            n_shared_experts=1,
            first_k_dense_replace=1,
            index_topk=8,
            index_head_dim=8,
            index_n_heads=2,
        )

        config = ArchitectureConfig.from_transformers(hf_config)

        assert config.model_type == "glm_moe_dsa"
        assert config.num_local_experts == 2
        assert config.shared_expert_intermediate_size == 32
        assert config.qk_nope_head_dim == 12
        assert config.qk_rope_head_dim == 4
        assert config.index_topk == 8
        assert config.rope_interleave is True

    def test_registry(self):
        assert registry.get("glm_moe_dsa") is GlmMoeDsaCausalLMModel

    def test_builds_full_attention_mla_moe_graph(self):
        config = _tiny_glm_moe_dsa_config()
        graph = build_from_module(GlmMoeDsaCausalLMModel(config), config)["model"].graph

        assert count_op_type(graph, "Attention") == config.num_hidden_layers
        assert count_op_type(graph, "TopK") == config.num_hidden_layers - 1
        assert count_op_type(graph, "Sigmoid") >= config.num_hidden_layers - 1
        assert not any("indexer" in name for name in graph.initializers)
        assert {value.name for value in graph.outputs} >= {
            "logits",
            "present.0.key",
            "present.0.value",
        }

    def test_quantized_graph_uses_matmul_nbits(self):
        config = _tiny_glm_moe_dsa_config(
            quantization=QuantizationConfig(
                bits=4,
                group_size=32,
                quant_method="gguf",
                sym=False,
            )
        )
        graph = build_from_module(GlmMoeDsaCausalLMModel(config), config)["model"].graph
        assert count_op_type(graph, "MatMulNBits") > 0

    def test_preprocess_drops_indexer_and_mtp_layer(self):
        config = _tiny_glm_moe_dsa_config()
        model = GlmMoeDsaCausalLMModel(config)
        state_dict = {
            "model.layers.1.mlp.gate.weight": torch.zeros(2, 64),
            "model.layers.1.mlp.experts.0.gate_proj.weight": torch.zeros(32, 64),
            "model.layers.1.indexer.wk.weight": torch.zeros(8, 64),
            "model.layers.4.self_attn.q_a_proj.weight": torch.zeros(16, 64),
        }

        result = model.preprocess_weights(state_dict)

        assert "model.layers.1.mlp.moe.gate.weight" in result
        assert "model.layers.1.mlp.moe.experts.0.gate_proj.weight" in result
        assert not any("indexer" in key for key in result)
        assert not any(".layers.4." in key for key in result)
