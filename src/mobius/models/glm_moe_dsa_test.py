# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from mobius._builder import build_from_module
from mobius._config_resolver import _default_task_for_model
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
            index_topk_freq=4,
            index_skip_topk_offset=3,
            index_share_for_mtp_iteration=True,
        )

        config = ArchitectureConfig.from_transformers(hf_config)

        assert config.model_type == "glm_moe_dsa"
        assert config.num_local_experts == 2
        assert config.shared_expert_intermediate_size == 32
        assert config.qk_nope_head_dim == 12
        assert config.qk_rope_head_dim == 4
        assert config.index_topk == 8
        assert config.index_topk_freq == 4
        assert config.index_skip_topk_offset == 3
        assert config.index_share_for_mtp_iteration is True
        assert config.rope_interleave is True

    def test_registry(self):
        assert registry.get("glm_moe_dsa") is GlmMoeDsaCausalLMModel
        assert _default_task_for_model("glm_moe_dsa") == "glm-moe-dsa"

    def test_builds_full_attention_mla_moe_graph(self):
        config = _tiny_glm_moe_dsa_config()
        package = build_from_module(
            GlmMoeDsaCausalLMModel(config),
            config,
            task="glm-moe-dsa",
        )
        graph = package["model"].graph

        assert count_op_type(graph, "Attention") == config.num_hidden_layers
        assert count_op_type(graph, "ScatterElements") >= 1
        assert count_op_type(graph, "Sigmoid") >= config.num_hidden_layers - 1
        assert any("layers.0.self_attn.indexer.wk.weight" in name for name in graph.initializers)
        assert not any("layers.1.self_attn.indexer" in name for name in graph.initializers)
        assert {value.name for value in graph.outputs} >= {
            "logits",
            "present.0.key",
            "present.0.value",
        }
        assert set(package) == {"model", "mtp"}

        mtp_graph = package["mtp"].graph
        assert count_op_type(mtp_graph, "Attention") == 1
        assert count_op_type(mtp_graph, "ScatterElements") == 1
        assert {value.name for value in mtp_graph.outputs} == {
            "mtp_hidden",
            "present.0.key",
            "present.0.value",
            "topk_indices",
        }

    def test_indexer_rotary_uses_full_rotation(self):
        """Regression: indexer RoPE must rotate the full index_head_dim.

        The indexer key shares the model's rotary_emb cos/sin cache (sized
        qk_rope_head_dim / 2). The opset-24 RotaryEmbedding op validates that
        the cos cache last dim equals head_size / 2 or rotary_embedding_dim / 2.
        Passing ``rotary_embedding_dim = index_head_dim // 2`` made the op
        expect a cache of that_value / 2, which mismatched the shared cache and
        raised "cos_cache dimension 2 should be same as head_size / 2 or
        rotary_embedding_dim / 2" at runtime. Full rotation (0) matches the
        shared cache like the main MLA q_rope/k_rope calls do.
        """
        config = _tiny_glm_moe_dsa_config()
        package = build_from_module(
            GlmMoeDsaCausalLMModel(config),
            config,
            task="glm-moe-dsa",
        )
        graph = package["model"].graph

        indexer_rope_nodes = [
            node
            for node in graph
            if node.op_type == "RotaryEmbedding" and "indexer" in (node.name or "")
        ]
        assert indexer_rope_nodes, "expected at least one indexer RotaryEmbedding node"
        for node in indexer_rope_nodes:
            attr = node.attributes.get("rotary_embedding_dim")
            dim = attr.value if attr is not None else 0
            assert dim == 0, (
                f"indexer RotaryEmbedding {node.name!r} must use full rotation "
                f"(rotary_embedding_dim=0) to match the shared cos cache, got {dim}"
            )

    def test_quantized_graph_uses_matmul_nbits(self):
        config = _tiny_glm_moe_dsa_config(
            quantization=QuantizationConfig(
                bits=4,
                group_size=32,
                quant_method="gguf",
                sym=False,
            )
        )
        graph = build_from_module(
            GlmMoeDsaCausalLMModel(config),
            config,
            task="glm-moe-dsa",
        )["model"].graph
        assert count_op_type(graph, "MatMulNBits") > 0

    def test_preprocess_maps_indexer_and_mtp_layer(self):
        config = _tiny_glm_moe_dsa_config()
        model = GlmMoeDsaCausalLMModel(config)
        state_dict = {
            "model.layers.1.mlp.gate.weight": torch.zeros(2, 64),
            "model.layers.1.mlp.experts.0.gate_proj.weight": torch.zeros(32, 64),
            "model.layers.0.self_attn.indexer.wk.weight": torch.zeros(8, 64),
            "model.layers.4.self_attn.q_a_proj.weight": torch.zeros(16, 64),
            "model.layers.4.eh_proj.weight": torch.zeros(64, 128),
            "model.layers.4.shared_head.norm.weight": torch.zeros(64),
        }

        result = model.preprocess_weights(state_dict)

        assert "model.layers.1.mlp.moe.gate.weight" in result
        assert "model.layers.1.mlp.moe.experts.0.gate_proj.weight" in result
        assert "model.layers.0.self_attn.indexer.wk.weight" in result
        assert "mtp.layer.self_attn.q_a_proj.weight" in result
        assert "mtp.eh_proj.weight" in result
        assert "mtp.shared_head.norm.weight" in result

    def test_full_attention_fallback(self):
        config = _tiny_glm_moe_dsa_config(
            use_dsa=False,
            num_nextn_predict_layers=0,
        )
        package = build_from_module(
            GlmMoeDsaCausalLMModel(config),
            config,
            task="glm-moe-dsa",
        )
        graph = package["model"].graph

        assert set(package) == {"model"}
        assert count_op_type(graph, "Attention") == config.num_hidden_layers
        assert count_op_type(graph, "ScatterElements") == 0
        assert not any("indexer" in name for name in graph.initializers)
