# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import torch

from mobius._configs import CodeShellConfig, Jais2Config, XverseConfig
from mobius.models.legacy_decoder import (
    CodeShellCausalLMModel,
    Jais2CausalLMModel,
    XverseCausalLMModel,
)
from mobius.tasks import CausalLMTask


def _codeshell_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="kclgpt",
        vocab_size=24,
        n_positions=32,
        n_embd=8,
        n_layer=1,
        n_head=2,
        n_inner=16,
        activation_function="gelu_pytorch_tanh",
        layer_norm_epsilon=1e-5,
        group_query_attention=True,
        num_query_groups=1,
        position_embedding_type="rope",
        rope_scaling=None,
    )


def test_specialized_model_config_classes_match_their_architectures() -> None:
    assert Jais2CausalLMModel.config_class is Jais2Config
    assert CodeShellCausalLMModel.config_class is CodeShellConfig
    assert XverseCausalLMModel.config_class is XverseConfig


def test_codeshell_real_hf_names_close_graph_and_preserve_qkv_values() -> None:
    config = CodeShellConfig.from_transformers(_codeshell_hf_config())
    module = CodeShellCausalLMModel(config)
    graph = CausalLMTask().build(module, config)["model"]

    hidden = config.hidden_size
    kv_hidden = config.num_key_value_heads * config.head_dim
    fused_rows = hidden + 2 * kv_hidden
    qkv_weight = torch.arange(fused_rows * hidden, dtype=torch.float32).reshape(
        fused_rows, hidden
    )
    qkv_bias = torch.arange(fused_rows, dtype=torch.float32)
    state = {
        "transformer.wte.weight": torch.zeros(config.vocab_size, hidden),
        "transformer.h.0.ln_1.weight": torch.ones(hidden),
        "transformer.h.0.ln_1.bias": torch.zeros(hidden),
        "transformer.h.0.attn.c_attn.weight": qkv_weight,
        "transformer.h.0.attn.c_attn.bias": qkv_bias,
        "transformer.h.0.attn.c_proj.weight": torch.zeros(hidden, hidden),
        "transformer.h.0.attn.c_proj.bias": torch.zeros(hidden),
        "transformer.h.0.ln_2.weight": torch.ones(hidden),
        "transformer.h.0.ln_2.bias": torch.zeros(hidden),
        "transformer.h.0.mlp.c_fc.weight": torch.zeros(config.intermediate_size, hidden),
        "transformer.h.0.mlp.c_fc.bias": torch.zeros(config.intermediate_size),
        "transformer.h.0.mlp.c_proj.weight": torch.zeros(hidden, config.intermediate_size),
        "transformer.h.0.mlp.c_proj.bias": torch.zeros(hidden),
        "transformer.h.0.attn.rotary_emb.inv_freq": torch.ones(config.head_dim // 2),
        "transformer.ln_f.weight": torch.ones(hidden),
        "transformer.ln_f.bias": torch.zeros(hidden),
    }

    processed = module.preprocess_weights(state)
    prefix = "model.layers.0.self_attn."
    torch.testing.assert_close(processed[prefix + "q_proj.weight"], qkv_weight[:hidden])
    torch.testing.assert_close(
        processed[prefix + "k_proj.weight"], qkv_weight[hidden : hidden + kv_hidden]
    )
    torch.testing.assert_close(
        processed[prefix + "v_proj.weight"], qkv_weight[hidden + kv_hidden :]
    )
    torch.testing.assert_close(processed[prefix + "q_proj.bias"], qkv_bias[:hidden])
    torch.testing.assert_close(
        processed[prefix + "k_proj.bias"], qkv_bias[hidden : hidden + kv_hidden]
    )
    torch.testing.assert_close(
        processed[prefix + "v_proj.bias"], qkv_bias[hidden + kv_hidden :]
    )

    graph_weights = {
        name
        for name in graph.graph.initializers
        if not name.startswith("const_") and ".rotary_emb." not in name
    }
    assert graph_weights == set(processed) - {"lm_head.weight"}
    assert "lm_head.weight" not in processed
    assert "model.embed_tokens.weight" in processed


def test_xverse_hf_config_builds_rotary_embeddings() -> None:
    hf_config = SimpleNamespace(
        model_type="xverse",
        vocab_size=24,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        max_position_embeddings=32,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )
    config = XverseConfig.from_transformers(hf_config)
    module = XverseCausalLMModel(config)
    graph = CausalLMTask().build(module, config)["model"]

    assert module.model.rotary_emb is not None
    assert any(".rotary_emb." in name for name in graph.graph.initializers)


def test_jais2_hf_config_emits_and_closes_all_bias_weights() -> None:
    hf_config = SimpleNamespace(
        model_type="jais2",
        vocab_size=24,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="relu2",
        max_position_embeddings=32,
        rope_parameters={"rope_theta": 10_000.0, "rope_type": "default"},
        attention_bias=True,
        mlp_bias=True,
        layer_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    config = Jais2Config.from_transformers(hf_config)
    module = Jais2CausalLMModel(config)
    graph = CausalLMTask().build(module, config)["model"]
    graph_weights = {
        name: value
        for name, value in graph.graph.initializers.items()
        if not name.startswith("const_") and ".rotary_emb." not in name
    }
    state = {
        name: torch.zeros(tuple(int(dim) for dim in value.shape))
        for name, value in graph_weights.items()
    }
    processed = module.preprocess_weights(state)

    assert set(graph_weights) == set(processed)
    expected_biases = {
        "model.layers.0.input_layernorm.bias",
        "model.layers.0.post_attention_layernorm.bias",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.0.self_attn.o_proj.bias",
        "model.layers.0.mlp.up_proj.bias",
        "model.layers.0.mlp.down_proj.bias",
        "model.norm.bias",
    }
    assert expected_biases <= set(graph_weights)
