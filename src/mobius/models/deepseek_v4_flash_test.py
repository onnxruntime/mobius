# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import torch

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig, QuantizationConfig
from mobius._testing import count_op_type, make_config
from mobius.integrations.ort_genai import export_package
from mobius.integrations.transformers._config_resolver import _default_task_for_model
from mobius.models.deepseek_v4 import DeepSeekV4CausalLMModel


def _tiny_config(**overrides):
    values = dict(
        model_type="deepseek_v4",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        o_groups=2,
        o_lora_rank=8,
        num_local_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        num_hash_layers=1,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        swiglu_limit=10.0,
        rope_interleave=True,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=4,
    )
    values.update(overrides)
    return make_config(**values)


def test_real_config_fields_extract():
    hf_config = SimpleNamespace(
        model_type="deepseek_v4",
        vocab_size=129280,
        hidden_size=4096,
        intermediate_size=None,
        num_hidden_layers=43,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        max_position_embeddings=1048576,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10000,
        rope_scaling={
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        o_groups=8,
        o_lora_rank=1024,
        n_routed_experts=256,
        num_experts_per_tok=6,
        moe_intermediate_size=2048,
        n_shared_experts=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        scoring_func="sqrtsoftplus",
        num_hash_layers=None,
        mlp_layer_types=["hash_moe", "hash_moe", "hash_moe", "moe"],
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        compress_ratios=None,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        },
        compress_rope_theta=160000,
        sliding_window=128,
        swiglu_limit=10.0,
        tie_word_embeddings=False,
        num_nextn_predict_layers=1,
        attention_bias=False,
        pad_token_id=0,
    )
    config = ArchitectureConfig.from_transformers(hf_config)
    assert config.model_type == "deepseek_v4"
    assert config.head_dim == 512
    assert config.qk_rope_head_dim == 64
    assert config.num_local_experts == 256
    assert config.num_hash_layers == 3
    assert config.hc_mult == 4
    assert config.compress_ratios == [0, 0, 4, 128]
    assert config.num_nextn_predict_layers == 1
    assert config.rope_type == "yarn"
    assert _default_task_for_model("deepseek_v4") == "deepseek-v4"


def test_tiny_graph_builds_v4_backbone():
    config = _tiny_config()
    graph = build_from_module(DeepSeekV4CausalLMModel(config), config)["model"].graph
    assert graph.num_nodes() > 0
    assert count_op_type(graph, "Attention") == 0
    assert count_op_type(graph, "Softmax") >= config.num_hidden_layers
    assert count_op_type(graph, "ScatterElements") == 0
    assert count_op_type(graph, "Softplus") >= config.num_hidden_layers
    assert count_op_type(graph, "TopK") >= 1
    assert count_op_type(graph, "Gather") >= 1
    assert count_op_type(graph, "RMSNormalization") >= 1
    assert sum("attn_sink" in name for name in graph.initializers) == config.num_hidden_layers
    assert "hidden_states" not in {value.name for value in graph.outputs}


def test_csa_schedule_exports_compressor_and_indexer_tensors_with_dense_attention():
    config = _tiny_config(
        num_hidden_layers=4,
        compress_ratios=[0, 0, 4, 128],
    )
    graph = build_from_module(DeepSeekV4CausalLMModel(config), config, task="deepseek-v4")[
        "model"
    ].graph
    names = set(graph.initializers)

    assert "model.layers.2.self_attn.compressor.wkv.weight" in names
    assert "model.layers.2.self_attn.compressor.wgate.weight" in names
    assert "model.layers.2.self_attn.compressor.ape" in names
    assert "model.layers.2.self_attn.compressor.norm.weight" in names
    assert "model.layers.2.self_attn.indexer.wq_b.weight" in names
    assert "model.layers.2.self_attn.indexer.weights_proj.weight" in names
    assert "model.layers.2.self_attn.indexer.compressor.wkv.weight" in names
    assert "model.layers.3.self_attn.compressor.wkv.weight" in names
    assert not any("model.layers.3.self_attn.indexer" in name for name in names)
    assert count_op_type(graph, "Attention") == 0
    assert count_op_type(graph, "ScatterElements") == 0
    assert count_op_type(graph, "Softmax") >= config.num_hidden_layers


def test_mtp_sidecar_exports_official_block_and_hyper_connection_state():
    config = _tiny_config(
        num_nextn_predict_layers=1,
        compress_ratios=[0, 0, 0],
    )
    package = build_from_module(DeepSeekV4CausalLMModel(config), config, task="deepseek-v4")

    assert set(package) == {"model", "mtp"}
    assert "hidden_states" in {value.name for value in package["model"].graph.outputs}
    mtp_graph = package["mtp"].graph
    names = set(mtp_graph.initializers)
    assert "mtp.0.e_proj.weight" in names
    assert "mtp.0.h_proj.weight" in names
    assert "mtp.0.hc_head_fn.weight" in names
    assert "mtp.0.self_attn.attn_sink" in names
    assert count_op_type(mtp_graph, "Softmax") >= 1
    assert {value.name for value in mtp_graph.outputs} == {
        "mtp_hidden",
        "present.0.key",
        "present.0.value",
    }


def test_ort_genai_sidecar_metadata(tmp_path):
    config = _tiny_config(
        num_nextn_predict_layers=1,
        compress_ratios=[0, 0, 0],
    )
    package = build_from_module(DeepSeekV4CausalLMModel(config), config, task="deepseek-v4")

    for model in package.values():
        for initializer in model.graph.initializers.values():
            if initializer.const_value is None:
                initializer.const_value = ir.tensor(
                    np.zeros(initializer.shape, dtype=initializer.dtype.numpy())
                )
    result = export_package(package, str(tmp_path), progress_bar=False)
    assert (tmp_path / "model" / "model.onnx").is_file()
    assert (tmp_path / "mtp" / "model.onnx").is_file()
    with open(result["genai_config"]) as f:
        genai_config = json.load(f)
    assert genai_config["model"]["decoder"]["filename"] == "model/model.onnx"
    with open(result["mtp_config"]) as f:
        sidecar = json.load(f)

    assert sidecar["model"]["filename"] == "mtp/model.onnx"
    assert sidecar["num_nextn_predict_layers"] == 1
    assert "hidden_states" in sidecar["inputs"]
    assert "mtp_hidden" in sidecar["outputs"]


def test_parameter_shapes_match_v4_projections():
    config = _tiny_config(num_hidden_layers=1)
    model = DeepSeekV4CausalLMModel(config)
    attn = model.model.layers[0].self_attn
    assert list(attn.q_b_proj.weight.shape) == [32, 8]
    assert list(attn.kv_proj.weight.shape) == [16, 32]
    assert list(attn.o_a_proj.weight.shape) == [16, 16]
    assert list(attn.o_b_proj.weight.shape) == [32, 16]
    assert list(model.model.layers[0].mlp.gate.bias.shape) == [config.num_local_experts]


def test_four_and_eight_bit_graphs_use_matmul_nbits():
    for bits in (4, 8):
        config = _tiny_config(
            num_hidden_layers=1,
            compress_ratios=[4],
            quantization=QuantizationConfig(
                bits=bits,
                group_size=16,
                quant_method="gguf",
                sym=False,
            ),
        )
        graph = build_from_module(DeepSeekV4CausalLMModel(config), config)["model"].graph
        assert count_op_type(graph, "MatMulNBits") > 0
        assert "model.layers.0.self_attn.compressor.wkv.scales" in graph.initializers
        model = DeepSeekV4CausalLMModel(config)
        assert model.model.layers[0].self_attn.compressor.wkv._gguf_quantized_linear


def test_official_weight_names_are_remapped():
    config = _tiny_config(
        num_hidden_layers=4,
        num_nextn_predict_layers=1,
        compress_ratios=[0, 0, 4, 128, 0],
    )
    model = DeepSeekV4CausalLMModel(config)
    weights = {
        "embed.weight": torch.zeros(config.vocab_size, config.hidden_size),
        "layers.0.attn.wq_a.weight": torch.zeros(config.q_lora_rank, config.hidden_size),
        "layers.0.ffn.experts.0.w1.weight": torch.zeros(
            config.moe_intermediate_size, config.hidden_size
        ),
        "layers.0.hc_attn_base": torch.zeros((2 + config.hc_mult) * config.hc_mult),
        "layers.2.attn.attn_sink": torch.zeros(config.num_attention_heads),
        "layers.2.attn.compressor.ape": torch.zeros(4, 2 * config.head_dim),
        "layers.2.attn.indexer.weights_proj.weight": torch.zeros(
            config.index_n_heads, config.hidden_size
        ),
        "mtp.0.e_proj.weight": torch.zeros(config.hidden_size, config.hidden_size),
        "mtp.0.hc_head_fn": torch.zeros(config.hc_mult, config.hc_mult * config.hidden_size),
    }
    result = model.preprocess_weights(weights)
    assert "model.embed_tokens.weight" in result
    assert "model.layers.0.self_attn.q_a_proj.weight" in result
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in result
    assert "model.layers.0.hc_attn_base" in result
    assert "model.layers.2.self_attn.attn_sink" in result
    assert "model.layers.2.self_attn.compressor.ape" in result
    assert "model.layers.2.self_attn.indexer.weights_proj.weight" in result
    assert "mtp.0.e_proj.weight" in result
    assert "mtp.0.hc_head_fn.weight" in result
