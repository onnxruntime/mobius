# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import onnx_ir as ir

from mobius import build_from_module
from mobius._configs import Lfm2Config
from mobius._registry import registry
from mobius.models.lfm2 import Lfm2CausalLMModel


def _hf_config(**overrides):
    fields = dict(
        model_type="lfm2",
        hidden_size=2048,
        intermediate_size=6656,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=1,
        vocab_size=65536,
        max_position_embeddings=32768,
        hidden_act="silu",
        head_dim=64,
        pad_token_id=0,
        norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
        layer_types=["full_attention"],
        block_auto_adjust_ff_dim=True,
        block_ffn_dim_multiplier=1.0,
        block_multiple_of=256,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_config_auto_adjusts_official_6656_width_to_4608():
    config = Lfm2Config.from_transformers(_hf_config())
    assert config.intermediate_size == 6656
    assert config.effective_intermediate_size == 4608


def test_config_auto_adjust_order_matches_transformers():
    config = Lfm2Config.from_transformers(_hf_config(block_ffn_dim_multiplier=1.5))
    # int(2 * 6656 / 3) -> 4437, int(1.5 * 4437) -> 6655,
    # then ceil to the next multiple of 256.
    assert config.effective_intermediate_size == 6656


def test_config_disabled_auto_adjust_keeps_raw_width():
    config = Lfm2Config.from_transformers(_hf_config(block_auto_adjust_ff_dim=False))
    assert config.effective_intermediate_size == 6656


def test_registry_uses_lfm2_config():
    assert registry.get_config_class("lfm2") is Lfm2Config


def test_model_uses_effective_intermediate_size():
    module = Lfm2CausalLMModel(Lfm2Config.from_transformers(_hf_config()))
    mlp = module.model.layers[0].feed_forward
    assert module.config.intermediate_size == 4608
    assert module.config.effective_intermediate_size == 4608
    assert tuple(mlp.gate_proj.weight.shape) == (4608, 2048)
    assert tuple(mlp.up_proj.weight.shape) == (4608, 2048)
    assert tuple(mlp.down_proj.weight.shape) == (2048, 4608)


def test_cuda_graph_uses_lfm2_fusions():
    config = Lfm2Config.from_transformers(
        _hf_config(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            vocab_size=256,
            head_dim=16,
            layer_types=["conv", "full_attention"],
            block_auto_adjust_ff_dim=False,
            conv_L_cache=3,
            conv_bias=False,
        )
    )
    config.dtype = ir.DataType.FLOAT16
    module = Lfm2CausalLMModel(config)
    model = build_from_module(
        module,
        config,
        task="hybrid-text-generation",
        execution_provider="cuda",
    )["model"]

    counts = Counter((node.domain or "", node.op_type) for node in model.graph)
    assert counts["", "Swish"] == 2
    # CUDA has no CausalConvWithState kernel. Its standard Conv function body
    # must be inlined or ORT aborts while assigning providers.
    assert counts["com.microsoft", "CausalConvWithState"] == 0
    assert counts["", "Conv"] == 1
    assert counts["com.microsoft", "SkipSimplifiedLayerNormalization"] == 4

    remaining_norms = [node for node in model.graph if node.op_type == "RMSNormalization"]
    assert remaining_norms
    assert all(
        node.attributes.get_int("stash_type") == ir.DataType.FLOAT for node in remaining_norms
    )
