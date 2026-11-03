# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LFM3 configuration, rotary layout, hybrid cache and export contract tests."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest

from mobius import build_from_module
from mobius._configs import Lfm3Config
from mobius._registry import registry
from mobius._testing import create_test_builder, create_test_input
from mobius.integrations.ort_genai import write_ort_genai_config
from mobius.models.lfm3 import Lfm3Attention, Lfm3CausalLMModel


def tiny_lfm3_config(**overrides) -> Lfm3Config:
    fields = dict(
        model_type="lfm3",
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=3,
        vocab_size=256,
        max_position_embeddings=128,
        layer_types=["conv", "full_attention", "sliding_attention"],
        sliding_window=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    fields.update(overrides)
    return Lfm3Config(**fields)


@pytest.mark.parametrize("raw", [True, False])
def test_lfm3_config_extraction_pinned_json_fields(raw):
    # Architectural fields from revision 875094441ea6517efc0f44a5685f05e381b72b14.
    fields = dict(
        model_type="lfm3",
        hidden_size=2048,
        intermediate_size=7680,
        num_attention_heads=16,
        num_key_value_heads=4,
        num_hidden_layers=24,
        head_dim=128,
        vocab_size=128000,
        max_position_embeddings=262144,
        norm_eps=1e-5,
        conv_L_cache=7,
        conv_bias=False,
        tie_word_embeddings=True,
        use_sandwich_norm=True,
        sliding_window=256,
        rope_parameters={
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 1e6,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000},
        },
        layer_types=[
            "full_attention"
            if i % 4 == 2
            else "sliding_attention"
            if i > 0 and i % 4 == 0
            else "conv"
            for i in range(24)
        ],
    )
    config = Lfm3Config.from_transformers(fields if raw else SimpleNamespace(**fields))
    assert config.intermediate_size == 7680
    assert config.max_position_embeddings == 262144
    assert config.short_conv_kernel == 7
    assert config.rms_norm_eps == pytest.approx(1e-5)
    assert config.layer_types == fields["layer_types"]
    assert config.rope_parameters == fields["rope_parameters"]
    assert config.use_sandwich_norm
    assert config.tie_word_embeddings
    assert registry.get_config_class("lfm3") is Lfm3Config


def test_lfm3_config_none_defaults_and_optional_norm_bias():
    config = Lfm3Config.from_transformers(
        dict(
            model_type="lfm3",
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=9,
            intermediate_size=130,
            vocab_size=256,
            layer_types=None,
            rope_parameters=None,
            use_sandwich_norm=False,
            conv_bias=True,
        )
    )
    assert config.layer_types == [
        "conv",
        "conv",
        "full_attention",
        "conv",
        "sliding_attention",
        "conv",
        "full_attention",
        "conv",
        "sliding_attention",
    ]
    module = Lfm3CausalLMModel(config)
    assert module.config.intermediate_size == 130
    assert list(module.model.layers[0].feed_forward.w1.weight.shape) == [130, 64]
    assert module.model.layers[0].post_operator_norm is None
    assert module.model.layers[0].post_ffn_norm is None
    assert module.model.layers[0].conv.in_proj.bias is not None


def test_lfm3_attention_contract_is_independent_of_config_label():
    config = tiny_lfm3_config(model_type="synthetic")
    module = Lfm3CausalLMModel(config)
    model = build_from_module(
        module, config, task="hybrid-text-generation", execution_provider="cuda"
    )["model"]
    assert sum(node.op_type == "Attention" for node in model.graph) == 2
    assert not any(node.op_type == "GroupQueryAttention" for node in model.graph)
    fields = {
        field.name: getattr(config, field.name)
        for field in dataclasses.fields(config)
        if field.name != "model_type"
    }
    assert Lfm3Config(**fields).model_type == "lfm3"


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"layer_types": ["conv"]}, "exactly"),
        ({"layer_types": ["conv", "invalid", "full_attention"]}, "Unsupported"),
        ({"rope_parameters": {"full_attention": {"rope_theta": 1e6}}}, "no entry"),
        ({"sliding_window": None}, "positive sliding_window"),
        ({"sliding_window": 0}, "positive sliding_window"),
        ({"head_dim": 15}, "even"),
        ({"short_conv_kernel": 1}, "at least"),
        (
            {
                "rope_parameters": {
                    "full_attention": {"rope_type": "dynamic", "rope_theta": 1e6},
                    "sliding_attention": {"rope_theta": 10000},
                }
            },
            "default zero-tail",
        ),
    ],
)
def test_lfm3_invalid_config_fails_explicitly(overrides, match):
    with pytest.raises(ValueError, match=match):
        tiny_lfm3_config(**overrides)


def test_lfm3_zero_tail_rope_keeps_full_dimension_frequencies():
    module = Lfm3CausalLMModel(tiny_lfm3_config())
    rope = module.model.rotary_emb[0]
    expected_freq = np.zeros(8, dtype=np.float32)
    expected_freq[:4] = 1 / (1e6 ** (np.arange(0, 8, 2, dtype=np.float32) / 16))
    np.testing.assert_allclose(
        rope.cos_cache.const_value.numpy()[3], np.cos(3 * expected_freq), atol=1e-7
    )
    np.testing.assert_allclose(
        rope.sin_cache.const_value.numpy()[3], np.sin(3 * expected_freq), atol=1e-7
    )
    sliding = module.model.rotary_emb[1]
    assert np.count_nonzero(sliding.sin_cache.const_value.numpy()[3]) == 8


def test_lfm3_rotary_layout_splits_full_head_and_uses_fp32():
    attention = Lfm3Attention(tiny_lfm3_config(dtype=ir.DataType.FLOAT16))
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1, 3, 64], dtype=ir.DataType.FLOAT16)
    cos = create_test_input(builder, "cos", [1, 3, 8], dtype=ir.DataType.FLOAT16)
    sin = create_test_input(builder, "sin", [1, 3, 8], dtype=ir.DataType.FLOAT16)
    result = attention._apply_rotary(op, x, (cos, sin), 4)
    builder._adapt_outputs([result], "")
    assert not any(node.op_type == "RotaryEmbedding" for node in graph)
    split = next(node for node in graph if node.op_type == "Split")
    assert split.inputs[0].producer().op_type == "Cast"
    assert split.inputs[0].producer().attributes.get_int("to") == ir.DataType.FLOAT
    reshape = split.inputs[0].producer().inputs[0].producer()
    np.testing.assert_array_equal(reshape.inputs[1].const_value.numpy(), [0, 0, -1, 16])


@pytest.mark.parametrize(
    "ep,dtype",
    [
        ("default", ir.DataType.FLOAT),
        ("onnx-standard", ir.DataType.FLOAT),
        ("cuda", ir.DataType.FLOAT),
    ],
)
def test_lfm3_portable_cuda_cache_graph(ep, dtype):
    config = tiny_lfm3_config(dtype=dtype)
    pkg = build_from_module(
        Lfm3CausalLMModel(config), config, task="hybrid-text-generation", execution_provider=ep
    )
    graph = pkg["model"].graph
    inputs = {value.name: value for value in graph.inputs}
    outputs = {value.name for value in graph.outputs}
    assert list(inputs["past_key_values.0.conv_state"].shape)[1:] == [64, 6]
    for i in [1, 2]:
        assert list(inputs[f"past_key_values.{i}.key"].shape)[1::2] == [2, 16]
        assert {f"present.{i}.key", f"present.{i}.value"} <= outputs
    assert "present.0.conv_state" in outputs
    attention_nodes = [
        node for node in graph if node.op_type in {"Attention", "GroupQueryAttention"}
    ]
    assert len(attention_nodes) == 2
    assert all(node.op_type == "Attention" for node in attention_nodes)
    assert not any(node.op_type == "RotaryEmbedding" for node in graph)
    if ep == "cuda":
        assert not any(node.op_type == "CausalConvWithState" for node in graph)
    if ep == "onnx-standard":
        assert all(not node.domain for node in graph.all_nodes())
    assert "Unpacked" in pkg["model"].metadata_props["mobius.input_contract"]


def test_lfm3_graph_derived_genai_metadata(tmp_path):
    config = tiny_lfm3_config()
    pkg = build_from_module(Lfm3CausalLMModel(config), config, task="hybrid-text-generation")
    write_ort_genai_config(pkg, str(tmp_path))
    metadata = json.loads((tmp_path / "genai_config.json").read_text())
    assert metadata["model"]["type"] == "lfm2"
    decoder = metadata["model"]["decoder"]
    assert decoder["num_hidden_layers"] == 3
    assert decoder["inputs"]["past_key_names"] == "past_key_values.%d.key"
    assert decoder["inputs"]["past_value_names"] == "past_key_values.%d.value"
    assert decoder["inputs"]["past_conv_names"] == "past_key_values.%d.conv_state"
    assert decoder["outputs"]["present_conv_names"] == "present.%d.conv_state"
    assert decoder["outputs"]["present_key_names"] == "present.%d.key"
    assert decoder["inputs"]["position_ids"] == "position_ids"
    assert decoder["head_size"] == 16


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_lfm3_rejects_unverified_reduced_precision(dtype):
    with pytest.raises(ValueError, match="float32 compute only"):
        Lfm3CausalLMModel(tiny_lfm3_config(dtype=dtype))


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_lfm3_rejects_dtype_changed_after_module_construction(dtype):
    config = tiny_lfm3_config()
    module = Lfm3CausalLMModel(config)
    config.dtype = dtype
    with pytest.raises(ValueError, match="float32 compute only"):
        build_from_module(module, config, task="hybrid-text-generation")


def test_lfm3_sandwich_norm_parameter_names_and_tied_embeddings():
    config = tiny_lfm3_config()
    module = Lfm3CausalLMModel(config)
    assert module.lm_head.weight is module.model.embed_tokens.weight
    names = {name for name, _ in module.named_parameters()}
    for i in range(3):
        assert f"model.layers.{i}.post_operator_norm.weight" in names
        assert f"model.layers.{i}.post_ffn_norm.weight" in names
        assert f"model.layers.{i}.feed_forward.w1.weight" in names
    assert "model.layers.1.self_attn.gate_proj.weight" in names
    without = Lfm3CausalLMModel(dataclasses.replace(config, use_sandwich_norm=False))
    assert not any("post_" in name for name, _ in without.named_parameters())
