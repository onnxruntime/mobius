# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Puzzle topology extraction tests; no remote code or checkpoint downloads."""

from __future__ import annotations

import copy
import types

import numpy as np
import onnx_ir as ir
import pytest

from mobius._builder import build_from_module
from mobius._configs import NemotronHPuzzleConfig
from mobius.components import Float32GatedRMSNorm
from mobius.integrations.transformers._config_resolver import _config_from_hf
from mobius.models.nemotron_h import NemotronHPuzzleCausalLMModel


def _source_config():
    return {
        "model_type": "nemotron_h_puzzle",
        "hidden_size": 16,
        "vocab_size": 32,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "mamba_num_heads": 4,
        "mamba_head_dim": 8,
        "n_groups": 2,
        "ssm_state_size": 4,
        "moe_latent_size": 8,
        "moe_shared_expert_intermediate_size": 24,
        "block_configs": [
            {"block_type": "mamba"},
            {
                "block_type": "moe",
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 8,
            },
            {"block_type": "attention"},
            {
                "block_type": "moe",
                "n_routed_experts": 3,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 12,
            },
        ],
    }


@pytest.mark.parametrize("native_blocks", [False, True])
def test_puzzle_extraction_preserves_each_layer(native_blocks):
    source = _source_config()
    original = copy.deepcopy(source)
    if native_blocks:
        source["block_configs"] = [
            types.SimpleNamespace(**block) for block in source["block_configs"]
        ]
    config = _config_from_hf(source)
    assert isinstance(config, NemotronHPuzzleConfig)
    assert config.num_hidden_layers == 4
    assert config.layer_types == ["mamba2", "moe", "full_attention", "moe"]
    assert config.for_layer(1).num_local_experts == 2
    assert config.for_layer(1).num_experts_per_tok == 1
    assert config.for_layer(1).moe_intermediate_size == 8
    assert config.for_layer(3).num_local_experts == 3
    assert config.for_layer(3).num_experts_per_tok == 2
    assert config.for_layer(3).moe_intermediate_size == 12
    assert config.for_layer(3).moe_latent_size == 8
    assert config.for_layer(3).shared_expert_intermediate_size == 24
    if not native_blocks:
        assert source == original


def test_puzzle_module_uses_resolved_expert_shapes():
    config = NemotronHPuzzleConfig.from_transformers(_source_config())
    module = NemotronHPuzzleCausalLMModel(config)
    params = dict(module.named_parameters())
    assert list(params["model.layers.1.moe.experts.1.up_proj.weight"].shape) == [8, 8]
    assert "model.layers.1.moe.experts.2.up_proj.weight" not in params
    assert list(params["model.layers.3.moe.experts.2.up_proj.weight"].shape) == [12, 8]
    assert list(params["model.layers.3.moe.experts.2.down_proj.weight"].shape) == [8, 12]
    assert list(params["model.layers.3.moe.gate.weight"].shape) == [3, 16]
    assert list(params["model.layers.3.moe.shared_experts.up_proj.weight"].shape) == [24, 16]
    assert isinstance(module.model.layers[0].mamba.norm, Float32GatedRMSNorm)
    assert module.model.layers[0].mamba.time_step_min == pytest.approx(0.0)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.BFLOAT16])
def test_puzzle_mixed_mamba_graph_roundtrip(tmp_path, dtype):
    config = NemotronHPuzzleConfig.from_transformers(_source_config())
    config.dtype = dtype
    module = NemotronHPuzzleCausalLMModel(config)
    for _, parameter in module.named_parameters():
        parameter.const_value = ir.tensor(
            np.zeros(list(parameter.shape), dtype=parameter.dtype.numpy())
        )
    package = build_from_module(module, config, task="hybrid-text-generation")
    path = tmp_path / "puzzle.onnx"
    ir.save(package["model"], path)
    graph = ir.load(path).graph
    inputs = {value.name: value for value in graph.inputs}
    assert inputs["past_key_values.0.ssm_state"].dtype == ir.DataType.FLOAT
    assert inputs["past_key_values.0.conv_state"].dtype == dtype
    assert inputs["past_key_values.0.conv_state"].shape[-1] == config.mamba_d_conv - 1
    assert "past_key_values.2.key" in inputs
    assert all(
        value is None or value.shape is None or len(value.shape) != 2
        for node in graph
        if node.op_type == "RMSNormalization"
        for value in node.inputs[1:2]
    )


def test_puzzle_global_moe_fields_apply_when_all_blocks_inherit():
    source = _source_config()
    for block in source["block_configs"]:
        if block["block_type"] == "moe":
            for field in ("n_routed_experts", "num_experts_per_tok", "moe_intermediate_size"):
                block.pop(field)
    source.update(n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=20)
    config = NemotronHPuzzleConfig.from_transformers(source)
    for index in (1, 3):
        layer = config.for_layer(index)
        assert (
            layer.num_local_experts,
            layer.num_experts_per_tok,
            layer.moe_intermediate_size,
        ) == (
            4,
            2,
            20,
        )


@pytest.mark.parametrize(
    "field", ["n_routed_experts", "num_experts_per_tok", "moe_intermediate_size"]
)
def test_puzzle_rejects_partial_blockwise_fields(field):
    source = _source_config()
    source["block_configs"][1].pop(field)
    with pytest.raises(ValueError, match="must be set for all MoE blocks"):
        NemotronHPuzzleConfig.from_transformers(source)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None])
def test_puzzle_rejects_invalid_expert_counts(value):
    source = _source_config()
    for index in (1, 3):
        source["block_configs"][index]["n_routed_experts"] = value
    with pytest.raises(ValueError, match="positive integer"):
        NemotronHPuzzleConfig.from_transformers(source)


def test_puzzle_rejects_topk_larger_than_expert_count():
    source = _source_config()
    source["block_configs"][1]["num_experts_per_tok"] = 3
    with pytest.raises(ValueError, match="exceeds"):
        NemotronHPuzzleConfig.from_transformers(source)


@pytest.mark.parametrize("blocks", [[], [{"block_type": "mlp"}], [{"block_type": "typo"}]])
def test_puzzle_rejects_missing_or_unknown_blocks(blocks):
    source = _source_config()
    source["block_configs"] = blocks
    with pytest.raises(ValueError, match="block"):
        NemotronHPuzzleConfig.from_transformers(source)


def test_puzzle_block_schedule_overrides_legacy_global_schedule():
    source = _source_config()
    source.update(num_hidden_layers=100, layers_block_type=["attention"] * 100)
    config = NemotronHPuzzleConfig.from_transformers(source)
    assert config.num_hidden_layers == 4
    assert config.layer_types == ["mamba2", "moe", "full_attention", "moe"]


@pytest.mark.parametrize("upper", [float("inf"), {"__float__": "Infinity"}])
def test_puzzle_does_not_clamp_to_dt_initialization_bound(upper):
    source = _source_config()
    source.update(time_step_min=0.001, time_step_limit=[0.0, upper])
    config = NemotronHPuzzleConfig.from_transformers(source)
    assert config.mamba_time_step_min == pytest.approx(0.0)
    assert config.mamba_ssm_dtype == ir.DataType.FLOAT


@pytest.mark.parametrize(
    ("name", "expected"),
    [("float32", ir.DataType.FLOAT), ("bfloat16", ir.DataType.BFLOAT16)],
)
def test_puzzle_preserves_declared_ssm_cache_dtype(name, expected):
    source = _source_config()
    source["mamba_ssm_cache_dtype"] = name
    assert NemotronHPuzzleConfig.from_transformers(source).mamba_ssm_dtype == expected


def test_puzzle_rejects_unknown_ssm_cache_dtype():
    source = _source_config()
    source["mamba_ssm_cache_dtype"] = "int8"
    with pytest.raises(ValueError, match="cache_dtype"):
        NemotronHPuzzleConfig.from_transformers(source)
