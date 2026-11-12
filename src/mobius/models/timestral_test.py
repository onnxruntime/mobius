# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses
import json

import onnx_ir as ir
import pytest
from onnx_ir.passes.common import CheckerPass

from mobius import ModelPackage, build_from_module
from mobius.integrations.transformers._config_resolver import _try_load_config_json
from mobius.models.timestral import TimestralConfig, TimestralModel
from tests.build_graph._support import _fill_dummy_weights


def test_timestral_graph_and_initializer_names():
    config = TimestralConfig(
        hidden_size=32, num_attention_heads=2, intermediate_size=48, patch_size=4
    )
    package = build_from_module(TimestralModel(config), config, "time-series-forecasting")
    model = package["model"]
    assert [value.name for value in model.graph.outputs] == ["quantiles"]
    assert {value.name for value in model.graph.inputs} == {
        "context",
        "observed_mask",
        "group_ids",
        "horizon_patches",
    }
    assert "representation.forecast_query" in model.graph.initializers
    assert "transformer.layers.1.attns.1._gate.weight" in model.graph.initializers
    _fill_dummy_weights(model)
    CheckerPass()(model)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_timestral_rejects_unqualified_precision_at_construction_and_build(dtype):
    config = TimestralConfig()
    module = TimestralModel(config)
    reduced = dataclasses.replace(config, dtype=dtype)
    with pytest.raises(ValueError, match="float32 only"):
        TimestralModel(reduced)
    with pytest.raises(ValueError, match="float32 only"):
        build_from_module(module, reduced, "time-series-forecasting")


def test_timestral_native_config_detection(tmp_path):
    raw = {
        "patch_size": 32,
        "d_model": 192,
        "num_heads": 3,
        "mlp_hidden_dim": 512,
        "block_schedule": ["time", "time_variate"],
        "quantile_knots": [0.1, 0.5, 0.9],
        "block_size": 32,
        "context_length": 32768,
        "time_attn_gate": "sdpa_sigmoid",
        "variate_attn_gate": "sdpa_sigmoid",
        "split_context_horizon_attn": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    parsed = _try_load_config_json(str(tmp_path))
    assert parsed.model_type == "timestral"
    config = TimestralConfig.from_transformers(parsed)
    assert config.hidden_size == 192
    assert config.block_schedule == ("time", "time_variate")
    del raw["quantile_knots"]
    (tmp_path / "config.json").write_text(json.dumps(raw))
    assert _try_load_config_json(str(tmp_path)) is None


def test_timestral_forecasting_settings_survive_package_roundtrip(tmp_path):
    config = TimestralConfig(
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=48,
        patch_size=4,
        block_schedule=("variate_time",),
        quantile_knots=(0.2, 0.5, 0.8),
        context_length=64,
        block_size=3,
        minimum_scale=0.01,
    )
    package = build_from_module(TimestralModel(config), config, "time-series-forecasting")
    _fill_dummy_weights(package["model"])
    package.save(str(tmp_path), progress_bar=False)
    restored = ModelPackage.load(str(tmp_path))
    assert TimestralConfig.from_onnx(restored["model"]) == config
