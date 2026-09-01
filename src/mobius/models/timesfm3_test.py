# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json

import numpy as np
import onnx_ir as ir
import pytest

from mobius import build_from_module
from mobius._registry import registry
from mobius.integrations.transformers._config_resolver import _try_load_config_json
from mobius.models.timesfm3 import TimesFM3Config, TimesFM3Model
from mobius.tasks import TimeSeriesForecastingTask, get_task


def _tiny_config() -> TimesFM3Config:
    return TimesFM3Config(
        input_patch_len=4,
        output_patch_len=8,
        quantiles=(0.1, 0.5, 0.9),
        num_layers=1,
        model_dims=16,
        transformer_hidden_dims=24,
        num_heads=4,
        max_variates=4,
    )


def _build_tiny():
    config = _tiny_config()
    module = TimesFM3Model(config)
    package = build_from_module(
        module,
        config,
        task=TimeSeriesForecastingTask(),
    )
    return config, module, package["model"]


def test_config_parses_official_nested_schema() -> None:
    config = TimesFM3Config.from_transformers(
        {
            "input_patch_len": 32,
            "output_patch_len": 64,
            "quantiles": [0.1, 0.5, 0.9],
            "use_iterative_cpm_revin": True,
            "use_variate_attention": True,
            "value_clip": 1e20,
            "residual_block_config": {"output_dims": 1280},
            "transformer_config": {
                "num_layers": 20,
                "transformer": {
                    "model_dims": 1280,
                    "hidden_dims": 1280,
                    "num_heads": 16,
                    "max_variates": 32,
                    "use_rope_seq": True,
                    "use_rope_var": False,
                },
            },
        }
    )

    assert config.input_patch_len == 32
    assert config.output_patch_len == 64
    assert config.num_layers == 20
    assert config.head_dim == 80
    assert config.quantiles == (0.1, 0.5, 0.9)
    assert config.rms_norm_eps == np.finfo(np.float32).eps


def test_raw_config_detection(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "input_patch_len": 32,
                "output_patch_len": 64,
                "quantiles": [0.1, 0.5, 0.9],
                "use_iterative_cpm_revin": True,
                "use_variate_attention": True,
                "residual_block_config": {"output_dims": 1280},
                "transformer_config": {
                    "num_layers": 20,
                    "transformer": {"model_dims": 1280, "max_variates": 32},
                },
            }
        )
    )

    config = _try_load_config_json(str(tmp_path))

    assert config is not None
    assert config.model_type == "timesfm3"


def test_graph_contract_and_checkpoint_weight_names() -> None:
    config, _, model = _build_tiny()

    assert {value.name for value in model.graph.inputs} == {
        "values",
        "masks",
        "patch_is_target",
        "patch_cpm_mask",
    }
    assert {value.name for value in model.graph.outputs} == {
        "logits",
        "revin_mean",
        "revin_std",
    }
    assert model.graph.inputs[0].shape[-1] == config.input_patch_len

    initializers = set(model.graph.initializers)
    assert "pre_transformer_resblock.hidden_layer.weight" in initializers
    assert "transformer_stack.layers.0.seq_attn.query_proj.weight" in initializers
    assert "transformer_stack.layers.0.var_attn.per_dim_scale.per_dim_scale" in initializers
    assert "output_head.weight" in initializers
    assert "output_head.bias" in initializers


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype"),
    [
        (ir.DataType.FLOAT, np.float32),
        (ir.DataType.FLOAT16, np.float16),
    ],
)
def test_tiny_model_runs_with_random_weights(dtype, numpy_dtype) -> None:
    from mobius._testing.ort_inference import OnnxModelSession

    config = _tiny_config()
    config.dtype = dtype
    model = build_from_module(
        TimesFM3Model(config),
        config,
        task=TimeSeriesForecastingTask(),
    )["model"]
    rng = np.random.default_rng(0)
    for initializer in model.graph.initializers.values():
        if initializer.const_value is None:
            initializer.const_value = ir.tensor(
                (rng.standard_normal(list(initializer.shape)) * 0.02).astype(numpy_dtype)
            )
    values = rng.standard_normal((2, 3, 5, config.input_patch_len)).astype(numpy_dtype)
    masks = np.zeros_like(values, dtype=np.bool_)
    masks[:, :, 0] = True
    patch_is_target = np.ones((2, 3, 5), dtype=np.bool_)
    patch_cpm_mask = np.zeros((2, 5), dtype=np.bool_)
    patch_cpm_mask[:, -2:] = True

    session = OnnxModelSession(model)
    outputs = session.run(
        {
            "values": values,
            "masks": masks,
            "patch_is_target": patch_is_target,
            "patch_cpm_mask": patch_cpm_mask,
        }
    )
    session.close()

    assert outputs["logits"].shape == (2, 3, 5, 8, 3)
    assert outputs["revin_mean"].shape == (2, 3, 5)
    assert outputs["revin_std"].shape == (2, 3, 5)
    assert np.isfinite(outputs["logits"]).all()


def test_running_revin_and_cpm_refinement_match_reference() -> None:
    from mobius._testing.ort_inference import OnnxModelSession

    config = TimesFM3Config(
        input_patch_len=2,
        output_patch_len=4,
        quantiles=(0.1, 0.5, 0.9),
        num_layers=0,
        model_dims=4,
        transformer_hidden_dims=4,
        num_heads=1,
        max_variates=1,
        use_variate_attention=False,
    )
    model = build_from_module(
        TimesFM3Model(config),
        config,
        task=TimeSeriesForecastingTask(),
    )["model"]
    for initializer in model.graph.initializers.values():
        if initializer.const_value is None:
            initializer.const_value = ir.tensor(
                np.zeros(list(initializer.shape), dtype=np.float32)
            )
    output_bias = np.zeros(12, dtype=np.float32)
    output_bias[1::3] = [1.0, 2.0, 3.0, 4.0]
    model.graph.initializers["output_head.bias"].const_value = ir.tensor(output_bias)

    session = OnnxModelSession(model)
    outputs = session.run(
        {
            "values": np.array(
                [[[[1.0, 3.0], [5.0, 7.0], [0.0, 0.0], [0.0, 0.0]]]],
                dtype=np.float32,
            ),
            "masks": np.array(
                [[[[False, False], [False, False], [True, True], [True, True]]]]
            ),
            "patch_is_target": np.ones((1, 1, 4), dtype=np.bool_),
            "patch_cpm_mask": np.array([[False, False, True, True]]),
        }
    )
    session.close()

    np.testing.assert_allclose(
        outputs["revin_mean"],
        np.array([[[2.0, 4.0, 4.0, 4.0]]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        outputs["revin_std"],
        np.array([[[1.0, np.sqrt(5.0), np.sqrt(5.0), np.sqrt(5.0)]]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        outputs["logits"][0, 0, :, :, 1],
        np.array(
            [
                [3.0, 4.0, 5.0, 6.0],
                [6.236068, 8.472136, 10.708204, 12.944272],
                [7.618034, 10.118034, 12.618034, 15.118034],
                [10.460805, 14.126524, 17.792244, 21.457964],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_registry_and_task_registration() -> None:
    assert registry.get("timesfm3") is TimesFM3Model
    assert registry.get_config_class("timesfm3") is TimesFM3Config
    assert isinstance(get_task("time-series-forecasting"), TimeSeriesForecastingTask)
