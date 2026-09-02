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
    return config, module, package


@pytest.fixture(scope="module")
def tiny_package():
    return _build_tiny()


def _run_pipeline(package, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from mobius._testing.ort_inference import OnnxModelSession

    preprocessor = OnnxModelSession(package["preprocessor"])
    preprocessed = preprocessor.run(feeds)
    preprocessor.close()

    model = OnnxModelSession(package["model"])
    raw_logits = model.run(
        {
            "model_inputs": preprocessed["model_inputs"],
            "patch_mask": preprocessed["patch_mask"],
        }
    )
    model.close()

    postprocessor = OnnxModelSession(package["postprocessor"])
    outputs = postprocessor.run(
        {
            "raw_logits": raw_logits["raw_logits"],
            "revin_count": preprocessed["revin_count"],
            "revin_mean": preprocessed["revin_mean"],
            "revin_std": preprocessed["revin_std"],
            "patch_cpm_mask": feeds["patch_cpm_mask"],
        }
    )
    postprocessor.close()
    return outputs


def _run_full_pipeline(
    package,
    feeds: dict[str, np.ndarray],
    *,
    make_positive: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    from mobius._testing.ort_inference import OnnxModelSession

    raw_session = OnnxModelSession(package["raw_preprocessor"])
    raw = raw_session.run(feeds)
    raw_session.close()
    patched = _run_pipeline(
        package,
        {name: raw[name] for name in ("values", "masks", "patch_is_target", "patch_cpm_mask")},
    )
    stitch_session = OnnxModelSession(package["stitcher"])
    outputs = stitch_session.run(
        {
            "logits": patched["logits"],
            "make_positive": np.array(make_positive),
            **{
                name: raw[name]
                for name in (
                    "trend_slope",
                    "trend_intercept",
                    "apply_detrend",
                    "target_mask",
                    "nonnegative_mask",
                    "context_lengths",
                    "horizon_lengths",
                    "context_patch_count",
                    "forecast_patch_counts",
                )
            },
        }
    )
    stitch_session.close()
    return raw, outputs


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
    assert config.use_linear_detrending
    assert config.linear_detrending_threshold == pytest.approx(0.5)


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


def test_graph_contract_and_checkpoint_weight_names(tiny_package) -> None:
    config, module, package = tiny_package

    assert set(package) == {
        "raw_preprocessor",
        "preprocessor",
        "model",
        "postprocessor",
        "stitcher",
    }
    assert {value.name for value in package["raw_preprocessor"].graph.inputs} == {
        "context_values",
        "context_observed",
        "future_values",
        "future_observed",
        "context_lengths",
        "horizon_lengths",
        "variate_roles",
    }
    assert {value.name for value in package["raw_preprocessor"].graph.outputs} == {
        "values",
        "masks",
        "patch_is_target",
        "patch_cpm_mask",
        "trend_slope",
        "trend_intercept",
        "apply_detrend",
        "target_mask",
        "nonnegative_mask",
        "context_lengths",
        "horizon_lengths",
        "context_patch_count",
        "forecast_patch_counts",
    }
    assert {value.name for value in package["preprocessor"].graph.inputs} == {
        "values",
        "masks",
        "patch_is_target",
        "patch_cpm_mask",
    }
    assert {value.name for value in package["preprocessor"].graph.outputs} == {
        "model_inputs",
        "patch_mask",
        "revin_count",
        "revin_mean",
        "revin_std",
    }
    assert {value.name for value in package["model"].graph.inputs} == {
        "model_inputs",
        "patch_mask",
    }
    assert {value.name for value in package["model"].graph.outputs} == {"raw_logits"}
    assert {value.name for value in package["postprocessor"].graph.inputs} == {
        "raw_logits",
        "revin_count",
        "revin_mean",
        "revin_std",
        "patch_cpm_mask",
    }
    assert {value.name for value in package["postprocessor"].graph.outputs} == {
        "logits",
        "revin_mean",
        "revin_std",
    }
    assert {value.name for value in package["stitcher"].graph.outputs} == {
        "point_forecast",
        "quantile_forecasts",
        "validity",
    }
    assert package["preprocessor"].graph.inputs[0].shape[-1] == config.input_patch_len
    assert not {node.op_type for node in package["model"].graph}.intersection(
        {"Scan", "Loop", "If"}
    )

    initializers = {
        name for component in package.values() for name in component.graph.initializers
    }
    assert "pre_transformer_resblock.hidden_layer.weight" in initializers
    assert "transformer_stack.layers.0.seq_attn.query_proj.weight" in initializers
    assert "transformer_stack.layers.0.var_attn.per_dim_scale.per_dim_scale" in initializers
    assert "output_head.weight" in initializers
    assert "output_head.bias" in initializers
    parameter_names = {name for name, _ in module.named_parameters()}
    assert parameter_names.issubset(initializers)
    assert all(
        sum(name in component.graph.initializers for component in package.values()) == 1
        for name in parameter_names
    )

    import torch

    state_dict = {
        name: torch.zeros(tuple(parameter.shape), dtype=torch.float32)
        for name, parameter in module.named_parameters()
    }
    package.apply_weights(state_dict, fold_constants=False)
    applied_names = {
        name
        for component in package.values()
        for name, initializer in component.graph.initializers.items()
        if initializer.const_value is not None
    }
    assert parameter_names.issubset(applied_names)
    assert len(list(TimesFM3Model(TimesFM3Config()).named_parameters())) == 445


def test_raw_preprocessor_matches_interpolation_detrending_and_padding(
    tiny_package,
) -> None:
    from mobius._testing.ort_inference import OnnxModelSession

    _, _, package = tiny_package
    context = np.zeros((2, 3, 6), dtype=np.float32)
    context[0, 0, -5:] = [0.0, np.nan, 2.0, 3.0, 4.0]
    context[0, 1, -5:] = [0.0, np.nan, 4.0, 9.0, 16.0]
    context[0, 2, -5:] = [10.0, np.nan, 14.0, 16.0, 18.0]
    context[1, 0, -3:] = [2.0, 4.0, 6.0]
    context[1, 1, -3:] = [1.0, 2.0, 5.0]
    context[1, 2, -3:] = [3.0, 6.0, 9.0]
    context_observed = np.isfinite(context)

    future = np.zeros((2, 3, 10), dtype=np.float32)
    future[0, 2] = np.arange(20.0, 40.0, 2.0)
    future[0, 2, 1] = np.nan
    future[1, 2, :3] = [12.0, np.nan, 18.0]
    future_observed = np.isfinite(future)

    session = OnnxModelSession(package["raw_preprocessor"])
    outputs = session.run(
        {
            "context_values": context,
            "context_observed": context_observed,
            "future_values": future,
            "future_observed": future_observed,
            "context_lengths": np.array([5, 3], dtype=np.int64),
            "horizon_lengths": np.array([10, 3], dtype=np.int64),
            "variate_roles": np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
        }
    )
    session.close()

    assert outputs["values"].shape == (2, 3, 5, 4)
    np.testing.assert_array_equal(
        outputs["patch_cpm_mask"],
        np.array([[False, False, True, True, True]] * 2),
    )
    np.testing.assert_array_equal(
        outputs["forecast_patch_counts"], np.array([2, 1], dtype=np.int64)
    )
    assert outputs["context_patch_count"] == 2

    flat = outputs["values"].reshape(2, 3, -1)
    masks = outputs["masks"].reshape(2, 3, -1)
    # Recover the interpolated nonlinear row from its emitted trend metadata.
    past_time = np.arange(-4, 1, dtype=np.float32) / 5.0
    restored_past = np.where(
        outputs["apply_detrend"][0, 1],
        flat[0, 1, 3:8]
        + outputs["trend_slope"][0, 1] * past_time
        + outputs["trend_intercept"][0, 1],
        flat[0, 1, 3:8],
    )
    np.testing.assert_allclose(restored_past, [0.0, 2.0, 4.0, 9.0, 16.0], atol=1e-5)
    np.testing.assert_array_equal(
        masks[0, 1, :8], [True, True, True, False, False, False, False, False]
    )
    # Perfectly linear target and past-future rows become zero, including
    # interpolated future covariates across the context/horizon boundary.
    np.testing.assert_allclose(flat[0, 0, 3:8], 0.0, atol=1e-6)
    np.testing.assert_allclose(flat[0, 2, 3:18], 0.0, atol=1e-6)
    np.testing.assert_allclose(outputs["trend_slope"][0, [0, 2]], [5.0, 10.0], atol=1e-5)
    np.testing.assert_allclose(outputs["trend_intercept"][0, [0, 2]], [4.0, 18.0], atol=1e-5)
    np.testing.assert_array_equal(outputs["apply_detrend"][0, [0, 2]], [True, True])
    np.testing.assert_array_equal(
        outputs["nonnegative_mask"],
        np.array([[True, False, False], [True, False, False]]),
    )


def test_stitcher_matches_overlap_trend_sort_mask_and_clipping(tiny_package) -> None:
    from mobius._testing.ort_inference import OnnxModelSession

    _, _, package = tiny_package
    logits = np.zeros((2, 2, 5, 8, 3), dtype=np.float32)
    quantile_offsets = np.array([2.0, -1.0, 1.0], dtype=np.float32)
    for batch_index in range(2):
        for variate_index in range(2):
            for forecast_index, patch_index in enumerate((1, 2)):
                center = (
                    -12.0
                    + 20.0 * batch_index
                    + 5.0 * variate_index
                    + 8.0 * forecast_index
                    + np.arange(8, dtype=np.float32)
                )
                logits[batch_index, variate_index, patch_index] = (
                    center[:, None] + quantile_offsets
                )

    trend_slope = np.array([[5.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    trend_intercept = np.array([[4.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    apply_detrend = np.array([[True, False], [False, False]])
    target_mask = np.array([[True, False], [True, True]])
    nonnegative = np.array([[True, False], [False, False]])
    context_lengths = np.array([5, 3], dtype=np.int64)
    horizon_lengths = np.array([6, 10], dtype=np.int64)
    forecast_patch_counts = np.array([1, 2], dtype=np.int64)

    session = OnnxModelSession(package["stitcher"])
    outputs = session.run(
        {
            "logits": logits,
            "trend_slope": trend_slope,
            "trend_intercept": trend_intercept,
            "apply_detrend": apply_detrend,
            "target_mask": target_mask,
            "nonnegative_mask": nonnegative,
            "make_positive": np.array(True),
            "context_lengths": context_lengths,
            "horizon_lengths": horizon_lengths,
            "context_patch_count": np.array(2, dtype=np.int64),
            "forecast_patch_counts": forecast_patch_counts,
        }
    )
    session.close()

    weights = np.linspace(1.0, 0.0, 4, dtype=np.float32)[None, :, None]
    expected = np.zeros((2, 2, 10, 3), dtype=np.float32)
    for batch_index, patch_count in enumerate(forecast_patch_counts):
        selected = logits[batch_index, :, 1 : 1 + patch_count]
        if patch_count == 1:
            stitched = selected[:, 0]
        else:
            stitched = np.concatenate(
                [
                    selected[:, 0, :4],
                    (
                        weights[0] * selected[:, 0, 4:8]
                        + (1.0 - weights[0]) * selected[:, 1, :4]
                    ),
                    selected[:, 1, 4:8],
                ],
                axis=1,
            )
        expected[batch_index, :, : horizon_lengths[batch_index]] = stitched[
            :, : horizon_lengths[batch_index]
        ]
    steps = np.arange(1, 11, dtype=np.float32)[None, None, :]
    trend = (
        trend_slope[:, :, None] * steps / context_lengths[:, None, None]
        + trend_intercept[:, :, None]
    )
    expected += np.where(apply_detrend[:, :, None], trend, 0.0)[..., None]
    expected = np.sort(expected, axis=-1)
    expected = np.where(nonnegative[:, :, None, None], np.maximum(expected, 0.0), expected)
    validity = target_mask[:, :, None] & (
        np.arange(10)[None, None, :] < horizon_lengths[:, None, None]
    )
    expected = np.where(validity[..., None], expected, 0.0)

    np.testing.assert_allclose(outputs["quantile_forecasts"], expected, atol=1e-6)
    np.testing.assert_allclose(outputs["point_forecast"], expected[..., 1], atol=1e-6)
    np.testing.assert_array_equal(outputs["validity"], validity)
    assert np.all(np.diff(outputs["quantile_forecasts"], axis=-1) >= 0)
    assert np.all(outputs["quantile_forecasts"][0, 0] >= 0)


def test_full_fp32_pipeline_matches_linear_extrapolation(tiny_package) -> None:
    _, module, package = tiny_package
    for name, parameter in module.named_parameters():
        for component in package.values():
            initializer = component.graph.initializers.get(name)
            if initializer is not None:
                initializer.const_value = ir.tensor(
                    np.zeros(list(parameter.shape), dtype=np.float32)
                )

    context = np.array(
        [
            [[0.0, 1.0, 2.0, 3.0, 4.0]],
            [[0.0, 0.0, 10.0, 12.0, 14.0]],
        ],
        dtype=np.float32,
    )
    _, outputs = _run_full_pipeline(
        package,
        {
            "context_values": context,
            "context_observed": np.array([[[True] * 5], [[False, False, True, True, True]]]),
            "future_values": np.zeros((2, 1, 6), dtype=np.float32),
            "future_observed": np.zeros((2, 1, 6), dtype=np.bool_),
            "context_lengths": np.array([5, 3], dtype=np.int64),
            "horizon_lengths": np.array([6, 2], dtype=np.int64),
            "variate_roles": np.zeros((2, 1), dtype=np.int64),
        },
    )

    expected = np.array(
        [[[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], [[16.0, 18.0, 0.0, 0.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(outputs["point_forecast"], expected, atol=1e-5)
    np.testing.assert_allclose(
        outputs["quantile_forecasts"],
        np.repeat(expected[..., None], 3, axis=-1),
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        outputs["validity"],
        np.array([[[True] * 6], [[True, True, False, False, False, False]]]),
    )


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype"),
    [
        (ir.DataType.FLOAT, np.float32),
        (ir.DataType.FLOAT16, np.float16),
    ],
)
def test_tiny_model_runs_with_random_weights(dtype, numpy_dtype) -> None:

    config = _tiny_config()
    config.dtype = dtype
    model = build_from_module(
        TimesFM3Model(config),
        config,
        task=TimeSeriesForecastingTask(),
    )
    rng = np.random.default_rng(0)
    for component in model.values():
        for initializer in component.graph.initializers.values():
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

    outputs = _run_pipeline(
        model,
        {
            "values": values,
            "masks": masks,
            "patch_is_target": patch_is_target,
            "patch_cpm_mask": patch_cpm_mask,
        },
    )

    assert outputs["logits"].shape == (2, 3, 5, 8, 3)
    assert outputs["revin_mean"].shape == (2, 3, 5)
    assert outputs["revin_std"].shape == (2, 3, 5)
    assert np.isfinite(outputs["logits"]).all()

    if dtype == ir.DataType.FLOAT:
        # A row's earlier forecast is invariant to another row forcing extra
        # left context patches and right horizon patches in the padded batch.
        context = np.zeros((2, 3, 9), dtype=np.float32)
        context[0, :, -5:] = np.array(
            [[1.0, 2.0, 4.0, 3.0, 5.0], [2.0, 1.0, 3.0, 2.0, 4.0], [0.0, 2.0, 1.0, 4.0, 3.0]]
        )
        context[1] = rng.standard_normal((3, 9)).astype(np.float32)
        future = np.zeros((2, 3, 10), dtype=np.float32)
        future[:, 2] = rng.standard_normal((2, 10)).astype(np.float32)
        common = {
            "context_values": context,
            "context_observed": np.ones_like(context, dtype=np.bool_),
            "future_values": future,
            "future_observed": np.ones_like(future, dtype=np.bool_),
            "variate_roles": np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
        }
        _, padded = _run_full_pipeline(
            model,
            {
                **common,
                "context_lengths": np.array([5, 9], dtype=np.int64),
                "horizon_lengths": np.array([3, 10], dtype=np.int64),
            },
        )
        _, alone = _run_full_pipeline(
            model,
            {name: value[:1] for name, value in common.items()}
            | {
                "context_lengths": np.array([5], dtype=np.int64),
                "horizon_lengths": np.array([3], dtype=np.int64),
            },
        )
        np.testing.assert_allclose(
            padded["quantile_forecasts"][0, :, :3],
            alone["quantile_forecasts"][0],
            rtol=1e-4,
            atol=1e-4,
        )


def test_running_revin_and_cpm_refinement_match_reference() -> None:

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
    )
    for component in model.values():
        for initializer in component.graph.initializers.values():
            if initializer.const_value is None:
                initializer.const_value = ir.tensor(
                    np.zeros(list(initializer.shape), dtype=np.float32)
                )
    output_bias = np.zeros(12, dtype=np.float32)
    output_bias[1::3] = [1.0, 2.0, 3.0, 4.0]
    model["model"].graph.initializers["output_head.bias"].const_value = ir.tensor(output_bias)

    outputs = _run_pipeline(
        model,
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
        },
    )

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
