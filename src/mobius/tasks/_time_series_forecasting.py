# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Padded-batch raw-series time-series forecasting task."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class TimeSeriesForecastingTask(ModelTask):
    """Build the TimesFM patched-input forecasting pipeline.

    ``raw_preprocessor`` accepts right-aligned context tensors ``[B, V, C]``
    and left-aligned future-covariate tensors ``[B, V, H]``. Boolean observed
    tensors distinguish missing values from padding; ``context_lengths`` and
    ``horizon_lengths`` define each row's valid extents. ``variate_roles`` uses
    0=target, 1=past-only, and 2=past-future. Lengths must be positive and no
    larger than their padded axes; invalid roles fail closed as masked variates.

    The package separates data-dependent preprocessing and postprocessing from
    the transformer so ``model`` contains no ONNX control-flow operators and
    can be captured independently by an execution provider. Symmetric
    averaging and outer z-normalization remain opt-in host wrapper policies.
    """

    model_roles: ClassVar[dict[str, str]] = {
        "raw_preprocessor": "encoder",
        "preprocessor": "encoder",
        "model": "encoder",
        "postprocessor": "encoder",
        "stitcher": "encoder",
    }

    def build(self, module: nn.Module, config: BaseModelConfig) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        variates = ir.SymbolicDim("variates")
        patches = ir.SymbolicDim("patches")
        patch_len = config.input_patch_len

        raw_graph, builder = _make_graph(name="timesfm3_raw_preprocessor")
        context = ir.SymbolicDim("context")
        horizon = ir.SymbolicDim("horizon")
        context_values = builder.input(
            "context_values",
            dtype=config.dtype,
            shape=[batch, variates, context],
        )
        context_observed = builder.input(
            "context_observed",
            dtype=ir.DataType.BOOL,
            shape=[batch, variates, context],
        )
        future_values = builder.input(
            "future_values",
            dtype=config.dtype,
            shape=[batch, variates, horizon],
        )
        future_observed = builder.input(
            "future_observed",
            dtype=ir.DataType.BOOL,
            shape=[batch, variates, horizon],
        )
        context_lengths = builder.input(
            "context_lengths",
            dtype=ir.DataType.INT64,
            shape=[batch],
        )
        horizon_lengths = builder.input(
            "horizon_lengths",
            dtype=ir.DataType.INT64,
            shape=[batch],
        )
        variate_roles = builder.input(
            "variate_roles",
            dtype=ir.DataType.INT64,
            shape=[batch, variates],
        )
        (
            raw_values,
            raw_masks,
            raw_patch_is_target,
            raw_patch_cpm_mask,
            trend_slope,
            trend_intercept,
            apply_detrend,
            target_mask,
            nonnegative_mask,
            raw_context_lengths,
            raw_horizon_lengths,
            context_patch_count,
            forecast_patch_counts,
        ) = module.prepare_raw_series(
            builder.op,
            context_values,
            context_observed,
            future_values,
            future_observed,
            context_lengths,
            horizon_lengths,
            variate_roles,
        )
        builder.add_output(raw_values, "values")
        builder.add_output(raw_masks, "masks")
        builder.add_output(raw_patch_is_target, "patch_is_target")
        builder.add_output(raw_patch_cpm_mask, "patch_cpm_mask")
        builder.add_output(trend_slope, "trend_slope")
        builder.add_output(trend_intercept, "trend_intercept")
        builder.add_output(apply_detrend, "apply_detrend")
        builder.add_output(target_mask, "target_mask")
        builder.add_output(nonnegative_mask, "nonnegative_mask")
        builder.add_output(raw_context_lengths, "context_lengths")
        builder.add_output(raw_horizon_lengths, "horizon_lengths")
        builder.add_output(context_patch_count, "context_patch_count")
        builder.add_output(forecast_patch_counts, "forecast_patch_counts")

        preprocess_graph, builder = _make_graph(name="timesfm3_preprocessor")
        values = builder.input(
            "values",
            dtype=config.dtype,
            shape=[batch, variates, patches, patch_len],
        )
        masks = builder.input(
            "masks",
            dtype=ir.DataType.BOOL,
            shape=[batch, variates, patches, patch_len],
        )
        patch_is_target = builder.input(
            "patch_is_target",
            dtype=ir.DataType.BOOL,
            shape=[batch, variates, patches],
        )
        patch_cpm_mask = builder.input(
            "patch_cpm_mask",
            dtype=ir.DataType.BOOL,
            shape=[batch, patches],
        )
        model_inputs, patch_mask, running_count, running_mean, running_std = module.preprocess(
            builder.op,
            values=values,
            masks=masks,
            patch_is_target=patch_is_target,
            patch_cpm_mask=patch_cpm_mask,
        )
        builder.add_output(model_inputs, "model_inputs")
        builder.add_output(patch_mask, "patch_mask")
        builder.add_output(running_count, "revin_count")
        builder.add_output(running_mean, "revin_mean")
        builder.add_output(running_std, "revin_std")

        model_graph, builder = _make_graph(name="timesfm3_model")
        model_inputs = builder.input(
            "model_inputs",
            dtype=config.dtype,
            shape=[
                batch,
                variates,
                patches,
                2 * (config.input_patch_len + config.output_patch_len),
            ],
        )
        patch_mask = builder.input(
            "patch_mask",
            dtype=ir.DataType.BOOL,
            shape=[batch, variates, patches],
        )
        raw_logits = module.forecast(builder.op, model_inputs, patch_mask)
        builder.add_output(raw_logits, "raw_logits")

        postprocess_graph, builder = _make_graph(name="timesfm3_postprocessor")
        raw_logits = builder.input(
            "raw_logits",
            dtype=config.dtype,
            shape=[
                batch,
                variates,
                patches,
                config.output_patch_len * len(config.quantiles),
            ],
        )
        running_count = builder.input(
            "revin_count",
            dtype=ir.DataType.FLOAT,
            shape=[batch, variates, patches],
        )
        running_mean = builder.input(
            "revin_mean",
            dtype=ir.DataType.FLOAT,
            shape=[batch, variates, patches],
        )
        running_std = builder.input(
            "revin_std",
            dtype=ir.DataType.FLOAT,
            shape=[batch, variates, patches],
        )
        patch_cpm_mask = builder.input(
            "patch_cpm_mask",
            dtype=ir.DataType.BOOL,
            shape=[batch, patches],
        )
        logits, running_mean, running_std = module.postprocess(
            builder.op,
            raw_logits,
            running_count,
            running_mean,
            running_std,
            patch_cpm_mask,
        )
        builder.add_output(logits, "logits")
        builder.add_output(running_mean, "revin_mean")
        builder.add_output(running_std, "revin_std")

        stitch_graph, builder = _make_graph(name="timesfm3_stitcher")
        logits = builder.input(
            "logits",
            dtype=config.dtype,
            shape=[
                batch,
                variates,
                patches,
                config.output_patch_len,
                len(config.quantiles),
            ],
        )
        trend_slope = builder.input(
            "trend_slope",
            dtype=ir.DataType.FLOAT,
            shape=[batch, variates],
        )
        trend_intercept = builder.input(
            "trend_intercept",
            dtype=ir.DataType.FLOAT,
            shape=[batch, variates],
        )
        apply_detrend = builder.input(
            "apply_detrend",
            dtype=ir.DataType.BOOL,
            shape=[batch, variates],
        )
        target_mask = builder.input(
            "target_mask",
            dtype=ir.DataType.BOOL,
            shape=[batch, variates],
        )
        nonnegative_mask = builder.input(
            "nonnegative_mask",
            dtype=ir.DataType.BOOL,
            shape=[batch, variates],
        )
        make_positive = builder.input(
            "make_positive",
            dtype=ir.DataType.BOOL,
            shape=[],
        )
        context_lengths = builder.input(
            "context_lengths",
            dtype=ir.DataType.INT64,
            shape=[batch],
        )
        horizon_lengths = builder.input(
            "horizon_lengths",
            dtype=ir.DataType.INT64,
            shape=[batch],
        )
        context_patch_count = builder.input(
            "context_patch_count",
            dtype=ir.DataType.INT64,
            shape=[],
        )
        forecast_patch_counts = builder.input(
            "forecast_patch_counts",
            dtype=ir.DataType.INT64,
            shape=[batch],
        )
        point_forecast, quantile_forecasts, validity = module.stitch_forecast(
            builder.op,
            logits,
            trend_slope,
            trend_intercept,
            apply_detrend,
            target_mask,
            nonnegative_mask,
            make_positive,
            context_lengths,
            horizon_lengths,
            context_patch_count,
            forecast_patch_counts,
        )
        builder.add_output(point_forecast, "point_forecast")
        builder.add_output(quantile_forecasts, "quantile_forecasts")
        builder.add_output(validity, "validity")
        return ModelPackage(
            {
                "raw_preprocessor": _make_model(raw_graph),
                "preprocessor": _make_model(preprocess_graph),
                "model": _make_model(model_graph),
                "postprocessor": _make_model(postprocess_graph),
                "stitcher": _make_model(stitch_graph),
            },
            config=config,
        )
