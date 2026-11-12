# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Target-only Timestral forecasting graph contract."""

from __future__ import annotations

import json
from typing import ClassVar

import onnx_ir as ir

from mobius._model_package import ModelPackage
from mobius.models.timestral import TimestralConfig
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class TimestralTask(ModelTask):
    """Forecast packed variates; matching group IDs identify a multivariate series.

    Context length must be a positive multiple of patch_size, bounded by the
    checkpoint context_length. horizon_patches must be in [1, block_size].
    Use the forecasting helper to validate and pad ordinary input arrays.
    """

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(self, module, config):
        config.validate()
        graph, builder = _make_graph()
        context = builder.input(
            "context", dtype=ir.DataType.FLOAT, shape=["rows", "context_length"]
        )
        observed = builder.input(
            "observed_mask", dtype=ir.DataType.BOOL, shape=["rows", "context_length"]
        )
        groups = builder.input("group_ids", dtype=ir.DataType.INT64, shape=["rows"])
        horizon = builder.input("horizon_patches", dtype=ir.DataType.INT64, shape=[1])
        result = module(builder.op, context, observed, groups, horizon)
        result.shape = ir.Shape(["rows", "prediction_length", len(config.quantile_knots)])
        result.dtype = ir.DataType.FLOAT
        builder.add_output(result, "quantiles")
        model = _make_model(graph)
        settings = {name: getattr(config, name) for name in TimestralConfig.__annotations__}
        model.metadata_props["mobius.timestral.config"] = json.dumps(settings)
        return ModelPackage({"model": model}, config=config)
