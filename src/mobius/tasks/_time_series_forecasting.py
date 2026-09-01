# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Patched-input time-series forecasting task."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class TimeSeriesForecastingTask(ModelTask):
    """Build the TimesFM patched-input forecasting graph.

    Inputs use the upstream core contract: values and masks are
    ``[batch, variates, patches, patch_len]``; ``patch_is_target`` marks target
    and past-only variates; ``patch_cpm_mask`` marks forecast-horizon patches.
    """

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(self, module: nn.Module, config: BaseModelConfig) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        variates = ir.SymbolicDim("variates")
        patches = ir.SymbolicDim("patches")
        patch_len = config.input_patch_len

        graph, builder = _make_graph(name="timesfm3")
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
        logits, running_mean, running_std = module(
            builder.op,
            values=values,
            masks=masks,
            patch_is_target=patch_is_target,
            patch_cpm_mask=patch_cpm_mask,
        )
        builder.add_output(logits, "logits")
        builder.add_output(running_mean, "revin_mean")
        builder.add_output(running_std, "revin_std")
        return ModelPackage({"model": _make_model(graph)}, config=config)
