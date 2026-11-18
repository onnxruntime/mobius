# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Independent Timestral synthetic and real-weight forecasting parity."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from mobius import build_from_module
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.timestral import TimestralConfig, TimestralModel


def _reference_config(config):
    from timestral.config import TimestralConfig as ReferenceConfig

    fields = dataclasses.asdict(config)
    fields.update(
        d_model=config.hidden_size,
        num_heads=config.num_attention_heads,
        mlp_hidden_dim=config.intermediate_size,
    )
    return ReferenceConfig.from_dict(
        {
            field.name: fields[field.name]
            for field in dataclasses.fields(ReferenceConfig)
            if field.name in fields
        }
    )


@pytest.mark.integration
@pytest.mark.parametrize("schedule", [("time", "time_variate"), ("variate_time", "variate")])
@pytest.mark.parametrize("attend_missing", [True, False])
def test_timestral_synthetic_parity(schedule, attend_missing):
    from timestral.model import TimestralModel as ReferenceModel

    config = TimestralConfig(
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=48,
        patch_size=4,
        block_schedule=schedule,
        attend_missing_patches=attend_missing,
    )
    torch.manual_seed(3)
    reference = ReferenceModel(_reference_config(config)).eval()
    package = build_from_module(TimestralModel(config), config, "time-series-forecasting")
    package.apply_weights(reference.state_dict())
    context = np.random.default_rng(42).normal(size=(3, 16)).astype(np.float32)
    observed = np.ones(context.shape, dtype=np.bool_)
    observed[0, :4] = False
    observed[1, 8:12] = False
    observed[2] = False
    groups = np.array([4, 4, 9], dtype=np.int64)
    feeds = {
        "context": context,
        "observed_mask": observed,
        "group_ids": groups,
        "horizon_patches": np.array([3], dtype=np.int64),
    }
    with torch.inference_mode():
        expected = reference.predict_block(
            torch.from_numpy(np.where(observed, context, 0)),
            torch.from_numpy(observed),
            torch.from_numpy(groups),
            3,
        ).numpy()
    actual = OnnxModelSession(package).run(feeds)["quantiles"]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3, strict=True)
    assert np.all(np.diff(actual, axis=-1) >= 0)


@pytest.mark.integration
def test_timestral_multiblock_refinement_and_feedback():
    from timestral import TimestralForecaster
    from timestral.model import TimestralModel as ReferenceModel

    from mobius.integrations.timestral import forecast_timestral

    config = TimestralConfig(
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=48,
        patch_size=4,
        context_length=16,
    )
    torch.manual_seed(9)
    reference_model = ReferenceModel(_reference_config(config)).eval()
    package = build_from_module(TimestralModel(config), config, "time-series-forecasting")
    package.apply_weights(reference_model.state_dict())
    context = np.random.default_rng(42).normal(size=(2, 2, 21)).astype(np.float32)
    context[0, 0, :4] = np.nan
    context[0, 1, 8:10] = np.nan
    context[1] = np.abs(context[1])
    forecaster = TimestralForecaster(reference_model, per_shot_horizon_patches=2)
    expected = forecaster.forecast(context, 21).quantiles.numpy()
    actual = forecast_timestral(
        OnnxModelSession(package),
        config,
        context,
        21,
        per_shot_horizon_patches=2,
    )
    assert actual.shape == (2, 2, 21, 3)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3, strict=True)
