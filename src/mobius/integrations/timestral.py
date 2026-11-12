# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NumPy host orchestration for native ONNX Timestral forecast blocks.

The neural network and causal scaling execute in ONNX. This module handles
padding, advancing-frame refinement and block feedback, without importing the
upstream reference implementation or using PyTorch for inference.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from mobius.models.timestral import TimestralConfig


class ForecastSession(Protocol):
    """A session exposing named NumPy feeds and outputs."""

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...


def _clamp_infinities(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    high = np.max(np.where(finite, values, -np.inf), axis=-1, keepdims=True)
    low = np.min(np.where(finite, values, np.inf), axis=-1, keepdims=True)
    result = np.where(np.isposinf(values), high, np.where(np.isneginf(values), low, values))
    return np.where(np.isinf(result), 0.0, result)


def _refine_block(
    quantiles: np.ndarray,
    context: np.ndarray,
    mask: np.ndarray,
    config: TimestralConfig,
) -> np.ndarray:
    """Re-anchor each forecast patch after folding in preceding patch medians."""
    ps = config.patch_size
    patches = quantiles.shape[1] // ps
    if patches == 1:
        return quantiles
    x = context.astype(np.float64)
    w = mask.astype(np.float64)
    count = w.sum(axis=1)
    mean = x.sum(axis=1) / np.maximum(count, 1)
    m2 = (w * (x - mean[:, None]) ** 2).sum(axis=1)
    scale = np.maximum(np.sqrt(m2 / np.maximum(count - 1, 1)), config.minimum_scale)
    loc0, scale0 = mean.astype(np.float32), scale.astype(np.float32)
    median = (
        quantiles[..., config.quantile_knots.index(0.5)]
        .reshape(-1, patches, ps)
        .astype(np.float64)
    )
    normalized_median = (median - loc0[:, None, None]) / scale0[:, None, None]
    offset = mean.copy()
    clip = np.finfo(np.float32).max ** 0.5 / 1e3
    locations, scales = [], []
    for patch in range(patches):
        scale = np.maximum(
            np.sqrt(np.maximum(m2 / np.maximum(count - 1, 1), 0)), config.minimum_scale
        )
        locations.append(loc0 if patch == 0 else mean.astype(np.float32))
        scales.append(scale0 if patch == 0 else scale.astype(np.float32))
        folded = (
            np.clip(
                normalized_median[:, patch] * scale[:, None] + mean[:, None] - offset[:, None],
                -clip,
                clip,
            )
            + offset[:, None]
        )
        batch_mean = folded.mean(axis=-1)
        batch_m2 = ((folded - batch_mean[:, None]) ** 2).sum(axis=-1)
        new_count = count + ps
        delta = batch_mean - mean
        mean = mean + delta * ps / new_count
        m2 = m2 + batch_m2 + delta**2 * count * ps / new_count
        count = new_count
    loc = np.repeat(np.stack(locations, axis=1), ps, axis=1)
    scale = np.repeat(np.stack(scales, axis=1), ps, axis=1)
    normalized = (quantiles - loc0[:, None, None]) / scale0[:, None, None]
    return _clamp_infinities(normalized * scale[..., None] + loc[..., None])


def forecast_timestral(
    session: ForecastSession,
    config: TimestralConfig,
    context: np.ndarray,
    prediction_length: int,
    *,
    observation_mask: np.ndarray | None = None,
    per_shot_horizon_patches: int = 32,
    symmetric_averaging: bool = True,
    clip_nonnegative: bool = True,
) -> np.ndarray:
    """Forecast target series, returning ``[batch, variates, horizon, quantiles]``.

    ``context`` accepts ``[time]``, ``[batch, time]`` or
    ``[batch, variates, time]``. NaN and infinity are missing observations.
    Rows within a batch item attend each other; different batch items do not.
    Known-future, past-only and categorical covariates are not supported.
    Defaults match the upstream target-only forecaster, including reflected
    averaging, nonnegative clipping, real-value caps and advancing frames.
    """
    config.validate()
    if (
        not isinstance(prediction_length, int)
        or isinstance(prediction_length, bool)
        or prediction_length <= 0
    ):
        raise ValueError(
            f"prediction_length must be a positive integer, got {prediction_length!r}."
        )
    if (
        not isinstance(per_shot_horizon_patches, int)
        or isinstance(per_shot_horizon_patches, bool)
        or per_shot_horizon_patches <= 0
    ):
        raise ValueError(
            "per_shot_horizon_patches must be a positive integer; use 32 for the released checkpoints."
        )
    values = np.asarray(context, dtype=np.float32)
    original_shape = values.shape
    if values.ndim == 1:
        values = values[None, None]
    elif values.ndim == 2:
        values = values[:, None]
    if values.ndim != 3 or any(size == 0 for size in values.shape):
        raise ValueError(
            f"Timestral context must be a nonempty [T], [B,T], or [B,V,T] array, got {original_shape}."
        )
    mask = np.isfinite(values)
    if observation_mask is not None:
        user_mask = np.asarray(observation_mask)
        if user_mask.shape != original_shape or user_mask.dtype != np.bool_:
            raise ValueError(
                f"observation_mask must be boolean with context shape {original_shape}, got {user_mask.shape}/{user_mask.dtype}."
            )
        mask &= user_mask.reshape(values.shape)
    values = np.where(mask, values, 0)
    batch, variates, _ = values.shape
    groups = np.repeat(np.arange(batch, dtype=np.int64), variates)
    ps = config.patch_size
    cap = max(ps, config.context_length // ps * ps)
    block_size = min(per_shot_horizon_patches, config.block_size)
    if symmetric_averaging and any(
        abs(a + b - 1) > 1e-9
        for a, b in zip(config.quantile_knots, reversed(config.quantile_knots))
    ):
        raise ValueError(
            "Symmetric averaging requires quantile levels symmetric around 0.5; disable it for asymmetric checkpoints."
        )

    def decode(source: np.ndarray) -> np.ndarray:
        rows = source.reshape(-1, source.shape[-1])[:, -cap:]
        observed = mask.reshape(rows.shape[0], -1)[:, -cap:]
        padding = (-rows.shape[-1]) % ps
        rows = np.pad(rows, ((0, 0), (padding, 0)))
        observed = np.pad(observed, ((0, 0), (padding, 0)))
        initial = rows.astype(np.float64)
        count = np.maximum(observed.sum(axis=-1), 1)
        mean = initial.sum(axis=-1) / count
        scale = np.maximum(
            np.sqrt(
                (observed * (initial - mean[:, None]) ** 2).sum(axis=-1)
                / np.maximum(count - 1, 1)
            ),
            config.minimum_scale,
        ).astype(np.float32)
        low = np.min(np.where(observed, rows, np.inf), axis=-1)
        high = np.max(np.where(observed, rows, -np.inf), axis=-1)
        low = np.where(np.isfinite(low), low, 0) - 1e4 * scale
        high = np.where(np.isfinite(high), high, 0) + 1e4 * scale
        remaining = math.ceil(prediction_length / ps)
        blocks = []
        while remaining:
            horizon = min(block_size, remaining)
            outputs = session.run(
                {
                    "context": np.ascontiguousarray(rows),
                    "observed_mask": np.ascontiguousarray(observed),
                    "group_ids": groups,
                    "horizon_patches": np.array([horizon], dtype=np.int64),
                }
            )
            quantiles = outputs["quantiles"]
            expected = (batch * variates, horizon * ps, len(config.quantile_knots))
            if quantiles.shape != expected or quantiles.dtype != np.float32:
                raise ValueError(
                    f"Timestral session returned {quantiles.shape}/{quantiles.dtype}; expected {expected}/float32. Use the matching Timestral package/config."
                )
            if not np.all(np.isfinite(quantiles)):
                raise ValueError(
                    "Timestral produced nonfinite forecast values; inspect the checkpoint, input magnitudes and execution provider."
                )
            quantiles = _refine_block(quantiles, rows, observed, config)
            blocks.append(quantiles)
            remaining -= horizon
            if remaining:
                feedback = quantiles[..., config.quantile_knots.index(0.5)]
                feedback = np.where(
                    np.isfinite(feedback) & (np.abs(feedback) < 1e30), feedback, 0
                )
                rows = np.concatenate((rows, feedback), axis=-1)[:, -cap:]
                observed = np.concatenate(
                    (observed, np.ones(feedback.shape, dtype=np.bool_)), axis=-1
                )[:, -cap:]
        result = np.concatenate(blocks, axis=1)[:, :prediction_length]
        result = np.clip(result, low[:, None, None], high[:, None, None])
        return result.reshape(batch, variates, prediction_length, -1)

    result = decode(values)
    if symmetric_averaging:
        result = 0.5 * (result - decode(-values)[..., ::-1])
    if clip_nonnegative:
        eligible = mask.any(axis=-1) & ((values >= 0) | ~mask).all(axis=-1)
        result = np.where(eligible[..., None, None], np.maximum(result, 0), result)
    return result
