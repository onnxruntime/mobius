# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Speech sampler checks for the LFM3 pipeline, against Transformers' logits processors."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
)

from mobius.integrations.lfm3_audio import _penalize, _sample


def test_repetition_penalty_matches_transformers():
    scores = np.random.default_rng(0).normal(size=17).astype(np.float32)
    history = np.array([3, 3, 5, 16, 0], dtype=np.int64)
    expected = RepetitionPenaltyLogitsProcessor(1.2)(
        torch.from_numpy(history)[None], torch.from_numpy(scores)[None]
    )[0].numpy()
    np.testing.assert_array_equal(_penalize(scores, history, 1.2), expected)
    assert _penalize(scores, history[:0], 1.2) is scores


@pytest.mark.parametrize("top_k", [5, None])
def test_sampler_draws_from_the_warped_distribution(top_k):
    scores = np.random.default_rng(1).normal(size=17).astype(np.float32)
    warped = TemperatureLogitsWarper(0.4)(None, torch.from_numpy(scores)[None])
    if top_k:
        warped = TopKLogitsWarper(top_k)(None, warped)
    probs = warped.softmax(-1)[0].numpy()
    rng = np.random.default_rng(0)
    counts = np.bincount(
        [_sample(scores, rng, 0.4, top_k) for _ in range(20_000)], minlength=17
    )
    assert set(np.flatnonzero(counts)) <= set(np.flatnonzero(probs))
    np.testing.assert_allclose(counts / counts.sum(), probs, atol=0.01)
