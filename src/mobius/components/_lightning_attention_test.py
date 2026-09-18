# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import math

import pytest

from mobius.components._lightning_attention import _compute_decay_log


def test_minimax_decay_matches_pinned_64_head_formula():
    actual = _compute_decay_log(layer_idx=7, num_layers=80, num_heads=64)
    factor = 1.0 - 7.0 / 79.0 + 1e-5
    expected = [-(2.0 ** (-(head + 1) / 8.0)) * factor for head in range(64)]

    assert actual == pytest.approx(expected)


def test_minimax_decay_supports_non_power_of_two_head_counts():
    actual = _compute_decay_log(layer_idx=0, num_layers=2, num_heads=6)
    start = 2.0 ** (-(2.0 ** -(math.log2(6) - 3.0)))
    expected = [-(start ** (head + 1)) * 1.00001 for head in range(6)]

    assert actual == pytest.approx(expected)
