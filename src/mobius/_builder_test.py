# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for _builder.py — model building API."""

from __future__ import annotations

import torch

from mobius._builder import _strip_weight_namespace


class TestStripWeightNamespace:
    """Tests for _strip_weight_namespace."""

    def test_filters_and_strips_prefix(self):
        state_dict = {
            "language_model.model.layers.0.weight": torch.tensor([1.0]),
            "language_model.lm_head.weight": torch.tensor([2.0]),
            "visual.encoder.layers.0.weight": torch.tensor([3.0]),
        }
        result = _strip_weight_namespace(state_dict, "language_model")
        assert set(result.keys()) == {
            "model.layers.0.weight",
            "lm_head.weight",
        }
        torch.testing.assert_close(result["model.layers.0.weight"], torch.tensor([1.0]))
        torch.testing.assert_close(result["lm_head.weight"], torch.tensor([2.0]))

    def test_namespace_with_trailing_dot(self):
        state_dict = {
            "language_model.model.weight": torch.tensor([1.0]),
        }
        result = _strip_weight_namespace(state_dict, "language_model.")
        assert set(result.keys()) == {"model.weight"}

    def test_empty_state_dict(self):
        result = _strip_weight_namespace({}, "language_model")
        assert result == {}

    def test_no_matching_keys(self):
        state_dict = {
            "visual.encoder.weight": torch.tensor([1.0]),
            "audio.encoder.weight": torch.tensor([2.0]),
        }
        result = _strip_weight_namespace(state_dict, "language_model")
        assert result == {}

    def test_does_not_match_partial_prefix(self):
        """Ensure 'language_model_v2.x' is not matched by 'language_model'."""
        state_dict = {
            "language_model_v2.weight": torch.tensor([1.0]),
            "language_model.weight": torch.tensor([2.0]),
        }
        result = _strip_weight_namespace(state_dict, "language_model")
        assert set(result.keys()) == {"weight"}
        torch.testing.assert_close(result["weight"], torch.tensor([2.0]))
