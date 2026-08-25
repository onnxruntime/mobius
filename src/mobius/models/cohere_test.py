# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import pytest

from mobius._configs import ArchitectureConfig
from mobius.models.cohere import CohereCausalLMModel
from mobius.tasks import CausalLMTask


@pytest.mark.parametrize("no_rope_layers", [[0], [0, 1]])
def test_cohere_short_or_complete_rope_schedule_builds(no_rope_layers: list[int]) -> None:
    config = ArchitectureConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
        max_position_embeddings=32,
        rope_type="default",
        layer_types=["full_attention", "full_attention"],
        no_rope_layers=no_rope_layers,
        tie_word_embeddings=True,
        logit_scale=0.0625,
    )
    package = CausalLMTask().build(CohereCausalLMModel(config), config)
    assert "model" in package
