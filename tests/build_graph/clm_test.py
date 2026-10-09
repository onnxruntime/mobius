# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dedicated L1 coverage for CLM's non-generative ranking contract."""

from __future__ import annotations

import pytest
from _test_configs import CONTRASTIVE_RANKING_CONFIGS, _base_config

from mobius import build_from_module
from mobius._configs import CLMConfig
from mobius._registry import registry
from mobius.tasks import ContrastiveRankingHeadsTask, get_task


@pytest.mark.parametrize("model_type,overrides,is_representative", CONTRASTIVE_RANKING_CONFIGS)
def test_clm_registered_head_graph(model_type, overrides, is_representative):
    config = _base_config(**overrides)
    assert is_representative
    assert model_type == "clm"
    assert registry.get_config_class("clm") is CLMConfig
    assert isinstance(get_task(registry.get("clm").default_task), ContrastiveRankingHeadsTask)
    package = build_from_module(
        registry.get("clm")(config), config, "contrastive-ranking-heads"
    )
    graph = package["model"].graph
    assert [value.name for value in graph.inputs] == ["state_embeddings", "action_embeddings"]
    assert [value.name for value in graph.outputs] == [
        "state_projections",
        "action_projections",
        "logits",
        "probabilities",
    ]
    assert (
        len(
            [
                name
                for name in graph.initializers
                if name.startswith(("state_head.", "action_head."))
            ]
        )
        == 16
    )
    assert not any(node.op_type == "Attention" for node in graph)
