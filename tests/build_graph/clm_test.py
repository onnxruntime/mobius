# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dedicated L1 coverage for CLM's non-generative ranking contract."""

from __future__ import annotations

from mobius import build_from_module
from mobius._configs import CLMConfig
from mobius._registry import registry
from mobius.tasks import ContrastiveRankingHeadsTask, get_task


def test_clm_registered_head_graph():
    config = CLMConfig(hidden_size=8, width=6, depth=3, projection_dim=4)
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
