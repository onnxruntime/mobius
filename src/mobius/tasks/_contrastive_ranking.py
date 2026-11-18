# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Projection-head-only CLM ranking task."""

from __future__ import annotations

from typing import ClassVar

from onnxscript import nn

from mobius._configs.clm import CLMConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class ContrastiveRankingHeadsTask(ModelTask):
    """Rank a shared candidate set from normalized encoder embeddings."""

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}
    output_names = ("state_projections", "action_projections", "logits", "probabilities")

    def build(self, module: nn.Module, config: CLMConfig) -> ModelPackage:
        config.validate()
        graph, builder = _make_graph("clm_ranking_heads")
        states = builder.input(
            "state_embeddings",
            dtype=config.dtype,
            shape=["states", config.hidden_size],
        )
        actions = builder.input(
            "action_embeddings",
            dtype=config.dtype,
            shape=["candidates", config.hidden_size],
        )
        for value, name in zip(
            module(builder.op, states, actions), self.output_names, strict=True
        ):
            builder.add_output(value, name)
        return ModelPackage({"model": _make_model(graph)}, config=config)
