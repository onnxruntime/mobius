# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from mobius import build_from_module
from mobius.models import GNMHeadModel
from mobius.models.gnm import GNMConfig
from mobius.tasks import MeshGenerationTask


def test_gnm_graph_builds():
    config = GNMConfig(num_vertices=7, identity_dim=2, expression_dim=3)
    package = build_from_module(GNMHeadModel(config), config, MeshGenerationTask())
    assert [value.name for value in package["model"].graph.inputs] == [
        "identity",
        "expression",
        "rotations",
        "translation",
    ]
    assert package["model"].graph.outputs[0].shape[-2:] == (7, 3)
