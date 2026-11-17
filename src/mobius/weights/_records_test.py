# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for typed logical checkpoint records."""

from __future__ import annotations

import pytest
import torch

from mobius._component_manifest import ComponentDescriptor
from mobius.weights import FloatWeight, WeightBundle, WeightRecord


def _component() -> ComponentDescriptor:
    return ComponentDescriptor(
        name="decoder",
        module_attribute_path="decoder",
        role="decoder",
        source_paths=("model.layers",),
    )


def test_bundle_tracks_source_keys():
    record = WeightRecord(
        name="model.norm.weight",
        component="decoder",
        storage=FloatWeight(
            value=torch.ones(4),
            source_key="model.norm.weight",
        ),
    )
    bundle = WeightBundle(_component(), {record.name: record})

    assert bundle.source_keys == frozenset({"model.norm.weight"})
    assert bundle["model.norm.weight"] is record


def test_bundle_rejects_wrong_component():
    record = WeightRecord(
        name="model.norm.weight",
        component="vision_encoder",
        storage=FloatWeight(torch.ones(4), "model.norm.weight"),
    )

    with pytest.raises(ValueError, match="belongs to component"):
        WeightBundle(_component(), {record.name: record})
