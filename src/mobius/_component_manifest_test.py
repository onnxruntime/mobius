# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for canonical component manifest resolution."""

from __future__ import annotations

from typing import ClassVar

import pytest

from mobius._component_manifest import (
    ComponentDescriptor,
    ComponentManifest,
    resolve_component_manifest,
)
from mobius.tasks import ComponentSpec


class _Task:
    model_roles: ClassVar[dict[str, str]] = {
        "decoder": "decoder",
        "vision_encoder": "encoder",
        "embedding": "embedding",
    }
    components = ComponentSpec(
        decoder="language",
        vision_encoder="vision.tower",
        embedding="embedding",
    )


class _Model:
    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "decoder": ("model.language_model.layers", "lm_head"),
        "vision_encoder": ("model.vision_tower", "model.projector"),
        "embedding": ("model.language_model.embed_tokens",),
    }


def test_manifest_combines_task_and_model_metadata():
    manifest = resolve_component_manifest(
        _Task(),
        module_class=_Model,
        model_type="test",
        hf_config=object(),
    )

    assert manifest.names == ("decoder", "vision_encoder", "embedding")
    assert manifest["decoder"] == ComponentDescriptor(
        name="decoder",
        module_path="language",
        role="decoder",
        source_paths=("model.language_model.layers", "lm_head"),
    )
    assert manifest["vision_encoder"].module_path == "vision.tower"
    assert manifest["vision_encoder"].role == "encoder"


def test_dynamic_source_resolver_is_authoritative():
    class _DynamicModel:
        @classmethod
        def get_hf_component_sources(cls, *, model_type, hf_config):
            assert model_type == "dynamic"
            assert hf_config == "config"
            return {"decoder": ("resolved.decoder",)}

    manifest = resolve_component_manifest(
        _Task(),
        module_class=_DynamicModel,
        model_type="dynamic",
        hf_config="config",
    )

    assert manifest["decoder"].source_paths == ("resolved.decoder",)
    assert manifest["vision_encoder"].source_paths == ()


def test_single_component_uses_root_module_path():
    class _SingleTask:
        model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}
        components = None

    manifest = resolve_component_manifest(_SingleTask())

    assert manifest["model"].module_path == ""
    assert manifest["model"].role == "encoder"


def test_duplicate_component_names_are_rejected():
    component = ComponentDescriptor("decoder", "decoder", "decoder")

    with pytest.raises(ValueError, match="more than once"):
        ComponentManifest((component, component))
