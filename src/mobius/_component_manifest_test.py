# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for canonical component manifest resolution."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from mobius._component_manifest import (
    ComponentDescriptor,
    ComponentManifest,
    get_hf_component_sources,
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
    HF_COMPONENT_MODULE_ALIASES: ClassVar[dict[str, dict[str, str]]] = {
        "vision_encoder": {
            "encoder": "model.vision_tower",
            "projector": "model.projector",
        }
    }


@pytest.mark.parametrize("model_type", ["test", None, ""])
def test_manifest_combines_task_and_model_metadata(model_type):
    manifest = resolve_component_manifest(
        _Task(),
        module_class=_Model,
        model_type=model_type,
        hf_config=object(),
    )

    assert manifest.names == ("decoder", "vision_encoder", "embedding")
    assert manifest["decoder"] == ComponentDescriptor(
        name="decoder",
        module_attribute_path="language",
        role="decoder",
        source_paths=("model.language_model.layers", "lm_head"),
    )
    assert manifest["vision_encoder"].module_attribute_path == "vision.tower"
    assert manifest["vision_encoder"].role == "encoder"
    assert manifest["vision_encoder"].source_module_names("encoder.layers.0.q_proj") == (
        "encoder.layers.0.q_proj",
        "model.vision_tower.layers.0.q_proj",
    )


@pytest.mark.parametrize(
    ("model_type", "hf_model_type"),
    [("dynamic", None), ("dynamic", "other"), (None, "dynamic"), ("", "dynamic")],
)
def test_dynamic_source_resolver_is_authoritative(model_type, hf_model_type):
    config = SimpleNamespace(model_type=hf_model_type)

    class _DynamicModel(_Model):
        @classmethod
        def get_hf_component_sources(cls, *, model_type, hf_config):
            assert model_type == "dynamic"
            assert hf_config is config
            return {"decoder": ("resolved.decoder",)}

    manifest = resolve_component_manifest(
        _Task(),
        module_class=_DynamicModel,
        model_type=model_type,
        hf_config=config,
    )

    assert manifest["decoder"].source_paths == ("resolved.decoder",)
    assert manifest["vision_encoder"].source_paths == ()
    assert get_hf_component_sources(_DynamicModel, model_type, config) == {
        "decoder": ("resolved.decoder",)
    }


@pytest.mark.parametrize("model_type", [None, ""])
@pytest.mark.parametrize(
    "hf_config", [object(), SimpleNamespace(model_type=None), SimpleNamespace(model_type="")]
)
def test_dynamic_source_resolver_is_skipped_without_model_type(model_type, hf_config):
    class _DynamicModel(_Model):
        @classmethod
        def get_hf_component_sources(cls, *, model_type, hf_config):
            pytest.fail("Dynamic source resolution requires a known model type")

    manifest = resolve_component_manifest(
        _Task(),
        module_class=_DynamicModel,
        model_type=model_type,
        hf_config=hf_config,
    )

    assert all(not component.source_paths for component in manifest.values())
    assert get_hf_component_sources(_DynamicModel, model_type, hf_config) == {}


def test_config_based_alias_resolver_does_not_require_model_type():
    config = object()

    class _DynamicAliasModel(_Model):
        @classmethod
        def get_hf_component_module_aliases(cls, *, hf_config):
            assert hf_config is config
            return {"decoder": {"blocks": "model.layers"}}

    manifest = resolve_component_manifest(
        _Task(),
        module_class=_DynamicAliasModel,
        hf_config=config,
    )

    assert manifest["decoder"].source_module_names("blocks.0.q_proj") == (
        "blocks.0.q_proj",
        "model.layers.0.q_proj",
    )
    assert manifest["vision_encoder"].source_path_aliases == ()


def test_descriptor_maps_local_path_to_huggingface_source_name():
    descriptor = ComponentDescriptor(
        name="decoder",
        module_attribute_path="decoder",
        role="decoder",
        source_paths=("model.language_model.layers", "lm_head"),
    )

    assert descriptor.source_module_names("model.layers.0.per_layer_input_gate") == (
        "model.layers.0.per_layer_input_gate",
        "model.language_model.layers.0.per_layer_input_gate",
    )
    assert descriptor.source_module_names("lm_head") == ("lm_head",)


def test_component_root_source_prefixes_local_descendants():
    descriptor = ComponentDescriptor(
        name="decoder",
        module_attribute_path="decoder",
        role="decoder",
        source_paths=("model.decoder", "lm_head"),
    )

    assert descriptor.source_module_names("block.0.self_attn.q_proj") == (
        "block.0.self_attn.q_proj",
        "model.decoder.block.0.self_attn.q_proj",
    )
    assert descriptor.source_module_names("lm_head") == ("lm_head",)
    assert descriptor.source_module_names("lm_head.adapter") == ("lm_head.adapter",)


def test_explicit_alias_takes_precedence_over_component_root_synthesis():
    descriptor = ComponentDescriptor(
        name="decoder",
        module_attribute_path="decoder",
        role="decoder",
        source_paths=("decoder", "lm_head"),
        source_path_aliases=(("output", "lm_head"),),
    )

    assert descriptor.source_module_names("output") == ("output", "lm_head")


@pytest.mark.parametrize("model_type", ["t5", None, ""])
def test_t5_aliases_follow_encoder_and_decoder_layer_counts(model_type):
    from mobius._configs import ArchitectureConfig
    from mobius.models.t5 import T5ForConditionalGeneration
    from mobius.tasks import get_task

    config = ArchitectureConfig(num_hidden_layers=1, num_decoder_layers=3)
    manifest = resolve_component_manifest(
        get_task("seq2seq"),
        module_class=T5ForConditionalGeneration,
        model_type=model_type,
        hf_config=config,
    )

    assert "encoder.block.0.layer.0.SelfAttention.q" in manifest[
        "encoder"
    ].source_module_names("block.0.self_attn.q_proj")
    assert "decoder.block.2.layer.1.EncDecAttention.k" in manifest[
        "decoder"
    ].source_module_names("block.2.cross_attn.k_proj")


def test_single_component_uses_root_module_path():
    class _SingleTask:
        model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}
        components = None

    manifest = resolve_component_manifest(_SingleTask())

    assert manifest["model"].module_attribute_path == ""
    assert manifest["model"].role == "encoder"


def test_duplicate_component_names_are_rejected():
    component = ComponentDescriptor("decoder", "decoder", "decoder")

    with pytest.raises(ValueError, match="more than once"):
        ComponentManifest((component, component))
