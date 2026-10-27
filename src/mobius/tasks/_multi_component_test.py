# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for generic non-generative multi-component tasks."""

from __future__ import annotations

from types import SimpleNamespace

import onnx_ir as ir
import pytest

from mobius._builder import build_from_module
from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks import (
    ComponentConfig,
    ComponentRole,
    ComponentSpec,
    MultiComponentModelTask,
)
from mobius.tasks._base import _make_graph, _make_model


class _BackboneAndHeadsTask(MultiComponentModelTask):
    components = ComponentSpec(
        feature_extractor=ComponentConfig(
            "encoder",
            ComponentRole.ENCODER,
        ),
        category_scores=ComponentConfig(
            "heads.category",
            ComponentRole.HEAD,
        ),
        quality_score=ComponentConfig(
            "heads.quality",
            ComponentRole.HEAD,
        ),
    )

    def __init__(self) -> None:
        self.built_modules: dict[str, object] = {}

    def build_component(self, name, component, module, config):
        self.built_modules[name] = module
        graph, builder = _make_graph(name=name)
        value = builder.input("input", ir.DataType.FLOAT, [1])
        builder.add_output(builder.op.Identity(value), "output")
        return _make_model(graph)


def _module():
    return SimpleNamespace(
        encoder=object(),
        heads=SimpleNamespace(category=object(), quality=object()),
    )


def test_manifest_resolves_names_paths_and_neutral_roles():
    manifest = _BackboneAndHeadsTask().component_manifest()

    assert _BackboneAndHeadsTask.model_roles == {
        "feature_extractor": "encoder",
        "category_scores": "head",
        "quality_score": "head",
    }
    assert manifest.names == (
        "feature_extractor",
        "category_scores",
        "quality_score",
    )
    assert manifest["feature_extractor"].module_attribute_path == "encoder"
    assert manifest["feature_extractor"].role == "encoder"
    assert manifest["category_scores"].module_attribute_path == "heads.category"
    assert manifest["category_scores"].role == "head"
    assert manifest["quality_score"].role == "head"


def test_build_packages_backbone_and_all_heads_together():
    module = _module()
    task = _BackboneAndHeadsTask()

    package = task.build(module, BaseModelConfig())

    assert isinstance(package, ModelPackage)
    assert tuple(package) == (
        "feature_extractor",
        "category_scores",
        "quality_score",
    )
    assert task.built_modules == {
        "feature_extractor": module.encoder,
        "category_scores": module.heads.category,
        "quality_score": module.heads.quality,
    }
    assert all(model.graph.name == name for name, model in package.items())


def test_build_optimization_uses_declared_roles(monkeypatch):
    optimized_roles = {}

    def record_role(model, **kwargs):
        optimized_roles[model.graph.name] = kwargs["model_role"]

    monkeypatch.setattr("mobius._builder.optimize_model", record_role)

    build_from_module(
        _module(),
        BaseModelConfig(),
        task=_BackboneAndHeadsTask(),
    )

    assert optimized_roles == {
        "feature_extractor": "encoder",
        "category_scores": "head",
        "quality_score": "head",
    }
    assert "decoder" not in optimized_roles.values()


@pytest.mark.parametrize("path", [None, 3])
def test_component_config_rejects_non_string_paths(path):
    with pytest.raises(TypeError, match="must be a string"):
        ComponentConfig(path, ComponentRole.HEAD)


@pytest.mark.parametrize("path", ["", ".", ".head", "heads.", "heads..classifier", " "])
def test_component_config_rejects_empty_attribute_path_segments(path):
    with pytest.raises(ValueError, match="non-empty dotted attribute path"):
        ComponentConfig(path, ComponentRole.HEAD)


@pytest.mark.parametrize("role", ["decoder", "", "classifier", 3])
def test_component_config_rejects_unsupported_roles(role):
    error = TypeError if role == 3 else ValueError
    with pytest.raises(error, match="component role"):
        ComponentConfig("head", role)
