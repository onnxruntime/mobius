# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the NeMo model builder."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock


def test_build_uses_transformers_config_resolver_for_default_task(monkeypatch) -> None:
    import mobius._builder as core_builder
    import mobius.integrations.nemo._config_mapping as config_mapping
    import mobius.integrations.nemo._reader as nemo_reader
    import mobius.integrations.transformers._config_resolver as config_resolver
    from mobius._registry import registry
    from mobius.integrations.nemo._builder import build_from_nemo

    config = SimpleNamespace(model_type="llama")
    weights = {"model.weight": object()}
    archive = SimpleNamespace(
        path="model.nemo",
        target="example.Model",
        config={},
        state_dict=lambda: weights,
    )
    module = object()
    module_class = mock.Mock(return_value=module)
    package = mock.MagicMock()
    resolve_task = mock.Mock(return_value="text-generation")
    build_module = mock.Mock(return_value=package)

    monkeypatch.setattr(nemo_reader, "NeMoArchive", mock.Mock(return_value=archive))
    monkeypatch.setattr(config_mapping, "nemo_to_config", mock.Mock(return_value=config))
    monkeypatch.setattr(registry, "get", mock.Mock(return_value=module_class))
    monkeypatch.setattr(config_resolver, "_default_task_for_model", resolve_task)
    monkeypatch.setattr(core_builder, "build_from_module", build_module)

    result = build_from_nemo("model.nemo", revision="immutable-revision")

    assert result is package
    resolve_task.assert_called_once_with("llama")
    build_module.assert_called_once_with(
        module,
        config,
        "text-generation",
        execution_provider="default",
    )
    package.apply_weights.assert_called_once_with(weights, prefix_map=None)
