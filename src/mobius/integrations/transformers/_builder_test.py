# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the Transformers integration builder."""

from __future__ import annotations

from unittest import mock

import onnx_ir as ir
from onnxscript import nn

from mobius._model_package import ModelPackage
from mobius._testing import make_config
from mobius.integrations import _builder as integration_builder
from mobius.integrations.transformers import _builder as transformers_builder
from mobius.integrations.transformers import _config_resolver


class _DummyModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


def test_legacy_weight_loader_patch_intercepts_transformers_build(monkeypatch) -> None:
    hf_config = type("HFConfig", (), {"model_type": "qwen2"})()
    config = make_config(model_type="qwen2")
    model = ir.Model(ir.Graph([], [], nodes=[], name="model"), ir_version=11)
    package = ModelPackage({"model": model}, config=config)
    download = mock.Mock(return_value={})

    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (hf_config, False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_select_primary_config",
        lambda value: (value, value, "qwen2"),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (_DummyModule, "text-generation", "qwen2"),
    )
    monkeypatch.setattr(_config_resolver, "_config_from_hf", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        integration_builder, "build_from_module", lambda *args, **kwargs: package
    )
    monkeypatch.setattr(integration_builder, "_download_weights", download)

    result = transformers_builder.build_transformers_model("fake/model")

    assert result is package
    download.assert_called_once_with("fake/model", revision=None)
