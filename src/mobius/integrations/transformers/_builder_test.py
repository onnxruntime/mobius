# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the Transformers integration builder."""

from __future__ import annotations

from unittest import mock

import onnx_ir as ir
from onnxscript import nn

from mobius._model_package import ModelPackage
from mobius._testing import make_config
from mobius.integrations.diffusers import _builder as diffusers_builder
from mobius.integrations.transformers import _builder as transformers_builder
from mobius.integrations.transformers import _config_resolver


class _DummyModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


def test_transformers_build_uses_canonical_weight_loader(monkeypatch) -> None:
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
        transformers_builder, "build_from_module", lambda *args, **kwargs: package
    )
    monkeypatch.setattr(transformers_builder, "_download_weights", download)

    result = transformers_builder.build_transformers_model("fake/model")

    assert result is package
    download.assert_called_once_with("fake/model", revision=None)


def test_build_threads_revision_to_diffusers_fallback(monkeypatch) -> None:
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("not transformers")),
    )
    monkeypatch.setattr(
        _config_resolver, "_try_load_config_json", lambda *args, **kwargs: None
    )
    expected = ModelPackage({})
    calls: list[tuple[tuple, dict]] = []

    def fake_build_diffusers(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(
        diffusers_builder,
        "build_diffusers_pipeline",
        fake_build_diffusers,
    )

    result = transformers_builder.build_transformers_model(
        "fake/diffusers",
        revision="pinned-revision",
        load_weights=False,
    )

    assert result is expected
    assert calls == [
        (
            ("fake/diffusers",),
            {
                "revision": "pinned-revision",
                "dtype": None,
                "load_weights": False,
                "execution_provider": "default",
            },
        )
    ]
