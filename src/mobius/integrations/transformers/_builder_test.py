# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the Transformers integration builder."""

from __future__ import annotations

from unittest import mock

import onnx_ir as ir
import pytest
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


def test_qwen4_composite_architecture_requires_text_only(monkeypatch) -> None:
    hf_config = type(
        "HFConfig",
        (),
        {
            "model_type": "renamed_qwen4",
            "architectures": ["Qwen4ExpForConditionalGeneration"],
        },
    )()
    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (hf_config, False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_select_primary_config",
        lambda value: (value, value, "qwen4_exp_text"),
    )

    with pytest.raises(ValueError, match="Pass text_only=True"):
        transformers_builder.build_transformers_model(
            "fake/qwen4",
            load_weights=False,
        )


def test_glm_full_attention_overrides_use_dsa_for_glm_moe_dsa(monkeypatch) -> None:
    """``--glm-full-attention`` forces ``config.use_dsa=False`` for GLM-5.2."""
    hf_config = type("HFConfig", (), {"model_type": "glm_moe_dsa"})()
    config = make_config(model_type="glm_moe_dsa", use_dsa=True)
    model = ir.Model(ir.Graph([], [], nodes=[], name="model"), ir_version=11)
    package = ModelPackage({"model": model}, config=config)
    captured_configs: list = []

    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (hf_config, False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_select_primary_config",
        lambda value: (value, value, "glm_moe_dsa"),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (_DummyModule, "text-generation", "glm_moe_dsa"),
    )
    monkeypatch.setattr(_config_resolver, "_config_from_hf", lambda *args, **kwargs: config)

    def fake_build_from_module(_module, built_config, *args, **kwargs):
        captured_configs.append(built_config)
        return package

    monkeypatch.setattr(transformers_builder, "build_from_module", fake_build_from_module)
    monkeypatch.setattr(transformers_builder, "_download_weights", mock.Mock(return_value={}))

    result = transformers_builder.build_transformers_model(
        "zai-org/GLM-5.2", glm_full_attention=True
    )

    assert result is package
    assert captured_configs[0].use_dsa is False


def test_glm_full_attention_rejects_non_glm_model_type(monkeypatch) -> None:
    """``--glm-full-attention`` is only meaningful for ``glm_moe_dsa``."""
    hf_config = type("HFConfig", (), {"model_type": "qwen2"})()
    config = make_config(model_type="qwen2")

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

    with pytest.raises(ValueError, match="glm_full_attention=True is not supported"):
        transformers_builder.build_transformers_model("fake/model", glm_full_attention=True)


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


def test_glm_full_attention_rejects_diffusers_dispatch(monkeypatch) -> None:
    """``--glm-full-attention`` must raise on the diffusers-dispatch branch.

    Mirrors the existing ``text_only`` guard on the same early-return path:
    a repo that doesn't resolve to a registered ``model_type`` (and so falls
    through to the Diffusers integration) can never be GLM-5.2, so silently
    ignoring the flag there would swallow a real user error.
    """
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("not transformers")),
    )
    monkeypatch.setattr(
        _config_resolver, "_try_load_config_json", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        diffusers_builder,
        "build_diffusers_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should not reach build_diffusers_pipeline")
        ),
    )

    with pytest.raises(ValueError, match="glm_full_attention=True is not supported"):
        transformers_builder.build_transformers_model(
            "fake/diffusers", glm_full_attention=True
        )
