# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mobius
from mobius._inspect import ComponentInfo, inspect_components


def _patch_autoconfig(monkeypatch, hf_config):
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *a, **k: hf_config),
    )


def test_inspect_components_exported_from_package():
    assert mobius.inspect_components is inspect_components
    assert mobius.ComponentInfo is ComponentInfo


def test_single_component_llm_returns_one_component(monkeypatch):
    _patch_autoconfig(
        monkeypatch, SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])
    )
    components = inspect_components("fake/llama")
    assert components == [ComponentInfo(name="model", kind="decoder")]


def test_vlm_returns_decoder_vision_embedding(monkeypatch):
    _patch_autoconfig(monkeypatch, SimpleNamespace(model_type="llava"))
    components = inspect_components("fake/llava")
    assert [(c.name, c.kind) for c in components] == [
        ("decoder", "decoder"),
        ("vision_encoder", "encoder"),
        ("embedding", "embedding"),
    ]


def test_encoder_decoder_returns_two_components(monkeypatch):
    _patch_autoconfig(monkeypatch, SimpleNamespace(model_type="t5"))
    names = {c.name for c in inspect_components("fake/t5")}
    assert names == {"encoder", "decoder"}


def test_explicit_task_overrides_model_type(monkeypatch):
    # AutoConfig must not even be consulted when a task is given.
    def _boom(*_a, **_k):
        raise AssertionError("AutoConfig should not be called when task is explicit")

    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(_boom))
    components = inspect_components("anything", task="vision-language")
    assert {c.name for c in components} == {"decoder", "vision_encoder", "embedding"}


def test_qwen3_5_moe_vl_detected_from_vision_config(monkeypatch):
    _patch_autoconfig(
        monkeypatch, SimpleNamespace(model_type="qwen3_5_moe", vision_config=SimpleNamespace())
    )
    names = {c.name for c in inspect_components("fake/qwen-vl")}
    assert "vision_encoder" in names and "decoder" in names


def test_unresolvable_config_raises(monkeypatch):
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(OSError("no config"))),
    )
    monkeypatch.setattr(
        "mobius._config_resolver._try_load_config_json", lambda *_a, **_k: None
    )
    with pytest.raises(ValueError, match="Could not load a HuggingFace config"):
        inspect_components("fake/diffusers-pipeline")
