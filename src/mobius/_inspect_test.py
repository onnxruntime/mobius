# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mobius
from mobius._inspect import ComponentInfo, inspect_components
from mobius._registry import registry
from mobius.tasks import get_task


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
    assert components == [ComponentInfo(name="model", role="decoder")]


def test_vlm_returns_decoder_vision_embedding(monkeypatch):
    _patch_autoconfig(monkeypatch, SimpleNamespace(model_type="llava"))
    components = inspect_components("fake/llava")
    assert [(c.name, c.role) for c in components] == [
        ("decoder", "decoder"),
        ("vision_encoder", "encoder"),
        ("embedding", "embedding"),
    ]


def test_qwen3_vl_returns_hf_source_paths(monkeypatch):
    # Qwen3-VL nests everything under ``model.``.
    _patch_autoconfig(monkeypatch, SimpleNamespace(model_type="qwen3_vl"))
    components = {c.name: c for c in inspect_components("fake/qwen-vl")}

    assert components["decoder"].source_paths == ("model.language_model",)
    assert components["vision_encoder"].source_paths == ("model.visual",)
    assert components["embedding"].source_paths == ("model.language_model.embed_tokens",)


def test_qwen2_5_vl_source_paths_differ_from_qwen3(monkeypatch):
    # Qwen2.5-VL uses a different HF layout: vision at top-level ``visual.*``,
    # text backbone under ``model.*``, ``lm_head`` at top level. This must NOT
    # be reported with the Qwen3 ``model.language_model`` paths.
    _patch_autoconfig(monkeypatch, SimpleNamespace(model_type="qwen2_5_vl"))
    components = {c.name: c for c in inspect_components("fake/qwen2.5-vl")}

    assert components["decoder"].source_paths == ("model", "lm_head")
    assert components["vision_encoder"].source_paths == ("visual",)
    assert components["embedding"].source_paths == ("model.embed_tokens",)


def test_phi4mm_decoder_maps_to_multiple_source_paths(monkeypatch):
    # A single component can span multiple disjoint HF sub-trees.
    _patch_autoconfig(monkeypatch, SimpleNamespace(model_type="phi4mm"))
    components = {c.name: c for c in inspect_components("fake/phi4mm")}

    assert components["decoder"].source_paths == ("model.layers", "model.norm", "lm_head")
    assert components["audio_encoder"].source_paths == (
        "model.embed_tokens_extend.audio_embed",
    )


def test_single_component_llm_has_empty_source_paths(monkeypatch):
    _patch_autoconfig(
        monkeypatch, SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])
    )
    (component,) = inspect_components("fake/llama")
    assert component.source_paths == ()


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


# Model types whose registered module class declares HF_COMPONENT_SOURCES.
_MULTI_COMPONENT_MODEL_TYPES = [
    # Qwen VL family
    "qwen2_5_vl",
    "qwen2_vl",
    "qwen3_vl",
    "qwen3_5_vl",
    "qwen3_5_moe_vl",
    # Other VLMs (3-model split)
    "llava",
    "blip-2",
    "internvl",
    "gemma3",
    "mllama",
    "hunyuan_vl_mot",
    # Multimodal 4-model split
    "phi4mm",
    "gemma4",
    "gemma4_unified",
    # Encoder-decoder (seq2seq / speech-to-text)
    "bart",
    "t5",
    "trocr",
    "whisper",
    # Speech-language / audio
    "qwen3_asr",
    "fun_asr",
    "qwen3_tts_tokenizer_12hz",
    "qwen3_tts",
    "fastconformer_rnnt",
]


@pytest.mark.parametrize("model_type", _MULTI_COMPONENT_MODEL_TYPES)
def test_hf_component_sources_keys_match_task_roles(model_type):
    # Drift guard: the source-path keys declared on the module class must be
    # exactly the package keys the task produces (its model_roles keys), so
    # inspect_components never mislabels or drops a component.
    module_class = registry.get(model_type)
    sources = module_class.HF_COMPONENT_SOURCES
    registration = registry.get_registration(model_type)
    task_name = registration.task or module_class.default_task
    roles = get_task(task_name).model_roles

    assert set(sources) == set(roles), (
        f"{model_type}: HF_COMPONENT_SOURCES keys {sorted(sources)} "
        f"!= model_roles keys {sorted(roles)}"
    )
    for name, paths in sources.items():
        assert isinstance(paths, tuple) and paths, (
            f"{model_type}.{name} must be a non-empty tuple"
        )
        assert all(isinstance(p, str) and p for p in paths)


def test_inspect_does_not_instantiate_module_class(monkeypatch):
    # inspect_components must read HF_COMPONENT_SOURCES off the class only —
    # never construct the (expensive) module. Make __init__ blow up to prove it.
    module_class = registry.get("qwen3_vl")

    def _boom(*_a, **_k):
        raise AssertionError("inspect_components must not instantiate the module class")

    monkeypatch.setattr(module_class, "__init__", _boom)
    _patch_autoconfig(monkeypatch, SimpleNamespace(model_type="qwen3_vl"))

    components = {c.name: c for c in inspect_components("fake/qwen-vl")}
    assert components["vision_encoder"].source_paths == ("model.visual",)


# Representative source_paths per model family, locking in the diverse HF layouts
# (standard llava naming, T5 without an outer ``model.``, Whisper/Bart with it,
# multi-path decoders, and shared encoder for RNNT streaming).
_SOURCE_PATH_EXPECTATIONS = {
    "llava": {
        "decoder": ("language_model",),
        "vision_encoder": ("vision_tower", "multi_modal_projector"),
        "embedding": ("language_model.model.embed_tokens",),
    },
    "internvl": {"vision_encoder": ("vision_model", "mlp1")},
    "blip-2": {"vision_encoder": ("vision_model", "qformer", "language_projection")},
    "hunyuan_vl_mot": {
        "decoder": ("model.language_model",),
        "vision_encoder": ("model.visual",),
    },
    "t5": {"encoder": ("encoder",), "decoder": ("decoder", "lm_head")},
    "bart": {"encoder": ("model.encoder",), "decoder": ("model.decoder", "lm_head")},
    "whisper": {"encoder": ("model.encoder",), "decoder": ("model.decoder",)},
    "qwen3_asr": {
        "audio_encoder": ("thinker.audio_tower",),
        "decoder": ("thinker.model.layers", "thinker.model.norm", "thinker.lm_head"),
    },
    "fastconformer_rnnt": {"encoder": ("encoder",), "encoder_streaming": ("encoder",)},
    # gemma4 (towers) vs gemma4_unified (encoder-free embedders) differ in vision/audio.
    "gemma4": {
        "vision_encoder": ("model.vision_tower", "model.embed_vision"),
        "audio_encoder": ("model.audio_tower", "model.embed_audio"),
    },
    "gemma4_unified": {
        "vision_encoder": ("model.vision_embedder", "model.embed_vision"),
        "audio_encoder": ("model.embed_audio",),
    },
}


@pytest.mark.parametrize("model_type,expected", _SOURCE_PATH_EXPECTATIONS.items())
def test_hf_component_sources_values(model_type, expected):
    sources = registry.get(model_type).HF_COMPONENT_SOURCES
    for key, paths in expected.items():
        assert sources[key] == paths, f"{model_type}.{key}: {sources[key]} != {paths}"
