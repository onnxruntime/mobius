# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mobius
from mobius._inspect import (
    ComponentInfo,
    _get_hf_component_sources,
    _resolve_task_model_type_and_config,
    inspect_components,
)
from mobius._registry import registry
from mobius.tasks import get_task


def _patch_autoconfig(monkeypatch, hf_config):
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *a, **k: hf_config),
    )


def _fake_hf_config(model_type):
    if model_type == "blip-2":
        return SimpleNamespace(
            model_type=model_type,
            text_config=SimpleNamespace(model_type="opt"),
            use_decoder_only_language_model=True,
        )
    return SimpleNamespace(model_type=model_type)


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
    # Runtime paths split the text backbone from the embedding component.
    _patch_autoconfig(monkeypatch, SimpleNamespace(model_type="qwen3_vl"))
    components = {c.name: c for c in inspect_components("fake/qwen-vl")}

    assert components["decoder"].source_paths == (
        "model.language_model.layers",
        "model.language_model.norm",
        "model.language_model.rotary_emb",
        "lm_head",
    )
    assert components["vision_encoder"].source_paths == ("model.visual",)
    assert components["embedding"].source_paths == ("model.language_model.embed_tokens",)


def test_qwen3_tts_embedders_return_shared_hf_source_paths(monkeypatch):
    _patch_autoconfig(monkeypatch, SimpleNamespace(model_type="qwen3_tts"))
    components = {c.name: c for c in inspect_components("fake/qwen3-tts")}

    assert components["talker_step_embedder"].source_paths == (
        "talker.model.codec_embedding",
        "talker.code_predictor.model.codec_embedding",
    )
    assert components["talker_prefill_embedder"].source_paths == (
        "talker.model.text_embedding",
        "talker.text_projection",
        "talker.model.codec_embedding",
    )


def test_qwen2_5_vl_uses_runtime_paths_not_checkpoint_prefixes(monkeypatch):
    # Modern HF Qwen2.5-VL uses the same runtime component roots as Qwen3-VL,
    # despite exposing different checkpoint weight prefixes.
    _patch_autoconfig(monkeypatch, SimpleNamespace(model_type="qwen2_5_vl"))
    components = {c.name: c for c in inspect_components("fake/qwen2.5-vl")}

    assert components["decoder"].source_paths == (
        "model.language_model.layers",
        "model.language_model.norm",
        "model.language_model.rotary_emb",
        "lm_head",
    )
    assert components["vision_encoder"].source_paths == ("model.visual",)
    assert components["embedding"].source_paths == ("model.language_model.embed_tokens",)


@pytest.mark.parametrize("model_type", ["idefics2", "idefics3", "smolvlm"])
def test_shared_llava_class_resolves_idefics_runtime_layout(monkeypatch, model_type):
    _patch_autoconfig(monkeypatch, SimpleNamespace(model_type=model_type))
    components = {c.name: c for c in inspect_components(f"fake/{model_type}")}

    assert components["decoder"].source_paths[0] == "model.text_model.layers"
    assert components["vision_encoder"].source_paths == (
        "model.vision_model",
        "model.connector",
    )
    assert components["embedding"].source_paths == ("model.text_model.embed_tokens",)


def test_shared_llava_class_does_not_guess_unverified_runtime_layout(monkeypatch):
    _patch_autoconfig(monkeypatch, SimpleNamespace(model_type="fuyu"))
    components = inspect_components("fake/fuyu")
    assert all(not component.source_paths for component in components)


@pytest.mark.parametrize(
    ("text_model_type", "decoder_only", "embedding_path"),
    [
        ("opt", True, "language_model.model.decoder.embed_tokens"),
        ("t5", False, "language_model.shared"),
    ],
)
def test_blip2_resolves_text_runtime_layout(
    monkeypatch,
    text_model_type,
    decoder_only,
    embedding_path,
):
    _patch_autoconfig(
        monkeypatch,
        SimpleNamespace(
            model_type="blip-2",
            text_config=SimpleNamespace(model_type=text_model_type),
            use_decoder_only_language_model=decoder_only,
        ),
    )
    components = {c.name: c for c in inspect_components("fake/blip2")}
    assert components["embedding"].source_paths == (embedding_path,)


def test_blip2_does_not_guess_unknown_text_runtime_layout(monkeypatch):
    _patch_autoconfig(
        monkeypatch,
        SimpleNamespace(
            model_type="blip-2",
            text_config=SimpleNamespace(model_type="llama"),
            use_decoder_only_language_model=True,
        ),
    )

    assert all(not component.source_paths for component in inspect_components("fake/blip2"))


def test_internvl_resolves_internlm2_runtime_layout(monkeypatch):
    _patch_autoconfig(
        monkeypatch,
        SimpleNamespace(
            model_type="internvl_chat",
            llm_config=SimpleNamespace(model_type="internlm2"),
            architectures=["InternVLChatModel"],
        ),
    )
    components = {c.name: c for c in inspect_components("fake/internvl")}

    assert components["decoder"].source_paths == (
        "language_model.model.layers",
        "language_model.model.norm",
        "language_model.output",
    )
    assert components["embedding"].source_paths == ("language_model.model.tok_embeddings",)


def test_internvl_does_not_guess_unknown_llm_runtime_layout(monkeypatch):
    _patch_autoconfig(
        monkeypatch,
        SimpleNamespace(
            model_type="internvl",
            llm_config=SimpleNamespace(model_type="qwen3"),
            architectures=[],
        ),
    )
    assert all(not component.source_paths for component in inspect_components("fake/internvl"))


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


def test_explicit_task_overrides_task_but_still_resolves_source_paths(monkeypatch):
    # The explicit task chooses the component layout, while the config still
    # identifies the registered model class that owns its runtime HF paths.
    _patch_autoconfig(monkeypatch, SimpleNamespace(model_type="qwen3_vl"))
    components = inspect_components("fake/qwen-vl", task="vision-language")
    assert {c.name for c in components} == {"decoder", "vision_encoder", "embedding"}
    by_name = {c.name: c for c in components}
    assert by_name["vision_encoder"].source_paths == ("model.visual",)


def test_explicit_task_without_config_returns_roles_without_source_paths(monkeypatch):
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(OSError("no config"))),
    )
    monkeypatch.setattr(
        "mobius.integrations.transformers._config_resolver._try_load_config_json",
        lambda *_a, **_k: None,
    )

    components = inspect_components("anything", task="vision-language")
    assert {c.name for c in components} == {"decoder", "vision_encoder", "embedding"}
    assert all(not component.source_paths for component in components)


def test_qwen3_5_moe_vl_detected_from_vision_config(monkeypatch):
    _patch_autoconfig(
        monkeypatch, SimpleNamespace(model_type="qwen3_5_moe", vision_config=SimpleNamespace())
    )
    names = {c.name for c in inspect_components("fake/qwen-vl")}
    assert "vision_encoder" in names and "decoder" in names


def test_ctc_architecture_uses_mms_registration(monkeypatch):
    hf_config = SimpleNamespace(
        model_type="wav2vec2",
        architectures=["Wav2Vec2ForCTC"],
    )
    _patch_autoconfig(monkeypatch, hf_config)

    task, model_type, resolved_config = _resolve_task_model_type_and_config(
        "fake/wav2vec2-ctc",
        task=None,
        trust_remote_code=False,
    )

    assert (task, model_type, resolved_config) == ("ctc-asr", "mms", hf_config)


def test_unresolvable_config_raises(monkeypatch):
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(OSError("no config"))),
    )
    monkeypatch.setattr(
        "mobius.integrations.transformers._config_resolver._try_load_config_json",
        lambda *_a, **_k: None,
    )
    with pytest.raises(ValueError, match="Could not load a HuggingFace config"):
        inspect_components("fake/diffusers-pipeline")


# Model types whose registered module class provides HF component sources.
_MULTI_COMPONENT_MODEL_TYPES = [
    # Qwen VL family
    "qwen2_5_vl",
    "qwen2_vl",
    "qwen3_vl",
    "qwen3_5_vl",
    "qwen3_5_moe_vl",
    # Other VLMs (3-model split)
    "llava",
    "idefics3",
    "smolvlm",
    "blip-2",
    "internvl",
    "internvl_chat",
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
_EMPTY_SOURCE_PATHS = {("trocr", "encoder")}


@pytest.mark.parametrize("model_type", _MULTI_COMPONENT_MODEL_TYPES)
def test_hf_component_sources_keys_match_task_roles(model_type):
    # Drift guard: the source-path keys provided by the module class must be
    # exactly the package keys the task produces (its model_roles keys), so
    # inspect_components never mislabels or drops a component.
    module_class = registry.get(model_type)
    sources = _get_hf_component_sources(
        module_class,
        model_type,
        _fake_hf_config(model_type),
    )
    registration = registry.get_registration(model_type)
    task_name = registration.task or module_class.default_task
    roles = get_task(task_name).model_roles

    assert set(sources) == set(roles), (
        f"{model_type}: HF_COMPONENT_SOURCES keys {sorted(sources)} "
        f"!= model_roles keys {sorted(roles)}"
    )
    for name, paths in sources.items():
        assert isinstance(paths, tuple)
        if not paths:
            assert (model_type, name) in _EMPTY_SOURCE_PATHS
            continue
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


def test_representative_vlm_source_paths_resolve_on_runtime_models():
    import torch
    import transformers

    text_config = {
        "vocab_size": 128,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "head_dim": 4,
    }
    qwen_vision_config = {
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 4,
        "in_channels": 3,
        "patch_size": 2,
        "spatial_merge_size": 1,
        "temporal_patch_size": 1,
        "window_size": 4,
        "out_hidden_size": 16,
        "fullatt_block_indexes": [0],
    }
    vision_config = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "image_size": 8,
        "patch_size": 2,
    }
    cases = [
        (
            "qwen2_5_vl",
            transformers.Qwen2_5_VLForConditionalGeneration,
            transformers.Qwen2_5_VLConfig(
                text_config=text_config,
                vision_config=qwen_vision_config,
            ),
        ),
        (
            "llava",
            transformers.LlavaForConditionalGeneration,
            transformers.LlavaConfig(
                text_config=text_config,
                vision_config=vision_config,
                image_token_index=127,
            ),
        ),
        (
            "idefics3",
            transformers.Idefics3ForConditionalGeneration,
            transformers.Idefics3Config(
                text_config={**text_config, "pad_token_id": 0},
                vision_config=vision_config,
                image_token_id=127,
                pad_token_id=0,
            ),
        ),
    ]

    for model_type, hf_model_class, hf_config in cases:
        with torch.device("meta"):
            hf_model = hf_model_class(hf_config)
        runtime_paths = {name for name, _ in hf_model.named_modules()}
        sources = _get_hf_component_sources(
            registry.get(model_type),
            model_type,
            hf_config,
        )
        missing = {
            f"{component}.{path}"
            for component, paths in sources.items()
            for path in paths
            if path not in runtime_paths
        }
        assert not missing, f"{model_type}: missing runtime paths {sorted(missing)}"
        for left_name, left_paths in sources.items():
            for right_name, right_paths in sources.items():
                if left_name >= right_name:
                    continue
                overlaps = {
                    (left, right)
                    for left in left_paths
                    for right in right_paths
                    if left == right
                    or left.startswith(f"{right}.")
                    or right.startswith(f"{left}.")
                }
                assert not overlaps, (
                    f"{model_type}: {left_name}/{right_name} paths overlap: {sorted(overlaps)}"
                )


# Representative source_paths per model family, locking in the diverse HF layouts
# (standard llava naming, T5 without an outer ``model.``, Whisper/Bart with it,
# multi-path decoders, and shared encoder for RNNT streaming).
_SOURCE_PATH_EXPECTATIONS = {
    "llava": {
        "decoder": (
            "model.language_model.layers",
            "model.language_model.norm",
            "model.language_model.rotary_emb",
            "lm_head",
        ),
        "vision_encoder": ("model.vision_tower", "model.multi_modal_projector"),
        "embedding": ("model.language_model.embed_tokens",),
    },
    "idefics3": {
        "decoder": (
            "model.text_model.layers",
            "model.text_model.norm",
            "model.text_model.rotary_emb",
            "lm_head",
        ),
        "vision_encoder": ("model.vision_model", "model.connector"),
        "embedding": ("model.text_model.embed_tokens",),
    },
    "gemma3": {
        "decoder": (
            "model.language_model.layers",
            "model.language_model.norm",
            "model.language_model.rotary_emb",
            "lm_head",
        ),
        "vision_encoder": ("model.vision_tower", "model.multi_modal_projector"),
        "embedding": ("model.language_model.embed_tokens",),
    },
    "mllama": {
        "decoder": (
            "model.language_model.layers",
            "model.language_model.norm",
            "model.language_model.rotary_emb",
            "lm_head",
        ),
        "vision_encoder": ("model.vision_model", "model.multi_modal_projector"),
        "embedding": ("model.language_model.embed_tokens",),
    },
    "internvl_chat": {
        "decoder": (
            "language_model.model.layers",
            "language_model.model.norm",
            "language_model.model.rotary_emb",
            "language_model.lm_head",
        ),
        "vision_encoder": ("vision_model", "mlp1"),
        "embedding": ("language_model.model.embed_tokens",),
    },
    "blip-2": {
        "decoder": (
            "language_model.model.decoder.embed_positions",
            "language_model.model.decoder.final_layer_norm",
            "language_model.model.decoder.layers",
            "language_model.lm_head",
        ),
        "vision_encoder": ("vision_model", "qformer", "language_projection"),
        "embedding": ("language_model.model.decoder.embed_tokens",),
    },
    "hunyuan_vl_mot": {
        "decoder": (
            "model.language_model.model.layers",
            "model.language_model.model.norm",
            "model.language_model.lm_head",
        ),
        "vision_encoder": ("model.visual",),
    },
    "t5": {"encoder": ("encoder",), "decoder": ("decoder", "lm_head")},
    "bart": {"encoder": ("model.encoder",), "decoder": ("model.decoder", "lm_head")},
    "trocr": {"encoder": (), "decoder": ("model.decoder", "output_projection")},
    "whisper": {"encoder": ("model.encoder",), "decoder": ("model.decoder", "proj_out")},
    "qwen3_asr": {
        "audio_encoder": ("thinker.audio_tower",),
        "decoder": ("thinker.model.layers", "thinker.model.norm", "thinker.lm_head"),
    },
    "fastconformer_rnnt": {"encoder": ("encoder",), "encoder_streaming": ("encoder",)},
    # gemma4 (towers) vs gemma4_unified (encoder-free embedders) differ in vision/audio.
    "gemma4": {
        "decoder": (
            "model.language_model.layers",
            "model.language_model.norm",
            "model.language_model.rotary_emb",
            "lm_head",
        ),
        "vision_encoder": ("model.vision_tower", "model.embed_vision"),
        "audio_encoder": ("model.audio_tower", "model.embed_audio"),
    },
    "gemma4_unified": {
        "decoder": (
            "model.language_model.layers",
            "model.language_model.norm",
            "model.language_model.rotary_emb",
            "lm_head",
        ),
        "vision_encoder": ("model.vision_embedder", "model.embed_vision"),
        "audio_encoder": ("model.embed_audio",),
    },
}


@pytest.mark.parametrize("model_type,expected", _SOURCE_PATH_EXPECTATIONS.items())
def test_hf_component_sources_values(model_type, expected):
    sources = _get_hf_component_sources(
        registry.get(model_type),
        model_type,
        _fake_hf_config(model_type),
    )
    for key, paths in expected.items():
        assert sources[key] == paths, f"{model_type}.{key}: {sources[key]} != {paths}"
