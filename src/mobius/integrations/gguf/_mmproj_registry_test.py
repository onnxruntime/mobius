# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Closure tests for the pinned clip projector capability registry."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import pytest

from mobius.integrations.gguf._mmproj_registry import (
    CLIP_METADATA_SCHEMA,
    LLAMA_CPP_MMPROJ_SHA,
    MMPROJ_ARTIFACT_PINS,
    MMProjModality,
    get_projector_spec,
    iter_projector_specs,
    projector_type_for_modality,
    supported_projector_types,
)
from mobius.integrations.gguf._spec import Support

_PINNED_PROJECTOR_STRINGS = (
    "mlp",
    "ldp",
    "ldpv2",
    "resampler",
    "adapter",
    "qwen2vl_merger",
    "qwen2.5vl_merger",
    "qwen3vl_merger",
    "step3vl",
    "gemma3",
    "gemma3nv",
    "gemma3na",
    "gemma4v",
    "gemma4a",
    "gemma4uv",
    "gemma4ua",
    "phi4",
    "idefics3",
    "pixtral",
    "ultravox",
    "internvl",
    "llama4",
    "qwen2a",
    "qwen3a",
    "glma",
    "qwen2.5o",
    "voxtral",
    "meralion",
    "musicflamingo",
    "lfm2",
    "kimivl",
    "paddleocr",
    "lightonocr",
    "cogvlm",
    "janus_pro",
    "dots_ocr",
    "dots3note_v",
    "dots3note_a",
    "deepseekocr",
    "deepseekocr2",
    "lfm2a",
    "glm4v",
    "youtuvl",
    "yasa2",
    "kimik25",
    "nemotron_v2_vl",
    "exaone4_5",
    "hunyuanvl",
    "minicpmv4_6",
    "granite_speech",
    "mimovl",
    "minimax_m3",
    "granite4_vision",
    "mimo_audio",
    "parakeet",
    "qwen3tts_spkenc",
    "qwen3tts_gen",
    "pockettts_spkenc",
    "pockettts_gen",
    "muse-glimmer",
)


def test_registry_is_the_exact_pinned_60_string_census() -> None:
    specs = iter_projector_specs()
    assert LLAMA_CPP_MMPROJ_SHA == "8d9af256337d1a501250f9bbf4c0859a654bddd6"
    assert tuple(spec.projector_type for spec in specs) == _PINNED_PROJECTOR_STRINGS
    assert len({spec.enum_name for spec in specs}) == 60


def test_registry_and_verdict_views_are_immutable() -> None:
    spec = get_projector_spec("gemma4v")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.projector_type = "mlp"  # type: ignore[misc]
    assert isinstance(spec.verdicts, MappingProxyType)
    with pytest.raises(TypeError):
        spec.verdicts["runtime"] = Support.DEFERRED  # type: ignore[index]


def test_graph_import_is_conservative_and_artifact_backed() -> None:
    assert supported_projector_types() == (
        "qwen2vl_merger",
        "qwen2.5vl_merger",
        "gemma3",
        "gemma4v",
        "muse-glimmer",
    )
    pins = {pin.artifact_id: pin for pin in MMPROJ_ARTIFACT_PINS}
    assert set(pins) == {
        "gemma3-4b-f16",
        "gemma4-e2b-f16",
        "muse-glimmer-30b-bf16",
        "qwen2-vl-2b-f16",
        "qwen25-vl-3b-f16",
    }
    for projector_type in supported_projector_types():
        spec = get_projector_spec(projector_type)
        assert spec.is_importable
        assert spec.runtime is Support.DEFERRED
        assert not spec.is_supported
        assert spec.real_artifact_ids
        assert all(artifact_id in pins for artifact_id in spec.real_artifact_ids)
        assert all(pins[artifact_id].parity_test for artifact_id in spec.real_artifact_ids)


def test_gemma3_real_artifact_pin_matches_huggingface_api_metadata() -> None:
    pin = next(pin for pin in MMPROJ_ARTIFACT_PINS if pin.artifact_id == "gemma3-4b-f16")
    assert pin.repository == "ggml-org/gemma-3-4b-it-GGUF"
    assert pin.revision == "ab31416aceb30cd095cb34cc27eea120940964e4"
    assert pin.filename == "mmproj-model-f16.gguf"
    assert pin.size == 851_251_104
    assert pin.lfs_sha256 == (
        "8c0fb064b019a6972856aaae2c7e4792858af3ca4561be2dbf649123ba6c40cb"
    )
    assert pin.tensor_qtypes == (("F32", 276), ("F16", 163))
    assert pin.tensor_count == 439


def test_qwen_processor_assets_and_real_contracts_are_exactly_pinned() -> None:
    pins = {pin.artifact_id: pin for pin in MMPROJ_ARTIFACT_PINS}
    qwen2 = pins["qwen2-vl-2b-f16"]
    qwen25 = pins["qwen25-vl-3b-f16"]

    assert qwen2.processor_repository == "Qwen/Qwen2-VL-2B-Instruct"
    assert qwen2.processor_revision == "895c3a49bc3fa70a340399125c650a463535e71c"
    assert qwen25.processor_repository == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert qwen25.processor_revision == "66285546d2b821cf421d4f5eb2576359d3770cd3"
    assert qwen2.processor_files == qwen25.processor_files == (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "chat_template.json",
    )
    for pin in (qwen2, qwen25):
        contract = dict(pin.processor_contract)
        assert contract["pixel_values"] == "float32[total_image_patches,1176]"
        assert contract["image_grid_thw"] == "int64[num_images,3]"
        assert contract["pixel_values_videos"] == "float32[total_video_patches,1176]"
        assert contract["video_grid_thw"] == "int64[num_videos,3]"
        assert contract["ordering"] == (
            "batch-major within independent image and video streams"
        )
        assert contract["empty_media"].startswith("omit ")
    assert dict(qwen25.processor_contract)["second_per_grid_ts"] == "float64[num_videos]"


def test_gemma3_processor_assets_and_real_contract_are_exactly_pinned() -> None:
    pins = {pin.artifact_id: pin for pin in MMPROJ_ARTIFACT_PINS}
    pin = pins["gemma3-4b-f16"]

    assert pin.processor_repository == "google/gemma-3-4b-it"
    assert pin.processor_revision == "093f9f388b31de276ce2de164bdc2081324b9767"
    assert pin.processor_class == "Gemma3Processor"
    assert pin.processor_files == (
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    )
    assert dict(pin.processor_contract) == {
        "pixel_values": "float32[num_images,3,896,896]",
        "vision_invocation": "split to one image row per vision graph call",
        "image_features": "concatenate 256 rows per image in processor row order",
        "empty_media": "omit pixel_values",
        "ordering": "batch-major image rows",
    }


def test_vlm_text_cohort_records_exact_companion_identity_without_support_claims() -> None:
    expected_targets = {
        "qwen3vl_merger": {"qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe"},
        "qwen3a": {"qwen3vl", "qwen3vlmoe"},
        "gemma3nv": {"gemma3n"},
        "gemma3na": {"gemma3n"},
        "pixtral": {"deepseek2", "llama", "mistral3", "mistral4"},
        "llama4": {"llama4"},
        "qwen2.5o": {"qwen2vl"},
        "paddleocr": {"paddleocr"},
        "cogvlm": {"cogvlm"},
        "deepseekocr": {"deepseek2-ocr"},
        "deepseekocr2": {"deepseek2-ocr"},
        "hunyuanvl": {"hunyuan_vl"},
    }
    for projector_type, targets in expected_targets.items():
        spec = get_projector_spec(projector_type)
        assert spec.target_architectures == frozenset(targets)
        assert not spec.is_supported
        assert set(spec.verdicts.values()) == {Support.DEFERRED}
        assert spec.builder is None
        assert spec.required_top_tensors == ()


def test_every_non_supported_verdict_has_an_actionable_reason() -> None:
    for spec in iter_projector_specs():
        if spec.is_supported:
            continue
        assert spec.reason
        assert len(spec.reason.split()) >= 5


def test_metadata_schema_captures_the_pinned_absence_of_a_text_encoder() -> None:
    assert len(CLIP_METADATA_SCHEMA) == 61
    fields = {field.key: field for field in CLIP_METADATA_SCHEMA}
    assert fields["clip.has_vision_encoder"].default is False
    assert fields["clip.has_audio_encoder"].default is False
    assert fields["clip.has_gen_audio_encoder"].default is False
    assert "Absent from the pinned ABI" in fields["clip.has_text_encoder"].note


def test_modality_projector_overrides_global_fallback() -> None:
    metadata = {
        "clip.projector_type": "gemma4v",
        "clip.audio.projector_type": "gemma4a",
    }
    assert projector_type_for_modality(metadata, MMProjModality.AUDIO) == "gemma4a"
    assert projector_type_for_modality(metadata, MMProjModality.VISION) == "gemma4v"


def test_missing_modality_and_global_projector_fails_closed() -> None:
    with pytest.raises(ValueError, match=r"neither 'clip\.projector_type' nor"):
        projector_type_for_modality({}, MMProjModality.GENERATED_AUDIO)


@pytest.mark.parametrize("projector_type", ["mlp", "gemma4a", "qwen2.5o"])
def test_deferred_projector_has_no_dispatch_or_loader_closure(projector_type: str) -> None:
    spec = get_projector_spec(projector_type)
    assert not spec.is_supported
    assert spec.builder is None
    assert spec.required_metadata == ()
    assert spec.required_top_tensors == ()
    assert spec.block_suffixes == ()


@pytest.mark.parametrize("projector_type", ["qwen3tts_gen", "pockettts_gen"])
def test_generated_audio_decoders_are_explicitly_rejected(projector_type: str) -> None:
    spec = get_projector_spec(projector_type)
    assert set(spec.verdicts.values()) == {Support.REJECTED}


def test_documented_projector_matrix_is_generated_from_registry() -> None:
    from mobius.integrations.gguf._docs import check_document

    assert check_document()
