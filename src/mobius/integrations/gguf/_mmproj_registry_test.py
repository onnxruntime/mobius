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
    MMPROJ_ARTIFACT_AVAILABILITY_PINS,
    MMPROJ_ARTIFACT_PINS,
    MMPROJ_SOURCE_EVIDENCE,
    MMProjModality,
    MMProjModelRole,
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
    assert LLAMA_CPP_MMPROJ_SHA == "86632248188c106d749fad34a1dcd237c95863d4"
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
        "mlp",
        "ldp",
        "ldpv2",
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
        "cogvlm",
        "janus_pro",
        "lfm2a",
        "glm4v",
        "yasa2",
        "kimik25",
        "nemotron_v2_vl",
        "exaone4_5",
        "hunyuanvl",
        "minicpmv4_6",
        "granite_speech",
        "mimovl",
        "minimax_m3",
        "mimo_audio",
        "parakeet",
        "qwen3tts_spkenc",
        "pockettts_spkenc",
        "muse-glimmer",
    )
    pins = {pin.artifact_id: pin for pin in MMPROJ_ARTIFACT_PINS}
    assert {
        "glm-edge-v-2b-adapter-f16",
        "gemma3-4b-f16",
        "gemma3n-e4b-f16",
        "gemma4-e2b-f16",
        "gemma4-unified-12b-f16",
        "llava-llama3-8b-mlp-f16",
        "minicpm-v2-resampler-f16",
        "mobilevlm-1.7b-ldp-f16",
        "mobilevlm-v2-1.7b-ldpv2-f16",
        "muse-glimmer-30b-bf16",
        "smolvlm-256m-idefics3-f16",
        "internvl25-1b-f16",
        "llama4-scout-f16",
        "pixtral-12b-f16",
        "qwen2-vl-2b-f16",
        "qwen25-vl-3b-f16",
        "qwen3-vl-projector-f16",
        "qwen3-audio-projector-bf16",
        "qwen2-audio-projector-f16",
        "qwen25-omni-projector-f16",
        "glm4v-projector-f16",
    } <= set(pins)
    for projector_type in supported_projector_types():
        spec = get_projector_spec(projector_type)
        assert spec.is_importable
        assert spec.runtime is Support.DEFERRED
        assert not spec.is_supported
        assert spec.real_artifact_ids or spec.source_evidence_ids
        assert all(artifact_id in pins for artifact_id in spec.real_artifact_ids)
        assert all(pins[artifact_id].parity_test for artifact_id in spec.real_artifact_ids)

    cohort = [
        pins[artifact_id]
        for artifact_id in (
            "llava-llama3-8b-mlp-f16",
            "mobilevlm-1.7b-ldp-f16",
            "mobilevlm-v2-1.7b-ldpv2-f16",
            "glm-edge-v-2b-adapter-f16",
            "minicpm-v2-resampler-f16",
        )
    ]
    assert all(pin.paired_text_revision for pin in cohort)
    processor_evidence = {pin.projector_types[0]: pin.processor_revision for pin in cohort}
    assert processor_evidence == {
        "mlp": "b20fb3040caaf5d0b3751c0d86a94efdf5bb007d",
        "ldp": None,
        "ldpv2": None,
        "adapter": "2053707733f99ab52e943904f43c2359a94301ef",
        "resampler": None,
    }
    assert sum(pin.size + (pin.paired_text_size or 0) for pin in cohort) <= 16 * 1024**3

    qwen_glm = [
        pins[artifact_id]
        for artifact_id in (
            "qwen3-vl-projector-f16",
            "qwen3-audio-projector-bf16",
            "qwen2-audio-projector-f16",
            "qwen25-omni-projector-f16",
            "glm4v-projector-f16",
        )
    ]
    assert sum(pin.size for pin in qwen_glm) == 5_980_273_312
    assert sum(pin.size for pin in qwen_glm) <= 16 * 1024**3


def test_deferred_and_packed_projector_artifacts_are_immutably_available() -> None:
    pins = {
        (pin.projector_type, pin.filename): pin for pin in MMPROJ_ARTIFACT_AVAILABILITY_PINS
    }
    assert set(pins) == {
        ("lfm2", "mmproj-LFM2-VL-1.6B-F16.gguf"),
        ("lfm2", "mmproj-LFM2-VL-1.6B-Q8_0.gguf"),
        ("pixtral", "mmproj-pixtral-12b-Q8_0.gguf"),
    }
    assert {
        (pin.repository, pin.revision, pin.size, pin.lfs_sha256) for pin in pins.values()
    } == {
        (
            "LiquidAI/LFM2-VL-1.6B-GGUF",
            "6121de267003bb4d4f325fe10abdc735aee06747",
            830_339_008,
            "b637bfa6060be2bc7503ec23ba48b407843d08c2ca83f52be206ea8563ccbae2",
        ),
        (
            "LiquidAI/LFM2-VL-1.6B-GGUF",
            "6121de267003bb4d4f325fe10abdc735aee06747",
            564_115_648,
            "65ec437db88d65fff93f472d00c145e09880769ac67fedff5cd1c0f8d8301d87",
        ),
        (
            "ggml-org/pixtral-12b-GGUF",
            "cba1ea4420bc2b4f15f50fdec59e30769880a63c",
            463_091_616,
            "5504fe00067629053e6f99abac05f628c653a50394f4929bcc185bc80a10daf4",
        ),
    }
    lfm2 = get_projector_spec("lfm2")
    assert lfm2.is_importable
    assert lfm2.real_artifact_ids == ("lfm2-vl-1-6b-f16-header",)
    assert get_projector_spec("pixtral").is_importable


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
    assert (
        qwen2.processor_files
        == qwen25.processor_files
        == (
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "chat_template.json",
        )
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
        "paddleocr": {"paddleocr"},
        "deepseekocr": {"deepseek2-ocr"},
        "deepseekocr2": {"deepseek2-ocr"},
    }
    for projector_type, targets in expected_targets.items():
        spec = get_projector_spec(projector_type)
        assert spec.target_architectures == frozenset(targets)
        assert not spec.is_supported
        assert set(spec.verdicts.values()) == {Support.DEFERRED}
        assert spec.builder is None
        assert spec.required_top_tensors == ()


def test_qwen_glm_routes_have_exact_standalone_roles_and_pairing() -> None:
    expected = {
        "glm4v": ({"glm4", "glm4moe"}, (MMProjModelRole.VISION_ENCODER,)),
        "glma": ({"llama"}, (MMProjModelRole.AUDIO_ENCODER,)),
        "qwen2.5o": (
            {"qwen2vl"},
            (MMProjModelRole.VISION_ENCODER, MMProjModelRole.AUDIO_ENCODER),
        ),
        "qwen2a": ({"qwen2"}, (MMProjModelRole.AUDIO_ENCODER,)),
        "qwen3a": ({"qwen3vl", "qwen3vlmoe"}, (MMProjModelRole.AUDIO_ENCODER,)),
        "qwen3vl_merger": (
            {"qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe"},
            (MMProjModelRole.VISION_ENCODER,),
        ),
        "qwen3tts_spkenc": (
            {"qwen3tts"},
            (MMProjModelRole.SPEAKER_ENCODER,),
        ),
    }
    for projector_type, (targets, roles) in expected.items():
        spec = get_projector_spec(projector_type)
        assert spec.target_architectures == frozenset(targets)
        assert spec.sidecar_builder == "qwen_glm_projector"
        assert spec.builder is None
        assert spec.model_roles == roles
        assert spec.is_importable
        assert spec.runtime is Support.DEFERRED

    alias = get_projector_spec("qwen2.5o")
    assert alias.primary_modality is MMProjModality.VISION
    assert alias.companion_tensors[0].modality is MMProjModality.AUDIO
    speaker = get_projector_spec("qwen3tts_spkenc")
    assert speaker.deferred_companions[0].projector_type == "qwen3tts_gen"
    assert speaker.deferred_companions[0].tensor_prefixes == ("a.gen.",)


def test_qwen_glm_source_blockers_are_immutable_and_not_model_gates() -> None:
    evidence = {record.evidence_id: record for record in MMPROJ_SOURCE_EVIDENCE}
    assert {
        "glma-converter-checkpoint-drift",
        "qwen3tts-speaker-runtime-boundary",
    } <= set(evidence)
    assert all(
        len(revision) == 40
        for record in evidence.values()
        for _, revision, _ in record.sources
    )
    assert "partial-RoPE" in evidence["glma-converter-checkpoint-drift"].finding
    assert "tts_pad" in evidence["qwen3tts-speaker-runtime-boundary"].finding


def test_every_non_supported_verdict_has_an_actionable_reason() -> None:
    for spec in iter_projector_specs():
        if spec.is_supported:
            continue
        assert spec.reason
        assert len(spec.reason.split()) >= 5


def test_metadata_schema_captures_the_pinned_absence_of_a_text_encoder() -> None:
    assert len(CLIP_METADATA_SCHEMA) == 62
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


def test_core_vlm_routes_preserve_roles_pairing_and_bounded_evidence() -> None:
    from mobius.integrations.gguf._mmproj_registry import MMProjModelRole

    expected = {
        "gemma3nv": ({"gemma3n"}, MMProjModality.VISION, MMProjModelRole.VISION_ENCODER),
        "gemma3na": ({"gemma3n"}, MMProjModality.AUDIO, MMProjModelRole.AUDIO_ENCODER),
        "gemma4a": ({"gemma4"}, MMProjModality.AUDIO, MMProjModelRole.AUDIO_ENCODER),
        "gemma4uv": ({"gemma4"}, MMProjModality.VISION, MMProjModelRole.VISION_ENCODER),
        "gemma4ua": ({"gemma4"}, MMProjModality.AUDIO, MMProjModelRole.AUDIO_ENCODER),
        "idefics3": ({"llama"}, MMProjModality.VISION, MMProjModelRole.VISION_ENCODER),
        "internvl": ({"qwen2"}, MMProjModality.VISION, MMProjModelRole.VISION_ENCODER),
        "llama4": ({"llama4"}, MMProjModality.VISION, MMProjModelRole.VISION_ENCODER),
        "pixtral": ({"llama"}, MMProjModality.VISION, MMProjModelRole.VISION_ENCODER),
    }
    pins = {pin.artifact_id: pin for pin in MMPROJ_ARTIFACT_PINS}
    for projector_type, (targets, modality, role) in expected.items():
        spec = get_projector_spec(projector_type)
        assert spec.target_architectures == frozenset(targets)
        assert spec.primary_modality is modality
        assert spec.model_roles == (role,)
        assert spec.sidecar_builder == "core_vlm_projector"
        assert spec.is_importable
        assert spec.runtime is Support.DEFERRED
        for artifact_id in spec.real_artifact_ids:
            pin = pins[artifact_id]
            assert pin.size <= 16 * 1024**3
            assert pin.parity_test
            assert len(pin.revision) == 40
            assert len(pin.lfs_sha256) == 64

    assert get_projector_spec("gemma3nv").modalities == frozenset({MMProjModality.VISION})
    assert get_projector_spec("gemma3na").modalities == frozenset({MMProjModality.AUDIO})
    assert get_projector_spec("gemma4a").required_top_tensors != (
        get_projector_spec("gemma4ua").required_top_tensors
    )


@pytest.mark.parametrize("projector_type", ["qwen3tts_gen", "pockettts_gen"])
def test_generated_audio_decoders_are_explicitly_rejected(projector_type: str) -> None:
    spec = get_projector_spec(projector_type)
    assert set(spec.verdicts.values()) == {Support.REJECTED}


def test_documented_projector_matrix_is_generated_from_registry() -> None:
    from mobius.integrations.gguf._docs import check_document

    assert check_document()
