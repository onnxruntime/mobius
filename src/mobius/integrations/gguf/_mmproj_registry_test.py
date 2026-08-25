# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Closure tests for the pinned clip projector capability registry."""

from __future__ import annotations

import dataclasses
import pathlib
from types import MappingProxyType

import pytest

from mobius.integrations.gguf._mmproj_registry import (
    CLIP_METADATA_SCHEMA,
    LLAMA_CPP_MMPROJ_SHA,
    MMPROJ_ARTIFACT_PINS,
    get_projector_spec,
    iter_projector_specs,
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


def test_support_is_conservative_and_artifact_backed() -> None:
    assert supported_projector_types() == ("gemma4v", "muse-glimmer")
    pins = {pin.artifact_id: pin for pin in MMPROJ_ARTIFACT_PINS}
    assert set(pins) == {"gemma4-e2b-f16", "muse-glimmer-30b-bf16"}
    for projector_type in supported_projector_types():
        spec = get_projector_spec(projector_type)
        assert spec.is_supported
        assert spec.real_artifact_ids
        assert all(artifact_id in pins for artifact_id in spec.real_artifact_ids)
        assert all(pins[artifact_id].parity_test for artifact_id in spec.real_artifact_ids)


def test_vlm_text_cohort_records_exact_companion_identity_without_support_claims() -> None:
    expected_targets = {
        "qwen2vl_merger": {"qwen2vl"},
        "qwen2.5vl_merger": {"qwen2vl"},
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
    doc = pathlib.Path(__file__).resolve().parents[4] / "docs" / "api" / "build_from_gguf.md"
    begin = "<!-- BEGIN GGUF MMPROJ SUPPORT MATRIX (generated; see _mmproj_registry.py) -->"
    end = "<!-- END GGUF MMPROJ SUPPORT MATRIX -->"
    block = doc.read_text(encoding="utf-8").split(begin, 1)[1].split(end, 1)[0]
    documented = [line for line in block.splitlines() if line.startswith("| `")]
    expected = []
    for spec in iter_projector_specs():
        modalities = ", ".join(
            modality.value for modality in sorted(spec.modalities, key=lambda item: item.value)
        )
        targets = (
            ", ".join(f"`{target}`" for target in sorted(spec.target_architectures)) or "—"
        )
        status = (
            "supported"
            if spec.is_supported
            else "; ".join(
                f"{name} {verdict.value}"
                for name, verdict in spec.verdicts.items()
                if verdict is not Support.SUPPORTED
            )
        )
        limitation = (
            spec.reason
            or "Exact registry-backed graph, tensor closure, target pairing, and component parity."
        )
        expected.append(
            f"| `{spec.projector_type}` | {modalities} | {targets} | {status} | {limitation} |"
        )
    assert documented == expected
