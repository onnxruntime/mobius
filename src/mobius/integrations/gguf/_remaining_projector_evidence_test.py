# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from mobius.integrations.gguf._artifact_blocker_evidence import (
    MAX_BOUNDED_ARTIFACT_BYTES,
)
from mobius.integrations.gguf._mmproj_registry import (
    MMPROJ_ARTIFACT_PINS,
    MMPROJ_SOURCE_EVIDENCE,
)

_ROUTES = {
    "cogvlm",
    "exaone4_5",
    "hunyuanvl",
    "janus_pro",
    "kimik25",
    "kimivl",
    "lfm2",
    "meralion",
    "mimovl",
    "minicpmv4_6",
    "minimax_m3",
    "nemotron_v2_vl",
    "step3vl",
    "yasa2",
}


def test_remaining_projector_source_evidence_is_complete_and_immutable() -> None:
    records = MMPROJ_SOURCE_EVIDENCE
    assert {record.evidence_id.removesuffix("-pinned-graph-source") for record in records} == {
        route.replace("_", "-") for route in _ROUTES
    }
    assert len({record.evidence_id for record in records}) == len(_ROUTES)
    for record in records:
        assert all(len(revision) == 40 for _, revision, _ in record.sources)


def test_bounded_header_evidence_is_isolated_below_the_payload_budget() -> None:
    pins = [pin for pin in MMPROJ_ARTIFACT_PINS if pin.bounded_header_bytes is not None]
    assert {pin.projector_types[0] for pin in pins} == _ROUTES - {
        "meralion",
        "minimax_m3",
    }
    assert all(pin.size <= MAX_BOUNDED_ARTIFACT_BYTES for pin in pins)
    assert sum(pin.bounded_header_bytes or 0 for pin in pins) <= MAX_BOUNDED_ARTIFACT_BYTES
    assert all(pin.processor_repository and pin.processor_revision for pin in pins)
    assert all(pin.processor_contract for pin in pins)


def test_minicpm_mislabeled_f16_alias_is_not_used_as_evidence() -> None:
    pin = next(
        pin
        for pin in MMPROJ_ARTIFACT_PINS
        if pin.artifact_id == "minicpm-v4-6-bf16-header"
    )
    assert pin.filename == "MiniCPM-V-4.6.mmproj-bf16.gguf"
    assert pin.tensor_qtypes == (("BF16", 170), ("F32", 289))
