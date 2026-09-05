# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Closure tests for GGUF MTP artifact and downstream-runtime status."""

from __future__ import annotations

import importlib.metadata
from dataclasses import replace

import pytest

from mobius.integrations.gguf._arch_registry import get_arch_spec
from mobius.integrations.gguf._mtp_runtime_evidence import (
    GGUFMtpArtifactLayout,
    iter_mtp_runtime_evidence,
    mtp_runtime_evidence,
)
from mobius.integrations.gguf._runtime_evidence import runtime_evidence


def _by_architecture():
    return {evidence.architecture: evidence for evidence in iter_mtp_runtime_evidence()}


def test_mtp_runtime_status_is_exactly_hy_v3_and_qwen35() -> None:
    records = _by_architecture()

    assert set(records) == {"hy_v3", "qwen35"}
    for record in records.values():
        assert mtp_runtime_evidence(record.evidence_id) is record
        assert runtime_evidence(record.evidence_id) is None
        assert get_arch_spec(record.architecture).runtime_evidence_ids == ()
        assert record.result == "runtime_unvalidated"
        assert record.graph_sha256 is None
        assert record.runtime_package_sha256 is None
        assert not record.source_fidelity
        assert not record.storage_fidelity


def test_qwen35_combined_fits_but_split_pair_exceeds_16_gib() -> None:
    record = _by_architecture()["qwen35"]
    layouts = {layout.name: layout for layout in record.layouts}

    assert record.bounded_complete_layout_available
    assert layouts["combined"].total_size == 16_998_719_584
    assert layouts["combined"].within_bounded_artifact_policy
    assert layouts["split"].total_size == 20_776_036_864
    assert not layouts["split"].within_bounded_artifact_policy

    combined = layouts["combined"].artifacts[0]
    target, mtp = layouts["split"].artifacts
    assert combined.trunk_block_count == 64
    assert combined.last_block_index == 64
    assert combined.nextn_tensor_count == 4
    assert target.role == "target"
    assert target.last_block_index == 63
    assert mtp.role == "mtp"
    assert mtp.first_block_index == mtp.last_block_index == 64
    assert target.tokenizer_metadata_sha256 == mtp.tokenizer_metadata_sha256


def test_hy_v3_target_only_and_combined_headers_are_physically_distinct() -> None:
    record = _by_architecture()["hy_v3"]
    combined = record.layouts[0].artifacts[0]
    target = record.target_only_discriminator

    assert target is not None
    assert not record.bounded_complete_layout_available
    assert combined.size == 91_756_066_272
    assert target.size == 89_446_312_384
    assert combined.lfs_sha256 != target.lfs_sha256
    assert combined.bounded_header_sha256 != target.bounded_header_sha256
    assert combined.trunk_block_count == target.block_count == 80
    assert combined.last_block_index == 80
    assert combined.nextn_tensor_count == 4
    assert target.last_block_index == 79
    assert target.nextn_tensor_count == 0


def test_split_layout_rejects_a_mutated_physical_mtp_block_index() -> None:
    layout = next(
        layout for layout in _by_architecture()["qwen35"].layouts if layout.name == "split"
    )
    target, mtp = layout.artifacts
    short_target = replace(
        target,
        block_count=63,
        last_block_index=62,
        physical_block_count=63,
    )

    with pytest.raises(ValueError, match="immediately follow every target block"):
        GGUFMtpArtifactLayout("split", (short_target, mtp))


def test_combined_layout_rejects_a_mutated_trailing_block_identity() -> None:
    combined = _by_architecture()["qwen35"].layouts[0].artifacts[0]

    with pytest.raises(ValueError, match="trailing block"):
        replace(combined, block_count=66)


def test_cache_topology_rejects_target_mtp_namespace_aliasing() -> None:
    topology = _by_architecture()["hy_v3"].cache_topology

    with pytest.raises(ValueError, match="namespaces must be non-empty and distinct"):
        replace(topology, mtp_namespace=topology.target_namespace)


def test_cache_topologies_keep_every_slot_component_qualified() -> None:
    for record in _by_architecture().values():
        topology = record.cache_topology
        target = {
            (topology.target_namespace, kind, index)
            for kind, count in topology.target_state_slots
            for index in range(count)
        }
        mtp = {
            (topology.mtp_namespace, kind, index)
            for kind, count in topology.mtp_state_slots
            for index in range(count)
        }
        assert target.isdisjoint(mtp)
        assert dict(topology.mtp_state_slots) == {
            "attention.key": 1,
            "attention.value": 1,
        }

    qwen = _by_architecture()["qwen35"]
    assert dict(qwen.synthetic_acceptance_statistics) == {
        "accepted": 1,
        "proposal_steps": 51,
        "rejected": 50,
        "rollbacks": 50,
    }
    assert qwen.synthetic_coordinator_test is not None
    assert _by_architecture()["hy_v3"].synthetic_coordinator_test is None


@pytest.mark.ort_genai_fast
def test_pinned_ort_genai_0152_public_abi_has_no_mtp_orchestration() -> None:
    if importlib.metadata.version("onnxruntime-genai") != "0.15.2":
        pytest.skip("This discriminator is pinned to onnxruntime-genai 0.15.2")
    if importlib.metadata.version("onnxruntime") != "1.29.0":
        pytest.skip("This discriminator is pinned to onnxruntime 1.29.0")

    import onnxruntime as ort
    import onnxruntime_genai as ort_genai

    record = _by_architecture()["qwen35"]
    assert ort_genai.__version__ == record.runtime_version
    assert record.execution_provider in ort.get_available_providers()

    # rewind_to only rewinds the one Generator. There is no second-model binding,
    # draft proposal transaction, independent MTP cache, or acceptance-statistics API.
    assert hasattr(ort_genai.Generator, "rewind_to")
    public_api = {
        name
        for cls in (ort_genai.Config, ort_genai.Generator, ort_genai.GeneratorParams)
        for name in dir(cls)
        if not name.startswith("_")
    }
    assert not public_api & {
        "accept_draft_tokens",
        "acceptance_statistics",
        "append_draft_model",
        "set_draft_model",
        "set_mtp_model",
        "set_proposer",
    }
    assert record.missing_runtime_capabilities == (
        "accept_reject_rollback",
        "acceptance_statistics",
        "independent_target_and_mtp_cache_threading",
        "two_model_draft_target_binding",
    )
