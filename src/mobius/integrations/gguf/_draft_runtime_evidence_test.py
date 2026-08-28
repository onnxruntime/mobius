# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _evidence() -> dict:
    path = (
        Path(__file__).parents[4]
        / "testdata"
        / "evidence"
        / "gguf_draft_runtime_evidence.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _independent_trace(record: dict) -> dict:
    metadata = record["independent_direct_ort_trace"]
    path = Path(__file__).parents[4] / "testdata" / "evidence" / metadata["filename"]
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
    return json.loads(payload)


def test_draft_runtime_evidence_is_bounded_immutable_and_complete() -> None:
    evidence = _evidence()
    policy = evidence["policy"]
    independent = evidence["independent_direct_ort"]
    assert evidence["schema_version"] == 1
    assert (
        policy["bound_target_draft_source_and_tokenizer_bytes"]
        <= policy["maximum_session_bytes"]
    )
    assert policy["higher_level_runtime_status"] == "runtime_unvalidated"
    assert independent["production_runner_imported"] is False
    assert independent["transition_helpers_imported"] is False
    assert independent["draft_mapping_source"] == "raw immutable draft GGUF metadata"
    assert {record["architecture"] for record in evidence["routes"]} == {
        "dflash",
        "eagle3",
    }
    bound_bytes = sum(
        record["target"]["size"]
        + record["draft"]["size"]
        + record["draft_source"]["config_size"]
        + record["draft_source"]["weights_size"]
        + record["target_config"]["config_size"]
        + record["target_config"]["tokenizer_size"]
        + record["target_config"]["tokenizer_config_size"]
        for record in evidence["routes"]
    )
    assert bound_bytes == policy["bound_target_draft_source_and_tokenizer_bytes"]

    for record in evidence["routes"]:
        assert record["result"] == "passed-direct-ort"
        for artifact in (record["target"], record["draft"]):
            assert re.fullmatch(r"[0-9a-f]{40}", artifact["revision"])
            assert re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
            assert artifact["size"] > 0
            assert artifact["tensor_count"] == sum(artifact["tensor_qtypes"].values())
        for package_hash in (
            record["mobius_packages"]["target_sha256"],
            record["mobius_packages"]["draft_sha256"],
        ):
            assert re.fullmatch(r"[0-9a-f]{64}", package_hash)


def test_draft_runtime_evidence_proves_real_speculative_work() -> None:
    for record in _evidence()["routes"]:
        result = record["direct_ort_result"]
        assert result["target_only_equal"] is True
        assert result["proposed_tokens"] > result["accepted_tokens"] > 0
        assert result["multi_token_rounds"] > 0
        assert result["rollback_events"] > 0
        assert result["target_cache_threaded"] is True
        assert result["draft_cache_threaded"] is True
        assert "no speedup claim" in result["timing_disposition"].lower() or (
            "not a benchmark claim" in result["timing_disposition"].lower()
        )
        assert "does not gate" in record["runtime_warning"]

        trace = _independent_trace(record)
        assert trace["tokens_equal"] is True
        assert trace["generated_tokens_sha256"] == result["generated_tokens_sha256"]
        assert trace["counters"] == {
            "accepted": result["accepted_tokens"],
            "multi_token_rounds": result["multi_token_rounds"],
            "proposed": result["proposed_tokens"],
            "rejections": result["rollback_events"],
            "rounds": result["rounds"],
        }
        assert trace["reorder"]["supported"] is False
        assert record["independent_direct_ort_trace"]["mutation_discriminators"] == {
            "draft_mapping": "semantic.draft_mapping",
            "proposal_order": "semantic.proposal_order",
            "cache_copy": "semantic.draft_cache_replay",
            "rollback": "semantic.target_replay_tokens",
        }
        assert any(round_trace["accepted_prefix"] > 1 for round_trace in trace["rounds"])
        assert any(round_trace["accepted_prefix"] == 0 for round_trace in trace["rounds"])
        assert trace["final_target_cache"]["length"] == 47
        assert (
            sum(len(item["proposal_ids"]) for item in trace["rounds"])
            == result["proposed_tokens"]
        )
        for round_trace in trace["rounds"]:
            assert 0 < len(round_trace["proposal_ids"]) <= result["draft_width"]
            assert len(round_trace["proposal_tokens"]) == len(round_trace["proposal_ids"])
            assert re.fullmatch(r"[0-9a-f]{64}", round_trace["proposal_logits_sha256"])
            assert round_trace["target"]["replay_tokens_match"] is True
            assert (
                round_trace["target"]["tentative_length"]
                <= trace["final_target_cache"]["length"]
            )
            assert (
                round_trace["target"]["before_length"]
                < (round_trace["target"]["tentative_length"])
            )
            assert (
                round_trace["target"]["committed_length"]
                == (round_trace["target"]["replay_length"])
            )
            assert (
                round_trace["draft"]["committed_length"]
                == (round_trace["draft"]["replay_length"])
            )


def test_draft_source_fidelity_dispositions_are_truthful() -> None:
    by_arch = {record["architecture"]: record for record in _evidence()["routes"]}
    dflash = by_arch["dflash"]["source_fidelity"]
    assert dflash["cosine"] > 0.999
    assert 0 < dflash["relative_l2"] < 0.01
    eagle3 = by_arch["eagle3"]["source_fidelity"]
    assert eagle3["all_types_shapes_values_equal"] is True
    assert eagle3["tensor_count"] == 15
