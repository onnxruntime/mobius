# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[4]
_EVIDENCE_DIR = _REPO_ROOT / "testdata" / "evidence"


def _evidence() -> dict:
    path = _EVIDENCE_DIR / "gguf_draft_runtime_evidence.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _independent_trace(record: dict) -> dict:
    metadata = record["independent_direct_ort_trace"]
    path = _EVIDENCE_DIR / metadata["filename"]
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
    return json.loads(payload)


def _effective_git_attributes(
    paths: list[Path], *, global_attributes: Path
) -> dict[str, dict[str, str]]:
    relative_paths = [path.relative_to(_REPO_ROOT).as_posix() for path in paths]
    result = subprocess.run(
        [
            "git",
            "-c",
            f"core.attributesFile={global_attributes}",
            "check-attr",
            "-z",
            "text",
            "eol",
            "--",
            *relative_paths,
        ],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
    )
    fields = result.stdout.decode("utf-8").split("\0")
    assert fields[-1] == ""
    attributes: dict[str, dict[str, str]] = {}
    for path, attribute, value in zip(fields[0::3], fields[1::3], fields[2::3]):
        attributes.setdefault(path, {})[attribute] = value
    return attributes


def test_evidence_json_files_are_lf_normalized_for_raw_hashes(tmp_path: Path) -> None:
    global_attributes = tmp_path / "global-attributes"
    global_attributes.touch()
    evidence_json = sorted(_EVIDENCE_DIR.rglob("*.json"))
    future_binary = _EVIDENCE_DIR / "future-evidence.bin"
    attributes = _effective_git_attributes(
        [*evidence_json, future_binary],
        global_attributes=global_attributes,
    )

    for path in evidence_json:
        relative_path = path.relative_to(_REPO_ROOT).as_posix()
        assert attributes[relative_path] == {"text": "set", "eol": "lf"}
    assert attributes[future_binary.relative_to(_REPO_ROOT).as_posix()] == {
        "text": "unspecified",
        "eol": "unspecified",
    }

    expected_crlf_sha256 = {
        "gguf_dflash_independent_ort_trace.json": (
            "647dfdca436f832ce6662e9131bacc59803f74a7b9c957b09dd4e32d52df7224"
        ),
        "gguf_eagle3_independent_ort_trace.json": (
            "180056770131135ebef3e23bda7800693a0f891cbffc1b662ec9105e836a2a73"
        ),
    }
    traces = {
        record["independent_direct_ort_trace"]["filename"]: record[
            "independent_direct_ort_trace"
        ]
        for record in _evidence()["routes"]
    }
    assert traces.keys() == expected_crlf_sha256.keys()
    for filename, metadata in traces.items():
        payload = (_EVIDENCE_DIR / filename).read_bytes()
        assert b"\n" in payload
        assert b"\r\n" not in payload
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
        crlf_payload = payload.replace(b"\n", b"\r\n")
        crlf_sha256 = hashlib.sha256(crlf_payload).hexdigest()
        assert crlf_sha256 == expected_crlf_sha256[filename]
        assert crlf_sha256 != metadata["sha256"]


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
