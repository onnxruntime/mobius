# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
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
    repository: Path, paths: list[str], *, global_attributes: Path
) -> dict[str, dict[str, str]]:
    environment = os.environ.copy()
    environment["GIT_ATTR_NOSYSTEM"] = "1"
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
            *paths,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        env=environment,
    )
    fields = result.stdout.decode("utf-8").split("\0")
    assert fields.pop() == ""
    assert len(fields) % 3 == 0
    attributes: dict[str, dict[str, str]] = {}
    for path, attribute, value in zip(fields[0::3], fields[1::3], fields[2::3], strict=True):
        attributes.setdefault(path, {})[attribute] = value
    return attributes


def test_evidence_json_files_are_lf_normalized_for_raw_hashes(tmp_path: Path) -> None:
    isolated_repo = tmp_path / "repository"
    isolated_repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=isolated_repo, check=True)
    shutil.copyfile(_REPO_ROOT / ".gitattributes", isolated_repo / ".gitattributes")

    global_attributes = tmp_path / "global-attributes"
    global_attributes.touch()
    evidence_json = sorted(
        path.relative_to(_REPO_ROOT).as_posix() for path in _EVIDENCE_DIR.rglob("*.json")
    )
    assert any(path.startswith("testdata/evidence/causal-lm/") for path in evidence_json)
    future_binary = "testdata/evidence/future-evidence.bin"
    for relative_path in [*evidence_json, future_binary]:
        path = isolated_repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    attributes = _effective_git_attributes(
        isolated_repo,
        [*evidence_json, future_binary],
        global_attributes=global_attributes,
    )

    for relative_path in evidence_json:
        assert attributes[relative_path] == {"text": "set", "eol": "lf"}
    assert attributes[future_binary] == {
        "text": "unspecified",
        "eol": "unspecified",
    }

    expected_trace_hashes = {
        "gguf_dflash_independent_ort_trace.json": {
            "crlf": "e0e70e909f33ed44aa87da39bab721ebbee0c3e730e2f89dbc509c6683701e42",
            "semantic": "ea1f1414d5bc28c2b7776fa66f2a3b4bc72eeda2b38ecaa8d1ca3f8f8bfe8fcf",
        },
        "gguf_eagle3_independent_ort_trace.json": {
            "crlf": "78aa94f24d5f1b9ae020de1ffa5a70c6a30f30ff45eb5310a540cdd2b3d8fab2",
            "semantic": "4dbd07a42b89f2067deb1eb249ce8f06ccb973cd700724592f3892deaf78a5db",
        },
    }
    traces = {
        record["independent_direct_ort_trace"]["filename"]: record[
            "independent_direct_ort_trace"
        ]
        for record in _evidence()["routes"]
    }
    assert traces.keys() == expected_trace_hashes.keys()
    for filename, metadata in traces.items():
        payload = (_EVIDENCE_DIR / filename).read_bytes()
        assert b"\n" in payload
        assert b"\r\n" not in payload
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
        semantic_payload = json.dumps(
            json.loads(payload), sort_keys=True, separators=(",", ":")
        ).encode()
        assert (
            hashlib.sha256(semantic_payload).hexdigest()
            == expected_trace_hashes[filename]["semantic"]
        )
        crlf_payload = payload.replace(b"\n", b"\r\n")
        crlf_sha256 = hashlib.sha256(crlf_payload).hexdigest()
        assert crlf_sha256 == expected_trace_hashes[filename]["crlf"]
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
