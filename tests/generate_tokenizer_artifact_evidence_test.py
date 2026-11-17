# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Network-free tests for the final tokenizer artifact evidence workflow."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import generate_tokenizer_artifact_evidence as generator  # noqa: E402

from mobius.integrations.gguf import _tokenizer  # noqa: E402
from mobius.integrations.gguf._tokenizer_evidence import (  # noqa: E402
    tokenizer_blocker_evidence,
)
from mobius.integrations.gguf._upstream import UPSTREAM_COMMIT  # noqa: E402

_FIXTURE_PATH = _ROOT / "tests/data/gguf_tokenizer_artifact_evidence.json"
_INPUTS_PATH = _ROOT / "tests/data/gguf_tokenizer_artifact_inputs.tar.xz"


def _fixture() -> dict:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _route_fixture(name: str) -> dict:
    return next(route for route in _fixture()["routes"] if route["name"] == name)


def _route(name: str) -> generator.Route:
    return next(route for route in generator.ROUTES if route.name == name)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_fixture_binds_generators_inputs_revisions_and_download_budget() -> None:
    fixture = _fixture()
    assert fixture["generator"] == "scripts/generate_tokenizer_artifact_evidence.py"
    assert fixture["generator_sha256"] == generator._generator_sha256(
        _ROOT / fixture["generator"]
    )
    assert fixture["llamacpp_commit"] == generator.LLAMACPP_COMMIT == UPSTREAM_COMMIT
    assert fixture["qualification_inputs"] == {
        "path": "tests/data/gguf_tokenizer_artifact_inputs.tar.xz",
        "size": _INPUTS_PATH.stat().st_size,
        "sha256": _sha256(_INPUTS_PATH.read_bytes()),
    }

    budget = fixture["download_budget"]
    selected = sum(route.artifact_size for route in generator.ROUTES)
    assert selected == 15_796_167_008
    assert budget["selected_artifact_bytes"] == selected
    assert budget["limit_bytes"] == 16 * 1024**3
    assert budget["headroom_bytes"] == budget["limit_bytes"] - selected
    assert selected < budget["limit_bytes"]
    assert all(len(route.artifact_revision) == 40 for route in generator.ROUTES)
    assert all(len(route.tokenizer_revision) == 40 for route in generator.ROUTES)

    inputs = generator.load_qualification_inputs(_INPUTS_PATH)
    assert set(inputs) == set(generator._qualification_specs())
    for name, (size, digest) in generator._qualification_specs().items():
        assert len(inputs[name]) == size
        assert _sha256(inputs[name]) == digest


def test_fixture_is_bound_to_artifact_scoped_blocker_records() -> None:
    fixture = _fixture()
    assert fixture["seed"] == generator.SEED
    assert fixture["random_count"] == generator.RANDOM_COUNT
    assert fixture["modes"] == [list(mode) for mode in generator.MODES]
    assert fixture["fixed_inputs"] == list(generator.FIXED_INPUTS)
    assert fixture["route_fixed_inputs"] == {
        name: list(values) for name, values in generator.ROUTE_FIXED_INPUTS.items()
    }

    for route in generator.ROUTES:
        actual = _route_fixture(route.name)
        blocker = tokenizer_blocker_evidence(route.evidence_id)
        assert blocker is not None
        assert actual["evidence_id"] == blocker.evidence_id
        assert actual["identifiers"] == list(blocker.blocked_identifiers)
        assert actual["declared_pre_identifier"] == blocker.pre_identifier
        assert actual["artifact_repository"] == blocker.repository
        assert actual["artifact_revision"] == blocker.revision
        assert actual["artifact_filename"] == blocker.filename
        assert actual["artifact_size"] == blocker.size
        assert actual["artifact_sha256"] == blocker.lfs_sha256
        assert actual["bounded_header_bytes"] == blocker.bounded_header_bytes
        assert actual["bounded_header_sha256"] == blocker.bounded_header_sha256
        assert actual["architecture"] == blocker.architecture
        assert actual["tensor_count"] == blocker.tensor_count
        assert actual["tensor_qtypes"] == [list(item) for item in blocker.tensor_qtypes]
        assert actual["tokenizer_repository"] == blocker.tokenizer_repository
        assert actual["tokenizer_revision"] == blocker.tokenizer_revision
        fixture_assets = tuple(tuple(asset) for asset in actual["tokenizer_assets"])
        assert fixture_assets == tuple(
            (asset.filename, asset.size, asset.sha256) for asset in route.source_assets
        )
        assert blocker.source_config_asset == next(
            asset for asset in fixture_assets if asset[0] == "config.json"
        )
        assert blocker.tokenizer_assets == tuple(
            asset for asset in fixture_assets if asset[0] != "config.json"
        )
        assert actual["tokenizer_metadata_sha256"] == blocker.tokenizer_metadata_sha256
        inventory = actual["inventory"]
        assert inventory["token_count"] == blocker.token_count
        assert inventory["source_token_count"] == blocker.source_token_count
        assert inventory["embedding_vocabulary_size"] == (blocker.embedding_vocabulary_size)
        assert tuple(inventory["deterministic_padding_range"]) == (
            blocker.deterministic_padding_range
        )
        assert inventory["ordered_vocabulary_sha256"] == (blocker.ordered_vocabulary_sha256)
        assert inventory["source_vocabulary_sha256"] == blocker.source_vocabulary_sha256
        assert inventory["merge_count"] == blocker.merge_count
        assert inventory["ordered_merges_sha256"] == blocker.ordered_merges_sha256
        assert inventory["ordered_merges_sha256"] == blocker.source_merges_sha256
        assert inventory["score_count"] == blocker.score_count
        assert inventory["ordered_scores_sha256"] == blocker.ordered_scores_sha256
        assert inventory["ordered_token_types_sha256"] == (blocker.ordered_token_types_sha256)
        assert inventory["ordered_source_added_tokens_sha256"] == (
            blocker.source_added_tokens_sha256
        )
        assert inventory["chat_template_sha256"] == blocker.chat_template_sha256
        assert inventory["normalizer"] == blocker.source_normalizer
        special_tokens = tuple(
            sorted(
                {
                    (token, token_id)
                    for token, token_id in inventory["special_token_ids"].values()
                }
            )
        )
        assert special_tokens == blocker.special_token_ids
        assert actual["case_count"] == blocker.llamacpp_oracle[1]
        assert actual["llamacpp_ordered_results_sha256"] == blocker.llamacpp_oracle[2]
        assert actual["official_source_ordered_results_sha256"] == (
            blocker.source_oracle_sha256
        )
        assert actual["materialized_ordered_results_sha256"] == (
            blocker.materialized_oracle_sha256
        )
        assert (
            actual["official_source_ordered_results_sha256"]
            == (actual["materialized_ordered_results_sha256"])
        )
        assert (
            actual["llamacpp_ordered_results_sha256"]
            != (actual["official_source_ordered_results_sha256"])
        )
        assert actual["dispatch_oracles"] == dict(blocker.dispatch_oracles)
        assert set(actual["dispatch_oracles"].values()) == {
            actual["llamacpp_ordered_results_sha256"]
        }
        assert actual["discriminator"] == {
            "pre_identifier": blocker.dispatch_discriminator[0],
            "mismatch_count": blocker.dispatch_discriminator[1],
            "ordered_results_sha256": blocker.dispatch_discriminator[2],
        }
        assert actual["discriminator"]["mismatch_count"] > 0
        assert (
            actual["discriminator"]["ordered_results_sha256"]
            != (actual["llamacpp_ordered_results_sha256"])
        )
        assert actual["tokenize_mismatch_count"] == blocker.oracle_mismatch_count
        assert tuple(actual["tokenize_mismatch_count_by_mode"]) == (
            blocker.oracle_mismatch_count_by_mode
        )
        first_tokenize = actual["first_tokenize_mismatch"]
        assert blocker.first_mismatch_mode == tuple(first_tokenize["mode"])
        assert blocker.mismatch == (
            first_tokenize["text"],
            tuple(first_tokenize["llamacpp_ids"]),
            tuple(first_tokenize["official_source_ids"]),
        )
        assert actual["detokenize_mismatch_count"] == (
            blocker.oracle_detokenize_mismatch_count
        )
        assert tuple(actual["detokenize_mismatch_count_by_mode"]) == (
            blocker.oracle_detokenize_mismatch_count_by_mode
        )
        first_detokenize = actual["first_detokenize_mismatch"]
        assert blocker.first_detokenize_mismatch == (
            None
            if first_detokenize is None
            else (
                first_detokenize["text"],
                first_detokenize["llamacpp_hex"],
                first_detokenize["official_source_hex"],
            )
        )
        assert inventory["materialized_tokenizer_sha256"] == (
            blocker.materialized_tokenizer_sha256
        )
        assert inventory["pipeline_sha256"] == blocker.source_pipeline_sha256
        assert inventory["tokenizer_config_sha256"] == (blocker.source_tokenizer_config_sha256)
        assert (
            tuple(sorted(inventory["pipeline_component_sha256"].items()))
            == blocker.source_pipeline_component_sha256
        )
        assert inventory["source_added_token_type_mismatch_count"] == (
            blocker.source_added_token_type_mismatch_count
        )
        assert actual["corpus_sha256"] == blocker.oracle_corpus_sha256
        assert actual["corpus_sha256"] == _sha256(
            generator._json_bytes(generator.build_corpus(route.name))
        )
        assert actual["case_count"] == len(generator.MODES) * len(
            generator.build_corpus(route.name)
        )


def test_evidence_captures_minimal_semantic_mismatches() -> None:
    bailing = _route_fixture("bailingmoe")
    assert bailing["inventory"]["normalizer"] == "NFC"
    assert bailing["first_tokenize_mismatch"] == {
        "mode": ["no-add", "no-parse-special"],
        "text": "e\u0301",
        "llamacpp_ids": [68, 150_766],
        "official_source_ids": [2_900],
    }
    assert bailing["inventory"]["source_added_token_type_mismatch_count"] == 0

    glm4 = _route_fixture("glm4")
    assert glm4["first_tokenize_mismatch"]["text"] == "' \u597d"
    assert glm4["first_detokenize_mismatch"] == {
        "mode": ["no-add", "no-parse-special"],
        "text": " .",
        "token_ids": [659],
        "llamacpp_hex": "2e",
        "official_source_hex": "202e",
    }
    assert glm4["inventory"]["source_added_token_type_mismatch_count"] == 7
    assert glm4["inventory"]["source_added_token_type_mismatch_ids"] == [
        154_838,
        154_839,
        154_840,
        154_852,
        154_853,
        154_854,
        154_855,
    ]

    tiny_aya = _route_fixture("tiny_aya")
    assert tiny_aya["first_tokenize_mismatch"]["text"] == "\t 9"
    assert tiny_aya["first_tokenize_mismatch"]["llamacpp_ids"] == [202, 225, 29]
    assert tiny_aya["first_tokenize_mismatch"]["official_source_ids"] == [
        13_396,
        29,
    ]
    assert tiny_aya["inventory"]["source_added_token_type_mismatch_count"] == 23
    assert tiny_aya["detokenize_mismatch_count"] == 0
    assert tiny_aya["first_detokenize_mismatch"] is None


def test_committed_inputs_replay_independent_materialized_identity_and_outputs() -> None:
    inputs = generator.load_qualification_inputs(_INPUTS_PATH)
    for route in generator.ROUTES:
        actual = _route_fixture(route.name)
        metadata = generator._read_header_bytes(
            inputs[f"{route.name}/header.gguf"],
            route,
        )
        source = generator._load_source_assets(
            route,
            {
                asset.filename: inputs[f"{route.name}/source/{asset.filename}"]
                for asset in route.source_assets
            },
        )
        materialized, _ = generator._independent_materialization(metadata, source)
        assert _sha256(materialized) == (actual["inventory"]["materialized_tokenizer_sha256"])
        assert (
            generator._semantic_inventory(metadata, source, materialized)
            == (actual["inventory"])
        )
        corpus = generator.build_corpus(route.name)
        source_outputs = generator._tokenizer_outputs(
            source["tokenizer.json"],
            corpus,
            metadata,
        )
        materialized_outputs = generator._tokenizer_outputs(
            materialized,
            corpus,
            metadata,
        )
        assert source_outputs == materialized_outputs
        assert (
            generator._results_sha256(source_outputs)
            == (actual["official_source_ordered_results_sha256"])
        )
        assert (
            generator._results_sha256(materialized_outputs)
            == (actual["materialized_ordered_results_sha256"])
        )


def test_production_alias_dispatch_reaches_one_reconstruction_or_rejection() -> None:
    inputs = generator.load_qualification_inputs(_INPUTS_PATH)
    for route in generator.ROUTES:
        actual = _route_fixture(route.name)
        metadata = generator._read_header_bytes(
            inputs[f"{route.name}/header.gguf"],
            route,
        )
        payloads = {
            asset.filename: inputs[f"{route.name}/source/{asset.filename}"]
            for asset in route.source_assets
            if asset.filename != "config.json"
        }
        outcomes = []
        for identifier in route.identifiers:
            alias_metadata = dict(metadata)
            alias_metadata["tokenizer.ggml.pre"] = identifier
            if route.name == "glm4":
                with pytest.raises(
                    ValueError,
                    match=r"eom_token_id.*eot_token_id|eot_token_id.*eom_token_id",
                ):
                    _tokenizer._validate_pinned_tokenizer(alias_metadata, payloads)
                outcomes.append("fail-closed")
            else:
                digest, materialized = _tokenizer._validate_pinned_tokenizer(
                    alias_metadata,
                    payloads,
                )
                assert digest == actual["inventory"]["materialized_tokenizer_sha256"]
                assert _sha256(materialized) == digest
                outcomes.append(digest)
        assert len(set(outcomes)) == 1


def test_production_padding_reconstruction_drift_fails_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route("bailingmoe")
    inputs = generator.load_qualification_inputs(_INPUTS_PATH)
    metadata = generator._read_header_bytes(
        inputs[f"{route.name}/header.gguf"],
        route,
    )
    payloads = {
        asset.filename: inputs[f"{route.name}/source/{asset.filename}"]
        for asset in route.source_assets
        if asset.filename != "config.json"
    }
    monkeypatch.setattr(_tokenizer, "_reconstruct_missing_added_tokens", lambda *_a: None)
    with pytest.raises(ValueError, match="vocabulary differs"):
        _tokenizer._validate_pinned_tokenizer(metadata, payloads)


def test_archive_identity_and_fixture_bytes_fail_closed(tmp_path: Path) -> None:
    tampered = bytearray(_INPUTS_PATH.read_bytes())
    tampered[len(tampered) // 2] ^= 1
    path = tmp_path / "tampered.tar.xz"
    path.write_bytes(tampered)
    with pytest.raises(ValueError, match="compressed identity differs"):
        generator.load_qualification_inputs(path)

    canonical = generator._render_fixture(_fixture())
    assert _FIXTURE_PATH.read_bytes() == canonical
    assert b"\r\n" not in canonical
