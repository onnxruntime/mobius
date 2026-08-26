# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for compact, network-free exact tokenizer evidence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from mobius.integrations.gguf import _tokenizer_evidence
from mobius.integrations.gguf._runtime_evidence import GGUFArtifactIdentity, runtime_evidence
from mobius.integrations.gguf._tokenizer_alias_evidence import (
    TOKENIZER_DISPATCH_SOURCE_PATH,
    TOKENIZER_DISPATCH_SOURCE_SHA256,
    tokenizer_alias_evidence,
)
from mobius.integrations.gguf._tokenizer_census import tokenizer_route_census
from mobius.integrations.gguf._tokenizer_evidence import (
    iter_tokenizer_blocker_evidence,
    iter_tokenizer_evidence,
    matching_tokenizer_evidence,
    tokenizer_blocker_evidence,
    tokenizer_evidence,
)


def test_qwen35_tokenizer_evidence_is_exact_and_runtime_independent() -> None:
    evidence = tokenizer_evidence("qwen3.5-0.8b-q4-tokenizer")
    assert evidence is not None
    assert evidence.revision == "8fea620810c4afa23dd6443f999a48574c1611a3"
    assert evidence.size == 563_036_064
    assert evidence.lfs_sha256 == (
        "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf"
    )
    assert evidence.tokenizer_revision == "2fc06364715b967f1860aea9cf38778875588b17"
    assert evidence.token_count == 248_320
    assert evidence.source_token_count == 248_077
    assert evidence.deterministic_padding_range == (248_077, 248_319)
    assert evidence.embedding_vocabulary_size == 248_320
    assert evidence.merge_count == 247_587
    assert evidence.ordered_token_types_sha256 == (
        "f6fdca1063d1ae1cc77ba1f5087d259f044c2634e64b65e31bc844ec00e9acab"
    )
    assert evidence.materialized_tokenizer_sha256 == (
        "a78b900eb4cd335bba249158066db523ce221f744e2b6144692bb81673d551af"
    )
    assert runtime_evidence("qwen3.5-0.8b-q4-tokenizer") is None


def test_first_tokenizer_evidence_batch_is_complete_and_artifact_scoped() -> None:
    records = iter_tokenizer_evidence()
    assert [(record.pre_identifier, record.architecture) for record in records] == [
        ("gpt-2", "gpt2"),
        ("jina-v2-code", "jina-bert-v2"),
        ("kanana2", "qwen3"),
        ("lfm2", "lfm2"),
        ("qwen2", "qwen2"),
        ("qwen35", "qwen35"),
        ("roberta-bpe", "bert"),
        ("smollm", "llama"),
        ("talkie", "talkie"),
    ]
    assert all(record.token_count > 0 for record in records)
    assert all(record.ordered_token_types_sha256 for record in records)
    assert all(record.representative_encodings for record in records)
    assert all(
        record.representative_special_encodings
        for record in records
        if record.pre_identifier in {"jina-v2-code", "roberta-bpe"}
    )
    assert all(record.source_config_asset[0] == "config.json" for record in records)


def test_plm_tokenizer_blocker_is_exact_and_architecture_scoped() -> None:
    blockers = iter_tokenizer_blocker_evidence()
    assert [record.evidence_id for record in blockers] == [
        "plm-1.8b-instruct-q4-k-m-tokenizer-blocker"
    ]
    blocker = blockers[0]
    assert blocker.architecture == "plm"
    assert blocker.pre_identifier == "qwen2"
    assert blocker.revision == "7bec6546983bcf0d99526c943580bd49e2237445"
    assert blocker.lfs_sha256 == (
        "b38570ee56ebec82a1e9ef45ab408c0d8230ececef1d7f1b267c49cff35638b8"
    )
    assert blocker.tokenizer_revision == "62d188c7d58843d7013d5b3ffe198db448787860"
    assert blocker.token_count == blocker.embedding_vocabulary_size == 151_936
    assert blocker.source_token_count == 151_646
    assert blocker.deterministic_padding_range == (151_646, 151_935)
    assert blocker.merge_count == 151_387
    assert blocker.score_count == 0
    assert blocker.ordered_scores_sha256 is None
    assert blocker.source_normalizer == "NFC"
    assert blocker.mismatch == ("é é", (68, 53839, 3958), (963, 3958))
    assert "NFC normalization" in blocker.disposition


def test_plm_llamacpp_oracle_fixture_is_bound_to_fail_closed_evidence() -> None:
    path = Path(__file__).parents[4] / "tests/data/gguf_plm_qwen2_tokenizer_blocker.json"
    oracle = json.loads(path.read_text(encoding="utf-8"))
    blocker = tokenizer_blocker_evidence("plm-1.8b-instruct-q4-k-m-tokenizer-blocker")
    assert blocker is not None
    assert blocker.llamacpp_oracle == (
        oracle["llamacpp_commit"],
        oracle["case_count"],
        oracle["ordered_results_sha256"],
    )
    assert blocker.lfs_sha256 == oracle["artifact_sha256"]
    assert blocker.tokenizer_revision == oracle["tokenizer_revision"]
    assert oracle["case_count"] == len(oracle["modes"]) * len(oracle["fixed_inputs"])
    assert oracle["mismatch_count"] == len(oracle["mismatch"]["modes"])
    assert blocker.source_pipeline_sha256 == oracle["source_pipeline_sha256"]
    assert blocker.chat_template_sha256 == oracle["chat_template_sha256"]
    assert blocker.source_normalizer == oracle["source_normalizer"]
    assert blocker.mismatch == (
        oracle["mismatch"]["text"],
        tuple(oracle["mismatch"]["llamacpp_ids"]),
        tuple(oracle["mismatch"]["official_source_ids"]),
    )
    assert oracle["source_normalizer"] == "NFC"
    assert oracle["default_add_bos_matches_no_add"]
    assert not oracle["scores_present"]


def test_evidence_schema_accepts_scores_instead_of_bpe_merges() -> None:
    evidence = tokenizer_evidence("qwen3.5-0.8b-q4-tokenizer")
    assert evidence is not None
    scores_only = dataclasses.replace(
        evidence,
        merge_count=0,
        ordered_merges_sha256=None,
        score_count=evidence.token_count,
        ordered_scores_sha256="a" * 64,
    )
    assert scores_only.ordered_merges_sha256 is None
    with pytest.raises(ValueError, match="ordered merges or ordered scores"):
        dataclasses.replace(
            evidence,
            merge_count=0,
            ordered_merges_sha256=None,
        )


def test_gpt4o_evidence_requires_oracle_at_dispatch_commit() -> None:
    evidence = tokenizer_evidence("talkie-13b-q4-native-tokenizer")
    assert evidence is not None
    with pytest.raises(ValueError, match=r"requires an exact pinned llama\.cpp oracle"):
        dataclasses.replace(evidence, llamacpp_oracle=None)
    with pytest.raises(ValueError, match="oracle evidence identity is invalid"):
        dataclasses.replace(
            evidence,
            llamacpp_oracle=("0" * 40, 444, evidence.llamacpp_oracle[2]),
        )


def test_registry_derived_census_has_a_concrete_disposition_for_every_alias() -> None:
    census = tokenizer_route_census()
    assert len(census) == 87
    assert len({record.identifier for record in census}) == 87
    statuses = {
        status: sum(record.current_status == status for record in census)
        for status in {record.current_status for record in census}
    }
    assert statuses == {
        "deferred-compiled-semantics": 45,
        "deferred-pinned-artifact-evidence": 12,
        "validated-pinned-source": 30,
    }
    for record in census:
        assert (record.evidence_id is None) == (record.blocker_category is not None)
        if record.evidence_id is None:
            assert (record.artifact_repository is None) == (
                record.candidate_disposition is None
            )
            assert (record.tokenizer_repository is None) == (
                record.candidate_disposition is None
            )
        else:
            assert record.artifact_revision is not None
            assert record.tokenizer_revision is not None
            assert record.tokenizer_assets
            assert record.candidate_disposition is None

    jina_v1 = next(record for record in census if record.identifier == "jina-v1-en")
    assert jina_v1.blocker_category == "pinned-candidate-source-token-mismatch"
    assert jina_v1.artifact_revision == "34fdafe5a08b64246bcbfdbf0b8a23f818baf8e3"
    assert jina_v1.artifact_sha256 == (
        "dbd88c851aaf373569d38e25d34203f8e7ab17a899f767f1f035245cb00b1188"
    )
    assert jina_v1.tokenizer_revision == "aca45de6945b5dc6399abcd2a9c55ded5dc9111f"
    assert jina_v1.candidate_disposition is not None
    assert "token id 5" in jina_v1.candidate_disposition

    gpt4o = next(record for record in census if record.identifier == "gpt-4o")
    assert gpt4o.blocker_category == "pinned-candidate-identifier-mismatch"
    assert gpt4o.artifact_revision == "41c1d48055e3192a907c0ffc2a886288e9040e33"
    assert gpt4o.artifact_sha256 == (
        "e14b14b73e0f7f35b234df7c5f1a585869d1fb3634331d489b4184b61cec5d29"
    )
    assert gpt4o.tokenizer_revision == "7956d98f2a83b2751a98ea7136fdf7fe6cf54e69"
    assert gpt4o.candidate_disposition is not None
    assert "dispatches llama-bpe, not gpt-4o" in gpt4o.candidate_disposition

    llama4 = next(record for record in census if record.identifier == "llama4")
    assert llama4.blocker_category == "pinned-candidate-incomplete-shard"
    assert llama4.artifact_revision == "42675345da11ade9203a5187595da7b74d4ff2ac"
    assert llama4.artifact_size == 15_511_520_608
    assert llama4.artifact_sha256 == (
        "53d9a61b90e38330daa4bb07afe56aa3e74a3d3aa31d344c053ebbdcfe5d59fe"
    )
    assert llama4.tokenizer_revision == "92f3b1597a195b523d8d9e5700e57e4fbb8f20d3"
    assert llama4.candidate_disposition is not None
    assert "145 of 628 tensors" in llama4.candidate_disposition

    talkie = tokenizer_evidence("talkie-13b-q4-native-tokenizer")
    assert talkie is not None
    assert talkie.reconstruct_gpt4o_from_gguf
    assert talkie.revision == "47b38329dd30e8b2d6ab8e2fc53f3f2ae789e694"
    assert talkie.size == 8_571_072_704
    assert talkie.lfs_sha256 == (
        "2d6c6c1d98a1b8ffa38b50916454891a31ad844ee69c686e525976867917d7b2"
    )
    assert talkie.tokenizer_revision == "6311dedf518470856a8503f2080bb4b54fcb3323"
    assert talkie.source_disposition is not None
    assert "65279 of 156379 source merges" in talkie.source_disposition
    assert talkie.llamacpp_oracle == (
        "8d9af256337d1a501250f9bbf4c0859a654bddd6",
        444,
        "484246b629d6eec375ebac3672e4f4d4fb29646d3b331917ec4d2cfe385c3b6a",
    )

    kanana2 = tokenizer_evidence("kanana2-1.3b-instruct-q8-tokenizer")
    assert kanana2 is not None
    assert kanana2.validated_identifiers == ("kanana2",)
    assert kanana2.token_count == kanana2.source_token_count == 128_256
    assert kanana2.embedding_vocabulary_size == 128_256
    assert kanana2.representative_special_encodings[0][1][0] == 128_000
    assert kanana2.llamacpp_oracle == (
        "8d9af256337d1a501250f9bbf4c0859a654bddd6",
        444,
        "ca7875445f21a03eb9a480c6aa96251bf4a8951a6e284dc480ef32eaedb796f5",
    )


def test_batch2_alias_fixture_matches_dispatch_proof_and_census() -> None:
    path = Path(__file__).parents[4] / "tests/data/gguf_tokenizer_alias_batch2.json"
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert len(expected) == 38
    assert len({row[0] for row in expected}) == 38
    census = {record.identifier: record for record in tokenizer_route_census()}
    proofs = tokenizer_alias_evidence()

    actual = [
        [
            identifier,
            census[identifier].semantic_group,
            census[identifier].pre_type,
            census[identifier].evidence_id,
        ]
        for identifier, *_ in expected
    ]
    assert actual == expected
    assert all(identifier in proofs for identifier, *_ in expected)
    assert {row[0] for row in expected if row[3] is not None} == {
        identifier
        for evidence in iter_tokenizer_evidence()
        for identifier in evidence.validated_identifiers
        if identifier in {row[0] for row in expected}
    }
    assert {proof.source_path for proof in proofs.values()} == {TOKENIZER_DISPATCH_SOURCE_PATH}
    assert {proof.source_sha256 for proof in proofs.values()} == {
        TOKENIZER_DISPATCH_SOURCE_SHA256
    }


def test_gpt4o_oracle_fixture_is_bound_to_talkie_evidence() -> None:
    path = Path(__file__).parents[4] / "tests/data/gguf_gpt4o_oracle.json"
    oracle = json.loads(path.read_text(encoding="utf-8"))
    evidence = tokenizer_evidence("talkie-13b-q4-native-tokenizer")
    assert evidence is not None
    assert evidence.llamacpp_oracle == (
        oracle["llamacpp_commit"],
        oracle["case_count"],
        oracle["ordered_results_sha256"],
    )
    assert evidence.lfs_sha256 == oracle["artifact_sha256"]
    assert oracle["seed"] == 648
    assert oracle["random_count"] == 128
    assert len(oracle["fixed_inputs"]) == 20
    assert oracle["case_count"] == (
        len(oracle["modes"]) * (len(oracle["fixed_inputs"]) + oracle["random_count"])
    )


def test_gpt4o_kanana2_oracle_fixture_is_bound_to_evidence() -> None:
    path = Path(__file__).parents[4] / "tests/data/gguf_gpt4o_kanana2_oracle.json"
    oracle = json.loads(path.read_text(encoding="utf-8"))
    evidence = tokenizer_evidence("kanana2-1.3b-instruct-q8-tokenizer")
    assert evidence is not None
    assert evidence.llamacpp_oracle == (
        oracle["llamacpp_commit"],
        oracle["case_count"],
        oracle["ordered_results_sha256"],
    )
    assert evidence.lfs_sha256 == oracle["artifact_sha256"]
    assert oracle["seed"] == 648
    assert oracle["random_count"] == 128
    assert oracle["fixed_count"] == 20
    assert oracle["case_count"] == len(oracle["modes"]) * (
        oracle["fixed_count"] + oracle["random_count"]
    )


def test_shared_evidence_requires_identical_pinned_dispatch_behavior() -> None:
    evidence = tokenizer_evidence("gpt2-q4-tokenizer")
    assert evidence is not None
    with pytest.raises(ValueError, match="exact pinned semantic group"):
        dataclasses.replace(
            evidence,
            validated_identifiers=("gpt-2", "qwen2"),
        )


def test_gpt2_add_sep_evidence_has_exact_special_token_witnesses() -> None:
    expected = {
        "jina-v2-code-q8-tokenizer": (0, 10564, 16, 7550, 5, 53737, 2),
        "roberta-bpe-q2-tokenizer": (0, 31414, 6, 232, 328, 17072, 1898, 2),
    }
    for evidence_id, token_ids in expected.items():
        evidence = tokenizer_evidence(evidence_id)
        assert evidence is not None
        assert evidence.representative_special_encodings == (
            ("Hello, world! 12345", token_ids),
        )
        assert evidence.special_token_ids[0] == ("</s>", 2)


def test_existing_qwen25_runtime_tokenizer_evidence_remains_pinned() -> None:
    evidence = runtime_evidence("qwen2.5-0.5b-instruct-q8-ort-genai-0.15.2")
    assert evidence is not None
    assert evidence.tokenizer_repository == "Qwen/Qwen2.5-0.5B-Instruct"
    assert evidence.tokenizer_revision == "a338b55dd21219a5f4da42bc11a9313d1a27d4cc"
    assert evidence.tokenizer_metadata_sha256 == (
        "8fc8ef848104e931f14ae03d9581699d54813a2ff952fb7caac0654e8aa27ee3"
    )


def _digest(values: list[object]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _small_evidence():
    evidence = tokenizer_evidence("qwen3.5-0.8b-q4-tokenizer")
    assert evidence is not None
    return dataclasses.replace(
        evidence,
        filename="tiny.gguf",
        size=4,
        lfs_sha256="a" * 64,
        tensor_count=1,
        tensor_qtypes=(("F32", 1),),
        token_count=2,
        source_token_count=2,
        embedding_vocabulary_size=2,
        deterministic_padding_range=(2, 1),
        ordered_vocabulary_sha256=_digest(["a", "<special>"]),
        merge_count=1,
        ordered_merges_sha256=_digest(["a <special>"]),
        ordered_token_types_sha256=_digest([1, 3]),
        special_token_ids=(("<special>", 1),),
    )


def _small_model():
    return SimpleNamespace(
        architecture="qwen35",
        metadata={
            "tokenizer.ggml.pre": "qwen35",
            "tokenizer.ggml.tokens": ["a", "<special>"],
            "tokenizer.ggml.merges": ["a <special>"],
            "tokenizer.ggml.token_type": [1, 3],
        },
        get_tensor_shape=lambda _name: (2, 4),
    )


def test_matching_evidence_binds_ordered_semantics_and_embedding_rows(
    tmp_path, monkeypatch
) -> None:
    evidence = _small_evidence()
    monkeypatch.setattr(
        _tokenizer_evidence,
        "_TOKENIZER_EVIDENCE",
        MappingProxyType({evidence.evidence_id: evidence}),
    )
    monkeypatch.setattr(
        _tokenizer_evidence,
        "gguf_artifact_identity",
        lambda *_a, **_k: GGUFArtifactIdentity(
            "qwen35",
            "tiny.gguf",
            4,
            "a" * 64,
            1,
            (("F32", 1),),
        ),
    )

    assert (
        matching_tokenizer_evidence(
            tmp_path / "tiny.gguf",
            _small_model(),
            metadata_sha256=evidence.tokenizer_metadata_sha256,
        )
        is evidence
    )


@pytest.mark.parametrize(
    "mismatch",
    ["metadata", "pre", "vocabulary", "merges", "scores", "token_types", "special", "rows"],
)
def test_matching_evidence_fails_closed_on_compact_identity_mismatch(
    tmp_path, monkeypatch, mismatch
) -> None:
    evidence = _small_evidence()
    model = _small_model()
    metadata_sha256 = evidence.tokenizer_metadata_sha256
    if mismatch == "metadata":
        metadata_sha256 = "b" * 64
    elif mismatch == "pre":
        model.metadata["tokenizer.ggml.pre"] = "qwen2"
    elif mismatch == "vocabulary":
        model.metadata["tokenizer.ggml.tokens"][0] = "changed"
    elif mismatch == "merges":
        model.metadata["tokenizer.ggml.merges"][0] = "<special> a"
    elif mismatch == "scores":
        model.metadata["tokenizer.ggml.scores"] = [0.0, 0.0]
    elif mismatch == "token_types":
        model.metadata["tokenizer.ggml.token_type"][0] = 2
    elif mismatch == "special":
        model.metadata["tokenizer.ggml.tokens"][1] = "<changed>"
    else:
        model.get_tensor_shape = lambda _name: (3, 4)
    monkeypatch.setattr(
        _tokenizer_evidence,
        "_TOKENIZER_EVIDENCE",
        MappingProxyType({evidence.evidence_id: evidence}),
    )
    monkeypatch.setattr(
        _tokenizer_evidence,
        "gguf_artifact_identity",
        lambda *_a, **_k: GGUFArtifactIdentity(
            "qwen35",
            "tiny.gguf",
            4,
            "a" * 64,
            1,
            (("F32", 1),),
        ),
    )

    with pytest.raises(ValueError, match="No unique exact tokenizer evidence"):
        matching_tokenizer_evidence(
            tmp_path / "tiny.gguf",
            model,
            metadata_sha256=metadata_sha256,
        )


def test_matching_plm_blocker_reports_exact_normalizer_mismatch(tmp_path, monkeypatch) -> None:
    blocker = tokenizer_blocker_evidence("plm-1.8b-instruct-q4-k-m-tokenizer-blocker")
    assert blocker is not None
    tiny = dataclasses.replace(
        blocker,
        filename="tiny.gguf",
        size=4,
        lfs_sha256="a" * 64,
        tensor_count=1,
        tensor_qtypes=(("F32", 1),),
        token_count=2,
        source_token_count=2,
        embedding_vocabulary_size=2,
        deterministic_padding_range=(2, 1),
        ordered_vocabulary_sha256=_digest(["a", "<special>"]),
        source_vocabulary_sha256=_digest(["a", "<special>"]),
        merge_count=1,
        ordered_merges_sha256=_digest(["a <special>"]),
        source_merges_sha256=_digest([["a", "<special>"]]),
        ordered_token_types_sha256=_digest([1, 3]),
        special_token_ids=(("<special>", 1),),
        mismatch=("é", (0,), (1,)),
    )
    monkeypatch.setattr(_tokenizer_evidence, "_TOKENIZER_EVIDENCE", MappingProxyType({}))
    monkeypatch.setattr(
        _tokenizer_evidence,
        "_TOKENIZER_BLOCKER_EVIDENCE",
        MappingProxyType({tiny.evidence_id: tiny}),
    )
    monkeypatch.setattr(
        _tokenizer_evidence,
        "gguf_artifact_identity",
        lambda *_a, **_k: GGUFArtifactIdentity(
            "plm",
            "tiny.gguf",
            4,
            "a" * 64,
            1,
            (("F32", 1),),
        ),
    )
    model = SimpleNamespace(
        architecture="plm",
        metadata={
            "tokenizer.ggml.pre": "qwen2",
            "tokenizer.ggml.tokens": ["a", "<special>"],
            "tokenizer.ggml.merges": ["a <special>"],
            "tokenizer.ggml.token_type": [1, 3],
        },
        get_tensor_shape=lambda _name: (2, 4),
    )

    with pytest.raises(ValueError, match=r"explicitly blocked.*NFC normalization"):
        matching_tokenizer_evidence(
            tmp_path / "tiny.gguf",
            model,
            metadata_sha256=tiny.tokenizer_metadata_sha256,
        )
