# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for compact, network-free exact tokenizer evidence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from types import MappingProxyType, SimpleNamespace

import pytest

from mobius.integrations.gguf import _tokenizer_evidence
from mobius.integrations.gguf._runtime_evidence import GGUFArtifactIdentity, runtime_evidence
from mobius.integrations.gguf._tokenizer_census import tokenizer_route_census
from mobius.integrations.gguf._tokenizer_evidence import (
    iter_tokenizer_evidence,
    matching_tokenizer_evidence,
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
        ("lfm2", "lfm2"),
        ("qwen2", "qwen2"),
        ("qwen35", "qwen35"),
        ("smollm", "llama"),
    ]
    assert all(record.token_count > 0 for record in records)
    assert all(record.ordered_token_types_sha256 for record in records)
    assert all(record.representative_encodings for record in records)
    assert all(record.source_config_asset[0] == "config.json" for record in records)


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


def test_registry_derived_census_has_a_concrete_disposition_for_every_alias() -> None:
    census = tokenizer_route_census()
    assert len(census) == 87
    assert len({record.identifier for record in census}) == 87
    statuses = {
        status: sum(record.current_status == status for record in census)
        for status in {record.current_status for record in census}
    }
    assert statuses == {
        "deferred-compiled-semantics": 83,
        "validated-pinned-source": 4,
    }
    for record in census:
        assert (record.evidence_id is None) == (record.blocker_category is not None)
        if record.evidence_id is None:
            assert record.artifact_repository is None
            assert record.tokenizer_repository is None
        else:
            assert record.artifact_revision is not None
            assert record.tokenizer_revision is not None
            assert record.tokenizer_assets


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
