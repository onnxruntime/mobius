# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Registry-derived, machine-readable disposition of every GGUF tokenizer route."""

from __future__ import annotations

__all__ = ["GGUFTokenizerRouteAudit", "tokenizer_route_census"]

import dataclasses
import functools
from typing import Literal

from mobius.integrations.gguf._tokenizer_alias_evidence import tokenizer_alias_evidence
from mobius.integrations.gguf._tokenizer_evidence import (
    GGUFTokenizerEvidence,
    iter_tokenizer_evidence,
)
from mobius.integrations.gguf._tokenizer_registry import tokenizer_pre_policies

TokenizerAuditStatus = Literal[
    "validated-pinned-source",
    "deferred-pinned-artifact-evidence",
    "deferred-compiled-semantics",
]
TokenizerBlocker = Literal[
    "pinned-artifact-source-parity-pending",
    "pinned-candidate-source-merge-mismatch",
    "pinned-candidate-source-token-mismatch",
    "compiled-llama.cpp-semantic-dependency",
]


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFTokenizerRouteAudit:
    """Exact policy, evidence identity, and disposition for one pre identifier."""

    identifier: str
    semantic_group: str
    pre_type: str
    default_policy: str
    current_status: TokenizerAuditStatus
    evidence_id: str | None
    artifact_repository: str | None
    artifact_revision: str | None
    artifact_filename: str | None
    artifact_size: int | None
    artifact_sha256: str | None
    tokenizer_repository: str | None
    tokenizer_revision: str | None
    tokenizer_assets: tuple[tuple[str, int, str], ...]
    blocker_category: TokenizerBlocker | None
    candidate_disposition: str | None


_PINNED_CANDIDATE_DISPOSITIONS = {
    "jina-v1-en": GGUFTokenizerRouteAudit(
        identifier="jina-v1-en",
        semantic_group="jina-v1-en",
        pre_type="GPT2_ADD_SEP",
        default_policy="deferred",
        current_status="deferred-pinned-artifact-evidence",
        evidence_id=None,
        artifact_repository="gpustack/jina-reranker-v1-tiny-en-GGUF",
        artifact_revision="34fdafe5a08b64246bcbfdbf0b8a23f818baf8e3",
        artifact_filename="jina-reranker-v1-tiny-en-Q2_K.gguf",
        artifact_size=31_645_024,
        artifact_sha256="dbd88c851aaf373569d38e25d34203f8e7ab17a899f767f1f035245cb00b1188",
        tokenizer_repository="jinaai/jina-reranker-v1-tiny-en",
        tokenizer_revision="aca45de6945b5dc6399abcd2a9c55ded5dc9111f",
        tokenizer_assets=(
            (
                "config.json",
                1_206,
                "dc70646aa6c9e75e3c513cc9c037f35ad54308001c3961d45c0f69749bcfb022",
            ),
            (
                "special_tokens_map.json",
                280,
                "06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f",
            ),
            (
                "tokenizer.json",
                2_030_772,
                "0046da43cc8c424b317f56b092b0512aaaa65c4f925d2f16af9d9eeb4d0ef902",
            ),
            (
                "tokenizer_config.json",
                1_215,
                "d291c6652d96d56ffdbcf1ea19d9bae5ed79003f7648c627e725a619227ce8fa",
            ),
        ),
        blocker_category="pinned-candidate-source-token-mismatch",
        candidate_disposition=(
            "ordered token id 5 differs: GGUF is empty while the official tokenizer is "
            "U+0000; deterministic padding starts only at id 60516"
        ),
    ),
    "talkie": GGUFTokenizerRouteAudit(
        identifier="talkie",
        semantic_group="gpt-4o",
        pre_type="GPT4O",
        default_policy="deferred",
        current_status="deferred-pinned-artifact-evidence",
        evidence_id=None,
        artifact_repository="PocketAiHub/talkie-1930-13b-it-GGUF",
        artifact_revision="47b38329dd30e8b2d6ab8e2fc53f3f2ae789e694",
        artifact_filename="talkie-1930-13b-it-Q4_K_M.gguf",
        artifact_size=8_571_072_704,
        artifact_sha256="2d6c6c1d98a1b8ffa38b50916454891a31ad844ee69c686e525976867917d7b2",
        tokenizer_repository="lewtun/talkie-1930-13b-it-hf",
        tokenizer_revision="6311dedf518470856a8503f2080bb4b54fcb3323",
        tokenizer_assets=(
            (
                "chat_template.jinja",
                343,
                "833a35215bfc10d1d9f27fb857123cc24bfef90f770fbc8d79ce37bf4ef4bc4",
            ),
            (
                "config.json",
                522,
                "e7f29da9cf0a69571d6a0521cd912dc5c2f0dd151d0e934b87541f4389a9ee30",
            ),
            (
                "tokenizer.json",
                8_870_742,
                "cc3813d9d674cf0e86e4171579ba276879c66c2171d993e5776fc5615756a03b",
            ),
            (
                "tokenizer_config.json",
                247,
                "e12d422a980eceaecd6ff388c3843b30dd461307d58ec19585953012d7386fc5",
            ),
        ),
        blocker_category="pinned-candidate-source-merge-mismatch",
        candidate_disposition=(
            "transformed GGUF retains 65279 of 156379 official merges; first ordered "
            "mismatch at index 4 is ('Ġ', 'the') versus ('Ġt', 'he')"
        ),
    ),
}


def _evidenced_route(
    identifier: str, evidence: GGUFTokenizerEvidence
) -> GGUFTokenizerRouteAudit:
    policy = tokenizer_pre_policies()[identifier]
    return GGUFTokenizerRouteAudit(
        identifier=policy.identifier,
        semantic_group=policy.canonical,
        pre_type=policy.pre_type,
        default_policy=policy.default_route,
        current_status="validated-pinned-source",
        evidence_id=evidence.evidence_id,
        artifact_repository=evidence.repository,
        artifact_revision=evidence.revision,
        artifact_filename=evidence.filename,
        artifact_size=evidence.size,
        artifact_sha256=evidence.lfs_sha256,
        tokenizer_repository=evidence.tokenizer_repository,
        tokenizer_revision=evidence.tokenizer_revision,
        tokenizer_assets=tuple(
            sorted((evidence.source_config_asset, *evidence.tokenizer_assets))
        ),
        blocker_category=None,
        candidate_disposition=None,
    )


@functools.lru_cache(maxsize=1)
def tokenizer_route_census() -> tuple[GGUFTokenizerRouteAudit, ...]:
    """Return all exact routes, including aliases, in identifier order."""
    evidenced = {
        identifier: _evidenced_route(identifier, evidence)
        for evidence in iter_tokenizer_evidence()
        for identifier in evidence.validated_identifiers
    }
    expected_count = sum(
        len(evidence.validated_identifiers) for evidence in iter_tokenizer_evidence()
    )
    if len(evidenced) != expected_count:
        raise RuntimeError("Tokenizer evidence contains duplicate validated identifiers")

    records = []
    alias_proofs = tokenizer_alias_evidence()
    for identifier, policy in sorted(tokenizer_pre_policies().items()):
        record = evidenced.get(identifier)
        if record is None:
            record = _PINNED_CANDIDATE_DISPOSITIONS.get(identifier)
        if record is None:
            dispatch_proven = identifier in alias_proofs
            record = GGUFTokenizerRouteAudit(
                identifier=identifier,
                semantic_group=policy.canonical,
                pre_type=policy.pre_type,
                default_policy=policy.default_route,
                current_status=(
                    "deferred-pinned-artifact-evidence"
                    if dispatch_proven
                    else "deferred-compiled-semantics"
                ),
                evidence_id=None,
                artifact_repository=None,
                artifact_revision=None,
                artifact_filename=None,
                artifact_size=None,
                artifact_sha256=None,
                tokenizer_repository=None,
                tokenizer_revision=None,
                tokenizer_assets=(),
                blocker_category=(
                    "pinned-artifact-source-parity-pending"
                    if dispatch_proven
                    else "compiled-llama.cpp-semantic-dependency"
                ),
                candidate_disposition=None,
            )
        records.append(record)
    return tuple(records)
