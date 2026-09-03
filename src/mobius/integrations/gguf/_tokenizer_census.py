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
    GGUFTokenizerBlockerEvidence,
    GGUFTokenizerEvidence,
    iter_tokenizer_blocker_evidence,
    iter_tokenizer_evidence,
)
from mobius.integrations.gguf._tokenizer_registry import tokenizer_pre_policies

TokenizerAuditStatus = Literal[
    "validated-pinned-source",
    "deferred-pinned-artifact-evidence",
    "deferred-pinned-artifact-mismatch",
    "deferred-compiled-semantics",
]
TokenizerBlocker = Literal[
    "pinned-artifact-source-parity-pending",
    "pinned-candidate-identifier-mismatch",
    "pinned-candidate-effective-pre-mismatch",
    "pinned-candidate-incomplete-shard",
    "pinned-candidate-source-merge-mismatch",
    "pinned-candidate-source-semantic-mismatch",
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
    artifact_architecture: str | None = None
    declared_pre_identifier: str | None = None
    effective_pre_identifier: str | None = None
    blocker_evidence_id: str | None = None


_PINNED_CANDIDATE_DISPOSITIONS = {
    "gpt-4o": GGUFTokenizerRouteAudit(
        identifier="gpt-4o",
        semantic_group="gpt-4o",
        pre_type="GPT4O",
        default_policy="deferred",
        current_status="deferred-pinned-artifact-evidence",
        evidence_id=None,
        artifact_repository="mradermacher/oh-dcft-v3.1-gpt-4o-mini-GGUF",
        artifact_revision="41c1d48055e3192a907c0ffc2a886288e9040e33",
        artifact_filename="oh-dcft-v3.1-gpt-4o-mini.Q2_K.gguf",
        artifact_size=3_179_132_928,
        artifact_sha256="e14b14b73e0f7f35b234df7c5f1a585869d1fb3634331d489b4184b61cec5d29",
        tokenizer_repository="Xenova/gpt-4o",
        tokenizer_revision="7956d98f2a83b2751a98ea7136fdf7fe6cf54e69",
        tokenizer_assets=(
            (
                "special_tokens_map.json",
                98,
                "7003e1e385ae2f4b32ca9ca8637c352553922adad120266cf82238155a21dd16",
            ),
            (
                "tokenizer.json",
                9_729_051,
                "43a3ad4618a6a938f8c2614154ea10c31ad53a62d5683ae0b0e6133575cef07e",
            ),
            (
                "tokenizer_config.json",
                236,
                "71424d4750066a6f9bcfea0699576c4284327ff25d388d42419506bcfb5535ab",
            ),
        ),
        blocker_category="pinned-candidate-identifier-mismatch",
        candidate_disposition=(
            "name-only candidate rejected: its complete GGUF header dispatches llama-bpe, "
            "not gpt-4o, and has 128256 tokens; the pinned llama.cpp fingerprint source has "
            "200000 vocabulary entries plus 2 added tokens and is tokenizer-only, with no "
            "model config, chat template, or embedding rows"
        ),
    ),
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
    "granite-embed-multi-311m": GGUFTokenizerRouteAudit(
        identifier="granite-embed-multi-311m",
        semantic_group="gemma4",
        pre_type="GEMMA4",
        default_policy="deferred",
        current_status="deferred-pinned-artifact-evidence",
        evidence_id=None,
        artifact_repository=(
            "SandLogicTechnologies/granite-embedding-311m-multilingual-r2-GGUF"
        ),
        artifact_revision="f142535239859391fbef67aaf886d96500ad9fa8",
        artifact_filename="granite-embedding-311M-multilingual-r2_IQ4_XS.gguf",
        artifact_size=240_602_528,
        artifact_sha256="4cddb0ecb0ee45fcd1da37c007a608662568477a76ce5e12c14f7b34f002709e",
        tokenizer_repository="ibm-granite/granite-embedding-311m-multilingual-r2",
        tokenizer_revision="44399559930365213510b1ee2eb15ded83374f0e",
        tokenizer_assets=(
            (
                "config.json",
                1_191,
                "e1e3fc842a8e0537e25d6e4c93879698b92ae96722e8c162bef334b57978a3b0",
            ),
            (
                "special_tokens_map.json",
                694,
                "cb9e60dcf4d8d314315cb3e761fe4c2e664fda8dbf66d7815372b2639e381182",
            ),
            (
                "tokenizer.json",
                33_384_821,
                "0087c868b33bad550a78a08d19798cfd7f713cde4f020803b8f51f405503e15f",
            ),
            (
                "tokenizer_config.json",
                1_155_500,
                "7947bdf0378520e69ca412b8c4dacd1cffa8aef099f851fdd5c65aa27c6b36a0",
            ),
        ),
        blocker_category="pinned-candidate-effective-pre-mismatch",
        candidate_disposition=(
            "complete modern-bert artifact rejected for this exact identifier: it declares no "
            "tokenizer.ggml.pre, so pinned llama.cpp uses fallback gemma4 rather than "
            "granite-embed-multi-311m. Its artifact-scoped fallback reconstruction matches 480 "
            "pinned tokenize/detokenize cases. GGUF-only IDs 262145..262151 are type-4 "
            "user-defined tokens; fallback gemma4 promotes only ID 262149 "
            "(<|tool_response>) to control/EOG. This cannot prove the absent identifier"
        ),
        artifact_architecture="modern-bert",
        declared_pre_identifier=None,
        effective_pre_identifier="gemma4",
    ),
    "llama4": GGUFTokenizerRouteAudit(
        identifier="llama4",
        semantic_group="gpt-4o",
        pre_type="GPT4O",
        default_policy="deferred",
        current_status="deferred-pinned-artifact-evidence",
        evidence_id=None,
        artifact_repository="ggml-org/Llama-4-Scout-17B-16E-Instruct-GGUF",
        artifact_revision="42675345da11ade9203a5187595da7b74d4ff2ac",
        artifact_filename="Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00002-of-00002.gguf",
        artifact_size=15_511_520_608,
        artifact_sha256="53d9a61b90e38330daa4bb07afe56aa3e74a3d3aa31d344c053ebbdcfe5d59fe",
        tokenizer_repository="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        tokenizer_revision="92f3b1597a195b523d8d9e5700e57e4fbb8f20d3",
        tokenizer_assets=(
            (
                "tokenizer.json",
                27_948_578,
                "172c9eb4beafc72601690da3ccfcede5c2e6806a8d5ec1fca33e22acea8023a4",
            ),
        ),
        blocker_category="pinned-candidate-incomplete-shard",
        candidate_disposition=(
            "the only pinned Q4_K_M file within 16 GiB is shard 2 of 2; its complete "
            "header has only split metadata, 145 of 628 tensors, no tokenizer fields, and "
            "no embedding tensor, while shard 1 is 49848377344 bytes"
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
        artifact_architecture=evidence.architecture,
        declared_pre_identifier=(
            None if evidence.uses_model_pre_fallback else evidence.pre_identifier
        ),
        effective_pre_identifier=evidence.pre_identifier,
    )


def _blocked_route(
    identifier: str, evidence: GGUFTokenizerBlockerEvidence
) -> GGUFTokenizerRouteAudit:
    policy = tokenizer_pre_policies()[identifier]
    return GGUFTokenizerRouteAudit(
        identifier=policy.identifier,
        semantic_group=policy.canonical,
        pre_type=policy.pre_type,
        default_policy=policy.default_route,
        current_status="deferred-pinned-artifact-mismatch",
        evidence_id=None,
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
        blocker_category="pinned-candidate-source-semantic-mismatch",
        candidate_disposition=evidence.disposition,
        artifact_architecture=evidence.architecture,
        declared_pre_identifier=evidence.pre_identifier,
        effective_pre_identifier=evidence.pre_identifier,
        blocker_evidence_id=evidence.evidence_id,
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
    blocked = {
        identifier: _blocked_route(identifier, evidence)
        for evidence in iter_tokenizer_blocker_evidence()
        for identifier in evidence.blocked_identifiers
    }
    expected_blocked_count = sum(
        len(evidence.blocked_identifiers) for evidence in iter_tokenizer_blocker_evidence()
    )
    if len(blocked) != expected_blocked_count:
        raise RuntimeError("Tokenizer blocker evidence contains duplicate identifiers")
    if set(evidenced) & set(blocked):
        raise RuntimeError("Tokenizer identifiers cannot be both validated and blocked")

    records = []
    alias_proofs = tokenizer_alias_evidence()
    for identifier, policy in sorted(tokenizer_pre_policies().items()):
        record = evidenced.get(identifier) or blocked.get(identifier)
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
