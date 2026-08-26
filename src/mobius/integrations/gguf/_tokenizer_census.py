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
            )
        records.append(record)
    return tuple(records)
