# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Registry-derived, machine-readable disposition of every GGUF tokenizer route."""

from __future__ import annotations

__all__ = ["GGUFTokenizerRouteAudit", "tokenizer_route_census"]

import dataclasses
import functools
from typing import Literal

from mobius.integrations.gguf._tokenizer_evidence import (
    GGUFTokenizerEvidence,
    iter_tokenizer_evidence,
)
from mobius.integrations.gguf._tokenizer_registry import tokenizer_pre_policies

TokenizerAuditStatus = Literal["validated-pinned-source", "deferred-compiled-semantics"]
TokenizerBlocker = Literal["compiled-llama.cpp-semantic-dependency"]


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


def _evidenced_route(evidence: GGUFTokenizerEvidence) -> GGUFTokenizerRouteAudit:
    policy = tokenizer_pre_policies()[evidence.pre_identifier]
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
        evidence.pre_identifier: _evidenced_route(evidence)
        for evidence in iter_tokenizer_evidence()
    }
    if len(evidenced) != len(iter_tokenizer_evidence()):
        raise RuntimeError("Tokenizer evidence contains duplicate exact pre identifiers")

    records = []
    for identifier, policy in sorted(tokenizer_pre_policies().items()):
        record = evidenced.get(identifier)
        if record is None:
            record = GGUFTokenizerRouteAudit(
                identifier=identifier,
                semantic_group=policy.canonical,
                pre_type=policy.pre_type,
                default_policy=policy.default_route,
                current_status="deferred-compiled-semantics",
                evidence_id=None,
                artifact_repository=None,
                artifact_revision=None,
                artifact_filename=None,
                artifact_size=None,
                artifact_sha256=None,
                tokenizer_repository=None,
                tokenizer_revision=None,
                tokenizer_assets=(),
                blocker_category="compiled-llama.cpp-semantic-dependency",
            )
        records.append(record)
    return tuple(records)
