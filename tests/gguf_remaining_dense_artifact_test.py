# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Metadata-only immutable artifact evidence for the remaining dense GGUF cohort."""

from __future__ import annotations

import pytest
from huggingface_hub import HfApi

from mobius.integrations.gguf._artifact_blocker_evidence import (
    MAX_BOUNDED_ARTIFACT_BYTES,
    GGUFArtifactBlockerEvidence,
    iter_artifact_blocker_evidence,
)


@pytest.mark.integration
@pytest.mark.parametrize(
    "artifact",
    iter_artifact_blocker_evidence(),
    ids=lambda item: item.architecture,
)
def test_real_candidate_identity_and_budget_blocker(
    artifact: GGUFArtifactBlockerEvidence,
) -> None:
    """Verify exact LFS identities without downloading any GGUF payload."""
    records = HfApi().get_paths_info(
        artifact.repository,
        [file.path for file in artifact.files],
        revision=artifact.revision,
    )
    observed = {record.path: record for record in records}
    assert set(observed) == {file.path for file in artifact.files}
    for expected in artifact.files:
        record = observed[expected.path]
        assert record.size == expected.size
        assert record.lfs is not None
        assert record.lfs.sha256 == expected.lfs_sha256
    assert artifact.total_size > MAX_BOUNDED_ARTIFACT_BYTES


def test_no_real_candidate_can_enter_the_bounded_payload_set() -> None:
    artifacts = iter_artifact_blocker_evidence()
    assert {artifact.architecture for artifact in artifacts} == {
        "glm-dsa",
        "minimax-m2",
        "mistral4",
    }
    assert all(artifact.total_size > MAX_BOUNDED_ARTIFACT_BYTES for artifact in artifacts)
