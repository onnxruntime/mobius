# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Immutable oversized-artifact evidence for graph routes using synthetic parity."""

from __future__ import annotations

import dataclasses
import re
from pathlib import PurePosixPath
from types import MappingProxyType

__all__ = [
    "GGUFArtifactBlockerEvidence",
    "GGUFArtifactFile",
    "MAX_BOUNDED_ARTIFACT_BYTES",
    "artifact_blocker_evidence",
    "iter_artifact_blocker_evidence",
]

MAX_BOUNDED_ARTIFACT_BYTES = 16 * 1024**3


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFArtifactFile:
    """One immutable Hub path in a complete logical GGUF candidate."""

    path: str
    size: int
    lfs_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.path
            or PurePosixPath(self.path).is_absolute()
            or ".." in PurePosixPath(self.path).parts
        ):
            raise ValueError("GGUF artifact evidence paths must be safe Hub-relative paths")
        if self.size <= 0 or re.fullmatch(r"[0-9a-f]{64}", self.lfs_sha256) is None:
            raise ValueError("GGUF artifact evidence requires a positive size and LFS SHA-256")


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFArtifactBlockerEvidence:
    """The smallest pinned real candidate for one synthetic-only graph route."""

    evidence_id: str
    architecture: str
    repository: str
    revision: str
    files: tuple[GGUFArtifactFile, ...]
    blocker: str

    def __post_init__(self) -> None:
        if (
            not self.evidence_id
            or not self.architecture
            or "/" not in self.repository
            or not self.blocker.strip()
        ):
            raise ValueError("GGUF artifact blocker evidence fields must be non-empty")
        if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ValueError("GGUF artifact blocker evidence requires an immutable revision")
        paths = tuple(file.path for file in self.files)
        if not paths or paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("GGUF artifact blocker files must be sorted and unique")
        if self.total_size <= MAX_BOUNDED_ARTIFACT_BYTES:
            raise ValueError("Artifact blocker candidates must exceed the 16 GiB policy")

    @property
    def total_size(self) -> int:
        """Complete logical GGUF size."""
        return sum(file.size for file in self.files)


_MINIMAX_M2 = GGUFArtifactBlockerEvidence(
    evidence_id="minimax-m2-iq1-s-artifact-budget-blocker",
    architecture="minimax-m2",
    repository="mradermacher/MiniMax-M2-i1-GGUF",
    revision="2d4f9b1a86d32ce4dfc47db312c8d6fcae8d7b37",
    files=(
        GGUFArtifactFile(
            "MiniMax-M2.i1-IQ1_S.gguf",
            46_514_882_176,
            "7bae986e3cd380c28c6177d612fce1d52373241a0dfa13a6fca25de79abf15fb",
        ),
    ),
    blocker=(
        "The smallest immutable public MiniMax-M2 GGUF is 46,514,882,176 bytes; "
        "real-weight parity cannot enter the 16 GiB bounded evidence set."
    ),
)

_MISTRAL4 = GGUFArtifactBlockerEvidence(
    evidence_id="mistral4-iq1-m-artifact-budget-blocker",
    architecture="mistral4",
    repository="unsloth/Mistral-Small-4-119B-2603-GGUF",
    revision="bd93c721735aa32c035c0f19e738cb3371fd56ff",
    files=(
        GGUFArtifactFile(
            "Mistral-Small-4-119B-2603-UD-IQ1_M.gguf",
            32_306_941_632,
            "40fcdee4869110938638c6b8bac253f442b196518d4623f4afdf4b885cd961c7",
        ),
    ),
    blocker=(
        "The smallest immutable public Mistral4 GGUF is 32,306,941,632 bytes; "
        "real-weight parity cannot enter the 16 GiB bounded evidence set."
    ),
)

_GLM_DSA = GGUFArtifactBlockerEvidence(
    evidence_id="glm-dsa-iq1-s-artifact-budget-blocker",
    architecture="glm-dsa",
    repository="unsloth/GLM-5.2-GGUF",
    revision="abc55e72527792c6e77069c99b4cb7de16fa9f23",
    files=(
        GGUFArtifactFile(
            "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00001-of-00006.gguf",
            9_423_744,
            "46b6148389219ae45167cb8124fbb18ef7d432daf619b4faf9e06ea80d3f4777",
        ),
        GGUFArtifactFile(
            "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00002-of-00006.gguf",
            49_208_128_256,
            "f2180207285e04fcaa5b8c53ba6e77ad5cc58666b6e7c6b04a5eded3fe8bef09",
        ),
        GGUFArtifactFile(
            "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00003-of-00006.gguf",
            49_684_417_024,
            "b1c0c5a302cc8d5d9ea0bcd4467c01db72c26839f820f7e882079582ea0a8d2b",
        ),
        GGUFArtifactFile(
            "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00004-of-00006.gguf",
            49_396_052_864,
            "a6a42da6975e29f89866dcde2956e9e50e6ea26635fb5063b74f3973f4f863b6",
        ),
        GGUFArtifactFile(
            "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00005-of-00006.gguf",
            49_246_275_936,
            "a4a9851a50db533f21ef824e5d8038f04e6782e7d602d18e5fdd6643f68ccccb",
        ),
        GGUFArtifactFile(
            "UD-IQ1_S/GLM-5.2-UD-IQ1_S-00006-of-00006.gguf",
            19_171_063_136,
            "3b767f55df64e0432d52fcf1a14eb47a1ef3bbc91339e2ae220f38602237d7d7",
        ),
    ),
    blocker=(
        "The smallest mainline GLM-5.2 GGUF is a 216,715,360,960-byte split set "
        "whose trunk also declares an unsupported routed DSA/MLA MTP block."
    ),
)

_ARTIFACT_BLOCKERS = MappingProxyType(
    {evidence.evidence_id: evidence for evidence in (_GLM_DSA, _MINIMAX_M2, _MISTRAL4)}
)


def artifact_blocker_evidence(
    evidence_id: str,
) -> GGUFArtifactBlockerEvidence | None:
    """Return one immutable oversized-artifact evidence record."""
    return _ARTIFACT_BLOCKERS.get(evidence_id)


def iter_artifact_blocker_evidence() -> tuple[GGUFArtifactBlockerEvidence, ...]:
    """Return all oversized graph-route candidates in stable order."""
    return tuple(_ARTIFACT_BLOCKERS[key] for key in sorted(_ARTIFACT_BLOCKERS))
