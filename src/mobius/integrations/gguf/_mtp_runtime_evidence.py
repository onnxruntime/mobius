# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Immutable artifact and downstream-runtime status for GGUF MTP sidecars."""

from __future__ import annotations

__all__ = [
    "GGUFMtpArtifact",
    "GGUFMtpArtifactLayout",
    "GGUFMtpCacheTopology",
    "GGUFMtpRuntimeEvidence",
    "iter_mtp_runtime_evidence",
    "mtp_runtime_evidence",
]

import dataclasses
import re
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Literal

_MAX_BOUNDED_ARTIFACT_BYTES = 16 * 1024**3
_BOUNDED_HEADER_BYTES = 16 * 1024**2


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFMtpArtifact:
    """One immutable GGUF header identity in an MTP artifact layout."""

    role: Literal["combined", "target", "mtp"]
    repository: str
    revision: str
    filename: str
    size: int
    lfs_sha256: str
    bounded_header_bytes: int
    bounded_header_sha256: str
    data_offset: int
    architecture: str
    model_name: str
    block_count: int
    nextn_predict_layers: int
    first_block_index: int
    last_block_index: int
    physical_block_count: int
    tensor_count: int
    tensor_qtypes: tuple[tuple[str, int], ...]
    nextn_tensor_count: int
    nextn_tensor_qtypes: tuple[tuple[str, int], ...]
    tokenizer_metadata_sha256: str

    def __post_init__(self) -> None:
        text_fields = (
            self.repository,
            self.filename,
            self.architecture,
            self.model_name,
        )
        if any(not value.strip() for value in text_fields) or "/" not in self.repository:
            raise ValueError("GGUF MTP artifact fields must be non-empty")
        path = PurePosixPath(self.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("GGUF MTP artifact filenames must be safe Hub-relative paths")
        if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ValueError("GGUF MTP artifacts require an immutable repository revision")
        digests = (
            self.lfs_sha256,
            self.bounded_header_sha256,
            self.tokenizer_metadata_sha256,
        )
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digests):
            raise ValueError("GGUF MTP artifacts require lowercase SHA-256 identities")
        positive = (
            self.size,
            self.bounded_header_bytes,
            self.data_offset,
            self.block_count,
            self.physical_block_count,
            self.tensor_count,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("GGUF MTP artifact sizes and counts must be positive")
        if self.data_offset > self.bounded_header_bytes:
            raise ValueError("Bounded GGUF MTP header must include the tensor-data offset")
        if self.first_block_index < 0 or self.last_block_index < self.first_block_index:
            raise ValueError("GGUF MTP artifact block indices are invalid")
        if self.physical_block_count != self.last_block_index - self.first_block_index + 1:
            raise ValueError("GGUF MTP physical block count contradicts its block range")
        self._validate_qtypes(self.tensor_qtypes, self.tensor_count, "tensor")
        self._validate_qtypes(
            self.nextn_tensor_qtypes,
            self.nextn_tensor_count,
            "NextN tensor",
        )

        if self.role == "target":
            if self.nextn_predict_layers != 0 or self.nextn_tensor_count != 0:
                raise ValueError(
                    "Target-only GGUF evidence must not contain a physical MTP head"
                )
            if (
                self.first_block_index != 0
                or self.last_block_index != self.block_count - 1
                or self.physical_block_count != self.block_count
            ):
                raise ValueError("Target-only GGUF evidence must contain every declared block")
            return

        if self.nextn_predict_layers != 1 or self.nextn_tensor_count <= 0:
            raise ValueError("MTP-bearing GGUF evidence must contain exactly one NextN head")
        mtp_block_index = self.block_count - self.nextn_predict_layers
        if self.last_block_index != mtp_block_index:
            raise ValueError(
                "GGUF MTP evidence must place NextN tensors in the trailing block"
            )
        if self.role == "combined":
            if self.first_block_index != 0 or self.physical_block_count != self.block_count:
                raise ValueError(
                    "Combined GGUF MTP evidence must contain trunk and MTP blocks"
                )
        elif self.role == "mtp":
            if self.first_block_index != mtp_block_index or self.physical_block_count != 1:
                raise ValueError("Split MTP evidence must contain only the trailing MTP block")
        else:
            raise ValueError(f"Unknown GGUF MTP artifact role {self.role!r}")

    @staticmethod
    def _validate_qtypes(
        records: tuple[tuple[str, int], ...],
        expected_count: int,
        label: str,
    ) -> None:
        names = tuple(name for name, _ in records)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError(f"GGUF MTP {label} qtypes must be sorted and unique")
        if any(not name or count <= 0 for name, count in records):
            raise ValueError(f"GGUF MTP {label} qtype counts must be positive")
        if sum(count for _, count in records) != expected_count:
            raise ValueError(f"GGUF MTP {label} qtypes must close over the tensor count")

    @property
    def trunk_block_count(self) -> int:
        """Number of target blocks declared by this artifact."""
        return self.block_count - self.nextn_predict_layers


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFMtpArtifactLayout:
    """One complete combined or split trunk-plus-MTP artifact layout."""

    name: Literal["combined", "split"]
    artifacts: tuple[GGUFMtpArtifact, ...]

    def __post_init__(self) -> None:
        if self.name == "combined":
            if len(self.artifacts) != 1 or self.artifacts[0].role != "combined":
                raise ValueError("Combined GGUF MTP layout requires one combined artifact")
            return
        if self.name != "split":
            raise ValueError(f"Unknown GGUF MTP layout {self.name!r}")
        if len(self.artifacts) != 2 or {item.role for item in self.artifacts} != {
            "target",
            "mtp",
        }:
            raise ValueError("Split GGUF MTP layout requires one target and one MTP artifact")
        target = next(item for item in self.artifacts if item.role == "target")
        mtp = next(item for item in self.artifacts if item.role == "mtp")
        if (target.repository, target.revision) != (mtp.repository, mtp.revision):
            raise ValueError(
                "Split GGUF MTP artifacts must share one immutable repository revision"
            )
        if target.architecture != mtp.architecture:
            raise ValueError("Split GGUF MTP artifacts must share one architecture")
        if target.tokenizer_metadata_sha256 != mtp.tokenizer_metadata_sha256:
            raise ValueError(
                "Split GGUF MTP artifacts must embed identical tokenizer metadata"
            )
        if target.block_count + 1 != mtp.block_count:
            raise ValueError("Split MTP block must immediately follow every target block")
        if mtp.first_block_index != target.block_count:
            raise ValueError(
                "Split MTP physical block index must equal the target block count"
            )

    @property
    def total_size(self) -> int:
        """Total bytes required to download this complete layout."""
        return sum(artifact.size for artifact in self.artifacts)

    @property
    def within_bounded_artifact_policy(self) -> bool:
        """Whether the complete layout is within the 16 GiB evidence budget."""
        return self.total_size <= _MAX_BOUNDED_ARTIFACT_BYTES


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFMtpCacheTopology:
    """Component-qualified target and MTP cache namespaces required at runtime."""

    target_namespace: str
    mtp_namespace: str
    target_state_slots: tuple[tuple[str, int], ...]
    mtp_state_slots: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if (
            not self.target_namespace
            or not self.mtp_namespace
            or self.target_namespace == self.mtp_namespace
        ):
            raise ValueError("Target and MTP cache namespaces must be non-empty and distinct")
        for records, label in (
            (self.target_state_slots, "target"),
            (self.mtp_state_slots, "MTP"),
        ):
            names = tuple(name for name, _ in records)
            if names != tuple(sorted(names)) or len(names) != len(set(names)):
                raise ValueError(f"{label} cache slots must be sorted and unique")
            if any(not name or count <= 0 for name, count in records):
                raise ValueError(f"{label} cache slot counts must be positive")
        target = {
            (self.target_namespace, name, index)
            for name, count in self.target_state_slots
            for index in range(count)
        }
        mtp = {
            (self.mtp_namespace, name, index)
            for name, count in self.mtp_state_slots
            for index in range(count)
        }
        if target & mtp:
            raise ValueError("Target and MTP cache identities must not alias")


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFMtpRuntimeEvidence:
    """Pinned artifact facts and explicitly unvalidated downstream runtime status."""

    evidence_id: str
    architecture: str
    layouts: tuple[GGUFMtpArtifactLayout, ...]
    target_only_discriminator: GGUFMtpArtifact | None
    config_repository: str
    config_revision: str
    config_sha256: str
    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_assets: tuple[tuple[str, int, str], ...]
    tokenizer_metadata_sha256: str
    cache_topology: GGUFMtpCacheTopology
    bounded_complete_layout_available: bool
    source_fidelity: bool
    storage_fidelity: bool
    graph_sha256: str | None
    runtime_package_sha256: str | None
    onnxruntime_version: str
    execution_provider: str
    runtime: str
    runtime_version: str
    runtime_source_revision: str
    missing_runtime_capabilities: tuple[str, ...]
    downstream_limitations: tuple[str, ...]
    separate_deferrals: tuple[str, ...]
    withheld_checks: tuple[str, ...]
    synthetic_coordinator_test: str | None
    synthetic_acceptance_statistics: tuple[tuple[str, int], ...]
    result: str = "runtime_unvalidated"

    def __post_init__(self) -> None:
        text_fields = (
            self.evidence_id,
            self.architecture,
            self.config_repository,
            self.tokenizer_repository,
            self.onnxruntime_version,
            self.execution_provider,
            self.runtime,
            self.runtime_version,
            self.result,
        )
        if any(not value.strip() for value in text_fields):
            raise ValueError("GGUF MTP runtime evidence fields must be non-empty")
        revisions = (
            self.config_revision,
            self.tokenizer_revision,
            self.runtime_source_revision,
        )
        if any(re.fullmatch(r"[0-9a-f]{40}", value) is None for value in revisions):
            raise ValueError("GGUF MTP runtime evidence revisions must be immutable")
        digests = (self.config_sha256, self.tokenizer_metadata_sha256)
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digests):
            raise ValueError("GGUF MTP runtime evidence digests must be lowercase SHA-256")
        layout_names = tuple(layout.name for layout in self.layouts)
        if (
            not self.layouts
            or layout_names != tuple(sorted(layout_names))
            or len(layout_names) != len(set(layout_names))
        ):
            raise ValueError("GGUF MTP artifact layouts must be sorted and unique")
        artifacts = tuple(artifact for layout in self.layouts for artifact in layout.artifacts)
        if any(artifact.architecture != self.architecture for artifact in artifacts):
            raise ValueError("GGUF MTP layouts must match the evidence architecture")
        has_bounded_layout = any(
            layout.within_bounded_artifact_policy for layout in self.layouts
        )
        if has_bounded_layout != self.bounded_complete_layout_available:
            raise ValueError("GGUF MTP bounded-layout verdict contradicts immutable sizes")
        discriminator = self.target_only_discriminator
        if discriminator is not None:
            if (
                discriminator.role != "target"
                or discriminator.architecture != self.architecture
            ):
                raise ValueError("MTP target-only discriminator must match the architecture")
            combined = [
                layout.artifacts[0] for layout in self.layouts if layout.name == "combined"
            ]
            if not combined or any(
                artifact.trunk_block_count != discriminator.block_count
                or artifact.tokenizer_metadata_sha256
                != discriminator.tokenizer_metadata_sha256
                for artifact in combined
            ):
                raise ValueError(
                    "Target-only discriminator must match each combined layout's trunk"
                )
        asset_names = tuple(name for name, _, _ in self.tokenizer_assets)
        if (
            not self.tokenizer_assets
            or "tokenizer.json" not in asset_names
            or asset_names != tuple(sorted(asset_names))
            or len(asset_names) != len(set(asset_names))
            or any(
                name != PurePosixPath(name).name
                or size <= 0
                or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
                for name, size, sha256 in self.tokenizer_assets
            )
        ):
            raise ValueError("GGUF MTP tokenizer assets must be sorted immutable identities")
        if (
            self.source_fidelity
            or self.storage_fidelity
            or self.graph_sha256 is not None
            or self.runtime_package_sha256 is not None
        ):
            raise ValueError(
                "Unvalidated GGUF MTP evidence cannot imply value or package fidelity"
            )
        for records, label in (
            (self.missing_runtime_capabilities, "missing runtime capabilities"),
            (self.withheld_checks, "withheld checks"),
        ):
            if (
                not records
                or records != tuple(sorted(records))
                or len(records) != len(set(records))
                or any(not value.strip() for value in records)
            ):
                raise ValueError(f"GGUF MTP {label} must be sorted, unique, and non-empty")
        if (
            self.result != "runtime_unvalidated"
            or not self.downstream_limitations
            or not self.separate_deferrals
            or any(
                not value.strip()
                for value in (*self.downstream_limitations, *self.separate_deferrals)
            )
        ):
            raise ValueError("Unvalidated GGUF MTP evidence cannot imply downstream support")
        statistic_names = tuple(name for name, _ in self.synthetic_acceptance_statistics)
        if (
            statistic_names != tuple(sorted(statistic_names))
            or len(statistic_names) != len(set(statistic_names))
            or any(
                not name or value < 0 for name, value in self.synthetic_acceptance_statistics
            )
        ):
            raise ValueError(
                "Synthetic MTP acceptance statistics must be sorted and non-negative"
            )
        if self.synthetic_coordinator_test is None:
            if self.synthetic_acceptance_statistics:
                raise ValueError("Synthetic MTP statistics require their exact test identity")
        else:
            if not self.synthetic_coordinator_test.strip():
                raise ValueError("Synthetic MTP coordinator test identity must be non-empty")
            statistics = dict(self.synthetic_acceptance_statistics)
            if statistics.get("proposal_steps") != statistics.get(
                "accepted", 0
            ) + statistics.get("rejected", 0) or statistics.get("rollbacks") != statistics.get(
                "rejected"
            ):
                raise ValueError("Synthetic MTP acceptance statistics do not close")


_QWEN_TOKENIZER_METADATA = "8d1070c727a7e6a03726687aa5746acece0255060a2a61a905c4c2285d353b68"
_QWEN_COMBINED = GGUFMtpArtifact(
    role="combined",
    repository="localweights/Qwen3.6-27B-MTP-Q4_K_M-Q8nextn-GGUF",
    revision="bdc8bc1ca4d45e1152d54c4e5f1389bc7bf6e439",
    filename="Qwen3.6-27B-MTP-Q4_K_M-Q8nextn.gguf",
    size=16_998_719_584,
    lfs_sha256="6cdb1aabb6ee711938df4192938b51228df9274f01c8d7a5ff404915ad342a7c",
    bounded_header_bytes=_BOUNDED_HEADER_BYTES,
    bounded_header_sha256=("ecf4be7b831ea56596022a4cd974930f59b37f5e64ce5b62bbe21686e02e07a8"),
    data_offset=10_993_760,
    architecture="qwen35",
    model_name="Qwen3.6 27B MTPpatched",
    block_count=65,
    nextn_predict_layers=1,
    first_block_index=0,
    last_block_index=64,
    physical_block_count=65,
    tensor_count=866,
    tensor_qtypes=(("F32", 360), ("Q4_K", 433), ("Q6_K", 65), ("Q8_0", 8)),
    nextn_tensor_count=4,
    nextn_tensor_qtypes=(("F32", 3), ("Q8_0", 1)),
    tokenizer_metadata_sha256=_QWEN_TOKENIZER_METADATA,
)
_QWEN_SPLIT_TARGET = GGUFMtpArtifact(
    role="target",
    repository="ggml-org/Qwen3.6-27B-GGUF",
    revision="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8",
    filename="Qwen3.6-27B-Q4_K_M.gguf",
    size=19_095_766_304,
    lfs_sha256="65b753ea835627f7b511143c6ceb976525c7f21f5df8c664bc0a9c23d1c49921",
    bounded_header_bytes=_BOUNDED_HEADER_BYTES,
    bounded_header_sha256=("a14696157c62d5cbc00cbc473f1fade79fe82a5406f0193ef9fd6839d9d75313"),
    data_offset=10_992_928,
    architecture="qwen35",
    model_name="Qwen3.6-27B",
    block_count=64,
    nextn_predict_layers=0,
    first_block_index=0,
    last_block_index=63,
    physical_block_count=64,
    tensor_count=851,
    tensor_qtypes=(("F32", 353), ("Q4_K", 193), ("Q6_K", 1), ("Q8_0", 304)),
    nextn_tensor_count=0,
    nextn_tensor_qtypes=(),
    tokenizer_metadata_sha256=_QWEN_TOKENIZER_METADATA,
)
_QWEN_SPLIT_MTP = GGUFMtpArtifact(
    role="mtp",
    repository="ggml-org/Qwen3.6-27B-GGUF",
    revision="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8",
    filename="mtp-Qwen3.6-27B-Q4_0.gguf",
    size=1_680_270_560,
    lfs_sha256="3d593f9e2788d59bb30d6024706b1efd5219fea466b6397c46159e3540937173",
    bounded_header_bytes=_BOUNDED_HEADER_BYTES,
    bounded_header_sha256=("082ed312178c514db10387e2f14db5c9b91cb86e2bed9fb163298d0bc081a7e0"),
    data_offset=10_943_712,
    architecture="qwen35",
    model_name="Qwen3.6-27B",
    block_count=65,
    nextn_predict_layers=1,
    first_block_index=64,
    last_block_index=64,
    physical_block_count=1,
    tensor_count=18,
    tensor_qtypes=(("F32", 8), ("Q4_0", 10)),
    nextn_tensor_count=4,
    nextn_tensor_qtypes=(("F32", 3), ("Q4_0", 1)),
    tokenizer_metadata_sha256=_QWEN_TOKENIZER_METADATA,
)

_HY_TOKENIZER_METADATA = "bff10dd3b99c0d4098b28fbeb1af46804184cefc2920895969b6f1ad2e548fa3"
_HY_COMBINED = GGUFMtpArtifact(
    role="combined",
    repository="AngelSlim/Hy3-GGUF",
    revision="31b453f4d9b647c74e4c4f5cba632eb512332c91",
    filename="Hy3-IQ1_M-mtp.gguf",
    size=91_756_066_272,
    lfs_sha256="c1fe984fef6f23fd9eb144e41ce6824e3941ed0a8b6b8f9f3e63da0c2a1320c2",
    bounded_header_bytes=_BOUNDED_HEADER_BYTES,
    bounded_header_sha256=("167bf00a520da2bfdaa9362ebf5ac47b4c8966ac1128b74fba0e8ca55a51341b"),
    data_offset=5_161_440,
    architecture="hy_v3",
    model_name="Hy3",
    block_count=81,
    nextn_predict_layers=1,
    first_block_index=0,
    last_block_index=80,
    physical_block_count=81,
    tensor_count=1_298,
    tensor_qtypes=(
        ("F32", 488),
        ("IQ1_M", 82),
        ("IQ2_XXS", 78),
        ("IQ3_XXS", 79),
        ("Q2_K", 1),
        ("Q4_K", 3),
        ("Q5_K", 242),
        ("Q6_K", 81),
        ("Q8_0", 244),
    ),
    nextn_tensor_count=4,
    nextn_tensor_qtypes=(("F32", 3), ("Q8_0", 1)),
    tokenizer_metadata_sha256=_HY_TOKENIZER_METADATA,
)
_HY_TARGET_ONLY = GGUFMtpArtifact(
    role="target",
    repository="AngelSlim/Hy3-GGUF",
    revision="31b453f4d9b647c74e4c4f5cba632eb512332c91",
    filename="Hy3-IQ1_M.gguf",
    size=89_446_312_384,
    lfs_sha256="8c4195718dc24384a38de1c492f4bca7d21447789714d232b0a4d9cd0bb0c806",
    bounded_header_bytes=_BOUNDED_HEADER_BYTES,
    bounded_header_sha256=("93b14dcd2cca8898bf295842e189d996a4d1138990ef2e275614d392eb1e4a22"),
    data_offset=5_160_128,
    architecture="hy_v3",
    model_name="Hy3",
    block_count=80,
    nextn_predict_layers=0,
    first_block_index=0,
    last_block_index=79,
    physical_block_count=80,
    tensor_count=1_278,
    tensor_qtypes=(
        ("F32", 479),
        ("IQ1_M", 82),
        ("IQ2_XXS", 78),
        ("IQ3_XXS", 79),
        ("Q2_K", 1),
        ("Q4_K", 1),
        ("Q5_K", 238),
        ("Q6_K", 80),
        ("Q8_0", 240),
    ),
    nextn_tensor_count=0,
    nextn_tensor_qtypes=(),
    tokenizer_metadata_sha256=_HY_TOKENIZER_METADATA,
)

_ORT_GENAI_SOURCE_REVISION = "ed5f4e87147731e5b07810f9f5c90103b3603cdf"
_MISSING_RUNTIME_CAPABILITIES = (
    "accept_reject_rollback",
    "acceptance_statistics",
    "independent_target_and_mtp_cache_threading",
    "two_model_draft_target_binding",
)
_WITHHELD_CHECKS = (
    "real-artifact accept/reject rollback and replay/reorder",
    "real-artifact deterministic output equality against target-only generation",
    "real-artifact full-logit parity from an independent oracle",
    "real-artifact multi-step acceptance statistics",
    "serialized graph and runtime-package SHA-256 identities",
    "source-value and target-storage fidelity",
)
_QWEN35_MTP_RUNTIME_EVIDENCE = GGUFMtpRuntimeEvidence(
    evidence_id="qwen3.6-27b-mtp-runtime-unvalidated",
    architecture="qwen35",
    layouts=(
        GGUFMtpArtifactLayout("combined", (_QWEN_COMBINED,)),
        GGUFMtpArtifactLayout("split", (_QWEN_SPLIT_TARGET, _QWEN_SPLIT_MTP)),
    ),
    target_only_discriminator=None,
    config_repository="Qwen/Qwen3.6-27B",
    config_revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    config_sha256="69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c",
    tokenizer_repository="Qwen/Qwen3.6-27B",
    tokenizer_revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    tokenizer_assets=(
        (
            "tokenizer.json",
            12_807_982,
            "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        ),
        (
            "tokenizer_config.json",
            16_718,
            "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
        ),
    ),
    tokenizer_metadata_sha256=_QWEN_TOKENIZER_METADATA,
    cache_topology=GGUFMtpCacheTopology(
        target_namespace="target",
        mtp_namespace="mtp",
        target_state_slots=(
            ("attention.key", 16),
            ("attention.value", 16),
            ("linear_attention.conv_state", 48),
            ("linear_attention.recurrent_state", 48),
        ),
        mtp_state_slots=(("attention.key", 1), ("attention.value", 1)),
    ),
    bounded_complete_layout_available=True,
    source_fidelity=False,
    storage_fidelity=False,
    graph_sha256=None,
    runtime_package_sha256=None,
    onnxruntime_version="1.29.0",
    execution_provider="CPUExecutionProvider",
    runtime="ort-genai",
    runtime_version="0.15.2",
    runtime_source_revision=_ORT_GENAI_SOURCE_REVISION,
    missing_runtime_capabilities=_MISSING_RUNTIME_CAPABILITIES,
    downstream_limitations=(
        (
            "ORT GenAI 0.15.2 constructs one Model per Generator and exposes no two-model "
            "draft/target binding, independent MTP-cache orchestration, proposal "
            "accept/reject rollback, or observable acceptance statistics. Its pinned release "
            "matrix places speculative decoding on the roadmap; Generator.rewind_to only "
            "rewinds the single target generator and is not an MTP acceptance loop."
        ),
    ),
    separate_deferrals=(
        (
            "The embedded tokenizer uses pre=qwen35, whose compiled llama.cpp behavior is not "
            "serialized in GGUF. This tokenizer limitation is separate from downstream "
            "two-model runtime validation."
        ),
        (
            "The under-budget combined artifact proves immutable header, physical trailing "
            "block, and layout identity only. No payload values, production tensor mapping, "
            "quantized storage execution, standalone MTP forward, graph, or package is "
            "promoted as runtime evidence. A separate tiny direct-ORT coordinator exercises "
            "target/MTP cache threading, accept/reject rollback, replay, reorder, and "
            "target-only equality without upgrading the real-artifact status."
        ),
        (
            "The 2026-08-28 qualification transfer fetched 2,167,418,880 of "
            "16,998,719,584 bytes before measured throughput implied a multi-hour transfer "
            "ahead of any 32 GiB-host build/runtime attempt. The partial payload was deleted; "
            "this operational deferral does not restrict export support."
        ),
    ),
    withheld_checks=_WITHHELD_CHECKS,
    synthetic_coordinator_test=(
        "src/mobius/integrations/gguf/_mtp_test.py::"
        "TestMtpAutoDetect::"
        "test_direct_ort_target_acceptance_threads_and_rolls_back_both_caches"
    ),
    synthetic_acceptance_statistics=(
        ("accepted", 1),
        ("proposal_steps", 51),
        ("rejected", 50),
        ("rollbacks", 50),
    ),
)

_HY_V3_MTP_RUNTIME_EVIDENCE = GGUFMtpRuntimeEvidence(
    evidence_id="hy3-iq1-m-mtp-runtime-unvalidated",
    architecture="hy_v3",
    layouts=(GGUFMtpArtifactLayout("combined", (_HY_COMBINED,)),),
    target_only_discriminator=_HY_TARGET_ONLY,
    config_repository="tencent/Hy3",
    config_revision="a960ebc3da325ba167f069f76c41eb62c9280d22",
    config_sha256="0c9daab42bff9cce1b6f058b10d7b730f76d583e583e28ad56e92b36373246f0",
    tokenizer_repository="tencent/Hy3",
    tokenizer_revision="a960ebc3da325ba167f069f76c41eb62c9280d22",
    tokenizer_assets=(
        (
            "tokenizer.json",
            9_527_406,
            "446e0b59cd941637c0ddfd84e12ee1f49480fd12097f86a9d2fec8ebd0c7ff6c",
        ),
        (
            "tokenizer_config.json",
            165_971,
            "1af226fc70b260371ca1f08053768b80a9d58b9d79e8eab718160f52783f7ceb",
        ),
    ),
    tokenizer_metadata_sha256=_HY_TOKENIZER_METADATA,
    cache_topology=GGUFMtpCacheTopology(
        target_namespace="target",
        mtp_namespace="mtp",
        target_state_slots=(("attention.key", 80), ("attention.value", 80)),
        mtp_state_slots=(("attention.key", 1), ("attention.value", 1)),
    ),
    bounded_complete_layout_available=False,
    source_fidelity=False,
    storage_fidelity=False,
    graph_sha256=None,
    runtime_package_sha256=None,
    onnxruntime_version="1.29.0",
    execution_provider="CPUExecutionProvider",
    runtime="ort-genai",
    runtime_version="0.15.2",
    runtime_source_revision=_ORT_GENAI_SOURCE_REVISION,
    missing_runtime_capabilities=_MISSING_RUNTIME_CAPABILITIES,
    downstream_limitations=(
        (
            "The smallest pinned public complete Hunyuan-V3 MTP artifact is "
            "91,756,066,272 bytes, above the 17,179,869,184-byte bounded-artifact policy. "
            "The matching target-only discriminator is 89,446,312,384 bytes."
        ),
        (
            "ORT GenAI 0.15.2 exposes no two-model draft/target binding, independent "
            "target/MTP cache orchestration, proposal rollback, or acceptance statistics."
        ),
    ),
    separate_deferrals=(
        (
            "The embedded tokenizer uses pre=hunyuan-dense, whose compiled llama.cpp "
            "behavior is not serialized in GGUF. This is separate from the real-artifact "
            "budget and downstream two-model runtime validation gaps."
        ),
        (
            "Bounded headers prove that the -mtp file has 81 physical blocks with trailing "
            "block 80 while the target-only file has exactly 80 blocks. No payload values, "
            "standalone MTP forward, graph, or package is promoted as runtime evidence."
        ),
    ),
    withheld_checks=_WITHHELD_CHECKS,
    synthetic_coordinator_test=None,
    synthetic_acceptance_statistics=(),
)

_MTP_RUNTIME_EVIDENCE = MappingProxyType(
    {
        evidence.evidence_id: evidence
        for evidence in (_HY_V3_MTP_RUNTIME_EVIDENCE, _QWEN35_MTP_RUNTIME_EVIDENCE)
    }
)


def mtp_runtime_evidence(evidence_id: str) -> GGUFMtpRuntimeEvidence | None:
    """Return one immutable GGUF MTP artifact/runtime-status record."""
    return _MTP_RUNTIME_EVIDENCE.get(evidence_id)


def iter_mtp_runtime_evidence() -> tuple[GGUFMtpRuntimeEvidence, ...]:
    """Return all GGUF MTP artifact/runtime-status records in stable order."""
    return tuple(_MTP_RUNTIME_EVIDENCE[key] for key in sorted(_MTP_RUNTIME_EVIDENCE))
