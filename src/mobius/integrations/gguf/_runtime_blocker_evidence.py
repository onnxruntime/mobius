# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Immutable real-artifact evidence for GGUF routes that remain fail-closed."""

from __future__ import annotations

__all__ = [
    "GGUFRuntimeBlockerEvidence",
    "iter_runtime_blocker_evidence",
    "runtime_blocker_evidence",
]

import dataclasses
import re
from pathlib import Path
from types import MappingProxyType


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFRuntimeBlockerEvidence:
    """One pinned candidate whose measured blockers prevent a runtime claim."""

    evidence_id: str
    architecture: str
    repository: str
    revision: str
    filename: str
    size: int
    lfs_sha256: str
    config_repository: str
    config_revision: str
    config_sha256: str
    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_metadata_sha256: str
    tokenizer_assets: tuple[tuple[str, int, str], ...]
    tensor_count: int
    tensor_qtypes: tuple[tuple[str, int], ...]
    logical_parameter_count: int
    explicit_float16_bytes: int
    explicit_float32_bytes: int
    bounded_header_bytes: int
    bounded_header_sha256: str
    expert_count: int
    experts_per_token: int
    layer_counts: tuple[tuple[str, int], ...]
    graph_node_count: int
    graph_initializer_count: int
    graph_matmul_count: int
    state_slots: tuple[tuple[str, int], ...]
    execution_provider: str
    onnxruntime_version: str
    runtime: str
    runtime_version: str
    runtime_schema_issue: str
    blockers: tuple[str, ...]
    withheld_checks: tuple[str, ...]
    result: str = "blocked"

    def __post_init__(self) -> None:
        text_fields = (
            self.evidence_id,
            self.architecture,
            self.repository,
            self.filename,
            self.config_repository,
            self.tokenizer_repository,
            self.execution_provider,
            self.onnxruntime_version,
            self.runtime,
            self.runtime_version,
            self.runtime_schema_issue,
            self.result,
        )
        if any(not value.strip() for value in text_fields):
            raise ValueError("GGUF runtime blocker evidence fields must be non-empty")
        revisions = (self.revision, self.config_revision, self.tokenizer_revision)
        digests = (
            self.lfs_sha256,
            self.config_sha256,
            self.tokenizer_metadata_sha256,
            self.bounded_header_sha256,
        )
        if any(re.fullmatch(r"[0-9a-f]{40}", value) is None for value in revisions):
            raise ValueError("GGUF runtime blocker evidence requires immutable revisions")
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digests):
            raise ValueError(
                "GGUF runtime blocker evidence requires lowercase SHA-256 digests"
            )
        positive = (
            self.size,
            self.tensor_count,
            self.logical_parameter_count,
            self.explicit_float16_bytes,
            self.explicit_float32_bytes,
            self.bounded_header_bytes,
            self.expert_count,
            self.experts_per_token,
            self.graph_node_count,
            self.graph_initializer_count,
            self.graph_matmul_count,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("GGUF runtime blocker evidence counts and sizes must be positive")
        if self.explicit_float16_bytes != self.logical_parameter_count * 2:
            raise ValueError("GGUF runtime blocker float16 size contradicts parameter count")
        if self.explicit_float32_bytes != self.logical_parameter_count * 4:
            raise ValueError("GGUF runtime blocker float32 size contradicts parameter count")
        if self.result != "blocked" or not self.blockers or not self.withheld_checks:
            raise ValueError("GGUF runtime blocker evidence cannot imply runtime support")
        if self.filename != Path(self.filename).name:
            raise ValueError("GGUF runtime blocker filename must be basename-only")
        for records, label in (
            (self.tensor_qtypes, "tensor qtypes"),
            (self.layer_counts, "layer counts"),
            (self.state_slots, "state slots"),
        ):
            names = tuple(name for name, _ in records)
            if names != tuple(sorted(names)) or len(set(names)) != len(names):
                raise ValueError(f"GGUF runtime blocker {label} must be sorted and unique")
            if any(not name or count <= 0 for name, count in records):
                raise ValueError(f"GGUF runtime blocker {label} must be positive")
        asset_names = tuple(name for name, _, _ in self.tokenizer_assets)
        if (
            "tokenizer.json" not in asset_names
            or asset_names != tuple(sorted(asset_names))
            or len(set(asset_names)) != len(asset_names)
            or any(
                name != Path(name).name
                or size <= 0
                or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
                for name, size, sha256 in self.tokenizer_assets
            )
        ):
            raise ValueError(
                "GGUF runtime blocker tokenizer assets must be sorted immutable identities"
            )


_NEMOTRON_H_MOE_30B_IQ2_XXS = GGUFRuntimeBlockerEvidence(
    evidence_id="nemotron-h-moe-30b-iq2-xxs-runtime-blocker",
    architecture="nemotron_h_moe",
    repository="bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF",
    revision="1fc64d5b160654ec892df2708aa893b0e96e6491",
    filename="nvidia_Nemotron-3-Nano-30B-A3B-IQ2_XXS.gguf",
    size=18_010_755_296,
    lfs_sha256="f3da710c046ce7cc6ff28a9b5f1a9153ac72e3f60603e51c7bb679d80716b58a",
    config_repository="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    config_revision="bf77c3174f68ad409e1c2aa60daeb46e32d1c606",
    config_sha256="dd9fa380ac107b0477db5a26108db9febe6378e7bb3966a107944853ec4f76f8",
    tokenizer_repository="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    tokenizer_revision="bf77c3174f68ad409e1c2aa60daeb46e32d1c606",
    tokenizer_metadata_sha256=(
        "6089bcaf08b3fe0d49379ca7e85bd3c93e8705bac6130636425c159212971225"
    ),
    tokenizer_assets=(
        (
            "chat_template.jinja",
            10_504,
            "ab7813c3abdd9cb655905a410728b26c7884eca45ddfab8d9f931553485a7862",
        ),
        (
            "special_tokens_map.json",
            420,
            "e3a4f63da745f02317a45e00e6476c17fc66ac41faf14bb1b0be1f3211b0ca53",
        ),
        (
            "tokenizer.json",
            17_077_485,
            "c6021eb6847e682f89aa52d5eb6e8c7d902a23acfc8137e25211cf84828f1592",
        ),
        (
            "tokenizer_config.json",
            188_049,
            "3d568e506d0905285ae90a3fdd1482be7b6d0bf0b8ca9514d75dad4257b0827a",
        ),
    ),
    tensor_count=401,
    tensor_qtypes=(
        ("F32", 237),
        ("IQ2_XXS", 23),
        ("IQ4_NL", 100),
        ("Q4_K", 6),
        ("Q5_0", 12),
        ("Q8_0", 23),
    ),
    logical_parameter_count=31_577_940_288,
    explicit_float16_bytes=63_155_880_576,
    explicit_float32_bytes=126_311_761_152,
    bounded_header_bytes=16_777_216,
    bounded_header_sha256=("db14a1223f2f4cfaaa265eafd763b760702c83a5007c72e2f3f265b0e7439dff"),
    expert_count=128,
    experts_per_token=6,
    layer_counts=(("full_attention", 6), ("mamba2", 23), ("moe", 23)),
    graph_node_count=40_167,
    graph_initializer_count=6_255,
    graph_matmul_count=6_028,
    state_slots=(
        ("attention.key", 6),
        ("attention.value", 6),
        ("mamba2.conv_state", 23),
        ("mamba2.ssm_state", 23),
    ),
    execution_provider="CPUExecutionProvider",
    onnxruntime_version="1.29.0",
    runtime="ort-genai",
    runtime_version="0.15.2",
    runtime_schema_issue="https://github.com/onnxruntime/mobius/issues/605",
    blockers=(
        (
            "The smallest model GGUF in the pinned 29-file repository revision is "
            "18,010,755,296 bytes, above the 16 GiB bounded-artifact policy; explicit "
            "float16/float32 weights require 63,155,880,576/126,311,761,152 bytes."
        ),
        (
            "Nemotron-H correction-biased sigmoid routing, unbiased sigmoid weights, ReLU2 "
            "experts, shared expert, and optional latent projections cannot use "
            "com.microsoft.MoE; the truthful production graph retains 40,167 nodes and "
            "6,028 MatMul nodes."
        ),
        (
            "ORT GenAI 0.15.2 cannot describe the 58 heterogeneous attention key/value and "
            "Mamba2 convolution/recurrent state slots required for package replay and reorder."
        ),
        (
            "The GGUF tokenizer declares pre=pixtral, whose compiled llama.cpp behavior is "
            "not serialized; exact ORT tokenizer materialization is unavailable."
        ),
    ),
    withheld_checks=(
        "cached decode and deterministic generation",
        "full-logit parity",
        "package and report roundtrip",
        "state replay, rollback, and reorder",
        "tensor value closure",
    ),
)

_RUNTIME_BLOCKER_EVIDENCE = MappingProxyType(
    {
        _NEMOTRON_H_MOE_30B_IQ2_XXS.evidence_id: _NEMOTRON_H_MOE_30B_IQ2_XXS,
    }
)


def runtime_blocker_evidence(evidence_id: str) -> GGUFRuntimeBlockerEvidence | None:
    """Return one immutable fail-closed runtime evidence record."""
    return _RUNTIME_BLOCKER_EVIDENCE.get(evidence_id)


def iter_runtime_blocker_evidence() -> tuple[GGUFRuntimeBlockerEvidence, ...]:
    """Return all fail-closed runtime evidence records in stable order."""
    return tuple(_RUNTIME_BLOCKER_EVIDENCE[key] for key in sorted(_RUNTIME_BLOCKER_EVIDENCE))
