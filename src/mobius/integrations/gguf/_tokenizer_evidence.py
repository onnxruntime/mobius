# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Artifact-scoped evidence for exact GGUF tokenizer materialization."""

from __future__ import annotations

__all__ = [
    "GGUFTokenizerEvidence",
    "matching_tokenizer_evidence",
    "tokenizer_evidence",
]

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mobius.integrations.gguf._runtime_evidence import gguf_artifact_identity
from mobius.integrations.gguf._tokenizer import GGUFTokenizerAsset, GGUFTokenizerSource


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFTokenizerEvidence:
    """Immutable artifact, source-tokenizer, and semantic validation evidence."""

    evidence_id: str
    architecture: str
    repository: str
    revision: str
    filename: str
    size: int
    lfs_sha256: str
    tensor_count: int
    tensor_qtypes: tuple[tuple[str, int], ...]
    tokenizer_repository: str
    tokenizer_revision: str
    source_config_asset: tuple[str, int, str]
    tokenizer_metadata_sha256: str
    tokenizer_assets: tuple[tuple[str, int, str], ...]
    token_count: int
    source_token_count: int
    embedding_vocabulary_size: int
    deterministic_padding_range: tuple[int, int]
    ordered_vocabulary_sha256: str
    merge_count: int
    ordered_merges_sha256: str
    materialized_tokenizer_sha256: str
    special_token_ids: tuple[tuple[str, int], ...]
    representative_encodings: tuple[tuple[str, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        revisions = (self.revision, self.tokenizer_revision)
        digests = (
            self.lfs_sha256,
            self.tokenizer_metadata_sha256,
            self.ordered_vocabulary_sha256,
            self.ordered_merges_sha256,
            self.materialized_tokenizer_sha256,
        )
        if any(re.fullmatch(r"[0-9a-f]{40}", value) is None for value in revisions):
            raise ValueError("Tokenizer evidence revisions must be immutable 40-hex commits")
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digests):
            raise ValueError("Tokenizer evidence digests must be lowercase SHA-256")
        if min(
            self.size,
            self.tensor_count,
            self.token_count,
            self.source_token_count,
            self.embedding_vocabulary_size,
            self.merge_count,
        ) <= 0:
            raise ValueError("Tokenizer evidence counts and artifact size must be positive")
        if (
            self.source_config_asset[0] != "config.json"
            or self.source_config_asset[1] <= 0
            or re.fullmatch(r"[0-9a-f]{64}", self.source_config_asset[2]) is None
        ):
            raise ValueError("Tokenizer evidence requires an exact source config identity")
        if (
            self.source_token_count > self.token_count
            or self.embedding_vocabulary_size != self.token_count
            or self.deterministic_padding_range
            != (self.source_token_count, self.token_count - 1)
        ):
            raise ValueError(
                "Tokenizer evidence source, deterministic padding, and embedding sizes disagree"
            )
        if tuple(sorted(self.tokenizer_assets)) != self.tokenizer_assets:
            raise ValueError("Tokenizer evidence assets must be sorted")
        GGUFTokenizerSource(
            self.tokenizer_repository,
            self.tokenizer_revision,
            tuple(GGUFTokenizerAsset(*asset) for asset in self.tokenizer_assets),
            self.tokenizer_metadata_sha256,
            self.materialized_tokenizer_sha256,
            self.representative_encodings,
        )
        if tuple(sorted(self.special_token_ids)) != self.special_token_ids:
            raise ValueError("Tokenizer evidence special token IDs must be sorted by token")
        if not self.representative_encodings or any(
            not text or not token_ids
            for text, token_ids in self.representative_encodings
        ):
            raise ValueError("Tokenizer evidence requires non-empty representative encodings")

    @property
    def source(self) -> GGUFTokenizerSource:
        """Return the exact source accepted by the materializer."""
        return GGUFTokenizerSource(
            self.tokenizer_repository,
            self.tokenizer_revision,
            tuple(GGUFTokenizerAsset(*asset) for asset in self.tokenizer_assets),
            self.tokenizer_metadata_sha256,
            self.materialized_tokenizer_sha256,
            self.representative_encodings,
        )


_QWEN35_08B_Q4_TOKENIZER = GGUFTokenizerEvidence(
    evidence_id="qwen3.5-0.8b-q4-tokenizer",
    architecture="qwen35",
    repository="ggml-org/Qwen3.5-0.8B-GGUF",
    revision="8fea620810c4afa23dd6443f999a48574c1611a3",
    filename="Qwen3.5-0.8B-Q4_0.gguf",
    size=563_036_064,
    lfs_sha256="57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf",
    tensor_count=320,
    tensor_qtypes=(("F32", 133), ("Q4_0", 186), ("Q8_0", 1)),
    tokenizer_repository="Qwen/Qwen3.5-0.8B",
    tokenizer_revision="2fc06364715b967f1860aea9cf38778875588b17",
    source_config_asset=(
        "config.json",
        2_907,
        "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    ),
    tokenizer_metadata_sha256="45302b58b2086a666a874652d0e9e1d5b4b26e786ffbaf9362a4f902eba0b10d",
    tokenizer_assets=(
        (
            "chat_template.jinja",
            7_755,
            "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
        ),
        (
            "tokenizer.json",
            12_807_982,
            "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        ),
        (
            "tokenizer_config.json",
            16_709,
            "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
        ),
    ),
    token_count=248_320,
    source_token_count=248_077,
    embedding_vocabulary_size=248_320,
    deterministic_padding_range=(248_077, 248_319),
    ordered_vocabulary_sha256="5ee0f927bcaa4b9fe85c244776ae9487468e427f83e053fc81f2a186f14936a3",
    merge_count=247_587,
    ordered_merges_sha256="7e299304d9ad9dc312acdbcb1f6ccf0dce1256bf1aa986d651f13814dfd27e7b",
    materialized_tokenizer_sha256=(
        "d91d6b29a588b072bd90f3598ee9097049b8082f0bc43e8a3b41da604bdfe1ee"
    ),
    special_token_ids=(
        ("<tts_pad>", 248_072),
        ("<tts_text_bos>", 248_073),
        ("<tts_text_bos_single>", 248_075),
        ("<tts_text_eod>", 248_074),
        ("<|audio_end|>", 248_071),
        ("<|audio_pad|>", 248_076),
        ("<|audio_start|>", 248_070),
        ("<|box_end|>", 248_050),
        ("<|box_start|>", 248_049),
        ("<|endoftext|>", 248_044),
        ("<|im_end|>", 248_046),
        ("<|im_start|>", 248_045),
        ("<|image_pad|>", 248_056),
        ("<|object_ref_end|>", 248_048),
        ("<|object_ref_start|>", 248_047),
        ("<|quad_end|>", 248_052),
        ("<|quad_start|>", 248_051),
        ("<|video_pad|>", 248_057),
        ("<|vision_end|>", 248_054),
        ("<|vision_pad|>", 248_055),
        ("<|vision_start|>", 248_053),
    ),
    representative_encodings=(
        ("Hello, world! 12345", (9419, 11, 1814, 0, 220, 16, 17, 18, 19, 20)),
        ("  spaced  text\n", (220, 61674, 220, 1414, 198)),
        ("你好，世界！", (109266, 3709, 96748, 6115)),
        ("Café — κόσμος 🚀", (34, 2492, 933, 1892, 166265, 203260, 10838, 248, 222)),
        (
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
            (248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198),
        ),
        ("<|audio_start|><|audio_pad|><|audio_end|>", (248070, 248076, 248071)),
    ),
)

_TOKENIZER_EVIDENCE = MappingProxyType(
    {_QWEN35_08B_Q4_TOKENIZER.evidence_id: _QWEN35_08B_Q4_TOKENIZER}
)


def tokenizer_evidence(evidence_id: str) -> GGUFTokenizerEvidence | None:
    """Return exact tokenizer evidence by stable ID."""
    return _TOKENIZER_EVIDENCE.get(evidence_id)


def matching_tokenizer_evidence(
    source_path: Path,
    gguf_model: Any,
    *,
    metadata_sha256: str | None,
) -> GGUFTokenizerEvidence:
    """Return the unique evidence record matching the complete artifact identity."""
    architecture = gguf_model.architecture
    identity = gguf_artifact_identity(
        source_path,
        gguf_model,
        architecture=architecture,
    )
    metadata = gguf_model.metadata

    def sequence_digest(key: str) -> tuple[int, str]:
        values = metadata.get(key)
        if not isinstance(values, list):
            return 0, ""
        payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
        return len(values), hashlib.sha256(payload).hexdigest()

    token_count, vocabulary_sha256 = sequence_digest("tokenizer.ggml.tokens")
    merge_count, merges_sha256 = sequence_digest("tokenizer.ggml.merges")
    token_types = metadata.get("tokenizer.ggml.token_type")
    matches = [
        evidence
        for evidence in _TOKENIZER_EVIDENCE.values()
        if evidence.architecture == architecture
        and evidence.filename == identity.filename
        and evidence.size == identity.size
        and evidence.lfs_sha256 == identity.sha256
        and evidence.tensor_count == identity.tensor_count
        and evidence.tensor_qtypes == identity.tensor_qtypes
        and evidence.tokenizer_metadata_sha256 == metadata_sha256
        and evidence.token_count == token_count
        and evidence.ordered_vocabulary_sha256 == vocabulary_sha256
        and evidence.merge_count == merge_count
        and evidence.ordered_merges_sha256 == merges_sha256
        and gguf_model.get_tensor_shape("token_embd.weight")[0]
        == evidence.embedding_vocabulary_size
        and all(
            token_id < token_count
            and metadata["tokenizer.ggml.tokens"][token_id] == token
            and isinstance(token_types, list)
            and token_types[token_id] in {2, 3}
            for token, token_id in evidence.special_token_ids
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "No unique exact tokenizer evidence matches "
            f"architecture={architecture!r}, artifact={identity!r}, "
            f"metadata_sha256={metadata_sha256!r}."
        )
    return matches[0]
