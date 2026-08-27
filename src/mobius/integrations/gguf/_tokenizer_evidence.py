# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Artifact-scoped evidence for exact GGUF tokenizer materialization."""

from __future__ import annotations

__all__ = [
    "GGUFTokenizerBlockerEvidence",
    "GGUFTokenizerEvidence",
    "iter_tokenizer_blocker_evidence",
    "iter_tokenizer_evidence",
    "matching_tokenizer_evidence",
    "tokenizer_blocker_evidence",
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
from mobius.integrations.gguf._tokenizer_alias_evidence import tokenizer_alias_evidence
from mobius.integrations.gguf._tokenizer_registry import tokenizer_pre_policies


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFTokenizerEvidence:
    """Immutable artifact, source-tokenizer, and semantic validation evidence."""

    evidence_id: str
    architecture: str
    pre_identifier: str
    validated_identifiers: tuple[str, ...]
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
    ordered_merges_sha256: str | None
    score_count: int
    ordered_scores_sha256: str | None
    ordered_token_types_sha256: str
    materialized_tokenizer_sha256: str
    special_token_ids: tuple[tuple[str, int], ...]
    representative_encodings: tuple[tuple[str, tuple[int, ...]], ...]
    representative_special_encodings: tuple[tuple[str, tuple[int, ...]], ...] = ()
    reconstruct_gpt4o_from_gguf: bool = False
    source_disposition: str | None = None
    llamacpp_oracle: tuple[str, int, str] | None = None

    def __post_init__(self) -> None:
        revisions = (self.revision, self.tokenizer_revision)
        digests = (
            self.lfs_sha256,
            self.tokenizer_metadata_sha256,
            self.ordered_vocabulary_sha256,
            self.ordered_token_types_sha256,
            self.materialized_tokenizer_sha256,
        )
        if any(re.fullmatch(r"[0-9a-f]{40}", value) is None for value in revisions):
            raise ValueError("Tokenizer evidence revisions must be immutable 40-hex commits")
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digests):
            raise ValueError("Tokenizer evidence digests must be lowercase SHA-256")
        if (
            min(
                self.size,
                self.tensor_count,
                self.token_count,
                self.source_token_count,
                self.embedding_vocabulary_size,
            )
            <= 0
        ):
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
        policies = tokenizer_pre_policies()
        if (
            not self.validated_identifiers
            or tuple(sorted(set(self.validated_identifiers))) != self.validated_identifiers
            or self.pre_identifier not in self.validated_identifiers
        ):
            raise ValueError(
                "Tokenizer evidence identifiers must be sorted, unique, and include pre"
            )
        try:
            witness = policies[self.pre_identifier]
            aliases = tuple(policies[identifier] for identifier in self.validated_identifiers)
        except KeyError as error:
            raise ValueError(
                f"Tokenizer evidence contains an unknown identifier: {error.args[0]}"
            ) from error
        if any(
            alias.canonical != witness.canonical or alias.pre_type != witness.pre_type
            for alias in aliases
        ):
            raise ValueError(
                "Tokenizer evidence may be shared only across an exact pinned semantic group"
            )
        if len(self.validated_identifiers) > 1:
            proofs = tokenizer_alias_evidence()
            if any(
                identifier not in proofs
                or proofs[identifier].pre_type != witness.pre_type
                or proofs[identifier].flag_overrides
                != proofs[self.pre_identifier].flag_overrides
                for identifier in self.validated_identifiers
            ):
                raise ValueError(
                    "Shared tokenizer evidence requires exact pinned dispatch proof for every alias"
                )
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
        if self.merge_count < 0 or (
            (self.merge_count == 0) != (self.ordered_merges_sha256 is None)
        ):
            raise ValueError("Tokenizer evidence merge count and digest disagree")
        if (
            self.ordered_merges_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.ordered_merges_sha256) is None
        ):
            raise ValueError("Tokenizer evidence merge digest must be lowercase SHA-256")
        if self.score_count < 0 or (
            (self.score_count == 0) != (self.ordered_scores_sha256 is None)
        ):
            raise ValueError("Tokenizer evidence score count and digest disagree")
        if (
            self.ordered_scores_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.ordered_scores_sha256) is None
        ):
            raise ValueError("Tokenizer evidence score digest must be lowercase SHA-256")
        if self.merge_count == 0 and self.score_count == 0:
            raise ValueError("Tokenizer evidence requires ordered merges or ordered scores")
        if not self.representative_encodings or any(
            not text or not token_ids for text, token_ids in self.representative_encodings
        ):
            raise ValueError("Tokenizer evidence requires non-empty representative encodings")
        if any(
            not text or not token_ids
            for text, token_ids in self.representative_special_encodings
        ):
            raise ValueError(
                "Tokenizer evidence representative special encodings must be non-empty"
            )
        if witness.pre_type == "GPT2_ADD_SEP" and not self.representative_special_encodings:
            raise ValueError(
                "GPT2_ADD_SEP evidence requires a representative special-token encoding"
            )
        if self.reconstruct_gpt4o_from_gguf != (self.source_disposition is not None):
            raise ValueError(
                "GGUF-native reconstruction requires an exact official-source disposition"
            )
        if self.reconstruct_gpt4o_from_gguf and witness.pre_type != "GPT4O":
            raise ValueError("GGUF-native reconstruction is supported only for GPT4O evidence")
        if witness.pre_type == "GPT4O" and self.llamacpp_oracle is None:
            raise ValueError("GPT4O evidence requires an exact pinned llama.cpp oracle")
        if self.llamacpp_oracle is not None:
            commit, case_count, digest = self.llamacpp_oracle
            dispatch = tokenizer_alias_evidence().get(self.pre_identifier)
            if (
                re.fullmatch(r"[0-9a-f]{40}", commit) is None
                or case_count <= 0
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or dispatch is None
                or commit != dispatch.source_commit
            ):
                raise ValueError("llama.cpp oracle evidence identity is invalid")

    @property
    def source(self) -> GGUFTokenizerSource:
        """The exact source accepted by the materializer."""
        return GGUFTokenizerSource(
            self.tokenizer_repository,
            self.tokenizer_revision,
            tuple(GGUFTokenizerAsset(*asset) for asset in self.tokenizer_assets),
            self.tokenizer_metadata_sha256,
            self.materialized_tokenizer_sha256,
            self.representative_encodings,
            self.representative_special_encodings,
            self.reconstruct_gpt4o_from_gguf,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFTokenizerBlockerEvidence:
    """Immutable evidence that a pinned artifact cannot use its official tokenizer."""

    evidence_id: str
    architecture: str
    pre_identifier: str
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
    tokenizer_assets: tuple[tuple[str, int, str], ...]
    tokenizer_metadata_sha256: str
    token_count: int
    source_token_count: int
    embedding_vocabulary_size: int
    deterministic_padding_range: tuple[int, int]
    ordered_vocabulary_sha256: str
    source_vocabulary_sha256: str
    merge_count: int
    ordered_merges_sha256: str
    source_merges_sha256: str
    score_count: int
    ordered_scores_sha256: str | None
    ordered_token_types_sha256: str
    source_added_tokens_sha256: str
    source_pipeline_sha256: str
    chat_template_sha256: str
    source_normalizer: str
    special_token_ids: tuple[tuple[str, int], ...]
    oracle_corpus_sha256: str
    llamacpp_oracle: tuple[str, int, str]
    mismatch: tuple[str, tuple[int, ...], tuple[int, ...]]
    disposition: str

    def __post_init__(self) -> None:
        digests = (
            self.lfs_sha256,
            self.tokenizer_metadata_sha256,
            self.ordered_vocabulary_sha256,
            self.source_vocabulary_sha256,
            self.ordered_merges_sha256,
            self.source_merges_sha256,
            self.ordered_token_types_sha256,
            self.source_added_tokens_sha256,
            self.source_pipeline_sha256,
            self.chat_template_sha256,
            self.oracle_corpus_sha256,
            self.llamacpp_oracle[2],
        )
        if any(
            re.fullmatch(r"[0-9a-f]{40}", revision) is None
            for revision in (self.revision, self.tokenizer_revision, self.llamacpp_oracle[0])
        ):
            raise ValueError("Tokenizer blocker revisions must be immutable 40-hex commits")
        if any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in digests):
            raise ValueError("Tokenizer blocker digests must be lowercase SHA-256")
        if self.score_count < 0 or (
            (self.score_count == 0) != (self.ordered_scores_sha256 is None)
        ):
            raise ValueError("Tokenizer blocker score count and digest disagree")
        if (
            self.ordered_scores_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.ordered_scores_sha256) is None
        ):
            raise ValueError("Tokenizer blocker score digest must be lowercase SHA-256")
        if (
            min(
                self.size,
                self.tensor_count,
                self.token_count,
                self.source_token_count,
                self.embedding_vocabulary_size,
                self.merge_count,
                self.llamacpp_oracle[1],
            )
            <= 0
        ):
            raise ValueError("Tokenizer blocker counts and artifact size must be positive")
        if (
            self.source_token_count > self.token_count
            or self.embedding_vocabulary_size != self.token_count
            or self.deterministic_padding_range
            != (self.source_token_count, self.token_count - 1)
        ):
            raise ValueError(
                "Tokenizer blocker vocabulary, padding, and embedding sizes disagree"
            )
        if tuple(sorted(self.tokenizer_assets)) != self.tokenizer_assets:
            raise ValueError("Tokenizer blocker assets must be sorted")
        assets = (self.source_config_asset, *self.tokenizer_assets)
        if self.source_config_asset[0] != "config.json" or any(
            not name or size <= 0 or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for name, size, digest in assets
        ):
            raise ValueError("Tokenizer blocker requires exact source asset identities")
        if tuple(sorted(self.special_token_ids)) != self.special_token_ids:
            raise ValueError("Tokenizer blocker special token IDs must be sorted")
        if any(
            token_id < 0 or token_id >= self.token_count
            for _, token_id in self.special_token_ids
        ):
            raise ValueError("Tokenizer blocker special token IDs must be in the vocabulary")
        if self.pre_identifier not in tokenizer_pre_policies():
            raise ValueError("Tokenizer blocker contains an unknown pre identifier")
        if not self.source_normalizer:
            raise ValueError("Tokenizer blocker requires explicit source normalizer semantics")
        dispatch = tokenizer_alias_evidence().get(self.pre_identifier)
        if dispatch is None or dispatch.source_commit != self.llamacpp_oracle[0]:
            raise ValueError("Tokenizer blocker oracle is not pinned to dispatch evidence")
        text, llamacpp_ids, source_ids = self.mismatch
        if not text or not llamacpp_ids or not source_ids or llamacpp_ids == source_ids:
            raise ValueError("Tokenizer blocker requires an exact non-empty mismatch witness")
        if any(
            token_id < 0 or token_id >= self.token_count
            for token_id in (*llamacpp_ids, *source_ids)
        ):
            raise ValueError("Tokenizer blocker mismatch IDs must be in the vocabulary")
        if not self.disposition:
            raise ValueError("Tokenizer blocker requires a fail-closed disposition")


_QWEN35_08B_Q4_TOKENIZER = GGUFTokenizerEvidence(
    evidence_id="qwen3.5-0.8b-q4-tokenizer",
    architecture="qwen35",
    pre_identifier="qwen35",
    validated_identifiers=("qwen35",),
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
    score_count=0,
    ordered_scores_sha256=None,
    ordered_token_types_sha256=(
        "f6fdca1063d1ae1cc77ba1f5087d259f044c2634e64b65e31bc844ec00e9acab"
    ),
    materialized_tokenizer_sha256=(
        "a78b900eb4cd335bba249158066db523ce221f744e2b6144692bb81673d551af"
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
        ("你好，世界！", (109266, 3709, 96748, 6115)),  # noqa: RUF001
        ("Café — κόσμος 🚀", (34, 2492, 933, 1892, 166265, 203260, 10838, 248, 222)),
        (
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
            (248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198),
        ),
        ("<|audio_start|><|audio_pad|><|audio_end|>", (248070, 248076, 248071)),
    ),
)

_GPT2_Q4_TOKENIZER = GGUFTokenizerEvidence(
    evidence_id="gpt2-q4-tokenizer",
    architecture="gpt2",
    pre_identifier="gpt-2",
    validated_identifiers=(
        "a.x-4.0",
        "exaone4",
        "gigachat",
        "gpt-2",
        "jina-de",
        "jina-es",
        "jina-v2-de",
        "jina-v2-es",
        "mellum",
        "modern-bert",
        "phi-2",
    ),
    repository="QuantFactory/gpt2-GGUF",
    revision="7eae6f079f0164bff66b86eea5159f7a368f9381",
    filename="gpt2.Q4_0.gguf",
    size=106_554_880,
    lfs_sha256="d52ac7ed12e1f87cbc93473912f5c213d7c7d6f2a0112ea9d78533d0d7bd3632",
    tensor_count=149,
    tensor_qtypes=(("F32", 99), ("Q4_0", 49), ("Q6_K", 1)),
    tokenizer_repository="openai-community/gpt2",
    tokenizer_revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
    source_config_asset=(
        "config.json",
        665,
        "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
    ),
    tokenizer_metadata_sha256="b2417176025f8500d864004b0bf93b1403dc3c52238f6628f82fb0e3c498977e",
    tokenizer_assets=(
        (
            "tokenizer.json",
            1_355_256,
            "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
        ),
        (
            "tokenizer_config.json",
            26,
            "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
        ),
    ),
    token_count=50_257,
    source_token_count=50_257,
    embedding_vocabulary_size=50_257,
    deterministic_padding_range=(50_257, 50_256),
    ordered_vocabulary_sha256="2d4e96560e324abcfaeaac6d24016c22f354ef9da213432b451e9a860b21d508",
    merge_count=50_000,
    ordered_merges_sha256="e707935c815087d8103fec742a07d7e8b50d1acf997ddcde53c73213db1141a4",
    score_count=0,
    ordered_scores_sha256=None,
    ordered_token_types_sha256=(
        "aeaf4bf6f00438b0ef9ee1edb6e58616d49b4735d520393af901b7ccf1e3a218"
    ),
    materialized_tokenizer_sha256=(
        "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
    ),
    special_token_ids=(("<|endoftext|>", 50_256),),
    representative_encodings=(
        ("Hello, world! 12345", (15496, 11, 995, 0, 17031, 2231)),
        ("  spaced  text\n", (220, 38980, 220, 2420, 198)),
        (
            "你好，世界！",  # noqa: RUF001
            (19526, 254, 25001, 121, 171, 120, 234, 10310, 244, 45911, 234, 171, 120, 223),
        ),
        (
            "Café — κόσμος 🚀",
            (
                34,
                1878,
                2634,
                851,
                7377,
                118,
                139,
                234,
                38392,
                34703,
                26517,
                35558,
                12520,
                248,
                222,
            ),
        ),
        ("<|endoftext|>", (50_256,)),
    ),
)

_KANANA2_13B_Q8_TOKENIZER = GGUFTokenizerEvidence(
    evidence_id="kanana2-1.3b-instruct-q8-tokenizer",
    architecture="qwen3",
    pre_identifier="kanana2",
    validated_identifiers=("kanana2",),
    repository="dummy9996/kanana-2-1.3b-instruct-GGUF",
    revision="6c998111f40f3ab7adf65620a6a752230d8c75f6",
    filename="kanana-2-1.3b-instruct-Q8_0.gguf",
    size=1_377_890_688,
    lfs_sha256="0b63b6b68f0c1f0e667ad070808dfee7a03db06fc41c3fe23c9d794841c6f801",
    tensor_count=354,
    tensor_qtypes=(("F32", 129), ("Q8_0", 225)),
    tokenizer_repository="kakaocorp/kanana-2-1.3b-instruct",
    tokenizer_revision="bf4786aa2a1908adce942d53976270132732f720",
    source_config_asset=(
        "config.json",
        2_019,
        "fe14b20b4b616d62ca0682312c2fcd2b90d9a836d14a1ff6448db3f533fd15a1",
    ),
    tokenizer_metadata_sha256="94c64f4813926cc68c2357c49a4f264a4788422f5b663299de0d6da63c4546e4",
    tokenizer_assets=(
        (
            "chat_template.jinja",
            10_725,
            "b8ee6b31575eada17ebbe73d3f1ac65d3efde64f0a25ff922031dec7e1cae3e3",
        ),
        (
            "tokenizer.json",
            10_057_457,
            "1c4be9ecf77c926456fb82d4cf07ff1218a91907f3408f44895d2b01e0f2b5ab",
        ),
        (
            "tokenizer_config.json",
            50_155,
            "1cdee8fcd4f6209e07e6d9966c8a3ff2d738830d79475193e94e448e153ae2d5",
        ),
    ),
    token_count=128_256,
    source_token_count=128_256,
    embedding_vocabulary_size=128_256,
    deterministic_padding_range=(128_256, 128_255),
    ordered_vocabulary_sha256="ba8fcb1c6a9186257d3e12f93bd8b77a50378a8e081c524c9c97ff652d34c941",
    merge_count=127_744,
    ordered_merges_sha256="04e73d514ad172c5f02929d7517a6a01a9b42630223e365f06b20d6628a8d1b6",
    score_count=0,
    ordered_scores_sha256=None,
    ordered_token_types_sha256=(
        "552780454e0de07e46ed452dba9c24001a780923e0ba1e4b4d0906cb02c2aeab"
    ),
    materialized_tokenizer_sha256=(
        "1fba3871de7549016b48a7890d46403fe24c59cf39afc9df395ea02c199d1917"
    ),
    special_token_ids=(
        ("<|begin_of_text|>", 128_000),
        ("<|end_of_text|>", 128_001),
        ("<|endoftext|>", 128_008),
        ("<|eom_id|>", 128_005),
        ("<|eot_id|>", 128_004),
        ("<|im_end|>", 128_010),
        ("<|im_start|>", 128_009),
    ),
    representative_encodings=(
        ("Hello, world! 12345", (17_263, 11, 1_666, 0, 220, 9_654, 2_995)),
        ("  spaced  text\n", (220, 49_580, 220, 2_620, 198)),
        ("你好，世界！", (117_006, 6_936, 9_428, 31_645)),  # noqa: RUF001
        (
            "Café — κόσμος 🚀",
            (34, 3_685, 1_989, 3_968, 77_914, 70_157, 23_526, 29_669, 98_218, 106_943, 222),
        ),
        (
            "<|im_start|>user\n안녕하세요<|im_end|>",
            (128_009, 3_043, 198, 15_191, 128_010),
        ),
    ),
    representative_special_encodings=(
        (
            "Hello, world! 12345",
            (128_000, 17_263, 11, 1_666, 0, 220, 9_654, 2_995),
        ),
    ),
    llamacpp_oracle=(
        "8d9af256337d1a501250f9bbf4c0859a654bddd6",
        444,
        "ca7875445f21a03eb9a480c6aa96251bf4a8951a6e284dc480ef32eaedb796f5",
    ),
)

_TALKIE_13B_Q4_TOKENIZER = GGUFTokenizerEvidence(
    evidence_id="talkie-13b-q4-native-tokenizer",
    architecture="talkie",
    pre_identifier="talkie",
    validated_identifiers=("talkie",),
    repository="PocketAiHub/talkie-1930-13b-it-GGUF",
    revision="47b38329dd30e8b2d6ab8e2fc53f3f2ae789e694",
    filename="talkie-1930-13b-it-Q4_K_M.gguf",
    size=8_571_072_704,
    lfs_sha256="2d6c6c1d98a1b8ffa38b50916454891a31ad844ee69c686e525976867917d7b2",
    tensor_count=362,
    tensor_qtypes=(
        ("BF16", 80),
        ("Q4_K", 221),
        ("Q5_0", 20),
        ("Q6_K", 21),
        ("Q8_0", 20),
    ),
    tokenizer_repository="lewtun/talkie-1930-13b-it-hf",
    tokenizer_revision="6311dedf518470856a8503f2080bb4b54fcb3323",
    source_config_asset=(
        "config.json",
        522,
        "e7f29da9cf0a69571d6a0521cd912dc5c2f0dd151d0e934b87541f4389a9ee30",
    ),
    tokenizer_metadata_sha256="7e14f443006afd16e49969f0bfbc5c995edde0075a829f2748e86b9fe4f2da81",
    tokenizer_assets=(
        (
            "chat_template.jinja",
            343,
            "833a35215bfc10d1d9f27fb857123cc24bfef90f770fbc8d79ce37bf4ef4bc4d",
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
    token_count=65_540,
    source_token_count=65_540,
    embedding_vocabulary_size=65_540,
    deterministic_padding_range=(65_540, 65_539),
    ordered_vocabulary_sha256="f88816a5099baf479e674c8d3c61ed31f97954bd8213d1bf269cbbb883012b9e",
    merge_count=65_279,
    ordered_merges_sha256="addf973bfd18babde5e7bfd7fe5f8e7fc3ae2f5fa01a7fdb4aba5c1898f0ec94",
    score_count=0,
    ordered_scores_sha256=None,
    ordered_token_types_sha256=(
        "511820f1cc6c9a5df9e0c93a062ff95f6d602208050aa12296fb66b798ad36cf"
    ),
    materialized_tokenizer_sha256=(
        "63eb55af29f6eb88b2a8caa7966e0202b59f799d4e560bc688b5ac7c5f0453de"
    ),
    special_token_ids=(
        ("<|assistant|>", 65_538),
        ("<|endoftext|>", 65_535),
        ("<|end|>", 65_536),
        ("<|system|>", 65_539),
        ("<|user|>", 65_537),
    ),
    representative_encodings=(
        ("Hello, world! 12345", (72, 22_882, 44, 1_490, 33, 32, 6_276, 1_400)),
        ("  spaced  text\n", (32, 25_156, 32, 5_272, 10)),
        (
            "你好，世界！",  # noqa: RUF001
            (
                228,
                189,
                160,
                229,
                165,
                189,
                239,
                188,
                140,
                57_632,
                150,
                231,
                149,
                140,
                239,
                188,
                129,
            ),
        ),
        (
            "Café — κόσμος 🚀",
            (
                67,
                1_063,
                1_238,
                461,
                12_887,
                12_562,
                6_076,
                7_938,
                14_917,
                32,
                240,
                159,
                154,
                128,
            ),
        ),
        ("<|user|>hello<|end|>", (65_537, 257, 12_227, 65_536)),
        (
            "<|system|>Be concise.<|end|><|user|>你好 12345!<|end|><|assistant|>",
            (
                65_539,
                3_664,
                32_185,
                46,
                65_536,
                65_537,
                228,
                189,
                160,
                229,
                165,
                189,
                32,
                6_276,
                1_400,
                33,
                65_536,
                65_538,
            ),
        ),
    ),
    reconstruct_gpt4o_from_gguf=True,
    source_disposition=(
        "official copy rejected: GGUF retains 65279 of 156379 source merges; first "
        "ordered mismatch at index 4 is ('Ġ', 'the') versus ('Ġt', 'he')"
    ),
    llamacpp_oracle=(
        "8d9af256337d1a501250f9bbf4c0859a654bddd6",
        444,
        "484246b629d6eec375ebac3672e4f4d4fb29646d3b331917ec4d2cfe385c3b6a",
    ),
)

_JINA_V2_CODE_Q8_TOKENIZER = GGUFTokenizerEvidence(
    evidence_id="jina-v2-code-q8-tokenizer",
    architecture="jina-bert-v2",
    pre_identifier="jina-v2-code",
    validated_identifiers=("jina-v2-code",),
    repository="ggml-org/jina-embeddings-v2-base-code-Q8_0-GGUF",
    revision="05e79e9a6c8b99491e92ebb28d753268f8601e3c",
    filename="jina-embeddings-v2-base-code-q8_0.gguf",
    size=172_869_280,
    lfs_sha256="3bd1722f09350209aa3ada93df55882666c58194bfbbbe81c30545d731cb4e7a",
    tensor_count=268,
    tensor_qtypes=(("F32", 183), ("Q8_0", 85)),
    tokenizer_repository="jinaai/jina-embeddings-v2-base-code",
    tokenizer_revision="516f4baf13dec4ddddda8631e019b5737c8bc250",
    source_config_asset=(
        "config.json",
        1_216,
        "e426aa684c7f9a95c5f020aa855faf93a24f065f5fad0c9e17b124670cabdea6",
    ),
    tokenizer_metadata_sha256="30161844cf4cd814a532f368f372a6ba0c7c2c7d86d9678f9816505122e889e5",
    tokenizer_assets=(
        (
            "special_tokens_map.json",
            280,
            "06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f",
        ),
        (
            "tokenizer.json",
            2_561_316,
            "b01c78a902aa4facb2f47f95449f48e2f7bbfea5d2472ee2f6ce92323c6f86e5",
        ),
        (
            "tokenizer_config.json",
            493,
            "f477aeb15ff9f78d3c1ddf2361d2b0b8b20cf55220f839f29a37f3a18efddd89",
        ),
    ),
    token_count=61_056,
    source_token_count=61_056,
    embedding_vocabulary_size=61_056,
    deterministic_padding_range=(61_056, 61_055),
    ordered_vocabulary_sha256="5a8d8f6a0dad37e10cb27a75ece22d44e96259419bdfe8f4e334a676483d9f78",
    merge_count=60_795,
    ordered_merges_sha256="3c0cca5349df26b361d55c0498b4a46fb52a92a05e185754cf709b8de160c5da",
    score_count=0,
    ordered_scores_sha256=None,
    ordered_token_types_sha256=(
        "821b8840b71bceb8d52cacc36f2cc574f0c2e73f8df002cc0936566e6ba20043"
    ),
    materialized_tokenizer_sha256=(
        "b01c78a902aa4facb2f47f95449f48e2f7bbfea5d2472ee2f6ce92323c6f86e5"
    ),
    special_token_ids=(
        ("</s>", 2),
        ("<mask>", 4),
        ("<pad>", 1),
        ("<s>", 0),
        ("<unk>", 3),
    ),
    representative_encodings=(
        ("Hello, world! 12345", (10564, 16, 7550, 5, 53737)),
        ("  spaced  text\n", (225, 4113, 72, 225, 1460, 203)),
        ("你好，世界！", (12552, 19692, 2397, 47406, 32039, 19513)),  # noqa: RUF001
        (
            "Café — κόσμος 🚀",
            (39, 1326, 2521, 25956, 31085, 20788, 10123, 59430, 14535, 30975),
        ),
        (
            (
                "def fibonacci(n: int) -> int:\n"
                "    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)"
            ),
            (
                406,
                23852,
                267,
                33404,
                12,
                82,
                30,
                577,
                13,
                1882,
                577,
                30,
                287,
                437,
                321,
                392,
                321,
                318,
                491,
                723,
                23852,
                267,
                33404,
                12,
                82,
                17,
                21,
                13,
                464,
                23852,
                267,
                33404,
                12,
                82,
                17,
                22,
                13,
            ),
        ),
    ),
    representative_special_encodings=(
        ("Hello, world! 12345", (0, 10564, 16, 7550, 5, 53737, 2)),
    ),
)

_ROBERTA_BPE_Q2_TOKENIZER = GGUFTokenizerEvidence(
    evidence_id="roberta-bpe-q2-tokenizer",
    architecture="bert",
    pre_identifier="roberta-bpe",
    validated_identifiers=("roberta-bpe",),
    repository="mradermacher/quora-roberta-base-GGUF",
    revision="7a6d5816bb01c2d917978fb36825d9fec3ce4ff4",
    filename="quora-roberta-base.Q2_K.gguf",
    size=67_888_768,
    lfs_sha256="1d31ba38f70d6f1456cfbd10c48dc6100a11a9b90558e110a9fb4d940b77cb49",
    tensor_count=201,
    tensor_qtypes=(("F16", 1), ("F32", 126), ("Q2_K", 37), ("Q3_K", 36), ("Q6_K", 1)),
    tokenizer_repository="sentence-transformers/stsb-roberta-base",
    tokenizer_revision="32d471df2968a46d1fe447d66a9275e8e63fcf12",
    source_config_asset=(
        "config.json",
        672,
        "05fac50b3f0e2782f88ba1349ede146230005edef945fb336eeb6f9a8d815940",
    ),
    tokenizer_metadata_sha256="9bc381b15c316f8ced2658ec079c0b2d5ea6c6dcddb615f2a6966bbb717bde74",
    tokenizer_assets=(
        (
            "special_tokens_map.json",
            239,
            "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d",
        ),
        (
            "tokenizer.json",
            1_355_881,
            "33465117406b9007673e8ba283f7f1383d9b5094df947481af60eec94ed7d7bd",
        ),
        (
            "tokenizer_config.json",
            1_172,
            "5992009790ef0a4ba5910d1e0dc04c4e0601d416131501080c694231548bf666",
        ),
    ),
    token_count=50_265,
    source_token_count=50_265,
    embedding_vocabulary_size=50_265,
    deterministic_padding_range=(50_265, 50_264),
    ordered_vocabulary_sha256="db935e2c7440742d76167108001403b9c51be6de99a5f677d70be0771e446cf5",
    merge_count=50_000,
    ordered_merges_sha256="e707935c815087d8103fec742a07d7e8b50d1acf997ddcde53c73213db1141a4",
    score_count=0,
    ordered_scores_sha256=None,
    ordered_token_types_sha256=(
        "70211118fea54968b4980f036554fd37269fc3c205dddbfc2b21004f570df7c3"
    ),
    materialized_tokenizer_sha256=(
        "33465117406b9007673e8ba283f7f1383d9b5094df947481af60eec94ed7d7bd"
    ),
    special_token_ids=(
        ("</s>", 2),
        ("<mask>", 50_264),
        ("<pad>", 1),
        ("<s>", 0),
        ("<unk>", 3),
    ),
    representative_encodings=(
        ("Hello, world! 12345", (31414, 6, 232, 328, 17072, 1898)),
        ("  spaced  text\n", (1437, 42926, 1437, 2788, 50118)),
        (
            "你好，世界！",  # noqa: RUF001
            (
                47856,
                21402,
                48975,
                10809,
                43251,
                4394,
                14285,
                46015,
                25448,
                49127,
                14285,
                43251,
                4394,
                10172,
            ),
        ),
        (
            "Café — κόσμος 🚀",
            (
                347,
                2001,
                1140,
                93,
                43662,
                3070,
                45704,
                14285,
                47721,
                47049,
                46122,
                47756,
                8103,
                15113,
                7471,
            ),
        ),
        (
            "The quick brown fox jumps over the lazy dog.",
            (133, 2119, 6219, 23602, 13855, 81, 5, 22414, 2335, 4),
        ),
    ),
    representative_special_encodings=(
        ("Hello, world! 12345", (0, 31414, 6, 232, 328, 17072, 1898, 2)),
    ),
)

_SMOLLM_135M_F16_TOKENIZER = GGUFTokenizerEvidence(
    evidence_id="smollm-135m-f16-tokenizer",
    architecture="llama",
    pre_identifier="smollm",
    validated_identifiers=("smollm",),
    repository="neopolita/smollm-135m-gguf",
    revision="22cca988936eafe92908e7558907c3964e10bba7",
    filename="ggml-model-f16.gguf",
    size=270_885_504,
    lfs_sha256="ec8c775c16944a7e4b5251f97b3f848500dcc3e701b0d492ce9055cea42138a2",
    tensor_count=272,
    tensor_qtypes=(("F16", 211), ("F32", 61)),
    tokenizer_repository="HuggingFaceTB/SmolLM-135M",
    tokenizer_revision="1d461723eec654e65efdc40cf49301c89c0c92f4",
    source_config_asset=(
        "config.json",
        724,
        "a1fe6f43e20f7a6c6dbc6380222af9526b5cef262446391a281c038249e3e3b7",
    ),
    tokenizer_metadata_sha256="46646ba36ecae43de6f9f649d217774b889e0fd405af92205319b882927493fc",
    tokenizer_assets=(
        (
            "special_tokens_map.json",
            831,
            "e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3",
        ),
        (
            "tokenizer.json",
            2_104_556,
            "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
        ),
        (
            "tokenizer_config.json",
            3_685,
            "238ad6b60d48e471624ea70bc79e92f2611844d5016471fee8c167854bcb98e8",
        ),
    ),
    token_count=49_152,
    source_token_count=49_152,
    embedding_vocabulary_size=49_152,
    deterministic_padding_range=(49_152, 49_151),
    ordered_vocabulary_sha256="ecc2f33f7cdf683196646ea97b005f82398e5ddbb0e143fbe95a402277eb1788",
    merge_count=48_900,
    ordered_merges_sha256="3d6f4016bc9b70ea16f0f01b1dadb4504ad99c5eaa8584b81997dc65168e136b",
    score_count=0,
    ordered_scores_sha256=None,
    ordered_token_types_sha256=(
        "3a92d63c9763834e17f2d93490d5a9643fa07057f2799168d67d7812d08e31aa"
    ),
    materialized_tokenizer_sha256=(
        "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
    ),
    special_token_ids=(("<|endoftext|>", 0),),
    representative_encodings=(
        ("Hello, world! 12345", (19556, 28, 905, 17, 216, 33, 34, 35, 36, 37)),
        ("  spaced  text\n", (216, 23861, 216, 1694, 198)),
        ("你好，世界！", (18645, 250, 48392, 138, 12831, 7906, 240, 178, 239, 230, 8083, 219)),  # noqa: RUF001
        (
            "Café — κόσμος 🚀",
            (51, 1939, 2756, 1841, 31953, 36180, 18751, 16674, 39346, 15107, 244, 218),
        ),
        ("<|endoftext|>", (0,)),
    ),
)

_QWEN25_05B_Q8_TOKENIZER = GGUFTokenizerEvidence(
    evidence_id="qwen2.5-0.5b-instruct-q8-tokenizer",
    architecture="qwen2",
    pre_identifier="qwen2",
    validated_identifiers=("deepseek-r1-qwen", "f2llmv2", "kormo", "qwen2"),
    repository="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    revision="9217f5db79a29953eb74d5343926648285ec7e67",
    filename="qwen2.5-0.5b-instruct-q8_0.gguf",
    size=675_710_816,
    lfs_sha256="ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
    tensor_count=291,
    tensor_qtypes=(("F32", 121), ("Q8_0", 170)),
    tokenizer_repository="Qwen/Qwen2.5-0.5B-Instruct",
    tokenizer_revision="a338b55dd21219a5f4da42bc11a9313d1a27d4cc",
    source_config_asset=(
        "config.json",
        659,
        "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    ),
    tokenizer_metadata_sha256="8fc8ef848104e931f14ae03d9581699d54813a2ff952fb7caac0654e8aa27ee3",
    tokenizer_assets=(
        (
            "tokenizer.json",
            7_031_645,
            "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        ),
        (
            "tokenizer_config.json",
            7_308,
            "5214600ee45ca2f887ce2eede8910378a0111ea99d657428bcbce94778e65a92",
        ),
    ),
    token_count=151_936,
    source_token_count=151_936,
    embedding_vocabulary_size=151_936,
    deterministic_padding_range=(151_936, 151_935),
    ordered_vocabulary_sha256="e2fadeac783c911f535d21f858f43127672a1d261af510d3f895e34bd2f6fb10",
    merge_count=151_387,
    ordered_merges_sha256="24fa2ae2a398e50784a1fff678482094af4f63e6783d35686726abacda8dc371",
    score_count=0,
    ordered_scores_sha256=None,
    ordered_token_types_sha256=(
        "17ccfa7767a8721474dc0fd21ca1308fdfd04e0f64036efbfa97f3e16e5f18f1"
    ),
    materialized_tokenizer_sha256=(
        "be55f66f0643df9d3c1b5dc55ae552b0e334f219a3a5f8338e6864f8eb3a8ac5"
    ),
    special_token_ids=(("<|endoftext|>", 151643), ("<|im_end|>", 151645)),
    representative_encodings=(
        ("Hello, world! 12345", (9707, 11, 1879, 0, 220, 16, 17, 18, 19, 20)),
        ("  spaced  text\n", (220, 63828, 220, 1467, 198)),
        ("你好，世界！", (108386, 3837, 99489, 6313)),  # noqa: RUF001
        (
            "Café — κόσμος 🚀",
            (34, 2577, 963, 1959, 71638, 75195, 43928, 43123, 27554, 45642, 11162, 248, 222),
        ),
        ("<|im_start|>user\nHello<|im_end|>\n", (151644, 872, 198, 9707, 151645, 198)),
    ),
)

_PLM_18B_Q4_K_M_TOKENIZER_BLOCKER = GGUFTokenizerBlockerEvidence(
    evidence_id="plm-1.8b-instruct-q4-k-m-tokenizer-blocker",
    architecture="plm",
    pre_identifier="qwen2",
    repository="PLM-Team/PLM-1.8B-Instruct-gguf",
    revision="7bec6546983bcf0d99526c943580bd49e2237445",
    filename="PLM-1.8B-Instruct-Q4_K_M.gguf",
    size=1_182_708_992,
    lfs_sha256="b38570ee56ebec82a1e9ef45ab408c0d8230ececef1d7f1b267c49cff35638b8",
    tensor_count=290,
    tensor_qtypes=(("F32", 97), ("Q4_K", 176), ("Q6_K", 17)),
    tokenizer_repository="PLM-Team/PLM-1.8B-Instruct",
    tokenizer_revision="62d188c7d58843d7013d5b3ffe198db448787860",
    source_config_asset=(
        "config.json",
        934,
        "91e6e13695a6de82556438667e64b60d9910269f300cd97f8c667d19e75f115e",
    ),
    tokenizer_assets=(
        (
            "merges.txt",
            1_671_853,
            "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        ),
        (
            "special_tokens_map.json",
            410,
            "c83747485fba9ef20c42793b4b02b05001f214250f0d787f573df216c91047a3",
        ),
        (
            "tokenizer.json",
            11_418_266,
            "bcfe42da0a4497e8b2b172c1f9f4ec423a46dc12907f4349c55025f670422ba9",
        ),
        (
            "tokenizer_config.json",
            1_327,
            "1becffcfa09c98935043f1724d988887c618c5f6e7a249087d3ae29eb70e2a6f",
        ),
        (
            "vocab.json",
            2_776_833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
    ),
    tokenizer_metadata_sha256="698bfa31cd069292437bc3509fea7be2445324a536d95a6e813d947c283bd989",
    token_count=151_936,
    source_token_count=151_646,
    embedding_vocabulary_size=151_936,
    deterministic_padding_range=(151_646, 151_935),
    ordered_vocabulary_sha256="f3ea8e8cf45bd58a8d5ad420306a3ccd925894cdb61a89062fd9e3a6de255a0e",
    source_vocabulary_sha256="696f26322524de87f49427fd1be6d1afce910574d9656c5d3f4f64064bdb83c1",
    merge_count=151_387,
    ordered_merges_sha256="24fa2ae2a398e50784a1fff678482094af4f63e6783d35686726abacda8dc371",
    source_merges_sha256="cc098baa4a74ce5156487605aa048a34e54f0eee6a704691a738c8fb22dafdd5",
    score_count=0,
    ordered_scores_sha256=None,
    ordered_token_types_sha256=(
        "0286431feb975d95a59a3f39957f8183d929295635fb627b6940627c63918bf1"
    ),
    source_added_tokens_sha256=(
        "e7b5f7013431aa26739424d92f40423f175a46f6d1fdc8453edf6005c99412f7"
    ),
    source_pipeline_sha256="97c53ee89fb584b10798f44b02c60c9a8b746165a32dc34737d178fc20618a69",
    chat_template_sha256="af9c0233881b083b52ff773580215222b5440ac3d0beeeca99b76329b048f8db",
    source_normalizer="NFC",
    special_token_ids=(
        ("<|endoftext|>", 151643),
        ("<|im_end|>", 151645),
        ("<|im_start|>", 151644),
    ),
    oracle_corpus_sha256="0f3c77d24208f2ac0833668128cb2a00adcb7e6b4b4eedc6e4783e3ec7b41f5a",
    llamacpp_oracle=(
        "8d9af256337d1a501250f9bbf4c0859a654bddd6",
        72,
        "66513168812575ccac974ecb454e916def5f4492d558c1866b8811d4f587a41d",
    ),
    mismatch=("é é", (68, 53839, 3958), (963, 3958)),
    disposition=(
        "official tokenizer.json applies NFC normalization, but pinned llama.cpp qwen2 "
        "preserves decomposed Unicode; exact materialization is blocked"
    ),
)

_LFM2_350M_F16_TOKENIZER = GGUFTokenizerEvidence(
    evidence_id="lfm2-350m-f16-tokenizer",
    architecture="lfm2",
    pre_identifier="lfm2",
    validated_identifiers=(
        "falcon-h1",
        "falcon3",
        "jina-v5-nano",
        "lfm2",
        "llama-bpe",
        "llama-v3",
        "llama3",
        "midm-2.0",
        "pixtral",
    ),
    repository="LiquidAI/LFM2-350M-GGUF",
    revision="8fdc9d526b7ed346b19257551b05816c7912ecc2",
    filename="LFM2-350M-F16.gguf",
    size=711_482_304,
    lfs_sha256="379ffdcbf08147c0313f6f1ce7ff558a2bc935eda633f4b46c52347032419c42",
    tensor_count=148,
    tensor_qtypes=(("F16", 93), ("F32", 55)),
    tokenizer_repository="LiquidAI/LFM2-350M",
    tokenizer_revision="73e3c253078a3b97c2e14b4c4665679f4d9b6d56",
    source_config_asset=(
        "config.json",
        999,
        "fd3b3fba4e50e7b9a22bd41cbab59e9b28e319b2de19668d7fd9777c8d1a9ba1",
    ),
    tokenizer_metadata_sha256="e5626d605bb50bc53fdb0fbfcf374fb33dfbaa0cc698d9746ba1e9b0b7e6d07d",
    tokenizer_assets=(
        (
            "chat_template.jinja",
            209,
            "a805e50fed68938a076b07e2e602639611b50b1ced0e50f11eb92f1ba25be4dc",
        ),
        (
            "special_tokens_map.json",
            434,
            "742aefe2b7dec496e8caffdba03a75d0c1a9925d53bd3f3e0d388c96b591b6f4",
        ),
        (
            "tokenizer.json",
            4_732_426,
            "98cff83b4f6d7e9d8929bebc62b07e92cf1b3f99c80d16bafe8b84a75448f40b",
        ),
        (
            "tokenizer_config.json",
            91_509,
            "36f511115e9d8952cbc9d15d9a20dfa7ce7d1444940e5c1dc42a762020c99bf5",
        ),
    ),
    token_count=65_536,
    source_token_count=65_536,
    embedding_vocabulary_size=65_536,
    deterministic_padding_range=(65_536, 65_535),
    ordered_vocabulary_sha256="c004fd0578dbfbff394335a7d5f95e78a8cdbbff6abc8c389ba2290637be58b6",
    merge_count=63_683,
    ordered_merges_sha256="c70042d0b5969460432a218556522dedee908735a3e4cf70f27936353c5b3f65",
    score_count=0,
    ordered_scores_sha256=None,
    ordered_token_types_sha256=(
        "ffe1ea561257dc6e1f2c257b99b4913d63e9d6b896cf2f9da1a3d2cac316d4b4"
    ),
    materialized_tokenizer_sha256=(
        "e7b7960966e2ed43a22b00431246cf820d5e2751bec58c44f0184cbe9b8d18c9"
    ),
    special_token_ids=(("<|im_end|>", 7), ("<|pad|>", 0), ("<|startoftext|>", 1)),
    representative_encodings=(
        ("Hello, world! 12345", (36309, 521, 2031, 510, 730, 10293, 2637)),
        ("  spaced  text\n", (730, 56551, 730, 3304, 708)),
        ("你好，世界！", (11754, 6400, 1198, 11370, 8668)),  # noqa: RUF001
        (
            "Café — κόσμος 🚀",
            (544, 2305, 860, 2180, 59955, 49122, 27443, 16883, 51332, 23805, 758, 732),
        ),
        ("<|startoftext|><|im_start|>user\nHello<|im_end|>", (1, 6, 6423, 708, 36309, 7)),
    ),
)

_TOKENIZER_EVIDENCE = MappingProxyType(
    {
        record.evidence_id: record
        for record in (
            _GPT2_Q4_TOKENIZER,
            _KANANA2_13B_Q8_TOKENIZER,
            _TALKIE_13B_Q4_TOKENIZER,
            _JINA_V2_CODE_Q8_TOKENIZER,
            _LFM2_350M_F16_TOKENIZER,
            _QWEN25_05B_Q8_TOKENIZER,
            _QWEN35_08B_Q4_TOKENIZER,
            _ROBERTA_BPE_Q2_TOKENIZER,
            _SMOLLM_135M_F16_TOKENIZER,
        )
    }
)

_TOKENIZER_BLOCKER_EVIDENCE = MappingProxyType(
    {_PLM_18B_Q4_K_M_TOKENIZER_BLOCKER.evidence_id: (_PLM_18B_Q4_K_M_TOKENIZER_BLOCKER)}
)


def tokenizer_evidence(evidence_id: str) -> GGUFTokenizerEvidence | None:
    """Return exact tokenizer evidence by stable ID."""
    return _TOKENIZER_EVIDENCE.get(evidence_id)


def iter_tokenizer_evidence() -> tuple[GGUFTokenizerEvidence, ...]:
    """Return every tokenizer evidence record in stable evidence-ID order."""
    return tuple(_TOKENIZER_EVIDENCE[key] for key in sorted(_TOKENIZER_EVIDENCE))


def tokenizer_blocker_evidence(evidence_id: str) -> GGUFTokenizerBlockerEvidence | None:
    """Return exact fail-closed tokenizer evidence by stable ID."""
    return _TOKENIZER_BLOCKER_EVIDENCE.get(evidence_id)


def iter_tokenizer_blocker_evidence() -> tuple[GGUFTokenizerBlockerEvidence, ...]:
    """Return every fail-closed tokenizer record in stable evidence-ID order."""
    return tuple(
        _TOKENIZER_BLOCKER_EVIDENCE[key] for key in sorted(_TOKENIZER_BLOCKER_EVIDENCE)
    )


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
    score_count, scores_sha256 = sequence_digest("tokenizer.ggml.scores")
    token_type_count, token_types_sha256 = sequence_digest("tokenizer.ggml.token_type")
    token_types = metadata.get("tokenizer.ggml.token_type")
    matches = [
        evidence
        for evidence in _TOKENIZER_EVIDENCE.values()
        if evidence.architecture == architecture
        and evidence.pre_identifier == metadata.get("tokenizer.ggml.pre")
        and evidence.filename == identity.filename
        and evidence.size == identity.size
        and evidence.lfs_sha256 == identity.sha256
        and evidence.tensor_count == identity.tensor_count
        and evidence.tensor_qtypes == identity.tensor_qtypes
        and evidence.tokenizer_metadata_sha256 == metadata_sha256
        and evidence.token_count == token_count
        and evidence.ordered_vocabulary_sha256 == vocabulary_sha256
        and evidence.merge_count == merge_count
        and evidence.ordered_merges_sha256 == (merges_sha256 or None)
        and evidence.score_count == score_count
        and evidence.ordered_scores_sha256 == (scores_sha256 or None)
        and evidence.token_count == token_type_count
        and evidence.ordered_token_types_sha256 == token_types_sha256
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
    blockers = [
        evidence
        for evidence in _TOKENIZER_BLOCKER_EVIDENCE.values()
        if evidence.architecture == architecture
        and evidence.pre_identifier == metadata.get("tokenizer.ggml.pre")
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
        and evidence.score_count == score_count
        and evidence.ordered_scores_sha256 == (scores_sha256 or None)
        and evidence.token_count == token_type_count
        and evidence.ordered_token_types_sha256 == token_types_sha256
        and gguf_model.get_tensor_shape("token_embd.weight")[0]
        == evidence.embedding_vocabulary_size
    ]
    if len(blockers) == 1:
        blocker = blockers[0]
        raise ValueError(
            f"Tokenizer materialization is explicitly blocked by {blocker.evidence_id!r}: "
            f"{blocker.disposition}"
        )
    if blockers:
        raise RuntimeError("Tokenizer blocker evidence contains duplicate artifact identities")
    if len(matches) != 1:
        raise ValueError(
            "No unique exact tokenizer evidence matches "
            f"architecture={architecture!r}, artifact={identity!r}, "
            f"metadata_sha256={metadata_sha256!r}."
        )
    return matches[0]
