# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned artifact evidence for tokenizer routes that must remain fail-closed."""

from __future__ import annotations

__all__ = [
    "GGUFTokenizerMismatchEvidence",
    "iter_tokenizer_mismatch_evidence",
    "tokenizer_mismatch_evidence",
]

import dataclasses
import re
from types import MappingProxyType

from mobius.integrations.gguf._upstream import UPSTREAM_COMMIT


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFTokenizerMismatchEvidence:
    """Immutable header, source, and oracle evidence for a rejected tokenizer route."""

    evidence_id: str
    architecture: str
    tokenizer_model: str
    pre_identifier: str
    repository: str
    revision: str
    filename: str
    size: int
    lfs_sha256: str
    bounded_header_bytes: int
    bounded_header_sha256: str
    tensor_count: int
    tensor_qtypes: tuple[tuple[str, int], ...]
    tokenizer_repository: str
    tokenizer_revision: str
    source_config_asset: tuple[str, int, str]
    tokenizer_assets: tuple[tuple[str, int, str], ...]
    tokenizer_metadata_sha256: str
    token_count: int
    source_model_token_count: int
    embedding_vocabulary_size: int
    ordered_vocabulary_sha256: str
    score_count: int
    ordered_scores_sha256: str
    source_score_mismatch_count: int
    ordered_token_types_sha256: str
    source_type_mismatch_count: int
    source_merge_count: int
    ordered_source_merges_sha256: str
    special_token_ids: tuple[tuple[str, int], ...]
    gguf_chat_template_sha256: str
    source_chat_template_sha256: str
    llamacpp_oracle: tuple[str, int, str]
    source_oracle_sha256: str
    oracle_mismatch_count: int
    oracle_mismatch_count_by_mode: tuple[int, ...]
    first_mismatch_mode: tuple[str, str]
    first_mismatch: tuple[str, tuple[int, ...], tuple[int, ...]]
    disposition: str

    def __post_init__(self) -> None:
        revisions = (self.revision, self.tokenizer_revision, self.llamacpp_oracle[0])
        if any(re.fullmatch(r"[0-9a-f]{40}", value) is None for value in revisions):
            raise ValueError("Tokenizer mismatch revisions must be immutable 40-hex commits")
        if self.llamacpp_oracle[0] != UPSTREAM_COMMIT:
            raise ValueError("Tokenizer mismatch oracle must use the pinned llama.cpp commit")
        digests = (
            self.lfs_sha256,
            self.bounded_header_sha256,
            self.tokenizer_metadata_sha256,
            self.ordered_vocabulary_sha256,
            self.ordered_scores_sha256,
            self.ordered_token_types_sha256,
            self.ordered_source_merges_sha256,
            self.gguf_chat_template_sha256,
            self.source_chat_template_sha256,
            self.llamacpp_oracle[2],
            self.source_oracle_sha256,
            self.source_config_asset[2],
            *(asset[2] for asset in self.tokenizer_assets),
        )
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digests):
            raise ValueError("Tokenizer mismatch digests must be lowercase SHA-256")
        counts = (
            self.size,
            self.bounded_header_bytes,
            self.tensor_count,
            self.token_count,
            self.source_model_token_count,
            self.embedding_vocabulary_size,
            self.score_count,
            self.source_score_mismatch_count,
            self.source_type_mismatch_count,
            self.source_merge_count,
            self.llamacpp_oracle[1],
            self.oracle_mismatch_count,
        )
        if min(counts) <= 0:
            raise ValueError("Tokenizer mismatch evidence counts must be positive")
        if self.embedding_vocabulary_size != self.token_count:
            raise ValueError("Tokenizer mismatch vocabulary and embedding rows disagree")
        if self.source_model_token_count > self.token_count:
            raise ValueError("Tokenizer mismatch source model exceeds the GGUF vocabulary")
        if sum(self.oracle_mismatch_count_by_mode) != self.oracle_mismatch_count:
            raise ValueError("Tokenizer mismatch per-mode counts disagree with the total")
        if self.first_mismatch_mode not in {
            ("no-add", "no-parse-special"),
            ("no-add", "parse-special"),
            ("add-special", "parse-special"),
        }:
            raise ValueError("Tokenizer mismatch first-mode witness is invalid")
        if tuple(sorted(self.tensor_qtypes)) != self.tensor_qtypes:
            raise ValueError("Tokenizer mismatch qtypes must be sorted")
        if sum(count for _, count in self.tensor_qtypes) != self.tensor_count:
            raise ValueError("Tokenizer mismatch qtype counts must cover every tensor")
        if tuple(sorted(self.tokenizer_assets)) != self.tokenizer_assets:
            raise ValueError("Tokenizer mismatch assets must be sorted")
        if tuple(sorted(self.special_token_ids)) != self.special_token_ids:
            raise ValueError("Tokenizer mismatch special IDs must be sorted by token")
        if self.gguf_chat_template_sha256 == self.source_chat_template_sha256:
            raise ValueError("Tokenizer mismatch evidence requires a chat-template divergence")
        if self.llamacpp_oracle[2] == self.source_oracle_sha256:
            raise ValueError("Tokenizer mismatch evidence requires an oracle divergence")
        text, actual_ids, source_ids = self.first_mismatch
        if not text or not actual_ids or not source_ids or actual_ids == source_ids:
            raise ValueError("Tokenizer mismatch evidence requires a concrete first mismatch")
        if not self.disposition:
            raise ValueError("Tokenizer mismatch evidence requires a fail-closed disposition")


_RECORDS = (
    GGUFTokenizerMismatchEvidence(
        evidence_id="minicpm-2b-q2-k-tokenizer-mismatch",
        architecture="minicpm",
        tokenizer_model="llama",
        pre_identifier="default",
        repository="mzwing/MiniCPM-2B-sft-bf16-GGUF",
        revision="121e7290609857006939fca0ec64981009b806b9",
        filename="MiniCPM-2B-sft-bf16.Q2_K.gguf",
        size=1_204_392_288,
        lfs_sha256="9e87235097895a22894c32a1e211f94b93b798d715cc7d99c8e846637927fd13",
        bounded_header_bytes=16_777_216,
        bounded_header_sha256=(
            "ff140dcf42ce544e61a8e3bd23a09edbfb1a573d7df3f266081bcbd06e33ab66"
        ),
        tensor_count=362,
        tensor_qtypes=(("F32", 81), ("IQ4_NL", 40), ("Q2_K", 160), ("Q3_K", 80), ("Q6_K", 1)),
        tokenizer_repository="openbmb/MiniCPM-2B-sft-bf16",
        tokenizer_revision="4ec16344ac13e6ef5010aeecaa533369ac8eb53c",
        source_config_asset=(
            "config.json",
            1_010,
            "41cf26cfdca93f49209a6c0c26c00b281d6408e010ddc8cca9531c825cc17fc4",
        ),
        tokenizer_assets=(
            (
                "special_tokens_map.json",
                414,
                "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
            ),
            (
                "tokenizer.json",
                6_202_715,
                "42f73d01995bd71c88647b13ac696b36d84d5126d0c7cbaef6f8d872d5c97dff",
            ),
            (
                "tokenizer.model",
                1_994_871,
                "c9aafcd7da1f5611dab6be545db74d5552a2ccc9c2a12c72ea7be63aac4a25d7",
            ),
            (
                "tokenizer_config.json",
                1_117,
                "9c87efade54e9b26d3374a42f294d266722cbf9e97c748a12183d8317da074c9",
            ),
        ),
        tokenizer_metadata_sha256=(
            "1a596fac46038c46f2fa273fd8ec10cf0721acd2ed6daf2121cfc0840da9ee3e"
        ),
        token_count=122_753,
        source_model_token_count=122_753,
        embedding_vocabulary_size=122_753,
        ordered_vocabulary_sha256=(
            "154c42653082e70f4a98ed1708e8a71d41494dc1de32fe9316bcb6046b993dbe"
        ),
        score_count=122_753,
        ordered_scores_sha256=(
            "c5bd4e305c0b57143fd35cfea8743ac4af7f0805c8a5787ff22cc0b4c06af085"
        ),
        source_score_mismatch_count=122_752,
        ordered_token_types_sha256=(
            "e89c4d81e916e4d1bee6e7d7842463429004c559d9106d7e8af73934e9e4e0eb"
        ),
        source_type_mismatch_count=1_087,
        source_merge_count=171_540,
        ordered_source_merges_sha256=(
            "4551998aaa2aa2468fc189317e85b7149a1999f38a9eddde19b523c3fa74e4b8"
        ),
        special_token_ids=(("</s>", 2), ("<s>", 1), ("<unk>", 0)),
        gguf_chat_template_sha256=(
            "ae9b050c5a5b0295cb09269e67bf832fa08675dd845c7b3ea4c2130bcacc5c26"
        ),
        source_chat_template_sha256=(
            "d9f25394f8be2d8a5fb234670a8596c7a12fa0d21896b5a757d77da8b8686944"
        ),
        llamacpp_oracle=(
            UPSTREAM_COMMIT,
            444,
            "8959cbd62821def331adc77db30e8b351e8fad221547e74804dc7aecb188df1a",
        ),
        source_oracle_sha256=(
            "a61293d6ad6fb5637f4ceafe225ac822897482d5ef5333b87f63b0e40611e9c2"
        ),
        oracle_mismatch_count=18,
        oracle_mismatch_count_by_mode=(6, 6, 6),
        first_mismatch_mode=("no-add", "no-parse-special"),
        first_mismatch=(
            "\u4f60\u597d\uff0c\u4e16\u754c\uff01",
            (29951, 95495, 65, 2925, 67),
            (95320, 23523, 65, 2925, 67),
        ),
        disposition=(
            "fail-closed: all GGUF scores are -1000, GGUF merges are absent, token types "
            "and chat whitespace semantics diverge, and pinned llama.cpp disagrees with the "
            "official tokenizer on multilingual and whitespace inputs"
        ),
    ),
    GGUFTokenizerMismatchEvidence(
        evidence_id="minicpm3-4b-q4-k-m-tokenizer-mismatch",
        architecture="minicpm3",
        tokenizer_model="llama",
        pre_identifier="default",
        repository="openbmb/MiniCPM3-4B-GGUF",
        revision="816dc79b35f92827e0d2d87aacea3567e49661a8",
        filename="minicpm3-4b-q4_k_m.gguf",
        size=2_469_791_584,
        lfs_sha256="64913247e927414ecf47fd3e9ea8e3f0c9acae293f583dfa7e24b8872e20fa4c",
        bounded_header_bytes=16_777_216,
        bounded_header_sha256=(
            "505ce706e29108bef3579b1b4dc38695fe44923b63f453163175d5024b4ea12e"
        ),
        tensor_count=748,
        tensor_qtypes=(("F32", 251), ("Q4_K", 466), ("Q6_K", 31)),
        tokenizer_repository="openbmb/MiniCPM3-4B",
        tokenizer_revision="d6b14ddaefdb11c624dd75c3c779549bc90b08cb",
        source_config_asset=(
            "config.json",
            1_929,
            "cf1d08cb7c1815c676e685bd6ce94eb8b85a57d53871e6e159ee8c650717d98a",
        ),
        tokenizer_assets=(
            (
                "added_tokens.json",
                216,
                "4760fcbf90bc193f33827ffe02f2e7ba1af1ec43644cc02ac22fdd611f6cca15",
            ),
            (
                "special_tokens_map.json",
                1_632,
                "068594063e37662c02b21acf42ebb334ef6a74fb810e68a2368f88f08351de76",
            ),
            (
                "tokenizer.json",
                3_676_758,
                "b00802b71a613e3f7df3899fe9643a3ff949736d333a2b892448a974383fe372",
            ),
            (
                "tokenizer.model",
                1_181_204,
                "bb74d51116831c3bf65db812c553f94ab0c88dcf97a5bbb37e3504f6d359c530",
            ),
            (
                "tokenizer_config.json",
                10_413,
                "25620d5a3f5727bba2fb403624f2c9a7bba55a7d00205829650cd1e3c646aae0",
            ),
        ),
        tokenizer_metadata_sha256=(
            "6dc004393b6fd1dd27f81c505cadbd8be953244999f8b6fde281fb68dff94c34"
        ),
        token_count=73_448,
        source_model_token_count=73_440,
        embedding_vocabulary_size=73_448,
        ordered_vocabulary_sha256=(
            "1046ac4e64873087a848a2e033be381d18a35974f1ec0c139326c3073ad6744c"
        ),
        score_count=73_448,
        ordered_scores_sha256=(
            "a6ca82ba25f969052af106d021d21d584bafdb21470737e7302f2a51a6ed5711"
        ),
        source_score_mismatch_count=73_439,
        ordered_token_types_sha256=(
            "ae14c00dfeff5f86796d58db7e91a7eee8fd44605d2cd3ec9846f56e52540a7e"
        ),
        source_type_mismatch_count=1_088,
        source_merge_count=104_297,
        ordered_source_merges_sha256=(
            "a6ae9d2ba560703a2f5933b92307f80f9fdebfabc052df1cb5c3542a98441cbc"
        ),
        special_token_ids=(
            ("</s>", 2),
            ("<s>", 1),
            ("<unk>", 0),
            ("<|execute_end|>", 73444),
            ("<|execute_start|>", 73443),
            ("<|fim_middle|>", 73446),
            ("<|fim_prefix|>", 73445),
            ("<|fim_suffix|>", 73447),
            ("<|im_end|>", 73440),
            ("<|im_start|>", 73441),
            ("<|tool_call|>", 73442),
        ),
        gguf_chat_template_sha256=(
            "153280e3ff55d19da1398bdb3914ee2a51b80429bfaedde11d7d216c39db80f3"
        ),
        source_chat_template_sha256=(
            "dbd75fe18b14711fa5968600a6f5c974d7d3e63e75fe163ecb99a1e5f94c38c9"
        ),
        llamacpp_oracle=(
            UPSTREAM_COMMIT,
            444,
            "d55374a7956f5379448b802ac23888c91f22aeb5fb6814f5b73efe74058fd475",
        ),
        source_oracle_sha256=(
            "d6beee40e9257575b1333e43f67709920eda0cd6249fa74cfbd8a5eaa978b4af"
        ),
        oracle_mismatch_count=9,
        oracle_mismatch_count_by_mode=(3, 3, 3),
        first_mismatch_mode=("no-add", "no-parse-special"),
        first_mismatch=(
            "\u4f60\u597d\uff0c\u4e16\u754c\uff01",
            (29951, 59495, 65, 2925, 67),
            (59320, 23523, 65, 2925, 67),
        ),
        disposition=(
            "fail-closed: all GGUF scores are -1000, GGUF merges are absent, token types "
            "diverge, the GGUF drops the official tool-aware chat template, and pinned "
            "llama.cpp disagrees with the official tokenizer on multilingual and whitespace inputs"
        ),
    ),
)

_BY_ID = MappingProxyType({record.evidence_id: record for record in _RECORDS})


def iter_tokenizer_mismatch_evidence() -> tuple[GGUFTokenizerMismatchEvidence, ...]:
    """Return deterministic architecture-scoped fail-closed evidence."""
    return _RECORDS


def tokenizer_mismatch_evidence(
    evidence_id: str,
) -> GGUFTokenizerMismatchEvidence | None:
    """Return one fail-closed evidence record by ID."""
    return _BY_ID.get(evidence_id)
