# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned llama.cpp dispatch evidence for tokenizer aliases validated in batch 2."""

from __future__ import annotations

__all__ = [
    "TOKENIZER_DISPATCH_SOURCE_PATH",
    "TOKENIZER_DISPATCH_SOURCE_SHA256",
    "TokenizerAliasEvidence",
    "tokenizer_alias_evidence",
]

import dataclasses
import functools

from mobius.integrations.gguf._tokenizer_registry import tokenizer_pre_policies
from mobius.integrations.gguf._upstream import UPSTREAM_COMMIT

TOKENIZER_DISPATCH_SOURCE_PATH = "src/llama-vocab.cpp"
TOKENIZER_DISPATCH_SOURCE_SHA256 = (
    "32b8b7ac4a023bc6367eb7832e56e988ab7373a676ddcca4977665f78932ab88"
)


@dataclasses.dataclass(frozen=True, slots=True)
class TokenizerAliasEvidence:
    """Exact dispatch line and effective non-default behavior at the pinned source."""

    identifier: str
    source_commit: str
    source_path: str
    source_sha256: str
    dispatch_line: int
    pre_type: str
    flag_overrides: tuple[str, ...]


_GROUP_BEHAVIOR = {
    "bailingmoe": ("clean_spaces=false",),
    "gemma4": ("escape_whitespaces=true",),
    "glm4": ("special_bos_id=null",),
    "gpt-2": (),
    "gpt-4o": ("clean_spaces=false",),
    "jina-v1-en": ("add_sep=true",),
    "llama3": ("add_bos=true", "ignore_merges=true"),
    "qwen2": ("clean_spaces=false",),
    "tiny_aya": ("clean_spaces=false",),
}

# Line numbers refer to TOKENIZER_DISPATCH_SOURCE_SHA256, not a moving branch.
_DISPATCH_LINES = {
    "llama3": 2148,
    "llama-v3": 2149,
    "llama-bpe": 2150,
    "falcon3": 2151,
    "falcon-h1": 2152,
    "pixtral": 2153,
    "midm-2.0": 2154,
    "lfm2": 2155,
    "jina-v5-nano": 2156,
    "gpt-2": 2187,
    "phi-2": 2188,
    "jina-es": 2189,
    "jina-de": 2190,
    "gigachat": 2191,
    "jina-v2-es": 2192,
    "jina-v2-de": 2193,
    "a.x-4.0": 2194,
    "mellum": 2195,
    "modern-bert": 2196,
    "gemma4": 2202,
    "granite-embed-multi-311m": 2203,
    "jina-v1-en": 2212,
    "jina-v2-code": 2213,
    "roberta-bpe": 2214,
    "qwen2": 2229,
    "deepseek-r1-qwen": 2230,
    "kormo": 2231,
    "f2llmv2": 2232,
    "glm4": 2256,
    "chatglm-bpe": 2257,
    "exaone4": 2290,
    "gpt-4o": 2307,
    "llama4": 2308,
    "kanana2": 2309,
    "talkie": 2310,
    "tiny_aya": 2319,
    "cohere2moe": 2320,
    "bailingmoe": 2336,
    "bailingmoe2": 2337,
    "llada-moe": 2338,
}


@functools.lru_cache(maxsize=1)
def tokenizer_alias_evidence() -> dict[str, TokenizerAliasEvidence]:
    """Return exact batch-2 identifier-to-implementation proofs."""
    policies = tokenizer_pre_policies()
    records = {
        identifier: TokenizerAliasEvidence(
            identifier=identifier,
            source_commit=UPSTREAM_COMMIT,
            source_path=TOKENIZER_DISPATCH_SOURCE_PATH,
            source_sha256=TOKENIZER_DISPATCH_SOURCE_SHA256,
            dispatch_line=line,
            pre_type=policies[identifier].pre_type,
            flag_overrides=_GROUP_BEHAVIOR[policies[identifier].canonical],
        )
        for identifier, line in _DISPATCH_LINES.items()
    }
    if set(records) != set(_DISPATCH_LINES):
        raise RuntimeError("Tokenizer alias evidence contains duplicate identifiers")
    return records
