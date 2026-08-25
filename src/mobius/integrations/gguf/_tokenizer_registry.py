# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Immutable tokenizer pre-type census from the pinned llama.cpp revision."""

from __future__ import annotations

__all__ = [
    "PINNED_TOKENIZER_PRE_COUNT",
    "TokenizerPrePolicy",
    "tokenizer_pre_policies",
]

import dataclasses
import functools

from mobius.integrations.gguf._upstream import UPSTREAM_COMMIT

PINNED_TOKENIZER_PRE_COUNT = 87


@dataclasses.dataclass(frozen=True, slots=True)
class TokenizerPrePolicy:
    """One exact ``tokenizer.ggml.pre`` dispatch accepted by pinned llama.cpp."""

    identifier: str
    canonical: str
    pre_type: str
    default_route: str = "deferred"


# Aliases share a canonical name only when the pinned C++ branch selects the
# same pre-type and the same hard-coded flag overrides. Regex similarity alone
# is not enough: e.g. DBRX uses the Llama-3 regex without Llama-3's BOS and
# ignore-merges overrides, so it remains a separate canonical policy.
_EXACT_ALIAS_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "llama3",
        "LLAMA3",
        (
            "llama3",
            "llama-v3",
            "llama-bpe",
            "falcon3",
            "falcon-h1",
            "pixtral",
            "midm-2.0",
            "lfm2",
            "jina-v5-nano",
        ),
    ),
    (
        "gpt-2",
        "GPT2",
        (
            "gpt-2",
            "phi-2",
            "jina-es",
            "jina-de",
            "gigachat",
            "jina-v2-es",
            "jina-v2-de",
            "a.x-4.0",
            "mellum",
            "modern-bert",
            "exaone4",
        ),
    ),
    ("gemma4", "GEMMA4", ("gemma4", "granite-embed-multi-311m")),
    ("jina-v1-en", "GPT2_ADD_SEP", ("jina-v1-en", "jina-v2-code", "roberta-bpe")),
    ("qwen2", "QWEN2", ("qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2")),
    ("glm4", "CHATGLM4", ("glm4", "chatglm-bpe")),
    ("gpt-4o", "GPT4O", ("gpt-4o", "llama4", "kanana2", "talkie")),
    ("tiny_aya", "TINY_AYA", ("tiny_aya", "cohere2moe")),
    ("bailingmoe", "BAILINGMOE", ("bailingmoe", "bailingmoe2", "llada-moe")),
)

_SINGLETONS: tuple[tuple[str, str], ...] = (
    ("default", "DEFAULT"),
    ("minicpm5", "MINICPM5"),
    ("deepseek-llm", "DEEPSEEK_LLM"),
    ("deepseek-coder", "DEEPSEEK_CODER"),
    ("deepseek-v3", "DEEPSEEK3_LLM"),
    ("youtu", "YOUTU"),
    ("falcon", "FALCON"),
    ("mpt", "MPT"),
    ("starcoder", "STARCODER"),
    ("jais-2", "JAIS2"),
    ("sarvam-moe", "SARVAM_MOE"),
    ("whitespace", "WHITESPACE"),
    ("refact", "REFACT"),
    ("command-r", "COMMAND_R"),
    ("qwen35", "QWEN35"),
    ("stablelm2", "STABLELM2"),
    ("olmo", "OLMO"),
    ("dbrx", "DBRX"),
    ("smaug-bpe", "SMAUG"),
    ("poro-chat", "PORO"),
    ("viking", "VIKING"),
    ("jais", "JAIS"),
    ("tekken", "TEKKEN"),
    ("smollm", "SMOLLM"),
    ("codeshell", "CODESHELL"),
    ("bloom", "BLOOM"),
    ("gpt3-finnish", "GPT3_FINNISH"),
    ("exaone", "EXAONE"),
    ("exaone-moe", "EXAONE_MOE"),
    ("chameleon", "CHAMELEON"),
    ("minerva-7b", "MINERVA"),
    ("megrez", "QWEN2_CLEAN_SPACES"),
    ("granite-embed-multi-97m", "GRANITE_EMB_MULTI"),
    ("superbpe", "SUPERBPE"),
    ("trillion", "TRILLION"),
    ("granite-docling", "GRANITE_DOCLING"),
    ("seed-coder", "SEED_CODER"),
    ("hunyuan", "HUNYUAN"),
    ("hunyuan-dense", "HUNYUAN_DENSE"),
    ("joyai-llm", "JOYAI_LLM"),
    ("kimi-k2", "KIMI_K2"),
    ("grok-2", "GROK_2"),
    ("afmoe", "AFMOE"),
    ("laguna", "LAGUNA"),
    ("minimax-m2", "MINIMAX_M2"),
    ("solar-open", "SOLAR_OPEN"),
    ("mellum2", "MELLUM2"),
)

_PINNED_IDENTIFIERS = (
    "default",
    "minicpm5",
    "llama3",
    "llama-v3",
    "llama-bpe",
    "falcon3",
    "falcon-h1",
    "pixtral",
    "midm-2.0",
    "lfm2",
    "jina-v5-nano",
    "deepseek-llm",
    "deepseek-coder",
    "deepseek-v3",
    "youtu",
    "falcon",
    "mpt",
    "starcoder",
    "gpt-2",
    "phi-2",
    "jina-es",
    "jina-de",
    "gigachat",
    "jina-v2-es",
    "jina-v2-de",
    "a.x-4.0",
    "mellum",
    "modern-bert",
    "jais-2",
    "gemma4",
    "granite-embed-multi-311m",
    "sarvam-moe",
    "jina-v1-en",
    "jina-v2-code",
    "roberta-bpe",
    "whitespace",
    "refact",
    "command-r",
    "qwen2",
    "deepseek-r1-qwen",
    "kormo",
    "f2llmv2",
    "qwen35",
    "stablelm2",
    "olmo",
    "dbrx",
    "smaug-bpe",
    "poro-chat",
    "glm4",
    "chatglm-bpe",
    "viking",
    "jais",
    "tekken",
    "smollm",
    "codeshell",
    "bloom",
    "gpt3-finnish",
    "exaone",
    "exaone4",
    "exaone-moe",
    "chameleon",
    "minerva-7b",
    "megrez",
    "gpt-4o",
    "llama4",
    "kanana2",
    "talkie",
    "granite-embed-multi-97m",
    "tiny_aya",
    "cohere2moe",
    "superbpe",
    "trillion",
    "granite-docling",
    "bailingmoe",
    "bailingmoe2",
    "llada-moe",
    "seed-coder",
    "hunyuan",
    "hunyuan-dense",
    "joyai-llm",
    "kimi-k2",
    "grok-2",
    "afmoe",
    "laguna",
    "minimax-m2",
    "solar-open",
    "mellum2",
)


@functools.lru_cache(maxsize=1)
def tokenizer_pre_policies() -> dict[str, TokenizerPrePolicy]:
    """Return all 87 pinned identifiers, keyed by their exact GGUF spelling."""
    unordered: dict[str, TokenizerPrePolicy] = {}
    for canonical, pre_type, identifiers in _EXACT_ALIAS_GROUPS:
        for identifier in identifiers:
            unordered[identifier] = TokenizerPrePolicy(identifier, canonical, pre_type)
    for identifier, pre_type in _SINGLETONS:
        unordered[identifier] = TokenizerPrePolicy(identifier, identifier, pre_type)
    policies = {identifier: unordered[identifier] for identifier in _PINNED_IDENTIFIERS}
    if len(policies) != PINNED_TOKENIZER_PRE_COUNT:
        raise RuntimeError(
            f"Tokenizer pre census for llama.cpp {UPSTREAM_COMMIT} has {len(policies)} "
            f"entries, expected {PINNED_TOKENIZER_PRE_COUNT}."
        )
    return policies
