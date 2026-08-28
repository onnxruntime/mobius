# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate GGUF tokenizer metadata and materialize only exact tokenizer assets."""

from __future__ import annotations

__all__ = [
    "GGUFTokenizerAsset",
    "GGUFTokenizerSource",
    "GGUFTokenizerVerdict",
    "inspect_gguf_tokenizer",
    "materialize_evidenced_gguf_tokenizer",
    "materialize_gguf_tokenizer",
    "write_gguf_tokenizer_json",
]

import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from mobius.integrations.gguf._tokenizer_registry import tokenizer_pre_policies

TokenizerRoute = Literal["copy", "pinned-source", "deferred"]

_BPE_MODELS = frozenset({"gpt2", "hybriddna", "whitespace", "gemma4"})
_KNOWN_MODELS = frozenset(
    {"llama", "bert", "t5", "rwkv", "plamo2", "gpt2", "hybriddna", "whitespace", "gemma4"}
)
_SPECIAL_ID_KEYS = (
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.eom_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.seperator_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.cls_token_id",
    "tokenizer.ggml.mask_token_id",
    "tokenizer.ggml.fim_pre_token_id",
    "tokenizer.ggml.fim_suf_token_id",
    "tokenizer.ggml.fim_mid_token_id",
    "tokenizer.ggml.fim_pad_token_id",
    "tokenizer.ggml.fim_rep_token_id",
    "tokenizer.ggml.fim_sep_token_id",
    "tokenizer.ggml.prefix_token_id",
    "tokenizer.ggml.suffix_token_id",
    "tokenizer.ggml.middle_token_id",
)
_BOOL_KEYS = (
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.add_sep_token",
    "tokenizer.ggml.add_space_prefix",
    "tokenizer.ggml.remove_extra_whitespaces",
    "tokenizer.ggml.normalizer.lowercase",
    "tokenizer.ggml.normalizer.strip_accents",
)
_TOKENIZER_ASSET_NAMES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
_MAX_TOKENIZER_ASSET_BYTES = 64 * 1024 * 1024
_SMOLLM_PIPELINE = {
    "normalizer": None,
    "pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [
            {"type": "Digits", "individual_digits": True},
            {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
        ],
    },
    "post_processor": None,
    "decoder": {
        "type": "ByteLevel",
        "add_prefix_space": True,
        "trim_offsets": True,
        "use_regex": True,
    },
}
_GPT4O_SOURCE_SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|"
    r"\s*[\r\n]+|\s+(?!\S)|\s+"
)
_GPT4O_SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*"
    r"((?=[\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|"
    r"'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?"
    r"((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*"
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|"
    r"\s*[\r\n]+|\s+(?!\S)|\s+"
)
_GPT4O_PRE_TOKENIZER = {
    "type": "Sequence",
    "pretokenizers": [
        {
            "type": "Split",
            "pattern": {"Regex": _GPT4O_SPLIT_PATTERN},
            "behavior": "Isolated",
            "invert": False,
        },
        {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
    ],
}
_GPT4O_POST_BYTE_LEVEL = {
    "type": "ByteLevel",
    "add_prefix_space": True,
    "trim_offsets": False,
    "use_regex": True,
}
_GPT4O_DECODER = {
    "type": "ByteLevel",
    "add_prefix_space": True,
    "trim_offsets": True,
    "use_regex": True,
}
_GEMMA4_NORMALIZER = {
    "type": "Replace",
    "pattern": {"String": " "},
    "content": "▁",
}
_GEMMA4_PRE_TOKENIZER = {
    "type": "Split",
    "pattern": {"String": " "},
    "behavior": "MergedWithPrevious",
    "invert": False,
}
_GEMMA4_SOURCE_POST_PROCESSOR = {
    "type": "TemplateProcessing",
    "single": [{"Sequence": {"id": "A", "type_id": 0}}],
    "pair": [
        {"Sequence": {"id": "A", "type_id": 0}},
        {"Sequence": {"id": "B", "type_id": 1}},
    ],
    "special_tokens": {},
}
_GEMMA4_SOURCE_DECODER = {
    "type": "Sequence",
    "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
    ],
}
_GEMMA4_CLEANUP_REPLACEMENTS = (
    (" ?", "?"),
    (" !", "!"),
    (" .", "."),
    (" ,", ","),
    (" ' ", "'"),
    (" 's", "'s"),
    (" 'm", "'m"),
    (" 're", "'re"),
    (" 've", "'ve"),
)
_GEMMA4_FORCED_CONTROL_TOKENS = frozenset({"<eos>", "<turn|>", "<|tool_response>"})

# Audited against the pinned C++ loader. ``tokenizer.huggingface.json`` and
# ``tokenizer.chat_templates`` are converter/extension fields; llama.cpp does
# not use the former to tokenize and uses the latter only to enumerate names.
LOADER_CONSUMED_TOKENIZER_FIELDS = frozenset(
    {
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.token_type_count",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.precompiled_charsmap",
        "tokenizer.ggml.suppress_tokens",
        "tokenizer.chat_template",
        *_SPECIAL_ID_KEYS,
        *_BOOL_KEYS,
    }
)
CONVERTER_OR_EXTENSION_TOKENIZER_FIELDS = frozenset(
    {"tokenizer.huggingface.json", "tokenizer.rwkv.world", "tokenizer.chat_templates"}
)


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFTokenizerVerdict:
    """Validated tokenizer policy for one GGUF source."""

    route: TokenizerRoute
    model: str | None
    pre: str | None
    canonical_pre: str | None
    reason: str
    token_count: int
    tokenizer_sha256: str | None = None
    metadata_sha256: str | None = None
    audit_status: str | None = None
    blocker_category: str | None = None
    evidence_id: str | None = None

    @property
    def materialized(self) -> bool:
        return self.route in {"copy", "pinned-source"}

    @property
    def route_identifier(self) -> str:
        """Exact serialized route discriminator used for diagnostics."""
        return self.pre or self.model or "absent"


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFTokenizerAsset:
    """Expected identity of one exact tokenizer source file."""

    filename: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if self.filename not in _TOKENIZER_ASSET_NAMES:
            raise ValueError(f"Unsupported tokenizer asset filename: {self.filename!r}")
        if self.size <= 0 or self.size > _MAX_TOKENIZER_ASSET_BYTES:
            raise ValueError(
                f"Invalid tokenizer asset size for {self.filename!r}: {self.size}"
            )
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError(f"Tokenizer asset {self.filename!r} requires a lowercase SHA-256")


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFTokenizerSource:
    """Immutable Hub source and exact file identities for tokenizer materialization."""

    repository: str
    revision: str
    assets: tuple[GGUFTokenizerAsset, ...]
    metadata_sha256: str
    materialized_tokenizer_sha256: str | None = None
    representative_encodings: tuple[tuple[str, tuple[int, ...]], ...] = ()
    representative_special_encodings: tuple[tuple[str, tuple[int, ...]], ...] = ()
    reconstruct_gpt4o_from_gguf: bool = False
    reconstruct_gemma4_from_gguf: bool = False

    def __post_init__(self) -> None:
        if self.repository.count("/") != 1 or not all(self.repository.split("/")):
            raise ValueError("Tokenizer source repository must be an owner/repository Hub ID")
        if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ValueError(
                "Tokenizer source revision must be an immutable 40-hex commit SHA"
            )
        if re.fullmatch(r"[0-9a-f]{64}", self.metadata_sha256) is None:
            raise ValueError(
                "Tokenizer source requires the exact GGUF tokenizer metadata SHA-256"
            )
        if (
            self.materialized_tokenizer_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.materialized_tokenizer_sha256) is None
        ):
            raise ValueError(
                "Tokenizer source materialized digest must be a lowercase SHA-256"
            )
        if any(not text or not token_ids for text, token_ids in self.representative_encodings):
            raise ValueError("Tokenizer source representative encodings must be non-empty")
        if any(
            not text or not token_ids
            for text, token_ids in self.representative_special_encodings
        ):
            raise ValueError(
                "Tokenizer source representative special encodings must be non-empty"
            )
        names = tuple(asset.filename for asset in self.assets)
        if "tokenizer.json" not in names:
            raise ValueError("Tokenizer source must include tokenizer.json")
        if len(set(names)) != len(names):
            raise ValueError("Tokenizer source contains duplicate asset filenames")
        if names != tuple(sorted(names)):
            raise ValueError("Tokenizer source assets must be sorted by filename")


def _require_list(metadata: Mapping[str, Any], key: str) -> list[Any] | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a GGUF array")
    return value


def _validate_chat_templates(metadata: Mapping[str, Any]) -> dict[str, str]:
    default = metadata.get("tokenizer.chat_template")
    if default is not None and not isinstance(default, str):
        raise ValueError("tokenizer.chat_template must be a string")
    qualified = {
        key.removeprefix("tokenizer.chat_template."): value
        for key, value in metadata.items()
        if key.startswith("tokenizer.chat_template.")
    }
    if any(not name or not isinstance(value, str) for name, value in qualified.items()):
        raise ValueError(
            "named tokenizer chat templates require non-empty names and string values"
        )
    names = _require_list(metadata, "tokenizer.chat_templates")
    if names is not None:
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("tokenizer.chat_templates must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("tokenizer.chat_templates contains duplicate names")
        expected = set(names) - {"default"}
        if expected != set(qualified):
            raise ValueError(
                "tokenizer.chat_templates does not exactly match named "
                "tokenizer.chat_template.<name> fields"
            )
        if ("default" in names) != (default is not None):
            raise ValueError(
                "tokenizer.chat_templates default entry contradicts tokenizer.chat_template"
            )
    templates = dict(sorted(qualified.items()))
    if default is not None:
        templates = {"default": default, **templates}
    return templates


def _validate_embedded_tokenizer_json(raw: str, tokens: list[str]) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("tokenizer.huggingface.json is not valid JSON") from error
    if not isinstance(payload, dict):
        raise TypeError("tokenizer.huggingface.json must contain a JSON object")
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_str(raw)
    except Exception as error:
        raise ValueError(
            "tokenizer.huggingface.json is not a loadable tokenizers tokenizer"
        ) from error
    actual = [tokenizer.id_to_token(index) for index in range(tokenizer.get_vocab_size())]
    if actual != tokens:
        mismatch = next(
            (
                index
                for index, (actual_token, expected_token) in enumerate(zip(actual, tokens))
                if actual_token != expected_token
            ),
            min(len(actual), len(tokens)),
        )
        raise ValueError(
            "tokenizer.huggingface.json vocabulary is not identical to "
            f"tokenizer.ggml.tokens (first mismatch at id {mismatch})"
        )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _tokenizer_metadata_sha256(metadata: Mapping[str, Any]) -> str:
    tokenizer_metadata = {
        key: value for key, value in metadata.items() if key.startswith("tokenizer.")
    }
    canonical = json.dumps(
        tokenizer_metadata,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _deferred_route_diagnostics(
    pre: str | None,
    *,
    reason: str,
) -> tuple[str | None, str | None, str | None, str]:
    """Resolve stable authoritative diagnostics for a deferred tokenizer route."""
    if pre is None:
        return (
            "deferred-incomplete-pipeline",
            "serialized-tokenizer-pipeline-incomplete",
            None,
            reason,
        )

    # Imported lazily because the census evidence records depend on the tokenizer
    # source dataclasses defined in this module.
    from mobius.integrations.gguf._tokenizer_census import tokenizer_route_census

    audit = next(
        (record for record in tokenizer_route_census() if record.identifier == pre),
        None,
    )
    if audit is None or audit.current_status == "validated-pinned-source":
        return None, None, None, reason
    return (
        audit.current_status,
        audit.blocker_category,
        audit.blocker_evidence_id or audit.evidence_id,
        audit.candidate_disposition or reason,
    )


def _incomplete_tokenizer_verdict(
    *,
    model: str | None,
    reason: str,
    token_count: int,
) -> GGUFTokenizerVerdict:
    """Return an authoritative deferred verdict for incomplete serialized metadata."""
    return GGUFTokenizerVerdict(
        "deferred",
        model,
        None,
        None,
        reason,
        token_count,
        audit_status="deferred-incomplete-pipeline",
        blocker_category="serialized-tokenizer-pipeline-incomplete",
    )


def _validate_incomplete_tokenizer_fields(
    metadata: Mapping[str, Any],
    *,
    source: str,
    model: str | None,
) -> int:
    """Validate every serialized field before downgrading an incomplete pipeline."""
    tokens_raw = _require_list(metadata, "tokenizer.ggml.tokens")
    tokens = list(tokens_raw or ())
    if any(not isinstance(token, str) for token in tokens):
        raise ValueError(f"{source} tokenizer.ggml.tokens must contain only UTF-8 strings")
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"{source} tokenizer.ggml.tokens contains duplicate token strings")

    pre = metadata.get("tokenizer.ggml.pre")
    if pre is not None:
        if not isinstance(pre, str) or not pre:
            raise ValueError("tokenizer.ggml.pre must be a non-empty string")
        if pre not in tokenizer_pre_policies():
            raise ValueError(f"{source} declares unknown tokenizer.ggml.pre {pre!r}")
        if model is not None and model not in _BPE_MODELS:
            architecture = metadata.get("general.architecture")
            legacy_default = pre == "default" and (
                model == "plamo2"
                or (
                    model == "llama"
                    and isinstance(architecture, str)
                    and architecture in {"minicpm", "minicpm3"}
                )
            )
            if not legacy_default:
                raise ValueError(
                    f"{source} declares tokenizer.ggml.pre for non-BPE model {model!r}; "
                    "the pinned loader does not consume that combination"
                )

    rwkv_world = metadata.get("tokenizer.rwkv.world")
    if rwkv_world is not None and not isinstance(rwkv_world, str):
        raise ValueError("tokenizer.rwkv.world must be a string")

    embedded = metadata.get("tokenizer.huggingface.json")
    if embedded is not None:
        if not isinstance(embedded, str):
            raise ValueError("tokenizer.huggingface.json must be a string")
        if tokens:
            _validate_embedded_tokenizer_json(embedded, tokens)
        else:
            try:
                payload = json.loads(embedded)
            except json.JSONDecodeError as error:
                raise ValueError("tokenizer.huggingface.json is not valid JSON") from error
            if not isinstance(payload, dict):
                raise TypeError("tokenizer.huggingface.json must contain a JSON object")
            try:
                from tokenizers import Tokenizer

                Tokenizer.from_str(embedded)
            except Exception as error:
                raise ValueError(
                    "tokenizer.huggingface.json is not a loadable tokenizers tokenizer"
                ) from error

    token_types = _require_list(metadata, "tokenizer.ggml.token_type")
    if token_types is not None:
        if len(token_types) != len(tokens):
            raise ValueError(
                "tokenizer.ggml.token_type length must equal tokenizer token count"
            )
        if any(type(value) is not int or value not in range(7) for value in token_types):
            raise ValueError("tokenizer.ggml.token_type values must be integers in [0, 6]")

    scores = _require_list(metadata, "tokenizer.ggml.scores")
    if scores is not None:
        if len(scores) != len(tokens):
            raise ValueError("tokenizer.ggml.scores length must equal tokenizer token count")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in scores
        ):
            raise ValueError("tokenizer.ggml.scores must contain finite numeric values")

    merges = _require_list(metadata, "tokenizer.ggml.merges")
    if merges is not None:
        pairs: list[tuple[str, str]] = []
        vocabulary = set(tokens)
        for index, merge in enumerate(merges):
            if not isinstance(merge, str):
                raise TypeError(f"tokenizer.ggml.merges[{index}] must be a string")
            separator = merge.find(" ", 1)
            if separator < 0:
                raise ValueError(f"tokenizer.ggml.merges[{index}] has no valid pair separator")
            pair = (merge[:separator], merge[separator + 1 :])
            if not all(pair) or pair[0] not in vocabulary or pair[1] not in vocabulary:
                raise ValueError(
                    f"tokenizer.ggml.merges[{index}] references a token outside the vocabulary"
                )
            pairs.append(pair)
        if len(set(pairs)) != len(pairs):
            raise ValueError("tokenizer.ggml.merges contains duplicate merge pairs")

    for key in _SPECIAL_ID_KEYS:
        value = metadata.get(key)
        if value is not None and (
            type(value) is not int or value < 0 or (tokens and value >= len(tokens))
        ):
            upper_bound = str(len(tokens)) if tokens else "the serialized token count"
            raise ValueError(f"{key} must be an integer in [0, {upper_bound})")
    for key in _BOOL_KEYS:
        value = metadata.get(key)
        if value is not None and type(value) is not bool:
            raise ValueError(f"{key} must be a boolean")

    type_count = metadata.get("tokenizer.ggml.token_type_count")
    if type_count is not None and (type(type_count) is not int or type_count <= 0):
        raise ValueError("tokenizer.ggml.token_type_count must be a positive integer")
    suppress = _require_list(metadata, "tokenizer.ggml.suppress_tokens")
    if suppress is not None and any(
        type(value) is not int or value < 0 or value >= len(tokens) for value in suppress
    ):
        raise ValueError("tokenizer.ggml.suppress_tokens contains an out-of-range token id")
    if suppress is not None and len(set(suppress)) != len(suppress):
        raise ValueError("tokenizer.ggml.suppress_tokens contains duplicate token ids")
    charsmap = _require_list(metadata, "tokenizer.ggml.precompiled_charsmap")
    if charsmap is not None and any(
        type(value) is not int or value < -128 or value > 255 for value in charsmap
    ):
        raise ValueError("tokenizer.ggml.precompiled_charsmap must contain byte values")
    if "tokenizer.ggml.byte_fallback" in metadata:
        raise ValueError(
            "tokenizer.ggml.byte_fallback is not a pinned GGUF key; byte-fallback semantics "
            "cannot be inferred from it"
        )
    _validate_chat_templates(metadata)
    return len(tokens)


def inspect_gguf_tokenizer(
    metadata: Mapping[str, Any],
    *,
    source: str = "<GGUF>",
    require_complete: bool = False,
) -> GGUFTokenizerVerdict:
    """Validate embedded tokenizer metadata and return an exact route verdict."""
    tokenizer_keys = [key for key in metadata if key.startswith("tokenizer.")]
    if not tokenizer_keys:
        return _incomplete_tokenizer_verdict(
            model=None,
            reason=f"{source} contains no tokenizer metadata",
            token_count=0,
        )

    model = metadata.get("tokenizer.ggml.model")
    if model is not None and (not isinstance(model, str) or not model):
        raise ValueError(f"{source} tokenizer.ggml.model must be a non-empty string")
    if model is None:
        if not require_complete:
            token_count = _validate_incomplete_tokenizer_fields(
                metadata,
                source=source,
                model=None,
            )
            return _incomplete_tokenizer_verdict(
                model=None,
                reason=(
                    f"{source} contains partial tokenizer metadata without "
                    "tokenizer.ggml.model"
                ),
                token_count=token_count,
            )
        raise ValueError(f"{source} tokenizer.ggml.model must be a non-empty string")
    if model not in _KNOWN_MODELS:
        raise ValueError(f"{source} declares unknown tokenizer.ggml.model {model!r}")

    tokens_raw = _require_list(metadata, "tokenizer.ggml.tokens")
    if not tokens_raw:
        if not require_complete:
            token_count = _validate_incomplete_tokenizer_fields(
                metadata,
                source=source,
                model=model,
            )
            return _incomplete_tokenizer_verdict(
                model=model,
                reason=f"{source} contains no complete tokenizer token table",
                token_count=token_count,
            )
        raise ValueError(f"{source} tokenizer.ggml.tokens must be a non-empty string array")
    if any(not isinstance(token, str) for token in tokens_raw):
        raise ValueError(f"{source} tokenizer.ggml.tokens must contain only UTF-8 strings")
    tokens = list(tokens_raw)
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"{source} tokenizer.ggml.tokens contains duplicate token strings")

    pre_value = metadata.get("tokenizer.ggml.pre")
    if pre_value is None and model == "gemma4":
        pre_value = "gemma4"
    if model in _BPE_MODELS:
        if not isinstance(pre_value, str) or not pre_value:
            raise ValueError(
                f"{source} BPE tokenizer requires tokenizer.ggml.pre; refusing llama.cpp's "
                "quality-degrading generic default"
            )
        policy = tokenizer_pre_policies().get(pre_value)
        if policy is None:
            raise ValueError(f"{source} declares unknown tokenizer.ggml.pre {pre_value!r}")
    elif pre_value is not None:
        # Public PLaMo2 and pinned MiniCPM conversions predate per-model
        # tokenizer cleanup and carry the inert legacy value "default".
        # Validate only those architecture-scoped combinations; exact artifact
        # audits keep tokenizer materialization deferred below.
        architecture = metadata.get("general.architecture")
        legacy_default = pre_value == "default" and (
            model == "plamo2"
            or (
                model == "llama"
                and isinstance(architecture, str)
                and architecture in {"minicpm", "minicpm3"}
            )
        )
        if not legacy_default:
            raise ValueError(
                f"{source} declares tokenizer.ggml.pre for non-BPE model {model!r}; "
                "the pinned loader does not consume that combination"
            )
        policy = tokenizer_pre_policies()[pre_value]
    else:
        policy = None

    token_types = _require_list(metadata, "tokenizer.ggml.token_type")
    if token_types is not None:
        if len(token_types) != len(tokens):
            raise ValueError(
                "tokenizer.ggml.token_type length must equal tokenizer token count"
            )
        if any(type(value) is not int or value not in range(7) for value in token_types):
            raise ValueError("tokenizer.ggml.token_type values must be integers in [0, 6]")

    scores = _require_list(metadata, "tokenizer.ggml.scores")
    if scores is not None:
        if len(scores) != len(tokens):
            raise ValueError("tokenizer.ggml.scores length must equal tokenizer token count")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in scores
        ):
            raise ValueError("tokenizer.ggml.scores must contain finite numeric values")

    merges = _require_list(metadata, "tokenizer.ggml.merges")
    if model in _BPE_MODELS and pre_value != "kimi-k2" and merges is None:
        raise ValueError(f"{source} BPE tokenizer requires tokenizer.ggml.merges")
    if merges is not None:
        pairs: list[tuple[str, str]] = []
        vocab = set(tokens)
        for index, merge in enumerate(merges):
            if not isinstance(merge, str):
                raise TypeError(f"tokenizer.ggml.merges[{index}] must be a string")
            separator = merge.find(" ", 1)
            if separator < 0:
                raise ValueError(f"tokenizer.ggml.merges[{index}] has no valid pair separator")
            pair = (merge[:separator], merge[separator + 1 :])
            if not all(pair) or pair[0] not in vocab or pair[1] not in vocab:
                raise ValueError(
                    f"tokenizer.ggml.merges[{index}] references a token outside the vocabulary"
                )
            pairs.append(pair)
        if len(set(pairs)) != len(pairs):
            raise ValueError("tokenizer.ggml.merges contains duplicate merge pairs")

    for key in _SPECIAL_ID_KEYS:
        value = metadata.get(key)
        if value is not None and (type(value) is not int or value < 0 or value >= len(tokens)):
            raise ValueError(f"{key} must be an integer in [0, {len(tokens)})")
    for key in _BOOL_KEYS:
        value = metadata.get(key)
        if value is not None and type(value) is not bool:
            raise ValueError(f"{key} must be a boolean")

    type_count = metadata.get("tokenizer.ggml.token_type_count")
    if type_count is not None and (type(type_count) is not int or type_count <= 0):
        raise ValueError("tokenizer.ggml.token_type_count must be a positive integer")
    suppress = _require_list(metadata, "tokenizer.ggml.suppress_tokens")
    if suppress is not None and any(
        type(value) is not int or value < 0 or value >= len(tokens) for value in suppress
    ):
        raise ValueError("tokenizer.ggml.suppress_tokens contains an out-of-range token id")
    if suppress is not None and len(set(suppress)) != len(suppress):
        raise ValueError("tokenizer.ggml.suppress_tokens contains duplicate token ids")
    charsmap = _require_list(metadata, "tokenizer.ggml.precompiled_charsmap")
    if charsmap is not None and any(
        type(value) is not int or value < -128 or value > 255 for value in charsmap
    ):
        raise ValueError("tokenizer.ggml.precompiled_charsmap must contain byte values")
    if "tokenizer.ggml.byte_fallback" in metadata:
        raise ValueError(
            "tokenizer.ggml.byte_fallback is not a pinned GGUF key; byte-fallback semantics "
            "cannot be inferred from it"
        )
    _validate_chat_templates(metadata)

    embedded = metadata.get("tokenizer.huggingface.json")
    if embedded is not None:
        if not isinstance(embedded, str):
            raise ValueError("tokenizer.huggingface.json must be a string")
        digest = _validate_embedded_tokenizer_json(embedded, tokens)
        return GGUFTokenizerVerdict(
            "copy",
            model,
            pre_value,
            policy.canonical if policy else None,
            "embedded tokenizers JSON is copied verbatim and its ordered vocabulary "
            "matches GGUF; pipeline execution is delegated to that artifact",
            len(tokens),
            digest,
            _tokenizer_metadata_sha256(metadata),
        )

    detail = (
        f"pre {pre_value!r} selects compiled llama.cpp behavior not serialized in GGUF"
        if pre_value is not None
        else f"model {model!r} omits the complete tokenizer pipeline"
    )
    reason = f"{detail}; exact ORT tokenizer materialization is unavailable"
    audit_status, blocker_category, evidence_id, reason = _deferred_route_diagnostics(
        pre_value,
        reason=reason,
    )
    return GGUFTokenizerVerdict(
        "deferred",
        model,
        pre_value,
        policy.canonical if policy else None,
        reason,
        len(tokens),
        metadata_sha256=_tokenizer_metadata_sha256(metadata),
        audit_status=audit_status,
        blocker_category=blocker_category,
        evidence_id=evidence_id,
    )


def _tokenizer_config(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    tokens = metadata["tokenizer.ggml.tokens"]
    config: dict[str, Any] = {}
    special_map: dict[str, str] = {}
    names = {
        "bos_token_id": "bos_token",
        "eos_token_id": "eos_token",
        "unknown_token_id": "unk_token",
        "padding_token_id": "pad_token",
        "seperator_token_id": "sep_token",
        "cls_token_id": "cls_token",
        "mask_token_id": "mask_token",
    }
    for suffix, config_name in names.items():
        value = metadata.get(f"tokenizer.ggml.{suffix}")
        if value is not None:
            token = tokens[value]
            config[config_name] = token
            special_map[config_name] = token
    for source_name, config_name in (
        ("add_bos_token", "add_bos_token"),
        ("add_eos_token", "add_eos_token"),
        ("add_sep_token", "add_sep_token"),
        ("add_space_prefix", "add_prefix_space"),
        ("remove_extra_whitespaces", "remove_extra_whitespaces"),
    ):
        value = metadata.get(f"tokenizer.ggml.{source_name}")
        if value is not None:
            config[config_name] = value
    templates = _validate_chat_templates(metadata)
    if templates:
        config["chat_template"] = (
            templates["default"]
            if set(templates) == {"default"}
            else dict(sorted(templates.items()))
        )
    return config, special_map


def write_gguf_tokenizer_json(
    gguf_path: str | Path,
    output_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    expected_metadata_sha256: str | None = None,
    source_identity: str | None = None,
) -> str:
    """Copy an exact embedded tokenizer and reproduce its GGUF-side config."""
    from mobius.integrations.gguf._reader import GGUFModel

    gguf_path = Path(gguf_path)
    if metadata is None:
        metadata = GGUFModel(gguf_path).metadata
    verdict = inspect_gguf_tokenizer(metadata, source=str(gguf_path), require_complete=True)
    if not verdict.materialized:
        raise ValueError(
            f"Cannot emit a complete ORT tokenizer from {gguf_path}: {verdict.reason}. "
            "No tokenizer files were written."
        )
    if (
        expected_metadata_sha256 is not None
        and verdict.metadata_sha256 != expected_metadata_sha256
    ):
        raise ValueError(
            "Tokenizer metadata does not match the expected graph-build identity; "
            "no tokenizer files were written."
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output / "tokenizer.json"
    tokenizer_path.write_bytes(metadata["tokenizer.huggingface.json"].encode("utf-8"))
    config, special_map = _tokenizer_config(metadata)
    (output / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "special_tokens_map.json").write_text(
        json.dumps(special_map, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    templates = _validate_chat_templates(metadata)
    chat_template_path = output / "chat_template.jinja"
    if "default" in templates:
        chat_template_path.write_text(
            templates["default"],
            encoding="utf-8",
        )
    elif chat_template_path.exists():
        chat_template_path.unlink()
    manifest = {
        "format_version": 1,
        "source": source_identity or str(gguf_path.resolve()),
        "route": verdict.route,
        "model": verdict.model,
        "pre": verdict.pre,
        "canonical_pre": verdict.canonical_pre,
        "token_count": verdict.token_count,
        "tokenizer_sha256": verdict.tokenizer_sha256,
        "metadata_sha256": verdict.metadata_sha256,
        "pipeline_semantics": "delegated_to_embedded_tokenizer_json",
        "ort_genai_compatible": "delegated",
    }
    (output / "gguf_tokenizer_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return str(tokenizer_path)


def _read_regular_file(path: Path, *, expected: GGUFTokenizerAsset) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or path.is_symlink():
            raise ValueError(f"Tokenizer asset must be a non-symlink regular file: {path}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > _MAX_TOKENIZER_ASSET_BYTES:
                raise ValueError(f"Tokenizer asset exceeds size limit: {expected.filename}")
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = path.stat()

        def identity(value: os.stat_result) -> tuple[int, int, int, int]:
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)

        if identity(before) != identity(after) or identity(after) != identity(current):
            raise ValueError(f"Tokenizer asset changed while it was being read: {path}")
        if size != expected.size or digest.hexdigest() != expected.sha256:
            raise ValueError(
                f"Tokenizer asset identity mismatch for {expected.filename}: "
                f"expected size={expected.size}, sha256={expected.sha256}; "
                f"got size={size}, sha256={digest.hexdigest()}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_asset_payload(payload: bytes, *, expected: GGUFTokenizerAsset) -> bytes:
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected.size or digest != expected.sha256:
        raise ValueError(
            f"Tokenizer asset identity mismatch for {expected.filename}: "
            f"expected size={expected.size}, sha256={expected.sha256}; "
            f"got size={len(payload)}, sha256={digest}"
        )
    return payload


def _download_tokenizer_assets(
    source: GGUFTokenizerSource,
    *,
    local_files_only: bool,
) -> dict[str, bytes]:
    from huggingface_hub import (
        get_hf_file_metadata,
        get_session,
        hf_hub_download,
        hf_hub_url,
    )
    from huggingface_hub.utils import build_hf_headers

    payloads: dict[str, bytes] = {}
    for asset in source.assets:
        if local_files_only:
            from huggingface_hub.constants import HF_HUB_CACHE

            path = Path(
                hf_hub_download(
                    repo_id=source.repository,
                    revision=source.revision,
                    filename=asset.filename,
                    local_files_only=True,
                )
            )
            try:
                cache_root = Path(HF_HUB_CACHE).resolve(strict=True)
                path.parent.resolve(strict=True).relative_to(cache_root)
                resolved_path = path.resolve(strict=True)
                resolved_path.relative_to(cache_root)
            except ValueError as error:
                raise ValueError(
                    f"Cached tokenizer asset is outside the trusted Hub cache: {path}"
                ) from error
            if path.is_symlink():
                path = resolved_path
        else:
            url = hf_hub_url(source.repository, asset.filename, revision=source.revision)
            metadata = get_hf_file_metadata(url)
            if metadata.commit_hash != source.revision:
                raise ValueError(
                    f"Hub resolved {source.repository}:{asset.filename} to "
                    f"{metadata.commit_hash!r}, not pinned revision {source.revision!r}"
                )
            location = urlparse(metadata.location)
            if (
                location.scheme != "https"
                or not location.netloc
                or location.username is not None
                or location.password is not None
            ):
                raise ValueError(
                    f"Hub returned an unsafe tokenizer asset location: {metadata.location!r}"
                )
            headers = build_hf_headers()
            if urlparse(url).netloc != location.netloc:
                for name in tuple(headers):
                    if name.lower() == "authorization":
                        headers.pop(name)
            session = get_session()

            def read_response(
                response: Any,
                *,
                location: str = metadata.location,
                expected_asset: GGUFTokenizerAsset = asset,
            ) -> bytes:
                if 300 <= response.status_code < 400:
                    raise ValueError(
                        "Tokenizer asset endpoint redirected after authorization policy "
                        f"was selected: {location}"
                    )
                response.raise_for_status()
                chunks: list[bytes] = []
                size = 0
                iterator = (
                    response.iter_bytes()
                    if hasattr(response, "iter_bytes")
                    else response.iter_content(chunk_size=1024 * 1024)
                )
                for chunk in iterator:
                    size += len(chunk)
                    if size > expected_asset.size or size > _MAX_TOKENIZER_ASSET_BYTES:
                        raise ValueError(
                            f"Tokenizer asset exceeded evidenced size: {expected_asset.filename}"
                        )
                    chunks.append(chunk)
                return b"".join(chunks)

            stream = getattr(session, "stream", None)
            if callable(stream):
                with stream(
                    "GET",
                    metadata.location,
                    headers=headers,
                    follow_redirects=False,
                ) as response:
                    payload = read_response(response)
            else:
                with session.get(
                    metadata.location,
                    headers=headers,
                    allow_redirects=False,
                    stream=True,
                ) as response:
                    payload = read_response(response)
            payloads[asset.filename] = _validate_asset_payload(payload, expected=asset)
            continue
        payloads[asset.filename] = _read_regular_file(path, expected=asset)
    return payloads


def _json_object(payload: bytes, *, filename: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{filename} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{filename} must contain a JSON object")
    return value


def _merge_pairs(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        raise TypeError("tokenizer.json model.merges must be an array")
    pairs: list[tuple[str, str]] = []
    for index, merge in enumerate(value):
        if isinstance(merge, str):
            separator = merge.find(" ", 1)
            if separator < 0:
                raise ValueError(f"tokenizer.json model.merges[{index}] is malformed")
            pair = (merge[:separator], merge[separator + 1 :])
        elif (
            isinstance(merge, list)
            and len(merge) == 2
            and all(isinstance(token, str) for token in merge)
        ):
            pair = (merge[0], merge[1])
        else:
            raise ValueError(f"tokenizer.json model.merges[{index}] is malformed")
        if not all(pair):
            raise ValueError(f"tokenizer.json model.merges[{index}] is malformed")
        pairs.append(pair)
    if len(set(pairs)) != len(pairs):
        raise ValueError("tokenizer.json model.merges contains duplicate pairs")
    return pairs


def _special_token_content(value: Any, *, key: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("content"), str):
        return value["content"]
    raise ValueError(f"tokenizer_config.json {key} must be a string or token object")


def _canonicalize_gpt4o_tokenizer_config(
    metadata: Mapping[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    config = dict(config)
    tokens = metadata["tokenizer.ggml.tokens"]
    for suffix, name in (
        ("bos_token_id", "bos_token"),
        ("eos_token_id", "eos_token"),
        ("unknown_token_id", "unk_token"),
        ("padding_token_id", "pad_token"),
    ):
        token_id = metadata.get(f"tokenizer.ggml.{suffix}")
        if token_id is not None:
            config[name] = tokens[token_id]
    for name in ("add_bos_token", "add_eos_token", "add_sep_token"):
        value = metadata.get(f"tokenizer.ggml.{name}")
        if value is not None:
            config[name] = value
    return config


_ADDED_TOKEN_FIELDS = frozenset(
    {"content", "single_word", "lstrip", "rstrip", "normalized", "special"}
)


def _validated_added_token(
    value: Any,
    *,
    token_id: int,
    expected_token: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ADDED_TOKEN_FIELDS:
        raise ValueError(
            f"tokenizer_config.json added_tokens_decoder[{token_id}] must contain "
            f"exactly {sorted(_ADDED_TOKEN_FIELDS)}"
        )
    if value["content"] != expected_token:
        raise ValueError(
            f"tokenizer_config.json added token {token_id} differs from GGUF vocabulary"
        )
    if any(
        type(value[field]) is not bool
        for field in ("single_word", "lstrip", "rstrip", "normalized", "special")
    ):
        raise ValueError(
            f"tokenizer_config.json added token {token_id} flags must be booleans"
        )
    return {"id": token_id, **value}


def _reconstruct_missing_added_tokens(
    tokenizer_json: dict[str, Any],
    config: Mapping[str, Any],
    expected_tokens: list[str],
    token_types: list[int] | None,
    actual_tokens: list[str | None],
) -> bytes | None:
    if actual_tokens != expected_tokens[: len(actual_tokens)] or len(actual_tokens) >= len(
        expected_tokens
    ):
        return None
    added_tokens = tokenizer_json.get("added_tokens")
    if not isinstance(added_tokens, list):
        return None
    model = tokenizer_json.get("model")
    if not isinstance(model, Mapping) or model.get("type") != "BPE":
        return None
    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        return None
    model_ids = list(vocab.values())
    if (
        any(type(token_id) is not int for token_id in model_ids)
        or len(set(model_ids)) != len(model_ids)
        or set(model_ids) != set(range(len(model_ids)))
    ):
        return None
    if any(
        vocab.get(token) != token_id
        for token_id, token in enumerate(expected_tokens[: len(model_ids)])
    ):
        return None

    decoder = config.get("added_tokens_decoder", {})
    if not isinstance(decoder, Mapping):
        raise TypeError("tokenizer_config.json added_tokens_decoder must be an object")
    decoded: dict[int, dict[str, Any]] = {}
    for raw_id, value in decoder.items():
        if not isinstance(raw_id, str) or not raw_id.isdecimal() or str(int(raw_id)) != raw_id:
            raise ValueError(
                "tokenizer_config.json added_tokens_decoder keys must be canonical token IDs"
            )
        token_id = int(raw_id)
        if token_id >= len(expected_tokens):
            raise ValueError(
                f"tokenizer_config.json added token {token_id} is outside the GGUF vocabulary"
            )
        decoded[token_id] = _validated_added_token(
            value,
            token_id=token_id,
            expected_token=expected_tokens[token_id],
        )

    source_added_tokens = {
        token["id"]: token
        for token in added_tokens
        if isinstance(token, Mapping) and isinstance(token.get("id"), int)
    }
    for token_id, token in decoded.items():
        source_token = source_added_tokens.get(token_id)
        if source_token is not None and source_token != token:
            raise ValueError(
                f"tokenizer.json and tokenizer_config.json contradict added token {token_id}"
            )

    for token_id in range(len(model_ids), len(expected_tokens)):
        expected = expected_tokens[token_id]
        if expected in vocab:
            return None
        vocab[expected] = token_id

    for token_id in range(len(actual_tokens), len(expected_tokens)):
        token = decoded.get(token_id)
        if token is None:
            expected = expected_tokens[token_id]
            if expected != f"[PAD{token_id}]":
                return None
            if token_types is None or token_types[token_id] != 5:
                raise ValueError("GGUF vocabulary padding must contain only unused tokens")
            continue
        added_tokens.append(token)
    return json.dumps(tokenizer_json, ensure_ascii=False, separators=(",", ":")).encode()


def _validate_unused_padding_is_non_matchable(
    tokenizer: Any,
    expected_tokens: list[str],
    token_types: list[int] | None,
    *,
    extension_start: int,
) -> None:
    if token_types is None:
        return
    for token_id in range(extension_start, len(expected_tokens)):
        token = expected_tokens[token_id]
        if token_types[token_id] != 5 or token != f"[PAD{token_id}]":
            continue
        for text in (token, f"ordinary{token}text"):
            if token_id in tokenizer.encode(text, add_special_tokens=False).ids:
                raise ValueError(
                    f"GGUF unused padding token {token_id} is matchable by ordinary input"
                )


def _roberta_processor_proves_special_insertion(
    tokenizer_json: Mapping[str, Any],
    metadata: Mapping[str, Any],
    gguf_name: str,
) -> bool:
    processor = tokenizer_json.get("post_processor")
    if not isinstance(processor, Mapping) or processor.get("type") != "RobertaProcessing":
        return False
    processor_name, token_id_name = {
        "add_bos_token": ("cls", "bos_token_id"),
        "add_eos_token": ("sep", "eos_token_id"),
        "add_sep_token": ("sep", "seperator_token_id"),
    }[gguf_name]
    token_id = metadata.get(f"tokenizer.ggml.{token_id_name}")
    value = processor.get(processor_name)
    tokens = metadata.get("tokenizer.ggml.tokens")
    return (
        type(token_id) is int
        and isinstance(tokens, list)
        and 0 <= token_id < len(tokens)
        and isinstance(value, list)
        and len(value) == 2
        and value == [tokens[token_id], token_id]
    )


def _template_processor_proves_bos_insertion(
    tokenizer_json: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> bool:
    processor = tokenizer_json.get("post_processor")
    if not isinstance(processor, Mapping) or processor.get("type") != "Sequence":
        return False
    processors = processor.get("processors")
    if not isinstance(processors, list) or len(processors) != 2:
        return False
    template = processors[1]
    token_id = metadata.get("tokenizer.ggml.bos_token_id")
    tokens = metadata.get("tokenizer.ggml.tokens")
    if (
        processors[0] != _GPT4O_POST_BYTE_LEVEL
        or not isinstance(template, Mapping)
        or template.get("type") != "TemplateProcessing"
        or type(token_id) is not int
        or not isinstance(tokens, list)
        or not 0 <= token_id < len(tokens)
    ):
        return False
    token = tokens[token_id]
    special = {"SpecialToken": {"id": token, "type_id": 0}}
    special_pair = {"SpecialToken": {"id": token, "type_id": 1}}
    return (
        template.get("single") == [special, {"Sequence": {"id": "A", "type_id": 0}}]
        and template.get("pair")
        == [
            special,
            {"Sequence": {"id": "A", "type_id": 0}},
            special_pair,
            {"Sequence": {"id": "B", "type_id": 1}},
        ]
        and template.get("special_tokens", {}).get(token)
        == {"id": token, "ids": [token_id], "tokens": [token]}
    )


def _canonicalize_gpt4o_pipeline(
    tokenizer_json: dict[str, Any],
    metadata: Mapping[str, Any],
    *,
    pre: str,
) -> None:
    pre_tokenizer = tokenizer_json.get("pre_tokenizer")
    source_pre_tokenizer = {
        **_GPT4O_PRE_TOKENIZER,
        "pretokenizers": [
            {
                **_GPT4O_PRE_TOKENIZER["pretokenizers"][0],
                "pattern": {"Regex": _GPT4O_SOURCE_SPLIT_PATTERN},
            },
            _GPT4O_PRE_TOKENIZER["pretokenizers"][1],
        ],
    }
    post_processor = tokenizer_json.get("post_processor")
    valid_post_processor = post_processor == _GPT4O_POST_BYTE_LEVEL or (
        metadata.get("tokenizer.ggml.add_bos_token") is True
        and _template_processor_proves_bos_insertion(tokenizer_json, metadata)
    )
    if (
        tokenizer_json.get("normalizer") is not None
        or pre_tokenizer not in (source_pre_tokenizer, _GPT4O_PRE_TOKENIZER)
        or tokenizer_json.get("decoder") != _GPT4O_DECODER
        or not valid_post_processor
    ):
        raise ValueError(f"Pinned tokenizer pipeline differs from GGUF pre {pre!r}")
    tokenizer_json["pre_tokenizer"] = _GPT4O_PRE_TOKENIZER
    model = tokenizer_json.get("model")
    if not isinstance(model, dict):
        raise TypeError("tokenizer.json must contain a mutable model object")
    model["ignore_merges"] = False


def _gemma4_post_processor(metadata: Mapping[str, Any]) -> dict[str, Any]:
    tokens = metadata.get("tokenizer.ggml.tokens")
    bos_id = metadata.get("tokenizer.ggml.bos_token_id")
    if (
        not isinstance(tokens, list)
        or type(bos_id) is not int
        or not 0 <= bos_id < len(tokens)
    ):
        raise ValueError("GGUF Gemma4 reconstruction requires an exact BOS token")
    bos = tokens[bos_id]
    return {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": bos, "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
        ],
        "pair": [
            {"SpecialToken": {"id": bos, "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": bos, "type_id": 1}},
            {"Sequence": {"id": "B", "type_id": 1}},
        ],
        "special_tokens": {bos: {"id": bos, "ids": [bos_id], "tokens": [bos]}},
    }


def _canonicalize_gemma4_tokenizer(
    tokenizer_json: dict[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    """Align official Gemma4 assets with the pinned llama.cpp GGUF semantics."""
    model = tokenizer_json.get("model")
    added_tokens = tokenizer_json.get("added_tokens")
    token_types = metadata.get("tokenizer.ggml.token_type")
    expected_tokens = metadata.get("tokenizer.ggml.tokens")
    if (
        not isinstance(model, dict)
        or model.get("type") != "BPE"
        or model.get("byte_fallback") is not True
        or not isinstance(added_tokens, list)
        or not isinstance(token_types, list)
        or not isinstance(expected_tokens, list)
        or len(token_types) != len(expected_tokens)
    ):
        raise ValueError("GGUF-native Gemma4 reconstruction requires exact BPE metadata")
    if (
        tokenizer_json.get("normalizer") != _GEMMA4_NORMALIZER
        or tokenizer_json.get("pre_tokenizer") != _GEMMA4_PRE_TOKENIZER
        or tokenizer_json.get("post_processor")
        not in (_GEMMA4_SOURCE_POST_PROCESSOR, _gemma4_post_processor(metadata))
        or tokenizer_json.get("decoder")
        not in (
            _GEMMA4_SOURCE_DECODER,
            {
                **_GEMMA4_SOURCE_DECODER,
                "decoders": [
                    *_GEMMA4_SOURCE_DECODER["decoders"],
                    *(
                        {
                            "type": "Replace",
                            "pattern": {"String": old},
                            "content": new,
                        }
                        for old, new in _GEMMA4_CLEANUP_REPLACEMENTS
                    ),
                ],
            },
        )
    ):
        raise ValueError("Pinned tokenizer pipeline differs from GGUF pre 'gemma4'")

    by_id: dict[int, dict[str, Any]] = {}
    for value in added_tokens:
        if (
            not isinstance(value, dict)
            or type(value.get("id")) is not int
            or not 0 <= value["id"] < len(expected_tokens)
            or value.get("content") != expected_tokens[value["id"]]
        ):
            raise ValueError("Pinned Gemma4 added-token inventory is invalid")
        by_id[value["id"]] = value
    required_added_ids = {
        token_id for token_id, token_type in enumerate(token_types) if token_type in {3, 4}
    }
    required_added_ids.update(
        token_id
        for token_id, token in enumerate(expected_tokens)
        if token in _GEMMA4_FORCED_CONTROL_TOKENS
    )
    if not required_added_ids.issubset(by_id):
        raise ValueError("Pinned Gemma4 tokenizer omits a llama.cpp special token")

    canonical_added = []
    for token_id in sorted(required_added_ids):
        value = dict(by_id[token_id])
        value["special"] = (
            token_types[token_id] == 3
            or expected_tokens[token_id] in _GEMMA4_FORCED_CONTROL_TOKENS
        )
        canonical_added.append(value)
    tokenizer_json["added_tokens"] = canonical_added
    tokenizer_json["post_processor"] = _gemma4_post_processor(metadata)
    tokenizer_json["decoder"] = {
        **_GEMMA4_SOURCE_DECODER,
        "decoders": [
            *_GEMMA4_SOURCE_DECODER["decoders"],
            *(
                {"type": "Replace", "pattern": {"String": old}, "content": new}
                for old, new in _GEMMA4_CLEANUP_REPLACEMENTS
            ),
        ],
    }


def _canonicalize_gemma4_tokenizer_config(
    metadata: Mapping[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    tokens = metadata.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list):
        raise TypeError("GGUF Gemma4 reconstruction requires an exact vocabulary")
    for gguf_name, config_name in (
        ("bos_token_id", "bos_token"),
        ("eos_token_id", "eos_token"),
        ("unknown_token_id", "unk_token"),
        ("padding_token_id", "pad_token"),
        ("mask_token_id", "mask_token"),
    ):
        token_id = metadata.get(f"tokenizer.ggml.{gguf_name}")
        if type(token_id) is int and 0 <= token_id < len(tokens):
            config[config_name] = tokens[token_id]
    config["add_bos_token"] = metadata.get("tokenizer.ggml.add_bos_token") is True
    templates = _validate_chat_templates(metadata)
    if set(templates) != {"default"}:
        raise ValueError("GGUF Gemma4 reconstruction requires one default chat template")
    config["chat_template"] = templates["default"]
    return config


def _validate_pinned_tokenizer(
    metadata: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    *,
    reconstruct_gpt4o_from_gguf: bool = False,
    reconstruct_gemma4_from_gguf: bool = False,
) -> tuple[str, bytes]:
    raw_tokenizer = payloads["tokenizer.json"]
    tokenizer_json = _json_object(raw_tokenizer, filename="tokenizer.json")
    config = _json_object(
        payloads.get("tokenizer_config.json", b"{}"), filename="tokenizer_config.json"
    )
    special_map = _json_object(
        payloads.get("special_tokens_map.json", b"{}"),
        filename="special_tokens_map.json",
    )
    expected_tokens = metadata["tokenizer.ggml.tokens"]
    model = tokenizer_json.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("tokenizer.json must contain a model object")
    pre = metadata.get("tokenizer.ggml.pre")
    if pre is None and metadata.get("tokenizer.ggml.model") == "gemma4":
        pre = "gemma4"
    policy = tokenizer_pre_policies().get(pre)
    if reconstruct_gpt4o_from_gguf:
        if policy is None or policy.pre_type != "GPT4O" or not isinstance(model, dict):
            raise ValueError("GGUF-native reconstruction is supported only for GPT4O BPE")
        model["merges"] = metadata.get("tokenizer.ggml.merges")
        config = _canonicalize_gpt4o_tokenizer_config(metadata, config)
    if reconstruct_gemma4_from_gguf:
        if (
            policy is None
            or policy.pre_type != "GEMMA4"
            or not isinstance(tokenizer_json, dict)
        ):
            raise ValueError("GGUF-native Gemma4 reconstruction requires GEMMA4 dispatch")
        _canonicalize_gemma4_tokenizer(tokenizer_json, metadata)
        config = _canonicalize_gemma4_tokenizer_config(metadata, config)
        raw_tokenizer = json.dumps(
            tokenizer_json, ensure_ascii=False, separators=(",", ":")
        ).encode()
    if policy is not None and policy.pre_type == "GPT4O":
        if not isinstance(pre, str):
            raise TypeError("GPT4O tokenizer reconstruction requires a string pre identifier")
        _canonicalize_gpt4o_pipeline(tokenizer_json, metadata, pre=pre)
        raw_tokenizer = json.dumps(
            tokenizer_json, ensure_ascii=False, separators=(",", ":")
        ).encode()

    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_str(raw_tokenizer.decode("utf-8"))
    except Exception as error:
        raise ValueError("tokenizer.json is not a loadable tokenizers tokenizer") from error

    actual_tokens = [
        tokenizer.id_to_token(index)
        for index in range(tokenizer.get_vocab_size(with_added_tokens=True))
    ]
    token_types = metadata.get("tokenizer.ggml.token_type")
    source_added_tokens = {
        token["id"]: token
        for token in tokenizer_json.get("added_tokens", ())
        if isinstance(token, Mapping) and isinstance(token.get("id"), int)
    }
    reconstructed = _reconstruct_missing_added_tokens(
        tokenizer_json,
        config,
        expected_tokens,
        token_types,
        actual_tokens,
    )
    if reconstructed is not None:
        extension_start = len(actual_tokens)
        raw_tokenizer = reconstructed
        try:
            tokenizer = Tokenizer.from_str(raw_tokenizer.decode("utf-8"))
        except Exception as error:
            raise ValueError(
                "tokenizer.json cannot represent exact GGUF added-token reconstruction"
            ) from error
        actual_tokens = [
            tokenizer.id_to_token(index)
            for index in range(tokenizer.get_vocab_size(with_added_tokens=True))
        ]
        _validate_unused_padding_is_non_matchable(
            tokenizer,
            expected_tokens,
            token_types,
            extension_start=extension_start,
        )
    if actual_tokens != expected_tokens:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(zip(actual_tokens, expected_tokens))
                if actual != expected
            ),
            min(len(actual_tokens), len(expected_tokens)),
        )
        raise ValueError(
            f"Pinned tokenizer vocabulary differs from GGUF at token id {mismatch}"
        )

    if _merge_pairs(model.get("merges")) != _merge_pairs(
        metadata.get("tokenizer.ggml.merges")
    ):
        raise ValueError(
            "Pinned tokenizer merge order differs from GGUF tokenizer.ggml.merges"
        )
    if (
        pre == "smollm"
        and {
            name: tokenizer_json.get(name)
            for name in ("normalizer", "pre_tokenizer", "post_processor", "decoder")
        }
        != _SMOLLM_PIPELINE
    ):
        raise ValueError(f"Pinned tokenizer pipeline differs from GGUF pre {pre!r}")
    if token_types is not None:
        gemma4_forced_control_tokens = (
            _GEMMA4_FORCED_CONTROL_TOKENS
            if policy is not None and policy.pre_type == "GEMMA4"
            else frozenset()
        )
        supported_types = {1, 2, 3, 4, 5}
        if policy is not None and policy.pre_type == "GEMMA4":
            supported_types.add(6)
        unsupported_types = sorted(set(token_types) - supported_types)
        if unsupported_types:
            raise ValueError(
                f"Pinned tokenizer identity cannot prove GGUF token types {unsupported_types}"
            )
        expected_non_special_added_ids = {
            index
            for index, token_type in enumerate(token_types)
            if token_type == 4 and expected_tokens[index] not in gemma4_forced_control_tokens
        }
        if any(
            index not in source_added_tokens
            or source_added_tokens[index].get("special") is not False
            for index in expected_non_special_added_ids
        ):
            raise ValueError(
                "Pinned tokenizer user-defined/unused-token inventory differs from GGUF"
            )
        source_special_ids = {
            token["id"]
            for token in source_added_tokens.values()
            if token.get("special") is True
        }
        if any(
            token_types[index] not in {2, 3}
            and expected_tokens[index] not in gemma4_forced_control_tokens
            for index in source_special_ids
        ):
            raise ValueError("Pinned tokenizer special-token inventory differs from GGUF")

    special_names = {
        "bos_token_id": "bos_token",
        "eos_token_id": "eos_token",
        "unknown_token_id": "unk_token",
        "padding_token_id": "pad_token",
        "seperator_token_id": "sep_token",
        "cls_token_id": "cls_token",
        "mask_token_id": "mask_token",
    }
    for gguf_suffix, config_name in special_names.items():
        expected_id = metadata.get(f"tokenizer.ggml.{gguf_suffix}")
        if expected_id is None:
            continue
        raw_value = config.get(config_name, special_map.get(config_name))
        if raw_value is None:
            source_token = source_added_tokens.get(expected_id)
            if (
                source_token is not None
                and source_token.get("content") == expected_tokens[expected_id]
                and source_token.get("special") is True
            ):
                continue
            if (
                config_name == "bos_token"
                and metadata.get("tokenizer.ggml.add_bos_token") is False
            ):
                continue
            raise ValueError(f"Pinned tokenizer omits GGUF special token {config_name}")
        token = _special_token_content(raw_value, key=config_name)
        if tokenizer.token_to_id(token) != expected_id:
            raise ValueError(f"Pinned tokenizer {config_name} id differs from GGUF")

    for gguf_name, config_name in (
        ("add_bos_token", "add_bos_token"),
        ("add_eos_token", "add_eos_token"),
        ("add_sep_token", "add_sep_token"),
    ):
        expected = metadata.get(f"tokenizer.ggml.{gguf_name}")
        actual = config.get(config_name)
        if expected is None:
            continue
        if actual is None:
            if expected is True and _roberta_processor_proves_special_insertion(
                tokenizer_json, metadata, gguf_name
            ):
                continue
            if (
                gguf_name == "add_bos_token"
                and expected is True
                and _template_processor_proves_bos_insertion(tokenizer_json, metadata)
            ):
                continue
            if expected is not False:
                raise ValueError(f"Pinned tokenizer cannot prove GGUF {config_name}")
        elif actual is not expected:
            raise ValueError(f"Pinned tokenizer {config_name} differs from GGUF")

    if (
        policy is not None
        and policy.pre_type == "GPT2_ADD_SEP"
        and not _roberta_processor_proves_special_insertion(
            tokenizer_json, metadata, "add_sep_token"
        )
    ):
        raise ValueError(
            f"Pinned tokenizer post-processor cannot prove GGUF pre {pre!r} SEP insertion"
        )

    expected_prefix = metadata.get("tokenizer.ggml.add_space_prefix")
    if expected_prefix is not None and policy is not None and policy.pre_type != "GEMMA4":
        pre_tokenizer = tokenizer_json.get("pre_tokenizer")
        queue = [pre_tokenizer]
        byte_level: list[Mapping[str, Any]] = []
        while queue:
            current = queue.pop()
            if not isinstance(current, Mapping):
                continue
            if current.get("type") == "ByteLevel":
                byte_level.append(current)
            children = current.get("pretokenizers")
            if isinstance(children, list):
                queue.extend(children)
        if (
            len(byte_level) != 1
            or byte_level[0].get("add_prefix_space") is not expected_prefix
        ):
            raise ValueError("Pinned tokenizer add_prefix_space differs from GGUF")

    suppress = metadata.get("tokenizer.ggml.suppress_tokens")
    if suppress is not None and config.get("suppress_tokens") != suppress:
        raise ValueError("Pinned tokenizer suppress_tokens differs from GGUF")
    validated_metadata = {
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.token_type_count",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.seperator_token_id",
        "tokenizer.ggml.cls_token_id",
        "tokenizer.ggml.mask_token_id",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
        "tokenizer.ggml.add_sep_token",
        "tokenizer.ggml.add_space_prefix",
        "tokenizer.ggml.suppress_tokens",
        "tokenizer.chat_template",
        "tokenizer.chat_templates",
    }
    scores = metadata.get("tokenizer.ggml.scores")
    if (
        policy is not None
        and policy.pre_type == "GEMMA4"
        and isinstance(scores, list)
        and len(scores) == len(expected_tokens)
        and set(scores) == {-1000.0}
    ):
        # llama.cpp ranks Gemma4 BPE merges from tokenizer.ggml.merges; the
        # converter's uniform sentinel scores do not participate in BPE.
        validated_metadata.add("tokenizer.ggml.scores")
    validated_metadata.update(
        key for key in metadata if key.startswith("tokenizer.chat_template.")
    )
    unsupported = sorted(
        key
        for key in metadata
        if key.startswith("tokenizer.")
        and key not in validated_metadata
        and key != "tokenizer.huggingface.json"
    )
    if unsupported:
        raise ValueError(
            f"Pinned tokenizer identity cannot prove GGUF tokenizer fields {unsupported}"
        )

    templates = _validate_chat_templates(metadata)
    source_template = config.get("chat_template")
    if "chat_template.jinja" in payloads and not reconstruct_gemma4_from_gguf:
        file_template = payloads["chat_template.jinja"].decode("utf-8")
        if source_template is not None and source_template != file_template:
            raise ValueError("Pinned tokenizer chat template assets contradict each other")
        source_template = file_template
    if templates:
        expected_template: Any = (
            templates["default"]
            if set(templates) == {"default"}
            else dict(sorted(templates.items()))
        )
        if source_template != expected_template:
            raise ValueError("Pinned tokenizer chat template differs from GGUF")

    return hashlib.sha256(raw_tokenizer).hexdigest(), raw_tokenizer


def materialize_evidenced_gguf_tokenizer(
    gguf_path: str | Path,
    output_dir: str | Path,
    *,
    local_files_only: bool = False,
) -> str:
    """Materialize a tokenizer only when the complete GGUF artifact has exact evidence."""
    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.integrations.gguf._runtime_package import _publish_directory_no_replace
    from mobius.integrations.gguf._tokenizer_evidence import matching_tokenizer_evidence

    gguf_path = Path(gguf_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(
            f"Evidenced tokenizer destination already exists: {output}. "
            "Refusing a non-atomic directory replacement."
        )
    model = GGUFModel(gguf_path)
    if not model.source_matches_path():
        raise ValueError(
            "The GGUF source changed while the canonical reader was opening it; "
            "refusing tokenizer publication."
        )
    verdict = inspect_gguf_tokenizer(
        model.metadata,
        source=str(gguf_path),
        require_complete=True,
    )
    evidence = matching_tokenizer_evidence(
        gguf_path,
        model,
        metadata_sha256=verdict.metadata_sha256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent))
    try:
        tokenizer_path = Path(
            materialize_gguf_tokenizer(
                gguf_path,
                stage,
                source=evidence.source,
                metadata=model.metadata,
                source_identity=(
                    f"hf://{evidence.repository}@{evidence.revision}/{evidence.filename}"
                    f"#sha256={evidence.lfs_sha256}"
                ),
                local_files_only=local_files_only,
            )
        )
        if not model.source_matches_path():
            raise ValueError(
                "The GGUF source changed while tokenizer assets were being validated; "
                "refusing tokenizer publication."
            )
        relative_tokenizer_path = tokenizer_path.relative_to(stage)
        _publish_directory_no_replace(stage, output)
        return str(output / relative_tokenizer_path)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def materialize_gguf_tokenizer(
    gguf_path: str | Path,
    output_dir: str | Path,
    *,
    source: GGUFTokenizerSource,
    metadata: Mapping[str, Any] | None = None,
    source_identity: str | None = None,
    local_files_only: bool = False,
) -> str:
    """Materialize exact pinned tokenizer assets after fail-closed semantic validation."""
    from mobius.integrations.gguf._reader import GGUFModel

    gguf_path = Path(gguf_path)
    if metadata is None:
        metadata = GGUFModel(gguf_path).metadata
    verdict = inspect_gguf_tokenizer(metadata, source=str(gguf_path), require_complete=True)
    if verdict.metadata_sha256 != source.metadata_sha256:
        raise ValueError("Pinned tokenizer evidence does not match GGUF tokenizer metadata")
    payloads = _download_tokenizer_assets(source, local_files_only=local_files_only)
    tokenizer_sha256, tokenizer_payload = _validate_pinned_tokenizer(
        metadata,
        payloads,
        reconstruct_gpt4o_from_gguf=source.reconstruct_gpt4o_from_gguf,
        reconstruct_gemma4_from_gguf=source.reconstruct_gemma4_from_gguf,
    )
    if source.reconstruct_gpt4o_from_gguf:
        config = _json_object(
            payloads.get("tokenizer_config.json", b"{}"),
            filename="tokenizer_config.json",
        )
        payloads["tokenizer_config.json"] = json.dumps(
            _canonicalize_gpt4o_tokenizer_config(metadata, config),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    if source.reconstruct_gemma4_from_gguf:
        config = _json_object(
            payloads.get("tokenizer_config.json", b"{}"),
            filename="tokenizer_config.json",
        )
        payloads["tokenizer_config.json"] = json.dumps(
            _canonicalize_gemma4_tokenizer_config(metadata, config),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        payloads["chat_template.jinja"] = metadata["tokenizer.chat_template"].encode()
    if (
        source.materialized_tokenizer_sha256 is not None
        and tokenizer_sha256 != source.materialized_tokenizer_sha256
    ):
        raise ValueError(
            "Materialized tokenizer digest differs from exact tokenizer evidence: "
            f"expected {source.materialized_tokenizer_sha256}, got {tokenizer_sha256}"
        )
    if source.representative_encodings or source.representative_special_encodings:
        try:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_str(tokenizer_payload.decode("utf-8"))
        except Exception as error:
            raise ValueError(
                "Materialized tokenizer is not loadable for evidence checks"
            ) from error
        for text, expected_ids in source.representative_encodings:
            actual_ids = tuple(tokenizer.encode(text, add_special_tokens=False).ids)
            if actual_ids != expected_ids:
                raise ValueError(
                    "Materialized tokenizer representative encoding differs for "
                    f"{text!r}: expected {expected_ids}, got {actual_ids}"
                )
        for text, expected_ids in source.representative_special_encodings:
            actual_ids = tuple(tokenizer.encode(text, add_special_tokens=True).ids)
            if actual_ids != expected_ids:
                raise ValueError(
                    "Materialized tokenizer representative special encoding differs for "
                    f"{text!r}: expected {expected_ids}, got {actual_ids}"
                )
    payloads["tokenizer.json"] = tokenizer_payload

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "source": source_identity or str(gguf_path.resolve()),
        "route": "pinned-source",
        "model": verdict.model,
        "pre": verdict.pre,
        "canonical_pre": verdict.canonical_pre,
        "token_count": verdict.token_count,
        "tokenizer_sha256": tokenizer_sha256,
        "metadata_sha256": verdict.metadata_sha256,
        "tokenizer_repository": source.repository,
        "tokenizer_revision": source.revision,
        "assets": [dataclasses.asdict(asset) for asset in source.assets],
        "pipeline_semantics": "exact_pinned_tokenizer_assets",
        "ort_genai_compatible": "validated",
    }
    writes = dict(payloads)
    writes["gguf_tokenizer_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    temporary: list[tuple[Path, Path]] = []
    try:
        for filename, payload in writes.items():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{filename}.", suffix=".tmp", dir=output
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.append((temporary_path, output / filename))
        for temporary_path, destination in temporary:
            os.replace(temporary_path, destination)
        return str(output / "tokenizer.json")
    finally:
        for temporary_path, _ in temporary:
            temporary_path.unlink(missing_ok=True)
