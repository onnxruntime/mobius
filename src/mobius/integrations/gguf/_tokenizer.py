# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate GGUF tokenizer metadata and materialize only exact tokenizer assets."""

from __future__ import annotations

__all__ = [
    "GGUFTokenizerVerdict",
    "inspect_gguf_tokenizer",
    "write_gguf_tokenizer_json",
]

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from mobius.integrations.gguf._tokenizer_registry import tokenizer_pre_policies

TokenizerRoute = Literal["copy", "deferred"]

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

    @property
    def materialized(self) -> bool:
        return self.route == "copy"


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


def inspect_gguf_tokenizer(
    metadata: Mapping[str, Any],
    *,
    source: str = "<GGUF>",
    require_complete: bool = False,
) -> GGUFTokenizerVerdict:
    """Validate embedded tokenizer metadata and return an exact route verdict."""
    tokenizer_keys = [key for key in metadata if key.startswith("tokenizer.")]
    if not tokenizer_keys:
        return GGUFTokenizerVerdict(
            "deferred",
            None,
            None,
            None,
            f"{source} contains no tokenizer metadata",
            0,
        )

    model = metadata.get("tokenizer.ggml.model")
    if not isinstance(model, str) or not model:
        if not require_complete:
            return GGUFTokenizerVerdict(
                "deferred",
                None,
                None,
                None,
                f"{source} contains partial tokenizer metadata without tokenizer.ggml.model",
                len(metadata.get("tokenizer.ggml.tokens", ())),
            )
        raise ValueError(f"{source} tokenizer.ggml.model must be a non-empty string")
    if model not in _KNOWN_MODELS:
        raise ValueError(f"{source} declares unknown tokenizer.ggml.model {model!r}")

    tokens_raw = _require_list(metadata, "tokenizer.ggml.tokens")
    if not tokens_raw:
        if not require_complete:
            return GGUFTokenizerVerdict(
                "deferred",
                model,
                None,
                None,
                f"{source} contains no complete tokenizer token table",
                0,
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
        raise ValueError(
            f"{source} declares tokenizer.ggml.pre for non-BPE model {model!r}; "
            "the pinned loader does not consume that combination"
        )
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
    return GGUFTokenizerVerdict(
        "deferred",
        model,
        pre_value,
        policy.canonical if policy else None,
        f"{detail}; exact ORT tokenizer materialization is unavailable",
        len(tokens),
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
    tokenizer_path.write_text(metadata["tokenizer.huggingface.json"], encoding="utf-8")
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
        "source": str(gguf_path.resolve()),
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
