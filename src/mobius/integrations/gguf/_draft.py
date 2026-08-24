# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fail-closed target pairing for speculative draft GGUF models."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from mobius._configs import DFlashConfig, Eagle3Config

_DRAFT_ARCHITECTURES = frozenset({"dflash", "eagle3"})
_MAX_CONFIG_JSON_BYTES = 4 * 1024 * 1024
_MAX_TOKENIZER_JSON_BYTES = 64 * 1024 * 1024
_MAX_TOKENIZER_VOCAB_SIZE = 2_000_000


def is_draft_architecture(architecture: str) -> bool:
    return architecture in _DRAFT_ARCHITECTURES


def _ordered_tokenizer_vocab(
    tokenizer_json: Mapping[str, Any],
    *,
    expected_vocab_size: int,
) -> list[str]:
    if (
        type(expected_vocab_size) is not int
        or expected_vocab_size <= 0
        or expected_vocab_size > _MAX_TOKENIZER_VOCAB_SIZE
    ):
        raise ValueError(
            "target vocab_size must be a positive integer no greater than "
            f"{_MAX_TOKENIZER_VOCAB_SIZE}, got {expected_vocab_size!r}"
        )
    model = tokenizer_json.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("target tokenizer.json has no model object")
    vocab = model.get("vocab")
    if isinstance(vocab, Mapping):
        if any(not isinstance(token, str) for token in vocab):
            raise ValueError("target tokenizer.json vocabulary keys must be strings")
        if any(type(index) is not int or index < 0 for index in vocab.values()):
            raise ValueError(
                "target tokenizer.json vocabulary ids must be non-negative integers"
            )
        if any(index >= expected_vocab_size for index in vocab.values()):
            raise ValueError(
                "draft/target tokenizer size mismatch: tokenizer vocabulary id "
                "exceeds target vocab_size"
            )
        tokens: list[str | None] = [None] * expected_vocab_size
        for token, index in vocab.items():
            if tokens[index] is not None:
                raise ValueError(
                    "target tokenizer.json has duplicate or invalid vocabulary ids"
                )
            tokens[index] = token
    elif isinstance(vocab, Sequence) and not isinstance(vocab, (str, bytes)):
        if len(vocab) > expected_vocab_size:
            raise ValueError(
                "draft/target tokenizer size mismatch: tokenizer vocabulary "
                "exceeds target vocab_size"
            )
        tokens = [None] * expected_vocab_size
        for index, entry in enumerate(vocab):
            token = (
                entry[0]
                if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes))
                else entry
            )
            if not isinstance(token, str):
                raise TypeError(
                    "target tokenizer.json vocabulary entries must contain strings"
                )
            tokens[index] = token
    else:
        raise TypeError("target tokenizer.json has no supported ordered vocabulary")

    added_tokens = tokenizer_json.get("added_tokens", ())
    if not isinstance(added_tokens, Sequence) or isinstance(added_tokens, (str, bytes)):
        raise TypeError("target tokenizer.json added_tokens must be an array")
    for added in added_tokens:
        if not isinstance(added, Mapping):
            raise TypeError("target tokenizer.json added_tokens entries must be objects")
        if "id" not in added or "content" not in added:
            raise ValueError(
                "target tokenizer.json added_tokens entries require id and content"
            )
        index = added["id"]
        token = added["content"]
        if type(index) is not int or index < 0 or not isinstance(token, str):
            raise ValueError(
                "target tokenizer.json added token ids/content must be non-negative integers/strings"
            )
        if index >= expected_vocab_size:
            raise ValueError(
                "draft/target tokenizer size mismatch: added token id exceeds "
                "target vocab_size"
            )
        if tokens[index] not in (None, token):
            raise ValueError(f"target tokenizer id {index} has conflicting token values")
        tokens[index] = token
    if any(token is None for token in tokens):
        missing = [index for index, token in enumerate(tokens) if token is None]
        raise ValueError(f"target tokenizer vocabulary has unmapped ids: {missing[:8]}")
    return [str(token) for token in tokens]


def _canonical_json_bytes(value: Any, *, label: str, limit: int) -> bytes:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not canonical JSON: {error}") from error
    if len(payload) > limit:
        raise ValueError(f"{label} exceeds the {limit}-byte canonical JSON limit")
    return payload


def _is_link_or_reparse_point(file_stat: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(file_stat.st_mode) or bool(
        getattr(file_stat, "st_file_attributes", 0) & reparse_flag
    )


def _supports_secure_dir_fd() -> bool:
    return os.open in os.supports_dir_fd and os.stat in os.supports_dir_fd


def _validate_path_open(
    root: Path,
    resource_path: Path,
    root_before: os.stat_result,
    resource_before: os.stat_result,
    descriptor: int,
) -> None:
    opened = os.fstat(descriptor)
    root_after = root.lstat()
    resource_after = resource_path.lstat()
    if (
        _is_link_or_reparse_point(root_after)
        or _is_link_or_reparse_point(resource_after)
        or not os.path.samestat(root_before, root_after)
        or not os.path.samestat(resource_before, resource_after)
        or not os.path.samestat(resource_before, opened)
    ):
        raise OSError("target root or resource changed while it was being opened")


def _open_resource_at_impl(root: Path, root_descriptor: int | None, resource: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    if root_descriptor is not None:
        return os.open(resource, flags, dir_fd=root_descriptor)

    # Windows cannot open a directory descriptor or use dir_fd. Reject
    # reparse points and verify identities before and after opening so a
    # path swap cannot silently redirect either the root or selected file.
    root_before = root.lstat()
    resource_path = root / resource
    resource_before = resource_path.lstat()
    if _is_link_or_reparse_point(root_before) or _is_link_or_reparse_point(resource_before):
        raise OSError("target root and resource must not be links or reparse points")
    descriptor = os.open(resource_path, flags)
    with ExitStack() as cleanup:
        cleanup.callback(os.close, descriptor)
        _validate_path_open(root, resource_path, root_before, resource_before, descriptor)
        cleanup.pop_all()
        return descriptor


def _open_resource_at(root: Path, root_descriptor: int | None, resource: str) -> int:
    try:
        return _open_resource_at_impl(root, root_descriptor, resource)
    except OSError as error:
        raise ValueError(
            f"target resource could not be opened safely inside the target root: {error}"
        ) from error


def _read_bounded_json_at(
    root: Path,
    root_descriptor: int | None,
    resource: str,
    *,
    label: str,
    limit: int,
) -> dict[str, Any]:
    try:
        descriptor = _open_resource_at(root, root_descriptor, resource)
    except ValueError as error:
        raise ValueError(f"{label} could not be opened safely: {error}") from error
    with os.fdopen(descriptor, "rb") as stream:
        file_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"{label} is not a regular file")
        if file_stat.st_size > limit:
            raise ValueError(f"{label} exceeds the {limit}-byte file-size limit")
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"{label} exceeds the {limit}-byte read limit")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} root must be a JSON object with string keys")
    return value


def _resource_exists_at(root: Path, root_descriptor: int | None, resource: str) -> bool:
    try:
        if root_descriptor is not None:
            os.stat(resource, dir_fd=root_descriptor, follow_symlinks=False)
        else:
            (root / resource).lstat()
    except FileNotFoundError:
        return False
    return True


def _load_target_resource_files(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        path_stat = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"target config does not exist: {path}") from error
    if _is_link_or_reparse_point(path_stat):
        raise ValueError("target config path must not be a symlink or reparse point")
    if stat.S_ISDIR(path_stat.st_mode):
        root = path
    else:
        if path.name != "config.json":
            raise ValueError("target config file must be named config.json")
        root = path.parent

    root_descriptor: int | None = None
    if _supports_secure_dir_fd():
        root_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            root_descriptor = os.open(root, root_flags)
        except OSError as error:
            raise ValueError(f"target root could not be opened safely: {error}") from error
    else:
        try:
            root_stat = root.lstat()
        except OSError as error:
            raise ValueError(f"target root could not be opened safely: {error}") from error
        if not stat.S_ISDIR(root_stat.st_mode) or _is_link_or_reparse_point(root_stat):
            raise ValueError("target root must be a directory, not a link or reparse point")
    try:
        if not _resource_exists_at(root, root_descriptor, "config.json"):
            raise ValueError("target config.json does not exist inside the target root")
        if not _resource_exists_at(root, root_descriptor, "tokenizer.json"):
            alternatives = [
                name
                for name in (
                    "tokenizer.model",
                    "vocab.json",
                    "merges.txt",
                    "tokenizer_config.json",
                )
                if _resource_exists_at(root, root_descriptor, name)
            ]
            suffix = (
                f"; found {alternatives}, but split tokenizer files cannot preserve the full "
                "normalizer/pre-tokenizer/added-token contract"
                if alternatives
                else ""
            )
            raise ValueError(
                "target tokenizer.json is required for exact draft pairing inside the "
                f"target root{suffix}"
            )
        data = _read_bounded_json_at(
            root,
            root_descriptor,
            "config.json",
            label="target config.json",
            limit=_MAX_CONFIG_JSON_BYTES,
        )
        tokenizer = _read_bounded_json_at(
            root,
            root_descriptor,
            "tokenizer.json",
            label="target tokenizer.json",
            limit=_MAX_TOKENIZER_JSON_BYTES,
        )
    finally:
        if root_descriptor is not None:
            os.close(root_descriptor)
    return data, tokenizer


def _load_target(
    target_config: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    if isinstance(target_config, Mapping):
        data = dict(target_config)
        tokenizer = data.pop("tokenizer_json", None)
        if not isinstance(tokenizer, Mapping):
            raise TypeError(
                "mapping target_config must include the complete tokenizer_json object; "
                "tokenizer_tokens alone cannot verify merges, normalization, or pre-tokenization"
            )
        if any(not isinstance(key, str) for key in data) or any(
            not isinstance(key, str) for key in tokenizer
        ):
            raise ValueError("target config and tokenizer mappings must have string keys")
        tokenizer = dict(tokenizer)
    else:
        data, tokenizer = _load_target_resource_files(Path(target_config))

    config_payload = _canonical_json_bytes(
        data,
        label="target config",
        limit=_MAX_CONFIG_JSON_BYTES,
    )
    tokenizer_payload = _canonical_json_bytes(
        tokenizer,
        label="target tokenizer",
        limit=_MAX_TOKENIZER_JSON_BYTES,
    )
    target = _flatten_text_config(data)
    ordered_vocab = _ordered_tokenizer_vocab(
        tokenizer,
        expected_vocab_size=target.get("vocab_size"),
    )
    from tokenizers import Tokenizer

    # The Rust parser is the authoritative schema validator for normalizers,
    # pre-tokenizers, models, post-processors, decoders, and added tokens.
    try:
        Tokenizer.from_str(tokenizer_payload.decode("utf-8"))
    except Exception as error:
        raise ValueError(f"target tokenizer.json has an invalid schema: {error}") from error
    return (
        data,
        ordered_vocab,
        {
            "config_resource": "config.json",
            "tokenizer_resource": "tokenizer.json",
            "config_sha256": hashlib.sha256(config_payload).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(tokenizer_payload).hexdigest(),
        },
    )


def _flatten_text_config(data: Mapping[str, Any]) -> dict[str, Any]:
    flattened = dict(data)
    text_config = data.get("text_config")
    if text_config is not None and not isinstance(text_config, Mapping):
        raise ValueError("target config text_config must be an object")
    if isinstance(text_config, Mapping):
        flattened.update(text_config)
    return flattened


def _tokenizer_digest(tokens: Sequence[str]) -> str:
    payload = json.dumps(list(tokens), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_d2t(gguf_model: Any) -> list[int] | None:
    if "d2t" not in gguf_model.tensor_names:
        return None
    for name, raw, qtype, shape in gguf_model.tensor_items_raw():
        if name != "d2t":
            continue
        qtype_name = getattr(qtype, "name", str(qtype))
        if qtype_name != "I64":
            raise ValueError(f"{gguf_model.architecture} d2t must use I64, got {qtype_name}")
        return [int(value) for value in raw.reshape(shape).reshape(-1)]
    raise AssertionError("d2t was indexed but not present in the GGUF tensor table")


def _validate_special_ids(metadata: Mapping[str, Any], target: Mapping[str, Any]) -> None:
    for gguf_suffix, config_key in (
        ("bos_token_id", "bos_token_id"),
        ("eos_token_id", "eos_token_id"),
        ("padding_token_id", "pad_token_id"),
    ):
        gguf_value = metadata.get(f"tokenizer.ggml.{gguf_suffix}")
        target_value = target.get(config_key)
        if gguf_value is None or target_value is None:
            continue
        if isinstance(target_value, Sequence) and not isinstance(target_value, (str, bytes)):
            if any(type(value) is not int or value < 0 for value in target_value):
                raise ValueError(
                    f"target config {config_key} must contain non-negative integers"
                )
            matches = int(gguf_value) in set(target_value)
        else:
            if type(target_value) is not int or target_value < 0:
                raise ValueError(f"target config {config_key} must be a non-negative integer")
            matches = int(gguf_value) == target_value
        if not matches:
            raise ValueError(
                f"draft/target {config_key} mismatch: GGUF={gguf_value!r}, "
                f"target={target_value!r}"
            )


def validate_draft_pairing(
    gguf_model: Any,
    config: DFlashConfig | Eagle3Config,
    target_config: str | Path | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate a speculative draft against its exact target before graph build."""
    architecture = gguf_model.architecture
    if target_config is None:
        raise ValueError(
            f"{architecture} GGUF is a speculative draft, not a standalone language model. "
            "Pass target_config (CLI: --target-config) for the exact paired target."
        )
    target_raw, target_tokens, target_identity = _load_target(target_config)
    target = _flatten_text_config(target_raw)
    required = ("hidden_size", "num_hidden_layers", "vocab_size")
    missing = [field for field in required if target.get(field) is None]
    if missing:
        raise ValueError(f"target config is missing required field(s): {missing}")
    invalid_fields = [
        field for field in required if type(target[field]) is not int or target[field] <= 0
    ]
    if invalid_fields:
        raise ValueError(f"target config field(s) must be positive integers: {invalid_fields}")
    if target.get("model_type") is not None and not isinstance(target["model_type"], str):
        raise ValueError("target config model_type must be a string")

    target_hidden = int(target["hidden_size"])
    target_layers = int(target["num_hidden_layers"])
    target_vocab = int(target["vocab_size"])
    gguf_tokens = [
        str(token) for token in gguf_model.metadata.get("tokenizer.ggml.tokens", ())
    ]
    if len(target_tokens) != target_vocab or len(gguf_tokens) != target_vocab:
        raise ValueError(
            "draft/target tokenizer size mismatch: "
            f"target config={target_vocab}, target tokenizer={len(target_tokens)}, "
            f"GGUF tokenizer={len(gguf_tokens)}"
        )
    if target_tokens != gguf_tokens:
        first = next(
            index
            for index, (target_token, gguf_token) in enumerate(zip(target_tokens, gguf_tokens))
            if target_token != gguf_token
        )
        raise ValueError(
            "draft GGUF tokenizer is not identical to the target tokenizer: "
            f"first mismatch at id {first}"
        )
    _validate_special_ids(gguf_model.metadata, target)

    layer_ids = list(config.target_layer_ids or [])
    if not layer_ids:
        raise ValueError(f"{architecture}.target_layers must be non-empty")
    if architecture == "eagle3" and len(layer_ids) != 3:
        raise ValueError("eagle3.target_layers must contain exactly three indices")
    if any(type(index) is not int for index in layer_ids):
        raise ValueError(f"{architecture}.target_layers must contain integers")
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError(f"{architecture}.target_layers contains duplicate indices")
    invalid = [index for index in layer_ids if index < 0 or index >= target_layers]
    if invalid:
        raise ValueError(
            f"{architecture}.target_layers contains indices outside target layer count "
            f"{target_layers}: {invalid}"
        )

    if architecture == "dflash":
        sliding_keys = (
            "dflash.attention.sliding_window",
            "dflash.attention.sliding_window_pattern",
        )
        present_sliding = [key for key in sliding_keys if key in gguf_model.metadata]
        if present_sliding:
            raise ValueError(
                "dflash sliding-window metadata is unsupported by the full-cache draft "
                f"graph: {present_sliding}"
            )
        if target_hidden != config.hidden_size:
            raise ValueError(
                f"dflash target hidden size {target_hidden} must equal draft hidden size "
                f"{config.hidden_size}"
            )
        if not config.block_size or config.block_size <= 0:
            raise ValueError("dflash.block_size must be a positive integer")
    else:
        if config.target_hidden_size != target_hidden:
            raise ValueError(
                f"eagle3.target_hidden_size={config.target_hidden_size} does not match "
                f"target hidden_size={target_hidden}"
            )
        if target_hidden != config.hidden_size:
            raise ValueError(
                "eagle3 target-shared embeddings require target_hidden_size to equal "
                f"draft hidden_size, got target={target_hidden}, draft={config.hidden_size}"
            )

    remap = _read_d2t(gguf_model)
    if remap is not None:
        if any(value < 0 or value >= target_vocab for value in remap):
            raise ValueError(
                f"{architecture} d2t contains target ids outside [0, {target_vocab})"
            )
        if len(set(remap)) != len(remap):
            raise ValueError(f"{architecture} d2t contains duplicate target ids")

    has_draft_head = "output.weight" in gguf_model.tensor_names
    output_name = "draft_logits" if has_draft_head else "draft_hidden"
    return {
        "format_version": 1,
        "kind": "speculative-draft",
        "architecture": architecture,
        "standalone": False,
        "runtime": "deferred",
        "target": {
            **target_identity,
            "model_type": target.get("model_type"),
            "hidden_size": target_hidden,
            "num_hidden_layers": target_layers,
            "vocab_size": target_vocab,
            "target_layers": layer_ids,
            "tokenizer_tokens_sha256": _tokenizer_digest(target_tokens),
        },
        "draft": {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "intermediate_size": config.intermediate_size,
            "block_size": getattr(config, "block_size", None),
            "draft_vocab_size": len(remap) if remap is not None else target_vocab,
        },
        "draft_to_target": remap,
        "orchestration": {
            "standalone_dispatch": "rejected",
            "embedding_source": "target",
            "lm_head_source": "draft" if has_draft_head else "target",
            "graph_output": output_name,
            "logits_vocabulary": (
                "draft; map proposed ids through draft_to_target"
                if remap is not None
                else "target"
            ),
            "target_hidden_layers": layer_ids,
            "cache_owner": "draft",
        },
    }


def validate_draft_tensor_contract(gguf_model: Any) -> None:
    """Validate the suffix-exact pinned llama.cpp draft tensor closure."""
    architecture = gguf_model.architecture
    if architecture not in _DRAFT_ARCHITECTURES:
        return
    metadata = gguf_model.metadata
    hidden = int(metadata[f"{architecture}.embedding_length"])
    intermediate = int(metadata[f"{architecture}.feed_forward_length"])
    layers = int(metadata[f"{architecture}.block_count"])
    heads = int(metadata[f"{architecture}.attention.head_count"])
    kv_heads = int(metadata.get(f"{architecture}.attention.head_count_kv", heads))
    head_dim = int(metadata.get(f"{architecture}.attention.key_length", hidden // heads))
    target_layers = list(metadata.get(f"{architecture}.target_layers", ()))
    target_hidden = (
        int(metadata["eagle3.target_hidden_size"]) if architecture == "eagle3" else hidden
    )
    target_vocab = len(metadata.get("tokenizer.ggml.tokens", ()))

    raw_items = list(gguf_model.tensor_items_raw())
    actual = {
        name: tuple(int(dim) for dim in shape) for name, _raw, _qtype, shape in raw_items
    }
    required: dict[str, tuple[int, ...]] = {
        "fc.weight": (hidden, len(target_layers) * target_hidden),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (target_vocab, hidden),
        "output.weight": (target_vocab, hidden),
    }
    if architecture == "dflash":
        required["enc.output_norm.weight"] = (hidden,)
        layer_shapes = {
            "attn_q.weight": (heads * head_dim, hidden),
            "attn_k.weight": (kv_heads * head_dim, hidden),
            "attn_v.weight": (kv_heads * head_dim, hidden),
            "attn_output.weight": (hidden, heads * head_dim),
            "attn_norm.weight": (hidden,),
            "attn_q_norm.weight": (head_dim,),
            "attn_k_norm.weight": (head_dim,),
            "ffn_norm.weight": (hidden,),
            "ffn_gate.weight": (intermediate, hidden),
            "ffn_up.weight": (intermediate, hidden),
            "ffn_down.weight": (hidden, intermediate),
        }
    else:
        if layers != 1:
            raise ValueError(f"eagle3.block_count must be 1, got {layers}")
        layer_shapes = {
            "attn_q.weight": (heads * head_dim, 2 * hidden),
            "attn_k.weight": (kv_heads * head_dim, 2 * hidden),
            "attn_v.weight": (kv_heads * head_dim, 2 * hidden),
            "attn_output.weight": (hidden, heads * head_dim),
            "attn_norm.weight": (hidden,),
            "attn_norm_2.weight": (hidden,),
            "ffn_norm.weight": (hidden,),
            "ffn_gate.weight": (intermediate, hidden),
            "ffn_up.weight": (intermediate, hidden),
            "ffn_down.weight": (hidden, intermediate),
        }
        if bool(metadata.get("eagle3.norm_before_fc", False)):
            required["enc.output_norm.weight"] = (3 * target_hidden,)

    for layer in range(layers):
        for suffix, shape in layer_shapes.items():
            required[f"blk.{layer}.{suffix}"] = shape
        if architecture == "eagle3":
            optional[f"blk.{layer}.rope_freqs.weight"] = (head_dim // 2,)

    if "d2t" in actual:
        if len(actual["d2t"]) != 1:
            raise ValueError(f"{architecture} d2t must be rank 1, got {actual['d2t']}")
        draft_vocab = actual["d2t"][0]
        optional["d2t"] = (draft_vocab,)
        optional["output.weight"] = (draft_vocab, hidden)
        if "output.weight" not in actual:
            raise ValueError(f"{architecture} reduced vocabulary requires output.weight")
    if architecture == "eagle3" and "output.weight" not in actual and "d2t" in actual:
        raise ValueError(
            "eagle3 cannot share a target lm_head with a reduced draft vocabulary"
        )
    if "token_embd.weight" in actual:
        raise ValueError(
            f"{architecture} GGUF contains a draft-owned token_embd.weight, but the Mobius "
            "draft graph consumes target-provided inputs_embeds and will not ignore it"
        )

    allowed_stems = set(required) | set(optional)
    unexpected = []
    for name in actual:
        base = name
        for suffix in (".input_scale", ".scale"):
            if name.endswith(suffix):
                base = name[: -len(suffix)] + ".weight"
                break
        if base not in allowed_stems:
            unexpected.append(name)
    if unexpected:
        raise ValueError(
            f"{architecture} GGUF has tensor suffixes outside the pinned loader closure: "
            f"{sorted(unexpected)}"
        )
    missing = sorted(set(required) - set(actual))
    if missing:
        raise ValueError(f"{architecture} GGUF is missing required tensor(s): {missing}")
    expected = {**optional, **required}
    malformed = {
        name: (expected[name], actual[name])
        for name in actual
        if name in expected and expected[name] != actual[name]
    }
    if malformed:
        raise ValueError(f"{architecture} GGUF has invalid tensor shape(s): {malformed}")


def write_draft_manifest(manifest: Mapping[str, Any], output_dir: str | Path) -> str:
    path = Path(output_dir) / "draft_manifest.json"
    path.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return str(path)
