# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fail-closed target pairing for speculative draft GGUF models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mobius._configs import DFlashConfig, Eagle3Config

_DRAFT_ARCHITECTURES = frozenset({"dflash", "eagle3"})


def is_draft_architecture(architecture: str) -> bool:
    return architecture in _DRAFT_ARCHITECTURES


def _ordered_tokenizer_vocab(tokenizer_json: Mapping[str, Any]) -> list[str]:
    model = tokenizer_json.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("target tokenizer.json has no model object")
    vocab = model.get("vocab")
    if isinstance(vocab, Mapping):
        size = max((int(index) for index in vocab.values()), default=-1) + 1
        tokens: list[str | None] = [None] * size
        for token, index in vocab.items():
            index = int(index)
            if index < 0 or index >= size or tokens[index] is not None:
                raise ValueError("target tokenizer.json has duplicate or invalid vocabulary ids")
            tokens[index] = str(token)
    elif isinstance(vocab, Sequence) and not isinstance(vocab, (str, bytes)):
        tokens = [
            str(entry[0] if isinstance(entry, Sequence) and not isinstance(entry, str) else entry)
            for entry in vocab
        ]
    else:
        raise ValueError("target tokenizer.json has no supported ordered vocabulary")

    for added in tokenizer_json.get("added_tokens", ()):
        if not isinstance(added, Mapping) or "id" not in added or "content" not in added:
            continue
        index = int(added["id"])
        if index >= len(tokens):
            tokens.extend([None] * (index + 1 - len(tokens)))
        token = str(added["content"])
        if tokens[index] not in (None, token):
            raise ValueError(f"target tokenizer id {index} has conflicting token values")
        tokens[index] = token
    if any(token is None for token in tokens):
        missing = [index for index, token in enumerate(tokens) if token is None]
        raise ValueError(f"target tokenizer vocabulary has unmapped ids: {missing[:8]}")
    return [str(token) for token in tokens]


def _load_target(
    target_config: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    if isinstance(target_config, Mapping):
        data = dict(target_config)
        tokens = data.pop("tokenizer_tokens", None)
        if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
            raise ValueError(
                "mapping target_config must include tokenizer_tokens for exact identity verification"
            )
        config_payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        return (
            data,
            [str(token) for token in tokens],
            {
                "source": str(data.get("target_model_id", "explicit-mapping")),
                "config_sha256": hashlib.sha256(config_payload).hexdigest(),
            },
        )

    path = Path(target_config)
    if path.is_dir():
        config_path = path / "config.json"
        tokenizer_path = path / "tokenizer.json"
    else:
        config_path = path
        tokenizer_path = path.parent / "tokenizer.json"
    if not config_path.is_file():
        raise ValueError(f"target config does not exist: {config_path}")
    if not tokenizer_path.is_file():
        raise ValueError(
            f"target tokenizer.json is required for exact draft pairing: {tokenizer_path}"
        )
    config_payload = config_path.read_bytes()
    data = json.loads(config_payload)
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    return (
        data,
        _ordered_tokenizer_vocab(tokenizer),
        {
            "source": str(config_path.resolve()),
            "config_sha256": hashlib.sha256(config_payload).hexdigest(),
        },
    )


def _flatten_text_config(data: Mapping[str, Any]) -> dict[str, Any]:
    flattened = dict(data)
    text_config = data.get("text_config")
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
            matches = int(gguf_value) in {int(value) for value in target_value}
        else:
            matches = int(gguf_value) == int(target_value)
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

    target_hidden = int(target["hidden_size"])
    target_layers = int(target["num_hidden_layers"])
    target_vocab = int(target["vocab_size"])
    gguf_tokens = [str(token) for token in gguf_model.metadata.get("tokenizer.ggml.tokens", ())]
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
            raise ValueError(f"{architecture} d2t contains target ids outside [0, {target_vocab})")
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
        raise ValueError("eagle3 cannot share a target lm_head with a reduced draft vocabulary")
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
    path.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)
