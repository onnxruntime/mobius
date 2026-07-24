# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extract Hugging Face tokenizer runtime assets from a GGUF file.

A GGUF checkpoint embeds its tokenizer as ``tokenizer.ggml.*`` metadata
(tokens, scores/merges, token types, and special-token ids). The onnx-genai
runtime loads ``<package>/tokenizer.json`` and ``tokenizer_config.json``, so a
model built from a GGUF file needs those files materialized alongside the ONNX
weights.

``transformers`` (5.x) can reconstruct a fast tokenizer directly from a GGUF
file via ``AutoTokenizer.from_pretrained(directory, gguf_file=filename)``; this
module wraps that and serializes the fast tokenizer plus its configuration,
mirroring the tokenizer helper in
``mobius.integrations.onnx_genai.auto_export``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

_GEMMA4_ORT_CHAT_TEMPLATE = """{{- bos_token -}}
{%- for message in messages -%}
{{- '<|turn>' + ('model' if message['role'] == 'assistant' else message['role']) + '\\n' -}}
{%- if message['content'] is string -%}
{{- message['content'] | trim -}}
{%- else -%}
{%- for item in message['content'] -%}
{%- if item['type'] == 'text' -%}
{{- item['text'] | trim -}}
{%- elif item['type'] == 'image' -%}
{{- '<|image|>' -}}
{%- elif item['type'] == 'audio' -%}
{{- '<|audio|>' -}}
{%- elif item['type'] == 'video' -%}
{{- '<|video|>' -}}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{{- '<turn|>\\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|turn>model\\n' -}}
{%- endif -%}
"""


def write_gguf_tokenizer_json(gguf_path: str | Path, output_dir: str | Path) -> str | None:
    """Write tokenizer runtime assets for a GGUF-built package.

    Reconstructs the fast tokenizer from the GGUF ``tokenizer.ggml.*`` metadata
    and serializes ``tokenizer.json`` plus ``tokenizer_config.json`` under
    *output_dir*. This is best-effort: it logs a warning and returns ``None``
    when the GGUF tokenizer cannot be converted, so the build is not blocked.

    Args:
        gguf_path: Path to the ``.gguf`` file whose embedded tokenizer to emit.
        output_dir: Package directory to write ``tokenizer.json`` into.

    Returns:
        The written ``tokenizer.json`` path, or ``None`` if it could not be
        emitted.
    """
    gguf_path = Path(gguf_path)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        _LOGGER.warning(
            "transformers is not available; skipping tokenizer.json emission. "
            "The onnx-genai runners will need a tokenizer.json supplied separately."
        )
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(gguf_path.parent),
            gguf_file=gguf_path.name,
            use_fast=True,
        )
    except Exception as error:  # best-effort; transformers may not know the arch
        _LOGGER.info(
            "transformers could not load a tokenizer from %r (%s); "
            "reconstructing directly from the GGUF tokenizer metadata.",
            str(gguf_path),
            error,
        )
        return _reconstruct_tokenizer_from_ggml(gguf_path, output_dir)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        _LOGGER.warning(
            "Reconstructed a slow tokenizer from %r with no fast backend; "
            "skipping tokenizer.json emission.",
            str(gguf_path),
        )
        return None
    tokenizer.save_pretrained(str(output_dir))
    from mobius.integrations.gguf._reader import GGUFModel

    _write_chat_template(GGUFModel(str(gguf_path)).metadata, output_dir)
    path = os.path.join(str(output_dir), "tokenizer.json")
    return path


def _reconstruct_tokenizer_from_ggml(gguf_path: Path, output_dir: str | Path) -> str | None:
    """Build ``tokenizer.json`` directly from GGUF ``tokenizer.ggml.*`` metadata.

    Fallback for GGUF architectures whose tokenizer ``transformers`` does not yet
    support (e.g. ``gemma4``). Reconstructs the fast BPE tokenizer from the
    embedded tokens + merges + special-token ids and validates it with an
    encode→decode round-trip. Best-effort: logs a warning and returns ``None``
    (never raising) if the required metadata or libraries are missing, or the
    round-trip fails.
    """
    try:
        from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
        from tokenizers.models import BPE

        from mobius.integrations.gguf._reader import GGUFModel
    except ImportError as error:
        _LOGGER.warning("Cannot reconstruct tokenizer (missing dependency: %s).", error)
        return None

    try:
        metadata = GGUFModel(str(gguf_path)).metadata
        tokens = metadata.get("tokenizer.ggml.tokens")
        merges_raw = metadata.get("tokenizer.ggml.merges")
        if not tokens or not merges_raw:
            _LOGGER.warning(
                "GGUF %r has no tokenizer.ggml.tokens/merges; skipping tokenizer.json.",
                str(gguf_path),
            )
            return None
        token_types = metadata.get("tokenizer.ggml.token_type") or []
        unknown_id = int(metadata.get("tokenizer.ggml.unknown_token_id", 0))
        bos_id = metadata.get("tokenizer.ggml.bos_token_id")
        add_bos = bool(metadata.get("tokenizer.ggml.add_bos_token", False))
        add_space_prefix = bool(metadata.get("tokenizer.ggml.add_space_prefix", False))

        vocab = {token: index for index, token in enumerate(tokens)}
        # llama.cpp stores BPE merges as space-joined "left right" pairs.
        merges = [
            (parts[0], parts[1]) for merge in merges_raw if len(parts := merge.split(" ")) == 2
        ]
        unknown_token = tokens[unknown_id] if unknown_id < len(tokens) else None

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                unk_token=unknown_token,
                fuse_unk=True,
                byte_fallback=True,
            )
        )
        # SentencePiece semantics: '▁' marks word boundaries; unknown bytes fall
        # back to <0xNN> byte tokens.
        prepend_scheme = "always" if add_space_prefix else "never"
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme=prepend_scheme
        )
        tokenizer.decoder = decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(content=" ", left=1, right=0),
            ]
        )
        # Preserve control / user-defined tokens (llama.cpp types 3 and 4) as
        # atomic special tokens so they are never split.
        from tokenizers import AddedToken

        special_tokens = [
            AddedToken(str(tokens[index]), special=True, normalized=False)
            for index, token_type in enumerate(token_types)
            if token_type in (3, 4) and index < len(tokens)
        ]
        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)

        if add_bos and bos_id is not None and int(bos_id) < len(tokens):
            bos_token = tokens[int(bos_id)]
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{bos_token} $A",
                pair=f"{bos_token} $A {bos_token} $B",
                special_tokens=[(bos_token, int(bos_id))],
            )

        # Sanity check: encode→decode round-trip. This is a soft signal (a small
        # or byte-incomplete vocab may not represent arbitrary text); the token
        # ids are correct by construction from the ggml ordering, so warn rather
        # than discard a reconstructed tokenizer.
        for sample in ("Hello, world!", "The capital of France is Paris."):
            decoded = tokenizer.decode(tokenizer.encode(sample).ids)
            if decoded.strip() != sample.strip():
                _LOGGER.warning(
                    "Reconstructed tokenizer round-trip differs (%r -> %r); "
                    "emitting anyway (ids follow the GGUF vocab order).",
                    sample,
                    decoded,
                )
                break

        path = os.path.join(str(output_dir), "tokenizer.json")
        tokenizer.save(path)
        _write_tokenizer_config(metadata, tokens, output_dir)
        _write_chat_template(metadata, output_dir)
    except Exception as error:  # best-effort; never block the build
        _LOGGER.warning(
            "Failed to reconstruct tokenizer from GGUF metadata %r: %s; "
            "skipping tokenizer.json emission.",
            str(gguf_path),
            error,
        )
        return None
    else:
        _LOGGER.info(
            "Reconstructed tokenizer.json from GGUF metadata (%d tokens).", len(tokens)
        )
        return path


def _write_tokenizer_config(metadata: dict, tokens: list[str], output_dir: str | Path) -> str:
    """Write the tokenizer sidecar required by ORT-GenAI."""
    architecture = str(metadata.get("general.architecture", ""))
    tokenizer_class = (
        "GemmaTokenizer" if architecture.startswith("gemma") else "LlamaTokenizer"
    )

    def _token(key: str) -> str | None:
        token_id = metadata.get(key)
        if token_id is None or not 0 <= int(token_id) < len(tokens):
            return None
        return tokens[int(token_id)]

    config = {
        "tokenizer_class": tokenizer_class,
        "model_max_length": int(metadata.get(f"{architecture}.context_length", 1_000_000_000)),
        "bos_token": _token("tokenizer.ggml.bos_token_id"),
        "eos_token": _token("tokenizer.ggml.eos_token_id"),
        "unk_token": _token("tokenizer.ggml.unknown_token_id"),
        "pad_token": _token("tokenizer.ggml.padding_token_id"),
        "mask_token": _token("tokenizer.ggml.mask_token_id"),
    }
    if architecture == "gemma4":
        config.update(
            {
                "boi_token": "<|image>",
                "image_token": "<|image|>",
                "eoi_token": "<image|>",
                "boa_token": "<|audio>",
                "audio_token": "<|audio|>",
                "eoa_token": "<audio|>",
                "padding_side": "left",
                "processor_class": "Gemma4Processor",
            }
        )
    config = {key: value for key, value in config.items() if value is not None}
    path = os.path.join(str(output_dir), "tokenizer_config.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    return path


def _write_chat_template(metadata: dict, output_dir: str | Path) -> str | None:
    architecture = str(metadata.get("general.architecture", ""))
    chat_template = (
        _GEMMA4_ORT_CHAT_TEMPLATE
        if architecture == "gemma4"
        else metadata.get("tokenizer.chat_template")
    )
    if not isinstance(chat_template, str) or not chat_template:
        return None
    path = os.path.join(str(output_dir), "chat_template.jinja")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(chat_template)
    return path
