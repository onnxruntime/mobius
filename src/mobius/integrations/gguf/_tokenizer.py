# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extract a Hugging Face ``tokenizer.json`` from a GGUF file.

A GGUF checkpoint embeds its tokenizer as ``tokenizer.ggml.*`` metadata
(tokens, scores/merges, token types, and special-token ids). The onnx-genai
runtime loads ``<package>/tokenizer.json`` — the ``tokenizers``-library fast
tokenizer serialization — so a model built from a GGUF file needs that file
materialized alongside the ONNX weights.

``transformers`` (5.x) can reconstruct a fast tokenizer directly from a GGUF
file via ``AutoTokenizer.from_pretrained(directory, gguf_file=filename)``; this
module wraps that and serializes the fast backend to ``tokenizer.json``,
mirroring the diffusion CLIP tokenizer helper in
``mobius.integrations.onnx_genai.auto_export``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


def write_gguf_tokenizer_json(gguf_path: str | Path, output_dir: str | Path) -> str | None:
    """Write ``tokenizer.json`` for a GGUF-built package.

    Reconstructs the fast tokenizer from the GGUF ``tokenizer.ggml.*`` metadata
    and serializes it to ``<output_dir>/tokenizer.json``. This is best-effort:
    it logs a warning and returns ``None`` (never raising) when ``transformers``
    is unavailable or the GGUF's tokenizer model cannot be converted, so the
    build is not blocked — the onnx-genai runners can be given a
    ``tokenizer.json`` separately.

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
    _ensure_bos_post_processor(backend, gguf_path)
    path = os.path.join(str(output_dir), "tokenizer.json")
    backend.save(path)
    return path


def _ensure_bos_post_processor(backend, gguf_path: Path) -> None:
    """Attach a BOS-prepending post-processor if the GGUF asks for one.

    ``transformers`` (5.x) can reconstruct a fast tokenizer from a GGUF's
    ``tokenizer.ggml.*`` metadata, but for some architectures (e.g. Gemma,
    whose GGUF tokenizer is loaded as a ``Unigram`` model) the resulting fast
    backend does **not** carry the ``add_bos_token`` post-processor. Models
    like Gemma require the ``<bos>`` prefix — without it, greedy decode
    degenerates into single-token repetition. This restores that post-processor
    from the GGUF metadata when the reconstructed tokenizer would otherwise omit
    it. Best-effort: never raises (a tokenizer without BOS still saves).
    """
    try:
        from tokenizers import processors

        from mobius.integrations.gguf._reader import GGUFModel

        metadata = GGUFModel(str(gguf_path)).metadata
        if not bool(metadata.get("tokenizer.ggml.add_bos_token", False)):
            return
        bos_id = metadata.get("tokenizer.ggml.bos_token_id")
        tokens = metadata.get("tokenizer.ggml.tokens")
        if bos_id is None or not tokens:
            return
        bos_id = int(bos_id)
        if bos_id < 0 or bos_id >= len(tokens):
            return
        bos_token = tokens[bos_id]
        # Probe: only attach BOS if the current pipeline does not already emit
        # it (avoids a doubled ``<bos>`` when transformers did wire it up).
        try:
            probe = backend.encode("probe").ids
            if probe and probe[0] == bos_id:
                return
        except Exception:  # pragma: no cover - probing is best-effort
            pass
        backend.post_processor = processors.TemplateProcessing(
            single=f"{bos_token} $A",
            pair=f"{bos_token} $A {bos_token} $B",
            special_tokens=[(bos_token, bos_id)],
        )
        _LOGGER.info(
            "Restored BOS post-processor (%r, id=%d) on GGUF-reconstructed tokenizer.",
            bos_token,
            bos_id,
        )
    except Exception as error:  # pragma: no cover - best-effort
        _LOGGER.warning(
            "Could not verify/restore BOS post-processor for %r: %s.",
            str(gguf_path),
            error,
        )


# Pre-tokenizer split rules keyed by `tokenizer.ggml.pre`, applied before the
# byte-level step (which then runs with `use_regex=False`). These mirror the
# per-family regexes llama.cpp keeps in `llama_vocab::init_tokenizer`; a family
# absent from this table falls back to the stock GPT-2 regex.
_PRE_TOKENIZER_SPLITS: dict[str, tuple[str, ...]] = {
    "hunyuan-dense": (
        r"\p{N}{1,3}",
        r"[\x{4E00}-\x{9FFF}\x{3040}-\x{309F}\x{30A0}-\x{30FF}]+",
        r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
    ),
}


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
        from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, processors
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
        # `tokenizer.ggml.model` names the tokenizer family. "gpt2" is byte-level
        # BPE, whose vocabulary encodes a leading space as the byte-mapped glyph
        # 'Ġ'; SentencePiece families ("llama", "t5", ...) instead use '▁'.
        # These are different alphabets, so the pre-tokenizer and decoder must
        # follow the declared family — applying Metaspace to a byte-level vocab
        # silently drops every inter-word space.
        ggml_model = str(metadata.get("tokenizer.ggml.model") or "").lower()
        byte_level = ggml_model in ("gpt2", "bloom", "falcon")

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
                unk_token=None if byte_level else unknown_token,
                fuse_unk=not byte_level,
                byte_fallback=not byte_level,
            )
        )
        if byte_level:
            # Byte-level BPE: every byte already maps to a printable glyph, so
            # there is no unknown token and no byte fallback. `add_prefix_space`
            # stays False because the vocabulary distinguishes 'world' from
            # 'Ġworld' and the caller's text must not be altered.
            #
            # `tokenizer.ggml.pre` names the *split* rule, which is not implied
            # by the byte-level model: llama.cpp keeps a table of per-family
            # regexes because families disagree on how digits, CJK, and
            # punctuation are grouped, and a mismatch changes token ids (the
            # text still round-trips, so only an id-level check catches it).
            split_patterns = _PRE_TOKENIZER_SPLITS.get(
                str(metadata.get("tokenizer.ggml.pre") or "").lower()
            )
            byte_level_pre = pre_tokenizers.ByteLevel(
                add_prefix_space=False, use_regex=not split_patterns
            )
            if split_patterns:
                tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                    [
                        pre_tokenizers.Split(
                            pattern=Regex(pattern), behavior="isolated"
                        )
                        for pattern in split_patterns
                    ]
                    + [byte_level_pre]
                )
            else:
                tokenizer.pre_tokenizer = byte_level_pre
            tokenizer.decoder = decoders.ByteLevel()
        else:
            # SentencePiece semantics: '▁' marks word boundaries; unknown bytes
            # fall back to <0xNN> byte tokens.
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

        if add_bos and bos_id is not None:
            bos_id = int(bos_id)
            if 0 <= bos_id < len(tokens):
                bos_token = tokens[bos_id]
                tokenizer.post_processor = processors.TemplateProcessing(
                    single=f"{bos_token} $A",
                    pair=f"{bos_token} $A {bos_token} $B",
                    special_tokens=[(bos_token, bos_id)],
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
