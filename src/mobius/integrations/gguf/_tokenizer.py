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
    except Exception as error:  # best-effort; never block the build
        _LOGGER.warning(
            "Could not reconstruct a tokenizer from the GGUF file %r: %s; "
            "skipping tokenizer.json emission.",
            str(gguf_path),
            error,
        )
        return None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        _LOGGER.warning(
            "Reconstructed a slow tokenizer from %r with no fast backend; "
            "skipping tokenizer.json emission.",
            str(gguf_path),
        )
        return None
    path = os.path.join(str(output_dir), "tokenizer.json")
    backend.save(path)
    return path
