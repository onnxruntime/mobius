# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build ONNX models from native Kyutai Moshi / Mimi checkpoints.

The builder resolves a Mimi codec ``safetensors`` checkpoint (local path or
HuggingFace Hub reference), constructs the :class:`~mobius.models.MimiModel`
ONNX graphs, and applies the converted weights.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path

from mobius._model_package import ModelPackage

logger = logging.getLogger(__name__)

# Default Mimi codec filename inside the personaplex / Moshi HF repos.
_MIMI_GLOB = "tokenizer-*.safetensors"


def _looks_like_hf_repo_id(value: str) -> bool:
    """Heuristic: ``owner/name`` with no filesystem separators beyond one ``/``."""
    return value.count("/") == 1 and not value.startswith((".", "/", "~"))


def _resolve_mimi_checkpoint(
    checkpoint: str | Path, *, revision: str | None
) -> str:
    """Return a local path to the Mimi ``safetensors`` file.

    Accepts a local file/dir, ``owner/repo`` (auto-discovers the
    ``tokenizer-*.safetensors`` file), or ``owner/repo:filename.safetensors``.
    """
    import fnmatch

    from huggingface_hub import HfApi, hf_hub_download

    raw = str(checkpoint)
    expanded = os.path.expanduser(raw)
    path = Path(expanded)
    if path.is_file():
        return expanded
    if path.is_dir():
        matches = sorted(path.glob(_MIMI_GLOB))
        if not matches:
            raise FileNotFoundError(
                f"No {_MIMI_GLOB!r} file found in directory {expanded!r}"
            )
        return str(matches[0])

    repo_id, _, filename = raw.partition(":")
    if not _looks_like_hf_repo_id(repo_id):
        raise FileNotFoundError(f"Mimi checkpoint not found: {raw!r}")

    if not filename:
        files = [
            f
            for f in HfApi().list_repo_files(repo_id, revision=revision)
            if fnmatch.fnmatch(os.path.basename(f), _MIMI_GLOB)
        ]
        if not files:
            raise FileNotFoundError(
                f"No {_MIMI_GLOB!r} file found in HF repo {repo_id!r}"
            )
        if len(files) > 1:
            raise ValueError(
                f"HF repo {repo_id!r} contains multiple Mimi checkpoints: "
                f"{files}. Specify one via '{repo_id}:<filename.safetensors>'."
            )
        filename = files[0]

    logger.info("Downloading %s from %s (revision=%s)", filename, repo_id, revision)
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def build_mimi(
    checkpoint: str | Path,
    *,
    dtype: str | None = None,
    execution_provider: str = "default",
    revision: str | None = None,
) -> ModelPackage:
    """Build the Mimi codec ONNX :class:`ModelPackage` from a Kyutai checkpoint.

    Args:
        checkpoint: Local path to a Mimi ``safetensors`` file or directory, or
            a HuggingFace Hub reference (``"owner/repo"`` or
            ``"owner/repo:tokenizer-*.safetensors"``).
        dtype: Override model dtype (e.g. ``"f16"``). When ``None``, float32.
        execution_provider: Target execution provider for EP-aware
            optimisations. Defaults to ``"default"`` (portable, no vendor
            fusions).
        revision: Optional HuggingFace Hub revision to pin downloads.

    Returns:
        A :class:`ModelPackage` with ``encoder`` (waveform -> codes) and
        ``decoder`` (codes -> waveform) graphs.
    """
    from safetensors.torch import load_file

    from mobius._builder import build_from_module, resolve_dtype
    from mobius.models.mimi import MimiModel, mimi_default_config
    from mobius.tasks import CodecTask

    ckpt_path = _resolve_mimi_checkpoint(checkpoint, revision=revision)
    logger.info("Loading Mimi checkpoint: %s", ckpt_path)

    config = mimi_default_config()
    if dtype is not None:
        resolved = resolve_dtype(dtype)
        if resolved is not None and resolved != config.dtype:
            config = dataclasses.replace(config, dtype=resolved)

    module = MimiModel(config)
    pkg = build_from_module(
        module, config, CodecTask(), execution_provider=execution_provider
    )

    state_dict = load_file(ckpt_path)
    logger.info("Loaded %d Mimi parameters", len(state_dict))
    pkg.apply_weights(module.preprocess_weights(state_dict))
    return pkg
