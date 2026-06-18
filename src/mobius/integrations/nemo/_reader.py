# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NeMo ``.nemo`` file reading and config/tensor extraction.

A ``.nemo`` file is an (uncompressed) tar archive produced by NVIDIA NeMo.
It bundles everything needed to restore a model:

- ``model_config.yaml`` — the Hydra/OmegaConf model configuration that
  describes the architecture (encoder, decoder, joint, tokenizer, ...).
- ``model_weights.ckpt`` — a PyTorch ``state_dict`` (``torch.save``) holding
  every parameter, keyed by the NeMo module path
  (e.g. ``encoder.layers.0.self_attn.linear_q.weight``).
- One or more tokenizer artifacts (SentencePiece ``*.model`` / ``*.vocab``)
  referenced from the config via ``nemo:<filename>`` URIs.

This module exposes :class:`NeMoArchive`, a thin reader that extracts those
pieces without depending on the (very heavy) ``nemo_toolkit`` package.
"""

from __future__ import annotations

import io
import logging
import os
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_CONFIG_NAMES = ("model_config.yaml", "model_config.yml")
_WEIGHTS_NAME = "model_weights.ckpt"


def _looks_like_hf_repo_id(value: str) -> bool:
    """Heuristic: ``value`` matches ``owner/repo`` (no path separators, no .nemo suffix)."""
    if value.startswith((".", "/", "~")):
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(p and not p.endswith(".nemo") for p in parts)


def _resolve_nemo_path(nemo_path: str | Path, revision: str | None = None) -> str:
    """Resolve a ``.nemo`` reference to a local file path.

    Accepts:
    - An existing local filesystem path (returned unchanged).
    - A HuggingFace Hub reference ``"owner/repo"`` — the repo must contain
      exactly one ``*.nemo`` file, which is downloaded.
    - A HuggingFace Hub reference ``"owner/repo:filename.nemo"`` to pick a
      specific file from a multi-file repo.

    Args:
        nemo_path: Local path or HuggingFace Hub reference.
        revision: Optional HuggingFace Hub revision (branch, tag, or commit
            SHA) used to pin downloads for reproducibility.
    """
    from huggingface_hub import HfApi, hf_hub_download

    raw = str(nemo_path)
    expanded = os.path.expanduser(raw)
    if Path(expanded).exists():
        return expanded

    repo_id, _, filename = raw.partition(":")
    if not _looks_like_hf_repo_id(repo_id):
        return raw  # Let NeMoArchive raise FileNotFoundError with the original path.

    if not filename:
        files = [
            f
            for f in HfApi().list_repo_files(repo_id, revision=revision)
            if f.endswith(".nemo")
        ]
        if not files:
            raise FileNotFoundError(f"No *.nemo files found in HF repo {repo_id!r}")
        if len(files) > 1:
            raise ValueError(
                f"HF repo {repo_id!r} contains multiple .nemo files: {files}. "
                f"Specify one via '{repo_id}:<filename.nemo>'."
            )
        filename = files[0]

    logger.info("Downloading %s from %s (revision=%s)", filename, repo_id, revision)
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


class NeMoArchive:
    """Reader for a NeMo ``.nemo`` archive.

    Parses the bundled ``model_config.yaml`` eagerly and exposes the model
    weights and tokenizer artifacts on demand.

    Args:
        nemo_path: Path to a local ``.nemo`` file, *or* a HuggingFace Hub
            reference (``"owner/repo"`` or ``"owner/repo:filename.nemo"``).
        revision: Optional HuggingFace Hub revision (branch, tag, or commit
            SHA) to pin downloads. Ignored for local paths.
    """

    def __init__(self, nemo_path: str | Path, revision: str | None = None):
        self.path = _resolve_nemo_path(nemo_path, revision=revision)
        if not Path(self.path).exists():
            raise FileNotFoundError(f"NeMo file not found: {self.path}")
        # Member name (basename) -> full archive member name, for lookups.
        self._members: dict[str, str] = {}
        self.config: dict[str, Any] = {}
        self._read_index_and_config()

    def _read_index_and_config(self) -> None:
        import yaml

        with tarfile.open(self.path, mode="r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                base = Path(member.name).name
                # Prefer the first occurrence; NeMo archives are flat.
                self._members.setdefault(base, member.name)

            config_member = next(
                (self._members[n] for n in _CONFIG_NAMES if n in self._members),
                None,
            )
            if config_member is None:
                raise ValueError(
                    f"{self.path!r} does not contain a model_config.yaml; "
                    "is it a valid .nemo archive?"
                )
            extracted = tar.extractfile(config_member)
            assert extracted is not None
            self.config = yaml.safe_load(extracted.read())

    @property
    def target(self) -> str:
        """The NeMo ``target`` class path (e.g. ``...EncDecRNNTBPEModel``)."""
        return str(self.config.get("target", ""))

    def read_file(self, basename: str) -> bytes:
        """Return the raw bytes of an archive member by basename."""
        member = self._members.get(basename)
        if member is None:
            raise KeyError(f"{basename!r} not found in {self.path!r}")
        with tarfile.open(self.path, mode="r:*") as tar:
            extracted = tar.extractfile(member)
            assert extracted is not None
            return extracted.read()

    def resolve_nemo_uri(self, uri: str | None) -> str | None:
        """Resolve a ``nemo:<basename>`` config URI to an archive basename.

        NeMo stores tokenizer paths as ``nemo:<hash>_<name>``; the prefix is
        stripped here so callers can fetch the member with :meth:`read_file`.
        Returns ``None`` for a missing/``None`` URI.
        """
        if not uri:
            return None
        return uri[len("nemo:") :] if uri.startswith("nemo:") else uri

    def extract_tokenizer(self, dest_dir: str | Path) -> dict[str, str]:
        """Extract tokenizer artifacts referenced by the config.

        Writes the SentencePiece ``model``/``vocab`` files into *dest_dir*
        and returns a mapping of config key (``model_path`` / ``vocab_path`` /
        ``spe_tokenizer_vocab``) to the written file path.
        """
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        tok_cfg = self.config.get("tokenizer", {}) or {}
        written: dict[str, str] = {}
        for key in ("model_path", "vocab_path", "spe_tokenizer_vocab"):
            base = self.resolve_nemo_uri(tok_cfg.get(key))
            if base is None or base not in self._members:
                continue
            out_path = dest / base
            out_path.write_bytes(self.read_file(base))
            written[key] = str(out_path)
        return written

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Load the PyTorch ``state_dict`` from ``model_weights.ckpt``.

        The checkpoint is a plain ``torch.save`` of a parameter dict, so it is
        loaded with ``weights_only=True`` (no arbitrary code execution).
        """
        import torch

        if _WEIGHTS_NAME not in self._members:
            raise ValueError(
                f"{self.path!r} does not contain {_WEIGHTS_NAME}; is it a valid .nemo archive?"
            )
        # Stream weights straight from the tar member to avoid buffering the
        # entire (multi-GB) checkpoint in memory. ``torch.load`` needs a
        # seekable stream for zip-format checkpoints; an uncompressed tar member
        # is seekable, so fall back to a buffered read only when it is not
        # (e.g. a compressed ``.tar.gz`` archive).
        member = self._members[_WEIGHTS_NAME]
        with tarfile.open(self.path, mode="r:*") as tar:
            extracted = tar.extractfile(member)
            assert extracted is not None
            if extracted.seekable():
                obj = torch.load(extracted, map_location="cpu", weights_only=True)
            else:
                obj = torch.load(
                    io.BytesIO(extracted.read()), map_location="cpu", weights_only=True
                )
        # NeMo checkpoints are usually a bare state_dict, but tolerate the
        # Lightning-style {"state_dict": {...}} wrapper too.
        if isinstance(obj, dict) and "state_dict" in obj and "encoder" not in obj:
            obj = obj["state_dict"]
        return obj
