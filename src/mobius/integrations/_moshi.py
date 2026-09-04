# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build PersonaPlex ONNX packages from native Kyutai Moshi / Mimi checkpoints.

Use :func:`mobius.build` with ``nvidia/personaplex-7b-v1``. The private helpers
in this module resolve the checkpoint's Mimi codec and Moshi LM weights and
assemble their four graphs into one :class:`~mobius.ModelPackage`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path

import onnx_ir as ir

from mobius._model_package import ModelPackage

logger = logging.getLogger(__name__)

_PERSONAPLEX_MODEL_ID = "nvidia/personaplex-7b-v1"
# Immutable Hub identity used by the native builders and the parity fixtures.
_PERSONAPLEX_REVISION = "fdaf4090a61cb315c138a1faee287ffd6c716309"
_PERSONAPLEX_DEP_Q = 16


@dataclasses.dataclass
class _PersonaPlexWorkflowConfig:
    delays: list[int] = dataclasses.field(
        default_factory=lambda: [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    )
    dep_q: int = _PERSONAPLEX_DEP_Q
    n_q: int = 8
    frame_size: int = 1920
    context: int = 3000
    text_initial_token_id: int = 32000
    initial_token_id: int = 2048


def _is_personaplex_checkpoint(checkpoint: str | Path) -> bool:
    """Recognize the canonical Hub ID or a strongly identified local checkpoint."""
    path = _local_checkpoint_path(checkpoint)
    if path is not None:
        return _is_local_personaplex_checkpoint(path)
    return str(checkpoint).casefold() == _PERSONAPLEX_MODEL_ID.casefold()


def _local_checkpoint_path(checkpoint: str | Path) -> Path | None:
    """Resolve an existing local checkpoint directory before interpreting Hub IDs."""
    path = Path(os.path.expanduser(str(checkpoint)))
    return path if path.is_dir() else None


def _is_local_personaplex_checkpoint(path: Path) -> bool:
    """Identify a native local checkpoint without reading tensor payloads."""
    config_path = path / "config.json"
    lm_path = path / _LM_GLOB
    mimi_paths = sorted(path.glob(_MIMI_GLOB))
    if not config_path.is_file() or not lm_path.is_file() or len(mimi_paths) != 1:
        return False
    from safetensors import SafetensorError, safe_open

    try:
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            return False

        with safe_open(lm_path, framework="pt", device="cpu") as handle:
            lm_keys = set(handle.keys())
            lm_shapes = {
                key: tuple(handle.get_slice(key).get_shape())
                for key in (
                    "text_emb.weight",
                    "text_linear.weight",
                    "depformer_in.15.weight",
                )
                if key in lm_keys
            }
        with safe_open(mimi_paths[0], framework="pt", device="cpu") as handle:
            mimi_keys = set(handle.keys())
    except (OSError, ValueError, json.JSONDecodeError, SafetensorError):
        return False

    # The two native files and these independent architecture signatures avoid
    # hijacking arbitrary Transformers repos that happen to use model.safetensors.
    return (
        {"text_emb.weight", "text_linear.weight", "depformer_in.15.weight"} <= lm_keys
        and lm_shapes["text_emb.weight"] == (32001, 4096)
        and lm_shapes["text_linear.weight"] == (32000, 4096)
        and lm_shapes["depformer_in.15.weight"] == (1024, 4096)
        and any(key.startswith("encoder.model.") for key in mimi_keys)
        and any(key.startswith("decoder.model.") for key in mimi_keys)
    )


def _personaplex_revision(checkpoint: str | Path, revision: str | None) -> str | None:
    if _local_checkpoint_path(checkpoint) is not None:
        return None
    if revision is not None:
        return revision
    if _is_personaplex_checkpoint(checkpoint):
        return _PERSONAPLEX_REVISION
    return None


def _build_personaplex(
    checkpoint: str | Path,
    *,
    dtype: str | ir.DataType | None = None,
    execution_provider: str = "default",
    revision: str | None = None,
    load_weights: bool = True,
) -> ModelPackage:
    """Build one flat PersonaPlex package for the standard Mobius build route."""
    from mobius._builder import resolve_dtype

    resolved_dtype = resolve_dtype(dtype) or ir.DataType.FLOAT
    if resolved_dtype != ir.DataType.FLOAT:
        raise ValueError(
            "PersonaPlex only supports dtype='f32': its Mimi codec does not have "
            "validated fp16/bf16 graph and runtime support"
        )
    revision = _personaplex_revision(checkpoint, revision)
    mimi = _build_mimi(
        checkpoint,
        dtype=resolved_dtype,
        execution_provider=execution_provider,
        revision=revision,
        load_weights=load_weights,
    )
    moshi = _build_moshi_lm(
        checkpoint,
        dtype=resolved_dtype,
        execution_provider=execution_provider,
        revision=revision,
        load_weights=load_weights,
        dep_q=_PERSONAPLEX_DEP_Q,
    )
    package = ModelPackage({**mimi, **moshi}, config=_PersonaPlexWorkflowConfig())
    if set(package) != {"encoder", "decoder", "temporal", "depformer"}:
        raise ValueError("PersonaPlex build did not produce the required four components")
    for model in package.values():
        model.metadata_props["mobius.source_revision"] = revision or "local"
    return package


# Default Mimi codec filename inside the personaplex / Moshi HF repos.
_MIMI_GLOB = "tokenizer-*.safetensors"


def _looks_like_hf_repo_id(value: str) -> bool:
    """Heuristic: ``owner/name`` with no filesystem separators beyond one ``/``."""
    return value.count("/") == 1 and not value.startswith((".", "/", "~"))


def _resolve_mimi_checkpoint(checkpoint: str | Path, *, revision: str | None) -> str:
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
            raise FileNotFoundError(f"No {_MIMI_GLOB!r} file found in directory {expanded!r}")
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
            raise FileNotFoundError(f"No {_MIMI_GLOB!r} file found in HF repo {repo_id!r}")
        if len(files) > 1:
            raise ValueError(
                f"HF repo {repo_id!r} contains multiple Mimi checkpoints: "
                f"{files}. Specify one via '{repo_id}:<filename.safetensors>'."
            )
        filename = files[0]

    logger.info("Downloading %s from %s (revision=%s)", filename, repo_id, revision)
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def _build_mimi(
    checkpoint: str | Path,
    *,
    dtype: str | ir.DataType | None = None,
    execution_provider: str = "default",
    revision: str | None = None,
    load_weights: bool = True,
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
        load_weights: Load and apply the native checkpoint payload. When false,
            graph construction uses only the fixed architecture metadata.

    Returns:
        A :class:`ModelPackage` with ``encoder`` (waveform -> codes) and
        ``decoder`` (codes -> waveform) graphs.
    """
    from mobius._builder import build_from_module, resolve_dtype
    from mobius.models.mimi import MimiModel, _mimi_default_config
    from mobius.tasks import CodecTask

    config = _mimi_default_config()
    if dtype is not None:
        resolved = resolve_dtype(dtype)
        if resolved is not None and resolved != config.dtype:
            config = dataclasses.replace(config, dtype=resolved)

    module = MimiModel(config)
    pkg = build_from_module(module, config, CodecTask(), execution_provider=execution_provider)

    if load_weights:
        from safetensors.torch import load_file

        ckpt_path = _resolve_mimi_checkpoint(checkpoint, revision=revision)
        logger.info("Loading Mimi checkpoint: %s", ckpt_path)
        state_dict = load_file(ckpt_path)
        logger.info("Loaded %d Mimi parameters", len(state_dict))
        pkg.apply_weights(module.preprocess_weights(state_dict))
    return pkg


# Default Moshi LM filename inside the personaplex / Moshi HF repos.
_LM_GLOB = "model.safetensors"


def _resolve_lm_checkpoint(checkpoint: str | Path, *, revision: str | None) -> str:
    """Return a local path to the Moshi LM ``safetensors`` file.

    Accepts a local file/dir, ``owner/repo`` (auto-discovers
    ``model.safetensors``), or ``owner/repo:filename.safetensors``.
    """
    from huggingface_hub import hf_hub_download

    raw = str(checkpoint)
    expanded = os.path.expanduser(raw)
    path = Path(expanded)
    if path.is_file():
        return expanded
    if path.is_dir():
        matches = sorted(path.glob(_LM_GLOB))
        if not matches:
            raise FileNotFoundError(f"No {_LM_GLOB!r} file found in directory {expanded!r}")
        return str(matches[0])

    repo_id, _, filename = raw.partition(":")
    if not _looks_like_hf_repo_id(repo_id):
        raise FileNotFoundError(f"Moshi LM checkpoint not found: {raw!r}")
    if not filename:
        filename = _LM_GLOB
    logger.info("Downloading %s from %s (revision=%s)", filename, repo_id, revision)
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def _build_moshi_lm(
    checkpoint: str | Path,
    *,
    dtype: str | ir.DataType | None = None,
    execution_provider: str = "default",
    revision: str | None = None,
    load_weights: bool = True,
    dep_q: int | None = None,
) -> ModelPackage:
    """Build the Moshi LM ONNX graphs from a native Kyutai checkpoint.

    Produces one package containing the 7B temporal transformer and per-substep
    depformer. Both load from the same ``model.safetensors`` checkpoint.

    The personaplex checkpoint already ships full 16-step depformer weights
    (``self_attn.in_proj_weight`` is ``[16 * 3 * 1024, 1024]``, ``gating`` /
    ``linears`` / ``depformer_in`` have all 16 entries), so the Kyutai loader's
    "expand / copy 0..7 -> 8..15" patches are not required here.

    Args:
        checkpoint: Local path to the LM ``safetensors`` file or directory, or
            a HuggingFace Hub reference (``"owner/repo"`` or
            ``"owner/repo:model.safetensors"``).
        dtype: Override model dtype (e.g. ``"f16"``). When ``None``, float32.
        execution_provider: Target execution provider for EP-aware
            optimisations. Defaults to ``"default"``.
        revision: Optional HuggingFace Hub revision to pin downloads.
        load_weights: Load and apply the native checkpoint payload.
        dep_q: Known depformer width. Required for graph-only construction.

    Returns:
        One package with ``temporal`` and ``depformer`` graph entries.
    """
    from mobius._builder import build_from_module, resolve_dtype
    from mobius.models.moshi import (
        MoshiDepformerModel,
        MoshiTemporalModel,
        _moshi_depformer_config,
        _moshi_temporal_config,
    )
    from mobius.tasks import MoshiDepformerTask, MoshiTemporalTask

    state_dict = None
    if load_weights:
        from safetensors.torch import load_file

        ckpt_path = _resolve_lm_checkpoint(checkpoint, revision=revision)
        logger.info("Loading Moshi LM checkpoint: %s", ckpt_path)
        state_dict = load_file(ckpt_path)
        logger.info("Loaded %d Moshi LM parameters", len(state_dict))
        detected_dep_q = sum(
            key.startswith("depformer_in.") and key.endswith(".weight") for key in state_dict
        )
        if dep_q is not None and detected_dep_q != dep_q:
            raise ValueError(f"Expected {dep_q} Moshi depformer steps, found {detected_dep_q}")
        dep_q = detected_dep_q
    if dep_q is None:
        raise ValueError("dep_q is required when building Moshi without checkpoint weights")
    if dep_q not in (8, 16):
        raise ValueError(
            "Unsupported Moshi depformer step count inferred from checkpoint: "
            f"{dep_q}; expected public Moshi/Moshiko (8) or PersonaPlex (16)"
        )
    logger.info("Detected %d Moshi depformer codebook steps", dep_q)

    resolved = resolve_dtype(dtype) if dtype is not None else None

    def _build(config, model_cls, task):
        if resolved is not None and resolved != config.dtype:
            config = dataclasses.replace(config, dtype=resolved)
        module = model_cls(config)
        pkg = build_from_module(module, config, task, execution_provider=execution_provider)
        if state_dict is not None:
            pkg.apply_weights(module.preprocess_weights(state_dict))
        return pkg

    temporal = _build(_moshi_temporal_config(), MoshiTemporalModel, MoshiTemporalTask())
    depformer = _build(
        _moshi_depformer_config(dep_q=dep_q), MoshiDepformerModel, MoshiDepformerTask()
    )
    return ModelPackage(
        {"temporal": temporal["model"], "depformer": depformer["model"]},
        config=temporal.config,
    )
