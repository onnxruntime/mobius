# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned configuration and checkpoint builder for MatterGen score-core exports."""

from __future__ import annotations

__all__ = ["build_mattergen", "is_mattergen_checkpoint"]

import dataclasses
from pathlib import Path

import onnx_ir as ir
import yaml
from huggingface_hub import hf_hub_download

from mobius._builder import build_from_module, resolve_dtype
from mobius._model_package import ModelPackage
from mobius.integrations.mattergen._configs import MatterGenConfig
from mobius.integrations.mattergen._contract import (
    MATTERGEN_HUB_ID,
    MATTERGEN_HUB_REVISION,
    OFFICIAL_CHECKPOINT_CONDITIONS,
)
from mobius.integrations.mattergen._weights import apply_mattergen_checkpoint
from mobius.models.mattergen import MatterGenModel
from mobius.tasks import MatterGenScoreTask


def _validate_checkpoint_family(family: str) -> str:
    """Return a declared official family or reject a path-like/checkpoint typo."""
    if family not in OFFICIAL_CHECKPOINT_CONDITIONS:
        options = ", ".join(sorted(OFFICIAL_CHECKPOINT_CONDITIONS))
        raise ValueError(f"Unknown MatterGen checkpoint family {family!r}. Available: {options}.")
    return family


def _local_checkpoint_file(root: Path, family: str, name: str) -> Path:
    """Resolve one local checkpoint artifact without following a link outside *root*."""
    candidate = root / "checkpoints" / family / name
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(f"MatterGen local artifact must be a regular file: {candidate}")
    resolved = candidate.resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError(f"MatterGen local artifact escapes its checkpoint root: {candidate}")
    return resolved


def is_mattergen_checkpoint(source: str | Path) -> bool:
    """Return whether *source* is the canonical Hub ID or a MatterGen directory."""
    if str(source) == MATTERGEN_HUB_ID:
        return True
    root = Path(source).expanduser()
    if not root.is_dir() or root.is_symlink():
        return False
    return any(
        (root / "checkpoints" / family / "config.yaml").is_file()
        for family in OFFICIAL_CHECKPOINT_CONDITIONS
    )


def _load_config(
    source: str | Path,
    family: str,
    revision: str | None,
    *,
    load_weights: bool,
) -> tuple[dict[object, object], Path | None, str]:
    """Load the family Hydra YAML from a validated local root or immutable Hub revision."""
    source_path = Path(source).expanduser()
    if source_path.is_dir():
        if revision is not None:
            raise ValueError("MatterGen local checkpoint directories cannot use --revision.")
        root = source_path.resolve(strict=True)
        if root.is_symlink():
            raise ValueError("MatterGen local checkpoint root must not be a symlink.")
        config_path = _local_checkpoint_file(root, family, "config.yaml")
        checkpoint_path = (
            _local_checkpoint_file(root, family, "checkpoints/last.ckpt")
            if load_weights
            else None
        )
        effective_revision = "local"
    else:
        if str(source) != MATTERGEN_HUB_ID:
            raise ValueError(
                f"MatterGen exports only support {MATTERGEN_HUB_ID!r} or a local checkpoint root."
            )
        effective_revision = MATTERGEN_HUB_REVISION if revision is None else revision
        if effective_revision != MATTERGEN_HUB_REVISION:
            raise ValueError(
                "MatterGen requires the pinned Hub revision "
                f"{MATTERGEN_HUB_REVISION}; got {effective_revision!r}."
            )
        config_path = Path(
            hf_hub_download(
                repo_id=MATTERGEN_HUB_ID,
                filename=f"checkpoints/{family}/config.yaml",
                revision=effective_revision,
            )
        )
        checkpoint_path = None

    with config_path.open(encoding="utf-8") as file:
        parsed = yaml.safe_load(file)
    if not isinstance(parsed, dict):
        raise TypeError(f"MatterGen Hydra config must be a mapping: {config_path}")
    return parsed, checkpoint_path, effective_revision


def build_mattergen(
    source: str | Path = MATTERGEN_HUB_ID,
    *,
    checkpoint: str = "mp_20_base",
    revision: str | None = None,
    dtype: str | ir.DataType | None = None,
    load_weights: bool = True,
    execution_provider: str = "default",
) -> ModelPackage:
    """Build one official MatterGen GemNet-T score core from its pinned Hydra config.

    ``source`` is either the immutable official Hub repository or a local
    checkpoint root containing ``checkpoints/<family>/config.yaml`` and, when
    weights are requested, ``checkpoints/<family>/checkpoints/last.ckpt``.
    The model intentionally exports neither MatterGen's dynamic periodic graph
    construction nor its stochastic crystal sampling host loop.
    """
    family = _validate_checkpoint_family(checkpoint)
    resolved_dtype = resolve_dtype(dtype)
    if resolved_dtype not in {None, ir.DataType.FLOAT}:
        raise ValueError(
            "MatterGen score-core export is assessed only for float32; f16 and bf16 are "
            "refused rather than changing the source float32 numerical contract."
        )
    if execution_provider not in {"default", "cpu"}:
        raise ValueError(
            "MatterGen score-core export is currently assessed only for default/CPU ONNX; "
            f"execution provider {execution_provider!r} is not supported."
        )

    hydra_config, local_checkpoint, effective_revision = _load_config(
        source,
        family,
        revision,
        load_weights=load_weights,
    )
    config = MatterGenConfig.from_hydra_config(
        hydra_config,
        model_id=str(source),
        revision=effective_revision,
        variant=family,
    )
    actual_conditions = tuple(spec.name for spec in config.condition_input_specs)
    expected_conditions = OFFICIAL_CHECKPOINT_CONDITIONS[family]
    if actual_conditions != expected_conditions:
        raise ValueError(
            f"MatterGen checkpoint family {family!r} declares condition inputs "
            f"{actual_conditions!r}; expected the pinned contract {expected_conditions!r}."
        )
    if resolved_dtype is not None:
        config = dataclasses.replace(config, dtype=resolved_dtype)
    config.validate()

    module = MatterGenModel(config)
    package = build_from_module(
        module,
        config,
        task=MatterGenScoreTask(),
        execution_provider=execution_provider,
    )
    package["model"].graph.name = f"{source}/{family}/model"
    if load_weights:
        checkpoint_path = local_checkpoint
        if checkpoint_path is None:
            checkpoint_path = Path(
                hf_hub_download(
                    repo_id=MATTERGEN_HUB_ID,
                    filename=f"checkpoints/{family}/checkpoints/last.ckpt",
                    revision=effective_revision,
                )
            )
        apply_mattergen_checkpoint(package, module, checkpoint_path)
    return package
