# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Safe, bounded CLM projection checkpoint loading; never load the backbone."""

from __future__ import annotations

import dataclasses
import hashlib
import re
from pathlib import Path

import torch

from mobius._builder import build_from_module, resolve_dtype
from mobius._configs.clm import CLMConfig
from mobius._model_package import ModelPackage
from mobius.models.clm import CLMRankingModel

CLM_MODEL_ID = "Contrastive-LM/CLM-v0.1-8B"
CLM_REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"
CLM_SOURCE_REVISION = "bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
CLM_CHECKPOINT = "CLM_v0.1-8B.pt"
CLM_CHECKPOINT_SHA256 = "b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5"


def load_clm_checkpoint(
    path: str | Path, config: CLMConfig
) -> tuple[CLMRankingModel, CLMConfig, dict[str, torch.Tensor]]:
    """Validate a weights-only checkpoint before binding any graph weights."""
    path = Path(path)
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("CLM head checkpoint exceeds the 256 MiB loading limit")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {
        "cfg",
        "state_head",
        "action_head",
        "logit_scale",
    }.issubset(checkpoint):
        raise ValueError("Malformed CLM checkpoint")
    cfg = checkpoint["cfg"]
    if not isinstance(cfg, dict):
        raise TypeError("CLM checkpoint cfg must be a dictionary")
    if not {"width", "depth"}.issubset(cfg):
        raise ValueError("CLM checkpoint cfg must specify width and depth")
    if cfg.get("hidden_size", 4096) != config.hidden_size:
        raise ValueError("CLM checkpoint encoder width disagrees with config")
    if checkpoint.get("hidden_size", config.hidden_size) != config.hidden_size:
        raise ValueError("CLM checkpoint hidden_size disagrees with config")
    if cfg.get("model", config.base_model) != config.base_model:
        raise ValueError("CLM checkpoint base model disagrees with config")
    projection_dim = checkpoint.get("projection_dim", cfg.get("projection_dim", 512))
    if not isinstance(projection_dim, int) or isinstance(projection_dim, bool):
        raise TypeError("CLM projection_dim must be a positive integer")
    if cfg.get("projection_dim", projection_dim) != projection_dim:
        raise ValueError("CLM checkpoint projection dimensions disagree")
    scale = checkpoint["logit_scale"]
    if (
        not isinstance(scale, torch.Tensor)
        or scale.numel() != 1
        or not scale.is_floating_point()
        or not torch.isfinite(scale).all()
    ):
        raise ValueError("CLM logit_scale must be one finite floating tensor")
    # Upstream computes float32 exp then clamps to 100, including large logs.
    scale_value = float(scale.float().exp().clamp(max=100))
    config = dataclasses.replace(
        config,
        width=cfg["width"],
        depth=cfg["depth"],
        projection_dim=projection_dim,
        activation=cfg.get("activation", "gelu"),
        layernorm=cfg.get("layernorm", False),
        residual=cfg.get("residual", False),
        scale=scale_value,
    )
    config.validate()
    # Bound module construction as well as file I/O for malformed topology metadata.
    block_params = config.width * config.width + config.width
    if config.layernorm:
        block_params += 2 * config.width
    head_params = (
        config.hidden_size * config.width
        + config.width
        + (config.depth - 2) * block_params
        + config.width * config.projection_dim
        + config.projection_dim
    )
    if 2 * head_params > 64 * 1024 * 1024:
        raise ValueError("CLM topology exceeds the 256 MiB float32 parameter limit")
    weights: dict[str, torch.Tensor] = {}
    head_shapes = {
        "inp.weight": (config.width, config.hidden_size),
        "inp.bias": (config.width,),
        "out.weight": (config.projection_dim, config.width),
        "out.bias": (config.projection_dim,),
    }
    for index in range(config.depth - 2):
        head_shapes[f"hidden.{index}.weight"] = (config.width, config.width)
        head_shapes[f"hidden.{index}.bias"] = (config.width,)
        if config.layernorm:
            head_shapes[f"norms.{index}.weight"] = (config.width,)
            head_shapes[f"norms.{index}.bias"] = (config.width,)
    expected = {
        f"{head}.{name}": shape
        for head in ("state_head", "action_head")
        for name, shape in head_shapes.items()
    }
    for head in ("state_head", "action_head"):
        values = checkpoint[head]
        if not isinstance(values, dict):
            raise TypeError(f"CLM {head} must be a tensor dictionary")
        for name, tensor in values.items():
            key = f"{head}.{name}"
            if (
                key not in expected
                or not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tuple(tensor.shape) != expected[key]
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid CLM head tensor: {key}")
            weights[key] = tensor
    if weights.keys() != expected.keys():
        raise ValueError("CLM checkpoint is missing head tensors")
    return CLMRankingModel(config), config, weights


def build_clm_heads(
    model_id: str,
    hf_config,
    *,
    revision: str | None,
    dtype,
    execution_provider: str,
    load_weights: bool,
) -> ModelPackage:
    """Build the ordinary ModelPackage using only the pinned head checkpoint."""
    from huggingface_hub import hf_hub_download

    config = CLMConfig.from_transformers(hf_config)
    config = dataclasses.replace(config, dtype=resolve_dtype(dtype) or config.dtype)
    config.validate()
    local = Path(model_id).is_dir()
    if not local and (revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None):
        raise ValueError("CLM head downloads require an immutable 40-character revision")
    path = (
        Path(model_id) / CLM_CHECKPOINT
        if local
        else Path(hf_hub_download(model_id, CLM_CHECKPOINT, revision=revision))
    )
    with path.open("rb") as stream:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
    if (
        model_id == CLM_MODEL_ID
        and revision == CLM_REVISION
        and digest != CLM_CHECKPOINT_SHA256
    ):
        raise ValueError("Pinned CLM checkpoint SHA256 mismatch")
    module, config, weights = load_clm_checkpoint(path, config)
    package = build_from_module(
        module, config, task="contrastive-ranking-heads", execution_provider=execution_provider
    )
    if load_weights:
        package.apply_weights(weights)
    model = package["model"]
    model.metadata_props.update(
        {
            "mobius.export_scope": "projection-head-only",
            "mobius.source_revision": revision or "local",
            "mobius.reference_source_revision": CLM_SOURCE_REVISION,
            "mobius.checkpoint_sha256": digest,
            "mobius.encoder_contract": "Qwen/Qwen3-8B; last-token; L2-normalized; external",
            "mobius.temperature": "1",
        }
    )
    return package
