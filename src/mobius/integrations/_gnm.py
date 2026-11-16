# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native, pickle-free Google GNM NPZ checkpoint importer."""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import onnx_ir as ir
import torch

from mobius._builder import build_from_module, resolve_dtype
from mobius.models.gnm import (
    GNM_MODEL_ID,
    GNM_REVISION,
    GNM_SHA256,
    GNMConfig,
    GNMHeadModel,
)


def gnm_archive_path(model_id: str, revision: str | None = None) -> pathlib.Path:
    """Resolve and integrity-check the native archive without executing remote code."""
    from huggingface_hub import hf_hub_download

    local = pathlib.Path(model_id)
    is_local = local.is_file()
    effective_revision = revision or GNM_REVISION
    if model_id == GNM_MODEL_ID and effective_revision != GNM_REVISION:
        raise ValueError(
            f"GNM is qualified only at revision {GNM_REVISION}; "
            "omit revision or pass that immutable release SHA."
        )
    path = (
        local
        if is_local
        else pathlib.Path(
            hf_hub_download(
                model_id,
                "v3_0/gnm_head.npz",
                revision=effective_revision,
            )
        )
    )
    if not is_local and model_id == GNM_MODEL_ID and effective_revision == GNM_REVISION:
        if hashlib.sha256(path.read_bytes()).hexdigest() != GNM_SHA256:
            raise ValueError(
                "GNM checkpoint SHA-256 differs from the pinned release; "
                "remove the cached archive and download it again."
            )
    return path


def read_gnm_config(archive) -> GNMConfig:
    """Read native dimensions and skeleton topology without synthesizing HF fields."""
    required = {
        "version",
        "variant",
        "template_vertex_positions",
        "template_joint_positions",
        "vertex_identity_basis",
        "joint_identity_basis",
        "expression_basis",
        "joint_parent_indices",
        "pose_correctives_regressor",
        "skinning_weights",
    }
    if missing := required - set(archive):
        raise ValueError(
            f"GNM archive lacks required arrays {sorted(missing)}; "
            "download an unmodified v3.0 head archive."
        )
    if str(archive["version"]) != "3.0" or str(archive["variant"]) != "head":
        raise ValueError("Expected a GNM 3.0 head archive; use google/gnm-v3.")
    parents = archive["joint_parent_indices"]
    if parents.ndim != 1 or parents.dtype.kind not in "iu":
        raise ValueError(
            "GNM joint_parent_indices must be a one-dimensional integer array; "
            "download an unmodified v3.0 head archive."
        )
    config = GNMConfig(
        num_vertices=archive["template_vertex_positions"].shape[0],
        identity_dim=archive["vertex_identity_basis"].shape[0],
        expression_dim=archive["expression_basis"].shape[0],
        joint_parent_indices=tuple(int(i) for i in parents),
    )
    config.validate()
    return config


def build_gnm(
    model_id: str = GNM_MODEL_ID,
    *,
    revision: str | None = None,
    dtype: str | ir.DataType | None = None,
    execution_provider: str = "default",
    load_weights: bool = True,
):
    """Build the complete GNM head from a pinned Hub archive or local NPZ."""
    from mobius.tasks import MeshGenerationTask

    resolved_dtype = resolve_dtype(dtype) or ir.DataType.FLOAT
    if resolved_dtype != ir.DataType.FLOAT:
        raise ValueError("GNM supports float32 only; select dtype='f32'.")
    path = gnm_archive_path(model_id, revision)
    is_local = pathlib.Path(model_id).is_file()
    with np.load(path, allow_pickle=False) as archive:
        config = read_gnm_config(archive)
        module = GNMHeadModel(config)
        package = build_from_module(
            module, config, MeshGenerationTask(), execution_provider=execution_provider
        )
        if load_weights:
            weights = {}
            for name, parameter in package["model"].graph.initializers.items():
                if parameter.const_value is not None:
                    continue
                value = archive[name]
                expected = tuple(parameter.shape)
                if value.dtype != np.float32 or value.shape != expected:
                    raise ValueError(
                        f"GNM tensor {name!r} has {value.dtype} {value.shape}; "
                        f"expected float32 {expected}. Use an unmodified v3.0 head archive."
                    )
                if not np.isfinite(value).all():
                    raise ValueError(
                        f"GNM tensor {name!r} contains nonfinite values; "
                        "download an unmodified v3.0 head archive."
                    )
                weights[name] = torch.from_numpy(np.array(value, copy=True))
            package.apply_weights(weights)
    metadata = package["model"].metadata_props
    metadata["mobius.source_revision"] = "local" if is_local else revision or GNM_REVISION
    metadata["mobius.task"] = "mesh-generation"
    return package
