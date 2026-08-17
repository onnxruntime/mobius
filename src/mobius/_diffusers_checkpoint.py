# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Read-only access to diffusers-style multi-component checkpoints.

A diffusers checkpoint is a directory (local or on the Hub) with a
``model_index.json`` at the root and one subdirectory per component, each
holding its own ``config.json`` and safetensors shards. World-model exporters
need to inspect such a checkpoint before building anything: which components
exist, what class each one declares, which tensors a component ships, and
which auxiliary files must travel with the exported package.

Every helper here is model-agnostic — it resolves and reads files without
interpreting their contents. Local paths are resolved with traversal
protection so a malicious index or component name cannot read outside the
checkpoint directory.
"""

from __future__ import annotations

__all__ = [
    "component_class",
    "component_shard_paths",
    "component_weight_names",
    "load_checkpoint_json",
    "load_component_weights",
    "load_optional_checkpoint_json",
    "resolve_assets",
    "resolve_checkpoint_file",
]

import json
import pathlib
from collections.abc import Iterable, Mapping
from typing import Any

import safetensors.torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors import safe_open

from mobius._diffusers_builder import _download_diffusers_component_weights
from mobius._weight_loading import _dequantize_fp8_weights

#: Weight-index basenames used by diffusers and transformers components.
_INDEX_NAMES: tuple[str, ...] = (
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
)
#: Single-shard weight filenames used by diffusers and transformers components.
_SINGLE_NAMES: tuple[str, ...] = (
    "diffusion_pytorch_model.safetensors",
    "model.safetensors",
)


def _local_component_directory(
    model_dir: pathlib.Path,
    component: str,
) -> pathlib.Path:
    """Resolve a component directory without allowing checkpoint-root escape."""
    root = model_dir.resolve()
    component_dir = (root / component).resolve()
    try:
        component_dir.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Unsafe component path {component!r}") from error
    return component_dir


def resolve_checkpoint_file(
    model_id: str,
    filename: str,
    *,
    required: bool = True,
) -> str | None:
    """Resolve one local-or-Hub checkpoint file without interpreting it.

    Args:
        model_id: Local checkpoint directory or Hub repository id.
        filename: ``/``-separated path relative to the checkpoint root.
        required: Whether a missing file is an error.

    Returns:
        An absolute local path, or ``None`` when an optional file is absent.

    Raises:
        ValueError: If *filename* resolves outside a local checkpoint.
        FileNotFoundError: If a required local file is missing.
    """
    root = pathlib.Path(model_id)
    if root.is_dir():
        path = (root / pathlib.PurePosixPath(filename)).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(
                f"Checkpoint file escapes model directory: {filename!r}"
            ) from error
        if path.is_file():
            return str(path)
        if required:
            raise FileNotFoundError(f"Required checkpoint file not found: {path}")
        return None
    try:
        return hf_hub_download(repo_id=model_id, filename=filename)
    except EntryNotFoundError:
        if required:
            raise
        return None


def load_checkpoint_json(model_id: str, filename: str) -> tuple[dict[str, Any], str]:
    """Load a required checkpoint JSON object and return it with its path."""
    path = resolve_checkpoint_file(model_id, filename)
    assert path is not None
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{filename!r} must contain a JSON object")
    return value, path


def load_optional_checkpoint_json(model_id: str, filename: str) -> dict[str, Any]:
    """Load an optional checkpoint JSON object, or ``{}`` when it is absent."""
    path = resolve_checkpoint_file(model_id, filename, required=False)
    if path is None:
        return {}
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{filename!r} must contain a JSON object")
    return value


def component_class(
    pipeline_index: Mapping[str, Any],
    component: str,
) -> str | None:
    """Return the class a ``model_index.json`` declares for *component*.

    Diffusers records each component as a ``[library, class_name]`` pair and
    marks an absent component as ``[null, null]``.
    """
    info = pipeline_index.get(component)
    if info in (None, [None, None]):
        return None
    if not isinstance(info, list) or len(info) != 2 or not isinstance(info[1], str):
        raise ValueError(f"Invalid model_index.json entry for {component!r}: {info!r}")
    return info[1]


def component_weight_names(model_id: str, component: str) -> set[str]:
    """Read only safetensors metadata to determine component graph shape.

    Tensor names alone reveal which optional towers a component ships, so no
    tensor data is downloaded or read here.
    """
    root = pathlib.Path(model_id)
    if root.is_dir():
        component_dir = _local_component_directory(root, component)
        for name in _INDEX_NAMES:
            path = component_dir / name
            if path.is_file():
                with path.open(encoding="utf-8") as handle:
                    index = json.load(handle)
                return set(index["weight_map"])
        for name in _SINGLE_NAMES:
            path = component_dir / name
            if path.is_file():
                with safe_open(str(path), framework="pt", device="cpu") as file:
                    return set(file.keys())
        raise FileNotFoundError(
            f"No safetensors checkpoint found for component {component!r} in {model_id!r}"
        )

    for name in _INDEX_NAMES:
        filename = f"{component}/{name}"
        path = resolve_checkpoint_file(model_id, filename, required=False)
        if path is not None:
            with open(path, encoding="utf-8") as handle:
                index = json.load(handle)
            return set(index["weight_map"])
    api = HfApi()
    for name in _SINGLE_NAMES:
        filename = f"{component}/{name}"
        try:
            metadata = api.parse_safetensors_file_metadata(model_id, filename)
        except EntryNotFoundError:
            continue
        return set(metadata.tensors)
    raise FileNotFoundError(
        f"No safetensors checkpoint found for component {component!r} in {model_id!r}"
    )


def component_shard_paths(
    model_dir: pathlib.Path,
    component: str,
) -> list[pathlib.Path]:
    """Resolve local component shards with traversal protection."""
    component_dir = _local_component_directory(model_dir, component)

    for basename in (
        "diffusion_pytorch_model",
        "model",
    ):
        index_path = component_dir / f"{basename}.safetensors.index.json"
        if not index_path.is_file():
            continue
        with index_path.open(encoding="utf-8") as handle:
            index = json.load(handle)
        paths: list[pathlib.Path] = []
        for filename in sorted(set(index["weight_map"].values())):
            relative = pathlib.PurePosixPath(str(filename).replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe component weight filename: {filename!r}")
            path = (component_dir / relative).resolve()
            try:
                path.relative_to(component_dir)
            except ValueError as error:
                raise ValueError(
                    f"Component weight filename escapes its directory: {filename!r}"
                ) from error
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
        return paths

    for basename in _SINGLE_NAMES:
        path = component_dir / basename
        if path.is_file():
            return [path]
    raise FileNotFoundError(
        f"No safetensors checkpoint found for component {component!r} in {model_dir}"
    )


def load_component_weights(model_id: str, component: str) -> dict[str, Any]:
    """Load one diffusers component, supporting Hub and local directories."""
    root = pathlib.Path(model_id)
    if not root.is_dir():
        return _download_diffusers_component_weights(model_id, component)

    state_dict: dict[str, Any] = {}
    for path in component_shard_paths(root, component):
        state_dict.update(safetensors.torch.load_file(str(path)))
    return _dequantize_fp8_weights(state_dict)


def resolve_assets(
    model_id: str,
    candidates: Iterable[tuple[str, bool]],
) -> dict[str, tuple[str, bool]]:
    """Resolve runtime asset candidates to ``destination -> (source, required)``.

    Candidates are ``(relative path, required)`` pairs. Required files must
    exist; optional files are skipped when absent. Contents are never read:
    an asset is copied into the exported package verbatim. The caller owns the
    candidate list, so which files a model family ships stays with that family.
    """
    assets: dict[str, tuple[str, bool]] = {}
    for destination, required in candidates:
        source = resolve_checkpoint_file(model_id, destination, required=required)
        if source is not None:
            assets[destination] = (source, required)
    return assets
