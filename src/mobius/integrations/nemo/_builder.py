# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NeMo ``.nemo`` → ONNX build pipeline.

Converts a NeMo ``.nemo`` archive to ONNX using the standard
``build_from_module`` pipeline:

1. Read ``model_config.yaml`` and map it to an :class:`ArchitectureConfig`.
2. Look up the model class + task from the registry by NeMo ``target``.
3. Build the ONNX graph(s).
4. Load the ``model_weights.ckpt`` ``state_dict``.
5. Run ``preprocess_weights`` (NeMo → ONNX name mapping) and apply weights.
"""

from __future__ import annotations

__all__ = ["build_from_nemo"]

import logging
from pathlib import Path

from mobius._model_package import ModelPackage

logger = logging.getLogger(__name__)


def build_from_nemo(
    nemo_path: str | Path,
    *,
    task: str | None = None,
    dtype: str | None = None,
    execution_provider: str = "default",
    revision: str | None = None,
) -> ModelPackage:
    """Build an ONNX :class:`ModelPackage` from a NeMo ``.nemo`` archive.

    Args:
        nemo_path: Path to a local ``.nemo`` file, or a HuggingFace Hub
            reference (``"owner/repo"`` or ``"owner/repo:filename.nemo"``).
        task: Override the model task. When ``None``, the task is the model
            class's ``default_task``.
        dtype: Override model dtype (e.g. ``"f16"``). When ``None``, defaults
            to float32.
        execution_provider: Target execution provider for EP-aware
            optimisations. Defaults to ``"default"`` (portable, no vendor
            fusions).
        revision: Optional HuggingFace Hub revision (branch, tag, or commit
            SHA) to pin downloads. Ignored for local ``.nemo`` paths.

    Returns:
        A :class:`ModelPackage` containing the built model(s).

    Raises:
        FileNotFoundError: If the ``.nemo`` file does not exist.
        KeyError: If the NeMo ``target`` is not supported / registered.
    """
    import dataclasses

    from mobius._builder import build_from_module, resolve_dtype
    from mobius._config_resolver import _default_task_for_model
    from mobius._registry import registry
    from mobius.integrations.nemo._config_mapping import nemo_to_config
    from mobius.integrations.nemo._reader import NeMoArchive

    archive = NeMoArchive(nemo_path, revision=revision)
    logger.info("Loaded NeMo archive: %s (target=%s)", archive.path, archive.target)

    config = nemo_to_config(archive.config)
    model_type = config.model_type

    if dtype is not None:
        resolved = resolve_dtype(dtype)
        if resolved is not None and resolved != config.dtype:
            # f16/bf16: float32-only ops (positional-encoding Sin/Cos) stay in
            # f32 and are cast to the compute dtype inside the model; all other
            # tensors follow config.dtype.
            config = dataclasses.replace(config, dtype=resolved)

    module_class = registry.get(model_type)
    if task is None:
        task = _default_task_for_model(model_type)

    module = module_class(config)
    pkg = build_from_module(module, config, task, execution_provider=execution_provider)
    logger.info("Built ONNX graph for %s (%d components)", model_type, len(pkg))

    state_dict = archive.state_dict()
    logger.info("Loaded %d parameters from model_weights.ckpt", len(state_dict))

    if hasattr(module, "preprocess_weights"):
        state_dict = module.preprocess_weights(state_dict)

    prefix_map = getattr(module, "weight_prefix_map", None)
    pkg.apply_weights(state_dict, prefix_map=prefix_map)
    return pkg
