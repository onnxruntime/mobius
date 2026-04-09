# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

# SECURITY: Do NOT use torch.load() or pickle deserialization anywhere in this
# module.  Only safetensors is permitted for weight loading to prevent arbitrary
# code execution from untrusted weight files.

"""Weight loading and application for ONNX models.

This module handles downloading model weights from HuggingFace Hub and
applying them to ONNX IR models. All weight loading uses the safetensors
format exclusively — no ``torch.load`` or pickle deserialization is used,
eliminating arbitrary code execution risks from untrusted weight files.
"""

from __future__ import annotations

__all__ = [
    "apply_weights",
]

import concurrent.futures
import json
import logging

import onnx_ir as ir
import torch
import tqdm
from onnx_ir import tensor_adapters

logger = logging.getLogger(__name__)


def _assign_weight(
    initializer: ir.Value,
    tensor: torch.Tensor,
    name: str,
) -> None:
    """Assign a weight tensor to an initializer with shape/dtype handling.

    This is the single source of truth for weight assignment logic:

    * **Shape mismatch error** — raises :class:`ValueError` when the
      tensor shape differs from the initializer shape.
    * **Lazy dtype cast** — when the tensor dtype differs from the
      initializer's declared ONNX type, wraps the tensor in
      ``ir.LazyTensor`` so the cast happens at serialization time,
      avoiding eager memory allocation.
    """
    # Raise on shape mismatch (initializers always have concrete int dims).
    init_shape = initializer.shape
    if init_shape is not None:
        expected = list(init_shape)
        actual = list(tensor.shape)
        if expected != actual:
            raise ValueError(
                f"Weight shape mismatch for '{name}': model expects {expected}, got {actual}"
            )

    onnx_dtype = initializer.dtype
    assert onnx_dtype is not None
    target_dtype = tensor_adapters.to_torch_dtype(onnx_dtype)

    if tensor.dtype != target_dtype:

        def tensor_func(t=tensor, dt=target_dtype, n=name) -> tensor_adapters.TorchTensor:
            return tensor_adapters.TorchTensor(t.to(dt), name=n)

        ir_tensor = ir.LazyTensor(
            tensor_func,
            dtype=onnx_dtype,
            shape=ir.Shape(tensor.shape),
            name=name,
        )
    else:
        ir_tensor = tensor_adapters.TorchTensor(tensor, name)
    initializer.const_value = ir_tensor


def apply_weights(model: ir.Model, state_dict: dict[str, torch.Tensor]) -> None:
    """Apply weights from a state dict to an ONNX model.

    Assigns each tensor in *state_dict* to the matching initializer in the
    model.  After all weights are assigned,
    :func:`~mobius._optimizations.fold_initializers_after_weights` is called to
    materialise any deferred folded initializers (created by stage 5 of
    :func:`~mobius._optimizations.optimize_model`) and remove the now-redundant
    source initializers.

    Stage 5 of :func:`~mobius._optimizations.optimize_model` folds structural
    ``Transpose`` and ``Concat`` nodes and removes the original source
    initializers from the graph.  Weights for those removed initializers still
    arrive in *state_dict* but are no longer present in
    ``model.graph.initializers``.  This function detects that case by inspecting
    ``pkg.mobius.fold_source`` / ``pkg.mobius.fold_sources`` metadata on folded
    initializers and computes the deferred tensor values directly from the state
    dict rather than requiring the intermediate initializers to be in the graph.

    Args:
        model: The ONNX IR model.
        state_dict: Mapping of parameter names to torch tensors.
    """
    import numpy as np

    from mobius._optimizations import fold_initializers_after_weights

    # Collect source names that were folded away by stage 5 so we can suppress
    # false-positive "weight not found" warnings for those entries.
    folded_sources: set[str] = set()
    for init in model.graph.initializers.values():
        fold_source = init.metadata_props.get("pkg.mobius.fold_source")
        if fold_source:
            folded_sources.add(fold_source)
        fold_sources_str = init.metadata_props.get("pkg.mobius.fold_sources")
        if fold_sources_str:
            folded_sources.update(fold_sources_str.split(","))

    # Step 1: Assign weights that are directly present in the graph.
    for name, tensor in state_dict.items():
        if name in model.graph.initializers:
            _assign_weight(model.graph.initializers[name], tensor, name)
        elif name not in folded_sources:
            logger.warning(
                "Weight '%s' not found in the model. Skipped applying.",
                name,
            )

    # Step 2: Populate deferred folded initializers whose source initializers
    # were removed from the graph by stage 5's RemoveUnusedNodesPass.
    # This covers two cases:
    #   a) Transposed concat — init has both pkg.mobius.fold_source and pkg.mobius.fold_sources:
    #      the value is np.concatenate(W_q, W_k, W_v, axis).T
    #   b) Pure concat — init has only pkg.mobius.fold_sources:
    #      the value is np.concatenate(bias_q, bias_k, bias_v, axis)
    for init in model.graph.initializers.values():
        if init.const_value is not None:
            continue  # already set by step 1

        fold_source = init.metadata_props.get("pkg.mobius.fold_source")
        fold_sources_str = init.metadata_props.get("pkg.mobius.fold_sources")

        if fold_sources_str is None:
            continue  # no packed-sources metadata; handled by _materialize_deferred

        source_names = fold_sources_str.split(",")
        axis = int(init.metadata_props.get("pkg.mobius.fold_axis", "0"))

        # Only apply this path when the sources were removed from the graph
        # (i.e. stage 5 + RemoveUnused pruned them).  If they are still present,
        # _materialize_deferred_initializers will handle them after step 1.
        if any(n in model.graph.initializers for n in source_names):
            continue

        if not all(n in state_dict for n in source_names):
            continue  # cannot compute; leave for materialise to handle or warn

        tensors = [state_dict[n] for n in source_names]
        onnx_dtype = init.dtype or ir.DataType.FLOAT
        target_dtype = tensor_adapters.to_torch_dtype(onnx_dtype)
        captured_axis = axis

        if fold_source is not None:
            # Transposed concat: concatenate along axis then transpose.
            init.const_value = ir.LazyTensor(
                lambda ts=tensors, ax=captured_axis, dt=target_dtype: ir.tensor(
                    np.concatenate([t.to(dt).numpy() for t in ts], axis=ax).T
                ),
                dtype=onnx_dtype,
                shape=init.shape,
                name=init.name,
            )
        else:
            # Pure concat: concatenate along axis.
            init.const_value = ir.LazyTensor(
                lambda ts=tensors, ax=captured_axis, dt=target_dtype: ir.tensor(
                    np.concatenate([t.to(dt).numpy() for t in ts], axis=ax)
                ),
                dtype=onnx_dtype,
                shape=init.shape,
                name=init.name,
            )

    fold_initializers_after_weights(model)


def _parallel_download(
    model_id: str, filenames: list[str], *, desc: str = "files"
) -> list[str]:
    """Download files from HuggingFace Hub in parallel.

    Uses a thread pool to download multiple safetensors shards
    concurrently, similar to how ``transformers`` handles sharded
    checkpoints.

    Args:
        model_id: HuggingFace model identifier.
        filenames: List of filenames to download.
        desc: Description for the progress bar.

    Returns:
        List of local file paths in the same order as *filenames*.
    """
    from huggingface_hub import hf_hub_download

    if len(filenames) <= 1:
        # No benefit from parallelism for a single file
        return [hf_hub_download(repo_id=model_id, filename=f) for f in filenames]

    print(f"Downloading {len(filenames)} {desc} files (parallel)...")
    path_map: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(hf_hub_download, repo_id=model_id, filename=f): f
            for f in filenames
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Downloading {desc}",
        ):
            fname = futures[future]
            path_map[fname] = future.result()

    # Return paths in original order
    return [path_map[f] for f in filenames]


def _download_weights(model_id: str) -> dict[str, torch.Tensor]:
    """Download weights from HuggingFace and return as a state dict.

    Uses parallel downloads when multiple safetensors shards exist.
    """
    import safetensors.torch
    from huggingface_hub import hf_hub_download

    try:
        index_path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors.index.json",
        )
        with open(index_path) as f:
            index = json.load(f)
        all_files = sorted(set(index["weight_map"].values()))
    except Exception as e:
        if "Entry Not Found" in str(e):
            all_files = ["model.safetensors"]
        else:
            raise

    paths = _parallel_download(model_id, all_files, desc="safetensors")

    state_dict: dict[str, torch.Tensor] = {}
    for path in tqdm.tqdm(paths, desc="Loading weights"):
        state_dict.update(safetensors.torch.load_file(path))
    return state_dict
