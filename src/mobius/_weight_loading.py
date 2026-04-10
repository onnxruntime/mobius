# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
import safetensors.torch
import torch
import tqdm
from huggingface_hub import hf_hub_download
from onnx_ir import tensor_adapters

from mobius._optimizations import fold_initializers_after_weights

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
    model.  When two entries in *state_dict* are the **same Python object**
    (i.e. genuinely tied weights such as ``lm_head.weight`` /
    ``embed_tokens.weight``), the second initializer's value is redirected to
    share the first one — a single ONNX initializer is emitted instead of two
    identical copies.

    After all weights are assigned,
    :func:`~mobius._optimizations.fold_initializers_after_weights` is called to
    fold ``Transpose`` and ``Concat`` nodes over initializers into pre-computed
    weights and remove unused source initializers.

    Args:
        model: The ONNX IR model.
        state_dict: Mapping of parameter names to torch tensors.
    """
    # Map tensor id → the ir.Value of the first initializer assigned for that tensor.
    # Enables genuine weight sharing: if lm_head.weight IS embed_tokens.weight
    # (same Python object), the second initializer is merged into the first.
    # Prefer embed_tokens.weight as the canonical name regardless of state_dict
    # insertion order — this keeps initializer names consistent even when a
    # checkpoint supplies only lm_head.weight and tie_word_embeddings() adds
    # model.embed_tokens.weight afterwards (making lm_head appear first).
    _embed_suffix = "embed_tokens.weight"
    tensor_id_to_value: dict[int, ir.Value] = {}

    for name, tensor in state_dict.items():
        if name not in model.graph.initializers:
            logger.warning(
                "Weight '%s' not found in the model. Skipped applying.",
                name,
            )
            continue

        initializer = model.graph.initializers[name]
        tid = id(tensor)

        if tid in tensor_id_to_value:
            canonical = tensor_id_to_value[tid]
            # If the current initializer has the preferred embedding name but the
            # existing canonical does not, swap them so embedding is always canonical.
            if name.endswith(_embed_suffix) and not (canonical.name or "").endswith(
                _embed_suffix
            ):
                # Promote this initializer to canonical; demote the previous one.
                _assign_weight(initializer, tensor, name)
                canonical.replace_all_uses_with(initializer)
                del model.graph.initializers[canonical.name]
                tensor_id_to_value[tid] = initializer
                logger.debug(
                    "Weight tying: '%s' promoted to canonical (was '%s')",
                    name,
                    canonical.name,
                )
            else:
                # Redirect all graph uses of this initializer to the canonical one,
                # then delete this initializer — genuine single-copy weight sharing.
                initializer.replace_all_uses_with(canonical)
                del model.graph.initializers[name]
                logger.debug(
                    "Weight tying: '%s' shares initializer with '%s'",
                    name,
                    canonical.name,
                )
        else:
            _assign_weight(initializer, tensor, name)
            tensor_id_to_value[tid] = initializer

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


def _dequantize_fp8_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Dequantize FP8 weights and return a new dict with float tensors.

    Some HuggingFace checkpoints (e.g. Ministral-3-3B) store linear layer
    weights as float8_e4m3fn with a scalar ``weight_scale_inv`` tensor.
    The real weight value is ``fp8_weight.to(bfloat16) * weight_scale_inv``.

    Dequantization targets bfloat16 because that is the native training
    dtype for FP8-quantized checkpoints — the FP8 values represent
    bfloat16 values scaled into the FP8 range.

    This function detects FP8 tensors, applies the scale, and removes the
    auxiliary ``weight_scale_inv``, ``activation_scale``, and
    ``input_scale`` tensors.

    Returns:
        A new dict with FP8 weights dequantized to bfloat16 and the
        auxiliary scale tensors removed.  Always returns a new dict,
        even when no FP8 weights are found.
    """
    fp8_dtypes = {torch.float8_e4m3fn, torch.float8_e5m2}
    fp8_keys = [k for k, v in state_dict.items() if v.dtype in fp8_dtypes]
    if not fp8_keys:
        return dict(state_dict)

    # Work on a copy to avoid mutating the caller's dict
    result = dict(state_dict)

    logger.info("Dequantizing %d FP8 weights", len(fp8_keys))
    for key in fp8_keys:
        # Derive scale key from weight key using suffix replacement to avoid
        # replacing '.weight' substrings that appear in the middle of the key
        # (e.g. 'model.weight_proj.weight' → 'model.weight_proj.weight_scale_inv').
        if key.endswith(".weight"):
            scale_key = key[: -len(".weight")] + ".weight_scale_inv"
        else:
            scale_key = key + "_scale_inv"

        if scale_key in result:
            # Cast scale to bfloat16 to guarantee the output dtype is bfloat16,
            # even when weight_scale_inv is stored as FP32 in the checkpoint.
            scale = result[scale_key].to(torch.bfloat16)
            result[key] = result[key].to(torch.bfloat16) * scale
        else:
            logger.warning("FP8 weight '%s' has no scale_inv — casting without scaling", key)
            result[key] = result[key].to(torch.bfloat16)

    # Remove auxiliary FP8 tensors (not needed in the ONNX graph)
    aux_suffixes = (".weight_scale_inv", ".activation_scale", ".input_scale")
    return {k: v for k, v in result.items() if not any(k.endswith(s) for s in aux_suffixes)}


def _download_weights(model_id: str) -> dict[str, torch.Tensor]:
    """Download weights from HuggingFace and return as a state dict.

    Uses parallel downloads when multiple safetensors shards exist.
    """
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

    state_dict = _dequantize_fp8_weights(state_dict)
    return state_dict
