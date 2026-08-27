# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# SECURITY: Prefer safetensors. Legacy PyTorch checkpoints are loaded only with
# ``weights_only=True``, which rejects arbitrary Python objects.

"""Weight loading and application for ONNX models.

This module handles downloading model weights from HuggingFace Hub and
applying them to ONNX IR models. Safetensors is preferred. Legacy HuggingFace
checkpoints that only publish ``pytorch_model.bin`` are loaded with
``torch.load(weights_only=True)`` so arbitrary Python objects are rejected.
"""

from __future__ import annotations

__all__ = [
    "StreamingExpertBankSource",
    "StreamingWeightPlan",
    "StreamingWeightSource",
    "apply_weights",
    "stream_qdq_safetensors_to_model",
    "stream_preprocessed_safetensors_to_model",
    "stream_safetensors_to_model",
    "external_data_checksums",
]

import concurrent.futures
import dataclasses
import hashlib
import json
import logging
import math
import pathlib
from collections.abc import Callable, Mapping
from typing import Literal

import onnx_ir as ir
import safetensors.torch
import torch
import tqdm
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from onnx_ir import tensor_adapters
from safetensors import safe_open

from mobius._optimizations import fold_initializers_after_weights

logger = logging.getLogger(__name__)

_WEIGHT_INDEX_NAME = "model.safetensors.index.json"
_SINGLE_WEIGHT_NAME = "model.safetensors"
_PYTORCH_WEIGHT_INDEX_NAME = "pytorch_model.bin.index.json"
_SINGLE_PYTORCH_WEIGHT_NAME = "pytorch_model.bin"
_SAFETENSORS_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
}
_SAFETENSORS_TO_IR_DTYPE = {
    "BF16": ir.DataType.BFLOAT16,
    "F16": ir.DataType.FLOAT16,
    "F32": ir.DataType.FLOAT,
    "F8_E4M3": ir.DataType.FLOAT8E4M3FN,
}


@dataclasses.dataclass(frozen=True)
class StreamingWeightSource:
    """One checkpoint tensor bound to one dense ONNX initializer."""

    source_name: str
    mode: Literal["direct", "fp8_scalar", "fp8_block_128"] = "direct"
    scale_name: str | None = None
    expected_scale: float | None = None


@dataclasses.dataclass(frozen=True)
class StreamingExpertBankSource:
    """Per-expert source matrices packed into one dense rank-3 initializer."""

    experts: tuple[tuple[StreamingWeightSource, ...], ...]


@dataclasses.dataclass(frozen=True)
class StreamingWeightPlan:
    """Complete fail-closed source classification for a streaming export."""

    targets: Mapping[str, StreamingWeightSource | StreamingExpertBankSource]
    ignored: Mapping[str, str] = dataclasses.field(default_factory=dict)
    constants: Mapping[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    report: Mapping[str, object] = dataclasses.field(default_factory=dict)


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
    # Map tensor storage identity → the ir.Value of the first initializer
    # assigned for that tensor.  Enables genuine weight sharing: if
    # lm_head.weight and embed_tokens.weight share the same underlying
    # storage (common when HF ties weights), only one ONNX initializer is
    # created and all graph uses point to it.
    #
    # We key on ``data_ptr()`` rather than ``id(tensor)`` because HF
    # safetensors deserialization may create distinct Python objects that
    # share the same storage (same data_ptr).  Using data_ptr catches both
    # cases: same Python object *and* same-storage-different-object.
    storage_to_value: dict[int, ir.Value] = {}

    for name, tensor in state_dict.items():
        if name not in model.graph.initializers:
            logger.warning(
                "Weight '%s' not found in the model. Skipped applying.",
                name,
            )
            continue

        initializer = model.graph.initializers[name]
        storage_key = tensor.data_ptr()

        if storage_key in storage_to_value:
            # This tensor shares storage with an already-assigned initializer.
            # Redirect all graph uses of this initializer to the canonical one,
            # then delete this initializer — genuine single-copy weight sharing.
            canonical = storage_to_value[storage_key]
            initializer.replace_all_uses_with(canonical)
            del model.graph.initializers[name]
            logger.debug(
                "Weight tying: '%s' shares initializer with '%s'",
                name,
                canonical.name,
            )
        else:
            _assign_weight(initializer, tensor, name)
            storage_to_value[storage_key] = initializer

    fold_initializers_after_weights(model)


def _parallel_download(
    model_id: str,
    filenames: list[str],
    *,
    revision: str | None = None,
    desc: str = "files",
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
        kwargs = {"repo_id": model_id}
        if revision is not None:
            kwargs["revision"] = revision
        return [hf_hub_download(filename=f, **kwargs) for f in filenames]

    logger.info("Downloading %d %s files in parallel", len(filenames), desc)
    path_map: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        kwargs = {"repo_id": model_id}
        if revision is not None:
            kwargs["revision"] = revision
        futures = {
            executor.submit(hf_hub_download, filename=f, **kwargs): f for f in filenames
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


def _validate_weight_filenames(filenames: list[str]) -> list[str]:
    """Validate filenames from a weight index.

    The HuggingFace index is model data, so reject absolute paths and path traversal
    before using entries as local filesystem paths or Hub filenames.
    """
    validated = []
    for filename in filenames:
        normalized = filename.replace("\\", "/")
        path = pathlib.PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe weight filename in weight index: {filename!r}")
        validated.append(normalized)
    return validated


def _weight_filenames_from_index(index_path: pathlib.Path) -> list[str]:
    with index_path.open() as f:
        index = json.load(f)
    return _validate_weight_filenames(sorted(set(index["weight_map"].values())))


def _local_weight_paths(model_dir: pathlib.Path) -> tuple[list[str], str] | None:
    """Return local weight paths and format for a HuggingFace checkpoint directory."""
    if not model_dir.is_dir():
        return None

    index_path = model_dir / _WEIGHT_INDEX_NAME
    if index_path.is_file():
        filenames = _weight_filenames_from_index(index_path)
        weight_format = "safetensors"
    elif (model_dir / _SINGLE_WEIGHT_NAME).is_file():
        filenames = [_SINGLE_WEIGHT_NAME]
        weight_format = "safetensors"
    elif (model_dir / _PYTORCH_WEIGHT_INDEX_NAME).is_file():
        filenames = _weight_filenames_from_index(model_dir / _PYTORCH_WEIGHT_INDEX_NAME)
        weight_format = "pytorch"
    elif (model_dir / _SINGLE_PYTORCH_WEIGHT_NAME).is_file():
        filenames = [_SINGLE_PYTORCH_WEIGHT_NAME]
        weight_format = "pytorch"
    else:
        raise FileNotFoundError(
            f"Local checkpoint directory has no '{_WEIGHT_INDEX_NAME}' or "
            f"'{_SINGLE_WEIGHT_NAME}', nor legacy PyTorch weights: {model_dir}"
        )

    root = model_dir.resolve()
    paths = []
    for filename in filenames:
        path = (model_dir / filename).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Unsafe weight filename in weight index: {filename!r}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Weight file referenced by index not found: {path}")
        paths.append(str(path))
    return paths, weight_format


def _dequantize_fp8_tensor(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    name: str,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct one scalar- or 128x128-block-scaled FP8 tensor."""
    if weight.dtype not in {torch.float8_e4m3fn, torch.float8_e5m2}:
        raise ValueError(f"Weight '{name}' is not an FP8 tensor: {weight.dtype}")
    scale = scale.to(target_dtype)
    if scale.ndim == 0:
        return weight.to(target_dtype) * scale
    if scale.ndim != 2:
        raise ValueError(
            f"FP8 weight '{name}' has scale with shape {tuple(scale.shape)}; "
            "expected a scalar or 2-D block scale grid"
        )
    if weight.ndim != 2:
        raise ValueError(
            f"FP8 weight '{name}' has a 2-D scale grid but is "
            f"{weight.ndim}-D; block scaling requires a 2-D weight"
        )

    rows, cols = weight.shape
    expected_grid_shape = ((rows + 127) // 128, (cols + 127) // 128)
    if tuple(scale.shape) != expected_grid_shape:
        raise ValueError(
            f"FP8 weight '{name}' has scale grid shape {tuple(scale.shape)}; "
            f"expected {expected_grid_shape} for weight shape {tuple(weight.shape)}"
        )

    # Mutate one dense target tensor tile-by-tile instead of materializing an
    # expanded scale grid the size of the weight.
    dequantized = weight.to(target_dtype)
    for block_row in range(expected_grid_shape[0]):
        row_start = block_row * 128
        row_end = min(row_start + 128, rows)
        for block_col in range(expected_grid_shape[1]):
            col_start = block_col * 128
            col_end = min(col_start + 128, cols)
            dequantized[row_start:row_end, col_start:col_end].mul_(scale[block_row, block_col])
    return dequantized


def _dequantize_fp8_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Dequantize FP8 weights and return a new dict with float tensors.

    Some HuggingFace checkpoints (e.g. Ministral-3-3B) store linear layer
    weights as float8_e4m3fn with a scalar ``weight_scale_inv`` tensor.
    Others use a two-dimensional grid of inverse scales, with each element
    applying to a 128-by-128 weight block. The real weight value is
    ``fp8_weight.to(bfloat16) * weight_scale_inv``.

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

        if scale_key not in result:
            raise ValueError(
                f"FP8 weight '{key}' has no '{scale_key}' tensor. Refusing to "
                "guess an implicit scale; an architecture-specific loader must "
                "explicitly classify any direct-cast FP8 storage."
            )
        result[key] = _dequantize_fp8_tensor(result[key], result[scale_key], name=key)

    # Remove auxiliary FP8 tensors (not needed in the ONNX graph)
    aux_suffixes = (".weight_scale_inv", ".activation_scale", ".input_scale")
    return {k: v for k, v in result.items() if not any(k.endswith(s) for s in aux_suffixes)}


def _resolve_shard_paths(model_id: str, revision: str | None = None) -> list[str]:
    """Resolve local safetensors shard paths for the streaming loader.

    Uses local files when *model_id* is a directory, otherwise downloads the
    shards from HuggingFace Hub (once) and returns their cache paths. The
    streaming loader reads tensors via ``safe_open``, so this is safetensors
    only and refuses a legacy PyTorch checkpoint (use the eager
    :func:`_download_weights` path for those).
    """
    local = _local_weight_paths(pathlib.Path(model_id))
    if local is not None:
        paths, weight_format = local
        if weight_format != "safetensors":
            raise ValueError(
                "stream_safetensors_to_model supports safetensors checkpoints "
                f"only, but {model_id!r} holds '{weight_format}' weights; use the "
                "eager apply_weights path."
            )
        return paths

    try:
        kwargs = {"repo_id": model_id, "filename": _WEIGHT_INDEX_NAME}
        if revision is not None:
            kwargs["revision"] = revision
        index_path = pathlib.Path(hf_hub_download(**kwargs))
        all_files = _weight_filenames_from_index(index_path)
    except EntryNotFoundError:
        all_files = [_SINGLE_WEIGHT_NAME]

    return _parallel_download(model_id, all_files, revision=revision, desc="safetensors")


def _download_weights(model_id: str, revision: str | None = None) -> dict[str, torch.Tensor]:
    """Download weights from HuggingFace and return as a state dict.

    Prefers safetensors and falls back to legacy PyTorch state dictionaries
    loaded with ``weights_only=True``. Uses parallel downloads for shards.

    .. note::
       This loader holds **every shard resident at once** — the returned state
       dict references the whole checkpoint. For a checkpoint larger than host
       RAM prefer :func:`stream_safetensors_to_model`, which keeps a bounded
       working set (safetensors only).
    """
    local_weights = _local_weight_paths(pathlib.Path(model_id))
    if local_weights is None:
        try:
            kwargs = {"repo_id": model_id, "filename": _WEIGHT_INDEX_NAME}
            if revision is not None:
                kwargs["revision"] = revision
            index_path = pathlib.Path(hf_hub_download(**kwargs))
            all_files = _weight_filenames_from_index(index_path)
            weight_format = "safetensors"
        except EntryNotFoundError:
            try:
                kwargs = {"repo_id": model_id, "filename": _SINGLE_WEIGHT_NAME}
                if revision is not None:
                    kwargs["revision"] = revision
                paths = [hf_hub_download(**kwargs)]
                weight_format = "safetensors"
            except EntryNotFoundError:
                try:
                    kwargs = {"repo_id": model_id, "filename": _PYTORCH_WEIGHT_INDEX_NAME}
                    if revision is not None:
                        kwargs["revision"] = revision
                    index_path = pathlib.Path(hf_hub_download(**kwargs))
                    all_files = _weight_filenames_from_index(index_path)
                    weight_format = "pytorch"
                except EntryNotFoundError:
                    all_files = [_SINGLE_PYTORCH_WEIGHT_NAME]
                    weight_format = "pytorch"
                paths = _parallel_download(
                    model_id,
                    all_files,
                    revision=revision,
                    desc=weight_format,
                )
        else:
            paths = _parallel_download(
                model_id,
                all_files,
                revision=revision,
                desc=weight_format,
            )
    else:
        paths, weight_format = local_weights

    state_dict: dict[str, torch.Tensor] = {}
    for path in tqdm.tqdm(paths, desc="Loading weights"):
        if weight_format == "safetensors":
            state_dict.update(safetensors.torch.load_file(path))
        else:
            shard = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(shard, dict) or not all(
                isinstance(name, str) and isinstance(value, torch.Tensor)
                for name, value in shard.items()
            ):
                raise TypeError(f"Legacy weight file is not a tensor state dict: {path}")
            state_dict.update(shard)

    state_dict = _dequantize_fp8_weights(state_dict)
    return state_dict


def _shard_key_index(paths: list[str]) -> dict[str, tuple[str, list[int], str]]:
    """Map each tensor key -> (shard_path, shape, dtype) by reading headers only.

    ``safe_open(...).keys()`` and ``get_slice(...).get_shape()`` read only the
    safetensors header, so this never materializes weight data. Duplicate keys
    are rejected because choosing a shard by traversal order is ambiguous.
    """
    key_index: dict[str, tuple[str, list[int], str]] = {}
    for path in paths:
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open handle is not directly iterable
                if key in key_index:
                    first_path = key_index[key][0]
                    raise ValueError(
                        f"Duplicate tensor key {key!r} across safetensors shards "
                        f"{first_path!r} and {path!r}"
                    )
                sliced = handle.get_slice(key)
                key_index[key] = (path, list(sliced.get_shape()), sliced.get_dtype())
    return key_index


def _assign_lazy_from_shard(
    initializer: ir.Value, shard_path: str, key: str, name: str
) -> None:
    """Assign an initializer a LazyTensor that reads its weight from a shard.

    The closure re-opens the shard and reads exactly one tensor at serialization
    time, so nothing is retained between assignment and ``ir.save`` and the peak
    host RAM is bounded by the largest single tensor, not the whole checkpoint.
    """
    onnx_dtype = initializer.dtype
    assert onnx_dtype is not None
    target_dtype = tensor_adapters.to_torch_dtype(onnx_dtype)

    def tensor_func(
        p: str = shard_path, k: str = key, dt=target_dtype, n: str = name
    ) -> tensor_adapters.TorchTensor:
        with safe_open(p, framework="pt") as handle:
            tensor = handle.get_tensor(k)
        if tensor.dtype != dt:
            tensor = tensor.to(dt)
        return tensor_adapters.TorchTensor(tensor, name=n)

    initializer.const_value = ir.LazyTensor(
        tensor_func,
        dtype=onnx_dtype,
        shape=ir.Shape(initializer.shape),
        name=name,
    )


def _materialize_preprocessed_source(
    source: StreamingWeightSource,
    key_index: Mapping[str, tuple[str, list[int], str]],
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Read and reconstruct one classified source tensor."""
    source_path, source_shape, source_dtype = key_index[source.source_name]
    del source_shape
    with safe_open(source_path, framework="pt") as handle:
        tensor = handle.get_tensor(source.source_name)
    if source.mode == "fp8_block_128":
        assert source.scale_name is not None
        scale_path = key_index[source.scale_name][0]
        with safe_open(scale_path, framework="pt") as handle:
            scale = handle.get_tensor(source.scale_name)
        tensor = _dequantize_fp8_tensor(
            tensor,
            scale,
            name=source.source_name,
            target_dtype=torch.bfloat16,
        )
        return tensor if target_dtype == torch.bfloat16 else tensor.to(target_dtype)
    if source.mode == "fp8_scalar":
        if not source_dtype.startswith("F8"):
            raise ValueError(
                f"Streaming source '{source.source_name}' was classified fp8_scalar "
                f"but has dtype {source_dtype}"
            )
        assert source.scale_name is not None
        scale_path = key_index[source.scale_name][0]
        with safe_open(scale_path, framework="pt") as handle:
            scale = handle.get_tensor(source.scale_name)
        actual_scale = float(scale.to(torch.float32).item())
        if source.expected_scale is None or actual_scale != source.expected_scale:
            raise ValueError(
                f"FP8 scalar source '{source.source_name}' has scale {actual_scale}; "
                f"expected pinned value {source.expected_scale}"
            )
        tensor = tensor.to(torch.bfloat16)
        tensor.mul_(scale.to(torch.bfloat16))
        return tensor if target_dtype == torch.bfloat16 else tensor.to(target_dtype)
    if source.mode == "direct":
        if source_dtype.startswith("F8"):
            raise ValueError(
                f"Streaming source '{source.source_name}' is FP8 but was not explicitly "
                "classified as scaled FP8"
            )
        return tensor if tensor.dtype == target_dtype else tensor.to(target_dtype)
    raise AssertionError(f"Unknown streaming weight mode: {source.mode}")


def _assign_lazy_preprocessed(
    initializer: ir.Value,
    source: StreamingWeightSource,
    key_index: Mapping[str, tuple[str, list[int], str]],
    target_name: str,
) -> None:
    """Bind a direct or dense-dequantized source with one-tensor working memory."""
    onnx_dtype = initializer.dtype
    assert onnx_dtype is not None
    target_dtype = tensor_adapters.to_torch_dtype(onnx_dtype)

    def tensor_func(
        source_spec: StreamingWeightSource = source,
        dt: torch.dtype = target_dtype,
        n: str = target_name,
    ) -> tensor_adapters.TorchTensor:
        tensor = _materialize_preprocessed_source(source_spec, key_index, dt)
        return tensor_adapters.TorchTensor(tensor, name=n)

    initializer.const_value = ir.LazyTensor(
        tensor_func,
        dtype=onnx_dtype,
        shape=ir.Shape(initializer.shape),
        name=target_name,
    )


def _assign_lazy_expert_bank(
    initializer: ir.Value,
    source: StreamingExpertBankSource,
    key_index: Mapping[str, tuple[str, list[int], str]],
    target_name: str,
) -> None:
    """Pack independently scaled expert matrices into one dense rank-3 bank."""
    onnx_dtype = initializer.dtype
    assert onnx_dtype is not None
    target_dtype = tensor_adapters.to_torch_dtype(onnx_dtype)
    target_shape = tuple(int(dim) for dim in initializer.shape)

    def tensor_func(
        source_spec: StreamingExpertBankSource = source,
        dt: torch.dtype = target_dtype,
        n: str = target_name,
    ) -> tensor_adapters.TorchTensor:
        output = torch.empty(target_shape, dtype=dt)
        for expert_index, projections in enumerate(source_spec.experts):
            row = 0
            for projection in projections:
                tensor = _materialize_preprocessed_source(projection, key_index, dt)
                next_row = row + tensor.shape[0]
                output[expert_index, row:next_row].copy_(tensor)
                row = next_row
            if row != target_shape[1]:
                raise ValueError(
                    f"Expert {expert_index} for '{n}' populated {row} rows; "
                    f"expected {target_shape[1]}"
                )
        return tensor_adapters.TorchTensor(output, name=n)

    initializer.const_value = ir.LazyTensor(
        tensor_func,
        dtype=onnx_dtype,
        shape=ir.Shape(target_shape),
        name=target_name,
    )


def stream_preprocessed_safetensors_to_model(
    model: ir.Model,
    model_id: str,
    planner: Callable[
        [Mapping[str, tuple[str, list[int], str]], Mapping[str, ir.Value]],
        StreamingWeightPlan,
    ],
    *,
    revision: str | None = None,
) -> dict[str, object]:
    """Stream a fully classified transformed checkpoint into a dense ONNX graph.

    The planner must classify every source tensor as a target, a consumed scale,
    a validated deterministic constant, or an explicitly ignored sidecar tensor.
    Any unclassified key, missing target, malformed FP8 grid, or changed constant
    fails before serialization. The resulting package is dense; this path never
    claims native FP8 preservation.
    """
    paths = _resolve_shard_paths(model_id, revision)
    key_index = _shard_key_index(paths)
    plan = planner(key_index, model.graph.initializers)

    consumed = set(plan.ignored) | set(plan.constants)
    assigned: set[str] = set()
    validated_scalar_scales: dict[str, float] = {}
    largest_source_tensor_bytes = 0
    largest_reconstruction_working_set_bytes = 0

    def validate_source(
        source: StreamingWeightSource,
    ) -> tuple[list[int], int, int]:
        if source.source_name not in key_index:
            raise ValueError(f"Streaming source '{source.source_name}' does not exist")
        _source_path, source_shape, source_dtype = key_index[source.source_name]
        if source.mode in {"fp8_block_128", "fp8_scalar"}:
            if source_dtype not in {"F8_E4M3", "F8_E5M2"}:
                raise ValueError(
                    f"Scaled FP8 source '{source.source_name}' has dtype {source_dtype}"
                )
            if source.scale_name is None or source.scale_name not in key_index:
                raise ValueError(
                    f"Scaled FP8 source '{source.source_name}' has no scale tensor"
                )
            scale_path, scale_shape, scale_dtype = key_index[source.scale_name]
            if source.mode == "fp8_block_128":
                expected_grid = [
                    (source_shape[0] + 127) // 128,
                    (source_shape[1] + 127) // 128,
                ]
                if len(source_shape) != 2 or scale_shape != expected_grid:
                    raise ValueError(
                        f"FP8 source '{source.source_name}' has scale grid {scale_shape}; "
                        f"expected {expected_grid} for strict 128x128 blocks"
                    )
                if scale_dtype not in {"BF16", "F32"}:
                    raise ValueError(
                        f"FP8 source '{source.source_name}' has unsupported scale dtype "
                        f"{scale_dtype}; expected BF16 or F32"
                    )
            else:
                if scale_shape != [1] or scale_dtype != "BF16":
                    raise ValueError(
                        f"FP8 scalar source '{source.source_name}' has scale "
                        f"dtype/shape {scale_dtype}/{scale_shape}; expected BF16/[1]"
                    )
                if source.expected_scale is None:
                    raise ValueError(
                        f"FP8 scalar source '{source.source_name}' has no pinned "
                        "expected scale value"
                    )
                actual_scale = validated_scalar_scales.get(source.scale_name)
                if actual_scale is None:
                    with safe_open(scale_path, framework="pt") as handle:
                        scale = handle.get_tensor(source.scale_name)
                    actual_scale = float(scale.to(torch.float32).item())
                    validated_scalar_scales[source.scale_name] = actual_scale
                if actual_scale != source.expected_scale:
                    raise ValueError(
                        f"FP8 scalar source '{source.source_name}' has scale "
                        f"{actual_scale}; expected pinned value {source.expected_scale}"
                    )
            consumed.add(source.scale_name)
        source_element_bytes = _SAFETENSORS_DTYPE_BYTES.get(source_dtype)
        if source_element_bytes is None:
            raise ValueError(
                f"Streaming source '{source.source_name}' has unsupported byte-size "
                f"dtype {source_dtype}"
            )
        source_bytes = math.prod(source_shape) * source_element_bytes
        scale_bytes = 0
        if source.scale_name is not None:
            _scale_path, scale_shape, scale_dtype = key_index[source.scale_name]
            scale_element_bytes = _SAFETENSORS_DTYPE_BYTES.get(scale_dtype)
            if scale_element_bytes is None:
                raise ValueError(
                    f"Streaming scale '{source.scale_name}' has unsupported "
                    f"byte-size dtype {scale_dtype}"
                )
            scale_bytes = math.prod(scale_shape) * scale_element_bytes
        consumed.add(source.source_name)
        return source_shape, source_bytes, scale_bytes

    for target_name, source in plan.targets.items():
        initializer = model.graph.initializers.get(target_name)
        if initializer is None:
            raise ValueError(f"Streaming plan targets unknown initializer '{target_name}'")
        if initializer.const_value is not None:
            raise ValueError(f"Streaming plan targets constant initializer '{target_name}'")
        assert initializer.dtype is not None
        expected_shape = [int(dim) for dim in initializer.shape]
        target_bytes = (math.prod(expected_shape) * initializer.dtype.bitwidth + 7) // 8
        if isinstance(source, StreamingWeightSource):
            source_shape, source_bytes, scale_bytes = validate_source(source)
            if expected_shape != source_shape:
                raise ValueError(
                    f"Weight shape mismatch for '{target_name}': model expects "
                    f"{expected_shape}, checkpoint source '{source.source_name}' has "
                    f"{source_shape}"
                )
            bf16_bytes = math.prod(source_shape) * 2
            cast_bytes = target_bytes if initializer.dtype != ir.DataType.BFLOAT16 else 0
            largest_source_tensor_bytes = max(largest_source_tensor_bytes, source_bytes)
            largest_reconstruction_working_set_bytes = max(
                largest_reconstruction_working_set_bytes,
                source_bytes + bf16_bytes + cast_bytes + scale_bytes,
            )
            _assign_lazy_preprocessed(initializer, source, key_index, target_name)
        else:
            if len(expected_shape) != 3 or len(source.experts) != expected_shape[0]:
                raise ValueError(
                    f"Expert bank '{target_name}' expects shape {expected_shape}, "
                    f"but the plan has {len(source.experts)} experts"
                )
            max_transient_bytes = 0
            for expert_index, projections in enumerate(source.experts):
                rows = 0
                for projection in projections:
                    source_shape, source_bytes, scale_bytes = validate_source(projection)
                    if len(source_shape) != 2 or source_shape[1] != expected_shape[2]:
                        raise ValueError(
                            f"Expert source '{projection.source_name}' has shape "
                            f"{source_shape}; expected [rows, {expected_shape[2]}]"
                        )
                    rows += source_shape[0]
                    dense_projection_bytes = (
                        math.prod(source_shape) * initializer.dtype.bitwidth + 7
                    ) // 8
                    bf16_projection_bytes = math.prod(source_shape) * 2
                    cast_projection_bytes = (
                        dense_projection_bytes
                        if initializer.dtype != ir.DataType.BFLOAT16
                        else 0
                    )
                    largest_source_tensor_bytes = max(
                        largest_source_tensor_bytes,
                        source_bytes,
                    )
                    max_transient_bytes = max(
                        max_transient_bytes,
                        source_bytes
                        + bf16_projection_bytes
                        + cast_projection_bytes
                        + scale_bytes,
                    )
                if rows != expected_shape[1]:
                    raise ValueError(
                        f"Expert {expert_index} for '{target_name}' has {rows} rows; "
                        f"expected {expected_shape[1]}"
                    )
            largest_reconstruction_working_set_bytes = max(
                largest_reconstruction_working_set_bytes,
                target_bytes + max_transient_bytes,
            )
            _assign_lazy_expert_bank(initializer, source, key_index, target_name)
        assigned.add(target_name)

    for source_name, expected in plan.constants.items():
        path, shape, _dtype = key_index[source_name]
        if list(expected.shape) != shape:
            raise ValueError(
                f"Deterministic source '{source_name}' has shape {shape}; "
                f"expected {list(expected.shape)}"
            )
        with safe_open(path, framework="pt") as handle:
            actual = handle.get_tensor(source_name)
        if not torch.equal(actual.cpu(), expected.cpu()):
            raise ValueError(
                f"Deterministic source '{source_name}' does not match the graph constant"
            )

    missing_targets = sorted(
        name
        for name, initializer in model.graph.initializers.items()
        if initializer.const_value is None and name not in assigned
    )
    if missing_targets:
        raise ValueError(
            f"{len(missing_targets)} graph initializer(s) are missing from the "
            f"streaming plan (e.g. {missing_targets[:5]})"
        )
    unclassified = sorted(set(key_index) - consumed)
    if unclassified:
        raise ValueError(
            f"{len(unclassified)} checkpoint tensor(s) are unclassified by the "
            f"streaming plan (e.g. {unclassified[:5]})"
        )

    report = {
        "format": "mobius.weight-loading-report.v1",
        "source": model_id,
        "revision": revision,
        "output_weight_format": "dense",
        "native_fp8": False,
        "assigned_tensors": len(assigned),
        "validated_constants": len(plan.constants),
        "ignored_tensors": len(plan.ignored),
        "largest_source_tensor_bytes": largest_source_tensor_bytes,
        "largest_reconstruction_working_set_bytes": (largest_reconstruction_working_set_bytes),
        **dict(plan.report),
    }
    model.metadata_props["mobius.weight_loading"] = json.dumps(report, sort_keys=True)
    return report


def _graph_constant(
    graph: ir.Graph,
    name: str,
    values: list[int],
    *,
    shape: tuple[int, ...] | None = None,
) -> ir.Value:
    array = torch.tensor(values, dtype=torch.int64)
    if shape is not None:
        array = array.reshape(shape)
    value = ir.Value(
        name=name,
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape(array.shape),
        const_value=tensor_adapters.TorchTensor(array, name),
    )
    graph.register_initializer(value)
    return value


def _append_standard_node(
    graph: ir.Graph,
    op_type: str,
    inputs: list[ir.Value | None],
    *,
    name: str,
    dtype: ir.DataType,
    shape: list[int],
    attributes: dict[str, object] | None = None,
) -> ir.Value:
    output = ir.Value(
        name=f"{name}.output",
        type=ir.TensorType(dtype),
        shape=ir.Shape(shape),
    )
    graph.append(
        ir.Node(
            "",
            op_type,
            inputs,
            outputs=[output],
            attributes=ir.convenience.convert_attributes(attributes or {}),
            name=name,
        )
    )
    return output


def _qdq_source_initializer(
    graph: ir.Graph,
    source_name: str,
    key_index: Mapping[str, tuple[str, list[int], str]],
) -> ir.Value:
    existing = graph.initializers.get(source_name)
    if existing is not None:
        return existing
    path, shape, safetensors_dtype = key_index[source_name]
    dtype = _SAFETENSORS_TO_IR_DTYPE.get(safetensors_dtype)
    if dtype is None:
        raise ValueError(
            f"QDQ source '{source_name}' has unsupported storage dtype {safetensors_dtype}"
        )
    value = ir.Value(
        name=source_name,
        type=ir.TensorType(dtype),
        shape=ir.Shape(shape),
    )
    _assign_lazy_from_shard(value, path, source_name, source_name)
    graph.register_initializer(value)
    return value


def _cast_qdq_output(
    graph: ir.Graph,
    value: ir.Value,
    target_dtype: ir.DataType,
    target_shape: list[int],
    *,
    prefix: str,
) -> ir.Value:
    if value.dtype == target_dtype:
        return value
    return _append_standard_node(
        graph,
        "Cast",
        [value],
        name=f"{prefix}.cast",
        dtype=target_dtype,
        shape=target_shape,
        attributes={"to": int(target_dtype.value)},
    )


def _build_scalar_fp8_qdq(
    graph: ir.Graph,
    source: StreamingWeightSource,
    key_index: Mapping[str, tuple[str, list[int], str]],
    *,
    target_dtype: ir.DataType,
    prefix: str,
) -> ir.Value:
    assert source.scale_name is not None
    codes = _qdq_source_initializer(graph, source.source_name, key_index)
    scale = _qdq_source_initializer(graph, source.scale_name, key_index)
    source_shape = [int(dim) for dim in codes.shape]
    scalar_shape = _graph_constant(graph, f"{prefix}.scalar_shape", [], shape=(0,))
    scalar = _append_standard_node(
        graph,
        "Reshape",
        [scale, scalar_shape],
        name=f"{prefix}.scale_scalar",
        dtype=ir.DataType.BFLOAT16,
        shape=[],
    )
    dequantized = _append_standard_node(
        graph,
        "DequantizeLinear",
        [codes, scalar],
        name=f"{prefix}.dequantize",
        dtype=ir.DataType.BFLOAT16,
        shape=source_shape,
    )
    return _cast_qdq_output(
        graph,
        dequantized,
        target_dtype,
        source_shape,
        prefix=prefix,
    )


def _build_block_fp8_qdq(
    graph: ir.Graph,
    source: StreamingWeightSource,
    key_index: Mapping[str, tuple[str, list[int], str]],
    *,
    target_dtype: ir.DataType,
    prefix: str,
) -> ir.Value:
    assert source.scale_name is not None
    codes = _qdq_source_initializer(graph, source.source_name, key_index)
    scale = _qdq_source_initializer(graph, source.scale_name, key_index)
    rows, cols = (int(dim) for dim in codes.shape)
    block_rows = (rows + 127) // 128
    block_cols = (cols + 127) // 128
    padded_rows = block_rows * 128
    padded_cols = block_cols * 128

    value = codes
    if (rows, cols) != (padded_rows, padded_cols):
        pads = _graph_constant(
            graph,
            f"{prefix}.pads",
            [0, 0, padded_rows - rows, padded_cols - cols],
        )
        value = _append_standard_node(
            graph,
            "Pad",
            [value, pads],
            name=f"{prefix}.pad",
            dtype=ir.DataType.FLOAT8E4M3FN,
            shape=[padded_rows, padded_cols],
            attributes={"mode": "constant"},
        )

    tiled_shape = _graph_constant(
        graph,
        f"{prefix}.tiled_shape",
        [block_rows, 128, block_cols, 128],
    )
    tiled = _append_standard_node(
        graph,
        "Reshape",
        [value, tiled_shape],
        name=f"{prefix}.tile_reshape",
        dtype=ir.DataType.FLOAT8E4M3FN,
        shape=[block_rows, 128, block_cols, 128],
    )
    tile_major = _append_standard_node(
        graph,
        "Transpose",
        [tiled],
        name=f"{prefix}.tile_transpose",
        dtype=ir.DataType.FLOAT8E4M3FN,
        shape=[block_rows, block_cols, 128, 128],
        attributes={"perm": [0, 2, 1, 3]},
    )
    flat_tile_shape = _graph_constant(
        graph,
        f"{prefix}.flat_tile_shape",
        [block_rows * block_cols, 128 * 128],
    )
    flat_tiles = _append_standard_node(
        graph,
        "Reshape",
        [tile_major, flat_tile_shape],
        name=f"{prefix}.flat_tiles",
        dtype=ir.DataType.FLOAT8E4M3FN,
        shape=[block_rows * block_cols, 128 * 128],
    )
    flat_scale_shape = _graph_constant(
        graph,
        f"{prefix}.flat_scale_shape",
        [block_rows * block_cols],
    )
    flat_scales = _append_standard_node(
        graph,
        "Reshape",
        [scale, flat_scale_shape],
        name=f"{prefix}.flat_scales",
        dtype=ir.DataType.BFLOAT16,
        shape=[block_rows * block_cols],
    )
    dequantized_tiles = _append_standard_node(
        graph,
        "DequantizeLinear",
        [flat_tiles, flat_scales],
        name=f"{prefix}.dequantize",
        dtype=ir.DataType.BFLOAT16,
        shape=[block_rows * block_cols, 128 * 128],
        attributes={"axis": 0},
    )
    tile_major_bf16 = _append_standard_node(
        graph,
        "Reshape",
        [
            dequantized_tiles,
            _graph_constant(
                graph,
                f"{prefix}.tile_major_shape",
                [block_rows, block_cols, 128, 128],
            ),
        ],
        name=f"{prefix}.tile_major_bf16",
        dtype=ir.DataType.BFLOAT16,
        shape=[block_rows, block_cols, 128, 128],
    )
    tiled_bf16 = _append_standard_node(
        graph,
        "Transpose",
        [tile_major_bf16],
        name=f"{prefix}.inverse_tile_transpose",
        dtype=ir.DataType.BFLOAT16,
        shape=[block_rows, 128, block_cols, 128],
        attributes={"perm": [0, 2, 1, 3]},
    )
    padded = _append_standard_node(
        graph,
        "Reshape",
        [
            tiled_bf16,
            _graph_constant(
                graph,
                f"{prefix}.padded_shape",
                [padded_rows, padded_cols],
            ),
        ],
        name=f"{prefix}.inverse_tile_reshape",
        dtype=ir.DataType.BFLOAT16,
        shape=[padded_rows, padded_cols],
    )
    if (rows, cols) != (padded_rows, padded_cols):
        padded = _append_standard_node(
            graph,
            "Slice",
            [
                padded,
                _graph_constant(graph, f"{prefix}.slice_starts", [0, 0]),
                _graph_constant(graph, f"{prefix}.slice_ends", [rows, cols]),
                _graph_constant(graph, f"{prefix}.slice_axes", [0, 1]),
            ],
            name=f"{prefix}.slice",
            dtype=ir.DataType.BFLOAT16,
            shape=[rows, cols],
        )
    return _cast_qdq_output(
        graph,
        padded,
        target_dtype,
        [rows, cols],
        prefix=prefix,
    )


def _build_fp8_qdq_source(
    graph: ir.Graph,
    source: StreamingWeightSource,
    key_index: Mapping[str, tuple[str, list[int], str]],
    *,
    target_dtype: ir.DataType,
    prefix: str,
) -> ir.Value:
    if source.mode == "fp8_scalar":
        return _build_scalar_fp8_qdq(
            graph,
            source,
            key_index,
            target_dtype=target_dtype,
            prefix=prefix,
        )
    if source.mode == "fp8_block_128":
        return _build_block_fp8_qdq(
            graph,
            source,
            key_index,
            target_dtype=target_dtype,
            prefix=prefix,
        )
    raise ValueError(f"QDQ requires an FP8 source, got mode {source.mode!r}")


def stream_qdq_safetensors_to_model(
    model: ir.Model,
    model_id: str,
    planner: Callable[
        [Mapping[str, tuple[str, list[int], str]], Mapping[str, ir.Value]],
        StreamingWeightPlan,
    ],
    *,
    revision: str | None = None,
) -> dict[str, object]:
    """Bind exact FP8/scales and reconstruct logical weights with standard QDQ."""
    paths = _resolve_shard_paths(model_id, revision)
    key_index = _shard_key_index(paths)
    plan = planner(key_index, model.graph.initializers)
    graph = model.graph
    consumed = set(plan.ignored) | set(plan.constants)
    assigned: set[str] = set()
    qdq_targets = 0
    code_to_target: dict[str, str] = {}
    scale_to_targets: dict[str, list[str]] = {}
    validated_scalar_scales: dict[str, float] = {}
    largest_source_cast_overlap_bytes = 0

    def record_qdq_source(source: StreamingWeightSource, target_label: str) -> None:
        prior = code_to_target.get(source.source_name)
        if prior is not None:
            raise ValueError(
                f"FP8 code tensor '{source.source_name}' maps to both {prior!r} "
                f"and {target_label!r}; code mapping must be bijective"
            )
        code_to_target[source.source_name] = target_label
        assert source.scale_name is not None
        _code_path, code_shape, code_dtype = key_index[source.source_name]
        _scale_path, scale_shape, scale_dtype = key_index[source.scale_name]
        if code_dtype != "F8_E4M3" or len(code_shape) != 2:
            raise ValueError(
                f"QDQ code '{source.source_name}' must be 2-D F8_E4M3, got "
                f"{code_dtype}/{code_shape}"
            )
        if source.mode == "fp8_block_128":
            expected_scale_shape = [
                (code_shape[0] + 127) // 128,
                (code_shape[1] + 127) // 128,
            ]
            if scale_dtype != "BF16" or scale_shape != expected_scale_shape:
                raise ValueError(
                    f"QDQ block scale '{source.scale_name}' must be BF16/"
                    f"{expected_scale_shape}, got {scale_dtype}/{scale_shape}"
                )
        elif source.mode == "fp8_scalar":
            if scale_dtype != "BF16" or scale_shape != [1]:
                raise ValueError(
                    f"QDQ scalar scale '{source.scale_name}' must be BF16/[1], "
                    f"got {scale_dtype}/{scale_shape}"
                )
            if source.expected_scale is None:
                raise ValueError(
                    f"QDQ scalar source '{source.source_name}' has no expected scale"
                )
            actual = validated_scalar_scales.get(source.scale_name)
            if actual is None:
                with safe_open(_scale_path, framework="pt") as handle:
                    actual = float(
                        handle.get_tensor(source.scale_name).to(torch.float32).item()
                    )
                validated_scalar_scales[source.scale_name] = actual
            if actual != source.expected_scale:
                raise ValueError(
                    f"QDQ scalar scale '{source.scale_name}' is {actual}; expected "
                    f"{source.expected_scale}"
                )
        else:
            raise ValueError(
                f"QDQ source '{source.source_name}' has unsupported mode {source.mode}"
            )
        scale_to_targets.setdefault(source.scale_name, []).append(target_label)

    # Validate the complete plan before registering any source initializer or
    # appending any QDQ node. A malformed late target must not leave a partially
    # mutated graph.
    for target_name, source in plan.targets.items():
        target = graph.initializers.get(target_name)
        if target is None:
            raise ValueError(f"QDQ plan targets unknown initializer '{target_name}'")
        if target.const_value is not None:
            raise ValueError(f"QDQ plan targets constant initializer '{target_name}'")
        if target.shape is None or target.dtype is None:
            raise ValueError(
                f"QDQ target '{target_name}' must have a concrete shape and dtype"
            )
        target_shape = [int(dim) for dim in target.shape]
        if isinstance(source, StreamingWeightSource):
            located = key_index.get(source.source_name)
            if located is None:
                raise ValueError(
                    f"QDQ target '{target_name}' is missing source '{source.source_name}'"
                )
            _path, source_shape, _dtype = located
            if source_shape != target_shape:
                raise ValueError(
                    f"QDQ source '{source.source_name}' has logical shape "
                    f"{source_shape}; target '{target_name}' expects {target_shape}"
                )
            if source.mode != "direct":
                record_qdq_source(source, target_name)
                assert source.scale_name is not None
                consumed.add(source.scale_name)
            consumed.add(source.source_name)
            continue

        if len(target_shape) != 3:
            raise ValueError(
                f"QDQ packed expert target '{target_name}' has shape {target_shape}; "
                "expected rank 3 [experts, rows, input_width]"
            )
        if len(source.experts) != target_shape[0]:
            raise ValueError(
                f"QDQ packed expert target '{target_name}' expects "
                f"{target_shape[0]} experts, but the plan provides "
                f"{len(source.experts)}"
            )
        for expert_index, projections in enumerate(source.experts):
            if not projections:
                raise ValueError(
                    f"QDQ packed expert target '{target_name}' expert "
                    f"{expert_index} has no source projections"
                )
            rows = 0
            for projection_index, projection in enumerate(projections):
                label = f"{target_name}[{expert_index}][{projection_index}]"
                record_qdq_source(projection, label)
                _path, projection_shape, _dtype = key_index[projection.source_name]
                if len(projection_shape) != 2:
                    raise ValueError(
                        f"QDQ expert source '{projection.source_name}' has shape "
                        f"{projection_shape}; target '{target_name}' requires 2-D "
                        "projections"
                    )
                if projection_shape[1] != target_shape[2]:
                    raise ValueError(
                        f"QDQ expert source '{projection.source_name}' has input "
                        f"width {projection_shape[1]}; target '{target_name}' expects "
                        f"{target_shape[2]}"
                    )
                rows += projection_shape[0]
                consumed.add(projection.source_name)
                assert projection.scale_name is not None
                consumed.add(projection.scale_name)
            if rows != target_shape[1]:
                raise ValueError(
                    f"QDQ packed expert target '{target_name}' expert "
                    f"{expert_index} source row sum is {rows}; expected "
                    f"{target_shape[1]}"
                )

    for source_name, expected in plan.constants.items():
        path, shape, _dtype = key_index[source_name]
        if list(expected.shape) != shape:
            raise ValueError(
                f"Deterministic source '{source_name}' has shape {shape}; "
                f"expected {list(expected.shape)}"
            )
        with safe_open(path, framework="pt") as handle:
            actual = handle.get_tensor(source_name)
        if not torch.equal(actual.cpu(), expected.cpu()):
            raise ValueError(
                f"Deterministic source '{source_name}' does not match the graph constant"
            )

    missing_targets = sorted(
        name
        for name, value in graph.initializers.items()
        if value.const_value is None and name not in plan.targets
    )
    if missing_targets:
        raise ValueError(f"QDQ plan leaves graph target(s) unassigned: {missing_targets[:5]}")
    unclassified = sorted(set(key_index) - consumed)
    if unclassified:
        raise ValueError(
            f"{len(unclassified)} checkpoint tensor(s) are unclassified by the "
            f"QDQ plan (e.g. {unclassified[:5]})"
        )

    for target_name, source in plan.targets.items():
        target = graph.initializers.get(target_name)
        if target is None:
            raise ValueError(f"QDQ plan targets unknown initializer '{target_name}'")
        if target.const_value is not None:
            raise ValueError(f"QDQ plan targets constant initializer '{target_name}'")
        assert target.dtype is not None
        prefix = f"fp8_qdq.{hashlib.sha256(target_name.encode()).hexdigest()[:16]}"

        if isinstance(source, StreamingWeightSource):
            if source.mode == "direct":
                path, shape, source_dtype = key_index[source.source_name]
                source_ir_dtype = _SAFETENSORS_TO_IR_DTYPE.get(source_dtype)
                if source_ir_dtype is None:
                    raise ValueError(
                        f"QDQ direct source '{source.source_name}' has unsupported "
                        f"storage dtype {source_dtype}"
                    )
                if source_ir_dtype != target.dtype:
                    source_bytes = math.prod(shape) * _SAFETENSORS_DTYPE_BYTES[source_dtype]
                    target_bytes = (
                        math.prod(int(dim) for dim in target.shape) * target.dtype.bitwidth + 7
                    ) // 8
                    largest_source_cast_overlap_bytes = max(
                        largest_source_cast_overlap_bytes,
                        source_bytes + target_bytes,
                    )
                _assign_lazy_from_shard(target, path, source.source_name, target_name)
                consumed.add(source.source_name)
                assigned.add(target_name)
                continue
            replacement = _build_fp8_qdq_source(
                graph,
                source,
                key_index,
                target_dtype=target.dtype,
                prefix=prefix,
            )
            consumed.add(source.source_name)
            assert source.scale_name is not None
            consumed.add(source.scale_name)
            qdq_targets += 1
        else:
            expert_values: list[ir.Value] = []
            for expert_index, projections in enumerate(source.experts):
                projection_values = []
                for projection_index, projection in enumerate(projections):
                    projection_values.append(
                        _build_fp8_qdq_source(
                            graph,
                            projection,
                            key_index,
                            target_dtype=target.dtype,
                            prefix=f"{prefix}.expert_{expert_index}.{projection_index}",
                        )
                    )
                    consumed.add(projection.source_name)
                    assert projection.scale_name is not None
                    consumed.add(projection.scale_name)
                expert = (
                    projection_values[0]
                    if len(projection_values) == 1
                    else _append_standard_node(
                        graph,
                        "Concat",
                        projection_values,
                        name=f"{prefix}.expert_{expert_index}.concat",
                        dtype=target.dtype,
                        shape=[
                            sum(int(value.shape[0]) for value in projection_values),
                            int(projection_values[0].shape[1]),
                        ],
                        attributes={"axis": 0},
                    )
                )
                expert_values.append(
                    _append_standard_node(
                        graph,
                        "Unsqueeze",
                        [
                            expert,
                            _graph_constant(
                                graph,
                                f"{prefix}.expert_{expert_index}.axis",
                                [0],
                            ),
                        ],
                        name=f"{prefix}.expert_{expert_index}.unsqueeze",
                        dtype=target.dtype,
                        shape=[1, *[int(dim) for dim in expert.shape]],
                    )
                )
            replacement = _append_standard_node(
                graph,
                "Concat",
                expert_values,
                name=f"{prefix}.expert_bank",
                dtype=target.dtype,
                shape=[int(dim) for dim in target.shape],
                attributes={"axis": 0},
            )
            qdq_targets += sum(len(projections) for projections in source.experts)

        target.replace_all_uses_with(replacement, replace_graph_outputs=True)
        del graph.initializers[target_name]
        assigned.add(target_name)

    graph.sort()
    fold_initializers_after_weights(model)
    canonical_mapping = "\n".join(
        f"{source}\t{code_to_target[source]}\n" for source in sorted(code_to_target)
    )
    stored_code_bytes = sum(
        math.prod(key_index[name][1]) * _SAFETENSORS_DTYPE_BYTES[key_index[name][2]]
        for name in code_to_target
    )
    stored_scale_bytes = sum(
        math.prod(key_index[name][1]) * _SAFETENSORS_DTYPE_BYTES[key_index[name][2]]
        for name in scale_to_targets
    )
    dense_equivalent_bytes = sum(math.prod(key_index[name][1]) * 2 for name in code_to_target)
    recipe = {
        "format": "mobius.fp8-qdq-recipe.v1",
        "block_shape": [128, 128],
        "block_transform": (
            "[R,C] -> pad -> [Br,128,Bc,128] -> transpose(0,2,1,3) "
            "-> [Br*Bc,16384] -> DequantizeLinear(axis=0) -> inverse -> slice"
        ),
        "ple_transform": "DequantizeLinear(float8_codes, reshape(bf16_scale,[ ]))",
        "source_codes_preserved": True,
        "source_scales_preserved": True,
        "native_fp8_compute": False,
        "runtime_execution_proven": False,
        "qdq_targets": qdq_targets,
        "source_code_tensors": len(code_to_target),
        "source_scale_tensors": len(scale_to_targets),
        "code_mapping": "bijective",
        "scale_mapping": (
            "one-to-one for block grids; the pinned PLE scalar is shared by "
            "all PLE shard code tensors"
        ),
        "canonical_code_mapping_sha256": hashlib.sha256(
            canonical_mapping.encode()
        ).hexdigest(),
    }
    report = {
        "format": "mobius.weight-loading-report.v1",
        "source": model_id,
        "revision": revision,
        "output_weight_format": "fp8_qdq",
        "storage_preserving": True,
        "native_fp8": False,
        "streaming_external_data": True,
        "largest_source_cast_overlap_bytes": largest_source_cast_overlap_bytes,
        "stored_fp8_code_bytes": stored_code_bytes,
        "stored_scale_bytes": stored_scale_bytes,
        "dense_equivalent_bytes": dense_equivalent_bytes,
        "assigned_targets": len(assigned),
        "validated_constants": len(plan.constants),
        "ignored_tensors": len(plan.ignored),
        "qdq_recipe": recipe,
        **dict(plan.report),
    }
    model.metadata_props["mobius.weight_loading"] = json.dumps(report, sort_keys=True)
    model.metadata_props["mobius.fp8_qdq_recipe"] = json.dumps(recipe, sort_keys=True)
    return report


def stream_safetensors_to_model(
    model: ir.Model,
    model_id: str,
    *,
    revision: str | None = None,
    require_passthrough: bool = True,
) -> set[str]:
    """Apply weights to *model* without holding the whole checkpoint in RAM.

    Every graph initializer is bound to a :class:`ir.LazyTensor` that reads its
    tensor from the owning safetensors shard on demand. Compared to
    :func:`apply_weights` fed by :func:`_download_weights` — which materializes
    the entire checkpoint as one state dict — this keeps at most one tensor
    resident, so a checkpoint far larger than host RAM can be re-serialized to
    ONNX external data.

    This is a *pass-through* loader: it maps each ONNX initializer 1:1 to a
    checkpoint tensor and only casts dtype. It deliberately does **not** perform
    weight fusion, qkv splitting, or quantization/dequantization. When
    *require_passthrough* is True (the default) it refuses two kinds of
    non-passthrough sources rather than silently emitting a wrong graph: (1) a
    quantized checkpoint (fp8 weights or ``*_scale_inv``/``*_scale`` tensors),
    which needs the eager dequant path, and (2) a graph with an initializer that
    has no matching checkpoint tensor — the signature of a model that needs
    preprocessing. Such models must use the eager :func:`apply_weights` path.

    Returns the set of initializer names that were bound.

    Raises:
        ValueError: on a shape mismatch; when the checkpoint is quantized; or
            (in pass-through mode) when a graph initializer has no corresponding
            checkpoint tensor.
    """
    paths = _resolve_shard_paths(model_id, revision)
    key_index = _shard_key_index(paths)

    if require_passthrough:
        # An fp8 checkpoint stores raw float8 weights under the same name the
        # graph expects for bf16, plus separate ``*_scale_inv``/``*_scale``
        # tensors. Casting fp8 -> bf16 without multiplying by the scale silently
        # produces wrong weights, and the scale keys never map to an
        # initializer (so the missing-tensor guard below never fires). Refuse
        # such quantized sources up front; they must use the eager
        # apply_weights path, which applies the scale.
        quant_signals = sorted(
            k
            for k, (_p, _s, dt) in key_index.items()
            if dt.startswith("F8") or k.endswith(("_scale_inv", "weight_scale"))
        )
        if quant_signals:
            raise ValueError(
                f"Checkpoint appears quantized (fp8 / scaled weights, e.g. "
                f"{quant_signals[:5]}); the pass-through streaming loader cannot "
                f"dequantize it and would drop the weight scale. Use the eager "
                f"apply_weights path (which applies weight_scale_inv)."
            )

    assigned: set[str] = set()
    missing: list[str] = []
    for name, initializer in list(model.graph.initializers.items()):
        if initializer.const_value is not None:
            continue
        located = key_index.get(name)
        if located is None:
            missing.append(name)
            continue
        shard_path, shard_shape, _shard_dtype = located
        if initializer.shape is not None:
            expected = [int(d) for d in initializer.shape]
            if expected != list(shard_shape):
                raise ValueError(
                    f"Weight shape mismatch for '{name}': model expects "
                    f"{expected}, checkpoint has {list(shard_shape)}"
                )
        _assign_lazy_from_shard(initializer, shard_path, name, name)
        assigned.add(name)

    if missing and require_passthrough:
        preview = missing[:5]
        raise ValueError(
            f"{len(missing)} graph initializer(s) have no matching checkpoint "
            f"tensor and cannot be streamed as pass-through weights "
            f"(e.g. {preview}). This model needs weight preprocessing "
            f"(fusion/split/quantization); use the eager apply_weights path."
        )
    if missing:
        logger.warning(
            "Streaming left %d initializer(s) unassigned (require_passthrough=False)",
            len(missing),
        )

    fold_initializers_after_weights(model)
    return assigned


def external_data_checksums(
    output_dir: str | pathlib.PathLike,
    *,
    pattern: str = "*.onnx.data",
    chunk_size: int = 1 << 20,
) -> dict[str, dict[str, object]]:
    """Compute a deterministic sha256 + size manifest for ONNX external data.

    Returns ``{filename: {"sha256": hex, "size": bytes}}`` sorted by filename so
    the manifest is byte-stable across runs. Use it to verify a re-export
    reproduced identical external data (deterministic naming *and* content).
    """
    directory = pathlib.Path(output_dir)
    manifest: dict[str, dict[str, object]] = {}
    for path in sorted(directory.glob(pattern)):
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(chunk_size), b""):
                digest.update(block)
                size += len(block)
        manifest[path.name] = {"sha256": digest.hexdigest(), "size": size}
    return manifest
