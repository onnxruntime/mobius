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
    "apply_weights",
    "stream_safetensors_to_model",
    "external_data_checksums",
]

import concurrent.futures
import hashlib
import json
import logging
import pathlib

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

        if scale_key in result:
            # Cast scale to bfloat16 to guarantee the output dtype is bfloat16,
            # even when weight_scale_inv is stored as FP32 in the checkpoint.
            scale = result[scale_key].to(torch.bfloat16)
            if scale.ndim == 0:
                result[key] = result[key].to(torch.bfloat16) * scale
            elif scale.ndim == 2:
                weight = result[key]
                if weight.ndim != 2:
                    raise ValueError(
                        f"FP8 weight '{key}' has a 2-D scale grid but is "
                        f"{weight.ndim}-D; block scaling requires a 2-D weight"
                    )

                rows, cols = weight.shape
                expected_grid_shape = ((rows + 127) // 128, (cols + 127) // 128)
                if tuple(scale.shape) != expected_grid_shape:
                    raise ValueError(
                        f"FP8 weight '{key}' has scale grid shape {tuple(scale.shape)}; "
                        f"expected {expected_grid_shape} for weight shape {tuple(weight.shape)}"
                    )

                # Scale each 128-by-128 tile in the BF16 output. This avoids
                # allocating a full expanded scale tensor for large expert weights.
                dequantized = weight.to(torch.bfloat16)
                for block_row in range(expected_grid_shape[0]):
                    row_start = block_row * 128
                    row_end = min(row_start + 128, rows)
                    for block_col in range(expected_grid_shape[1]):
                        col_start = block_col * 128
                        col_end = min(col_start + 128, cols)
                        dequantized[row_start:row_end, col_start:col_end].mul_(
                            scale[block_row, block_col]
                        )
                result[key] = dequantized
            else:
                raise ValueError(
                    f"FP8 weight '{key}' has scale with shape {tuple(scale.shape)}; "
                    "expected a scalar or 2-D block scale grid"
                )
        else:
            logger.warning("FP8 weight '%s' has no scale_inv — casting without scaling", key)
            result[key] = result[key].to(torch.bfloat16)

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
    safetensors header, so this never materializes weight data. When a key
    appears in more than one shard the first shard wins and a warning is logged.
    """
    key_index: dict[str, tuple[str, list[int], str]] = {}
    for path in paths:
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open handle is not directly iterable
                if key in key_index:
                    logger.warning("Duplicate tensor key %r across shards; keeping first", key)
                    continue
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
