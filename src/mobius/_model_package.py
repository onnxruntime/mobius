# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ModelPackage: a collection of named ONNX models forming a complete model.

For text-only models, a package contains a single component (e.g. ``"model"``).
For multimodal models, it may contain multiple (e.g. ``"vision_encoder"``,
``"text_decoder"``).

Example::

    from mobius import build

    pkg = build("meta-llama/Llama-3-8B")
    pkg["model"]              # ir.Model
    pkg.save("/output/llama/")  # saves model.onnx + model.onnx.data
"""

from __future__ import annotations

__all__ = ["ModelPackage"]

import json
import logging
import os
from collections import UserDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnx_ir as ir
import torch
import tqdm

from mobius._optimizations import fold_initializers_after_weights
from mobius._weight_loading import _assign_weight

logger = logging.getLogger(__name__)


class ModelPackage(UserDict[str, ir.Model]):
    """A dict-like collection of named ``ir.Model`` objects.

    Attributes:
        config: The architecture configuration used to build the models,
            or ``None`` if not available (e.g. after :meth:`load`).
    """

    def __init__(
        self,
        models: dict[str, ir.Model] | None = None,
        config: object | None = None,
    ) -> None:
        super().__init__(models or {})
        self.config = config

    def __repr__(self) -> str:
        names = ", ".join(repr(k) for k in self.data)
        return f"ModelPackage({{{names}}})"

    # -- Persistence -------------------------------------------------------

    def save(
        self,
        directory: str,
        *,
        external_data: str = "onnx",
        max_shard_size_bytes: int | None = None,
        components: Callable[[str], bool] | None = None,
        progress_bar: bool = True,
        check_weights: bool = True,
    ) -> None:
        """Save all component models to a directory.

        When the package contains a single model, it is saved directly as
        ``model.onnx`` in *directory*.  When multiple models are present,
        each is saved in its own subfolder as ``{name}/model.onnx``.

        .. note::
            This method writes ONNX files only.  If you need a directory that
            ``onnxruntime-genai`` can load (i.e. with ``genai_config.json`` and
            tokenizer files), use
            :func:`mobius.integrations.ort_genai.export_package` instead. For
            onnx-genai, call
            :func:`mobius.integrations.onnx_genai.write_onnx_genai_config`
            after saving, or use ``mobius build --runtime onnx-genai``.

        Args:
            directory: Path to the output directory (created if needed).
            external_data: External data format. ``"onnx"`` (default) saves
                weights to ``model.onnx.data``. ``"safetensors"`` saves
                weights in safetensors format.

                .. warning::
                    The safetensors format does not guarantee 256-byte offset
                    alignment for tensor data within the file.  This can cause
                    ``CUBLAS_STATUS_INVALID_VALUE`` ("misaligned address")
                    errors on some CUDA/cuBLAS versions when loading weights
                    via memory-mapped I/O.  Use ``"onnx"`` (the default) for
                    models targeting CUDA execution.
            max_shard_size_bytes: Maximum shard size in bytes for safetensors
                format.  Only used when *external_data* is ``"safetensors"``.
            components: Optional predicate ``(name) -> bool`` that selects
                which components to save.  When ``None`` (default), all
                components are saved.  Examples::

                    # Allow list
                    components=lambda name: name in {"transformer", "vae"}
                    # Block list
                    components=lambda name: name != "text_encoder"

            progress_bar: Whether to display a tqdm progress bar while
                saving tensors.  Defaults to ``True``.
            check_weights: Whether to verify that all initializers have
                weight data before saving.  Defaults to ``True``.
                Set to ``False`` when saving skeleton models without weights.

        Raises:
            ValueError: If *external_data* is not ``"onnx"`` or
                ``"safetensors"``, or if *check_weights* is ``True`` and
                any initializer is missing its ``const_value``.
        """
        if external_data not in {"onnx", "safetensors"}:
            raise ValueError(
                f"Unknown external_data format {external_data!r}. "
                "Expected 'onnx' or 'safetensors'."
            )
        os.makedirs(directory, exist_ok=True)
        callback = _make_progress_callback() if progress_bar else None

        selected = {
            name: model
            for name, model in self.data.items()
            if components is None or components(name)
        }
        use_subfolders = len(selected) > 1

        for name, model in selected.items():
            if check_weights:
                _check_weights(name, model)
            if use_subfolders:
                model_dir = os.path.join(directory, name)
                os.makedirs(model_dir, exist_ok=True)
            else:
                model_dir = directory
            path = os.path.join(model_dir, "model.onnx")
            if external_data == "safetensors":
                _save_safetensors_external_data(
                    model,
                    path,
                    max_shard_size_bytes=max_shard_size_bytes,
                    callback=callback,
                )
            else:
                ir.save(model, path, external_data="model.onnx.data", callback=callback)

    @classmethod
    def load(cls, directory: str) -> ModelPackage:
        """Load all ``.onnx`` files from a directory into a package.

        Supports two layouts:

        - **Flat**: ``model.onnx`` directly in *directory* → single-component
          package keyed ``"model"``.
        - **Subfolder**: each subdirectory contains ``model.onnx`` → multi-
          component package keyed by subfolder name.

        Args:
            directory: Path to the directory containing models.

        Returns:
            A new ``ModelPackage`` with one entry per model found.
        """
        models: dict[str, ir.Model] = {}
        # Check for subfolder layout first
        for entry in sorted(os.listdir(directory)):
            subdir = os.path.join(directory, entry)
            model_path = os.path.join(subdir, "model.onnx")
            if os.path.isdir(subdir) and os.path.isfile(model_path):
                models[entry] = ir.load(model_path)
        if models:
            return cls(models)
        # Fall back to flat layout
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".onnx"):
                name = filename.removesuffix(".onnx")
                models[name] = ir.load(os.path.join(directory, filename))
        return cls(models)

    # -- Weight application ------------------------------------------------

    def apply_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix_map: dict[str, str] | None = None,
    ) -> None:
        """Apply weights from a state dict across component models.

        For single-component packages, all weights are applied to the sole
        model. For multi-component packages, use ``prefix_map`` to route
        weights by prefix to the correct component.

        Args:
            state_dict: Mapping of parameter names to torch tensors.
            prefix_map: Optional mapping from weight-name prefix to component
                name. For example::

                    {"model.vision": "vision_encoder", "model.language": "text_decoder"}

                Weights whose name starts with a prefix are applied to the
                named component (with the prefix stripped). Unmatched weights
                are applied to all components.
        """
        applied: set[str] = set()

        if len(self.data) == 1:
            # Single component — apply all weights directly
            model = next(iter(self.data.values()))
            applied = _apply_weights_to_model(model, state_dict)
        elif prefix_map is None:
            # No routing info — try every weight against every model
            for model in self.data.values():
                applied |= _apply_weights_to_model(model, state_dict)
        else:
            # Route by prefix
            routed: dict[str, dict[str, torch.Tensor]] = {name: {} for name in self.data}
            unmatched: dict[str, torch.Tensor] = {}
            # Track original HF names for weights that get stripped
            stripped_to_original: dict[str, str] = {}

            for weight_name, tensor in state_dict.items():
                matched = False
                for prefix, component in prefix_map.items():
                    if weight_name.startswith(prefix):
                        stripped = weight_name[len(prefix) :].lstrip(".")
                        routed[component][stripped] = tensor
                        stripped_to_original[stripped] = weight_name
                        matched = True
                        break
                if not matched:
                    unmatched[weight_name] = tensor

            for component_name, component_weights in routed.items():
                applied_stripped = _apply_weights_to_model(
                    self.data[component_name], component_weights
                )
                for s in applied_stripped:
                    applied.add(stripped_to_original.get(s, s))

            # Try unmatched weights against all models
            if unmatched:
                for model in self.data.values():
                    applied |= _apply_weights_to_model(model, unmatched)

        _log_weight_mapping(state_dict, applied)

        # Fold constants now that weights have been loaded.
        # PackQKV emits Concat(w_q, w_k, w_v) in the graph; those nodes can only
        # be constant-folded once the weight tensors carry their const_value.
        for model in self.data.values():
            fold_initializers_after_weights(model)


def _make_progress_callback():
    """Create a tqdm progress-bar callback for ``ir.save``."""
    pbar = tqdm.tqdm()
    total_set = False

    def callback(tensor: ir.TensorProtocol, metadata: ir.external_data.CallbackInfo) -> None:
        nonlocal total_set
        if not total_set:
            pbar.total = metadata.total
            total_set = True
        pbar.update()
        pbar.set_description(
            f"Saving {tensor.name} ({tensor.dtype.short_name()}, {tensor.shape})"
        )
        pbar.set_postfix(written_gib=f"{metadata.offset / (1024**3):.2f}")

    return callback


def _safetensors_write_workers(total_shards: int) -> int:
    """Return the number of shard writer threads for safetensors output."""
    raw = os.environ.get("MOBIUS_SAFETENSORS_WRITE_WORKERS")
    if raw:
        try:
            workers = int(raw)
        except ValueError:
            logger.warning(
                "Ignoring invalid MOBIUS_SAFETENSORS_WRITE_WORKERS=%r; expected an integer",
                raw,
            )
        else:
            if workers > 0:
                return min(workers, total_shards)
            logger.warning(
                "Ignoring invalid MOBIUS_SAFETENSORS_WRITE_WORKERS=%r; expected > 0",
                raw,
            )
    return max(1, min(4, total_shards))


def _tensor_to_safetensors_array(tensor: ir.TensorProtocol) -> np.ndarray:
    """Return a NumPy array with the storage representation safetensors expects."""
    if tensor.dtype.bitwidth < 8:
        return np.frombuffer(tensor.tobytes(), dtype=np.uint8)
    return np.asarray(tensor)


def _save_safetensors_external_data(
    model: ir.Model,
    path: str,
    *,
    max_shard_size_bytes: int | None,
    callback: Callable[[ir.TensorProtocol, ir.external_data.CallbackInfo], None] | None = None,
) -> None:
    """Save ONNX with sharded safetensors external data.

    ``onnx_ir.save_safetensors`` shards but writes each shard serially and, with
    newer ``safetensors`` releases, can fail because the low-level
    ``serialize_file`` API now expects ``TensorSpec`` objects. This compatibility
    saver uses the public NumPy safetensors API and writes independent shards in
    a bounded thread pool.
    """
    import safetensors.numpy

    safe_globals = ir.save_safetensors.__globals__
    shard_tensors = safe_globals["_shard_tensors"]
    get_shard_filename = safe_globals["_get_shard_filename"]
    read_safetensors = safe_globals["_read_safetensors"]
    migrate_tensor_shape_dtype = safe_globals["_migrate_tensor_shape_dtype"]

    path_str = str(path)
    base_name = (
        os.path.splitext(os.path.basename(path_str))[0]
        if "." in os.path.basename(path_str)
        else os.path.basename(path_str)
    )
    location = f"{base_name}.safetensors"
    base_dir = os.path.dirname(path)

    value_tensor_pairs: list[tuple[ir.Value, ir.TensorProtocol]] = []
    initializer_names: set[str] = set()
    for graph in model.graphs():
        for value in graph.initializers.values():
            tensor = value.const_value
            name = value.name
            if name is None:
                raise ValueError(
                    f"Initializer value '{value!r}' has no name (in graph {graph.name!r}). "
                    "All initializers must have names."
                )
            if tensor is None:
                continue
            if name in initializer_names:
                raise ValueError(
                    f"Duplicate initializer name found: {name} (in graph {graph.name!r}). "
                    "Rename the initializers to have unique names before saving to safetensors."
                )
            initializer_names.add(name)
            value_tensor_pairs.append((value, tensor))

    values_to_save: list[ir.Value] = []
    tensors_to_save: list[ir.TensorProtocol] = []
    for value, tensor in value_tensor_pairs:
        if tensor.nbytes >= 256:
            values_to_save.append(value)
            tensors_to_save.append(tensor)

    if not tensors_to_save:
        ir.save(model, path)
        return

    tensor_shards = shard_tensors(tensors_to_save, max_shard_size_bytes)
    total_shards = len(tensor_shards)
    total_tensors = len(tensors_to_save)
    total_size = sum(tensor.nbytes for tensor in tensors_to_save)
    filenames = [
        get_shard_filename(location, shard_idx, total_shards)
        for shard_idx in range(1, total_shards + 1)
    ]
    workers = _safetensors_write_workers(total_shards)
    logger.info(
        "Writing %d safetensors shard(s) with %d worker thread(s)", total_shards, workers
    )

    # Precompute stable progress metadata and the HF-style weight map before
    # shard writes run concurrently.
    tensor_progress: dict[str, tuple[int, int]] = {}
    weight_map: dict[str, str] = {}
    current_offset = 0
    current_index = 0
    for filename, shard in zip(filenames, tensor_shards, strict=True):
        for tensor in shard:
            assert tensor.name is not None
            tensor_progress[tensor.name] = (current_index, current_offset)
            weight_map[tensor.name] = filename
            current_offset += tensor.nbytes
            current_index += 1

    def write_shard(filename: str, shard: list[ir.TensorProtocol]) -> None:
        shard_dict: dict[str, np.ndarray] = {}
        for tensor in shard:
            assert tensor.name is not None
            shard_dict[tensor.name] = _tensor_to_safetensors_array(tensor)
        safetensors.numpy.save_file(shard_dict, os.path.join(base_dir, filename))
        if callback is not None:
            for tensor in shard:
                assert tensor.name is not None
                index, offset = tensor_progress[tensor.name]
                callback(
                    tensor,
                    ir.external_data.CallbackInfo(
                        total=total_tensors,
                        index=index,
                        offset=offset + tensor.nbytes,
                        filename=filename,
                    ),
                )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(write_shard, filename, shard)
            for filename, shard in zip(filenames, tensor_shards, strict=True)
        ]
        for future in futures:
            future.result()

    if total_shards > 1:
        index_filename = location.rsplit(".safetensors", 1)[0] + ".safetensors.index.json"
        with open(os.path.join(base_dir, index_filename), "w") as f:
            json.dump(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": weight_map,
                },
                f,
                indent=2,
            )

    try:
        value_map: dict[str, ir.Value] = {value.name: value for value in values_to_save}  # type: ignore[misc]
        for filename in filenames:
            for name, safe_tensor in read_safetensors(filename, base_dir=base_dir).items():
                value = value_map[name]
                model_tensor = value.const_value
                assert model_tensor is not None
                value.const_value = migrate_tensor_shape_dtype(model_tensor, safe_tensor)
        ir.save(model, path)
    finally:
        for value, tensor in value_tensor_pairs:
            value.const_value = tensor


def _check_weights(component_name: str, model: ir.Model) -> None:
    """Raise if any initializer is missing its weight data."""
    unset = [
        name for name, init in model.graph.initializers.items() if init.const_value is None
    ]
    if unset:
        examples = ", ".join(f"'{n}'" for n in unset[:5])
        suffix = f" (and {len(unset) - 5} more)" if len(unset) > 5 else ""
        raise ValueError(
            f"Component '{component_name}' has {len(unset)} initializer(s) "
            f"without weights: {examples}{suffix}. "
            f"Ensure all weights are loaded before saving. Check if the preprocess_weights logic is correct."
        )


def _log_weight_mapping(
    state_dict: dict[str, torch.Tensor],
    applied: set[str],
) -> None:
    """Log applied and unmapped weights for debugging.

    Logs each unmapped weight at INFO level (with name and shape),
    and the full mapping summary at DEBUG level.
    """
    all_names = set(state_dict.keys())
    unmapped = all_names - applied

    if unmapped:
        lines = sorted(f"  {name} {tuple(state_dict[name].shape)}" for name in unmapped)
        logger.info(
            "%d weight(s) not applied to ONNX model (may be tied or unused):\n%s",
            len(unmapped),
            "\n".join(lines),
        )

    if logger.isEnabledFor(logging.DEBUG):
        mapped_lines = sorted(f"  {name} {tuple(state_dict[name].shape)}" for name in applied)
        logger.debug(
            "Applied %d of %d weight(s):\n%s",
            len(applied),
            len(state_dict),
            "\n".join(mapped_lines),
        )


def _apply_weights_to_model(model: ir.Model, state_dict: dict[str, torch.Tensor]) -> set[str]:
    """Apply weights to a single model (internal helper).

    Uses :func:`_assign_weight` from ``_weight_loading`` for shape
    checking and lazy dtype casting.

    Returns:
        Set of weight names from *state_dict* that were applied.
    """
    applied: set[str] = set()
    for name, tensor in state_dict.items():
        if name not in model.graph.initializers:
            continue

        _assign_weight(model.graph.initializers[name], tensor, name)
        applied.add(name)
    return applied
