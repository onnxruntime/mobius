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

import inspect
import logging
import os
import threading
from collections import UserDict
from collections.abc import Callable
from typing import Any

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

    def __setitem__(self, name: str, model: ir.Model) -> None:
        """Store a component whose ONNX graph is available for downstream tooling."""
        if not isinstance(model, ir.Model) or model.graph is None:
            raise TypeError(
                f"ModelPackage component {name!r} must be an ir.Model with a graph"
            )
        super().__setitem__(name, model)

    # -- Persistence -------------------------------------------------------

    def save(
        self,
        directory: str,
        *,
        external_data: str = "onnx",
        max_shard_size_bytes: int | None = None,
        max_workers: int = 8,
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
            max_shard_size_bytes: Maximum external-data shard size in bytes.
                Used by both ONNX and safetensors external-data formats. A
                single tensor larger than this value is written in its own
                oversized shard.
            max_workers: Number of threads used to write ONNX external data.
                Defaults to 8. Set to 1 to save serially. Older ``onnx_ir``
                versions that do not support concurrent saves fall back to
                serial behavior.
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
                ``"safetensors"``, if *max_workers* is not positive, or if
                *check_weights* is ``True`` and any initializer is missing its
                ``const_value``.
        """
        if external_data not in {"onnx", "safetensors"}:
            raise ValueError(
                f"Unknown external_data format {external_data!r}. "
                "Expected 'onnx' or 'safetensors'."
            )
        if max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {max_workers}.")
        os.makedirs(directory, exist_ok=True)
        selected = {
            name: model
            for name, model in self.data.items()
            if components is None or components(name)
        }
        use_subfolders = len(selected) > 1

        for name, model in selected.items():
            callback = _make_progress_callback() if progress_bar else None
            if check_weights:
                _check_weights(name, model)
            if use_subfolders:
                model_dir = os.path.join(directory, name)
                os.makedirs(model_dir, exist_ok=True)
            else:
                model_dir = directory
            path = os.path.join(model_dir, "model.onnx")
            if external_data == "safetensors":
                ir.save_safetensors(
                    model,
                    path,
                    max_shard_size_bytes=max_shard_size_bytes,
                    callback=callback,
                )
            else:
                save_kwargs: dict[str, Any] = {
                    "external_data": "model.onnx.data",
                    "max_shard_size_bytes": max_shard_size_bytes,
                    "callback": callback,
                }
                if "max_workers" in inspect.signature(ir.save).parameters:
                    save_kwargs["max_workers"] = max_workers
                ir.save(model, path, **save_kwargs)

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
    """Create a thread-safe tqdm progress-bar callback for ``ir.save``.

    Newer ``onnx_ir`` versions may invoke callbacks concurrently and out of
    index order. Count invocations instead of tracking ``metadata.index`` and
    derive each bar's position from its shard filename so rendering order is
    deterministic. Serialize all progress-bar mutations. This remains compatible
    with ``onnx_ir`` 1.0, where callbacks are invoked serially.
    """
    lock = threading.Lock()
    bars: dict[str, tqdm.tqdm] = {}

    def callback(tensor: ir.TensorProtocol, metadata: ir.external_data.CallbackInfo) -> None:
        with lock:
            shard_total = getattr(metadata, "shard_total", None)
            key = metadata.filename if shard_total is not None else "__all__"
            pbar = bars.get(key)
            if pbar is None:
                description = (
                    f"Saving {metadata.filename}"
                    if shard_total is not None
                    else "Saving external data"
                )
                position = 0
                if shard_total is not None:
                    shard_prefix, separator, _ = metadata.filename.rpartition("-of-")
                    shard_number = shard_prefix.rpartition("-")[2]
                    if separator and shard_number.isdigit():
                        position = int(shard_number) - 1
                pbar = tqdm.tqdm(
                    total=shard_total if shard_total is not None else metadata.total,
                    desc=description,
                    position=position,
                    leave=True,
                )
                bars[key] = pbar
            pbar.update()
            pbar.set_postfix_str(
                f"{tensor.name} ({tensor.dtype.short_name()}, {tensor.shape})"
            )
            if pbar.n >= pbar.total:
                pbar.close()

    return callback


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
