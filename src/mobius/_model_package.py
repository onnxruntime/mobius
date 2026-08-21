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
import shutil
import threading
from collections import UserDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import onnx_ir as ir
import rfc8785
import torch
import tqdm

from mobius._optimizations import fold_initializers_after_weights
from mobius.adapters import (
    AdapterArtifact,
    AdapterServiceOptions,
    AdapterTargetManifest,
)
from mobius.generation import PolicyComponent
from mobius.integrations._weight_loading import _assign_weight

logger = logging.getLogger(__name__)


def _adapter_dtype_name(dtype: ir.DataType) -> str:
    names = {
        ir.DataType.FLOAT16: "float16",
        ir.DataType.FLOAT: "float32",
        ir.DataType.BFLOAT16: "bfloat16",
    }
    try:
        return names[dtype]
    except KeyError as error:
        raise ValueError(
            f"adapter dtype {dtype.name} must be a floating-point adapter dtype"
        ) from error


def _adapter_source_file(artifact: AdapterArtifact) -> str:
    if artifact.source.path is None:
        raise ValueError(
            f"adapter {artifact.name!r} cannot preserve an in-memory source format"
        )
    if artifact.source.format == "onnx_adapter":
        return artifact.source.path
    raise ValueError(
        f"adapter {artifact.name!r} source format {artifact.source.format!r} "
        "cannot be preserved"
    )


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
        policy_components: dict[str, PolicyComponent] | None = None,
        adapter_artifacts: dict[str, AdapterArtifact] | None = None,
        adapter_target_manifest: AdapterTargetManifest | None = None,
        adapter_service_options: AdapterServiceOptions | None = None,
    ) -> None:
        super().__init__(models or {})
        self.config = config
        self.policy_components = dict(policy_components or {})
        self.adapter_target_manifest = adapter_target_manifest
        self.adapter_service_options = adapter_service_options or AdapterServiceOptions()
        if adapter_target_manifest is not None:
            adapter_target_manifest.validate(self.data)
        self.adapter_artifacts: dict[str, AdapterArtifact] = {}
        for name, artifact in (adapter_artifacts or {}).items():
            if name != artifact.name:
                raise ValueError(
                    f"adapter catalog key {name!r} does not match artifact name "
                    f"{artifact.name!r}"
                )
            self.add_adapter_artifact(artifact)

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
        include_policy_components: bool = True,
        include_adapter_artifacts: bool = True,
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
            include_policy_components: Save attached generation-policy ONNX
                components under ``policies/``. Defaults to ``True``.
            include_adapter_artifacts: Save attached parameter-adapter bundles
                under ``adapters/``. Defaults to ``True``.

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
            with _namespaced_symbolic_dimensions(model, f"component.{name}") as saved_model:
                if external_data == "safetensors":
                    ir.save_safetensors(
                        saved_model,
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
                    ir.save(saved_model, path, **save_kwargs)

        if include_policy_components:
            self.save_policy_components(directory, check_weights=check_weights)
        if include_adapter_artifacts:
            self.save_adapter_artifacts(directory)

    def add_policy_component(self, name: str, component: PolicyComponent) -> None:
        """Attach a reusable generation-policy graph to this package."""
        if not name or "/" in name or "\\" in name:
            raise ValueError("Policy component name must be a non-empty path segment")
        self.policy_components[name] = component

    def add_adapter_artifact(
        self, artifact: AdapterArtifact, *, validate_base: bool = True
    ) -> None:
        """Attach a model-agnostic adapter artifact to this package.

        Persistence and ONNX GenAI metadata emission intentionally remain separate
        until the runtime artifact/schema contract is finalized.
        """
        if artifact.name in self.adapter_artifacts:
            raise ValueError(f"adapter artifact {artifact.name!r} is already attached")
        if validate_base:
            artifact.validate_base(
                self.data,
                fingerprint_targets=(
                    self.adapter_target_manifest.targets
                    if self.adapter_target_manifest is not None
                    else None
                ),
            )
        if self.adapter_target_manifest is not None:
            if artifact.base_fingerprint != self.adapter_target_manifest.base_fingerprint:
                raise ValueError(
                    f"adapter {artifact.name!r} base fingerprint does not match "
                    "the authoritative target manifest"
                )
            missing = artifact.target_bindings - self.adapter_target_manifest.bindings
            if missing:
                raise ValueError(
                    f"adapter artifact {artifact.name!r} contains targets outside "
                    f"the authoritative manifest: {sorted(map(str, missing))}"
                )
        self.adapter_artifacts[artifact.name] = artifact

    def save_adapter_artifacts(self, directory: str) -> dict[str, dict[str, object]]:
        """Save exact adapter bundles and return ONNX GenAI catalog entries."""
        if not self.adapter_artifacts:
            return {}
        if self.adapter_target_manifest is None:
            raise ValueError(
                "adapter artifacts require an authoritative adapter target manifest"
            )
        self.adapter_target_manifest.validate(self.data)
        adapter_dir = os.path.join(directory, "adapters")
        os.makedirs(adapter_dir, exist_ok=True)
        descriptors = {
            descriptor.target: descriptor
            for descriptor in self.adapter_target_manifest.targets
        }
        manifest_targets = {
            target["id"]: target
            for target in self.adapter_target_manifest_metadata()["targets"]
        }
        catalog: dict[str, dict[str, object]] = {}
        identities: set[tuple[str, str]] = set()
        for artifact_index, (alias, artifact) in enumerate(
            sorted(self.adapter_artifacts.items())
        ):
            identity_version = (artifact.stable_identity, artifact.version)
            if identity_version in identities:
                raise ValueError(
                    f"adapter identity/version {identity_version[0]}@"
                    f"{identity_version[1]} must be unique"
                )
            identities.add(identity_version)
            ordered_weights = sorted(
                artifact.weights,
                key=lambda item: (
                    item.target.component,
                    item.target.parameter,
                    item.target_id or "",
                ),
            )
            dtypes = {weight.dtype for weight in artifact.weights}
            if len(dtypes) != 1:
                raise ValueError(
                    f"adapter {alias!r} has heterogeneous target dtypes, "
                    "which the ONNX GenAI artifact contract cannot represent"
                )
            rank = ordered_weights[0].rank
            alpha = ordered_weights[0].alpha
            dtype = _adapter_dtype_name(dtypes.pop())
            bindings: list[dict[str, object]] = []
            portable_targets: dict[str, dict[str, list[float]]] = {}
            for weight in ordered_weights:
                descriptor = descriptors[weight.target]
                target_id = weight.target_id or descriptor.semantic_name
                if target_id not in manifest_targets:
                    raise ValueError(
                        f"adapter {alias!r} references target ID {target_id!r} "
                        "outside the authoritative manifest"
                    )
                target_policy = manifest_targets[target_id]
                if target_policy.get("rank", weight.rank) != weight.rank:
                    raise ValueError(
                        f"adapter {alias!r} target {target_id!r} rank {weight.rank} "
                        f"violates manifest policy {target_policy['rank']}"
                    )
                if target_policy.get("alpha", weight.alpha) != weight.alpha:
                    raise ValueError(
                        f"adapter {alias!r} target {target_id!r} alpha {weight.alpha} "
                        f"violates manifest policy {target_policy['alpha']}"
                    )
                slice_policy = target_policy.get("output_slice")
                if isinstance(slice_policy, dict):
                    if slice_policy.get("rank", weight.rank) != weight.rank:
                        raise ValueError(
                            f"adapter {alias!r} target {target_id!r} rank {weight.rank} "
                            f"violates output-slice policy {slice_policy['rank']}"
                        )
                    if slice_policy.get("alpha", weight.alpha) != weight.alpha:
                        raise ValueError(
                            f"adapter {alias!r} target {target_id!r} alpha {weight.alpha} "
                            f"violates output-slice policy {slice_policy['alpha']}"
                        )
                weight_key = weight.weight_key or descriptor.semantic_name
                binding: dict[str, object] = {
                    "target": target_id,
                    "weight_key": weight_key,
                }
                if weight.rank != rank:
                    binding["rank"] = weight.rank
                if weight.alpha != alpha:
                    binding["alpha"] = weight.alpha
                bindings.append(binding)
                portable_targets[weight_key] = {
                    "a": weight.a.numpy().reshape(-1).astype("float32").tolist(),
                    "b": weight.b.numpy().reshape(-1).astype("float32").tolist(),
                }

            artifact_dir = os.path.join(directory, "adapters", alias)
            os.makedirs(artifact_dir, exist_ok=True)
            weight_artifacts: list[dict[str, object]] = []
            if self.adapter_service_options.portable_fallback:
                relative_location = f"adapters/{alias}/adapter.json"
                destination = os.path.join(directory, relative_location)
                payload = rfc8785.dumps({"targets": portable_targets})
                with open(destination, "wb") as handle:
                    handle.write(payload)
                weight_artifacts.append(
                    {
                        "location": relative_location,
                        "loader_capability": "onnx-genai.adapters.json@1",
                        "scale_encoding": "alpha_over_rank",
                        "format": "json",
                    }
                )
            if (
                self.adapter_service_options.preserve_source_format
                and artifact.source.format != "in_memory"
            ):
                if artifact.source.format == "onnx_adapter":
                    source_path = _adapter_source_file(artifact)
                    relative_location = f"adapters/{alias}/adapter.onnx_adapter"
                    destination = os.path.join(directory, relative_location)
                    shutil.copyfile(source_path, destination)
                    weight_artifacts.append(
                        {
                            "location": relative_location,
                            "loader_capability": "onnxruntime.lora-adapter@1",
                            "scale_encoding": "baked",
                            "format": "ort_genai",
                        }
                    )
                elif artifact.source.format == "peft_safetensors":
                    if artifact.source.path is None:
                        raise ValueError(f"adapter {alias!r} PEFT source path is absent")
                    source_dir = artifact.source.path
                    source_weights = os.path.join(source_dir, "adapter_model.safetensors")
                    source_config = os.path.join(source_dir, "adapter_config.json")
                    relative_location = f"adapters/{alias}/adapter_model.safetensors"
                    relative_config = f"adapters/{alias}/adapter_config.json"
                    destination = os.path.join(directory, relative_location)
                    config_destination = os.path.join(directory, relative_config)
                    shutil.copyfile(source_weights, destination)
                    shutil.copyfile(source_config, config_destination)
                    weight_artifacts.append(
                        {
                            "location": relative_location,
                            "loader_capability": "onnx-genai.adapters.hf-peft@1",
                            "config_location": relative_config,
                            "scale_encoding": "alpha_over_rank",
                            "format": "hf_peft",
                        }
                    )
                else:
                    raise ValueError(
                        f"adapter {alias!r} source format {artifact.source.format!r} "
                        "cannot be preserved"
                    )
            if not weight_artifacts:
                raise ValueError(
                    f"adapter {alias!r} must emit a portable or preserved source artifact"
                )
            provenance = {"producer": artifact.source.producer}
            if artifact.source.base_model:
                provenance["source"] = artifact.source.base_model
            if artifact.source.revision:
                provenance["revision"] = artifact.source.revision
            catalog[alias] = {
                "index": artifact_index,
                "identity": artifact.stable_identity,
                "version": artifact.version,
                "rank": rank,
                "alpha": alpha,
                "dtype": dtype,
                "provenance": provenance,
                "weights": weight_artifacts,
                "bindings": bindings,
            }
        return catalog

    def adapter_target_manifest_metadata(self) -> dict[str, object]:
        """Serialize the authoritative generic LoRA target manifest."""
        if self.adapter_target_manifest is None:
            raise ValueError("adapter target manifest is not attached")
        targets: list[dict[str, object]] = []
        for descriptor in sorted(
            self.adapter_target_manifest.targets,
            key=lambda item: item.semantic_name,
        ):
            initializer = self.data[descriptor.target.component].graph.initializers[
                descriptor.target.parameter
            ]
            activation_dtype = _adapter_dtype_name(
                descriptor.activation_dtype or initializer.dtype
            )
            base: dict[str, object] = {
                "id": descriptor.semantic_name,
                "component": descriptor.target.component,
                "initializer": descriptor.target.parameter,
                "node_name": descriptor.node_name,
                "output_name": descriptor.output_name,
                "activation_dtype": activation_dtype,
                "input_features": descriptor.input_size,
                "output_features": descriptor.output_size,
            }
            if descriptor.layer_index is not None:
                base["layer_index"] = descriptor.layer_index
            if descriptor.rank is not None:
                base["rank"] = descriptor.rank
            if descriptor.alpha is not None:
                base["alpha"] = descriptor.alpha
            if descriptor.graph_input_a is not None:
                graph_inputs = {
                    "a": descriptor.graph_input_a,
                    "b": descriptor.graph_input_b,
                }
                if descriptor.graph_input_scale is not None:
                    graph_inputs["scale"] = descriptor.graph_input_scale
                base["graph_inputs"] = graph_inputs
            targets.append(base)
            for target_slice in descriptor.slices:
                sliced = dict(base)
                sliced["id"] = f"{descriptor.semantic_name}.{target_slice.role}"
                output_slice: dict[str, object] = {
                    "role": target_slice.role,
                    "offset": target_slice.offset,
                    "width": target_slice.width,
                }
                if target_slice.rank is not None:
                    output_slice["rank"] = target_slice.rank
                if target_slice.alpha is not None:
                    output_slice["alpha"] = target_slice.alpha
                sliced["output_slice"] = output_slice
                targets.append(sliced)
        return {"targets": targets}

    def save_policy_components(
        self,
        directory: str,
        *,
        check_weights: bool = True,
    ) -> dict[str, str]:
        """Save attached policy graphs and return package-relative artifact paths."""
        if not self.policy_components:
            return {}
        policy_dir = os.path.join(directory, "policies")
        os.makedirs(policy_dir, exist_ok=True)
        artifacts: dict[str, str] = {}
        for name, component in self.policy_components.items():
            if check_weights:
                _check_weights(name, component.model)
            relative_path = f"policies/{name}.onnx"
            # Policy components are separate ONNX artifacts with public ABI
            # dimension names such as ``batch``. Namespacing those dimensions
            # makes an otherwise exact semantic contract ineligible for runtime fusion.
            ir.save(component.model, os.path.join(directory, relative_path))
            artifacts[name] = relative_path
        return artifacts

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
            package = cls(models)
            package._load_policy_components(directory)
            return package
        # Fall back to flat layout
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".onnx"):
                name = filename.removesuffix(".onnx")
                models[name] = ir.load(os.path.join(directory, filename))
        package = cls(models)
        package._load_policy_components(directory)
        return package

    def _load_policy_components(self, directory: str) -> None:
        policy_dir = os.path.join(directory, "policies")
        if not os.path.isdir(policy_dir):
            return
        for filename in sorted(os.listdir(policy_dir)):
            if not filename.endswith(".onnx"):
                continue
            model = ir.load(os.path.join(policy_dir, filename))
            self.policy_components[filename.removesuffix(".onnx")] = (
                PolicyComponent.from_model(model)
            )

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


@contextmanager
def _namespaced_symbolic_dimensions(
    model: ir.Model,
    namespace: str,
) -> Iterator[ir.Model]:
    """Namespace interface symbols and discard non-contractual intermediate aliases."""
    interface_values = {id(value) for value in (*model.graph.inputs, *model.graph.outputs)}
    values: dict[int, ir.Value] = {}
    graphs: list[ir.GraphProtocol] = []
    nodes = ir.traversal.RecursiveGraphIterator(model.graph, enter_graph=graphs.append)
    for node in nodes:
        for value in (*node.inputs, *node.outputs):
            if value is not None:
                values[id(value)] = value
    for graph in graphs:
        for value in (*graph.inputs, *graph.outputs, *graph.initializers.values()):
            values[id(value)] = value

    originals: list[tuple[ir.Value, ir.Shape]] = []
    symbols: dict[str, str] = {}
    try:
        for value in values.values():
            if value.shape is None:
                continue
            dimensions: list[int | str | ir.SymbolicDim] = []
            changed = False
            for dimension in value.shape:
                if isinstance(dimension, int):
                    dimensions.append(dimension)
                    continue
                if dimension.value is None:
                    dimensions.append(dimension)
                    continue
                if id(value) not in interface_values:
                    dimensions.append(ir.SymbolicDim(None))
                    changed = True
                    continue
                text = str(dimension)
                dimensions.append(symbols.setdefault(text, f"{namespace}.{text}"))
                changed = True
            if changed:
                originals.append((value, value.shape))
                denotations = [
                    value.shape.get_denotation(index) for index in range(len(value.shape))
                ]
                value.shape = ir.Shape(dimensions, denotations)
        yield model
    finally:
        for value, shape in originals:
            value.shape = shape


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
