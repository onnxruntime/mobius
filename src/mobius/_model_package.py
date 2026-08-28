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
import json
import logging
import math
import os
import shutil
import tempfile
import threading
from collections import UserDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx_ir as ir
import rfc8785
import torch
import tqdm

from mobius._export_report import ComponentExportReport
from mobius._optimizations import fold_initializers_after_weights
from mobius.adapters import (
    AdapterArtifact,
    AdapterServiceOptions,
    AdapterTargetManifest,
)
from mobius.generation import PolicyComponent
from mobius.integrations._weight_loading import _assign_weight

if TYPE_CHECKING:
    from mobius.integrations.gguf._quantization_report import GGUFQuantizationReport

logger = logging.getLogger(__name__)

_PACKAGE_MANIFEST = ".mobius-package.json"
_PACKAGE_MANIFEST_FORMAT = "mobius.model-package.v1"
_MTP_SIDECAR_BASENAME = ".mobius-mtp"
_QUANTIZATION_REPORT = "quantization_report.json"
_EXPORT_REPORT = "export_report.json"
_WEIGHT_LOADING_REPORT = "weight-loading-report.json"
_DEFAULT_STREAMING_DENSE_SHARD_BYTES = 1 << 30
_MAX_STREAMING_DENSE_SHARD_BYTES = 5_000_000_000


def _read_mtp_sidecar_name(directory: str) -> str | None:
    manifest_path = os.path.join(directory, _PACKAGE_MANIFEST)
    if not os.path.lexists(manifest_path):
        return None
    if os.path.islink(manifest_path):
        raise ValueError("ModelPackage manifest must not be a symlink.")
    if not os.path.isfile(manifest_path):
        raise ValueError(f"ModelPackage manifest is not a file: {manifest_path}.")
    with open(manifest_path) as file:
        manifest = json.load(file)
    sidecar_name = manifest.get("mtp_head") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != _PACKAGE_MANIFEST_FORMAT
        or not isinstance(sidecar_name, str)
        or not sidecar_name
        or sidecar_name in {".", ".."}
        or os.path.basename(sidecar_name) != sidecar_name
    ):
        raise ValueError(f"Invalid ModelPackage manifest: {manifest_path}.")
    return sidecar_name


def _write_mtp_manifest(directory: str, sidecar_name: str) -> None:
    payload = json.dumps(
        {"format": _PACKAGE_MANIFEST_FORMAT, "mtp_head": sidecar_name},
        indent=2,
        sort_keys=True,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.path.join(directory, _PACKAGE_MANIFEST), flags, 0o666)
    with os.fdopen(descriptor, "w") as file:
        file.write(payload)
        file.write("\n")


def _mtp_sidecar_name(package: ModelPackage) -> str:
    """Choose a sidecar directory that cannot hide a model component."""
    component_keys = {name.casefold() for name in package}
    name = _MTP_SIDECAR_BASENAME
    suffix = 0
    while name.casefold() in component_keys:
        suffix += 1
        name = f"{_MTP_SIDECAR_BASENAME}-{suffix}"
    return name


def _validate_mtp_chain(package: ModelPackage) -> None:
    """Reject malformed or cyclic MTP package chains before saving any files."""
    seen: set[int] = set()
    current = package
    while True:
        identity = id(current)
        if identity in seen:
            raise ValueError("ModelPackage mtp_head attachments must not contain a cycle.")
        seen.add(identity)
        component_keys: set[str] = set()
        for name in current:
            if (
                not name
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
                or os.path.isabs(name)
            ):
                raise ValueError(
                    f"ModelPackage component name {name!r} must be a non-empty path segment."
                )
            collision_key = name.casefold()
            if collision_key in component_keys:
                raise ValueError(
                    "ModelPackage component names must be distinct when compared "
                    f"case-insensitively; {name!r} collides with another component."
                )
            component_keys.add(collision_key)
            if collision_key in {
                _PACKAGE_MANIFEST.casefold(),
                _EXPORT_REPORT.casefold(),
                _QUANTIZATION_REPORT.casefold(),
                _WEIGHT_LOADING_REPORT.casefold(),
            }:
                raise ValueError(
                    f"ModelPackage component name {name!r} is reserved for package metadata."
                )
        sidecar = current.mtp_head
        if sidecar is None:
            return
        if not isinstance(sidecar, ModelPackage):
            raise TypeError("ModelPackage mtp_head must be another ModelPackage or None.")
        current = sidecar


def _validate_output_directory(
    directory: str, *, kind: str, inspect_contents: bool = True
) -> None:
    if not os.path.lexists(directory):
        return
    if os.path.islink(directory) or not os.path.isdir(directory):
        raise ValueError(f"ModelPackage {kind} output must be a real directory.")
    if inspect_contents:
        for root, directories, files in os.walk(directory):
            if any(os.path.islink(os.path.join(root, entry)) for entry in directories + files):
                raise ValueError(f"ModelPackage {kind} output must not contain symlinks.")


def _validate_mtp_save_paths(
    package: ModelPackage,
    directory: str,
    components: Callable[[str], bool] | None = None,
    *,
    external_data: str = "onnx",
    include_policy_components: bool = True,
    include_adapter_artifacts: bool = True,
) -> None:
    """Reject unsafe output paths throughout the package chain before any writes."""
    _validate_output_directory(directory, kind="package", inspect_contents=False)
    if package.export_report is not None and components is not None:
        raise ValueError(
            "Component-filtered saves cannot preserve an attached export report; "
            "save the complete package or detach and replace the report."
        )
    report_path = os.path.join(directory, _QUANTIZATION_REPORT)
    if os.path.lexists(report_path) and (
        os.path.islink(report_path) or not os.path.isfile(report_path)
    ):
        raise ValueError("ModelPackage quantization report output must be a real file.")
    if (
        package.gguf_quantization_report is None
        and package.gguf_reuse_plan is None
        and os.path.isfile(report_path)
    ):
        raise ValueError(
            "ModelPackage output contains a stale quantization_report.json, but the "
            "package being saved has no GGUF quantization report. Remove or clean the "
            "output directory before saving."
        )
    export_report_path = os.path.join(directory, _EXPORT_REPORT)
    if os.path.lexists(export_report_path) and (
        os.path.islink(export_report_path) or not os.path.isfile(export_report_path)
    ):
        raise ValueError("ModelPackage component export report output must be a real file.")
    if (
        package.export_report is None
        and package.gguf_reuse_plan is None
        and os.path.isfile(export_report_path)
    ):
        raise ValueError(
            "ModelPackage output contains a stale export_report.json, but the package "
            "being saved has no component export report. Remove or clean the output "
            "directory before saving."
        )
    if (
        package.export_report is not None
        and package.gguf_reuse_plan is None
        and (
            package.export_report.status == "partial"
            or package.export_report.runtime_validation_status == "unvalidated"
        )
        and os.path.isdir(directory)
        and os.listdir(directory)
    ):
        raise FileExistsError(
            "Partial or runtime-unvalidated component exports require an empty output "
            "directory so omitted or unverified components cannot survive from a "
            "previous package."
        )
    weight_report_path = os.path.join(directory, _WEIGHT_LOADING_REPORT)
    if os.path.lexists(weight_report_path) and (
        os.path.islink(weight_report_path) or not os.path.isfile(weight_report_path)
    ):
        raise ValueError("ModelPackage weight-loading report output must be a real file.")
    previous_sidecar_name = _read_mtp_sidecar_name(directory)
    selected = [name for name in package if components is None or components(name)]
    if len(selected) > 1:
        reserved = {"policies", "adapters"}.intersection(name.casefold() for name in selected)
        if reserved:
            name = sorted(reserved)[0]
            raise ValueError(
                f"ModelPackage component name {name!r} is reserved for package artifacts."
            )
    if len(selected) > 1:
        for name in selected:
            _validate_output_directory(
                os.path.join(directory, name),
                kind=f"component {name!r}",
            )
    else:
        for entry in os.listdir(directory) if os.path.isdir(directory) else ():
            if (
                entry == "model.onnx"
                or (
                    entry.startswith("model")
                    and (
                        entry.endswith(".data")
                        or (external_data == "safetensors" and ".safetensors" in entry)
                    )
                )
            ) and os.path.islink(os.path.join(directory, entry)):
                raise ValueError("ModelPackage model output must not be a symlink.")
    if include_policy_components and package.policy_components:
        _validate_output_directory(
            os.path.join(directory, "policies"),
            kind="policy artifact",
        )
    if include_adapter_artifacts and package.adapter_artifacts:
        _validate_output_directory(
            os.path.join(directory, "adapters"),
            kind="adapter artifact",
        )
    active_artifact_names = {
        name
        for name, active in (
            ("policies", include_policy_components and bool(package.policy_components)),
            ("adapters", include_adapter_artifacts and bool(package.adapter_artifacts)),
        )
        if active
    }
    if (
        previous_sidecar_name is not None
        and previous_sidecar_name.casefold() in active_artifact_names
    ):
        raise ValueError(
            "ModelPackage manifest MTP sidecar collides with an active artifact namespace."
        )
    if package.mtp_head is None:
        return
    sidecar_directory = os.path.join(directory, _mtp_sidecar_name(package))
    _validate_output_directory(sidecar_directory, kind="MTP sidecar")
    _validate_mtp_save_paths(
        package.mtp_head,
        sidecar_directory,
        external_data=external_data,
        include_policy_components=include_policy_components,
        include_adapter_artifacts=include_adapter_artifacts,
    )


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
        gguf_quantization_report: Optional persisted report describing GGUF
            source-storage preservation and conversion behavior.
        export_report: Optional persisted component-level report describing
            independently supported/exported and deferred/omitted package parts.
        quantization_report: Optional transient, loader-specific metadata describing
            quantization handling. The concrete report type and schema belong to the
            integration that produced it (for example, ``CompressedTensorsLoadReport``);
            they are not a stable shared ``ModelPackage`` API and are not persisted by
            :meth:`save`.
        mtp_head: Optional nested MTP sidecar package. :meth:`save` records its
            collision-free directory in an explicit package manifest, and
            :meth:`load` restores it automatically.
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
        self.gguf_quantization_report: GGUFQuantizationReport | None = None
        self.gguf_architecture = ""
        self.gguf_artifact_identity: Any = None
        self.gguf_execution_provider = ""
        self.gguf_import_route = ""
        self.gguf_source_filename = ""
        self.gguf_source_identity: object | None = None
        self.gguf_source_path = ""
        self.gguf_tokenizer_verdict: Any = None
        self.export_report: ComponentExportReport | None = None
        self.quantization_report: object | None = None
        self.weight_loading_report: dict[str, object] | None = None
        # Optional persistence policy attached by the GGUF importer.
        self.gguf_reuse_plan: Any = None
        self.draft_manifest: Any = None
        self.mtp_head: ModelPackage | None = None
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
        _atomic_export_report: bool = True,
    ) -> None:
        """Save all component models to a directory.

        When the package contains a single model, it is saved directly as
        ``model.onnx`` in *directory*.  When multiple models are present,
        each is saved in its own subfolder as ``{name}/model.onnx``. An attached
        :attr:`mtp_head` package is saved under a collision-free directory named
        by ``.mobius-package.json``.

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
        _validate_mtp_chain(self)
        _validate_mtp_save_paths(
            self,
            directory,
            components,
            external_data=external_data,
            include_policy_components=include_policy_components,
            include_adapter_artifacts=include_adapter_artifacts,
        )
        if external_data not in {"onnx", "safetensors"}:
            raise ValueError(
                f"Unknown external_data format {external_data!r}. "
                "Expected 'onnx' or 'safetensors'."
            )
        if max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {max_workers}.")
        reuse_plan = getattr(self, "gguf_reuse_plan", None)
        if self.export_report is not None and _atomic_export_report and reuse_plan is None:
            destination = os.path.abspath(directory)
            if os.path.lexists(destination):
                raise FileExistsError(
                    "Component-report package destination already exists; refusing "
                    "non-atomic replacement."
                )
            parent = os.path.dirname(destination)
            os.makedirs(parent, exist_ok=True)
            stage = tempfile.mkdtemp(
                prefix=f".{os.path.basename(destination)}.",
                suffix=".tmp",
                dir=parent,
            )
            try:
                self.save(
                    stage,
                    external_data=external_data,
                    max_shard_size_bytes=max_shard_size_bytes,
                    max_workers=max_workers,
                    components=components,
                    progress_bar=progress_bar,
                    check_weights=check_weights,
                    include_policy_components=include_policy_components,
                    include_adapter_artifacts=include_adapter_artifacts,
                    _atomic_export_report=False,
                )
                if self.draft_manifest is not None:
                    from mobius.integrations.gguf._draft import write_draft_manifest

                    write_draft_manifest(self.draft_manifest, stage)
                from mobius.integrations.gguf._runtime_package import (
                    _publish_directory_no_replace,
                )

                _publish_directory_no_replace(Path(stage), Path(destination))
            finally:
                if os.path.isdir(stage):
                    shutil.rmtree(stage)
            return
        if reuse_plan is not None:
            if external_data != "onnx":
                raise ValueError(
                    "GGUF weight reuse requires external_data='onnx'; safetensors "
                    "cannot preserve arbitrary GGUF byte ranges."
                )
            if max_shard_size_bytes is not None:
                raise ValueError(
                    "GGUF weight reuse writes one converted-weight sidecar and does "
                    "not support max_shard_size_bytes."
                )
        previous_sidecar_name = _read_mtp_sidecar_name(directory)
        os.makedirs(directory, exist_ok=True)
        quantization_report_path = os.path.join(directory, _QUANTIZATION_REPORT)
        if self.gguf_quantization_report is not None and reuse_plan is None:
            self.gguf_quantization_report.write_json(quantization_report_path)
        export_report_path = os.path.join(directory, _EXPORT_REPORT)
        selected = {
            name: model
            for name, model in self.data.items()
            if components is None or components(name)
        }
        report = self.weight_loading_report
        dense_stream = (
            report is not None
            and report.get("output_weight_format") == "dense"
            and report.get("native_fp8") is False
        )
        preserved_stream = report is not None and report.get("streaming_external_data") is True
        if dense_stream or preserved_stream:
            assert report is not None
            # onnx-ir may materialize external-data shards concurrently. Keep
            # this fallback serial so the documented bound remains one output
            # shard plus one reconstructed tensor instead of max_workers shards.
            max_workers = 1
            if max_shard_size_bytes is None:
                max_shard_size_bytes = _DEFAULT_STREAMING_DENSE_SHARD_BYTES
            if max_shard_size_bytes > _MAX_STREAMING_DENSE_SHARD_BYTES:
                raise ValueError(
                    "Streaming dense fallback packages require "
                    f"max_shard_size_bytes <= {_MAX_STREAMING_DENSE_SHARD_BYTES}; "
                    f"got {max_shard_size_bytes}. The serializer buffers one output "
                    "shard before flushing it."
                )
            largest_tensor_bytes = 0
            for model in selected.values():
                for initializer in model.graph.initializers.values():
                    if initializer.shape is None or initializer.dtype is None:
                        continue
                    num_elements = math.prod(int(dim) for dim in initializer.shape)
                    tensor_bytes = (num_elements * initializer.dtype.bitwidth + 7) // 8
                    largest_tensor_bytes = max(largest_tensor_bytes, tensor_bytes)
            self.weight_loading_report = {
                **report,
                "external_data_shard_limit_bytes": max_shard_size_bytes,
                "largest_dense_tensor_bytes": largest_tensor_bytes,
                "largest_reconstruction_working_set_bytes": max(
                    int(
                        report.get(
                            "largest_reconstruction_working_set_bytes",
                            0,
                        )
                    ),
                    int(report.get("largest_source_cast_overlap_bytes", 0)),
                    largest_tensor_bytes,
                ),
                "serializer_max_workers": max_workers,
                "serialization_memory_bound": (
                    "one output shard plus the largest source/reconstruction "
                    "working set and serializer overhead; "
                    "external-data serialization is forced to one worker"
                ),
            }
        use_subfolders = len(selected) > 1
        if reuse_plan is not None and (len(selected) != 1 or use_subfolders):
            raise ValueError("GGUF weight reuse currently supports one flat ONNX model only.")

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
                if reuse_plan is not None:
                    from mobius.integrations.gguf._reuse import save_reuse_package

                    package_metadata: dict[str, bytes] = {}
                    if self.gguf_quantization_report is not None:
                        package_metadata[_QUANTIZATION_REPORT] = (
                            json.dumps(
                                self.gguf_quantization_report.to_dict(),
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n"
                        ).encode()
                    if self.export_report is not None:
                        package_metadata[_EXPORT_REPORT] = (
                            self.export_report.to_json().encode()
                        )
                    if self.draft_manifest is not None:
                        package_metadata["draft_manifest.json"] = (
                            json.dumps(self.draft_manifest, indent=2, sort_keys=True) + "\n"
                        ).encode()
                    save_reuse_package(
                        saved_model,
                        path,
                        reuse_plan,
                        callback=callback,
                        package_metadata=package_metadata,
                    )
                elif external_data == "safetensors":
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

        if reuse_plan is None:
            # Re-saving a loaded reuse package through the ordinary saver copies
            # all weights into ONNX external data, so any old reuse manifest in
            # the destination would become a false provenance claim.
            stale_reuse_manifest = os.path.join(directory, "gguf-reuse.json")
            if os.path.isfile(stale_reuse_manifest):
                os.remove(stale_reuse_manifest)
        report_path = os.path.join(directory, _WEIGHT_LOADING_REPORT)
        if self.weight_loading_report is not None:
            if os.path.islink(report_path):
                raise ValueError("Weight-loading report must not be a symlink.")
            with open(report_path, "w", encoding="utf-8") as file:
                json.dump(self.weight_loading_report, file, indent=2, sort_keys=True)
                file.write("\n")
        elif os.path.isfile(report_path):
            os.remove(report_path)
        if include_policy_components:
            self.save_policy_components(directory, check_weights=check_weights)
        if include_adapter_artifacts:
            self.save_adapter_artifacts(directory)
        if self.mtp_head is not None:
            sidecar_name = _mtp_sidecar_name(self)
            sidecar_directory = os.path.join(directory, sidecar_name)
            if os.path.isdir(sidecar_directory):
                shutil.rmtree(sidecar_directory)
            self.mtp_head.save(
                sidecar_directory,
                external_data=external_data,
                max_shard_size_bytes=max_shard_size_bytes,
                max_workers=max_workers,
                progress_bar=progress_bar,
                check_weights=check_weights,
                include_policy_components=include_policy_components,
                include_adapter_artifacts=include_adapter_artifacts,
            )
            _write_mtp_manifest(directory, sidecar_name)
        else:
            sidecar_name = None
            manifest_path = os.path.join(directory, _PACKAGE_MANIFEST)
            if os.path.isfile(manifest_path):
                os.remove(manifest_path)
        if previous_sidecar_name is not None and previous_sidecar_name != sidecar_name:
            previous_sidecar = os.path.join(directory, previous_sidecar_name)
            current_directories = (
                [os.path.join(directory, name) for name in selected] if use_subfolders else []
            )
            if sidecar_name is not None:
                current_directories.append(os.path.join(directory, sidecar_name))
            if include_policy_components and self.policy_components:
                current_directories.append(os.path.join(directory, "policies"))
            if include_adapter_artifacts and self.adapter_artifacts:
                current_directories.append(os.path.join(directory, "adapters"))
            aliases_current_output = any(
                os.path.exists(current)
                and os.path.exists(previous_sidecar)
                and os.path.samefile(previous_sidecar, current)
                for current in current_directories
            )
            if (
                not aliases_current_output
                and os.path.isdir(previous_sidecar)
                and not os.path.islink(previous_sidecar)
            ):
                shutil.rmtree(previous_sidecar)
        if self.export_report is not None and reuse_plan is None:
            if os.path.islink(export_report_path):
                raise ValueError("Component export report must not be a symlink.")
            self.export_report.write_json(export_report_path)

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

        An MTP sidecar named by ``.mobius-package.json`` is restored as
        :attr:`mtp_head`. Unmarked subdirectories, including ``mtp/``, remain
        ordinary model components.

        Args:
            directory: Path to the directory containing models.

        Returns:
            A new ``ModelPackage`` with one entry per model found.
        """
        return cls._load(directory, frozenset())

    @classmethod
    def _load(cls, directory: str, ancestors: frozenset[str]) -> ModelPackage:
        resolved_directory = os.path.realpath(directory)
        if resolved_directory in ancestors:
            raise ValueError("ModelPackage MTP sidecar manifests must not contain a cycle.")
        ancestors = ancestors | {resolved_directory}
        sidecar_name = _read_mtp_sidecar_name(directory)
        mtp_dir: str | None = None
        if sidecar_name is not None:
            mtp_dir = os.path.join(directory, sidecar_name)
            if not os.path.isdir(mtp_dir):
                raise ValueError(
                    f"ModelPackage manifest references missing MTP sidecar {sidecar_name!r}."
                )
            if os.path.islink(mtp_dir):
                raise ValueError("ModelPackage MTP sidecar directory must not be a symlink.")
            resolved_sidecar = os.path.realpath(mtp_dir)
            if os.path.dirname(resolved_sidecar) != resolved_directory:
                raise ValueError("ModelPackage MTP sidecar must remain inside its package.")

        models: dict[str, ir.Model] = {}
        for entry in sorted(os.listdir(directory)):
            subdir = os.path.join(directory, entry)
            if (
                mtp_dir is not None
                and os.path.isdir(subdir)
                and os.path.samefile(subdir, mtp_dir)
            ):
                continue
            model_path = os.path.join(subdir, "model.onnx")
            if os.path.isdir(subdir) and os.path.isfile(model_path):
                models[entry] = ir.load(model_path)
        if not models:
            for filename in sorted(os.listdir(directory)):
                if filename.endswith(".onnx"):
                    name = filename.removesuffix(".onnx")
                    models[name] = ir.load(os.path.join(directory, filename))
        package = cls(models)
        quantization_report_path = os.path.join(directory, _QUANTIZATION_REPORT)
        if os.path.lexists(quantization_report_path):
            if os.path.islink(quantization_report_path) or not os.path.isfile(
                quantization_report_path
            ):
                raise ValueError("ModelPackage quantization report must be a real file.")
            from mobius.integrations.gguf._quantization_report import GGUFQuantizationReport

            package.gguf_quantization_report = GGUFQuantizationReport.read_json(
                quantization_report_path
            )
        export_report_path = os.path.join(directory, _EXPORT_REPORT)
        if os.path.lexists(export_report_path):
            if os.path.islink(export_report_path) or not os.path.isfile(export_report_path):
                raise ValueError("ModelPackage component export report must be a real file.")
            package.export_report = ComponentExportReport.read_json(export_report_path)
        report_path = os.path.join(directory, _WEIGHT_LOADING_REPORT)
        if os.path.lexists(report_path):
            if os.path.islink(report_path) or not os.path.isfile(report_path):
                raise ValueError("Weight-loading report must be a regular file.")
            with open(report_path, encoding="utf-8") as file:
                report = json.load(file)
            if (
                not isinstance(report, dict)
                or report.get("format") != "mobius.weight-loading-report.v1"
            ):
                raise ValueError("Invalid weight-loading report.")
            package.weight_loading_report = report
        package._load_policy_components(directory)
        if mtp_dir is not None:
            package.mtp_head = cls._load(mtp_dir, ancestors)
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
        *,
        fold_constants: bool = True,
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

        if not fold_constants:
            return

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
