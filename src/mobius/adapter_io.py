# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Load standard adapter sources into the model-agnostic Mobius representation."""

from __future__ import annotations

__all__ = [
    "adapter_source_from_onnx_adapter",
    "attach_peft_adapter",
    "load_peft_adapter",
]

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnx_ir.passes import common as common_passes
from safetensors.numpy import load_file

from mobius._constants import (
    LAZY_SOURCE_FILES_META_KEY,
    LEGACY_QKV_PROJECTION_SOURCES_METADATA_KEY,
    QKV_PROJECTION_SOURCES_METADATA_KEY,
)
from mobius.adapters import (
    AdapterArtifact,
    AdapterServiceOptions,
    AdapterSource,
    AdapterTarget,
    AdapterTargetDescriptor,
    AdapterTargetManifest,
    AdapterWeights,
    fingerprint_model_weights,
)

if TYPE_CHECKING:
    from mobius._model_package import ModelPackage

_PEFT_CONFIG = "adapter_config.json"
_PEFT_WEIGHTS = "adapter_model.safetensors"
_FACTOR_PATTERN = re.compile(r"^(?P<module>.+)\.lora_(?P<factor>[AB])(?:\.[^.]+)?\.weight$")


def _source_checksum(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _pattern_value(patterns: Mapping[str, object], module_key: str, default: object) -> object:
    matches = [(key, value) for key, value in patterns.items() if module_key.endswith(key)]
    if not matches:
        return default
    return max(matches, key=lambda item: len(item[0]))[1]


def _resolve_target(
    module_key: str, targets: Mapping[str, AdapterTarget]
) -> tuple[str, AdapterTarget]:
    if module_key in targets:
        return module_key, targets[module_key]
    matches = [(name, target) for name, target in targets.items() if module_key.endswith(name)]
    if len(matches) != 1:
        raise ValueError(
            f"PEFT module {module_key!r} resolves to {len(matches)} producer targets; "
            "provide one exact or unique suffix binding"
        )
    return matches[0]


def load_peft_adapter(
    directory: str | Path,
    *,
    target_bindings: Mapping[str, AdapterTarget],
    base_fingerprint: str,
    name: str | None = None,
) -> AdapterArtifact:
    """Load PEFT config/safetensors using producer-declared, model-agnostic bindings."""
    directory = Path(directory)
    config_path = directory / _PEFT_CONFIG
    weights_path = directory / _PEFT_WEIGHTS
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError(
            f"PEFT adapter directory {directory} must contain "
            f"{_PEFT_CONFIG} and {_PEFT_WEIGHTS}"
        )
    config = json.loads(config_path.read_text())
    tensors = load_file(weights_path)
    target_modules = tuple(config.get("target_modules", ()))
    if not target_modules:
        raise ValueError("PEFT adapter target_modules must not be empty")
    default_rank = int(config.get("r", 0))
    default_alpha = float(config.get("lora_alpha", default_rank))
    rank_pattern = config.get("rank_pattern", {})
    alpha_pattern = config.get("alpha_pattern", {})

    pending: dict[str, dict[str, np.ndarray]] = {}
    for key, tensor in tensors.items():
        match = _FACTOR_PATTERN.match(key)
        if match is None:
            continue
        module_key = match.group("module")
        if not any(module_key.endswith(target) for target in target_modules):
            raise ValueError(f"PEFT tensor {key!r} is not covered by target_modules")
        pending.setdefault(module_key, {})[match.group("factor")] = tensor
    if not pending:
        raise ValueError("PEFT adapter contains no LoRA A/B tensors")

    loaded_weights: list[AdapterWeights] = []
    for module_key, factors in sorted(pending.items()):
        if set(factors) != {"A", "B"}:
            raise ValueError(f"PEFT module {module_key!r} must contain paired A/B factors")
        a = np.ascontiguousarray(factors["A"])
        b = np.ascontiguousarray(factors["B"])
        rank = int(_pattern_value(rank_pattern, module_key, default_rank))
        alpha = float(_pattern_value(alpha_pattern, module_key, default_alpha))
        if rank <= 0 or not math.isfinite(alpha):
            raise ValueError(
                f"PEFT module {module_key!r} has invalid rank/alpha {rank}/{alpha}"
            )
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(
                f"PEFT module {module_key!r} factors must have shapes "
                f"[rank,K]/[N,rank] for rank {rank}, got {a.shape}/{b.shape}"
            )
        weight_key, target = _resolve_target(module_key, target_bindings)
        loaded_weights.append(
            AdapterWeights(
                target,
                ir.tensor(a),
                ir.tensor(b),
                alpha,
                weight_key=weight_key,
            )
        )

    source = AdapterSource(
        "peft_safetensors",
        path=str(directory),
        checksum=_source_checksum([config_path, weights_path]),
        base_model=config.get("base_model_name_or_path"),
        revision=config.get("revision"),
    )
    return AdapterArtifact(
        name=name or directory.name,
        base_fingerprint=base_fingerprint,
        weights=tuple(loaded_weights),
        source=source,
    )


def _peft_module_keys(tensors: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                match.group("module")
                for key in tensors
                if (match := _FACTOR_PATTERN.match(key)) is not None
            }
        )
    )


def _packed_projection_sources(node: ir.Node, metadata: str) -> list[tuple[str, int]]:
    record: object = json.loads(metadata)
    if (
        not isinstance(record, dict)
        or type(record.get("version")) is not int
        or record["version"] != 1
    ):
        raise ValueError(f"unsupported packed-projection metadata on {node.name!r}")
    names, widths = record.get("weights"), record.get("widths")
    if (
        not isinstance(names, list)
        or not isinstance(widths, list)
        or len(names) != len(widths)
        or len(names) not in (2, 3)
    ):
        raise ValueError(f"invalid packed-projection source list on {node.name!r}")
    sources = []
    for name, width in zip(names, widths, strict=True):
        if not isinstance(name, str) or not name or type(width) is not int or width <= 0:
            raise ValueError(f"invalid packed-projection source on {node.name!r}")
        original = name.removesuffix("_t") if name.endswith(".weight_t") else name
        sources.append((original, width))
    return sources


def _packed_float_projection(
    model: ir.Model, split: ir.Node, source_count: int, module_key: str
) -> ir.Node:
    """Require the packer's MatMul-to-Split topology before restoring PEFT targets.

    The packed activation must be private to Split; replacing a shared or public
    activation could change consumers outside the adapter's projection group.
    """
    packed = split.inputs[0] if split.inputs else None
    projection = packed.producer() if packed is not None else None
    if (
        split.domain != ""
        or split.op_type != "Split"
        or split.overload
        or split.attributes.get_int("axis", 0) not in (-1, 2)
        or len(split.outputs) != source_count
        or packed is None
        or projection is None
        or projection.domain != ""
        or projection.op_type != "MatMul"
        or projection.overload
        or projection.attributes
        or len(projection.inputs) != 2
        or len(projection.outputs) != 1
        or packed in model.graph.outputs
        or len(packed.uses()) != 1
    ):
        raise ValueError(f"PEFT module {module_key!r} has an unsupported packed projection")
    return projection


def _loaded_packed_projection_inputs(
    model: ir.Model, projection: ir.Node, total_width: int, module_key: str
) -> tuple[ir.Value, ir.Value]:
    """Require a loaded static float weight before lazily slicing its columns.

    Overridable, quantized, or symbolic weights cannot be restored by this
    float-only path. Validate without materializing the packed payload.
    """
    x, weight = projection.inputs
    if (
        x is None
        or weight is None
        or not weight.is_initializer()
        or weight in model.graph.inputs
        or weight.const_value is None
        or weight.shape is None
        or len(weight.shape) != 2
        or weight.dtype not in (ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16)
        or not all(isinstance(dim, int) and dim > 0 for dim in weight.shape)
        or weight.shape[1] != total_width
    ):
        raise ValueError(
            f"PEFT module {module_key!r} requires a loaded, folded packed projection weight"
        )
    return x, weight


def _restore_packed_peft_projections(
    model: ir.Model, module_key: str, candidates: list[str]
) -> None:
    """Restore independent float projections before attaching a PEFT adapter."""
    suffixes = tuple(
        suffix
        for candidate in candidates
        for suffix in (f"{candidate}.weight", f"{candidate}.weight_t")
    )
    groups = []
    for node in model.graph:
        metadata = node.metadata_props.get(QKV_PROJECTION_SOURCES_METADATA_KEY)
        if metadata is None:
            # Read previews written before adopting the persistent-key namespace.
            metadata = node.metadata_props.get(LEGACY_QKV_PROJECTION_SOURCES_METADATA_KEY)
        if metadata is None:
            continue
        sources = _packed_projection_sources(node, metadata)
        if any(name.endswith(suffixes) for name, _ in sources):
            groups.append((node, sources))
    if not groups:
        return
    if (
        len(groups) != 1
        or sum(name.endswith(suffixes) for name, _ in groups[0][1]) != 1
        or any(name.endswith(suffixes) for name in model.graph.initializers)
    ):
        raise ValueError(f"PEFT module {module_key!r} has ambiguous packed projection targets")
    split, sources = groups[0]
    projection = _packed_float_projection(model, split, len(sources), module_key)
    x, weight = _loaded_packed_projection_inputs(
        model, projection, sum(width for _, width in sources), module_key
    )
    assert weight.shape is not None and weight.const_value is not None
    names = [name for name, _ in sources]
    if len(set(names)) != len(names) or any(
        name in model.graph.initializers for name in names
    ):
        raise ValueError(f"PEFT module {module_key!r} has conflicting source weights")
    if any(
        output.shape is None
        or len(output.shape) != 3
        or output.shape[-1] != width
        or output.dtype != weight.dtype
        for output, (_, width) in zip(split.outputs, sources, strict=True)
    ):
        raise ValueError(f"PEFT module {module_key!r} has inconsistent packed output metadata")
    nodes: list[ir.Node] = []
    outputs = []
    initializers = []
    start = 0
    tensor = weight.const_value
    assert weight.dtype is not None
    for original, (name, width) in zip(split.outputs, sources, strict=True):
        end = start + width
        parameter_shape = ir.Shape([width, weight.shape[0]])
        parameter = ir.Value(name=name, type=weight.type, shape=parameter_shape)

        def load_projection(
            source: ir.TensorProtocol = tensor, lo: int = start, hi: int = end
        ) -> ir.TensorProtocol:
            return ir.tensor(np.ascontiguousarray(source.numpy()[:, lo:hi].T))

        parameter.const_value = ir.LazyTensor(
            load_projection, dtype=weight.dtype, shape=parameter_shape, name=name
        )
        # RemoveUnusedNodes drops the packed ExternalTensor from the graph.
        # Keep its file ownership visible to ModelPackage's existing pre-write
        # alias guard without materializing the full packed checkpoint.
        source_files = (
            (tensor.path,)
            if isinstance(tensor, ir.ExternalTensor)
            else tensor.meta.get(LAZY_SOURCE_FILES_META_KEY, ())
        )
        if source_files:
            parameter.const_value.meta[LAZY_SOURCE_FILES_META_KEY] = tuple(source_files)
            # Folding may replace this tensor; ownership belongs to the model too.
            model.meta[LAZY_SOURCE_FILES_META_KEY] = frozenset(
                (*model.meta.get(LAZY_SOURCE_FILES_META_KEY, ()), *source_files)
            )
        initializers.append(parameter)
        transposed = ir.Value(type=weight.type, shape=ir.Shape([weight.shape[0], width]))
        nodes.append(
            ir.Node(
                "",
                "Transpose",
                [parameter],
                {"perm": ir.AttrInt64s("perm", [1, 0])},
                outputs=[transposed],
            )
        )
        output = ir.Value(
            name=original.name,
            shape=original.shape,
            type=original.type,
            doc_string=original.doc_string,
        )
        output.metadata_props.update(original.metadata_props)
        output.meta.update(original.meta)
        outputs.append(output)
        nodes.append(ir.Node("", "MatMul", [x, transposed], outputs=[output]))
        start = end
    for initializer in initializers:
        model.graph.initializers.add(initializer)
    ir.convenience.replace_nodes_and_values(
        model.graph,
        insertion_point=projection,
        old_nodes=[projection, split],
        new_nodes=nodes,
        old_values=list(split.outputs),
        new_values=outputs,
    )
    common_passes.RemoveUnusedNodesPass()(model)


def _resolve_initializer_name(model: ir.Model, module_key: str) -> str:
    candidates = [module_key]
    for prefix in ("base_model.model.", "base_model.", "model."):
        if module_key.startswith(prefix):
            candidates.append(module_key[len(prefix) :])
    _restore_packed_peft_projections(model, module_key, candidates)
    matches = [
        name
        for name in model.graph.initializers
        if any(
            name.endswith((f"{candidate}.weight", f"{candidate}.weight_t"))
            for candidate in candidates
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"PEFT module {module_key!r} resolves to {len(matches)} ONNX initializers: "
            f"{matches}"
        )
    return matches[0]


def _restore_peft_weight_orientation(
    model: ir.Model, initializer_name: str
) -> tuple[str, ir.Node]:
    initializer = model.graph.initializers[initializer_name]
    consumers = list(initializer.uses())
    if initializer_name.endswith(".weight_t"):
        if len(consumers) != 1:
            raise ValueError(f"folded PEFT target {initializer_name!r} must have one consumer")
        consumer, input_index = consumers[0]
        if consumer.op_type != "MatMul":
            raise ValueError(
                f"folded PEFT target {initializer_name!r} is consumed by "
                f"{consumer.op_type}, expected MatMul"
            )
        original_name = initializer_name.removesuffix("_t")
        assert initializer.const_value is not None
        original_array = np.ascontiguousarray(initializer.const_value.numpy().T)
        original = ir.Value(
            name=original_name,
            const_value=ir.tensor(original_array, name=original_name),
            type=ir.TensorType(initializer.dtype),
            shape=ir.Shape(original_array.shape),
        )
        transpose = ir.Node(
            "",
            "Transpose",
            inputs=[original],
            attributes=[ir.Attr("perm", ir.AttributeType.INTS, [1, 0])],
            num_outputs=1,
            name=f"{consumer.name}/adapter_weight_transpose",
        )
        transposed = transpose.outputs[0]
        transposed.name = f"{original_name}.transposed"
        transposed.dtype = initializer.dtype
        transposed.shape = initializer.shape
        consumer.replace_input_with(input_index, transposed)
        model.graph.insert_before(consumer, transpose)
        del model.graph.initializers[initializer_name]
        model.graph.initializers.add(original)
        return original_name, transpose

    if len(consumers) != 1:
        raise ValueError(f"PEFT target {initializer_name!r} must have one consumer")
    consumer, _ = consumers[0]
    if consumer.op_type != "Transpose":
        raise ValueError(
            f"PEFT target {initializer_name!r} is consumed by {consumer.op_type}, "
            "expected Transpose"
        )
    return initializer_name, consumer


def attach_peft_adapter(
    package: ModelPackage,
    directory: str | Path,
    *,
    component: str = "model",
    name: str | None = None,
    max_adapters: int = 1,
    cache_max_entries: int = 1,
    preserve_source_format: bool = True,
) -> AdapterArtifact:
    """Attach a real PEFT LoRA to a weighted package and publish exact ONNX targets.

    Weight folding transposes ``Linear`` parameters into MatMul-ready ``*_t``
    initializers. PEFT factors use the original ``[out, in]`` orientation, so
    this restores the explicit Transpose node before producing the target
    manifest. The resulting base graph remains numerically identical while the
    adapter ABI can address standard PEFT A/B factors without family-specific
    aliases.
    """
    directory = Path(directory)
    config = json.loads((directory / _PEFT_CONFIG).read_text())
    tensors = load_file(directory / _PEFT_WEIGHTS)
    model = package[component]
    rank = int(config.get("r", 0))
    alpha = float(config.get("lora_alpha", rank))
    descriptors: list[AdapterTargetDescriptor] = []
    bindings: dict[str, AdapterTarget] = {}
    for module_key in _peft_module_keys(tensors):
        initializer_name = _resolve_initializer_name(model, module_key)
        parameter_name, transpose = _restore_peft_weight_orientation(model, initializer_name)
        initializer = model.graph.initializers[parameter_name]
        output_size, input_size = (int(dimension) for dimension in initializer.shape)
        semantic_name = module_key
        for prefix in ("base_model.model.", "base_model.", "model."):
            if semantic_name.startswith(prefix):
                semantic_name = semantic_name[len(prefix) :]
                break
        layer_match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", semantic_name)
        target = AdapterTarget(component, parameter_name)
        descriptor = AdapterTargetDescriptor(
            target,
            semantic_name=semantic_name,
            node_name=transpose.name,
            output_name=transpose.outputs[0].name,
            input_size=input_size,
            output_size=output_size,
            layer_index=int(layer_match.group(1)) if layer_match else None,
            rank=rank,
            alpha=alpha,
            activation_dtype=initializer.dtype,
        )
        descriptors.append(descriptor)
        bindings[module_key] = target

    manifest_targets = tuple(descriptors)
    fingerprint = fingerprint_model_weights(package, manifest_targets)
    package.adapter_target_manifest = AdapterTargetManifest(fingerprint, manifest_targets)
    package.adapter_service_options = AdapterServiceOptions(
        max_adapters=max_adapters,
        cache_max_entries=cache_max_entries,
        preserve_source_format=preserve_source_format,
    )
    artifact = load_peft_adapter(
        directory,
        target_bindings=bindings,
        base_fingerprint=fingerprint,
        name=name,
    )
    cast_weights = []
    casted = False
    for weight in artifact.weights:
        target_dtype = model.graph.initializers[weight.target.parameter].dtype
        assert target_dtype is not None
        if weight.dtype == target_dtype:
            cast_weights.append(weight)
            continue
        cast_weights.append(
            AdapterWeights(
                weight.target,
                ir.tensor(weight.a.numpy().astype(target_dtype.numpy())),
                ir.tensor(weight.b.numpy().astype(target_dtype.numpy())),
                weight.alpha,
                weight_key=weight.weight_key,
                target_id=weight.target_id,
            )
        )
        casted = True
    if casted:
        artifact = AdapterArtifact(
            artifact.name,
            artifact.base_fingerprint,
            tuple(cast_weights),
            source=artifact.source,
            identity=artifact.identity,
            version=artifact.version,
        )
    package.add_adapter_artifact(artifact)
    return artifact


def adapter_source_from_onnx_adapter(
    path: str | Path,
) -> AdapterSource:
    """Declare an ORT FlatBuffers adapter source without making it mandatory."""
    path = Path(path)
    payload = path.read_bytes()
    if len(payload) < 8 or payload[4:8] != b"TORT":
        raise ValueError(f"ONNX adapter {path} does not contain the TORT identifier")
    return AdapterSource(
        "onnx_adapter",
        path=str(path),
        checksum=f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )
