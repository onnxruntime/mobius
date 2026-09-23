# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configure and load independently quantized model-package components."""

from __future__ import annotations

__all__ = [
    "attach_hf_component_sources",
    "configure_component_quantization",
    "normalize_component_quantized_weights",
    "validate_quantized_component_bindings",
    "preprocess_component_quantized_state_dict",
]

from collections.abc import Iterable, Mapping
from typing import Any

import onnx_ir as ir
import torch
from onnxscript import nn

from mobius._component_manifest import ComponentDescriptor, ComponentManifest
from mobius._configs import BaseModelConfig, QuantizationConfig
from mobius._weight_utils import is_packed_quant_key
from mobius.components import (
    ClippableLinear,
    ClippableQuantizedLinear,
    Embedding,
    Linear,
    MoELayer,
    QuantizedEmbedding,
    QuantizedLinear,
    make_clippable_quantized_linear_factory,
    make_quantized_linear_factory,
)
from mobius.tasks import ModelTask, get_task
from mobius.weights import FloatWeight, PackedWeight, codec_registry

_AFFINE_METHODS = frozenset({"olive", "gptq", "awq"})
_TOKEN_EMBEDDING_NAMES = frozenset(
    {"embed_in", "embed_tokens", "shared", "word_embeddings", "wte"}
)
_KNOWN_SPLIT_COMPONENTS = frozenset(
    {
        "decoder",
        "encoder",
        "vision",
        "vision_encoder",
        "audio",
        "audio_encoder",
        "embedding",
        "model",
    }
)


def _resolve_module(root: nn.Module, path: str) -> nn.Module | None:
    if not path:
        return root
    current: object = root
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current if isinstance(current, nn.Module) else None


def _replace_child(root: nn.Module, path: str, replacement: nn.Module) -> None:
    if not path:
        raise ValueError("Cannot replace a component's root module")
    parts = path.split(".")
    parent: object = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    old = getattr(parent, parts[-1])
    if hasattr(replacement, "_set_name") and hasattr(old, "name"):
        replacement._set_name(old.name)
    setattr(parent, parts[-1], replacement)


def _component_quantization(
    config: BaseModelConfig,
    component: str,
) -> QuantizationConfig | None:
    resolver = getattr(config, "quantization_for", None)
    if resolver is not None:
        return resolver(component)
    mapping = getattr(config, "component_quantization", None)
    if mapping is None:
        return getattr(config, "quantization", None)
    if component in mapping:
        return mapping[component]
    if component == "model":
        return mapping.get("decoder")
    if component == "decoder":
        return mapping.get("model")
    return None


def _uses_global_module_rules(
    config: BaseModelConfig,
    manifest: ComponentManifest,
) -> bool:
    quantization = getattr(config, "quantization", None)
    return len(manifest) == 1 and quantization is not None and quantization.has_module_plan


def _linear_factory(
    config: BaseModelConfig,
    quantization: QuantizationConfig,
) -> type[nn.Module]:
    zero_point_dtype = config.dtype if quantization.float_zero_point else ir.DataType.UINT8
    return make_quantized_linear_factory(
        bits=quantization.bits,
        block_size=quantization.group_size,
        has_zero_point=not quantization.sym,
        zero_point_dtype=zero_point_dtype,
    )


def _clippable_linear_factory(
    config: BaseModelConfig,
    quantization: QuantizationConfig,
) -> type[nn.Module]:
    zero_point_dtype = config.dtype if quantization.float_zero_point else ir.DataType.UINT8
    return make_clippable_quantized_linear_factory(
        bits=quantization.bits,
        block_size=quantization.group_size,
        has_zero_point=not quantization.sym,
        zero_point_dtype=zero_point_dtype,
    )


def _float_linear(module: QuantizedLinear) -> Linear:
    return Linear(module._k, module._n, bias=module.bias is not None)


def _float_embedding(module: QuantizedEmbedding) -> Embedding:
    return Embedding(
        int(module.qweight.shape[0]),
        module._embedding_dim,
        module.padding_idx,
    )


def _linear_layout_matches(
    module: QuantizedLinear,
    quantization: QuantizationConfig,
) -> bool:
    expected_zero_point_dtype = (
        module.scales.dtype if quantization.float_zero_point else ir.DataType.UINT8
    )
    return (
        module._bits == quantization.bits
        and module._block_size == quantization.group_size
        and (module.zero_points is None) is quantization.sym
        and (
            module.zero_points is None or module.zero_points.dtype == expected_zero_point_dtype
        )
    )


def _embedding_layout_matches(
    module: QuantizedEmbedding,
    quantization: QuantizationConfig,
) -> bool:
    return (
        quantization.quantize_embeddings
        and module._bits == quantization.bits
        and module._block_size == quantization.group_size
        and (module.zero_points is None) is quantization.sym
    )


def _component_output_head_paths(module: nn.Module, component: str) -> tuple[str, ...]:
    mapping = getattr(type(module), "COMPONENT_OUTPUT_HEADS", {})
    aliases = {
        "decoder": ("decoder", "model"),
        "model": ("model", "decoder"),
    }.get(component, (component,))
    declared = next((tuple(mapping[name]) for name in aliases if name in mapping), ())
    return tuple(dict.fromkeys(("lm_head", *declared)))


def _excluded_from_component_quantization(
    root: nn.Module,
    path: str,
    quantization: QuantizationConfig,
) -> bool:
    parts = path.split(".")
    for end in range(len(parts) + 1):
        ancestor = _resolve_module(root, ".".join(parts[:end]))
        methods = getattr(ancestor, "component_quantization_excluded_methods", ())
        if quantization.quant_method in methods:
            return True
    return False


def _effective_module_quantization(
    component_quantization: QuantizationConfig | None,
    descriptor: ComponentDescriptor,
    local_module_path: str,
    *,
    source_module_names: tuple[str, ...] | None = None,
    component_module: nn.Module | None = None,
    output_head_paths: tuple[str, ...] = ("lm_head",),
) -> QuantizationConfig | None:
    if component_quantization is None or component_quantization.quant_method == "none":
        return None
    source_names = (
        source_module_names
        if source_module_names is not None
        else descriptor.source_module_names(local_module_path)
    )
    quantization = component_quantization.for_module(source_names)
    if quantization is None:
        return None
    if not quantization.quantize_lm_head and any(
        local_module_path == path or local_module_path.endswith(f".{path}")
        for path in output_head_paths
    ):
        return None
    if component_module is not None and _excluded_from_component_quantization(
        component_module, local_module_path, quantization
    ):
        return None
    return quantization


def _source_module_names(
    descriptor: ComponentDescriptor,
    local_module_path: str,
    module: nn.Module,
) -> tuple[str, ...]:
    names = descriptor.source_module_names(local_module_path)
    if isinstance(module, (ClippableLinear, ClippableQuantizedLinear)):
        names = (*names, *(f"{name}.linear" for name in names))
    return tuple(dict.fromkeys(names))


def _configure_component_module(
    component_module: nn.Module,
    descriptor: ComponentDescriptor,
    config: BaseModelConfig,
    component_quantization: QuantizationConfig | None,
    *,
    owned_by_other_components: tuple[str, ...] = (),
    output_head_paths: tuple[str, ...] = ("lm_head",),
) -> None:
    replacements: list[tuple[str, nn.Module]] = []
    for local_path, child in list(component_module.named_modules()):
        if not local_path:
            continue
        if any(
            local_path == prefix or local_path.startswith(f"{prefix}.")
            for prefix in owned_by_other_components
        ):
            continue
        source_names = _source_module_names(descriptor, local_path, child)
        quantization = _effective_module_quantization(
            component_quantization,
            descriptor,
            local_path,
            source_module_names=source_names,
            component_module=component_module,
            output_head_paths=output_head_paths,
        )

        if isinstance(child, ClippableQuantizedLinear):
            if type(child).forward is not ClippableQuantizedLinear.forward:
                raise TypeError(
                    f"Component plan cannot rewrite specialized clipped "
                    f"quantized module {local_path!r} "
                    f"({type(child).__name__}); provide a model weight adapter "
                    "for this component."
                )
            replacement: nn.Module
            if quantization is None:
                replacement = ClippableLinear(
                    child._k,
                    child._n,
                    bias=child.bias is not None,
                )
            else:
                replacement = _clippable_linear_factory(config, quantization)(
                    child._k,
                    child._n,
                    bias=child.bias is not None,
                )
            replacements.append((local_path, replacement))
            continue

        if isinstance(child, QuantizedLinear):
            if type(child).forward is not QuantizedLinear.forward:
                if quantization is not None and _linear_layout_matches(
                    child,
                    quantization,
                ):
                    continue
                raise TypeError(
                    f"Component plan cannot rewrite specialized quantized "
                    f"module {local_path!r} ({type(child).__name__}); provide "
                    "a model weight adapter for this component."
                )
            replacement = (
                _float_linear(child)
                if quantization is None
                else _linear_factory(config, quantization)(
                    child._k,
                    child._n,
                    bias=child.bias is not None,
                )
            )
            replacements.append((local_path, replacement))
            continue

        if isinstance(child, QuantizedEmbedding):
            if type(child).forward is not QuantizedEmbedding.forward:
                if quantization is not None and _embedding_layout_matches(
                    child,
                    quantization,
                ):
                    continue
                raise TypeError(
                    f"Component plan cannot rewrite specialized quantized "
                    f"embedding {local_path!r} ({type(child).__name__}); "
                    "provide a model weight adapter for this component."
                )
            if quantization is None or not quantization.quantize_embeddings:
                replacements.append((local_path, _float_embedding(child)))
            elif not _embedding_layout_matches(child, quantization):
                assert child.qweight.shape is not None
                num_embeddings = child.qweight.shape[0]
                assert isinstance(num_embeddings, int)
                replacements.append(
                    (
                        local_path,
                        QuantizedEmbedding(
                            num_embeddings,
                            child._embedding_dim,
                            padding_idx=child.padding_idx,
                            bits=quantization.bits,
                            block_size=quantization.group_size,
                            has_zero_point=not quantization.sym,
                        ),
                    )
                )
            continue

        if quantization is None:
            continue
        if quantization.quant_method not in _AFFINE_METHODS:
            continue

        if isinstance(child, Embedding) and type(child).forward is Embedding.forward:
            embedding_dim = int(child.weight.shape[1])
            if (
                quantization.quantize_embeddings
                and any(
                    name.rsplit(".", 1)[-1] in _TOKEN_EMBEDDING_NAMES for name in source_names
                )
                and embedding_dim % quantization.group_size == 0
            ):
                num_embeddings = int(child.weight.shape[0])
                replacements.append(
                    (
                        local_path,
                        QuantizedEmbedding(
                            num_embeddings,
                            embedding_dim,
                            bits=quantization.bits,
                            block_size=quantization.group_size,
                            has_zero_point=not quantization.sym,
                            padding_idx=child.padding_idx,
                        ),
                    )
                )
            continue

        if isinstance(child, Linear) and type(child).forward is Linear.forward:
            out_features, in_features = (int(dim) for dim in child.weight.shape)
            replacements.append(
                (
                    local_path,
                    _linear_factory(config, quantization)(
                        in_features,
                        out_features,
                        bias=child.bias is not None,
                    ),
                )
            )
        elif type(child) is ClippableLinear:
            out_features, in_features = (int(dim) for dim in child.weight.shape)
            replacements.append(
                (
                    local_path,
                    _clippable_linear_factory(config, quantization)(
                        in_features,
                        out_features,
                        bias=child.bias is not None,
                    ),
                )
            )

    # Replace deepest children first so replacing a parent never invalidates a
    # path that still needs to be visited.
    for path, replacement in sorted(
        replacements,
        key=lambda item: item[0].count("."),
        reverse=True,
    ):
        _replace_child(component_module, path, replacement)


def _default_manifest(
    module: nn.Module,
    config: BaseModelConfig,
    task: str | ModelTask,
) -> ComponentManifest:
    resolved_task = get_task(task)
    model_type = getattr(config, "model_type", None)
    return resolved_task.component_manifest(
        module_class=type(module),
        model_type=model_type,
        hf_config=config,
    )


def configure_component_quantization(
    module: nn.Module,
    config: BaseModelConfig,
    task: str | ModelTask,
    *,
    manifest: ComponentManifest | None = None,
) -> ComponentManifest:
    """Apply authoritative component plans to graph parameter scaffolding."""
    manifest = manifest or _default_manifest(module, config, task)
    mapping = getattr(config, "component_quantization", None)
    if mapping is None and not _uses_global_module_rules(config, manifest):
        return manifest

    unresolved = set(mapping) - set(manifest) if mapping is not None else set()
    if "model" in manifest:
        unresolved.discard("decoder")
    if "decoder" in manifest:
        unresolved.discard("model")
    if manifest.names == ("model",):
        unresolved -= _KNOWN_SPLIT_COMPONENTS
    if unresolved:
        raise ValueError(
            f"Component quantization references components not produced by "
            f"{type(get_task(task)).__name__}: {sorted(unresolved)}. "
            f"Available components: {sorted(manifest)}"
        )

    for descriptor in manifest.values():
        component_module = _resolve_module(
            module,
            descriptor.module_attribute_path,
        )
        quantization = _component_quantization(config, descriptor.name)
        if component_module is None:
            continue
        owned_elsewhere = (
            tuple(
                other.module_attribute_path
                for other in manifest.values()
                if other.name != descriptor.name and other.module_attribute_path
            )
            if not descriptor.module_attribute_path
            else ()
        )
        _configure_component_module(
            component_module,
            descriptor,
            config,
            quantization,
            owned_by_other_components=owned_elsewhere,
            output_head_paths=_component_output_head_paths(module, descriptor.name),
        )
    return manifest


def _quantized_parameter_groups(
    module: nn.Module,
) -> Iterable[tuple[str, dict[str, nn.Parameter]]]:
    if isinstance(module, MoELayer) and module.experts is None:
        for prefix in ("fc1", "fc2"):
            weight_name = f"{prefix}_experts_weights"
            parameters = {
                weight_name: getattr(module, weight_name),
                f"{prefix}_scales": getattr(module, f"{prefix}_scales"),
            }
            zero_points_name = f"{prefix}_experts_zero_points"
            zero_points = getattr(module, zero_points_name)
            if zero_points is not None:
                parameters[zero_points_name] = zero_points
            yield weight_name, parameters
    elif isinstance(module, (QuantizedEmbedding, QuantizedLinear)):
        weight_name = "qweight" if isinstance(module, QuantizedEmbedding) else "weight"
        parameters = {
            weight_name: getattr(module, weight_name),
            "scales": module.scales,
        }
        if module.zero_points is not None:
            parameters["zero_points"] = module.zero_points
        yield weight_name, parameters


def _canonical_component_parameter_keys(
    module: nn.Module,
    descriptor: ComponentDescriptor,
    weights: Mapping[str, torch.Tensor],
) -> frozenset[str]:
    """Identify complete canonical groups, not individual shared sidecar names."""
    component_module = _resolve_module(
        module,
        descriptor.module_attribute_path,
    )
    if component_module is None:
        return frozenset()

    keys: set[str] = set()
    prefixes = {
        prefix for prefix in (descriptor.name, descriptor.module_attribute_path) if prefix
    }
    for local_path, child in component_module.named_modules():
        if not local_path:
            continue
        stems = {
            f"{prefix}.{local_path}" if prefix else local_path for prefix in (*prefixes, "")
        }
        for weight_name, parameters in _quantized_parameter_groups(child):
            for stem in stems:
                packed_key = f"{stem}.{weight_name}"
                if packed_key not in weights or weights[packed_key].dtype != torch.uint8:
                    continue
                if any(
                    parameter.shape is None
                    or f"{stem}.{name}" not in weights
                    or tuple(weights[f"{stem}.{name}"].shape) != tuple(parameter.shape)
                    for name, parameter in parameters.items()
                ):
                    continue
                if isinstance(child, QuantizedLinear) and any(
                    key in weights for key in (f"{stem}.qweight", f"{stem}.weight_qweight")
                ):
                    continue
                keys.update(f"{stem}.{name}" for name in parameters)
    return frozenset(keys)


def _route_component_weights(
    state_dict: Mapping[str, Any],
    manifest: ComponentManifest,
    component_names: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    if len(component_names) == 1:
        return {component_names[0]: dict(state_dict)}

    prefixes = {
        name: {
            prefix
            for prefix in (
                name,
                manifest[name].module_attribute_path,
            )
            if prefix
        }
        for name in component_names
    }

    def owner(key: str) -> str | None:
        matches = [
            (len(prefix), component)
            for component, component_prefixes in prefixes.items()
            for prefix in component_prefixes
            if key.startswith(f"{prefix}.")
        ]
        if not matches:
            root_components = [
                name for name in component_names if not manifest[name].module_attribute_path
            ]
            return root_components[0] if len(root_components) == 1 else None
        max_length = max(length for length, _ in matches)
        owners = {component for length, component in matches if length == max_length}
        if len(owners) != 1:
            raise ValueError(
                f"Checkpoint weight {key!r} matches multiple components "
                f"{sorted(owners)} at the same prefix depth"
            )
        return next(iter(owners))

    routed = {name: {} for name in component_names}
    for key, value in state_dict.items():
        component = owner(key)
        if component is not None:
            routed[component][key] = value
    return routed


def _local_weight_module_path(
    record_name: str,
    descriptor: ComponentDescriptor,
) -> str:
    name = record_name.removesuffix(".weight")
    if not descriptor.module_attribute_path:
        return name
    for prefix in (descriptor.module_attribute_path, descriptor.name):
        if prefix and name.startswith(f"{prefix}."):
            return name[len(prefix) + 1 :]
    return name


def normalize_component_quantized_weights(
    state_dict: dict[str, Any],
    module: nn.Module,
    config: BaseModelConfig,
    component_names: Iterable[str],
    *,
    manifest: ComponentManifest | None = None,
    task: str | ModelTask,
) -> dict[str, Any]:
    """Normalize existing packed sidecars with each component's own plan."""
    component_names = tuple(component_names)
    manifest = manifest or _default_manifest(module, config, task)
    mapping = getattr(config, "component_quantization", None)
    if mapping is None and not _uses_global_module_rules(config, manifest):
        return state_dict
    routed = _route_component_weights(state_dict, manifest, component_names)
    result = dict(state_dict)

    for component in component_names:
        weights = routed[component]
        canonical_keys = _canonical_component_parameter_keys(
            module,
            manifest[component],
            weights,
        )
        source_weights = {
            key: value for key, value in weights.items() if key not in canonical_keys
        }
        if not any(is_packed_quant_key(key) for key in source_weights):
            continue

        descriptor = manifest[component]
        component_module = _resolve_module(module, descriptor.module_attribute_path)
        output_head_paths = _component_output_head_paths(module, component)
        component_quantization = _component_quantization(config, component)
        if component_quantization is None:
            packed_key = next(key for key in source_weights if is_packed_quant_key(key))
            raise ValueError(
                f"Component {component!r} is floating point but checkpoint "
                f"contains packed weight {packed_key!r}"
            )
        if component_quantization.quant_method not in codec_registry:
            raise KeyError(
                f"No packed-weight codec for component {component!r} method "
                f"{component_quantization.quant_method!r}"
            )

        codec = codec_registry.get(component_quantization.quant_method)
        bundle = codec.group(
            descriptor,
            source_weights,
            component_quantization,
        )
        for source_key in bundle.source_keys:
            result.pop(source_key, None)
        for record in bundle.values():
            if isinstance(record.storage, FloatWeight):
                result[record.storage.source_key] = record.storage.value
                continue
            assert isinstance(record.storage, PackedWeight)
            if "expert" in record.name:
                raise NotImplementedError(
                    f"Packed expert weight {record.name!r} requires a "
                    "component-specific QMoE weight adapter."
                )
            local_path = _local_weight_module_path(record.name, descriptor)
            local_module = (
                _resolve_module(component_module, local_path)
                if component_module is not None
                else None
            )
            source_names = (
                _source_module_names(descriptor, local_path, local_module)
                if local_module is not None
                else descriptor.source_module_names(local_path)
            )
            quantization = _effective_module_quantization(
                component_quantization,
                descriptor,
                local_path,
                source_module_names=source_names,
                component_module=component_module,
                output_head_paths=output_head_paths,
            )
            if quantization is None:
                raise ValueError(
                    f"Packed checkpoint weight {record.name!r} targets a module "
                    f"excluded from component {component!r} quantization"
                )
            if local_module is None:
                raise ValueError(
                    f"Packed checkpoint weight {record.name!r} has no target module "
                    f"in component {component!r}"
                )
            if not isinstance(local_module, (QuantizedLinear, QuantizedEmbedding)):
                if quantization.tie_word_embeddings and any(
                    token in record.name for token in ("embed_tokens", "lm_head")
                ):
                    raise NotImplementedError(
                        f"Tied packed table {record.name!r} requires a "
                        "component-specific tied-weight adapter."
                    )
                raise TypeError(
                    f"Packed checkpoint weight {record.name!r} targets a "
                    f"module unsupported by the affine codec: {type(local_module).__name__}"
                )
            result.update(
                codec.normalize(
                    record,
                    quantization,
                    kind="embedding"
                    if isinstance(local_module, QuantizedEmbedding)
                    else "linear",
                )
            )

    canonical_keys = frozenset(
        key
        for component in component_names
        for key in _canonical_component_parameter_keys(
            module,
            manifest[component],
            result,
        )
    )
    remaining = next(
        (key for key in result if is_packed_quant_key(key) and key not in canonical_keys),
        None,
    )
    if remaining is not None:
        raise ValueError(
            f"Packed checkpoint weight {remaining!r} was not routed to any "
            "ModelPackage component"
        )
    return result


def validate_quantized_component_bindings(
    models: Mapping[str, ir.Model],
    config: BaseModelConfig,
) -> None:
    """Require every affine quantized op input to carry a bound value."""
    quantized_input_slots = {
        "MatMulNBits": (1, 2, 3),
        "GatherBlockQuantized": (0, 2, 3),
    }
    for component, model in models.items():
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            slots = quantized_input_slots.get(node.op_type)
            if slots is None:
                continue
            for index in slots:
                if index >= len(node.inputs):
                    continue
                value = node.inputs[index]
                if value is None or value.producer() is not None:
                    continue
                if value.const_value is None:
                    raise ValueError(
                        f"Quantized component {component!r} has unbound "
                        f"{node.op_type} parameter {value.name!r}"
                    )


def attach_hf_component_sources(
    module: nn.Module,
    *,
    model_type: str,
    hf_config: object,
) -> None:
    """Attach the runtime HF component map selected for this concrete model."""
    resolver = getattr(type(module), "get_hf_component_sources", None)
    if resolver is not None:
        source_map = resolver(model_type=model_type, hf_config=hf_config)
    else:
        source_map = getattr(type(module), "HF_COMPONENT_SOURCES", {})
    module._hf_component_sources = {
        component: tuple(paths) for component, paths in source_map.items()
    }


def preprocess_component_quantized_state_dict(
    state_dict: dict[str, torch.Tensor],
    module: nn.Module,
    config: BaseModelConfig,
    task: ModelTask | str,
    package_components: Iterable[str],
) -> dict[str, torch.Tensor]:
    """Normalize component weights with an explicit task ownership contract."""
    if task is None:
        raise ValueError("task must be provided to normalize component quantized weights")
    resolved_task = get_task(task)
    return normalize_component_quantized_weights(
        state_dict,
        module,
        config,
        package_components,
        task=resolved_task,
    )
