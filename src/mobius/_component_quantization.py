# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configure and load independently quantized model-package components."""

from __future__ import annotations

__all__ = [
    "configure_component_quantization",
    "normalize_component_quantized_weights",
]

from collections.abc import Iterable, Mapping
from typing import Any

import onnx_ir as ir
from onnxscript import nn

from mobius._component_manifest import ComponentDescriptor, ComponentManifest
from mobius._configs import BaseModelConfig, QuantizationConfig
from mobius.components import (
    ClippableLinear,
    ClippableQuantizedLinear,
    Embedding,
    Linear,
    QuantizedEmbedding,
    QuantizedLinear,
    make_clippable_quantized_linear_factory,
    make_quantized_linear_factory,
)
from mobius.tasks import ModelTask, get_task
from mobius.weights import FloatWeight, PackedWeight, codec_registry

_AFFINE_METHODS = frozenset({"olive", "gptq", "awq"})
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


def _effective_module_quantization(
    component_quantization: QuantizationConfig | None,
    descriptor: ComponentDescriptor,
    local_module_path: str,
) -> QuantizationConfig | None:
    if component_quantization is None or component_quantization.quant_method == "none":
        return None
    source_names = descriptor.source_module_names(local_module_path)
    return component_quantization.for_module(source_names)


def _configure_component_module(
    component_module: nn.Module,
    descriptor: ComponentDescriptor,
    config: BaseModelConfig,
    component_quantization: QuantizationConfig | None,
    *,
    owned_by_other_components: tuple[str, ...] = (),
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
        quantization = _effective_module_quantization(
            component_quantization,
            descriptor,
            local_path,
        )
        is_lm_head = local_path == "lm_head" or local_path.endswith(".lm_head")
        if quantization is not None and is_lm_head and not quantization.quantize_lm_head:
            quantization = None

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
                raise TypeError(
                    f"Component plan cannot rewrite specialized quantized "
                    f"embedding {local_path!r} ({type(child).__name__}); "
                    "provide a model weight adapter for this component."
                )
            if quantization is None or not quantization.quantize_embeddings:
                replacements.append((local_path, _float_embedding(child)))
            continue

        if quantization is None:
            continue
        if quantization.quant_method not in _AFFINE_METHODS:
            continue

        if isinstance(child, Embedding) and type(child).forward is Embedding.forward:
            embedding_dim = int(child.weight.shape[1])
            if (
                quantization.quantize_embeddings
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
    root_quantization = getattr(config, "quantization", None)
    single_component_rules = (
        manifest.names == ("model",)
        and root_quantization is not None
        and (root_quantization.modules_to_not_convert or root_quantization.overrides)
    )
    if mapping is None and not single_component_rules:
        return manifest

    unresolved = set(mapping) - set(manifest)
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
        )
    return manifest


def _raw_qweight_key(name: str) -> bool:
    return name.endswith(("_qweight", ".qweight"))


def _canonical_component_parameter_keys(
    module: nn.Module,
    descriptor: ComponentDescriptor,
) -> frozenset[str]:
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
        if isinstance(child, QuantizedEmbedding):
            for stem in stems:
                keys.update(
                    {
                        f"{stem}.qweight",
                        f"{stem}.scales",
                        f"{stem}.zero_points",
                    }
                )
        elif isinstance(child, QuantizedLinear):
            for stem in stems:
                keys.update(
                    {
                        f"{stem}.weight",
                        f"{stem}.scales",
                        f"{stem}.zero_points",
                    }
                )
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
    mapping = getattr(config, "component_quantization", None)
    root_quantization = getattr(config, "quantization", None)
    single_component_rules = (
        len(component_names) == 1
        and root_quantization is not None
        and (root_quantization.modules_to_not_convert or root_quantization.overrides)
    )
    if mapping is None and not single_component_rules:
        return state_dict
    manifest = manifest or _default_manifest(module, config, task)
    manifest = manifest or _default_manifest(module, config, task)
    routed = _route_component_weights(state_dict, manifest, component_names)
    result = dict(state_dict)

    for component in component_names:
        weights = routed[component]
        canonical_keys = _canonical_component_parameter_keys(
            module,
            manifest[component],
        )
        source_weights = {
            key: value for key, value in weights.items() if key not in canonical_keys
        }
        if not any(_raw_qweight_key(key) for key in source_weights):
            continue

        descriptor = manifest[component]
        component_quantization = _component_quantization(config, component)
        if component_quantization is None:
            packed_key = next(key for key in weights if _raw_qweight_key(key))
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
            if component_quantization.tie_word_embeddings and any(
                token in record.name for token in ("embed_tokens", "lm_head")
            ):
                raise NotImplementedError(
                    f"Tied packed table {record.name!r} requires a "
                    "component-specific tied-weight adapter."
                )
            local_path = _local_weight_module_path(record.name, descriptor)
            quantization = _effective_module_quantization(
                component_quantization,
                descriptor,
                local_path,
            )
            if quantization is None:
                raise ValueError(
                    f"Packed checkpoint weight {record.name!r} targets a module "
                    f"excluded from component {component!r} quantization"
                )
            result.update(codec.normalize(record, quantization))

    canonical_keys = frozenset(
        key
        for component in component_names
        for key in _canonical_component_parameter_keys(
            module,
            manifest[component],
        )
    )
    remaining = next(
        (key for key in result if _raw_qweight_key(key) and key not in canonical_keys),
        None,
    )
    if remaining is not None:
        raise ValueError(
            f"Packed checkpoint weight {remaining!r} was not routed to any "
            "ModelPackage component"
        )
    return result
