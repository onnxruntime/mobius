# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Apply independent weight-quantization layouts to package components."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig, QuantizationConfig
from mobius._weight_utils import preprocess_quantized_weights
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

_AFFINE_QUANT_METHODS = frozenset({"olive", "gptq", "awq"})
_COMPONENT_ATTRIBUTE_ALIASES = {
    "vision": "vision_encoder",
    "audio": "audio_encoder",
    "speech": "speech_encoder",
}
_KNOWN_COMPONENT_NAMES = frozenset(
    {
        "model",
        "decoder",
        "encoder",
        "vision",
        "vision_encoder",
        "audio",
        "audio_encoder",
        "embedding",
    }
)


def _resolve_path(root: nn.Module, path: str) -> nn.Module | None:
    current: object = root
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current if isinstance(current, nn.Module) else None


def _component_module_paths(
    module: nn.Module,
    task: ModelTask,
) -> dict[str, str]:
    """Resolve package component names to ONNXScript module paths."""
    paths: dict[str, str] = {}
    if task.components is not None:
        paths.update(
            {
                component: path
                for component, path in task.components.items()
                if _resolve_path(module, path) is not None
            }
        )

    for component in task.model_roles:
        if component in paths:
            continue
        if component == "model":
            paths[component] = ""
            continue
        candidates = (
            component,
            _COMPONENT_ATTRIBUTE_ALIASES.get(component, component),
        )
        for candidate in candidates:
            if _resolve_path(module, candidate) is not None:
                paths[component] = candidate
                break
    return paths


def _component_module(module: nn.Module, path: str) -> nn.Module:
    if not path:
        return module
    resolved = _resolve_path(module, path)
    if resolved is None:
        raise ValueError(f"Cannot resolve component module path {path!r}")
    return resolved


def _replace_child_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    """Replace a named ONNXScript child while retaining its graph name."""
    if not path:
        raise ValueError("Cannot replace the root component module")
    parts = path.split(".")
    parent: object = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    child_name = parts[-1]
    old = getattr(parent, child_name)
    if hasattr(replacement, "_set_name") and hasattr(old, "name"):
        replacement._set_name(old.name)
    setattr(parent, child_name, replacement)


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


def _effective_component_quantization(
    module: nn.Module,
    config: BaseModelConfig,
    component: str,
) -> QuantizationConfig | None:
    resolver = getattr(config, "quantization_for", None)
    if resolver is not None:
        quantization = resolver(component)
    else:
        component_quantization = getattr(config, "component_quantization", None)
        quantization = (
            component_quantization.get(component)
            if component_quantization is not None
            else getattr(config, "quantization", None)
        )
    if quantization is None or quantization.quant_method == "none":
        return None
    if not quantization.has_module_plan:
        return quantization

    source_map = getattr(type(module), "HF_COMPONENT_SOURCES", {})
    source_paths = tuple(source_map.get(component, ()))
    if not source_paths:
        raise ValueError(
            f"Component {component!r} carries module-level quantization rules, "
            f"but {type(module).__name__} declares no HF_COMPONENT_SOURCES entry "
            "from which Mobius can derive a uniform component layout."
        )
    return quantization.for_source_paths(source_paths, component=component)


def _float_linear(module: QuantizedLinear) -> Linear:
    return Linear(module._k, module._n, bias=module.bias is not None)


def _float_embedding(module: QuantizedEmbedding) -> Embedding:
    return Embedding(
        int(module.qweight.shape[0]),
        module._embedding_dim,
        module.padding_idx,
    )


def _configure_component_module(
    component_module: nn.Module,
    config: BaseModelConfig,
    quantization: QuantizationConfig | None,
) -> None:
    """Rewrite float/quantized projection scaffolding for one component."""
    named_modules = list(component_module.named_modules())

    linear_factory = (
        _linear_factory(config, quantization) if quantization is not None else None
    )
    clippable_factory = (
        _clippable_linear_factory(config, quantization) if quantization is not None else None
    )
    replacements: list[tuple[str, nn.Module]] = []

    for name, child in named_modules:
        if not name:
            continue
        is_lm_head = name == "lm_head" or name.endswith(".lm_head")

        if isinstance(child, ClippableQuantizedLinear):
            if quantization is None or (is_lm_head and not quantization.quantize_lm_head):
                replacement: nn.Module = ClippableLinear(
                    child._k,
                    child._n,
                    bias=child.bias is not None,
                )
            else:
                assert clippable_factory is not None
                replacement = clippable_factory(
                    child._k,
                    child._n,
                    bias=child.bias is not None,
                )
            replacements.append((name, replacement))
            continue

        if isinstance(child, QuantizedLinear):
            if quantization is None or (is_lm_head and not quantization.quantize_lm_head):
                replacement = _float_linear(child)
            else:
                assert linear_factory is not None
                replacement = linear_factory(
                    child._k,
                    child._n,
                    bias=child.bias is not None,
                )
            replacements.append((name, replacement))
            continue

        if isinstance(child, QuantizedEmbedding):
            if quantization is None or not quantization.quantize_embeddings:
                replacements.append((name, _float_embedding(child)))
            continue

        if quantization is None:
            continue
        if is_lm_head and not quantization.quantize_lm_head:
            continue
        if name.split(".")[-1] in {"router", "shared_expert_gate"}:
            continue
        if isinstance(child, Embedding) and type(child).forward is Embedding.forward:
            if (
                quantization.quantize_embeddings
                and int(child.weight.shape[1]) % quantization.group_size == 0
            ):
                num_embeddings, embedding_dim = (int(dim) for dim in child.weight.shape)
                replacements.append(
                    (
                        name,
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
            assert linear_factory is not None
            out_features, in_features = (int(dim) for dim in child.weight.shape)
            replacements.append(
                (
                    name,
                    linear_factory(
                        in_features,
                        out_features,
                        bias=child.bias is not None,
                    ),
                )
            )
        elif type(child) is ClippableLinear:
            assert clippable_factory is not None
            out_features, in_features = (int(dim) for dim in child.weight.shape)
            replacements.append(
                (
                    name,
                    clippable_factory(
                        in_features,
                        out_features,
                        bias=child.bias is not None,
                    ),
                )
            )

    for path, replacement in sorted(
        replacements,
        key=lambda item: item[0].count("."),
        reverse=True,
    ):
        _replace_child_module(component_module, path, replacement)


def configure_component_quantization(
    module: nn.Module,
    config: BaseModelConfig,
    task: str | ModelTask,
) -> None:
    """Configure every task component from ``config.component_quantization``."""
    component_quantization = getattr(config, "component_quantization", None)
    if component_quantization is None:
        return
    resolved_task = get_task(task)
    paths = _component_module_paths(module, resolved_task)
    unresolved = set(component_quantization) - set(paths)
    if "model" in paths:
        unresolved.discard("decoder")
    if "decoder" in paths:
        unresolved.discard("model")
    if set(paths) == {"model"}:
        unresolved -= _KNOWN_COMPONENT_NAMES
    if unresolved:
        raise ValueError(
            f"{type(resolved_task).__name__} cannot resolve component module(s) "
            f"{sorted(unresolved)} on {type(module).__name__}. Resolved components: "
            f"{sorted(paths)}"
        )

    for component, path in paths.items():
        quantization = _effective_component_quantization(module, config, component)
        if quantization is not None and quantization.quant_method not in _AFFINE_QUANT_METHODS:
            component_module = _component_module(module, path)
            if not any(
                isinstance(child, QuantizedLinear)
                for name, child in component_module.named_modules()
                if name
            ):
                raise NotImplementedError(
                    f"Generic component quantization cannot construct "
                    f"{quantization.quant_method!r} projections for component "
                    f"{component!r}; the model must provide a specialized component."
                )
            continue
        _configure_component_module(
            _component_module(module, path),
            config,
            quantization,
        )


def _has_raw_packed_weight(names: Iterable[str]) -> bool:
    return any(name.endswith(("_qweight", ".qweight")) for name in names)


def preprocess_component_quantized_state_dict(
    state_dict: dict[str, Any],
    module: nn.Module,
    config: BaseModelConfig,
    task: str | ModelTask,
    component_names: Iterable[str],
) -> dict[str, Any]:
    """Convert remaining raw packed sidecars with each component's layout."""
    if getattr(config, "component_quantization", None) is None:
        return state_dict

    resolved_task = get_task(task)
    component_paths = _component_module_paths(module, resolved_task)
    component_names = tuple(component_names)
    routing_prefixes = {
        component: {prefix for prefix in (component, component_paths.get(component)) if prefix}
        for component in component_names
    }

    def routed_component(key: str) -> str | None:
        matches = [
            (len(prefix), component)
            for component, prefixes in routing_prefixes.items()
            for prefix in prefixes
            if key.startswith(f"{prefix}.")
        ]
        if not matches:
            return None
        max_length = max(length for length, _ in matches)
        owners = {component for length, component in matches if length == max_length}
        if len(owners) != 1:
            raise ValueError(
                f"Checkpoint weight {key!r} matches multiple components "
                f"{sorted(owners)} at the same prefix depth."
            )
        return next(iter(owners))

    result = dict(state_dict)
    for component in component_names:
        component_path = component_paths.get(component, component)
        if len(component_names) == 1 or not component_path:
            component_weights = dict(result)
        else:
            component_weights = {
                key: value
                for key, value in result.items()
                if routed_component(key) == component
            }
        if not component_weights or not _has_raw_packed_weight(component_weights):
            continue

        quantization = _effective_component_quantization(module, config, component)
        if quantization is None:
            packed_key = next(
                key for key in component_weights if key.endswith(("_qweight", ".qweight"))
            )
            raise ValueError(
                f"Component {component!r} is configured as floating point, but "
                f"packed checkpoint weight {packed_key!r} was found."
            )
        if quantization.quant_method not in _AFFINE_QUANT_METHODS:
            raise NotImplementedError(
                f"Generic packed-weight preprocessing does not support "
                f"{quantization.quant_method!r} for component {component!r}."
            )
        packed_expert_key = next(
            (
                key
                for key in component_weights
                if key.endswith(("_qweight", ".qweight")) and "expert" in key
            ),
            None,
        )
        if packed_expert_key is not None:
            raise NotImplementedError(
                f"Component {component!r} still contains packed expert weight "
                f"{packed_expert_key!r} after model preprocessing. Its model "
                "must provide a component-aware QMoE conversion."
            )
        packed_lm_head = next(
            (
                key
                for key in component_weights
                if key.endswith(("_qweight", ".qweight")) and "lm_head" in key
            ),
            None,
        )
        if packed_lm_head is not None and not quantization.quantize_lm_head:
            raise ValueError(
                f"Component {component!r} keeps lm_head floating point, but "
                f"packed checkpoint weight {packed_lm_head!r} was found."
            )
        packed_embedding = next(
            (
                key
                for key in component_weights
                if key.endswith(("_qweight", ".qweight"))
                and any(token in key for token in ("embed_tokens", "embedding"))
            ),
            None,
        )
        if packed_embedding is not None and not quantization.quantize_embeddings:
            raise ValueError(
                f"Component {component!r} keeps embeddings floating point, but "
                f"packed checkpoint weight {packed_embedding!r} was found."
            )

        converted = preprocess_quantized_weights(
            component_weights,
            quantization,
            tie_embeddings=False,
            qmoe_target_path=None,
        )
        if len(component_names) == 1:
            result = converted
        else:
            for key in component_weights:
                result.pop(key, None)
            result.update(converted)
    remaining_packed_key = next(
        (key for key in result if key.endswith(("_qweight", ".qweight"))),
        None,
    )
    if remaining_packed_key is not None:
        raise ValueError(
            f"Packed checkpoint weight {remaining_packed_key!r} was not routed "
            "to any ModelPackage component."
        )
    return result
