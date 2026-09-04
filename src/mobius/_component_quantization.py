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
_TOKEN_EMBEDDING_NAMES = frozenset(
    {
        "embed_in",
        "embed_tokens",
        "shared",
        "word_embeddings",
        "wte",
    }
)
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


def _component_source_map(module: nn.Module) -> dict[str, tuple[str, ...]]:
    return getattr(
        module,
        "_hf_component_sources",
        getattr(type(module), "HF_COMPONENT_SOURCES", {}),
    )


def _component_output_head_paths(module: nn.Module, component: str) -> tuple[str, ...]:
    mapping = getattr(type(module), "COMPONENT_OUTPUT_HEADS", {})
    aliases = {
        "decoder": ("decoder", "model"),
        "model": ("model", "decoder"),
    }.get(component, (component,))
    declared = next((tuple(mapping[name]) for name in aliases if name in mapping), ())
    return tuple(dict.fromkeys(("lm_head", *declared)))


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

    source_map = _component_source_map(module)
    source_paths = tuple(source_map.get(component, ()))
    if not source_paths:
        raise ValueError(
            f"Component {component!r} carries module-level quantization rules, "
            f"but {type(module).__name__} declares no HF_COMPONENT_SOURCES entry "
            "from which Mobius can derive a uniform component layout."
        )
    return config.quantization_for_source_paths(
        component,
        source_paths,
        ignored_source_names=(
            *_component_output_head_paths(module, component),
            "embed_tokens",
        ),
    )


def _float_linear(module: QuantizedLinear) -> Linear:
    return Linear(module._k, module._n, bias=module.bias is not None)


class _ScaledEmbedding(Embedding):
    """Float token embedding that preserves a model's post-gather scale."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None,
        *,
        embed_scale: float,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, op, input_ids):
        return op.Mul(super().forward(op, input_ids), self.embed_scale)


class _ScaledQuantizedEmbedding(QuantizedEmbedding):
    """Quantized token embedding that preserves a model's post-gather scale."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        bits: int,
        block_size: int,
        has_zero_point: bool,
        padding_idx: int | None,
        embed_scale: float,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            bits=bits,
            block_size=block_size,
            has_zero_point=has_zero_point,
            padding_idx=padding_idx,
        )
        self.embed_scale = embed_scale

    def forward(self, op, input_ids):
        return op.Mul(super().forward(op, input_ids), self.embed_scale)


def _float_embedding(module: QuantizedEmbedding) -> Embedding:
    args = (
        int(module.qweight.shape[0]),
        module._embedding_dim,
        module.padding_idx,
    )
    embed_scale = getattr(module, "embed_scale", None)
    if embed_scale is not None:
        return _ScaledEmbedding(*args, embed_scale=float(embed_scale))
    if type(module).forward is not QuantizedEmbedding.forward:
        raise TypeError(
            "Component plan cannot convert specialized quantized embedding "
            f"{type(module).__name__} to a plain embedding without dropping "
            "its forward semantics."
        )
    return Embedding(*args)


def _quantized_embedding(
    module: Embedding | QuantizedEmbedding,
    quantization: QuantizationConfig,
) -> QuantizedEmbedding:
    if isinstance(module, QuantizedEmbedding):
        num_embeddings = int(module.qweight.shape[0])
        embedding_dim = module._embedding_dim
    else:
        num_embeddings, embedding_dim = (int(dim) for dim in module.weight.shape)
    embed_scale = getattr(module, "embed_scale", None)
    if embed_scale is not None:
        return _ScaledQuantizedEmbedding(
            num_embeddings,
            embedding_dim,
            bits=quantization.bits,
            block_size=quantization.group_size,
            has_zero_point=not quantization.sym,
            padding_idx=module.padding_idx,
            embed_scale=float(embed_scale),
        )
    if (
        isinstance(module, QuantizedEmbedding)
        and type(module).forward is not QuantizedEmbedding.forward
    ) or (isinstance(module, Embedding) and type(module).forward is not Embedding.forward):
        raise TypeError(
            "Component plan cannot quantize specialized embedding "
            f"{type(module).__name__} without dropping its forward semantics."
        )
    return QuantizedEmbedding(
        num_embeddings,
        embedding_dim,
        bits=quantization.bits,
        block_size=quantization.group_size,
        has_zero_point=not quantization.sym,
        padding_idx=module.padding_idx,
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


def _excluded_from_component_quantization(
    root: nn.Module,
    path: str,
    quantization: QuantizationConfig,
) -> bool:
    """Return whether a model-declared subtree stays float for this method."""
    parts = path.split(".")
    for end in range(1, len(parts) + 1):
        module = _resolve_path(root, ".".join(parts[:end]))
        methods = getattr(module, "component_quantization_excluded_methods", ())
        if quantization.quant_method in methods:
            return True
    return False


def _component_token_embedding_keys(
    module: nn.Module,
    component: str,
    component_path: str,
) -> tuple[str, ...]:
    """Return canonical float keys for token tables owned by one component."""
    component_module = _component_module(module, component_path)
    prefixes = {component_path, component} if component_path else {""}
    keys: set[str] = set()
    for name, child in component_module.named_modules():
        if (
            name
            and isinstance(child, (Embedding, QuantizedEmbedding))
            and name.rsplit(".", 1)[-1] in _TOKEN_EMBEDDING_NAMES
        ):
            for prefix in prefixes:
                path = f"{prefix}.{name}" if prefix else name
                keys.add(f"{path}.weight")
    return tuple(sorted(keys))


def _packed_qweight_for(key: str, float_key: str) -> bool:
    owner = float_key.removesuffix(".weight")
    return key in {f"{float_key}_qweight", f"{owner}.qweight"}


def _configure_component_module(
    component_module: nn.Module,
    config: BaseModelConfig,
    quantization: QuantizationConfig | None,
    *,
    output_head_paths: tuple[str, ...],
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
        is_lm_head = (
            name == "lm_head" or name.endswith(".lm_head") or name in output_head_paths
        )
        excluded = quantization is not None and _excluded_from_component_quantization(
            component_module,
            name,
            quantization,
        )

        if isinstance(child, ClippableQuantizedLinear):
            if (
                quantization is None
                or excluded
                or (is_lm_head and not quantization.quantize_lm_head)
            ):
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
            if (
                quantization is None
                or excluded
                or (is_lm_head and not quantization.quantize_lm_head)
            ):
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
            elif not _embedding_layout_matches(child, quantization):
                replacements.append((name, _quantized_embedding(child, quantization)))
            continue

        if quantization is None:
            continue
        if excluded:
            continue
        if is_lm_head and not quantization.quantize_lm_head:
            continue
        if name.split(".")[-1] in {"router", "shared_expert_gate"}:
            continue
        if isinstance(child, Embedding) and name.rsplit(".", 1)[-1] in (
            _TOKEN_EMBEDDING_NAMES
        ):
            if quantization.quantize_embeddings:
                if int(child.weight.shape[1]) % quantization.group_size != 0:
                    raise ValueError(
                        f"Embedding {name!r} dimension {int(child.weight.shape[1])} "
                        f"is not divisible by quantization group size "
                        f"{quantization.group_size}."
                    )
                replacements.append((name, _quantized_embedding(child, quantization)))
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
            output_head_paths=_component_output_head_paths(module, component),
        )


def _has_raw_packed_weight(
    names: Iterable[str],
    parameter_names: frozenset[str],
) -> bool:
    return any(
        name.endswith(("_qweight", ".qweight")) and name not in parameter_names
        for name in names
    )


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
    parameter_names = frozenset(name for name, _ in module.named_parameters())
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
        if not component_weights or not _has_raw_packed_weight(
            component_weights,
            parameter_names,
        ):
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
        output_head_paths = _component_output_head_paths(module, component)
        packed_lm_head = next(
            (
                key
                for key in component_weights
                if key.endswith(("_qweight", ".qweight"))
                and any(
                    key.startswith(f"{head_path}.") or f".{head_path}." in key
                    for head_path in output_head_paths
                )
            ),
            None,
        )
        if packed_lm_head is not None and not quantization.quantize_lm_head:
            raise ValueError(
                f"Component {component!r} keeps lm_head floating point, but "
                f"packed checkpoint weight {packed_lm_head!r} was found."
            )
        embedding_keys = _component_token_embedding_keys(
            module,
            component,
            component_path,
        )
        packed_embeddings = [
            (key, embedding_key)
            for key in component_weights
            for embedding_key in embedding_keys
            if _packed_qweight_for(key, embedding_key)
        ]
        packed_embedding = packed_embeddings[0][0] if packed_embeddings else None
        if packed_embedding is not None and not quantization.quantize_embeddings:
            raise ValueError(
                f"Component {component!r} keeps embeddings floating point, but "
                f"packed checkpoint weight {packed_embedding!r} was found."
            )
        if len(packed_embeddings) > 1:
            raise NotImplementedError(
                f"Component {component!r} contains multiple packed token tables; "
                "generic component preprocessing currently supports one per component."
            )
        if (
            packed_embedding is not None
            and quantization.quant_method != "olive"
            and quantization.quantize_embeddings
        ):
            raise NotImplementedError(
                "Generic component preprocessing supports packed token embeddings "
                "only for Olive checkpoints."
            )
        embed_key = (
            packed_embeddings[0][1]
            if packed_embeddings
            else embedding_keys[0]
            if len(embedding_keys) == 1
            else "__mobius_no_token_embedding__.weight"
        )

        converted = preprocess_quantized_weights(
            component_weights,
            quantization,
            tie_embeddings=False,
            embed_key=embed_key,
            qmoe_target_path=None,
        )
        if len(component_names) == 1:
            result = converted
        else:
            for key in component_weights:
                result.pop(key, None)
            result.update(converted)
    remaining_packed_key = next(
        (
            key
            for key in result
            if key.endswith(("_qweight", ".qweight")) and key not in parameter_names
        ),
        None,
    )
    if remaining_packed_key is not None:
        raise ValueError(
            f"Packed checkpoint weight {remaining_packed_key!r} was not routed "
            "to any ModelPackage component."
        )
    return result
