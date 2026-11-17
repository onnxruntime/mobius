# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bounded-memory safetensors loading for the Qwen4-Exp text core."""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import torch
from onnx_ir import tensor_adapters
from safetensors import safe_open

from mobius._configs import Qwen4ExpConfig
from mobius._model_package import ModelPackage
from mobius._optimizations import fold_initializers_after_weights
from mobius.integrations._weight_loading import _resolve_shard_paths, _shard_key_index
from mobius.models.qwen4_exp import _qwen4_exp_ple_buffer_values

_PLE_TABLE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight"


def _source_name(target_name: str) -> str:
    if target_name.startswith("decoder."):
        return _source_name(target_name[len("decoder.") :])
    if target_name == "embedding.embed_tokens.weight":
        return "model.language_model.embed_tokens.weight"
    if target_name.startswith("vision_encoder.visual."):
        source_name = f"model.visual.{target_name[len('vision_encoder.visual.') :]}"
        source_name = source_name.replace(".mlp.up_proj.", ".mlp.linear_fc1.")
        return source_name.replace(".mlp.down_proj.", ".mlp.linear_fc2.")
    if target_name.startswith("model."):
        return f"model.language_model.{target_name[len('model.') :]}"
    return target_name


def _validate_shape(name: str, actual: list[int], expected: list[int]) -> None:
    if actual != expected:
        raise ValueError(
            f"Qwen4-Exp streamed weight '{name}' has shape {actual}, expected {expected}"
        )


def _lazy_source_tensor(
    initializer: ir.Value,
    source_path: str,
    source_name: str,
    *,
    source_slice: tuple[int | slice, ...] | None = None,
) -> ir.LazyTensor:
    onnx_dtype = initializer.dtype
    assert onnx_dtype is not None
    target_dtype = tensor_adapters.to_torch_dtype(onnx_dtype)
    target_name = initializer.name
    assert target_name is not None

    def load() -> tensor_adapters.TorchTensor:
        with safe_open(source_path, framework="pt") as handle:
            tensor = (
                handle.get_tensor(source_name)
                if source_slice is None
                else handle.get_slice(source_name)[source_slice]
            )
        if tensor.dtype != target_dtype:
            tensor = tensor.to(target_dtype)
        return tensor_adapters.TorchTensor(tensor, name=target_name)

    return ir.LazyTensor(
        load,
        dtype=onnx_dtype,
        shape=ir.Shape(initializer.shape),
        name=target_name,
    )


def _lazy_concat_tensor(
    initializer: ir.Value,
    sources: list[tuple[str, str]],
) -> ir.LazyTensor:
    onnx_dtype = initializer.dtype
    assert onnx_dtype is not None
    target_dtype = tensor_adapters.to_torch_dtype(onnx_dtype)
    target_name = initializer.name
    assert target_name is not None
    target_shape = tuple(int(dim) for dim in initializer.shape)

    def load() -> tensor_adapters.TorchTensor:
        # Allocate the final table once and copy one source shard at a time.
        # Peak host memory is the 95 GiB target plus one PLE shard, rather than
        # the full checkpoint plus a second concatenated table.
        output = torch.empty(target_shape, dtype=target_dtype)
        row = 0
        for source_path, source_name in sources:
            with safe_open(source_path, framework="pt") as handle:
                shard = handle.get_tensor(source_name)
            if shard.dtype != target_dtype:
                shard = shard.to(target_dtype)
            next_row = row + shard.shape[0]
            output[row:next_row].copy_(shard)
            row = next_row
        if row != target_shape[0]:
            raise ValueError(
                f"Qwen4-Exp streamed PLE table populated {row} rows, "
                f"expected {target_shape[0]}"
            )
        return tensor_adapters.TorchTensor(output, name=target_name)

    return ir.LazyTensor(
        load,
        dtype=onnx_dtype,
        shape=ir.Shape(initializer.shape),
        name=target_name,
    )


@dataclasses.dataclass(frozen=True)
class Qwen4ExpStreamingReport:
    """Bounded-memory accounting for one transactional package binding."""

    model_count: int
    initializer_count: int
    lazy_initializer_count: int
    eagerly_validated_bytes: int
    retained_source_tensor_count: int = 0


def _validate_unquantized_checkpoint(key_index) -> None:
    quantized = sorted(
        name
        for name, (_path, _shape, dtype) in key_index.items()
        if dtype.startswith("F8") or name.endswith(("_scale_inv", "weight_scale"))
    )
    if quantized:
        raise ValueError(
            "Qwen4-Exp streaming requires the unquantized BF16 checkpoint; "
            f"found quantized tensors such as {quantized[:5]}"
        )


def _validate_deterministic_ple_buffers(
    config: Qwen4ExpConfig,
    key_index,
) -> int:
    validated_bytes = 0
    for ple_layer_index, layer_id in enumerate(config.ple_layer_ids or []):
        expected_values, _padded_vocab_size = _qwen4_exp_ple_buffer_values(
            config,
            ple_layer_index,
        )
        prefix = f"model.language_model.layers.{layer_id - 1}.ple.ple_embedding."
        for buffer_name, expected_array in expected_values.items():
            source_name = f"{prefix}{buffer_name}"
            located = key_index.get(source_name)
            if located is None:
                raise ValueError(
                    f"Qwen4-Exp checkpoint is missing deterministic buffer '{source_name}'"
                )
            source_path, source_shape, _dtype = located
            _validate_shape(source_name, source_shape, list(expected_array.shape))
            with safe_open(source_path, framework="pt") as handle:
                actual = handle.get_tensor(source_name)
            expected = torch.from_numpy(expected_array)
            if not torch.equal(actual.cpu(), expected):
                raise ValueError(
                    f"Qwen4-Exp deterministic buffer {source_name} does not "
                    "match the pinned hash construction"
                )
            validated_bytes += actual.numel() * actual.element_size()
    return validated_bytes


def _plan_model_bindings(
    model: ir.Model,
    config: Qwen4ExpConfig,
    key_index,
) -> tuple[list[tuple[ir.Value, ir.TensorProtocol]], int]:
    bindings: list[tuple[ir.Value, ir.TensorProtocol]] = []
    eagerly_validated_bytes = 0
    parameter_names = set(model.graph.initializers)
    for target_name, initializer in model.graph.initializers.items():
        if initializer.const_value is not None:
            continue

        if target_name.endswith(_PLE_TABLE_SUFFIX):
            prefix = target_name[: -len(".weight")]
            sources = []
            source_rows = 0
            expected_width = int(initializer.shape[1])
            for shard_index in range(config.split_ngram_parts):
                source_name = _source_name(f"{prefix}.shard_{shard_index}.weight")
                located = key_index.get(source_name)
                if located is None:
                    raise ValueError(
                        f"Qwen4-Exp checkpoint is missing PLE shard '{source_name}'"
                    )
                source_path, source_shape, _dtype = located
                if len(source_shape) != 2 or source_shape[1] != expected_width:
                    raise ValueError(
                        f"Qwen4-Exp PLE shard '{source_name}' has shape {source_shape}, "
                        f"expected [rows, {expected_width}]"
                    )
                source_rows += source_shape[0]
                sources.append((source_path, source_name))
            if source_rows != int(initializer.shape[0]):
                raise ValueError(
                    f"Qwen4-Exp PLE shards contain {source_rows} rows, "
                    f"expected {int(initializer.shape[0])}"
                )
            bindings.append((initializer, _lazy_concat_tensor(initializer, sources)))
            continue

        source_name = _source_name(target_name)
        located = key_index.get(source_name)
        if located is None:
            raise ValueError(
                f"Qwen4-Exp checkpoint has no source tensor for initializer "
                f"'{target_name}' (expected '{source_name}')"
            )
        source_path, source_shape, _dtype = located
        _validate_shape(source_name, source_shape, [int(d) for d in initializer.shape])
        bindings.append(
            (initializer, _lazy_source_tensor(initializer, source_path, source_name))
        )

    assigned = {initializer.name for initializer, _value in bindings}
    unassigned = [
        name
        for name in parameter_names
        if model.graph.initializers[name].const_value is None and name not in assigned
    ]
    if unassigned:
        raise ValueError(f"Qwen4-Exp streaming left initializers unassigned: {unassigned[:5]}")
    return bindings, eagerly_validated_bytes


def _stage_model(
    model: ir.Model,
    config: Qwen4ExpConfig,
    key_index,
) -> tuple[ir.Model, int, int]:
    staged = model.clone()
    bindings, validated_bytes = _plan_model_bindings(staged, config, key_index)
    for initializer, value in bindings:
        initializer.const_value = value
    fold_initializers_after_weights(staged)
    return staged, len(bindings), validated_bytes


def stream_qwen4_exp_safetensors_to_model(
    model: ir.Model,
    model_id: str,
    config: Qwen4ExpConfig,
    *,
    revision: str | None = None,
) -> Qwen4ExpStreamingReport:
    """Bind one Qwen4-Exp graph without retaining the checkpoint state dict."""
    paths = _resolve_shard_paths(model_id, revision)
    key_index = _shard_key_index(paths)
    _validate_unquantized_checkpoint(key_index)
    deterministic_bytes = _validate_deterministic_ple_buffers(config, key_index)
    staged, lazy_count, validated_bytes = _stage_model(model, config, key_index)
    model.graph = staged.graph
    return Qwen4ExpStreamingReport(
        model_count=1,
        initializer_count=len(model.graph.initializers),
        lazy_initializer_count=lazy_count,
        eagerly_validated_bytes=deterministic_bytes + validated_bytes,
    )


def stream_qwen4_exp_safetensors_to_package(
    package: ModelPackage,
    model_id: str,
    config: Qwen4ExpConfig,
    *,
    revision: str | None = None,
) -> Qwen4ExpStreamingReport:
    """Transactionally bind every Qwen4-Exp package component from one shard index."""
    paths = _resolve_shard_paths(model_id, revision)
    key_index = _shard_key_index(paths)
    _validate_unquantized_checkpoint(key_index)
    validated_bytes = _validate_deterministic_ple_buffers(config, key_index)

    staged_models: dict[str, ir.Model] = {}
    initializer_count = 0
    lazy_initializer_count = 0
    for name, model in package.items():
        staged, model_lazy_count, model_validated_bytes = _stage_model(
            model,
            config,
            key_index,
        )
        staged_models[name] = staged
        validated_bytes += model_validated_bytes
        initializer_count += len(staged.graph.initializers)
        lazy_initializer_count += model_lazy_count

    # Commit only after every cloned component has bound and folded successfully.
    package.update(staged_models)
    return Qwen4ExpStreamingReport(
        model_count=len(package),
        initializer_count=initializer_count,
        lazy_initializer_count=lazy_initializer_count,
        eagerly_validated_bytes=validated_bytes,
    )
