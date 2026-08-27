# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import onnx_ir as ir
import pytest
import safetensors.torch
import torch

from mobius._builder import build_from_module
from mobius.integrations.transformers import _qwen4_exp_weights as qwen4_weights
from mobius.integrations.transformers._qwen4_exp_weights import (
    _source_name,
    stream_qwen4_exp_safetensors_to_model,
    stream_qwen4_exp_safetensors_to_package,
)
from mobius.models.qwen4_exp import (
    Qwen4ExpCausalLMModel,
    Qwen4ExpForConditionalGeneration,
    _qwen4_exp_ple_buffer_values,
)
from mobius.models.qwen4_exp_test import _config, _vl_config


def _official_state(module: Qwen4ExpCausalLMModel) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for target_name, parameter in module.named_parameters():
        value = (
            torch.from_numpy(parameter._const_value.numpy().copy())
            if parameter._const_value is not None
            else torch.randn(tuple(int(dim) for dim in parameter.shape))
        )
        source_name = (
            f"model.language_model.{target_name[len('model.') :]}"
            if target_name.startswith("model.")
            else target_name
        )
        if target_name.endswith(".ple.ple_embedding.ngram_embedding.weight"):
            rows = value.shape[0]
            for shard_index, shard in enumerate(
                torch.tensor_split(value, module.config.split_ngram_parts, dim=0)
            ):
                state[f"{source_name[: -len('.weight')]}.shard_{shard_index}.weight"] = (
                    shard.contiguous()
                )
            assert (
                sum(
                    tensor.shape[0]
                    for name, tensor in state.items()
                    if name.startswith(source_name[: -len(".weight")])
                )
                == rows
            )
            continue
        state[source_name] = value
    return state


def test_streaming_loader_maps_packed_experts_and_ple_without_eager_checkpoint(
    tmp_path,
):
    config = _config(split_ngram_parts=2)
    module = Qwen4ExpCausalLMModel(config)
    package = build_from_module(
        module,
        config,
        task="qwen4-exp-text-generation",
    )
    source = _official_state(module)
    safetensors.torch.save_file(source, tmp_path / "model.safetensors")

    stream_qwen4_exp_safetensors_to_model(
        package["model"],
        str(tmp_path),
        config,
    )

    initializers = package["model"].graph.initializers
    assert all(value.const_value is not None for value in initializers.values())
    assert (
        initializers["model.layers.0.mlp.experts.gate_up_proj"].const_value.dtype
        == ir.DataType.FLOAT
    )
    ple_initializers = [
        initializer
        for initializer in initializers.values()
        if ".ple.ple_embedding.ngram_embedding.shard_" in initializer.name
        and initializer.name.endswith(".weight")
    ]
    assert len(ple_initializers) == config.split_ngram_parts
    assert all(
        isinstance(initializer.const_value, ir.LazyTensor) for initializer in ple_initializers
    )


def _official_package_state(package, config) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for model in package.values():
        for target_name, initializer in model.graph.initializers.items():
            if initializer.const_value is not None:
                if not target_name.endswith(
                    (
                        ".ple_embedding.layer_multipliers",
                        ".ple_embedding.ngram_heads_vocab_sizes",
                        ".ple_embedding.ngram_heads_offsets",
                    )
                ):
                    continue
                value = torch.from_numpy(initializer.const_value.numpy().copy())
            else:
                value = torch.randn(tuple(int(dim) for dim in initializer.shape))
            source_name = _source_name(target_name)
            if target_name.endswith(".ple.ple_embedding.ngram_embedding.weight"):
                for shard_index, shard in enumerate(
                    torch.tensor_split(value, config.split_ngram_parts, dim=0)
                ):
                    state[f"{source_name[: -len('.weight')]}.shard_{shard_index}.weight"] = (
                        shard.contiguous()
                    )
            else:
                existing = state.setdefault(source_name, value)
                assert existing.shape == value.shape
    for ple_layer_index, layer_id in enumerate(config.ple_layer_ids or []):
        buffers, _padded_vocab_size = _qwen4_exp_ple_buffer_values(
            config,
            ple_layer_index,
        )
        prefix = f"model.language_model.layers.{layer_id - 1}.ple.ple_embedding."
        for name, value in buffers.items():
            state[f"{prefix}{name}"] = torch.from_numpy(value.copy())
    return state


def test_multimodal_streaming_is_transactional_and_retains_no_source_tensors(
    tmp_path,
    monkeypatch,
):
    config = _vl_config(split_ngram_parts=2)
    package = build_from_module(
        Qwen4ExpForConditionalGeneration(config),
        config,
        task="qwen4-exp-vision-language",
    )
    source = _official_package_state(package, config)
    safetensors.torch.save_file(source, tmp_path / "model.safetensors")

    report = stream_qwen4_exp_safetensors_to_package(
        package,
        str(tmp_path),
        config,
    )

    assert report.model_count == 3
    assert report.lazy_initializer_count > 0
    assert report.initializer_count >= report.lazy_initializer_count
    assert 0 < report.eagerly_validated_bytes < 4096
    assert report.retained_source_tensor_count == 0
    for model in package.values():
        assert all(
            initializer.const_value is not None
            for initializer in model.graph.initializers.values()
        )
    ple_initializers = [
        initializer
        for initializer in package["decoder"].graph.initializers.values()
        if ".ple.ple_embedding.ngram_embedding.shard_" in initializer.name
        and initializer.name.endswith(".weight")
    ]
    assert len(ple_initializers) == config.split_ngram_parts
    assert all(
        isinstance(initializer.const_value, ir.LazyTensor) for initializer in ple_initializers
    )

    failed_package = build_from_module(
        Qwen4ExpForConditionalGeneration(config),
        config,
        task="qwen4-exp-vision-language",
    )
    missing_source = dict(source)
    missing_name = next(
        name for name in missing_source if name.startswith("model.visual.patch_embed")
    )
    missing_source.pop(missing_name)
    failed_dir = tmp_path / "missing"
    failed_dir.mkdir()
    safetensors.torch.save_file(
        missing_source,
        failed_dir / "model.safetensors",
    )
    unset_before = {
        (model_name, initializer.name)
        for model_name, model in failed_package.items()
        for initializer in model.graph.initializers.values()
        if initializer.const_value is None
    }
    with pytest.raises(ValueError, match="checkpoint has no source tensor"):
        stream_qwen4_exp_safetensors_to_package(
            failed_package,
            str(failed_dir),
            config,
        )
    unset_after = {
        (model_name, initializer.name)
        for model_name, model in failed_package.items()
        for initializer in model.graph.initializers.values()
        if initializer.const_value is None
    }
    assert unset_after == unset_before

    corrupt_package = build_from_module(
        Qwen4ExpForConditionalGeneration(config),
        config,
        task="qwen4-exp-vision-language",
    )
    corrupt_source = dict(source)
    multiplier_name = next(
        name for name in corrupt_source if name.endswith("layer_multipliers")
    )
    corrupt_source[multiplier_name] = torch.full_like(
        corrupt_source[multiplier_name],
        999,
    )
    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    safetensors.torch.save_file(
        corrupt_source,
        corrupt_dir / "model.safetensors",
    )
    corrupt_unset_before = {
        (model_name, initializer.name)
        for model_name, model in corrupt_package.items()
        for initializer in model.graph.initializers.values()
        if initializer.const_value is None
    }
    with pytest.raises(ValueError, match="does not match the pinned hash"):
        stream_qwen4_exp_safetensors_to_package(
            corrupt_package,
            str(corrupt_dir),
            config,
        )
    corrupt_unset_after = {
        (model_name, initializer.name)
        for model_name, model in corrupt_package.items()
        for initializer in model.graph.initializers.values()
        if initializer.const_value is None
    }
    assert corrupt_unset_after == corrupt_unset_before

    fold_failed_package = build_from_module(
        Qwen4ExpForConditionalGeneration(config),
        config,
        task="qwen4-exp-vision-language",
    )
    original_models = dict(fold_failed_package)
    unset_before_fold = {
        (model_name, initializer.name)
        for model_name, model in fold_failed_package.items()
        for initializer in model.graph.initializers.values()
        if initializer.const_value is None
    }
    real_fold = qwen4_weights.fold_initializers_after_weights
    fold_calls = 0

    def fail_second_fold(model):
        nonlocal fold_calls
        fold_calls += 1
        if fold_calls == 2:
            raise RuntimeError("synthetic fold failure")
        real_fold(model)

    monkeypatch.setattr(
        qwen4_weights,
        "fold_initializers_after_weights",
        fail_second_fold,
    )
    with pytest.raises(RuntimeError, match="synthetic fold failure"):
        stream_qwen4_exp_safetensors_to_package(
            fold_failed_package,
            str(tmp_path),
            config,
        )
    assert all(fold_failed_package[name] is model for name, model in original_models.items())
    unset_after_fold = {
        (model_name, initializer.name)
        for model_name, model in fold_failed_package.items()
        for initializer in model.graph.initializers.values()
        if initializer.const_value is None
    }
    assert unset_after_fold == unset_before_fold
