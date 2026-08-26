# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import onnx_ir as ir
import safetensors.torch
import torch

from mobius._builder import build_from_module
from mobius.models.qwen4_exp import Qwen4ExpCausalLMModel
from mobius.models.qwen4_exp_test import _config

from ._qwen4_exp_weights import stream_qwen4_exp_safetensors_to_model


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
                state[
                    f"{source_name[: -len('.weight')]}.shard_{shard_index}.weight"
                ] = shard.contiguous()
            assert sum(
                tensor.shape[0]
                for name, tensor in state.items()
                if name.startswith(source_name[: -len(".weight")])
            ) == rows
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
    assert initializers[
        "model.layers.0.mlp.experts.gate_up_proj"
    ].const_value.dtype == ir.DataType.FLOAT
    assert initializers[
        "model.layers.0.ple.ple_embedding.ngram_embedding.weight"
    ].const_value.shape == module.model.layers[0].ple.ple_embedding.ngram_embedding.weight.shape
