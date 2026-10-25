# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Flint checkpoint accounting and bounded expert-bank loading."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import safetensors.torch
import torch

from mobius._builder import build_from_module
from mobius.integrations._weight_loading import stream_preprocessed_safetensors_to_model
from mobius.integrations.transformers._flint_weights import build_flint_streaming_plan
from mobius.models.flint import FlintCausalLMModel
from mobius.models.flint_test import _config


def _checkpoint(module):
    state = {}
    for name, parameter in module.named_parameters():
        if parameter.const_value is not None:
            continue
        shape = tuple(parameter.shape)
        if name.endswith((".fc1_experts_weights", ".fc2_experts_weights")):
            prefix, bank = name.rsplit(".", 1)
            projections = (
                ("gate_proj", "up_proj") if bank == "fc1_experts_weights" else ("down_proj",)
            )
            rows = shape[1] // len(projections)
            for expert in range(shape[0]):
                for index, projection in enumerate(projections):
                    state[f"{prefix}.experts.{expert}.{projection}.weight"] = torch.full(
                        (rows, shape[2]), 10 * expert + index + 1, dtype=torch.bfloat16
                    )
        else:
            state[name] = torch.full(shape, 0.5, dtype=torch.bfloat16)
    return state


def test_stream_flint_experts_and_fp32_head(tmp_path):
    config = _config(dtype=ir.DataType.BFLOAT16)
    module = FlintCausalLMModel(config)
    package = build_from_module(module, config)
    source = _checkpoint(module)
    safetensors.torch.save_file(source, tmp_path / "model.safetensors")
    model = package["model"]
    report = stream_preprocessed_safetensors_to_model(
        model,
        str(tmp_path),
        lambda index, initializers: build_flint_streaming_plan(config, index, initializers),
    )
    assert report["accounted_source_tensor_count"] == len(source)
    for layer in (1, 2):
        prefix = f"model.layers.{layer}.mlp"
        fc1 = model.graph.initializers[f"{prefix}.fc1_experts_weights"]
        bank = fc1.const_value.numpy().astype(np.float32)
        for expert in range(config.num_local_experts):
            np.testing.assert_array_equal(
                bank[expert, : config.moe_intermediate_size], 10 * expert + 1
            )
            np.testing.assert_array_equal(
                bank[expert, config.moe_intermediate_size :], 10 * expert + 2
            )
    head = model.graph.initializers["lm_head.weight"]
    assert head.const_value.dtype == ir.DataType.FLOAT
    np.testing.assert_array_equal(head.const_value.numpy(), 0.5)


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "dtype"])
def test_flint_plan_rejects_unaccounted_or_malformed_sources(mutation):
    config = _config()
    module = FlintCausalLMModel(config)
    package = build_from_module(module, config)
    state = _checkpoint(module)
    index = {name: ("unused", list(value.shape), "BF16") for name, value in state.items()}
    name = "model.layers.1.mlp.experts.0.gate_proj.weight"
    if mutation == "missing":
        del index[name]
    elif mutation == "extra":
        index["unmapped.weight"] = ("unused", [1], "BF16")
    elif mutation == "shape":
        index[name] = ("unused", [1, 2], "BF16")
    else:
        index[name] = ("unused", list(state[name].shape), "I64")
    with pytest.raises(ValueError):
        build_flint_streaming_plan(config, index, package["model"].graph.initializers)
