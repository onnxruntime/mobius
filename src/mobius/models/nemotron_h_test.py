# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from collections import Counter

import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import NemotronHConfig, QuantizationConfig
from mobius._registry import registry
from mobius.integrations.gguf._architecture import validate_package_state_dict
from mobius.integrations.transformers import _default_task_for_model


def _config(*, quantization: QuantizationConfig | None = None) -> NemotronHConfig:
    return NemotronHConfig(
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="relu2",
        pad_token_id=0,
        layer_types=["mamba2", "moe", "full_attention"],
        num_local_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=64,
        mamba_n_heads=2,
        mamba_d_head=32,
        mamba_d_state=16,
        mamba_n_groups=2,
        mamba_d_conv=4,
        quantization=quantization,
    )


def _build_package(config: NemotronHConfig):
    module = registry.get("nemotron_h")(config)
    return build_from_module(
        module,
        config,
        _default_task_for_model("nemotron_h"),
    )


def _build_graph(config: NemotronHConfig):
    return _build_package(config)["model"].graph


def test_float_nemotron_retains_safetensors_projection_contract() -> None:
    graph = _build_graph(_config())
    ops = Counter((node.domain, node.op_type) for node in graph.all_nodes())

    assert ops["com.microsoft", "MatMulNBits"] == 0
    assert ops["com.microsoft", "GatherBlockQuantized"] == 0
    assert len(graph.initializers) == 34
    assert graph.initializers["model.layers.0.mamba.in_proj.weight"].shape == [194, 64]
    assert graph.initializers["model.layers.1.moe.experts.0.up_proj.weight"].shape == [32, 64]
    assert graph.initializers["model.layers.2.self_attn.q_proj.weight"].shape == [64, 64]
    assert graph.initializers["model.embed_tokens.weight"].shape == [256, 64]
    assert graph.initializers["lm_head.weight"].shape == [256, 64]
    assert not any(name.endswith((".scales", ".zero_points")) for name in graph.initializers)


def test_quantized_nemotron_wires_every_projection_family() -> None:
    config = _config(
        quantization=QuantizationConfig(
            bits=8,
            group_size=32,
            quant_method="gguf",
            sym=False,
            quantize_embeddings=True,
            quantize_lm_head=True,
        )
    )
    graph = _build_graph(config)
    ops = Counter((node.domain, node.op_type) for node in graph.all_nodes())

    assert ops["com.microsoft", "MatMulNBits"] == 13
    assert ops["com.microsoft", "GatherBlockQuantized"] == 1
    assert len(graph.initializers) == 62
    assert graph.initializers["model.layers.0.mamba.in_proj.weight"].shape == [
        194,
        2,
        32,
    ]
    assert graph.initializers["model.layers.1.moe.experts.0.up_proj.weight"].shape == [
        32,
        2,
        32,
    ]
    assert graph.initializers["model.layers.2.self_attn.q_proj.weight"].shape == [
        64,
        2,
        32,
    ]
    assert graph.initializers["model.embed_tokens.qweight"].shape == [256, 64]
    assert graph.initializers["lm_head.weight"].shape == [256, 2, 32]


def test_gguf_state_dict_requires_exact_initializer_coverage() -> None:
    package = _build_package(_config())
    required = {
        name: torch.empty(0)
        for name, value in package["model"].graph.initializers.items()
        if value.const_value is None
    }

    validate_package_state_dict(package, required)
    missing = dict(required)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="1 missing"):
        validate_package_state_dict(package, missing)
    with pytest.raises(ValueError, match="1 unexpected"):
        validate_package_state_dict(package, {**required, "unexpected.weight": torch.empty(0)})
