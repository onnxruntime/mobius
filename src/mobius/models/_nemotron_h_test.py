# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Lightning config, checkpoint accounting, routing dtype and recurrent ABI regressions."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch

from mobius import build_from_module
from mobius._builder import _cast_module_dtype
from mobius._configs import NemotronHConfig
from mobius._testing import create_test_builder, create_test_input
from mobius.integrations._weight_loading import apply_weights
from mobius.models.nemotron_h import (
    NemotronHCausalLMModel,
    NemotronHMoEBlock,
    _rename_nemotron_h_weight,
)

# Selected architecture fields from pinned BF16 config
# a9904d24bcc1d289a1950fa9d2b978c47cf903b9 (raw SHA-256 in the recipe).
# NVFP4 uses the same topology, but max_position_embeddings=1048576.
_LIGHTNING = {
    "model_type": "nemotron_h",
    "hidden_size": 2688,
    "intermediate_size": 1856,
    "vocab_size": 131072,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "num_hidden_layers": 52,
    "max_position_embeddings": 262144,
    "num_nextn_predict_layers": 1,
    "layer_norm_epsilon": 1e-5,
    "layers_block_type": [
        "mamba",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "attention",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "attention",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "attention",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "attention",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "attention",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "attention",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "moe",
    ],
    "mamba_num_heads": 64,
    "mamba_head_dim": 64,
    "n_groups": 8,
    "ssm_state_size": 128,
    "mamba_ssm_cache_dtype": "float32",
    "n_routed_experts": 128,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 1856,
    "moe_shared_expert_intermediate_size": 3712,
    "routed_scaling_factor": 2.5,
    "norm_topk_prob": True,
    "dtype": "bfloat16",
}


def _tiny_config(**kwargs):
    config = NemotronHConfig(
        model_type="nemotron_h",
        hidden_size=32,
        vocab_size=48,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["mamba2", "moe", "full_attention"],
        mamba_n_heads=4,
        mamba_d_head=8,
        mamba_d_state=8,
        mamba_n_groups=2,
        mamba_ssm_dtype=ir.DataType.FLOAT,
        hidden_act="relu2",
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=64,
        routed_scaling_factor=2.5,
        rms_norm_eps=1e-5,
    )
    return dataclasses.replace(config, **kwargs)


@pytest.mark.parametrize("context", [262144, 1048576])
def test_pinned_lightning_architecture_fields(context):
    raw = SimpleNamespace(**{**_LIGHTNING, "max_position_embeddings": context})
    config = NemotronHConfig.from_transformers(raw)
    assert len(config.layer_types) == 52
    assert config.layer_types.count("mamba2") == 23
    assert config.layer_types.count("moe") == 23
    assert config.layer_types.count("full_attention") == 6
    assert config.num_local_experts == 128 and config.num_experts_per_tok == 6
    assert config.shared_expert_intermediate_size == 3712
    assert config.mamba_n_heads == 64 and config.mamba_d_head == 64
    assert config.mamba_ssm_dtype == ir.DataType.FLOAT
    assert config.rms_norm_eps == pytest.approx(1e-5) and config.hidden_act == "relu2"
    assert config.max_position_embeddings == context
    with pytest.raises(NotImplementedError, match="NextN/MTP"):
        NemotronHCausalLMModel(config)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_mamba_cache_abi_preserves_configured_float32_ssm(dtype):
    config = _tiny_config(dtype=dtype)
    package = build_from_module(
        NemotronHCausalLMModel(config), config, "hybrid-text-generation"
    )
    graph = package["model"].graph
    inputs = {x.name: x for x in graph.inputs}
    outputs = {x.name: x for x in graph.outputs}
    assert inputs["past_key_values.0.conv_state"].dtype == dtype
    assert inputs["past_key_values.0.ssm_state"].dtype == ir.DataType.FLOAT
    assert list(inputs["past_key_values.0.ssm_state"].shape)[1:] == [4, 8, 8]
    assert outputs["present.0.ssm_state"].dtype == ir.DataType.FLOAT
    assert inputs["past_key_values.2.key"].dtype == dtype
    assert not any("recurrent_state" in name for name in inputs)
    # Every statically typed MatMul/Mul has homogeneous input types, including
    # FP32 routing and accumulation. ONNX has no implicit mixed-type promotion.
    for node in graph:
        if node.op_type in {"MatMul", "Mul", "Add"}:
            dtypes = {x.dtype for x in node.inputs if x is not None and x.dtype is not None}
            assert len(dtypes) <= 1, (node.name, dtypes)


def test_expert_routing_relu2_and_sigmoid_matches_independent_reference(tmp_path):
    config = _tiny_config()
    block = NemotronHMoEBlock(config)
    builder, op, graph = create_test_builder()
    value = create_test_input(builder, "hidden", [2, 3, 32])
    builder.add_output(block(op, value), "out")
    model = ir.Model(graph, ir_version=11)
    rng = np.random.default_rng(19)
    weights = {
        name: torch.from_numpy(rng.normal(0, 0.12, list(x.shape)).astype(np.float32))
        for name, x in graph.initializers.items()
        if x.const_value is None
    }
    # Nonzero correction bias must affect selection, not the mixture weights.
    weights["gate.e_score_correction_bias"] = torch.tensor([0.8, -0.4, 0.3, -0.6])
    apply_weights(model, weights)
    path = tmp_path / "moe.onnx"
    ir.save(model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    hidden = torch.from_numpy(rng.normal(size=(2, 3, 32)).astype(np.float32))
    scores = torch.sigmoid(hidden @ weights["gate.weight"].T)
    selected = torch.topk(scores + weights["gate.e_score_correction_bias"], 2, dim=-1).indices
    mixture = scores.gather(-1, selected)
    mixture = mixture / (mixture.sum(-1, keepdim=True) + 1e-20) * 2.5
    expected = torch.zeros_like(hidden)
    for idx in range(4):
        up = torch.relu(hidden @ weights[f"experts.{idx}.up_proj.weight"].T).square()
        expert = up @ weights[f"experts.{idx}.down_proj.weight"].T
        mass = (mixture * (selected == idx)).sum(-1, keepdim=True)
        expected += expert * mass
    shared = torch.relu(hidden @ weights["shared_experts.up_proj.weight"].T).square()
    expected += shared @ weights["shared_experts.down_proj.weight"].T
    actual = session.run(None, {"hidden": hidden.numpy()})[0]
    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6, rtol=1e-5)


def test_router_correction_bias_remains_float32():
    block = NemotronHMoEBlock(_tiny_config())
    _cast_module_dtype(block, ir.DataType.BFLOAT16)
    assert block.gate.e_score_correction_bias.dtype == ir.DataType.FLOAT
    assert block.gate.weight.dtype == ir.DataType.BFLOAT16


def test_mtp_and_duplicate_source_weights_are_not_silently_dropped():
    module = NemotronHCausalLMModel(_tiny_config())
    with pytest.raises(NotImplementedError, match="MTP tensors"):
        module.preprocess_weights({"mtp.layers.0.hnorm.weight": torch.ones(32)})
    with pytest.raises(ValueError, match="Duplicate"):
        module.preprocess_weights(
            {
                "backbone.embeddings.weight": torch.ones(48, 32),
                "model.embeddings.weight": torch.ones(48, 32),
            }
        )
    with pytest.raises(ValueError, match="out of range"):
        _rename_nemotron_h_weight("backbone.layers.3.mixer.q_proj.weight", ["mamba2"])


def test_reject_unimplemented_grouped_expert_routing():
    with pytest.raises(NotImplementedError, match="grouped expert"):
        NemotronHConfig.from_transformers(SimpleNamespace(**{**_LIGHTNING, "n_group": 2}))
