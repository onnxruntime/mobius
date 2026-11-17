# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph and executable synthetic-reference tests for Hunyuan-V3."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import torch
from onnxscript import nn

from mobius import build_from_module
from mobius._configs import HyV3Config, HyV3MtpConfig
from mobius._optimizations import SymbolicShapeInferencePass
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.hy_v3 import (
    HyV3CausalLMModel,
    HyV3MtpModel,
    HyV3TopKGate,
    _preprocess_hy_v3_weights,
)
from mobius.tasks import CausalLMTask, HyV3MtpTask


def _config(config_class=HyV3Config, **overrides):
    fields = dict(
        model_type="hy_v3",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        rope_theta=10_000.0,
        rope_type="default",
        tie_word_embeddings=False,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        first_k_dense_replace=1,
        attn_qk_norm=True,
        attn_qk_norm_full=False,
        norm_topk_prob=True,
        routing_weight_normalization_floor=6.103515625e-5,
        routing_weight_normalization_epsilon=None,
        routed_scaling_factor=2.826,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        use_expert_bias=True,
        disable_qmoe=True,
    )
    fields.update(overrides)
    return config_class(**fields)


def test_hy_v3_builds_dense_prefix_and_full_attention_graph() -> None:
    config = _config()
    package = CausalLMTask().build(HyV3CausalLMModel(config), config)
    model = package["model"]
    names = set(model.graph.initializers)

    assert sum(node.op_type == "Attention" for node in model.graph) == 2
    assert "model.layers.0.mlp.gate_proj.weight" in names
    assert "model.layers.1.mlp.gate.weight" in names
    assert "model.layers.1.mlp.e_score_correction_bias" in names
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in names
    assert "model.layers.1.self_attn.q_norm.weight" in names
    assert "model.layers.1.self_attn.k_norm.weight" in names
    assert not any("hyper" in name for name in names)


def test_hy_v3_preprocesses_separate_stacked_experts() -> None:
    gate = np.arange(4 * 8 * 16, dtype=np.float32).reshape(4, 8, 16)
    up = gate + 1000
    result = _preprocess_hy_v3_weights(
        {
            "model.layers.1.mlp.experts.gate_proj.weight": torch.from_numpy(gate),
            "model.layers.1.mlp.experts.up_proj.weight": torch.from_numpy(up),
        }
    )

    assert set(result) == {
        f"model.layers.1.mlp.experts.{expert}.{projection}_proj.weight"
        for expert in range(4)
        for projection in ("gate", "up")
    }
    np.testing.assert_array_equal(
        result["model.layers.1.mlp.experts.2.gate_proj.weight"], gate[2]
    )
    np.testing.assert_array_equal(result["model.layers.1.mlp.experts.2.up_proj.weight"], up[2])


def test_hy_v3_preprocesses_official_raw_weight_names_collision_safely() -> None:
    router = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    expert_bias = torch.arange(4, dtype=torch.float32)
    shared_gate = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
    shared_up = shared_gate + 1
    shared_down = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
    result = _preprocess_hy_v3_weights(
        {
            "model.layers.1.mlp.router.gate.weight": router,
            "model.layers.1.mlp.expert_bias": expert_bias,
            "model.layers.1.mlp.shared_mlp.gate_proj.weight": shared_gate,
            "model.layers.1.mlp.shared_mlp.up_proj.weight": shared_up,
            "model.layers.1.mlp.shared_mlp.down_proj.weight": shared_down,
        }
    )

    assert result["model.layers.1.mlp.gate.weight"] is router
    assert result["model.layers.1.mlp.e_score_correction_bias"] is expert_bias
    assert result["model.layers.1.mlp.shared_experts.gate_proj.weight"] is shared_gate
    assert result["model.layers.1.mlp.shared_experts.up_proj.weight"] is shared_up
    assert result["model.layers.1.mlp.shared_experts.down_proj.weight"] is shared_down

    with pytest.raises(ValueError, match="HYV3 weight rename collision"):
        _preprocess_hy_v3_weights(
            {
                "model.layers.1.mlp.router.gate.weight": router,
                "model.layers.1.mlp.gate.weight": router,
            }
        )


def test_hy_v3_mtp_uses_exactly_one_independent_cache_layer() -> None:
    config = _config(
        HyV3MtpConfig,
        num_hidden_layers=1,
        first_k_dense_replace=0,
        use_dedicated_embeddings=False,
        use_dedicated_lm_head=False,
    )
    package = HyV3MtpTask().build(HyV3MtpModel(config), config)
    model = package["model"]

    assert [value.name for value in model.graph.inputs if "past_key_values" in value.name] == [
        "past_key_values.0.key",
        "past_key_values.0.value",
    ]
    assert [value.name for value in model.graph.outputs if "present" in value.name] == [
        "present.0.key",
        "present.0.value",
    ]
    assert model.graph.outputs[0].name == "mtp_hidden"


def test_hy_v3_router_matches_selection_biased_sigmoid_reference() -> None:
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 2, 3], dtype=ir.DataType.FLOAT16)
    correction = create_test_input(builder, "correction", [4], dtype=ir.DataType.FLOAT16)
    gate = HyV3TopKGate(
        3,
        4,
        2,
        normalize=True,
        normalization_floor=6.103515625e-5,
        normalization_epsilon=None,
        routed_scaling_factor=2.826,
    )
    weights, experts = gate(op, hidden, correction)
    weights.name = "weights"
    experts.name = "experts"
    graph.outputs.extend((weights, experts))
    model = ir.Model(graph, ir_version=10)

    matrix = np.array(
        [[0.2, -0.4, 0.8], [-0.7, 0.3, 0.1], [0.5, 0.6, -0.2], [-0.1, -0.8, 0.4]],
        dtype=np.float32,
    )
    model.graph.initializers["weight"].const_value = ir.tensor(matrix)
    hidden_value = np.array([[[0.4, -0.2, 0.7], [-0.3, 0.8, 0.1]]], dtype=np.float16)
    correction_value = np.array([0.0, 0.35, -0.1, 0.2], dtype=np.float16)
    session = OnnxModelSession(model, device="cpu")
    actual = session.run({"hidden": hidden_value, "correction": correction_value})
    session.close()

    probabilities = 1.0 / (1.0 + np.exp(-(hidden_value @ matrix.T)))
    selected = np.argpartition(probabilities + correction_value, -2, axis=-1)[..., -2:]
    gathered = np.take_along_axis(probabilities, selected, axis=-1)
    expected_weights = gathered / np.maximum(
        gathered.sum(axis=-1, keepdims=True), 6.103515625e-5
    )
    expected_weights *= 2.826

    for token in range(hidden_value.shape[1]):
        actual_by_expert = dict(
            zip(
                actual["experts"][0, token].tolist(),
                actual["weights"][0, token].tolist(),
            )
        )
        expected_by_expert = dict(
            zip(selected[0, token].tolist(), expected_weights[0, token].tolist())
        )
        assert actual_by_expert.keys() == expected_by_expert.keys()
        for expert, expected in expected_by_expert.items():
            np.testing.assert_allclose(
                actual_by_expert[expert], expected, rtol=1e-6, atol=1e-6
            )


@pytest.mark.parametrize(
    "dtype",
    [ir.DataType.FLOAT16, ir.DataType.BFLOAT16],
)
@pytest.mark.parametrize(
    "config_class,task,num_hidden_layers,first_k_dense_replace",
    [
        (HyV3Config, "text-generation", 2, 1),
        (HyV3MtpConfig, HyV3MtpTask(), 1, 0),
    ],
)
def test_hy_v3_reduced_precision_keeps_selection_bias_in_float32(
    config_class,
    task,
    num_hidden_layers: int,
    first_k_dense_replace: int,
    dtype: ir.DataType,
) -> None:
    overrides = dict(
        dtype=dtype,
        num_hidden_layers=num_hidden_layers,
        first_k_dense_replace=first_k_dense_replace,
    )
    if config_class is HyV3MtpConfig:
        overrides.update(use_dedicated_embeddings=False, use_dedicated_lm_head=False)
    config = _config(config_class, **overrides)
    module = HyV3CausalLMModel(config) if config_class is HyV3Config else HyV3MtpModel(config)
    model = build_from_module(module, config, task=task)["model"]
    SymbolicShapeInferencePass()(model)

    bias_name = (
        "model.layers.1.mlp.e_score_correction_bias"
        if config_class is HyV3Config
        else "layers.0.mlp.e_score_correction_bias"
    )
    assert model.graph.initializers[bias_name].dtype == ir.DataType.FLOAT
    selection_add = next(
        node
        for node in model.graph
        if node.op_type == "Add"
        and any(value is not None and value.name == bias_name for value in node.inputs)
    )
    assert all(
        value is None or value.dtype == ir.DataType.FLOAT for value in selection_add.inputs
    )


def test_hy_v3_float32_bias_preserves_near_boundary_expert_selection() -> None:
    correction_values = np.array([1.0002, 1.0, 2.0, -1.0], dtype=np.float32)
    router_logits = np.array([0.0, 0.0004, 0.0, 0.0], dtype=np.float32)

    def run(correction_dtype: ir.DataType) -> set[int]:
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 1, 1], dtype=ir.DataType.FLOAT16)
        gate = HyV3TopKGate(
            1,
            4,
            2,
            normalize=False,
            normalization_floor=None,
            normalization_epsilon=None,
            routed_scaling_factor=1.0,
        )
        correction = nn.Parameter([4], dtype=correction_dtype)
        correction.const_value = ir.tensor(
            correction_values.astype(
                np.float32 if correction_dtype is ir.DataType.FLOAT else np.float16
            )
        )
        experts = gate(op, hidden, correction)[1]
        experts.name = "experts"
        graph.outputs.append(experts)
        correction.name = "correction"
        graph.register_initializer(correction)
        model = ir.Model(graph, ir_version=10)
        model.graph.initializers["weight"].const_value = ir.tensor(router_logits[:, None])

        session = OnnxModelSession(model, device="cpu")
        actual = session.run({"hidden": np.ones((1, 1, 1), dtype=np.float16)})["experts"]
        session.close()
        return set(actual.reshape(-1).tolist())

    probabilities = 1.0 / (1.0 + np.exp(-router_logits))
    expected = np.argpartition(probabilities + correction_values, -2)[-2:]
    assert run(ir.DataType.FLOAT) == set(expected.tolist()) == {0, 2}
    assert run(ir.DataType.FLOAT16) == {1, 2}


def test_hy_v3_official_router_uses_additive_normalization_epsilon() -> None:
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 1, 1])
    correction = create_test_input(builder, "correction", [2])
    gate = HyV3TopKGate(
        1,
        2,
        2,
        normalize=True,
        normalization_floor=None,
        normalization_epsilon=1e-20,
        routed_scaling_factor=1.0,
    )
    weights, _ = gate(op, hidden, correction)
    weights.name = "weights"
    graph.outputs.append(weights)
    model = ir.Model(graph, ir_version=10)
    model.graph.initializers["weight"].const_value = ir.tensor(
        np.array([[-12.0], [-12.0]], dtype=np.float32)
    )

    session = OnnxModelSession(model, device="cpu")
    actual = session.run(
        {
            "hidden": np.ones((1, 1, 1), dtype=np.float32),
            "correction": np.zeros(2, dtype=np.float32),
        }
    )["weights"]
    session.close()

    np.testing.assert_allclose(actual, np.array([[[0.5, 0.5]]]), rtol=1e-6, atol=1e-6)


def test_hy_v3_llamacpp_router_clamps_tiny_probability_sum() -> None:
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 1, 1])
    correction = create_test_input(builder, "correction", [2])
    floor = 6.103515625e-5
    gate = HyV3TopKGate(
        1,
        2,
        2,
        normalize=True,
        normalization_floor=floor,
        normalization_epsilon=None,
        routed_scaling_factor=1.0,
    )
    weights, _ = gate(op, hidden, correction)
    weights.name = "weights"
    graph.outputs.append(weights)
    model = ir.Model(graph, ir_version=10)
    model.graph.initializers["weight"].const_value = ir.tensor(
        np.array([[-12.0], [-12.0]], dtype=np.float32)
    )

    session = OnnxModelSession(model, device="cpu")
    actual = session.run(
        {
            "hidden": np.ones((1, 1, 1), dtype=np.float32),
            "correction": np.zeros(2, dtype=np.float32),
        }
    )["weights"]
    session.close()

    probability = 1.0 / (1.0 + np.exp(12.0))
    expected = np.full((1, 1, 2), probability / floor, dtype=np.float32)
    # Stock ORT's tail sigmoid approximation is within 1% of the analytic value.
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-8)
    assert not np.allclose(actual, np.array([[[0.5, 0.5]]], dtype=np.float32))
