# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from onnxscript import GraphBuilder

from mobius._constants import OPSET_VERSION
from mobius._testing import create_test_builder, create_test_input, make_config
from mobius.models.deepseek import DeepSeekMoEGate, DeepSeekV3CausalLMModel, _DeepSeekMoEFFN


def _silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-value))


def _mlp(hidden_states: np.ndarray, weights: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    gate = hidden_states @ weights[f"{prefix}gate_proj.weight"].T
    up = hidden_states @ weights[f"{prefix}up_proj.weight"].T
    return (_silu(gate) * up) @ weights[f"{prefix}down_proj.weight"].T


@pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
def test_deepseek_dense_moe_fallback_matches_numpy_reference(scoring_func: str):
    """Tiny grouped, bias-corrected MoE export is a portable correctness oracle."""
    rng = np.random.default_rng(7)
    config = make_config(
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=3,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        n_shared_experts=1,
        scoring_func=scoring_func,
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=1.25,
        mlp_bias=False,
    )
    module = _DeepSeekMoEFFN(config, DeepSeekMoEGate(config))

    hidden = ir.Value(
        name="hidden_states",
        shape=ir.Shape([1, 3, config.hidden_size]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    graph = ir.Graph(
        inputs=[hidden],
        outputs=[],
        nodes=[],
        name="tiny_deepseek_dense_moe",
        opset_imports={"": OPSET_VERSION},
    )
    builder = GraphBuilder(graph)
    output = module(builder.op, hidden)
    output.name = "output"
    graph.outputs.append(output)

    weights: dict[str, np.ndarray] = {}
    for name, parameter in module.named_parameters():
        values = rng.standard_normal(tuple(parameter.shape)).astype(np.float32) * 0.2
        if name == "moe.gate.weight":
            values = np.zeros(tuple(parameter.shape), dtype=np.float32)
        elif name == "moe.gate.e_score_correction_bias":
            # Group 0 wins group selection, but its corrected expert scores are
            # negative. Excluded groups must therefore be masked with -inf, not 0.
            values = np.array([-0.7, -0.6, -1.5, -1.4], dtype=np.float32)
        parameter.const_value = ir.tensor(values)
        weights[name] = values

    model = ir.Model(graph, ir_version=11)
    proto = ir.to_proto(model)
    session = ort.InferenceSession(
        proto.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    hidden_values = rng.standard_normal((1, 3, config.hidden_size)).astype(np.float32)
    actual = session.run(None, {"hidden_states": hidden_values})[0]

    # Uniform unbiased scores are normalized after bias-corrected group selection,
    # so both score functions select experts 1 and 0 with equal aggregation weight.
    selected_experts = (1, 0)
    routing_weight = (0.5 / (0.5 + 0.5)) * config.routed_scaling_factor
    expected = sum(
        _mlp(hidden_values, weights, f"moe.experts.{expert}.") * routing_weight
        for expert in selected_experts
    )
    expected += _mlp(hidden_values, weights, "shared_experts.")

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert any(node.op_type == "MatMul" for node in graph)
    assert all(node.domain != "com.microsoft" for node in graph)


def test_sigmoid_gate_without_optional_correction_bias_has_no_bias_parameter():
    config = make_config(
        hidden_size=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        scoring_func="sigmoid",
        use_expert_bias=False,
    )
    gate = DeepSeekMoEGate(config)

    assert [name for name, _ in gate.named_parameters()] == ["weight"]


def test_dots1_routing_normalization_uses_pinned_llama_floor():
    config = make_config(
        hidden_size=1,
        num_local_experts=2,
        num_experts_per_tok=2,
        scoring_func="sigmoid",
        use_expert_bias=False,
        norm_topk_prob=True,
        routing_weight_normalization_floor=6.103515625e-5,
    )
    gate = DeepSeekMoEGate(config)
    gate.weight.const_value = ir.tensor(np.ones((2, 1), dtype=np.float32))

    hidden = ir.Value(
        name="hidden_states",
        shape=ir.Shape([1, 1, 1]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    graph = ir.Graph(
        inputs=[hidden],
        outputs=[],
        nodes=[],
        name="dots1_routing_floor",
        opset_imports={"": OPSET_VERSION},
    )
    builder = GraphBuilder(graph)
    routing_weights, _ = gate(builder.op, hidden)
    routing_weights.name = "routing_weights"
    graph.outputs.append(routing_weights)

    session = ort.InferenceSession(
        ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(None, {"hidden_states": np.array([[[-11.0]]], dtype=np.float32)})[0]
    score = 1.0 / (1.0 + np.exp(11.0))
    expected = np.full((1, 1, 2), score / 6.103515625e-5, dtype=np.float32)

    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-8)
    assert actual.sum() < 1.0


@pytest.mark.parametrize(
    ("scoring_func", "expects_bias"),
    [("softmax", False), ("sigmoid", True)],
)
def test_unspecified_correction_bias_preserves_hf_routing_defaults(
    scoring_func: str, expects_bias: bool
):
    config = make_config(
        hidden_size=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        scoring_func=scoring_func,
        use_expert_bias=None,
    )
    names = {name for name, _ in DeepSeekMoEGate(config).named_parameters()}
    assert ("e_score_correction_bias" in names) is expects_bias


def test_gguf_expanded_expert_values_are_renamed_without_repacking():
    config = make_config(
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=3,
        num_local_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=1,
    )
    model = DeepSeekV3CausalLMModel(config)
    gate = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    down = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    processed = model.preprocess_weights(
        {
            "model.layers.0.mlp.experts.1.gate_proj.weight": gate,
            "model.layers.0.mlp.experts.1.down_proj.weight": down,
        }
    )

    torch.testing.assert_close(
        processed["model.layers.0.mlp.moe.experts.1.gate_proj.weight"], gate
    )
    torch.testing.assert_close(
        processed["model.layers.0.mlp.moe.experts.1.down_proj.weight"], down
    )


def test_deepseek_moe_ffn_linear_class_reaches_routed_experts():
    """``linear_class`` must reach the routed dense-loop experts, not just the shared expert.

    Regression test for a bug where ``_DeepSeekMoEFFN`` constructed its
    ``MoELayer`` without forwarding ``linear_class``, so a quantized config
    quantized attention/dense-FFN/shared-expert linears but silently left the
    routed MoE experts as plain float ``MatMul`` -- losing quantization and
    breaking the ``fuse_dense_moe_to_qmoe`` post-hoc rewrite, which only
    matches a quantized ``MatMulNBits`` dense-fallback pattern.
    """
    from mobius.components._common import Linear

    created: list[Linear] = []

    class TrackingLinear(Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    config = make_config(
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=3,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        n_shared_experts=1,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
    )
    module = _DeepSeekMoEFFN(config, DeepSeekMoEGate(config), linear_class=TrackingLinear)

    # The dense loop-over-experts fallback must have been built with
    # TrackingLinear for every routed expert's projections. Check
    # `module.moe.experts` directly (not the global `created` list) so this
    # assertion can't pass merely because the shared expert below picked up
    # the class -- that path already worked before this fix.
    assert module.moe.experts is not None
    routed_linears = [m for m in module.moe.experts.modules() if isinstance(m, Linear)]
    assert len(routed_linears) > 0
    assert all(isinstance(m, TrackingLinear) for m in routed_linears)
    # The shared expert must still be quantized too (it already worked
    # before this fix; guard against a future regression there as well).
    shared_linears = [m for m in module.shared_experts.modules() if isinstance(m, Linear)]
    assert len(shared_linears) > 0
    assert all(isinstance(m, TrackingLinear) for m in shared_linears)
    # Sanity: the tracking list saw both groups (routed + shared).
    assert len(created) == len(routed_linears) + len(shared_linears)


def test_noaux_tc_supports_single_expert_groups():
    config = make_config(
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=3,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_group=4,
        topk_group=2,
        n_shared_experts=1,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
    )
    module = _DeepSeekMoEFFN(config, DeepSeekMoEGate(config))
    hidden = ir.Value(
        name="hidden_states",
        shape=ir.Shape([1, 2, config.hidden_size]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    graph = ir.Graph(
        inputs=[hidden],
        outputs=[],
        nodes=[],
        name="single_expert_groups",
        opset_imports={"": OPSET_VERSION},
    )
    output = module(GraphBuilder(graph).op, hidden)

    assert output is not None
    assert any(node.op_type == "TopK" for node in graph)


def test_qmoe_routing_uses_raw_logits_and_casts_scores_to_hidden_dtype():
    """qmoe_routing()'s router_probs/router_weights must match hidden_states' dtype.

    QMoE's contrib-op schema binds router_probs/router_weights to the same
    type constraint ("T") as hidden_states. _routing_scores() computes in
    FLOAT32 for numerical stability (matching HF's fp32 routing), so
    qmoe_routing() must cast the result back to hidden_states' dtype --
    otherwise a fp16/bf16 model export would hit a QMoE type-mismatch error.
    """
    config = make_config(
        hidden_size=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        scoring_func="softmax",
        topk_method="greedy",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )
    gate = DeepSeekMoEGate(config)
    builder, op, graph = create_test_builder()
    hidden = create_test_input(
        builder, "hidden", [1, 2, config.hidden_size], ir.DataType.FLOAT16
    )

    router_probs, router_weights, _normalize, _scale = gate.qmoe_routing(op, hidden)

    assert router_probs.dtype == ir.DataType.FLOAT16
    assert router_weights.dtype == ir.DataType.FLOAT16
    assert router_probs.producer().op_type == "CastLike"
    assert router_probs.producer().inputs[0].producer().op_type == "MatMul"
    assert any(node.op_type == "Softmax" for node in graph)


def test_qmoe_routing_preserves_existing_cpu_grouped_encoding():
    config = make_config(
        hidden_size=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        use_expert_bias=True,
    )
    gate = DeepSeekMoEGate(config)
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 2, 4], ir.DataType.FLOAT)

    router_probs, router_weights, _normalize, _scale = gate.qmoe_routing(op, hidden)

    assert router_probs is not None
    assert router_weights is not None
    assert any(node.op_type == "Sigmoid" for node in graph)


def test_grouped_routing_rejects_non_divisible_expert_count():
    config = make_config(
        hidden_size=4,
        num_local_experts=5,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
    )

    with pytest.raises(ValueError, match="evenly divisible"):
        DeepSeekMoEGate(config)


@pytest.mark.parametrize(("topk_group", "n_group"), [(0, 2), (3, 2)])
def test_grouped_routing_rejects_invalid_topk_group(topk_group, n_group):
    config = make_config(
        hidden_size=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_group=n_group,
        topk_group=topk_group,
    )

    with pytest.raises(ValueError, match="1 <= topk_group <= n_group"):
        DeepSeekMoEGate(config)
