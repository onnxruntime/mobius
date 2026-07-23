# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
from onnxscript import GraphBuilder

from mobius._constants import OPSET_VERSION
from mobius._testing import make_config
from mobius.models.deepseek import DeepSeekMoEGate, _DeepSeekMoEFFN


def _silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-value))


def _mlp(hidden_states: np.ndarray, weights: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    gate = hidden_states @ weights[f"{prefix}gate_proj.weight"].T
    up = hidden_states @ weights[f"{prefix}up_proj.weight"].T
    return (_silu(gate) * up) @ weights[f"{prefix}down_proj.weight"].T


def test_deepseek_dense_moe_fallback_matches_numpy_reference():
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
        scoring_func="sigmoid",
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

    # Router logits are zero, so all original sigmoid aggregation scores are 0.5.
    # Bias-corrected grouped TopK selects experts 1 and 0 from group 0.
    selected_experts = (1, 0)
    routing_weight = (0.5 / (0.5 + 0.5)) * config.routed_scaling_factor
    expected = sum(
        _mlp(hidden_values, weights, f"moe.experts.{expert}.") * routing_weight
        for expert in selected_experts
    )
    expected += _mlp(hidden_values, weights, "shared_experts.")

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert sum(node.op_type == "MatMul" for node in graph) == 16
    assert all(node.domain != "com.microsoft" for node in graph)
