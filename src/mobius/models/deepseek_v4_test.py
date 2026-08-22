# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Correctness/structure fixture for DeepSeek-V4's fused QMoE export.

These tests prove the QMoE export (mobius.components._moe.MoELayer +
DeepSeekV4Gate.qmoe_routing) is numerically and structurally equivalent to
the portable dense loop-over-experts representation for DeepSeek-V4's
hash-routed, clipped-SwiGLU MoE, without altering the gate's selection math
or its output.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnxscript import GraphBuilder

from mobius._configs import QuantizationConfig
from mobius._constants import OPSET_VERSION
from mobius._testing import make_config
from mobius.components import MoELayer
from mobius.models.deepseek_v4 import DeepSeekV4Gate, _DeepSeekV4Expert


def _pack_int4_sym(codes: np.ndarray) -> np.ndarray:
    """Pack symmetric int4 codes (0..15) into QMoE's 2-codes-per-byte layout.

    Matches the convention validated against the CPU QMoE kernel: low nibble
    holds the even-indexed code, high nibble the odd-indexed code, along the
    last (contraction) axis.
    """
    low = codes[..., 0::2].astype(np.uint8)
    high = codes[..., 1::2].astype(np.uint8)
    return (low | (high << 4)).astype(np.uint8)


def _dequant_int4_sym(packed: np.ndarray, scales: np.ndarray, block_size: int) -> np.ndarray:
    """Inverse of ``_pack_int4_sym`` plus per-block symmetric scaling."""
    low = packed & 0x0F
    high = packed >> 4
    codes = np.stack([low, high], axis=-1).reshape(*packed.shape[:-1], -1).astype(np.float32)
    blocks = codes.shape[-1] // block_size
    values = codes.reshape(*codes.shape[:-1], blocks, block_size) - 8.0
    return (values * scales[..., None]).reshape(codes.shape)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _clipped_swiglu(gate: np.ndarray, up: np.ndarray, limit: float) -> np.ndarray:
    """Reference for _DeepSeekV4Expert's clipped SwiGLU (alpha=1, beta=0)."""
    if limit > 0:
        gate = np.minimum(gate, limit)
        up = np.clip(up, -limit, limit)
    return _silu(gate) * up


def _reference_routed_moe(
    hidden: np.ndarray,
    gate_weight: np.ndarray,
    tid2eid: np.ndarray,
    input_ids: np.ndarray,
    scoring_func: str,
    route_scale: float,
    fc1: np.ndarray,
    fc2: np.ndarray,
    intermediate_size: int,
    swiglu_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-numpy oracle for DeepSeekV4Gate hash routing + routed experts.

    Excludes the shared expert -- see module docstring for why that's
    excluded.

    Returns (output, selected_experts, routing_weights) so callers can also
    assert routing identity directly.
    """
    logits = hidden @ gate_weight.T
    if scoring_func == "sigmoid":
        scores = 1.0 / (1.0 + np.exp(-logits))
    elif scoring_func == "softmax":
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        scores = exp / exp.sum(axis=-1, keepdims=True)
    else:  # sqrtsoftplus
        scores = np.sqrt(np.log1p(np.exp(logits)))

    selected_experts = tid2eid[input_ids]  # [tokens, top_k]
    routing_weights = np.take_along_axis(scores, selected_experts, axis=-1)
    if scoring_func != "softmax":
        weight_sum = routing_weights.sum(axis=-1, keepdims=True)
        routing_weights = routing_weights / (weight_sum + 1e-20)
    if route_scale != 1.0:  # noqa: RUF069
        routing_weights = routing_weights * route_scale

    num_tokens, top_k = selected_experts.shape
    result = np.zeros_like(hidden)
    for token in range(num_tokens):
        for k in range(top_k):
            expert = int(selected_experts[token, k])
            weight = routing_weights[token, k]
            projected = hidden[token] @ fc1[expert].T
            gate_part = projected[:intermediate_size]
            up_part = projected[intermediate_size:]
            activated = _clipped_swiglu(gate_part, up_part, swiglu_limit)
            expert_out = activated @ fc2[expert].T
            result[token] += weight * expert_out
    return result, selected_experts, routing_weights


def _build_hash_routed_qmoe_layer(
    rng,
    *,
    num_experts,
    top_k,
    hidden_size,
    intermediate_size,
    block_size,
    vocab_size,
    swiglu_limit,
    scoring_func,
    route_scale,
):
    """Construct a quantized, hash-routed MoELayer plus its numpy ground truth.

    Returns the layer and everything it should be numerically identical to.
    """
    config = make_config(
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        num_local_experts=num_experts,
        num_experts_per_tok=top_k,
        n_shared_experts=1,
        vocab_size=vocab_size,
        num_hash_layers=1,  # layer_id=0 below is hash-routed.
        scoring_func=scoring_func,
        routed_scaling_factor=route_scale,
        swiglu_limit=swiglu_limit,
        quantization=QuantizationConfig(
            bits=4, group_size=block_size, quant_method="gptq", sym=True
        ),
    )
    gate = DeepSeekV4Gate(config, layer_id=0)
    layer = MoELayer(
        config,
        gate=gate,
        expert_factory=lambda cfg, _lc: _DeepSeekV4Expert(cfg, cfg.intermediate_size),
        activation_alpha=1.0,
        activation_beta=0.0,
        swiglu_limit=swiglu_limit,
    )
    assert layer.experts is None, "quantized hash-routed gate must take the fused QMoE path"

    fc1_out = 2 * intermediate_size
    gate_weight = rng.standard_normal((num_experts, hidden_size)).astype(np.float32) * 0.3
    # Real hash tables route each token to `top_k` *distinct* experts (that's
    # the whole point of top-k MoE); sample without replacement per row so the
    # fixture doesn't exercise the ill-defined "same expert twice" case no
    # gate in this file (hash or learned top-k) ever actually produces.
    tid2eid = np.stack(
        [rng.choice(num_experts, size=top_k, replace=False) for _ in range(vocab_size)]
    ).astype(np.int32)

    fc1_codes = rng.integers(0, 16, size=(num_experts, fc1_out, hidden_size)).astype(np.int64)
    fc2_codes = rng.integers(0, 16, size=(num_experts, hidden_size, intermediate_size)).astype(
        np.int64
    )
    fc1_scales = rng.random((num_experts, fc1_out, hidden_size // block_size)) * 0.1 + 0.01
    fc1_scales = fc1_scales.astype(np.float32)
    fc2_scales = (
        rng.random((num_experts, hidden_size, intermediate_size // block_size)) * 0.1 + 0.01
    )
    fc2_scales = fc2_scales.astype(np.float32)

    layer.gate.weight.const_value = ir.tensor(gate_weight)
    layer.gate.bias.const_value = ir.tensor(
        np.zeros(num_experts, dtype=np.float32)
    )  # unused by hash routing, but must have a value to build the graph.
    layer.gate.tid2eid.const_value = ir.tensor(tid2eid)
    layer.fc1_experts_weights.const_value = ir.tensor(_pack_int4_sym(fc1_codes))
    layer.fc1_scales.const_value = ir.tensor(fc1_scales)
    layer.fc2_experts_weights.const_value = ir.tensor(_pack_int4_sym(fc2_codes))
    layer.fc2_scales.const_value = ir.tensor(fc2_scales)

    fc1_dense = _dequant_int4_sym(_pack_int4_sym(fc1_codes), fc1_scales, block_size)
    fc2_dense = _dequant_int4_sym(_pack_int4_sym(fc2_codes), fc2_scales, block_size)

    return layer, dict(
        gate_weight=gate_weight,
        tid2eid=tid2eid,
        fc1_dense=fc1_dense,
        fc2_dense=fc2_dense,
        config=config,
    )


def _run_layer(layer, hidden_values: np.ndarray, input_ids_values: np.ndarray) -> np.ndarray:
    graph = ir.Graph(
        inputs=[],
        outputs=[],
        nodes=[],
        name="deepseek_v4_qmoe_fixture",
        opset_imports={"": OPSET_VERSION, "com.microsoft": 1},
    )
    builder = GraphBuilder(graph)
    hidden = ir.Value(
        name="hidden_states",
        shape=ir.Shape(list(hidden_values.shape)),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    input_ids = ir.Value(
        name="input_ids",
        shape=ir.Shape(list(input_ids_values.shape)),
        type=ir.TensorType(ir.DataType.INT64),
    )
    graph.inputs.extend([hidden, input_ids])
    output = layer(builder.op, hidden, input_ids)
    output.name = "output"
    graph.outputs.append(output)

    model = ir.Model(graph, ir_version=11)
    proto = ir.to_proto(model)
    session = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(
        None, {"hidden_states": hidden_values, "input_ids": input_ids_values}
    )
    return actual, graph


class TestDeepSeekV4QMoEExport:
    """Fused-QMoE export must match the portable dense-loop reference."""

    @pytest.mark.parametrize(
        ("hidden_size", "intermediate_size", "block_size"),
        [
            (16, 16, 16),  # single quantization block per row (block_size == hidden_size)
            (32, 32, 16),  # multiple quantization blocks per row (2 blocks)
        ],
        ids=["single_block", "multi_block"],
    )
    def test_hash_routed_clipped_swiglu_qmoe_matches_dense_reference(
        self, hidden_size, intermediate_size, block_size
    ):
        rng = np.random.default_rng(1234)
        num_experts, top_k = 4, 2
        vocab_size = 6
        swiglu_limit = 2.0  # Small on purpose: forces clipping to actually engage.
        scoring_func = "sigmoid"
        route_scale = 1.7

        layer, ref = _build_hash_routed_qmoe_layer(
            rng,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            block_size=block_size,
            vocab_size=vocab_size,
            swiglu_limit=swiglu_limit,
            scoring_func=scoring_func,
            route_scale=route_scale,
        )

        num_tokens = vocab_size
        hidden_values = (rng.standard_normal((num_tokens, hidden_size)) * 3.0).astype(
            np.float32
        )
        input_ids_values = np.arange(num_tokens, dtype=np.int64)

        actual, graph = _run_layer(layer, hidden_values, input_ids_values)

        expected, selected_experts, routing_weights = _reference_routed_moe(
            hidden_values,
            ref["gate_weight"],
            ref["tid2eid"],
            input_ids_values,
            scoring_func,
            route_scale,
            ref["fc1_dense"],
            ref["fc2_dense"],
            intermediate_size,
            swiglu_limit,
        )
        # Sanity: hash routing and clipping are both actually exercised.
        assert len(set(selected_experts.flatten().tolist())) > 1
        assert np.any(routing_weights > 0)

        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=5e-3)

        # Structural evidence: routed experts collapse to exactly one QMoE
        # node, with no per-expert MatMulNBits/mask-loop left over.
        assert sum(node.op_type == "QMoE" for node in graph) == 1
        assert sum(node.op_type == "MatMulNBits" for node in graph) == 0
        assert sum(node.op_type == "Equal" for node in graph) == 0

    def test_qmoe_routing_preserves_hash_selected_experts_and_weights(self):
        """qmoe_routing()'s adapter must reproduce forward()'s exact routing.

        Not merely a plausible selection/weights, but the exact hash-routed
        ones.
        """
        rng = np.random.default_rng(99)
        num_experts, top_k = 5, 2
        hidden_size = 8
        vocab_size = 7

        config = make_config(
            hidden_size=hidden_size,
            moe_intermediate_size=8,
            num_local_experts=num_experts,
            num_experts_per_tok=top_k,
            n_shared_experts=1,
            vocab_size=vocab_size,
            num_hash_layers=1,
            scoring_func="sigmoid",
            routed_scaling_factor=1.0,
        )
        gate = DeepSeekV4Gate(config, layer_id=0)
        gate.weight.const_value = ir.tensor(
            (rng.standard_normal((num_experts, hidden_size)) * 0.3).astype(np.float32)
        )
        gate.bias.const_value = ir.tensor(np.zeros(num_experts, dtype=np.float32))
        tid2eid = np.stack(
            [rng.choice(num_experts, size=top_k, replace=False) for _ in range(vocab_size)]
        ).astype(np.int32)
        gate.tid2eid.const_value = ir.tensor(tid2eid)

        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            name="qmoe_routing_identity",
            opset_imports={"": OPSET_VERSION},
        )
        builder = GraphBuilder(graph)
        hidden = ir.Value(
            name="hidden_states",
            shape=ir.Shape([vocab_size, hidden_size]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([vocab_size]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        graph.inputs.extend([hidden, input_ids])
        # Mirror MoELayer._qmoe_forward's manual module-scope push: qmoe_routing
        # is normally invoked that way (not via gate.__call__), which is what
        # realizes/qualifies the gate's own parameters.
        builder.push_module(gate.name or "gate", type(gate).__qualname__)
        try:
            for param in gate.parameters(recurse=False):
                param._realize(builder)
            router_probs, router_weights, normalize, output_scale = gate.qmoe_routing(
                builder.op, hidden, input_ids
            )
        finally:
            builder.pop_module()
        assert normalize is False
        assert output_scale == 1.0  # noqa: RUF069
        router_probs.name = "router_probs"
        router_weights.name = "router_weights"
        graph.outputs.extend([router_probs, router_weights])

        model = ir.Model(graph, ir_version=11)
        proto = ir.to_proto(model)
        session = ort.InferenceSession(
            proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        hidden_values = (rng.standard_normal((vocab_size, hidden_size)) * 2.0).astype(
            np.float32
        )
        input_ids_values = np.arange(vocab_size, dtype=np.int64)
        probs, weights = session.run(
            None, {"hidden_states": hidden_values, "input_ids": input_ids_values}
        )

        expected_selected = tid2eid[input_ids_values]
        _, _, forward_weights = _forward_reference(
            hidden_values, gate.weight.const_value.numpy(), tid2eid, input_ids_values
        )
        for token in range(vocab_size):
            top = np.argsort(-probs[token])[:top_k]
            assert set(top.tolist()) == set(expected_selected[token].tolist())
            for k, expert in enumerate(expected_selected[token]):
                np.testing.assert_allclose(
                    weights[token, expert], forward_weights[token, k], rtol=1e-5, atol=1e-6
                )


def _forward_reference(hidden, gate_weight, tid2eid, input_ids):
    """Minimal re-derivation of DeepSeekV4Gate.forward's hash routing.

    Uses sigmoid scoring, independent of qmoe_routing, for a cross-check.
    """
    logits = hidden @ gate_weight.T
    scores = 1.0 / (1.0 + np.exp(-logits))
    selected_experts = tid2eid[input_ids]
    routing_weights = np.take_along_axis(scores, selected_experts, axis=-1)
    weight_sum = routing_weights.sum(axis=-1, keepdims=True)
    routing_weights = routing_weights / (weight_sum + 1e-20)
    return scores, selected_experts, routing_weights
