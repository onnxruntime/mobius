# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma4's raw-logit SwiGLU boundary, without changing generic MoE routing."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._build_context import build_context
from mobius._builder import build_from_module
from mobius._execution_providers import ep_registry
from mobius._flags import override_flags
from mobius.models import Gemma4CausalLMModel
from mobius.models.gemma4_test import _build_gemma4_text, _tiny_gemma4_text_moe_config
from mobius.tasks import CausalLMTask

pytestmark = pytest.mark.opset29


@pytest.mark.parametrize(
    "enabled,dtype,activation",
    [
        (False, ir.DataType.FLOAT, "silu"),
        (True, ir.DataType.FLOAT, "silu"),
        (True, ir.DataType.FLOAT16, "swish"),
        (True, ir.DataType.BFLOAT16, "silu"),
        (True, ir.DataType.FLOAT, "gelu_pytorch_tanh"),
        (True, ir.DataType.FLOAT, "gelu"),
    ],
)
def test_fallback_ep_keeps_model_specific_dense_dispatch(enabled, dtype, activation):
    config = _tiny_gemma4_text_moe_config(dtype=dtype, hidden_act=activation)
    with (
        override_flags(experimental_opset29=enabled),
        build_context(ep_registry.require("onnx-standard"), dtype),
    ):
        model = _build_gemma4_text(config)
    assert not any(node.op_type in {"MoE", "QMoE", "GroupedMatMul"} for node in model.graph)
    assert model.graph.opset_imports[""] == (29 if enabled else 24)
    assert any(node.op_type == "Softmax" for node in model.graph)
    assert any(node.op_type == "Einsum" for node in model.graph)


def test_normal_build_keeps_gemma4_dense_at29():
    config = _tiny_gemma4_text_moe_config(dtype=ir.DataType.FLOAT, hidden_act="silu")
    with override_flags(experimental_opset29=True):
        package = build_from_module(
            Gemma4CausalLMModel(config), config, execution_provider="onnx-standard"
        )
    model = package["model"]
    assert model.graph.opset_imports[""] == 29
    assert not any(node.op_type == "GroupedMatMul" for node in model.graph)
    equations = [
        node.attributes.get_string("equation")
        for node in model.graph
        if node.op_type == "Einsum"
    ]
    assert equations.count("th,eih->tei") == 4  # Separate gate/up for each layer.
    assert equations.count("tei,ehi->teh") == 2  # One down projection per layer.
    assert not any(node.op_type in {"MoE", "QMoE"} for node in model.graph)
    assert all(not node.domain for node in model.graph.all_nodes())


@pytest.mark.parametrize("enabled", [False, True])
def test_fallback_dispatch_uses_snapshotted_build_opset(enabled):
    config = _tiny_gemma4_text_moe_config(dtype=ir.DataType.FLOAT, hidden_act="silu")
    with (
        override_flags(experimental_opset29=enabled),
        build_context(ep_registry.require("onnx-standard"), ir.DataType.FLOAT),
        override_flags(experimental_opset29=not enabled),
    ):
        model = _build_gemma4_text(config)
    assert model.graph.opset_imports[""] == (29 if enabled else 24)
    assert not any(node.op_type in {"MoE", "QMoE", "GroupedMatMul"} for node in model.graph)
    assert any(node.op_type == "Einsum" for node in model.graph)


def _expert_boundary_model(*, enabled, dtype, fc1, fc2, scale, update, top_k=2):
    """Extract the expert boundary from the actual decoder forward/weight pipeline."""
    config = _tiny_gemma4_text_moe_config(
        dtype=dtype,
        hidden_act="silu",
        num_local_experts=2,
        num_experts_per_tok=top_k,
        moe_intermediate_size=fc2.shape[-1],
        tie_word_embeddings=False,  # Permit expert-only partial checkpoint updates.
    )
    module = Gemma4CausalLMModel(config)
    layer = module.model.layers[0]
    if update != "folded":
        layer.fc1_experts_weights.const_value = ir.tensor(fc1)
        layer.fc2_experts_weights.const_value = ir.tensor(fc2)
        layer.router.per_expert_scale.const_value = ir.tensor(
            np.ones_like(scale) if update == "scale-only" else scale
        )
    with (
        override_flags(experimental_opset29=enabled),
        build_context(ep_registry.require("onnx-standard"), dtype),
    ):
        package = CausalLMTask().build(module, config)
        prefix = "model.layers.0."
        if update == "scale-only":
            weights = module.preprocess_weights(
                {
                    prefix + "router.per_expert_scale": torch.from_numpy(scale),
                }
            )
            assert package.apply_weights(weights, fold_constants=False) == set(weights)
        elif update == "folded":
            weights = module.preprocess_weights(
                {
                    prefix + "experts.gate_up_proj": torch.from_numpy(fc1),
                    prefix + "experts.down_proj": torch.from_numpy(fc2),
                    prefix + "router.per_expert_scale": torch.from_numpy(scale),
                }
            )
            torch.testing.assert_close(
                weights[prefix + "router.per_expert_scale"], torch.ones(2)
            )
            assert package.apply_weights(weights, fold_constants=False) == set(weights)
        model = package["model"]
        nodes = list(model.graph)
        assert not any(node.op_type in {"MoE", "QMoE", "GroupedMatMul"} for node in nodes)
        assert prefix + "router.per_expert_scale" in model.graph.initializers
        normed = next(node for node in nodes if node.op_type == "Einsum").inputs[0]
        logits = next(node for node in nodes if node.op_type == "Softmax").inputs[0]
        down = next(
            node
            for node in nodes
            if node.op_type == "Einsum"
            and node.attributes.get_string("equation") == "tei,ehi->teh"
        )
        output = next(
            node for node in nodes[nodes.index(down) + 1 :] if node.op_type == "ReduceSum"
        ).outputs[0]
        normed.name, logits.name, output.name = "normed", "logits", "experts"
        normed.dtype, logits.dtype = dtype, ir.DataType.FLOAT
        normed.shape, logits.shape = ir.Shape(["tokens", 64]), ir.Shape(["tokens", 2])
        output.shape, output.dtype = normed.shape.copy(), dtype
        graph = ir.convenience.extract(model.graph, [normed, logits], [output])
        return ir.Model(graph, ir_version=10)


def _run_experts(model, hidden, logits, tmp_path):
    from onnx.reference import ReferenceEvaluator

    # Execute dense 24/29 graphs at their original imports; ORT cannot load 29 yet.
    path = tmp_path / "experts.onnx"
    ir.save(model, path)
    return ReferenceEvaluator(str(path)).run(
        None, {"normed": hidden, "logits": logits.astype(np.float32)}
    )[0]


@pytest.mark.parametrize("update", ["preinitialized", "scale-only", "folded"])
def test_actual_float_dispatch_matches_dense_and_hf_with_remaining_scale(update, tmp_path):
    rng = np.random.default_rng(108)
    fc1 = rng.normal(0, 0.2, (2, 8, 64)).astype(np.float32)
    fc2 = rng.normal(0, 0.2, (2, 64, 4)).astype(np.float32)
    scale = np.array([0.3, 1.7], dtype=np.float32)
    hidden = rng.normal(0, 0.5, (3, 64)).astype(np.float32)
    logits = np.array([[0.2, -0.3], [-0.7, 0.8], [1.5, -0.8]], dtype=np.float32)
    actual = []
    for enabled in (False, True):
        model = _expert_boundary_model(
            enabled=enabled,
            dtype=ir.DataType.FLOAT,
            fc1=fc1,
            fc2=fc2,
            scale=scale,
            update=update,
        )
        actual.append(_run_experts(model, hidden, logits, tmp_path))
    # HF: chunked gate/up -> SiLU -> down -> normalized routing * learned scale.
    gate_up = torch.einsum("th,eih->tei", torch.from_numpy(hidden), torch.from_numpy(fc1))
    gate, up = gate_up.chunk(2, dim=-1)
    expert = torch.einsum(
        "tei,ehi->teh", torch.nn.functional.silu(gate) * up, torch.from_numpy(fc2)
    )
    weights = torch.softmax(torch.from_numpy(logits), dim=-1) * torch.from_numpy(scale)
    expected = (expert * weights.unsqueeze(-1)).sum(1).numpy()
    np.testing.assert_allclose(actual[0], expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual[1], expected, atol=1e-5, rtol=1e-5)


def test_actual_float_top1_dispatch_scales_only_selected_expert(tmp_path):
    fc1 = np.zeros((2, 2, 64), dtype=np.float32)
    fc1[:, 0, 0], fc1[:, 1, 0] = [1, 2], 1
    fc2 = np.zeros((2, 64, 1), dtype=np.float32)
    fc2[:, :, 0] = np.array([2, -3], dtype=np.float32)[:, None]
    scale = np.array([0.3, 1.7], dtype=np.float32)
    hidden = np.zeros((2, 64), dtype=np.float32)
    hidden[:, 0] = [1, 2]
    logits = np.array([[2, -1], [-1, 2]], dtype=np.float32)
    # Token 0 selects expert 0; token 1 selects expert 1. Top-1 normalization
    # gives unit routing weights BEFORE applying the selected learned scale.
    expected = (
        torch.nn.functional.silu(torch.tensor([1.0, 4.0]))
        * torch.tensor([1.0, 2.0])
        * torch.tensor([2.0, -3.0])
        * torch.from_numpy(scale)
    ).numpy()
    expected = np.repeat(expected[:, None], 64, axis=1)
    for enabled in (False, True):
        model = _expert_boundary_model(
            enabled=enabled,
            dtype=ir.DataType.FLOAT,
            fc1=fc1,
            fc2=fc2,
            scale=scale,
            update="preinitialized",
            top_k=1,
        )
        np.testing.assert_allclose(
            _run_experts(model, hidden, logits, tmp_path), expected, atol=1e-5, rtol=1e-5
        )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("update", ["preinitialized", "scale-only"])
def test_tied_logits_top1_selects_lower_expert_id(enabled, update, tmp_path):
    fc1 = np.zeros((2, 2, 64), dtype=np.float32)
    fc1[:, :, 0] = 1
    fc2 = np.zeros((2, 64, 1), dtype=np.float32)
    fc2[:, :, 0] = np.array([2, -3], dtype=np.float32)[:, None]
    scale = np.array([0.3, 1.7], dtype=np.float32)
    hidden = np.zeros((1, 64), dtype=np.float32)
    hidden[0, 0] = 1
    logits = np.zeros((1, 2), dtype=np.float32)
    model = _expert_boundary_model(
        enabled=enabled,
        dtype=ir.DataType.FLOAT,
        fc1=fc1,
        fc2=fc2,
        scale=scale,
        update=update,
        top_k=1,
    )
    # Equal logits select expert 0 in Gemma's dense TopK, unlike native CPU MoE.
    routing_weight = np.float32(1) * scale[0]
    contribution = torch.nn.functional.silu(torch.tensor(1.0)).numpy() * fc2[0, :, 0]
    expected = (routing_weight * contribution)[None, :]
    np.testing.assert_allclose(
        _run_experts(model, hidden, logits, tmp_path), expected, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("update", ["preinitialized", "scale-only"])
def test_remaining_scale_does_not_overflow_finite_expert_output(enabled, update, tmp_path):
    fc1 = np.zeros((2, 2, 64), dtype=np.float32)
    # SiLU(1) * up gives an intermediate of approximately 0.1.
    fc1[:, 0, 0] = 1
    fc1[:, 1, 0] = np.float32(0.1) / torch.nn.functional.silu(torch.tensor(1.0)).item()
    fc2 = np.full((2, 64, 1), np.float32(2e38), dtype=np.float32)
    scale = np.full(2, 2, dtype=np.float32)
    hidden = np.zeros((1, 64), dtype=np.float32)
    hidden[0, 0] = 1
    logits = np.array([[2, -1]], dtype=np.float32)
    model = _expert_boundary_model(
        enabled=enabled,
        dtype=ir.DataType.FLOAT,
        fc1=fc1,
        fc2=fc2,
        scale=scale,
        update=update,
        top_k=1,
    )
    # Preserve the dense arithmetic: normalized routing * live scale, then
    # expert contribution, then reduction. Pre-scaling W2 would overflow.
    gate_up = torch.einsum("th,eih->tei", torch.from_numpy(hidden), torch.from_numpy(fc1))
    gate, up = gate_up.chunk(2, dim=-1)
    expert = torch.einsum(
        "tei,ehi->teh", torch.nn.functional.silu(gate) * up, torch.from_numpy(fc2)
    )
    routing_norm = torch.tensor([[1.0, 0.0]])
    weights = routing_norm * torch.from_numpy(scale)
    expected = (expert * weights.unsqueeze(-1)).sum(1).numpy()
    assert np.isfinite(expected).all()
    actual = _run_experts(model, hidden, logits, tmp_path)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_half_adversarial_cancellation_keeps_dense_rounding_with_flag_on(tmp_path):
    from onnx.reference import ReferenceEvaluator

    fc1 = np.zeros((2, 2, 64), dtype=np.float16)
    fc1[:, 0, 0], fc1[:, 1, 0] = 10, 1
    fc2 = np.zeros((2, 64, 1), dtype=np.float16)
    fc2[:, :, 0] = np.array([6000, -6000], dtype=np.float16)[:, None]
    hidden = np.zeros((1, 64), dtype=np.float16)
    hidden[0, 0] = 1
    logits = np.array([[0.0002, -0.0002]], dtype=np.float16)
    results = []
    for enabled in (False, True):
        model = _expert_boundary_model(
            enabled=enabled,
            dtype=ir.DataType.FLOAT16,
            fc1=fc1,
            fc2=fc2,
            scale=np.ones(2, dtype=np.float16),
            update="preinitialized",
        )
        # Execute the actual dense graph without relabeling its 24/29 imports.
        # ORT cannot load 29 yet; the ONNX reference evaluator supports these ops.
        path = tmp_path / f"dense-{enabled}.onnx"
        ir.save(model, path)
        results.append(
            ReferenceEvaluator(str(path)).run(
                None, {"normed": hidden, "logits": logits.astype(np.float32)}
            )[0]
        )
    weights = torch.softmax(torch.from_numpy(logits).float(), dim=-1)
    expert = torch.tensor([60000, -60000], dtype=torch.float16)
    expected = (expert.float() * weights).half().sum(-1)
    unrounded = (expert.float() * weights).sum(-1).half()
    assert expected.item() == 0 and unrounded.item() == 12
    np.testing.assert_array_equal(results[0], np.zeros((1, 64), dtype=np.float16))
    np.testing.assert_array_equal(results[1], results[0])
