# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for exact GGUF grouped-MoE module replacement."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import torch
from onnxscript import nn

from mobius._configs import Lfm2MoeConfig
from mobius._testing import make_config
from mobius._testing.ort_inference import OnnxModelSession
from mobius.components import (
    GroupedGatedSharedMoELayer,
    GroupedMoELayer,
    GroupedUngatedSharedMoELayer,
    MoELayer,
)
from mobius.functions._lowering_test_utils import expand_grouped_matmul
from mobius.integrations._weight_loading import apply_weights
from mobius.integrations.gguf._builder import _replace_grouped_moe_layers
from mobius.models.lfm2 import GroupedLfm2MoEFeedForward, Lfm2MoEFeedForward
from mobius.models.moe import Qwen2MoELayer, UngatedSharedMoELayer
from mobius.tasks._base import _make_graph, _make_model


class _Root(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


def _config(**kwargs):
    return make_config(
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=48,
        **kwargs,
    )


def test_replaces_exact_supported_compositions() -> None:
    cases = [
        (MoELayer(_config(mlp_bias=True)), GroupedMoELayer),
        (Qwen2MoELayer(_config()), GroupedGatedSharedMoELayer),
        (UngatedSharedMoELayer(_config()), GroupedUngatedSharedMoELayer),
    ]
    for source, expected_type in cases:
        root = _Root(source)
        assert _replace_grouped_moe_layers(root) == ["mlp"]
        assert type(root.mlp) is expected_type


def test_does_not_blanket_replace_specialized_subclasses() -> None:
    class SpecializedMoE(MoELayer):
        pass

    root = _Root(SpecializedMoE(_config()))
    assert _replace_grouped_moe_layers(root) == []
    assert type(root.mlp) is SpecializedMoE


@pytest.mark.parametrize(
    ("biased", "normalize"), [(False, False), (True, False), (True, True)]
)
def test_grouped_lfm2_preserves_selection_only_bias(biased: bool, normalize: bool) -> None:
    config = Lfm2MoeConfig(
        hidden_size=4,
        intermediate_size=6,
        moe_intermediate_size=6,
        num_local_experts=4,
        num_experts_per_tok=2,
        hidden_act="silu",
        use_expert_bias=biased,
        norm_topk_prob=normalize,
        routed_scaling_factor=1.7,
    )
    root = _Root(Lfm2MoEFeedForward(config))
    assert _replace_grouped_moe_layers(root) == ["mlp"]
    assert type(root.mlp) is GroupedLfm2MoEFeedForward
    graph, builder = _make_graph(name="lfm2_grouped")
    hidden = builder.input("hidden", dtype=ir.DataType.FLOAT, shape=[2, 3, 4])
    builder.add_output(root.mlp(builder.op, hidden), "output")
    model = _make_model(graph)
    model.graph.opset_imports[""] = 29
    assert sum(node.op_type == "GroupedMatMul" for node in graph) == 3
    assert not any(node.op_type == "Scan" for node in graph)
    rng = np.random.default_rng(42)
    weights = {
        name: rng.normal(0, 0.2, tuple(value.shape)).astype(np.float32)
        for name, value in graph.initializers.items()
        if not name.startswith("const")
    }
    if biased:
        weights["mlp.expert_bias"] = np.array([0, 5, -5, 2], dtype=np.float32)
    apply_weights(model, {name: torch.from_numpy(value) for name, value in weights.items()})
    values = rng.normal(0, 0.5, (2, 3, 4)).astype(np.float32)
    actual = OnnxModelSession(expand_grouped_matmul(model), device="cpu").run(
        {"hidden": values}
    )["output"]
    x = torch.from_numpy(values)
    probs = torch.sigmoid(x @ torch.from_numpy(weights["mlp.gate.weight"]).T)
    selection = probs + torch.from_numpy(weights["mlp.expert_bias"]) if biased else probs
    ids = selection.topk(2, dim=-1).indices
    if biased:
        assert not torch.equal(ids, probs.topk(2, dim=-1).indices)
    route_weights = probs.gather(-1, ids)
    if normalize:
        route_weights /= route_weights.sum(dim=-1, keepdim=True) + 1e-6
    route_weights *= 1.7
    expected = torch.zeros_like(x)
    for expert in range(4):
        prefix = f"mlp.experts.{expert}."
        gate = x @ torch.from_numpy(weights[prefix + "gate_proj.weight"]).T
        up = x @ torch.from_numpy(weights[prefix + "up_proj.weight"]).T
        output = (torch.nn.functional.silu(gate) * up) @ torch.from_numpy(
            weights[prefix + "down_proj.weight"]
        ).T
        scale = (route_weights * (ids == expert)).sum(dim=-1, keepdim=True)
        expected += output * scale
    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-5, atol=2e-6)
