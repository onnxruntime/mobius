# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Focused graph-batching regressions; requires the new Olive MoE pass."""

from __future__ import annotations

import onnx_ir as ir
import pytest

pytest.importorskip("olive.passes.onnx.moe_quantization")

from examples.olive.flint.quantize import _batch_routes


def _model(mismatch=None):
    x = ir.Value(name="hidden")
    gate, down, bias = (ir.Value(name=name) for name in ("gate", "down", "bias"))
    routers = [ir.Value(name=f"router_{i}") for i in range(6)]
    nodes = []
    for index, router in enumerate(routers):
        node = ir.node(
            "MoE",
            domain="com.microsoft",
            inputs=[
                x,
                router,
                gate,
                bias if mismatch == "bias" and index == 5 else None,
                down,
            ],
            attributes={
                "k": 1,
                "normalize_routing_weights": 1,
                "swiglu_fusion": 1,
                "activation_type": "swiglu",
                "use_sparse_mixer": int(mismatch == "attributes" and index == 5),
            },
            name=f"route_{index}",
        )
        node.outputs[0].name = f"output_{index}"
        nodes.append(node)
    return ir.Model(
        ir.Graph(
            [x, gate, down, bias, *routers],
            [node.outputs[0] for node in nodes],
            nodes=nodes,
            opset_imports={"": 24, "com.microsoft": 1},
        ),
        ir_version=10,
    )


def test_batch_routes_keeps_each_route_output():
    model = _model()
    assert _batch_routes(model) == 1
    assert [node.op_type for node in model.graph] == ["Concat", "Concat", "MoE", "Split"]
    assert [output.name for output in model.graph.outputs] == [
        f"output_{index}" for index in range(6)
    ]


@pytest.mark.parametrize("mismatch", ["bias", "attributes"])
def test_batch_routes_rejects_different_expert_parameters(mismatch):
    with pytest.raises(ValueError, match="share all expert inputs and attributes"):
        _batch_routes(_model(mismatch))
