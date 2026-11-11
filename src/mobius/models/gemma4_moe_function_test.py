# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma4's raw-logit SwiGLU boundary, without changing generic MoE routing."""

from __future__ import annotations

import onnx
import onnx_ir as ir
import pytest

from mobius._build_context import build_context
from mobius._builder import build_from_module
from mobius._execution_providers import ep_registry
from mobius._flags import override_flags
from mobius.models import Gemma4CausalLMModel
from mobius.models.gemma4_test import _build_gemma4_text, _tiny_gemma4_text_moe_config


@pytest.mark.parametrize(
    "enabled,dtype,activation,native",
    [
        (False, ir.DataType.FLOAT, "silu", False),
        (True, ir.DataType.FLOAT, "silu", True),
        (True, ir.DataType.FLOAT16, "swish", True),
        (True, ir.DataType.BFLOAT16, "silu", False),
        (True, ir.DataType.FLOAT, "gelu_pytorch_tanh", False),
        (True, ir.DataType.FLOAT, "gelu", False),
    ],
)
def test_fallback_ep_reuses_native_call_only_for_supported_raw_logit_swiglu(
    enabled, dtype, activation, native
):
    config = _tiny_gemma4_text_moe_config(dtype=dtype, hidden_act=activation)
    with (
        override_flags(experimental_opset29=enabled),
        build_context(ep_registry.require("onnx-standard"), dtype),
    ):
        model = _build_gemma4_text(config)
    nodes = [node for node in model.graph if node.op_type == "MoE"]
    assert bool(nodes) == native
    assert model.graph.opset_imports[""] == (29 if enabled else 24)
    for node in nodes:
        assert node.attributes.get_string("activation_type") == "swiglu"
        router = node.inputs[1]
        while router.producer().op_type == "Reshape":
            router = router.producer().inputs[0]
        assert router.producer().op_type == "MatMul"  # Raw logits, never probabilities.
    if not native:
        assert any(node.op_type == "Softmax" for node in model.graph)
        assert not any(node.op_type == "GroupedMatMul" for node in model.graph)


@pytest.mark.skipif(
    not onnx.defs.has("GroupedMatMul", 29), reason="GroupedMatMul-29 schemas unavailable"
)
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
def test_normal_build_inlines_gemma4_swiglu_at29(dtype):
    config = _tiny_gemma4_text_moe_config(dtype=dtype, hidden_act="silu")
    with override_flags(experimental_opset29=True):
        package = build_from_module(
            Gemma4CausalLMModel(config), config, execution_provider="onnx-standard"
        )
    model = package["model"]
    assert model.graph.opset_imports[""] == 29
    assert sum(node.op_type == "GroupedMatMul" for node in model.graph) == 4
    assert not any(node.op_type in {"MoE", "QMoE"} for node in model.graph)
    assert all(not node.domain for node in model.graph.all_nodes())
