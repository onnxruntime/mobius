# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Structural tests for the packed Qwen3.5 hybrid serving graph."""

from __future__ import annotations

import inspect

import onnx_ir as ir
import pytest

from mobius._testing import make_config
from mobius.models.qwen35 import (
    Qwen35CausalLMModel,
    Qwen35DecoderLayer,
    Qwen35TextModel,
)
from mobius.tasks import PagedHybridCausalLMTask


def _config(*, layers: int = 2, layer_types: list[str] | None = None):
    return make_config(
        model_type="qwen3_5_text",
        dtype=ir.DataType.FLOAT16,
        num_hidden_layers=layers,
        layer_types=layer_types or ["full_attention", "linear_attention"],
        partial_rotary_factor=0.5,
        mrope_section=[2, 1, 1],
        mrope_interleaved=True,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
    )


def _build(config, *, prune: bool = False):
    return PagedHybridCausalLMTask(prune_prefill_prefix=prune).build(
        Qwen35CausalLMModel(config), config
    )["model"]


def _nodes(model, op_type: str):
    return [node for node in model.graph if node.op_type == op_type]


def test_packed_hybrid_io_and_native_operands():
    model = _build(_config())
    inputs = {value.name: value for value in model.graph.inputs}
    assert {"input_ids", "position_ids", "block_table"} <= inputs.keys()
    assert "attention_mask" not in inputs
    assert "slot_mapping" not in inputs
    assert inputs["input_ids"].shape == ir.Shape(["num_tokens"])
    assert inputs["position_ids"].shape == ir.Shape([3, "num_tokens"])
    assert inputs["attention_metadata"].dtype == ir.DataType.INT32
    assert inputs["past_key_values.0.key"].shape[1:] == ir.Shape([256, 2, 16])
    assert inputs["past_key_values.1.recurrent_state"].shape[1:] == ir.Shape([4, 16, 16])
    assert inputs["past_key_values.1.recurrent_state"].dtype == ir.DataType.FLOAT

    paged = _nodes(model, "PagedAttention")
    conv = _nodes(model, "VarlenCausalConvWithState")
    delta = _nodes(model, "GatedDeltaNet")
    assert len(paged) == len(conv) == len(delta) == 1
    assert paged[0].domain == conv[0].domain == delta[0].domain == "com.microsoft"
    assert len(paged[0].inputs) == 17
    assert all(value is None for value in paged[0].inputs[8:16])
    assert paged[0].inputs[16].name == "attention_metadata"
    attrs = {name: attr.value for name, attr in paged[0].attributes.items()}
    assert attrs["kv_cache_layout"] == "SEPARATE"
    assert attrs["do_rotary"] == 0
    delta_attrs = {name: attr.value for name, attr in delta[0].attributes.items()}
    assert delta_attrs["gate_activation"] == "qwen"
    assert delta_attrs["qk_l2_norm"] == 1
    assert delta[0].inputs[4].dtype == ir.DataType.FLOAT
    assert delta[0].inputs[5].dtype == ir.DataType.FLOAT

    assert model.metadata_props["mobius.paged_hybrid"] == "qwen3_5_text"
    assert model.metadata_props["mobius.paged_block_size"] == "256"
    assert "mobius.ort_revision" in model.metadata_props
    assert "mobius.genai_revision" in model.metadata_props


def test_packed_context_has_an_explicit_component_api():
    causal_parameters = inspect.signature(Qwen35CausalLMModel.forward).parameters
    text_parameters = inspect.signature(Qwen35TextModel.forward).parameters
    layer_parameters = inspect.signature(Qwen35DecoderLayer.forward).parameters

    assert causal_parameters["paged_context"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "paged_context" in text_parameters
    assert "paged_context" in layer_parameters
    assert "PagedHybridContext" not in str(causal_parameters["attention_mask"].annotation)
    assert "PagedHybridContext" not in str(layer_parameters["attention_bias"].annotation)


def test_fp32_native_decay_parameters_and_qualified_weight_names():
    model = _build(_config())
    initializers = model.graph.initializers
    for name in (
        "model.layers.1.linear_attn.A_log",
        "model.layers.1.linear_attn.dt_bias",
    ):
        assert initializers[name].dtype == ir.DataType.FLOAT
    assert "model.layers.0.self_attn.q_norm.weight" in initializers
    assert "model.layers.1.linear_attn.conv1d.weight" in initializers


def test_pruning_gathers_each_packed_row_end_before_lm_head():
    model = _build(_config(), prune=True)
    assert model.graph.outputs[0].shape == ir.Shape(["batch", 100])
    assert any(
        node.op_type == "Gather"
        and node.inputs[1].producer() is not None
        and node.inputs[1].producer().op_type == "Sub"
        for node in model.graph
    )


def test_production_topology_emits_16_paged_and_48_native_layers():
    schedule = [
        "full_attention" if index % 4 == 3 else "linear_attention" for index in range(64)
    ]
    model = _build(_config(layers=64, layer_types=schedule))
    assert len(_nodes(model, "PagedAttention")) == 16
    assert len(_nodes(model, "VarlenCausalConvWithState")) == 48
    assert len(_nodes(model, "GatedDeltaNet")) == 48


@pytest.mark.parametrize("block_size", [0, 1, 255, 257])
def test_block_size_must_be_positive_multiple_of_256(block_size: int):
    with pytest.raises(ValueError, match="positive multiple of 256"):
        PagedHybridCausalLMTask(paged_block_size=block_size)
