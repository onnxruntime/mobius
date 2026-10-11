# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph conversion tests for the experimental ONNX kv_cache example."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import onnx
import pytest
from onnx import helper

_CONVERTER_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "kvcache" / "convert_to_kv_cache.py"
)
_SPEC = importlib.util.spec_from_file_location("convert_to_kv_cache", _CONVERTER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CONVERTER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONVERTER)


def _tensor_value(name: str, shape: list[str | int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _tensor_cache_model(
    *,
    cache_shape_consumer: bool = False,
    diagnostic_output: bool = False,
) -> onnx.ModelProto:
    graph_inputs = [
        helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, ["batch", "seq"]),
        helper.make_tensor_value_info(
            "attention_mask", onnx.TensorProto.BOOL, ["batch", 1, "seq", "total"]
        ),
        helper.make_tensor_value_info(
            "position_ids", onnx.TensorProto.INT64, ["batch", "seq"]
        ),
    ]
    graph_outputs = []
    nodes = []
    for layer in range(2):
        past_key = f"past_key_values.{layer}.key"
        past_value = f"past_key_values.{layer}.value"
        present_key = f"present.{layer}.key"
        present_value = f"present.{layer}.value"
        graph_inputs.extend(
            [
                _tensor_value(past_key, ["batch", 2, "past", 8]),
                _tensor_value(past_value, ["batch", 2, "past", 8]),
            ]
        )
        graph_outputs.extend(
            [
                _tensor_value(present_key, ["batch", 2, "total", 8]),
                _tensor_value(present_value, ["batch", 2, "total", 8]),
            ]
        )
        attention_outputs = [f"y_{layer}", present_key, present_value]
        if diagnostic_output:
            attention_outputs.append(f"attention_scores_{layer}")
        nodes.append(
            helper.make_node(
                "Attention",
                [
                    f"q_{layer}",
                    f"k_{layer}",
                    f"v_{layer}",
                    "attention_mask",
                    past_key,
                    past_value,
                ],
                attention_outputs,
                name=f"attention_{layer}",
                q_num_heads=4,
                kv_num_heads=2,
                scale=8**-0.5,
                is_causal=1,
            )
        )
        if cache_shape_consumer and layer == 0:
            nodes.append(helper.make_node("Shape", [past_key], ["past_key_shape_0"]))
        if layer == 1:
            graph_outputs.append(_tensor_value("logits", ["batch", "seq", 32]))
            nodes.append(helper.make_node("Identity", [f"y_{layer}"], ["logits"]))
    for layer in range(2):
        graph_inputs.extend(
            [
                _tensor_value(f"q_{layer}", ["batch", "seq", 32]),
                _tensor_value(f"k_{layer}", ["batch", "seq", 16]),
                _tensor_value(f"v_{layer}", ["batch", "seq", 16]),
            ]
        )
    model = helper.make_model(
        helper.make_graph(nodes, "decoder", graph_inputs, graph_outputs),
        opset_imports=[helper.make_opsetid("", 24)],
    )
    model.ir_version = 12
    return model


def test_convert_two_layer_dynamic_attention_to_kv_cache():
    if not hasattr(helper, "make_kv_cache_type"):
        pytest.skip("Requires the experimental ONNX kv_cache fork.")
    config = {
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "sliding_window": 4,
        "layer_types": ["full_attention", "sliding_attention"],
    }

    converted = _CONVERTER.convert_model(_tensor_cache_model(), config)

    assert converted.ir_version == 15
    assert {item.domain: item.version for item in converted.opset_import}[""] == 29
    assert not any(node.op_type == "Attention" for node in converted.graph.node)
    cache_nodes = [
        node for node in converted.graph.node if node.op_type == "AttentionWithCache"
    ]
    assert len(cache_nodes) == 2
    assert [node.name for node in cache_nodes] == [
        "AttentionWithCache_0",
        "AttentionWithCache_1",
    ]
    windows = []
    for node in cache_nodes:
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        windows.append(attributes["sliding_window"])
        assert attributes["is_causal"] == 1
    assert windows == [0, 4]
    assert len(cache_nodes[0].input) == 6
    assert cache_nodes[0].input[4] == "attention_mask"
    assert cache_nodes[0].input[5] == "position_ids"
    assert cache_nodes[1].input[3] == "past_cache_1"
    assert cache_nodes[1].output[1] == "present_cache_1"
    assert [
        value.name for value in converted.graph.input if value.type.HasField("kv_cache_type")
    ] == [
        "past_cache_0",
        "past_cache_1",
    ]
    assert [
        value.name for value in converted.graph.output if value.type.HasField("kv_cache_type")
    ] == [
        "present_cache_0",
        "present_cache_1",
    ]
    for value in [*converted.graph.input, *converted.graph.output]:
        if value.type.HasField("kv_cache_type"):
            assert value.type.kv_cache_type.elem_type == onnx.TensorProto.FLOAT
            assert value.type.kv_cache_type.num_kv_heads.dim_value == 2
            assert value.type.kv_cache_type.key_head_size.dim_value == 8
    assert converted.graph.input[-1].type.kv_cache_type.sliding_window == 4


def test_converter_updates_function_opsets_with_model_opset():
    if not hasattr(helper, "make_kv_cache_type"):
        pytest.skip("Requires the experimental ONNX kv_cache fork.")
    config = {
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "hidden_size": 32,
        "num_hidden_layers": 2,
    }
    model = _tensor_cache_model()
    function = onnx.FunctionProto(
        domain="com.microsoft", name="SkipSimplifiedLayerNormalization"
    )
    function.opset_import.extend(
        [helper.make_opsetid("", 24), helper.make_opsetid("com.microsoft", 1)]
    )
    model.functions.append(function)

    converted = _CONVERTER.convert_model(model, config)

    assert [(item.domain, item.version) for item in converted.functions[0].opset_import] == [
        ("", 29),
        ("com.microsoft", 1),
    ]


def test_converter_preserves_explicit_zero_scale_and_float_softmax_precision():
    if not hasattr(helper, "make_kv_cache_type"):
        pytest.skip("Requires the experimental ONNX kv_cache fork.")
    config = {
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "hidden_size": 32,
        "num_hidden_layers": 2,
    }
    model = _tensor_cache_model()
    for node in model.graph.node:
        if node.op_type == "Attention":
            for index, attribute in enumerate(node.attribute):
                if attribute.name == "scale":
                    del node.attribute[index]
                    break
            node.attribute.extend(
                [
                    helper.make_attribute("scale", 0.0),
                    helper.make_attribute("softmax_precision", onnx.TensorProto.FLOAT),
                ]
            )

    converted = _CONVERTER.convert_model(model, config)

    attributes = {
        attr.name: helper.get_attribute_value(attr)
        for node in converted.graph.node
        if node.op_type == "AttentionWithCache"
        for attr in node.attribute
    }
    assert math.isclose(attributes["scale"], 0.0, abs_tol=0.0)
    assert attributes["softmax_precision"] == onnx.TensorProto.FLOAT


@pytest.mark.parametrize(
    ("attribute_name", "attribute_value", "message"),
    [
        ("softmax_precision", onnx.TensorProto.FLOAT16, "non-FLOAT"),
        ("q_num_heads", 8, "checkpoint config requires"),
        ("extra_attention_attribute", 1, "unsupported attribute"),
    ],
)
def test_converter_rejects_attention_semantics_it_cannot_preserve(
    attribute_name: str,
    attribute_value: int,
    message: str,
):
    if not hasattr(helper, "make_kv_cache_type"):
        pytest.skip("Requires the experimental ONNX kv_cache fork.")
    config = {
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "hidden_size": 32,
        "num_hidden_layers": 2,
    }
    model = _tensor_cache_model()
    model.graph.node[0].attribute.append(
        helper.make_attribute(attribute_name, attribute_value)
    )
    with pytest.raises(ValueError, match=message):
        _CONVERTER.convert_model(model, config)


def test_converter_rejects_dynamic_cache_nonpad_lengths():
    if not hasattr(helper, "make_kv_cache_type"):
        pytest.skip("Requires the experimental ONNX kv_cache fork.")
    config = {
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "hidden_size": 32,
        "num_hidden_layers": 2,
    }
    model = _tensor_cache_model()
    model.graph.node[0].input.append("nonpad_kv_seqlen")

    with pytest.raises(ValueError, match="nonpad_kv_seqlen"):
        _CONVERTER.convert_model(model, config)


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (_tensor_cache_model(cache_shape_consumer=True), "consumed outside"),
        (_tensor_cache_model(diagnostic_output=True), "diagnostic output"),
    ],
)
def test_converter_rejects_unpreserved_cache_edges(model: onnx.ModelProto, message: str):
    if not hasattr(helper, "make_kv_cache_type"):
        pytest.skip("Requires the experimental ONNX kv_cache fork.")
    config = {
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "hidden_size": 32,
        "num_hidden_layers": 2,
    }
    with pytest.raises(ValueError, match=message):
        _CONVERTER.convert_model(model, config)
