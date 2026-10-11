# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Convert a Mobius tensor-cache decoder graph to experimental kv_cache ONNX."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

_OPSET_VERSION = 29
_CACHE_INPUT = re.compile(r"^past_key_values\.(\d+)\.(key|value)$")
_CACHE_OUTPUT = re.compile(r"^present\.(\d+)\.(key|value)$")


def _decoder_config(config: dict) -> dict:
    """Return the text decoder fields from either a text or multimodal config."""
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Expected {name} to be a positive integer, got {value!r}.")
    return value


def _attention_attributes(node: onnx.NodeProto) -> dict[str, int | float]:
    supported = {
        "is_causal",
        "kv_num_heads",
        "q_num_heads",
        "scale",
        "softcap",
        "softmax_precision",
    }
    attributes: dict[str, int | float] = {}
    for attribute in node.attribute:
        if attribute.name not in supported:
            raise ValueError(
                f"Attention node {node.name!r} has unsupported attribute {attribute.name!r}."
            )
        value = helper.get_attribute_value(attribute)
        if attribute.name == "softmax_precision" and value != onnx.TensorProto.FLOAT:
            raise ValueError(
                f"Attention node {node.name!r} requests non-FLOAT softmax_precision "
                f"{value!r}; only FLOAT is supported."
            )
        attributes[attribute.name] = value
    return attributes


def convert_model(model: onnx.ModelProto, config: dict) -> onnx.ModelProto:
    """Replace each standard tensor-cache Attention with AttentionWithCache.

    The original graph must be an unfused Mobius dynamic-cache decoder graph:
    one default-domain Attention node and paired past/present key/value values
    per decoder layer. Q/K/V are reshaped from BS(H*D) to BNSD around the new
    operator. Existing projection, Q/K normalization, RoPE, masks, and logits
    computation are otherwise untouched.
    """
    decoder_config = _decoder_config(config)
    num_heads = _integer(decoder_config.get("num_attention_heads"), "num_attention_heads")
    num_kv_heads = _integer(
        decoder_config.get("num_key_value_heads", num_heads), "num_key_value_heads"
    )
    head_dim = _integer(
        decoder_config.get("head_dim") or decoder_config.get("hidden_size", 0) // num_heads,
        "head_dim",
    )
    num_layers = _integer(decoder_config.get("num_hidden_layers"), "num_hidden_layers")
    sliding_window = decoder_config.get("sliding_window") or 0
    if not isinstance(sliding_window, int) or sliding_window < 0:
        raise ValueError(f"Invalid sliding_window: {sliding_window!r}.")
    layer_types = decoder_config.get("layer_types")
    if layer_types is not None and (
        not isinstance(layer_types, list) or len(layer_types) != num_layers
    ):
        raise ValueError("config.layer_types must contain one value per decoder layer.")
    if sliding_window and layer_types is None:
        raise ValueError(
            "A windowed checkpoint must provide explicit per-layer layer_types; "
            "the converter will not infer a full/sliding schedule."
        )
    if layer_types is not None and any(
        layer_type not in {"full_attention", "sliding_attention"} for layer_type in layer_types
    ):
        raise ValueError("Only full_attention and sliding_attention layers are supported.")

    def window_for_layer(layer: int) -> int:
        return (
            sliding_window
            if layer_types is not None and layer_types[layer] == "sliding_attention"
            else 0
        )

    cache_dtype: int | None = None
    input_cache_layers: dict[int, dict[str, str]] = {}
    output_cache_layers: dict[int, dict[str, str]] = {}
    for value in model.graph.input:
        match = _CACHE_INPUT.match(value.name)
        if match:
            layer = int(match.group(1))
            kind = match.group(2)
            input_cache_layers.setdefault(layer, {})[kind] = value.name
            if kind == "key":
                cache_dtype = value.type.tensor_type.elem_type
    for value in model.graph.output:
        match = _CACHE_OUTPUT.match(value.name)
        if match:
            layer = int(match.group(1))
            kind = match.group(2)
            output_cache_layers.setdefault(layer, {})[kind] = value.name

    expected_layers = set(range(num_layers))
    if set(input_cache_layers) != expected_layers or any(
        set(values) != {"key", "value"} for values in input_cache_layers.values()
    ):
        raise ValueError(
            "Expected paired past_key_values.<layer>.key/value graph inputs for "
            f"layers 0..{num_layers - 1}; found {input_cache_layers!r}."
        )
    if set(output_cache_layers) != expected_layers or any(
        set(values) != {"key", "value"} for values in output_cache_layers.values()
    ):
        raise ValueError(
            "Expected paired present.<layer>.key/value graph outputs for "
            f"layers 0..{num_layers - 1}; found {output_cache_layers!r}."
        )
    if cache_dtype is None or cache_dtype not in {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.BFLOAT16,
    }:
        raise ValueError(f"Unsupported or missing KV cache element type: {cache_dtype!r}.")

    def check_cache_geometry(value: onnx.ValueInfoProto, expected_width: int) -> None:
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type != cache_dtype or len(tensor_type.shape.dim) != 4:
            raise ValueError(f"Unexpected tensor cache type on {value.name!r}.")
        dims = tensor_type.shape.dim
        for axis, expected in ((1, num_kv_heads), (3, expected_width)):
            if dims[axis].HasField("dim_value") and dims[axis].dim_value != expected:
                raise ValueError(
                    f"Cache tensor {value.name!r} axis {axis} is {dims[axis].dim_value}, "
                    f"but the checkpoint config requires {expected}."
                )

    input_by_name = {value.name: value for value in model.graph.input}
    for layer in range(num_layers):
        check_cache_geometry(input_by_name[input_cache_layers[layer]["key"]], head_dim)
        check_cache_geometry(input_by_name[input_cache_layers[layer]["value"]], head_dim)

    old_cache_inputs = {
        name for layer in input_cache_layers.values() for name in layer.values()
    }
    old_cache_outputs = {
        name for layer in output_cache_layers.values() for name in layer.values()
    }
    for node in model.graph.node:
        for input_index, name in enumerate(node.input):
            if name in old_cache_inputs and not (
                node.domain in ("", "ai.onnx")
                and node.op_type == "Attention"
                and input_index in (4, 5)
            ):
                raise ValueError(
                    f"Cache tensor {name!r} is consumed outside a standard Attention "
                    f"past_key/past_value input (node {node.name!r}, input {input_index}). "
                    "Length/mask plumbing must use logical position inputs, not cache "
                    "tensor shape."
                )
            if name in old_cache_outputs:
                raise ValueError(
                    f"Present cache tensor {name!r} is consumed by node {node.name!r}; "
                    "the converter cannot preserve this non-attention cache consumer."
                )

    converted_nodes: list[onnx.NodeProto] = []
    seen_layers: set[int] = set()
    shape_initializers: list[onnx.TensorProto] = []
    for node_index, node in enumerate(model.graph.node):
        if node.domain not in ("", "ai.onnx") or node.op_type != "Attention":
            converted_nodes.append(node)
            continue
        if len(node.input) < 6 or len(node.output) < 3:
            raise ValueError(
                f"Attention node {node.name!r} is not a dynamic-cache "
                "Attention with past_key/past_value and present outputs."
            )
        if len(node.input) > 6 and node.input[6]:
            raise ValueError(
                f"Attention node {node.name!r} uses nonpad_kv_seqlen with a dynamic cache."
            )
        if any(node.output[3:]):
            raise ValueError(
                f"Attention node {node.name!r} requests a diagnostic output that "
                "AttentionWithCache does not preserve."
            )
        layer = next(
            (
                index
                for index, names in input_cache_layers.items()
                if node.input[4] == names["key"] and node.input[5] == names["value"]
            ),
            None,
        )
        if layer is None or layer in seen_layers:
            raise ValueError(
                f"Could not uniquely map Attention node {node.name!r} to its KV inputs."
            )
        if (
            node.output[1] != output_cache_layers[layer]["key"]
            or node.output[2] != output_cache_layers[layer]["value"]
        ):
            raise ValueError(f"Attention node {node.name!r} has unexpected KV outputs.")
        seen_layers.add(layer)

        window = window_for_layer(layer)
        cache_in = f"past_cache_{layer}"
        cache_out = f"present_cache_{layer}"
        q_bsh = f"kvcache_{node_index}_q_bsh"
        k_bsh = f"kvcache_{node_index}_k_bsh"
        v_bsh = f"kvcache_{node_index}_v_bsh"
        q_bnsd = f"kvcache_{node_index}_q_bnsd"
        k_bnsd = f"kvcache_{node_index}_k_bnsd"
        v_bnsd = f"kvcache_{node_index}_v_bnsd"
        y_bnsd = f"kvcache_{node_index}_y_bnsd"
        y_bsh = f"kvcache_{node_index}_y_bsh"
        q_shape = f"kvcache_{node_index}_q_shape"
        k_shape = f"kvcache_{node_index}_k_shape"
        v_shape = f"kvcache_{node_index}_v_shape"
        y_shape = f"kvcache_{node_index}_y_shape"
        for name, shape in (
            (q_shape, [0, 0, num_heads, head_dim]),
            (k_shape, [0, 0, num_kv_heads, head_dim]),
            (v_shape, [0, 0, num_kv_heads, head_dim]),
            (y_shape, [0, 0, -1]),
        ):
            shape_initializers.append(
                onnx.numpy_helper.from_array(np.asarray(shape, dtype=np.int64), name=name)
            )

        converted_nodes.extend(
            [
                helper.make_node("Reshape", [node.input[0], q_shape], [q_bsh]),
                helper.make_node("Transpose", [q_bsh], [q_bnsd], perm=[0, 2, 1, 3]),
                helper.make_node("Reshape", [node.input[1], k_shape], [k_bsh]),
                helper.make_node("Transpose", [k_bsh], [k_bnsd], perm=[0, 2, 1, 3]),
                helper.make_node("Reshape", [node.input[2], v_shape], [v_bsh]),
                helper.make_node("Transpose", [v_bsh], [v_bnsd], perm=[0, 2, 1, 3]),
            ]
        )
        attention_inputs = [
            q_bnsd,
            k_bnsd,
            v_bnsd,
            cache_in,
            node.input[3] if len(node.input) > 3 else "",
            "position_ids"
            if any(value.name == "position_ids" for value in model.graph.input)
            else "",
        ]
        attributes: dict[str, int | float] = _attention_attributes(node)
        for name, expected in (
            ("q_num_heads", num_heads),
            ("kv_num_heads", num_kv_heads),
        ):
            if name in attributes and attributes[name] != expected:
                raise ValueError(
                    f"Attention node {node.name!r} has {name}={attributes[name]!r}, "
                    f"but the checkpoint config requires {expected}."
                )
        attributes.update(
            q_num_heads=num_heads,
            kv_num_heads=num_kv_heads,
            sliding_window=window,
        )
        converted_nodes.append(
            helper.make_node(
                "AttentionWithCache",
                attention_inputs,
                [y_bnsd, cache_out],
                name=f"AttentionWithCache_{layer}",
                domain="",
                **attributes,
            )
        )
        converted_nodes.extend(
            [
                helper.make_node("Transpose", [y_bnsd], [y_bsh], perm=[0, 2, 1, 3]),
                helper.make_node("Reshape", [y_bsh, y_shape], [node.output[0]]),
            ]
        )
    if seen_layers != expected_layers:
        raise ValueError(
            "Expected exactly one standard Attention node per KV layer; "
            f"converted {sorted(seen_layers)!r}, expected {sorted(expected_layers)!r}."
        )

    del model.graph.node[:]
    model.graph.node.extend(converted_nodes)
    model.graph.initializer.extend(shape_initializers)

    retained_inputs = [
        value for value in model.graph.input if value.name not in old_cache_inputs
    ]
    del model.graph.input[:]
    model.graph.input.extend(retained_inputs)
    for layer in range(num_layers):
        model.graph.input.append(
            helper.make_kv_cache_type(
                name=f"past_cache_{layer}",
                elem_type=cache_dtype,
                batch_size="batch",
                num_kv_heads=num_kv_heads,
                key_head_size=head_dim,
                value_head_size=head_dim,
                sliding_window=window_for_layer(layer),
            )
        )
    retained_outputs = [
        value for value in model.graph.output if value.name not in old_cache_outputs
    ]
    del model.graph.output[:]
    model.graph.output.extend(retained_outputs)
    for layer in range(num_layers):
        model.graph.output.append(
            helper.make_kv_cache_type(
                name=f"present_cache_{layer}",
                elem_type=cache_dtype,
                batch_size="batch",
                num_kv_heads=num_kv_heads,
                key_head_size=head_dim,
                value_head_size=head_dim,
                sliding_window=window_for_layer(layer),
            )
        )
    retained_value_info = [
        value
        for value in model.graph.value_info
        if value.name not in old_cache_inputs | old_cache_outputs
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)

    default_import = next(
        (item for item in model.opset_import if item.domain in ("", "ai.onnx")),
        None,
    )
    if default_import is None:
        model.opset_import.append(helper.make_opsetid("", _OPSET_VERSION))
    else:
        default_import.domain = ""
        default_import.version = _OPSET_VERSION
    for function in model.functions:
        for imported_opset in function.opset_import:
            if imported_opset.domain in ("", "ai.onnx"):
                imported_opset.domain = ""
                imported_opset.version = _OPSET_VERSION
    model.ir_version = 15
    model.doc_string = (
        f"{model.doc_string}\n"
        "Converted for the experimental ONNX kv_cache type (IR 15, opset 29)."
    ).strip()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_model", type=Path)
    parser.add_argument("config", type=Path, help="Hugging Face config.json")
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()
    if args.input_model.resolve().parent != args.output_model.resolve().parent:
        raise ValueError(
            "Keep the converted model next to its source model so ONNX external "
            "weight-data locations continue to resolve."
        )
    with args.config.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    model = onnx.load_model(args.input_model, load_external_data=False)
    onnx.save_model(convert_model(model, config), args.output_model)
    onnx.checker.check_model(str(args.output_model))
    print(f"Wrote {args.output_model}")


if __name__ == "__main__":
    main()
