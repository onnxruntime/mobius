# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph-verified packed hybrid ABI shared by runtime metadata exporters."""

from __future__ import annotations

import re
from dataclasses import dataclass

import onnx_ir as ir

_COMPUTE_DTYPES = {ir.DataType.FLOAT16, ir.DataType.BFLOAT16}

_OPERANDS = {
    "PagedAttention": {
        "query": 0,
        "key": 1,
        "value": 2,
        "key_cache": 3,
        "value_cache": 4,
        "cumulative_sequence_lengths": 5,
        "past_sequence_lengths": 6,
        "block_table": 7,
        "attention_metadata": 16,
    },
    "VarlenCausalConvWithState": {
        "packed": 0,
        "weight": 1,
        "cumulative_sequence_lengths": 2,
        "bias": 3,
        "state": 4,
    },
    "GatedDeltaNet": {
        "query": 0,
        "key": 1,
        "value": 2,
        "cumulative_sequence_lengths": 3,
        "raw_a": 4,
        "raw_b": 5,
        "state": 6,
        "a_log": 7,
        "dt_bias": 8,
    },
}
_STATE_OPERANDS = {
    op_type: operands[state_name]
    for op_type, operands, state_name in (
        ("PagedAttention", _OPERANDS["PagedAttention"], "key_cache"),
        ("VarlenCausalConvWithState", _OPERANDS["VarlenCausalConvWithState"], "state"),
        ("GatedDeltaNet", _OPERANDS["GatedDeltaNet"], "state"),
    )
}


@dataclass(frozen=True)
class PagedHybridAbi:
    block_size: int
    full_layers: tuple[int, ...]
    linear_layers: tuple[int, ...]

    def state_groups(self) -> list[dict]:
        return [
            {"kind": "paged_kv", "layer_ids": list(self.full_layers)},
            {"kind": "fixed_conv", "layer_ids": list(self.linear_layers)},
            {"kind": "fixed_recurrent", "layer_ids": list(self.linear_layers)},
        ]


def _require_value(
    value: ir.Value | None,
    *,
    dtype: ir.DataType,
    rank: int,
    description: str,
) -> ir.Value:
    if (
        value is None
        or value.dtype != dtype
        or value.shape is None
        or len(value.shape) != rank
    ):
        raise ValueError(f"{description} must have {dtype} rank {rank}")
    return value


def _attribute_value(node: ir.Node, name: str):
    attribute = node.attributes.get(name)
    return None if attribute is None else attribute.value


def _has_default_attribute(node: ir.Node, name: str, default) -> bool:
    value = _attribute_value(node, name)
    return value is None or value == default


def _operand(node: ir.Node, name: str) -> ir.Value | None:
    return node.inputs[_OPERANDS[node.op_type][name]]


def _require_native_outputs(
    node: ir.Node,
    count: int,
    state: ir.Value,
    present: ir.Value,
    layer: int,
) -> None:
    if len(node.outputs) != count or node.outputs[-1] is not present:
        raise ValueError(
            f"Native {node.op_type} state output is disconnected at layer {layer}"
        )
    native_state = node.outputs[-1]
    if native_state.dtype != state.dtype or not _shapes_compatible(
        native_state.shape, state.shape
    ):
        raise ValueError(f"Invalid native {node.op_type} state output at layer {layer}")


def _known_dimension_disagrees(left, right) -> bool:
    """Compare dimensions only when both are concrete graph facts."""
    return isinstance(left, int) and isinstance(right, int) and left != right


def _shapes_compatible(left: ir.Shape | None, right: ir.Shape | None) -> bool:
    """Reject only rank or concrete-dimension disagreements."""
    return (
        left is not None
        and right is not None
        and len(left) == len(right)
        and not any(_known_dimension_disagrees(a, b) for a, b in zip(left, right))
    )


def _validate_request_dimensions(
    inputs: dict[str, ir.Value], layers: dict[str, set[int]]
) -> None:
    batch_dimensions = {
        "block_table": inputs["block_table"].shape[0],
        "past_sequence_lengths": inputs["past_sequence_lengths"].shape[0],
    }
    for layer in layers["conv_state"]:
        batch_dimensions[f"conv_state at layer {layer}"] = inputs[
            f"past_key_values.{layer}.conv_state"
        ].shape[0]
        batch_dimensions[f"recurrent_state at layer {layer}"] = inputs[
            f"past_key_values.{layer}.recurrent_state"
        ].shape[0]
    concrete_batches = {
        dimension for dimension in batch_dimensions.values() if isinstance(dimension, int)
    }
    if len(concrete_batches) > 1:
        raise ValueError("Packed hybrid request-aligned batch dimensions disagree")
    cumulative_extent = inputs["cumulative_sequence_lengths"].shape[0]
    if concrete_batches and isinstance(cumulative_extent, int):
        batch = next(iter(concrete_batches))
        if cumulative_extent != batch + 1:
            raise ValueError(
                "Packed hybrid cumulative_sequence_lengths extent must equal batch + 1"
            )


def _validate_paged_attention(
    node: ir.Node,
    *,
    layer: int,
    inputs: dict[str, ir.Value],
    outputs: dict[str, ir.Value],
) -> None:
    key_cache = inputs[f"past_key_values.{layer}.key"]
    value_cache = inputs[f"past_key_values.{layer}.value"]
    if key_cache.dtype != value_cache.dtype or not _shapes_compatible(
        key_cache.shape, value_cache.shape
    ):
        raise ValueError(f"Paged key/value cache layouts disagree at layer {layer}")
    num_heads = _attribute_value(node, "num_heads")
    kv_num_heads = _attribute_value(node, "kv_num_heads")
    if (
        len(node.inputs) != 17
        or _operand(node, "key_cache") is not key_cache
        or _operand(node, "value_cache") is not value_cache
        or _operand(node, "cumulative_sequence_lengths")
        is not inputs["cumulative_sequence_lengths"]
        or _operand(node, "past_sequence_lengths") is not inputs["past_sequence_lengths"]
        or _operand(node, "block_table") is not inputs["block_table"]
        or any(operand is not None for operand in node.inputs[8:16])
        or _operand(node, "attention_metadata") is not inputs["attention_metadata"]
        or not _has_default_attribute(node, "kv_cache_layout", "SEPARATE")
        or not _has_default_attribute(node, "do_rotary", 0)
        or not _has_default_attribute(node, "is_causal", 1)
        or not _has_default_attribute(node, "local_window_size", -1)
        or not _has_default_attribute(node, "softcap", 0.0)
        or not _has_default_attribute(node, "scale", 0.0)
        or not isinstance(num_heads, int)
        or num_heads <= 0
        or not isinstance(kv_num_heads, int)
        or kv_num_heads <= 0
        or num_heads % kv_num_heads != 0
        or _known_dimension_disagrees(kv_num_heads, key_cache.shape[2])
        or len(node.outputs) != 3
        or node.outputs[1] is not outputs[f"present.{layer}.key"]
        or node.outputs[2] is not outputs[f"present.{layer}.value"]
    ):
        raise ValueError(f"Invalid SEPARATE PagedAttention contract at layer {layer}")

    query, key, value = (_operand(node, name) for name in ("query", "key", "value"))
    if any(
        operand is None
        or operand.dtype not in _COMPUTE_DTYPES
        or operand.shape is None
        or len(operand.shape) != 2
        for operand in (query, key, value)
    ):
        raise ValueError(f"Invalid PagedAttention QKV operands at layer {layer}")
    assert query is not None and query.shape is not None
    assert key is not None and key.shape is not None
    assert value is not None and value.shape is not None
    if (
        key.dtype != query.dtype
        or value.dtype != query.dtype
        or _known_dimension_disagrees(query.shape[0], key.shape[0])
        or _known_dimension_disagrees(query.shape[0], value.shape[0])
        or _known_dimension_disagrees(key.shape[0], value.shape[0])
    ):
        raise ValueError(f"Invalid PagedAttention QKV layout at layer {layer}")
    head_dim = key_cache.shape[3]
    if isinstance(head_dim, int) and (
        _known_dimension_disagrees(query.shape[1], num_heads * head_dim)
        or _known_dimension_disagrees(key.shape[1], kv_num_heads * head_dim)
        or _known_dimension_disagrees(value.shape[1], kv_num_heads * head_dim)
    ):
        raise ValueError(f"Invalid PagedAttention QKV width at layer {layer}")

    data_output = node.outputs[0]
    if (
        data_output.dtype not in _COMPUTE_DTYPES
        or data_output.dtype != query.dtype
        or data_output.shape is None
        or len(data_output.shape) != 2
        or _known_dimension_disagrees(data_output.shape[0], query.shape[0])
        or _known_dimension_disagrees(data_output.shape[1], query.shape[1])
        or not data_output.uses()
    ):
        raise ValueError(
            f"Invalid or disconnected PagedAttention data output at layer {layer}"
        )
    if (
        node.outputs[1].dtype != key_cache.dtype
        or not _shapes_compatible(node.outputs[1].shape, key_cache.shape)
        or node.outputs[2].dtype != value_cache.dtype
        or not _shapes_compatible(node.outputs[2].shape, value_cache.shape)
    ):
        raise ValueError(f"Invalid PagedAttention state outputs at layer {layer}")


def _validate_varlen_conv(
    node: ir.Node,
    *,
    layer: int,
    inputs: dict[str, ir.Value],
    outputs: dict[str, ir.Value],
) -> None:
    state = inputs[f"past_key_values.{layer}.conv_state"]
    if (
        len(node.inputs) != 5
        or _operand(node, "cumulative_sequence_lengths")
        is not inputs["cumulative_sequence_lengths"]
        or _operand(node, "state") is not state
        or _attribute_value(node, "activation") != "silu"
        or not _has_default_attribute(node, "dilation", 1)
        or not _has_default_attribute(node, "state_update_capacity", 0)
    ):
        raise ValueError(f"Invalid VarlenCausalConvWithState contract at layer {layer}")
    _require_native_outputs(node, 2, state, outputs[f"present.{layer}.conv_state"], layer)
    packed = _operand(node, "packed")
    weight = _operand(node, "weight")
    bias = _operand(node, "bias")
    if (
        packed is None
        or packed.dtype not in _COMPUTE_DTYPES
        or packed.shape is None
        or len(packed.shape) != 2
        or weight is None
        or weight.dtype != packed.dtype
        or weight.shape is None
        or len(weight.shape) != 3
        or _known_dimension_disagrees(weight.shape[1], 1)
        or bias is None
        or bias.dtype != packed.dtype
        or bias.shape is None
        or len(bias.shape) != 1
        or _known_dimension_disagrees(packed.shape[1], state.shape[1])
        or _known_dimension_disagrees(weight.shape[0], state.shape[1])
        or _known_dimension_disagrees(bias.shape[0], state.shape[1])
    ):
        raise ValueError(f"Invalid VarlenCausalConvWithState operands at layer {layer}")
    if (
        isinstance(weight.shape[2], int)
        and isinstance(state.shape[2], int)
        and weight.shape[2] != state.shape[2] + 1
    ):
        raise ValueError(f"Invalid VarlenCausalConvWithState layout at layer {layer}")


def _validate_gated_delta_net(
    node: ir.Node,
    *,
    layer: int,
    inputs: dict[str, ir.Value],
    outputs: dict[str, ir.Value],
) -> None:
    state = inputs[f"past_key_values.{layer}.recurrent_state"]
    required_attributes = {
        "gate_activation": "qwen",
        "beta_activation": "sigmoid",
        "qk_l2_norm": 1,
        "update_rule": "gated_delta",
    }
    if (
        len(node.inputs) != 9
        or _operand(node, "cumulative_sequence_lengths")
        is not inputs["cumulative_sequence_lengths"]
        or _operand(node, "state") is not state
        or any(
            (
                not _has_default_attribute(node, name, expected)
                if name == "update_rule"
                else _attribute_value(node, name) != expected
            )
            for name, expected in required_attributes.items()
        )
        or not _has_default_attribute(node, "scale", 0.0)
        or not _has_default_attribute(node, "state_update_capacity", 0)
    ):
        raise ValueError(f"Invalid GatedDeltaNet contract at layer {layer}")
    _require_native_outputs(node, 2, state, outputs[f"present.{layer}.recurrent_state"], layer)
    query, key, value = (_operand(node, name) for name in ("query", "key", "value"))
    if any(
        operand is None
        or operand.dtype not in _COMPUTE_DTYPES
        or operand.shape is None
        or len(operand.shape) != 3
        for operand in (query, key, value)
    ):
        raise ValueError(f"Invalid GatedDeltaNet QKV operands at layer {layer}")
    assert query is not None and query.shape is not None
    assert key is not None and key.shape is not None
    assert value is not None and value.shape is not None
    if (
        key.dtype != query.dtype
        or value.dtype != query.dtype
        or any(
            _known_dimension_disagrees(query.shape[axis], key.shape[axis]) for axis in range(3)
        )
        or _known_dimension_disagrees(query.shape[0], value.shape[0])
    ):
        raise ValueError(f"Invalid GatedDeltaNet QKV layout at layer {layer}")
    raw_a = _operand(node, "raw_a")
    raw_b = _operand(node, "raw_b")
    a_log = _operand(node, "a_log")
    dt_bias = _operand(node, "dt_bias")
    for name, operand, rank in (
        ("raw_a", raw_a, 2),
        ("raw_b", raw_b, 2),
        ("A_log", a_log, 1),
        ("dt_bias", dt_bias, 1),
    ):
        _require_value(
            operand,
            dtype=ir.DataType.FLOAT,
            rank=rank,
            description=f"GatedDeltaNet {name} at layer {layer}",
        )
    assert raw_a is not None and raw_a.shape is not None
    assert raw_b is not None and raw_b.shape is not None
    assert a_log is not None and a_log.shape is not None
    assert dt_bias is not None and dt_bias.shape is not None
    if (
        any(
            _known_dimension_disagrees(raw_a.shape[axis], raw_b.shape[axis])
            for axis in range(2)
        )
        or _known_dimension_disagrees(query.shape[0], raw_a.shape[0])
        or _known_dimension_disagrees(query.shape[0], raw_b.shape[0])
        or _known_dimension_disagrees(value.shape[1], state.shape[1])
        or _known_dimension_disagrees(value.shape[2], state.shape[2])
        or _known_dimension_disagrees(query.shape[2], state.shape[3])
        or _known_dimension_disagrees(raw_a.shape[1], state.shape[1])
        or _known_dimension_disagrees(a_log.shape[0], state.shape[1])
        or _known_dimension_disagrees(dt_bias.shape[0], state.shape[1])
        or (
            isinstance(query.shape[1], int)
            and (
                query.shape[1] <= 0
                or (isinstance(state.shape[1], int) and state.shape[1] % query.shape[1] != 0)
            )
        )
    ):
        raise ValueError(f"Invalid V-major GatedDeltaNet layout at layer {layer}")
    data_output = node.outputs[0]
    if (
        data_output.dtype != value.dtype
        or data_output.shape is None
        or len(data_output.shape) != 3
        or _known_dimension_disagrees(data_output.shape[0], value.shape[0])
        or _known_dimension_disagrees(data_output.shape[1], state.shape[1])
        or _known_dimension_disagrees(data_output.shape[2], state.shape[2])
    ):
        raise ValueError(f"Invalid GatedDeltaNet data output at layer {layer}")


def inspect_paged_hybrid(model: ir.Model) -> PagedHybridAbi | None:
    """Require all three state disciplines, not merely a block-table input."""
    if "mobius.paged_hybrid" not in model.metadata_props:
        return None
    if model.metadata_props["mobius.paged_hybrid"] != "qwen3_5_text":
        raise ValueError("Unknown packed hybrid ABI")
    input_names = [value.name for value in model.graph.inputs]
    output_names = [value.name for value in model.graph.outputs]
    if len(input_names) != len(set(input_names)) or len(output_names) != len(
        set(output_names)
    ):
        raise ValueError("Packed hybrid graph ports must have unique names")
    inputs = {value.name: value for value in model.graph.inputs}
    outputs = {value.name: value for value in model.graph.outputs}
    required = {
        "input_ids": (ir.DataType.INT64, 1),
        "position_ids": (ir.DataType.INT64, 2),
        "block_table": (ir.DataType.INT32, 2),
        "cumulative_sequence_lengths": (ir.DataType.INT32, 1),
        "past_sequence_lengths": (ir.DataType.INT32, 1),
        "attention_metadata": (ir.DataType.INT32, 1),
    }
    for name, (dtype, rank) in required.items():
        value = inputs.get(name)
        if (
            value is None
            or value.dtype != dtype
            or value.shape is None
            or len(value.shape) != rank
        ):
            raise ValueError(f"Packed hybrid input {name!r} must have {dtype} rank {rank}")
    if inputs["position_ids"].shape[0] != 3 or inputs["attention_metadata"].shape != ir.Shape(
        [3]
    ):
        raise ValueError("Packed hybrid requires three position planes and three CPU bounds")
    if "attention_mask" in inputs or "slot_mapping" in inputs:
        raise ValueError("Packed hybrid does not expose attention_mask or slot_mapping")
    logits = outputs.get("logits")
    if (
        logits is None
        or logits.dtype != ir.DataType.FLOAT
        or logits.shape is None
        or len(logits.shape) != 2
    ):
        raise ValueError("Packed hybrid logits must be rank-2 FLOAT")
    block_size = int(model.metadata_props.get("mobius.paged_block_size", "0"))
    if block_size <= 0 or block_size % 256:
        raise ValueError("Packed hybrid block size must be a positive multiple of 256")
    layers: dict[str, set[int]] = {
        kind: set() for kind in ("key", "value", "conv_state", "recurrent_state")
    }
    for name, past in inputs.items():
        match = re.fullmatch(
            r"past_key_values\.(\d+)\.(key|value|conv_state|recurrent_state)", name or ""
        )
        if match is None:
            if name not in required:
                raise ValueError(f"Unknown packed hybrid input {name!r}")
            continue
        layer, kind = int(match[1]), match[2]
        layers[kind].add(layer)
        present = outputs.get(f"present.{layer}.{kind}")
        rank = 3 if kind == "conv_state" else 4
        dtype = ir.DataType.FLOAT if kind == "recurrent_state" else past.dtype
        if (
            past.shape is None
            or len(past.shape) != rank
            or dtype not in _COMPUTE_DTYPES | {ir.DataType.FLOAT}
            or (kind != "recurrent_state" and dtype not in _COMPUTE_DTYPES)
            or past.dtype != dtype
            or present is None
            or present.dtype != dtype
            or not _shapes_compatible(present.shape, past.shape)
        ):
            raise ValueError(f"Invalid packed hybrid state pair {name!r}")
        if kind in {"key", "value"} and past.shape[1] != block_size:
            raise ValueError(f"Page extent disagrees with block size for {name!r}")
    full, linear = layers["key"], layers["conv_state"]
    if (
        not full
        or not linear
        or full != layers["value"]
        or linear != layers["recurrent_state"]
        or full & linear
        or full | linear != set(range(max(full | linear) + 1))
    ):
        raise ValueError(
            "Packed hybrid requires disjoint, complete paged and fixed layer groups"
        )
    expected_outputs = {"logits"} | {
        f"present.{layer}.{kind}" for kind, ids in layers.items() for layer in ids
    }
    if set(outputs) != expected_outputs:
        raise ValueError("Packed hybrid graph has unknown or missing outputs")
    _validate_request_dimensions(inputs, layers)

    native: dict[tuple[str, str], ir.Node] = {}
    for node in model.graph:
        if node.domain != "com.microsoft" or node.op_type not in _STATE_OPERANDS:
            continue
        operand = _STATE_OPERANDS[node.op_type]
        if len(node.inputs) <= operand or node.inputs[operand] is None:
            raise ValueError(f"Missing native {node.op_type} state operand")
        key = (node.op_type, node.inputs[operand].name)
        if key in native:
            raise ValueError(f"Duplicate native state consumer {key}")
        native[key] = node

    expected_native_keys: set[tuple[str, str]] = set()
    for layer in full:
        key = inputs[f"past_key_values.{layer}.key"]
        node_key = ("PagedAttention", key.name)
        expected_native_keys.add(node_key)
        node = native.get(node_key)
        if node is None:
            raise ValueError(f"Missing native PagedAttention for layer {layer}")
        _validate_paged_attention(node, layer=layer, inputs=inputs, outputs=outputs)

    for layer in linear:
        conv_state = inputs[f"past_key_values.{layer}.conv_state"]
        conv_key = ("VarlenCausalConvWithState", conv_state.name)
        expected_native_keys.add(conv_key)
        conv = native.get(conv_key)
        if conv is None:
            raise ValueError(f"Missing native VarlenCausalConvWithState for layer {layer}")
        _validate_varlen_conv(conv, layer=layer, inputs=inputs, outputs=outputs)

        recurrent_state = inputs[f"past_key_values.{layer}.recurrent_state"]
        delta_key = ("GatedDeltaNet", recurrent_state.name)
        expected_native_keys.add(delta_key)
        delta = native.get(delta_key)
        if delta is None:
            raise ValueError(f"Missing native GatedDeltaNet for layer {layer}")
        _validate_gated_delta_net(delta, layer=layer, inputs=inputs, outputs=outputs)

    if set(native) != expected_native_keys:
        raise ValueError("Packed hybrid graph has extra or cross-bound native state bindings")
    return PagedHybridAbi(block_size, tuple(sorted(full)), tuple(sorted(linear)))
