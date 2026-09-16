# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph-verified packed hybrid ABI shared by runtime metadata exporters."""

from __future__ import annotations

import re
from dataclasses import dataclass

import onnx_ir as ir


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


def inspect_paged_hybrid(model: ir.Model) -> PagedHybridAbi | None:
    """Require all three state disciplines, not merely a block-table input."""
    if "mobius.paged_hybrid" not in model.metadata_props:
        return None
    if model.metadata_props["mobius.paged_hybrid"] != "qwen3_5_text":
        raise ValueError("Unknown packed hybrid ABI")
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
            or dtype not in {ir.DataType.FLOAT16, ir.DataType.BFLOAT16, ir.DataType.FLOAT}
            or (kind != "recurrent_state" and dtype == ir.DataType.FLOAT)
            or past.dtype != dtype
            or present is None
            or present.dtype != dtype
            or present.shape != past.shape
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
    native = {}
    state_operands = {"PagedAttention": 3, "VarlenCausalConvWithState": 4, "GatedDeltaNet": 6}
    for node in model.graph:
        if node.domain != "com.microsoft" or node.op_type not in state_operands:
            continue
        operand = state_operands[node.op_type]
        if len(node.inputs) <= operand or node.inputs[operand] is None:
            raise ValueError(f"Missing native {node.op_type} state operand")
        key = (node.op_type, node.inputs[operand].name)
        if key in native:
            raise ValueError(f"Duplicate native state consumer {key}")
        native[key] = node
    for kind, ids, op_type in (
        ("key", full, "PagedAttention"),
        ("conv_state", linear, "VarlenCausalConvWithState"),
        ("recurrent_state", linear, "GatedDeltaNet"),
    ):
        for layer in ids:
            node = native.get((op_type, f"past_key_values.{layer}.{kind}"))
            if node is None:
                raise ValueError(f"Missing native {op_type} for layer {layer}")
            cu_index = (
                5
                if op_type == "PagedAttention"
                else (2 if op_type == "VarlenCausalConvWithState" else 3)
            )
            if node.inputs[cu_index] is not inputs["cumulative_sequence_lengths"]:
                raise ValueError(f"Missing packed sequence boundaries at layer {layer}")
            if op_type == "PagedAttention" and (
                node.attributes.get_string("kv_cache_layout", "") != "SEPARATE"
                or len(node.inputs) != 17
                or any(value is not None for value in node.inputs[8:16])
                or node.inputs[16] is not inputs["attention_metadata"]
                or node.inputs[4] is not inputs[f"past_key_values.{layer}.value"]
                or node.inputs[6] is not inputs["past_sequence_lengths"]
                or node.inputs[7] is not inputs["block_table"]
            ):
                raise ValueError(f"Invalid SEPARATE PagedAttention contract at layer {layer}")
    return PagedHybridAbi(block_size, tuple(sorted(full)), tuple(sorted(linear)))
