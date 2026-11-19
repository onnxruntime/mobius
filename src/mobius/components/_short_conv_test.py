# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from mobius._testing import create_test_builder, create_test_input
from mobius.components import GatedShortConv


def test_gated_short_conv_builds_stateful_graph():
    component = GatedShortConv(hidden_size=16, kernel_size=3)
    builder, op, graph = create_test_builder()
    hidden_states = create_test_input(builder, "hidden_states", [1, 4, 16])
    conv_state = create_test_input(builder, "conv_state", [1, 16, 2])
    attention_mask = create_test_input(builder, "attention_mask", [1, 4])

    output, present_state = component(op, hidden_states, conv_state, attention_mask)
    builder._adapt_outputs([output, present_state], "")

    assert any(node.op_type == "CausalConvWithState" for node in graph)
    assert {name for name, _ in component.named_parameters()} == {
        "in_proj.weight",
        "conv.weight",
        "out_proj.weight",
    }


def test_gated_short_conv_masks_biased_projection_without_changing_positional_api():
    component = GatedShortConv(hidden_size=16, kernel_size=7, bias=True)
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 4, 16])
    state = create_test_input(builder, "state", [1, 16, 6])
    mask = create_test_input(builder, "mask", [1, 4])
    output, present = component(op, hidden, state, mask, mask)
    builder._adapt_outputs([output, present], "")
    conv = next(node for node in graph if node.op_type == "CausalConvWithState")
    masked = conv.inputs[0].producer()
    assert masked.op_type == "Mul"
    assert masked.inputs[1].producer().op_type == "CastLike"
