# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from mobius._testing import create_test_builder, create_test_input
from mobius.components import GatedShortConv


def test_gated_short_conv_builds_stateful_graph():
    component = GatedShortConv(hidden_size=16, kernel_size=3)
    builder, op, graph = create_test_builder()
    hidden_states = create_test_input(builder, "hidden_states", [1, 4, 16])
    conv_state = create_test_input(builder, "conv_state", [1, 16, 3])
    attention_mask = create_test_input(builder, "attention_mask", [1, 4])

    output, present_state = component(op, hidden_states, conv_state, attention_mask)
    builder._adapt_outputs([output, present_state], "")

    assert any(node.op_type == "Conv" for node in graph)
    assert {name for name, _ in component.named_parameters()} == {
        "in_proj.weight",
        "conv.weight",
        "out_proj.weight",
    }
