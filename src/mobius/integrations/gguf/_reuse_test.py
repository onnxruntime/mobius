# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for GGUF external-data graph transforms."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from mobius.integrations.gguf._reuse import (
    GGUFReuseCandidate,
    _insert_external_transform,
)


def test_external_transform_precedes_all_consumers():
    initializer = ir.Value(
        name="weight",
        shape=ir.Shape([2, 2]),
        type=ir.TensorType(ir.DataType.FLOAT),
        const_value=ir.tensor(np.ones((2, 2), dtype=np.float32)),
    )
    early_output = ir.Value(name="early_output")
    late_output = ir.Value(name="late_output")

    # Construct uses in the opposite order from the graph to ensure insertion
    # follows graph topology rather than Value.uses() registration order.
    late = ir.Node("", "Identity", inputs=[initializer], outputs=[late_output], name="late")
    early = ir.Node("", "Identity", inputs=[initializer], outputs=[early_output], name="early")
    graph = ir.Graph([], [early_output, late_output], nodes=[early, late], name="test")
    graph.initializers[initializer.name] = initializer
    candidate = GGUFReuseCandidate(
        source_name="weight",
        offset=0,
        length=initializer.const_value.nbytes,
        qtype=0,
        transform="transpose",
        source_shape=(2, 2),
    )

    _insert_external_transform(graph, initializer, candidate)

    transform = next(node for node in graph if node.name == "weight.gguf_reuse.Transpose")
    assert graph.index(transform) < graph.index(early)
    assert graph.index(transform) < graph.index(late)
    assert early.inputs[0] is transform.outputs[0]
    assert late.inputs[0] is transform.outputs[0]
