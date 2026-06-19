# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for :mod:`mobius._builder` opset-lowering logic.

These tests are CPU-authorable: they construct tiny ONNX-IR graphs directly
(no weights, no network) and exercise the conditional opset 24→23 lowering
that preserves the static-cache Flash-attention path.
"""

from __future__ import annotations

import onnx_ir as ir

from mobius._builder import _graph_requires_opset24


def _make_value(name: str) -> ir.Value:
    return ir.Value(name=name)


def _graph_with(nodes: list[ir.Node]) -> ir.Graph:
    return ir.Graph(inputs=[], outputs=[], nodes=nodes, opset_imports={"": 24})


def test_graph_requires_opset24_tensor_scatter() -> None:
    # A TensorScatter node (opset-24-only) must force opset 24 retention.
    node = ir.Node(
        "",
        "TensorScatter",
        inputs=[_make_value("past"), _make_value("update"), _make_value("idx")],
    )
    assert _graph_requires_opset24(_graph_with([node])) is True


def test_graph_requires_opset24_attention_nonpad_kv_seqlen() -> None:
    # Attention consuming input #6 (nonpad_kv_seqlen) is opset-24-only.
    inputs = [
        _make_value("q"),
        _make_value("k"),
        _make_value("v"),
        None,  # attn_mask
        None,  # past_key
        None,  # past_value
        _make_value("nonpad_kv_seqlen"),  # input #6
    ]
    node = ir.Node("", "Attention", inputs=inputs)
    assert _graph_requires_opset24(_graph_with([node])) is True


def test_graph_requires_opset24_attention_without_nonpad() -> None:
    # A plain Attention (no input #6) does not require opset 24.
    inputs = [_make_value("q"), _make_value("k"), _make_value("v")]
    node = ir.Node("", "Attention", inputs=inputs)
    assert _graph_requires_opset24(_graph_with([node])) is False


def test_graph_requires_opset24_standard_ops_only() -> None:
    # Standard ops (Reshape) are safe to lower.
    node = ir.Node("", "Reshape", inputs=[_make_value("x"), _make_value("shape")])
    assert _graph_requires_opset24(_graph_with([node])) is False


def test_graph_requires_opset24_ignores_custom_domain() -> None:
    # Same op names in a non-default domain must not trigger retention.
    node = ir.Node(
        "com.example",
        "TensorScatter",
        inputs=[_make_value("a"), _make_value("b")],
    )
    assert _graph_requires_opset24(_graph_with([node])) is False


def test_lowering_skipped_for_static_cache_graph() -> None:
    # End-to-end of the lowering branch: a static-cache graph keeps opset 24
    # while a standard-op graph is lowered to 23.
    attn_inputs = [
        _make_value("q"),
        _make_value("k"),
        _make_value("v"),
        None,
        None,
        None,
        _make_value("nonpad_kv_seqlen"),
    ]
    static_cache_graph = _graph_with(
        [
            ir.Node(
                "",
                "TensorScatter",
                inputs=[_make_value("p"), _make_value("u"), _make_value("i")],
            ),
            ir.Node("", "Attention", inputs=attn_inputs, num_outputs=1),
        ]
    )
    standard_graph = _graph_with(
        [ir.Node("", "Reshape", inputs=[_make_value("x"), _make_value("shape")])]
    )

    # Mirror the lowering decision in build_from_module.
    for graph, expected in ((static_cache_graph, 24), (standard_graph, 23)):
        if not _graph_requires_opset24(graph):
            graph.opset_imports[""] = 23
        assert graph.opset_imports[""] == expected
