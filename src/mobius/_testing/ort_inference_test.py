# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for :func:`mobius._testing.ort_inference._should_lower_opset`.

CPU-authorable: construct tiny ONNX-IR models directly and assert the opset
24→23 lowering decision.  In particular these prove the M2 alignment — the
decision now delegates to ``_builder._graph_requires_opset24``, so an
opset-24-only op (TensorScatter, or Attention with a non-empty input #6
``nonpad_kv_seqlen``) buried inside a control-flow subgraph blocks lowering,
which the previous top-level-only scan missed.
"""

from __future__ import annotations

import tempfile

import onnx_ir as ir
import pytest

from mobius._flags import flags
from mobius._testing.ort_inference import OnnxModelSession, _MAX_EP_OPSET, _should_lower_opset


def _make_value(name: str) -> ir.Value:
    return ir.Value(name=name)


def _model_with(graph: ir.Graph) -> ir.Model:
    return ir.Model(graph, ir_version=10)


def _graph_with(nodes: list[ir.Node], *, opset: int = 24) -> ir.Graph:
    return ir.Graph(inputs=[], outputs=[], nodes=nodes, opset_imports={"": opset})


def _attention_nonpad_node() -> ir.Node:
    # q, k, v, attn_mask, past_key, past_value, nonpad_kv_seqlen (input #6).
    inputs = [
        _make_value("q"),
        _make_value("k"),
        _make_value("v"),
        None,
        None,
        None,
        _make_value("nonpad_kv_seqlen"),
    ]
    return ir.Node("", "Attention", inputs=inputs, num_outputs=1)


@pytest.fixture
def lowering_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flags, "ort_lower_opset_for_ep", True)


def test_nested_attention_nonpad_blocks_lowering(lowering_enabled: None) -> None:
    # M2: an Attention(nonpad_kv_seqlen, input #6) buried inside an If body must
    # block lowering. The old top-level-only scan missed both the recursion and
    # the input-#6 check, so it would have (wrongly) returned True here.
    then_branch = _graph_with([_attention_nonpad_node()])
    else_branch = _graph_with([ir.Node("", "Identity", inputs=[_make_value("b")])])
    if_node = ir.Node(
        "",
        "If",
        inputs=[_make_value("cond")],
        attributes=[
            ir.AttrGraph("then_branch", then_branch),
            ir.AttrGraph("else_branch", else_branch),
        ],
    )
    model = _model_with(_graph_with([if_node]))

    assert _should_lower_opset(model, device="cuda") is False


def test_top_level_tensorscatter_blocks_lowering(lowering_enabled: None) -> None:
    node = ir.Node("", "TensorScatter", inputs=[_make_value("p"), _make_value("u")])
    model = _model_with(_graph_with([node]))

    assert _should_lower_opset(model, device="cuda") is False


def test_standard_opset24_graph_allows_lowering(lowering_enabled: None) -> None:
    # A plain opset-24 graph with no opset-24-only semantics is safe to lower.
    node = ir.Node("", "Reshape", inputs=[_make_value("x"), _make_value("shape")])
    model = _model_with(_graph_with([node]))

    assert _should_lower_opset(model, device="cuda") is True


def test_cpu_device_never_lowers(lowering_enabled: None) -> None:
    # The CPU EP already has opset-24 kernels, so lowering is skipped for it.
    node = ir.Node("", "Reshape", inputs=[_make_value("x"), _make_value("shape")])
    model = _model_with(_graph_with([node]))

    assert _should_lower_opset(model, device="cpu") is False


def test_flag_disabled_never_lowers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flags, "ort_lower_opset_for_ep", False)
    node = ir.Node("", "Reshape", inputs=[_make_value("x"), _make_value("shape")])
    model = _model_with(_graph_with([node]))

    assert _should_lower_opset(model, device="cuda") is False


def test_opset_at_or_below_max_not_lowered(lowering_enabled: None) -> None:
    # Already at the EP-max opset: nothing to lower.
    node = ir.Node("", "Reshape", inputs=[_make_value("x"), _make_value("shape")])
    model = _model_with(_graph_with([node], opset=_MAX_EP_OPSET))

    assert _should_lower_opset(model, device="cuda") is False


def test_close_releases_session_reference() -> None:
    session = OnnxModelSession.__new__(OnnxModelSession)
    session._session = object()  # type: ignore[attr-defined]
    session._tmpdir = tempfile.TemporaryDirectory()  # type: ignore[attr-defined]

    session.close()

    assert not hasattr(session, "_session")
