# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for :mod:`mobius._builder` opset-lowering logic.

These tests are CPU-authorable: they construct tiny ONNX-IR graphs directly
(no weights, no network) and exercise the conditional opset 24→23 lowering
that preserves the static-cache Flash-attention path.
"""

from __future__ import annotations

import onnx_ir as ir
import pytest

from mobius._builder import (
    _enable_prefill_prefix_pruning_task,
    _graph_requires_opset24,
    _maybe_apply_opset_lowering,
    build,
    flags,
)
from mobius._model_package import ModelPackage


def _make_value(name: str) -> ir.Value:
    return ir.Value(name=name)


def _graph_with(nodes: list[ir.Node], *, opset: int = 24) -> ir.Graph:
    return ir.Graph(inputs=[], outputs=[], nodes=nodes, opset_imports={"": opset})


def _model_with(nodes: list[ir.Node], *, opset: int = 24) -> ir.Model:
    return ir.Model(_graph_with(nodes, opset=opset), ir_version=10)


def _attention_nonpad_inputs() -> list[ir.Value | None]:
    # q, k, v, attn_mask, past_key, past_value, nonpad_kv_seqlen (input #6).
    return [
        _make_value("q"),
        _make_value("k"),
        _make_value("v"),
        None,
        None,
        None,
        _make_value("nonpad_kv_seqlen"),
    ]


def _static_cache_nodes() -> list[ir.Node]:
    return [
        ir.Node(
            "",
            "TensorScatter",
            inputs=[_make_value("p"), _make_value("u"), _make_value("i")],
        ),
        ir.Node("", "Attention", inputs=_attention_nonpad_inputs(), num_outputs=1),
    ]


def _standard_nodes() -> list[ir.Node]:
    return [ir.Node("", "Reshape", inputs=[_make_value("x"), _make_value("shape")])]


def test_prefill_prefix_pruning_error_lists_supported_tasks() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "text-generation, hybrid-text-generation, gemma4-text-generation, and gemma4 tasks"
        ),
    ):
        _enable_prefill_prefix_pruning_task("feature-extraction")


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
    node = ir.Node("", "Attention", inputs=_attention_nonpad_inputs())
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


def test_graph_requires_opset24_detects_nested_subgraph() -> None:
    # An opset-24-only op buried inside an If/Loop/Scan body must still be
    # detected: the scan is recursive (RecursiveGraphIterator).
    then_branch = _graph_with([ir.Node("", "TensorScatter", inputs=[_make_value("a")])])
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
    outer = _graph_with([if_node])
    assert _graph_requires_opset24(outer) is True


def test_maybe_apply_opset_lowering_mixed_package(monkeypatch: pytest.MonkeyPatch) -> None:
    # Drive the REAL lowering branch in build_from_module via _maybe_apply_opset_lowering.
    # A mixed package must be handled per sub-model: the static-cache sub-model
    # keeps opset 24, the standard sub-model is lowered to 23.
    monkeypatch.setattr(flags, "ort_lower_opset_for_ep", True)
    pkg = ModelPackage(
        {
            "model": _model_with(_static_cache_nodes()),
            "embedding": _model_with(_standard_nodes()),
        }
    )

    _maybe_apply_opset_lowering(pkg, execution_provider="cuda")

    assert pkg["model"].graph.opset_imports[""] == 24
    assert pkg["embedding"].graph.opset_imports[""] == 23


def test_maybe_apply_opset_lowering_skipped_for_default_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The "default" EP gate: lowering never fires even with the flag enabled.
    monkeypatch.setattr(flags, "ort_lower_opset_for_ep", True)
    pkg = ModelPackage({"embedding": _model_with(_standard_nodes())})

    _maybe_apply_opset_lowering(pkg, execution_provider="default")

    assert pkg["embedding"].graph.opset_imports[""] == 24


def test_maybe_apply_opset_lowering_skipped_for_cpu_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The "cpu" EP gate: the CPU EP already registers opset-24 kernels, so
    # lowering never fires for it even with the flag enabled (mirrors the
    # inference-side ort_inference._should_lower_opset CPU skip).
    monkeypatch.setattr(flags, "ort_lower_opset_for_ep", True)
    pkg = ModelPackage({"embedding": _model_with(_standard_nodes())})

    _maybe_apply_opset_lowering(pkg, execution_provider="cpu")

    assert pkg["embedding"].graph.opset_imports[""] == 24


def test_maybe_apply_opset_lowering_skipped_when_flag_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The flag gate: lowering never fires when the flag is off (the default).
    monkeypatch.setattr(flags, "ort_lower_opset_for_ep", False)
    pkg = ModelPackage({"embedding": _model_with(_standard_nodes())})

    _maybe_apply_opset_lowering(pkg, execution_provider="cuda")

    assert pkg["embedding"].graph.opset_imports[""] == 24


def test_build_threads_revision_to_diffusers_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    import mobius._config_resolver as config_resolver
    import mobius._diffusers_builder as diffusers_builder

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("not transformers")),
    )
    monkeypatch.setattr(config_resolver, "_try_load_config_json", lambda *args, **kwargs: None)
    expected = ModelPackage({})
    calls: list[tuple[tuple, dict]] = []

    def fake_build_diffusers(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(
        diffusers_builder,
        "build_diffusers_pipeline",
        fake_build_diffusers,
    )

    result = build(
        "fake/diffusers",
        revision="pinned-revision",
        load_weights=False,
    )

    assert result is expected
    assert calls == [
        (
            ("fake/diffusers",),
            {
                "revision": "pinned-revision",
                "dtype": None,
                "load_weights": False,
                "execution_provider": "default",
            },
        )
    ]
