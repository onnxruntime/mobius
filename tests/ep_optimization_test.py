# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for EP-aware optimization pipeline.

Verifies that ``build_from_module`` produces EP-specific fused ops
when ``execution_provider`` is set, and that role gating prevents
decoder-only fusions from applying to vision or embedding models.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
from _test_configs import _base_config

from mobius._builder import build_from_module
from mobius._optimizations import _count_ops, optimize_model, validate_standard_onnx
from mobius._pipeline_contract import declare_arbitrary_attention_mask
from mobius._registry import registry
from mobius._testing import create_test_builder, create_test_input
from mobius.components._scan_utils import create_body_graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llama_pkg(ep: str, dtype: ir.DataType = ir.DataType.FLOAT):
    """Build a tiny Llama model for the given EP and dtype."""
    config = _base_config(dtype=dtype)
    module_cls = registry.get("llama")
    module = module_cls(config)
    return build_from_module(module, config, execution_provider=ep)


def _make_qwen2_pkg(ep: str, dtype: ir.DataType = ir.DataType.FLOAT):
    """Build a tiny Qwen2 model for the given EP and dtype."""
    config = _base_config(dtype=dtype)
    module_cls = registry.get("qwen2")
    module = module_cls(config)
    return build_from_module(module, config, execution_provider=ep)


def _make_qwen2_biased_pkg(ep: str, dtype: ir.DataType = ir.DataType.FLOAT):
    """Build a tiny Qwen2 model with QKV bias enabled (mirrors Qwen2.5)."""
    config = _base_config(dtype=dtype, attn_qkv_bias=True)
    module_cls = registry.get("qwen2")
    module = module_cls(config)
    return build_from_module(module, config, execution_provider=ep)


def _check_op_constraint(
    model: ir.Model,
    op_type: str,
    constraint: str,
    actual: int,
) -> None:
    """Assert an op count constraint. constraint is one of '>0', '==0'."""
    if constraint == ">0":
        assert actual > 0, f"Expected {op_type} count > 0, got {actual}"
    elif constraint == "==0":
        assert actual == 0, f"Expected {op_type} count == 0, got {actual}"
    else:
        raise ValueError(f"Unknown constraint: {constraint!r}")


# ---------------------------------------------------------------------------
# EP fusion expectations table
# ---------------------------------------------------------------------------

# (ep, dtype, op_type, constraint) — decoder model expectations
_EP_FUSION_EXPECTATIONS = [
    # CPU with FLOAT: GQA supported per support matrix, norm fusions apply
    ("cpu", ir.DataType.FLOAT, "GroupQueryAttention", ">0"),
    # CUDA with FLOAT16: GQA + norm fusions
    ("cuda", ir.DataType.FLOAT16, "GroupQueryAttention", ">0"),
    # CUDA with BFLOAT16: GQA supported
    ("cuda", ir.DataType.BFLOAT16, "GroupQueryAttention", ">0"),
    # DML with FLOAT16: GQA supported
    ("dml", ir.DataType.FLOAT16, "GroupQueryAttention", ">0"),
    # WebGPU with FLOAT: GQA supported
    ("webgpu", ir.DataType.FLOAT, "GroupQueryAttention", ">0"),
    # WebGPU with FLOAT16: GQA supported
    ("webgpu", ir.DataType.FLOAT16, "GroupQueryAttention", ">0"),
    # TRT-RTX with FLOAT16: GQA supported
    ("trt-rtx", ir.DataType.FLOAT16, "GroupQueryAttention", ">0"),
    # TRT-RTX with BFLOAT16: GQA supported
    ("trt-rtx", ir.DataType.BFLOAT16, "GroupQueryAttention", ">0"),
    # CPU with FLOAT16: GQA NOT supported (not in support matrix)
    ("cpu", ir.DataType.FLOAT16, "GroupQueryAttention", "==0"),
    # DML with FLOAT: GQA NOT supported
    ("dml", ir.DataType.FLOAT, "GroupQueryAttention", "==0"),
]


@pytest.mark.parametrize("ep,dtype,op_type,constraint", _EP_FUSION_EXPECTATIONS)
def test_ep_produces_expected_ops_llama(ep, dtype, op_type, constraint):
    """Verify EP-specific GQA fusion on Llama (GQA-compatible model)."""
    pkg = _make_llama_pkg(ep=ep, dtype=dtype)
    model = pkg["model"]
    actual = _count_ops(model, op_type)
    _check_op_constraint(model, op_type, constraint, actual)


# ---------------------------------------------------------------------------
# Role gating: GQA must NOT apply to vision/embedding models
# ---------------------------------------------------------------------------


def test_ep_no_gqa_for_vision_encoder():
    """Vision encoder models must not get GQA fusion."""
    vit_cls = registry.get("vit")
    # ViTModel uses ArchitectureConfig; provide image-specific overrides
    vit_config = _base_config(
        hidden_act="gelu",
        image_size=32,
        patch_size=8,
        num_channels=3,
        dtype=ir.DataType.FLOAT16,
    )
    vit_module = vit_cls(vit_config)
    pkg = build_from_module(
        vit_module, vit_config, "image-classification", execution_provider="cuda"
    )
    # Vision encoder uses encoder-style attention — the GQA rewrite pattern
    # does not match. Verify no GQA was produced.
    model = pkg["model"]
    gqa_count = _count_ops(model, "GroupQueryAttention")
    assert gqa_count == 0, (
        f"Vision encoder should not have GQA fusion but found {gqa_count} GQA nodes"
    )


# ---------------------------------------------------------------------------
# CPU default: no EP fusions beyond base cleanup
# ---------------------------------------------------------------------------


def test_cpu_default_produces_attention_not_gqa():
    """CPU with FLOAT16 produces standard Attention, not GroupQueryAttention."""
    pkg = _make_llama_pkg(ep="cpu", dtype=ir.DataType.FLOAT16)
    model = pkg["model"]
    gqa_count = _count_ops(model, "GroupQueryAttention")
    attn_count = _count_ops(model, "Attention")
    assert gqa_count == 0, f"CPU+FLOAT16 should not produce GQA, got {gqa_count}"
    assert attn_count > 0, f"CPU+FLOAT16 should have Attention nodes, got {attn_count}"


def test_default_ep_produces_portable_onnx():
    """build_from_module with no EP arg defaults to 'default' (portable ONNX).

    'default' applies standard fusions (SkipNorm, FusedMatMul, Gelu) but no
    EP-specific ops (no GQA, no PackQKV). Custom ops like FusedMatMul and
    SkipLayerNormalization are kept when fused since their ONNX function bodies
    serve as portable fallbacks. Llama uses RMSNorm (not LayerNorm), so no
    SkipLayerNormalization nodes appear here.
    """
    config = _base_config(dtype=ir.DataType.FLOAT16)
    module_cls = registry.get("llama")
    module = module_cls(config)
    # 'default' EP: no GQA fusion regardless of dtype
    pkg = build_from_module(module, config)
    model = pkg["model"]
    assert _count_ops(model, "GroupQueryAttention") == 0
    # Llama uses RMSNorm; SkipLayerNorm fusion won't match
    assert _count_ops(model, "SkipLayerNormalization") == 0


def test_cuda_float16_has_gqa_llama():
    """Llama on CUDA+FLOAT16 must produce GroupQueryAttention."""
    pkg = _make_llama_pkg(ep="cuda", dtype=ir.DataType.FLOAT16)
    model = pkg["model"]
    gqa_count = _count_ops(model, "GroupQueryAttention")
    assert gqa_count > 0, f"CUDA+FLOAT16 Llama should have GQA, got {gqa_count}"


def test_cuda_float16_has_gqa_qwen2():
    """Qwen2 on CUDA+FLOAT16 must produce GroupQueryAttention."""
    pkg = _make_qwen2_pkg(ep="cuda", dtype=ir.DataType.FLOAT16)
    model = pkg["model"]
    gqa_count = _count_ops(model, "GroupQueryAttention")
    assert gqa_count > 0, f"CUDA+FLOAT16 Qwen2 should have GQA, got {gqa_count}"


def test_arbitrary_mask_contract_disables_decoder_gqa():
    """A suffix-valid mask retains standard Attention even on a GQA-capable EP.

    This is the generic fusion gate used by VibeVoice's negative CFG cache
    branch; the normal CPU float Llama test above still proves GQA fuses.
    """
    model = _make_llama_pkg(ep="default", dtype=ir.DataType.FLOAT)["model"]
    declare_arbitrary_attention_mask(model.graph)

    optimize_model(model, ep="cpu", dtype=ir.DataType.FLOAT, model_role="decoder")

    assert _count_ops(model, "GroupQueryAttention") == 0
    assert _count_ops(model, "Attention") > 0


def test_cuda_float16_qwen2_has_packed_qkv():
    """Qwen2 with QKV bias on CUDA+FLOAT16 has packed QKV (k=None, v=None).

    Qwen2.5 uses attn_qkv_bias=True, so PackQKVWithBiasForGQA must fire.
    The packed projection is Add(MatMul(hidden, packed_w), packed_bias).
    """
    pkg = _make_qwen2_biased_pkg(ep="cuda", dtype=ir.DataType.FLOAT16)
    model = pkg["model"]
    gqa_nodes = [n for n in model.graph if n.op_type == "GroupQueryAttention"]
    assert gqa_nodes, "Expected GQA nodes on CUDA+FLOAT16"
    for gqa in gqa_nodes:
        assert gqa.inputs[1] is None, (
            "Qwen2 GQA should be in packed mode (k=None) after PackQKVWithBiasForGQA"
        )
        assert gqa.inputs[2] is None, (
            "Qwen2 GQA should be in packed mode (v=None) after PackQKVWithBiasForGQA"
        )
        packed_proj = gqa.inputs[0]
        assert packed_proj is not None
        # Packed projection must be Add(MatMul, bias) — not bare MatMul
        producer = packed_proj.producer()
        assert producer is not None and producer.op_type == "Add", (
            "Biased packed QKV should produce Add(MatMul, packed_bias)"
        )


# ---------------------------------------------------------------------------
# Trace optimization
# ---------------------------------------------------------------------------


def test_trace_optimization_produces_output(caplog):
    """trace_optimization=True emits INFO logs with stage headers and rule names."""
    config = _base_config(dtype=ir.DataType.FLOAT)
    module_cls = registry.get("llama")
    module = module_cls(config)

    with caplog.at_level(logging.INFO, logger="mobius._optimizations"):
        build_from_module(module, config, execution_provider="cpu", trace_optimization=True)

    messages = [r.message for r in caplog.records]

    # Header line identifies target/dtype/role
    assert any("[EP Trace] Target:" in m for m in messages), (
        "Expected '[EP Trace] Target:' header in trace output"
    )
    # Stage labels are present
    assert any("Stage 2: Fusion" in m for m in messages), (
        "Expected 'Stage 2: Fusion' in trace output"
    )
    assert any("Stage 3: Lowering" in m for m in messages), (
        "Expected 'Stage 3: Lowering' in trace output"
    )
    # At least one rule name is logged
    assert any("GQAFusion" in m for m in messages), (
        "Expected 'GQAFusion' rule name in trace output (cpu+FLOAT runs GQA fusion)"
    )
    # Summary table is emitted
    assert any("Summary" in m for m in messages), "Expected 'Summary' table in trace output"


def test_trace_optimization_no_matches_shows_zero(caplog):
    """When a rule has no matches, trace output says 'no matches (0 nodes affected)'."""
    # CPU+FLOAT16 skips GQA fusion — GQAFusion should show no matches.
    config = _base_config(dtype=ir.DataType.FLOAT16)
    module_cls = registry.get("llama")
    module = module_cls(config)

    with caplog.at_level(logging.INFO, logger="mobius._optimizations"):
        build_from_module(module, config, execution_provider="cpu", trace_optimization=True)

    messages = [r.message for r in caplog.records]
    # CPU+FLOAT16 has no GQA fusion in the support matrix, so GQAFusion should
    # NOT appear at all in the trace (it's not added to the stage list).
    # But SkipLayerNorm or GeluFusion may show zero matches depending on the model.
    # Verify that the "no matches" format appears at least once (GeluFusion won't match Llama).
    assert any("no matches (0 nodes affected)" in m for m in messages), (
        "Expected at least one 'no matches' entry — GeluFusion should not match Llama (uses SiLU)"
    )


def test_trace_optimization_is_noop_without_flag():
    """Without trace_optimization, build_from_module logs nothing at EP Trace level."""
    import logging as _logging

    config = _base_config(dtype=ir.DataType.FLOAT)
    module_cls = registry.get("llama")
    module = module_cls(config)

    captured: list[str] = []

    class _Capture(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            captured.append(record.getMessage())

    builder_logger = _logging.getLogger("mobius._optimizations")
    handler = _Capture()
    handler.setLevel(_logging.INFO)
    builder_logger.addHandler(handler)
    try:
        build_from_module(module, config, execution_provider="cpu", trace_optimization=False)
    finally:
        builder_logger.removeHandler(handler)

    assert not any("[EP Trace]" in m for m in captured), (
        "Expected no '[EP Trace]' log output when trace_optimization=False"
    )


# ---------------------------------------------------------------------------
# DML / WebGPU lowering pipeline tests
# ---------------------------------------------------------------------------


def test_dml_lowers_rope_and_qkv():
    """DML: after GQA fusion, SeparateRoPE and UnpackQKV lowering must fire."""
    pkg = _make_llama_pkg(ep="dml", dtype=ir.DataType.FLOAT16)
    model = pkg["model"]
    # DML does not support fused RoPE inside GQA — must be separated out.
    # The result is standard RotaryEmbedding nodes and split Q/K/V MatMuls.
    gqa_count = _count_ops(model, "GroupQueryAttention")
    rope_count = _count_ops(model, "RotaryEmbedding")
    assert gqa_count > 0, f"DML+FLOAT16 should still have GQA, got {gqa_count}"
    assert rope_count > 0, (
        f"DML SeparateRoPE should add RotaryEmbedding nodes, got {rope_count}"
    )
    # DML has qkv_pack_dtypes=frozenset() — PackQKV must not fire.
    pack_qkv_count = _count_ops(model, "PackQKV")
    assert pack_qkv_count == 0, (
        f"DML has qkv_pack_dtypes=frozenset(), so PackQKV must not be present, "
        f"got {pack_qkv_count}"
    )


def test_webgpu_supports_shape_nodes():
    """WebGPU: Shape ops are now supported natively — graph must retain them."""
    pkg = _make_llama_pkg(ep="webgpu", dtype=ir.DataType.FLOAT16)
    model = pkg["model"]
    shape_count = _count_ops(model, "Shape")
    assert shape_count > 0, (
        f"WebGPU graph should still contain Shape nodes (Shape is now supported), "
        f"got {shape_count}"
    )


# ---------------------------------------------------------------------------
# Encoder / embedding role gating
# ---------------------------------------------------------------------------


def test_encoder_role_no_gqa():
    """model_role='encoder' must not receive GQA fusion even on cuda+FLOAT16."""
    # Build with 'default' EP (no vendor fusions) to get a clean baseline model.
    config = _base_config(dtype=ir.DataType.FLOAT16)
    module_cls = registry.get("llama")
    module = module_cls(config)
    pkg = build_from_module(module, config, execution_provider="default")
    model = pkg["model"]
    assert _count_ops(model, "GroupQueryAttention") == 0, (
        "baseline default EP should have no GQA"
    )

    # Now re-optimize as encoder role with cuda — GQA should still not be added.
    optimize_model(model, ep="cuda", dtype=ir.DataType.FLOAT16, model_role="encoder")
    gqa_count = _count_ops(model, "GroupQueryAttention")
    assert gqa_count == 0, (
        f"encoder role should produce no GQA even on cuda+FLOAT16, got {gqa_count}"
    )


# ---------------------------------------------------------------------------
# Unknown EP error handling
# ---------------------------------------------------------------------------


def test_unknown_ep_raises():
    """Passing an unrecognised EP string must raise ValueError immediately."""
    config = _base_config(dtype=ir.DataType.FLOAT)
    module_cls = registry.get("llama")
    module = module_cls(config)
    with pytest.raises(ValueError, match="Unknown execution provider"):
        build_from_module(module, config, execution_provider="nonexistent-ep")


# ---------------------------------------------------------------------------
# onnx-standard EP: zero custom-domain nodes
# ---------------------------------------------------------------------------


_STANDARD_ONNX_DOMAINS = frozenset({"", "ai.onnx", "ai.onnx.ml"})


@pytest.mark.parametrize("location", ["main", "If", "Scan", "graphs", "signature", "body"])
@pytest.mark.parametrize(
    "domain,operator,overload,is_standard",
    [
        ("test.custom", "Leak", "", False),
        ("", "NotAnOnnxOp", "", False),
        ("", "Identity", "custom", False),
        ("ai.onnx.ml", "Normalizer", "", True),
        ("ai.onnx.ml", "NotAnOnnxOp", "", False),
        ("ai.onnx.ml", "Identity", "", False),
        ("ai.onnx.ml", "Normalizer", "custom", False),
    ],
)
def test_onnx_standard_validator_checks_recursive_identifiers(
    location, domain, operator, overload, is_standard
):
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1])
    graph.opset_imports["ai.onnx.ml"] = 1

    def emit(op_builder, op_type, op_domain, op_overload):
        value = getattr(op_builder, op_type)(x, _domain=op_domain)
        producer = value.producer()
        assert producer is not None
        producer.overload = op_overload
        return value

    if location == "main":
        graph.outputs.append(emit(op, operator, domain, overload))
    elif location in {"If", "Scan", "graphs"}:
        body, body_builder = create_body_graph([], [])
        body.outputs.append(emit(body_builder.op, operator, domain, overload))
        if location == "If":
            node = ir.Node(
                "",
                "If",
                [x],
                attributes=[
                    ir.AttrGraph("then_branch", body),
                    ir.AttrGraph("else_branch", body),
                ],
                num_outputs=1,
            )
        elif location == "Scan":
            node = ir.Node(
                "",
                "Scan",
                [x],
                attributes=[ir.AttrGraph("body", body), ir.AttrInt64("num_scan_inputs", 1)],
                num_outputs=1,
            )
        else:
            node = ir.Node(
                "",
                "Identity",
                [x],
                attributes=[ir.AttrGraphs("nested", [body])],
                num_outputs=1,
            )
        graph.append(node)
        graph.outputs.append(node.outputs[0])
    else:
        graph.outputs.append(op.Identity(x))
    model = ir.Model(graph, ir_version=11)
    if location in {"signature", "body"}:
        body, body_builder = create_body_graph([], [])
        body.opset_imports["ai.onnx.ml"] = 1
        body.outputs.append(
            emit(
                body_builder.op,
                operator if location == "body" else "Identity",
                domain if location == "body" else "",
                overload if location == "body" else "",
            )
        )
        function = ir.Function(
            domain if location == "signature" else "",
            operator if location == "signature" else "Identity",
            overload=overload if location == "signature" else "",
            graph=body,
            attributes={},
        )
        model.functions[function.identifier()] = function
    if is_standard:
        validate_standard_onnx(model, component="decoder")
        return
    with pytest.raises(
        ValueError,
        match=rf"component 'decoder'.*{operator!r}.*overload={overload!r}.*decomposition",
    ):
        validate_standard_onnx(model, component="decoder")


def test_onnx_standard_ml_export_roundtrip(tmp_path: Path):
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1, 2])
    y = op.Normalizer(x, norm="L2", _domain="ai.onnx.ml")
    y.type, y.shape = x.type, x.shape
    graph.outputs.append(y)
    graph.opset_imports["ai.onnx.ml"] = 1
    model = ir.Model(graph, ir_version=10)
    optimize_model(model, ep="onnx-standard")
    assert any(node.domain == "ai.onnx.ml" for node in model.graph)

    path = tmp_path / "normalizer.onnx"
    try:
        ir.save(model, path)
        onnx.checker.check_model(str(path), full_check=True)
        for candidate in (model, ir.load(path)):
            validate_standard_onnx(candidate)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        (actual,) = session.run(None, {"x": np.array([[3, 4]], dtype=np.float32)})
        np.testing.assert_allclose(
            actual, np.array([[0.6, 0.8]], dtype=np.float32), rtol=1e-6, atol=1e-7
        )
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("include_unused_ml_import", [False, True])
def test_onnx_standard_retained_ml_function(tmp_path: Path, include_unused_ml_import):
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1, 2])
    y = op.Binarizer(x, _domain="ai.onnx.ml")
    y.type, y.shape = x.type, x.shape
    graph.outputs.append(y)
    graph.opset_imports["ai.onnx.ml"] = 1
    model = ir.Model(graph, ir_version=10)

    body_builder, body_op, body = create_test_builder()
    body_x = create_test_input(body_builder, "body_x", [1, 2])
    zero = body_op.Constant(value_float=0.0)
    body_y = body_op.Cast(body_op.Greater(body_x, zero), to=ir.DataType.FLOAT)
    body_y.type, body_y.shape = body_x.type, body_x.shape
    body.outputs.append(body_y)
    if include_unused_ml_import:
        body.opset_imports["ai.onnx.ml"] = 1
    function = ir.Function("ai.onnx.ml", "Binarizer", graph=body, attributes={})
    model.functions[function.identifier()] = function

    validate_standard_onnx(model)
    optimize_model(model, ep="onnx-standard")
    assert function.identifier() in model.functions
    assert "ai.onnx.ml" not in function.opset_imports
    validate_standard_onnx(model)
    path = tmp_path / "ml_function.onnx"
    try:
        ir.save(model, path)
        onnx.checker.check_model(str(path), full_check=True)
        validate_standard_onnx(ir.load(path))
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("location", ["main", "signature", "body"])
@pytest.mark.parametrize("ml_opset", [None, 0])
def test_onnx_standard_ml_requires_own_opset(location, ml_opset):
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1, 2])
    graph.outputs.append(
        op.Normalizer(x, _domain="ai.onnx.ml") if location == "main" else op.Identity(x)
    )
    model = ir.Model(graph, ir_version=11)
    target_graph = graph
    if location != "main":
        body_builder, body_op, body = create_test_builder()
        body_x = create_test_input(body_builder, "body_x", [1, 2])
        body.outputs.append(
            body_op.Normalizer(body_x, _domain="ai.onnx.ml")
            if location == "body"
            else body_op.Identity(body_x)
        )
        function = ir.Function(
            "ai.onnx.ml" if location == "signature" else "",
            "Normalizer" if location == "signature" else "Identity",
            graph=body,
            attributes={},
        )
        model.functions[function.identifier()] = function
        target_graph = graph if location == "signature" else body
    target_graph.opset_imports.pop("ai.onnx.ml", None)
    if ml_opset is not None:
        target_graph.opset_imports["ai.onnx.ml"] = ml_opset
    with pytest.raises(ValueError, match=rf"ai.onnx.ml.*opset={ml_opset!r}.*no standard"):
        validate_standard_onnx(model)


@pytest.mark.parametrize("domain,import_domain", [("", "ai.onnx"), ("ai.onnx", "")])
def test_onnx_standard_core_domain_aliases(domain, import_domain):
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1])
    graph.outputs.append(op.Identity(x, _domain=domain))
    graph.opset_imports.clear()
    graph.opset_imports[import_domain] = 24
    validate_standard_onnx(ir.Model(graph, ir_version=11))


@pytest.mark.parametrize("location", ["call", "definition"])
def test_onnx_standard_validator_rejects_declared_overloads_after_roundtrip(
    tmp_path: Path, location
):
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1])
    output = op.Identity(x)
    output.type, output.shape = x.type, x.shape
    node = output.producer()
    assert node is not None
    node.overload = "custom" if location == "call" else ""
    graph.outputs.append(output)
    function_builder, function_op, function_graph = create_test_builder()
    function_x = create_test_input(function_builder, "function_x", [1])
    function_y = function_op.Neg(function_x)
    function_y.type, function_y.shape = function_x.type, function_x.shape
    function_graph.outputs.append(function_y)
    function = ir.Function(
        "", "Identity", overload="custom", graph=function_graph, attributes={}
    )
    model = ir.Model(graph, ir_version=11)
    model.functions[function.identifier()] = function
    path = tmp_path / "overloaded.onnx"
    expected_path = "graph/" if location == "call" else "function/::Identity"
    try:
        ir.save(model, path)
        for candidate in (model, ir.load(path)):
            with pytest.raises(
                ValueError, match=rf"{expected_path}.*overload='custom'.*no standard"
            ):
                validate_standard_onnx(candidate)
    finally:
        path.unlink(missing_ok=True)


def test_onnx_standard_optimizer_rejects_unresolved_custom_node():
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1])
    graph.outputs.append(op.Leak(x, _domain="test.custom"))
    model = ir.Model(graph, ir_version=11)
    with pytest.raises(ValueError, match=r"test.custom.*Leak"):
        optimize_model(model, ep="onnx-standard")


def test_onnx_standard_validates_after_opset_lowering(monkeypatch):
    def insert_leak(package, _ep):
        model = package["model"]
        output = model.graph.outputs[0]
        node = ir.Node("test.custom", "AfterLowering", [output], num_outputs=1)
        model.graph.append(node)
        model.graph.outputs[0] = node.outputs[0]

    monkeypatch.setattr("mobius._builder._maybe_apply_opset_lowering", insert_leak)
    with pytest.raises(ValueError, match=r"component 'model'.*AfterLowering"):
        _make_llama_pkg("onnx-standard")


def test_onnx_standard_validator_rejects_inapplicable_schema_version():
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1])
    graph.outputs.append(op.Identity(x))
    graph.opset_imports[""] = 0
    with pytest.raises(ValueError, match=r"opset=0.*no standard"):
        validate_standard_onnx(ir.Model(graph, ir_version=11))


def _make_nested_swish_model(parent_opset, child_opset, location):
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1]))

    def make_body(name):
        body_input = (
            ir.Value(name=f"{name}_x", type=x.type, shape=ir.Shape([]))
            if location == "Scan"
            else x
        )
        output = ir.Value(name=f"{name}_y", type=x.type, shape=body_input.shape)
        return ir.Graph(
            inputs=[body_input] if location == "Scan" else [],
            outputs=[output],
            nodes=[ir.Node("", "Swish", [body_input], outputs=[output])],
            opset_imports={} if child_opset is None else {"": child_opset},
            name=name,
        )

    inputs = [x]
    if location == "If":
        condition = ir.Value(
            name="condition", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
        )
        inputs.append(condition)
        node_inputs = [condition]
        attributes = [
            ir.AttrGraph("then_branch", make_body("then")),
            ir.AttrGraph("else_branch", make_body("else")),
        ]
    elif location == "Scan":
        node_inputs = [x]
        attributes = [
            ir.AttrGraph("body", make_body("scan")),
            ir.AttrInt64("num_scan_inputs", 1),
        ]
    else:
        node_inputs = [x]
        # Synthetic GRAPHS attribute exercises the validator's repeated-graph traversal.
        attributes = [ir.AttrGraphs("nested", [make_body("first"), make_body("second")])]
    output = ir.Value(name="y", type=x.type, shape=x.shape)
    node = ir.Node(
        "",
        "Identity" if location == "graphs" else location,
        node_inputs,
        attributes=attributes,
        outputs=[output],
    )
    return ir.Model(
        ir.Graph(
            inputs=inputs,
            outputs=[output],
            nodes=[node],
            opset_imports={"": parent_opset},
            name="parent",
        ),
        ir_version=11,
    )


@pytest.mark.parametrize("location", ["If", "Scan", "graphs"])
@pytest.mark.parametrize("parent_opset,child_opset", [(23, 24), (24, 23), (24, None)])
def test_onnx_standard_validator_subgraphs_inherit_parent_opset(
    location, parent_opset, child_opset
):
    model = _make_nested_swish_model(parent_opset, child_opset, location)
    if parent_opset == 23:
        with pytest.raises(ValueError, match=r"Swish.*opset=23.*no standard"):
            validate_standard_onnx(model)
    else:
        validate_standard_onnx(model)


@pytest.mark.parametrize(
    "model_opset,function_opset,child_opset",
    [(23, 24, 23), (24, 23, 24), (23, 24, None)],
)
def test_onnx_standard_validator_function_subgraphs_inherit_function_opset(
    model_opset, function_opset, child_opset
):
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1])
    graph.outputs.append(op.Identity(x))
    graph.opset_imports[""] = model_opset
    model = ir.Model(graph, ir_version=11)
    function = ir.Function(
        "",
        "Identity",
        graph=_make_nested_swish_model(function_opset, child_opset, "Scan").graph,
        attributes={},
    )
    model.functions[function.identifier()] = function
    if function_opset == 23:
        with pytest.raises(ValueError, match=r"function/::Identity.*Swish.*opset=23"):
            validate_standard_onnx(model)
    else:
        validate_standard_onnx(model)


@pytest.mark.parametrize("parent_opset", [23, 24])
def test_onnx_standard_validator_subgraph_opset_matches_roundtrip(
    tmp_path: Path, parent_opset
):
    model = _make_nested_swish_model(parent_opset, 24, "If")
    path = tmp_path / "nested.onnx"
    try:
        ir.save(model, path)
        for candidate in (model, ir.load(path)):
            if parent_opset == 23:
                with pytest.raises(ValueError, match=r"Swish.*opset=23.*no standard"):
                    validate_standard_onnx(candidate)
            else:
                validate_standard_onnx(candidate)
    finally:
        path.unlink(missing_ok=True)


def test_onnx_standard_validates_after_transformers_weight_binding(monkeypatch):
    from types import SimpleNamespace

    from mobius.integrations.transformers import _builder as transformers_builder
    from mobius.integrations.transformers import _config_resolver

    config = _base_config()
    package = _make_llama_pkg("onnx-standard")
    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (SimpleNamespace(model_type="llama"), False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (registry.get("llama"), "text-generation", "llama"),
    )
    monkeypatch.setattr(_config_resolver, "_config_from_hf", lambda *args, **kwargs: config)
    monkeypatch.setattr(transformers_builder, "build_from_module", lambda *a, **k: package)
    monkeypatch.setattr(transformers_builder, "_download_weights", lambda *a, **k: {})

    def bind(_self, *args, **kwargs):
        model = package["model"]
        node = ir.Node("test.custom", "AfterWeights", [model.graph.outputs[0]], num_outputs=1)
        model.graph.append(node)
        model.graph.outputs[0] = node.outputs[0]

    monkeypatch.setattr(type(package), "apply_weights", bind)
    with pytest.raises(ValueError, match=r"component 'model'.*AfterWeights"):
        transformers_builder.build_transformers_model(
            "local-synthetic-llama", execution_provider="onnx-standard"
        )


def _non_standard_nodes(model: ir.Model) -> list[tuple[str, str]]:
    """Return (domain, op_type) pairs for all nodes with non-standard domains."""
    return [
        (node.domain, node.op_type)
        for node in model.graph.all_nodes()
        if node.domain not in _STANDARD_ONNX_DOMAINS
    ]


def test_onnx_standard_ep_llama_has_no_custom_domain_nodes():
    """'onnx-standard' EP must produce zero nodes from non-standard domains on Llama.

    Verifies that FusedMatMul (com.microsoft) is expanded via InlinePass to
    standard Transpose+MatMul, and that no GQA/PackQKV/SkipLayerNorm ops
    are added during optimization.
    """
    pkg = _make_llama_pkg(ep="onnx-standard", dtype=ir.DataType.FLOAT)
    model = pkg["model"]
    bad = _non_standard_nodes(model)
    assert not bad, (
        f"'onnx-standard' EP produced {len(bad)} non-standard-domain node(s): {bad}"
    )


def test_onnx_standard_ep_qwen2_has_no_custom_domain_nodes():
    """'onnx-standard' EP must produce zero non-standard-domain nodes on Qwen2."""
    pkg = _make_qwen2_pkg(ep="onnx-standard", dtype=ir.DataType.FLOAT)
    model = pkg["model"]
    bad = _non_standard_nodes(model)
    assert not bad, (
        f"'onnx-standard' EP produced {len(bad)} non-standard-domain node(s): {bad}"
    )


def test_onnx_standard_ep_qwen2_biased_has_no_custom_domain_nodes():
    """'onnx-standard' EP must produce zero non-standard-domain nodes on Qwen2 with QKV bias."""
    pkg = _make_qwen2_biased_pkg(ep="onnx-standard", dtype=ir.DataType.FLOAT)
    model = pkg["model"]
    bad = _non_standard_nodes(model)
    assert not bad, (
        f"'onnx-standard' EP produced {len(bad)} non-standard-domain node(s): {bad}"
    )


def test_onnx_standard_ep_has_matmul_not_fused():
    """'onnx-standard' EP must expand FusedMatMul to standard MatMul+Transpose.

    FusedMatMul is the primary com.microsoft op emitted by Linear components.
    After onnx-standard EP optimization, all MatMul ops should be standard.
    """
    pkg = _make_llama_pkg(ep="onnx-standard", dtype=ir.DataType.FLOAT)
    model = pkg["model"]
    fused_mm_count = _count_ops(model, "FusedMatMul")
    matmul_count = _count_ops(model, "MatMul")
    assert fused_mm_count == 0, (
        f"'onnx-standard' EP should have no FusedMatMul, got {fused_mm_count}"
    )
    assert matmul_count > 0, (
        f"'onnx-standard' EP should have MatMul nodes after FusedMatMul expansion, got {matmul_count}"
    )


def test_onnx_standard_ep_has_no_gqa():
    """'onnx-standard' EP must not produce GroupQueryAttention."""
    pkg = _make_llama_pkg(ep="onnx-standard", dtype=ir.DataType.FLOAT)
    model = pkg["model"]
    gqa_count = _count_ops(model, "GroupQueryAttention")
    assert gqa_count == 0, f"'onnx-standard' EP should have no GQA, got {gqa_count}"


def test_optimize_model_eliminates_weight_transpose_after_weights():
    """fold_initializers_after_weights folds Transpose(weight) nodes once weights are set.

    Linear emits Transpose(weight, perm=[1,0]) → MatMul.  After the fold pass
    runs (post-weight-load), no Transpose nodes should target initializers —
    they are replaced by pre-transposed weight_t initializers.
    """
    import numpy as np

    from mobius._optimizations import fold_initializers_after_weights

    pkg = _make_llama_pkg(ep="cpu", dtype=ir.DataType.FLOAT)
    model = pkg["model"]

    # Initialise all uninitialised parameters so the fold can materialise.
    for init in model.graph.initializers.values():
        if init.const_value is None:
            shape = [int(d) for d in init.shape]
            init.const_value = ir.Tensor(np.random.randn(*shape).astype(np.float32))

    fold_initializers_after_weights(model)

    remaining_transpose = sum(
        1
        for node in model.graph.all_nodes()
        if node.op_type == "Transpose"
        and node.inputs[0] is not None
        and node.inputs[0].name in model.graph.initializers
        and list(node.attributes["perm"].value) == [1, 0]
    )
    assert remaining_transpose == 0, (
        f"fold_initializers_after_weights should fold all Transpose(initializer) nodes, "
        f"{remaining_transpose} remain"
    )
    transposed_inits = [k for k in model.graph.initializers if k.endswith("_t")]
    assert len(transposed_inits) > 0, "Expected pre-transposed weight_t initializers"


def test_optimize_model_cuda_eliminates_qkv_concat_after_weights():
    """fold_initializers_after_weights folds PackQKV Concat+Transpose after weights are set.

    On CUDA EP, PackQKV rewrites Q/K/V projections into Concat(W_q,W_k,W_v) →
    Transpose → MatMul.  After fold_initializers_after_weights, no Concat nodes
    should remain over initializers.
    """
    import numpy as np

    from mobius._optimizations import fold_initializers_after_weights

    pkg = _make_llama_pkg(ep="cuda", dtype=ir.DataType.FLOAT)
    model = pkg["model"]

    for init in model.graph.initializers.values():
        if init.const_value is None:
            shape = [int(d) for d in init.shape]
            init.const_value = ir.Tensor(np.random.randn(*shape).astype(np.float32))

    fold_initializers_after_weights(model)

    remaining_concat = sum(
        1
        for node in model.graph.all_nodes()
        if node.op_type == "Concat"
        and all(
            inp is not None and inp.name in model.graph.initializers
            for inp in node.inputs
            if inp is not None
        )
    )
    assert remaining_concat == 0, (
        f"fold_initializers_after_weights should fold all Concat(initializer, …) nodes, "
        f"{remaining_concat} remain"
    )
