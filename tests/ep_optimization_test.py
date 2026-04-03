# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for EP-aware optimization pipeline.

Verifies that ``build_from_module`` produces EP-specific fused ops
when ``execution_provider`` is set, and that role gating prevents
decoder-only fusions from applying to vision or embedding models.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import onnx_ir as ir
import pytest
from _test_configs import _base_config

from mobius._builder import build_from_module
from mobius._optimizations import _count_ops, optimize_model
from mobius._registry import registry

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

    'default' applies only cleanup + constant folding. No GQA fusion.
    Custom ops (SkipLayerNorm, etc.) are kept when present since their
    ONNX function bodies serve as portable fallbacks. Llama uses RMSNorm
    (not LayerNorm), so no SkipLayerNormalization nodes appear here.
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
    # But SkipLayerNorm or BiasGelu may show zero matches depending on the model.
    # Verify that the "no matches" format appears at least once (BiasGelu won't match Llama).
    assert any("no matches (0 nodes affected)" in m for m in messages), (
        "Expected at least one 'no matches' entry — BiasGelu should not match Llama (uses SiLU)"
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


def test_webgpu_no_shape_nodes():
    """WebGPU: EliminateShape lowering must remove Shape ops from the graph."""
    pkg = _make_llama_pkg(ep="webgpu", dtype=ir.DataType.FLOAT16)
    model = pkg["model"]
    shape_count = _count_ops(model, "Shape")
    assert shape_count == 0, (
        f"WebGPU should have no Shape nodes after lowering, got {shape_count}"
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
