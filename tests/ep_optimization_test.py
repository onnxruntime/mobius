# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for EP-aware optimization pipeline.

Verifies that ``build_from_module`` produces EP-specific fused ops
when ``execution_provider`` is set, and that role gating prevents
decoder-only fusions from applying to vision or embedding models.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import pytest
import onnx_ir as ir
from _test_configs import _base_config, TINY_HIDDEN, TINY_HEADS, TINY_KV_HEADS

from mobius._builder import _count_ops, build_from_module
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


def test_default_ep_is_cpu():
    """build_from_module with no EP arg defaults to cpu (no behavioral change)."""
    config = _base_config(dtype=ir.DataType.FLOAT16)
    module_cls = registry.get("llama")
    module = module_cls(config)
    # Should not raise and should behave like ep="cpu"
    pkg = build_from_module(module, config)
    model = pkg["model"]
    # cpu + FLOAT16 → no GQA
    assert _count_ops(model, "GroupQueryAttention") == 0


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
