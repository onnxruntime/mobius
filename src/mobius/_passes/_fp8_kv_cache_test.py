# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the FP8 KV-cache pass (:mod:`mobius._passes._fp8_kv_cache`).

Two layers of coverage:

* **Structural (no GPU):** build a tiny GQA decoder on the CUDA EP with
  ``fp8_kv_cache=True`` and assert the exported graph types every
  ``past_key_values`` input and ``present`` output as ``FLOAT8E4M3FN``, adds
  per-layer ``k_scale`` / ``v_scale`` initializers at GQA input slots 12/13,
  and sets the quantization attributes.
* **Runtime (CUDA only):** apply the pass to a hand-built ``GroupQueryAttention``
  graph and execute it on ``CUDAExecutionProvider`` to prove the emitted op
  signature is accepted by ORT's FP8 KV-cache kernel.
"""

from __future__ import annotations

import json
import sys

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

sys.path.insert(0, "tests")

from _test_configs import _base_config

from mobius import build_from_module
from mobius._passes._fp8_kv_cache import (
    Fp8KvCachePass,
    load_kv_cache_scale_file,
)
from mobius._registry import registry

_FP8 = ir.DataType.FLOAT8E4M3FN
_CUDA = "CUDAExecutionProvider"
_HAS_CUDA = _CUDA in ort.get_available_providers()


def _cuda_supports_fp8() -> bool:
    """Best-effort check that the CUDA device has an FP8 GQA kernel (SM89+).

    Conservative: returns ``False`` when the compute capability cannot be
    determined (e.g. torch unavailable), so the runtime test is skipped rather
    than run on hardware without the FP8 kernel (e.g. T4 / SM75).
    """
    if not _HAS_CUDA:
        return False
    try:
        import torch

        return torch.cuda.get_device_capability() >= (8, 9)
    except Exception:
        return False


_FP8_CUDA = _cuda_supports_fp8()


def _build_fp8_decoder(fp8_kv_cache=True, kv_cache_scales=None):
    """Build a tiny fp16 qwen2 decoder on the CUDA EP with the FP8 KV pass."""
    config = _base_config(dtype=ir.DataType.FLOAT16)
    module = registry.get("qwen2")(config)
    pkg = build_from_module(
        module,
        config,
        "text-generation",
        execution_provider="cuda",
        fp8_kv_cache=fp8_kv_cache,
        kv_cache_scales=kv_cache_scales,
    )
    return pkg["model"], config


class TestFp8KvCacheGraph:
    def test_all_kv_io_typed_fp8(self):
        model, config = _build_fp8_decoder()
        ins = {v.name: v for v in model.graph.inputs}
        outs = {v.name: v for v in model.graph.outputs}
        for i in range(config.num_hidden_layers):
            assert ins[f"past_key_values.{i}.key"].dtype == _FP8
            assert ins[f"past_key_values.{i}.value"].dtype == _FP8
            assert outs[f"present.{i}.key"].dtype == _FP8
            assert outs[f"present.{i}.value"].dtype == _FP8

    def test_gqa_scale_inputs_and_attrs(self):
        model, _ = _build_fp8_decoder()
        gqa_nodes = [
            n
            for n in model.graph
            if n.domain == "com.microsoft" and n.op_type == "GroupQueryAttention"
        ]
        assert gqa_nodes, "expected GroupQueryAttention nodes"
        for node in gqa_nodes:
            assert len(node.inputs) == 14
            assert node.inputs[12] is not None and node.inputs[12].name.endswith(
                "key.fp8_scale"
            )
            assert node.inputs[13] is not None and node.inputs[13].name.endswith(
                "value.fp8_scale"
            )
            assert node.attributes.get_string("k_quant_type") == "PER_TENSOR"
            assert node.attributes.get_string("v_quant_type") == "PER_TENSOR"
            assert node.attributes.get_int("kv_cache_bit_width") == 8

    def test_default_unit_scales(self):
        model, config = _build_fp8_decoder()
        for i in range(config.num_hidden_layers):
            k = model.graph.initializers[f"past_key_values.{i}.key.fp8_scale"]
            v = model.graph.initializers[f"past_key_values.{i}.value.fp8_scale"]
            np.testing.assert_array_equal(k.const_value.numpy(), np.array([1.0], np.float32))
            np.testing.assert_array_equal(v.const_value.numpy(), np.array([1.0], np.float32))

    def test_calibrated_scales_applied(self):
        scales = {0: (0.25, 0.5), 1: (2.0, 4.0)}
        model, _config = _build_fp8_decoder(kv_cache_scales=scales)
        for i, (kv_k, kv_v) in scales.items():
            k = model.graph.initializers[f"past_key_values.{i}.key.fp8_scale"]
            v = model.graph.initializers[f"past_key_values.{i}.value.fp8_scale"]
            np.testing.assert_array_equal(k.const_value.numpy(), np.array([kv_k], np.float32))
            np.testing.assert_array_equal(v.const_value.numpy(), np.array([kv_v], np.float32))

    def test_disabled_keeps_fp16_kv(self):
        model, _config = _build_fp8_decoder(fp8_kv_cache=False)
        ins = {v.name: v for v in model.graph.inputs}
        assert ins["past_key_values.0.key"].dtype == ir.DataType.FLOAT16

    def test_ignored_and_warns_on_non_gqa_ep(self):
        """fp8_kv_cache on a non-GQA EP is a no-op with a warning."""
        config = _base_config(dtype=ir.DataType.FLOAT)
        module = registry.get("qwen2")(config)
        with pytest.warns(UserWarning, match="fp8_kv_cache=True"):
            pkg = build_from_module(
                module,
                config,
                "text-generation",
                execution_provider="default",
                fp8_kv_cache=True,
            )
        ins = {v.name: v for v in pkg["model"].graph.inputs}
        assert ins["past_key_values.0.key"].dtype == ir.DataType.FLOAT

    def test_pass_is_idempotent(self):
        model, _config = _build_fp8_decoder()
        # Re-running the pass must not add extra inputs or re-type anything.
        Fp8KvCachePass()(model)
        for node in model.graph:
            if node.op_type == "GroupQueryAttention":
                assert len(node.inputs) == 14

    def test_skips_nonempty_initializer_cache(self):
        """A non-empty FLOAT initializer-backed cache is left untouched (warns)."""
        f16, i32 = ir.DataType.FLOAT16, ir.DataType.INT32

        def val(name, dt, shape):
            return ir.Value(name=name, type=ir.TensorType(dt), shape=ir.Shape(shape))

        query = val("query", f16, [1, 1, 32])
        seqlens_k = val("seqlens_k", i32, [1])
        total_seq = val("total_seq", i32, [1])
        # Non-empty fp16 past initializers (NOT graph inputs).
        past_key = val("past_key_values.0.key", f16, [1, 1, 1, 16])
        past_key.const_value = ir.tensor(
            np.ones((1, 1, 1, 16), np.float16), name="past_key_values.0.key"
        )
        past_value = val("past_key_values.0.value", f16, [1, 1, 1, 16])
        past_value.const_value = ir.tensor(
            np.ones((1, 1, 1, 16), np.float16), name="past_key_values.0.value"
        )
        out = val("out", f16, [1, 1, 32])
        pk = val("present.0.key", f16, [1, 1, 1, 16])
        pv = val("present.0.value", f16, [1, 1, 1, 16])
        node = ir.Node(
            "com.microsoft",
            "GroupQueryAttention",
            [query, None, None, past_key, past_value, seqlens_k, total_seq],
            attributes={
                "num_heads": ir.AttrInt64("num_heads", 2),
                "kv_num_heads": ir.AttrInt64("kv_num_heads", 1),
            },
            outputs=[out, pk, pv],
            num_outputs=3,
        )
        graph = ir.Graph(
            [query, seqlens_k, total_seq],
            [out, pk, pv],
            nodes=[node],
            initializers=[past_key, past_value],
            name="g",
            opset_imports={"": 24, "com.microsoft": 1},
        )
        model = ir.Model(graph, ir_version=10)
        with pytest.warns(UserWarning, match="non-empty initializer"):
            Fp8KvCachePass()(model)
        assert past_key.dtype == f16
        assert len(node.inputs) == 7  # unchanged: no scale inputs added


class TestLoadScaleFile:
    def test_parses_ort_genai_format(self, tmp_path):
        path = tmp_path / "scales.json"
        path.write_text(
            json.dumps({"scales": {"k_scales": [0.1, 0.2], "v_scales": [0.3, 0.4]}})
        )
        scales = load_kv_cache_scale_file(str(path))
        assert scales == {0: (0.1, 0.3), 1: (0.2, 0.4)}

    def test_rejects_missing_keys(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"scales": {"k_scales": [0.1]}}))
        with pytest.raises(ValueError, match="k_scales and scales"):
            load_kv_cache_scale_file(str(path))

    def test_rejects_length_mismatch(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"scales": {"k_scales": [0.1, 0.2], "v_scales": [0.3]}}))
        with pytest.raises(ValueError, match="equal length"):
            load_kv_cache_scale_file(str(path))

    def test_rejects_non_positive_scale(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps({"scales": {"k_scales": [0.1, 0.0], "v_scales": [0.3, 0.4]}})
        )
        with pytest.raises(ValueError, match="non-finite"):
            load_kv_cache_scale_file(str(path))

    def test_rejects_non_array(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"scales": {"k_scales": "0.1", "v_scales": "0.3"}}))
        with pytest.raises(ValueError, match="must be JSON arrays"):
            load_kv_cache_scale_file(str(path))


@pytest.mark.skipif(not _FP8_CUDA, reason="requires CUDA FP8 GQA kernel (SM89+)")
def test_fp8_kv_gqa_runs_on_cuda(tmp_path):
    """The emitted FP8 GQA op signature is accepted and computes on CUDA.

    Builds a single ``GroupQueryAttention`` graph with fp16 q/k/v, empty FP8
    past cache (as internal constants so no fp8 numpy feed is needed — a known
    ORT-Python limitation), runs the pass, and executes on the CUDA EP.
    """
    b, s, h, kv, d = 1, 4, 2, 1, 16
    f16 = ir.DataType.FLOAT16
    i32 = ir.DataType.INT32

    def val(name, dt, shape):
        return ir.Value(name=name, type=ir.TensorType(dt), shape=ir.Shape(shape))

    def const(name, arr):
        v = ir.Value(
            name=name, type=ir.TensorType(ir.tensor(arr).dtype), shape=ir.Shape(arr.shape)
        )
        v.const_value = ir.tensor(arr, name=name)
        return v

    query = val("query", f16, [b, s, h * d])
    key = val("key", f16, [b, s, kv * d])
    value = val("value", f16, [b, s, kv * d])
    seqlens_k = val("seqlens_k", i32, [b])
    total_seq = val("total_seq", i32, [1])
    # Empty (0-length) fp16 past constants: the pass retypes them to FP8 (there
    # are no elements to reinterpret) so no fp8 numpy feed is needed.
    past_key = const("past_key", np.zeros((b, kv, 0, d), dtype=np.float16))
    past_value = const("past_value", np.zeros((b, kv, 0, d), dtype=np.float16))
    out = val("out", f16, [b, s, h * d])
    pk = val("present_key", f16, [b, kv, s, d])
    pv = val("present_value", f16, [b, kv, s, d])

    node = ir.Node(
        "com.microsoft",
        "GroupQueryAttention",
        [query, key, value, past_key, past_value, seqlens_k, total_seq],
        attributes={
            "num_heads": ir.AttrInt64("num_heads", h),
            "kv_num_heads": ir.AttrInt64("kv_num_heads", kv),
        },
        outputs=[out, pk, pv],
        num_outputs=3,
    )
    graph = ir.Graph(
        [query, key, value, seqlens_k, total_seq],
        [out, pk, pv],
        nodes=[node],
        initializers=[past_key, past_value],
        name="g",
        opset_imports={"": 24, "com.microsoft": 1},
    )
    model = ir.Model(graph, ir_version=10)
    Fp8KvCachePass()(model)

    # Present outputs are now FP8 and the node carries k_scale/v_scale + attrs.
    assert graph.outputs[1].dtype == _FP8
    assert node.attributes.get_int("kv_cache_bit_width") == 8

    # Match the empty past constants' data to their new FP8 declared type.
    past_key.const_value = ir.tensor(
        np.zeros((b, kv, 0, d), dtype=ml_dtypes.float8_e4m3fn), name="past_key"
    )
    past_value.const_value = ir.tensor(
        np.zeros((b, kv, 0, d), dtype=ml_dtypes.float8_e4m3fn), name="past_value"
    )

    path = tmp_path / "gqa.onnx"
    ir.save(model, str(path))
    sess = ort.InferenceSession(str(path), providers=[_CUDA, "CPUExecutionProvider"])
    outs = sess.run(
        None,
        {
            "query": (np.random.randn(b, s, h * d) * 0.1).astype(np.float16),
            "key": (np.random.randn(b, s, kv * d) * 0.1).astype(np.float16),
            "value": (np.random.randn(b, s, kv * d) * 0.1).astype(np.float16),
            "seqlens_k": np.array([s - 1], dtype=np.int32),
            "total_seq": np.array([s], dtype=np.int32),
        },
    )
    assert np.isfinite(outs[0].astype(np.float32)).all()
