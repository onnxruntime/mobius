# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the standard-ONNX ``com.microsoft::MatMulNBits`` function body.

The body is inlined by :class:`onnx_ir.passes.common.InlinePass` to expand the
blockwise-INT4 ``MatMulNBits`` contrib op into a QDQ (``DequantizeLinear`` +
``MatMul``) form for EPs that lack a native ``MatMulNBits`` kernel (QNN HTP).

The critical invariants:

1. **Numerical parity** — the inlined QDQ body must produce identical output to
   the native ``MatMulNBits`` op for the 4-bit / blocked layout mobius emits.
2. **EP gating** — a ``qnn`` build inlines the op to QDQ; a ``cpu`` build keeps
   the compact contrib op (its native kernel takes precedence over the
   registered-but-uninlined function body).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx_ir.passes.common import InlinePass

from mobius.functions import get_function, matmul_nbits, register_function_bodies


def _run(model: ir.Model, feeds: dict) -> np.ndarray:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "m.onnx"
        ir.save(model, path)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        output = sess.run(None, feeds)[0]
        del sess
        return output


def _const(name: str, arr: np.ndarray) -> ir.Value:
    v = ir.Value(name=name)
    t = ir.tensor(arr)
    v.const_value = t
    v.shape = ir.Shape(arr.shape)
    v.dtype = t.dtype
    return v


def _build_matmulnbits_model(
    n_out: int,
    k_in: int,
    block: int,
    seed: int = 0,
    *,
    accuracy_level: int | None = None,
    has_zero_points: bool = True,
    keep_empty_slot: bool = False,
    dtype: ir.DataType = ir.DataType.FLOAT,
) -> tuple[ir.Model, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Build a single-node MatMulNBits model with random 4-bit weights.

    Returns ``(model, feeds, packed_weight, scales, packed_zero_points)``.
    """
    nb = k_in // block
    blob = block // 2  # 4-bit → 2 values per byte
    rng = np.random.default_rng(seed)
    numpy_dtype = np.float16 if dtype == ir.DataType.FLOAT16 else np.float32
    packed = rng.integers(0, 256, (n_out, nb, blob), dtype=np.uint8)
    scales = (rng.random((n_out, nb)).astype(np.float32) * 0.1 + 0.01).astype(numpy_dtype)
    zero_points = rng.integers(0, 256, (n_out, (nb + 1) // 2), dtype=np.uint8)
    a = rng.standard_normal((2, 3, k_in)).astype(numpy_dtype)

    a_val = ir.Value(name="A", shape=ir.Shape([2, 3, k_in]), type=ir.TensorType(dtype))
    b_c = _const("B", packed)
    sc_c = _const("scales", scales)
    zp_c = _const("zero_points", zero_points)
    y = ir.Value(name="Y")
    attributes = {"K": k_in, "N": n_out, "bits": 4, "block_size": block}
    if accuracy_level is not None:
        attributes["accuracy_level"] = accuracy_level
    inputs = [a_val, b_c, sc_c]
    initializers = [b_c, sc_c]
    if has_zero_points:
        inputs.append(zp_c)
        initializers.append(zp_c)
    elif keep_empty_slot:
        inputs.append(None)
    node = ir.Node(
        "com.microsoft",
        "MatMulNBits",
        inputs=inputs,
        outputs=[y],
        attributes=ir.convenience.convert_attributes(attributes),
    )
    graph = ir.Graph(
        inputs=[a_val],
        outputs=[y],
        nodes=[node],
        initializers=initializers,
        opset_imports={"": 24, "com.microsoft": 1},
        name="mnb",
    )
    return ir.Model(graph, ir_version=10), {"A": a}, packed, scales, zero_points


class TestMatMulNBitsFunctionSignature:
    def test_op_identity(self):
        fn = matmul_nbits()
        assert fn.domain == "com.microsoft"
        assert fn.name == "MatMulNBits"
        assert len(fn.inputs) == 4  # A, B, scales, zero_points
        assert len(fn.outputs) == 1

    def test_default_zero_points_overload(self):
        identifier = ("com.microsoft", "MatMulNBits", "default_zero_points")
        function = get_function(identifier)
        assert function is not None
        assert function.identifier() == identifier
        assert not function.inputs[3].uses()

    def test_caller_provided_function_is_preserved(self):
        model, *_ = _build_matmulnbits_model(n_out=8, k_in=64, block=32, has_zero_points=False)
        function = matmul_nbits()
        model.functions[function.identifier()] = function

        register_function_bodies(model)

        assert model.functions[function.identifier()] is function
        assert next(iter(model.graph)).overload == ""

    def test_new_call_is_specialized_after_registration(self):
        model, *_ = _build_matmulnbits_model(n_out=8, k_in=64, block=32)
        register_function_bodies(model)
        node = next(iter(model.graph))
        node.resize_inputs(3)

        register_function_bodies(model)
        register_function_bodies(model)

        assert node.overload == "default_zero_points"
        assert node.op_identifier() in model.functions


class TestMatMulNBitsInlineParity:
    def _inline(self, model: ir.Model) -> None:
        register_function_bodies(model)
        InlinePass(criteria=lambda f: f.domain == "com.microsoft" and f.name == "MatMulNBits")(
            model
        )

    def test_inline_matches_native_op(self):
        """The inlined QDQ body must equal the native MatMulNBits op bit-for-bit."""
        model, feeds, *_ = _build_matmulnbits_model(n_out=8, k_in=64, block=32)
        reference = _run(model, feeds)

        inlined, feeds2, packed, *_ = _build_matmulnbits_model(n_out=8, k_in=64, block=32)
        self._inline(inlined)
        # Op is expanded: no MatMulNBits left, DequantizeLinear present.
        ops = [n.op_type for n in inlined.graph]
        assert "MatMulNBits" not in ops
        assert "DequantizeLinear" in ops
        packed_initializer = inlined.graph.initializers["B"]
        assert packed_initializer.dtype == ir.DataType.UINT8
        np.testing.assert_array_equal(np.asarray(packed_initializer.const_value), packed)
        got = _run(inlined, feeds2)

        np.testing.assert_allclose(got, reference, rtol=0, atol=0)

    def test_accuracy_four_matches_explicit_asymmetric_dequantization(self):
        """The CPU kernel consumes Mobius's packed weights and zero points correctly."""
        model, feeds, packed, scales, packed_zero_points = _build_matmulnbits_model(
            n_out=8,
            k_in=64,
            block=32,
            seed=11,
            accuracy_level=4,
        )
        got = _run(model, feeds)

        quants = np.empty((8, 2, 32), dtype=np.float32)
        quants[..., 0::2] = packed & 0x0F
        quants[..., 1::2] = packed >> 4
        zero_points = np.empty((8, 2), dtype=np.float32)
        zero_points[..., 0::2] = packed_zero_points & 0x0F
        zero_points[..., 1::2] = packed_zero_points >> 4
        weights = ((quants - zero_points[..., None]) * scales[..., None]).reshape(8, 64)
        expected = feeds["A"] @ weights.T

        error = got - expected
        assert float(np.max(np.abs(error))) < 0.1
        assert float(np.linalg.norm(error) / np.linalg.norm(expected)) < 0.02

    def test_inline_matches_native_op_odd_blocks(self):
        """Odd n_blocks exercises the zero-point nibble slice (ceil(nb/2))."""
        # K=96, block=32 → nb=3 (odd)
        model, feeds, *_ = _build_matmulnbits_model(n_out=4, k_in=96, block=32, seed=1)
        reference = _run(model, feeds)

        inlined, feeds2, *_ = _build_matmulnbits_model(n_out=4, k_in=96, block=32, seed=1)
        self._inline(inlined)
        got = _run(inlined, feeds2)
        np.testing.assert_allclose(got, reference, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
    @pytest.mark.parametrize("keep_empty_slot", [False, True])
    @pytest.mark.parametrize("num_blocks", [1, 2, 3])
    def test_default_zero_points_match_native_op(self, dtype, keep_empty_slot, num_blocks):
        kwargs = {
            "n_out": 8,
            "k_in": num_blocks * 32,
            "block": 32,
            "has_zero_points": False,
            "keep_empty_slot": keep_empty_slot,
            "dtype": dtype,
        }
        model, feeds, *_ = _build_matmulnbits_model(**kwargs)
        reference = _run(model, feeds)
        inlined, feeds, packed, *_ = _build_matmulnbits_model(**kwargs)

        self._inline(inlined)

        assert not any(node.op_type == "MatMulNBits" for node in inlined.graph)
        assert any(node.op_type == "DequantizeLinear" for node in inlined.graph)
        for node in inlined.graph.all_nodes():
            if node.op_type in {"BitwiseAnd", "BitShift"}:
                assert all(value is not None for value in node.inputs)
        np.testing.assert_array_equal(inlined.graph.initializers["B"].const_value, packed)
        got = _run(inlined, feeds)
        tolerance = 1e-2 if dtype == ir.DataType.FLOAT16 else 0
        np.testing.assert_allclose(got, reference, rtol=tolerance, atol=tolerance)

    def test_mixed_zero_point_forms_keep_distinct_semantics(self):
        def build():
            model, feeds, *_ = _build_matmulnbits_model(n_out=8, k_in=96, block=32)
            asymmetric = next(iter(model.graph))
            symmetric_output = ir.Value(name="symmetric_output")
            model.graph.append(
                ir.Node(
                    "com.microsoft",
                    "MatMulNBits",
                    inputs=asymmetric.inputs[:3],
                    outputs=[symmetric_output],
                    attributes=dict(asymmetric.attributes),
                )
            )
            combined = ir.Value(name="combined")
            model.graph.append(
                ir.Node(
                    "",
                    "Concat",
                    inputs=[asymmetric.outputs[0], symmetric_output],
                    outputs=[combined],
                    attributes={"axis": ir.AttrInt64("axis", 0)},
                )
            )
            model.graph.outputs.clear()
            model.graph.outputs.append(combined)
            return model, feeds

        model, feeds = build()
        reference = _run(model, feeds)
        inlined, feeds = build()
        self._inline(inlined)

        np.testing.assert_allclose(_run(inlined, feeds), reference, rtol=0, atol=0)

    def test_registered_default_body_keeps_native_kernel(self, tmp_path):
        model, feeds, packed, *_ = _build_matmulnbits_model(
            n_out=8, k_in=64, block=32, has_zero_points=False
        )
        reference = _run(model, feeds)
        register_function_bodies(model)
        node = next(iter(model.graph))
        assert len(node.inputs) == 3
        assert node.op_type == "MatMulNBits"
        np.testing.assert_array_equal(model.graph.initializers["B"].const_value, packed)
        path = tmp_path / "native.onnx"
        ir.save(model, path)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.enable_profiling = True
        options.profile_file_prefix = str(tmp_path / "native-profile")
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        got = session.run(None, feeds)[0]
        profile = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
        operations = {
            event["args"]["op_name"]
            for event in profile
            if event.get("cat") == "Node" and "op_name" in event.get("args", {})
        }
        assert "MatMulNBits" in operations
        assert "DequantizeLinear" not in operations
        np.testing.assert_array_equal(got, reference)

    @pytest.mark.parametrize("keep_empty_slot", [False, True])
    def test_default_zero_points_remain_convertible_by_openvino(
        self, tmp_path, keep_empty_slot
    ):
        ov = pytest.importorskip("openvino")
        model, feeds, *_ = _build_matmulnbits_model(
            n_out=8,
            k_in=96,
            block=32,
            has_zero_points=False,
            keep_empty_slot=keep_empty_slot,
        )
        reference = _run(model, feeds)
        register_function_bodies(model)
        path = tmp_path / "openvino.onnx"
        ir.save(model, path)
        converted = ov.convert_model(path)
        # Compare exact f32 arithmetic, not the CPU plugin's dynamic activation quantization.
        compiled = ov.Core().compile_model(
            converted,
            "CPU",
            {
                "INFERENCE_PRECISION_HINT": ov.Type.f32,
                "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0,
            },
        )
        got = compiled({"A": feeds["A"]})[0]
        np.testing.assert_allclose(got, reference, rtol=1e-5, atol=1e-5)


class TestMatMulNBitsEpGating:
    """EPs without native MatMulNBits support lower blockwise INT4 to QDQ."""

    def _build(self, ep: str, *, sym: bool = False):
        import dataclasses
        from collections import Counter

        from mobius._builder import build_from_module
        from mobius._configs import CausalLMConfig, QuantizationConfig
        from mobius._optimizations import optimize_model
        from mobius._registry import registry
        from mobius.integrations.transformers._config_resolver import (
            _default_task_for_model,
        )

        cfg = CausalLMConfig(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            max_position_embeddings=64,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
        )
        cfg.model_type = "qwen2"
        cfg = dataclasses.replace(
            cfg,
            quantization=QuantizationConfig(
                bits=4, group_size=32, quant_method="gguf", sym=sym
            ),
        )
        module = registry.get("qwen2")(cfg)
        model = build_from_module(
            module, cfg, task=_default_task_for_model("qwen2"), execution_provider=ep
        )["model"]
        optimize_model(model, ep=ep, dtype=cfg.dtype, model_role="decoder")
        for node in model.graph.all_nodes():
            if node.op_type in {"BitwiseAnd", "BitShift"}:
                assert all(value is not None for value in node.inputs)
        return Counter(n.op_type for n in model.graph)

    def test_cpu_keeps_matmulnbits(self):
        ops = self._build("cpu")
        assert ops.get("MatMulNBits", 0) > 0
        assert ops.get("DequantizeLinear", 0) == 0

    def test_qnn_lowers_to_qdq(self):
        ops = self._build("qnn")
        assert ops.get("MatMulNBits", 0) == 0
        assert ops.get("DequantizeLinear", 0) > 0

    def test_trt_rtx_lowers_to_qdq(self):
        ops = self._build("trt-rtx")
        assert ops.get("MatMulNBits", 0) == 0
        assert ops.get("DequantizeLinear", 0) > 0

    @pytest.mark.parametrize("ep", ["onnx-standard", "qnn", "trt-rtx"])
    def test_symmetric_quantization_lowers_to_valid_qdq(self, ep):
        ops = self._build(ep, sym=True)
        assert ops.get("MatMulNBits", 0) == 0
        assert ops.get("DequantizeLinear", 0) > 0

    def test_openvino_keeps_symmetric_matmulnbits(self):
        ops = self._build("openvino", sym=True)
        assert ops.get("MatMulNBits", 0) > 0
        assert ops.get("DequantizeLinear", 0) == 0
