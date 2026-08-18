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

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
from onnx_ir.passes.common import InlinePass

from mobius.functions import matmul_nbits, register_function_bodies


def _run(model: ir.Model, feeds: dict) -> np.ndarray:
    import os
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "m.onnx")
    ir.save(model, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)[0]


def _const(name: str, arr: np.ndarray) -> ir.Value:
    v = ir.Value(name=name)
    t = ir.tensor(arr)
    v.const_value = t
    v.shape = ir.Shape(arr.shape)
    v.dtype = t.dtype
    return v


def _build_matmulnbits_model(
    n_out: int, k_in: int, block: int, seed: int = 0
) -> tuple[ir.Model, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Build a single-node MatMulNBits model with random 4-bit weights.

    Returns ``(model, feeds, packed_weight, scales, packed_zero_points)``.
    """
    nb = k_in // block
    blob = block // 2  # 4-bit → 2 values per byte
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, (n_out, nb, blob), dtype=np.uint8)
    scales = rng.random((n_out, nb)).astype(np.float32) * 0.1 + 0.01
    zero_points = rng.integers(0, 256, (n_out, (nb + 1) // 2), dtype=np.uint8)
    a = rng.standard_normal((2, 3, k_in)).astype(np.float32)

    a_val = ir.Value(
        name="A", shape=ir.Shape([2, 3, k_in]), type=ir.TensorType(ir.DataType.FLOAT)
    )
    b_c = _const("B", packed)
    sc_c = _const("scales", scales)
    zp_c = _const("zero_points", zero_points)
    y = ir.Value(name="Y")
    node = ir.Node(
        "com.microsoft",
        "MatMulNBits",
        inputs=[a_val, b_c, sc_c, zp_c],
        outputs=[y],
        attributes=ir.convenience.convert_attributes(
            {"K": k_in, "N": n_out, "bits": 4, "block_size": block}
        ),
    )
    graph = ir.Graph(
        inputs=[a_val],
        outputs=[y],
        nodes=[node],
        initializers=[b_c, sc_c, zp_c],
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

        inlined, feeds2, *_ = _build_matmulnbits_model(n_out=8, k_in=64, block=32)
        self._inline(inlined)
        # Op is expanded: no MatMulNBits left, DequantizeLinear present.
        ops = [n.op_type for n in inlined.graph]
        assert "MatMulNBits" not in ops
        assert "DequantizeLinear" in ops
        got = _run(inlined, feeds2)

        np.testing.assert_allclose(got, reference, rtol=0, atol=0)

    def test_inline_matches_native_op_odd_blocks(self):
        """Odd n_blocks exercises the zero-point nibble slice (ceil(nb/2))."""
        # K=96, block=32 → nb=3 (odd)
        model, feeds, *_ = _build_matmulnbits_model(n_out=4, k_in=96, block=32, seed=1)
        reference = _run(model, feeds)

        inlined, feeds2, *_ = _build_matmulnbits_model(n_out=4, k_in=96, block=32, seed=1)
        self._inline(inlined)
        got = _run(inlined, feeds2)
        np.testing.assert_allclose(got, reference, rtol=0, atol=0)


class TestMatMulNBitsEpGating:
    """A qnn build lowers MatMulNBits to QDQ; a cpu build keeps the contrib op."""

    def _build(self, ep: str):
        import dataclasses
        from collections import Counter

        from mobius._builder import build_from_module
        from mobius._configs import CausalLMConfig, QuantizationConfig
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
                bits=4, group_size=32, quant_method="gguf", sym=False
            ),
        )
        module = registry.get("qwen2")(cfg)
        model = build_from_module(
            module, cfg, task=_default_task_for_model("qwen2"), execution_provider=ep
        )["model"]
        return Counter(n.op_type for n in model.graph)

    def test_cpu_keeps_matmulnbits(self):
        ops = self._build("cpu")
        assert ops.get("MatMulNBits", 0) > 0
        assert ops.get("DequantizeLinear", 0) == 0

    def test_qnn_lowers_to_qdq(self):
        ops = self._build("qnn")
        assert ops.get("MatMulNBits", 0) == 0
        assert ops.get("DequantizeLinear", 0) > 0
