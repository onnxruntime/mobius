# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Parity tests: mobius RMSNorm vs transformers Phi3RMSNorm.

Phi3RMSNorm (from ``transformers.models.phi3.modeling_phi3``) is mathematically
equivalent to standard RMSNorm but always upcasts to float32 for the variance
computation, then casts the result back to the original input dtype:

    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)

The ONNX ``RMSNormalization`` op used by mobius also computes in float32
internally. This test suite documents the numerical agreement between the two
implementations across dtypes and shapes.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
import torch

from mobius._constants import OPSET_VERSION
from mobius._testing.ort_inference import OnnxModelSession

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@functools.cache
def _get_cached_rms_norm_session(
    hidden_size: int, eps: float, batch: int, seq: int
) -> OnnxModelSession:
    """Build and cache an ORT session for a single-RMSNorm ONNX model.

    Weight is exposed as a named graph input (not an initializer) so that
    the same session can be reused across tests that differ only in weight
    values.  Sessions are cached by ``(hidden_size, eps, batch, seq)``; the
    cache is module-scoped and cleaned up at process exit.

    Args:
        hidden_size: Feature dimension.
        eps: Variance epsilon.
        batch: Batch dimension of the ``x`` input.
        seq: Sequence dimension of the ``x`` input.

    Returns:
        A ready-to-use :class:`OnnxModelSession` whose inputs are
        ``"x"`` (shape ``[batch, seq, hidden_size]``) and
        ``"weight"`` (shape ``[hidden_size]``), both float32.
    """
    import onnx_ir as ir
    from onnxscript._internal.builder import GraphBuilder

    from mobius.components._rms_norm import apply_rms_norm

    graph = ir.Graph(
        [],
        [],
        nodes=[],
        name="rms_norm_graph",
        opset_imports={"": OPSET_VERSION},
    )
    b = GraphBuilder(graph)
    op = b.op

    # Both x and weight are graph inputs so a single session serves all tests
    # with the same shape config, regardless of the weight values used.
    x_val = ir.Value(
        name="x",
        shape=ir.Shape([batch, seq, hidden_size]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    weight_val = ir.Value(
        name="weight",
        shape=ir.Shape([hidden_size]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    graph.inputs.extend([x_val, weight_val])

    out = apply_rms_norm(op, x_val, weight_val, eps)
    graph.outputs.append(out)

    model = ir.Model(graph, ir_version=10)
    return OnnxModelSession(model)


def _run_phi3_rms_norm(
    weight_np: np.ndarray,
    x_np: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Run the HuggingFace Phi3RMSNorm on ``x_np`` and return the result as numpy.

    Args:
        weight_np: Float32 weight array of shape ``(hidden_size,)``.
        x_np: Input array; may be float32 or float16.
        eps: Variance epsilon.

    Returns:
        Output numpy array. Note that because ``weight_np`` is float32 and
        PyTorch promotes the computation, float16 inputs will produce a
        float32 output (matching ``Phi3RMSNorm``), while float32 inputs
        produce a float32 output.
    """
    try:
        from transformers.models.phi3.modeling_phi3 import Phi3RMSNorm
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "Phi3RMSNorm is not available in the installed transformers version; "
            "skipping Phi-3 RMSNorm parity tests.",
            allow_module_level=True,
        )

    hidden_size = weight_np.shape[0]
    phi3_norm = Phi3RMSNorm(hidden_size, eps=eps).eval()

    with torch.no_grad():
        phi3_norm.weight.copy_(torch.from_numpy(weight_np))
        x_t = torch.from_numpy(x_np)
        return phi3_norm(x_t).numpy()


def _run_mobius_rms_norm(
    hidden_size: int,
    eps: float,
    batch: int,
    seq: int,
    weight_np: np.ndarray,
    x_np: np.ndarray,
) -> np.ndarray:
    """Run mobius RMSNorm via a cached ORT session.

    Reuses the session cached by :func:`_get_cached_rms_norm_session` for the
    given ``(hidden_size, eps, batch, seq)`` configuration.  Weight and input
    are passed as feeds on every call, so different weight values can be used
    without rebuilding the session.

    Args:
        hidden_size: Feature dimension.
        eps: Variance epsilon.
        batch: Batch dimension.
        seq: Sequence dimension.
        weight_np: Float32 weight array of shape ``(hidden_size,)``.
        x_np: Input array; converted to float32 before feeding.

    Returns:
        Float32 output array of shape ``(batch, seq, hidden_size)``.
    """
    session = _get_cached_rms_norm_session(hidden_size, eps, batch, seq)
    result = session.run(
        {
            "x": x_np.astype(np.float32),
            "weight": weight_np.astype(np.float32),
        }
    )
    return result[next(iter(result.keys()))]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhi3RMSNormParityFloat32:
    """Numerical parity between Phi3RMSNorm and mobius RMSNorm for float32 inputs.

    For float32 inputs the two implementations should agree within a small
    absolute tolerance (atol up to 5e-6 in these tests), matching the expected
    behavior of the ONNX ``RMSNormalization`` op.
    """

    def test_basic_float32_parity(self):
        """Single-batch, multi-token float32 input matches within tight tolerance."""
        rng = np.random.default_rng(0)
        hidden_size, batch, seq = 64, 1, 4
        eps = 1e-6

        x_np = rng.standard_normal((batch, seq, hidden_size)).astype(np.float32)
        weight_np = rng.standard_normal(hidden_size).astype(np.float32)

        onnx_out = _run_mobius_rms_norm(hidden_size, eps, batch, seq, weight_np, x_np)
        hf_out = _run_phi3_rms_norm(weight_np, x_np, eps)

        np.testing.assert_allclose(
            onnx_out,
            hf_out,
            atol=5e-6,
            rtol=0,
            err_msg="float32 parity failed: mobius RMSNorm vs Phi3RMSNorm",
        )

    def test_multibatch_float32_parity(self):
        """Multi-batch float32 input matches within tight tolerance."""
        rng = np.random.default_rng(1)
        hidden_size, batch, seq = 128, 2, 8
        eps = 1e-5

        x_np = rng.standard_normal((batch, seq, hidden_size)).astype(np.float32)
        weight_np = rng.standard_normal(hidden_size).astype(np.float32)

        onnx_out = _run_mobius_rms_norm(hidden_size, eps, batch, seq, weight_np, x_np)
        hf_out = _run_phi3_rms_norm(weight_np, x_np, eps)

        np.testing.assert_allclose(
            onnx_out,
            hf_out,
            atol=1e-6,
            rtol=0,
            err_msg="multi-batch float32 parity failed",
        )

    def test_single_token_float32_parity(self):
        """Single-token (seq=1) decode step matches within tight tolerance."""
        rng = np.random.default_rng(2)
        hidden_size, batch, seq = 64, 1, 1
        eps = 1e-6

        x_np = rng.standard_normal((batch, seq, hidden_size)).astype(np.float32)
        weight_np = rng.standard_normal(hidden_size).astype(np.float32)

        onnx_out = _run_mobius_rms_norm(hidden_size, eps, batch, seq, weight_np, x_np)
        hf_out = _run_phi3_rms_norm(weight_np, x_np, eps)

        np.testing.assert_allclose(
            onnx_out,
            hf_out,
            atol=1e-6,
            rtol=0,
            err_msg="single-token float32 parity failed",
        )

    @pytest.mark.parametrize("eps", [1e-5, 1e-6, 1e-8])
    def test_various_eps_float32_parity(self, eps: float):
        """Various epsilon values all produce matching outputs."""
        rng = np.random.default_rng(3)
        hidden_size, batch, seq = 64, 1, 4

        x_np = rng.standard_normal((batch, seq, hidden_size)).astype(np.float32)
        weight_np = rng.standard_normal(hidden_size).astype(np.float32)

        onnx_out = _run_mobius_rms_norm(hidden_size, eps, batch, seq, weight_np, x_np)
        hf_out = _run_phi3_rms_norm(weight_np, x_np, eps)

        np.testing.assert_allclose(
            onnx_out,
            hf_out,
            atol=1e-6,
            rtol=0,
            err_msg=f"float32 parity failed for eps={eps}",
        )

    @pytest.mark.parametrize("hidden_size", [32, 64, 128, 256])
    def test_various_hidden_sizes_float32_parity(self, hidden_size: int):
        """Various hidden_size values all produce matching outputs.

        Larger hidden sizes accumulate more rounding error during the
        reduction (mean of squares), so we allow a slightly looser tolerance
        than 1 ULP: atol=5e-6 ≈ 4 ULP for float32 at O(1) activations.
        """
        rng = np.random.default_rng(4)
        batch, seq = 1, 4
        eps = 1e-6

        x_np = rng.standard_normal((batch, seq, hidden_size)).astype(np.float32)
        weight_np = rng.standard_normal(hidden_size).astype(np.float32)

        onnx_out = _run_mobius_rms_norm(hidden_size, eps, batch, seq, weight_np, x_np)
        hf_out = _run_phi3_rms_norm(weight_np, x_np, eps)

        np.testing.assert_allclose(
            onnx_out,
            hf_out,
            atol=5e-6,
            rtol=0,
            err_msg=f"float32 parity failed for hidden_size={hidden_size}",
        )

    def test_near_zero_inputs_float32_parity(self):
        """Near-zero inputs (stress-tests epsilon behaviour) match."""
        rng = np.random.default_rng(5)
        hidden_size, batch, seq = 64, 1, 4
        eps = 1e-6

        # Very small magnitudes trigger the epsilon term
        x_np = (rng.standard_normal((batch, seq, hidden_size)) * 1e-4).astype(np.float32)
        weight_np = rng.standard_normal(hidden_size).astype(np.float32)

        onnx_out = _run_mobius_rms_norm(hidden_size, eps, batch, seq, weight_np, x_np)
        hf_out = _run_phi3_rms_norm(weight_np, x_np, eps)

        np.testing.assert_allclose(
            onnx_out,
            hf_out,
            atol=1e-5,
            rtol=0,
            err_msg="near-zero float32 parity failed",
        )

    def test_large_magnitude_inputs_float32_parity(self):
        """Large-magnitude inputs match (no overflow for float32)."""
        rng = np.random.default_rng(6)
        hidden_size, batch, seq = 64, 1, 4
        eps = 1e-6

        x_np = (rng.standard_normal((batch, seq, hidden_size)) * 100.0).astype(np.float32)
        weight_np = rng.standard_normal(hidden_size).astype(np.float32)

        onnx_out = _run_mobius_rms_norm(hidden_size, eps, batch, seq, weight_np, x_np)
        hf_out = _run_phi3_rms_norm(weight_np, x_np, eps)

        np.testing.assert_allclose(
            onnx_out,
            hf_out,
            atol=1e-5,
            rtol=0,
            err_msg="large-magnitude float32 parity failed",
        )


class TestPhi3RMSNormParityFloat16:
    """Parity tests for float16 inputs.

    Phi3RMSNorm always promotes to float32 for the variance computation and
    then casts the result back to float16. The ONNX ``RMSNormalization`` op
    may follow a similar promotion strategy internally, so both paths should
    agree to within a loose float16 tolerance (atol ≤ 2e-3).
    """

    def test_float16_parity_within_loose_tolerance(self):
        """float16 inputs: mobius (float32 ONNX) agrees with Phi3RMSNorm within fp16 tolerance.

        Phi3RMSNorm promotes float16 input to float32, computes the
        normalization, and then casts the normalized tensor back to float16
        **before** multiplying by the weight:

            hidden_states = hidden_states * torch.rsqrt(variance + eps)
            return self.weight * hidden_states.to(input_dtype)   # ← cast first

        The ONNX model accepts a float32 input (constructed from the same
        float16 values so the numbers are identical) and keeps everything in
        float32, so the weight multiplication is done at higher precision.
        The maximum difference is bounded by one float16 ULP at the output
        magnitude, which is ≈ ``|output| * 2^-10`` ≈ 2e-3 for activations of
        magnitude ~2.
        """
        rng = np.random.default_rng(10)
        hidden_size, batch, seq = 64, 1, 4
        eps = 1e-6

        x_fp16 = rng.standard_normal((batch, seq, hidden_size)).astype(np.float16)
        weight_np = rng.standard_normal(hidden_size).astype(np.float32)

        # Feed float16 values promoted to float32; the ONNX model computes in float32.
        onnx_out_f32 = _run_mobius_rms_norm(hidden_size, eps, batch, seq, weight_np, x_fp16)

        # Phi3RMSNorm returns float32 because weight (float32) * norm_f16 → float32
        hf_out = _run_phi3_rms_norm(weight_np, x_fp16, eps)

        np.testing.assert_allclose(
            onnx_out_f32,
            hf_out,
            atol=2e-3,
            rtol=0,
            err_msg="float16 parity failed: mobius RMSNorm vs Phi3RMSNorm",
        )

    def test_float16_discrepancy_is_bounded(self):
        """Document that float16 ↔ float32 discrepancy stays within float16 precision.

        The maximum absolute difference between the float32 ONNX output and
        Phi3RMSNorm's float32-internal / float16-output result is bounded by
        the float16 quantisation error (~1e-3 for typical activations).
        """
        rng = np.random.default_rng(11)
        hidden_size, batch, seq = 128, 2, 8
        eps = 1e-5

        x_fp16 = rng.standard_normal((batch, seq, hidden_size)).astype(np.float16)
        weight_np = rng.standard_normal(hidden_size).astype(np.float32)

        # Feed float16 values promoted to float32; the ONNX model computes in float32.
        onnx_out_f32 = _run_mobius_rms_norm(hidden_size, eps, batch, seq, weight_np, x_fp16)

        hf_out_f32_internal = _run_phi3_rms_norm(weight_np, x_fp16, eps)
        hf_out_f32 = hf_out_f32_internal.astype(np.float32)

        max_diff = float(np.max(np.abs(onnx_out_f32 - hf_out_f32)))
        # float16 machine epsilon is ~9.77e-4 ≈ 1e-3; normalized activations
        # after RMSNorm are O(1), so differences > 1e-2 indicate a real bug.
        assert max_diff < 1e-2, (
            f"float16 discrepancy too large: max_abs_diff={max_diff:.6f} "
            f"(expected < 1e-2 for float16 inputs)"
        )
