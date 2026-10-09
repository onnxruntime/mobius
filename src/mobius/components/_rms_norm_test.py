# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for RMSNorm component."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from onnxscript import nn

from mobius._flags import override_flags
from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius._testing.ort_inference import OnnxModelSession
from mobius.components._rms_norm import (
    GroupRMSNorm,
    PerHeadRMSNorm,
    RMSNorm,
    apply_rms_norm,
)


class TestRMSNorm:
    def test_rms_norm_creates_parameters(self):
        norm = RMSNorm(64, eps=1e-6)
        params = list(norm.parameters())
        # Should have weight only (eps is now a float attribute, not a parameter)
        assert len(params) == 1

    def test_rms_norm_weight_shape(self):
        norm = RMSNorm(128)
        assert list(norm.weight.shape) == [128]

    def test_rms_norm_forward_builds_graph(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        norm = RMSNorm(64, eps=1e-5)
        result = norm(op, x)
        assert result is not None
        # Should use the ONNX RMSNormalization op
        assert count_op_type(graph, "RMSNormalization") >= 1

    def test_rms_norm_different_eps(self):
        norm1 = RMSNorm(64, eps=1e-5)
        norm2 = RMSNorm(64, eps=1e-6)
        assert norm1.variance_epsilon != norm2.variance_epsilon

    def test_apply_rms_norm_function(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        weight = nn.Parameter([64], name="test_weight")
        weight._realize(builder)

        result = apply_rms_norm(op, x, weight, 1e-6)
        assert result is not None
        assert count_op_type(graph, "RMSNormalization") >= 1

    def test_multiple_rms_norms_in_same_graph(self):
        """Ensure two RMSNorm modules can coexist in the same graph."""
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])

        norm1 = RMSNorm(64, eps=1e-6)
        norm2 = RMSNorm(64, eps=1e-6)

        # Manually set names to avoid collision
        builder.push_module("norm1")
        for p in norm1._parameters.values():
            p._realize(builder)
        r1 = norm1.forward(op, x)
        builder.pop_module()

        builder.push_module("norm2")
        for p in norm2._parameters.values():
            p._realize(builder)
        r2 = norm2.forward(op, x)
        builder.pop_module()

        assert r1 is not None
        assert r2 is not None
        assert count_op_type(graph, "RMSNormalization") == 2


def _run_module(module, x: np.ndarray, weight: np.ndarray):
    """Build a tiny graph around ``module``, bake ``weight`` in, and execute via ORT."""
    builder, op, graph = create_test_builder()
    x_input = create_test_input(builder, "x", list(x.shape))
    module.weight.const_value = ir.tensor(weight.astype(np.float32))
    output = module(op, x_input)
    output.name = "output"
    graph.outputs.append(output)
    model = ir.Model(graph, ir_version=11)
    return OnnxModelSession(model, device="cpu").run({"x": x.astype(np.float32)})["output"]


def _group_rms_norm_reference(
    x: np.ndarray, weight: np.ndarray, n_groups: int, eps: float
) -> np.ndarray:
    hidden_size = x.shape[-1]
    group_size = hidden_size // n_groups
    grouped = x.reshape(*x.shape[:-1], n_groups, group_size)
    variance = np.mean(grouped**2, axis=-1, keepdims=True)
    normed = grouped / np.sqrt(variance + eps)
    return normed.reshape(x.shape) * weight


def _per_head_rms_norm_reference(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(x**2, axis=-1, keepdims=True)
    normed = x / np.sqrt(variance + eps)
    return normed * weight


class TestGroupRMSNorm:
    def test_invalid_n_groups_raises(self):
        with pytest.raises(ValueError, match="evenly divisible"):
            GroupRMSNorm(64, n_groups=3)
        with pytest.raises(ValueError, match="evenly divisible"):
            GroupRMSNorm(64, n_groups=0)

    def test_creates_single_full_width_weight(self):
        norm = GroupRMSNorm(64, n_groups=4)
        params = list(norm.parameters())
        assert len(params) == 1
        # Unlike a per-group weight, GroupRMSNorm's weight is full-width,
        # not (n_groups, group_size).
        assert list(norm.weight.shape) == [64]

    def test_n_groups_one_degenerates_to_rms_normalization(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        norm = GroupRMSNorm(64, n_groups=1)
        result = norm(op, x)
        assert result is not None
        assert count_op_type(graph, "RMSNormalization") >= 1

    def test_grouped_forward_matches_reference(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2, 3, 8)).astype(np.float32)
        weight = rng.normal(size=(8,)).astype(np.float32)
        norm = GroupRMSNorm(8, n_groups=4, eps=1e-6)
        actual = _run_module(norm, x, weight)
        expected = _group_rms_norm_reference(x, weight, n_groups=4, eps=1e-6)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_cuda_workaround_path_matches_reference(self):
        """The decomposed (fp32-upcast) CUDA workaround path must match the plain path.

        Regression test for the fp16-overflow bug fixed by upcasting
        ``grouped * grouped`` before the ``RMSNormalization`` comparison.
        """
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2, 3, 8)).astype(np.float32)
        weight = rng.normal(size=(8,)).astype(np.float32)
        with override_flags(ort_cuda_grouped_rmsnorm_workaround=True):
            norm = GroupRMSNorm(8, n_groups=4, eps=1e-6)
            actual = _run_module(norm, x, weight)
        expected = _group_rms_norm_reference(x, weight, n_groups=4, eps=1e-6)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


class TestPerHeadRMSNorm:
    def test_weight_shape_is_per_head(self):
        norm = PerHeadRMSNorm(num_heads=4, head_dim=16)
        params = list(norm.parameters())
        assert len(params) == 1
        assert list(norm.weight.shape) == [4, 16]

    def test_forward_builds_graph(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 4, 16])
        norm = PerHeadRMSNorm(num_heads=4, head_dim=16)
        result = norm(op, x)
        assert result is not None
        assert count_op_type(graph, "Mul") >= 1

    def test_forward_matches_reference(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(2, 3, 4, 16)).astype(np.float32)
        weight = rng.normal(size=(4, 16)).astype(np.float32)
        norm = PerHeadRMSNorm(num_heads=4, head_dim=16, eps=1e-6)
        actual = _run_module(norm, x, weight)
        expected = _per_head_rms_norm_reference(x, weight, eps=1e-6)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_distinct_per_head_weight_changes_output(self):
        """Each head must use its own weight row, not a broadcast shared one."""
        x = np.ones((1, 1, 2, 4), dtype=np.float32)
        weight = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
        norm = PerHeadRMSNorm(num_heads=2, head_dim=4, eps=1e-6)
        actual = _run_module(norm, x, weight)
        # Head 0 scaled by 1, head 1 scaled by 2 -- outputs must differ.
        assert not np.allclose(actual[..., 0, :], actual[..., 1, :])
        np.testing.assert_allclose(actual[..., 1, :], 2.0 * actual[..., 0, :], rtol=1e-4)
