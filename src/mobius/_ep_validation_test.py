# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for _ep_validation.validate_ep_support()."""

from __future__ import annotations

import pytest

from mobius._ep_validation import KNOWN_EPS, validate_ep_support


class TestValidateEpSupport:
    def test_valid_combinations_pass(self):
        """Common valid combos should not raise."""
        valid = [
            ("llama", "cpu"),
            ("llama", "cuda"),
            ("llama", "dml"),
            ("llama", "webgpu"),
            ("llama", "trt-rtx"),
            ("gpt2", "cpu"),
            ("qwen2", "cuda"),
            ("phi3", "dml"),
        ]
        for model_type, ep in valid:
            validate_ep_support(model_type, ep)  # Should not raise

    def test_moe_on_dml_rejected(self):
        """MoE models should be rejected on DML."""
        moe_types = ["mixtral", "phimoe", "qwen2_moe", "qwen3_moe", "dbrx"]
        for model_type in moe_types:
            with pytest.raises(ValueError, match="DML"):
                validate_ep_support(model_type, "dml")

    def test_moe_on_webgpu_rejected(self):
        """MoE models should be rejected on WebGPU."""
        with pytest.raises(ValueError, match="WebGPU"):
            validate_ep_support("mixtral", "webgpu")

    def test_mamba_on_webgpu_rejected(self):
        """Mamba SSM models should be rejected on WebGPU."""
        with pytest.raises(ValueError, match="WebGPU"):
            validate_ep_support("mamba", "webgpu")

    def test_jamba_on_trt_rtx_rejected(self):
        """Jamba hybrid is unsupported on TRT-RTX."""
        with pytest.raises(ValueError, match="TRT-RTX"):
            validate_ep_support("jamba", "trt-rtx")

    def test_unknown_ep_rejected(self):
        """Unknown EP names should be rejected."""
        with pytest.raises(ValueError, match="Unknown execution provider"):
            validate_ep_support("llama", "rocm")

    def test_known_eps_frozenset(self):
        """KNOWN_EPS should contain the canonical EP set."""
        assert KNOWN_EPS == {"cpu", "cuda", "dml", "webgpu", "trt-rtx"}

    def test_moe_on_cuda_allowed(self):
        """MoE models should work fine on CUDA."""
        validate_ep_support("mixtral", "cuda")  # Should not raise

    def test_moe_on_cpu_allowed(self):
        """MoE models should work fine on CPU."""
        validate_ep_support("mixtral", "cpu")  # Should not raise
