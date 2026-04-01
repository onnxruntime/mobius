# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests that float constants are CastLike-wrapped for non-float32 dtypes.

PR #58 wraps bare float32 constants with op.CastLike() to prevent dtype
mismatches when model tensors are float16 or bfloat16. These tests build
component graphs with float16 inputs and verify CastLike ops are present.
"""

from __future__ import annotations

import onnx_ir as ir
import pytest

from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius.components._activations import quick_gelu
from mobius.components._diffusion import AdaLayerNormOutput
from mobius.components._rms_norm import OffsetRMSNorm


class TestCastLikeDtypeSafety:
    """Verify CastLike wrapping of float constants for non-float32 dtypes."""

    @pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
    def test_offset_rms_norm_casts_constant(self, dtype: ir.DataType):
        """OffsetRMSNorm adds 1.0 to weight — must CastLike for non-f32."""
        norm = OffsetRMSNorm(hidden_size=64, eps=1e-6)
        builder_, op, graph = create_test_builder()
        x = create_test_input(builder_, "x", [2, 3, 64], dtype)
        result = norm(op, x)
        graph.outputs.append(result)
        assert count_op_type(graph, "CastLike") >= 1

    @pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
    def test_quick_gelu_casts_constant(self, dtype: ir.DataType):
        """quick_gelu uses 1.702 constant — must CastLike for non-f32."""
        builder_, op, graph = create_test_builder()
        x = create_test_input(builder_, "x", [2, 3, 64], dtype)
        result = quick_gelu(op, x)
        graph.outputs.append(result)
        assert count_op_type(graph, "CastLike") >= 1

    @pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
    def test_ada_layer_norm_output_casts_constant(self, dtype: ir.DataType):
        """AdaLayerNormOutput adds 1.0 to scale — must CastLike for non-f32."""
        mod = AdaLayerNormOutput(hidden_size=64, eps=1e-6)
        builder_, op, graph = create_test_builder()
        hidden = create_test_input(builder_, "hidden", [1, 4, 64], dtype)
        temb = create_test_input(builder_, "temb", [1, 64], dtype)
        result = mod(op, hidden, temb)
        graph.outputs.append(result)
        assert count_op_type(graph, "CastLike") >= 1

    def test_no_castlike_needed_for_float32(self):
        """With float32 inputs, CastLike is still present but benign.

        This verifies the pattern is always applied regardless of dtype,
        so there's no conditional logic that could miss a case.
        """
        norm = OffsetRMSNorm(hidden_size=64, eps=1e-6)
        builder_, op, graph = create_test_builder()
        x = create_test_input(builder_, "x", [2, 3, 64], ir.DataType.FLOAT)
        result = norm(op, x)
        graph.outputs.append(result)
        # CastLike is always emitted (no-op for float32, but present)
        assert count_op_type(graph, "CastLike") >= 1
