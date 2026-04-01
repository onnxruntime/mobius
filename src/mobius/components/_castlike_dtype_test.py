# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests that Python float literals auto-cast to match operand dtypes.

onnxscript auto-casts Python scalars (int/float/bool) to match the dtype of
the other operand in a binary op. These tests verify that components using
Python literals produce output in the correct dtype — no spurious FLOAT32
constants widening BF16/FP16 computations.
"""

from __future__ import annotations

import onnx_ir as ir
import pytest

from mobius._builder import _cast_module_dtype
from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius.components._activations import quick_gelu
from mobius.components._diffusion import AdaLayerNormOutput
from mobius.components._rms_norm import OffsetRMSNorm


def _get_output_dtype(graph: ir.Graph) -> ir.DataType | None:
    """Return the dtype of the first graph output, or None."""
    if graph.outputs:
        return graph.outputs[0].dtype
    return None


class TestPythonLiteralAutocast:
    """Verify that Python float literals auto-cast to match tensor operand dtypes."""

    @pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
    def test_offset_rms_norm_constant_autocasts(self, dtype: ir.DataType):
        """OffsetRMSNorm: `1.0` literal in op.Add auto-casts to weight dtype.

        After _cast_module_dtype, the weight param is BF16/FP16. The Python
        literal 1.0 in op.Add(self.weight, 1.0) must auto-cast to match.
        """
        norm = OffsetRMSNorm(hidden_size=64, eps=1e-6)
        _cast_module_dtype(norm, dtype)
        builder_, op, graph = create_test_builder()
        x = create_test_input(builder_, "x", [2, 3, 64], dtype)
        result = norm(op, x)
        graph.outputs.append(result)
        # No CastLike — auto-cast handles it; output stays in the expected dtype
        assert count_op_type(graph, "CastLike") == 0
        assert _get_output_dtype(graph) == dtype

    @pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
    def test_quick_gelu_constant_autocasts(self, dtype: ir.DataType):
        """quick_gelu: `1.702` literal in op.Mul auto-casts to input dtype."""
        builder_, op, graph = create_test_builder()
        x = create_test_input(builder_, "x", [2, 3, 64], dtype)
        result = quick_gelu(op, x)
        graph.outputs.append(result)
        assert count_op_type(graph, "CastLike") == 0
        assert _get_output_dtype(graph) == dtype

    @pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
    def test_ada_layer_norm_output_constant_autocasts(self, dtype: ir.DataType):
        """AdaLayerNormOutput: `1.0` literal auto-casts to scale tensor dtype."""
        mod = AdaLayerNormOutput(hidden_size=64, eps=1e-6)
        _cast_module_dtype(mod, dtype)
        builder_, op, graph = create_test_builder()
        hidden = create_test_input(builder_, "hidden", [1, 4, 64], dtype)
        temb = create_test_input(builder_, "temb", [1, 64], dtype)
        result = mod(op, hidden, temb)
        graph.outputs.append(result)
        assert count_op_type(graph, "CastLike") == 0
        assert _get_output_dtype(graph) == dtype

    def test_float32_inputs_produce_float32_output(self):
        """Float32 inputs — no special casting needed, output stays float32."""
        norm = OffsetRMSNorm(hidden_size=64, eps=1e-6)
        builder_, op, graph = create_test_builder()
        x = create_test_input(builder_, "x", [2, 3, 64], ir.DataType.FLOAT)
        result = norm(op, x)
        graph.outputs.append(result)
        assert count_op_type(graph, "CastLike") == 0
        assert _get_output_dtype(graph) == ir.DataType.FLOAT
