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
from mobius._configs import Gemma3nConfig
from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius.components._activations import quick_gelu
from mobius.components._audio import ConformerEncoderLayer
from mobius.components._diffusion import AdaLayerNormOutput
from mobius.components._moe import SigmoidTopKGate, SparseMixerGate
from mobius.components._rms_norm import OffsetRMSNorm
from mobius.models.gemma3n import Gemma3nAltUp


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


class TestSigmoidTopKGate:
    """SigmoidTopKGate: verify 1e-9 epsilon and routed_scaling_factor auto-cast."""

    @pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
    def test_routing_weights_dtype(self, dtype: ir.DataType):
        """Routing weights stay in input dtype — no FP32 widening from 1e-9 literal."""
        gate = SigmoidTopKGate(
            hidden_size=32,
            num_experts=4,
            top_k=2,
            norm_topk_prob=True,
        )
        _cast_module_dtype(gate, dtype)
        builder_, op, graph = create_test_builder()
        x = create_test_input(builder_, "x", [1, 3, 32], dtype)
        routing_weights, selected_experts = gate(op, x)
        graph.outputs.extend([routing_weights, selected_experts])
        # 1e-9 in op.Add(weight_sum, 1e-9) must auto-cast to routing dtype
        assert count_op_type(graph, "CastLike") == 0
        assert routing_weights.dtype == dtype

    @pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
    def test_routed_scaling_factor_autocasts(self, dtype: ir.DataType):
        """routed_scaling_factor Python float literal auto-casts to routing dtype."""
        gate = SigmoidTopKGate(
            hidden_size=32,
            num_experts=4,
            top_k=2,
            norm_topk_prob=False,
            routed_scaling_factor=2.5,
        )
        _cast_module_dtype(gate, dtype)
        builder_, op, graph = create_test_builder()
        x = create_test_input(builder_, "x", [1, 3, 32], dtype)
        routing_weights, _ = gate(op, x)
        graph.outputs.append(routing_weights)
        # routed_scaling_factor=2.5 in op.Mul must auto-cast, not widen to FP32
        assert count_op_type(graph, "CastLike") == 0
        assert routing_weights.dtype == dtype


class TestSparseMixerGate:
    """SparseMixerGate: verify CastLike(-1e30) preserves input dtype."""

    @pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
    def test_castlike_neg_inf_uses_input_dtype(self, dtype: ir.DataType):
        """op.CastLike(-1e30, scores) must cast the constant to the scores dtype.

        Without CastLike the -1e30 literal would be FLOAT32, causing type
        mismatches in the op.Where and op.Expand downstream ops.
        """
        gate = SparseMixerGate(hidden_size=32, num_experts=4, top_k=2)
        _cast_module_dtype(gate, dtype)
        builder_, op, graph = create_test_builder()
        x = create_test_input(builder_, "x", [1, 3, 32], dtype)
        routing_weights, selected_experts = gate(op, x)
        graph.outputs.extend([routing_weights, selected_experts])
        # CastLike nodes should be present (the pattern is intentional for -1e30)
        assert count_op_type(graph, "CastLike") > 0
        # Final routing weights must remain in the input dtype
        assert routing_weights.dtype == dtype


class TestConformerEncoderLayer:
    """ConformerEncoderLayer: verify 0.5 Macaron weight auto-casts."""

    @pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
    def test_macaron_half_weight_autocasts(self, dtype: ir.DataType):
        """0.5 literal in op.Mul(feed_forward(x), 0.5) auto-casts to input dtype.

        The Macaron structure applies half-weight feed-forward modules:
        ``x += 0.5 * feed_forward_in(x)`` and ``x += 0.5 * feed_forward_out(x)``.
        Both 0.5 literals must auto-cast to the hidden state dtype.
        """
        layer = ConformerEncoderLayer(d_model=32, num_heads=4, d_inner=64, kernel_size=3)
        _cast_module_dtype(layer, dtype)
        builder_, op, graph = create_test_builder()
        x = create_test_input(builder_, "x", [1, 5, 32], dtype)
        # relative_attention_bias: [num_heads, q_len, kv_len]
        bias = create_test_input(builder_, "bias", [4, 5, 5], dtype)
        result = layer(op, x, bias)
        graph.outputs.append(result)
        # No CastLike needed — Python float 0.5 auto-casts
        assert count_op_type(graph, "CastLike") == 0
        assert _get_output_dtype(graph) == dtype


class TestGemma3nAltUp:
    """Gemma3nAltUp: verify router_input_scale Python float auto-casts."""

    def _make_config(self, hidden_size: int = 32) -> Gemma3nConfig:
        from mobius._configs import ArchitectureConfig

        base = ArchitectureConfig(
            hidden_size=hidden_size,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=1,
            vocab_size=256,
        )
        return Gemma3nConfig(
            **{k: getattr(base, k) for k in base.__dataclass_fields__ if hasattr(base, k)},
            altup_num_inputs=2,
            altup_active_idx=0,
            altup_correct_scale=True,
            laurel_rank=8,
            hidden_size_per_layer_input=16,
            vocab_size_per_layer_input=256,
        )

    @pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
    def test_router_input_scale_autocasts(self, dtype: ir.DataType):
        """router_input_scale = hidden_size**-1.0 auto-casts in op.Mul.

        AltUp._compute_router_modalities multiplies a normalized hidden state
        by self.router_input_scale (a Python float). This must not widen
        BF16/FP16 computations to FP32.
        """
        config = self._make_config(hidden_size=32)
        altup = Gemma3nAltUp(config)
        _cast_module_dtype(altup, dtype)
        builder_, op, graph = create_test_builder()
        # altup_num_inputs=2 — provide two hidden state tensors
        hs0 = create_test_input(builder_, "hs0", [1, 3, 32], dtype)
        hs1 = create_test_input(builder_, "hs1", [1, 3, 32], dtype)
        predicted = altup.predict(op, [hs0, hs1])
        graph.outputs.extend(predicted)
        # router_input_scale (float) in op.Mul must auto-cast
        assert count_op_type(graph, "CastLike") == 0
        assert predicted[0].dtype == dtype
