# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the standard-ONNX SkipLayerNormalization function bodies.

These bodies are inlined by :class:`onnx_ir.passes.common.InlinePass` to
expand the ``com.microsoft`` custom ops for EPs that do not support them.
The key invariant is that each function's output arity and ordering match
the ``com.microsoft`` op spec so InlinePass can reconnect downstream
consumers by position — in particular ``input_skip_bias_sum`` must stay at
output index 3, not collapse into the optional ``mean`` slot at index 1.
"""

from __future__ import annotations

import onnx_ir as ir
from onnx_ir.passes.common import InlinePass

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._registry import registry
from mobius.functions import register_function_bodies
from mobius.functions.skip_layer_normalization import (
    skip_layer_normalization,
    skip_simplified_layer_normalization,
)


def _tiny_config() -> ArchitectureConfig:
    return ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10000.0,
        pad_token_id=0,
    )


def _count(model: ir.Model, op_type: str) -> int:
    return sum(1 for n in model.graph.all_nodes() if n.op_type == op_type)


def _has_dangling_inputs(model: ir.Model) -> bool:
    """Return True if any node consumes a value with no producer/initializer/input."""
    known: set[ir.Value] = set(model.graph.inputs)
    known.update(model.graph.initializers.values())
    for node in model.graph.all_nodes():
        known.update(node.outputs)
    for node in model.graph.all_nodes():
        for inp in node.inputs:
            if inp is not None and inp.name and inp not in known:
                return True
    return False


class TestSkipSimplifiedFunctionSignature:
    def test_output_arity_matches_spec(self):
        """SkipSimplifiedLayerNormalization must expose 4 positional outputs.

        Spec order: output(0), mean(1), inv_std_var(2), input_skip_bias_sum(3).
        """
        fn = skip_simplified_layer_normalization()
        assert len(fn.outputs) == 4
        # The residual sum must live at index 3 (not index 1).
        assert fn.outputs[3].name == "add_out"
        assert fn.outputs[0].name == "norm_out"

    def test_skip_layer_norm_output_arity(self):
        """The non-simplified variant also exposes 4 outputs with sum at index 3."""
        fn = skip_layer_normalization()
        assert len(fn.outputs) == 4
        assert fn.outputs[3].name == "add_out"


class TestSkipSimplifiedInline:
    def _inline_skip_norm(self, model: ir.Model) -> None:
        register_function_bodies(model)

        def criteria(func: ir.Function) -> bool:
            return func.domain == "com.microsoft" and func.name in (
                "SkipLayerNormalization",
                "SkipSimplifiedLayerNormalization",
            )

        InlinePass(criteria=criteria)(model)

    def test_inline_preserves_residual(self):
        """Inlining the fallback must expand every fused op and keep the graph valid.

        Regression test: with the old 2-output body, InlinePass raised
        ``ValueError`` (output-count mismatch) because the 4-output node's
        ``input_skip_bias_sum`` (index 3) had no replacement value. The fixed
        4-output body reconnects the residual and leaves no dangling inputs.
        """
        config = _tiny_config()
        model = build_from_module(registry.get("qwen2")(config), config)["model"]

        fused = _count(model, "SkipSimplifiedLayerNormalization")
        assert fused > 0, "expected the build pipeline to fuse Add+RMSNorm"

        self._inline_skip_norm(model)

        # All fused ops expanded back to Add + RMSNormalization.
        assert _count(model, "SkipSimplifiedLayerNormalization") == 0
        # Each expansion restores one residual Add.
        assert _count(model, "Add") >= fused
        # The optional mean/inv_std placeholders are unused and pruned-safe;
        # more importantly the residual reconnected with no dangling inputs.
        assert not _has_dangling_inputs(model)
