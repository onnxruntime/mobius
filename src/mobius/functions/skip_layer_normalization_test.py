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

from mobius.functions.skip_layer_normalization import (
    skip_layer_normalization,
    skip_simplified_layer_normalization,
)


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
