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

from mobius.functions import register_function_bodies
from mobius.functions.skip_layer_normalization import (
    skip_layer_normalization,
    skip_simplified_layer_normalization,
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
        """Inlining the fallback must expand the custom op and keep the graph valid.

        Regression test: with the old 2-output body, InlinePass raised
        ``ValueError`` (output-count mismatch) because the 4-output node's
        ``input_skip_bias_sum`` (index 3) had no replacement value. The fixed
        4-output body reconnects the residual and leaves no dangling inputs.
        """
        value_type = ir.TensorType(ir.DataType.FLOAT)
        x = ir.Value(name="x", shape=ir.Shape([1, 2, 4]), type=value_type)
        skip = ir.Value(name="skip", shape=ir.Shape([1, 2, 4]), type=value_type)
        weight = ir.Value(name="weight", shape=ir.Shape([4]), type=value_type)
        outputs = [ir.Value(name=name) for name in ("norm", "mean", "inv_std", "sum")]
        fused_node = ir.Node(
            "com.microsoft",
            "SkipSimplifiedLayerNormalization",
            inputs=[x, skip, weight],
            outputs=outputs,
            attributes=ir.convenience.convert_attributes({"epsilon": 1e-5}),
        )
        residual = ir.Value(name="residual")
        consumer = ir.Node("", "Identity", inputs=[outputs[3]], outputs=[residual])
        model = ir.Model(
            ir.Graph(
                inputs=[x, skip, weight],
                outputs=[outputs[0], residual],
                nodes=[fused_node, consumer],
                opset_imports={"": 24, "com.microsoft": 1},
                name="skip_simplified",
            ),
            ir_version=11,
        )

        fused = _count(model, "SkipSimplifiedLayerNormalization")
        assert fused == 1

        self._inline_skip_norm(model)

        # All fused ops expanded back to Add + RMSNormalization.
        assert _count(model, "SkipSimplifiedLayerNormalization") == 0
        # Each expansion restores one residual Add.
        assert _count(model, "Add") >= fused
        # The optional mean/inv_std placeholders are unused and pruned-safe;
        # more importantly the residual reconnected with no dangling inputs.
        assert not _has_dangling_inputs(model)
