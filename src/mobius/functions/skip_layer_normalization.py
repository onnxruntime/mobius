# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standard-ONNX ir.Function bodies for SkipLayerNormalization operators.

Provides portable fallback implementations for the ``com.microsoft``
``SkipLayerNormalization`` and ``SkipSimplifiedLayerNormalization`` custom ops.
When registered in ``model.functions``, :class:`onnx_ir.passes.common.InlinePass`
can expand them for EPs that do not support the custom ops natively.

Naming convention:
    Python factory functions are snake_case (e.g. ``skip_layer_normalization``)
    while the ``ir.Function`` op type strings are PascalCase
    (``"SkipLayerNormalization"``). PascalCase aliases are provided for
    discoverability, matching the convention in
    :mod:`~mobius.functions.causal_conv` and
    :mod:`~mobius.functions.linear_attention`.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript._internal.builder import build_function

from mobius._constants import OPSET_VERSION

DOMAIN = "com.microsoft"


def skip_layer_normalization() -> ir.Function:
    """Build an ``ir.Function`` for ``SkipLayerNormalization``.

    Standard-ONNX body::

        add_out  = Add(input, skip)
        norm_out, mean_out, inv_std_out = LayerNormalization(
            add_out, weight, bias, axis=-1, epsilon=<forwarded>
        )

    Inputs:  ``[input, skip, weight, bias]``  (``bias`` may be absent at
             the call site; ``LayerNormalization`` accepts optional B).
    Outputs: ``[norm_out, mean_out, inv_std_out, add_out]``
    Attr:    ``epsilon`` (float)
    """

    def body(op, v_input, v_skip, v_weight, v_bias):
        add_out = op.Add(v_input, v_skip)

        # ref_attr_name forwards the caller's epsilon at InlinePass expand time.
        epsilon_attr = ir.Attr(
            "epsilon",
            ir.AttributeType.FLOAT,
            1e-5,
            ref_attr_name="epsilon",
        )
        norm_out, mean_out, inv_std_out = op.LayerNormalization(
            add_out,
            v_weight,
            v_bias,
            axis=-1,
            epsilon=epsilon_attr,
            _outputs=3,
        )

        norm_out.name = "norm_out"
        mean_out.name = "mean_out"
        inv_std_out.name = "inv_std_out"
        add_out.name = "add_out"
        return norm_out, mean_out, inv_std_out, add_out

    return build_function(
        body,
        [
            ir.Value(name="input"),
            ir.Value(name="skip"),
            ir.Value(name="weight"),
            ir.Value(name="bias"),
        ],
        domain=DOMAIN,
        name="SkipLayerNormalization",
        attributes=[
            ir.Attr("epsilon", ir.AttributeType.FLOAT, 1e-5),
        ],
        opset_imports={"": OPSET_VERSION},
    )


def skip_simplified_layer_normalization() -> ir.Function:
    """Build an ``ir.Function`` for ``SkipSimplifiedLayerNormalization``.

    Standard-ONNX body::

        add_out  = Add(input, skip)
        norm_out = RMSNormalization(add_out, weight, epsilon=<forwarded>)

    Inputs:  ``[input, skip, weight]``
    Outputs: ``[norm_out, add_out]``
    Attr:    ``epsilon`` (float)
    """

    def body(op, v_input, v_skip, v_weight):
        add_out = op.Add(v_input, v_skip)

        # ref_attr_name forwards caller's epsilon at InlinePass expand time.
        epsilon_attr = ir.Attr(
            "epsilon",
            ir.AttributeType.FLOAT,
            1e-5,
            ref_attr_name="epsilon",
        )
        norm_out = op.RMSNormalization(add_out, v_weight, epsilon=epsilon_attr)

        norm_out.name = "norm_out"
        add_out.name = "add_out"
        return norm_out, add_out

    return build_function(
        body,
        [
            ir.Value(name="input"),
            ir.Value(name="skip"),
            ir.Value(name="weight"),
        ],
        domain=DOMAIN,
        name="SkipSimplifiedLayerNormalization",
        attributes=[
            ir.Attr("epsilon", ir.AttributeType.FLOAT, 1e-5),
        ],
        opset_imports={"": OPSET_VERSION},
    )
