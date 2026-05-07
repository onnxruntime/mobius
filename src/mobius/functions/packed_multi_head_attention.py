# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standard-ONNX ir.Function body for PackedMultiHeadAttention.

Provides a portable fallback decomposition for
``com.microsoft::PackedMultiHeadAttention``.  When registered in
``model.functions``, :class:`onnx_ir.passes.common.InlinePass` can expand
the op for runtimes that do not have the native kernel.

The function body rebuilds the block-diagonal attention bias from
``cumulative_sequence_length`` and delegates to the standard ONNX
``Attention`` op::

    # Compute segment IDs from cumulative sequence lengths
    segment_ids = ReduceSum(Cast(GreaterOrEqual(range, cu_seqlens))) - 1

    # Build block-diagonal bias (0 for same segment, -inf for different)
    same = Equal(segment_ids[:, None], segment_ids[None, :])
    bias = Where(same, 0.0, -10000.0)

    # Standard Attention
    output = Attention(query, key, value, bias,
                       q_num_heads=<num_heads>, kv_num_heads=<num_heads>,
                       scale=<scale>)

The ``token_offset`` input is consumed by the native kernel but is
unused by the fallback body (segment boundaries from
``cumulative_sequence_length`` are sufficient).

Attributes:
    num_heads (int): Number of attention heads.
    scale (float): Attention scale factor (default 1/sqrt(head_dim)).
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript._internal.builder import build_function

from mobius._constants import OPSET_VERSION

DOMAIN = "com.microsoft"


def packed_multi_head_attention() -> ir.Function:
    """Build an ``ir.Function`` for ``PackedMultiHeadAttention``.

    Standard-ONNX body builds a block-diagonal attention bias from
    ``cumulative_sequence_length`` and calls the standard ``Attention`` op.

    Inputs:
        query:  (token_count, hidden_size)
        key:    (token_count, hidden_size)
        value:  (token_count, v_hidden_size)
        token_offset: (batch_size, sequence_length) — unused in fallback
        cumulative_sequence_length: (batch_size + 1,) INT32

    Output:
        output: (token_count, v_hidden_size)

    Attrs:
        num_heads (int): Number of attention heads.
        scale (float): Attention scale (default 1.0).
    """

    def body(
        op,
        query_input,
        key_input,
        value_input,
        token_offset_input,
        cumulative_sequence_length_input,
    ):
        # --- Compute sequence length from query shape ---
        # query: (token_count, hidden_size)
        token_count = op.Shape(query_input, start=0, end=1)
        token_count_scalar = op.Squeeze(token_count)

        # --- Build block-diagonal attention bias from cu_seqlens ---
        # Create range [0, 1, ..., token_count - 1]
        positions = op.Range(
            op.Constant(value_int=0),
            token_count_scalar,
            op.Constant(value_int=1),
        )

        # Compute segment IDs: for each position i, count how many
        # cu_seqlens boundaries it has passed.
        # segment_ids[i] = sum(i >= cu_seqlens[j] for all j) - 1
        positions_column = op.Unsqueeze(positions, [1])  # (N, 1)
        cu_seqlens_int64 = op.Cast(cumulative_sequence_length_input, to=7)  # INT64
        cu_seqlens_row = op.Unsqueeze(cu_seqlens_int64, [0])  # (1, S+1)

        # ge_mask[i, j] = (position_i >= cu_seqlens_j)
        greater_or_equal_mask = op.GreaterOrEqual(positions_column, cu_seqlens_row)
        greater_or_equal_int = op.Cast(greater_or_equal_mask, to=7)

        segment_ids = op.Sub(
            op.ReduceSum(greater_or_equal_int, [1], keepdims=False),
            op.Constant(value_int=1),
        )  # (token_count,)

        # Build same-segment mask: same_segment[i, j] = (seg[i] == seg[j])
        segment_ids_row = op.Unsqueeze(segment_ids, [1])  # (N, 1)
        segment_ids_column = op.Unsqueeze(segment_ids, [0])  # (1, N)
        same_segment = op.Equal(segment_ids_row, segment_ids_column)  # (N, N)

        # Convert to attention bias: 0 for same segment, -10000 for different
        attention_bias = op.Where(
            same_segment,
            op.Constant(value_float=0.0),
            op.Constant(value_float=-10000.0),
        )
        # Reshape for Attention: (1, 1, N, N)
        attention_bias = op.Unsqueeze(attention_bias, [0, 1])

        # --- Add batch dimension for Attention op ---
        # (token_count, hidden) → (1, token_count, hidden)
        query_batched = op.Unsqueeze(query_input, [0])
        key_batched = op.Unsqueeze(key_input, [0])
        value_batched = op.Unsqueeze(value_input, [0])

        # --- Call standard Attention op ---
        # ir.Attr with ref_attr_name forwards num_heads and scale from
        # the function's formal attributes to the inner Attention node.
        attention_output = op.Attention(
            query_batched,
            key_batched,
            value_batched,
            attention_bias,
            q_num_heads=ir.Attr(
                "q_num_heads",
                ir.AttributeType.INT,
                1,
                ref_attr_name="num_heads",
            ),
            kv_num_heads=ir.Attr(
                "kv_num_heads",
                ir.AttributeType.INT,
                1,
                ref_attr_name="num_heads",
            ),
            scale=ir.Attr(
                "scale",
                ir.AttributeType.FLOAT,
                1.0,
                ref_attr_name="scale",
            ),
        )

        # --- Remove batch dimension ---
        # (1, token_count, v_hidden) → (token_count, v_hidden)
        return op.Squeeze(attention_output, [0])

    return build_function(
        body,
        [
            ir.Value(name="query"),
            ir.Value(name="key"),
            ir.Value(name="value"),
            ir.Value(name="token_offset"),
            ir.Value(name="cumulative_sequence_length"),
        ],
        domain=DOMAIN,
        name="PackedMultiHeadAttention",
        attributes=[
            ir.Attr("num_heads", ir.AttributeType.INT, 1),
            ir.Attr("scale", ir.AttributeType.FLOAT, 1.0),
        ],
        opset_imports={"": OPSET_VERSION},
    )
