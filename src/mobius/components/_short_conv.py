# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Double-gated causal short convolution with recurrent state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius.components._common import INT64_MAX, Linear

if TYPE_CHECKING:
    import onnx_ir as ir


class _DepthwiseShortConv1d(nn.Module):
    """Depthwise Conv1D backed by ``com.microsoft.CausalConvWithState``."""

    def __init__(self, channels: int, kernel_size: int, *, bias: bool):
        super().__init__()
        self.weight = nn.Parameter([channels, 1, kernel_size])
        self.bias = nn.Parameter([channels]) if bias else None
        self._channels = channels

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        bias = self.bias
        if bias is None:
            bias = op.Expand(
                op.CastLike(op.Constant(value_float=0.0), self.weight),
                op.Constant(value_ints=[self._channels]),
            )
        return op.CausalConvWithState(
            hidden_states,
            self.weight,
            bias,
            conv_state,
            activation="none",
            ndim=1,
            _domain="com.microsoft",
            _outputs=2,
        )


class GatedShortConv(nn.Module):
    """Double-gated depthwise causal convolution used by hybrid language models.

    The input projection produces ``B``, ``C``, and ``x`` branches. The
    depthwise convolution consumes ``B * x``, then the second gate forms
    ``C * conv(B * x)`` before the output projection.
    """

    def __init__(self, hidden_size: int, kernel_size: int, *, bias: bool = False):
        super().__init__()
        self.in_proj = Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.conv = _DepthwiseShortConv1d(hidden_size, kernel_size, bias=bias)
        self.out_proj = Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        attention_mask: ir.Value | None = None,
    ) -> tuple[ir.Value, ir.Value]:
        """Apply the gated convolution and return output plus updated state.

        Args:
            hidden_states: Input activations shaped ``(batch, seq, hidden)``.
            conv_state: Previous convolution history shaped
                ``(batch, hidden, kernel_size - 1)``.
            attention_mask: Optional padding mask shaped
                ``(batch, past_seq + seq)``.
        """
        if attention_mask is not None:
            # Recurrent layers only consume the mask for the current token span.
            seq_len = op.Shape(hidden_states, start=1, end=2)
            current_mask = op.Slice(
                attention_mask,
                op.Neg(seq_len),
                op.Constant(value_ints=[INT64_MAX]),
                op.Constant(value_ints=[1]),
            )
            current_mask = op.Unsqueeze(current_mask, op.Constant(value_ints=[-1]))
            hidden_states = op.Mul(hidden_states, op.CastLike(current_mask, hidden_states))

        # (B, T, H) -> (B, 3H, T), split into the two gates and conv input.
        projected = op.Transpose(self.in_proj(op, hidden_states), perm=[0, 2, 1])
        gate_b, gate_c, conv_input = op.Split(projected, num_outputs=3, axis=1, _outputs=3)
        conv_input = op.Mul(gate_b, conv_input)  # (B, H, T)
        conv_output, present_state = self.conv(op, conv_input, conv_state)
        output = op.Mul(gate_c, conv_output)  # (B, H, T)
        output = self.out_proj(op, op.Transpose(output, perm=[0, 2, 1]))
        return output, present_state
