# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ShortConv: gated causal depthwise Conv1d for LFM2 conv layers.

Implements the ``Lfm2ShortConv`` layer from HuggingFace transformers:

1. ``in_proj(x)`` -> split into B, C, x chunks  (3 x hidden_size)
2. ``B * x`` (element-wise gating)
3. Causal depthwise Conv1d on ``Bx``
4. ``C * conv_out`` (output gating)
5. ``out_proj(y)``

The convolution is depthwise (groups=hidden_size) with causal padding
(left-pad by kernel_size-1). During inference, the ``conv_state``
(B, hidden_size, kernel_size-1) is carried across steps.

HuggingFace weight names::

    conv.conv.weight   → self.conv_weight  (hidden_size, 1, kernel_size)
    conv.in_proj.weight → self.in_proj.weight  (3*hidden_size, hidden_size)
    conv.out_proj.weight → self.out_proj.weight  (hidden_size, hidden_size)

HuggingFace reference: ``Lfm2ShortConv`` in ``modeling_lfm2.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius.components._common import INT64_MAX, Linear

if TYPE_CHECKING:
    import onnx_ir as ir


class ShortConv(nn.Module):
    """Gated causal depthwise Conv1d block (LFM2 ShortConv).

    Data flow::

        x  →  in_proj  →  [B, C, x]
                               ↓
                           B * x = Bx
                               ↓
                     causal depthwise Conv1d(Bx)
                               ↓
                          C * conv_out
                               ↓
                           out_proj  →  y

    The convolution uses ``groups=hidden_size`` (depthwise) and left-pads
    by ``kernel_size - 1`` for causal behavior during prefill.  During
    generation (single-step), the cached ``conv_state`` (B, hidden_size,
    kernel_size - 1) replaces left-padding.

    Args:
        hidden_size: Model hidden dimension.
        kernel_size: Convolution kernel size (typically 3-4).
        bias: Whether conv/proj layers include bias.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self._kernel_size = kernel_size
        self._bias = bias

        # in_proj: hidden_size → 3 * hidden_size (B, C, x)
        self.in_proj = Linear(hidden_size, 3 * hidden_size, bias=bias)
        # out_proj: hidden_size → hidden_size
        self.out_proj = Linear(hidden_size, hidden_size, bias=bias)

        # Depthwise conv weight: (hidden_size, 1, kernel_size)
        # Stored as a plain parameter (not a sub-module) to match HF naming
        self.conv_weight = nn.Parameter(
            shape=[hidden_size, 1, kernel_size],
        )
        if bias:
            self.conv_bias = nn.Parameter(shape=[hidden_size])
        else:
            self.conv_bias = None

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value | None = None,
    ) -> tuple[ir.Value, ir.Value]:
        """Forward pass.

        Args:
            hidden_states: (B, S, hidden_size) input tensor.
            conv_state: (B, hidden_size, kernel_size-1) past conv state,
                or None for first step / prefill.

        Returns:
            (output, new_conv_state):
                output: (B, S, hidden_size)
                new_conv_state: (B, hidden_size, kernel_size-1)
        """
        # in_proj → (B, S, 3*hidden_size) → transpose to (B, 3*hidden_size, S)
        projected = self.in_proj(op, hidden_states)
        # Transpose to channels-first: (B, 3*H, S)
        projected = op.Transpose(projected, perm=[0, 2, 1])

        # Split into B, C, x along dim=1: each (B, hidden_size, S)
        h = self._hidden_size
        b_gate = op.Slice(
            projected,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[h]),
            op.Constant(value_ints=[1]),
        )
        c_gate = op.Slice(
            projected,
            op.Constant(value_ints=[h]),
            op.Constant(value_ints=[2 * h]),
            op.Constant(value_ints=[1]),
        )
        x = op.Slice(
            projected,
            op.Constant(value_ints=[2 * h]),
            op.Constant(value_ints=[3 * h]),
            op.Constant(value_ints=[1]),
        )

        # Bx = B * x  (element-wise gating)
        bx = op.Mul(b_gate, x)  # (B, hidden_size, S)

        # Causal depthwise Conv1d on Bx
        # Left-pad by (kernel_size - 1) for causal convolution
        pad_left = self._kernel_size - 1
        if conv_state is not None:
            # Inference: prepend cached state (replaces left-padding)
            # conv_state: (B, hidden_size, kernel_size-1)
            bx_padded = op.Concat(conv_state, bx, axis=2)
        else:
            # Prefill or no cache: explicit left-padding
            pads = op.Constant(value_ints=[0, 0, pad_left, 0, 0, 0])
            bx_padded = op.Pad(bx, pads, mode="constant")

        # Extract new conv_state: last (kernel_size - 1) timesteps of bx_padded
        # bx_padded shape: (B, hidden_size, S + kernel_size - 1)
        new_conv_state = op.Slice(
            bx_padded,
            op.Constant(value_ints=[-(self._kernel_size - 1)]),
            op.Constant(value_ints=[INT64_MAX]),
            op.Constant(value_ints=[2]),
        )

        # Depthwise Conv1d: groups=hidden_size
        conv_inputs = [bx_padded, self.conv_weight]
        if self.conv_bias is not None:
            conv_inputs.append(self.conv_bias)

        conv_out = op.Conv(
            *conv_inputs,
            group=self._hidden_size,
        )

        # Output gating: y = C * conv_out
        y = op.Mul(c_gate, conv_out)  # (B, hidden_size, S)

        # Transpose back to (B, S, hidden_size) for out_proj
        y = op.Transpose(y, perm=[0, 2, 1])
        y = self.out_proj(op, y)

        return y, new_conv_state
