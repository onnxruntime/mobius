# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SANM (Self-Attention with Normalization and Memory) components for Fun-ASR.

Provides encoder building blocks for the Fun-ASR Nano speech model:

- ``SANMAttention``: Multi-head self-attention with an FSMN (Feedforward
  Sequential Memory Network) memory block applied to values before the
  attention computation.
- ``SANMFFN``: Simple two-layer feed-forward network with ReLU activation.
- ``SANMEncoderLayer``: Pre-norm encoder layer combining SANM attention,
  FSMN memory, and FFN with residual connections.

The FSMN block applies a depthwise 1-D convolution over the value
projection, giving the model a local-context memory mechanism that
complements the global self-attention.

Weight attribute names are chosen to match the Fun-ASR checkpoint naming
so that ``preprocess_weights`` needs minimal renames.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._common import LayerNorm, Linear

if TYPE_CHECKING:
    import onnx_ir as ir


class _FSMNBlock(nn.Module):
    """FSMN depthwise conv1d block.

    Applies a depthwise 1-D convolution over the input (channels-first)
    using a learnable weight.  Weight shape: ``[n_feat, 1, kernel_size]``
    (groups = n_feat for depthwise convolution).

    The forward method must be called through ``self(op, x)`` so that
    onnxscript qualifies the weight parameter name correctly in the graph.
    """

    def __init__(self, n_feat: int, kernel_size: int):
        super().__init__()
        self.weight = nn.Parameter([n_feat, 1, kernel_size])
        self._n_feat = n_feat
        self._kernel_size = kernel_size

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        """Apply depthwise Conv1d.

        Args:
            x: ``(batch, n_feat, time)`` — channels-first input.

        Returns:
            ``(batch, n_feat, time)`` — convolved output.
        """
        left_pad = (self._kernel_size - 1) // 2
        right_pad = self._kernel_size - 1 - left_pad
        return op.Conv(
            x,
            self.weight,
            kernel_shape=[self._kernel_size],
            pads=[left_pad, right_pad],
            group=self._n_feat,
        )


class SANMAttention(nn.Module):
    """SANM self-attention with FSMN memory block.

    Forward pass:
        1. Fused QKV projection → split into Q, K, V  [B, T, n_feat] each
        2. FSMN memory: depthwise Conv1d on V + residual → fsmn_memory
        3. Scaled dot-product attention on Q, K, V (original V, NOT fsmn V)
        4. Output = linear_out(attention_output + fsmn_memory)

    The FSMN memory is ADDED to the attention output, not fed as V
    into the attention computation. This matches the reference:
    ``return att_outs + fsmn_memory``

    Args:
        in_size: Input feature dimension (may differ from out_size).
        out_size: Output feature / attention hidden dimension (``n_feat``).
        n_heads: Number of attention heads.
        kernel_size: FSMN depthwise conv kernel size.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        n_heads: int,
        kernel_size: int,
    ):
        super().__init__()
        # Fused Q/K/V projection: [in_size] → [3 * out_size]
        self.linear_q_k_v = Linear(in_size, out_size * 3, bias=True)

        # FSMN depthwise conv1d on values
        # Weight shape: [n_feat, 1, kernel_size] (groups=n_feat)
        self.fsmn_block = _FSMNBlock(out_size, kernel_size)

        # Output projection: [out_size] → [out_size]
        self.linear_out = Linear(out_size, out_size, bias=True)

        self._n_heads = n_heads
        self._head_dim = out_size // n_heads

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # hidden_states: [B, T, in_size]

        # Fused QKV → [B, T, 3 * out_size]
        qkv = self.linear_q_k_v(op, hidden_states)
        # Split into Q, K, V each [B, T, out_size]
        q, k, v = op.Split(qkv, axis=-1, num_outputs=3, _outputs=3)

        # --- FSMN path (parallel): Conv1d on V with residual ---
        v_t = op.Transpose(v, perm=[0, 2, 1])  # [B, n_feat, T]
        v_conv = self.fsmn_block(op, v_t)  # [B, n_feat, T]
        v_conv = op.Transpose(v_conv, perm=[0, 2, 1])  # [B, T, n_feat]
        fsmn_memory = op.Add(v_conv, v)  # [B, T, out_size]

        # --- Attention path (parallel): uses ORIGINAL v ---
        scale = self._head_dim**-0.5
        attn_output, _, _ = op.Attention(
            q,
            k,
            v,
            q_num_heads=self._n_heads,
            kv_num_heads=self._n_heads,
            scale=scale,
            _outputs=3,
        )  # [B, T, out_size]
        attn_output = self.linear_out(op, attn_output)  # [B, T, out_size]

        # Sum parallel paths: attention + FSMN memory
        return op.Add(attn_output, fsmn_memory)  # [B, T, out_size]


class SANMFFN(nn.Module):
    """Simple two-layer FFN with ReLU activation for SANM encoder.

    Forward: ``x → w_1 → ReLU → w_2``

    Args:
        hidden_size: Input and output dimension.
        ffn_dim: Inner (intermediate) dimension.
    """

    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.w_1 = Linear(hidden_size, ffn_dim, bias=True)
        self.w_2 = Linear(ffn_dim, hidden_size, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        # x: [B, T, hidden_size]
        return self.w_2(op, op.Relu(self.w_1(op, x)))  # [B, T, hidden_size]


class SANMEncoderLayer(nn.Module):
    """SANM encoder layer: pre-norm attention + FSMN + pre-norm FFN.

    Architecture::

        LayerNorm(in_size) → SANMAttention → Residual → LayerNorm(out_size) → FFN → Residual

    When ``in_size != out_size`` (e.g. the first encoder layer projecting
    from 560 → 512), the residual connection around the attention block is
    skipped because the dimensions are incompatible.

    Args:
        in_size: Input feature dimension.
        out_size: Output / hidden dimension for attention and FFN.
        n_heads: Number of attention heads.
        ffn_dim: FFN intermediate dimension.
        kernel_size: FSMN depthwise conv kernel size.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        n_heads: int,
        ffn_dim: int,
        kernel_size: int,
    ):
        super().__init__()
        self.norm1 = LayerNorm(in_size)
        self.self_attn = SANMAttention(in_size, out_size, n_heads, kernel_size)
        self.norm2 = LayerNorm(out_size)
        self.feed_forward = SANMFFN(out_size, ffn_dim)
        # When in_size != out_size the attention changes dimensionality,
        # so the residual around the attention block must be skipped.
        self._has_residual = in_size == out_size

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # hidden_states: [B, T, in_size]

        # Pre-norm attention with optional residual
        residual = hidden_states
        hidden_states = self.norm1(op, hidden_states)  # [B, T, in_size]
        hidden_states = self.self_attn(op, hidden_states)  # [B, T, out_size]
        if self._has_residual:
            hidden_states = op.Add(hidden_states, residual)

        # Pre-norm FFN with residual
        residual = hidden_states
        hidden_states = self.norm2(op, hidden_states)  # [B, T, out_size]
        hidden_states = self.feed_forward(op, hidden_states)  # [B, T, out_size]
        hidden_states = op.Add(hidden_states, residual)

        return hidden_states  # [B, T, out_size]
