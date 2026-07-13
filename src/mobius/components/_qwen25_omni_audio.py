# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen2.5-Omni audio encoder components.

Packed bidirectional transformer layers with LayerNorm.

Reference: Transformers
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius._build_context import get_build_dtype
from mobius.components._common import LayerNorm, Linear


class Qwen25OmniAudioAttention(nn.Module):
    """Bidirectional multi-head attention for Qwen2_5Omni audio encoder.

    Unlike WhisperAttention, all projections (Q, V, Out) have bias and K does not have bias.
    No causal masking — the encoder uses full bidirectional attention.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.q_proj = Linear(d_model, d_model, bias=True)
        self.k_proj = Linear(d_model, d_model, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=True)
        self.out_proj = Linear(d_model, d_model, bias=True)
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
    ):
        """Bidirectional self-attention.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        seq_len = op.Shape(hidden_states, start=0, end=1)
        packed_shape = op.Concat(seq_len, [self._num_heads, self._head_dim], axis=0)
        q = op.Reshape(self.q_proj(op, hidden_states), packed_shape)
        k = op.Reshape(self.k_proj(op, hidden_states), packed_shape)
        v = op.Reshape(self.v_proj(op, hidden_states), packed_shape)

        # Build the block-diagonal mask represented by HF's cu_seqlens.
        positions = op.Range(0, op.Squeeze(seq_len, [0]), 1)
        segment_ids = op.Sub(
            op.ReduceSum(
                op.Cast(
                    op.GreaterOrEqual(
                        op.Unsqueeze(positions, [1]),
                        op.Unsqueeze(op.Cast(cu_seqlens, to=7), [0]),
                    ),
                    to=7,
                ),
                [1],
                keepdims=False,
            ),
            1,
        )
        same_segment = op.Equal(
            op.Unsqueeze(segment_ids, [1]),
            op.Unsqueeze(segment_ids, [0]),
        )
        attention_bias = op.Where(
            same_segment,
            op.CastLike(0.0, q),
            op.CastLike(-1e9, q),
        )
        attention_bias = op.Unsqueeze(attention_bias, [0, 1])

        q = op.Unsqueeze(op.Transpose(q, perm=[1, 0, 2]), [0])
        k = op.Unsqueeze(op.Transpose(k, perm=[1, 0, 2]), [0])
        v = op.Unsqueeze(op.Transpose(v, perm=[1, 0, 2]), [0])
        attn_output = op.Attention(
            q,
            k,
            v,
            attention_bias,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            scale=float(self._head_dim**-0.5),
        )
        attn_output = op.Transpose(op.Squeeze(attn_output, [0]), perm=[1, 0, 2])
        attn_output = op.Reshape(attn_output, op.Concat(seq_len, [-1], axis=0))
        return self.out_proj(op, attn_output)


class Qwen25OmniAudioEncoderLayer(nn.Module):
    """Qwen25-Omni audio encoder layer.

    Pre-norm pattern:  LayerNorm → self-attn → residual
    → LayerNorm → FFN → residual.
    Uses GELU activation in the FFN.

    Huggingface class: ``Qwen2_5OmniAudioEncoder``
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn = Qwen25OmniAudioAttention(d_model, num_heads)
        self.self_attn_layer_norm = LayerNorm(d_model, eps=eps)
        self.fc1 = Linear(d_model, ffn_dim, bias=True)
        self.fc2 = Linear(ffn_dim, d_model, bias=True)
        self.final_layer_norm = LayerNorm(d_model, eps=eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
    ):
        """Pre-norm encoder layer with bidirectional attention.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            hidden_states: (batch, seq_len, d_model)
        """
        # Self-attention with pre-norm and residual
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states, cu_seqlens)
        hidden_states = op.Add(residual, hidden_states)

        # FFN with pre-norm, GELU, and residual
        residual = hidden_states
        hidden_states = self.final_layer_norm(op, hidden_states)
        hidden_states = self.fc1(op, hidden_states)
        hidden_states = op.Gelu(hidden_states)
        hidden_states = self.fc2(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        if get_build_dtype() == ir.DataType.FLOAT16:
            hidden_states = op.Clip(
                hidden_states,
                op.CastLike(-64504.0, hidden_states),
                op.CastLike(64504.0, hidden_states),
            )

        return hidden_states
