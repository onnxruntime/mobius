# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Encoder-decoder attention module for seq2seq models (BART, T5, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components._common import Embedding, Linear

if TYPE_CHECKING:
    import onnx_ir as ir


class EncoderDecoderAttention(nn.Module):
    """Multi-head attention for encoder-decoder models.

    Supports self-attention and cross-attention (via ``key_value_states``),
    optional relative position bias (T5-style), configurable causality,
    and KV cache for autoregressive decoding.

    Named by pattern, not model: consolidates the identical attention patterns
    used in BART, T5, and similar encoder-decoder architectures.

    Args:
        config: Architecture configuration.
        is_causal: Whether to use causal (unidirectional) attention.
        has_relative_attention_bias: Whether to include T5-style learned
            relative position bias.
        bias: Whether projection layers use bias. Default True (BART-style).
            Pass False for T5-style.
        scale: Attention score scale factor. Defaults to ``1/sqrt(head_dim)``.
            T5 uses ``scale=1.0`` (no scaling).
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        *,
        is_causal: bool = False,
        has_relative_attention_bias: bool = False,
        bias: bool = True,
        scale: float | None = None,
        linear_class: type | None = None,
        use_cross_attention_cache: bool = False,
    ):
        super().__init__()
        if linear_class is None:
            linear_class = Linear
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.is_causal = is_causal
        self.use_cross_attention_cache = use_cross_attention_cache
        self._scale = scale if scale is not None else float(self.head_dim**-0.5)

        self.q_proj = linear_class(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = linear_class(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.v_proj = linear_class(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.out_proj = linear_class(
            self.num_heads * self.head_dim, self.hidden_size, bias=bias
        )

        if has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                config.relative_attention_num_buckets, self.num_heads
            )
        else:
            self.relative_attention_bias = None

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        key_value_states: ir.Value | None = None,
        attention_bias: ir.Value | None = None,
        past_key_value: tuple | None = None,
    ):
        """Forward pass.

        Args:
            op: ONNX op builder.
            hidden_states: Query source tensor.
            key_value_states: If provided, K/V are projected from this tensor
                (cross-attention). Otherwise, self-attention is performed.
            attention_bias: Optional attention bias (e.g., relative position bias).
            past_key_value: Cached (key, value) tuple for incremental decoding.

        Returns:
            Tuple of (output, (present_key, present_value)).
        """
        query_states = self.q_proj(op, hidden_states)

        if key_value_states is not None:
            # Cache-enabled cross-attention accepts an empty encoder sequence
            # during decode; other callers continue to project full encoder
            # states on every step.
            key_states = self.k_proj(op, key_value_states)
            value_states = self.v_proj(op, key_value_states)
            if past_key_value is not None:
                past_key, past_value = past_key_value
                if self.use_cross_attention_cache:
                    # Attention requires a non-empty current K/V sequence even
                    # when past is supplied. Flatten the cached head layout and
                    # concatenate it with the newly projected encoder suffix.
                    flat_shape = op.Concat(
                        op.Shape(past_key, start=0, end=1),
                        op.Shape(past_key, start=2, end=3),
                        op.Constant(value_ints=[self.num_heads * self.head_dim]),
                        axis=0,
                    )
                    flat_past_key = op.Reshape(
                        op.Transpose(past_key, perm=[0, 2, 1, 3]), flat_shape
                    )
                    flat_past_value = op.Reshape(
                        op.Transpose(past_value, perm=[0, 2, 1, 3]), flat_shape
                    )
                    key_states = op.Concat(flat_past_key, key_states, axis=1)
                    value_states = op.Concat(flat_past_value, value_states, axis=1)
                # Cross-attention presents the complete K/V sequence as current
                # input. Non-caching callers keep the historical behavior of
                # recomputing it from full encoder states.
                past_key = None
                past_value = None
            else:
                past_key = None
                past_value = None
        else:
            key_states = self.k_proj(op, hidden_states)
            value_states = self.v_proj(op, hidden_states)
            if past_key_value is not None:
                past_key, past_value = past_key_value
            else:
                past_key = None
                past_value = None

        attn_output, present_key, present_value = op.Attention(
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key,
            past_value,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            scale=self._scale,
            is_causal=1 if self.is_causal else 0,
            _outputs=3,
        )
        if key_value_states is not None:
            present_shape = op.Concat(
                op.Shape(key_states, start=0, end=1),
                op.Constant(value_ints=[self.num_heads]),
                op.Shape(key_states, start=1, end=2),
                op.Constant(value_ints=[self.head_dim]),
                axis=0,
            )
            present_key = op.Reshape(present_key, present_shape)
            present_value = op.Reshape(present_value, present_shape)

        output = self.out_proj(op, attn_output)
        return output, (present_key, present_value)
