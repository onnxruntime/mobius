# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""GroupQueryAttention component for ORT GenAI-compatible model generation.

Emits the ``com.microsoft::GroupQueryAttention`` contrib op directly,
instead of the standard ONNX ``Attention`` op.  This produces models
compatible with the onnxruntime-genai runtime, which uses in-place KV
cache updates (``past_present_share_buffer``) for efficient generation.

When ``do_rotary=True``, RoPE is fused inside the GQA op using
``cos_cache`` / ``sin_cache`` graph initializers — no external
``RotaryEmbedding`` nodes are needed and ``position_ids`` is not a
graph input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._common import Linear
from mobius.components._rms_norm import RMSNorm

if TYPE_CHECKING:
    import onnx_ir as ir


class GQAContext(NamedTuple):
    """Graph-level context for GroupQueryAttention layers.

    Created once by the task and passed through the model to each
    GQAAttention instance.  All layers share the same ``seqlens_k``
    and ``total_seq_len`` values (computed from ``attention_mask``).

    Fields:
        seqlens_k: Per-batch actual KV length ``[batch]`` INT32.
            Computed as ``ReduceSum(attention_mask, axis=1) - 1``.
        total_seq_len: Scalar total sequence length INT32.
            Computed as ``Shape(attention_mask)[1]``.
        cos_cache: Full cosine RoPE table ``[max_seq_len, rotary_dim]``.
        sin_cache: Full sine RoPE table ``[max_seq_len, rotary_dim]``.
    """

    seqlens_k: ir.Value
    total_seq_len: ir.Value
    cos_cache: ir.Value
    sin_cache: ir.Value


class GQAAttention(nn.Module):
    """Multi-head attention using com.microsoft::GroupQueryAttention.

    Drop-in replacement for :class:`Attention` that targets the ORT GenAI
    runtime.  Supports GQA (grouped query attention), optional QK norm,
    and fused rotary embeddings.

    The GQA op handles RoPE internally when ``cos_cache``/``sin_cache``
    are provided via :class:`GQAContext`, eliminating the need for
    external ``RotaryEmbedding`` nodes.

    Args:
        config: Architecture configuration.
        rms_norm_class: Norm class for Q/K normalization (default: RMSNorm).
        scale: Custom attention scale factor (default: 1/sqrt(head_dim)).
        linear_class: Factory for projection layers (default: Linear).
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        rms_norm_class: type[nn.Module] | None = None,
        scale: float | None = None,
        linear_class: type | None = None,
    ):
        super().__init__()
        if linear_class is None:
            linear_class = Linear

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = scale if scale is not None else self.head_dim**-0.5
        self._rope_interleave = config.rope_interleave
        self._window_size = getattr(config, "sliding_window", None) or -1
        self._softcap = getattr(config, "attn_logit_softcapping", None) or 0.0

        self.q_proj = linear_class(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.k_proj = linear_class(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.v_proj = linear_class(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.o_proj = linear_class(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=config.attn_o_bias,
        )

        if config.attn_qk_norm:
            rms_norm_class = RMSNorm if rms_norm_class is None else rms_norm_class
            self._qk_norm_full = config.attn_qk_norm_full
            if self._qk_norm_full:
                self.q_norm = rms_norm_class(
                    self.num_attention_heads * self.head_dim, eps=config.rms_norm_eps
                )
                self.k_norm = rms_norm_class(
                    self.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps
                )
            else:
                self.q_norm = rms_norm_class(self.head_dim, eps=config.rms_norm_eps)
                self.k_norm = rms_norm_class(self.head_dim, eps=config.rms_norm_eps)
        else:
            self._qk_norm_full = False
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        gqa_context: GQAContext,
        past_key_value: tuple | None = None,
    ):
        """Forward pass emitting com.microsoft::GroupQueryAttention.

        Args:
            op: ONNX op builder.
            hidden_states: ``[batch, seq_len, hidden_size]``.
            gqa_context: Shared GQA context with seqlens_k, total_seq_len,
                cos_cache, sin_cache.
            past_key_value: Optional ``(past_key, past_value)`` tuple.

        Returns:
            ``(attn_output, (present_key, present_value))``
        """
        query_states = self.q_proj(op, hidden_states)
        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)

        # Optional QK norm (applied before GQA, which handles RoPE)
        if self.q_norm is not None and self.k_norm is not None:
            if self._qk_norm_full:
                query_states = self.q_norm(op, query_states)
                key_states = self.k_norm(op, key_states)
            else:
                query_states = op.Reshape(query_states, [0, 0, -1, self.head_dim])
                key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
                query_states = self.q_norm(op, query_states)
                key_states = self.k_norm(op, key_states)
                query_states = op.Reshape(query_states, [0, 0, -1])
                key_states = op.Reshape(key_states, [0, 0, -1])

        past_key = past_key_value[0] if past_key_value is not None else None
        past_value = past_key_value[1] if past_key_value is not None else None

        # Emit com.microsoft::GroupQueryAttention
        # GQA handles RoPE internally via cos_cache/sin_cache and do_rotary=1.
        # It also handles KV cache concatenation and causal masking via
        # seqlens_k/total_seq_len.
        attn_output, present_key, present_value = op.GroupQueryAttention(
            query_states,
            key_states,
            value_states,
            past_key,
            past_value,
            gqa_context.seqlens_k,
            gqa_context.total_seq_len,
            gqa_context.cos_cache,
            gqa_context.sin_cache,
            _domain="com.microsoft",
            num_heads=self.num_attention_heads,
            kv_num_heads=self.num_key_value_heads,
            scale=self.scaling,
            do_rotary=1,
            rotary_interleaved=1 if self._rope_interleave else 0,
            local_window_size=self._window_size,
            softcap=self._softcap,
            _outputs=3,
        )

        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)
