# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import math
from typing import NamedTuple

import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._common import Linear
from mobius.components._rms_norm import OffsetRMSNorm, RMSNorm
from mobius.components._rotary_embedding import apply_rotary_pos_emb


class GQAContext(NamedTuple):
    """Context for direct ``com.microsoft::GroupQueryAttention`` emission.

    Created once per graph by :class:`~mobius.models.base.TextModel` when the
    active EP (from :func:`~mobius._build_context.ep_capabilities`) supports
    GQA for the current build dtype.  Passed through DecoderLayer as the
    ``attention_bias`` argument so that :class:`Attention` can detect it and
    emit ``GroupQueryAttention`` directly instead of the generic
    ``Attention + RotaryEmbedding`` sequence.

    Using this context skips the post-hoc
    :class:`~mobius.rewrite_rules._group_query_attention.RotaryAttentionToGQA`
    rewrite rule for models that use the standard :class:`TextModel` backbone.
    The rewrite rule remains as a fallback for models with non-standard RoPE
    (e.g. Qwen3.5 with 3D mRoPE).

    Fields:
        seqlens_k: Per-batch last valid KV index ``[batch]`` INT32.
            Computed as ``ReduceSum(attention_mask, axis=1) - 1``; this is a
            0-based index into the valid KV tokens, not the KV length itself.
        total_seq_len: Scalar total sequence length INT32.
            Computed as ``Shape(attention_mask)[1]``.
        cos_cache: Full cosine RoPE table ``[max_seq_len, rotary_dim]`` FLOAT.
            Taken directly from the model's ``rotary_emb.cos_cache`` parameter.
        sin_cache: Full sine RoPE table ``[max_seq_len, rotary_dim]`` FLOAT.
            Taken directly from the model's ``rotary_emb.sin_cache`` parameter.
    """

    seqlens_k: ir.Value
    total_seq_len: ir.Value
    cos_cache: ir.Value
    sin_cache: ir.Value


class StaticCacheState(NamedTuple):
    """Static KV cache state for opset-24 TensorScatter + Attention.

    When used, the caller manages the KV cache statically. New key/value
    tokens are scattered into the pre-allocated cache via TensorScatter,
    and the full cache is passed to the Attention op with
    ``nonpad_kv_seqlen`` to indicate valid token counts.

    Fields:
        key_cache: Pre-allocated key cache [B, max_seq, kv_hidden] 3D.
        value_cache: Pre-allocated value cache [B, max_seq, kv_hidden] 3D.
        write_indices: Position to write new tokens [B] int64.
        nonpad_kv_seqlen: Valid KV length per batch entry [B] int64.
    """

    key_cache: ir.Value
    value_cache: ir.Value
    write_indices: ir.Value
    nonpad_kv_seqlen: ir.Value


def _apply_attention(
    op: builder.OpBuilder,
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    attn_mask: ir.Value | None,
    past_key: ir.Value | None,
    past_value: ir.Value | None,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    scale: float,
    softcap: float = 0.0,
    static_cache: StaticCacheState | None = None,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """Apply the ONNX Attention op with internal or static KV cache.

    Dynamic cache mode (``static_cache is None``):
        Concatenates ``past_key``/``past_value`` with new key/value
        internally.  Uses ``is_causal=1`` so callers only need to provide
        a bool padding mask (not a full causal+padding float bias).
        Returns ``(attn_output, present_key, present_value)``.

    Static cache mode (``static_cache is not None``):
        Scatters new key/value into the static cache via TensorScatter,
        then attends over the full cache using ``nonpad_kv_seqlen``.
        Also uses ``is_causal=1``.
        Returns ``(attn_output, updated_key_cache, updated_value_cache)``.

    Note:
        Both paths set ``is_causal=1`` on the Attention op, which enables
        built-in causal masking. This means ``attn_mask`` should encode
        only padding information (as a bool mask), not causality.

    Note:
        In static cache mode, RoPE must be applied to key *before*
        calling this function so that cached entries have RoPE baked in.
    """
    if static_cache is not None:
        # Scatter new K/V into the pre-allocated cache at write_indices.
        # write_indices [B] is a START POSITION per batch item, not
        # per-token.  TensorScatter writes:
        #   cache[b, write_indices[b] + t] = update[b, t]
        # for all t in range(seq_len).  This handles both prefill
        # (write_indices=0, seq_len=N) and decode (write_indices=N,
        # seq_len=1) with the same graph.
        updated_k = op.TensorScatter(
            static_cache.key_cache,
            key,
            static_cache.write_indices,
            axis=1,
        )  # [B, max_seq, kv_hidden]
        updated_v = op.TensorScatter(
            static_cache.value_cache,
            value,
            static_cache.write_indices,
            axis=1,
        )  # [B, max_seq, kv_hidden]

        # Attend over the full cache.  We pass None for attn_mask and use
        # is_causal=1 instead — the Attention op handles causal + padding
        # masking internally via is_causal + nonpad_kv_seqlen.  Using
        # create_attention_bias() here would produce incorrect causality
        # during prefill because it cannot represent the relationship
        # between query positions and the full cache length.
        #
        # NOTE: The ONNX Attention spec supports attn_mask alongside
        # nonpad_kv_seqlen for custom masking (e.g., user-defined masks
        # beyond causal + padding).  Currently we rely on is_causal=1 +
        # nonpad_kv_seqlen for standard LLM causal + padding masking.
        # TODO(titaiwang): Support user-provided attn_mask in external
        # cache mode for advanced use cases (e.g., prefix masking,
        # document boundaries in batched inference).
        # TODO(titaiwang): Support sliding window (circular cache mode)
        # with static cache for long-context models that use local
        # attention windows.
        attn_output, _, _ = op.Attention(
            query,
            updated_k,
            updated_v,
            None,  # no attn_mask — is_causal handles masking
            None,  # no past_key (full cache is already provided)
            None,  # no past_value
            static_cache.nonpad_kv_seqlen,
            q_num_heads=num_attention_heads,
            kv_num_heads=num_key_value_heads,
            scale=scale,
            softcap=softcap,
            is_causal=1,
            _outputs=3,
        )
        return attn_output, updated_k, updated_v

    # Dynamic cache mode: standard Attention with past KV concatenation.
    # is_causal=1 enables built-in causal masking, eliminating the need for
    # callers to embed causality in the attn_mask. This allows attn_mask to
    # be a simple bool padding mask, which unlocks Flash Attention eligibility.
    attn_output, present_key, present_value = op.Attention(
        query,
        key,
        value,
        attn_mask,
        past_key,
        past_value,
        q_num_heads=num_attention_heads,
        kv_num_heads=num_key_value_heads,
        scale=scale,
        softcap=softcap,
        is_causal=1,
        _outputs=3,
    )
    return attn_output, present_key, present_value


class Attention(nn.Module):
    """Multi-head attention module using ONNX ops.

    Supports GQA (grouped query attention), optional Q/K normalization,
    and optional rotary position embeddings.

    Args:
        config: Architecture configuration.
        rms_norm_class: Norm class for Q/K normalization (default: RMSNorm).
        scale: Custom attention scale factor (default: 1/sqrt(head_dim)).
        linear_class: Factory callable ``(in_features, out_features, bias=...)``
            for creating projection layers. Defaults to ``Linear``. Pass a
            ``LoRALinear`` factory for LoRA-adapted attention.
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
        self.rotary_embedding_dim = (
            0
            if math.isclose(config.partial_rotary_factor, 1.0)
            else int(self.head_dim * config.partial_rotary_factor)
        )
        self._rope_interleave = config.rope_interleave
        # Gemma2-style logit soft-capping; 0.0 means disabled.
        self._softcap = getattr(config, "attn_logit_softcapping", 0.0) or 0.0

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
        attention_bias: ir.Value | GQAContext | None,
        position_embeddings: tuple | None = None,
        past_key_value: tuple | None = None,
        static_cache: StaticCacheState | None = None,
    ):
        query_states = self.q_proj(op, hidden_states)
        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)

        if self.q_norm is not None and self.k_norm is not None:
            if self._qk_norm_full:
                # Apply norm on 3D tensor (across all heads)
                query_states = self.q_norm(op, query_states)
                key_states = self.k_norm(op, key_states)
            else:
                # Apply norm per-head on 4D tensor
                query_states = op.Reshape(query_states, [0, 0, -1, self.head_dim])
                key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
                query_states = self.q_norm(op, query_states)
                key_states = self.k_norm(op, key_states)
                query_states = op.Reshape(query_states, [0, 0, -1])
                key_states = op.Reshape(key_states, [0, 0, -1])

        # Direct GroupQueryAttention path: skip external RoPE, fuse everything.
        if isinstance(attention_bias, GQAContext):
            return self._forward_gqa(
                op,
                query_states,
                key_states,
                value_states,
                attention_bias,
                past_key_value,
            )

        # Apply rotary position embeddings (skip when not provided)
        if position_embeddings is not None:
            # Apply llama_4_attn_scale if present (Ministral3/Mistral4).
            # The scale is computed from position_ids by the RoPE module
            # and passed as the 3rd element of position_embeddings.
            # Applied BEFORE RoPE so the graph keeps the
            # RotaryEmbedding → Attention pattern that the
            # RotaryAttentionToGQA rewrite rule matches. Scaling
            # commutes with rotation: scale(RoPE(q)) == RoPE(scale(q)).
            if len(position_embeddings) > 2:
                attn_scale = position_embeddings[2]
                query_states = op.Mul(query_states, attn_scale)

            query_states = apply_rotary_pos_emb(
                op,
                x=query_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_attention_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )
            key_states = apply_rotary_pos_emb(
                op,
                x=key_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_key_value_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )

        attn_output, present_key, present_value = _apply_attention(
            op,
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key_value[0] if past_key_value is not None else None,
            past_key_value[1] if past_key_value is not None else None,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale=self.scaling,
            softcap=self._softcap,
            static_cache=static_cache,
        )

        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)

    def _forward_gqa(
        self,
        op: builder.OpBuilder,
        query_states: ir.Value,
        key_states: ir.Value,
        value_states: ir.Value,
        gqa_ctx: GQAContext,
        past_key_value: tuple | None,
    ):
        """Emit ``com.microsoft::GroupQueryAttention`` directly.

        Called from :meth:`forward` when ``attention_bias`` is a
        :class:`GQAContext`.  Bypasses the external
        :class:`~mobius.components._rotary_embedding.RotaryEmbeddingBase`
        forward pass and the post-hoc
        :class:`~mobius.rewrite_rules._group_query_attention.RotaryAttentionToGQA`
        rewrite rule; RoPE is handled by the ``do_rotary=1`` attribute instead.

        Returns ``(attn_output, (present_key, present_value))`` in the same
        shape as the standard :meth:`forward` path.
        """
        past_key = past_key_value[0] if past_key_value is not None else None
        past_value = past_key_value[1] if past_key_value is not None else None

        gqa_attrs: dict = {
            "num_heads": self.num_attention_heads,
            "kv_num_heads": self.num_key_value_heads,
            "scale": self.scaling,
            "do_rotary": 1,
            "rotary_interleaved": int(self._rope_interleave),
        }
        if self._softcap:
            gqa_attrs["softcap"] = self._softcap
        if self.rotary_embedding_dim:
            # Partial RoPE: only rotate the first rotary_embedding_dim elements.
            gqa_attrs["rotary_embedding_dim"] = self.rotary_embedding_dim

        # Emit GroupQueryAttention: RoPE + attention + KV cache in one op.
        # Outputs: (attn_output [B, S, hidden], present_key, present_value)
        attn_out, present_key, present_value = op.GroupQueryAttention(
            query_states,  # [B, S, num_heads * head_dim]
            key_states,  # [B, S, kv_heads * head_dim]
            value_states,  # [B, S, kv_heads * head_dim]
            past_key,  # [B, kv_heads, past_S, head_dim] or None
            past_value,  # [B, kv_heads, past_S, head_dim] or None
            gqa_ctx.seqlens_k,  # [B] INT32
            gqa_ctx.total_seq_len,  # scalar INT32
            gqa_ctx.cos_cache,  # [max_seq, rotary_dim]
            gqa_ctx.sin_cache,  # [max_seq, rotary_dim]
            _domain="com.microsoft",
            _outputs=3,
            **gqa_attrs,
        )

        attn_out = self.o_proj(op, attn_out)
        return attn_out, (present_key, present_value)


class Qwen35Attention(nn.Module):
    """Multi-head attention with output gating for Qwen3.5.

    Differences from base Attention:
    - Q projection is doubled to produce both Q and a gating signal
    - Per-head Q/K RMSNorm with +1 offset (OffsetRMSNorm)
    - Partial RoPE (rotary_embedding_dim < head_dim)
    - Output gating: attn_output * sigmoid(gate)
    """

    def __init__(
        self,
        config: ArchitectureConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.rotary_embedding_dim = (
            0
            if math.isclose(config.partial_rotary_factor, 1.0)
            else int(self.head_dim * config.partial_rotary_factor)
        )
        self._rope_interleave = config.rope_interleave

        q_dim = self.num_attention_heads * self.head_dim
        self.q_proj = Linear(
            self.hidden_size,
            q_dim * 2,
            bias=config.attn_qkv_bias,
        )
        self.k_proj = Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.v_proj = Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.o_proj = Linear(
            q_dim,
            self.hidden_size,
            bias=config.attn_o_bias,
        )

        self.q_norm = OffsetRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = OffsetRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
        static_cache: StaticCacheState | None = None,
    ):
        # Q projection (doubled) → split into Q and gate per head
        q_gate = self.q_proj(op, hidden_states)
        # Reshape to per-head view so split separates Q/gate within each head
        q_gate = op.Reshape(
            q_gate,
            [0, 0, self.num_attention_heads, self.head_dim * 2],
        )
        query_states, gate = op.Split(q_gate, num_outputs=2, axis=-1, _outputs=2)
        gate = op.Reshape(gate, [0, 0, -1])

        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)

        # Per-head RMSNorm on 4D tensors (query_states already 4D)
        key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
        query_states = self.q_norm(op, query_states)
        key_states = self.k_norm(op, key_states)
        query_states = op.Reshape(query_states, [0, 0, -1])
        key_states = op.Reshape(key_states, [0, 0, -1])

        # Apply rotary position embeddings
        query_states = apply_rotary_pos_emb(
            op,
            x=query_states,
            position_embeddings=position_embeddings,
            num_heads=self.num_attention_heads,
            rotary_embedding_dim=self.rotary_embedding_dim,
            interleaved=self._rope_interleave,
        )
        key_states = apply_rotary_pos_emb(
            op,
            x=key_states,
            position_embeddings=position_embeddings,
            num_heads=self.num_key_value_heads,
            rotary_embedding_dim=self.rotary_embedding_dim,
            interleaved=self._rope_interleave,
        )

        attn_output, present_key, present_value = _apply_attention(
            op,
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key_value[0] if past_key_value is not None else None,
            past_key_value[1] if past_key_value is not None else None,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale=self.scaling,
            static_cache=static_cache,
        )

        # Output gating: attn_output * sigmoid(gate)
        attn_output = op.Mul(attn_output, op.Sigmoid(gate))

        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)
