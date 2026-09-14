# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import math
from typing import NamedTuple

import onnx_ir as ir
from onnxscript import OpBuilder, nn

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
        local_window_size: Left window size for local/sliding-window attention.
            ``-1`` means unused (full causal attention).  Maps to the
            ``local_window_size`` attribute on ``GroupQueryAttention``.
    """

    seqlens_k: ir.Value
    total_seq_len: ir.Value
    cos_cache: ir.Value
    sin_cache: ir.Value
    local_window_size: int = -1


class StaticCacheState(NamedTuple):
    """Static KV cache state for opset-24 TensorScatter + Attention.

    When used, the caller manages the KV cache statically. New key/value
    tokens are scattered into the pre-allocated cache via TensorScatter,
    and the full cache is passed to the Attention op with
    ``nonpad_kv_seqlen`` to indicate valid token counts.

    Fields:
        key_cache: Pre-allocated key cache [B, max_seq_len, kv_hidden] 3D.
        value_cache: Pre-allocated value cache [B, max_seq_len, kv_hidden] 3D.
        write_indices: Position to write new tokens [B] int64.
        nonpad_kv_seqlen: Valid KV length per batch entry [B] int64.
    """

    key_cache: ir.Value
    value_cache: ir.Value
    write_indices: ir.Value
    nonpad_kv_seqlen: ir.Value


def _apply_attention(
    op: OpBuilder,
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
    is_causal: int = 1,
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
        Uses ``is_causal=1`` when ``attn_mask`` is ``None`` (maskless
        default), or ``is_causal=0`` when ``attn_mask`` is a float additive
        bias (the bias then carries the full causal + sliding + block +
        padding mask).  The incoming ``is_causal`` argument is ignored in
        this mode; causality is derived from ``attn_mask`` presence so the
        two can never disagree.
        Returns ``(attn_output, updated_key_cache, updated_value_cache)``.

    Args:
        is_causal: Whether the Attention op applies its built-in causal
            mask (default ``1``). Set to ``0`` when ``attn_mask`` already
            bakes the FULL mask (causal + sliding + padding, and any
            bidirectional unmasking such as Gemma4's vision-block overlay)
            into a float additive bias. Leaving ``is_causal=1`` in that
            case would re-apply causality and cancel any future-position
            unmasking encoded in the bias.

    Note:
        This applies to the DYNAMIC cache path only. There, the Attention op
        defaults to ``is_causal=1`` for built-in causal masking, so
        ``attn_mask`` should encode only padding information (as a bool mask),
        not causality, unless ``is_causal=0`` is passed explicitly. In STATIC
        cache mode the incoming ``is_causal`` argument is ignored — causality
        is derived from ``attn_mask`` presence (see above).

    Note:
        ``nonpad_kv_seqlen`` (input #6) is only valid in static cache mode
        (no ``past_key``/``past_value``). ORT asserts that it cannot be
        combined with dynamic KV cache inputs.

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

        # External-cache masking.  Two modes, selected by whether the caller
        # supplied a float additive bias:
        #
        #   * attn_mask is None (default, maskless): pass None and use
        #     is_causal=1 — the Attention op derives causal + padding masking
        #     internally from is_causal + nonpad_kv_seqlen.  This is the
        #     Flash-eligible form (onnx#8068 / onnxruntime#28958).
        #   * attn_mask is not None (float-bias decoders): pass the bias and
        #     STRICTLY pair it with is_causal=0.  The bias already bakes in the
        #     FULL mask (causal + sliding + Gemma4 block overlay + padding), so
        #     leaving is_causal=1 would double-apply causality and cancel any
        #     bidirectional unmasking encoded in the bias.  This routes ORT to
        #     the MEA external-cache path (Flash is precluded by any bias).
        #
        # nonpad_kv_seqlen stays as input #6 in BOTH modes: it bounds the valid
        # KV prefix and, on the CUDA Flash path, drives the fully-masked-row
        # zero guard (LaunchZeroFullyMaskedRows).  In bias mode the additive
        # bias already encodes the same ``slot < nonpad`` validity.  The
        # cross-repo invariant is ``nonpad == write_indices + valid_token_count``
        # (the count of UNPADDED query tokens), which equals
        # ``write_indices + S_q`` only when the chunk is unpadded — S_q is the
        # PADDED chunk width.  When the chunk is unpadded, every query row keeps
        # its own diagonal slot valid, so a fully-masked (all-``dtype.min``) row
        # never arises.  With intra-prompt padding plus a sliding window,
        # however, a pad-token query row CAN fall outside every valid slot and
        # become fully masked.  In that case the CPU MEA path this bias mode uses
        # does NOT apply the Flash zero-guard: it returns a finite mean-of-V row
        # (not NaN, not exactly 0).  This finite-row behavior was empirically
        # verified on ORT 1.27 CPU MEA; it is an observed ORT-version behavior,
        # not a permanent op-spec invariant — see test_fully_masked_row_stays_finite.
        if attn_mask is not None:
            mask_arg, causal = attn_mask, 0
        else:
            mask_arg, causal = None, 1
        attn_output, _, _ = op.Attention(
            query,
            updated_k,
            updated_v,
            mask_arg,
            None,  # no past_key (full cache is already provided)
            None,  # no past_value
            static_cache.nonpad_kv_seqlen,
            q_num_heads=num_attention_heads,
            kv_num_heads=num_key_value_heads,
            scale=scale,
            softcap=softcap,
            is_causal=causal,
            _outputs=3,
        )
        return attn_output, updated_k, updated_v

    # Dynamic cache mode: standard Attention with past KV concatenation.
    # is_causal=1 enables built-in causal masking, eliminating the need for
    # callers to embed causality in the attn_mask. This allows attn_mask to
    # be a simple bool padding mask, which unlocks Flash Attention eligibility.
    #
    # NOTE: nonpad_kv_seqlen cannot be used here — ORT requires that
    # nonpad_kv_seqlen is NOT combined with past_key/past_value inputs.
    # It is only valid in static cache mode (no past_key/past_value).
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
        is_causal=is_causal,
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
        self._key_multiplier = getattr(config, "key_multiplier", 1.0)
        # NoPE models leave ``partial_rotary_factor`` as ``None``; treat as
        # the inert 1.0 for the purpose of computing ``rotary_embedding_dim``
        # (which will itself be 0, i.e. no partial-RoPE splitting). The
        # actual RoPE call sites are guarded by ``position_embeddings is not
        # None``, so this value is never consumed for NoPE models.
        prf = config.partial_rotary_factor if config.partial_rotary_factor is not None else 1.0
        self.rotary_embedding_dim = 0 if math.isclose(prf, 1.0) else int(self.head_dim * prf)
        self._rope_interleave = config.rope_interleave
        # Gemma2-style logit soft-capping; 0.0 means disabled.
        self._softcap = getattr(config, "attn_logit_softcapping", 0.0) or 0.0
        # Whether the Attention op applies its own causal mask. Subclasses that
        # feed a float additive bias already encoding causality *and* some
        # bidirectional unmasking (e.g. a PrefixLM / vision-block overlay) set
        # this to 0 so the built-in mask does not cancel that unmasking. It is a
        # static per-graph property, so it lives on the module rather than being
        # threaded through every forward signature.
        self._is_causal = 1

        self._init_qkv_projections(config, linear_class)
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

    def _init_qkv_projections(self, config: ArchitectureConfig, linear_class: type) -> None:
        """Create the default independent query, key, and value projections."""
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

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | GQAContext | None,
        position_embeddings: tuple | None = None,
        past_key_value: tuple | None = None,
        static_cache: StaticCacheState | None = None,
    ):
        query_states, key_states, value_states = self._project_qkv(op, hidden_states)
        if not math.isclose(self._key_multiplier, 1.0):
            key_states = op.Mul(key_states, self._key_multiplier)

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
                hidden_states=hidden_states,
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
            is_causal=self._is_causal,
        )

        attn_output = self._post_attention(op, attn_output, hidden_states)
        attn_output = self._project_output(op, attn_output)
        return attn_output, (present_key, present_value)

    def _project_qkv(
        self, op: OpBuilder, hidden_states: ir.Value
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        """Project hidden states into Q, K, and V tensors."""
        return (
            self.q_proj(op, hidden_states),
            self.k_proj(op, hidden_states),
            self.v_proj(op, hidden_states),
        )

    def _project_output(self, op: OpBuilder, attn_output: ir.Value) -> ir.Value:
        """Apply architecture-specific processing before the output projection."""
        return self.o_proj(op, attn_output)

    def _post_attention(
        self,
        op: OpBuilder,
        attn_output: ir.Value,
        hidden_states: ir.Value,
    ) -> ir.Value:
        """Transform the attention output before the ``o_proj`` projection.

        Extension point for architectures that post-process the attended
        values while still reusing the whole Q/K/V + RoPE + cache pipeline of
        this class. The base implementation is the identity, so it is inert
        for every standard model.

        Args:
            attn_output: Attention result ``[B, S, num_heads * head_dim]``.
            hidden_states: The (already normalized) layer input that produced
                Q/K/V, so a subclass can derive a gate from the same tensor
                HuggingFace does.
        """
        del op, hidden_states
        return attn_output

    def _forward_gqa(
        self,
        op: OpBuilder,
        query_states: ir.Value,
        key_states: ir.Value,
        value_states: ir.Value,
        gqa_ctx: GQAContext,
        past_key_value: tuple | None,
        hidden_states: ir.Value | None = None,
    ):
        """Emit ``com.microsoft::GroupQueryAttention`` directly.

        Called from :meth:`forward` when ``attention_bias`` is a
        :class:`GQAContext`.  Bypasses the external
        :class:`~mobius.components._rotary_embedding.RotaryEmbeddingBase`
        forward pass and the post-hoc
        :class:`~mobius.rewrite_rules._group_query_attention.RotaryAttentionToGQA`
        rewrite rule; RoPE is handled by the ``do_rotary=1`` attribute instead.

        ``hidden_states`` is only forwarded to :meth:`_post_attention`; it is
        optional so that existing positional callers keep working.

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
        if gqa_ctx.local_window_size > 0:
            gqa_attrs["local_window_size"] = gqa_ctx.local_window_size

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

        if hidden_states is not None:
            attn_out = self._post_attention(op, attn_out, hidden_states)
        attn_out = self._project_output(op, attn_out)
        return attn_out, (present_key, present_value)


class FusedQKVAttention(Attention):
    """Attention whose checkpoint stores one contiguous Q/K/V projection.

    The projection rows are ordered ``[Q | K | V]``. An optional symmetric
    activation clamp is applied to the fused output before it is split, so
    callers can represent architectures where clipping is part of the fused
    projection contract rather than an independent per-projection transform.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        *,
        clamp: float | None = None,
        linear_class: type | None = None,
    ):
        super().__init__(config, linear_class=linear_class)
        self._clamp = clamp

    def _init_qkv_projections(self, config: ArchitectureConfig, linear_class: type) -> None:
        self._q_width = self.num_attention_heads * self.head_dim
        self._kv_width = self.num_key_value_heads * self.head_dim
        self.qkv_proj = linear_class(
            self.hidden_size,
            self._q_width + 2 * self._kv_width,
            bias=config.attn_qkv_bias,
        )

    def _project_qkv(
        self, op: OpBuilder, hidden_states: ir.Value
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        # Keep clipping ahead of the split: this is observably different from
        # clipping only Q/K after RoPE or leaving V unclipped.
        qkv = self.qkv_proj(op, hidden_states)  # (B, S, Q + K + V)
        if self._clamp is not None and self._clamp > 0:
            limit = op.CastLike(self._clamp, qkv)
            qkv = op.Clip(qkv, op.Neg(limit), limit)
        return op.Split(
            qkv,
            [self._q_width, self._kv_width, self._kv_width],
            axis=-1,
            _outputs=3,
        )


class Qwen35Attention(nn.Module):
    """Multi-head attention with output gating for Qwen3.5.

    Differences from base Attention:
    - Q projection is doubled to produce both Q and a gating signal
    - Per-head Q/K RMSNorm with +1 offset (OffsetRMSNorm)
    - Partial RoPE (rotary_embedding_dim < head_dim)
    - Output gating: attn_output * sigmoid(gate)

    Args:
        config: Architecture configuration.
        linear_class: Factory callable ``(in_features, out_features, bias=...)``
            for creating projection layers. Defaults to ``Linear``. Pass a
            quantized-linear factory (see ``make_quantized_linear_factory``)
            to emit ``MatMulNBits`` for q/k/v/o projections instead.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        linear_class: type | None = None,
    ):
        super().__init__()
        if linear_class is None:
            linear_class = Linear
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        # NoPE-safe: ``partial_rotary_factor`` may be ``None`` for models
        # without RoPE. Treat ``None`` as the inert 1.0.
        prf = config.partial_rotary_factor if config.partial_rotary_factor is not None else 1.0
        self.rotary_embedding_dim = 0 if math.isclose(prf, 1.0) else int(self.head_dim * prf)
        self._rope_interleave = config.rope_interleave

        q_dim = self.num_attention_heads * self.head_dim
        self.q_proj = linear_class(
            self.hidden_size,
            q_dim * 2,
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
            q_dim,
            self.hidden_size,
            bias=config.attn_o_bias,
        )

        self.q_norm = OffsetRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = OffsetRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
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
