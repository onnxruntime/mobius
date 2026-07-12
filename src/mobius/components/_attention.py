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


class PagedCacheState(NamedTuple):
    """Paged (block-table) KV cache state for opset-24 attention.

    Implements the paged / block-table KV cache described in onnx-genai
    ``docs/DESIGN.md`` §39.4 Option C ("ONNX Scatter/GatherElements in
    Graph").  The KV for the whole batch lives in a shared *page pool* of
    fixed-size pages that are NOT contiguous per sequence.  A per-sequence
    ``block_table`` maps logical page slots to physical page indices, and a
    ``slot_mapping`` gives the flat physical slot for each newly written
    token.  This is the format used by vLLM PagedAttention and, because
    multiple sequences can list the *same* physical page in their
    ``block_table``, it also supports SGLang RadixAttention (shared prefix
    pages) with no graph change — the sharing lives entirely in the runtime's
    ``block_table`` / ``slot_mapping`` bookkeeping.

    The paged attention body is (per layer):

        1. ``ScatterND(pool_flat, slot_mapping, new_kv)`` — write the new
           tokens' K/V into their physical slots in the pool.
        2. ``Gather(updated_pool, block_table, axis=0)`` — assemble this
           sequence's pages into a contiguous ``[num_blocks, page_size, ...]``
           tensor, reshaped to ``[1, num_blocks * page_size, kv_hidden]``.
        3. ``Attention(query, K_gathered, V_gathered, nonpad_kv_seqlen,
           is_causal=1)`` — attend over the contiguous KV, bounded to the
           valid prefix by ``nonpad_kv_seqlen`` (same op contract as the
           static cache).

    Only standard ONNX ops are used (``Reshape``/``Shape``/``Unsqueeze``/
    ``ScatterND``/``Gather``/``Attention``); no custom op is required.

    .. note::
        The current implementation targets a single active sequence per
        forward (``batch == 1``), so ``block_table`` and ``slot_mapping`` are
        1-D.  Multi-sequence batching (2-D block tables + per-row gather) is a
        documented TODO.

    Fields:
        key_pool: Physical key page pool ``[num_pages, page_size, kv_hidden]``.
        value_pool: Physical value page pool ``[num_pages, page_size, kv_hidden]``.
        block_table: Physical page indices for the active sequence in logical
            order ``[num_blocks]`` int64.
        slot_mapping: Flat physical slot (``page_id * page_size + offset``) for
            each newly written token ``[seq_len]`` int64.
        nonpad_kv_seqlen: Valid KV length for the active sequence ``[batch]``
            int64 (``write_start + valid_token_count``).
    """

    key_pool: ir.Value
    value_pool: ir.Value
    block_table: ir.Value
    slot_mapping: ir.Value
    nonpad_kv_seqlen: ir.Value


def _apply_paged_attention(
    op: OpBuilder,
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    paged_cache: PagedCacheState,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    scale: float,
    softcap: float = 0.0,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """Apply attention against a paged (block-table) KV cache.

    See :class:`PagedCacheState` for the I/O contract and op sequence.  RoPE
    must already be baked into ``key`` (applied by the caller) so cached page
    entries carry the rotated keys, exactly as in the static cache path.

    Args:
        query/key/value: 3-D ``[1, seq_len, heads * head_dim]`` projections.
        head_dim: Per-head dim; used to recover ``kv_hidden`` for the pool
            reshape.

    Returns:
        ``(attn_output, updated_key_pool, updated_value_pool)`` where the
        updated pools have the same ``[num_pages, page_size, kv_hidden]`` shape
        as the inputs and are registered as graph outputs by the task.
    """
    kv_hidden = num_key_value_heads * head_dim

    # Original 3-D pool shapes [num_pages, page_size, kv_hidden]; we scatter on
    # a 2-D flattened view then restore the pool shape for the graph output.
    key_pool_shape = op.Shape(paged_cache.key_pool)
    value_pool_shape = op.Shape(paged_cache.value_pool)

    key_pool_flat = op.Reshape(paged_cache.key_pool, [-1, kv_hidden])
    value_pool_flat = op.Reshape(paged_cache.value_pool, [-1, kv_hidden])

    # New tokens' K/V as [seq_len, kv_hidden] (batch == 1).
    key_rows = op.Reshape(key, [-1, kv_hidden])
    value_rows = op.Reshape(value, [-1, kv_hidden])

    # ScatterND row-writes: pool_flat[slot_mapping[t]] = new_kv[t].
    slot_indices = op.Unsqueeze(paged_cache.slot_mapping, [-1])  # [seq_len, 1]
    updated_key_flat = op.ScatterND(key_pool_flat, slot_indices, key_rows)
    updated_value_flat = op.ScatterND(value_pool_flat, slot_indices, value_rows)

    updated_key_pool = op.Reshape(updated_key_flat, key_pool_shape)
    updated_value_pool = op.Reshape(updated_value_flat, value_pool_shape)

    # Gather this sequence's physical pages into logical order, then flatten the
    # (num_blocks, page_size) axes into a contiguous KV sequence for Attention.
    gathered_key = op.Gather(updated_key_pool, paged_cache.block_table, axis=0)
    gathered_value = op.Gather(updated_value_pool, paged_cache.block_table, axis=0)
    gathered_key = op.Reshape(gathered_key, [1, -1, kv_hidden])
    gathered_value = op.Reshape(gathered_value, [1, -1, kv_hidden])

    # Maskless is_causal=1 + nonpad_kv_seqlen (Attention input #6): identical op
    # contract to the static cache path — causal + padding masking is derived
    # internally, bounding attention to the valid prefix of the gathered pages.
    attn_output, _, _ = op.Attention(
        query,
        gathered_key,
        gathered_value,
        None,  # no attn_mask — is_causal handles masking
        None,  # no past_key (full gathered KV is provided)
        None,  # no past_value
        paged_cache.nonpad_kv_seqlen,
        q_num_heads=num_attention_heads,
        kv_num_heads=num_key_value_heads,
        scale=scale,
        softcap=softcap,
        is_causal=1,
        _outputs=3,
    )
    return attn_output, updated_key_pool, updated_value_pool


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
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | GQAContext | None,
        position_embeddings: tuple | None = None,
        past_key_value: tuple | None = None,
        static_cache: StaticCacheState | None = None,
        paged_cache: PagedCacheState | None = None,
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

        if paged_cache is not None:
            # Paged (block-table) KV cache: scatter new K/V into the physical
            # page pool, gather this sequence's pages contiguously, then attend.
            # present_* here are the UPDATED page pools (registered as outputs).
            attn_output, present_key, present_value = _apply_paged_attention(
                op,
                query_states,
                key_states,
                value_states,
                paged_cache,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                scale=self.scaling,
                softcap=self._softcap,
            )
        else:
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
        op: OpBuilder,
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
        # NoPE-safe: ``partial_rotary_factor`` may be ``None`` for models
        # without RoPE. Treat ``None`` as the inert 1.0.
        prf = config.partial_rotary_factor if config.partial_rotary_factor is not None else 1.0
        self.rotary_embedding_dim = 0 if math.isclose(prf, 1.0) else int(self.head_dim * prf)
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
