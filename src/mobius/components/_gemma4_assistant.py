# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma4-Assistant attention and decoder layer.

Standalone implementation that does not touch existing Gemma4 components
— avoids risking regressions in the production Gemma4 path while the
assistant family is being brought up.

The assistant attention is structurally simpler than the standard
Gemma4 attention because every layer borrows its K and V entirely from
the target model (no own K/V projections, no own KV cache, no
KV-share-source-layer arithmetic).  The Q-side stays the same: Q
projection + per-head RMSNorm + RoPE, mirroring Gemma4TextAttention.

Per upstream comments in
``Gemma4AssistantForCausalLM.create_attention_masks``:
    "There is no difference for the edge case of q_len == 1 as it acts
    as full attention no matter what"
— and the standard speculative-decoding loop only ever calls the
assistant with ``q_len = 1`` (one new draft token at a time).  We
therefore omit the bidirectional mask machinery entirely and rely on
``is_causal=0`` + ``attention_bias=None``, which is full attention.
That stays correct for any ``q_len`` so long as the caller does not
require causality among Q positions; the assistant's autoregressive
draft loop satisfies that by construction (each Q token corresponds to
a previously-predicted position).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import Gemma4AssistantConfig
from mobius.components._attention import GQAContext, _apply_attention
from mobius.components._common import Linear, create_attention_bias
from mobius.components._mlp import GatedMLP
from mobius.components._rms_norm import RMSNorm
from mobius.components._rotary_embedding import apply_rotary_pos_emb

if TYPE_CHECKING:
    pass


class Gemma4AssistantAttention(nn.Module):
    """Q-only attention against externally-supplied K/V (from the target's cache).

    Two emission paths:

    * **Generic (Attention op)** — when ``gqa_ctx`` is None.  Transposes the
      shared K/V from BNSH to BSH and runs ``op.Attention`` with
      ``is_causal=0``.  Works on every EP and any dtype.

    * **GroupQueryAttention** — when ``gqa_ctx`` is given (active EP supports
      GQA fusion for the build dtype).  Routes the shared K/V as
      ``past_key`` / ``past_value`` directly in BNSH (no transpose), passes an
      empty new K/V (``kv_sequence_length=0`` so nothing is appended), and
      lets GQA's ``seqlens_k`` + ``do_rotary`` handle masking and RoPE on Q
      automatically.  Mirrors the gemma4 KV-shared layer pattern at
      ``mobius/models/gemma4.py:758-803``.

    The GQA path is what lets the orchestrator share a single max-cache-len KV
    buffer between target and assistant (the assistant just reads the buffer
    in-place; ``seqlens_k`` drives the mask).
    """

    def __init__(
        self,
        config: Gemma4AssistantConfig,
        layer_type: str,
        rotary_embedding_dim: int = 0,
    ):
        super().__init__()
        is_full = layer_type == "full_attention"
        self.layer_type = layer_type
        # Sliding layers limit attention to the most recent ``sliding_window``
        # keys; full layers attend the whole shared-KV buffer.
        self.sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )
        self.head_dim = (
            (config.global_head_dim or config.head_dim) if is_full else config.head_dim
        )
        self.num_attention_heads = config.num_attention_heads
        # Full-attention layers may use a distinct kv-head count; sliding always
        # uses the standard num_key_value_heads.  Mirrors Gemma4TextAttention.
        if is_full and config.num_global_key_value_heads is not None:
            self.num_key_value_heads = config.num_global_key_value_heads
        else:
            self.num_key_value_heads = config.num_key_value_heads
        # Gemma4 hardcodes scale=1.0 (not 1/sqrt(d)); attn_logit_softcapping
        # is wired through to the ONNX Attention op's native ``softcap``.
        self.scaling = 1.0
        self.softcap = config.attn_logit_softcapping or 0.0
        self.rotary_embedding_dim = rotary_embedding_dim
        self._rope_interleave = config.rope_interleave

        self.q_proj = Linear(
            config.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = Linear(
            self.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        shared_key: ir.Value,
        shared_value: ir.Value,
        position_embeddings: tuple | None,
        gqa_ctx: GQAContext | None = None,
        attention_mask: ir.Value | None = None,
    ) -> ir.Value:
        # Q projection + per-head Q norm.
        q = self.q_proj(op, hidden_states)
        q = op.Reshape(q, [0, 0, -1, self.head_dim])
        q = self.q_norm(op, q)
        q = op.Reshape(q, [0, 0, -1])

        if gqa_ctx is not None:
            # ===== GQA path =====
            # Route shared K/V (already RoPE'd by the target) directly as
            # past_key / past_value in BNSH.  Pass an empty new K/V so GQA
            # does not try to append anything.
            #
            # GQA with do_rotary=1 applies RoPE to Q internally using the
            # per-batch position derived from seqlens_k (= sum(mask)-1) and
            # the new_seqlen=0 of the K input, so the Q's position becomes
            # ``past_seq_len = seqlens_k + 1``, i.e. the position of the
            # next token to predict.
            batch_dim = op.Shape(q, start=0, end=1)
            kv_hidden = self.num_key_value_heads * self.head_dim
            empty_shape = op.Concat(
                batch_dim,
                op.Constant(value_ints=[0, kv_hidden]),
                axis=0,
            )
            empty_kv = op.CastLike(op.ConstantOfShape(empty_shape), q)

            gqa_attrs: dict = {
                "num_heads": self.num_attention_heads,
                "kv_num_heads": self.num_key_value_heads,
                "scale": self.scaling,
                "do_rotary": 1,
                "rotary_interleaved": int(self._rope_interleave),
            }
            if self.softcap:
                gqa_attrs["softcap"] = self.softcap
            if self.rotary_embedding_dim:
                gqa_attrs["rotary_embedding_dim"] = self.rotary_embedding_dim
            if gqa_ctx.local_window_size > 0:
                gqa_attrs["local_window_size"] = gqa_ctx.local_window_size

            attn_output, _present_k, _present_v = op.GroupQueryAttention(
                q,
                empty_kv,
                empty_kv,
                shared_key,
                shared_value,
                gqa_ctx.seqlens_k,
                gqa_ctx.total_seq_len,
                gqa_ctx.cos_cache,
                gqa_ctx.sin_cache,
                _domain="com.microsoft",
                _outputs=3,
                **gqa_attrs,
            )
            return self.o_proj(op, attn_output)

        # ===== Generic Attention path =====
        # Apply RoPE to Q.  The shared K from the target is already RoPE'd by
        # the target's forward pass; we only need to RoPE the assistant Q.
        q = apply_rotary_pos_emb(
            op,
            x=q,
            position_embeddings=position_embeddings,
            num_heads=self.num_attention_heads,
            rotary_embedding_dim=self.rotary_embedding_dim,
            interleaved=self._rope_interleave,
        )

        # Shared K, V arrive in BNSH; transpose to BSH for op.Attention.
        # Mirrors the non-GQA KV-shared path in mobius/models/gemma4.py:812-815.
        k = op.Transpose(shared_key, perm=[0, 2, 1, 3])
        k = op.Reshape(k, [0, 0, -1])
        v = op.Transpose(shared_value, perm=[0, 2, 1, 3])
        v = op.Reshape(v, [0, 0, -1])

        # Full attention: is_causal=0, no attention_bias.  See module
        # docstring for why this is correct.  For q_len < kv_len the two EPs
        # disagree on ``is_causal`` alignment; with is_causal=0 + no bias the
        # graph is correct on every EP.
        #
        # Sliding layers, however, must constrain attention to the most recent
        # ``sliding_window`` keys, otherwise they diverge from HF once
        # ``kv_len > sliding_window``.  When a padding mask is available we
        # reuse the well-tested causal+sliding bias builder (causal is a no-op
        # for the q_len==1 draft step this drafter runs).
        attn_bias = None
        if self.sliding_window and attention_mask is not None:
            from mobius._build_context import get_build_dtype

            # The shared KV holds ``kv_len`` real positions, but ``attention_mask``
            # is the target's full buffer-width mask (kv_len + 1 for the slot of
            # the token being predicted — GQA consumes that extra slot via
            # ``total_seq_len`` while the generic path passes only the shared KV
            # as K). Slice the mask to ``kv_len`` so the additive bias width
            # matches the Attention op's total_sequence_length (= K's seq dim);
            # otherwise ORT rejects the model with an inconsistent
            # total_sequence_length error.
            kv_len = op.Shape(shared_key, start=2, end=3)  # [1] = shared KV seq len
            mask_kv = op.Slice(attention_mask, [0], kv_len, [1], [1])  # [B, kv_len]
            attn_bias = create_attention_bias(
                op,
                input_ids=q,
                attention_mask=mask_kv,
                sliding_window=self.sliding_window,
                dtype=get_build_dtype(),
            )
        attn_output, _present_k, _present_v = _apply_attention(
            op,
            q,
            k,
            v,
            attn_mask=attn_bias,
            past_key=None,
            past_value=None,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale=self.scaling,
            softcap=self.softcap,
            is_causal=0,
        )
        return self.o_proj(op, attn_output)


class Gemma4AssistantDecoderLayer(nn.Module):
    """One Gemma4-Assistant transformer block.

    Mirrors upstream ``Gemma4TextDecoderLayer.forward`` (lines 1398-1455
    of transformers/models/gemma4/modeling_gemma4.py): the Gemma4 layer
    is **4-norm + layer_scalar**, not the standard 2-norm pre-norm
    pattern.  Algorithmically::

        # Attention block
        h_in  = x
        h     = input_layernorm(x)
        h     = self_attn(h)
        h     = post_attention_layernorm(h)     # NORM BEFORE RESIDUAL
        h     = h_in + h

        # MLP block
        h_in  = h
        h     = pre_feedforward_layernorm(h)
        h     = mlp(h)
        h     = post_feedforward_layernorm(h)   # NORM BEFORE RESIDUAL
        h     = h_in + h

        # Per-layer scalar multiplier (a per-layer persistent buffer,
        # ``register_buffer("layer_scalar", torch.ones(1))``, in HF).
        return h * layer_scalar

    The assistant has no per-layer input gating and no MoE (validated by
    :meth:`Gemma4AssistantConfig.validate`), so we skip those branches
    that the standard Gemma4 layer carries.
    """

    def __init__(
        self,
        config: Gemma4AssistantConfig,
        layer_type: str,
        rotary_embedding_dim: int = 0,
    ):
        super().__init__()
        self.layer_type = layer_type
        self.self_attn = Gemma4AssistantAttention(
            config, layer_type=layer_type, rotary_embedding_dim=rotary_embedding_dim
        )
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=config.hidden_act,
            bias=False,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # ``layer_scalar`` is a per-layer persistent buffer in HF
        # (``register_buffer("layer_scalar", torch.ones(1))`` on every layer,
        # so the state-dict key is ``model.layers.{i}.layer_scalar``).  In
        # mobius we declare it as a Parameter so the weight loader populates it
        # from the HF state dict; at runtime it's applied as the last op
        # in this layer.
        self.layer_scalar = nn.Parameter([1])

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        shared_key: ir.Value,
        shared_value: ir.Value,
        position_embeddings: tuple | None,
        gqa_ctx: GQAContext | None = None,
        attention_mask: ir.Value | None = None,
    ) -> ir.Value:
        # --- Attention block: pre-norm -> attn -> post-norm -> residual ---
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_output = self.self_attn(
            op,
            hidden_states=hidden_states,
            shared_key=shared_key,
            shared_value=shared_value,
            position_embeddings=position_embeddings,
            gqa_ctx=gqa_ctx,
            attention_mask=attention_mask,
        )
        hidden_states = self.post_attention_layernorm(op, attn_output)
        hidden_states = op.Add(residual, hidden_states)

        # --- MLP block: pre-norm -> mlp -> post-norm -> residual ---
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = self.post_feedforward_layernorm(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        # --- Final per-layer scalar multiplier ---
        hidden_states = op.Mul(hidden_states, self.layer_scalar)
        return hidden_states


class Gemma4AssistantMaskedEmbedder(nn.Module):
    """Centroid-routed sparse LM head used when ``use_ordered_embeddings=True``.

    Mirrors upstream ``Gemma4AssistantMaskedEmbedder.forward``:

        centroid_logits  = centroids(hidden)                          # [B, L, C]
        top_k_indices    = TopK(centroid_logits, k=top_k)             # [B, L, top_k]
        canonical_per_c  = token_ordering.view(C, K)                  # [C, K]
        selected_canon   = canonical_per_c[top_k_indices]             # [B, L, top_k, K]
        selected_emb     = lm_head_weight[selected_canon.flatten()]   # gather rows
        selected_logits  = hidden @ selected_emb.T                    # [B, L, top_k*K]
        mask_value       = selected_logits.min() - 1
        output           = full([B, L, vocab], mask_value)
        scatter_idx      = selected_canon.view(B, L, top_k*K)
        output.scatter_(dim=-1, index=scatter_idx, src=selected_logits)

    Where ``C = num_centroids`` and ``K = vocab_size // num_centroids`` is the
    number of vocab tokens routed by each centroid.  For E2B-it-assistant
    these are ``C=2048``, ``top_k=32``, ``K=128`` (so ``top_k*K = 4096`` ≪
    vocab=262144 — only 1.6% of vocab positions ever get a non-mask logit).

    Weight layout (matches HF):
        centroids.weight       [num_centroids, hidden_size]  float
        token_ordering         [vocab_size]                  int64

    The ``lm_head.weight`` (separately owned by
    :class:`Gemma4AssistantCausalLMModel`) is passed in as ``lm_head_weight``;
    we do not duplicate it inside this module.
    """

    def __init__(self, config: Gemma4AssistantConfig):
        super().__init__()
        if config.vocab_size % config.num_centroids != 0:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) must be divisible by "
                f"num_centroids ({config.num_centroids})"
            )
        self.num_centroids = config.num_centroids
        self.centroid_intermediate_top_k = config.centroid_intermediate_top_k
        self.vocab_size = config.vocab_size
        self.vocab_size_per_centroid = config.vocab_size // config.num_centroids
        self.hidden_size = config.hidden_size

        self.centroids = Linear(config.hidden_size, config.num_centroids, bias=False)
        # token_ordering is a learnable INT64 buffer in upstream (registered
        # via register_buffer); in mobius we declare it as an INT64 Parameter
        # so the weight loader populates it from the HF state_dict key
        # ``masked_embedding.token_ordering``.
        self.token_ordering = nn.Parameter(
            [config.vocab_size],
            dtype=ir.DataType.INT64,
            name="token_ordering",
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        lm_head_weight: ir.Value,
    ) -> ir.Value:
        # ``top_k * vocab_size_per_centroid`` is the number of vocab positions
        # we score per token; everything else gets ``mask_value`` (effectively
        # -inf for argmax / softmax).
        top_k = self.centroid_intermediate_top_k
        K = self.vocab_size_per_centroid  # noqa: N806 — matrix-dim convention
        top_kK = top_k * K  # noqa: N806

        # centroid_logits: [B, L, num_centroids]
        centroid_logits = self.centroids(op, hidden_states)

        # TopK over the last dim → top_k centroid indices per (B, L).
        # _values is discarded; we only need indices.
        _vals, top_k_indices = op.TopK(
            centroid_logits,
            op.Constant(value_ints=[top_k]),
            axis=-1,
            largest=1,
            sorted=0,
            _outputs=2,
        )
        # top_k_indices: [B, L, top_k] INT64

        # Reshape the INT64 token_ordering buffer [vocab] → [num_centroids, K].
        canonical_per_cluster = op.Reshape(
            self.token_ordering,
            op.Constant(value_ints=[self.num_centroids, K]),
        )  # [C, K] INT64

        # Gather canonical vocab positions for each selected centroid.
        # canonical_per_cluster: [C, K], top_k_indices: [B, L, top_k]
        # axis=0 → [B, L, top_k, K] INT64
        selected_canonical = op.Gather(canonical_per_cluster, top_k_indices, axis=0)

        # Gather LM-head rows at those canonical positions.
        # lm_head_weight: [vocab, hidden], selected_canonical: [B, L, top_k, K]
        # axis=0 → [B, L, top_k, K, hidden] (model dtype)
        selected_embeddings = op.Gather(lm_head_weight, selected_canonical, axis=0)

        # Flatten the (top_k, K) axes to one (top_k*K) axis so we can do a
        # single batched matmul.  Shape: [B, L, top_k*K, hidden].
        selected_emb_flat = op.Reshape(
            selected_embeddings,
            op.Constant(value_ints=[0, 0, top_kK, self.hidden_size]),
        )

        # Compute the selected logits.  Upstream does:
        #   hidden.unsqueeze(-2) @ selected_embeddings.transpose(-1, -2)
        # i.e. [B, L, 1, hidden] @ [B, L, hidden, top_k*K] → [B, L, 1, top_k*K]
        # → squeeze(-2) → [B, L, top_k*K].
        h_4d = op.Unsqueeze(hidden_states, op.Constant(value_ints=[-2]))
        emb_t = op.Transpose(selected_emb_flat, perm=[0, 1, 3, 2])
        sel_logits_4d = op.MatMul(h_4d, emb_t)
        selected_logits = op.Squeeze(sel_logits_4d, op.Constant(value_ints=[-2]))
        # selected_logits: [B, L, top_k*K] (model dtype)

        # mask_value = min(selected_logits) - 1  (scalar, in model dtype)
        min_logit = op.ReduceMin(selected_logits, keepdims=0)
        one_like = op.CastLike(op.Constant(value_float=1.0), selected_logits)
        mask_value = op.Sub(min_logit, one_like)

        # Build output buffer [B, L, vocab] filled with mask_value.
        # Take dims [B, L] from hidden_states and append vocab.
        bl_shape = op.Shape(hidden_states, start=0, end=2)
        vocab_dim = op.Constant(value_ints=[self.vocab_size])
        out_shape = op.Concat(bl_shape, vocab_dim, axis=0)
        # Reshape scalar mask_value → [1, 1, 1] so Expand can broadcast it
        # across the full vocab dim.
        mask_3d = op.Reshape(mask_value, op.Constant(value_ints=[1, 1, 1]))
        output = op.Expand(mask_3d, out_shape)

        # Scatter selected_logits into output at the per-position canonical
        # vocab indices.  axis=-1 (the vocab dim).
        scatter_idx = op.Reshape(
            selected_canonical,
            op.Constant(value_ints=[0, 0, top_kK]),
        )  # [B, L, top_k*K] INT64
        return op.ScatterElements(output, scatter_idx, selected_logits, axis=-1)
