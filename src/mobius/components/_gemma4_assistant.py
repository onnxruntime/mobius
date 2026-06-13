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
from mobius.components._attention import _apply_attention
from mobius.components._common import Linear
from mobius.components._mlp import GatedMLP
from mobius.components._rms_norm import RMSNorm
from mobius.components._rotary_embedding import apply_rotary_pos_emb

if TYPE_CHECKING:
    pass


class Gemma4AssistantAttention(nn.Module):
    """Q-only attention against externally-supplied K/V.

    The K/V tensors arrive in ``[B, num_kv_heads, kv_len, head_dim]``
    (BNSH) — the layout produced by the target's KV cache — and are
    transposed to ``[B, kv_len, num_kv_heads * head_dim]`` (BSH) before
    being fed to ``op.Attention``.

    Args:
        config: The flattened :class:`Gemma4AssistantConfig`.
        layer_type: Either ``"sliding_attention"`` or ``"full_attention"``;
            controls which per-layer-type head_dim / KV head count is used.
        rotary_embedding_dim: Dimensions to rotate inside RoPE.  Always 0
            (full rotation) for Gemma4 — both DefaultRope (sliding) and
            ProportionalRope (full, zero-padded) handle partial rotation
            inside their cos/sin tables.
    """

    def __init__(
        self,
        config: Gemma4AssistantConfig,
        layer_type: str,
        rotary_embedding_dim: int = 0,
    ):
        super().__init__()
        is_full = layer_type == "full_attention"
        self.head_dim = (config.global_head_dim or config.head_dim) if is_full else config.head_dim
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
        position_embeddings: tuple,
    ) -> ir.Value:
        # Q projection + per-head Q norm.
        q = self.q_proj(op, hidden_states)
        q = op.Reshape(q, [0, 0, -1, self.head_dim])
        q = self.q_norm(op, q)
        q = op.Reshape(q, [0, 0, -1])

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
        # docstring for why this is correct.
        #
        # WHY NOT is_causal=1?  For ``q_len < kv_len`` (always the case
        # here — assistant Q is ~1 token, target K is the full prefill+
        # generated prefix) the two EPs disagree on alignment: per the
        # ONNX spec ``is_causal`` is upper-left-aligned, so CUDA makes
        # Q[0] attend to K[0] only (silently wrong), while CPU bottom-
        # right-aligns to the correct K[0..kv_len-1].  This is the same
        # pitfall documented for the Gemma4 KV-shared path in
        # ``mobius/models/gemma4.py:818-829``.  Using is_causal=0 with
        # no bias is correct on every EP.
        #
        # FOR FUTURE GQA FUSION: the right path is NOT is_causal=1 on
        # ``op.Attention``; it is to emit ``com.microsoft::GroupQueryAttention``
        # directly with the shared K/V routed as past_key/past_value and
        # empty new key/value, exactly as ``gemma4.py:758-803`` does for
        # the KV-shared layers.  GQA's masking is governed by seqlens_k
        # + total_seq_len, which align correctly across EPs; it does not
        # consult ``is_causal`` on the inner Attention op at all.  Adding
        # that dispatch here is a localised follow-up (~50 LOC) and does
        # not require any change to this fallback path.
        attn_output, _present_k, _present_v = _apply_attention(
            op,
            q,
            k,
            v,
            attn_mask=None,
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

    Pre-norm pattern matching upstream
    ``Gemma4TextDecoderLayer`` for the parts the assistant uses
    (no per-layer input gating, no double-wide MLP, no MoE — those are
    rejected by :meth:`Gemma4AssistantConfig.validate`).
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
        # SwiGLU MLP with standard intermediate_size (no double-wide; assistant
        # config validates that use_double_wide_mlp is False).
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=config.hidden_act,
            bias=False,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        shared_key: ir.Value,
        shared_value: ir.Value,
        position_embeddings: tuple,
    ) -> ir.Value:
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_output = self.self_attn(
            op,
            hidden_states=hidden_states,
            shared_key=shared_key,
            shared_value=shared_value,
            position_embeddings=position_embeddings,
        )
        hidden_states = op.Add(residual, attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        return op.Add(residual, hidden_states)
