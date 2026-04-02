# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""ESMFold — protein structure prediction from sequence.

ESMFold combines an ESM-2 protein language model backbone with an
AlphaFold2-style folding trunk to predict 3D protein structures directly
from amino-acid sequences.

Architecture overview:

```
input_ids → ESM-2 encoder (36-layer transformer, rotary position embeddings)
    ↓
  single representations (B, L, 1024)
  + pairwise from outer-product (B, L, L, 128)
    ↓
  Folding trunk (48 × TriangularSelfAttentionBlock)
    - sequence ↔ pair exchange
    - triangle multiplicative updates (outgoing + incoming)
    - triangle attention (start + end)
    - sequence + pair MLPs
    ↓
  Structure module (IPA + rigid-body updates)
    ↓
  atom14 coordinates, pLDDT, distogram, pTM
```

**Phase 1 (this file)**: ESM-2 backbone with RoPE, trunk projection
(``esm_s_mlp``), LM head, config, registry.

**Phase 2** (future): Folding trunk — triangular attention & multiplicative
updates, pairwise representations.

**Phase 3** (future): Structure module — invariant point attention (IPA),
rigid-body transforms, angle prediction.

Replicates HuggingFace's ``EsmForProteinFolding``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._common import Embedding, LayerNorm, Linear
from mobius.components._rotary_embedding import (
    apply_rotary_pos_emb,
    initialize_rope,
)
from mobius.models.bert import (
    _BertAttention,
    _BertIntermediate,
    _BertOutput,
    _rename_bert_weight,
)

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# ESM-2 encoder components (BERT-like + rotary position embeddings)
# ---------------------------------------------------------------------------


class _EsmEmbeddings(nn.Module):
    """ESM-2 embeddings: token embedding + optional pre-LayerNorm.

    ESM-2 uses rotary position embeddings in the attention layers, so the
    embedding layer contains only the token (word) embedding.

    ``padding_idx`` is set so that pad tokens produce zero embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        pad_token_id: int,
        layer_norm_eps: float,
        emb_layer_norm_before: bool = False,
    ):
        super().__init__()
        self.word_embeddings = Embedding(vocab_size, hidden_size)
        self._emb_layer_norm_before = emb_layer_norm_before
        if emb_layer_norm_before:
            self.layer_norm = LayerNorm(hidden_size, eps=layer_norm_eps)
        self._pad_token_id = pad_token_id

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value):
        hidden_states = self.word_embeddings(op, input_ids)
        if self._emb_layer_norm_before:
            hidden_states = self.layer_norm(op, hidden_states)

        # Zero out padding positions
        # mask: (B, L) -> (B, L, 1) for broadcast
        pad_mask = op.Equal(input_ids, self._pad_token_id)
        pad_mask = op.Unsqueeze(pad_mask, [-1])
        zero = op.CastLike(0.0, hidden_states)
        hidden_states = op.Where(pad_mask, zero, hidden_states)
        return hidden_states


class _EsmAttention(_BertAttention):
    """ESM-2 attention: BERT-style post-norm attention with RoPE.

    Extends :class:`_BertAttention` by applying rotary position embeddings
    to Q and K before the ``op.Attention`` call.  Weight names remain
    identical to BERT (``attention.query``, ``attention.key``, etc.).
    """

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_embeddings: tuple | None = None,
    ):
        self_attn = self.self
        query = self_attn.query(op, hidden_states)  # (B, L, H)
        key = self_attn.key(op, hidden_states)  # (B, L, H)
        value = self_attn.value(op, hidden_states)  # (B, L, H)

        # Apply rotary position embeddings to Q and K
        if position_embeddings is not None:
            query = apply_rotary_pos_emb(
                op,
                query,
                position_embeddings,
                num_heads=self_attn.num_heads,
                rotary_embedding_dim=0,
            )
            key = apply_rotary_pos_emb(
                op,
                key,
                position_embeddings,
                num_heads=self_attn.num_heads,
                rotary_embedding_dim=0,
            )

        attn_out = op.Attention(
            query,
            key,
            value,
            attention_mask,
            q_num_heads=self_attn.num_heads,
            kv_num_heads=self_attn.num_heads,
            scale=float(self_attn.head_dim**-0.5),
        )

        attn_out = self.output.dense(op, attn_out)
        return self.output.LayerNorm(op, op.Add(hidden_states, attn_out))


class _EsmEncoderLayer(nn.Module):
    """ESM-2 encoder layer: post-norm with RoPE-enabled attention.

    Same structure as ``_BertEncoderLayer`` but uses :class:`_EsmAttention`
    and threads ``position_embeddings`` through.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        bias: bool = True,
    ):
        super().__init__()
        self.attention = _EsmAttention(
            hidden_size, num_attention_heads, layer_norm_eps, bias
        )
        self.intermediate = _BertIntermediate(
            hidden_size, intermediate_size, hidden_act, bias
        )
        self.output = _BertOutput(
            intermediate_size, hidden_size, layer_norm_eps, bias
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
        position_embeddings: tuple | None = None,
    ):
        # Self-attention with post-norm and RoPE
        hidden_states = self.attention(
            op, hidden_states, attention_mask, position_embeddings
        )
        # MLP with post-norm
        intermediate = self.intermediate(op, hidden_states)
        mlp_out = self.output.dense(op, intermediate)
        hidden_states = self.output.LayerNorm(
            op, op.Add(hidden_states, mlp_out)
        )
        return hidden_states


class _EsmEncoder(nn.Module):
    """ESM-2 encoder: stack of post-norm layers with rotary embeddings."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                _EsmEncoderLayer(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    layer_norm_eps=config.rms_norm_eps,
                    bias=True,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_embeddings: tuple | None = None,
    ):
        for layer in self.layer:
            hidden_states = layer(
                op, hidden_states, attention_mask, position_embeddings
            )
        return hidden_states


# ---------------------------------------------------------------------------
# Trunk projection
# ---------------------------------------------------------------------------


class _EsmSMlp(nn.Module):
    """ESM-2 → trunk projection: LayerNorm → Linear → ReLU → Linear.

    Maps ESM-2 hidden states (dim ``hidden_size``) to the folding trunk
    sequence dimension (``trunk_seq_dim``).  The intermediate dimension
    equals ``trunk_seq_dim``.

    HuggingFace layout::

        esm_s_mlp = Sequential(
            LayerNorm(hidden_size),
            Linear(hidden_size, trunk_seq_dim),
            ReLU(),
            Linear(trunk_seq_dim, trunk_seq_dim),
        )
    """

    def __init__(
        self, hidden_size: int, trunk_seq_dim: int, layer_norm_eps: float
    ):
        super().__init__()
        # Indexed 0..3 to match HF Sequential indices for weight alignment
        self._0 = LayerNorm(hidden_size, eps=layer_norm_eps)
        self._1 = Linear(hidden_size, trunk_seq_dim, bias=True)
        self._3 = Linear(trunk_seq_dim, trunk_seq_dim, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = self._0(op, x)
        x = self._1(op, x)
        x = op.Relu(x)
        x = self._3(op, x)
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class EsmFoldModel(nn.Module):
    """ESMFold protein structure prediction model (Phase 1: backbone only).

    This Phase 1 implementation builds the ESM-2 encoder backbone which
    produces per-residue representations projected to trunk dimension.
    The folding trunk and structure module are not yet implemented
    (Phase 2/3).

    Top-level architecture (HuggingFace ``EsmForProteinFolding``):

    - ``esm``: ESM-2 protein language model (encoder, rotary pos emb)
    - ``esm_s_mlp``: Projects ESM-2 hidden states to trunk sequence dim
    - ``embedding``: Amino-acid embedding for trunk input
    - ``trunk``: Folding trunk (48 triangular self-attention blocks)
    - ``distogram_head``: Predicts inter-residue distances
    - ``ptm_head``: Predicts template modeling score
    - ``lm_head``: Masked language model head
    - ``lddt_head``: Predicts per-residue confidence (pLDDT)

    Inputs: ``input_ids`` (amino acid token IDs), ``attention_mask``.
    Outputs: Trunk-projected hidden states ``(B, L, trunk_seq_dim)``.
    """

    default_task = "feature-extraction"
    category = "Protein Structure"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config

        # ESM-2 backbone with rotary position embeddings
        self.esm_embeddings = _EsmEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            pad_token_id=config.pad_token_id or 1,
            layer_norm_eps=config.rms_norm_eps,
            emb_layer_norm_before=getattr(
                config, "emb_layer_norm_before", False
            ),
        )
        self.esm_encoder = _EsmEncoder(config)
        self.rotary_emb = initialize_rope(config)

        # Projection from ESM-2 hidden dim to trunk sequence dim
        # HF layout: Sequential(LayerNorm → Linear → ReLU → Linear)
        trunk_seq_dim = getattr(
            config, "trunk_sequence_state_dim", 1024
        )
        self.esm_s_mlp = _EsmSMlp(
            config.hidden_size, trunk_seq_dim, config.rms_norm_eps
        )

        # LM head for masked language modeling (auxiliary objective)
        self.lm_head = Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        token_type_ids: ir.Value | None = None,
    ):
        """Forward pass: ESM-2 backbone → esm_s_mlp → trunk-dim output.

        Returns trunk-projected per-residue representations.
        Phase 2/3 will feed these into the folding trunk and structure
        module.

        ``token_type_ids`` is accepted for compatibility with the
        feature-extraction task but is not used by ESM-2.
        """
        # ESM-2 embeddings: (B, L) -> (B, L, hidden_size)
        hidden_states = self.esm_embeddings(op, input_ids)

        # Compute position_ids from sequence length for RoPE
        # ESM-2 encoder positions are simply [0, 1, ..., L-1]
        seq_len = op.Shape(input_ids, start=1, end=2)  # [1]
        position_ids = op.Unsqueeze(
            op.Range(
                op.Constant(value_int=0),
                op.Squeeze(seq_len),
                op.Constant(value_int=1),
            ),
            [0],
        )  # (1, L)
        position_embeddings = self.rotary_emb(
            op, position_ids
        )  # (cos: (1, L, rotary_dim), sin: (1, L, rotary_dim))

        # ESM-2 encoder: (B, L, hidden_size) -> (B, L, hidden_size)
        hidden_states = self.esm_encoder(
            op, hidden_states, attention_mask, position_embeddings
        )

        # Project to trunk sequence dimension: (B, L, trunk_seq_dim)
        hidden_states = self.esm_s_mlp(op, hidden_states)

        return hidden_states

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace ESMFold weight names to our parameter names.

        HuggingFace layout:
        - ``esm.embeddings.*`` → ``esm_embeddings.*``
        - ``esm.encoder.layer.N.*`` → ``esm_encoder.layer.N.*``
        - ``esm_s_mlp.{0,1,3}.*`` → ``esm_s_mlp._{0,1,3}.*``
        - ``lm_head.*`` → ``lm_head.*``
        - Trunk/structure weights are dropped in Phase 1.
        """
        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _rename_esmfold_weight(key)
            if new_key is not None:
                new_state_dict[new_key] = value
        return new_state_dict


def _rename_esmfold_weight(name: str) -> str | None:
    """Rename a single HuggingFace ESMFold weight key.

    Returns the new name, or None to drop the weight.
    """
    # ESM-2 embeddings: esm.embeddings.* → esm_embeddings.*
    if name.startswith("esm.embeddings."):
        suffix = name[len("esm.embeddings."):]
        # Skip position_ids (buffer, not a parameter)
        if suffix == "position_ids":
            return None
        return f"esm_embeddings.{suffix}"

    # ESM-2 encoder: esm.encoder.layer.* → esm_encoder.layer.*
    if name.startswith("esm.encoder."):
        suffix = name[len("esm.encoder."):]
        # Drop rotary_embeddings buffers (pre-computed in our model)
        if "rotary_embeddings" in suffix:
            return None
        # Reuse BERT weight renaming for the encoder internals
        renamed = _rename_bert_weight(f"bert.encoder.{suffix}")
        if renamed is None:
            return None
        # Strip "encoder." prefix that _rename_bert_weight produces
        return f"esm_{renamed}"

    # ESM MLP projection: esm_s_mlp.{0,1,3}.* → esm_s_mlp._{0,1,3}.*
    # HF uses Sequential indices; we prefix with _ since Python attrs
    # can't start with a digit.  Index 2 is ReLU (no parameters).
    if name.startswith("esm_s_mlp."):
        suffix = name[len("esm_s_mlp."):]
        return f"esm_s_mlp._{suffix}"

    # LM head
    if name.startswith("lm_head."):
        return name

    # Drop ESM contact head (not used in folding)
    if name.startswith("esm.contact_head."):
        return None

    # Drop trunk and structure module weights (Phase 2/3)
    if name.startswith("trunk.") or name.startswith("embedding."):
        return None
    if name.startswith("distogram_head.") or name.startswith("ptm_head."):
        return None
    if name.startswith("lddt_head."):
        return None

    # Drop unknown weights
    return None
