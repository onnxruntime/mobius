"""ESMFold — protein structure prediction from sequence.

ESMFold combines an ESM-2 protein language model backbone with an
AlphaFold2-style folding trunk to predict 3D protein structures directly
from amino-acid sequences.

Architecture overview:

```
input_ids → ESM-2 encoder (36-layer transformer)
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

**Phase 1 (this file)**: ESM-2 backbone reuse from BertModel, config,
registry, skeleton for trunk placeholders.

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
from mobius.components._common import Embedding, Linear
from mobius.models.bert import _BertEncoder, _rename_bert_weight

if TYPE_CHECKING:
    import onnx_ir as ir


class _EsmEmbeddings(nn.Module):
    """ESM-2 embeddings: token + positional (rotary applied in attention).

    ESM-2 uses rotary position embeddings in the attention layers rather
    than learned absolute position embeddings.  The embedding layer only
    contains the token (word) embedding and an optional layer norm.

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
            from mobius.components._common import LayerNorm

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


class _EsmEncoder(_BertEncoder):
    """ESM-2 encoder: stack of transformer layers.

    Reuses the BERT encoder implementation.  ESM-2 uses rotary position
    embeddings which are applied inside each attention layer rather than
    in the embedding layer, but for the ONNX graph the standard BERT
    attention pattern with full attention mask is compatible.
    """


class EsmFoldModel(nn.Module):
    """ESMFold protein structure prediction model (Phase 1: backbone only).

    This Phase 1 implementation builds the ESM-2 encoder backbone which
    produces per-residue representations.  The folding trunk and structure
    module are placeholders for Phase 2/3.

    Top-level architecture (HuggingFace ``EsmForProteinFolding``):

    - ``esm``: ESM-2 protein language model (encoder)
    - ``esm_s_mlp``: Projects ESM-2 hidden states to trunk sequence dim
    - ``embedding``: Amino-acid embedding for trunk input
    - ``trunk``: Folding trunk (48 triangular self-attention blocks)
    - ``distogram_head``: Predicts inter-residue distances
    - ``ptm_head``: Predicts template modeling score
    - ``lm_head``: Masked language model head
    - ``lddt_head``: Predicts per-residue confidence (pLDDT)

    Inputs: ``input_ids`` (amino acid token IDs), ``attention_mask``.
    Outputs: Per-residue hidden states from the ESM-2 backbone.
    """

    default_task = "protein-folding"
    category = "Protein Structure"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config

        # ESM-2 backbone (same as BertModel but with ESM embeddings)
        self.esm_embeddings = _EsmEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            pad_token_id=config.pad_token_id or 1,
            layer_norm_eps=config.rms_norm_eps,
            emb_layer_norm_before=getattr(config, "emb_layer_norm_before", False),
        )
        self.esm_encoder = _BertEncoder(config)

        # Projection from ESM-2 hidden dim to trunk sequence dim
        trunk_seq_dim = getattr(config, "trunk_sequence_state_dim", 1024)
        self.esm_s_mlp = nn.Sequential(
            Linear(config.hidden_size, config.hidden_size, bias=True),
            Linear(config.hidden_size, config.hidden_size, bias=True),
            Linear(config.hidden_size, trunk_seq_dim, bias=True),
        )

        # LM head for masked language modeling (auxiliary objective)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        token_type_ids: ir.Value | None = None,
    ):
        """Forward pass: ESM-2 backbone → per-residue representations.

        Phase 1 returns the ESM-2 hidden states and LM logits.
        Phase 2/3 will add the folding trunk and structure module.

        ``token_type_ids`` is accepted for compatibility with the
        feature-extraction task but is not used by ESM-2.
        """
        # ESM-2 encoder: (B, L) -> (B, L, hidden_size)
        hidden_states = self.esm_embeddings(op, input_ids)
        hidden_states = self.esm_encoder(op, hidden_states, attention_mask)

        # LM head: (B, L, vocab_size)
        logits = self.lm_head(op, hidden_states)

        return logits

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace ESMFold weight names to our parameter names.

        HuggingFace layout:
        - ``esm.embeddings.*`` → ``esm_embeddings.*``
        - ``esm.encoder.layer.N.*`` → ``esm_encoder.layer.N.*``
        - ``esm_s_mlp.0/1/2.*`` → ``esm_s_mlp.0/1/2.*``
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
        suffix = name[len("esm.embeddings.") :]
        # Skip position_ids (buffer, not a parameter)
        if suffix == "position_ids":
            return None
        return f"esm_embeddings.{suffix}"

    # ESM-2 encoder: esm.encoder.layer.* → esm_encoder.layer.*
    if name.startswith("esm.encoder."):
        suffix = name[len("esm.encoder.") :]
        # Reuse BERT weight renaming for the encoder internals
        renamed = _rename_bert_weight(f"bert.encoder.{suffix}")
        if renamed is None:
            return None
        # Strip "encoder." prefix that _rename_bert_weight produces
        return f"esm_{renamed}"

    # ESM MLP projection: esm_s_mlp.* → esm_s_mlp.*
    if name.startswith("esm_s_mlp."):
        return name

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
