# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""BERT encoder-only model and masked LM with HF-aligned weight naming.

HF BERT uses deeply nested naming for encoder layers:
  attention.self.query / attention.self.key / attention.self.value
  attention.output.dense / attention.output.LayerNorm
  intermediate.dense
  output.dense / output.LayerNorm
  embeddings.LayerNorm

Module attributes here match HF conventions to eliminate the
rename dict entirely. Only prefix stripping (bert./roberta./esm.)
and gamma/beta compat remain in preprocess_weights.

BertForMaskedLM adds a masked language model head on top of the
encoder for predicting masked tokens. Supports BERT, RoBERTa, ESM-2,
and similar encoder-only architectures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components._activations import ACT2FN
from mobius.components._common import Embedding, LayerNorm, Linear

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# BERT components with HF-aligned attribute names
# ---------------------------------------------------------------------------


class _BertSelfAttention(nn.Module):
    """Self-attention projections: query, key, value (HF naming)."""

    def __init__(self, hidden_size: int, num_heads: int, bias: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = Linear(hidden_size, hidden_size, bias=bias)
        self.key = Linear(hidden_size, hidden_size, bias=bias)
        self.value = Linear(hidden_size, hidden_size, bias=bias)


class _BertAttentionOutput(nn.Module):
    """Attention output projection + LayerNorm (HF naming)."""

    def __init__(self, hidden_size: int, eps: float, bias: bool):
        super().__init__()
        self.dense = Linear(hidden_size, hidden_size, bias=bias)
        # Capital 'LayerNorm' matches HF BERT naming
        self.LayerNorm = LayerNorm(hidden_size, eps=eps)


class _BertAttention(nn.Module):
    """BERT attention block with HF-compatible nesting.

    Produces parameter paths like:
      attention.self.query.weight
      attention.output.dense.weight
      attention.output.LayerNorm.weight
    """

    def __init__(self, hidden_size: int, num_heads: int, eps: float, bias: bool):
        super().__init__()
        # 'self' is a valid Python attribute name (HF uses it)
        self_attn = _BertSelfAttention(hidden_size, num_heads, bias)
        self.self = self_attn
        self.output = _BertAttentionOutput(hidden_size, eps, bias)

    def forward(self, op: OpBuilder, hidden_states: ir.Value, attention_mask: ir.Value):
        self_attn = self.self
        query = self_attn.query(op, hidden_states)
        key = self_attn.key(op, hidden_states)
        value = self_attn.value(op, hidden_states)

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


class _BertIntermediate(nn.Module):
    """BERT intermediate (up-projection + activation, HF naming)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        bias: bool,
    ):
        super().__init__()
        self.dense = Linear(hidden_size, intermediate_size, bias=bias)
        self._act_fn = ACT2FN[hidden_act]

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        return self._act_fn(op, self.dense(op, hidden_states))


class _BertOutput(nn.Module):
    """BERT output (down-projection + LayerNorm, HF naming)."""

    def __init__(self, intermediate_size: int, hidden_size: int, eps: float, bias: bool):
        super().__init__()
        self.dense = Linear(intermediate_size, hidden_size, bias=bias)
        self.LayerNorm = LayerNorm(hidden_size, eps=eps)


class _BertEncoderLayer(nn.Module):
    """Post-norm encoder layer with HF-aligned attribute names.

    Produces parameter paths like:
      layer.N.attention.self.query.weight
      layer.N.attention.output.LayerNorm.weight
      layer.N.intermediate.dense.weight
      layer.N.output.dense.weight
      layer.N.output.LayerNorm.weight
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
        self.attention = _BertAttention(hidden_size, num_attention_heads, layer_norm_eps, bias)
        self.intermediate = _BertIntermediate(hidden_size, intermediate_size, hidden_act, bias)
        self.output = _BertOutput(intermediate_size, hidden_size, layer_norm_eps, bias)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        # Self-attention with post-norm (inside _BertAttention)
        hidden_states = self.attention(op, hidden_states, attention_mask)
        # MLP with post-norm
        intermediate = self.intermediate(op, hidden_states)
        mlp_out = self.output.dense(op, intermediate)
        hidden_states = self.output.LayerNorm(op, op.Add(hidden_states, mlp_out))
        return hidden_states


class _BertEmbeddings(nn.Module):
    """BERT embeddings with HF-compatible LayerNorm naming."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.word_embeddings = Embedding(vocab_size, hidden_size, pad_token_id)
        self.position_embeddings = Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = Embedding(type_vocab_size, hidden_size)
        # Capital 'LayerNorm' matches HF BERT naming
        self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, op: OpBuilder, input_ids: ir.Value, token_type_ids: ir.Value):
        word_embeds = self.word_embeddings(op, input_ids)
        seq_len = op.Shape(input_ids, start=1, end=2)
        position_ids = op.Range(
            op.Constant(value_int=0),
            seq_len,
            op.Constant(value_int=1),
        )
        position_ids = op.Cast(position_ids, to=7)  # INT64
        position_ids = op.Unsqueeze(position_ids, [0])
        position_embeds = self.position_embeddings(op, position_ids)
        token_type_embeds = self.token_type_embeddings(op, token_type_ids)
        embeddings = op.Add(op.Add(word_embeds, position_embeds), token_type_embeds)
        return self.LayerNorm(op, embeddings)


# ---------------------------------------------------------------------------
# BERT Model
# ---------------------------------------------------------------------------


class BertModel(nn.Module):
    """BERT encoder-only model for feature extraction.

    Supports BERT, RoBERTa, and similar encoder architectures.
    Output is the last hidden state (no pooler head).

    Module attributes match HF naming to eliminate most renames.
    Only prefix stripping and gamma/beta compat remain.

    Replicates HuggingFace's ``BertModel`` / ``RobertaModel``.
    """

    default_task = "feature-extraction"
    category = "encoder"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embeddings = _BertEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=getattr(config, "type_vocab_size", 2),
            layer_norm_eps=config.rms_norm_eps,
            pad_token_id=config.pad_token_id or 0,
        )
        self.encoder = _BertEncoder(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        token_type_ids: ir.Value,
    ):
        hidden_states = self.embeddings(op, input_ids, token_type_ids)
        hidden_states = self.encoder(op, hidden_states, attention_mask)
        return hidden_states

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HF BERT weight names to match our parameter names.

        After attribute alignment, only prefix stripping (bert./roberta.)
        and old-BERT gamma/beta compat remain.
        """
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_bert_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


class _BertEncoder(nn.Module):
    """BERT encoder: stack of post-norm encoder layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                _BertEncoderLayer(
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

    def forward(self, op: OpBuilder, hidden_states: ir.Value, attention_mask: ir.Value):
        for layer in self.layer:
            hidden_states = layer(op, hidden_states, attention_mask)
        return hidden_states


# Old BERT checkpoints use gamma/beta instead of weight/bias
_PARAM_RENAMES = {"gamma": "weight", "beta": "bias"}


def _rename_bert_encoder_weight(name: str) -> str:
    """Collapse nested HF naming and handle gamma/beta compat.

    This is the shared encoder-weight rename logic used by both
    ``_rename_bert_weight`` and ``_rename_masked_lm_weight``.
    Assumes model prefix (bert./roberta./esm.) is already stripped.
    """
    # Collapse nested HF naming to match flat ONNX paths:
    #   attention.self.query → attention.query
    #   attention.output.dense → attention.dense
    #   layer.N.output.dense → layer.N.dense
    name = name.replace(".attention.self.", ".attention.")
    name = name.replace(".attention.output.", ".attention.")
    name = name.replace(".output.dense.", ".dense.")
    name = name.replace(".output.LayerNorm.", ".LayerNorm.")

    # Rename gamma/beta to weight/bias (old BERT compat)
    parts = name.rsplit(".", 1)
    if len(parts) == 2 and parts[1] in _PARAM_RENAMES:
        name = f"{parts[0]}.{_PARAM_RENAMES[parts[1]]}"

    return name


def _rename_bert_weight(name: str) -> str | None:
    """Rename a single HF BERT weight to our convention.

    Strips model prefixes (bert./roberta./esm.), collapses nested HF naming
    (.self./.output.) to match the flat ONNX initializer paths, and
    handles old-BERT gamma/beta compat. Returns None for pooler/cls
    weights we don't need.
    """
    # Strip "bert." / "roberta." / "esm." prefix if present
    if name.startswith("bert."):
        name = name[5:]
    elif name.startswith("roberta."):
        name = name[8:]
    elif name.startswith("esm."):
        name = name[4:]

    # Skip pooler and classification heads
    if name.startswith(("pooler.", "cls.")):
        return None

    return _rename_bert_encoder_weight(name)


# ---------------------------------------------------------------------------
# Masked LM Head and BertForMaskedLM
# ---------------------------------------------------------------------------

# BERT cls.predictions.transform.* -> our lm_head.* naming
_BERT_CLS_TO_LM_HEAD = {
    "cls.predictions.transform.dense.": "lm_head.dense.",
    "cls.predictions.transform.LayerNorm.": "lm_head.layer_norm.",
    "cls.predictions.decoder.": "lm_head.decoder.",
    # Top-level bias on the BERT predictions head
    "cls.predictions.bias": "lm_head.decoder.bias",
}


class _MaskedLMHead(nn.Module):
    """Masked LM prediction head: dense -> activation -> LayerNorm -> decoder.

    Matches HF ESM/RoBERTa naming: lm_head.dense, lm_head.layer_norm,
    lm_head.decoder. BERT uses different naming (cls.predictions.*) which
    is handled by preprocess_weights.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.dense = Linear(hidden_size, hidden_size, bias=True)
        self.layer_norm = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = Linear(hidden_size, vocab_size, bias=True)
        self._act_fn = ACT2FN[hidden_act]

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        # hidden_states: (batch, seq, hidden_size) -> logits: (batch, seq, vocab)
        x = self.dense(op, hidden_states)
        x = self._act_fn(op, x)
        x = self.layer_norm(op, x)
        return self.decoder(op, x)


class BertForMaskedLM(nn.Module):
    """BERT/ESM/RoBERTa encoder with masked language model head.

    Predicts vocabulary logits for each token position, used for
    masked token prediction (fill-mask).

    Supports ESM-2 (protein masked LM), BERT, RoBERTa, and similar
    encoder architectures. Output is per-token logits over the vocabulary.

    Replicates HuggingFace's ``EsmForMaskedLM`` / ``BertForMaskedLM`` /
    ``RobertaForMaskedLM``.
    """

    default_task = "masked-lm"
    category = "encoder"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embeddings = _BertEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=getattr(config, "type_vocab_size", 2),
            layer_norm_eps=config.rms_norm_eps,
            pad_token_id=config.pad_token_id or 0,
        )
        self.encoder = _BertEncoder(config)
        self.lm_head = _MaskedLMHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            hidden_act=config.hidden_act,
            layer_norm_eps=config.rms_norm_eps,
        )

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        token_type_ids: ir.Value,
    ):
        hidden_states = self.embeddings(op, input_ids, token_type_ids)
        hidden_states = self.encoder(op, hidden_states, attention_mask)
        logits = self.lm_head(op, hidden_states)
        return logits

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HF weight names to match our parameter names.

        Handles ESM (esm.* prefix + lm_head.*), RoBERTa (roberta.* +
        lm_head.*), and BERT (bert.* + cls.predictions.*) conventions.
        """
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_masked_lm_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


def _rename_masked_lm_weight(name: str) -> str | None:
    """Rename a single HF masked LM weight to our convention.

    Handles LM head weights (BERT cls.predictions.* -> lm_head.*,
    ESM/RoBERTa lm_head.* passes through) and delegates encoder
    weights to ``_rename_bert_weight``.
    """
    # Strip "bert." / "roberta." / "esm." prefix from encoder weights
    if name.startswith("bert."):
        name = name[5:]
    elif name.startswith("roberta."):
        name = name[8:]
    elif name.startswith("esm."):
        name = name[4:]

    # Skip pooler (not needed for MLM)
    if name.startswith("pooler."):
        return None

    # Skip contact_head (ESM-specific, not needed for MLM)
    if name.startswith("contact_head."):
        return None

    # Map BERT cls.predictions.* -> lm_head.*
    for bert_prefix, our_prefix in _BERT_CLS_TO_LM_HEAD.items():
        if name.startswith(bert_prefix):
            name = our_prefix + name[len(bert_prefix) :]
            return name

    # lm_head.* passes through directly (ESM/RoBERTa convention)
    if name.startswith("lm_head."):
        return name

    # Skip other cls.* heads (e.g. cls.seq_relationship for NSP)
    if name.startswith("cls."):
        return None

    # Encoder weights: delegate to shared rename logic.
    # Prefix is already stripped, so pass directly to the encoder
    # rename which handles HF naming collapse + gamma/beta compat.
    return _rename_bert_encoder_weight(name)
