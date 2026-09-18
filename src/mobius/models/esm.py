# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ESM-2 protein language model encoder.

Replicates HuggingFace's ``EsmModel`` (``transformers.models.esm``) for the
``feature-extraction`` task: amino-acid token ids in, per-residue contextual
embeddings out.

ESM-2 is *not* a BERT clone, and the differences all change the numbers:

* **Rotary position embeddings.** ``config.position_embedding_type`` is
  ``"rotary"``; the learned ``embeddings.position_embeddings`` table shipped in
  the checkpoint is dead weight and is dropped. Rotary is applied to Q and K
  inside every layer, using positions ``0..seq_len-1``.
* **Pre-LayerNorm blocks.** ``attention.LayerNorm`` runs *before* self-attention
  and the layer's own ``LayerNorm`` runs *before* the feed-forward, with plain
  residual adds after each. BERT is post-norm, so reusing the BERT block would
  put every norm in the wrong place.
* **No token-type embeddings and no embedding LayerNorm.** ``token_type_ids``
  is accepted for task-signature compatibility and ignored;
  ``emb_layer_norm_before`` is ``False`` for the released ESM-2 checkpoints.
* **A final ``emb_layer_norm_after``** closes the encoder stack.
* **Token dropout.** When ``config.token_dropout`` is set, masked positions are
  zeroed and the whole embedding is rescaled by
  ``(1 - 0.15*0.8) / (1 - observed_mask_ratio)`` — a factor of ``0.88`` even
  when no ``<mask>`` token is present, so it cannot be skipped.
* **Padding is zeroed in the embedding**, not only masked in attention.

Inputs:
    input_ids: ``(batch, sequence_len)`` INT64 amino-acid token ids.
    attention_mask: ``(batch, sequence_len)`` INT64, 1 = residue, 0 = padding.
    token_type_ids: ``(batch, sequence_len)`` INT64, unused (task signature).

Outputs:
    last_hidden_state: ``(batch, sequence_len, hidden_size)`` per-residue
    embeddings.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components._activations import ACT2FN
from mobius.components._common import (
    Embedding,
    LayerNorm,
    Linear,
    create_padding_mask,
)
from mobius.components._rotary_embedding import DefaultRope, apply_rotary_pos_emb

if TYPE_CHECKING:
    import onnx_ir as ir

#: HuggingFace's ``EsmEmbeddings`` hard-codes the training-time masking rate as
#: ``0.15 * 0.8`` (15% of positions selected, 80% of those replaced by
#: ``<mask>``). It is a property of how ESM-2 was trained, not a config value.
_MASK_RATIO_TRAIN = 0.15 * 0.8


@dataclasses.dataclass
class EsmConfig(ArchitectureConfig):
    """ESM-2 architecture config.

    Adds the four ESM-specific switches that decide *which* graph is built.
    They are read from the HuggingFace config rather than assumed, because the
    ESM family ships checkpoints on both sides of each one (ESM-1b uses
    absolute positions and a pre-encoder LayerNorm; ESM-2 uses rotary and none).
    """

    position_embedding_type: str = "rotary"
    emb_layer_norm_before: bool = False
    token_dropout: bool = True
    mask_token_id: int = 32

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> EsmConfig:
        base = super().from_transformers(config, parent_config=parent_config)
        # ESM-2 declares rotary through ``position_embedding_type`` alone: its
        # HuggingFace config carries no ``rope_theta`` / ``rope_scaling``, so
        # the generic RoPE extractor sees no signal and leaves the fields unset.
        # HF's ``EsmRotaryEmbedding`` hard-codes base 10000 over the full head
        # dimension, which is what these values restate.
        return dataclasses.replace(
            base,
            position_embedding_type=getattr(config, "position_embedding_type", "rotary"),
            emb_layer_norm_before=bool(getattr(config, "emb_layer_norm_before", False)),
            token_dropout=bool(getattr(config, "token_dropout", False)),
            mask_token_id=int(getattr(config, "mask_token_id", 32) or 32),
            rope_type="default",
            rope_theta=10000.0,
            partial_rotary_factor=1.0,
        )


class _EsmEmbeddings(nn.Module):
    """Word embeddings + ESM token dropout + padding zeroing.

    Mirrors ``EsmEmbeddings``. The learned ``position_embeddings`` table is not
    instantiated: for ``position_embedding_type == "rotary"`` HuggingFace never
    reads it, and materializing it would add an initializer the graph never
    uses.
    """

    def __init__(self, config: EsmConfig):
        super().__init__()
        self.word_embeddings = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id or 0
        )
        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id
        self.layer_norm = (
            LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.emb_layer_norm_before
            else None
        )

    def forward(self, op: OpBuilder, input_ids: ir.Value, attention_mask: ir.Value):
        # (batch, seq) -> (batch, seq, hidden)
        embeddings = self.word_embeddings(op, input_ids)

        if self.token_dropout:
            # Zero the <mask> rows, then rescale so the expected embedding
            # magnitude matches training. With no <mask> present this is still
            # a 0.88 scale, so it is not an inference-time no-op.
            is_mask = op.Equal(
                input_ids, op.Constant(value_int=self.mask_token_id)
            )  # (batch, seq)
            zero = op.CastLike(op.Constant(value_float=0.0), embeddings)
            embeddings = op.Where(op.Unsqueeze(is_mask, [-1]), zero, embeddings)

            mask_float = op.CastLike(is_mask, embeddings)
            valid_float = op.CastLike(attention_mask, embeddings)
            # (batch,) counts of masked residues and of real residues
            masked_count = op.ReduceSum(mask_float, [-1], keepdims=0)
            src_lengths = op.ReduceSum(valid_float, [-1], keepdims=0)
            observed = op.Div(masked_count, src_lengths)
            one = op.CastLike(op.Constant(value_float=1.0), embeddings)
            kept = op.CastLike(op.Constant(value_float=1.0 - _MASK_RATIO_TRAIN), embeddings)
            scale = op.Div(kept, op.Sub(one, observed))  # (batch,)
            # (batch,) -> (batch, 1, 1) so it scales every residue of a row
            embeddings = op.Mul(embeddings, op.Unsqueeze(scale, [-1, -2]))

        if self.layer_norm is not None:
            embeddings = self.layer_norm(op, embeddings)

        # ESM zeroes padded positions in the embedding itself, in addition to
        # masking them in attention.
        pad_scale = op.Unsqueeze(op.CastLike(attention_mask, embeddings), [-1])
        return op.Mul(embeddings, pad_scale)


class _EsmSelfAttention(nn.Module):
    """Rotary bidirectional self-attention (HF ``EsmSelfAttention``)."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = Linear(hidden_size, hidden_size, bias=True)
        self.key = Linear(hidden_size, hidden_size, bias=True)
        self.value = Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_embeddings: tuple,
    ):
        query = self.query(op, hidden_states)
        key = self.key(op, hidden_states)
        value = self.value(op, hidden_states)

        # Rotary acts within each head; HF scales Q by head_dim**-0.5 before
        # the rotation, which commutes with it, so the equivalent `scale`
        # attribute below reproduces the same scores.
        query = apply_rotary_pos_emb(
            op, x=query, position_embeddings=position_embeddings, num_heads=self.num_heads
        )
        key = apply_rotary_pos_emb(
            op, x=key, position_embeddings=position_embeddings, num_heads=self.num_heads
        )

        return op.Attention(
            query,
            key,
            value,
            attention_mask,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            scale=float(self.head_dim**-0.5),
        )


class _EsmSelfOutput(nn.Module):
    """Attention output projection + residual (no LayerNorm — ESM is pre-norm)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = Linear(hidden_size, hidden_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value, input_tensor: ir.Value):
        return op.Add(self.dense(op, hidden_states), input_tensor)


class _EsmAttention(nn.Module):
    """Pre-norm self-attention block.

    Parameter paths match HuggingFace:
      ``attention.LayerNorm`` / ``attention.self.query`` /
      ``attention.output.dense``.
    """

    def __init__(self, hidden_size: int, num_heads: int, eps: float):
        super().__init__()
        self.self = _EsmSelfAttention(hidden_size, num_heads)
        self.output = _EsmSelfOutput(hidden_size)
        # Capital 'LayerNorm' matches HF ESM naming; it runs *before* attention.
        self.LayerNorm = LayerNorm(hidden_size, eps=eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_embeddings: tuple,
    ):
        normed = self.LayerNorm(op, hidden_states)
        attn_out = self.self(op, normed, attention_mask, position_embeddings)
        return self.output(op, attn_out, hidden_states)


class _EsmIntermediate(nn.Module):
    """Feed-forward up-projection + activation (HF naming)."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.dense = Linear(hidden_size, intermediate_size, bias=True)
        self._act_fn = ACT2FN[hidden_act]

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        return self._act_fn(op, self.dense(op, hidden_states))


class _EsmOutput(nn.Module):
    """Feed-forward down-projection + residual (no LayerNorm — ESM is pre-norm)."""

    def __init__(self, intermediate_size: int, hidden_size: int):
        super().__init__()
        self.dense = Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value, input_tensor: ir.Value):
        return op.Add(self.dense(op, hidden_states), input_tensor)


class _EsmLayer(nn.Module):
    """One pre-norm ESM encoder layer.

    Parameter paths match HuggingFace:
      ``layer.N.attention.*`` / ``layer.N.LayerNorm`` /
      ``layer.N.intermediate.dense`` / ``layer.N.output.dense``.
    """

    def __init__(self, config: EsmConfig):
        super().__init__()
        self.attention = _EsmAttention(
            config.hidden_size, config.num_attention_heads, config.rms_norm_eps
        )
        self.intermediate = _EsmIntermediate(
            config.hidden_size, config.intermediate_size, config.hidden_act
        )
        self.output = _EsmOutput(config.intermediate_size, config.hidden_size)
        # Pre-feed-forward norm.
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_embeddings: tuple,
    ):
        hidden_states = self.attention(op, hidden_states, attention_mask, position_embeddings)
        normed = self.LayerNorm(op, hidden_states)
        return self.output(op, self.intermediate(op, normed), hidden_states)


class _EsmEncoder(nn.Module):
    """Stack of pre-norm layers closed by ``emb_layer_norm_after``."""

    def __init__(self, config: EsmConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [_EsmLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.emb_layer_norm_after = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_embeddings: tuple,
    ):
        for layer in self.layer:
            hidden_states = layer(op, hidden_states, attention_mask, position_embeddings)
        return self.emb_layer_norm_after(op, hidden_states)


class EsmModel(nn.Module):
    """ESM-2 protein encoder for per-residue feature extraction.

    Replicates HuggingFace's ``EsmModel``; the output is ``last_hidden_state``
    (the pooler and the contact head are not part of the embedding contract and
    are dropped).
    """

    default_task = "feature-extraction"
    category = "encoder"
    config_class = EsmConfig

    def __init__(self, config: EsmConfig):
        super().__init__()
        if config.position_embedding_type != "rotary":
            raise ValueError(
                "EsmModel currently builds the rotary ESM-2 variant; got "
                f"position_embedding_type={config.position_embedding_type!r}"
            )
        self.config = config
        self.embeddings = _EsmEmbeddings(config)
        self.encoder = _EsmEncoder(config)
        # One shared rotary table: HF instantiates a RotaryEmbedding per layer,
        # but every copy holds the same inv_freq, so a single cache is emitted.
        self.rotary_emb = DefaultRope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        token_type_ids: ir.Value,  # Unused: ESM has no token-type embeddings.
    ):
        del token_type_ids
        hidden_states = self.embeddings(op, input_ids, attention_mask)

        # Rotary positions are 0..seq_len-1 for every row, matching HF, which
        # derives them from the tensor shape rather than the mask. The ONNX
        # ``RotaryEmbedding`` op requires cos/sin to carry the same batch extent
        # as ``x``, so the row vector is expanded to (batch, seq) rather than
        # left at (1, seq).
        batch = op.Shape(input_ids, start=0, end=1)
        seq_len = op.Shape(input_ids, start=1, end=2)
        position_ids = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(seq_len),
            op.Constant(value_int=1),
        )
        position_ids = op.Cast(position_ids, to=7)  # INT64
        position_ids = op.Unsqueeze(position_ids, [0])  # (1, seq)
        position_ids = op.Expand(position_ids, op.Concat(batch, seq_len, axis=0))
        position_embeddings = self.rotary_emb(op, position_ids)

        # (batch, 1, seq, seq) bool mask; a rank-2 int mask cannot broadcast
        # onto the (batch, heads, q, kv) score tensor once batch > 1.
        padding_mask = create_padding_mask(op, input_ids, attention_mask)
        return self.encoder(op, hidden_states, padding_mask, position_embeddings)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace ESM weight names onto this module tree."""
        new_state_dict: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            new_name = _rename_esm_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


def _rename_esm_weight(name: str) -> str | None:
    """Rename one HuggingFace ESM weight, or drop it.

    Dropped: the masked-LM head, the contact head, the pooler, the unused
    absolute ``position_embeddings`` table, the ``position_ids`` buffer, and the
    per-layer ``inv_freq`` rotary buffers (this module emits one shared cos/sin
    cache computed from ``rope_theta`` instead).
    """
    if name.startswith("esm."):
        name = name[4:]

    if name.startswith(("lm_head.", "contact_head.", "pooler.", "cls.")):
        return None
    if name in ("embeddings.position_ids", "embeddings.position_embeddings.weight"):
        return None
    if name.endswith(".rotary_embeddings.inv_freq"):
        return None

    # attention.self.query -> attention.self.query (kept: HF nesting is mirrored
    # by the module tree), attention.output.dense -> attention.output.dense.
    return name
