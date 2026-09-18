# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma 3n multimodal embedder.

Bridges the vision and audio towers into the text embedding space.  Gemma 3n
uses one of these per modality (``model.embed_vision.`` and
``model.embed_audio.`` in the checkpoint, 4 tensors each), and both are
consumed through *two* distinct paths:

* the **soft** path embeds continuous encoder features (the vision tower's
  256 soft tokens, the audio tower's 188);
* the **hard** path embeds the reserved placeholder *token ids* that the
  processor splices into the prompt, via a 128-entry lookup table offset by
  ``vocab_offset``.

HF reference: ``Gemma3nMultimodalEmbedder`` in
``transformers.models.gemma3n.modeling_gemma3n``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius.components._common import Embedding, Linear
from mobius.components._rms_norm import RMSNorm, ScaleFreeRMSNorm

if TYPE_CHECKING:
    import onnx_ir as ir


class Gemma3nMultimodalEmbedder(nn.Module):
    """Projects modality features (or placeholder token ids) into text space.

    Weight names match the checkpoint exactly (``embedding``,
    ``hard_embedding_norm``, ``soft_embedding_norm``, ``embedding_projection``),
    so no renaming is needed in ``preprocess_weights``.
    ``embedding_post_projection_norm`` is scale-free — HF builds it with
    ``with_scale=False`` and the checkpoint ships no weight for it, which is
    why it is a :class:`ScaleFreeRMSNorm` rather than a :class:`RMSNorm`.

    Args:
        multimodal_hidden_size: Width of the encoder output (vision 2048,
            audio 1536 for E4B).
        text_hidden_size: Decoder width to project into.
        vocab_size: Number of reserved placeholder tokens (128).
        vocab_offset: First placeholder token id (262144 vision, 262272 audio).
        eps: RMSNorm epsilon (``rms_norm_eps`` of the sub-config).
    """

    def __init__(
        self,
        multimodal_hidden_size: int,
        text_hidden_size: int,
        vocab_size: int = 128,
        vocab_offset: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.multimodal_hidden_size = multimodal_hidden_size
        self.text_hidden_size = text_hidden_size
        self.vocab_size = vocab_size
        self.vocab_offset = vocab_offset

        self.embedding = Embedding(vocab_size, multimodal_hidden_size)
        self.hard_embedding_norm = RMSNorm(multimodal_hidden_size, eps=eps)
        self.soft_embedding_norm = RMSNorm(multimodal_hidden_size, eps=eps)
        self.embedding_projection = Linear(
            multimodal_hidden_size, text_hidden_size, bias=False
        )
        self.embedding_post_projection_norm = ScaleFreeRMSNorm(text_hidden_size, eps=eps)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value | None = None,
        input_ids: ir.Value | None = None,
    ) -> ir.Value:
        """Embed either encoder features (soft) or placeholder token ids (hard).

        Exactly one argument must be given, matching HF's signature.  Both
        paths share the projection tail, and the branch is resolved at graph
        *build* time — the unused branch emits no nodes, so a graph that only
        needs one path carries only that path's initializers.

        Args:
            inputs_embeds: ``[batch, tokens, multimodal_hidden_size]`` encoder
                output (the soft path).
            input_ids: Token ids in ``[vocab_offset, vocab_offset + vocab_size)``
                (the hard path).  Callers must substitute an in-range dummy id
                at non-placeholder positions, as HF does, because ONNX
                ``Gather`` does not bounds-check.

        Returns:
            ``[..., text_hidden_size]``.
        """
        if (inputs_embeds is None) == (input_ids is None):
            raise ValueError("Specify exactly one of inputs_embeds or input_ids")

        if inputs_embeds is not None:
            normed = self.soft_embedding_norm(op, inputs_embeds)
        else:
            # CastLike keeps the offset in the caller's index dtype (int32/int64).
            offset = op.CastLike(op.Constant(value_int=self.vocab_offset), input_ids)
            hard_embeds = self.embedding(op, op.Sub(input_ids, offset))
            normed = self.hard_embedding_norm(op, hard_embeds)

        projected = self.embedding_projection(op, normed)
        return self.embedding_post_projection_norm(op, projected)
