# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""LFM2-VL vision-language model (3-model split).

Architecture:
    vision_encoder: SigLIP2 encoder + MLP projector
        pixel_values → image_features (num_image_tokens, text_hidden)
    embedding: token embed + image feature scatter
        input_ids + image_features → inputs_embeds
    decoder: LFM2 text backbone
        inputs_embeds → logits + hybrid KV cache

Projector (vision_hidden → projector_hidden → text_hidden):
    Linear(vision_hidden, projector_hidden) → GELU → Linear(projector_hidden, text_hidden)

HuggingFace weight prefixes:
    language_model.*     → decoder.*  (+ embedding.embed_tokens.*)
    vision_tower.*       → vision_encoder.vision_tower.*
    projector.linear_1.* → vision_encoder.projector.linear_1.*
    projector.linear_2.* → vision_encoder.projector.linear_2.*

HuggingFace reference: ``Lfm2VlForConditionalGeneration``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import Lfm2VlConfig
from mobius.components import Embedding, Linear, VisionModel
from mobius.models.lfm2 import _Lfm2TextModel, _rename_lfm2_weight

if TYPE_CHECKING:
    import onnx_ir as ir


class _Lfm2VlProjector(nn.Module):
    """Two-layer MLP projector: Linear → GELU → Linear.

    Maps vision features from vision_hidden to text_hidden space via an
    intermediate projector_hidden dimension.

    vision_hidden → projector_hidden → text_hidden
    """

    def __init__(self, config: Lfm2VlConfig):
        super().__init__()
        vision_hidden = config.vision.hidden_size if config.vision else 1152
        self.linear_1 = Linear(
            vision_hidden, config.projector_hidden_size, bias=config.projector_bias
        )
        self.linear_2 = Linear(
            config.projector_hidden_size, config.hidden_size, bias=config.projector_bias
        )

    def forward(self, op: builder.OpBuilder, features: ir.Value):
        # features: (batch, num_patches, vision_hidden)
        hidden = self.linear_1(op, features)
        hidden = op.Gelu(hidden)  # (batch, num_patches, projector_hidden)
        return self.linear_2(op, hidden)  # (batch, num_patches, text_hidden)


class _Lfm2VlVisionModel(nn.Module):
    """LFM2-VL vision encoder: SigLIP2 + MLP projector.

    Encodes pixel_values into projected image features.
    Output shape: (num_image_tokens, text_hidden) — batch is folded in.
    """

    def __init__(self, config: Lfm2VlConfig):
        super().__init__()
        self.vision_tower = VisionModel(config)
        self.projector = _Lfm2VlProjector(config)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # pixel_values: (batch, 3, H, W)
        vision_features = self.vision_tower(op, pixel_values)
        # vision_features: (batch, num_patches, vision_hidden)
        projected = self.projector(op, vision_features)
        # projected: (batch, num_patches, text_hidden)
        # Flatten batch*num_patches → num_image_tokens for embedding model
        text_hidden_size = op.Shape(projected, start=2, end=3)
        minus_one = op.Constant(value_ints=[-1])
        flat_shape = op.Concat(minus_one, text_hidden_size, axis=0)
        return op.Reshape(projected, flat_shape)  # (num_image_tokens, text_hidden)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in state_dict.items()
            if key.startswith(("vision_tower.", "projector."))
        }


class _Lfm2VlEmbedding(nn.Module):
    """LFM2-VL embedding model: token lookup + image feature scatter.

    Replaces image token positions in the text embedding sequence with
    the projected vision features from the vision encoder.
    """

    def __init__(self, config: Lfm2VlConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.image_token_id = config.image_token_id

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value, image_features: ir.Value):
        # input_ids: (batch, seq_len)
        # image_features: (num_image_tokens, text_hidden)
        text_embeds = self.embed_tokens(op, input_ids)  # (batch, seq_len, text_hidden)

        # Build mask for image token positions
        image_mask = op.Equal(
            input_ids,
            op.Constant(value_int=self.image_token_id),
        )  # (batch, seq_len)
        image_mask_3d = op.Unsqueeze(image_mask, [-1])  # (batch, seq_len, 1)

        # CumSum-based index into image_features
        mask_int = op.Cast(image_mask, to=7)  # INT64
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        gathered = op.Gather(image_features, indices, axis=0)
        return op.Where(image_mask_3d, gathered, text_embeds)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {key: value for key, value in state_dict.items() if "embed_tokens" in key}


class _Lfm2VlDecoder(nn.Module):
    """LFM2-VL text decoder: inputs_embeds → logits + hybrid KV cache.

    Takes pre-embedded inputs (from the embedding model) and runs the
    LFM2 text backbone. Uses hybrid cache (conv_state for conv layers,
    KV cache for attention layers).
    """

    def __init__(self, config: Lfm2VlConfig):
        super().__init__()
        self.config = config
        self.model = _Lfm2TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _rename_lfm2_weight(key)
            renamed[new_key] = value
        # Handle weight tying
        if self.config.tie_word_embeddings:
            embed_key = "model.embed_tokens.weight"
            head_key = "lm_head.weight"
            if head_key not in renamed and embed_key in renamed:
                renamed[head_key] = renamed[embed_key]
        return renamed


class Lfm2VlModel(nn.Module):
    """LFM2-VL vision-language model (3-model ONNX split).

    Produces three separate ONNX models:
    - ``decoder``: LFM2 text decoder (inputs_embeds → logits + hybrid cache)
    - ``vision``: SigLIP2 encoder + MLP projector (pixel_values → image_features)
    - ``embedding``: token lookup + image feature scatter
      (input_ids + image_features → inputs_embeds)

    HuggingFace reference: ``Lfm2VlForConditionalGeneration``.
    """

    default_task: str = "hybrid-vision-language"
    category: str = "Multimodal"
    config_class: type = Lfm2VlConfig

    def __init__(self, config: Lfm2VlConfig):
        super().__init__()
        self.config = config
        self.decoder = _Lfm2VlDecoder(config)
        self.vision_encoder = _Lfm2VlVisionModel(config)
        self.embedding = _Lfm2VlEmbedding(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "Lfm2VlModel uses HybridVisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HF LFM2-VL weights to ONNX initializer names.

        HF checkpoint layout → ONNX initializer names::

            ``language_model.`` → ``decoder.``
            ``vision_tower.``   → ``vision_encoder.vision_tower.``
            ``projector.``      → ``vision_encoder.projector.``

        ``language_model.model.embed_tokens.weight`` is also copied into
        ``embedding.embed_tokens.weight`` for the embedding sub-model.
        """
        # Handle weight tying in language model
        if self.config.tie_word_embeddings:
            lm_embed = "language_model.model.embed_tokens.weight"
            lm_head = "language_model.lm_head.weight"
            if lm_head not in state_dict and lm_embed in state_dict:
                state_dict[lm_head] = state_dict[lm_embed]

        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("language_model."):
                # Strip 'language_model.' prefix and apply LFM2 renames
                inner_key = key[len("language_model.") :]
                renamed_inner = _rename_lfm2_weight(inner_key)
                renamed[f"decoder.{renamed_inner}"] = value
                # Duplicate embed_tokens for the embedding sub-model
                if key == "language_model.model.embed_tokens.weight":
                    renamed["embedding.embed_tokens.weight"] = value
            elif key.startswith("vision_tower."):
                renamed[f"vision_encoder.{key}"] = value
            elif key.startswith("projector."):
                renamed[f"vision_encoder.{key}"] = value
            else:
                renamed[key] = value

        return renamed
