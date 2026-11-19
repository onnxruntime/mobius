# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LiquidAI LFM3-Omni: LFM3-Audio's speech input and output plus LFM3-VL's image input.

Builds the LFM3-Audio graphs (audio encoder, embedding, decoder, depth decoder)
and the LFM3-VL vision encoder, which takes one image per call. The embedding
scatters image features, audio features and summed speech-frame codebooks onto
their placeholder tokens. Video input is not supported.

HuggingFace weight prefixes -> ONNX initializer names::

    model.vision_tower.*            -> vision_encoder.vision_tower.*
    model.multi_modal_projector.*   -> vision_encoder.multi_modal_projector.*
    everything else                 -> as for LFM3-Audio
"""

from __future__ import annotations

import onnx_ir as ir
import torch
from onnxscript import OpBuilder

from mobius._configs.lfm3_omni import Lfm3OmniConfig
from mobius.models.lfm3_audio import (
    Lfm3AudioEmbedding,
    Lfm3AudioForConditionalGeneration,
)
from mobius.models.lfm3_vl import _Lfm3VlVisionEncoderModel

_VISION_PREFIXES = ("model.vision_tower.", "model.multi_modal_projector.")


class Lfm3OmniEmbedding(Lfm3AudioEmbedding):
    """LFM3-Audio's fusion, then image features at ``image_token_id`` slots."""

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
        audio_features: ir.Value,
        audio_out_codes: ir.Value,
    ) -> ir.Value:
        embeddings = super().forward(op, input_ids, audio_features, audio_out_codes)
        return self._replace(
            op, embeddings, input_ids, image_features, self.config.image_token_id
        )


class Lfm3OmniForConditionalGeneration(Lfm3AudioForConditionalGeneration):
    """LiquidAI LFM3-Omni image, audio and text input with text and speech-code output.

    Replicates the remote-code ``Lfm3OmniForConditionalGeneration`` of
    ``LiquidAI/LFM3-Omni-2B``: LFM3-Audio's speech encoder, LFM3 decoder and
    MaskGIT depth decoder, plus LFM3-VL's vision tower and projector. LiquidCodec
    converts the generated 12-codebook frames to 24-kHz audio. Float32 only.
    """

    default_task: str = "lfm3-omni"
    category: str = "Multimodal"
    config_class: type = Lfm3OmniConfig

    def __init__(self, config: Lfm3OmniConfig):
        super().__init__(config)
        self.embedding = Lfm3OmniEmbedding(config)
        self.vision_encoder = _Lfm3VlVisionEncoderModel(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        weights = super().preprocess_weights(state_dict)
        return {
            "vision_encoder." + name.removeprefix("model.")
            if name.startswith(_VISION_PREFIXES)
            else name: value
            for name, value in weights.items()
        }
