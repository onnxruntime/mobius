# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generic three-model wrapper for llama.cpp ``clip`` projector sidecars."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from onnxscript import OpBuilder, nn

from mobius.components import (
    GGUFMLPProjector,
    GLMEdgeAdapterProjector,
    MiniCPMResamplerProjector,
    MobileLDPProjector,
    MobileLDPV2Projector,
)
from mobius.models.base import embedding_for_config
from mobius.models.clip import ClipVisionConfigView, CLIPVisionModel, SigLIPVisionModel

if TYPE_CHECKING:
    import onnx_ir as ir

    from mobius._configs import ArchitectureConfig


class _DecoderFromEmbeds(nn.Module):
    """Expose any supported causal model as an ``inputs_embeds`` decoder."""

    def __init__(self, causal_lm: Any):
        super().__init__()
        self.model = causal_lm.model
        self.lm_head = causal_lm.lm_head

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        result = self.model(
            op,
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        hidden_states, present_key_values = result[:2]
        return self.lm_head(op, hidden_states), present_key_values


class _EmbeddingWithImageFeatures(nn.Module):
    """Token embedding lookup followed by image-placeholder replacement."""

    def __init__(
        self,
        config: ArchitectureConfig,
        image_token_id: int,
        *,
        image_start_token_id: int | None = None,
        image_end_token_id: int | None = None,
    ):
        super().__init__()
        if (image_start_token_id is None) != (image_end_token_id is None):
            raise ValueError("Image start and end token IDs must be specified together")
        self.embed_tokens = embedding_for_config(config)
        self._image_token_id = image_token_id
        self._image_start_token_id = image_start_token_id
        self._image_end_token_id = image_end_token_id
        self._embedding_multiplier = float(config.embedding_multiplier)

    def forward(self, op: OpBuilder, input_ids: ir.Value, image_features: ir.Value):
        image_mask = op.Equal(input_ids, op.Constant(value_int=self._image_token_id))
        if self._image_start_token_id is not None:
            starts = op.CumSum(
                op.Cast(
                    op.Equal(
                        input_ids,
                        op.Constant(value_int=self._image_start_token_id),
                    ),
                    to=7,
                ),
                op.Constant(value_int=1),
            )
            ends = op.CumSum(
                op.Cast(
                    op.Equal(
                        input_ids,
                        op.Constant(value_int=self._image_end_token_id),
                    ),
                    to=7,
                ),
                op.Constant(value_int=1),
            )
            image_mask = op.And(image_mask, op.Greater(starts, ends))

        # Some processors use an out-of-vocabulary sentinel (for example -200).
        # Replace media slots before embedding lookup so Gather never sees it.
        safe_input_ids = op.Where(
            image_mask,
            op.CastLike(0, input_ids),
            input_ids,
        )
        text_embeds = self.embed_tokens(op, safe_input_ids)
        if not math.isclose(self._embedding_multiplier, 1.0):
            text_embeds = op.Mul(
                text_embeds,
                op.CastLike(self._embedding_multiplier, text_embeds),
            )

        image_mask_3d = op.Unsqueeze(image_mask, [-1])
        flat_mask = op.Reshape(image_mask, [-1])
        indices = op.Sub(
            op.CumSum(op.Cast(flat_mask, to=7), op.Constant(value_int=0)),
            op.Constant(value_int=1),
        )
        indices = op.Clip(indices, op.Constant(value_int=0))
        indices = op.Reshape(indices, op.Shape(input_ids))

        # Keep text-only calls valid even when the feature input has zero rows.
        zero_row = op.Expand(
            op.CastLike(0.0, image_features),
            op.Concat([1], op.Shape(image_features, start=1, end=2), axis=0),
        )
        features = op.Concat(image_features, zero_row, axis=0)
        return op.Where(image_mask_3d, op.Gather(features, indices, axis=0), text_embeds)


class _ProjectorVisionEncoder(nn.Module):
    """CLIP/SigLIP tower plus one exact generic llama.cpp projector."""

    def __init__(
        self,
        config: ArchitectureConfig,
        *,
        projector_type: str,
        projector_hidden_size: int,
        projector_intermediate_size: int | None = None,
        num_queries: int | None = None,
        mlp_has_second_layer: bool = True,
    ):
        super().__init__()
        if config.vision is None:
            raise ValueError("Generic GGUF projector requires a vision configuration")
        vision = config.vision
        if (
            vision.image_size is None
            or vision.patch_size is None
            or vision.hidden_size is None
        ):
            raise ValueError("Generic GGUF projector vision dimensions must be defined")
        vision_width = int(vision.hidden_size)
        grid = int(vision.image_size) // int(vision.patch_size)
        clip_view = ClipVisionConfigView(vision)
        self.vision_tower: CLIPVisionModel | SigLIPVisionModel

        if projector_type in {"mlp", "ldp", "ldpv2"}:
            # Legacy CLIP sidecars serialize the selected layers only and omit
            # post_ln. The pinned loader excludes the final serialized block
            # when no explicit feature layer is present.
            self.vision_tower = CLIPVisionModel(
                clip_view,
                feature_layer=-2,
                drop_class_token=True,
            )
        elif projector_type in {"adapter", "resampler"}:
            self.vision_tower = SigLIPVisionModel(clip_view)
        else:
            raise ValueError(f"Unknown generic GGUF projector type {projector_type!r}")

        self.projector: nn.Module
        if projector_type == "mlp":
            self.projector = GGUFMLPProjector(
                vision_width,
                projector_hidden_size,
                has_second_layer=mlp_has_second_layer,
            )
        elif projector_type == "ldp":
            self.projector = MobileLDPProjector(
                vision_width,
                projector_hidden_size,
                grid_size=grid,
                eps=vision.norm_eps,
            )
        elif projector_type == "ldpv2":
            self.projector = MobileLDPV2Projector(
                vision_width,
                projector_hidden_size,
                grid_size=grid,
            )
        elif projector_type == "adapter":
            if projector_intermediate_size is None:
                raise ValueError("GLM-Edge adapter requires its gated intermediate size")
            self.projector = GLMEdgeAdapterProjector(
                vision_width,
                projector_hidden_size,
                projector_intermediate_size,
                grid_size=grid,
                eps=vision.norm_eps,
            )
        else:
            if num_queries is None:
                raise ValueError("MiniCPM resampler requires a query count")
            self.projector = MiniCPMResamplerProjector(
                vision_width,
                projector_hidden_size,
                num_queries=num_queries,
                grid_size=grid,
                eps=vision.norm_eps,
            )

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        # Processor boundary stays FLOAT; reduced-precision packages cast once.
        pixel_values = op.CastLike(
            pixel_values, self.vision_tower.embeddings.patch_embedding.projection.weight
        )
        return self.projector(op, self.vision_tower(op, pixel_values))


class GenericGGUFProjectorModel(nn.Module):
    """Decoder, vision-projector, and embedding modules for generic sidecars."""

    def __init__(
        self,
        config: ArchitectureConfig,
        causal_lm: Any,
        *,
        projector_type: str,
        projector_hidden_size: int,
        image_token_id: int,
        projector_intermediate_size: int | None = None,
        num_queries: int | None = None,
        mlp_has_second_layer: bool = True,
        image_start_token_id: int | None = None,
        image_end_token_id: int | None = None,
    ):
        super().__init__()
        self.decoder = _DecoderFromEmbeds(causal_lm)
        self.vision_encoder = _ProjectorVisionEncoder(
            config,
            projector_type=projector_type,
            projector_hidden_size=projector_hidden_size,
            projector_intermediate_size=projector_intermediate_size,
            num_queries=num_queries,
            mlp_has_second_layer=mlp_has_second_layer,
        )
        self.embedding = _EmbeddingWithImageFeatures(
            config,
            image_token_id,
            image_start_token_id=image_start_token_id,
            image_end_token_id=image_end_token_id,
        )

    def forward(self, op: OpBuilder, **kwargs):
        del op, kwargs
        raise NotImplementedError(
            "GenericGGUFProjectorModel is built as decoder, vision, and embedding graphs."
        )
