# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph modules for the core architecture-specific GGUF VLM sidecars."""

from __future__ import annotations

from typing import cast

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, Gemma3nMultiModalConfig, Gemma4Config
from mobius.components import (
    Idefics3Projector,
    InternVLProjector,
    Llama4Projector,
    Llama4VisionTower,
    PixtralProjector,
)
from mobius.models.clip import ClipVisionConfigView, SigLIPVisionModel
from mobius.models.gemma3n import (
    _Gemma3nAudioEncoderModel,
    _Gemma3nVisionEncoderModel,
)
from mobius.models.gemma4 import (
    _Gemma4AudioEncoderModel,
    _Gemma4UnifiedAudioEmbedderModel,
    _Gemma4UnifiedVisionEmbedderModel,
)
from mobius.models.internvl import _InternVisionModel


class _Idefics3VisionEncoder(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vision = config.vision
        if vision is None:
            raise ValueError("Idefics3 GGUF sidecar requires a vision configuration")
        grid = int(vision.image_size or 0) // int(vision.patch_size or 1)
        merge = int(vision.spatial_merge_size)
        self.vision_tower = SigLIPVisionModel(ClipVisionConfigView(vision))
        self.projector = Idefics3Projector(
            int(vision.hidden_size or 0),
            config.hidden_size,
            grid_size=grid,
            scale_factor=merge,
        )
        self._output_hidden_size = config.hidden_size
        self.input_schema = (
            (
                "pixel_values",
                ir.DataType.FLOAT,
                ("num_tiles", 3, vision.image_size, vision.image_size),
            ),
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        pixel_values = op.CastLike(
            pixel_values,
            self.vision_tower.embeddings.patch_embedding.projection.weight,
        )
        # [tiles, rows_per_tile, hidden] -> [tiles * rows_per_tile, hidden].
        # Reshape preserves processor tile order, then raster order within each tile.
        return op.Reshape(
            self.projector(op, self.vision_tower(op, pixel_values)),
            [-1, self._output_hidden_size],
        )


class _InternVLVisionEncoder(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vision = config.vision
        if vision is None:
            raise ValueError("InternVL GGUF sidecar requires a vision configuration")
        grid = int(vision.image_size or 0) // int(vision.patch_size or 1)
        merge = int(vision.spatial_merge_size)
        # The GGUF reference appends CLS and assigns patch i position row i.
        self.vision_tower = _InternVisionModel(config, class_token_at_end=True)
        self.projector = InternVLProjector(
            int(vision.hidden_size or 0),
            config.hidden_size,
            grid_size=grid,
            scale_factor=merge,
        )
        self._output_hidden_size = config.hidden_size
        self.input_schema = (
            (
                "pixel_values",
                ir.DataType.FLOAT,
                ("num_tiles", 3, vision.image_size, vision.image_size),
            ),
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        pixel_values = op.CastLike(
            pixel_values,
            self.vision_tower.embeddings.patch_embedding.weight,
        )
        hidden_states = self.vision_tower(op, pixel_values)
        # llama.cpp drops the appended CLS row after the encoder.
        hidden_states = op.Slice(hidden_states, [0], [-1], [1])
        # Flatten tiles without inserting separators: InternVL's processor
        # already orders refined raster tiles before the overview tile.
        return op.Reshape(
            self.projector(op, hidden_states),
            [-1, self._output_hidden_size],
        )


class _Llama4VisionEncoder(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vision = config.vision
        if vision is None:
            raise ValueError("Llama4 GGUF sidecar requires a vision configuration")
        grid = int(vision.image_size or 0) // int(vision.patch_size or 1)
        merge = int(vision.spatial_merge_size)
        adapter_hidden = int(vision.projector_intermediate_size or 0)
        self.vision_tower = Llama4VisionTower(vision)
        self.projector = Llama4Projector(
            int(vision.hidden_size or 0),
            adapter_hidden,
            config.hidden_size,
            grid_size=grid,
            scale_factor=merge,
        )
        self._output_hidden_size = config.hidden_size
        self.input_schema = (
            (
                "pixel_values",
                ir.DataType.FLOAT,
                ("num_tiles", 3, vision.image_size, vision.image_size),
            ),
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        pixel_values = op.CastLike(
            pixel_values,
            self.vision_tower.embeddings.patch_embedding,
        )
        # Tile-major flattening preserves the processor's exact crop/overview order.
        return op.Reshape(
            self.projector(op, self.vision_tower(op, pixel_values)),
            [-1, self._output_hidden_size],
        )


class _PixtralVisionEncoder(nn.Module):
    def __init__(self, config: ArchitectureConfig, *, with_image_break: bool):
        super().__init__()
        from mobius.components import PixtralVisionTower

        vision = config.vision
        if vision is None:
            raise ValueError("Pixtral GGUF sidecar requires a vision configuration")
        self.vision_tower = PixtralVisionTower(config)
        self.projector = PixtralProjector(
            int(vision.hidden_size or 0),
            config.hidden_size,
            with_image_break=with_image_break,
        )
        self.input_schema = (
            ("pixel_values", ir.DataType.FLOAT, (1, 3, "image_height", "image_width")),
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        pixel_values = op.CastLike(pixel_values, self.vision_tower.patch_conv.weight)
        hidden_states, grid_height, grid_width = self.vision_tower(op, pixel_values)
        return self.projector(
            op,
            hidden_states,
            grid_height,
            grid_width,
        )


class _Gemma3nVisionSidecar(_Gemma3nVisionEncoderModel):
    def __init__(self, config: Gemma3nMultiModalConfig):
        super().__init__(config)
        image_size = int(config.vision.image_size)
        self.input_schema = (
            ("pixel_values", ir.DataType.FLOAT, (1, 3, image_size, image_size)),
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        pixel_values = op.CastLike(pixel_values, self.encoder.conv_stem.conv.weight)
        return super().forward(op, pixel_values)


class _Gemma4UnifiedVisionSidecar(_Gemma4UnifiedVisionEmbedderModel):
    def __init__(self, config: Gemma4Config):
        super().__init__(config)
        patch_size = int(config.vision.patch_size)
        pooling = int(config.vision.pooling_kernel_size)
        self.input_schema = (
            (
                "pixel_values",
                ir.DataType.FLOAT,
                (1, "num_patches", 3 * (patch_size * pooling) ** 2),
            ),
            (
                "pixel_position_ids",
                ir.DataType.INT64,
                (1, "num_patches", 2),
            ),
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        pixel_position_ids: ir.Value,
    ) -> ir.Value:
        pixel_values = op.CastLike(pixel_values, self.patch_ln1.weight)
        return super().forward(op, pixel_values, pixel_position_ids)


class CoreVLMProjectorModel(nn.Module):
    """One explicitly selected encoder role from a core GGUF projector sidecar."""

    default_task = "gguf-core-projector"
    category = "Multimodal"
    vision_encoder: nn.Module
    audio_encoder: nn.Module

    def __init__(
        self,
        config: ArchitectureConfig,
        projector_type: str,
        *,
        with_image_break: bool = False,
    ):
        super().__init__()
        self.projector_type = projector_type
        if projector_type == "gemma3nv":
            self.vision_encoder = _Gemma3nVisionSidecar(cast(Gemma3nMultiModalConfig, config))
        elif projector_type == "gemma3na":
            self.audio_encoder = _Gemma3nAudioEncoderModel(
                cast(Gemma3nMultiModalConfig, config)
            )
        elif projector_type == "gemma4a":
            self.audio_encoder = _Gemma4AudioEncoderModel(cast(Gemma4Config, config))
        elif projector_type == "gemma4uv":
            self.vision_encoder = _Gemma4UnifiedVisionSidecar(cast(Gemma4Config, config))
        elif projector_type == "gemma4ua":
            self.audio_encoder = _Gemma4UnifiedAudioEmbedderModel(cast(Gemma4Config, config))
        elif projector_type == "idefics3":
            self.vision_encoder = _Idefics3VisionEncoder(config)
        elif projector_type == "internvl":
            self.vision_encoder = _InternVLVisionEncoder(config)
        elif projector_type == "llama4":
            self.vision_encoder = _Llama4VisionEncoder(config)
        elif projector_type == "pixtral":
            self.vision_encoder = _PixtralVisionEncoder(
                config,
                with_image_break=with_image_break,
            )
        else:
            raise ValueError(f"Unknown core GGUF projector type {projector_type!r}")

    def forward(self, op: OpBuilder, **kwargs):
        del op, kwargs
        raise NotImplementedError(
            "CoreVLMProjectorModel is split into an explicit vision or audio encoder role."
        )
