# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standalone Qwen/GLM components for GGUF ``clip`` sidecars."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Glm4VVisionModel,
    GlmOcrVisionModel,
    Qwen3VLVisionModel,
    Qwen25VLVisionModel,
    SpeakerEncoder,
)
from mobius.models.qwen3_asr import Qwen3ASRAudioEncoder


def _guard_merged_grid_contract(
    op: OpBuilder,
    pixel_values: ir.Value,
    image_grid_thw: ir.Value,
) -> tuple[ir.Value, ir.Value]:
    temporal = op.Slice(image_grid_thw, [0], [1], [1], [1])
    height = op.Slice(image_grid_thw, [1], [2], [1], [1])
    width = op.Slice(image_grid_thw, [2], [3], [1], [1])
    invalid_grid = op.Or(
        op.LessOrEqual(image_grid_thw, op.Constant(value_int=0)),
        op.Concat(
            op.Equal(temporal, op.Constant(value_int=-1)),
            op.Not(op.Equal(op.Mod(height, 2), 0)),
            op.Not(op.Equal(op.Mod(width, 2), 0)),
            axis=1,
        ),
    )
    invalid_count = op.ReduceSum(
        op.Cast(invalid_grid, to=ir.DataType.INT64),
        keepdims=False,
    )
    expected_patches = op.ReduceSum(
        op.Mul(temporal, op.Mul(height, width)),
        keepdims=False,
    )
    actual_patches = op.Squeeze(op.Shape(pixel_values, start=0, end=1), [0])
    invalid_count = op.Add(
        invalid_count,
        op.Cast(
            op.Not(op.Equal(expected_patches, actual_patches)),
            to=ir.DataType.INT64,
        ),
    )
    guard = op.Gather(op.Constant(value_ints=[0]), invalid_count, axis=0)
    return (
        op.Add(pixel_values, op.CastLike(guard, pixel_values)),
        op.Add(image_grid_thw, op.CastLike(guard, image_grid_thw)),
    )


class GGUFGlm4VVisionProjector(nn.Module):
    """GLM4V vision encoder plus its downsampler and gated projector."""

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self._dtype = config.dtype
        vision = config.vision
        if vision is None:
            raise ValueError("GLM4V sidecar requires a vision configuration")
        required = (
            vision.hidden_size,
            vision.intermediate_size,
            vision.num_hidden_layers,
            vision.num_attention_heads,
            vision.patch_size,
            vision.out_hidden_size,
        )
        if any(value is None for value in required):
            raise ValueError("GLM4V sidecar vision dimensions must be complete")
        assert vision.hidden_size is not None
        assert vision.intermediate_size is not None
        assert vision.num_hidden_layers is not None
        assert vision.num_attention_heads is not None
        assert vision.patch_size is not None
        assert vision.out_hidden_size is not None
        self.visual = Glm4VVisionModel(
            depth=int(vision.num_hidden_layers),
            hidden_size=int(vision.hidden_size),
            intermediate_size=int(vision.intermediate_size),
            num_heads=int(vision.num_attention_heads),
            patch_size=int(vision.patch_size),
            temporal_patch_size=int(vision.temporal_patch_size),
            in_channels=int(vision.in_channels),
            out_hidden_size=int(vision.out_hidden_size),
            spatial_merge_size=int(vision.spatial_merge_size),
            norm_eps=float(vision.norm_eps),
            hidden_act=vision.hidden_act or "silu",
            num_position_embeddings=vision.num_position_embeddings,
            projector_intermediate_size=vision.projector_intermediate_size,
        )
        pixel_width = (
            int(vision.in_channels)
            * int(vision.temporal_patch_size)
            * int(vision.patch_size) ** 2
        )
        self.input_schema = (
            (
                "pixel_values",
                ir.DataType.FLOAT,
                (ir.SymbolicDim("total_patches"), pixel_width),
            ),
            (
                "image_grid_thw",
                ir.DataType.INT64,
                (ir.SymbolicDim("num_media"), 3),
            ),
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ) -> ir.Value:
        pixel_values, image_grid_thw = _guard_merged_grid_contract(
            op,
            pixel_values,
            image_grid_thw,
        )
        pixel_values = op.Cast(pixel_values, to=self._dtype)
        return self.visual(op, pixel_values, image_grid_thw)


class GGUFGlmOcrVisionProjector(nn.Module):
    """GLM-OCR tensor variant serialized under the glm4v projector route."""

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        vision = config.vision
        if vision is None:
            raise ValueError("GLM-OCR sidecar requires a vision configuration")
        required = (
            vision.hidden_size,
            vision.intermediate_size,
            vision.num_hidden_layers,
            vision.num_attention_heads,
            vision.patch_size,
            vision.out_hidden_size,
        )
        if any(value is None for value in required):
            raise ValueError("GLM-OCR sidecar vision dimensions must be complete")
        assert vision.hidden_size is not None
        assert vision.intermediate_size is not None
        assert vision.num_hidden_layers is not None
        assert vision.num_attention_heads is not None
        assert vision.patch_size is not None
        assert vision.out_hidden_size is not None
        self.visual = GlmOcrVisionModel(
            depth=int(vision.num_hidden_layers),
            hidden_size=int(vision.hidden_size),
            intermediate_size=int(vision.intermediate_size),
            num_heads=int(vision.num_attention_heads),
            patch_size=int(vision.patch_size),
            temporal_patch_size=int(vision.temporal_patch_size),
            in_channels=int(vision.in_channels),
            out_hidden_size=int(vision.out_hidden_size),
            spatial_merge_size=int(vision.spatial_merge_size),
            norm_eps=float(vision.norm_eps),
        )
        pixel_width = (
            int(vision.in_channels)
            * int(vision.temporal_patch_size)
            * int(vision.patch_size) ** 2
        )
        self.input_schema = (
            (
                "pixel_values",
                ir.DataType.FLOAT,
                (ir.SymbolicDim("total_patches"), pixel_width),
            ),
            (
                "image_grid_thw",
                ir.DataType.INT64,
                (ir.SymbolicDim("num_media"), 3),
            ),
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ) -> ir.Value:
        pixel_values, image_grid_thw = _guard_merged_grid_contract(
            op,
            pixel_values,
            image_grid_thw,
        )
        pixel_values = op.CastLike(pixel_values, self.visual.patch_embed.weight)
        return self.visual(op, pixel_values, image_grid_thw)


class GGUFQwen3VLProjector(nn.Module):
    """Qwen3-VL vision tower with final and DeepStack mergers packed by width."""

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        vision = config.vision
        if vision is None:
            raise ValueError("Qwen3-VL sidecar requires a vision configuration")
        required = (
            vision.hidden_size,
            vision.intermediate_size,
            vision.num_hidden_layers,
            vision.num_attention_heads,
            vision.patch_size,
            vision.out_hidden_size,
            vision.num_position_embeddings,
        )
        if any(value is None for value in required):
            raise ValueError("Qwen3-VL sidecar vision dimensions must be complete")
        assert vision.hidden_size is not None
        assert vision.intermediate_size is not None
        assert vision.num_hidden_layers is not None
        assert vision.num_attention_heads is not None
        assert vision.patch_size is not None
        assert vision.out_hidden_size is not None
        assert vision.num_position_embeddings is not None
        if int(vision.spatial_merge_size) != 2:
            raise ValueError("Qwen3-VL GGUF merger requires spatial_merge_size=2")
        self.visual = Qwen3VLVisionModel(
            depth=int(vision.num_hidden_layers),
            hidden_size=int(vision.hidden_size),
            intermediate_size=int(vision.intermediate_size),
            num_heads=int(vision.num_attention_heads),
            patch_size=int(vision.patch_size),
            temporal_patch_size=int(vision.temporal_patch_size),
            in_channels=int(vision.in_channels),
            out_hidden_size=int(vision.out_hidden_size),
            spatial_merge_size=2,
            num_position_embeddings=int(vision.num_position_embeddings),
            deepstack_visual_indexes=vision.deepstack_visual_indexes or [],
            # llama.cpp's FFN_GELU is the tanh approximation for both the
            # final merger and every DeepStack merger.
            merger_gelu_approximate="tanh",
        )
        pixel_width = (
            int(vision.in_channels)
            * int(vision.temporal_patch_size)
            * int(vision.patch_size) ** 2
        )
        self.input_schema = (
            (
                "pixel_values",
                ir.DataType.FLOAT,
                (ir.SymbolicDim("total_patches"), pixel_width),
            ),
            (
                "image_grid_thw",
                ir.DataType.INT64,
                (ir.SymbolicDim("num_media"), 3),
            ),
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ) -> ir.Value:
        pixel_values, image_grid_thw = _guard_merged_grid_contract(
            op,
            pixel_values,
            image_grid_thw,
        )
        pixel_values = op.CastLike(pixel_values, self.visual.patch_embed.weight)
        outputs = self.visual(op, pixel_values, image_grid_thw)
        merged = outputs[0]
        if len(outputs) == 1:
            return merged
        # llama.cpp concatenates DeepStack maps after the final map along the
        # embedding width: [tokens, text_hidden * (1 + deepstack_layers)].
        return op.Concat(merged, *outputs[1:], axis=1)


class GGUFQwen25OmniVisionProjector(nn.Module):
    """Qwen2.5-Omni legacy vision context resolved as qwen2.5vl_merger."""

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self._dtype = config.dtype
        vision = config.vision
        if vision is None:
            raise ValueError("Qwen2.5-Omni sidecar requires a vision configuration")
        required = (
            vision.hidden_size,
            vision.intermediate_size,
            vision.num_hidden_layers,
            vision.num_attention_heads,
            vision.patch_size,
            vision.out_hidden_size,
        )
        if any(value is None for value in required):
            raise ValueError("Qwen2.5-Omni vision dimensions must be complete")
        assert vision.hidden_size is not None
        assert vision.intermediate_size is not None
        assert vision.num_hidden_layers is not None
        assert vision.num_attention_heads is not None
        assert vision.patch_size is not None
        assert vision.out_hidden_size is not None
        self.visual = Qwen25VLVisionModel(
            depth=int(vision.num_hidden_layers),
            hidden_size=int(vision.hidden_size),
            intermediate_size=int(vision.intermediate_size),
            num_heads=int(vision.num_attention_heads),
            patch_size=int(vision.patch_size),
            temporal_patch_size=int(vision.temporal_patch_size),
            in_channels=int(vision.in_channels),
            out_hidden_size=int(vision.out_hidden_size),
            spatial_merge_size=int(vision.spatial_merge_size),
            fullatt_block_indexes=vision.fullatt_block_indexes,
            window_size=vision.window_size or 112,
        )
        pixel_width = (
            int(vision.in_channels)
            * int(vision.temporal_patch_size)
            * int(vision.patch_size) ** 2
        )
        self.input_schema = (
            (
                "pixel_values",
                ir.DataType.FLOAT,
                (ir.SymbolicDim("total_patches"), pixel_width),
            ),
            (
                "image_grid_thw",
                ir.DataType.INT64,
                (ir.SymbolicDim("num_media"), 3),
            ),
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ) -> ir.Value:
        pixel_values, image_grid_thw = _guard_merged_grid_contract(
            op,
            pixel_values,
            image_grid_thw,
        )
        pixel_values = op.Cast(pixel_values, to=self._dtype)
        return self.visual(op, pixel_values, image_grid_thw)


class GGUFQwen3AudioProjector(nn.Module):
    """Qwen3 audio encoder for one preprocessor-padded 100-frame window group."""

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        if config.audio is None:
            raise ValueError("Qwen3 audio sidecar requires an audio configuration")
        self.audio_tower = Qwen3ASRAudioEncoder(config)
        self.input_schema = (
            (
                "input_features",
                ir.DataType.FLOAT,
                (
                    1,
                    config.audio.num_mel_bins or 128,
                    ir.SymbolicDim("audio_frames_multiple_of_100"),
                ),
            ),
        )

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        input_features = op.CastLike(
            input_features,
            self.audio_tower.conv2d1.weight,
        )
        batch = op.Shape(input_features, start=0, end=1)
        frames = op.Shape(input_features, start=2, end=3)
        frame_count = op.Squeeze(frames, [0])
        invalid_frames = op.Cast(
            op.Or(
                op.LessOrEqual(frame_count, op.Constant(value_int=0)),
                op.Not(op.Equal(op.Mod(frame_count, 100), op.Constant(value_int=0))),
            ),
            to=ir.DataType.INT64,
        )
        guard = op.Gather(op.Constant(value_ints=[0]), invalid_frames, axis=0)
        input_features = op.Add(input_features, op.CastLike(guard, input_features))
        mask_shape = op.Concat(batch, frames, axis=0)
        feature_mask = op.Expand(
            op.Constant(value_int=1),
            mask_shape,
        )
        audio_features, _ = self.audio_tower(op, input_features, feature_mask)
        return op.Squeeze(audio_features, [0])


class GGUFQwen3TTSSpeakerProjector(nn.Module):
    """Qwen3-TTS ECAPA-TDNN speaker embedding sidecar."""

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        tts = config.tts
        speaker = tts.speaker_encoder if tts else None
        if speaker is None:
            raise ValueError("Qwen3-TTS sidecar requires a speaker encoder configuration")
        self.encoder = SpeakerEncoder(
            config,
            mel_dim=speaker.mel_dim,
            enc_dim=speaker.enc_dim,
            enc_channels=speaker.enc_channels,
            enc_kernel_sizes=speaker.enc_kernel_sizes,
            enc_dilations=speaker.enc_dilations,
            enc_attention_channels=speaker.enc_attention_channels,
            enc_res2net_scale=speaker.enc_res2net_scale,
            enc_se_channels=speaker.enc_se_channels,
        )
        self.input_schema = (
            (
                "mel_features",
                ir.DataType.FLOAT,
                (ir.SymbolicDim("audio_frames"), speaker.mel_dim),
            ),
        )

    def forward(self, op: OpBuilder, mel_features: ir.Value) -> ir.Value:
        # mtmd preprocessing produces one unbatched [frames, mel_bins] clip.
        return self.encoder(op, op.Unsqueeze(mel_features, [0]))
