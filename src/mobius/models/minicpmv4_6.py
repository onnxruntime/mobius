# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""OpenBMB MiniCPM-V-4.6 vision-language model.

Replicates ``MiniCPMV4_6ForConditionalGeneration`` from Transformers 5.7:
packed variable-resolution SigLIP2 vision, an in-tower window-attention
merger, a second MLP spatial merger, and the Qwen3.5 hybrid text decoder.
"""

from __future__ import annotations

from typing import cast

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Embedding,
    LayerNorm,
    Linear,
    VisionAttention,
    VisionEncoderLayer,
    get_activation,
)
from mobius.models.qwen35 import Qwen35TextModel


class MiniCPMV46Config(ArchitectureConfig):
    """Mobius configuration preserving MiniCPM's composite model identity."""

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> MiniCPMV46Config:
        result = super().from_transformers(config, parent_config=parent_config)
        composite = parent_config or config
        result.model_type = "minicpmv4_6"
        result.video_token_id = getattr(composite, "video_token_id", None)
        result.downsample_mode = getattr(composite, "downsample_mode", "16x")
        return cast(MiniCPMV46Config, result)


def _grid_columns(op: OpBuilder, target_sizes: ir.Value) -> tuple[ir.Value, ir.Value]:
    """Return the per-visual-unit patch-grid heights and widths."""
    target_sizes = op.Cast(target_sizes, to=7)
    heights = op.Squeeze(
        op.Slice(target_sizes, [0], [1], axes=[1]),
        [1],
    )
    widths = op.Squeeze(
        op.Slice(target_sizes, [1], [2], axes=[1]),
        [1],
    )
    return heights, widths


def _packed_patch_coordinates(
    op: OpBuilder, target_sizes: ir.Value
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """Map each packed patch to ``(visual_unit, row, column)``."""
    heights, widths = _grid_columns(op, target_sizes)
    lengths = op.Mul(heights, widths)
    ends = op.CumSum(lengths, 0)
    starts = op.Sub(ends, lengths)
    total = op.ReduceSum(lengths, keepdims=0)
    packed_index = op.Range(0, total, 1)

    # The first cumulative end greater than each packed index identifies its
    # source image crop or video frame.
    reached_end = op.GreaterOrEqual(
        op.Unsqueeze(packed_index, [1]),
        op.Unsqueeze(ends, [0]),
    )
    visual_index = op.ReduceSum(
        op.Cast(reached_end, to=7),
        [1],
        keepdims=0,
    )
    local_index = op.Sub(packed_index, op.Gather(starts, visual_index))
    local_width = op.Gather(widths, visual_index)
    rows = op.Div(local_index, local_width)
    columns = op.Mod(local_index, local_width)
    return visual_index, rows, columns


def _max_grid_size(op: OpBuilder, target_sizes: ir.Value) -> tuple[ir.Value, ir.Value]:
    heights, widths = _grid_columns(op, target_sizes)
    return (
        op.ReduceMax(heights, keepdims=1),
        op.ReduceMax(widths, keepdims=1),
    )


def _grid_mask(
    op: OpBuilder,
    target_sizes: ir.Value,
    max_height: ir.Value,
    max_width: ir.Value,
) -> ir.Value:
    """Boolean ``[N, max_h, max_w]`` mask for a padded ragged patch grid."""
    heights, widths = _grid_columns(op, target_sizes)
    row_ids = op.Range(0, op.Squeeze(max_height, [0]), 1)
    column_ids = op.Range(0, op.Squeeze(max_width, [0]), 1)
    valid_rows = op.Less(
        op.Unsqueeze(row_ids, [0]),
        op.Unsqueeze(heights, [1]),
    )
    valid_columns = op.Less(
        op.Unsqueeze(column_ids, [0]),
        op.Unsqueeze(widths, [1]),
    )
    return op.And(
        op.Unsqueeze(valid_rows, [2]),
        op.Unsqueeze(valid_columns, [1]),
    )


def _unpack_padded_grid(
    op: OpBuilder,
    hidden_states: ir.Value,
    target_sizes: ir.Value,
    hidden_size: int,
) -> ir.Value:
    """Scatter packed patches into ``[N, max_h*max_w, D]`` padded batches."""
    visual_index, rows, columns = _packed_patch_coordinates(op, target_sizes)
    coordinates = op.Concat(
        op.Unsqueeze(visual_index, [1]),
        op.Unsqueeze(rows, [1]),
        op.Unsqueeze(columns, [1]),
        axis=1,
    )
    max_height, max_width = _max_grid_size(op, target_sizes)
    grid_shape = op.Concat(
        op.Shape(target_sizes, start=0, end=1),
        max_height,
        max_width,
        op.Constant(value_ints=[hidden_size]),
        axis=0,
    )
    patches = op.Squeeze(hidden_states, [0])
    padded = op.Expand(op.CastLike(0.0, patches), grid_shape)
    padded = op.ScatterND(padded, coordinates, patches)
    return op.Reshape(
        padded,
        op.Concat(
            op.Shape(target_sizes, start=0, end=1),
            op.Mul(max_height, max_width),
            op.Constant(value_ints=[hidden_size]),
            axis=0,
        ),
    )


def _vision_attention_bias(
    op: OpBuilder,
    hidden_states: ir.Value,
    target_sizes: ir.Value,
) -> ir.Value:
    """Build a full additive mask for padded per-image patch batches."""
    max_height, max_width = _max_grid_size(op, target_sizes)
    valid = op.Reshape(
        _grid_mask(op, target_sizes, max_height, max_width),
        op.Concat(
            op.Shape(target_sizes, start=0, end=1),
            op.Mul(max_height, max_width),
            axis=0,
        ),
    )
    key_bias = op.Where(
        op.Unsqueeze(valid, [1, 2]),
        op.CastLike(0.0, hidden_states),
        op.CastLike(-10_000.0, hidden_states),
    )
    sequence_length = op.Mul(max_height, max_width)
    return op.Expand(
        key_bias,
        op.Concat(
            op.Shape(target_sizes, start=0, end=1),
            op.Constant(value_ints=[1]),
            sequence_length,
            sequence_length,
            axis=0,
        ),
    )


def _spatial_windows(
    op: OpBuilder,
    hidden_states: ir.Value,
    target_sizes: ir.Value,
    hidden_size: int,
    kernel_size: tuple[int, int],
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """Arrange ``[B, H*W, D]`` as ``[B, H/k, W/k, k, k, D]`` windows."""
    batch = op.Shape(target_sizes, start=0, end=1)
    height, width = _max_grid_size(op, target_sizes)
    kernel_h, kernel_w = kernel_size
    merged_h = op.Div(height, op.Constant(value_ints=[kernel_h]))
    merged_w = op.Div(width, op.Constant(value_ints=[kernel_w]))
    grid = op.Reshape(
        hidden_states,
        op.Concat(
            batch,
            merged_h,
            op.Constant(value_ints=[kernel_h]),
            merged_w,
            op.Constant(value_ints=[kernel_w, hidden_size]),
            axis=0,
        ),
    )
    # (B, H/k, k, W/k, k, D) -> (B, H/k, W/k, k, k, D)
    grid = op.Transpose(grid, perm=[0, 1, 3, 2, 4, 5])
    return grid, merged_h, merged_w


class _MiniCPMVisionEmbeddings(nn.Module):
    """NaViT patch embedding with nearest-neighbor learned 2D positions."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        num_channels: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.position_side = image_size // patch_size
        self.patch_embedding = nn.Parameter(
            [hidden_size, num_channels, patch_size, patch_size],
            name="patch_embedding.weight",
        )
        self.patch_embedding_bias = nn.Parameter([hidden_size], name="patch_embedding.bias")
        self.position_embedding = nn.Parameter(
            [self.position_side**2, hidden_size],
            name="position_embedding.weight",
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        target_sizes: ir.Value,
    ):
        # NaViT input is packed horizontally: (1, C, patch_size, total_patch_width).
        patches = op.Conv(
            pixel_values,
            self.patch_embedding,
            self.patch_embedding_bias,
            kernel_shape=[self.patch_size, self.patch_size],
            strides=[self.patch_size, self.patch_size],
        )
        # (packed_batch, D, 1, patches) -> (1, total_patches, D)
        patches = op.Transpose(patches, perm=[0, 2, 3, 1])
        patches = op.Reshape(
            patches,
            op.Constant(value_ints=[1, -1, self.hidden_size]),
        )

        visual_index, rows, columns = _packed_patch_coordinates(op, target_sizes)
        heights, widths = _grid_columns(op, target_sizes)
        patch_heights = op.Gather(heights, visual_index)
        patch_widths = op.Gather(widths, visual_index)
        side = op.Constant(value_int=self.position_side)
        bucket_h = op.Div(op.Mul(rows, side), patch_heights)
        bucket_w = op.Div(op.Mul(columns, side), patch_widths)
        position_ids = op.Add(op.Mul(bucket_h, side), bucket_w)
        position_embeddings = op.Gather(self.position_embedding, position_ids)
        return op.Add(patches, op.Unsqueeze(position_embeddings, [0]))


class _MiniCPMViTWindowAttentionMerger(nn.Module):
    """2x2 local attention followed by spatial MLP compression."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vision = config.vision
        assert vision is not None
        assert vision.hidden_size is not None
        assert vision.intermediate_size is not None
        self.hidden_size = vision.hidden_size
        self.kernel_size = vision.window_kernel_size
        self._needs_bf16_cast = config.dtype == ir.DataType.BFLOAT16
        window_tokens = self.kernel_size[0] * self.kernel_size[1]
        window_hidden = self.hidden_size * window_tokens
        window_intermediate = vision.intermediate_size * window_tokens
        self.self_attn = VisionAttention(self.hidden_size, vision.num_attention_heads or 1)
        self.layer_norm1 = LayerNorm(self.hidden_size, eps=vision.norm_eps)
        self.pre_norm = LayerNorm(window_hidden, eps=vision.norm_eps)
        self.linear_1 = Linear(window_hidden, window_intermediate, bias=True)
        self.act = get_activation("gelu_pytorch_tanh")
        self.linear_2 = Linear(window_intermediate, self.hidden_size, bias=True)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        target_sizes: ir.Value,
    ):
        residual = hidden_states
        normalized = self.layer_norm1(op, hidden_states)
        grid, merged_h, merged_w = _spatial_windows(
            op, normalized, target_sizes, self.hidden_size, self.kernel_size
        )
        window_tokens = self.kernel_size[0] * self.kernel_size[1]
        windows = op.Reshape(
            grid,
            op.Constant(value_ints=[-1, window_tokens, self.hidden_size]),
        )
        windows = self.self_attn(op, windows)
        grid = op.Reshape(
            windows,
            op.Concat(
                op.Shape(target_sizes, start=0, end=1),
                merged_h,
                merged_w,
                op.Constant(
                    value_ints=[
                        self.kernel_size[0],
                        self.kernel_size[1],
                        self.hidden_size,
                    ]
                ),
                axis=0,
            ),
        )
        # Restore raster order before the attention residual.
        grid = op.Transpose(grid, perm=[0, 1, 3, 2, 4, 5])
        attended = op.Reshape(
            grid,
            op.Concat(
                op.Shape(target_sizes, start=0, end=1),
                op.Mul(
                    op.Mul(merged_h, merged_w),
                    op.Constant(value_ints=[window_tokens]),
                ),
                op.Constant(value_ints=[self.hidden_size]),
                axis=0,
            ),
        )
        hidden_states = op.Add(residual, attended)

        # Merge each 2x2 block: (B, H, W, D) -> (B*H/2*W/2, 4D).
        merged_grid, _, _ = _spatial_windows(
            op, hidden_states, target_sizes, self.hidden_size, self.kernel_size
        )
        if self._needs_bf16_cast:
            # ReduceMean does not have an ORT BF16 kernel on CPU or CUDA.
            merge_residual = op.CastLike(
                op.ReduceMean(
                    op.CastLike(merged_grid, 0.0),
                    [3, 4],
                    keepdims=0,
                ),
                merged_grid,
            )
        else:
            merge_residual = op.ReduceMean(merged_grid, [3, 4], keepdims=0)
        merge_residual = op.Reshape(
            merge_residual,
            op.Constant(value_ints=[-1, self.hidden_size]),
        )
        flat = op.Reshape(
            merged_grid,
            op.Constant(
                value_ints=[
                    -1,
                    window_tokens * self.hidden_size,
                ]
            ),
        )
        flat = self.pre_norm(op, flat)
        flat = self.linear_1(op, flat)
        flat = self.act(op, flat)
        flat = self.linear_2(op, flat)
        flat = op.Add(flat, merge_residual)
        return op.Reshape(
            flat,
            op.Concat(
                op.Shape(target_sizes, start=0, end=1),
                op.Mul(merged_h, merged_w),
                op.Constant(value_ints=[self.hidden_size]),
                axis=0,
            ),
        )


class _MiniCPMVisionTower(nn.Module):
    """Variable-resolution SigLIP2 tower with an inserted attention merger."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vision = config.vision
        assert vision is not None
        assert vision.image_size is not None
        assert vision.patch_size is not None
        assert vision.hidden_size is not None
        assert vision.intermediate_size is not None
        assert vision.num_hidden_layers is not None
        assert vision.num_attention_heads is not None
        self.hidden_size = vision.hidden_size
        self.insert_layer_id = (
            vision.insert_layer_id if vision.insert_layer_id is not None else 6
        )
        self.use_vit_merger = config.downsample_mode != "4x"
        self.embeddings = _MiniCPMVisionEmbeddings(
            vision.image_size,
            vision.patch_size,
            vision.hidden_size,
            vision.in_channels,
        )
        self.encoder = _MiniCPMVisionEncoder(config)
        self.post_layernorm = LayerNorm(vision.hidden_size, eps=vision.norm_eps)

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        target_sizes: ir.Value,
    ):
        hidden_states = self.embeddings(op, pixel_values, target_sizes)
        hidden_states = _unpack_padded_grid(
            op,
            hidden_states,
            target_sizes,
            self.hidden_size,
        )
        hidden_states = self.encoder(
            op,
            hidden_states,
            target_sizes=target_sizes,
            insert_layer_id=self.insert_layer_id if self.use_vit_merger else -1,
        )
        return self.post_layernorm(op, hidden_states)


class _MiniCPMVisionEncoder(nn.Module):
    """Container matching HuggingFace's ``vision_tower.encoder.layers`` names."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vision = config.vision
        assert vision is not None
        assert vision.hidden_size is not None
        assert vision.intermediate_size is not None
        assert vision.num_hidden_layers is not None
        assert vision.num_attention_heads is not None
        self.layers = nn.ModuleList(
            [
                VisionEncoderLayer(
                    vision.hidden_size,
                    vision.intermediate_size,
                    vision.num_attention_heads,
                    vision.norm_eps,
                )
                for _ in range(vision.num_hidden_layers)
            ]
        )
        # The upstream module registers vit_merger beside encoder. Keeping it
        # here lets the encoder call every layer through its own scope so layer
        # initializer names retain ``encoder.layers.N``; preprocessing adds
        # this one extra ``encoder.`` segment for the merger weights.
        self.vit_merger = _MiniCPMViTWindowAttentionMerger(config)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        target_sizes: ir.Value,
        insert_layer_id: int,
    ):
        current_sizes = op.Cast(target_sizes, to=7)
        attention_bias = _vision_attention_bias(
            op,
            hidden_states,
            current_sizes,
        )
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(op, hidden_states, attention_bias)
            if layer_idx == insert_layer_id:
                hidden_states = self.vit_merger(op, hidden_states, current_sizes)
                current_sizes = op.Div(current_sizes, 2)
                attention_bias = _vision_attention_bias(
                    op,
                    hidden_states,
                    current_sizes,
                )
        return hidden_states


class _MiniCPMDownsampleMLP(nn.Module):
    """One 2x2 spatial merge and projection, matching ``merger.mlp.N``."""

    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        merged_hidden_size = hidden_size * 4
        self.pre_norm = LayerNorm(merged_hidden_size, eps=1e-6)
        self.linear_1 = Linear(merged_hidden_size, merged_hidden_size, bias=True)
        self.act = get_activation("gelu")
        self.linear_2 = Linear(merged_hidden_size, output_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        hidden_states = self.pre_norm(op, hidden_states)
        hidden_states = self.linear_1(op, hidden_states)
        hidden_states = self.act(op, hidden_states)
        return self.linear_2(op, hidden_states)


class _MiniCPMMerger(nn.Module):
    """Final spatial merger from vision width into the text embedding width."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vision = config.vision
        assert vision is not None and vision.hidden_size is not None
        self.hidden_size = vision.hidden_size
        self.kernel_size = vision.merge_kernel_size
        self.merger_times = vision.merger_times
        self.vision_downsample = 1 if config.downsample_mode == "4x" else 2
        self._needs_bf16_cast = config.dtype == ir.DataType.BFLOAT16
        self.output_sizes = [
            self.hidden_size if i < self.merger_times - 1 else config.hidden_size
            for i in range(self.merger_times)
        ]
        mlps: list[nn.Module] = [
            _MiniCPMDownsampleMLP(
                self.hidden_size,
                self.output_sizes[i],
            )
            for i in range(self.merger_times)
        ]
        self.mlp = nn.ModuleList(mlps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        target_sizes: ir.Value,
    ):
        # Default 16x mode enters after the in-tower 2x2 merger. In 4x mode
        # the tower is unmerged, so this projector performs the only 2x2 merge.
        target_sizes = op.Div(
            op.Cast(target_sizes, to=7),
            self.vision_downsample,
        )
        current_hidden = self.hidden_size
        for index, mlp in enumerate(self.mlp):
            grid, merged_h, merged_w = _spatial_windows(
                op,
                hidden_states,
                target_sizes,
                current_hidden,
                self.kernel_size,
            )
            merged_dim = self.kernel_size[0] * self.kernel_size[1] * current_hidden
            hidden_states = op.Reshape(
                grid,
                op.Constant(value_ints=[-1, merged_dim]),
            )
            hidden_states = mlp(op, hidden_states)
            target_sizes = op.Concat(
                op.Div(
                    op.Slice(target_sizes, [0], [1], axes=[1]),
                    self.kernel_size[0],
                ),
                op.Div(
                    op.Slice(target_sizes, [1], [2], axes=[1]),
                    self.kernel_size[1],
                ),
                axis=1,
            )
            current_hidden = self.output_sizes[index]
            hidden_states = op.Reshape(
                hidden_states,
                op.Concat(
                    op.Shape(target_sizes, start=0, end=1),
                    op.Mul(merged_h, merged_w),
                    op.Shape(hidden_states, start=1, end=2),
                    axis=0,
                ),
            )
        max_height, max_width = _max_grid_size(op, target_sizes)
        valid = op.Reshape(
            _grid_mask(op, target_sizes, max_height, max_width),
            [-1],
        )
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                op.Constant(value_ints=[-1]),
                op.Shape(hidden_states, start=2, end=3),
                axis=0,
            ),
        )
        if self._needs_bf16_cast:
            # ONNX Compress excludes BF16 from its type constraint. Compaction
            # is data movement only, so cast through FLOAT and restore dtype.
            compacted = op.Compress(
                op.CastLike(hidden_states, 0.0),
                valid,
                axis=0,
            )
            return op.CastLike(compacted, hidden_states)
        return op.Compress(hidden_states, valid, axis=0)


class _MiniCPMVisionEncoderModel(nn.Module):
    """Packed SigLIP2 encoder plus both MiniCPM visual token mergers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.vision_tower = _MiniCPMVisionTower(config)
        self.merger = _MiniCPMMerger(config)

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        target_sizes: ir.Value,
    ):
        pixel_values = op.CastLike(
            pixel_values,
            self.vision_tower.embeddings.patch_embedding,
        )
        hidden_states = self.vision_tower(op, pixel_values, target_sizes)
        return self.merger(op, hidden_states, target_sizes)


class _MiniCPMDecoderModel(nn.Module):
    """Qwen3.5 hybrid decoder taking pre-fused ``inputs_embeds``."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.model = Qwen35TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
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
        return self.lm_head(op, hidden_states), present_key_values


class _MiniCPMEmbeddingModel(nn.Module):
    """Fuse packed image or video features into the token embedding sequence."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.image_token_id = config.image_token_id or 248056
        self.video_token_id = config.video_token_id or 248057

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
    ):
        text_embeds = self.embed_tokens(op, input_ids)
        media_mask = op.Or(
            op.Equal(input_ids, self.image_token_id),
            op.Equal(input_ids, self.video_token_id),
        )
        # HF masked_scatter consumes the packed feature stream in batch-major
        # order, so flatten before CumSum instead of restarting at each row.
        flat_media_mask = op.Reshape(media_mask, [-1])
        media_indices = op.Clip(
            op.Sub(
                op.CumSum(op.Cast(flat_media_mask, to=ir.DataType.INT64), 0),
                1,
            ),
            0,
        )
        # Keep text-only inference valid when the feature tensor has zero rows.
        padding = op.Expand(
            op.CastLike(0.0, image_features),
            op.Concat(
                op.Constant(value_ints=[1]),
                op.Shape(image_features, start=1, end=2),
                axis=0,
            ),
        )
        features = op.Gather(
            op.Concat(image_features, padding, axis=0),
            media_indices,
            axis=0,
        )
        features = op.Reshape(features, op.Shape(text_embeds))
        return op.Where(op.Unsqueeze(media_mask, [-1]), features, text_embeds)


class MiniCPMV46ForConditionalGeneration(nn.Module):
    """OpenBMB MiniCPM-V-4.6 image/video conditional generation model."""

    default_task: str = "minicpm-vl"
    category: str = "Multimodal"
    config_class: type = MiniCPMV46Config

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _MiniCPMDecoderModel(config)
        self.vision_encoder = _MiniCPMVisionEncoderModel(config)
        self.embedding = _MiniCPMEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "MiniCPMV46ForConditionalGeneration uses MiniCPMVLTask, which "
            "builds decoder, vision_encoder, and embedding graphs separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route the HuggingFace checkpoint into the three ONNX sub-models."""
        renamed: dict[str, torch.Tensor] = {}
        embed_weight: torch.Tensor | None = None
        for key, value in state_dict.items():
            if key.startswith(("mtp_", "mtp.")):
                continue
            if key.startswith("model.vision_tower."):
                name = "vision_encoder." + key[len("model.") :]
                name = name.replace(".mlp.fc1.", ".mlp.up_proj.")
                name = name.replace(".mlp.fc2.", ".mlp.down_proj.")
                name = name.replace(
                    ".vision_tower.vit_merger.",
                    ".vision_tower.encoder.vit_merger.",
                )
                renamed[name] = value
            elif key.startswith("model.merger."):
                renamed["vision_encoder." + key[len("model.") :]] = value
            elif key == "model.language_model.embed_tokens.weight":
                embed_weight = value
                renamed["decoder.model.embed_tokens.weight"] = value
                renamed["embedding.embed_tokens.weight"] = value
            elif key.startswith("model.language_model."):
                suffix = key[len("model.language_model.") :]
                renamed[f"decoder.model.{suffix}"] = value
            elif key == "lm_head.weight":
                renamed["decoder.lm_head.weight"] = value

        if self.config.tie_word_embeddings and embed_weight is not None:
            renamed["decoder.lm_head.weight"] = embed_weight
        return renamed
