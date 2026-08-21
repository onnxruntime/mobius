# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LiquidAI LFM2-VL vision-language model — 3-model split.

Replicates ``Lfm2VlForConditionalGeneration`` from Transformers.  Builds
three ONNX models:

- **decoder**: LFM2 hybrid short-conv / GQA decoder taking ``inputs_embeds``
- **vision_encoder**: SigLIP2 NaFlex tower + pixel-unshuffle MLP projector
- **embedding**: token embedding + image-feature fusion

Pipeline for ``LiquidAI/LFM2.5-VL-3B``::

    pixel_values (N, max_patches, 3*16*16)   [NaFlex, per-image patch grid]
        -> SigLIP2 NaFlex tower              -> (N, max_patches, 1152)
        -> unpad + pixel unshuffle (f=2)     -> (T, 4608)
        -> Linear(4608 -> 2048) -> GELU
           -> Linear(2048 -> 2048)           -> (T, 2048)
        -> scattered onto ``image_token_id`` positions of inputs_embeds
        -> LFM2 decoder (30 layers, 2048-dim, GQA 32/8, conv+full_attention)

HuggingFace weight prefixes -> ONNX initializer names::

    model.vision_tower.[vision_model.]*
        -> vision_encoder.vision_tower.*     (mlp.fc1/fc2 -> up_proj/down_proj)
    model.multi_modal_projector.linear_{1,2}.*
        -> vision_encoder.multi_modal_projector.linear_{1,2}.*
    model.language_model.embed_tokens.weight
        -> decoder.model.embed_tokens.weight
        -> embedding.embed_tokens.weight     (duplicated)
        -> decoder.lm_head.weight            (tie_word_embeddings)
    model.language_model.*
        -> decoder.model.*                   (LFM2 projection renames)

The checkpoint has no ``lm_head.weight``; it is tied to
``model.language_model.embed_tokens.weight``.

Reference: ``LiquidAI/LFM2.5-VL-3B``.
"""

from __future__ import annotations

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, Lfm2VlConfig
from mobius.components import (
    Embedding,
    LayerNorm,
    Linear,
    Siglip2NaFlexVisionModel,
    get_activation,
)
from mobius.models.lfm2 import (
    Lfm2TextModel,
    apply_lfm2_config_defaults,
    rename_lfm2_weight_key,
)

# HuggingFace ``Lfm2VlConfig.image_token_id`` for LFM2.5-VL.
_IMAGE_TOKEN_ID = 124907


class Lfm2VlMultiModalProjector(nn.Module):
    """Pixel-unshuffle spatial merge followed by a two-layer MLP.

    Upstream operates one image at a time on a ``(1, h, w, C)`` feature map.
    Here every image is processed in one shot: the padded tower output is
    addressed with explicit gather indices derived from ``spatial_shapes``,
    which both unpads and applies the unshuffle in a single ``Gather``.

    Deriving the channel order from upstream's reshape/permute chain, with
    ``f = downsample_factor``::

        out[a, b, (s*f + r)*C + c] = in[f*a + s, f*b + r, c]

    i.e. the ``f x f`` sub-patch block is flattened row-major over
    (row offset, column offset) and each sub-patch contributes a contiguous
    ``C``-wide slice.

    The index arithmetic replaces a ``Scan``-plus-``Compress`` round trip: the
    enumeration of merged tokens is already the compacted, image-ordered
    output stream, so the padding never enters the projection at all.
    """

    def __init__(self, config: Lfm2VlConfig):
        super().__init__()
        assert config.vision is not None and config.vision.hidden_size is not None
        self.factor = config.downsample_factor
        self.vision_hidden_size = config.vision.hidden_size
        merged_size = self.vision_hidden_size * self.factor**2
        self.use_layer_norm = config.projector_use_layernorm
        if self.use_layer_norm:
            # torch nn.LayerNorm default eps; LFM2.5-VL ships it disabled.
            self.layer_norm = LayerNorm(merged_size, eps=1e-5)
        self.linear_1 = Linear(
            merged_size,
            config.projector_hidden_size,
            bias=config.projector_bias,
        )
        self.act = get_activation(config.projector_hidden_act)
        self.linear_2 = Linear(
            config.projector_hidden_size,
            config.hidden_size,
            bias=config.projector_bias,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        spatial_shapes: ir.Value,
    ) -> ir.Value:
        # (N, max_patches, C) -> (total_merged_tokens, C * f^2)
        merged = self._pixel_unshuffle(op, hidden_states, spatial_shapes)
        if self.use_layer_norm:
            merged = self.layer_norm(op, merged)
        merged = self.linear_1(op, merged)
        merged = self.act(op, merged)
        return self.linear_2(op, merged)

    def _pixel_unshuffle(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        spatial_shapes: ir.Value,
    ) -> ir.Value:
        factor = self.factor
        shapes = op.Cast(spatial_shapes, to=ir.DataType.INT64)
        heights = op.Squeeze(op.Slice(shapes, [0], [1], axes=[1]), [1])  # (N,)
        widths = op.Squeeze(op.Slice(shapes, [1], [2], axes=[1]), [1])  # (N,)
        merged_heights = op.Div(heights, factor)
        merged_widths = op.Div(widths, factor)

        # Flat enumeration of every merged token across every image, in image
        # order — the concatenation order upstream builds with a Python loop.
        tokens_per_image = op.Mul(merged_heights, merged_widths)  # (N,)
        ends = op.CumSum(tokens_per_image, 0)
        starts = op.Sub(ends, tokens_per_image)
        total_tokens = op.ReduceSum(tokens_per_image, keepdims=0)
        token_index = op.Range(0, total_tokens, 1)  # (T,)

        # The first cumulative end past each token identifies its source image.
        reached_end = op.GreaterOrEqual(
            op.Unsqueeze(token_index, [1]),
            op.Unsqueeze(ends, [0]),
        )
        image_index = op.ReduceSum(
            op.Cast(reached_end, to=ir.DataType.INT64),
            [1],
            keepdims=0,
        )  # (T,)
        local_index = op.Sub(token_index, op.Gather(starts, image_index))
        merged_width = op.Gather(merged_widths, image_index)
        merged_row = op.Div(local_index, merged_width)  # a
        merged_column = op.Mod(local_index, merged_width)  # b

        # Top-left source patch of each f x f block, flattened over the padded
        # (N, max_patches) grid: valid patches occupy the first h*w slots
        # row-major, so patch (row, col) of image i lives at
        # i * max_patches + row * w + col.
        image_width = op.Gather(widths, image_index)  # (T,)
        max_patches = op.Shape(hidden_states, start=1, end=2)  # (1,)
        base_index = op.Add(
            op.Mul(image_index, max_patches),
            op.Add(
                op.Mul(op.Mul(merged_row, factor), image_width),
                op.Mul(merged_column, factor),
            ),
        )  # (T,)

        # Offsets of the f x f sub-patches, row-major over (row, column).
        block = op.Range(0, factor, 1)  # (f,)
        row_offsets = op.Unsqueeze(block, [0, 2])  # (1, f, 1)
        column_offsets = op.Unsqueeze(block, [0, 1])  # (1, 1, f)
        offsets = op.Reshape(
            op.Add(
                op.Mul(row_offsets, op.Unsqueeze(image_width, [1, 2])),
                column_offsets,
            ),
            op.Constant(value_ints=[-1, factor * factor]),
        )  # (T, f*f)
        indices = op.Add(op.Unsqueeze(base_index, [1]), offsets)  # (T, f*f)

        flat_states = op.Reshape(
            hidden_states,
            op.Concat(
                op.Constant(value_ints=[-1]),
                op.Shape(hidden_states, start=2, end=3),
                axis=0,
            ),
        )  # (N * max_patches, C)
        gathered = op.Gather(flat_states, indices, axis=0)  # (T, f*f, C)
        return op.Reshape(
            gathered,
            op.Constant(value_ints=[-1, factor * factor * self.vision_hidden_size]),
        )


class _Lfm2VlVisionEncoderModel(nn.Module):
    """SigLIP2 NaFlex tower plus the LFM2-VL multimodal projector."""

    def __init__(self, config: Lfm2VlConfig):
        super().__init__()
        assert config.vision is not None
        self.vision_tower = Siglip2NaFlexVisionModel(config.vision)
        self.multi_modal_projector = Lfm2VlMultiModalProjector(config)

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        pixel_attention_mask: ir.Value,
        spatial_shapes: ir.Value,
    ) -> ir.Value:
        hidden_states = self.vision_tower(
            op,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )  # (N, max_patches, vision_hidden)
        return self.multi_modal_projector(op, hidden_states, spatial_shapes)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Extract the vision tower and projector weights."""
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            name = _rename_vision_key(key)
            if name is not None:
                renamed[name] = value
        return renamed


class _Lfm2VlDecoderModel(nn.Module):
    """LFM2 hybrid decoder taking pre-fused ``inputs_embeds``."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.model = Lfm2TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            # Share one ONNX initializer with the embedding table.
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list[tuple[ir.Value, ...]] | None = None,
    ) -> tuple[ir.Value, list[tuple[ir.Value, ...]]]:
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        return self.lm_head(op, hidden_states), present_key_values


class _Lfm2VlEmbeddingModel(nn.Module):
    """Fuse projected image features into the token embedding sequence."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.image_token_id = (
            config.image_token_id if config.image_token_id is not None else _IMAGE_TOKEN_ID
        )

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
    ) -> ir.Value:
        text_embeds = self.embed_tokens(op, input_ids)  # (B, T) -> (B, T, H)
        image_mask = op.Equal(input_ids, self.image_token_id)

        # HF masked_scatter consumes the flat feature stream in batch-major
        # order, so flatten before CumSum instead of restarting at each row.
        flat_mask = op.Reshape(image_mask, [-1])
        feature_indices = op.Clip(
            op.Sub(op.CumSum(op.Cast(flat_mask, to=ir.DataType.INT64), 0), 1),
            0,
        )
        # Cached decode steps pass a zero-row feature tensor; the sentinel row
        # keeps the Gather in range and is discarded by the Where below.
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
            feature_indices,
            axis=0,
        )
        features = op.Reshape(features, op.Shape(text_embeds))
        return op.Where(op.Unsqueeze(image_mask, [-1]), features, text_embeds)


def _rename_vision_key(key: str) -> str | None:
    """Map a checkpoint key to its ``vision_encoder.`` initializer name."""
    vision_tower_prefix = "model.vision_tower."
    projector_prefix = "model.multi_modal_projector."

    if key.startswith(vision_tower_prefix):
        suffix = key[len(vision_tower_prefix) :]
        # Checkpoints saved before the Transformers v5 flattening nest the
        # tower under an extra ``vision_model.`` scope.
        if suffix.startswith("vision_model."):
            suffix = suffix[len("vision_model.") :]
        # Shared VisionEncoderLayer names its FC MLP up_proj/down_proj.
        suffix = suffix.replace(".mlp.fc1.", ".mlp.up_proj.")
        suffix = suffix.replace(".mlp.fc2.", ".mlp.down_proj.")
        return f"vision_encoder.vision_tower.{suffix}"
    if key.startswith(projector_prefix):
        return f"vision_encoder.multi_modal_projector.{key[len(projector_prefix) :]}"
    return None


class Lfm2VlForConditionalGeneration(nn.Module):
    """LiquidAI LFM2-VL image-conditioned generation model (3-model split)."""

    default_task: str = "lfm2-vl"
    category: str = "Multimodal"
    config_class: type = Lfm2VlConfig

    def __init__(self, config: ArchitectureConfig):
        assert isinstance(config, Lfm2VlConfig), (
            "Lfm2VlForConditionalGeneration requires an Lfm2VlConfig "
            f"(projector fields), got {type(config).__name__}"
        )
        # The decoder is a plain LFM2 backbone, so it needs the same
        # construction-time config normalisation as the text-only model.
        config = apply_lfm2_config_defaults(config)
        super().__init__()
        self.config = config
        self.decoder = _Lfm2VlDecoderModel(config)
        self.vision_encoder = _Lfm2VlVisionEncoderModel(config)
        self.embedding = _Lfm2VlEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "Lfm2VlForConditionalGeneration uses Lfm2VlTask, which builds "
            "decoder, vision_encoder, and embedding graphs separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route the HuggingFace checkpoint into the three ONNX sub-models."""
        renamed: dict[str, torch.Tensor] = {}
        embed_weight: torch.Tensor | None = None
        text_prefix = "model.language_model."

        for key, value in state_dict.items():
            vision_name = _rename_vision_key(key)
            if vision_name is not None:
                # Vision weights keep the SigLIP2 names; only the LFM2 decoder
                # gets the projection renames (both use ``self_attn.out_proj``).
                renamed[vision_name] = value
            elif key == f"{text_prefix}embed_tokens.weight":
                embed_weight = value
                # The decoder consumes inputs_embeds, so its own table stays
                # unused; emitting it keeps the routing uniform and costs
                # nothing (ModelPackage skips names with no initializer).
                renamed["decoder.model.embed_tokens.weight"] = value
                renamed["embedding.embed_tokens.weight"] = value
            elif key.startswith(text_prefix):
                suffix = key[len(text_prefix) :]
                renamed[f"decoder.model.{rename_lfm2_weight_key(suffix)}"] = value
            elif key == "lm_head.weight":
                renamed["decoder.lm_head.weight"] = value

        # LFM2.5-VL ships no lm_head; it is tied to the embedding table.
        if self.config.tie_word_embeddings and embed_weight is not None:
            renamed["decoder.lm_head.weight"] = embed_weight

        return renamed
