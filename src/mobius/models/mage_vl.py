# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mage-VL streaming image/video-language model with a custom Mage-ViT encoder.

Replicates Hugging Face ``MageVLForConditionalGeneration`` as a standardized
decoder, vision-encoder, and embedding package. The vision tower consumes packed
Qwen2-VL patches, applies 3D 4:6:6 RoPE at explicit sampled-frame positions, and
limits bidirectional attention to independent four-frame windows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities, get_build_dtype
from mobius._configs import ArchitectureConfig
from mobius.components import (
    Conv2dNoBias,
    Embedding,
    LayerNorm,
    Linear,
    build_packed_token_offset,
)
from mobius.models.base import TextModel

if TYPE_CHECKING:
    from onnx_ir import Value


class _GELU(nn.Module):
    def forward(self, op: OpBuilder, hidden_states: Value):
        return op.Gelu(hidden_states)


class MageVLVisionRotaryEmbedding(nn.Module):
    """Construct Mage-VL's 4:6:6 temporal/height/width rotary frequencies."""

    def __init__(self, head_dim: int, rope_theta: float):
        super().__init__()
        if head_dim % 32 != 0:
            raise ValueError(
                f"Mage-VL vision head_dim must be divisible by 32, got {head_dim}"
            )
        half_dim = head_dim // 2
        unit = half_dim // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit
        self.rope_theta = rope_theta

    def _axis_freqs(self, op: OpBuilder, positions: Value, size: int):
        indices = op.Range(
            op.Constant(value_int=0),
            op.Constant(value_int=size),
            op.Constant(value_int=1),
        )
        indices = op.Cast(indices, to=ir.DataType.FLOAT)
        exponents = op.Div(indices, float(size))
        inv_freq = op.Reciprocal(
            op.Pow(op.Constant(value_float=float(self.rope_theta)), exponents)
        )
        positions = op.Cast(positions, to=ir.DataType.FLOAT)
        return op.Mul(op.Unsqueeze(positions, [-1]), op.Unsqueeze(inv_freq, [0]))

    def forward(self, op: OpBuilder, patch_positions: Value):
        # Explicit temporal positions preserve the original sampled video frame indices.
        t_pos = op.Gather(patch_positions, op.Constant(value_int=0), axis=1)
        h_pos = op.Gather(patch_positions, op.Constant(value_int=1), axis=1)
        w_pos = op.Gather(patch_positions, op.Constant(value_int=2), axis=1)
        half_freqs = op.Concat(
            self._axis_freqs(op, t_pos, self.t_size),
            self._axis_freqs(op, h_pos, self.h_size),
            self._axis_freqs(op, w_pos, self.w_size),
            axis=-1,
        )
        # The reference concatenates D/2 frequencies with themselves before
        # applying its interleaved rotate-half implementation.
        freqs = op.Concat(half_freqs, half_freqs, axis=-1)  # (patches, head_dim)
        freqs = op.Unsqueeze(freqs, [0])  # (1, patches, head_dim)
        return op.Cos(freqs), op.Sin(freqs)


class MageVLVisionAttention(nn.Module):
    """Fused-QKV bidirectional attention matching MageVLVisionAttention."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (hidden_size // num_heads) ** -0.5
        self.qkv = Linear(hidden_size, 3 * hidden_size)
        self.proj = Linear(hidden_size, hidden_size)

    def _apply_rope(self, op: OpBuilder, x: Value, cos: Value, sin: Value):
        # Mage-VL's reference duplicates the complete D/2 frequency vector,
        # then rotates adjacent even/odd channels. This differs from the ONNX
        # RotaryEmbedding interleaved contract, which shares one frequency per
        # pair, so reproduce the reference explicitly in FP32.
        x_shape = op.Shape(x)
        x_heads = op.Reshape(
            x,
            op.Concat(
                op.Shape(x, start=0, end=2),
                op.Constant(value_ints=[self.num_heads, -1]),
                axis=0,
            ),
        )
        x_float = op.Cast(x_heads, to=ir.DataType.FLOAT)
        even = op.Slice(x_float, starts=[0], ends=[9223372036854775807], axes=[3], steps=[2])
        odd = op.Slice(x_float, starts=[1], ends=[9223372036854775807], axes=[3], steps=[2])
        rotated = op.Reshape(
            op.Concat(op.Unsqueeze(op.Neg(odd), [-1]), op.Unsqueeze(even, [-1]), axis=-1),
            op.Shape(x_float),
        )
        cos = op.Unsqueeze(cos, [2])
        sin = op.Unsqueeze(sin, [2])
        embedded = op.Add(op.Mul(x_float, cos), op.Mul(rotated, sin))
        return op.Reshape(op.CastLike(embedded, x), x_shape)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: Value,
        attention_mask: Value | None,
        cu_seqlens: Value,
        cos: Value,
        sin: Value,
    ):
        q, k, v = op.Split(
            self.qkv(op, hidden_states),
            num_outputs=3,
            axis=-1,
            _outputs=3,
        )
        q = self._apply_rope(op, q, cos, sin)
        k = self._apply_rope(op, k, cos, sin)
        if ep_capabilities().supports_packed_multi_head_attention:
            query = op.Squeeze(q, [0])
            key = op.Squeeze(k, [0])
            value = op.Squeeze(v, [0])
            token_offset = build_packed_token_offset(op, cu_seqlens)
            if get_build_dtype() == ir.DataType.BFLOAT16:
                query_mha = op.Cast(query, to=ir.DataType.FLOAT16)
                key_mha = op.Cast(key, to=ir.DataType.FLOAT16)
                value_mha = op.Cast(value, to=ir.DataType.FLOAT16)
            else:
                query_mha, key_mha, value_mha = query, key, value
            hidden_states = op.PackedMultiHeadAttention(
                query_mha,
                key_mha,
                value_mha,
                None,
                token_offset,
                op.Cast(cu_seqlens, to=ir.DataType.INT32),
                num_heads=self.num_heads,
                scale=self.scale,
                _domain="com.microsoft",
                _outputs=1,
            )
            hidden_states = op.Unsqueeze(op.CastLike(hidden_states, query), [0])
        else:
            hidden_states = op.Attention(
                q,
                k,
                v,
                attention_mask,
                q_num_heads=self.num_heads,
                kv_num_heads=self.num_heads,
                scale=self.scale,
                is_causal=0,
                _outputs=1,
            )
        return self.proj(op, hidden_states)


class MageVLVisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = Linear(hidden_size, intermediate_size)
        self.fc2 = Linear(intermediate_size, hidden_size)

    def forward(self, op: OpBuilder, hidden_states: Value):
        return self.fc2(op, op.Gelu(self.fc1(op, hidden_states)))


class MageVLVisionEncoderLayer(nn.Module):
    """Pre-norm Mage-ViT block with fused-QKV attention and GELU MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
    ):
        super().__init__()
        self.self_attn = MageVLVisionAttention(hidden_size, num_heads)
        self.layer_norm1 = LayerNorm(hidden_size, eps=norm_eps)
        self.mlp = MageVLVisionMLP(hidden_size, intermediate_size)
        self.layer_norm2 = LayerNorm(hidden_size, eps=norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: Value,
        attention_mask: Value | None,
        cu_seqlens: Value,
        cos: Value,
        sin: Value,
    ):
        residual = hidden_states
        hidden_states = self.self_attn(
            op,
            self.layer_norm1(op, hidden_states),
            attention_mask,
            cu_seqlens,
            cos,
            sin,
        )
        hidden_states = op.Add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(op, self.layer_norm2(op, hidden_states))
        return op.Add(residual, hidden_states)


class MageVLVisionEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MageVLVisionEncoderLayer(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: Value,
        attention_mask: Value | None,
        cu_seqlens: Value,
        cos: Value,
        sin: Value,
    ):
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, attention_mask, cu_seqlens, cos, sin)
        return hidden_states


class MageVLVisionEmbeddings(nn.Module):
    """Bias-free Conv2d projection for pre-extracted Qwen2-VL patches."""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.patch_embedding = Conv2dNoBias(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, op: OpBuilder, hidden_states: Value):
        patches = op.Reshape(
            hidden_states,
            op.Constant(value_ints=[-1, self.in_channels, self.patch_size, self.patch_size]),
        )
        hidden_states = self.patch_embedding(op, patches)  # (patches, hidden, 1, 1)
        return op.Reshape(
            hidden_states,
            op.Constant(value_ints=[1, -1, self.hidden_size]),
        )  # (1, patches, hidden)


class MageVLVisionPatchMerger(nn.Module):
    """Merge each contiguous 2x2 packed patch group into one text-width token."""

    def __init__(
        self,
        output_size: int,
        context_size: int,
        spatial_merge_size: int,
        norm_eps: float,
    ):
        super().__init__()
        merged_size = context_size * spatial_merge_size**2
        self.ln_q = LayerNorm(context_size, eps=norm_eps)
        self.mlp = nn.Sequential(
            Linear(merged_size, merged_size),
            _GELU(),
            Linear(merged_size, output_size),
        )
        self.merged_size = merged_size

    def forward(self, op: OpBuilder, hidden_states: Value):
        hidden_states = self.ln_q(op, hidden_states)
        hidden_states = op.Reshape(
            hidden_states,
            op.Constant(value_ints=[-1, self.merged_size]),
        )
        return self.mlp(op, hidden_states)


class MageVLVisionPretrainedModel(nn.Module):
    """Custom Mage-ViT backbone used by microsoft/Mage-VL."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        if vc is None or vc.hidden_size is None:
            raise ValueError("Mage-VL requires a complete vision configuration")
        if vc.num_hidden_layers is None or vc.num_attention_heads is None:
            raise ValueError("Mage-VL vision layer and head counts are required")
        hidden_size = vc.hidden_size
        num_heads = vc.num_attention_heads
        norm_eps = vc.norm_eps
        self.frame_windows_size = vc.frame_windows_size
        self.embeddings = MageVLVisionEmbeddings(
            vc.in_channels,
            hidden_size,
            vc.patch_size or 16,
        )
        self.layernorm_pre = LayerNorm(hidden_size, eps=norm_eps)
        self.encoder = MageVLVisionEncoder(
            vc.num_hidden_layers,
            hidden_size,
            vc.intermediate_size or 4 * hidden_size,
            num_heads,
            norm_eps,
        )
        self.video_rope = MageVLVisionRotaryEmbedding(
            hidden_size // num_heads,
            vc.rope_theta or 10_000.0,
        )
        self.merger = MageVLVisionPatchMerger(
            vc.out_hidden_size or config.hidden_size,
            hidden_size,
            vc.spatial_merge_size,
            norm_eps,
        )

    def _attention_metadata(self, op: OpBuilder, grid_thw: Value, hidden_states: Value):
        # Build per-window lengths directly from (T, H, W). The processor emits
        # one row per image/video frame, while custom callers may pack T>1 into a
        # row; padding to max_windows keeps this O(num_visuals * max_windows)
        # instead of materializing an O(total_patches * num_visuals) table.
        temporal = op.Gather(grid_thw, op.Constant(value_int=0), axis=1)
        spatial = op.Mul(
            op.Gather(grid_thw, op.Constant(value_int=1), axis=1),
            op.Gather(grid_thw, op.Constant(value_int=2), axis=1),
        )
        window = op.Constant(value_int=self.frame_windows_size)
        window_counts = op.Div(op.Add(temporal, self.frame_windows_size - 1), window)
        max_windows = op.Squeeze(op.ReduceMax(window_counts), [0])
        window_ids = op.Range(
            op.Constant(value_int=0),
            max_windows,
            op.Constant(value_int=1),
        )
        valid_windows = op.Less(
            op.Unsqueeze(window_ids, [0]),
            op.Unsqueeze(window_counts, [1]),
        )
        remaining_frames = op.Sub(
            op.Unsqueeze(temporal, [1]),
            op.Mul(op.Unsqueeze(window_ids, [0]), window),
        )
        window_frames = op.Min(remaining_frames, window)
        window_lengths = op.Mul(window_frames, op.Unsqueeze(spatial, [1]))
        lengths = op.Compress(
            op.Reshape(window_lengths, op.Constant(value_ints=[-1])),
            op.Reshape(valid_windows, op.Constant(value_ints=[-1])),
        )
        cu_seqlens = op.Concat(
            op.Constant(value_ints=[0]),
            op.CumSum(lengths, op.Constant(value_int=0)),
            axis=0,
        )
        if ep_capabilities().supports_packed_multi_head_attention:
            return None, cu_seqlens

        # Portable ONNX Attention needs an explicit block-diagonal mask.
        total = op.Shape(hidden_states, start=1, end=2)
        patch_ids = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(total, [0]),
            op.Constant(value_int=1),
        )
        segment_ends = op.Slice(
            cu_seqlens,
            starts=[1],
            ends=[9223372036854775807],
            axes=[0],
        )
        segment_ids = op.ReduceSum(
            op.Cast(
                op.GreaterOrEqual(
                    op.Unsqueeze(patch_ids, [1]),
                    op.Unsqueeze(segment_ends, [0]),
                ),
                to=ir.DataType.INT64,
            ),
            axes=[1],
            keepdims=0,
        )
        attention_mask = op.Equal(
            op.Unsqueeze(segment_ids, [1]),
            op.Unsqueeze(segment_ids, [0]),
        )
        return op.Unsqueeze(attention_mask, [0, 1]), cu_seqlens

    def forward(
        self,
        op: OpBuilder,
        hidden_state: Value,
        grid_thw: Value,
        patch_positions: Value,
    ):
        if get_build_dtype() != ir.DataType.FLOAT:
            hidden_state = op.Cast(hidden_state, to=get_build_dtype())
        hidden_states = self.embeddings(op, hidden_state)
        cos, sin = self.video_rope(op, patch_positions)
        attention_mask, cu_seqlens = self._attention_metadata(op, grid_thw, hidden_states)
        hidden_states = self.layernorm_pre(op, hidden_states)
        hidden_states = self.encoder(
            op,
            hidden_states,
            attention_mask,
            cu_seqlens,
            cos,
            sin,
        )
        return self.merger(op, hidden_states)


class _MageVLModelBody(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.language_model = TextModel(config)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: Value,
        attention_mask: Value,
        position_ids: Value,
        past_key_values: list | None,
    ):
        return self.language_model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )


class MageVLDecoderModel(nn.Module):
    """Qwen3 decoder with the checkpoint's ``model.language_model`` hierarchy."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.model = _MageVLModelBody(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: Value,
        attention_mask: Value,
        position_ids: Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            inputs_embeds,
            attention_mask,
            position_ids,
            past_key_values,
        )
        return self.lm_head(op, hidden_states), present_key_values


class _MageVLVisualBody(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.visual = MageVLVisionPretrainedModel(config)


class MageVLVisionEncoderModel(nn.Module):
    """Standalone Mage-ViT encoder for image and sampled-video patches."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.model = _MageVLVisualBody(config)

    def forward(
        self,
        op: OpBuilder,
        pixel_values: Value,
        image_grid_thw: Value,
        patch_positions: Value,
    ):
        return self.model.visual(op, pixel_values, image_grid_thw, patch_positions)


class MageVLEmbeddingModel(nn.Module):
    """Scatter packed image/video features at Mage-VL visual placeholder tokens."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        if config.image_token_id is None:
            raise ValueError("Mage-VL requires image_token_id")
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id

    def forward(self, op: OpBuilder, input_ids: Value, image_features: Value):
        text_embeddings = self.embed_tokens(op, input_ids)
        visual_mask = op.Equal(input_ids, op.Constant(value_int=self.image_token_id))
        if self.video_token_id is not None:
            visual_mask = op.Or(
                visual_mask,
                op.Equal(input_ids, op.Constant(value_int=self.video_token_id)),
            )
        flat_visual_mask = op.Reshape(
            visual_mask,
            op.Constant(value_ints=[-1]),
        )
        flat_indices = op.CumSum(
            op.Cast(flat_visual_mask, to=ir.DataType.INT64),
            op.Constant(value_int=0),
        )
        indices = op.Reshape(flat_indices, op.Shape(input_ids))
        flat_text = op.Reshape(
            text_embeddings,
            op.Constant(value_ints=[-1, self.hidden_size]),
        )
        zero_feature = op.Mul(
            op.Slice(flat_text, starts=[0], ends=[1], axes=[0]),
            0.0,
        )
        padded_features = op.Concat(zero_feature, image_features, axis=0)
        visual_embeddings = op.Gather(padded_features, indices, axis=0)
        return op.Where(
            op.Unsqueeze(visual_mask, [-1]),
            visual_embeddings,
            text_embeddings,
        )


class MageVLForConditionalGeneration(nn.Module):
    """Mage-VL streaming image/video-language model with a Qwen3 decoder."""

    default_task: str = "mage-vl"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = MageVLDecoderModel(config)
        self.vision_encoder = MageVLVisionEncoderModel(config)
        self.embedding = MageVLEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError("MageVLTask exports each pipeline component separately")

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Route every Hugging Face checkpoint tensor to its package component."""
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("model.language_model."):
                if key.startswith("model.language_model.embed_tokens."):
                    suffix = key[len("model.language_model.") :]
                    renamed[f"embedding.{suffix}"] = value
                else:
                    renamed[f"decoder.{key}"] = value
            elif key.startswith("model.visual."):
                renamed[f"vision_encoder.{key[len('model.') :]}"] = value
            elif key.startswith("lm_head."):
                renamed[f"decoder.{key}"] = value
        return renamed
