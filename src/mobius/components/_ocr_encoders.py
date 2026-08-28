# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Packed vision and audio encoders used by OCR GGUF sidecars."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities
from mobius._configs import ArchitectureConfig
from mobius.components._common import LayerNorm, Linear
from mobius.components._conv import Conv2d
from mobius.components._mlp import FCMLP, GatedMLP
from mobius.components._ocr_projectors import (
    Dots3NoteAudioProjector,
    DotsOCRProjector,
    LightOnOCRProjector,
    PaddleOCRProjector,
    YouTuVLProjector,
)
from mobius.components._pixtral_vision import PixtralVisionTower
from mobius.components._qwen25_vl_vision import (
    Qwen25VLVisionAttention,
    Qwen25VLVisionModel,
    Qwen25VLVisionRotaryEmbedding,
)
from mobius.components._rms_norm import RMSNorm
from mobius.components._sam_vision import SAMVisionEncoder

if TYPE_CHECKING:
    from collections.abc import Sequence


class PackedVisionPatchEmbed(nn.Module):
    """Apply a 2-D patch kernel to processor-packed patch rows."""

    def __init__(
        self,
        hidden_size: int,
        *,
        in_channels: int,
        patch_size: int,
        bias: bool,
    ):
        super().__init__()
        self.weight = nn.Parameter([hidden_size, in_channels, patch_size, patch_size])
        self.bias = nn.Parameter([hidden_size]) if bias else None
        self._hidden_size = hidden_size
        self._pixel_size = in_channels * patch_size * patch_size

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        pixel_values = op.CastLike(pixel_values, self.weight)
        patches = op.Reshape(pixel_values, [-1, self._pixel_size])
        weight = op.Reshape(self.weight, [self._hidden_size, -1])
        hidden_states = op.MatMul(patches, op.Transpose(weight, perm=[1, 0]))
        if self.bias is not None:
            hidden_states = op.Add(hidden_states, self.bias)
        return hidden_states


class PackedLinearPatchEmbed(nn.Module):
    """Apply a serialized linear patch projection to packed patch rows."""

    def __init__(self, pixel_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter([hidden_size, pixel_size])
        self.bias = nn.Parameter([hidden_size])
        self._pixel_size = pixel_size

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        pixel_values = op.CastLike(pixel_values, self.weight)
        pixel_values = op.Reshape(pixel_values, [-1, self._pixel_size])
        return op.Add(
            op.MatMul(pixel_values, op.Transpose(self.weight, perm=[1, 0])),
            self.bias,
        )


class FusedQKVVisionAttention(Qwen25VLVisionAttention):
    """Packed fused-QKV vision attention with optional per-head Q/K RMSNorm."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        norm_eps: float,
        qk_norm: bool,
    ):
        super().__init__(hidden_size, num_heads)
        self.qkv = Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
        cos: ir.Value,
        sin: ir.Value,
    ) -> ir.Value:
        seq_len = op.Shape(hidden_states, start=0, end=1)
        qkv = self.qkv(op, hidden_states)
        qkv = op.Reshape(
            qkv,
            op.Concat(seq_len, [3, self.num_heads, self.head_dim], axis=0),
        )
        qkv = op.Transpose(qkv, perm=[1, 0, 2, 3])
        query, key, value = (
            op.Squeeze(part, [0]) for part in op.Split(qkv, [1, 1, 1], axis=0, _outputs=3)
        )
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(op, query)
            key = self.k_norm(op, key)
        query = self._apply_rotary(op, query, cos, sin)
        key = self._apply_rotary(op, key, cos, sin)
        if ep_capabilities().supports_packed_multi_head_attention:
            output = self._emit_packed_mha(
                op,
                query,
                key,
                value,
                cu_seqlens,
                seq_len,
            )
        else:
            output = self._emit_standard_attention(
                op,
                query,
                key,
                value,
                cu_seqlens,
                seq_len,
            )
        return self.proj(op, output)


class SigmoidTopKVisionMoE(nn.Module):
    """Explicit-float sigmoid top-k MoE with selection-only correction bias."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
    ):
        super().__init__()
        if not 0 < top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        self.ffn_gate_inp = nn.Parameter([num_experts, hidden_size])
        self.ffn_gate_exps = nn.Parameter([num_experts, intermediate_size, hidden_size])
        self.ffn_up_exps = nn.Parameter([num_experts, intermediate_size, hidden_size])
        self.ffn_down_exps = nn.Parameter([num_experts, hidden_size, intermediate_size])
        self.exp_probs_b = nn.Parameter([num_experts])
        self._num_experts = num_experts
        self._top_k = top_k

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # The router and correction-biased selection run in float32.
        router_input = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        router = op.MatMul(
            router_input,
            op.Transpose(op.Cast(self.ffn_gate_inp, to=ir.DataType.FLOAT), perm=[1, 0]),
        )
        probabilities = op.Sigmoid(router)
        selection = op.Add(
            probabilities,
            op.Cast(self.exp_probs_b, to=ir.DataType.FLOAT),
        )
        _, expert_indices = op.TopK(
            selection,
            op.Constant(value_ints=[self._top_k]),
            axis=-1,
            largest=1,
            sorted=0,
            _outputs=2,
        )
        selected_weights = op.GatherElements(probabilities, expert_indices, axis=1)
        selected_weights = op.Div(
            selected_weights,
            op.ReduceSum(selected_weights, [1], keepdims=True),
        )

        # Evaluate every explicit-float expert, then mask to the selected top-k.
        hidden_by_expert = op.Unsqueeze(hidden_states, [0])
        gate = op.MatMul(
            hidden_by_expert,
            op.Transpose(self.ffn_gate_exps, perm=[0, 2, 1]),
        )
        up = op.MatMul(
            hidden_by_expert,
            op.Transpose(self.ffn_up_exps, perm=[0, 2, 1]),
        )
        activated = op.Mul(op.Mul(gate, op.Sigmoid(gate)), up)
        expert_output = op.MatMul(
            activated,
            op.Transpose(self.ffn_down_exps, perm=[0, 2, 1]),
        )
        expert_output = op.Transpose(expert_output, perm=[1, 0, 2])  # (N, E, H)

        one_hot = op.OneHot(
            expert_indices,
            op.Constant(value_int=self._num_experts),
            op.Constant(value_floats=[0.0, 1.0]),
        )
        routing = op.ReduceSum(
            op.Mul(one_hot, op.Unsqueeze(selected_weights, [-1])),
            [1],
            keepdims=False,
        )
        routing = op.CastLike(op.Unsqueeze(routing, [-1]), expert_output)
        return op.ReduceSum(op.Mul(expert_output, routing), [1], keepdims=False)


class DotsVisionBlock(nn.Module):
    """Dots pre-norm vision block with dense or progressive-MoE FFN."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        *,
        norm_eps: float,
        qk_norm: bool,
        num_experts: int = 0,
        expert_intermediate_size: int | None = None,
        top_k: int = 0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = FusedQKVVisionAttention(
            hidden_size,
            num_heads,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
        )
        if num_experts:
            if expert_intermediate_size is None:
                raise ValueError("MoE vision blocks require expert_intermediate_size")
            self.mlp: nn.Module = SigmoidTopKVisionMoE(
                hidden_size,
                expert_intermediate_size,
                num_experts,
                min(top_k, num_experts),
            )
        else:
            self.mlp = GatedMLP(
                hidden_size,
                intermediate_size,
                activation="silu",
                bias=False,
            )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
        cos: ir.Value,
        sin: ir.Value,
    ) -> ir.Value:
        residual = hidden_states
        hidden_states = self.attn(
            op,
            self.norm1(op, hidden_states),
            cu_seqlens,
            cos,
            sin,
        )
        hidden_states = op.Add(residual, hidden_states)
        return op.Add(hidden_states, self.mlp(op, self.norm2(op, hidden_states)))


class DotsVisionEncoder(Qwen25VLVisionModel):
    """Packed Dots vision tower with optional progressive-MoE blocks."""

    def __init__(
        self,
        *,
        depth: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        patch_size: int,
        output_size: int,
        projector_intermediate_size: int,
        spatial_merge_size: int,
        norm_eps: float,
        qk_norm: bool,
        expert_counts: Sequence[int] | None = None,
        expert_intermediate_size: int | None = None,
        top_k: int = 0,
    ):
        nn.Module.__init__(self)
        counts = tuple(expert_counts or (0,) * depth)
        if len(counts) != depth:
            raise ValueError("expert_counts must have one entry per vision block")
        self._spatial_merge_size = spatial_merge_size
        self._spatial_merge_unit = spatial_merge_size * spatial_merge_size
        self.input_schema = (
            ("pixel_values", ir.DataType.FLOAT, ("total_patches", 3 * patch_size**2)),
            ("image_grid_thw", ir.DataType.INT64, ("num_images", 3)),
        )
        self.patch_embed = PackedVisionPatchEmbed(  # type: ignore[assignment]
            hidden_size,
            in_channels=3,
            patch_size=patch_size,
            bias=True,
        )
        self.pre_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        self.rotary_pos_emb = Qwen25VLVisionRotaryEmbedding(
            hidden_size // num_heads // 2,
        )
        self.blocks = nn.ModuleList(
            [
                DotsVisionBlock(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    num_experts=num_experts,
                    expert_intermediate_size=expert_intermediate_size,
                    top_k=top_k,
                )
                for num_experts in counts
            ]
        )
        self.post_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        self.projector = DotsOCRProjector(
            hidden_size,
            projector_intermediate_size,
            output_size,
            merge_size=spatial_merge_size,
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ) -> ir.Value:
        hidden_states = self.pre_layernorm(op, self.patch_embed(op, pixel_values))
        rotary_pos_ids = self._compute_rotary_pos_ids(op, image_grid_thw)
        cu_seqlens = self._compute_cu_seqlens(op, image_grid_thw)
        cos, sin = self.rotary_pos_emb(op, rotary_pos_ids)
        for block in self.blocks:
            hidden_states = block(op, hidden_states, cu_seqlens, cos, sin)
        hidden_states = self.post_layernorm(op, hidden_states)
        return self.projector(op, hidden_states)


class SplitQKVVisionAttention(Qwen25VLVisionAttention):
    """Bidirectional split-QKV vision attention over one packed image."""

    def __init__(self, hidden_size: int, num_heads: int):
        nn.Module.__init__(self)
        self.q_proj = Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = Linear(hidden_size, hidden_size, bias=True)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
        cos: ir.Value,
        sin: ir.Value,
    ) -> ir.Value:
        seq_len = op.Shape(hidden_states, start=0, end=1)

        def heads(value: ir.Value) -> ir.Value:
            return op.Reshape(
                value,
                op.Concat(seq_len, [self.num_heads, self.head_dim], axis=0),
            )

        query = self._apply_rotary(op, heads(self.q_proj(op, hidden_states)), cos, sin)
        key = self._apply_rotary(op, heads(self.k_proj(op, hidden_states)), cos, sin)
        value = heads(self.v_proj(op, hidden_states))
        if ep_capabilities().supports_packed_multi_head_attention:
            output = self._emit_packed_mha(
                op,
                query,
                key,
                value,
                cu_seqlens,
                seq_len,
            )
        else:
            output = self._emit_standard_attention(
                op,
                query,
                key,
                value,
                cu_seqlens,
                seq_len,
            )
        return self.out_proj(op, output)


class SplitQKVVisionBlock(nn.Module):
    """LayerNorm split-QKV vision block with a biased GELU FFN."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        *,
        norm_eps: float,
    ):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps=norm_eps)
        self.norm2 = LayerNorm(hidden_size, eps=norm_eps)
        self.attn = SplitQKVVisionAttention(hidden_size, num_heads)
        self.mlp = FCMLP(
            hidden_size,
            intermediate_size,
            activation="gelu_pytorch_tanh",
            bias=True,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
        cos: ir.Value,
        sin: ir.Value,
    ) -> ir.Value:
        residual = hidden_states
        hidden_states = self.attn(
            op,
            self.norm1(op, hidden_states),
            cu_seqlens,
            cos,
            sin,
        )
        hidden_states = op.Add(residual, hidden_states)
        return op.Add(hidden_states, self.mlp(op, self.norm2(op, hidden_states)))


def _raster_rotary_pos_ids(
    op: OpBuilder,
    grid_h: ir.Value,
    grid_w: ir.Value,
) -> ir.Value:
    h_range = op.Range(op.Constant(value_int=0), grid_h, op.Constant(value_int=1))
    w_range = op.Range(op.Constant(value_int=0), grid_w, op.Constant(value_int=1))
    h_grid = op.Expand(
        op.Unsqueeze(h_range, [1]),
        op.Concat(op.Reshape(grid_h, [1]), op.Reshape(grid_w, [1]), axis=0),
    )
    w_grid = op.Expand(
        op.Unsqueeze(w_range, [0]),
        op.Concat(op.Reshape(grid_h, [1]), op.Reshape(grid_w, [1]), axis=0),
    )
    return op.Concat(
        op.Reshape(h_grid, [-1, 1]),
        op.Reshape(w_grid, [-1, 1]),
        axis=1,
    )


class PaddleOCRVisionEncoder(Qwen25VLVisionModel):
    """PaddleOCR dynamic packed-patch ViT and MLP-AR projector."""

    def __init__(
        self,
        *,
        depth: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        patch_size: int,
        position_size: int,
        output_size: int,
        projector_intermediate_size: int,
        spatial_merge_size: int = 2,
        norm_eps: float = 1e-6,
    ):
        nn.Module.__init__(self)
        source_side = math.isqrt(position_size)
        if source_side * source_side != position_size:
            raise ValueError("PaddleOCR learned position table must be square")
        self._source_side = source_side
        self._hidden_size = hidden_size
        self.input_schema = (
            (
                "pixel_values",
                ir.DataType.FLOAT,
                ("total_patches", 3, patch_size, patch_size),
            ),
            ("image_grid_thw", ir.DataType.INT64, (1, 3)),
        )
        self.patch_embed = PackedVisionPatchEmbed(  # type: ignore[assignment]
            hidden_size,
            in_channels=3,
            patch_size=patch_size,
            bias=True,
        )
        self.position_embedding = nn.Parameter([position_size, hidden_size])
        self.rotary_pos_emb = Qwen25VLVisionRotaryEmbedding(
            hidden_size // num_heads // 2,
        )
        self.blocks = nn.ModuleList(
            [
                SplitQKVVisionBlock(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    norm_eps=norm_eps,
                )
                for _ in range(depth)
            ]
        )
        self.post_layernorm = LayerNorm(hidden_size, eps=norm_eps)
        self.projector = PaddleOCRProjector(
            hidden_size,
            projector_intermediate_size,
            output_size,
            merge_size=spatial_merge_size,
        )

    def _resized_position_embedding(
        self,
        op: OpBuilder,
        grid_h: ir.Value,
        grid_w: ir.Value,
    ) -> ir.Value:
        table = op.Reshape(
            op.Transpose(self.position_embedding, perm=[1, 0]),
            [1, self._hidden_size, self._source_side, self._source_side],
        )
        sizes = op.Concat(
            [1, self._hidden_size],
            op.Reshape(grid_h, [1]),
            op.Reshape(grid_w, [1]),
            axis=0,
        )
        table = op.Resize(
            table,
            None,
            None,
            sizes,
            mode="linear",
            coordinate_transformation_mode="half_pixel",
            antialias=1,
            exclude_outside=1,
        )
        return op.Transpose(op.Reshape(table, [self._hidden_size, -1]), perm=[1, 0])

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ) -> ir.Value:
        grid_h = op.Squeeze(op.Gather(image_grid_thw, [1], axis=1))
        grid_w = op.Squeeze(op.Gather(image_grid_thw, [2], axis=1))
        hidden_states = self.patch_embed(op, pixel_values)
        hidden_states = op.Add(
            hidden_states,
            op.CastLike(self._resized_position_embedding(op, grid_h, grid_w), hidden_states),
        )
        rotary_pos_ids = _raster_rotary_pos_ids(op, grid_h, grid_w)
        cos, sin = self.rotary_pos_emb(op, rotary_pos_ids)
        cu_seqlens = op.Concat(
            op.Constant(value_ints=[0]),
            op.Reshape(op.Mul(grid_h, grid_w), [1]),
            axis=0,
        )
        for block in self.blocks:
            hidden_states = block(op, hidden_states, cu_seqlens, cos, sin)
        hidden_states = self.post_layernorm(op, hidden_states)
        return self.projector(op, hidden_states, grid_h, grid_w)


class YouTuVLVisionEncoder(Qwen25VLVisionModel):
    """YouTu-VL packed SigLIP2 tower with irregular window-attention layers."""

    def __init__(
        self,
        *,
        depth: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        pixel_size: int,
        patch_size: int,
        output_size: int,
        projector_intermediate_size: int,
        spatial_merge_size: int,
        window_size: int,
        full_attention_layers: Sequence[int],
        norm_eps: float,
    ):
        nn.Module.__init__(self)
        self._fullatt_block_indexes = set(full_attention_layers)
        self._spatial_merge_size = spatial_merge_size
        self._spatial_merge_unit = spatial_merge_size * spatial_merge_size
        self._patch_size = patch_size
        self._hidden_size = hidden_size
        self._vit_merger_window_size = window_size // spatial_merge_size // patch_size
        self.input_schema = (
            ("pixel_values", ir.DataType.FLOAT, (1, "total_patches", pixel_size)),
            ("spatial_shapes", ir.DataType.INT64, (1, 2)),
        )
        self.patch_embed = PackedLinearPatchEmbed(  # type: ignore[assignment]
            pixel_size,
            hidden_size,
        )
        self.rotary_pos_emb = Qwen25VLVisionRotaryEmbedding(
            hidden_size // num_heads // 2,
        )
        self.blocks = nn.ModuleList(
            [
                SplitQKVVisionBlock(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    norm_eps=norm_eps,
                )
                for _ in range(depth)
            ]
        )
        self.post_layernorm = LayerNorm(hidden_size, eps=norm_eps)
        self.projector = YouTuVLProjector(
            hidden_size,
            projector_intermediate_size,
            output_size,
            merge_size=spatial_merge_size,
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        spatial_shapes: ir.Value,
    ) -> ir.Value:
        image_grid_thw = op.Concat(
            op.Reshape(op.Constant(value_int=1), [1, 1]),
            spatial_shapes,
            axis=1,
        )
        merge_unit = self._spatial_merge_unit
        hidden_states = self.patch_embed(op, pixel_values)
        rotary_pos_ids = self._compute_rotary_pos_ids(op, image_grid_thw)
        window_index, cu_window_seqlens = self._compute_window_index(op, image_grid_thw)
        cu_seqlens = self._compute_cu_seqlens(op, image_grid_thw)

        seq_len = op.Shape(hidden_states, start=0, end=1)
        merged_len = op.Div(seq_len, op.Constant(value_ints=[merge_unit]))
        hidden_dim = op.Shape(hidden_states, start=1, end=2)
        grouped_shape = op.Concat(merged_len, [merge_unit], hidden_dim, axis=0)
        hidden_states = op.Reshape(hidden_states, grouped_shape)
        hidden_states = op.Gather(hidden_states, window_index, axis=0)
        hidden_states = op.Reshape(hidden_states, [-1, self._hidden_size])
        rotary_pos_ids = op.Reshape(rotary_pos_ids, [-1, merge_unit, 2])
        rotary_pos_ids = op.Reshape(
            op.Gather(rotary_pos_ids, window_index, axis=0),
            [-1, 2],
        )
        cos, sin = self.rotary_pos_emb(op, rotary_pos_ids)

        for layer_index, block in enumerate(self.blocks):
            block_cu = (
                cu_seqlens if layer_index in self._fullatt_block_indexes else cu_window_seqlens
            )
            hidden_states = block(op, hidden_states, block_cu, cos, sin)
        hidden_states = self.post_layernorm(op, hidden_states)
        merged = self.projector(op, hidden_states)

        k = op.Shape(window_index, start=0, end=1)
        _, reverse_index = op.TopK(
            op.Cast(window_index, to=ir.DataType.FLOAT),
            k,
            largest=0,
            sorted=1,
            _outputs=2,
        )
        return op.Gather(merged, reverse_index, axis=0)


class LightOnOCRVisionEncoder(nn.Module):
    """Dynamic Pixtral tower and LightOnOCR spatial projector."""

    def __init__(
        self,
        config: ArchitectureConfig,
        *,
        first_bias: bool = False,
        second_bias: bool = False,
    ):
        super().__init__()
        vision = config.vision
        if vision is None or vision.hidden_size is None or vision.spatial_merge_size is None:
            raise ValueError("LightOnOCR requires complete vision dimensions")
        self.vision_tower = PixtralVisionTower(config)
        self.projector = LightOnOCRProjector(
            int(vision.hidden_size),
            int(config.hidden_size),
            merge_size=int(vision.spatial_merge_size),
            eps=vision.norm_eps,
            first_bias=first_bias,
            second_bias=second_bias,
        )
        self.input_schema = (("pixel_values", ir.DataType.FLOAT, (1, 3, "height", "width")),)

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        pixel_values = op.CastLike(
            pixel_values,
            self.vision_tower.patch_conv.weight,
        )
        hidden_states, grid_h, grid_w = self.vision_tower(op, pixel_values)
        return self.projector(op, hidden_states, grid_h, grid_w)


class _Granite4Attention(nn.Module):
    """Biased dense attention used by the Granite tower and QFormer."""

    def __init__(self, hidden_size: int, num_heads: int, *, kv_size: int | None = None):
        super().__init__()
        kv_size = hidden_size if kv_size is None else kv_size
        self.q_proj = Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = Linear(kv_size, hidden_size, bias=True)
        self.v_proj = Linear(kv_size, hidden_size, bias=True)
        self.out_proj = Linear(hidden_size, hidden_size, bias=True)
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        key_value_states: ir.Value | None = None,
    ) -> ir.Value:
        source = hidden_states if key_value_states is None else key_value_states
        output = op.Attention(
            self.q_proj(op, hidden_states),
            self.k_proj(op, source),
            self.v_proj(op, source),
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            scale=float(self._head_dim**-0.5),
            is_causal=0,
        )
        return self.out_proj(op, output)


class _Granite4VisionBlock(nn.Module):
    """LayerNorm transformer block for the Granite SigLIP tower."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
    ):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps=norm_eps)
        self.norm2 = LayerNorm(hidden_size, eps=norm_eps)
        self.attn = _Granite4Attention(hidden_size, num_heads)
        self.mlp = FCMLP(
            hidden_size,
            intermediate_size,
            activation="gelu_pytorch_tanh",
            bias=True,
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = op.Add(
            hidden_states,
            self.attn(op, self.norm1(op, hidden_states)),
        )
        return op.Add(hidden_states, self.mlp(op, self.norm2(op, hidden_states)))


class _Granite4QFormerLayer(nn.Module):
    """One post-norm self/cross/FFN QFormer layer."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        if hidden_size % 64:
            raise ValueError("Granite4 QFormer hidden size must be divisible by 64")
        heads = hidden_size // 64
        self.self_attn = _Granite4Attention(hidden_size, heads)
        self.self_attn_norm = LayerNorm(hidden_size, eps=1e-12)
        self.cross_attn = _Granite4Attention(hidden_size, heads)
        self.cross_attn_norm = LayerNorm(hidden_size, eps=1e-12)
        self.ffn_up = Linear(hidden_size, intermediate_size, bias=True)
        self.ffn_down = Linear(intermediate_size, hidden_size, bias=True)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-12)

    def forward(
        self,
        op: OpBuilder,
        query_states: ir.Value,
        encoder_states: ir.Value,
    ) -> ir.Value:
        query_states = self.self_attn_norm(
            op,
            op.Add(query_states, self.self_attn(op, query_states)),
        )
        query_states = self.cross_attn_norm(
            op,
            op.Add(
                query_states,
                self.cross_attn(op, query_states, encoder_states),
            ),
        )
        projected = self.ffn_down(op, op.Gelu(self.ffn_up(op, query_states)))
        return self.ffn_norm(op, op.Add(query_states, projected))


class Granite4WindowQFormerProjector(nn.Module):
    """Window-QFormer stream over one selected Granite vision layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        *,
        image_side: int,
        window_side: int,
        query_side: int,
        spatial_offset: int,
        norm_eps: float,
    ):
        super().__init__()
        if image_side % window_side:
            raise ValueError("Granite4 image_side must be divisible by window_side")
        if spatial_offset not in {-1, 0, 1, 2, 3}:
            raise ValueError("Granite4 spatial_offset must be -1 or one of four 2x2 offsets")
        windows_per_side = image_side // window_side
        downsampled_side = windows_per_side * query_side
        if downsampled_side * 2 != image_side:
            raise ValueError("Granite4 pinned projector requires 2x spatial downsampling")
        self.norm = LayerNorm(hidden_size, eps=norm_eps)
        self.query = nn.Parameter([1, query_side * query_side, hidden_size])
        self.image_positions = nn.Parameter([1, window_side * window_side, hidden_size])
        self.post_norm = LayerNorm(hidden_size, eps=1e-12)
        self.qformer = _Granite4QFormerLayer(hidden_size, intermediate_size)
        self.linear = Linear(hidden_size, output_size, bias=True)
        self._hidden_size = hidden_size
        self._image_side = image_side
        self._window_side = window_side
        self._query_side = query_side
        self._spatial_offset = spatial_offset
        self._windows_per_side = windows_per_side
        self._downsampled_side = downsampled_side

    def _window(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        *,
        side: int,
        window: int,
    ) -> ir.Value:
        batch = op.Shape(hidden_states, start=0, end=1)
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(batch, [side, side, self._hidden_size], axis=0),
        )
        count = side // window
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                batch,
                [count, window, count, window, self._hidden_size],
                axis=0,
            ),
        )
        hidden_states = op.Transpose(hidden_states, perm=[0, 1, 3, 2, 4, 5])
        return op.Reshape(hidden_states, [-1, window * window, self._hidden_size])

    def _downsample(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        batch = op.Shape(hidden_states, start=0, end=1)
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                batch,
                [self._image_side, self._image_side, self._hidden_size],
                axis=0,
            ),
        )
        if self._spatial_offset < 0:
            hidden_states = op.Transpose(hidden_states, perm=[0, 3, 1, 2])
            hidden_states = op.AveragePool(
                hidden_states,
                kernel_shape=[2, 2],
                strides=[2, 2],
            )
            return op.Transpose(hidden_states, perm=[0, 2, 3, 1])
        offset_y = (self._spatial_offset >> 1) & 1
        offset_x = self._spatial_offset & 1
        return op.Slice(
            hidden_states,
            [offset_y, offset_x],
            [self._image_side, self._image_side],
            [1, 2],
            [2, 2],
        )

    def _unwindow(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        windows = self._windows_per_side
        query = self._query_side
        batch = op.Div(
            op.Shape(hidden_states, start=0, end=1),
            op.Constant(value_ints=[windows * windows]),
        )
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                batch,
                [windows, windows, query, query, self._hidden_size],
                axis=0,
            ),
        )
        hidden_states = op.Transpose(hidden_states, perm=[0, 1, 3, 2, 4, 5])
        return op.Reshape(
            hidden_states,
            op.Concat(
                batch,
                [self._downsampled_side**2, self._hidden_size],
                axis=0,
            ),
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = self.norm(op, hidden_states)
        encoder_states = self._window(
            op,
            hidden_states,
            side=self._image_side,
            window=self._window_side,
        )
        encoder_states = op.Add(encoder_states, self.image_positions)

        downsampled = self._downsample(op, hidden_states)
        downsampled = op.Reshape(
            downsampled,
            [-1, self._downsampled_side**2, self._hidden_size],
        )
        query_states = self._window(
            op,
            downsampled,
            side=self._downsampled_side,
            window=self._query_side,
        )
        query_states = op.Add(query_states, self.query)
        query_states = self.post_norm(op, query_states)
        query_states = self.qformer(op, query_states, encoder_states)
        query_states = self._unwindow(op, query_states)
        return self.linear(op, query_states)


class Granite4VisionEncoder(nn.Module):
    """Granite4 SigLIP tower and concatenated Window-QFormer streams."""

    def __init__(
        self,
        *,
        depth: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        image_size: int,
        patch_size: int,
        feature_layers: Sequence[int],
        spatial_offsets: Sequence[int],
        query_side: int,
        window_side: int,
        output_size: int,
        qformer_intermediate_size: int,
        norm_eps: float,
    ):
        super().__init__()
        if len(feature_layers) != len(spatial_offsets) or not feature_layers:
            raise ValueError("Granite4 feature layers and spatial offsets must be nonempty")
        if any(not 0 <= layer < depth for layer in feature_layers):
            raise ValueError("Granite4 feature layer is outside the vision tower")
        image_side = image_size // patch_size
        self.patch_embedding = Conv2d(
            3,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embedding = nn.Parameter([image_side * image_side, hidden_size])
        self.blocks = nn.ModuleList(
            [
                _Granite4VisionBlock(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    norm_eps,
                )
                for _ in range(depth)
            ]
        )
        self.projectors = nn.ModuleList(
            [
                Granite4WindowQFormerProjector(
                    hidden_size,
                    qformer_intermediate_size,
                    output_size,
                    image_side=image_side,
                    window_side=window_side,
                    query_side=query_side,
                    spatial_offset=offset,
                    norm_eps=norm_eps,
                )
                for offset in spatial_offsets
            ]
        )
        self.image_newline = nn.Parameter([output_size])
        self._feature_layers = tuple(feature_layers)
        self._hidden_size = hidden_size
        self._projector_count = len(feature_layers)
        self._output_size = output_size * len(feature_layers)
        self._output_side = image_side // window_side * query_side
        self.input_schema = (
            ("pixel_values", ir.DataType.FLOAT, (1, "num_tiles", 3, image_size, image_size)),
            ("image_sizes", ir.DataType.INT64, (1, 2)),
            ("tile_grid", ir.DataType.INT64, (2,)),
        )

    def _newline_row(self, op: OpBuilder) -> ir.Value:
        return op.Reshape(
            op.Tile(
                self.image_newline,
                op.Constant(value_ints=[self._projector_count]),
            ),
            [1, self._output_size],
        )

    def _assemble_tiles(
        self,
        op: OpBuilder,
        features: ir.Value,
        image_sizes: ir.Value,
        tile_grid: ir.Value,
    ) -> ir.Value:
        grid_h = op.Squeeze(op.Gather(tile_grid, [0]))
        grid_w = op.Squeeze(op.Gather(tile_grid, [1]))
        tiles = op.Slice(features, [1], [2**31 - 1], [0])
        tiles = op.Reshape(
            tiles,
            op.Concat(
                op.Reshape(grid_h, [1]),
                op.Reshape(grid_w, [1]),
                [self._output_side, self._output_side, self._output_size],
                axis=0,
            ),
        )
        tiles = op.Transpose(tiles, perm=[0, 2, 1, 3, 4])
        current_h = op.Mul(grid_h, op.Constant(value_int=self._output_side))
        current_w = op.Mul(grid_w, op.Constant(value_int=self._output_side))
        tiles = op.Reshape(
            tiles,
            op.Concat(
                op.Reshape(current_h, [1]),
                op.Reshape(current_w, [1]),
                [self._output_size],
                axis=0,
            ),
        )

        original_h = op.Squeeze(op.Gather(image_sizes, [0], axis=1))
        original_w = op.Squeeze(op.Gather(image_sizes, [1], axis=1))
        width_limited = op.Greater(
            op.Mul(original_w, current_h),
            op.Mul(original_h, current_w),
        )
        resized_h = op.Div(op.Mul(original_h, current_w), original_w)
        resized_w = op.Div(op.Mul(original_w, current_h), original_h)
        pad_h = op.Div(op.Sub(current_h, resized_h), op.Constant(value_int=2))
        pad_w = op.Div(op.Sub(current_w, resized_w), op.Constant(value_int=2))
        retained_h = op.Sub(current_h, op.Mul(pad_h, op.Constant(value_int=2)))
        retained_w = op.Sub(current_w, op.Mul(pad_w, op.Constant(value_int=2)))
        rows = op.Range(op.Constant(value_int=0), current_h, op.Constant(value_int=1))
        columns = op.Range(op.Constant(value_int=0), current_w, op.Constant(value_int=1))
        valid_rows = op.Or(
            op.Not(width_limited),
            op.And(
                op.GreaterOrEqual(rows, pad_h),
                op.Less(rows, op.Add(pad_h, retained_h)),
            ),
        )
        valid_columns = op.Or(
            width_limited,
            op.And(
                op.GreaterOrEqual(columns, pad_w),
                op.Less(columns, op.Add(pad_w, retained_w)),
            ),
        )
        tiles = op.Compress(tiles, valid_rows, axis=0)
        tiles = op.Compress(tiles, valid_columns, axis=1)
        unpadded_h = op.Squeeze(op.Shape(tiles, start=0, end=1))
        newline = op.Expand(
            op.Unsqueeze(self._newline_row(op), [0]),
            op.Concat(
                op.Reshape(unpadded_h, [1]),
                [1, self._output_size],
                axis=0,
            ),
        )
        return op.Reshape(op.Concat(tiles, newline, axis=1), [-1, self._output_size])

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_sizes: ir.Value,
        tile_grid: ir.Value,
    ) -> ir.Value:
        # Processor supplies one image's [overview, row-major tiles] stack.
        pixel_values = op.Reshape(
            pixel_values,
            op.Concat(
                op.Constant(value_ints=[-1]),
                op.Shape(pixel_values, start=2, end=5),
                axis=0,
            ),
        )
        pixel_values = op.CastLike(pixel_values, self.patch_embedding.weight)
        hidden_states = self.patch_embedding(op, pixel_values)
        batch = op.Shape(hidden_states, start=0, end=1)
        hidden_states = op.Transpose(
            op.Reshape(
                hidden_states,
                op.Concat(batch, [self._hidden_size, -1], axis=0),
            ),
            perm=[0, 2, 1],
        )
        hidden_states = op.Add(hidden_states, self.position_embedding)
        layer_outputs: list[ir.Value] = []
        for block in self.blocks:
            hidden_states = block(op, hidden_states)
            layer_outputs.append(hidden_states)
        streams = [
            projector(op, layer_outputs[layer_index])
            for projector, layer_index in zip(self.projectors, self._feature_layers)
        ]
        features = op.Concat(*streams, axis=-1)
        overview = op.Squeeze(op.Gather(features, [0], axis=0), [0])
        tiled = self._assemble_tiles(op, features, image_sizes, tile_grid)
        return op.Concat(overview, tiled, axis=0)


class _DeepSeekCLIPAttention(nn.Module):
    """Fused-QKV CLIP attention used after the DeepSeek SAM tower."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.qkv = Linear(hidden_size, hidden_size * 3, bias=True)
        self.out_proj = Linear(hidden_size, hidden_size, bias=True)
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        batch = op.Shape(hidden_states, start=0, end=1)
        seq = op.Shape(hidden_states, start=1, end=2)
        qkv = self.qkv(op, hidden_states)
        qkv = op.Reshape(
            qkv,
            op.Concat(batch, seq, [3, self._num_heads, self._head_dim], axis=0),
        )
        qkv = op.Transpose(qkv, perm=[2, 0, 3, 1, 4])
        query, key, value = (
            op.Squeeze(part, [0]) for part in op.Split(qkv, [1, 1, 1], axis=0, _outputs=3)
        )
        output = op.Attention(
            query,
            key,
            value,
            scale=float(self._head_dim**-0.5),
            is_causal=0,
        )
        output = op.Transpose(output, perm=[0, 2, 1, 3])
        output = op.Reshape(
            output,
            op.Concat(batch, seq, [self._num_heads * self._head_dim], axis=0),
        )
        return self.out_proj(op, output)


class _DeepSeekCLIPBlock(nn.Module):
    """Pre-norm CLIP block with QuickGELU."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
    ):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps=norm_eps)
        self.norm2 = LayerNorm(hidden_size, eps=norm_eps)
        self.attn = _DeepSeekCLIPAttention(hidden_size, num_heads)
        self.mlp = FCMLP(
            hidden_size,
            intermediate_size,
            activation="quick_gelu",
            bias=True,
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = op.Add(
            hidden_states,
            self.attn(op, self.norm1(op, hidden_states)),
        )
        return op.Add(hidden_states, self.mlp(op, self.norm2(op, hidden_states)))


class DeepSeekOCRCLIPEncoder(nn.Module):
    """CLIP transformer over SAM feature patches for DeepSeek-OCR v1."""

    def __init__(
        self,
        *,
        depth: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        position_size: int,
        norm_eps: float,
    ):
        super().__init__()
        source_side = math.isqrt(position_size - 1)
        if source_side * source_side + 1 != position_size:
            raise ValueError("DeepSeek CLIP position table must be square plus one row")
        self.class_embedding = nn.Parameter([hidden_size])
        self.position_embedding = nn.Parameter([position_size, hidden_size])
        self.pre_layernorm = LayerNorm(hidden_size, eps=norm_eps)
        self.blocks = nn.ModuleList(
            [
                _DeepSeekCLIPBlock(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    norm_eps,
                )
                for _ in range(depth)
            ]
        )
        self._hidden_size = hidden_size
        self._source_side = source_side

    def _positions(self, op: OpBuilder, target_side: ir.Value) -> ir.Value:
        patch_positions = op.Slice(
            self.position_embedding,
            [0],
            [self._source_side * self._source_side],
            [0],
        )
        patch_positions = op.Transpose(patch_positions, perm=[1, 0])
        patch_positions = op.Reshape(
            patch_positions,
            [1, self._hidden_size, self._source_side, self._source_side],
        )
        sizes = op.Concat(
            [1, self._hidden_size],
            op.Reshape(target_side, [1]),
            op.Reshape(target_side, [1]),
            axis=0,
        )
        patch_positions = op.Resize(
            patch_positions,
            None,
            None,
            sizes,
            mode="cubic",
            coordinate_transformation_mode="half_pixel",
        )
        patch_positions = op.Transpose(
            op.Reshape(patch_positions, [self._hidden_size, -1]),
            perm=[1, 0],
        )
        class_position = op.Slice(
            self.position_embedding,
            [self._source_side * self._source_side],
            [self._source_side * self._source_side + 1],
            [0],
        )
        return op.Concat(patch_positions, class_position, axis=0)

    def forward(self, op: OpBuilder, sam_features: ir.Value) -> ir.Value:
        batch = op.Shape(sam_features, start=0, end=1)
        side = op.Squeeze(op.Shape(sam_features, start=2, end=3))
        hidden_states = op.Transpose(sam_features, perm=[0, 2, 3, 1])
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(batch, [-1, self._hidden_size], axis=0),
        )
        class_token = op.Expand(
            op.Reshape(self.class_embedding, [1, 1, self._hidden_size]),
            op.Concat(batch, [1, self._hidden_size], axis=0),
        )
        hidden_states = op.Concat(class_token, hidden_states, axis=1)
        hidden_states = op.Add(
            hidden_states,
            op.CastLike(op.Unsqueeze(self._positions(op, side), [0]), hidden_states),
        )
        hidden_states = self.pre_layernorm(op, hidden_states)
        for block in self.blocks:
            hidden_states = block(op, hidden_states)
        return op.Slice(hidden_states, [1], [2**31 - 1], [1])


class DeepSeekOCRVisionEncoder(nn.Module):
    """DeepSeek-OCR v1 SAM + CLIP fusion for one square processor view."""

    def __init__(
        self,
        *,
        image_size: int,
        sam_hidden_size: int,
        sam_num_heads: int,
        sam_depth: int,
        sam_window_size: int,
        clip_hidden_size: int,
        clip_intermediate_size: int,
        clip_num_heads: int,
        clip_depth: int,
        output_size: int,
    ):
        super().__init__()
        self.sam = SAMVisionEncoder(
            img_size=image_size,
            patch_size=16,
            embed_dim=sam_hidden_size,
            depth=sam_depth,
            num_heads=sam_num_heads,
            window_size=sam_window_size,
            downsample_channels=(512, clip_hidden_size),
            position_size=64,
            rel_pos_size=64,
            mlp_activation="gelu_pytorch_tanh",
        )
        self.clip = DeepSeekOCRCLIPEncoder(
            depth=clip_depth,
            hidden_size=clip_hidden_size,
            intermediate_size=clip_intermediate_size,
            num_heads=clip_num_heads,
            position_size=257,
            norm_eps=1e-5,
        )
        from mobius.components._ocr_projectors import DeepSeekOCRProjector

        self.projector = DeepSeekOCRProjector(
            clip_hidden_size,
            clip_hidden_size,
            output_size,
        )
        self._clip_hidden_size = clip_hidden_size
        self.input_schema = (
            ("pixel_values", ir.DataType.FLOAT, (1, 3, image_size, image_size)),
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        pixel_values = op.CastLike(pixel_values, self.sam.patch_embed.weight)
        sam_map = self.sam(op, pixel_values)
        clip_features = self.clip(op, sam_map)
        sam_features = op.Transpose(sam_map, perm=[0, 2, 3, 1])
        batch = op.Shape(sam_features, start=0, end=1)
        sam_features = op.Reshape(
            sam_features,
            op.Concat(batch, [-1, self._clip_hidden_size], axis=0),
        )
        return self.projector(op, clip_features, sam_features)


class DeepSeekOCRFullImageEncoder(nn.Module):
    """DeepSeek-OCR v1 local-tile mosaic followed by the global overview."""

    def __init__(
        self,
        *,
        sam_hidden_size: int,
        sam_num_heads: int,
        sam_depth: int,
        sam_window_size: int,
        clip_hidden_size: int,
        clip_intermediate_size: int,
        clip_num_heads: int,
        clip_depth: int,
        output_size: int,
    ):
        super().__init__()
        common = {
            "sam_hidden_size": sam_hidden_size,
            "sam_num_heads": sam_num_heads,
            "sam_depth": sam_depth,
            "sam_window_size": sam_window_size,
            "clip_hidden_size": clip_hidden_size,
            "clip_intermediate_size": clip_intermediate_size,
            "clip_num_heads": clip_num_heads,
            "clip_depth": clip_depth,
            "output_size": output_size,
        }
        self.global_encoder = DeepSeekOCRVisionEncoder(image_size=1024, **common)
        self.local_encoder = DeepSeekOCRVisionEncoder(image_size=640, **common)
        self.image_newline = nn.Parameter([output_size])
        self.view_separator = nn.Parameter([output_size])
        self._output_size = output_size
        self._local_side = 10
        self._global_side = 16
        self.input_schema = (
            ("global_pixel_values", ir.DataType.FLOAT, (1, 3, 1024, 1024)),
            (
                "local_pixel_values",
                ir.DataType.FLOAT,
                ("num_tile_rows", 3, 640, "local_row_width"),
            ),
        )

    def _append_newlines(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        height: ir.Value,
        width: ir.Value,
    ) -> ir.Value:
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                op.Reshape(height, [1]),
                op.Reshape(width, [1]),
                [self._output_size],
                axis=0,
            ),
        )
        newline = op.Expand(
            op.Reshape(self.image_newline, [1, 1, self._output_size]),
            op.Concat(op.Reshape(height, [1]), [1, self._output_size], axis=0),
        )
        return op.Reshape(op.Concat(hidden_states, newline, axis=1), [-1, self._output_size])

    def forward(
        self,
        op: OpBuilder,
        global_pixel_values: ir.Value,
        local_pixel_values: ir.Value,
    ) -> ir.Value:
        grid_h = op.Squeeze(op.Shape(local_pixel_values, start=0, end=1))
        packed_width = op.Squeeze(op.Shape(local_pixel_values, start=3, end=4))
        grid_w = op.Div(packed_width, op.Constant(value_int=640))
        local_tiles = op.Reshape(
            local_pixel_values,
            op.Concat(
                op.Reshape(grid_h, [1]),
                [3, 640],
                op.Reshape(grid_w, [1]),
                [640],
                axis=0,
            ),
        )
        local_tiles = op.Transpose(local_tiles, perm=[0, 3, 1, 2, 4])
        local_tiles = op.Reshape(local_tiles, [-1, 3, 640, 640])
        local = self.local_encoder(op, local_tiles)
        local = op.Reshape(
            local,
            op.Concat(
                op.Reshape(grid_h, [1]),
                op.Reshape(grid_w, [1]),
                [self._local_side, self._local_side, self._output_size],
                axis=0,
            ),
        )
        # Tile-major -> one raster mosaic, then one learned newline per row.
        local = op.Transpose(local, perm=[0, 2, 1, 3, 4])
        local_h = op.Mul(grid_h, op.Constant(value_int=self._local_side))
        local_w = op.Mul(grid_w, op.Constant(value_int=self._local_side))
        local = self._append_newlines(op, local, local_h, local_w)

        overview = op.Squeeze(self.global_encoder(op, global_pixel_values), [0])
        overview = self._append_newlines(
            op,
            overview,
            op.Constant(value_int=self._global_side),
            op.Constant(value_int=self._global_side),
        )
        overview = op.Concat(
            overview,
            op.Reshape(self.view_separator, [1, self._output_size]),
            axis=0,
        )
        return op.Concat(local, overview, axis=0)


class _DeepSeekQueryAttention(nn.Module):
    """Qwen2 GQA used as DeepSeek-OCR-2's query encoder."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float,
    ):
        super().__init__()
        head_dim = hidden_size // num_heads
        self.q_proj = Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.out_proj = Linear(hidden_size, hidden_size, bias=False)
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        inv_freq = 1.0 / (
            rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
        )
        self.inv_freq = nn.Parameter([len(inv_freq)], data=ir.tensor(inv_freq))
        self.inv_freq._keep_float32 = True

    def _rope(self, op: OpBuilder, states: ir.Value) -> ir.Value:
        seq_len = op.Squeeze(op.Shape(states, start=2, end=3))
        positions = op.Cast(
            op.Range(op.Constant(value_int=0), seq_len, op.Constant(value_int=1)),
            to=ir.DataType.FLOAT,
        )
        inv_freq = op.Cast(self.inv_freq, to=ir.DataType.FLOAT)
        freqs = op.Mul(op.Unsqueeze(positions, [1]), op.Unsqueeze(inv_freq, [0]))
        cos = op.CastLike(op.Unsqueeze(op.Cos(freqs), [0, 1]), states)
        sin = op.CastLike(op.Unsqueeze(op.Sin(freqs), [0, 1]), states)
        half = self._head_dim // 2
        first = op.Slice(states, [0], [half], [3])
        second = op.Slice(states, [half], [self._head_dim], [3])
        return op.Concat(
            op.Sub(op.Mul(first, cos), op.Mul(second, sin)),
            op.Add(op.Mul(first, sin), op.Mul(second, cos)),
            axis=3,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
    ) -> ir.Value:
        batch = op.Shape(hidden_states, start=0, end=1)
        seq = op.Shape(hidden_states, start=1, end=2)

        def heads(states: ir.Value, count: int) -> ir.Value:
            states = op.Reshape(
                states,
                op.Concat(batch, seq, [count, self._head_dim], axis=0),
            )
            return op.Transpose(states, perm=[0, 2, 1, 3])

        query = self._rope(op, heads(self.q_proj(op, hidden_states), self._num_heads))
        key = self._rope(op, heads(self.k_proj(op, hidden_states), self._num_kv_heads))
        value = heads(self.v_proj(op, hidden_states), self._num_kv_heads)
        output = op.Attention(
            query,
            key,
            value,
            attention_bias,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_kv_heads,
            scale=float(self._head_dim**-0.5),
            is_causal=0,
        )
        output = op.Transpose(output, perm=[0, 2, 1, 3])
        output = op.Reshape(
            output,
            op.Concat(batch, seq, [self._num_heads * self._head_dim], axis=0),
        )
        return self.out_proj(op, output)


class _DeepSeekQueryBlock(nn.Module):
    """Qwen2 pre-norm GQA block used by the OCR-2 visual query encoder."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
        rope_theta: float,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = _DeepSeekQueryAttention(
            hidden_size,
            num_heads,
            num_kv_heads,
            rope_theta,
        )
        self.mlp = GatedMLP(
            hidden_size,
            intermediate_size,
            activation="silu",
            bias=False,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
    ) -> ir.Value:
        hidden_states = op.Add(
            hidden_states,
            self.attn(op, self.norm1(op, hidden_states), attention_bias),
        )
        return op.Add(hidden_states, self.mlp(op, self.norm2(op, hidden_states)))


class DeepSeekOCR2QueryEncoder(nn.Module):
    """Dynamic 144/256-query Qwen2 encoder over DeepSeek SAM features."""

    def __init__(
        self,
        *,
        depth: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
        rope_theta: float = 1_000_000.0,
    ):
        super().__init__()
        self.query_768 = nn.Parameter([144, hidden_size])
        self.query_1024 = nn.Parameter([256, hidden_size])
        self.blocks = nn.ModuleList(
            [
                _DeepSeekQueryBlock(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    num_kv_heads,
                    norm_eps,
                    rope_theta,
                )
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self._hidden_size = hidden_size

    def _dual_mask(
        self,
        op: OpBuilder,
        visual_tokens: ir.Value,
        total_tokens: ir.Value,
        like: ir.Value,
    ) -> ir.Value:
        positions = op.Range(
            op.Constant(value_int=0),
            total_tokens,
            op.Constant(value_int=1),
        )
        query = op.Unsqueeze(positions, [1])
        key = op.Unsqueeze(positions, [0])
        query_is_visual = op.Less(query, visual_tokens)
        key_is_visual = op.Less(key, visual_tokens)
        allowed = op.Or(
            op.And(query_is_visual, key_is_visual),
            op.And(
                op.Not(query_is_visual),
                op.Or(key_is_visual, op.LessOrEqual(key, query)),
            ),
        )
        return op.Unsqueeze(
            op.Where(
                allowed,
                op.CastLike(0.0, like),
                op.CastLike(-1e9, like),
            ),
            [0, 1],
        )

    def forward(self, op: OpBuilder, sam_features: ir.Value) -> ir.Value:
        batch = op.Shape(sam_features, start=0, end=1)
        hidden_states = op.Transpose(sam_features, perm=[0, 2, 3, 1])
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(batch, [-1, self._hidden_size], axis=0),
        )
        visual_tokens = op.Squeeze(op.Shape(hidden_states, start=1, end=2))
        valid = op.Or(
            op.Equal(visual_tokens, op.Constant(value_int=144)),
            op.Equal(visual_tokens, op.Constant(value_int=256)),
        )
        invalid = op.Cast(op.Not(valid), to=ir.DataType.INT64)
        guard = op.Gather(op.Constant(value_ints=[0]), invalid)
        hidden_states = op.Add(hidden_states, op.CastLike(guard, hidden_states))

        query_768 = op.Pad(
            self.query_768,
            op.Constant(value_ints=[0, 0, 112, 0]),
            op.CastLike(0.0, self.query_768),
        )
        queries = op.Where(
            op.Equal(visual_tokens, op.Constant(value_int=256)),
            self.query_1024,
            query_768,
        )
        queries = op.Slice(queries, [0], op.Reshape(visual_tokens, [1]), [0])
        queries = op.Expand(
            op.Unsqueeze(queries, [0]),
            op.Concat(batch, op.Shape(queries), axis=0),
        )
        hidden_states = op.Concat(hidden_states, queries, axis=1)
        total_tokens = op.Mul(visual_tokens, op.Constant(value_int=2))
        attention_bias = self._dual_mask(
            op,
            visual_tokens,
            total_tokens,
            hidden_states,
        )
        for block in self.blocks:
            hidden_states = block(op, hidden_states, attention_bias)
        hidden_states = self.norm(op, hidden_states)
        return op.Slice(
            hidden_states,
            op.Reshape(visual_tokens, [1]),
            op.Reshape(total_tokens, [1]),
            [1],
        )


class DeepSeekOCR2VisionEncoder(nn.Module):
    """DeepSeek-OCR-2 SAM + dynamic query encoder for one processor view."""

    def __init__(
        self,
        *,
        image_size: int,
        sam_hidden_size: int,
        sam_num_heads: int,
        sam_depth: int,
        sam_window_size: int,
        hidden_size: int,
        intermediate_size: int,
        depth: int,
        num_heads: int,
        num_kv_heads: int,
        output_size: int,
        norm_eps: float,
    ):
        super().__init__()
        self.sam = SAMVisionEncoder(
            img_size=image_size,
            patch_size=16,
            embed_dim=sam_hidden_size,
            depth=sam_depth,
            num_heads=sam_num_heads,
            window_size=sam_window_size,
            downsample_channels=(512, hidden_size),
            position_size=64,
            rel_pos_size=64,
            mlp_activation="gelu_pytorch_tanh",
        )
        self.query_encoder = DeepSeekOCR2QueryEncoder(
            depth=depth,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            norm_eps=norm_eps,
        )
        self.projector = Linear(hidden_size, output_size, bias=True)
        self.input_schema = (
            ("pixel_values", ir.DataType.FLOAT, (1, 3, image_size, image_size)),
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        pixel_values = op.CastLike(pixel_values, self.sam.patch_embed.weight)
        hidden_states = self.sam(op, pixel_values)
        hidden_states = self.query_encoder(op, hidden_states)
        return self.projector(op, hidden_states)


class DeepSeekOCR2FullImageEncoder(nn.Module):
    """DeepSeek-OCR-2 local 768 views followed by its 1024 overview."""

    def __init__(
        self,
        *,
        sam_hidden_size: int,
        sam_num_heads: int,
        sam_depth: int,
        sam_window_size: int,
        hidden_size: int,
        intermediate_size: int,
        depth: int,
        num_heads: int,
        num_kv_heads: int,
        output_size: int,
        norm_eps: float,
    ):
        super().__init__()
        self.global_encoder = DeepSeekOCR2VisionEncoder(
            image_size=1024,
            sam_hidden_size=sam_hidden_size,
            sam_num_heads=sam_num_heads,
            sam_depth=sam_depth,
            sam_window_size=sam_window_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            depth=depth,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            output_size=output_size,
            norm_eps=norm_eps,
        )
        self.local_encoder = DeepSeekOCR2VisionEncoder(
            image_size=768,
            sam_hidden_size=sam_hidden_size,
            sam_num_heads=sam_num_heads,
            sam_depth=sam_depth,
            sam_window_size=sam_window_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            depth=depth,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            output_size=output_size,
            norm_eps=norm_eps,
        )
        self.view_separator = nn.Parameter([output_size])
        self._output_size = output_size
        self.input_schema = (
            ("global_pixel_values", ir.DataType.FLOAT, (1, 3, 1024, 1024)),
            ("local_pixel_values", ir.DataType.FLOAT, ("num_tiles", 3, 768, 768)),
        )

    def forward(
        self,
        op: OpBuilder,
        global_pixel_values: ir.Value,
        local_pixel_values: ir.Value,
    ) -> ir.Value:
        local = self.local_encoder(op, local_pixel_values)
        local = op.Reshape(local, [-1, self._output_size])
        overview = op.Squeeze(self.global_encoder(op, global_pixel_values), [0])
        overview = op.Concat(
            overview,
            op.Reshape(self.view_separator, [1, self._output_size]),
            axis=0,
        )
        return op.Concat(local, overview, axis=0)


class PartialRotaryAudioAttention(nn.Module):
    """Bidirectional audio attention with NEOX rotary on half of each head."""

    def __init__(self, hidden_size: int, num_heads: int, rope_theta: float):
        super().__init__()
        self.q_proj = Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = Linear(hidden_size, hidden_size, bias=True)
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads
        self._rotary_dim = self._head_dim // 2
        inv_freq = 1.0 / (
            rope_theta
            ** (np.arange(0, self._rotary_dim, 2, dtype=np.float32) / self._rotary_dim)
        )
        self.inv_freq = nn.Parameter([len(inv_freq)], data=ir.tensor(inv_freq))
        self.inv_freq._keep_float32 = True

    def _apply_rope(self, op: OpBuilder, states: ir.Value) -> ir.Value:
        seq_len = op.Squeeze(op.Shape(states, start=2, end=3))
        positions = op.Cast(
            op.Range(op.Constant(value_int=0), seq_len, op.Constant(value_int=1)),
            to=ir.DataType.FLOAT,
        )
        inv_freq = op.Cast(self.inv_freq, to=ir.DataType.FLOAT)
        freqs = op.Mul(op.Unsqueeze(positions, [1]), op.Unsqueeze(inv_freq, [0]))
        cos = op.CastLike(op.Unsqueeze(op.Cos(freqs), [0, 1]), states)
        sin = op.CastLike(op.Unsqueeze(op.Sin(freqs), [0, 1]), states)
        half = self._rotary_dim // 2
        first = op.Slice(states, [0], [half], [3])
        second = op.Slice(states, [half], [self._rotary_dim], [3])
        tail = op.Slice(states, [self._rotary_dim], [self._head_dim], [3])
        rotated_first = op.Sub(op.Mul(first, cos), op.Mul(second, sin))
        rotated_second = op.Add(op.Mul(first, sin), op.Mul(second, cos))
        return op.Concat(rotated_first, rotated_second, tail, axis=3)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        batch = op.Shape(hidden_states, start=0, end=1)
        seq = op.Shape(hidden_states, start=1, end=2)

        def project(linear: Linear) -> ir.Value:
            states = linear(op, hidden_states)
            states = op.Reshape(
                states,
                op.Concat(batch, seq, [self._num_heads, self._head_dim], axis=0),
            )
            return op.Transpose(states, perm=[0, 2, 1, 3])

        query = self._apply_rope(op, project(self.q_proj))
        key = self._apply_rope(op, project(self.k_proj))
        value = project(self.v_proj)
        output = op.Attention(
            query,
            key,
            value,
            scale=float(self._head_dim**-0.5),
            is_causal=0,
        )
        output = op.Transpose(output, perm=[0, 2, 1, 3])
        output = op.Reshape(
            output,
            op.Concat(batch, seq, [self._num_heads * self._head_dim], axis=0),
        )
        return self.out_proj(op, output)


class Dots3NoteAudioBlock(nn.Module):
    """Dots3Note pre-norm audio transformer block."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        *,
        norm_eps: float,
        rope_theta: float,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = PartialRotaryAudioAttention(hidden_size, num_heads, rope_theta)
        self.mlp = GatedMLP(
            hidden_size,
            intermediate_size,
            activation="silu",
            bias=True,
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        residual = hidden_states
        hidden_states = op.Add(
            residual,
            self.attn(op, self.norm1(op, hidden_states)),
        )
        return op.Add(hidden_states, self.mlp(op, self.norm2(op, hidden_states)))


class Dots3NoteAudioEncoder(nn.Module):
    """Dots3Note mel Conv2d stem, rotary transformer, and audio adapter."""

    def __init__(
        self,
        *,
        num_mel_bins: int,
        conv_channels: int,
        depth: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        output_size: int,
        norm_eps: float,
        rope_theta: float = 10_000.0,
    ):
        super().__init__()
        self.conv2d = nn.ModuleList(
            [
                Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),
                Conv2d(
                    conv_channels,
                    conv_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                Conv2d(
                    conv_channels,
                    conv_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        mel_after_stem = (num_mel_bins + 7) // 8
        self.conv_out = Linear(
            conv_channels * mel_after_stem,
            hidden_size,
            bias=False,
        )
        self.blocks = nn.ModuleList(
            [
                Dots3NoteAudioBlock(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    norm_eps=norm_eps,
                    rope_theta=rope_theta,
                )
                for _ in range(depth)
            ]
        )
        self.post_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        self.projector = Dots3NoteAudioProjector(
            hidden_size,
            output_size,
            output_size,
        )
        self.input_schema = (
            ("input_features", ir.DataType.FLOAT, (1, num_mel_bins, "audio_frames")),
        )

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        # Processor contract: float32 (B, mel, frames).
        first_conv = self.conv2d[0]
        if not isinstance(first_conv, Conv2d):
            raise TypeError("Dots3Note audio stem must begin with Conv2d")
        input_features = op.CastLike(input_features, first_conv.weight)
        hidden_states = op.Unsqueeze(input_features, [1])
        for conv in self.conv2d:
            hidden_states = op.Gelu(conv(op, hidden_states))
        # (B, C, mel/8, frames/8) -> (B, frames/8, C*mel/8).
        hidden_states = op.Transpose(hidden_states, perm=[0, 3, 1, 2])
        batch = op.Shape(hidden_states, start=0, end=1)
        frames = op.Shape(hidden_states, start=1, end=2)
        hidden_states = op.Reshape(hidden_states, op.Concat(batch, frames, [-1], axis=0))
        hidden_states = self.conv_out(op, hidden_states)
        for block in self.blocks:
            hidden_states = block(op, hidden_states)
        hidden_states = self.post_layernorm(op, hidden_states)
        return self.projector(op, hidden_states)
