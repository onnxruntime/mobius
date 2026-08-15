# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen3-VL vision encoder components.

Provides modules for the Qwen3-VL vision backbone:

- ``Qwen3VLPatchEmbed``: Conv3d patch tokenisation (temporal + spatial).
- ``Qwen3VLVisionRotaryEmbedding``: 2D rotary embeddings from grid positions.
- ``Qwen3VLVisionAttention``: Packed bidirectional MHA with ``cu_seqlens``.
- ``Qwen3VLVisionBlock``: Pre-norm transformer block (uses ``FCMLP``).
- ``Qwen3VLPatchMerger``: Spatial merge to reduce token count.
- ``Qwen3VLVisionModel``: Full encoder stack with DeepStack outputs.

Packed attention loops over sub-sequences indicated by ``cu_seqlens``.
This uses standard ONNX ops; the ``rewrite_rules`` submodule provides
optional rules to replace the loop with a custom packed-attention op.
"""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities, get_build_dtype
from mobius.components._common import LayerNorm, Linear, build_packed_token_offset
from mobius.components._mlp import FCMLP


class Qwen3VLPatchEmbed(nn.Module):
    """Conv3d patch embedding for video / image tokens.

    Reshapes flat input ``(total_patches, C * T_p * P * P)`` into 5-D,
    applies Conv3d with kernel = stride = ``(T_p, P, P)``, and flattens
    back to ``(total_patches, hidden_size)``.
    """

    def __init__(
        self,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        hidden_size: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        # Conv3d weight: [out_channels, in_channels, kD, kH, kW]
        self.weight = nn.Parameter(
            [hidden_size, in_channels, temporal_patch_size, patch_size, patch_size],
            name="proj.weight",
        )
        self.bias = nn.Parameter([hidden_size], name="proj.bias")

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        # hidden_states: (total_patches, C * T_p * P * P)
        # Reshape to (total_patches, C, T_p, P, P) for Conv3d
        x = op.Reshape(
            hidden_states,
            [-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size],
        )
        x = op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self.temporal_patch_size, self.patch_size, self.patch_size],
            strides=[self.temporal_patch_size, self.patch_size, self.patch_size],
        )
        # x: (total_patches, hidden_size, 1, 1, 1) → flatten to (total_patches, hidden_size)
        return op.Reshape(x, [-1, self.hidden_size])


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """2D rotary position embeddings for the vision encoder.

    Precomputes a frequency table from which cos/sin values are looked up
    using 2D grid position IDs.

    In HuggingFace this is ``Qwen3VLVisionModel.rot_pos_emb()``.
    """

    def __init__(self, head_dim: int, max_grid_size: int = 4096):
        super().__init__()
        dim = head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))

        pos = np.arange(0, max_grid_size, dtype=np.float32)
        angles = np.outer(pos, inv_freq)
        cos_table = np.cos(angles).astype(np.float32)
        sin_table = np.sin(angles).astype(np.float32)

        self.cos_table = nn.Parameter(
            list(cos_table.shape),
            name="cos_table",
            data=ir.tensor(cos_table),
        )
        self.sin_table = nn.Parameter(
            list(sin_table.shape),
            name="sin_table",
            data=ir.tensor(sin_table),
        )

    def forward(self, op: OpBuilder, position_ids: ir.Value):
        """Look up cos/sin for 2D position IDs.

        Args:
            op: ONNX op builder.
            position_ids: ``(total_tokens, 2)`` with [h_pos, w_pos] per token.

        Returns:
            Tuple of ``(cos, sin)`` each ``(total_tokens, head_dim // 2)``.
        """
        # position_ids: (total_tokens, 2) — h_indices and w_indices
        h_pos = op.Gather(position_ids, [0], axis=1)  # (total_tokens, 1)
        w_pos = op.Gather(position_ids, [1], axis=1)

        h_pos = op.Squeeze(h_pos, [1])  # (total_tokens,)
        w_pos = op.Squeeze(w_pos, [1])

        cos_h = op.Gather(self.cos_table, h_pos)  # (total_tokens, dim//2)
        sin_h = op.Gather(self.sin_table, h_pos)
        cos_w = op.Gather(self.cos_table, w_pos)
        sin_w = op.Gather(self.sin_table, w_pos)

        cos = op.Concat(cos_h, cos_w, axis=-1)  # (total_tokens, dim)
        sin = op.Concat(sin_h, sin_w, axis=-1)
        return cos, sin


class Qwen3VLVisionAttention(nn.Module):
    """Packed bidirectional multi-head attention for the vision encoder.

    Iterates over sub-sequences delimited by ``cu_seqlens`` and applies
    standard ONNX Attention (opset 24) to each independently.  This avoids
    cross-image attention while processing all patches in a single flat
    sequence.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        # Fused QKV projection (matches HF weight name ``attn.qkv``)
        self.qkv = Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
        position_embeddings: tuple,
    ):
        # hidden_states: (total_seq, hidden_size)  — flat packed sequence
        # cu_seqlens: (num_sub_seqs + 1,) — cumulative sequence lengths
        # position_embeddings: (cos, sin) each (total_seq, rotary_dim)

        cos, sin = position_embeddings

        qkv = self.qkv(op, hidden_states)

        # Split into Q, K, V: each (total_seq, hidden_size)
        q, k, v = op.Split(qkv, axis=-1, num_outputs=3, _outputs=3)

        # Reshape to (total_seq, num_heads, head_dim) for RoPE
        q = op.Reshape(q, [0, self.num_heads, self.head_dim])
        k = op.Reshape(k, [0, self.num_heads, self.head_dim])

        # Apply rotary embedding (vision uses full rotation, no partial)
        # Cast cos/sin to match query dtype (tables are float32, model may be f16/bf16)
        cos = op.CastLike(cos, q)
        sin = op.CastLike(sin, q)
        cos = op.Unsqueeze(cos, [1])  # (total_seq, 1, rotary_dim)
        sin = op.Unsqueeze(sin, [1])

        half = self.head_dim // 2
        q1, q2 = op.Split(q, [half, half], axis=-1, _outputs=2)
        k1, k2 = op.Split(k, [half, half], axis=-1, _outputs=2)

        q_rot = op.Concat(
            op.Sub(op.Mul(cos, q1), op.Mul(sin, q2)),
            op.Add(op.Mul(sin, q1), op.Mul(cos, q2)),
            axis=-1,
        )
        k_rot = op.Concat(
            op.Sub(op.Mul(cos, k1), op.Mul(sin, k2)),
            op.Add(op.Mul(sin, k1), op.Mul(cos, k2)),
            axis=-1,
        )

        # Flatten back to (total_seq, hidden_size)
        q_rot = op.Reshape(q_rot, [0, -1])
        k_rot = op.Reshape(k_rot, [0, -1])

        capabilities = ep_capabilities()
        if capabilities.supports_packed_multi_head_attention:
            attn_output = self._emit_packed_mha(op, q_rot, k_rot, v, cu_seqlens, hidden_states)
        else:
            attn_output = self._emit_standard_attention(
                op, q_rot, k_rot, v, cu_seqlens, hidden_states
            )

        return self.proj(op, attn_output)

    def _emit_packed_mha(self, op, query, key, value, cu_seqlens, hidden_states):
        """Emit com.microsoft.PackedMultiHeadAttention.

        Uses cu_seqlens natively, avoiding the O(N^2) block-diagonal bias.
        Each sub-sequence delimited by ``cu_seqlens`` (one per image/frame)
        is one packed batch element, so block-diagonal masking is expressed
        through the varlen contract rather than an explicit bias.

        Args:
            query, key: (total_seq, hidden_size) after rotary embedding
            value: (total_seq, hidden_size) from QKV split
            cu_seqlens: (num_sub_seqs + 1,) cumulative sequence lengths
            hidden_states: original input, used only for shape
        """
        # token_offset: (num_sub_seqs, max_seq_len) mapping packed tokens to
        # their padded (batch, seq) layout. ORT derives batch_size from
        # token_offset.shape[0] and requires cumulative_sequence_length to have
        # length batch_size + 1, so this MUST encode every sub-sequence (image
        # or frame), not a single (1, N) batch — otherwise the kernel rejects
        # multi-sequence cu_seqlens and computes full instead of block-diagonal
        # attention.
        token_offset = build_packed_token_offset(op, cu_seqlens)

        cu_seqlens_int32 = op.Cast(cu_seqlens, to=6)  # INT32

        # PackedMHA supports float32 and float16 only; cast bfloat16 builds to
        # float16 and leave float32/float16 native to preserve precision.
        if get_build_dtype() == ir.DataType.BFLOAT16:
            query_mha = op.Cast(query, to=ir.DataType.FLOAT16)
            key_mha = op.Cast(key, to=ir.DataType.FLOAT16)
            value_mha = op.Cast(value, to=ir.DataType.FLOAT16)
        else:
            query_mha, key_mha, value_mha = query, key, value

        attn_output = op.PackedMultiHeadAttention(
            query_mha,
            key_mha,
            value_mha,
            None,  # bias (optional, not used)
            token_offset,
            cu_seqlens_int32,
            num_heads=self.num_heads,
            scale=self.scale,
            _domain="com.microsoft",
            _outputs=["packed_attn_out"],
        )  # (total_seq, hidden_size)

        # Cast back to original dtype
        attn_output = op.CastLike(attn_output, query)

        return attn_output

    def _emit_standard_attention(self, op, query, key, value, cu_seqlens, hidden_states):
        """Emit standard Attention with block-diagonal bias from cu_seqlens.

        Fallback path for EPs without PackedMultiHeadAttention support.

        Args:
            query, key: (total_seq, hidden_size) after rotary embedding
            value: (total_seq, hidden_size) from QKV split
            cu_seqlens: (num_sub_seqs + 1,) cumulative lengths
            hidden_states: original input, used only for shape
        """
        # Build block-diagonal attention bias from cu_seqlens
        # Each token only attends to tokens in the same sub-sequence
        total_seq = op.Shape(hidden_states, start=0, end=1)
        total_seq_scalar = op.Squeeze(total_seq)
        positions = op.Range(
            op.Constant(value_int=0),
            total_seq_scalar,
            op.Constant(value_int=1),
        )
        positions_2d = op.Unsqueeze(positions, [1])  # (total_seq, 1)
        cu_seqlens_2d = op.Unsqueeze(cu_seqlens, [0])  # (1, num_sub_seqs+1)
        greater_or_equal = op.GreaterOrEqual(
            positions_2d, cu_seqlens_2d
        )  # (total_seq, num_sub_seqs+1)
        greater_or_equal_int = op.Cast(greater_or_equal, to=7)  # INT64
        segment_ids = op.Sub(
            op.ReduceSum(greater_or_equal_int, [1], keepdims=False),
            op.Constant(value_int=1),
        )  # (total_seq,) — segment ID per token

        segment_row = op.Unsqueeze(segment_ids, [1])  # (total_seq, 1)
        segment_column = op.Unsqueeze(segment_ids, [0])  # (1, total_seq)
        same_segment = op.Equal(segment_row, segment_column)  # (total_seq, total_seq)
        attn_bias = op.Where(
            same_segment,
            op.CastLike(0.0, query),
            op.CastLike(-10000.0, query),
        )
        # Reshape for Attention: (1, 1, total_seq, total_seq)
        attn_bias = op.Unsqueeze(attn_bias, [0, 1])

        # Add batch dim: (1, total_seq, hidden_size)
        query_batched = op.Unsqueeze(query, [0])
        key_batched = op.Unsqueeze(key, [0])
        value_batched = op.Unsqueeze(value, [0])

        attn_output = op.Attention(
            query_batched,
            key_batched,
            value_batched,
            attn_bias,
            kv_num_heads=self.num_heads,
            q_num_heads=self.num_heads,
            scale=self.scale,
            _outputs=1,
        )

        # Remove batch dim: (total_seq, hidden_size)
        attn_output = op.Squeeze(attn_output, [0])

        return attn_output


class Qwen3VLVisionBlock(nn.Module):
    """Pre-norm vision transformer block with packed attention.

    Structure: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(hidden_size, num_heads)
        self.norm2 = LayerNorm(hidden_size, eps=1e-6)
        # GELU (tanh approx) MLP with bias (HF linear_fc1/linear_fc2 → up_proj/down_proj)
        self.mlp = FCMLP(hidden_size, intermediate_size, activation="gelu_new", bias=True)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        cu_seqlens: ir.Value,
        position_embeddings: tuple,
    ):
        residual = hidden_states
        hidden_states = self.norm1(op, hidden_states)
        hidden_states = self.attn(op, hidden_states, cu_seqlens, position_embeddings)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.norm2(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class Qwen3VLPatchMerger(nn.Module):
    """Spatial merge — reduces spatial resolution by merging adjacent patches.

    Reshapes tokens by the merge factor, normalises, then projects.
    ``use_postshuffle_norm=False`` (final merger): LayerNorm before reshape.
    ``use_postshuffle_norm=True`` (deepstack mergers): LayerNorm after reshape.
    """

    def __init__(
        self,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        self.merged_size = hidden_size * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        norm_dim = self.merged_size if use_postshuffle_norm else hidden_size
        self.norm = LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = Linear(self.merged_size, self.merged_size, bias=True)
        self.linear_fc2 = Linear(self.merged_size, out_hidden_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        if self.use_postshuffle_norm:
            # Reshape to merged dim, then normalise
            x = op.Reshape(hidden_states, [-1, self.merged_size])
            x = self.norm(op, x)
        else:
            # Normalise first, then reshape to merged dim
            x = self.norm(op, hidden_states)
            x = op.Reshape(x, [-1, self.merged_size])

        x = self.linear_fc1(op, x)
        x = op.Gelu(x, approximate="none")
        return self.linear_fc2(op, x)


class Qwen3VLVisionModel(nn.Module):
    """Full Qwen3-VL vision encoder with DeepStack outputs.

    Processes packed image/video patches through Conv3d embedding,
    bilinear-interpolated position embeddings, transformer blocks with
    packed attention, and spatial merge.

    Args:
        depth: Number of vision transformer blocks.
        hidden_size: Hidden dimension of the vision encoder.
        intermediate_size: MLP intermediate dimension.
        num_heads: Number of attention heads.
        patch_size: Spatial patch size.
        temporal_patch_size: Temporal patch size.
        in_channels: Number of input channels.
        out_hidden_size: Output projection dimension (after merge).
        spatial_merge_size: Factor for spatial merge.
        num_position_embeddings: Size of the learned 2D position grid.
        deepstack_visual_indexes: Layer indices for DeepStack features.
    """

    def __init__(
        self,
        depth: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        out_hidden_size: int | None = None,
        spatial_merge_size: int = 2,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: list[int] | None = None,
    ):
        super().__init__()
        if out_hidden_size is None:
            out_hidden_size = hidden_size
        if deepstack_visual_indexes is None:
            deepstack_visual_indexes = []

        self.depth = depth
        self.hidden_size = hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.deepstack_visual_indexes = deepstack_visual_indexes
        self.num_grid_per_side = math.isqrt(num_position_embeddings)

        head_dim = hidden_size // num_heads

        self.patch_embed = Qwen3VLPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
        )

        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(
            head_dim=head_dim,
        )

        # Learned position embedding grid (num_grid_per_side^2, hidden_size)
        self.pos_embed = nn.Parameter(
            [num_position_embeddings, hidden_size],
            name="pos_embed.weight",
        )

        self.blocks = nn.ModuleList(
            [
                Qwen3VLVisionBlock(hidden_size, intermediate_size, num_heads)
                for _ in range(depth)
            ]
        )

        self.merger = Qwen3VLPatchMerger(
            hidden_size=hidden_size,
            out_hidden_size=out_hidden_size,
            spatial_merge_size=spatial_merge_size,
            use_postshuffle_norm=False,
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLPatchMerger(
                    hidden_size=hidden_size,
                    out_hidden_size=out_hidden_size,
                    spatial_merge_size=spatial_merge_size,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(deepstack_visual_indexes))
            ]
        )

    def _flat_grid_coordinates(self, op, grid_thw):
        """Map each packed patch to its media row and merge-permuted H/W coordinates.

        Qwen3-VL stores each media item's patches in
        ``(T, H // ms, W // ms, ms, ms)`` order. Computing coordinates over
        the concatenated patch stream avoids control-flow subgraphs while
        preserving arbitrary image/video sizes and order.
        """
        ms = self.spatial_merge_size
        T_col = op.Squeeze(op.Slice(grid_thw, [0], [1], [1], [1]), [1])  # noqa: N806
        H_col = op.Squeeze(op.Slice(grid_thw, [1], [2], [1], [1]), [1])  # noqa: N806
        W_col = op.Squeeze(op.Slice(grid_thw, [2], [3], [1], [1]), [1])  # noqa: N806
        patches_per_media = op.Mul(T_col, op.Mul(H_col, W_col))
        patch_ends = op.CumSum(patches_per_media, op.Constant(value_int=0))
        patch_starts = op.Pad(
            patch_ends,
            op.Constant(value_ints=[1, 0]),
            op.Constant(value_int=0),
        )

        total_patches = op.ReduceSum(patches_per_media, keepdims=False)
        patch_ids = op.Range(
            op.Constant(value_int=0),
            total_patches,
            op.Constant(value_int=1),
        )
        # The number of completed media ranges is the owning media row.
        media_ids = op.ReduceSum(
            op.Cast(
                op.GreaterOrEqual(
                    op.Unsqueeze(patch_ids, [1]),
                    op.Unsqueeze(patch_ends, [0]),
                ),
                to=7,
            ),
            [1],
            keepdims=False,
        )
        local_ids = op.Sub(patch_ids, op.Gather(patch_starts, media_ids))

        H = op.Gather(H_col, media_ids)  # noqa: N806
        W = op.Gather(W_col, media_ids)  # noqa: N806
        patches_per_frame = op.Mul(H, W)
        frame_local_ids = op.Mod(local_ids, patches_per_frame)

        merge_area = op.Constant(value_int=ms * ms)
        merge_block_ids = op.Div(frame_local_ids, merge_area)
        intra_merge_ids = op.Mod(frame_local_ids, merge_area)
        W_m = op.Div(W, op.Constant(value_int=ms))  # noqa: N806
        block_rows = op.Div(merge_block_ids, W_m)
        block_cols = op.Mod(merge_block_ids, W_m)
        intra_rows = op.Div(intra_merge_ids, op.Constant(value_int=ms))
        intra_cols = op.Mod(intra_merge_ids, op.Constant(value_int=ms))
        rows = op.Add(op.Mul(block_rows, op.Constant(value_int=ms)), intra_rows)
        cols = op.Add(op.Mul(block_cols, op.Constant(value_int=ms)), intra_cols)
        return rows, cols, H, W

    def _interpolate_pos_embed(self, op, grid_thw):
        """Bilinearly interpolate learned positions for the packed media stream.

        Matches HuggingFace ``Qwen3VLVisionModel.fast_pos_embed_interpolate``.

        Args:
            op: OpBuilder instance.
            grid_thw: ``(num_images, 3)`` INT64 with ``[T, H, W]`` per image.

        Returns:
            Position embeddings ``(total_patches, hidden_size)``.
        """
        n = self.num_grid_per_side
        rows, cols, H, W = self._flat_grid_coordinates(op, grid_thw)  # noqa: N806
        rows_f = op.Cast(rows, to=1)
        cols_f = op.Cast(cols, to=1)
        H_f = op.Cast(H, to=1)  # noqa: N806
        W_f = op.Cast(W, to=1)  # noqa: N806
        rows_scaled = op.Div(op.Mul(rows_f, float(n - 1)), op.Sub(H_f, 1.0))
        cols_scaled = op.Div(op.Mul(cols_f, float(n - 1)), op.Sub(W_f, 1.0))

        row_floor = op.Cast(op.Floor(rows_scaled), to=7)
        col_floor = op.Cast(op.Floor(cols_scaled), to=7)
        clip_max = op.Constant(value_int=n - 1)
        row_ceil = op.Min(op.Add(row_floor, op.Constant(value_int=1)), clip_max)
        col_ceil = op.Min(op.Add(col_floor, op.Constant(value_int=1)), clip_max)

        row_delta = op.Sub(rows_scaled, op.Cast(row_floor, to=1))
        col_delta = op.Sub(cols_scaled, op.Cast(col_floor, to=1))
        one_minus_row = op.Sub(1.0, row_delta)
        one_minus_col = op.Sub(1.0, col_delta)
        w_00 = op.Unsqueeze(op.Mul(one_minus_row, one_minus_col), [1])
        w_01 = op.Unsqueeze(op.Mul(one_minus_row, col_delta), [1])
        w_10 = op.Unsqueeze(op.Mul(row_delta, one_minus_col), [1])
        w_11 = op.Unsqueeze(op.Mul(row_delta, col_delta), [1])

        row_floor_base = op.Mul(row_floor, op.Constant(value_int=n))
        row_ceil_base = op.Mul(row_ceil, op.Constant(value_int=n))
        idx_00 = op.Add(row_floor_base, col_floor)
        idx_01 = op.Add(row_floor_base, col_ceil)
        idx_10 = op.Add(row_ceil_base, col_floor)
        idx_11 = op.Add(row_ceil_base, col_ceil)

        # Interpolate in float32 even when the learned table is f16/bf16.
        e_00 = op.Mul(op.Cast(op.Gather(self.pos_embed, idx_00), to=1), w_00)
        e_01 = op.Mul(op.Cast(op.Gather(self.pos_embed, idx_01), to=1), w_01)
        e_10 = op.Mul(op.Cast(op.Gather(self.pos_embed, idx_10), to=1), w_10)
        e_11 = op.Mul(op.Cast(op.Gather(self.pos_embed, idx_11), to=1), w_11)
        return op.Add(op.Add(e_00, e_01), op.Add(e_10, e_11))

    def _compute_rotary_pos_ids(self, op, grid_thw):
        """Compute 2D rotary position IDs for all images via ONNX Scan.

        Matches HF ``Qwen3VLVisionModel.rot_pos_emb()`` position indexing.

        Returns ``(total_patches, 2)`` INT64 with ``[h_pos, w_pos]`` per patch.
        """
        rows, cols, _, _ = self._flat_grid_coordinates(op, grid_thw)
        return op.Concat(op.Unsqueeze(rows, [1]), op.Unsqueeze(cols, [1]), axis=1)

    def _compute_cu_seqlens(self, op, grid_thw):
        """Compute full-attention cu_seqlens for all images.

        Produces per-frame boundaries across all images without a control-flow
        subgraph, equivalent to ``repeat_interleave(H * W, T)`` + CumSum.

        Returns ``(total_frames + 1,)`` INT64.
        """
        T_col = op.Squeeze(op.Slice(grid_thw, [0], [1], [1], [1]), [1])  # noqa: N806
        H_col = op.Squeeze(op.Slice(grid_thw, [1], [2], [1], [1]), [1])  # noqa: N806
        W_col = op.Squeeze(op.Slice(grid_thw, [2], [3], [1], [1]), [1])  # noqa: N806
        frame_ends = op.CumSum(T_col, op.Constant(value_int=0))
        total_frames = op.ReduceSum(T_col, keepdims=False)
        frame_ids = op.Range(
            op.Constant(value_int=0),
            total_frames,
            op.Constant(value_int=1),
        )
        media_ids = op.ReduceSum(
            op.Cast(
                op.GreaterOrEqual(
                    op.Unsqueeze(frame_ids, [1]),
                    op.Unsqueeze(frame_ends, [0]),
                ),
                to=7,
            ),
            [1],
            keepdims=False,
        )
        hw_per_media = op.Mul(H_col, W_col)
        hw_per_frame = op.Gather(hw_per_media, media_ids)
        cu = op.CumSum(hw_per_frame, op.Constant(value_int=0))
        return op.Pad(cu, op.Constant(value_ints=[1, 0]), op.Constant(value_int=0))

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        grid_thw: ir.Value,
    ):
        """Run the vision encoder.

        Args:
            hidden_states: Flat patches ``(total_patches, C * T_p * P * P)``.
            grid_thw: ``(num_images, 3)`` INT64 with ``[T, H, W]`` per image.

        Returns:
            Tuple of ``(merged_hidden_states, *deepstack_features)`` where
            ``merged_hidden_states`` has shape
            ``(total_merged_patches, out_hidden_size)`` and each deepstack
            feature tensor has the same shape.
        """
        # Patch embedding
        hidden_states = self.patch_embed(op, hidden_states)

        # Bilinear-interpolated position embeddings from learned grid.
        # Cast to match hidden_states dtype (interpolation computes in float32).
        pos_embeds = self._interpolate_pos_embed(op, grid_thw)
        pos_embeds = op.CastLike(pos_embeds, hidden_states)
        hidden_states = op.Add(hidden_states, pos_embeds)

        # Compute rotary position IDs and embeddings from grid_thw
        rotary_pos_ids = self._compute_rotary_pos_ids(op, grid_thw)
        position_embeddings = self.rotary_pos_emb(op, rotary_pos_ids)

        # Compute cu_seqlens from grid_thw
        cu_seqlens = self._compute_cu_seqlens(op, grid_thw)

        # Transformer blocks
        deepstack_features = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(op, hidden_states, cu_seqlens, position_embeddings)

            if layer_idx in self.deepstack_visual_indexes:
                ds_idx = self.deepstack_visual_indexes.index(layer_idx)
                ds_feature = self.deepstack_merger_list[ds_idx](op, hidden_states)
                deepstack_features.append(ds_feature)

        # Final spatial merge
        merged = self.merger(op, hidden_states)

        return merged, *deepstack_features
