# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Segment Anything Model (SAM) — facebook/sam-vit-base.

Replicates ``transformers.SamModel``: a ViT vision encoder, prompt
encoder (point / box inputs), and two-way-transformer mask decoder.

The architecture produces three sub-models at export time:

- **vision_encoder**: pixel_values → image_embeddings
- **decoder**: image_embeddings + input_points + input_labels
  → pred_masks + iou_predictions
"""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import SamConfig
from mobius.components._activations import get_activation
from mobius.components._common import Embedding, LayerNorm, Linear
from mobius.components._conv import Conv2d, Conv2dNoBias, ConvTranspose2d

# ── helpers ──────────────────────────────────────────────────────────


def _channels_first_layer_norm(
    op: builder.OpBuilder,
    layer_norm: LayerNorm,
    x,
):
    """Apply LayerNorm to channels-first data (B, C, H, W).

    Transposes to (B, H, W, C), normalises, transposes back.
    """
    x = op.Transpose(x, perm=[0, 2, 3, 1])  # (B, H, W, C)
    x = layer_norm(op, x)
    return op.Transpose(x, perm=[0, 3, 1, 2])  # (B, C, H, W)


# ── Vision Encoder ───────────────────────────────────────────────────


class _SamPatchEmbeddings(nn.Module):
    """Conv2d patch projection: (B, 3, H, W) → (B, H/p, W/p, C)."""

    def __init__(self, config: SamConfig):
        super().__init__()
        self.projection = Conv2d(
            3,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # (B, 3, H, W) → Conv → (B, C, H/p, W/p)
        x = self.projection(op, pixel_values)
        # → (B, H/p, W/p, C)
        return op.Transpose(x, perm=[0, 2, 3, 1])


class _SamVisionAttention(nn.Module):
    """Global multi-head self-attention with fused QKV.

    Replicates ``transformers.models.sam.SamVisionAttention`` (global
    attention path, no windowed/relative position bias).
    """

    def __init__(self, config: SamConfig):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        self._num_heads = num_heads
        self._scale = 1.0 / math.sqrt(hidden_size // num_heads)

        self.qkv = Linear(hidden_size, hidden_size * 3, bias=config.qkv_bias)
        self.proj = Linear(hidden_size, hidden_size, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: (B, S, C)
        qkv = self.qkv(op, x)  # (B, S, 3*C)
        q, k, v = op.Split(qkv, num_outputs=3, axis=-1, _outputs=3)

        attn_output = op.Attention(
            q,
            k,
            v,
            kv_num_heads=self._num_heads,
            q_num_heads=self._num_heads,
            scale=self._scale,
            _outputs=1,
        )
        return self.proj(op, attn_output)  # (B, S, C)


class _SamVisionMLP(nn.Module):
    """Two-layer MLP in the vision encoder (GELU activation).

    Replicates ``transformers.models.sam.SamMLPBlock``.
    """

    def __init__(self, config: SamConfig):
        super().__init__()
        self.lin1 = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.lin2 = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self._act = get_activation("gelu")

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        x = self._act(op, self.lin1(op, x))
        return self.lin2(op, x)


class _SamVisionLayer(nn.Module):
    """Single vision-encoder layer: LN → Attention → residual → LN → MLP → residual.

    Input/output: (B, S, C) where S = H_grid * W_grid.
    Replicates ``transformers.models.sam.SamVisionLayer``.
    """

    def __init__(self, config: SamConfig):
        super().__init__()
        self.layer_norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = _SamVisionAttention(config)
        self.layer_norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = _SamVisionMLP(config)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: (B, S, C)
        residual = x
        x = self.layer_norm1(op, x)
        x = self.attn(op, x)
        x = op.Add(residual, x)

        residual = x
        x = self.layer_norm2(op, x)
        x = self.mlp(op, x)
        return op.Add(residual, x)


class _SamVisionNeck(nn.Module):
    """Neck: 1x1 conv, LN, 3x3 conv, LN.

    Maps (B, H, W, hidden) to (B, output_channels, H, W).
    Replicates ``transformers.models.sam.SamVisionNeck``.
    """

    def __init__(self, config: SamConfig):
        super().__init__()
        self.conv1 = Conv2dNoBias(
            config.hidden_size,
            config.output_channels,
            kernel_size=1,
        )
        self.layer_norm1 = LayerNorm(config.output_channels, eps=config.layer_norm_eps)
        self.conv2 = Conv2dNoBias(
            config.output_channels,
            config.output_channels,
            kernel_size=3,
            padding=1,
        )
        self.layer_norm2 = LayerNorm(config.output_channels, eps=config.layer_norm_eps)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: (B, H, W, C) → (B, C, H, W) for Conv
        x = op.Transpose(x, perm=[0, 3, 1, 2])

        x = self.conv1(op, x)  # (B, output_channels, H, W)
        x = _channels_first_layer_norm(op, self.layer_norm1, x)
        x = self.conv2(op, x)
        x = _channels_first_layer_norm(op, self.layer_norm2, x)
        return x  # (B, output_channels, H, W)


class SamVisionEncoder(nn.Module):
    """ViT backbone with patch embedding, positional embedding, and neck.

    Input: pixel_values (B, 3, H, W)
    Output: image_embeddings (B, output_channels, H/p, W/p)

    Replicates ``transformers.models.sam.SamVisionEncoder``.
    """

    def __init__(self, config: SamConfig):
        super().__init__()
        image_embedding_size = config.image_embedding_size
        hidden_size = config.hidden_size

        self.patch_embed = _SamPatchEmbeddings(config)

        # Frozen positional embedding: (1, H_grid, W_grid, C)
        # Weights loaded from checkpoint; init with zeros as placeholder.
        self.pos_embed = nn.Parameter(
            [1, image_embedding_size, image_embedding_size, hidden_size],
            name="pos_embed",
            data=ir.tensor(
                np.zeros(
                    (1, image_embedding_size, image_embedding_size, hidden_size),
                    dtype=np.float32,
                )
            ),
        )

        self.layers = nn.ModuleList(
            [_SamVisionLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.neck = _SamVisionNeck(config)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # pixel_values: (B, 3, H, W)
        x = self.patch_embed(op, pixel_values)  # (B, H_g, W_g, C)
        x = op.Add(x, self.pos_embed)  # add positional embedding

        # Flatten spatial dims for transformer layers
        # (B, H_g, W_g, C) → (B, H_g*W_g, C)
        batch_dim = op.Shape(x, start=0, end=1)
        h_dim = op.Shape(x, start=1, end=2)
        w_dim = op.Shape(x, start=2, end=3)
        c_dim = op.Shape(x, start=3, end=4)
        hw = op.Mul(h_dim, w_dim)
        flat_shape = op.Concat(batch_dim, hw, c_dim, axis=0)
        x = op.Reshape(x, flat_shape)  # (B, H_g*W_g, C)

        for layer in self.layers:
            x = layer(op, x)

        # Restore spatial: (B, H_g*W_g, C) → (B, H_g, W_g, C)
        spatial_shape = op.Concat(batch_dim, h_dim, w_dim, c_dim, axis=0)
        x = op.Reshape(x, spatial_shape)

        # Neck: (B, H_g, W_g, C) → (B, output_channels, H_g, W_g)
        return self.neck(op, x)


# ── Positional Embedding ─────────────────────────────────────────────


class _SamPositionalEmbedding(nn.Module):
    """Fourier positional embedding for normalized [0, 1] coordinates.

    Learned ``positional_embedding`` of shape (2, num_pos_feats) maps 2-D
    coordinates to ``2 * num_pos_feats``-dimensional features.

    Replicates ``transformers.models.sam.SamPositionEmbedding``.
    """

    def __init__(self, config: SamConfig):
        super().__init__()
        self._num_pos_feats = config.num_pos_feats
        self._image_embedding_size = config.image_embedding_size

        self.positional_embedding = nn.Parameter([2, config.num_pos_feats])

    def forward(self, op: builder.OpBuilder, coords: ir.Value):
        """Encode coordinates.

        Args:
            coords: (*, 2) tensor with values in [0, 1].

        Returns:
            (*, 2 * num_pos_feats) positional features.
        """
        # Scale from [0, 1] to [-1, 1]
        scaled = op.Sub(op.Mul(coords, 2.0), 1.0)
        # Project: (*, 2) @ (2, D) → (*, D)
        proj = op.MatMul(scaled, self.positional_embedding)
        proj = op.Mul(proj, float(2 * math.pi))
        # Concat sin and cos: (*, 2*D)
        return op.Concat(op.Sin(proj), op.Cos(proj), axis=-1)

    def get_image_pe(self, op: builder.OpBuilder, height: int, width: int):
        """Compute image positional encoding grid.

        Returns a (1, 2*num_pos_feats, H, W) tensor suitable for adding to
        image embeddings.
        """
        # Create constant (H, W, 2) grid of normalised coordinates
        grid_y = np.linspace(0, 1, height, dtype=np.float32)
        grid_x = np.linspace(0, 1, width, dtype=np.float32)
        yy, xx = np.meshgrid(grid_y, grid_x, indexing="ij")
        grid = np.stack([xx, yy], axis=-1)  # (H, W, 2)
        grid_const = op.Constant(value=ir.tensor(grid))

        # Apply learned positional encoding
        pe = self.forward(op, grid_const)  # (H, W, 2*num_pos_feats)

        # Reshape to (1, 2*num_pos_feats, H, W)
        pe = op.Unsqueeze(pe, [0])  # (1, H, W, 2*num_pos_feats)
        return op.Transpose(pe, perm=[0, 3, 1, 2])


# ── Prompt Encoder ───────────────────────────────────────────────────


class _SamMaskEmbedding(nn.Module):
    """Downscale a (1, H, W) mask to (hidden_size, H/4, W/4) embedding.

    Two strided 2x2 convolutions (each halves spatial dims) followed by
    a 1x1 projection to hidden_size.

    Replicates ``transformers.models.sam.SamMaskEmbedding``.
    """

    def __init__(self, config: SamConfig):
        super().__init__()
        mask_channels = config.mask_input_channels
        self.conv1 = Conv2d(1, mask_channels // 4, kernel_size=2, stride=2)
        self.layer_norm1 = LayerNorm(mask_channels // 4, eps=config.layer_norm_eps)
        self.conv2 = Conv2d(mask_channels // 4, mask_channels, kernel_size=2, stride=2)
        self.layer_norm2 = LayerNorm(mask_channels, eps=config.layer_norm_eps)
        self.conv3 = Conv2d(mask_channels, config.output_channels, kernel_size=1)
        self._act = get_activation("gelu")

    def forward(self, op: builder.OpBuilder, masks: ir.Value):
        # masks: (B, 1, H, W)
        x = self.conv1(op, masks)  # (B, mask_ch//4, H/2, W/2)
        x = _channels_first_layer_norm(op, self.layer_norm1, x)
        x = self._act(op, x)
        x = self.conv2(op, x)  # (B, mask_ch, H/4, W/4)
        x = _channels_first_layer_norm(op, self.layer_norm2, x)
        x = self._act(op, x)
        return self.conv3(op, x)  # (B, hidden, H/4, W/4)


class _SamPromptEncoder(nn.Module):
    """Encode point/box prompts into sparse + dense embeddings.

    Replicates ``transformers.models.sam.SamPromptEncoder``.

    Inputs:
        input_points: (B, N, 2) — normalised point coordinates.
        input_labels: (B, N) — INT64 labels: 0=bg, 1=fg, 2=box_tl,
            3=box_br, -1=padding.

    Outputs:
        sparse_embeddings: (B, N, hidden_size)
        dense_embeddings:  (B, hidden_size, H_emb, W_emb)  (from no_mask)
    """

    def __init__(self, config: SamConfig):
        super().__init__()
        hidden_size = config.output_channels
        self._hidden_size = hidden_size
        self._image_embedding_size = config.image_embedding_size
        self._image_size = config.image_size

        # Point-type embeddings  (HF: point_embed.0 … point_embed.3)
        self.point_embed = nn.ModuleList(
            [Embedding(1, hidden_size) for _ in range(config.num_point_embeddings)]
        )
        self.not_a_point_embed = Embedding(1, hidden_size)

        # Dense embedding for "no mask" branch
        self.no_mask_embed = Embedding(1, hidden_size)

        # Mask downscaler (used when a real mask is provided; kept for
        # weight compatibility, not exercised in the default point-only path)
        self.mask_embed = _SamMaskEmbedding(config)

        # Shared positional embedding (injected from top-level model)
        self.shared_embedding: _SamPositionalEmbedding  # set externally

    def forward(
        self,
        op: builder.OpBuilder,
        input_points: ir.Value,
        input_labels: ir.Value,
        shared_embedding: _SamPositionalEmbedding,
    ):
        # ── Normalise points to [0, 1] ──
        # input_points: (B, N, 2) in pixel coords → divide by image_size
        points = op.Div(input_points, float(self._image_size))

        # ── Position encoding ──
        point_pe = shared_embedding(op, points)  # (B, N, hidden)

        # ── Type embeddings ──
        # Stack: [not_a_point, point_embed.0, .1, .2, .3]  (5, hidden)
        # Labels shifted by +1 so -1 → index 0 (not_a_point)
        # Use Embedding.forward() to trigger parameter realization
        idx_0 = op.Constant(value_ints=[0])
        all_type_weights = op.Concat(
            self.not_a_point_embed(op, idx_0),  # (1, hidden)
            self.point_embed[0](op, idx_0),
            self.point_embed[1](op, idx_0),
            self.point_embed[2](op, idx_0),
            self.point_embed[3](op, idx_0),
            axis=0,
        )  # (5, hidden)

        # Shift labels: -1 → 0, 0 → 1, … , 3 → 4
        shifted = op.Add(input_labels, 1)  # (B, N)
        type_embed = op.Gather(all_type_weights, shifted)  # (B, N, hidden)

        sparse_embeddings = op.Add(point_pe, type_embed)  # (B, N, hidden)

        # ── Dense embeddings (no-mask branch) ──
        # Expand no_mask_embed to (B, hidden, H_emb, W_emb)
        batch_dim = op.Shape(input_points, start=0, end=1)
        emb_size = self._image_embedding_size
        # no_mask_embed(op, 0): (hidden,) → reshape to (1, hidden, 1, 1)
        no_mask = op.Reshape(
            self.no_mask_embed(op, idx_0),
            op.Constant(value_ints=[1, self._hidden_size, 1, 1]),
        )
        target_shape = op.Concat(
            batch_dim,
            op.Constant(value_ints=[self._hidden_size, emb_size, emb_size]),
            axis=0,
        )
        dense_embeddings = op.Expand(
            no_mask,
            target_shape,
        )  # (B, hidden, H_emb, W_emb)

        return sparse_embeddings, dense_embeddings


# ── Mask Decoder ─────────────────────────────────────────────────────


class _SamAttention(nn.Module):
    """Multi-head attention with optional downsampling for the mask decoder.

    Replicates ``transformers.models.sam.SamAttention``.
    """

    def __init__(self, hidden_size: int, num_heads: int, downsample_rate: int = 1):
        super().__init__()
        internal_dim = hidden_size // downsample_rate
        self._num_heads = num_heads
        self._scale = 1.0 / math.sqrt(internal_dim // num_heads)

        self.q_proj = Linear(hidden_size, internal_dim, bias=True)
        self.k_proj = Linear(hidden_size, internal_dim, bias=True)
        self.v_proj = Linear(hidden_size, internal_dim, bias=True)
        self.out_proj = Linear(internal_dim, hidden_size, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        query: ir.Value,
        key: ir.Value,
        value: ir.Value,
    ):
        q = self.q_proj(op, query)  # (B, S_q, internal)
        k = self.k_proj(op, key)  # (B, S_k, internal)
        v = self.v_proj(op, value)  # (B, S_k, internal)

        attn_output = op.Attention(
            q,
            k,
            v,
            kv_num_heads=self._num_heads,
            q_num_heads=self._num_heads,
            scale=self._scale,
            _outputs=1,
        )
        return self.out_proj(op, attn_output)  # (B, S_q, hidden)


class _SamMLPBlock(nn.Module):
    """Two-layer MLP in the mask decoder two-way transformer.

    Replicates the MLP inside ``SamTwoWayAttentionBlock``.
    """

    def __init__(self, hidden_size: int, mlp_dim: int):
        super().__init__()
        self.lin1 = Linear(hidden_size, mlp_dim, bias=True)
        self.lin2 = Linear(mlp_dim, hidden_size, bias=True)
        self._act = get_activation("relu")

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return self.lin2(op, self._act(op, self.lin1(op, x)))


class _SamFeedForward(nn.Module):
    """Variable-depth MLP (used for hypernetwork + IoU head).

    Replicates ``transformers.models.sam.SamFeedForward``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.proj_in = Linear(input_dim, hidden_dim, bias=True)
        self.proj_out = Linear(hidden_dim, output_dim, bias=True)
        self.layers = nn.ModuleList(
            [Linear(hidden_dim, hidden_dim, bias=True) for _ in range(num_layers - 2)]
        )
        self._act = get_activation("relu")

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        x = self._act(op, self.proj_in(op, x))
        for layer in self.layers:
            x = self._act(op, layer(op, x))
        return self.proj_out(op, x)


class _SamTwoWayAttentionBlock(nn.Module):
    """Bidirectional cross-attention block between tokens and image features.

    Replicates ``transformers.models.sam.SamTwoWayAttentionBlock``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_dim: int,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ):
        super().__init__()
        self._skip_first_layer_pe = skip_first_layer_pe

        self.self_attn = _SamAttention(hidden_size, num_heads, downsample_rate=1)
        self.layer_norm1 = LayerNorm(hidden_size)
        self.cross_attn_token_to_image = _SamAttention(
            hidden_size,
            num_heads,
            downsample_rate=attention_downsample_rate,
        )
        self.layer_norm2 = LayerNorm(hidden_size)
        self.mlp = _SamMLPBlock(hidden_size, mlp_dim)
        self.layer_norm3 = LayerNorm(hidden_size)
        self.cross_attn_image_to_token = _SamAttention(
            hidden_size,
            num_heads,
            downsample_rate=attention_downsample_rate,
        )
        self.layer_norm4 = LayerNorm(hidden_size)

    def forward(
        self,
        op: builder.OpBuilder,
        queries: ir.Value,
        keys: ir.Value,
        query_pe: ir.Value,
        key_pe: ir.Value,
    ):
        # 1. Self-attention on queries
        if self._skip_first_layer_pe:
            q_for_sa = queries
        else:
            q_for_sa = op.Add(queries, query_pe)
        attn_out = self.self_attn(op, q_for_sa, q_for_sa, queries)
        queries = self.layer_norm1(op, op.Add(queries, attn_out))

        # 2. Cross-attention: token → image
        q = op.Add(queries, query_pe)
        k = op.Add(keys, key_pe)
        attn_out = self.cross_attn_token_to_image(op, q, k, keys)
        queries = self.layer_norm2(op, op.Add(queries, attn_out))

        # 3. MLP on queries
        mlp_out = self.mlp(op, queries)
        queries = self.layer_norm3(op, op.Add(queries, mlp_out))

        # 4. Cross-attention: image → token
        q = op.Add(keys, key_pe)
        k = op.Add(queries, query_pe)
        attn_out = self.cross_attn_image_to_token(op, q, k, queries)
        keys = self.layer_norm4(op, op.Add(keys, attn_out))

        return queries, keys


class _SamTwoWayTransformer(nn.Module):
    """Stacked two-way attention blocks + final token-to-image attention.

    Replicates ``transformers.models.sam.SamTwoWayTransformer``.
    """

    def __init__(self, config: SamConfig):
        super().__init__()
        hidden_size = config.output_channels
        num_heads = config.mask_num_attention_heads
        mlp_dim = config.mask_intermediate_size
        downsample_rate = config.attention_downsample_rate

        self.layers = nn.ModuleList(
            [
                _SamTwoWayAttentionBlock(
                    hidden_size,
                    num_heads,
                    mlp_dim,
                    attention_downsample_rate=downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
                for i in range(config.mask_num_hidden_layers)
            ]
        )
        self.final_attn_token_to_image = _SamAttention(
            hidden_size,
            num_heads,
            downsample_rate=downsample_rate,
        )
        self.layer_norm_final_attn = LayerNorm(hidden_size)

    def forward(
        self,
        op: builder.OpBuilder,
        point_embeddings: ir.Value,
        image_embeddings: ir.Value,
        image_pe: ir.Value,
    ):
        """Run two-way transformer on tokens and image features.

        Args:
            point_embeddings: (B, T, C) token queries (iou + mask + prompt).
            image_embeddings: (B, H*W, C) flattened image features.
            image_pe:         (B, H*W, C) flattened image PE.

        Returns:
            (queries, keys) updated token and image representations.
        """
        queries = point_embeddings
        keys = image_embeddings

        for layer in self.layers:
            queries, keys = layer(op, queries, keys, point_embeddings, image_pe)

        # Final token-to-image cross-attention
        q = op.Add(queries, point_embeddings)
        k = op.Add(keys, image_pe)
        attn_out = self.final_attn_token_to_image(op, q, k, keys)
        queries = self.layer_norm_final_attn(op, op.Add(queries, attn_out))

        return queries, keys


class SamMaskDecoder(nn.Module):
    """Predict segmentation masks and IoU scores from image + prompt embeddings.

    Replicates ``transformers.models.sam.SamMaskDecoder``.

    Inputs:
        image_embeddings:        (B, C, H, W)
        image_pe:                (B, C, H, W)
        sparse_prompt_embeddings: (B, N_prompt, C)
        dense_prompt_embeddings:  (B, C, H, W)

    Outputs:
        pred_masks:     (B, num_mask_tokens, 4*H, 4*W)
        iou_predictions: (B, num_mask_tokens)
    """

    def __init__(self, config: SamConfig):
        super().__init__()
        hidden_size = config.output_channels
        num_mask_tokens = config.num_multimask_outputs + 1
        self._num_mask_tokens = num_mask_tokens

        self.iou_token = Embedding(1, hidden_size)
        self.mask_tokens = Embedding(num_mask_tokens, hidden_size)

        self.transformer = _SamTwoWayTransformer(config)

        # Upscaling path: 2x ConvTranspose -> 4x spatial upscale
        self.upscale_conv1 = ConvTranspose2d(
            hidden_size,
            hidden_size // 4,
            kernel_size=2,
            stride=2,
        )
        self.upscale_layer_norm = LayerNorm(
            hidden_size // 4,
            eps=config.layer_norm_eps,
        )
        self.upscale_conv2 = ConvTranspose2d(
            hidden_size // 4,
            hidden_size // 8,
            kernel_size=2,
            stride=2,
        )

        # Per-mask hypernetwork MLPs
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                _SamFeedForward(hidden_size, hidden_size, hidden_size // 8, 3)
                for _ in range(num_mask_tokens)
            ]
        )

        # IoU prediction head
        self.iou_prediction_head = _SamFeedForward(
            hidden_size,
            config.iou_head_hidden_dim,
            num_mask_tokens,
            config.iou_head_depth,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        image_embeddings: ir.Value,
        image_pe: ir.Value,
        sparse_prompt_embeddings: ir.Value,
        dense_prompt_embeddings: ir.Value,
    ):
        # ── Prepare tokens ──
        # output_tokens: (1 + num_mask_tokens, C)
        # Use forward() to trigger parameter realization
        idx_0 = op.Constant(value_ints=[0])
        # iou_token(op, [0]) → Gather → (1, C) shape since index is 1-D
        iou_token_weight = self.iou_token(op, idx_0)  # (1, C)
        mask_range = op.Constant(value_ints=list(range(self._num_mask_tokens)))
        mask_tokens_weight = self.mask_tokens(op, mask_range)  # (num_mask_tokens, C)
        output_tokens = op.Concat(
            iou_token_weight,
            mask_tokens_weight,
            axis=0,
        )  # (1 + num_mask_tokens, C)

        # Expand to (B, 1+num_mask_tokens, C) and concat with sparse prompts
        batch_dim = op.Shape(sparse_prompt_embeddings, start=0, end=1)
        num_tokens = 1 + self._num_mask_tokens
        c_dim = op.Shape(sparse_prompt_embeddings, start=2, end=3)
        expand_shape = op.Concat(
            batch_dim,
            op.Constant(value_ints=[num_tokens]),
            c_dim,
            axis=0,
        )
        output_tokens = op.Expand(op.Unsqueeze(output_tokens, [0]), expand_shape)
        tokens = op.Concat(
            output_tokens,
            sparse_prompt_embeddings,
            axis=1,
        )  # (B, num_tokens + N_prompt, C)

        # ── Flatten image features ──
        # (B, C, H, W) → (B, H*W, C)
        src = op.Add(image_embeddings, dense_prompt_embeddings)
        src = op.Transpose(src, perm=[0, 2, 3, 1])  # (B, H, W, C)
        src_b = op.Shape(src, start=0, end=1)
        src_h = op.Shape(src, start=1, end=2)
        src_w = op.Shape(src, start=2, end=3)
        src_c = op.Shape(src, start=3, end=4)
        src_hw = op.Mul(src_h, src_w)
        src = op.Reshape(src, op.Concat(src_b, src_hw, src_c, axis=0))

        # Flatten image PE the same way
        pos_src = op.Transpose(image_pe, perm=[0, 2, 3, 1])  # (B, H, W, C)
        pos_src = op.Reshape(pos_src, op.Concat(src_b, src_hw, src_c, axis=0))

        # ── Two-way transformer ──
        hs, src = self.transformer(op, tokens, src, pos_src)

        # ── Split output tokens ──
        iou_token_out = op.Gather(
            hs,
            op.Constant(value_ints=[0]),
            axis=1,
        )  # (B, 1, C) → squeeze → (B, C)
        iou_token_out = op.Squeeze(iou_token_out, [1])

        mask_tokens_out = op.Slice(
            hs,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[1 + self._num_mask_tokens]),
            op.Constant(value_ints=[1]),
        )  # (B, num_mask_tokens, C)

        # ── Upscale image features ──
        # (B, H*W, C) → (B, C, H, W)
        src_spatial = op.Transpose(
            op.Reshape(src, op.Concat(src_b, src_h, src_w, src_c, axis=0)),
            perm=[0, 3, 1, 2],
        )  # (B, C, H, W)

        upscaled = self.upscale_conv1(op, src_spatial)  # (B, C//4, 2H, 2W)
        upscaled = _channels_first_layer_norm(op, self.upscale_layer_norm, upscaled)
        upscaled = op.Gelu(upscaled)
        upscaled = self.upscale_conv2(op, upscaled)  # (B, C//8, 4H, 4W)
        upscaled = op.Gelu(upscaled)

        # ── Hypernetwork MLPs ──
        # Each MLP maps a mask token (B, C) → (B, C//8)
        hyper_in_list = []
        for i, mlp in enumerate(self.output_hypernetworks_mlps):
            # Slice token i: (B, 1, C) → squeeze → (B, C)
            tok = op.Gather(
                mask_tokens_out,
                op.Constant(value_ints=[i]),
                axis=1,
            )
            tok = op.Squeeze(tok, [1])
            hyper_in_list.append(op.Unsqueeze(mlp(op, tok), [1]))  # (B, 1, C//8)
        hyper_in = op.Concat(*hyper_in_list, axis=1)  # (B, num_mask_tokens, C//8)

        # ── Generate masks ──
        # upscaled flat: (B, C//8, 4H*4W)
        up_b = op.Shape(upscaled, start=0, end=1)
        up_c = op.Shape(upscaled, start=1, end=2)
        up_h = op.Shape(upscaled, start=2, end=3)
        up_w = op.Shape(upscaled, start=3, end=4)
        up_hw = op.Mul(up_h, up_w)
        upscaled_flat = op.Reshape(
            upscaled,
            op.Concat(up_b, up_c, up_hw, axis=0),
        )  # (B, C//8, 4H*4W)

        # (B, num_mask_tokens, C//8) @ (B, C//8, 4H*4W) → (B, num_mask_tokens, 4H*4W)
        masks = op.MatMul(hyper_in, upscaled_flat)

        # Reshape to (B, num_mask_tokens, 4H, 4W)
        masks = op.Reshape(
            masks,
            op.Concat(
                up_b,
                op.Constant(value_ints=[self._num_mask_tokens]),
                up_h,
                up_w,
                axis=0,
            ),
        )

        # ── IoU predictions ──
        iou_predictions = self.iou_prediction_head(op, iou_token_out)  # (B, num_mask_tokens)

        return masks, iou_predictions


# ── Top-Level Model ──────────────────────────────────────────────────


class SamModel(nn.Module):
    """Segment Anything Model (SAM).

    Composed of a ViT vision encoder, a prompt encoder, and a mask
    decoder.  Exported as two ONNX models (vision_encoder and decoder)
    via :class:`~mobius.tasks.SamSegmentationTask`.

    Replicates ``transformers.SamModel``.
    """

    default_task: str = "sam-segmentation"
    category: str = "Image Segmentation"
    config_class: type = SamConfig

    def __init__(self, config: SamConfig):
        super().__init__()
        self.config = config
        self.shared_image_embedding = _SamPositionalEmbedding(config)
        self.vision_encoder = SamVisionEncoder(config)
        self.prompt_encoder = _SamPromptEncoder(config)
        self.mask_decoder = SamMaskDecoder(config)

    def preprocess_weights(
        self,
        state_dict: dict[str, object],
    ) -> dict[str, object]:
        """Drop windowed-attention relative position biases (not used)."""
        new: dict[str, object] = {}
        for key, value in state_dict.items():
            if "rel_pos_h" in key or "rel_pos_w" in key:
                continue
            new[key] = value
        return new
