# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NVIDIA Cosmos3-Edge SigLIP2 vision tower and pixel-shuffle merger projector.

Replicates ``Cosmos3EdgeVisionModel`` / ``Cosmos3EdgePatchMerger`` from
``transformers.models.cosmos3_edge.modular_cosmos3_edge`` (cross-checked
against ``vllm/model_executor/models/cosmos3_edge.py``).

The tower is a **variable-resolution, packed** SigLIP2 encoder, not a
fixed-square ViT:

- ``pixel_values`` arrives **already patchified** as
  ``[total_patches, patch_size**2 * channels * temporal_patch_size]``.  The
  processor emits patches in **block-major** order (each ``merge x merge``
  block of adjacent patches is contiguous) and stores the values inside a
  patch as ``(patch_h, patch_w, channel)`` — channel-**last**.  The checkpoint
  therefore ships ``embeddings.patch_embedding`` as an ``nn.Linear``
  ``[hidden, patch_dim]``, not a ``Conv2d`` kernel.
- The learned ``num_patches`` (16x16) position table is bilinearly resampled
  (antialiased, ``align_corners=False``) to the item's ``(grid_h, grid_w)``
  and then reordered into the same block-major layout.
- Attention is non-causal and runs **independently per frame**
  (``cu_seqlens`` delimits every frame), which for a single packed item with a
  shared ``(grid_h, grid_w)`` is exactly batched attention over ``grid_t``
  equal-length sequences.

The merger projector groups ``spatial_merge_size**2`` consecutive
(block-major) patches, applies the pre-shuffle ``LayerNorm``
(``use_postshuffle_norm=false`` in the public checkpoint), then
``linear_fc1 → GELU → linear_fc2``.
"""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._vision import VisionEncoder, VisionLayerNorm, _VisionLinear

__all__ = [
    "Cosmos3EdgePatchMerger",
    "Cosmos3EdgeVisionEmbeddings",
    "Cosmos3EdgeVisionTower",
]


def _resample_weights(op: OpBuilder, source: int, target: ir.Value) -> ir.Value:
    """Build the 1-D antialiased-bilinear resample matrix ``[target, source]``.

    Reproduces PyTorch's ``F.interpolate(..., mode="bilinear",
    align_corners=False, antialias=True)`` separable filter:

    - ``scale = source / target`` (``align_corners=False`` area scale),
    - ``support = max(scale, 1)`` — antialiasing only widens the triangle
      filter when *downsampling*,
    - ``w[i, j] = max(0, 1 - |j + 0.5 - scale * (i + 0.5)| / support)``,
      renormalised over the ``source`` window.

    ONNX ``Resize(antialias=1)`` is *not* used: onnxruntime's antialias filter
    does not match PyTorch's when downsampling, and CUDA/DML do not implement
    it at all.  An explicit ``[target, source]`` matrix is exact and portable.
    """
    source_f = float(source)
    # centers[i] = (i + 0.5) * source / target
    target_f = op.Cast(target, to=ir.DataType.FLOAT)
    scale = op.Div(op.Constant(value_float=source_f), target_f)
    positions = op.Cast(
        op.Range(
            op.Constant(value_int=0),
            op.Reshape(target, op.Constant(value_ints=[])),
            op.Constant(value_int=1),
        ),
        to=ir.DataType.FLOAT,
    )
    centers = op.Mul(op.Add(positions, op.Constant(value_float=0.5)), scale)
    support = op.Max(scale, op.Constant(value_float=1.0))

    # Source pixel centres: [0.5, 1.5, ..., source - 0.5]
    source_centers = op.Constant(
        value=ir.tensor(np.arange(source, dtype=np.float32) + 0.5, name="source_centers")
    )
    distance = op.Sub(op.Unsqueeze(source_centers, [0]), op.Unsqueeze(centers, [-1]))
    weights = op.Sub(
        op.Constant(value_float=1.0),
        op.Div(op.Abs(distance), op.Unsqueeze(support, [0])),
    )
    weights = op.Relu(weights)  # triangle filter: max(0, 1 - |x| / support)
    total = op.ReduceSum(weights, op.Constant(value_ints=[-1]), keepdims=1)
    return op.Div(weights, total)


class Cosmos3EdgeVisionEmbeddings(nn.Module):
    """Packed patch embedding + resampled block-major position embedding.

    HF reference: ``Cosmos3EdgeVisionEmbeddings`` (a ``Siglip2VisionEmbeddings``
    subclass whose ``resize_positional_embeddings`` additionally reorders the
    resampled grid into the processor's block-major 2x2 layout).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        patch_size: int,
        num_channels: int,
        num_patches: int,
        temporal_patch_size: int = 1,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        grid = math.isqrt(num_patches)
        if grid * grid != num_patches:
            raise ValueError(
                f"num_patches must form a square reference grid, got {num_patches}"
            )
        self.position_embedding_size = grid
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size
        self.patch_dim = patch_size * patch_size * num_channels * temporal_patch_size
        # nn.Linear over flattened (patch_h, patch_w, channel) values.
        self.patch_embedding = _VisionLinear(self.patch_dim, hidden_size)
        self.position_embedding = nn.Parameter(
            [num_patches, hidden_size], name="position_embedding.weight"
        )

    def _resized_position_embedding(
        self, op: OpBuilder, grid_h: ir.Value, grid_w: ir.Value
    ) -> ir.Value:
        """Resample the learned grid to ``(grid_h, grid_w)``, block-major.

        ``[g, g, D]`` -> bilinear/antialias -> ``[H, W, D]`` -> block-major
        ``[H*W, D]`` so element ``k`` matches the processor's patch ``k``.
        """
        grid = self.position_embedding_size
        dim = self.hidden_size
        merge = self.spatial_merge_size

        table = op.Reshape(self.position_embedding, op.Constant(value_ints=[grid, grid * dim]))
        # Height pass: [H, g] @ [g, g*D] -> [H, g, D]
        height_weights = _resample_weights(op, grid, grid_h)
        table = op.MatMul(op.CastLike(height_weights, table), table)
        table = op.Reshape(
            table,
            op.Concat(
                op.Reshape(grid_h, op.Constant(value_ints=[1])),
                op.Constant(value_ints=[grid, dim]),
                axis=0,
            ),
        )
        # Width pass: [W, g] @ [g, H*D] -> [W, H, D]
        table = op.Transpose(table, perm=[1, 0, 2])  # [g, H, D]
        table = op.Reshape(table, op.Constant(value_ints=[grid, -1]))
        width_weights = _resample_weights(op, grid, grid_w)
        table = op.MatMul(op.CastLike(width_weights, table), table)
        table = op.Reshape(
            table,
            op.Concat(
                op.Reshape(grid_w, op.Constant(value_ints=[1])),
                op.Constant(value_ints=[-1, dim]),
                axis=0,
            ),
        )
        table = op.Transpose(table, perm=[1, 0, 2])  # [H, W, D]

        # Block-major reorder: (H, W) -> (H/m, m, W/m, m) -> (H/m, W/m, m, m)
        merge_const = op.Constant(value_ints=[merge])
        blocks_h = op.Div(op.Reshape(grid_h, op.Constant(value_ints=[1])), merge_const)
        blocks_w = op.Div(op.Reshape(grid_w, op.Constant(value_ints=[1])), merge_const)
        table = op.Reshape(
            table,
            op.Concat(
                blocks_h,
                merge_const,
                blocks_w,
                merge_const,
                op.Constant(value_ints=[dim]),
                axis=0,
            ),
        )
        table = op.Transpose(table, perm=[0, 2, 1, 3, 4])
        return op.Reshape(table, op.Constant(value_ints=[-1, dim]))

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        grid_t: ir.Value,
        grid_h: ir.Value,
        grid_w: ir.Value,
    ):
        # pixel_values: [T*H*W, patch_dim] -> [T*H*W, hidden]
        patch_embeds = self.patch_embedding(op, pixel_values)
        # Split frames out so the per-frame position grid broadcasts over T,
        # which is what the reference's ``resized.repeat(temporal, 1)`` does.
        frame_shape = op.Concat(
            op.Reshape(grid_t, op.Constant(value_ints=[1])),
            op.Reshape(op.Mul(grid_h, grid_w), op.Constant(value_ints=[1])),
            op.Constant(value_ints=[self.hidden_size]),
            axis=0,
        )
        patch_embeds = op.Reshape(patch_embeds, frame_shape)
        position_embeds = self._resized_position_embedding(op, grid_h, grid_w)
        # [T, H*W, D] + [H*W, D] -> [T, H*W, D]
        return op.Add(patch_embeds, position_embeds)


class Cosmos3EdgeVisionTower(nn.Module):
    """SigLIP2 encoder stack over packed, per-frame attention sequences.

    ``pixel_values [T*H*W, patch_dim]`` -> ``[T*H*W, vision_hidden]``.  The
    patches are reshaped to ``[T, H*W, D]`` so the ONNX ``Attention`` op sees
    ``T`` independent sequences — the exact semantics of the reference's
    per-frame ``cu_seqlens`` (every frame of a packed item shares one
    ``(H, W)`` grid, so all sequences have equal length and no mask is needed).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        patch_size: int,
        num_channels: int,
        num_patches: int,
        norm_eps: float = 1e-6,
        temporal_patch_size: int = 1,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embeddings = Cosmos3EdgeVisionEmbeddings(
            hidden_size=hidden_size,
            patch_size=patch_size,
            num_channels=num_channels,
            num_patches=num_patches,
            temporal_patch_size=temporal_patch_size,
            spatial_merge_size=spatial_merge_size,
        )
        self.encoder = VisionEncoder(
            num_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_attention_heads,
            norm_eps=norm_eps,
        )
        self.post_layernorm = VisionLayerNorm(hidden_size, eps=norm_eps)

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        grid_t: ir.Value,
        grid_h: ir.Value,
        grid_w: ir.Value,
    ):
        # [T, H*W, D]: one non-causal attention sequence per frame.
        hidden_states = self.embeddings(op, pixel_values, grid_t, grid_h, grid_w)
        hidden_states = self.encoder(op, hidden_states)
        hidden_states = self.post_layernorm(op, hidden_states)
        # Back to packed rows: [T, H*W, D] -> [T*H*W, D]
        return op.Reshape(hidden_states, op.Constant(value_ints=[-1, self.hidden_size]))


class Cosmos3EdgePatchMerger(nn.Module):
    """Cosmos3-Edge pixel-shuffle merger projector.

    ``LayerNorm -> spatial merge -> linear_fc1 -> GELU -> linear_fc2``.

    The processor already emits patches in block-major order, so merging is a
    plain reshape of ``spatial_merge_size**2`` *consecutive* rows — matching
    ``Cosmos3EdgePatchMerger.forward``'s
    ``x.reshape(-1, spatial_merge_size**2, input_hidden_size)``.

    ``use_postshuffle_norm`` selects whether the ``LayerNorm`` normalises the
    raw ``vision_hidden_size`` features (``false``, the public checkpoint) or
    the merged ``spatial_merge_size**2 * vision_hidden_size`` vector.

    HF weights (``model.projector.*``): ``norm.{weight,bias}``,
    ``linear_fc1.{weight,bias}``, ``linear_fc2.{weight,bias}``.
    """

    def __init__(
        self,
        *,
        vision_hidden_size: int,
        text_hidden_size: int,
        intermediate_size: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if spatial_merge_size <= 0:
            raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size}")
        self.spatial_merge_size = spatial_merge_size
        self.vision_hidden_size = vision_hidden_size
        self.merged_size = vision_hidden_size * spatial_merge_size * spatial_merge_size
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = VisionLayerNorm(
            self.merged_size if use_postshuffle_norm else vision_hidden_size,
            eps=norm_eps,
        )
        self.linear_fc1 = _VisionLinear(self.merged_size, intermediate_size)
        self.linear_fc2 = _VisionLinear(intermediate_size, text_hidden_size)

    def forward(self, op: OpBuilder, vision_features: ir.Value):
        # vision_features: [total_patches, vision_hidden]
        merged_shape = op.Constant(value_ints=[-1, self.merged_size])
        if self.use_postshuffle_norm:
            hidden = self.norm(op, op.Reshape(vision_features, merged_shape))
        else:
            grouped = op.Reshape(
                vision_features,
                op.Constant(
                    value_ints=[
                        -1,
                        self.spatial_merge_size * self.spatial_merge_size,
                        self.vision_hidden_size,
                    ]
                ),
            )
            hidden = op.Reshape(self.norm(op, grouped), merged_shape)
        hidden = self.linear_fc1(op, hidden)
        hidden = op.Gelu(hidden)  # nn.GELU() -> exact erf GELU
        return self.linear_fc2(op, hidden)
