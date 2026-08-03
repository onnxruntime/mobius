# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Multimodal components for bridging vision and text models.

Provides modules for connecting vision encoder outputs to text model inputs:

Projectors (vision → text embedding space):
- ``Gemma3MultiModalProjector``: AvgPool2d → RMSNorm → MatMul (Gemma3)
- ``MLPMultiModalProjector``: Linear → Act → Linear (LLaVA, Phi4MM)
- ``LinearMultiModalProjector``: Single Linear (PaliGemma)

Mixer:
- ``InputMixer``: Merges vision embeddings into text embeddings at
  placeholder token positions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius.components._common import Linear
from mobius.components._rms_norm import RMSNorm
from mobius.components._vision import VisionLayerNorm

if TYPE_CHECKING:
    import onnx_ir as ir


class Gemma3MultiModalProjector(nn.Module):
    """AvgPool2d → RMSNorm → MatMul projector (Gemma3).

    Reshapes vision features into a 2-D spatial grid, applies 2-D average
    pooling to reduce the number of tokens, normalises with RMSNorm, then
    projects via a learnable matrix multiplication.

    HF reference: ``Gemma3MultiModalProjector`` in
    ``transformers.models.gemma3.modeling_gemma3``.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        patches_per_image: int,
        tokens_per_image: int,
        norm: RMSNorm | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.mm_soft_emb_norm = norm or RMSNorm(vision_hidden_size, eps=eps)
        self.patches_per_image = patches_per_image
        tokens_per_side = int(tokens_per_image**0.5)
        self.pool_kernel = patches_per_image // tokens_per_side
        self.mm_input_projection_weight = nn.Parameter([vision_hidden_size, text_hidden_size])

    def forward(self, op: OpBuilder, vision_features: ir.Value):
        # vision_features: [batch, num_patches, vision_hidden_size]
        batch_size = op.Shape(vision_features, start=0, end=1)
        hidden_size = op.Shape(vision_features, start=2, end=3)
        patches = op.Constant(value_ints=[self.patches_per_image])

        # Transpose to [batch, hidden, num_patches]
        hidden = op.Transpose(vision_features, perm=[0, 2, 1])
        # Reshape to [batch, hidden, patches_per_image, patches_per_image]
        new_shape = op.Concat(batch_size, hidden_size, patches, patches, axis=0)
        hidden = op.Reshape(hidden, new_shape)

        # 2D average pooling
        hidden = op.AveragePool(
            hidden,
            kernel_shape=[self.pool_kernel, self.pool_kernel],
            strides=[self.pool_kernel, self.pool_kernel],
        )

        # Flatten spatial dims: [batch, hidden, h, w] → [batch, hidden, h*w]
        minus_one = op.Constant(value_ints=[-1])
        flat_shape = op.Concat(batch_size, hidden_size, minus_one, axis=0)
        hidden = op.Reshape(hidden, flat_shape)
        # Transpose to [batch, tokens, hidden]
        hidden = op.Transpose(hidden, perm=[0, 2, 1])

        # RMSNorm after pooling
        hidden = self.mm_soft_emb_norm(op, hidden)

        # Linear projection: vision_hidden → text_hidden
        projected = op.MatMul(hidden, self.mm_input_projection_weight)
        return projected


class MLPMultiModalProjector(nn.Module):
    """Two-layer MLP projector: Linear → GELU → Linear.

    The most common projector pattern, used by LLaVA, LLaVA-NeXT, VipLLaVA,
    Phi-4-multimodal and others.

    HF reference: ``LlavaMultiModalProjector`` in
    ``transformers.models.llava.modeling_llava``.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.linear_1 = Linear(vision_hidden_size, text_hidden_size, bias=bias)
        self.linear_2 = Linear(text_hidden_size, text_hidden_size, bias=bias)

    def forward(self, op: OpBuilder, vision_features: ir.Value):
        # vision_features: [batch, num_patches, vision_hidden_size]
        hidden = self.linear_1(op, vision_features)
        hidden = op.Gelu(hidden)
        hidden = self.linear_2(op, hidden)
        return hidden


class Cosmos3EdgeMultiModalProjector(nn.Module):
    """Cosmos3-Edge pixel-shuffle merger projector.

    ``LayerNorm → spatial 2x2 pixel-shuffle → Linear(fc1) → GELU → Linear(fc2)``

    The SigLIP vision encoder emits a fixed ``grid x grid`` patch grid
    (``num_patches`` patches, e.g. 16x16 = 256). ``use_postshuffle_norm=false``
    means the ``LayerNorm`` is applied on the raw ``vision_hidden_size`` (1152)
    features **before** the spatial merge. The merge concatenates each
    ``spatial_merge_size x spatial_merge_size`` block of adjacent patches into a
    single ``spatial_merge_size**2 * vision_hidden_size`` (4608) vector, which
    ``linear_fc1`` maps to ``intermediate_size`` (11520) and ``linear_fc2`` maps
    to ``text_hidden_size`` (2048).

    HF weights (``model.projector.*``):
    - ``norm.{weight,bias}`` (pre-shuffle LayerNorm)
    - ``linear_fc1.{weight,bias}``
    - ``linear_fc2.{weight,bias}``
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        intermediate_size: int,
        grid_size: int,
        spatial_merge_size: int = 2,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self._grid = grid_size
        self._ms = spatial_merge_size
        self._vision_hidden = vision_hidden_size
        merged_dim = vision_hidden_size * spatial_merge_size * spatial_merge_size
        # Pre-shuffle LayerNorm over the raw vision hidden size.
        self.norm = VisionLayerNorm(vision_hidden_size, eps=norm_eps)
        self.linear_fc1 = Linear(merged_dim, intermediate_size, bias=True)
        self.linear_fc2 = Linear(intermediate_size, text_hidden_size, bias=True)

    def forward(self, op: OpBuilder, vision_features: ir.Value):
        # vision_features: [batch, grid*grid, vision_hidden]
        ms = self._ms
        g = self._grid
        gm = g // ms
        d = self._vision_hidden

        # Pre-shuffle LayerNorm (use_postshuffle_norm=false).
        x = self.norm(op, vision_features)

        batch = op.Shape(vision_features, start=0, end=1)  # dynamic [1]

        # [B, g*g, D] -> [B, g/ms, ms, g/ms, ms, D]
        shape_6d = op.Concat(
            batch,
            op.Constant(value_ints=[gm, ms, gm, ms, d]),
            axis=0,
        )
        x = op.Reshape(x, shape_6d)
        # Group hidden dim outermost per merged block (HF F.unfold ordering):
        # [B, g/ms, ms, g/ms, ms, D] -> [B, g/ms, g/ms, D, ms, ms]
        x = op.Transpose(x, perm=[0, 1, 3, 5, 2, 4])
        # Flatten to [B, (g/ms)^2, D*ms*ms]
        shape_3d = op.Concat(
            batch,
            op.Constant(value_ints=[gm * gm, d * ms * ms]),
            axis=0,
        )
        x = op.Reshape(x, shape_3d)

        x = self.linear_fc1(op, x)
        x = op.Gelu(x)
        x = self.linear_fc2(op, x)
        return x


class LinearMultiModalProjector(nn.Module):
    """Single linear projection (PaliGemma, Qwen2-Audio).

    The simplest projector — a single ``nn.Linear`` mapping vision features
    directly to the text hidden dimension.

    HF reference: ``PaliGemmaMultiModalProjector`` in
    ``transformers.models.paligemma.modeling_paligemma``.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = Linear(vision_hidden_size, text_hidden_size, bias=bias)

    def forward(self, op: OpBuilder, vision_features: ir.Value):
        return self.linear(op, vision_features)


class InputMixer(nn.Module):
    """Merges vision embeddings into text embeddings at placeholder positions.

    Replaces image_token_id positions in text embeddings with projected
    vision embeddings using scatter-like operations.
    """

    def __init__(self, image_token_id: int):
        super().__init__()
        self.image_token_id = image_token_id

    def forward(
        self,
        op: OpBuilder,
        text_embeddings: ir.Value,
        vision_embeddings: ir.Value,
        input_ids: ir.Value,
    ):
        # text_embeddings: [batch, text_seq, hidden]
        # vision_embeddings: [batch, vision_seq, hidden]
        # input_ids: [batch, text_seq]

        # Create mask where input_ids == image_token_id
        token_id = op.Constant(value_int=self.image_token_id)
        mask = op.Equal(input_ids, token_id)  # [batch, text_seq]
        # Expand mask to [batch, text_seq, 1] for broadcasting
        mask_expanded = op.Unsqueeze(mask, [-1])

        # Pad vision_embeddings with a single zero row so GatherElements
        # always has at least one row to index into.  When the modality
        # is absent (vision_seq == 0), all indices clamp to 0 and the
        # mask ensures the padding is never selected.
        batch_dim = op.Shape(vision_embeddings, start=0, end=1)
        hidden_dim = op.Shape(vision_embeddings, start=2, end=3)
        pad_shape = op.Concat(
            batch_dim,
            op.Constant(value_ints=[1]),
            hidden_dim,
            axis=0,
        )
        zero_pad = op.Expand(
            op.CastLike(0.0, vision_embeddings),
            pad_shape,
        )
        # [batch, vision_seq + 1, hidden]
        vision_padded = op.Concat(vision_embeddings, zero_pad, axis=1)

        # Create full-size vision tensor at text positions
        # Use cumulative sum of mask to index into vision_embeddings
        mask_int = op.Cast(mask, to=7)  # INT64
        # cumsum along seq dim gives position indices into vision embeddings
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        # Subtract 1 for 0-based indexing, clamp to 0
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        # Gather vision embeddings at computed indices
        # indices: [batch, text_seq], vision_padded: [batch, vision_seq+1, hidden]
        indices_3d = op.Unsqueeze(indices, [-1])
        expand_shape = op.Concat(
            op.Constant(value_ints=[1, 1]),
            hidden_dim,
            axis=0,
        )
        indices_expanded = op.Expand(indices_3d, expand_shape)
        scattered_vision = op.GatherElements(vision_padded, indices_expanded, axis=1)

        # Mix: where mask, use vision; else use text
        mixed = op.Where(mask_expanded, scattered_vision, text_embeddings)
        return mixed
