# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SigLIP2 NaFlex vision tower (variable aspect ratio, pre-patchified input).

Replicates HuggingFace ``Siglip2VisionModel`` for the NaFlex checkpoints
(``google/siglip2-*-naflex``).  Unlike fixed-resolution SigLIP towers, the
image processor emits *already patchified* pixels, so the patch embedding is
a plain ``Linear`` over ``num_channels * patch_size**2`` values rather than a
strided ``Conv2d``.  Each image keeps its own ``(height, width)`` patch grid;
images are right-padded to a common ``max_num_patches`` and a per-image
padding mask keeps the bidirectional attention from crossing image
boundaries.

Inputs
    pixel_values: ``(num_images, max_num_patches, C * patch**2)``, model dtype.
    pixel_attention_mask: ``(num_images, max_num_patches)``, 1 = real patch.
    spatial_shapes: ``(num_images, 2)`` INT64 ``[grid_h, grid_w]`` per image.

Output
    ``(num_images, max_num_patches, hidden_size)`` — post-layernorm hidden
    states, still padded.  Callers unpad using ``pixel_attention_mask``.

The learned position table is a square ``sqrt(num_patches)`` grid that is
bilinearly resampled to each image's ``(grid_h, grid_w)``.  HuggingFace uses
``F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)``;
the ONNX equivalent is ``Resize`` with ``coordinate_transformation_mode=
"half_pixel"``, ``antialias=1`` and ``exclude_outside=1`` (PyTorch normalises
the antialias filter over the clipped support, which is what
``exclude_outside=1`` expresses).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._common import LayerNorm, Linear
from mobius.components._scan_utils import create_body_graph, rename_subgraph_values
from mobius.components._vision import VisionEncoder

if TYPE_CHECKING:
    from mobius._configs import VisionConfig


class Siglip2NaFlexVisionEmbeddings(nn.Module):
    """Linear patch embedding plus per-image resampled position embeddings."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        num_patches: int,
        in_channels: int = 3,
    ):
        super().__init__()
        position_embedding_size = math.isqrt(num_patches)
        if position_embedding_size * position_embedding_size != num_patches:
            raise ValueError(
                f"SigLIP2 NaFlex requires a square position grid, got "
                f"num_patches={num_patches}"
            )
        self.hidden_size = hidden_size
        self.position_embedding_size = position_embedding_size
        # Patches arrive pre-flattened from the image processor, so the
        # "convolution" degenerates to a single Linear over the patch pixels.
        self.patch_embedding = Linear(
            in_channels * patch_size * patch_size,
            hidden_size,
            bias=True,
        )
        # The position table is consumed by Resize (not Gather), so it is a
        # plain Parameter rather than an Embedding submodule; an unused
        # submodule would never realize its initializer. The explicit name
        # keeps the HuggingFace ``position_embedding.weight`` key.
        self.position_embedding = nn.Parameter(
            [num_patches, hidden_size],
            name="position_embedding.weight",
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        spatial_shapes: ir.Value,
    ) -> ir.Value:
        # (N, P, C*k*k) -> (N, P, D)
        patch_embeds = self.patch_embedding(op, pixel_values)
        max_patches = op.Shape(pixel_values, start=1, end=2)  # (1,)
        position_embeds = self._resize_position_embedding(
            op,
            op.Cast(spatial_shapes, to=ir.DataType.INT64),
            max_patches,
        )
        return op.Add(patch_embeds, op.CastLike(position_embeds, patch_embeds))

    def _resize_position_embedding(
        self,
        op: OpBuilder,
        spatial_shapes: ir.Value,
        max_patches: ir.Value,
    ) -> ir.Value:
        """Resample the square position table onto each image's patch grid.

        Returns ``(num_images, max_num_patches, hidden_size)`` in float32.
        Iterating with ``Scan`` is required because ``Resize`` takes one
        ``sizes`` vector per call while every image has its own grid.
        """
        side = self.position_embedding_size
        hidden_size = self.hidden_size

        # (side*side, D) -> (1, D, side, side): Resize consumes NCHW.
        # Interpolate in float32; the stored table may be fp16/bf16 and the
        # antialias filter is numerically sensitive.
        table = op.Cast(self.position_embedding, to=ir.DataType.FLOAT)
        table = op.Reshape(table, op.Constant(value_ints=[side, side, hidden_size]))
        table = op.Transpose(table, perm=[2, 0, 1])
        table = op.Unsqueeze(table, [0])

        body_shape = ir.Value(
            name="body_spatial_shape",
            shape=ir.Shape([2]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        body_graph, body_builder = create_body_graph([], [body_shape])
        body_op = body_builder.op

        grid_h = body_op.Slice(body_shape, [0], [1])  # (1,)
        grid_w = body_op.Slice(body_shape, [1], [2])  # (1,)
        sizes = body_op.Concat(
            body_op.Constant(value_ints=[1, hidden_size]),
            grid_h,
            grid_w,
            axis=0,
        )
        # ``antialias=1`` + ``exclude_outside=1`` + ``half_pixel`` reproduces
        # torch ``F.interpolate(bilinear, align_corners=False, antialias=True)``.
        resized = body_op.Resize(
            table,
            None,  # roi
            None,  # scales
            sizes,
            mode="linear",
            coordinate_transformation_mode="half_pixel",
            antialias=1,
            exclude_outside=1,
        )
        # (1, D, h, w) -> (D, h*w) -> (h*w, D)
        resized = body_op.Reshape(resized, body_op.Constant(value_ints=[hidden_size, -1]))
        resized = body_op.Transpose(resized, perm=[1, 0])

        # Right-pad to max_num_patches. HuggingFace fills the padding with the
        # first resampled row; padded positions are masked out of attention and
        # dropped downstream, so only the shape matters, but matching upstream
        # keeps the intermediate tensors comparable.
        pad_length = body_op.Sub(max_patches, body_op.Mul(grid_h, grid_w))
        first_row = body_op.Slice(resized, [0], [1], axes=[0])
        padding = body_op.Tile(
            first_row,
            body_op.Concat(pad_length, body_op.Constant(value_ints=[1]), axis=0),
        )
        padded = body_op.Concat(resized, padding, axis=0)
        padded.name = "resized_position_embedding"
        body_graph.outputs.append(padded)
        rename_subgraph_values(body_graph, "siglip2_pos_")

        return op.Scan(
            spatial_shapes,
            body=body_graph,
            num_scan_inputs=1,
            _outputs=1,
        )  # (N, P, D)


def siglip2_naflex_attention_mask(
    op: OpBuilder,
    pixel_attention_mask: ir.Value,
) -> ir.Value:
    """Expand a per-patch padding mask into a bidirectional attention mask.

    Returns a ``(num_images, 1, max_patches, max_patches)`` bool tensor where
    ``True`` means "attend".  The ONNX Attention op requires the query axis to
    match ``q_sequence_length`` (a broadcast ``1`` is rejected), hence the
    explicit expansion.  A bool mask (rather than an additive float bias)
    keeps the ORT Flash-Attention kernel eligible.
    """
    valid = op.Cast(pixel_attention_mask, to=ir.DataType.BOOL)  # (N, P)
    key_mask = op.Unsqueeze(valid, [1, 2])  # (N, 1, 1, P)
    num_images = op.Shape(pixel_attention_mask, start=0, end=1)
    num_patches = op.Shape(pixel_attention_mask, start=1, end=2)
    return op.Expand(
        key_mask,
        op.Concat(
            num_images,
            op.Constant(value_ints=[1]),
            num_patches,
            num_patches,
            axis=0,
        ),
    )


class Siglip2NaFlexVisionModel(nn.Module):
    """SigLIP2 NaFlex tower: embeddings, encoder stack, and post-layernorm.

    Attribute names mirror HuggingFace ``Siglip2VisionModel`` so that the
    checkpoint keys ``embeddings.* / encoder.layers.N.* / post_layernorm.*``
    map across with no renaming beyond the MLP ``fc1``/``fc2`` naming that
    :class:`~mobius.components.VisionEncoderLayer` normalises to
    ``up_proj``/``down_proj``.
    """

    def __init__(self, vision_config: VisionConfig):
        super().__init__()
        assert vision_config.hidden_size is not None
        assert vision_config.intermediate_size is not None
        assert vision_config.num_hidden_layers is not None
        assert vision_config.num_attention_heads is not None
        assert vision_config.patch_size is not None
        num_patches = vision_config.num_position_embeddings
        assert num_patches is not None, (
            "SigLIP2 NaFlex needs the learned position table size "
            "(HF vision_config.num_patches)"
        )

        self.embeddings = Siglip2NaFlexVisionEmbeddings(
            hidden_size=vision_config.hidden_size,
            patch_size=vision_config.patch_size,
            num_patches=num_patches,
            in_channels=vision_config.in_channels,
        )
        self.encoder = VisionEncoder(
            num_layers=vision_config.num_hidden_layers,
            hidden_size=vision_config.hidden_size,
            intermediate_size=vision_config.intermediate_size,
            num_heads=vision_config.num_attention_heads,
            norm_eps=vision_config.norm_eps,
        )
        self.post_layernorm = LayerNorm(
            vision_config.hidden_size,
            eps=vision_config.norm_eps,
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        pixel_attention_mask: ir.Value,
        spatial_shapes: ir.Value,
    ) -> ir.Value:
        # (N, P, C*k*k) -> (N, P, D)
        hidden_states = self.embeddings(op, pixel_values, spatial_shapes)
        attention_mask = siglip2_naflex_attention_mask(op, pixel_attention_mask)
        hidden_states = self.encoder(op, hidden_states, attention_mask)
        return self.post_layernorm(op, hidden_states)
