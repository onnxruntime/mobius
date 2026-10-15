# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""C-RADIOv4 ``RadioModel`` backbone with external input conditioning.

Matches ``modeling_radio.py`` at HuggingFace revision
``c6cc2e8ffa353403213d6b9b18aaf8eb684d705a``: normalized BCHW images (or
temporally packed channels) produce flattened CLS summaries and spatial features.
Parameter names retain the released checkpoint's ``patch_generator`` / ``blocks``
layout, before the HuggingFace loader's conversion to ``embeddings`` / ``encoder``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._common import INT64_MAX, Linear
from mobius.components._radio_vision import RadioBlock, _RadioClsToken

if TYPE_CHECKING:
    from collections.abc import Sequence


class _RadioV4PatchGenerator(nn.Module):
    def __init__(
        self,
        patch_size: int,
        max_grid_size: int,
        hidden_size: int,
        num_prefix_tokens: int,
        video_temporal_patch_size: int | None,
    ):
        super().__init__()
        self.embedder = Linear(3 * patch_size**2, hidden_size, bias=False)
        self.video_embedder = (
            Linear(video_temporal_patch_size * 3 * patch_size**2, hidden_size, bias=False)
            if video_temporal_patch_size is not None
            else None
        )
        self.pos_embed = nn.Parameter([1, max_grid_size**2, hidden_size])
        self.cls_token = _RadioClsToken(num_prefix_tokens, hidden_size)
        self._patch_size = patch_size
        self._max_grid_size = max_grid_size
        self._hidden_size = hidden_size
        self._video_temporal_patch_size = video_temporal_patch_size

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        use_video_patch_projection: bool = False,
    ) -> ir.Value:
        if not isinstance(use_video_patch_projection, bool):
            raise TypeError("use_video_patch_projection must be an export-time bool")
        projection = self.video_embedder if use_video_patch_projection else self.embedder
        if projection is None:
            raise ValueError("video projection requires video_temporal_patch_size")
        channels = (
            3 * (self._video_temporal_patch_size or 1) if use_video_patch_projection else 3
        )
        if pixel_values.shape is not None:
            if len(pixel_values.shape) != 4:
                raise ValueError("RADIOv4 expects BCHW input")
            channel_dim = pixel_values.shape[1]
            if isinstance(channel_dim, int) and channel_dim != channels:
                raise ValueError(f"RADIOv4 expects {channels} input channels")
            for dim in pixel_values.shape[2:]:
                if isinstance(dim, int) and (dim <= 0 or dim % self._patch_size):
                    raise ValueError(
                        "RADIOv4 image dimensions must be positive patch multiples"
                    )

        batch = op.Shape(pixel_values, start=0, end=1)
        grid = op.Div(op.Shape(pixel_values, start=2, end=4), [self._patch_size])
        rows = op.Slice(grid, [0], [1])
        cols = op.Slice(grid, [1], [2])
        # Keep source reshape semantics: non-divisible dynamic H/W must fail,
        # rather than silently dropping border pixels as a strided Conv would.
        patches = op.Reshape(
            pixel_values,
            op.Concat(
                batch, [channels], rows, [self._patch_size], cols, [self._patch_size], axis=0
            ),
        )  # (B, C*T, rows, patch_h, cols, patch_w)
        patches = op.Transpose(patches, perm=[0, 2, 4, 1, 3, 5])
        patches = op.Reshape(
            patches,
            op.Concat(batch, [-1, channels * self._patch_size**2], axis=0),
        )
        patches = projection(op, patches)  # (B, rows*cols, hidden)

        # CPE resizes the learned square in float32 to max(rows, cols), then
        # crops its top-left corner. Direct rectangular resize is not equivalent.
        pos = op.Reshape(
            self.pos_embed,
            [1, self._max_grid_size, self._max_grid_size, self._hidden_size],
        )
        pos = op.Transpose(pos, perm=[0, 3, 1, 2])
        max_dim = op.ReduceMax(grid, keepdims=True)
        pos = op.Resize(
            op.Cast(pos, to=ir.DataType.FLOAT),
            None,
            None,
            op.Concat(max_dim, max_dim, axis=0),
            axes=[2, 3],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
        )
        pos = op.CastLike(pos, patches)
        pos = op.Slice(pos, [0, 0], grid, [2, 3])
        pos = op.Transpose(pos, perm=[0, 2, 3, 1])
        pos = op.Reshape(pos, [1, -1, self._hidden_size])
        return self.cls_token(op, op.Add(patches, pos))


class RadioV4VisionModel(nn.Module):
    """C-RADIOv4 GELU backbone, without normalization or a downstream projector.

    ``forward(op, pixel_values, use_video_patch_projection=False)`` accepts
    normalized ``(B, 3, H, W)`` images. The video path accepts
    ``(B, 3 * video_temporal_patch_size, H, W)`` with consecutive frames packed
    into channels, as in the source wrapper. Frame padding/grouping is external.
    H/W may be symbolic but must be positive multiples of ``patch_size``.

    Returns ``(summary, features)`` with shapes ``(B, len(summary_idxs)*hidden)``
    and ``(B, (H/patch_size)*(W/patch_size), hidden)``. There is no final norm.
    Only the checkpoint's biased QKV, exact GELU, identity LayerScale architecture
    is supported; SwiGLU and learned LayerScale variants are not represented.
    Image normalization is deliberately external, matching
    ``RadioModel.make_preprocessor_external()`` in the source wrapper.
    """

    def __init__(
        self,
        *,
        patch_size: int,
        max_grid_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        norm_eps: float = 1e-6,
        video_temporal_patch_size: int | None = None,
        num_cls_tokens: int = 3,
        num_registers: int = 7,
        summary_idxs: Sequence[int] = (0, 1),
    ):
        super().__init__()
        if min(patch_size, max_grid_size, hidden_size, intermediate_size, num_heads) <= 0:
            raise ValueError(
                "RADIOv4 patch, grid, hidden, MLP and head sizes must be positive"
            )
        if hidden_size % num_heads:
            raise ValueError("RADIOv4 hidden_size must be divisible by num_heads")
        if num_layers < 0 or num_cls_tokens <= 0 or num_registers < 0 or norm_eps <= 0:
            raise ValueError("Invalid RADIOv4 layer, prefix token count or norm epsilon")
        if video_temporal_patch_size is not None and video_temporal_patch_size <= 0:
            raise ValueError("video_temporal_patch_size must be positive")
        if not summary_idxs or any(i < 0 or i >= num_cls_tokens for i in summary_idxs):
            raise ValueError("summary_idxs must select at least one valid CLS token")
        self._summary_idxs = tuple(summary_idxs)
        self._num_prefix_tokens = num_cls_tokens + num_registers
        self.patch_generator = _RadioV4PatchGenerator(
            patch_size,
            max_grid_size,
            hidden_size,
            self._num_prefix_tokens,
            video_temporal_patch_size,
        )
        self.blocks = nn.ModuleList(
            [
                RadioBlock(hidden_size, intermediate_size, num_heads, norm_eps)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        use_video_patch_projection: bool = False,
    ) -> tuple[ir.Value, ir.Value]:
        hidden_states = self.patch_generator(op, pixel_values, use_video_patch_projection)
        for block in self.blocks:
            hidden_states = block(op, hidden_states)

        summary = op.Gather(hidden_states, list(self._summary_idxs), axis=1)
        summary = op.Reshape(summary, [0, -1])
        features = op.Slice(hidden_states, [self._num_prefix_tokens], [INT64_MAX], [1])
        return summary, features
