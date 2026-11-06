# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Source-faithful Omni vision projection and Efficient Video Sampling (EVS).

Matches ``modeling_nemotron_h_omni.py`` at HuggingFace revision
``c6cc2e8ffa353403213d6b9b18aaf8eb684d705a``. These helpers do not own the RADIO
tower, processor, input-token scatter, or language model. Pixels must already
be normalized and cast to the tower's compute dtype.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components import INT64_MAX, Linear, RMSNorm

if TYPE_CHECKING:
    from mobius.components import RadioV4VisionModel


class _ProjectorRMSNorm(RMSNorm):
    """Normalize and apply gamma in float32, then round once to model dtype."""

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        result = op.RMSNormalization(
            op.Cast(hidden_states, to=ir.DataType.FLOAT),
            op.Cast(self.weight, to=ir.DataType.FLOAT),
            epsilon=self.variance_epsilon,
            axis=-1,
        )
        return op.CastLike(result, hidden_states)


class _ReluSquared(nn.Module):
    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = op.Relu(hidden_states)
        return op.Mul(hidden_states, hidden_states)


class NemotronHOmniVisionProjector(nn.Module):
    """V2 pixel shuffle -> fp32 RMSNorm -> native linear/ReLU²/linear.

    ``spatial_features`` is ``(B, rows*cols, vision_hidden_size)`` and
    ``grid_hw`` is an int64 vector ``[rows, cols]``. Returns
    ``(B, (rows/merge_size)*(cols/merge_size), text_hidden_size)``.
    Grid dimensions must be positive multiples of ``merge_size``. B may be zero.

    Names ``mlp1.0.weight``, ``mlp1.1.weight``, ``mlp1.3.weight`` match the
    original checkpoint (the HF loader converts these to norm/linear1/linear2).
    Bolt explicitly has ``vision_final_layernorm=False``. Other final-norm
    configurations and legacy transposed pixel shuffle are rejected.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        projector_hidden_size: int,
        text_hidden_size: int,
        *,
        downsample_ratio: float = 0.5,
        ps_version: str = "v2",
        vision_final_layernorm: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        if min(vision_hidden_size, projector_hidden_size, text_hidden_size) <= 0:
            raise ValueError("Omni projector dimensions must be positive")
        if downsample_ratio not in {0.5}:
            raise ValueError("Only Bolt downsample_ratio=0.5 is supported")
        merge_size = 2
        if ps_version != "v2":
            raise ValueError("Only source ps_version='v2' pixel shuffle is supported")
        if vision_final_layernorm is not False:
            raise ValueError("Bolt requires explicit vision_final_layernorm=False")
        if not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be positive and finite")
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.merge_size = merge_size
        merged_size = vision_hidden_size * merge_size**2
        self.mlp1 = nn.Sequential(
            _ProjectorRMSNorm(merged_size, eps=norm_eps),
            Linear(merged_size, projector_hidden_size, bias=False),
            _ReluSquared(),
            Linear(projector_hidden_size, text_hidden_size, bias=False),
        )

    def pixel_shuffle(
        self, op: OpBuilder, spatial_features: ir.Value, grid_hw: ir.Value
    ) -> ir.Value:
        """Exact source v2 order, including its intermediate width/height swap."""
        batch = op.Shape(spatial_features, start=0, end=1)
        rows = op.Slice(grid_hw, [0], [1])
        cols = op.Slice(grid_hw, [1], [2])
        merged_rows = op.Div(rows, [self.merge_size])
        merged_cols = op.Div(cols, [self.merge_size])
        states = op.Reshape(
            spatial_features,
            op.Concat(
                batch, rows, merged_cols, [self.vision_hidden_size * self.merge_size], axis=0
            ),
            allowzero=1,
        )
        states = op.Transpose(states, perm=[0, 2, 1, 3])
        states = op.Reshape(
            states,
            op.Concat(
                batch,
                merged_cols,
                merged_rows,
                [self.vision_hidden_size * self.merge_size**2],
                axis=0,
            ),
            allowzero=1,
        )
        states = op.Transpose(states, perm=[0, 2, 1, 3])
        return op.Reshape(
            states,
            op.Concat(
                batch,
                op.Mul(merged_rows, merged_cols),
                [self.vision_hidden_size * self.merge_size**2],
                axis=0,
            ),
            allowzero=1,
        )  # (B, merged_rows*merged_cols, row_offset*col_offset*channel)

    def forward(
        self, op: OpBuilder, spatial_features: ir.Value, grid_hw: ir.Value
    ) -> ir.Value:
        return self.mlp1(op, self.pixel_shuffle(op, spatial_features, grid_hw))


def pack_omni_video_frames(
    op: OpBuilder, pixel_values: ir.Value, temporal_patch_size: int = 2
) -> ir.Value:
    """Pad by repeating the final frame, then pack consecutive frames in channels.

    Input ``(N, 3, H, W)`` -> ``(ceil(N/T), T*3, H, W)``. One invocation is one
    clip; callers must not concatenate unrelated clips before temporal packing.
    """
    if temporal_patch_size <= 0:
        raise ValueError("temporal_patch_size must be positive")
    count = op.Shape(pixel_values, start=0, end=1)
    remaining = op.Mod(count, [temporal_patch_size])
    pad_count = op.Mod(op.Sub([temporal_patch_size], remaining), [temporal_patch_size])
    last = op.Slice(pixel_values, [-1], [INT64_MAX], [0])
    padding = op.Expand(
        last, op.Concat(pad_count, op.Shape(pixel_values, start=1, end=4), axis=0)
    )
    padded = op.Concat(pixel_values, padding, axis=0)
    return op.Reshape(
        padded,
        op.Concat(
            op.Div(op.Add(count, pad_count), [temporal_patch_size]),
            [temporal_patch_size * 3],
            op.Shape(pixel_values, start=2, end=4),
            axis=0,
        ),
        allowzero=1,
    )


def compact_omni_video_features(
    op: OpBuilder, video_features: ir.Value, pruning_rate: float
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """EVS for one clip, returning compact features, dense mask, and kept indices.

    Input is post-projector ``(T, P, D)``. Outputs are ``(K, D)``, bool ``(T*P,)``,
    and int64 ``(K,)``. Indices are ascending original feature positions, not
    descending EVS scores. Preserve this mask when compacting the corresponding
    video placeholder positions in input_ids, attention masks, and embeddings.
    Do not scatter compact features into the original unpruned prompt.

    This is the core source ``compute_retention_mask`` with spatial_merge_size=1;
    P is the number of projected patches per frame. Empty T is supported.
    Cosine scores must be finite. In particular, float16 zero vectors make the
    source's epsilon underflow and produce NaNs; that undefined ranking case
    fails explicitly instead of arbitrarily violating first-frame retention.
    The source wrapper's separate sqrt(P) square-grid inference is not performed
    here; the video pipeline below preserves that wrapper restriction.
    Dynamic output sizes and the finite-score guard require runtime shape
    synchronization; this helper does not promise CUDA graph capture.
    """
    if not math.isfinite(pruning_rate) or not 0 <= pruning_rate <= 1:
        raise ValueError("pruning_rate must be finite and in [0, 1]")
    frames = op.Shape(video_features, start=0, end=1)
    patches = op.Shape(video_features, start=1, end=2)
    hidden = op.Shape(video_features, start=2, end=3)
    total = op.Mul(frames, patches)
    flattened = op.Reshape(video_features, op.Concat(total, hidden, axis=0), allowzero=1)
    if pruning_rate == 0:
        mask = op.ConstantOfShape(total, value=ir.tensor(np.array([True], dtype=np.bool_)))
        indices = op.Range(0, op.Squeeze(total, [0]), 1)
        return flattened, mask, indices

    current = op.Slice(video_features, [1], [INT64_MAX], [0])
    previous = op.Slice(video_features, [0], [-1], [0])
    epsilon = op.CastLike(op.Constant(value_float=1e-8), video_features)
    # Match torch cosine_similarity's separate norm clamps and native-dtype
    # normalized-vector product, not dot(a,b)/(norm(a)*norm(b)).
    # PyTorch accumulates reduced-precision vector norms in float32, then
    # rounds the norm to the input dtype before clamp/division. ORT does not
    # provide a bfloat16 ReduceL2 kernel, so make that accumulation explicit.
    current_norm = op.CastLike(
        op.ReduceL2(op.Cast(current, to=ir.DataType.FLOAT), [-1], keepdims=True), current
    )
    previous_norm = op.CastLike(
        op.ReduceL2(op.Cast(previous, to=ir.DataType.FLOAT), [-1], keepdims=True), previous
    )
    current = op.Div(current, op.Max(current_norm, epsilon))
    previous = op.Div(previous, op.Max(previous_norm, epsilon))
    product = op.Mul(current, previous)
    similarity = op.CastLike(
        op.ReduceSum(op.Cast(product, to=ir.DataType.FLOAT), [-1], keepdims=False), product
    )
    dissimilarity = op.Sub(op.CastLike(1.0, video_features), similarity)
    first_frame = op.ConstantOfShape(
        op.Concat(op.Min(frames, [1]), patches, axis=0),
        value=ir.tensor(np.array([255], dtype=np.float32)),
    )
    scores = op.Reshape(
        op.Concat(op.CastLike(first_frame, video_features), dissimilarity, axis=0),
        total,
        allowzero=1,
    )
    # ONNX TopK does not specify NaN ordering. Reject nonfinite source scores
    # using a shape invariant: the requested length equals total iff all scores
    # are finite. This never substitutes or perturbs an invalid score.
    invalid_count = op.ReduceSum(
        op.Cast(op.Or(op.IsNaN(scores), op.IsInf(scores)), to=ir.DataType.INT64),
        keepdims=True,
    )
    scores = op.Reshape(scores, op.Add(total, invalid_count), allowzero=1)
    # Python int(T*P*(1-q)) uses double precision before truncation. Float32
    # arithmetic changes boundary counts (e.g. T*P=30 and q=0.8 keeps 5, not 6).
    count = op.Cast(
        op.Mul(
            op.Cast(total, to=ir.DataType.DOUBLE),
            op.Constant(value=ir.tensor(np.array(1 - pruning_rate, dtype=np.float64))),
        ),
        to=ir.DataType.INT64,
    )
    keep_count = op.Min(total, op.Max(patches, count))
    # ONNX TopK breaks equal-value ties by lower original index, exactly as
    # descending torch.argsort(stable=True). Never perturb scores to break ties.
    # Casting the already-rounded scores is exact and preserves all ties;
    # it also avoids requiring a bfloat16 TopK kernel.
    _, top_indices = op.TopK(
        op.Cast(scores, to=ir.DataType.FLOAT),
        keep_count,
        axis=0,
        largest=1,
        sorted=1,
        _outputs=2,
    )
    mask = op.ConstantOfShape(total, value=ir.tensor(np.array([False], dtype=np.bool_)))
    selected = op.ConstantOfShape(
        keep_count, value=ir.tensor(np.array([True], dtype=np.bool_))
    )
    mask = op.ScatterElements(mask, top_indices, selected, axis=0)
    indices = op.Squeeze(op.NonZero(mask), [0])
    return op.Gather(flattened, indices, axis=0), mask, indices


def build_omni_image_features(
    op: OpBuilder,
    pixel_values: ir.Value,
    *,
    vision_model: RadioV4VisionModel,
    projector: NemotronHOmniVisionProjector,
    patch_size: int,
) -> ir.Value:
    """Nonempty image batch -> dense ``(B, P, D)`` projected features.

    Missing media is represented by not invoking the encoder, as in the source
    wrapper's ``pixel_values is None`` branch, not by a zero-size RADIO batch.
    """
    _, spatial = vision_model(op, pixel_values)
    grid = op.Div(op.Shape(pixel_values, start=2, end=4), [patch_size])
    return projector(op, spatial, grid)


def build_omni_video_features(
    op: OpBuilder,
    pixel_values: ir.Value,
    *,
    vision_model: RadioV4VisionModel,
    projector: NemotronHOmniVisionProjector,
    patch_size: int,
    temporal_patch_size: int = 2,
    pruning_rate: float = 0.7,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """One nonempty clip -> compact projected features and original-position mask.

    Mirrors the source wrapper's EVS square-grid inference; non-square projected
    patch counts fail its reshape instead of silently changing wrapper behavior.
    Image and video streams stay separate: the source uses img_context_token_id
    for video scatter too (Bolt's video_context_token_id is None).
    """
    packed = pack_omni_video_frames(op, pixel_values, temporal_patch_size)
    _, spatial = vision_model(op, packed, True)
    grid = op.Div(op.Shape(pixel_values, start=2, end=4), [patch_size])
    projected = projector(op, spatial, grid)
    if pruning_rate > 0:
        count = op.Shape(projected, start=1, end=2)
        side = op.Cast(op.Sqrt(op.Cast(count, to=ir.DataType.DOUBLE)), to=ir.DataType.INT64)
        projected = op.Reshape(
            projected,
            op.Concat(
                op.Shape(projected, start=0, end=1),
                op.Mul(side, side),
                [projector.text_hidden_size],
                axis=0,
            ),
        )
    return compact_omni_video_features(op, projected, pruning_rate)
