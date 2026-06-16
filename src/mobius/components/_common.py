# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

# Used as Slice "end" to mean "all remaining elements along this axis".
INT64_MAX = 9223372036854775807


class Linear(nn.Module):
    """Linear (fully-connected) layer using ONNX ops."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter([out_features, in_features])
        self.bias = nn.Parameter([out_features]) if bias else None

    def forward(self, op: OpBuilder, x: ir.Value):
        # Transpose weight from [out_features, in_features] → [in_features, out_features]
        # so MatMul(x, w_t) computes x @ weight.T.
        # FoldTransposedInitializerPass (applied after weight loading) will
        # pre-compute this transpose and eliminate the runtime Transpose node.
        w_t = op.Transpose(self.weight, perm=[1, 0])
        result = op.MatMul(x, w_t)
        if self.bias is not None:
            result = op.Add(result, self.bias)
        return result


class Embedding(nn.Module):
    """Embedding layer using ONNX Gather op."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter([num_embeddings, embedding_dim])
        self.padding_idx = padding_idx

    def forward(self, op: OpBuilder, input_ids: ir.Value):
        return op.Gather(self.weight, input_ids)


class LayerNorm(nn.Module):
    """Layer Normalization using ONNX LayerNormalization op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self.bias = nn.Parameter([hidden_size])
        self.eps = eps

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        return op.LayerNormalization(
            hidden_states,
            self.weight,
            self.bias,
            epsilon=self.eps,
            axis=-1,
        )


class OffsetLayerNorm(nn.Module):
    """Layer Normalization with +1 offset on weight: output = LN(x, weight+1, bias).

    Used by Nemotron where the HF checkpoint stores weights initialized
    to zero, and the effective multiplier is (1 + weight).  Analogous to
    ``OffsetRMSNorm`` but for full LayerNorm (with bias).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self.bias = nn.Parameter([hidden_size])
        self.eps = eps

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        effective_weight = op.Add(self.weight, 1.0)
        return op.LayerNormalization(
            hidden_states,
            effective_weight,
            self.bias,
            epsilon=self.eps,
            axis=-1,
        )


class LayerNormNoBias(nn.Module):
    """Layer Normalization with weight-only affine (no bias).

    Used by models like Cohere whose layer norms have only a ``weight``
    parameter, matching ``nn.LayerNorm(elementwise_affine=True, bias=False)``.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self.eps = eps

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        return op.LayerNormalization(
            hidden_states,
            self.weight,
            epsilon=self.eps,
            axis=-1,
        )


class LayerNormNoAffine(nn.Module):
    """Layer Normalization without learnable affine parameters.

    Used in AdaLayerNorm where scale/shift come from a modulation projection,
    matching ``nn.LayerNorm(elementwise_affine=False)`` in PyTorch.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self._hidden_size = hidden_size
        self._eps = eps

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        # ONNX LayerNormalization requires a Scale input; use all-ones
        # since this is the no-affine variant (scale/shift come externally).
        # CastLike ensures Scale matches the input dtype (fp16/bf16/fp32).
        scale = op.Constant(value=ir.tensor(np.ones(self._hidden_size, dtype=np.float32)))
        scale = op.CastLike(scale, hidden_states)
        return op.LayerNormalization(hidden_states, scale, axis=-1, epsilon=self._eps)


class GroupNorm(nn.Module):
    """Group Normalization using ONNX GroupNormalization op."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter((num_channels,))
        self.bias = nn.Parameter((num_channels,))
        self._num_groups = num_groups
        self._eps = eps

    def forward(self, op: OpBuilder, x: ir.Value):
        return op.GroupNormalization(
            x, self.weight, self.bias, num_groups=self._num_groups, epsilon=self._eps
        )


def create_attention_bias(
    op: OpBuilder,
    input_ids,
    attention_mask,
    sliding_window: int | None = None,
    dtype: ir.DataType = ir.DataType.FLOAT,
    block_sequence_ids=None,
):
    """Create causal attention bias for use in attention mechanisms.

    Args:
        op: The OpBuilder.
        input_ids: Input tensor of shape (batch_size, query_length).
        attention_mask: Attention mask of shape (batch_size, total_length).
        sliding_window: Optional sliding window size for local attention.
        dtype: Data type for the bias tensor. The masked value uses the
            minimum representable value for this dtype (e.g. -65504 for
            float16, -3.4e38 for float32).
        block_sequence_ids: Optional INT tensor of shape
            (batch_size, query_length) giving a contiguous vision-block id
            per current-sequence position (``>= 0`` for vision tokens in the
            same block, ``-1`` for text). When provided, a bidirectional
            "blockwise overlay" is OR-ed onto the causal (and sliding) mask:
            two positions in the same block (same id ``>= 0``) may attend to
            each other regardless of causal order. This mirrors HuggingFace
            ``blockwise_overlay`` for Gemma4 ``use_bidirectional_attention``.
            The returned bias bakes in causal + sliding + padding + blockwise,
            so the consuming ``Attention`` op MUST be called with
            ``is_causal=0`` to avoid re-applying the causal constraint and
            cancelling the bidirectional unmasking.

    Returns:
        Attention bias tensor of shape (batch_size, 1, query_length, total_length).
    """
    # cumsum on attention_mask gives indices
    all_indices = op.CumSum(attention_mask, 1)  # axis=1

    # kv_indices: (batch_size, 1, total_length)
    kv_indices = op.Unsqueeze(all_indices, [1])

    # q_indices: take last query_length elements
    # We use Gather with negative indices via dynamic slicing
    # For simplicity, use the full indices and let the Attention op handle masking
    # Actually we need to implement this with shape ops

    # Get query_length and total_length from shapes.
    # query_length comes from input_ids dim 1 (the query; e.g. 1 during decode).
    # total_length comes from attention_mask dim 1 (past + current tokens).
    # Using input_ids for query_length is semantically correct: during decode
    # input_ids is (batch, 1), so query_length=1 and start = total_length - 1,
    # giving the last row of q_indices.
    query_length = op.Shape(input_ids, start=1, end=2)  # 1-D [1]
    total_length = op.Shape(attention_mask, start=1, end=2)  # 1-D [1]
    start = op.Sub(total_length, query_length)
    # q_indices_2d: (batch_size, query_length)
    q_indices_2d = op.Slice(all_indices, start, total_length, [1])
    # q_indices: (batch_size, query_length, 1)
    q_indices = op.Unsqueeze(q_indices_2d, [2])

    # Causal mask: q_indices >= kv_indices
    full_mask = op.GreaterOrEqual(q_indices, kv_indices)

    if sliding_window is not None:
        # Also mask out positions too far away
        dist = op.Sub(q_indices, kv_indices)
        within_window = op.Less(dist, sliding_window)
        full_mask = op.And(full_mask, within_window)

    if block_sequence_ids is not None:
        # Bidirectional vision-block overlay (OR-ed onto causal/sliding mask,
        # BEFORE the padding AND, matching HF blockwise_overlay ordering).
        #
        # q_group: block id per query position -> (batch, query_length, 1).
        # block_sequence_ids covers the current input (== the query), so it
        # aligns 1:1 with the query positions.
        q_group = op.Unsqueeze(block_sequence_ids, [2])  # (B, q_len, 1)
        # kv_group: block id per kv position -> (batch, 1, total_length).
        # The kv axis spans past + current; past positions are text in the
        # cache, so left-pad with -1 to width total_length.
        pad_width = op.Sub(total_length, query_length)  # [1], == past length
        zero_1d = op.Constant(value_ints=[0])
        # Pad spec for a 2-D tensor [B, q_len]: [b_begin, s_begin, b_end, s_end].
        pads = op.Concat(zero_1d, pad_width, zero_1d, zero_1d, axis=0)
        kv_group_2d = op.Pad(
            block_sequence_ids,
            pads,
            op.Constant(value_int=-1),
        )  # (B, total_length)
        kv_group = op.Unsqueeze(kv_group_2d, [1])  # (B, 1, total_length)
        # same_block = (q_group == kv_group) AND (q_group >= 0)
        same_block = op.And(
            op.Equal(q_group, kv_group),
            op.GreaterOrEqual(q_group, op.Constant(value_int=0)),
        )
        full_mask = op.Or(full_mask, same_block)

    # Combine with attention_mask
    attn_mask_bool = op.Cast(op.Unsqueeze(attention_mask, [1]), to=ir.DataType.BOOL)
    full_mask = op.And(attn_mask_bool, full_mask)

    # Convert to float bias: 0 where attended, dtype.min where masked
    mask_value = float(dtype.min)
    attention_bias = op.Where(full_mask, 0.0, mask_value)
    attention_bias = op.Cast(attention_bias, to=dtype)

    # Unsqueeze to (batch_size, 1, query_length, total_length)
    return op.Unsqueeze(attention_bias, [1])


def build_packed_token_offset(op: OpBuilder, cu_seqlens) -> ir.Value:
    """Build ``token_offset`` for ``com.microsoft::PackedMultiHeadAttention``.

    ORT's ``PackedMultiHeadAttention`` treats the packed
    ``(token_count, hidden)`` query/key/value as ``batch_size``
    variable-length sub-sequences (delimited by ``cu_seqlens``) with
    right-padding removed, and computes block-diagonal attention *within*
    each sub-sequence.  The kernel derives ``batch_size`` from
    ``token_offset.shape[0]`` and requires ``cumulative_sequence_length``
    (i.e. ``cu_seqlens``) to have length ``batch_size + 1``.

    ``token_offset`` has shape ``(batch_size, max_seq_len)`` and follows
    ORT's ``GetPaddingOffset`` convention: the first ``token_count`` entries
    are the padded-layout indices (``b * max_seq_len + s``) of the valid
    tokens in packed (sub-sequence-major) order, followed by the
    padded-layout indices of the padding slots.

    Example: ``cu_seqlens = [0, 1, 3]`` (two sub-sequences of length 1 and
    2, so ``max_seq_len = 2``) yields ``[[0, 2], [3, 1]]``.

    Args:
        op: The OpBuilder.
        cu_seqlens: ``(batch_size + 1,)`` cumulative sequence lengths
            (INT32 or INT64).

    Returns:
        ``(batch_size, max_seq_len)`` INT32 ``token_offset`` tensor.
    """
    cu_i32 = op.Cast(cu_seqlens, to=ir.DataType.INT32)

    # batch_size = len(cu_seqlens) - 1  (number of packed sub-sequences).
    batch_size = op.Cast(
        op.Sub(op.Size(cu_seqlens), op.Constant(value_int=1)),
        to=ir.DataType.INT32,
    )

    # Per-sub-sequence lengths and the padded sequence dimension.
    starts = op.Slice(cu_i32, [0], [-1], [0])  # cu[:-1]
    ends = op.Slice(cu_i32, [1], [INT64_MAX], [0])  # cu[1:]
    lengths = op.Sub(ends, starts)  # (batch_size,)
    max_len = op.Squeeze(op.ReduceMax(lengths), [0])  # scalar INT32

    # Padded position grid: pos[b, s] = b * max_len + s.
    zero_i32 = op.Cast(op.Constant(value_int=0), to=ir.DataType.INT32)
    one_i32 = op.Cast(op.Constant(value_int=1), to=ir.DataType.INT32)
    rows = op.Range(zero_i32, batch_size, one_i32)  # (batch_size,)
    cols = op.Range(zero_i32, max_len, one_i32)  # (max_len,)
    pos_matrix = op.Add(
        op.Mul(op.Unsqueeze(rows, [1]), max_len),
        op.Unsqueeze(cols, [0]),
    )  # (batch_size, max_len) INT32
    pos_shape = op.Shape(pos_matrix)

    # Column s is a valid token for row b iff s < lengths[b].
    valid_mask = op.Less(op.Unsqueeze(cols, [0]), op.Unsqueeze(lengths, [1]))
    valid_mask_1d = op.Reshape(valid_mask, [-1])
    pos_1d = op.Reshape(pos_matrix, [-1])

    # Valid positions first (packed order), then padding-slot positions.
    valid = op.Compress(pos_1d, valid_mask_1d)
    padding = op.Compress(pos_1d, op.Not(valid_mask_1d))
    return op.Reshape(op.Concat(valid, padding, axis=0), pos_shape)


def create_padding_mask(
    op: OpBuilder,
    input_ids,
    attention_mask,
):
    """Create a bool padding mask for the ONNX Attention op.

    When used with ``is_causal=1`` on the Attention op, this provides a
    minimal mask that encodes only padding information. Causal masking is
    handled natively by the Attention op, avoiding the overhead of the
    CumSum/GreaterOrEqual/Where chain in ``create_attention_bias()``.

    Using a bool mask (instead of float additive bias) also unlocks Flash
    Attention eligibility in ORT, since Flash requires ``attn_mask`` to be
    either ``nullptr`` or ``bool`` type.

    The output is a 4D ``(batch_size, 1, q_len, total_length)`` bool tensor.
    The ORT Attention op requires ``mask_dim[-2] == q_sequence_length`` and a
    head dimension (``mask_dim[-3]``) that is either ``1`` or ``q_num_heads``
    (validated in ``attention_helper.h:ComputeOutputShapeForAttention``).  An
    explicit singleton head dim is required so that, for ``batch_size > 1``, the
    batch dimension is not right-aligned onto (and misread as) the head
    dimension — a 3D ``(batch, q_len, total)`` mask broadcasts as
    ``(q_num_heads=batch, ...)`` and is rejected once ``batch != 1``.

    Args:
        op: The OpBuilder.
        input_ids: Input tensor of shape ``(batch_size, q_length)`` or
            ``(batch_size, q_length, hidden_size)``, used to derive the
            query sequence length for mask expansion. Only dims 0 and 1
            are read, so 3D hidden_states (inputs_embeds path) are safe.
        attention_mask: Attention mask of shape ``(batch_size, total_length)``.
            INT64 tensor with ``1`` = valid token, ``0`` = padding.

    Returns:
        Bool mask of shape ``(batch_size, 1, q_length, total_length)``.
        ``True`` = attend, ``False`` = mask out.
    """
    bool_mask = op.Cast(attention_mask, to=ir.DataType.BOOL)
    # Unsqueeze to [B, 1, total_len] for broadcasting across q_len.
    mask_3d = op.Unsqueeze(bool_mask, [1])
    # Build target shape [B, q_len, total_len] using explicit slices.
    # input_ids may be 2D (input_ids) or 3D (hidden_states when inputs_embeds is
    # used), so we extract the batch dimension from it individually.
    # q_len comes from input_ids (the query, e.g. 1 in decode step); total_len
    # comes from attention_mask (covers both past and current tokens).
    batch_size = op.Shape(input_ids, start=0, end=1)
    # q_len comes from input_ids dim 1 (query length, e.g. 1 during decode).
    # total_len comes from attention_mask dim 1 (past + current tokens).
    q_len = op.Shape(input_ids, start=1, end=2)
    total_len = op.Shape(attention_mask, start=1, end=2)
    target_shape = op.Concat(batch_size, q_len, total_len, axis=0)
    mask_3d = op.Expand(mask_3d, target_shape)  # (B, q_len, total_len)
    # Insert a singleton head dim -> (B, 1, q_len, total_len) so the head axis
    # is explicit and the batch axis is never misread as q_num_heads.
    return op.Unsqueeze(mask_3d, [1])


def create_sliding_window_mask(
    op: OpBuilder,
    input_ids: ir.Value,
    attention_mask: ir.Value,
    window_size: int,
):
    """Create a bool mask combining padding and sliding-window constraints.

    When used with ``is_causal=1`` on the Attention op, causality is
    handled by the op.  This mask encodes two additional constraints:

    1. **Padding**: tokens with ``attention_mask == 0`` are masked out.
    2. **Sliding window**: each query position only attends to the
       ``window_size`` most recent key positions.

    The output is a 4D ``(batch_size, 1, q_len, total_length)`` bool tensor.
    The explicit singleton head dim (``mask_dim[-3]``) ensures the batch axis is
    not misread as ``q_num_heads`` by the ORT Attention op when ``batch > 1``.

    Args:
        op: The OpBuilder.
        input_ids: ``(batch_size, q_length)`` — used for query length.
        attention_mask: ``(batch_size, total_length)`` INT64 (1=valid).
        window_size: The number of recent tokens each position attends to.

    Returns:
        Bool mask ``(batch_size, 1, q_len, total_length)``.
    """
    # Position indices via cumsum on attention_mask
    # CumSum gives 1-based indices for non-padding tokens, 0 stays 0
    all_indices = op.CumSum(attention_mask, op.Constant(value_int=1))

    # kv_indices: (batch, 1, total_len)
    kv_indices = op.Unsqueeze(all_indices, [1])

    # q_indices: last q_len positions → (batch, q_len, 1)
    q_len = op.Shape(input_ids, start=1, end=2)
    total_len = op.Shape(attention_mask, start=1, end=2)
    start = op.Sub(total_len, q_len)
    q_indices = op.Unsqueeze(op.Slice(all_indices, start, total_len, [1]), [2])

    # Sliding window: distance < window_size
    # dist = q_pos - kv_pos; within window when dist < window_size
    dist = op.Sub(q_indices, kv_indices)
    within_window = op.Less(dist, op.Constant(value_int=window_size))

    # Combine with padding mask
    padding_mask = op.Cast(op.Unsqueeze(attention_mask, [1]), to=ir.DataType.BOOL)
    mask_3d = op.And(within_window, padding_mask)  # (B, q_len, total_len)
    # Insert a singleton head dim -> (B, 1, q_len, total_len) so the head axis
    # is explicit and the batch axis is never misread as q_num_heads.
    return op.Unsqueeze(mask_3d, [1])
