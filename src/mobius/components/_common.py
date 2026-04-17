# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius._flags import flags

if TYPE_CHECKING:
    import torch

# Used as Slice "end" to mean "all remaining elements along this axis".
INT64_MAX = 9223372036854775807

# ORT ≤1.24.x CUDA Gather kernel uses int32 for element offset
# computation (CUDA_LONG = int32_t).  Tensors with >2^31 elements
# cause integer overflow → cudaErrorIllegalAddress.
# See: https://github.com/microsoft/onnxruntime/issues/28107
_MAX_GATHER_ELEMENTS = 2**31 - 1


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

    def forward(self, op: builder.OpBuilder, x: ir.Value):
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
    """Embedding layer using ONNX Gather op.

    When :attr:`~mobius._flags._Flags.ort_shard_large_gathers` is enabled
    and the embedding table exceeds ~2.1B elements, creates sharded weight
    parameters and emits a sharded Gather subgraph to work around the ORT
    CUDA int32 overflow (onnxruntime#28107).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        total_elements = num_embeddings * embedding_dim
        if flags.ort_shard_large_gathers and total_elements > _MAX_GATHER_ELEMENTS:
            max_rows = _MAX_GATHER_ELEMENTS // embedding_dim
            self._num_shards = math.ceil(num_embeddings / max_rows)
            self._shard_size = math.ceil(num_embeddings / self._num_shards)

            self.weight = None  # type: ignore[assignment]
            for i in range(self._num_shards):
                start = i * self._shard_size
                end = min(start + self._shard_size, num_embeddings)
                setattr(self, f"weight_shard{i}", nn.Parameter([end - start, embedding_dim]))
        else:
            self._num_shards = 0
            self._shard_size = 0
            self.weight = nn.Parameter([num_embeddings, embedding_dim])

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value):
        if self._num_shards == 0:
            return op.Gather(self.weight, input_ids)
        return self._sharded_gather(op, input_ids)

    def _sharded_gather(self, op: builder.OpBuilder, input_ids: ir.Value):
        """Emit a sharded Gather that avoids int32 offset overflow.

        Splits the embedding lookup across N shards.  Routes indices to
        the correct shard using ``Less`` / ``Where`` / ``Sub`` ops.
        """
        # Build routing: nested Where from the last shard backwards
        boundaries = [i * self._shard_size for i in range(self._num_shards)]
        result: ir.Value | None = None

        for i in reversed(range(self._num_shards)):
            boundary = boundaries[i]
            shard_weight = getattr(self, f"weight_shard{i}")

            # local_idx = input_ids - boundary
            if boundary == 0:
                local_idx = input_ids
            else:
                local_idx = op.Sub(input_ids, op.Constant(value_int=boundary))

            shard_result = op.Gather(shard_weight, local_idx)

            if result is None:
                # Last shard is the fallback
                result = shard_result
            else:
                # cond = input_ids < boundary[i+1]
                next_boundary = boundaries[i + 1]
                cond = op.Less(input_ids, op.Constant(value_int=next_boundary))
                # Unsqueeze for broadcasting: e.g. [B, S] → [B, S, 1]
                cond = op.Unsqueeze(cond, op.Constant(value_int=-1))
                result = op.Where(cond, shard_result, result)

        assert result is not None
        return result

    def shard_weight_dict(
        self, original_key: str, tensor: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Split an embedding weight tensor into shard key/tensor pairs.

        Call this from ``preprocess_weights`` to map the single HF embedding
        weight into the per-shard parameter names expected by the ONNX model.

        Args:
            original_key: The ONNX parameter name for the unsplit weight
                (e.g. ``"decoder.model.embed_tokens_per_layer.weight"``).
            tensor: The full embedding weight tensor [V, D].

        Returns:
            Dict mapping shard keys to sliced tensors.  If sharding is
            inactive, returns ``{original_key: tensor}`` unchanged.
        """
        if self._num_shards == 0:
            return {original_key: tensor}

        # Replace ".weight" suffix with ".weight_shardN"
        prefix = original_key.rsplit(".weight", 1)[0]
        result: dict[str, torch.Tensor] = {}
        for i in range(self._num_shards):
            start = i * self._shard_size
            end = min(start + self._shard_size, self.num_embeddings)
            result[f"{prefix}.weight_shard{i}"] = tensor[start:end]
        return result


class LayerNorm(nn.Module):
    """Layer Normalization using ONNX LayerNormalization op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self.bias = nn.Parameter([hidden_size])
        self.eps = eps

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
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

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
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

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
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

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
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

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        return op.GroupNormalization(
            x, self.weight, self.bias, num_groups=self._num_groups, epsilon=self._eps
        )


def create_attention_bias(
    op: builder.OpBuilder,
    input_ids,
    attention_mask,
    sliding_window: int | None = None,
    dtype: ir.DataType = ir.DataType.FLOAT,
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

    # Combine with attention_mask
    attn_mask_bool = op.Cast(op.Unsqueeze(attention_mask, [1]), to=ir.DataType.BOOL)
    full_mask = op.And(attn_mask_bool, full_mask)

    # Convert to float bias: 0 where attended, dtype.min where masked
    mask_value = float(dtype.min)
    attention_bias = op.Where(full_mask, 0.0, mask_value)
    attention_bias = op.Cast(attention_bias, to=dtype)

    # Unsqueeze to (batch_size, 1, query_length, total_length)
    return op.Unsqueeze(attention_bias, [1])


def create_padding_mask(
    op: builder.OpBuilder,
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

    The output is a 3D ``(batch_size, q_len, total_length)`` bool tensor.
    The ORT Attention op requires ``mask_dim[-2] == q_sequence_length``
    (validated in ``attention_helper.h:ComputeOutputShapeForAttention``),
    so the padding mask is broadcast-expanded along the query dimension.

    Args:
        op: The OpBuilder.
        input_ids: Input tensor of shape ``(batch_size, q_length)`` or
            ``(batch_size, q_length, hidden_size)``, used to derive the
            query sequence length for mask expansion. Only dims 0 and 1
            are read, so 3D hidden_states (inputs_embeds path) are safe.
        attention_mask: Attention mask of shape ``(batch_size, total_length)``.
            INT64 tensor with ``1`` = valid token, ``0`` = padding.

    Returns:
        Bool mask of shape ``(batch_size, q_length, total_length)``.
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
    return op.Expand(mask_3d, target_shape)
