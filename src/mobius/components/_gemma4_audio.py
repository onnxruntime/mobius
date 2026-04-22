# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma4 audio encoder components (Universal Speech Model architecture).

Implements the Conformer-based audio encoder for Gemma4 any-to-any variants
(E2B, E4B). Architecture from ``Gemma4AudioModel`` in HuggingFace transformers
(model_type=gemma4_audio), based on the Universal Speech Model (USM).

Public components::

    Gemma4ConvSubsampling       - 2-stage Conv2d → LayerNorm → ReLU → Linear
    Gemma4FeedForward           - Standard 2-layer MLP with RMSNorm & gradient clip
    Gemma4LightConv1d           - Linear-GLU → CausalDepthwiseConv1d → Linear
    Gemma4Attention             - Custom chunked attention (offline equivalent)
    Gemma4AudioLayer            - FF1 → Attention → LightConv1d → FF2 → RMSNorm
    Gemma4AudioEncoder          - Full 12-layer encoder with output projection

Default config (google/gemma-4-E2B-it audio_config)::

    hidden_size=1024, num_attention_heads=8, num_hidden_layers=12,
    subsampling_conv_channels=[128, 32], conv_kernel_size=5,
    attention_context_left=13, output_proj_dims=1536, rms_norm_eps=1e-6,
    residual_weight=0.5, gradient_clipping=1e10

Notes:
- ``use_clipped_linears=True``: ``Gemma4ClippableLinear`` has learned
  ``input_{min,max}`` and ``output_{min,max}`` buffers that clamp activations
  before and after the linear projection. Implemented as ``ClippableLinear``.
- ``attention_logit_cap=50.0``: Soft-capping ``tanh(logits/cap)*cap`` IS
  implemented using standard ONNX ops (MatMul/Tanh/Div/Mul). The ONNX
  Attention op's native ``softcap`` attribute cannot be used because the
  relative position bias must be added *before* softcap (see
  ``Gemma4Attention`` docstring for details).
- The blocked chunked attention is simplified to full offline MHA with a
  causal sliding-window mask — equivalent output for offline (non-streaming)
  inference. The relative position bias is fully implemented.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._common import LayerNormNoBias, Linear
from mobius.components._conv import CausalDepthwiseConv1d, Conv2dNoBias
from mobius.components._rms_norm import RMSNorm

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _gradient_clip(op: builder.OpBuilder, x: ir.Value, clip_val: float = 1e9) -> ir.Value:
    """Clamp activations to ±clip_val (gradient clipping for numerical stability)."""
    lo = op.CastLike(op.Constant(value_float=-clip_val), x)
    hi = op.CastLike(op.Constant(value_float=clip_val), x)
    return op.Clip(x, lo, hi)


def _glu(op: builder.OpBuilder, x: ir.Value) -> ir.Value:
    """GLU: split last dim in half → a * sigmoid(b)."""
    a, b = op.Split(x, axis=-1, num_outputs=2, _outputs=2)
    return op.Mul(a, op.Sigmoid(b))


def _swish(op: builder.OpBuilder, x: ir.Value) -> ir.Value:
    """SiLU/Swish activation: x * sigmoid(x)."""
    return op.Mul(x, op.Sigmoid(x))


# ---------------------------------------------------------------------------
# Public components
# ---------------------------------------------------------------------------


class ClippableLinear(nn.Module):
    """Linear layer with learned input/output activation clamping.

    Matches ``Gemma4ClippableLinear`` in HuggingFace transformers.
    The checkpoint stores learned ``input_{min,max}`` and ``output_{min,max}``
    scalars that clamp activations before and after the linear projection::

        x = clamp(x, input_min, input_max)
        x = x @ weight.T [+ bias]
        x = clamp(x, output_min, output_max)

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        bias: Whether to include a bias term (default: False).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter([out_features, in_features])
        self.bias = nn.Parameter([out_features]) if bias else None
        # Learned activation clipping bounds (scalar)
        self.input_min = nn.Parameter([])
        self.input_max = nn.Parameter([])
        self.output_min = nn.Parameter([])
        self.output_max = nn.Parameter([])

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        # Clamp input activations
        x = op.Clip(x, self.input_min, self.input_max)
        # Linear: x @ weight.T [+ bias]
        w_t = op.Transpose(self.weight, perm=[1, 0])
        result = op.MatMul(x, w_t)
        if self.bias is not None:
            result = op.Add(result, self.bias)
        # Clamp output activations
        return op.Clip(result, self.output_min, self.output_max)


class Gemma4ConvSubsampling(nn.Module):
    """2-stage Conv2d subsampling for Gemma4 audio features.

    Each stage: ``Conv2dNoBias(stride=2) → LayerNorm(no bias) → ReLU``.
    A final linear projection maps the flattened features to ``hidden_size``.

    Channel progression: ``1 → conv_channels[0] → conv_channels[1]``.
    Time dimension is reduced by 4x; frequency by 4x as well.

    Layout inside each conv stage (matching HuggingFace)::

        x = conv(x)                          # [B, C, T', F']
        x = act(norm(x.T_CF).T_FC)          # LayerNorm over channels, then ReLU

    Args:
        input_size: Input mel-spectrogram frequency bins (default 128).
        conv_channels: ``[c0, c1]`` per-stage output channels.
        hidden_size: Output feature dimension.
    """

    def __init__(
        self,
        input_size: int = 128,
        conv_channels: list[int] | None = None,
        hidden_size: int = 1024,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [128, 32]
        c0, c1 = conv_channels

        # Frequency dim after 2x stride-2 Conv2d with pad=1, kernel=3:
        #   F_out = (F_in + 2 - 3) // 2 + 1 = (F_in - 1) // 2 + 1
        freq = input_size
        for _ in range(2):
            freq = (freq - 3 + 2) // 2 + 1

        self.conv0 = Conv2dNoBias(1, c0, kernel_size=3, stride=2, padding=1)
        self.norm0 = LayerNormNoBias(c0, eps=norm_eps)

        self.conv1 = Conv2dNoBias(c0, c1, kernel_size=3, stride=2, padding=1)
        self.norm1 = LayerNormNoBias(c1, eps=norm_eps)

        # Linear: (c1 * freq_after_2_stages) → hidden_size
        # HF Gemma4AudioSubSampleConvProjection uses nn.Linear(bias=False)
        # for this projection, so no bias initializer should be created.
        self.input_proj_linear = Linear(c1 * freq, hidden_size, bias=False)

    def _conv_norm_relu(
        self,
        op: builder.OpBuilder,
        x: ir.Value,
        conv: Conv2dNoBias,
        norm: LayerNormNoBias,
    ) -> ir.Value:
        # x: [B, C_in, T, F]
        x = conv(op, x)  # [B, C_out, T', F']
        # LayerNorm over channel dim: permute so channels are last, norm, permute back
        x = op.Transpose(x, perm=[0, 2, 3, 1])  # [B, T', F', C_out]
        x = norm(op, x)  # [B, T', F', C_out]
        x = op.Transpose(x, perm=[0, 3, 1, 2])  # [B, C_out, T', F']
        return op.Relu(x)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: [B, T, input_size]
        x = op.Unsqueeze(x, [1])  # [B, 1, T, input_size]

        x = self._conv_norm_relu(op, x, self.conv0, self.norm0)  # [B, c0, T//2, F//2]
        x = self._conv_norm_relu(op, x, self.conv1, self.norm1)  # [B, c1, T//4, F//4]

        # [B, c1, T', F'] → [B, T', F'*c1]  (permute so T is dim 1, flatten C and F)
        x = op.Transpose(x, perm=[0, 2, 3, 1])  # [B, T', F', c1]
        batch = op.Shape(x, start=0, end=1)
        t_out = op.Shape(x, start=1, end=2)
        x = op.Reshape(x, op.Concat(batch, t_out, op.Constant(value_ints=[-1]), axis=0))

        return self.input_proj_linear(op, x)  # [B, T', hidden_size]


class Gemma4FeedForward(nn.Module):
    """Gemma4 audio feed-forward module (standard MLP with residual and norms).

    Matches ``Gemma4AudioFeedForward``::

        residual = x
        x = clip(x, ±GC)
        x = pre_rms_norm(x)
        x = linear1(x)     # h → 4h
        x = silu(x)
        x = linear2(x)     # 4h → h
        x = clip(x, ±GC)
        x = post_rms_norm(x)
        x = x * residual_weight   # 0.5
        return x + residual

    Args:
        hidden_size: Model hidden dimension.
        rms_norm_eps: Epsilon for RMSNorm.
        residual_weight: Scale applied to FF output before adding residual (0.5).
        gradient_clipping: Clamp value for numerical stability (1e9).
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        rms_norm_eps: float = 1e-6,
        residual_weight: float = 0.5,
        gradient_clipping: float = 1e9,
    ):
        super().__init__()
        self._residual_weight = residual_weight
        self._gradient_clipping = gradient_clipping

        self.pre_layer_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ffw_layer_1 = ClippableLinear(hidden_size, hidden_size * 4, bias=False)
        self.ffw_layer_2 = ClippableLinear(hidden_size * 4, hidden_size, bias=False)
        self.post_layer_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        residual = x
        x = _gradient_clip(op, x, self._gradient_clipping)
        x = self.pre_layer_norm(op, x)
        x = self.ffw_layer_1(op, x)  # [B, T, 4h]
        x = _swish(op, x)
        x = self.ffw_layer_2(op, x)  # [B, T, h]
        x = _gradient_clip(op, x, self._gradient_clipping)
        x = self.post_layer_norm(op, x)
        x = op.Mul(x, op.CastLike(op.Constant(value_float=self._residual_weight), x))
        return op.Add(x, residual)


class Gemma4LightConv1d(nn.Module):
    """Gemma4 audio lightweight conv1d (causal) module.

    Matches ``Gemma4AudioLightConv1d``::

        residual = x
        x = pre_rms_norm(x)
        x = linear_start(x)    # h → 2h (ClippableLinear ≈ plain Linear)
        x = glu(x)             # 2h → h  (a * sigmoid(b))
        x = causal_dw_conv1d(x.T).T
        x = clip(x, ±GC)
        x = conv_norm(x)
        x = silu(x)
        x = linear_end(x)      # h → h
        return x + residual

    Args:
        hidden_size: Model hidden dimension.
        conv_kernel_size: Depthwise conv kernel size (Gemma4 default: 5).
        rms_norm_eps: Epsilon for RMSNorm.
        gradient_clipping: Clamp value for numerical stability.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        conv_kernel_size: int = 5,
        rms_norm_eps: float = 1e-6,
        gradient_clipping: float = 1e9,
    ):
        super().__init__()
        self._gradient_clipping = gradient_clipping

        self.pre_layer_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.linear_start = ClippableLinear(hidden_size, hidden_size * 2, bias=False)
        self.depthwise_conv1d = CausalDepthwiseConv1d(hidden_size, conv_kernel_size)
        self.conv_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.linear_end = ClippableLinear(hidden_size, hidden_size, bias=False)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        residual = x
        x = self.pre_layer_norm(op, x)  # [B, T, h]
        x = self.linear_start(op, x)  # [B, T, 2h]
        x = _glu(op, x)  # [B, T, h]

        # Causal depthwise conv expects [B, C, T] layout
        x = op.Transpose(x, perm=[0, 2, 1])  # [B, h, T]
        x = self.depthwise_conv1d(op, x)  # [B, h, T]
        x = op.Transpose(x, perm=[0, 2, 1])  # [B, T, h]

        x = _gradient_clip(op, x, self._gradient_clipping)
        x = self.conv_norm(op, x)
        x = _swish(op, x)
        x = self.linear_end(op, x)  # [B, T, h]
        return op.Add(x, residual)


class Gemma4Attention(nn.Module):
    """Gemma4 audio attention (offline-equivalent causal sliding-window).

    Faithfully implements:
    - Custom Q scale: ``head_dim^-0.5 / log(2)``
    - Custom K scale: ``log(1+e) / log(2)``
    - Learnable per-dimension Q scale (``per_dim_scale``)
    - Relative position bias via ``relative_k_proj`` + sinusoidal embeddings
    - Soft-capping: ``tanh(scores / cap) * cap``
    - Causal sliding-window mask (``attention_context_left`` left frames)

    For offline (non-streaming) inference the blocked computation of the
    original implementation is replaced by full TxT MHA with a local mask,
    which produces identical outputs.

    **Why the ONNX Attention op's native ``softcap`` attribute cannot be used:**

    The ONNX ``Attention`` op (opset 24) has a ``softcap`` attribute that applies
    ``tanh(qk / cap) * cap``, but its internal pipeline is fixed as:

        QK matmul → scale → softcap → attn_mask add → softmax

    This audio encoder requires a **different order**:

        QK matmul → scale → relative_position_bias add → softcap → window_mask add → softmax

    The relative position bias must be inside the softcap (i.e. softcap is applied
    to ``qk + rel_bias``), but the ONNX op's ``softcap`` fires before the attention
    mask, with no hook for pre-softcap bias injection.

    Additionally, the non-standard Q/K scaling factors and the ``per_dim_scale``
    learnable Q scaling cannot be expressed as ONNX Attention op attributes at all.

    Therefore this class uses lower-level ``MatMul``, ``Softmax``, etc. ops to
    implement the full attention computation manually.

    TODO: If the ONNX Attention spec is extended to support a pre-softcap additive
    bias input (separate from ``attn_mask``), the manual Tanh/Div/Mul ops (lines
    ~454-456) could be replaced by the native ``softcap`` attribute. Track at:
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#Attention

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        attention_context_left: Left-context window size, including the
            current frame (Gemma4 default: 13).
        attention_logit_cap: Soft-capping value (Gemma4 default: 50.0).
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 8,
        attention_context_left: int = 13,
        attention_logit_cap: float = 50.0,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads
        self._attention_context_left = attention_context_left
        self._attention_logit_cap = attention_logit_cap

        # Non-standard Q/K scaling factors (from Gemma4 attention)
        self._q_scale = (self._head_dim**-0.5) / math.log(2)
        self._k_scale = math.log(1 + math.e) / math.log(2)

        # Q/K/V: no bias (HF nn.Linear(..., bias=False))
        self.q_proj = ClippableLinear(hidden_size, hidden_size, bias=False)
        self.k_proj = ClippableLinear(hidden_size, hidden_size, bias=False)
        self.v_proj = ClippableLinear(hidden_size, hidden_size, bias=False)
        # post: no bias (HF has no .bias key for self_attn.post in checkpoint)
        self.post = ClippableLinear(hidden_size, hidden_size, bias=False)

        # Learnable per-head-dim scale applied to Q after projection
        self.per_dim_scale = nn.Parameter([self._head_dim])

        # Relative position key projection: no bias (HF nn.Linear(..., bias=False))
        self.relative_k_proj = Linear(hidden_size, hidden_size, bias=False)

        # Precomputed sinusoidal relative position embeddings [context_left, hidden_size].
        # Positions are ordered [context_left-1, ..., 1, 0] (descending relative distance).
        # These are constants derived from hyperparameters — not part of the HF state dict.
        pos_embed = self._compute_pos_embeddings(attention_context_left, hidden_size)
        self.pos_embed = nn.Parameter(
            [attention_context_left, hidden_size],
            data=ir.Tensor(pos_embed, dtype=ir.DataType.FLOAT),
        )

    @staticmethod
    def _compute_pos_embeddings(context_left: int, hidden_size: int) -> np.ndarray:
        """Sinusoidal relative position embeddings [context_left, hidden_size]."""
        num_ts = hidden_size // 2
        log_ts_inc = math.log(10000.0) / max(num_ts - 1, 1)
        inv_ts = np.array([math.exp(-i * log_ts_inc) for i in range(num_ts)], dtype=np.float32)
        # Descending relative distances: [context_left-1, ..., 1, 0]
        pos_ids = np.arange(context_left - 1, -1, -1, dtype=np.float32).reshape(
            context_left, 1
        )
        scaled = pos_ids * inv_ts[None, :]  # [context_left, hidden_size//2]
        return np.concatenate(
            [np.sin(scaled), np.cos(scaled)], axis=-1
        )  # [context_left, hidden_size]

    def _build_causal_window_mask(self, op: builder.OpBuilder, seq_len: ir.Value) -> ir.Value:
        """Build [1, 1, T, T] causal sliding-window attention bias."""
        zero = op.Constant(value_int=0)
        one = op.Constant(value_int=1)
        positions = op.Range(zero, seq_len, one)  # [T] int64
        q_pos = op.Unsqueeze(positions, [1])  # [T, 1]
        k_pos = op.Unsqueeze(positions, [0])  # [1, T]
        diff = op.Sub(q_pos, k_pos)  # [T, T]: i - j

        # Causal: j ≤ i  ↔  diff ≥ 0
        causal = op.GreaterOrEqual(diff, zero)  # bool [T, T]
        # In window: j ≥ i - (context_left - 2)  ↔  diff < context_left - 1
        # HF uses left_window_size = attention_context_left - 1 (e.g. 12 for
        # config value 13) so the window covers positions [i-11, i] (12 frames).
        ctx = op.Constant(value_int=self._attention_context_left - 1)
        in_window = op.Less(diff, ctx)  # bool [T, T]

        allowed = op.And(causal, in_window)
        bias = op.Where(
            allowed,
            op.Constant(value_float=0.0),
            op.Constant(value_float=-1e9),
        )  # [T, T] float32
        return op.Unsqueeze(bias, [0, 1])  # [1, 1, T, T]

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: [B, T, hidden_size]
        batch = op.Shape(x, start=0, end=1)
        seq_len = op.Shape(x, start=1, end=2)
        num_heads = self._num_heads
        head_dim = self._head_dim

        # Q/K/V projections
        q = self.q_proj(op, x)  # [B, T, H*D]
        k = self.k_proj(op, x)
        v = self.v_proj(op, x)

        # Cast to float32 for numerically stable attention (matches HF .float())
        q = op.Cast(q, to=ir.DataType.FLOAT)
        k = op.Cast(k, to=ir.DataType.FLOAT)
        v = op.Cast(v, to=ir.DataType.FLOAT)

        # Reshape: [B, T, H*D] → [B, T, num_heads, head_dim]
        hd_shape = op.Concat(
            batch, seq_len, op.Constant(value_ints=[num_heads, head_dim]), axis=0
        )
        q = op.Reshape(q, hd_shape)  # [B, T, num_heads, head_dim]
        k = op.Reshape(k, hd_shape)
        v = op.Reshape(v, hd_shape)

        # Apply custom Q scale and per-dim scale: q *= q_scale * softplus(per_dim_scale)
        # Softplus(x) = log(1 + exp(x)); for numerical stability, cast scale param to fp32
        per_dim_f32 = op.Cast(self.per_dim_scale, to=ir.DataType.FLOAT)
        softplus_scale = op.Log(
            op.Add(op.Constant(value_float=1.0), op.Exp(per_dim_f32))
        )  # [head_dim] softplus(per_dim_scale)
        q_scale = op.Constant(value_float=self._q_scale)
        q = op.Mul(q, op.Mul(q_scale, softplus_scale))  # [B, T, num_heads, head_dim]

        # Apply K scale (constant)
        k_scale = op.Constant(value_float=self._k_scale)
        k = op.Mul(k, k_scale)  # [B, T, num_heads, head_dim]

        # Transpose to [B, num_heads, T, head_dim] for batched matmul
        q = op.Transpose(q, perm=[0, 2, 1, 3])  # [B, num_heads, T, head_dim]
        k = op.Transpose(k, perm=[0, 2, 1, 3])
        v = op.Transpose(v, perm=[0, 2, 1, 3])

        # Content attention scores: matrix_ac = Q @ K^T → [B, H, T, T]
        k_t = op.Transpose(k, perm=[0, 1, 3, 2])  # [B, H, D, T]
        scores = op.MatMul(q, k_t)  # [B, H, T, T]

        # Relative position bias: matrix_bd
        # pos_embed: [context_left, hidden_size] - project in model dtype, then cast
        rel_k = self.relative_k_proj(op, self.pos_embed)  # [context_left, num_heads*head_dim]
        rel_k = op.Cast(rel_k, to=ir.DataType.FLOAT)
        rel_k = op.Reshape(
            rel_k, op.Constant(value_ints=[-1, num_heads, head_dim])
        )  # [ctx, H, D]
        rel_k = op.Transpose(rel_k, perm=[1, 2, 0])  # [num_heads, head_dim, ctx]
        rel_k = op.Unsqueeze(rel_k, [0])  # [1, num_heads, head_dim, ctx]

        # rel_scores[b, h, t, d] = q[b, h, t, :] @ rel_k[h, :, d]
        rel_scores = op.MatMul(q, rel_k)  # [B, H, T, context_left]

        # Build offset matrix [T, T]: offset[i,j] = clip((ctx_left-1) - (i-j), 0, ctx_left-1)
        # pos_embed is ordered [ctx_left-1, ..., 0], so distance d maps to index ctx_left-1-d
        zero_i = op.Constant(value_int=0)
        one_i = op.Constant(value_int=1)
        positions = op.Range(zero_i, seq_len, one_i)  # [T]
        q_pos_i = op.Unsqueeze(positions, [1])  # [T, 1]
        k_pos_i = op.Unsqueeze(positions, [0])  # [1, T]
        diff_i = op.Sub(q_pos_i, k_pos_i)  # [T, T]: i-j
        ctx_m1 = op.Constant(value_int=self._attention_context_left - 1)
        ctx_val = op.Constant(value_int=self._attention_context_left - 1)
        offset_mat = op.Clip(op.Sub(ctx_m1, diff_i), zero_i, ctx_val)  # [T, T] int64

        # Expand offset_mat to [B, H, T, T] for GatherElements
        offset_2d = op.Unsqueeze(offset_mat, [0, 1])  # [1, 1, T, T]
        bh = op.Shape(scores, start=0, end=2)  # [B, H] as shape prefix
        tt = op.Concat(seq_len, seq_len, axis=0)  # [T, T] as shape suffix
        expand_shape = op.Concat(bh, tt, axis=0)  # [B, H, T, T]
        offset_expanded = op.Expand(offset_2d, expand_shape)  # [B, H, T, T]

        # Gather relative scores: rel_bias[b, h, i, j] = rel_scores[b, h, i, offset[i,j]]
        rel_bias = op.GatherElements(rel_scores, offset_expanded, axis=3)  # [B, H, T, T]

        scores = op.Add(scores, rel_bias)  # [B, H, T, T]

        # Soft-capping: tanh(scores / cap) * cap  [B, H, T, T]
        # Relative bias is already included in scores — this is why we cannot
        # use the ONNX Attention op's native softcap attribute (it fires before
        # any mask/bias addition in the op's fixed pipeline).
        cap = op.Constant(value_float=self._attention_logit_cap)
        scores = op.Mul(op.Tanh(op.Div(scores, cap)), cap)

        # Causal sliding-window mask (add bias: 0 for allowed, -1e9 for blocked)
        window_mask = self._build_causal_window_mask(op, seq_len)
        scores = op.Add(scores, window_mask)

        # Softmax in float32 → attention weights
        attn_weights = op.Softmax(scores, axis=-1)  # [B, H, T, T]

        # Weighted sum of values:
        # [B, H, T, T] @ [B, num_heads, T, head_dim] = [B, num_heads, T, head_dim]
        context = op.MatMul(attn_weights, v)  # [B, num_heads, T, head_dim]

        # Reshape [B, num_heads, T, head_dim] → [B, T, H*D]
        context = op.Transpose(context, perm=[0, 2, 1, 3])  # [B, T, num_heads, head_dim]
        out_shape = op.Concat(batch, seq_len, op.Constant(value_ints=[-1]), axis=0)
        context = op.Reshape(context, out_shape)  # [B, T, H*D]

        # Cast back to input dtype and apply output projection
        context = op.CastLike(context, x)
        return self.post(op, context)  # [B, T, hidden_size]


class Gemma4AudioLayer(nn.Module):
    """Single Gemma4 audio encoder layer.

    Matches ``Gemma4AudioLayer``::

        x = feed_forward1(x)              # FF + residual handled inside
        residual = x
        x = clip(x) → norm_pre_attn(x) → attention(x) → clip(x) → norm_post_attn(x)
        x = x + residual
        x = lconv1d(x)                    # LightConv1d + residual handled inside
        x = feed_forward2(x)              # FF + residual handled inside
        x = clip(x) → norm_out(x)        # final norm (no residual)

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        conv_kernel_size: Depthwise conv kernel size.
        attention_context_left: Causal sliding-window size.
        attention_logit_cap: Soft-capping value for attention logits.
        rms_norm_eps: Epsilon for all RMSNorm layers.
        residual_weight: Feed-forward output scale before adding residual.
        gradient_clipping: Activation clamp value.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 8,
        conv_kernel_size: int = 5,
        attention_context_left: int = 13,
        attention_logit_cap: float = 50.0,
        rms_norm_eps: float = 1e-6,
        residual_weight: float = 0.5,
        gradient_clipping: float = 1e9,
    ):
        super().__init__()
        self._gradient_clipping = gradient_clipping

        self.feed_forward1 = Gemma4FeedForward(
            hidden_size, rms_norm_eps, residual_weight, gradient_clipping
        )
        self.self_attn = Gemma4Attention(
            hidden_size, num_heads, attention_context_left, attention_logit_cap
        )
        self.lconv1d = Gemma4LightConv1d(
            hidden_size, conv_kernel_size, rms_norm_eps, gradient_clipping
        )
        self.feed_forward2 = Gemma4FeedForward(
            hidden_size, rms_norm_eps, residual_weight, gradient_clipping
        )
        self.norm_pre_attn = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm_post_attn = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm_out = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        gc = self._gradient_clipping
        x = self.feed_forward1(op, x)  # FF1 (handles residual internally)

        residual = x
        x = _gradient_clip(op, x, gc)
        x = self.norm_pre_attn(op, x)
        x = self.self_attn(op, x)
        x = _gradient_clip(op, x, gc)
        x = self.norm_post_attn(op, x)
        x = op.Add(x, residual)

        x = self.lconv1d(op, x)  # LightConv1d (handles residual internally)
        x = self.feed_forward2(op, x)  # FF2 (handles residual internally)

        x = _gradient_clip(op, x, gc)
        return self.norm_out(op, x)


class Gemma4AudioEncoder(nn.Module):
    """Gemma4 Conformer audio encoder (Universal Speech Model architecture).

    Used only by Gemma4 any-to-any variants (E2B and E4B).
    The Image-Text-to-Text variants (26B-A4B, 31B) have no audio encoder.

    Architecture::

        input_features [B, T, input_size]
        → Gemma4ConvSubsampling (4x time reduction)
        → N x Gemma4AudioLayer
        → Linear(hidden_size, output_proj_dims, bias=True)  [B, T//4, output_proj_dims]

    Default values match ``google/gemma-4-E2B-it`` audio_config.
    The output projection has bias=True in HF (``nn.Linear(..., bias=True)``), but
    since ORT fuses ``Add(1D bias) + SimplifiedLayerNorm`` into
    ``SkipSimplifiedLayerNorm`` with the 1D bias as the skip — which ORT rejects
    (requires 2D+) — we store the bias separately as ``output_proj_bias [1, 1, D]``
    and add it after the weight matmul.

    Args:
        input_size: Mel-spectrogram frequency bins.
        hidden_size: Conformer hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of encoder layers.
        conv_kernel_size: Depthwise conv kernel size in ``Gemma4LightConv1d``.
        conv_channels: Subsampling channel counts per stage.
        attention_context_left: Causal sliding-window size.
        attention_logit_cap: Soft-capping for attention logits.
        output_proj_dims: Output projection dimension (text model hidden dim).
        rms_norm_eps: Epsilon for all RMSNorm layers.
        residual_weight: Feed-forward residual scale.
        gradient_clipping: Activation clamp for numerical stability.
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 1024,
        num_heads: int = 8,
        num_layers: int = 12,
        conv_kernel_size: int = 5,
        conv_channels: list[int] | None = None,
        attention_context_left: int = 13,
        attention_logit_cap: float = 50.0,
        output_proj_dims: int = 1536,
        rms_norm_eps: float = 1e-6,
        residual_weight: float = 0.5,
        gradient_clipping: float = 1e9,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [128, 32]

        self.subsample_conv_projection = Gemma4ConvSubsampling(
            input_size, conv_channels, hidden_size, rms_norm_eps
        )
        self.layers = nn.ModuleList(
            [
                Gemma4AudioLayer(
                    hidden_size,
                    num_heads,
                    conv_kernel_size,
                    attention_context_left,
                    attention_logit_cap,
                    rms_norm_eps,
                    residual_weight,
                    gradient_clipping,
                )
                for _ in range(num_layers)
            ]
        )
        # HF uses nn.Linear(..., bias=True) for the output projection.
        # The bias would normally cause ORT to fuse Add(1D bias) + LayerNorm into
        # SkipSimplifiedLayerNorm (with 1D skip, which ORT rejects).  This is avoided
        # because _Gemma4ScaleFreeRMSNorm uses manual primitive ops rather than
        # op.RMSNormalization, preventing ORT from recognizing the fusion pattern.
        self.output_proj = Linear(hidden_size, output_proj_dims, bias=True)

    def forward(self, op: builder.OpBuilder, input_features: ir.Value):
        # input_features: [B, T, input_size]
        x = self.subsample_conv_projection(op, input_features)  # [B, T//4, hidden_size]

        for layer in self.layers:
            x = layer(op, x)  # [B, T//4, hidden_size]

        return self.output_proj(op, x)  # [B, T//4, output_proj_dims]
