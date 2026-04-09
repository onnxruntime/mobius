# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma4 audio encoder components (Conformer-style).

Implements the Conformer-based audio encoder for Gemma4 multimodal models
(E2B and E4B any-to-any variants). Architecture derived from Gemma4 audio
config (model_type=gemma4_audio):

  - ``Gemma4ConvSubsampling``: 2-stage Conv2d subsampling (4x time reduction),
    conv_channels=[128, 32], SiLU activation.
  - ``Gemma4CausalChunkedAttention``: multi-head self-attention with a causal
    sliding-window mask; each position attends to at most
    ``attention_context_left`` frames to its left (default 13, right=0).
  - ``Gemma4ConformerEncoderLayer``: Macaron Conformer block using RMSNorm
    (not LayerNorm) and causal sliding-window attention.
  - ``Gemma4AudioEncoder``: 12-layer encoder with a linear output projection
    (hidden_size=1024 → output_proj_dims=1536).

Default config values match google/gemma-4-E2B-it::

    hidden_size=1024, num_attention_heads=8, num_hidden_layers=12,
    subsampling_conv_channels=[128, 32], conv_kernel_size=5,
    attention_context_left=13, output_proj_dims=1536, rms_norm_eps=1e-6

Note on ``attention_logit_cap=50.0`` (from config): Gemma4 applies soft-capping
``tanh(logits / cap) * cap`` to raw attention logits. The ONNX opset-23
``Attention`` op does not expose raw logits, so this cap is not applied here.
A custom attention kernel would be needed for exact numerical parity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._audio import (
    ConformerConvModule,
    ConformerFeedForward,
    _Conv2d,  # private: local to components package
    _swish,   # private: SiLU/Swish activation helper
)
from mobius.components._common import Linear
from mobius.components._rms_norm import RMSNorm

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# Private norm-swapped variants (RMSNorm in place of LayerNorm)
# ---------------------------------------------------------------------------


class _Gemma4FeedForward(ConformerFeedForward):
    """ConformerFeedForward with RMSNorm in place of LayerNorm.

    Gemma4 uses RMSNorm (rms_norm_eps=1e-6) throughout; replacing the base
    class's LayerNorm keeps all other feed-forward logic identical.
    """

    def __init__(self, d_model: int, d_inner: int, rms_norm_eps: float = 1e-6):
        super().__init__(d_model, d_inner)
        self.layer_norm = RMSNorm(d_model, eps=rms_norm_eps)


class _Gemma4ConvModule(ConformerConvModule):
    """ConformerConvModule with RMSNorm in place of LayerNorm."""

    def __init__(self, channels: int, kernel_size: int, rms_norm_eps: float = 1e-6):
        super().__init__(channels, kernel_size)
        self.layer_norm = RMSNorm(channels, eps=rms_norm_eps)


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------


class Gemma4ConvSubsampling(nn.Module):
    """2-stage Conv2d subsampling for Gemma4 audio features.

    Two stride-2 Conv2d stages reduce the time and frequency dimensions by
    4x each.  Channel counts for the two stages are configured separately
    to match Gemma4's ``subsampling_conv_channels=[128, 32]`` config field.
    SiLU (swish) activation is used between stages, matching ``hidden_act=silu``.

    Structure::

        [B, T, input_size]
        → unsqueeze → [B, 1, T, input_size]
        → Conv2d(1→c0, stride=2, pad=1) + SiLU → [B, c0, T//2, F//2]
        → Conv2d(c0→c1, stride=2, pad=1) + SiLU → [B, c1, T//4, F//4]
        → transpose + flatten → [B, T//4, c1 * F//4]
        → Linear(c1 * F//4, hidden_size) → [B, T//4, hidden_size]

    Args:
        input_size: Number of input mel-spectrogram bins (e.g., 128).
        conv_channels: Two-element list ``[c0, c1]`` — output channels for
            stage-1 and stage-2 convolutions (Gemma4 default: [128, 32]).
        hidden_size: Output feature dimension after the linear projection.
    """

    def __init__(
        self,
        input_size: int,
        conv_channels: list[int],
        hidden_size: int,
    ):
        super().__init__()
        c0, c1 = conv_channels

        # Frequency dimension after each stride-2 conv with pad=1, kernel=3:
        #   F_out = (F_in + 2*pad - kernel) // stride + 1 = (F + 2 - 3) // 2 + 1
        freq = input_size
        for _ in range(2):
            freq = (freq - 3 + 2) // 2 + 1

        self.conv0 = _Conv2d(1, c0, kernel_size=3, stride=2, padding=1)
        self.conv1 = _Conv2d(c0, c1, kernel_size=3, stride=2, padding=1)
        self.out = Linear(c1 * freq, hidden_size)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: [B, T, input_size]
        x = op.Unsqueeze(x, [1])  # [B, 1, T, input_size]

        x = self.conv0(op, x)  # [B, c0, T//2, F//2]
        x = _swish(op, x)      # SiLU: x * sigmoid(x)

        x = self.conv1(op, x)  # [B, c1, T//4, F//4]
        x = _swish(op, x)

        # [B, c1, T', F'] → [B, T', c1*F']
        x = op.Transpose(x, perm=[0, 2, 1, 3])
        x = op.Reshape(x, [0, 0, -1])

        return self.out(op, x)  # [B, T', hidden_size]


# ---------------------------------------------------------------------------
# Causal sliding-window attention
# ---------------------------------------------------------------------------


class Gemma4CausalChunkedAttention(nn.Module):
    """Causal sliding-window self-attention for the Gemma4 audio encoder.

    Each position can only attend to positions within a left context window
    of size ``attention_context_left`` (including itself) and no future frames
    (``attention_context_right=0``).  This matches Gemma4's audio config::

        attention_context_left=13, attention_context_right=0

    The mask is computed dynamically from the runtime sequence length and cast
    to match the QKV compute dtype for mixed-precision safety.

    Note: ``attention_logit_cap=50.0`` in the Gemma4 audio config applies
    ``tanh(logits / 50) * 50`` soft-capping to raw QK scores.  This cannot
    be expressed through the ONNX opset-23 ``Attention`` op's interface and
    is omitted here.  Integration tests may show small numerical divergence.

    Args:
        d_model: Model (hidden) dimension.
        num_heads: Number of attention heads.
        attention_context_left: Sliding window size including the current
            position (Gemma4 default: 13 → attends to 12 past + 1 current).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attention_context_left: int = 13,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self._scale = float(self._head_dim) ** -0.5
        self._attention_context_left = attention_context_left

        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model)
        self.linear_v = Linear(d_model, d_model)
        self.linear_out = Linear(d_model, d_model)

    def _build_causal_sliding_window_mask(
        self, op: builder.OpBuilder, seq_len: ir.Value
    ) -> ir.Value:
        """Build a float attention bias of shape [1, 1, T, T].

        Positions where attention is **allowed** (i - (context_left-1) ≤ j ≤ i)
        receive bias 0.0; blocked positions receive -1e9 (effectively −∞).

        The mask is causal (j ≤ i) and windowed (j ≥ i − context_left + 1):
            diff = i − j ∈ [0, context_left − 1]  →  allowed
        """
        zero = op.Constant(value_int=0)
        one = op.Constant(value_int=1)
        positions = op.Range(zero, seq_len, one)  # [T] int64

        q_pos = op.Unsqueeze(positions, [1])  # [T, 1]
        k_pos = op.Unsqueeze(positions, [0])  # [1, T]
        diff = op.Sub(q_pos, k_pos)  # [T, T]:  diff[i,j] = i - j

        # Causal: j ≤ i  ↔  diff ≥ 0
        causal = op.GreaterOrEqual(diff, zero)  # [T, T] bool

        # In window: j ≥ i − (context_left − 1)  ↔  diff < context_left
        context_left_const = op.Constant(value_int=self._attention_context_left)
        in_window = op.Less(diff, context_left_const)  # [T, T] bool

        allowed = op.And(causal, in_window)  # [T, T] bool

        # Bias: 0.0 where attending is allowed, -1e9 (masked) elsewhere
        bias = op.Where(
            allowed,
            op.Constant(value_float=0.0),
            op.Constant(value_float=-1e9),
        )  # [T, T] float32

        return op.Unsqueeze(bias, [0, 1])  # [1, 1, T, T]

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: [B, T, d_model]
        q = self.linear_q(op, x)
        k = self.linear_k(op, x)
        v = self.linear_v(op, x)

        # Dynamic causal sliding-window mask, cast to match QKV compute dtype
        seq_len = op.Shape(x, start=1, end=2)
        attn_bias = self._build_causal_sliding_window_mask(op, seq_len)
        attn_bias = op.CastLike(attn_bias, q)  # [1, 1, T, T] in QKV dtype

        attn_output = op.Attention(
            q,
            k,
            v,
            attn_bias,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            scale=self._scale,
            _outputs=1,
        )  # [B, T, d_model]

        return self.linear_out(op, attn_output)


# ---------------------------------------------------------------------------
# Conformer encoder layer
# ---------------------------------------------------------------------------


class Gemma4ConformerEncoderLayer(nn.Module):
    """Single Conformer encoder layer for the Gemma4 audio encoder.

    Macaron structure with causal sliding-window attention and RMSNorm::

        x += residual_weight * feed_forward_in(x)     # pre-norm inside FF
        x += causal_sliding_window_attn(rms_norm(x))
        x += conv(x)                                   # pre-norm inside conv
        x += residual_weight * feed_forward_out(x)
        x  = rms_norm(x)

    The ``residual_weight=0.5`` from the Gemma4 config matches the standard
    Macaron feed-forward half-step scaling.

    Args:
        d_model: Model (hidden) dimension.
        num_heads: Number of attention heads.
        d_inner: Inner dimension of Macaron feed-forward modules.
        conv_kernel_size: Depthwise conv kernel size (Gemma4 default: 5).
        attention_context_left: Sliding window for causal attention
            (Gemma4 default: 13).
        rms_norm_eps: Epsilon for RMSNorm layers (Gemma4 default: 1e-6).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_inner: int,
        conv_kernel_size: int,
        attention_context_left: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.feed_forward_in = _Gemma4FeedForward(d_model, d_inner, rms_norm_eps)
        self.self_attn = Gemma4CausalChunkedAttention(
            d_model, num_heads, attention_context_left
        )
        self.conv = _Gemma4ConvModule(d_model, conv_kernel_size, rms_norm_eps)
        self.feed_forward_out = _Gemma4FeedForward(d_model, d_inner, rms_norm_eps)
        self.layer_norm_att = RMSNorm(d_model, eps=rms_norm_eps)
        self.layer_norm = RMSNorm(d_model, eps=rms_norm_eps)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # residual_weight=0.5 matches Gemma4 config field
        half = op.Constant(value_float=0.5)

        # Macaron feed-forward in (pre-norm is inside _Gemma4FeedForward)
        x = op.Add(x, op.Mul(self.feed_forward_in(op, x), half))

        # Causal sliding-window self-attention with pre-RMSNorm
        norm_x = self.layer_norm_att(op, x)
        x = op.Add(x, self.self_attn(op, norm_x))

        # Convolution module (pre-norm is inside _Gemma4ConvModule)
        x = op.Add(x, self.conv(op, x))

        # Macaron feed-forward out
        x = op.Add(x, op.Mul(self.feed_forward_out(op, x), half))

        return self.layer_norm(op, x)


# ---------------------------------------------------------------------------
# Top-level encoder
# ---------------------------------------------------------------------------


class Gemma4AudioEncoder(nn.Module):
    """Gemma4 Conformer audio encoder.

    Combines:
    1. 2-stage Conv2d subsampling (4× time reduction, channels ``[128, 32]``,
       SiLU activation)
    2. ``num_layers`` Conformer blocks with causal sliding-window attention
       and RMSNorm
    3. A linear projection from ``hidden_size`` to ``output_proj_dims``

    This encoder is used only by Gemma4 any-to-any variants (E2B, E4B).
    The Image-Text-to-Text variants (26B-A4B, 31B) have no audio encoder.

    Default configuration matches ``google/gemma-4-E2B-it`` audio config::

        hidden_size=1024, num_attention_heads=8, num_hidden_layers=12,
        subsampling_conv_channels=[128, 32], conv_kernel_size=5,
        attention_context_left=13, output_proj_dims=1536, rms_norm_eps=1e-6

    Input:  ``[B, T, input_size]``  (mel spectrogram; default 128-dim bins)
    Output: ``[B, T//4, output_proj_dims]``  (default 1536-dim)

    Args:
        input_size: Input mel-spectrogram bins.
        hidden_size: Conformer hidden dimension.
        num_heads: Number of attention heads (``head_dim = hidden_size // num_heads``).
        num_layers: Number of Conformer encoder layers.
        ffn_inner_size: Inner dimension of Macaron feed-forward modules.
        conv_kernel_size: Depthwise conv kernel size (Gemma4: 5).
        conv_channels: ``[c0, c1]`` subsampling channel counts.
        attention_context_left: Causal sliding-window size (Gemma4: 13).
        rms_norm_eps: Epsilon for all RMSNorm layers.
        output_proj_dims: Output projection dimension; should match the text
            model's hidden dimension (Gemma4: 1536).
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 1024,
        num_heads: int = 8,
        num_layers: int = 12,
        ffn_inner_size: int = 4096,
        conv_kernel_size: int = 5,
        conv_channels: list[int] | None = None,
        attention_context_left: int = 13,
        rms_norm_eps: float = 1e-6,
        output_proj_dims: int = 1536,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [128, 32]

        self.subsampling = Gemma4ConvSubsampling(input_size, conv_channels, hidden_size)

        self.encoders = nn.ModuleList(
            [
                Gemma4ConformerEncoderLayer(
                    hidden_size,
                    num_heads,
                    ffn_inner_size,
                    conv_kernel_size,
                    attention_context_left,
                    rms_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        # Bridge Conformer dim → text model hidden dim
        self.output_projection = Linear(hidden_size, output_proj_dims)

    def forward(self, op: builder.OpBuilder, input_features: ir.Value):
        # input_features: [B, T, input_size]
        x = self.subsampling(op, input_features)  # [B, T//4, hidden_size]

        for layer in self.encoders:
            x = layer(op, x)  # [B, T//4, hidden_size]

        return self.output_projection(op, x)  # [B, T//4, output_proj_dims]
