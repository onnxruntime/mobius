# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Moonshine Streaming raw-waveform encoder-decoder model for speech recognition.

Replicates Hugging Face ``MoonshineStreamingForConditionalGeneration`` as separate
encoder and cached-decoder ONNX graphs.

Compared with offline Moonshine the streaming variant changes the whole audio
front end and the encoder's positional scheme:

* **Framing front end** — the waveform is reshaped into fixed ``frame_ms`` frames
  (80 raw samples at 16 kHz / 5 ms), per-frame mean/RMS normalised (CMVN),
  compressed with a learned ``asinh(exp(log_k) * x)`` gain, and projected by a
  single bias-free linear layer.
* **Causal downsampling** — two left-padded stride-2 convolutions replace
  Moonshine's centred convolution stem, so no future frame ever leaks backwards.
* **No encoder RoPE** — encoder self-attention is purely content based; ordering
  comes from the causal stem plus per-layer asymmetric ``(left, right)`` sliding
  windows. ``right`` is the strict lookahead in encoder frames and is ``0`` for
  the fully causal layers, which is what bounds streaming latency.
* **Unit-offset LayerNorm** — encoder norms are affine-free ``LayerNorm`` scaled
  by ``gamma + 1``.
* **Context adapter** — the decoder adds a learned absolute position table
  (``pos_emb``) to the encoder output, then optionally projects it to the decoder
  width, before cross-attention.

The decoder itself (partial interleaved RoPE, cached causal self-attention,
cross-attention, fused gate/up SiLU MLP, bias-free LayerNorms) is identical to
offline Moonshine and is reused directly.
"""

from __future__ import annotations

from typing import overload

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import MoonshineStreamingConfig
from mobius.components import Conv1d, Embedding, Linear, SiLU, get_activation
from mobius.models.moonshine import (
    MoonshineAttention,
    MoonshineDecoderModel,
    MoonshineForConditionalGeneration,
)


class MoonshineStreamingLayerNorm(nn.Module):
    """Affine-free LayerNorm scaled by ``gamma + 1``.

    Mirrors HF ``MoonshineStreamingLayerNorm``: an ``nn.LayerNorm`` with
    ``elementwise_affine=False`` followed by a multiplication with
    ``gamma + unit_offset``. The checkpoint therefore stores ``gamma`` (not
    ``weight``), initialised to zero-mean around the unit scale.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter([hidden_size])
        self._eps = eps

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # LayerNorm without affine followed by ``* (gamma + 1)`` is exactly
        # LayerNormalization with scale = gamma + 1 and no bias.
        return op.LayerNormalization(
            hidden_states,
            op.Add(self.gamma, 1.0),
            None,
            epsilon=self._eps,
            axis=-1,
        )


class MoonshineStreamingAsinhCompression(nn.Module):
    """Learned log-domain gain followed by ``asinh`` dynamic-range compression."""

    def __init__(self):
        super().__init__()
        # Scalar parameter; kept in float32 because it feeds the float32
        # normalisation stage of the front end.
        self.log_k = nn.Parameter([])
        self.log_k._keep_float32 = True

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return op.Asinh(op.Mul(op.Exp(self.log_k), hidden_states))


class MoonshineStreamingCausalConv1d(Conv1d):
    """Left-padded strided 1-D convolution that also carries the frame mask.

    HF ``MoonshineStreamingCausalConv1d`` pads the input by
    ``(kernel_size - 1) * dilation`` on the left only, so an output frame never
    depends on a future input frame. The padding mask is pushed through the same
    receptive field with an "any valid" reduction and used to zero invalid output
    frames.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1, 0),
            bias=True,
        )
        self._left_pad = kernel_size - 1
        self._window = kernel_size
        self._stride = stride

    @overload
    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value: ...

    @overload
    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        mask: ir.Value,
        dtype: ir.DataType,
    ) -> tuple[ir.Value, ir.Value]: ...

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        mask: ir.Value | None = None,
        dtype: ir.DataType = ir.DataType.FLOAT,
    ) -> ir.Value | tuple[ir.Value, ir.Value]:
        """Run the convolution and downsample the mask.

        Args:
            hidden_states: ``(B, C_in, T)`` channel-first frames.
            mask: ``(B, T)`` bool frame-validity mask.
            dtype: Compute dtype used to zero masked output frames.

        Returns:
            ``((B, C_out, T_out), (B, T_out))`` hidden states and mask.
        """
        hidden_states = super().forward(op, hidden_states)  # (B, C_out, T_out)
        if mask is None:
            return hidden_states

        # An output frame is valid when *any* input frame in its (left-padded)
        # receptive field is valid — max-pooling over the same window with the
        # same stride is exactly HF's ``conv1d(mask, ones) > 0``.
        mask_values = op.Cast(op.Unsqueeze(mask, [1]), to=ir.DataType.FLOAT)  # (B, 1, T)
        mask_values = op.Pad(mask_values, [0, 0, self._left_pad, 0, 0, 0])
        mask_values = op.MaxPool(
            mask_values, kernel_shape=[self._window], strides=[self._stride]
        )
        mask_4d = op.Greater(mask_values, 0.0)  # (B, 1, T_out) bool
        hidden_states = op.Mul(hidden_states, op.Cast(mask_4d, to=dtype))
        return hidden_states, op.Squeeze(mask_4d, [1])


class MoonshineStreamingEncoderEmbedder(nn.Module):
    """Raw-waveform framing front end: CMVN, asinh compression, causal convs."""

    def __init__(self, config: MoonshineStreamingConfig):
        super().__init__()
        hidden_size = config.encoder_hidden_size
        self.comp = MoonshineStreamingAsinhCompression()
        self.conv1 = MoonshineStreamingCausalConv1d(
            hidden_size, 2 * hidden_size, kernel_size=5, stride=2
        )
        self.conv2 = MoonshineStreamingCausalConv1d(
            2 * hidden_size, hidden_size, kernel_size=5, stride=2
        )
        self.linear = Linear(config.frame_length, hidden_size, bias=False)
        self.activation = SiLU()
        self._frame_length = config.frame_length
        self._cmvn_eps = config.encoder_cmvn_eps
        self._dtype = config.dtype

    def forward(
        self,
        op: OpBuilder,
        input_values: ir.Value,
        attention_mask: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        # Raw audio (B, L) -> (B, T, frame_len). ``L`` must be a multiple of
        # ``frame_len``; the processor enforces this with pad_to_multiple_of.
        frames = op.Reshape(input_values, [0, -1, self._frame_length])

        # Per-frame CMVN and the asinh gain run in float32: squared waveform
        # amplitudes are ~1e-6 and would collapse into float16 subnormals.
        frames = op.Cast(frames, to=ir.DataType.FLOAT)
        mean = op.ReduceMean(frames, [-1], keepdims=1)  # (B, T, 1)
        centered = op.Sub(frames, mean)
        variance = op.ReduceMean(op.Mul(centered, centered), [-1], keepdims=1)
        normalized = op.Div(centered, op.Sqrt(op.Add(variance, self._cmvn_eps)))
        hidden_states = self.comp(op, normalized)  # (B, T, frame_len)
        if self._dtype != ir.DataType.FLOAT:
            hidden_states = op.Cast(hidden_states, to=self._dtype)
        hidden_states = self.activation(op, self.linear(op, hidden_states))  # (B, T, D)

        # Only frames fully covered by real samples are valid; HF uses integer
        # division of the sample-level mask by the frame length.
        valid_frames = op.Div(
            op.ReduceSum(attention_mask, [-1], keepdims=1), self._frame_length
        )  # (B, 1)
        frame_count = op.Shape(hidden_states, start=1, end=2)
        positions = op.Unsqueeze(op.Range(0, op.Squeeze(frame_count, [0]), 1), [0])
        frame_mask = op.Less(positions, valid_frames)  # (B, T) bool
        hidden_states = op.Mul(
            hidden_states, op.Cast(op.Unsqueeze(frame_mask, [2]), to=self._dtype)
        )

        # Channel-first for the convolutions; each halves the frame rate.
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])  # (B, D, T)
        hidden_states, frame_mask = self.conv1(op, hidden_states, frame_mask, self._dtype)
        hidden_states = self.activation(op, hidden_states)
        hidden_states, frame_mask = self.conv2(op, hidden_states, frame_mask, self._dtype)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])  # (B, T2, D)
        return hidden_states, frame_mask


class MoonshineStreamingEncoderMLP(nn.Module):
    """Feed-forward network of an encoder layer (``fc1`` -> act -> ``fc2``)."""

    def __init__(self, config: MoonshineStreamingConfig):
        super().__init__()
        self.fc1 = Linear(config.encoder_hidden_size, config.encoder_intermediate_size)
        self.fc2 = Linear(config.encoder_intermediate_size, config.encoder_hidden_size)
        self._activation = get_activation(config.encoder_hidden_act)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return self.fc2(op, self._activation(op, self.fc1(op, hidden_states)))


class MoonshineStreamingEncoderLayer(nn.Module):
    """Pre-norm encoder layer with unit-offset norms and no positional rotation."""

    def __init__(self, config: MoonshineStreamingConfig):
        super().__init__()
        self.input_layernorm = MoonshineStreamingLayerNorm(
            config.encoder_hidden_size, eps=config.layer_norm_eps
        )
        self.self_attn = MoonshineAttention(
            config,
            num_heads=config.encoder_num_attention_heads,
            num_key_value_heads=config.encoder_num_key_value_heads,
            hidden_size=config.encoder_hidden_size,
            head_dim=config.encoder_head_dim,
            # Upstream gates every encoder projection — including o_proj — on
            # the encoder sub-config's attention_bias.
            qkv_bias=config.encoder_attention_bias,
            o_bias=config.encoder_attention_bias,
        )
        self.post_attention_layernorm = MoonshineStreamingLayerNorm(
            config.encoder_hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = MoonshineStreamingEncoderMLP(config)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
    ) -> ir.Value:
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        # Streaming encoder attention carries no RoPE — the sliding window and
        # the causal convolution stem supply all positional structure.
        hidden_states, _ = self.self_attn(op, hidden_states, attention_bias=attention_bias)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        return op.Add(residual, hidden_states)


def _sliding_window_attention_bias(
    op: OpBuilder,
    frame_mask: ir.Value,
    frame_count: ir.Value,
    window: tuple[int, int],
    dtype: ir.DataType,
) -> ir.Value:
    """Additive ``[B, 1, T, T]`` bias for one asymmetric bidirectional window.

    HF combines the padding mask with
    ``(0 <= q - kv < left) or (0 < kv - q < right)``. Expressed as bounds on
    ``dist = q - kv`` that is ``-(right - 1) <= dist <= left - 1``, with the
    lower bound clamped to ``0`` when ``right == 0`` (fully causal layer).

    Args:
        frame_mask: ``(B, T)`` bool mask of valid encoder frames.
        frame_count: ``(1,)`` int64 encoder sequence length.
        window: ``(left, right)`` window of the layer.
        dtype: Compute dtype of the emitted bias.
    """
    left, right = window
    lower = -(right - 1) if right >= 1 else 0
    upper = left - 1

    positions = op.Range(0, op.Squeeze(frame_count, [0]), 1)  # (T,)
    distance = op.Sub(op.Unsqueeze(positions, [1]), op.Unsqueeze(positions, [0]))  # (T, T)
    in_window = op.And(
        op.GreaterOrEqual(distance, lower),
        op.LessOrEqual(distance, upper),
    )
    # (1, 1, T, T) window AND (B, 1, 1, T) key validity -> (B, 1, T, T).
    allowed = op.And(op.Unsqueeze(in_window, [0, 1]), op.Unsqueeze(frame_mask, [1, 2]))
    return op.Cast(op.Where(allowed, 0.0, float(dtype.min)), to=dtype)


class MoonshineStreamingEncoderModel(nn.Module):
    """Streaming audio encoder: framing front end plus windowed transformer."""

    def __init__(self, config: MoonshineStreamingConfig):
        super().__init__()
        self.embedder = MoonshineStreamingEncoderEmbedder(config)
        self.layers = nn.ModuleList(
            [
                MoonshineStreamingEncoderLayer(config)
                for _ in range(config.encoder_num_hidden_layers)
            ]
        )
        self.final_norm = MoonshineStreamingLayerNorm(
            config.encoder_hidden_size, eps=config.layer_norm_eps
        )
        self._windows = config.encoder_sliding_windows
        self._dtype = config.dtype

    def forward(
        self,
        op: OpBuilder,
        input_values: ir.Value,
        attention_mask: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        hidden_states, frame_mask = self.embedder(op, input_values, attention_mask)
        frame_count = op.Shape(hidden_states, start=1, end=2)

        # Layers share only a handful of distinct windows, so build each bias
        # once and reuse it across the layers that declare it.
        biases: dict[tuple[int, int], ir.Value] = {}
        for window in self._windows:
            if window not in biases:
                biases[window] = _sliding_window_attention_bias(
                    op, frame_mask, frame_count, window, self._dtype
                )

        for layer, window in zip(self.layers, self._windows):
            hidden_states = layer(op, hidden_states, biases[window])

        hidden_states = self.final_norm(op, hidden_states)
        encoder_attention_mask = op.Cast(frame_mask, to=ir.DataType.INT64)
        return hidden_states, encoder_attention_mask


class MoonshineStreamingDecoderModel(MoonshineDecoderModel):
    """Moonshine decoder with the streaming context adapter on encoder states."""

    def __init__(self, config: MoonshineStreamingConfig):
        super().__init__(config)
        # Absolute learned positions added to the encoder output. The table is
        # sized by the decoder's ``max_position_embeddings`` but indexed by the
        # encoder frame position, exactly as upstream does.
        self.pos_emb = Embedding(config.max_position_embeddings, config.encoder_hidden_size)
        if config.encoder_hidden_size != config.hidden_size:
            self.proj = Linear(config.encoder_hidden_size, config.hidden_size, bias=False)
        else:
            # HF uses ``nn.Identity`` here, which contributes no checkpoint entry.
            self.proj = None

    def forward(
        self,
        op: OpBuilder,
        decoder_input_ids: ir.Value,
        encoder_hidden_states: ir.Value,
        encoder_attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list[tuple[ir.Value, ir.Value]] | None = None,
    ) -> tuple[ir.Value, list[tuple[ir.Value, ir.Value]]]:
        # Context adapter: encoder_hidden_states (B, E, De) + pos_emb[0:E]
        encoder_length = op.Shape(encoder_hidden_states, start=1, end=2)
        encoder_positions = op.Range(0, op.Squeeze(encoder_length, [0]), 1)  # (E,)
        encoder_hidden_states = op.Add(
            encoder_hidden_states, self.pos_emb(op, encoder_positions)
        )
        if self.proj is not None:
            encoder_hidden_states = self.proj(op, encoder_hidden_states)  # (B, E, D)

        return super().forward(
            op,
            decoder_input_ids,
            encoder_hidden_states,
            encoder_attention_mask,
            position_ids,
            past_key_values,
        )


class _MoonshineStreamingModel(nn.Module):
    def __init__(self, config: MoonshineStreamingConfig):
        super().__init__()
        self.encoder = MoonshineStreamingEncoderModel(config)
        self.decoder = MoonshineStreamingDecoderModel(config)


class MoonshineStreamingForConditionalGeneration(MoonshineForConditionalGeneration):
    """Moonshine Streaming encoder-decoder model for low-latency ASR.

    Replicates Hugging Face ``MoonshineStreamingForConditionalGeneration``: a
    causal framing/convolution front end feeding a windowed (bounded-lookahead)
    transformer encoder, and a cached Moonshine decoder that cross-attends to the
    position-adapted encoder context.
    """

    default_task: str = "speech-to-text"
    category: str = "Speech-to-Text"
    config_class: type = MoonshineStreamingConfig

    def __init__(self, config: MoonshineStreamingConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = _MoonshineStreamingModel(config)
        self.proj_out = self.model.decoder.proj_out


__all__ = ["MoonshineStreamingForConditionalGeneration"]
