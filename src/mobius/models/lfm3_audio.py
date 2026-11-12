# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LiquidAI LFM3 Audio spectrogram frontend."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import Lfm2Config
from mobius.components import Conv1d, Conv2d, SiLU, initialize_rope
from mobius.models.lfm2 import (
    Lfm2DecoderLayer,
    Lfm2RMSNorm,
    apply_lfm2_config_defaults,
)


class Lfm3AudioDownsampler(nn.Module):
    """Replicate HuggingFace Lfm3AudioDownsampler's masked stride-eight frontend.

    Inputs are float log-mel features (B, mel_bins, frames) and int64 valid
    frame lengths (B,). Outputs have shape (B, hidden_size, ceil(frames / 8)).
    Padding must be masked between convolutions, not only at graph entry:
    convolution biases otherwise leak padded activations into valid frames.
    """

    def __init__(self, num_mel_bins: int, conv_channels: int, hidden_size: int):
        super().__init__()
        if num_mel_bins <= 0 or num_mel_bins % 8:
            raise ValueError(
                f"LFM3 Audio num_mel_bins must be a positive multiple of 8, "
                f"got {num_mel_bins}; use the checkpoint's audio encoder configuration."
            )
        self.conv_in = Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1)
        self.conv_mid = Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=conv_channels,
        )
        self.lin_mid = Conv2d(conv_channels, conv_channels, kernel_size=1)
        self.conv_end = Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=conv_channels,
        )
        self.lin_end = Conv2d(conv_channels, conv_channels, kernel_size=1)
        self.lin_out = Conv1d(num_mel_bins * conv_channels // 8, hidden_size, kernel_size=1)
        self.act = SiLU()

    @staticmethod
    def _mask_padding(op: OpBuilder, hidden_states: ir.Value, lengths: ir.Value) -> ir.Value:
        frames = op.Squeeze(op.Shape(hidden_states, start=3, end=4), [0])
        indices = op.Range(0, frames, 1)
        valid = op.Less(op.Unsqueeze(indices, [0]), op.Unsqueeze(lengths, [1]))
        return op.Mul(hidden_states, op.CastLike(op.Unsqueeze(valid, [1, 2]), hidden_states))

    def forward(
        self, op: OpBuilder, input_features: ir.Value, feature_lengths: ir.Value
    ) -> ir.Value:
        hidden_states = self._mask_padding(
            op, op.Unsqueeze(input_features, [1]), feature_lengths
        )  # (B, 1, mel_bins, T)
        hidden_states = self.act(op, self.conv_in(op, hidden_states))
        lengths = op.Div(op.Add(feature_lengths, 1), 2)
        hidden_states = self._mask_padding(op, hidden_states, lengths)
        hidden_states = self.act(op, self.lin_mid(op, self.conv_mid(op, hidden_states)))
        lengths = op.Div(op.Add(lengths, 1), 2)
        hidden_states = self._mask_padding(op, hidden_states, lengths)
        hidden_states = self.act(op, self.lin_end(op, self.conv_end(op, hidden_states)))
        # Merge channel and mel axes without collapsing batch or time.
        shape = op.Concat(
            op.Shape(hidden_states, start=0, end=1),
            [-1],
            op.Shape(hidden_states, start=3, end=4),
            axis=0,
        )
        return self.lin_out(op, op.Reshape(hidden_states, shape))


class _Lfm3AudioEncoderBackbone(nn.Module):
    """LFM2 over downsampled frames with causal sliding-window attention."""

    def __init__(self, config: Lfm2Config, sliding_window: int):
        super().__init__()
        self.config = apply_lfm2_config_defaults(config)
        if (
            self.config.layer_types is None
            or len(self.config.layer_types) != self.config.num_hidden_layers
            or set(self.config.layer_types) - {"conv", "full_attention"}
        ):
            raise ValueError(
                "LFM3 Audio encoder requires one 'conv' or 'full_attention' layer type "
                "per layer; use the checkpoint's audio encoder LFM2 layer_types."
            )
        self.sliding_window = sliding_window
        self.layers = nn.ModuleList(
            [
                Lfm2DecoderLayer(self.config, index)
                for index in range(self.config.num_hidden_layers)
            ]
        )
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(self.config)
        if self.rotary_emb is None:
            raise ValueError(
                "LFM3 Audio encoder requires rotary position embeddings; "
                "use the checkpoint's LFM2 rope_parameters."
            )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        num_frames = op.Squeeze(op.Shape(hidden_states, start=1, end=2), [0])
        positions = op.Range(0, num_frames, 1)
        position_ids = op.Expand(
            op.Unsqueeze(positions, [0]), op.Shape(hidden_states, start=0, end=2)
        )
        assert self.rotary_emb is not None
        position_embeddings = self.rotary_emb(op, position_ids)
        distance = op.Sub(op.Unsqueeze(positions, [1]), op.Unsqueeze(positions, [0]))
        allowed = op.And(
            op.GreaterOrEqual(distance, 0), op.Less(distance, self.sliding_window)
        )
        attention_bias = op.Unsqueeze(allowed, [0, 1])  # (1, 1, T, T)
        attention_mask = op.Expand(
            op.Constant(value_int=1), op.Shape(hidden_states, start=0, end=2)
        )
        layer_types = self.config.layer_types
        assert layer_types is not None
        for layer, kind in zip(self.layers, layer_types, strict=True):
            if kind == "conv":
                state_shape = op.Concat(
                    op.Shape(hidden_states, start=0, end=1),
                    [self.config.hidden_size, self.config.short_conv_kernel - 1],
                    axis=0,
                )
                state = op.Expand(op.CastLike(0.0, hidden_states), state_shape)
                past = (state,)
            else:
                # No cache is reused in the offline encoder.
                past = None
            hidden_states, _ = layer(
                op,
                hidden_states,
                attention_mask,
                attention_bias,
                position_embeddings,
                past,
            )
        return self.embedding_norm(op, hidden_states)


class Lfm3AudioEncoder(nn.Module):
    """Offline HuggingFace Lfm3AudioEncoder: log-mel frames to causal features.

    Inputs are (B, mel_bins, T) float32 log-mel features and (B,) int64
    valid lengths. Outputs are (B, ceil(T / 8), hidden_size) features and
    (B,) downsampled lengths; callers trim padding before placeholder fusion.
    """

    def __init__(
        self,
        config: Lfm2Config,
        *,
        num_mel_bins: int = 128,
        conv_channels: int = 256,
        sliding_window: int = 32,
    ):
        super().__init__()
        if sliding_window <= 0:
            raise ValueError(
                f"LFM3 Audio encoder sliding_window must be positive, got {sliding_window}; "
                "use the checkpoint's audio encoder configuration."
            )
        if config.dtype != ir.DataType.FLOAT:
            raise ValueError(
                "LFM3 Audio encoder currently requires float32 compute; "
                "construct its Lfm2Config with dtype=ir.DataType.FLOAT."
            )
        self.downsample = Lfm3AudioDownsampler(num_mel_bins, conv_channels, config.hidden_size)
        self.lfm = _Lfm3AudioEncoderBackbone(config, sliding_window)

    def forward(
        self, op: OpBuilder, input_features: ir.Value, feature_lengths: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        if input_features.dtype != ir.DataType.FLOAT:
            raise ValueError(
                "LFM3 Audio encoder input_features must be float32 log-mel features; "
                "retain the processor's float32 output at the graph boundary."
            )
        hidden_states = self.downsample(op, input_features, feature_lengths)
        hidden_states = self.lfm(op, op.Transpose(hidden_states, perm=[0, 2, 1]))
        return hidden_states, op.Div(op.Add(feature_lengths, 7), 8)
