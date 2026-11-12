# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline LiquidCodec synthesis from discrete codes to 24 kHz waveform.

This module implements the decoder bundled in ``LiquidAI/LFM3-Audio-2B``:
12 finite-scalar-quantizer codebooks are projected into an eight-layer LFM3
backbone, converted to complex STFT frames, and overlap-added into audio.
Streaming state is intentionally outside this standalone decoder contract.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import numpy as np
import onnx_ir as ir
from onnxscript import GraphBuilder, OpBuilder, nn

from mobius._configs import Lfm3Config
from mobius._constants import OPSET_VERSION
from mobius.components import Linear
from mobius.models.lfm3 import Lfm3TextModel, _sliding_mask


@dataclasses.dataclass
class LiquidCodecConfig:
    """Configuration of the released LiquidAI 24 kHz codec decoder."""

    encoder_dim: int = 512
    quantizer_dim: int = 256
    fsq_levels: tuple[int, ...] = (8, 7, 6, 6)
    n_q: int = 12
    decoder_window: int = 30
    stft_n_fft: int = 1280
    stft_hop_length: int = 320
    stft_win_length: int = 1280
    stft_frame_stack: int = 6
    dtype: ir.DataType = ir.DataType.FLOAT

    @property
    def codebook_size(self) -> int:
        return math.prod(self.fsq_levels)

    @classmethod
    def from_transformers(cls, config) -> LiquidCodecConfig:
        """Create a decoder config from the pinned HF codec config."""
        return cls(
            encoder_dim=config.encoder_dim,
            quantizer_dim=config.quantizer_dim,
            fsq_levels=tuple(config.fsq_levels),
            n_q=config.n_q,
            decoder_window=config.decoder_window,
            stft_n_fft=config.stft_n_fft,
            stft_hop_length=config.stft_hop_length,
            stft_win_length=config.stft_win_length,
            stft_frame_stack=config.stft_frame_stack,
        )

    def lfm_config(self) -> Lfm3Config:
        """Return the exact synthesis-backbone configuration."""
        layer_types = ["sliding_attention" if i in (2, 4, 6) else "conv" for i in range(8)]
        return Lfm3Config(
            vocab_size=1,
            hidden_size=self.encoder_dim,
            intermediate_size=2304,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=131_072,
            rms_norm_eps=1e-5,
            short_conv_kernel=7,
            short_conv_bias=False,
            use_sandwich_norm=True,
            layer_types=layer_types,
            sliding_window=self.decoder_window,
            rope_parameters={
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 1.0,
                }
            },
            tie_word_embeddings=False,
            dtype=self.dtype,
        )


class LiquidCodecQuantizer(nn.Module):
    """Invert the released residual FSQ codebooks."""

    def __init__(self, config: LiquidCodecConfig):
        super().__init__()
        effective_dim = len(config.fsq_levels) * config.n_q
        self.levels = tuple(config.fsq_levels)
        self.basis = tuple(
            math.prod(config.fsq_levels[:index]) for index in range(len(config.fsq_levels))
        )
        self.project_out = Linear(effective_dim, config.quantizer_dim)

    def forward(self, op: OpBuilder, codes: ir.Value) -> ir.Value:
        # (B, Q, T) -> (B, T, Q, 1), then independently invert each FSQ digit.
        indices = op.Unsqueeze(op.Transpose(codes, perm=[0, 2, 1]), [-1])
        digits = []
        for level, basis in zip(self.levels, self.basis, strict=True):
            digit = op.Mod(op.Div(indices, basis), level)
            half_width = level // 2
            digit = op.Cast(op.Sub(digit, half_width), to=ir.DataType.FLOAT)
            digits.append(op.Div(digit, float(half_width)))
        latents = op.Concat(*digits, axis=-1)  # (B, T, Q, 4)
        latents = op.Reshape(latents, [0, 0, -1])  # (B, T, 48)
        return self.project_out(op, latents)  # (B, T, 256)


class _LiquidCodecLfm(Lfm3TextModel):
    """LFM3 backbone variant that consumes continuous codec embeddings."""

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        if hidden_states.dtype != ir.DataType.FLOAT:
            raise ValueError("LiquidCodec decoder currently requires float32 embeddings")

        batch = op.Shape(hidden_states, start=0, end=1)
        length = op.Shape(hidden_states, start=1, end=2)
        attention_mask = op.ConstantOfShape(
            op.Concat(batch, length, axis=0), value=ir.tensor(np.array([1], np.int64))
        )
        position_ids = op.Expand(
            op.Unsqueeze(op.Range(0, op.Squeeze(length, [0]), 1), [0]),
            op.Concat(batch, length, axis=0),
        )
        embeddings = {
            kind: rope(op, position_ids)
            for kind, rope in zip(self._rope_types, self.rotary_emb, strict=True)
        }
        window = self.config.sliding_window
        assert window is not None
        attention_bias = _sliding_mask(op, hidden_states, attention_mask, window)

        layer_types = self.config.layer_types
        assert layer_types is not None
        for layer, kind in zip(self.layers, layer_types, strict=True):
            if kind == "conv":
                cache_shape = op.Concat(batch, [self.config.hidden_size, 6], axis=0)
                past = (
                    op.ConstantOfShape(
                        cache_shape, value=ir.tensor(np.array([0], np.float32))
                    ),
                )
            else:
                cache_shape = op.Concat(
                    batch,
                    [self.config.num_key_value_heads, 0, self.config.head_dim],
                    axis=0,
                )
                empty = op.ConstantOfShape(
                    cache_shape, value=ir.tensor(np.array([0], np.float32))
                )
                past = (empty, empty)
            hidden_states, _ = layer(
                op,
                hidden_states,
                attention_mask,
                attention_mask,
                attention_bias if kind == "sliding_attention" else None,
                embeddings.get(kind),
                past,
            )
        return self.embedding_norm(op, hidden_states)


class LiquidCodecISTFT(nn.Module):
    """Float32 inverse real FFT, Hann synthesis window, and overlap-add."""

    def __init__(self, config: LiquidCodecConfig):
        super().__init__()
        if config.stft_win_length != config.stft_n_fft:
            raise ValueError("LiquidCodec currently requires win_length == n_fft")
        self.n_fft = config.stft_n_fft
        self.hop_length = config.stft_hop_length
        self.trim = (config.stft_win_length - config.stft_hop_length) // 2
        window = np.hanning(config.stft_win_length + 1)[:-1].astype(np.float32)
        self.window = nn.Parameter(
            [config.stft_win_length], data=ir.tensor(window), name="window"
        )

    def forward(
        self, op: OpBuilder, real: ir.Value, imag: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        complex_spec = op.Concat(op.Unsqueeze(real, [-1]), op.Unsqueeze(imag, [-1]), axis=-1)
        complex_spec = op.Transpose(complex_spec, perm=[0, 2, 1, 3])  # (B, T, F, 2)
        frames = op.DFT(complex_spec, self.n_fft, 2, inverse=1, onesided=1)
        frames = op.Squeeze(op.Slice(frames, [0], [1], [3]), [3])
        frames = op.Transpose(frames, perm=[0, 2, 1])  # (B, N, T)
        frames = op.Mul(frames, op.Reshape(self.window, [1, self.n_fft, 1]))

        num_frames = op.Shape(frames, start=2, end=3)
        output_size = op.Add(op.Mul(op.Sub(num_frames, [1]), [self.hop_length]), [self.n_fft])
        image_shape = op.Concat([1], output_size, axis=0)
        audio = op.Col2Im(
            frames,
            image_shape,
            [1, self.n_fft],
            strides=[1, self.hop_length],
        )
        audio = op.Squeeze(audio, [1, 2])

        envelope_frames = op.Expand(
            op.Reshape(op.Mul(self.window, self.window), [1, self.n_fft, 1]),
            op.Concat([1, self.n_fft], num_frames, axis=0),
        )
        envelope = op.Col2Im(
            envelope_frames,
            image_shape,
            [1, self.n_fft],
            strides=[1, self.hop_length],
        )
        envelope = op.Squeeze(envelope, [0, 1, 2])
        end = op.Sub(op.Shape(audio, start=1, end=2), [self.trim])
        audio = op.Slice(audio, [self.trim], end, [1])
        envelope = op.Slice(envelope, [self.trim], end, [0])
        return audio, envelope


class LiquidCodecDecoder(nn.Module):
    """Independent offline ``codes[B,12,T] -> waveform[B,T*1920]`` decoder."""

    def __init__(self, config: LiquidCodecConfig | None = None):
        super().__init__()
        self.config = config or LiquidCodecConfig()
        if self.config.dtype != ir.DataType.FLOAT:
            raise ValueError("LiquidCodec decoder currently supports float32 only")
        if self.config.n_q != 12 or len(self.config.fsq_levels) != 4:
            raise ValueError("LiquidCodec decoder requires 12 four-dimensional FSQ codebooks")
        self.quantizer = LiquidCodecQuantizer(self.config)
        self.from_quantizer = Linear(self.config.quantizer_dim, self.config.encoder_dim)
        self.decoder = _LiquidCodecSynthesis(self.config)

    def forward(self, op: OpBuilder, codes: ir.Value) -> ir.Value:
        if codes.dtype != ir.DataType.INT64:
            raise ValueError("LiquidCodec codes must be int64")
        latents = self.quantizer(op, codes)  # (B, T, 256)
        embeddings = self.from_quantizer(op, latents)  # (B, T, 512)
        audio, envelope = self.decoder(op, embeddings)
        return op.Div(audio, envelope)

    @staticmethod
    def preprocess_weights(state_dict: Mapping[str, object]) -> dict[str, object]:
        """Keep decoder weights and normalize legacy checkpoint prefixes.

        Encoder-only tensors are deliberately rejected from the returned map;
        the standalone decoder never pretends that the main 4 GB checkpoint
        contains its separate ``audio_tokenizer`` weights.
        """
        kept_prefixes = ("quantizer.project_out.", "from_quantizer.", "decoder.")
        result = {}
        for name, value in state_dict.items():
            name = name.removeprefix("codec.").removeprefix("model.")
            if name.startswith(kept_prefixes):
                result[name] = value
        return result


class _LiquidCodecSynthesis(nn.Module):
    def __init__(self, config: LiquidCodecConfig):
        super().__init__()
        self.config = config
        self.frame_stack = config.stft_frame_stack
        self.lfm = _LiquidCodecLfm(config.lfm_config())
        self.lin = Linear(config.encoder_dim, config.stft_n_fft + 2)
        self.istft = LiquidCodecISTFT(config)

    def forward(self, op: OpBuilder, embeddings: ir.Value) -> tuple[ir.Value, ir.Value]:
        # Nearest-exact temporal interpolation is repetition for integer factor six.
        embeddings = op.Unsqueeze(embeddings, [2])
        embeddings = op.Expand(
            embeddings,
            op.Concat(
                op.Shape(embeddings, start=0, end=2),
                [self.frame_stack],
                op.Shape(embeddings, start=3, end=4),
                axis=0,
            ),
        )
        embeddings = op.Reshape(embeddings, [0, -1, self.config.encoder_dim])
        hidden_states = self.lfm(op, embeddings)
        spectrum = op.Transpose(self.lin(op, hidden_states), perm=[0, 2, 1])
        log_magnitude, angle = op.Split(spectrum, num_outputs=2, axis=1, _outputs=2)
        magnitude = op.Exp(log_magnitude)
        return self.istft(
            op, op.Mul(magnitude, op.Cos(angle)), op.Mul(magnitude, op.Sin(angle))
        )


def build_liquid_codec_decoder(
    config: LiquidCodecConfig | None = None,
) -> ir.Model:
    """Build the standalone offline decoder graph without registry integration."""
    module = LiquidCodecDecoder(config)
    graph = ir.Graph([], [], nodes=[], opset_imports={"": OPSET_VERSION}, name="liquid_codec")
    builder = GraphBuilder(graph)
    codes = builder.input(
        "codes",
        dtype=ir.DataType.INT64,
        shape=[ir.SymbolicDim("batch"), 12, ir.SymbolicDim("codec_frames")],
    )
    builder.add_output(module(builder.op, codes), "waveform")
    return ir.Model(graph, ir_version=10)
