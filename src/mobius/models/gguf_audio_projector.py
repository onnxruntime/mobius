# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable audio encoder/projector graphs for GGUF ``clip`` sidecars.

The modules in this file consume the processor-native single-clip boundary used
by llama.cpp: rank-2 log-mel features ``(frames, mel_bins)`` or a rank-1 mono
waveform. They return rank-2 projected feature rows ``(audio_tokens, text_dim)``.
No text decoder, media-token mixer, or generated-audio decoder is implied.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ParakeetCTCConfig
from mobius.components import (
    CodecEncoderTransformerModel,
    Conv1d,
    Conv2d,
    LayerNorm,
    Linear,
    MeralionAudioSidecar,
    ParakeetFastConformerEncoder,
    RMSNorm,
    WhisperEncoderLayer,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


TensorShapes = Mapping[str, tuple[int, ...]]


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFAudioProcessorABI:
    """Revision-neutral processor-to-graph contract for one audio route."""

    sample_rate: int
    channels: int
    graph_input: str
    graph_layout: str
    preprocessing: str
    n_fft: int | None = None
    window_length: int | None = None
    hop_length: int | None = None
    chunk_seconds: int | None = None
    frame_multiple: int | None = None
    max_seconds: int | None = None

    @property
    def feature_contract(self) -> str:
        """Compatibility alias used by the merged generic audio task."""
        return self.preprocessing


AUDIO_PROCESSOR_ABIS: Mapping[str, GGUFAudioProcessorABI] = {
    "ultravox": GGUFAudioProcessorABI(
        16_000,
        1,
        "input_features",
        "float32[frames,128]",
        "Whisper log10 mel, right-padded to 3000 frames",
        n_fft=400,
        window_length=400,
        hop_length=160,
        chunk_seconds=30,
    ),
    "voxtral": GGUFAudioProcessorABI(
        16_000,
        1,
        "input_features",
        "float32[frames,128]",
        "Whisper log10 mel, right-padded to 3000 frames",
        n_fft=400,
        window_length=400,
        hop_length=160,
        chunk_seconds=30,
    ),
    "musicflamingo": GGUFAudioProcessorABI(
        16_000,
        1,
        "input_features",
        "float32[frames,128]",
        "Whisper log10 mel, right-padded to 3000 frames",
        n_fft=400,
        window_length=400,
        hop_length=160,
        chunk_seconds=30,
    ),
    "lfm2a": GGUFAudioProcessorABI(
        16_000,
        1,
        "input_features",
        "float32[frames,128]",
        "centered pre-emphasized natural-log mel with per-feature normalization",
        n_fft=512,
        window_length=400,
        hop_length=160,
        chunk_seconds=1,
    ),
    "granite_speech": GGUFAudioProcessorABI(
        16_000,
        1,
        "input_features",
        "float32[stacked_frames,160]",
        "centered half-mel filterbank, clamp/scale, then concatenate frame pairs",
        n_fft=512,
        window_length=400,
        hop_length=160,
    ),
    "parakeet": GGUFAudioProcessorABI(
        16_000,
        1,
        "input_features",
        "float32[frames,mel_bins]",
        "centered pre-emphasized power mel with stored window/filterbank and normalization",
        n_fft=512,
        window_length=400,
        hop_length=160,
    ),
    "mimo_audio": GGUFAudioProcessorABI(
        24_000,
        1,
        "input_features",
        "float32[frames,128]",
        "centered HTK magnitude mel followed by natural log",
        n_fft=960,
        window_length=960,
        hop_length=240,
    ),
    "pockettts_spkenc": GGUFAudioProcessorABI(
        24_000,
        1,
        "input_values",
        "float32[samples]",
        "raw mono PCM, zero-padded to complete 1920-sample conditioning frames",
        frame_multiple=1_920,
        max_seconds=30,
    ),
    "meralion": GGUFAudioProcessorABI(
        16_000,
        1,
        "input_features",
        "float32[3000,128]",
        "Whisper log10 mel for one right-padded 30-second chunk",
        n_fft=400,
        window_length=400,
        hop_length=160,
        chunk_seconds=30,
        max_seconds=300,
    ),
}


def _shape(shapes: TensorShapes, name: str, rank: int | None = None) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in shapes[name])
    except KeyError as exc:
        raise ValueError(f"GGUF audio sidecar is missing tensor {name!r}.") from exc
    if rank is not None and len(shape) != rank:
        raise ValueError(
            f"GGUF audio tensor {name!r} has shape {shape}, expected rank {rank}."
        )
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"GGUF audio tensor {name!r} has non-positive shape {shape}.")
    return shape


def _metadata_int(metadata: Mapping[str, object], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}.")
    return int(value)


def _metadata_float(metadata: Mapping[str, object], key: str) -> float:
    value = metadata.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{key} must be a positive finite number, got {value!r}.")
    return float(value)


def _cast_boundary(op: OpBuilder, value: ir.Value, weight: ir.Value) -> ir.Value:
    return op.CastLike(value, weight)


def _stack_frames(
    op: OpBuilder,
    hidden_states: ir.Value,
    *,
    stack_factor: int,
    hidden_size: int,
) -> ir.Value:
    """Right-pad and concatenate consecutive frames along the feature axis."""
    if stack_factor <= 1:
        return hidden_states
    time = op.Shape(hidden_states, start=1, end=2)
    factor = op.Constant(value_ints=[stack_factor])
    padded_time = op.Mul(
        op.Div(op.Add(time, op.Constant(value_ints=[stack_factor - 1])), factor),
        factor,
    )
    pad = op.Sub(padded_time, time)
    pads = op.Concat(
        op.Constant(value_ints=[0, 0, 0, 0]),
        pad,
        op.Constant(value_ints=[0]),
        axis=0,
    )
    hidden_states = op.Pad(hidden_states, pads)
    batch = op.Shape(hidden_states, start=0, end=1)
    stacked_time = op.Div(padded_time, factor)
    return op.Reshape(
        hidden_states,
        op.Concat(
            batch,
            stacked_time,
            op.Constant(value_ints=[hidden_size * stack_factor]),
            axis=0,
        ),
    )


def _average_pool_frames(op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
    hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
    hidden_states = op.AveragePool(
        hidden_states,
        kernel_shape=[2],
        strides=[2],
        pads=[0, 0],
        count_include_pad=0,
    )
    return op.Transpose(hidden_states, perm=[0, 2, 1])


class _GeluProjector(nn.Module):
    """Two-layer exact-GELU projector with route-specific optional biases."""

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        *,
        first_bias: bool,
        second_bias: bool,
    ):
        super().__init__()
        self.linear_1 = Linear(input_size, intermediate_size, bias=first_bias)
        self.linear_2 = Linear(intermediate_size, output_size, bias=second_bias)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = self.linear_1(op, hidden_states)
        hidden_states = op.Gelu(hidden_states)
        return self.linear_2(op, hidden_states)


class _UltravoxProjector(nn.Module):
    """Ultravox RMSNorm -> swapped SwiGLU -> RMSNorm -> linear projector."""

    def __init__(self, input_size: int, expanded_size: int, output_size: int):
        super().__init__()
        if expanded_size % 2:
            raise ValueError(
                f"Ultravox projector expansion must be even, got {expanded_size}."
            )
        self.norm_pre = RMSNorm(input_size, eps=1e-6)
        self.linear_1 = Linear(input_size, expanded_size, bias=False)
        self.norm_mid = RMSNorm(expanded_size // 2, eps=1e-6)
        self.linear_2 = Linear(expanded_size // 2, output_size, bias=False)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = self.norm_pre(op, hidden_states)
        first, second = op.Split(
            self.linear_1(op, hidden_states),
            axis=-1,
            num_outputs=2,
            _outputs=2,
        )
        # llama.cpp's ``ggml_swiglu_swapped`` applies SiLU to the second half.
        hidden_states = op.Mul(first, op.Swish(second))
        hidden_states = self.norm_mid(op, hidden_states)
        return self.linear_2(op, hidden_states)


class GGUFWhisperAudioProjector(nn.Module):
    """Whisper encoder shared by ``ultravox``, ``voxtral``, and ``musicflamingo``."""

    def __init__(
        self,
        projector_type: str,
        metadata: Mapping[str, object],
        shapes: TensorShapes,
    ):
        super().__init__()
        if projector_type not in {"ultravox", "voxtral", "musicflamingo"}:
            raise ValueError(f"Unsupported Whisper GGUF projector {projector_type!r}.")
        hidden_size = _metadata_int(metadata, "clip.audio.embedding_length")
        intermediate_size = _metadata_int(metadata, "clip.audio.feed_forward_length")
        num_layers = _metadata_int(metadata, "clip.audio.block_count")
        num_heads = _metadata_int(metadata, "clip.audio.attention.head_count")
        eps = _metadata_float(metadata, "clip.audio.attention.layer_norm_epsilon")
        self.num_mel_bins = _metadata_int(metadata, "clip.audio.num_mel_bins")
        self.input_schema = (
            (
                "input_features",
                ir.DataType.FLOAT,
                (ir.SymbolicDim("frames"), self.num_mel_bins),
            ),
        )
        if hidden_size % num_heads:
            raise ValueError(
                f"{projector_type} hidden size {hidden_size} is not divisible by {num_heads} heads."
            )

        conv1_shape = _shape(shapes, "a.conv1d.1.weight", 3)
        conv2_shape = _shape(shapes, "a.conv1d.2.weight", 3)
        if conv1_shape[:2] != (hidden_size, self.num_mel_bins):
            raise ValueError(
                f"{projector_type} first convolution has shape {conv1_shape}, expected "
                f"({hidden_size}, {self.num_mel_bins}, kernel)."
            )
        if conv2_shape[:2] != (hidden_size, hidden_size):
            raise ValueError(
                f"{projector_type} second convolution has shape {conv2_shape}, expected "
                f"({hidden_size}, {hidden_size}, kernel)."
            )
        self.conv1 = Conv1d(
            self.num_mel_bins,
            hidden_size,
            conv1_shape[2],
            stride=1,
            padding=conv1_shape[2] // 2,
            bias=True,
        )
        self.conv2 = Conv1d(
            hidden_size,
            hidden_size,
            conv2_shape[2],
            stride=2,
            padding=conv2_shape[2] // 2,
            bias=True,
        )
        position_shape = _shape(shapes, "a.position_embd.weight", 2)
        if position_shape[1] != hidden_size:
            raise ValueError(
                f"{projector_type} position table width {position_shape[1]} != {hidden_size}."
            )
        self.position_embeddings = nn.Parameter(position_shape)
        self.layers = nn.ModuleList(
            [
                WhisperEncoderLayer(
                    hidden_size,
                    num_heads,
                    intermediate_size,
                    "gelu",
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.post_layernorm = LayerNorm(hidden_size, eps=eps)
        self._hidden_size = hidden_size
        self._max_positions = position_shape[0]
        self._stack_factor = (
            _metadata_int(metadata, "clip.audio.projector.stack_factor")
            if projector_type in {"ultravox", "voxtral"}
            else 1
        )
        self._use_average_pool = projector_type in {"voxtral", "musicflamingo"}

        projector_input = hidden_size * self._stack_factor
        first_shape = _shape(shapes, "mm.a.mlp.1.weight", 2)
        second_shape = _shape(shapes, "mm.a.mlp.2.weight", 2)
        if first_shape[1] != projector_input or second_shape[1] != first_shape[0] // (
            2 if projector_type == "ultravox" else 1
        ):
            raise ValueError(
                f"{projector_type} projector shapes {first_shape}, {second_shape} do not "
                f"compose from input width {projector_input}."
            )
        if projector_type == "ultravox":
            self.projector: nn.Module = _UltravoxProjector(
                projector_input,
                first_shape[0],
                second_shape[0],
            )
        else:
            self.projector = _GeluProjector(
                projector_input,
                first_shape[0],
                second_shape[0],
                first_bias="mm.a.mlp.1.bias" in shapes,
                second_bias="mm.a.mlp.2.bias" in shapes,
            )

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        # Processor boundary: (frames, mel) -> Whisper Conv1d (1, mel, frames).
        hidden_states = op.Transpose(input_features, perm=[1, 0])
        hidden_states = op.Unsqueeze(hidden_states, [0])
        hidden_states = _cast_boundary(op, hidden_states, self.conv1.weight)
        hidden_states = op.Gelu(self.conv1(op, hidden_states))
        hidden_states = op.Gelu(self.conv2(op, hidden_states))
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])

        time = op.Shape(hidden_states, start=1, end=2)
        positions = op.Slice(
            self.position_embeddings,
            op.Constant(value_ints=[0]),
            time,
            op.Constant(value_ints=[0]),
        )
        hidden_states = op.Add(hidden_states, op.Unsqueeze(positions, [0]))
        for layer in self.layers:
            hidden_states = layer(op, hidden_states)
        if self._use_average_pool:
            hidden_states = _average_pool_frames(op, hidden_states)
        hidden_states = self.post_layernorm(op, hidden_states)
        hidden_states = _stack_frames(
            op,
            hidden_states,
            stack_factor=self._stack_factor,
            hidden_size=self._hidden_size,
        )
        hidden_states = self.projector(op, hidden_states)
        return op.Squeeze(hidden_states, [0])


class _SquaredReLUProjector(nn.Module):
    """Parakeet RMSNorm -> Linear -> ReLU squared -> Linear projection."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        *,
        first_bias: bool,
        second_bias: bool,
    ):
        super().__init__()
        self.norm_pre = RMSNorm(hidden_size, eps=1e-6)
        self.linear_1 = Linear(hidden_size, intermediate_size, bias=first_bias)
        self.linear_2 = Linear(intermediate_size, output_size, bias=second_bias)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = self.norm_pre(op, hidden_states)
        hidden_states = op.Relu(self.linear_1(op, hidden_states))
        hidden_states = op.Mul(hidden_states, hidden_states)
        return self.linear_2(op, hidden_states)


class GGUFParakeetAudioProjector(nn.Module):
    """NVIDIA Parakeet FastConformer encoder followed by its sound projection."""

    def __init__(self, metadata: Mapping[str, object], shapes: TensorShapes):
        super().__init__()
        hidden_size = _metadata_int(metadata, "clip.audio.embedding_length")
        intermediate_size = _metadata_int(metadata, "clip.audio.feed_forward_length")
        num_layers = _metadata_int(metadata, "clip.audio.block_count")
        num_heads = _metadata_int(metadata, "clip.audio.attention.head_count")
        eps = _metadata_float(metadata, "clip.audio.attention.layer_norm_epsilon")
        self.num_mel_bins = _metadata_int(metadata, "clip.audio.num_mel_bins")
        self.input_schema = (
            (
                "input_features",
                ir.DataType.FLOAT,
                (ir.SymbolicDim("frames"), self.num_mel_bins),
            ),
        )
        subsampling_factor = _metadata_int(metadata, "clip.audio.subsampling_factor")
        conv_kernel_size = _metadata_int(metadata, "clip.audio.conv_kernel_size")
        if subsampling_factor != 8:
            raise ValueError(
                f"Parakeet GGUF subsampling_factor must be 8, got {subsampling_factor}."
            )
        first_conv = _shape(shapes, "a.conv1d.0.weight", 4)
        subsampling_channels = first_conv[0]
        subsampling_kernel = first_conv[-1]
        if first_conv[1] != 1 or first_conv[-2] != subsampling_kernel:
            raise ValueError(f"Parakeet first Conv2d has unsupported shape {first_conv}.")

        config = ParakeetCTCConfig(
            model_type="gguf_parakeet",
            vocab_size=1,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=hidden_size // num_heads,
            max_position_embeddings=65_536,
            hidden_act="silu",
            num_mel_bins=self.num_mel_bins,
            subsampling_factor=subsampling_factor,
            subsampling_conv_channels=subsampling_channels,
            subsampling_conv_kernel_size=subsampling_kernel,
            subsampling_conv_stride=2,
            conv_kernel_size=conv_kernel_size,
            attention_bias="a.blk.0.attn_q.bias" in shapes,
            convolution_bias="a.blk.0.conv_pw1.bias" in shapes,
            scale_input=False,
            layer_norm_eps=eps,
        )
        self.encoder = ParakeetFastConformerEncoder(config)

        first_shape = _shape(shapes, "mm.a.mlp.1.weight", 2)
        second_shape = _shape(shapes, "mm.a.mlp.2.weight", 2)
        if first_shape[1] != hidden_size or second_shape[1] != first_shape[0]:
            raise ValueError(
                f"Parakeet projector shapes {first_shape}, {second_shape} do not compose."
            )
        self.projector = _SquaredReLUProjector(
            hidden_size,
            first_shape[0],
            second_shape[0],
            first_bias="mm.a.mlp.1.bias" in shapes,
            second_bias="mm.a.mlp.2.bias" in shapes,
        )

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        hidden_states = op.Unsqueeze(input_features, [0])
        hidden_states = _cast_boundary(
            op,
            hidden_states,
            self.encoder.subsampling.layers[0].weight,
        )
        frames = op.Shape(hidden_states, start=1, end=2)
        mask_shape = op.Concat(op.Constant(value_ints=[1]), frames, axis=0)
        attention_mask = op.Cast(
            op.ConstantOfShape(mask_shape, value=ir.tensor(np.ones(1, dtype=np.float32))),
            to=ir.DataType.BOOL,
        )
        hidden_states, _ = self.encoder(op, hidden_states, attention_mask)
        hidden_states = self.projector(op, hidden_states)
        return op.Squeeze(hidden_states, [0])


class _LFM2Subsampling(nn.Module):
    """Three-stage depthwise-separable Conv2d subsampler used by LFM2-Audio."""

    def __init__(
        self,
        num_mel_bins: int,
        hidden_size: int,
        conv_channels: int,
        *,
        kernel_size: int = 3,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.ModuleList(
            [
                Conv2d(
                    1,
                    conv_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                ),
                _ReLU(),
                Conv2d(
                    conv_channels,
                    conv_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    groups=conv_channels,
                ),
                Conv2d(conv_channels, conv_channels, kernel_size=1),
                _ReLU(),
                Conv2d(
                    conv_channels,
                    conv_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    groups=conv_channels,
                ),
                Conv2d(conv_channels, conv_channels, kernel_size=1),
                _ReLU(),
            ]
        )
        frequency = num_mel_bins
        for _ in range(3):
            frequency = (frequency + 2 * padding - kernel_size) // 2 + 1
        self.output = Linear(conv_channels * frequency, hidden_size, bias=True)

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        hidden_states = op.Unsqueeze(input_features, [0, 1])
        for layer in self.layers:
            hidden_states = layer(op, hidden_states)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1, 3])
        hidden_states = op.Reshape(hidden_states, [0, 0, -1])
        return self.output(op, hidden_states)


class _ReLU(nn.Module):
    def forward(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        return op.Relu(value)


class _RelativePositionEncoding(nn.Module):
    """Interleaved Transformer-XL relative sinusoid over ``[-T+1, T-1]``."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self._hidden_size = hidden_size
        self._inv_freq = 1.0 / (
            10_000.0 ** (np.arange(0, hidden_size, 2, dtype=np.float32) / hidden_size)
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        time = op.Squeeze(op.Shape(hidden_states, start=1, end=2), [0])
        time_f = op.Cast(time, to=ir.DataType.FLOAT)
        one = op.Constant(value=ir.tensor(np.float32(1.0)))
        positions = op.Range(op.Sub(time_f, one), op.Neg(time_f), op.Neg(one))
        angles = op.Mul(
            op.Unsqueeze(positions, [1]),
            op.Unsqueeze(op.Constant(value=ir.tensor(self._inv_freq)), [0]),
        )
        sin = op.Unsqueeze(op.Sin(angles), [-1])
        cos = op.Unsqueeze(op.Cos(angles), [-1])
        embeddings = op.Reshape(
            op.Concat(sin, cos, axis=-1),
            op.Concat(
                op.Shape(angles, start=0, end=1),
                op.Constant(value_ints=[self._hidden_size]),
                axis=0,
            ),
        )
        return op.Unsqueeze(op.CastLike(embeddings, hidden_states), [0])


class _RelativeConformerAttention(nn.Module):
    """Transformer-XL relative-position attention shared by LFM2-Audio."""

    def __init__(self, hidden_size: int, num_heads: int, *, bias: bool):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads
        self.q_proj = Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = Linear(hidden_size, hidden_size, bias=bias)
        self.linear_pos = Linear(hidden_size, hidden_size, bias=False)
        self.pos_bias_u = nn.Parameter([num_heads, self._head_dim])
        self.pos_bias_v = nn.Parameter([num_heads, self._head_dim])

    def _heads(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        return op.Reshape(
            value,
            op.Concat(
                op.Shape(value, start=0, end=1),
                op.Shape(value, start=1, end=2),
                op.Constant(value_ints=[self._num_heads, self._head_dim]),
                axis=0,
            ),
        )

    @staticmethod
    def _relative_shift(op: OpBuilder, scores: ir.Value) -> ir.Value:
        zero = op.Expand(
            op.CastLike(0.0, scores),
            op.Concat(
                op.Shape(scores, start=0, end=3),
                op.Constant(value_ints=[1]),
                axis=0,
            ),
        )
        scores = op.Concat(zero, scores, axis=-1)
        scores = op.Reshape(
            scores,
            op.Concat(
                op.Shape(scores, start=0, end=2),
                op.Constant(value_ints=[-1]),
                op.Shape(scores, start=2, end=3),
                axis=0,
            ),
        )
        scores = op.Slice(
            scores,
            op.Constant(value_ints=[1]),
            op.Shape(scores, start=2, end=3),
            op.Constant(value_ints=[2]),
        )
        return op.Reshape(
            scores,
            op.Concat(
                op.Shape(scores, start=0, end=2),
                op.Shape(scores, start=3, end=4),
                op.Constant(value_ints=[-1]),
                axis=0,
            ),
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: ir.Value,
    ) -> ir.Value:
        scale = float(self._head_dim**-0.5)
        query = self._heads(op, self.q_proj(op, hidden_states))
        key = self._heads(op, self.k_proj(op, hidden_states))
        value = self._heads(op, self.v_proj(op, hidden_states))
        query_u = op.Add(query, op.Unsqueeze(op.Unsqueeze(self.pos_bias_u, [0]), [0]))
        query_v = op.Add(query, op.Unsqueeze(op.Unsqueeze(self.pos_bias_v, [0]), [0]))
        query_u = op.Transpose(query_u, perm=[0, 2, 1, 3])
        query_v = op.Transpose(query_v, perm=[0, 2, 1, 3])
        key = op.Transpose(key, perm=[0, 2, 1, 3])
        value = op.Transpose(value, perm=[0, 2, 1, 3])

        content_scores = op.MatMul(query_u, op.Transpose(key, perm=[0, 1, 3, 2]))
        relative_key = self._heads(op, self.linear_pos(op, position_embeddings))
        relative_key = op.Transpose(relative_key, perm=[0, 2, 1, 3])
        relative_scores = op.MatMul(
            query_v,
            op.Transpose(relative_key, perm=[0, 1, 3, 2]),
        )
        relative_scores = self._relative_shift(op, relative_scores)
        relative_scores = op.Slice(
            relative_scores,
            op.Constant(value_ints=[0]),
            op.Shape(key, start=2, end=3),
            op.Constant(value_ints=[3]),
        )
        probabilities = op.Softmax(
            op.Mul(op.Add(content_scores, relative_scores), scale),
            axis=-1,
        )
        context = op.MatMul(probabilities, value)
        context = op.Transpose(context, perm=[0, 2, 1, 3])
        context = op.Reshape(context, [0, 0, self._num_heads * self._head_dim])
        return self.out_proj(op, context)


class _ConformerFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return self.down_proj(op, op.Swish(self.up_proj(op, hidden_states)))


class _ConformerConvolution(nn.Module):
    """LFM2 Conformer GLU/depthwise-convolution module."""

    def __init__(self, hidden_size: int, kernel_size: int):
        super().__init__()
        self.pointwise_conv1 = Linear(hidden_size, hidden_size * 2, bias=True)
        self.depthwise_conv = Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            padding=kernel_size // 2,
            groups=hidden_size,
            bias=True,
        )
        self.conv_norm_weight = nn.Parameter([hidden_size])
        self.conv_norm_bias = nn.Parameter([hidden_size])
        self.pointwise_conv2 = Linear(hidden_size, hidden_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        first, gate = op.Split(
            self.pointwise_conv1(op, hidden_states),
            axis=-1,
            num_outputs=2,
            _outputs=2,
        )
        hidden_states = op.Mul(first, op.Sigmoid(gate))
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = self.depthwise_conv(op, hidden_states)
        hidden_states = op.Add(
            op.Mul(hidden_states, op.Unsqueeze(self.conv_norm_weight, [0, 2])),
            op.Unsqueeze(self.conv_norm_bias, [0, 2]),
        )
        hidden_states = op.Swish(hidden_states)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        return self.pointwise_conv2(op, hidden_states)


class _LFM2ConformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        kernel_size: int,
        eps: float,
    ):
        super().__init__()
        self.feed_forward1 = _ConformerFeedForward(hidden_size, intermediate_size)
        self.norm_feed_forward1 = LayerNorm(hidden_size, eps=eps)
        self.self_attn = _RelativeConformerAttention(hidden_size, num_heads, bias=True)
        self.norm_self_attn = LayerNorm(hidden_size, eps=eps)
        self.conv = _ConformerConvolution(hidden_size, kernel_size)
        self.norm_conv = LayerNorm(hidden_size, eps=eps)
        self.feed_forward2 = _ConformerFeedForward(hidden_size, intermediate_size)
        self.norm_feed_forward2 = LayerNorm(hidden_size, eps=eps)
        self.norm_out = LayerNorm(hidden_size, eps=eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: ir.Value,
    ) -> ir.Value:
        half = op.CastLike(0.5, hidden_states)
        hidden_states = op.Add(
            hidden_states,
            op.Mul(
                self.feed_forward1(op, self.norm_feed_forward1(op, hidden_states)),
                half,
            ),
        )
        hidden_states = op.Add(
            hidden_states,
            self.self_attn(
                op,
                self.norm_self_attn(op, hidden_states),
                position_embeddings,
            ),
        )
        hidden_states = op.Add(
            hidden_states,
            self.conv(op, self.norm_conv(op, hidden_states)),
        )
        hidden_states = op.Add(
            hidden_states,
            op.Mul(
                self.feed_forward2(op, self.norm_feed_forward2(op, hidden_states)),
                half,
            ),
        )
        return self.norm_out(op, hidden_states)


class _LFM2AudioAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        eps: float,
    ):
        super().__init__()
        self.norm = LayerNorm(hidden_size, eps=eps)
        self.linear_1 = Linear(hidden_size, intermediate_size, bias=True)
        self.linear_2 = Linear(intermediate_size, output_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = self.norm(op, hidden_states)
        hidden_states = op.Gelu(self.linear_1(op, hidden_states))
        return self.linear_2(op, hidden_states)


class GGUFLFM2AudioProjector(nn.Module):
    """LFM2-Audio Conformer encoder and GELU adapter."""

    def __init__(self, metadata: Mapping[str, object], shapes: TensorShapes):
        super().__init__()
        hidden_size = _metadata_int(metadata, "clip.audio.embedding_length")
        _metadata_int(metadata, "clip.audio.feed_forward_length")
        intermediate_size = _shape(shapes, "a.blk.0.ffn_up.weight", 2)[0]
        num_layers = _metadata_int(metadata, "clip.audio.block_count")
        num_heads = _metadata_int(metadata, "clip.audio.attention.head_count")
        eps = _metadata_float(metadata, "clip.audio.attention.layer_norm_epsilon")
        self.num_mel_bins = _metadata_int(metadata, "clip.audio.num_mel_bins")
        self.input_schema = (
            (
                "input_features",
                ir.DataType.FLOAT,
                (ir.SymbolicDim("frames"), self.num_mel_bins),
            ),
        )
        first_conv = _shape(shapes, "a.conv1d.0.weight", 4)
        conv_channels = first_conv[0]
        self.pre_encode = _LFM2Subsampling(
            self.num_mel_bins,
            hidden_size,
            conv_channels,
            kernel_size=first_conv[-1],
        )
        self.position_encoding = _RelativePositionEncoding(hidden_size)
        depthwise_shape = _shape(shapes, "a.blk.0.conv_dw.weight")
        kernel_size = depthwise_shape[-1]
        self.layers = nn.ModuleList(
            [
                _LFM2ConformerLayer(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    kernel_size,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )
        first_projector = _shape(shapes, "mm.a.mlp.1.weight", 2)
        second_projector = _shape(shapes, "mm.a.mlp.3.weight", 2)
        if first_projector[1] != hidden_size or second_projector[1] != first_projector[0]:
            raise ValueError(
                f"LFM2A adapter shapes {first_projector}, {second_projector} do not compose."
            )
        self.projector = _LFM2AudioAdapter(
            hidden_size,
            first_projector[0],
            second_projector[0],
            eps,
        )

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        hidden_states = _cast_boundary(
            op,
            input_features,
            self.pre_encode.layers[0].weight,
        )
        hidden_states = self.pre_encode(op, hidden_states)
        position_embeddings = self.position_encoding(op, hidden_states)
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, position_embeddings)
        hidden_states = self.projector(op, hidden_states)
        return op.Squeeze(hidden_states, [0])


class _GraniteSpeechFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return self.down_proj(op, op.Swish(self.up_proj(op, hidden_states)))


class _GraniteSpeechAttention(nn.Module):
    """Chunked Shaw-relative attention used by Granite Speech."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_size: int,
        max_position: int,
        relative_rows: int,
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads
        self._context_size = context_size
        distances = np.arange(context_size)[:, None] - np.arange(context_size)[None, :]
        distances = np.clip(distances, -context_size, context_size) + max_position
        if int(distances.max()) >= relative_rows:
            raise ValueError(
                f"Granite Speech relative-position table has {relative_rows} rows, "
                f"but chunk distances require index {int(distances.max())}."
            )
        self._distance_indices = distances.astype(np.int64)
        self.relative_positions = nn.Parameter([relative_rows, self._head_dim])
        self.q_proj = Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = Linear(hidden_size, hidden_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        time = op.Shape(hidden_states, start=1, end=2)
        context = op.Constant(value_ints=[self._context_size])
        blocks = op.Div(
            op.Add(time, op.Constant(value_ints=[self._context_size - 1])),
            context,
        )
        padded_time = op.Mul(blocks, context)
        pad = op.Sub(padded_time, time)
        hidden_states = op.Pad(
            hidden_states,
            op.Concat(
                op.Constant(value_ints=[0, 0, 0, 0]),
                pad,
                op.Constant(value_ints=[0]),
                axis=0,
            ),
        )
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                op.Constant(value_ints=[1]),
                blocks,
                context,
                op.Constant(value_ints=[self._hidden_size]),
                axis=0,
            ),
        )

        def split_heads(value: ir.Value) -> ir.Value:
            value = op.Reshape(
                value,
                op.Concat(
                    op.Constant(value_ints=[1]),
                    blocks,
                    context,
                    op.Constant(value_ints=[self._num_heads, self._head_dim]),
                    axis=0,
                ),
            )
            return op.Transpose(value, perm=[0, 1, 3, 2, 4])

        query = split_heads(self.q_proj(op, hidden_states))
        key = split_heads(self.k_proj(op, hidden_states))
        value = split_heads(self.v_proj(op, hidden_states))
        scores = op.MatMul(query, op.Transpose(key, perm=[0, 1, 2, 4, 3]))
        relative = op.Gather(
            self.relative_positions,
            op.Constant(value=ir.tensor(self._distance_indices)),
        )
        relative_scores = op.Einsum(
            query,
            relative,
            equation="bnhqd,qkd->bnhqk",
        )
        scores = op.Mul(
            op.Add(scores, relative_scores),
            op.CastLike(float(self._head_dim**-0.5), scores),
        )

        frame_ids = op.Range(op.Constant(value_int=0), op.Squeeze(padded_time), 1)
        valid = op.Less(frame_ids, op.Squeeze(time))
        valid = op.Reshape(valid, op.Concat(blocks, context, axis=0))
        key_valid = op.Unsqueeze(valid, [0, 2, 3])
        scores = op.Where(
            key_valid,
            scores,
            op.CastLike(float("-inf"), scores),
        )
        probabilities = op.Softmax(scores, axis=-1)
        hidden_states = op.MatMul(probabilities, value)
        hidden_states = op.Transpose(hidden_states, perm=[0, 1, 3, 2, 4])
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                op.Constant(value_ints=[1]),
                padded_time,
                op.Constant(value_ints=[self._hidden_size]),
                axis=0,
            ),
        )
        hidden_states = self.out_proj(op, hidden_states)
        return op.Slice(
            hidden_states,
            op.Constant(value_ints=[0]),
            time,
            op.Constant(value_ints=[1]),
        )


class _GraniteSpeechConvolution(nn.Module):
    def __init__(self, hidden_size: int, conv_size: int, kernel_size: int):
        super().__init__()
        self.pointwise_conv1 = Linear(hidden_size, conv_size * 2, bias=True)
        self.depthwise_conv = Conv1d(
            conv_size,
            conv_size,
            kernel_size,
            padding=kernel_size // 2,
            groups=conv_size,
            bias=False,
        )
        self.batch_norm_weight = nn.Parameter([conv_size])
        self.batch_norm_bias = nn.Parameter([conv_size])
        self.pointwise_conv2 = Linear(conv_size, hidden_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        first, gate = op.Split(
            self.pointwise_conv1(op, hidden_states),
            axis=-1,
            num_outputs=2,
            _outputs=2,
        )
        hidden_states = op.Mul(first, op.Sigmoid(gate))
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = self.depthwise_conv(op, hidden_states)
        hidden_states = op.Add(
            op.Mul(hidden_states, op.Unsqueeze(self.batch_norm_weight, [0, 2])),
            op.Unsqueeze(self.batch_norm_bias, [0, 2]),
        )
        hidden_states = op.Swish(hidden_states)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        return self.pointwise_conv2(op, hidden_states)


class _GraniteSpeechEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        context_size: int,
        max_position: int,
        relative_rows: int,
        conv_size: int,
        kernel_size: int,
        eps: float,
    ):
        super().__init__()
        self.ffn1_norm = LayerNorm(hidden_size, eps=eps)
        self.feed_forward1 = _GraniteSpeechFeedForward(hidden_size, intermediate_size)
        self.attention_norm = LayerNorm(hidden_size, eps=eps)
        self.self_attn = _GraniteSpeechAttention(
            hidden_size,
            num_heads,
            context_size,
            max_position,
            relative_rows,
        )
        self.conv_norm = LayerNorm(hidden_size, eps=eps)
        self.conv = _GraniteSpeechConvolution(hidden_size, conv_size, kernel_size)
        self.ffn2_norm = LayerNorm(hidden_size, eps=eps)
        self.feed_forward2 = _GraniteSpeechFeedForward(hidden_size, intermediate_size)
        self.output_norm = LayerNorm(hidden_size, eps=eps)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        half = op.CastLike(0.5, hidden_states)
        hidden_states = op.Add(
            hidden_states,
            op.Mul(self.feed_forward1(op, self.ffn1_norm(op, hidden_states)), half),
        )
        hidden_states = op.Add(
            hidden_states,
            self.self_attn(op, self.attention_norm(op, hidden_states)),
        )
        hidden_states = op.Add(
            hidden_states,
            self.conv(op, self.conv_norm(op, hidden_states)),
        )
        hidden_states = op.Add(
            hidden_states,
            op.Mul(self.feed_forward2(op, self.ffn2_norm(op, hidden_states)), half),
        )
        return self.output_norm(op, hidden_states)


class _GraniteSpeechQFormer(nn.Module):
    """Windowed two-layer post-norm Q-Former used by Granite Speech."""

    def __init__(
        self,
        encoder_size: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        query_shape: tuple[int, int, int],
        num_layers: int,
        output_size: int,
    ):
        super().__init__()
        from mobius.components import QFormerLayer

        self.query_tokens = nn.Parameter(query_shape)
        self.query_norm = LayerNorm(hidden_size, eps=1e-12)
        self.layers = nn.ModuleList(
            [
                QFormerLayer(
                    hidden_size,
                    num_heads,
                    intermediate_size,
                    encoder_hidden_size=encoder_size,
                    hidden_act="gelu",
                    layer_norm_eps=1e-12,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.output = Linear(hidden_size, output_size, bias=True)
        self._num_queries = query_shape[1]
        self._hidden_size = hidden_size

    def forward(self, op: OpBuilder, encoder_windows: ir.Value) -> ir.Value:
        queries = self.query_norm(op, self.query_tokens)
        batch = op.Shape(encoder_windows, start=0, end=1)
        queries = op.Expand(
            queries,
            op.Concat(
                batch,
                op.Constant(value_ints=[self._num_queries, self._hidden_size]),
                axis=0,
            ),
        )
        for layer in self.layers:
            queries = layer(op, queries, encoder_windows)
        return self.output(op, queries)


class GGUFGraniteSpeechProjector(nn.Module):
    """Granite Speech chunked Conformer, CTC branch, and Q-Former projector."""

    input_kind = "features"

    def __init__(self, metadata: Mapping[str, object], shapes: TensorShapes):
        super().__init__()
        hidden_size = _metadata_int(metadata, "clip.audio.embedding_length")
        _metadata_int(metadata, "clip.audio.feed_forward_length")
        num_layers = _metadata_int(metadata, "clip.audio.block_count")
        num_heads = _metadata_int(metadata, "clip.audio.attention.head_count")
        eps = _metadata_float(metadata, "clip.audio.attention.layer_norm_epsilon")
        self.num_mel_bins = _metadata_int(metadata, "clip.audio.num_mel_bins")
        self.input_schema = (
            (
                "input_features",
                ir.DataType.FLOAT,
                (ir.SymbolicDim("frames"), self.num_mel_bins),
            ),
        )
        context_size = _metadata_int(metadata, "clip.audio.chunk_size")
        max_position = _metadata_int(metadata, "clip.audio.max_pos_emb")
        projector_window = _metadata_int(metadata, "clip.audio.projector.window_size")
        downsample_rate = _metadata_int(
            metadata,
            "clip.audio.projector.downsample_rate",
        )
        projector_heads = _metadata_int(metadata, "clip.audio.projector.head_count")
        if projector_window % downsample_rate:
            raise ValueError(
                f"Granite Speech projector window {projector_window} is not divisible by "
                f"downsample rate {downsample_rate}."
            )

        input_projection = _shape(shapes, "a.input_projection.weight", 2)
        if input_projection != (hidden_size, self.num_mel_bins):
            raise ValueError(
                f"Granite Speech input projection has shape {input_projection}, expected "
                f"({hidden_size}, {self.num_mel_bins})."
            )
        self.input_projection = Linear(self.num_mel_bins, hidden_size, bias=True)
        intermediate_size = _shape(shapes, "a.blk.0.ffn_up.weight", 2)[0]
        relative_shape = _shape(shapes, "a.blk.0.attn_rel_pos_emb", 2)
        conv_shape = _shape(shapes, "a.blk.0.conv_dw.weight", 2)
        conv_size, kernel_size = conv_shape
        self.layers = nn.ModuleList(
            [
                _GraniteSpeechEncoderLayer(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    context_size,
                    max_position,
                    relative_shape[0],
                    conv_size,
                    kernel_size,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.ctc_out = Linear(
            hidden_size,
            _shape(shapes, "a.enc_ctc_out.weight", 2)[0],
            bias=True,
        )
        ctc_mid_shape = _shape(shapes, "a.enc_ctc_out_mid.weight", 2)
        self.ctc_mid = Linear(ctc_mid_shape[1], ctc_mid_shape[0], bias=True)
        self._ctc_layer = num_layers // 2
        raw_feature_layers = metadata.get("clip.audio.feature_layer", ())
        if raw_feature_layers is None:
            raw_feature_layers = ()
        if not isinstance(raw_feature_layers, (list, tuple)):
            raise TypeError("clip.audio.feature_layer must be an integer array when present.")
        self._feature_layers = tuple(int(index) for index in raw_feature_layers)
        if any(index < 0 or index > num_layers for index in self._feature_layers):
            raise ValueError(
                f"Granite Speech feature layers {self._feature_layers} are outside "
                f"[0, {num_layers}]."
            )

        encoder_width = hidden_size * (len(self._feature_layers) + 1)
        query_shape = _shape(shapes, "a.proj_query", 3)
        num_queries = projector_window // downsample_rate
        if query_shape[0] != 1 or query_shape[1] != num_queries:
            raise ValueError(
                f"Granite Speech query shape {query_shape} does not contain "
                f"{num_queries} queries."
            )
        projector_intermediate = _shape(shapes, "a.proj_blk.0.ffn_up.weight", 2)[0]
        output_size = _shape(shapes, "a.proj_linear.weight", 2)[0]
        projector_layers = 0
        while f"a.proj_blk.{projector_layers}.self_attn_q.weight" in shapes:
            projector_layers += 1
        if projector_layers == 0:
            raise ValueError("Granite Speech sidecar has no Q-Former layers.")
        self.projector = _GraniteSpeechQFormer(
            encoder_width,
            query_shape[2],
            projector_heads,
            projector_intermediate,
            query_shape,
            projector_layers,
            output_size,
        )
        self._projector_window = projector_window

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        hidden_states = _cast_boundary(
            op,
            input_features,
            self.input_projection.weight,
        )
        hidden_states = op.Unsqueeze(self.input_projection(op, hidden_states), [0])
        captured: list[ir.Value] = []
        if 0 in self._feature_layers:
            captured.append(hidden_states)
        for index, layer in enumerate(self.layers, start=1):
            hidden_states = layer(op, hidden_states)
            if index == self._ctc_layer:
                ctc = op.Softmax(self.ctc_out(op, hidden_states), axis=-1)
                hidden_states = op.Add(hidden_states, self.ctc_mid(op, ctc))
            if index in self._feature_layers:
                captured.append(hidden_states)
        captured.append(hidden_states)
        hidden_states = op.Concat(*captured, axis=-1) if len(captured) > 1 else captured[0]

        time = op.Shape(hidden_states, start=1, end=2)
        window = op.Constant(value_ints=[self._projector_window])
        blocks = op.Div(
            op.Add(time, op.Constant(value_ints=[self._projector_window - 1])),
            window,
        )
        padded_time = op.Mul(blocks, window)
        hidden_states = op.Pad(
            hidden_states,
            op.Concat(
                op.Constant(value_ints=[0, 0, 0, 0]),
                op.Sub(padded_time, time),
                op.Constant(value_ints=[0]),
                axis=0,
            ),
        )
        width = op.Shape(hidden_states, start=2, end=3)
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(blocks, window, width, axis=0),
        )
        hidden_states = self.projector(op, hidden_states)
        return op.Reshape(
            hidden_states,
            op.Concat(
                op.Constant(value_ints=[-1]),
                op.Shape(hidden_states, start=2, end=3),
                axis=0,
            ),
        )


def _rotary_half(
    op: OpBuilder,
    value: ir.Value,
    position_ids: ir.Value,
    *,
    head_dim: int,
    theta: float,
) -> ir.Value:
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = op.Mul(
        op.Unsqueeze(op.Cast(position_ids, to=ir.DataType.FLOAT), [-1]),
        op.Constant(value=ir.tensor(inv_freq)),
    )
    cos = op.Concat(op.Cos(angles), op.Cos(angles), axis=-1)
    sin = op.Concat(op.Sin(angles), op.Sin(angles), axis=-1)
    cos = op.CastLike(op.Unsqueeze(cos, [1]), value)
    sin = op.CastLike(op.Unsqueeze(sin, [1]), value)
    first, second = op.Split(value, axis=-1, num_outputs=2, _outputs=2)
    rotated = op.Concat(op.Neg(second), first, axis=-1)
    return op.Add(op.Mul(value, cos), op.Mul(rotated, sin))


def _causal_audio_mask(
    op: OpBuilder,
    time: ir.Value,
    reference: ir.Value,
    *,
    window_size: int | None,
) -> ir.Value:
    positions = op.Range(op.Constant(value_int=0), op.Squeeze(time), 1)
    query = op.Unsqueeze(positions, [1])
    key = op.Unsqueeze(positions, [0])
    allowed = op.LessOrEqual(key, query)
    if window_size is not None:
        allowed = op.And(
            allowed,
            op.LessOrEqual(op.Sub(query, key), op.Constant(value_int=window_size)),
        )
    return op.Unsqueeze(
        op.Unsqueeze(
            op.Where(
                allowed,
                op.CastLike(0.0, reference),
                op.CastLike(float("-inf"), reference),
            ),
            [0],
        ),
        [0],
    )


class _MimoAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        q_bias: bool,
        k_bias: bool,
        v_bias: bool,
        out_bias: bool,
        rope_theta: float,
    ):
        super().__init__()
        self.q_proj = Linear(hidden_size, hidden_size, bias=q_bias)
        self.k_proj = Linear(hidden_size, hidden_size, bias=k_bias)
        self.v_proj = Linear(hidden_size, hidden_size, bias=v_bias)
        self.out_proj = Linear(hidden_size, hidden_size, bias=out_bias)
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads
        self._rope_theta = rope_theta

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_ids: ir.Value,
        attention_bias: ir.Value,
    ) -> ir.Value:
        shape = op.Concat(
            op.Shape(hidden_states, start=0, end=2),
            op.Constant(value_ints=[self._num_heads, self._head_dim]),
            axis=0,
        )
        query = op.Transpose(
            op.Reshape(self.q_proj(op, hidden_states), shape),
            perm=[0, 2, 1, 3],
        )
        key = op.Transpose(
            op.Reshape(self.k_proj(op, hidden_states), shape),
            perm=[0, 2, 1, 3],
        )
        value = op.Transpose(
            op.Reshape(self.v_proj(op, hidden_states), shape),
            perm=[0, 2, 1, 3],
        )
        query = _rotary_half(
            op,
            query,
            position_ids,
            head_dim=self._head_dim,
            theta=self._rope_theta,
        )
        key = _rotary_half(
            op,
            key,
            position_ids,
            head_dim=self._head_dim,
            theta=self._rope_theta,
        )
        scores = op.MatMul(query, op.Transpose(key, perm=[0, 1, 3, 2]))
        scores = op.Mul(scores, op.CastLike(float(self._head_dim**-0.5), scores))
        scores = op.Add(scores, attention_bias)
        context = op.MatMul(op.Softmax(scores, axis=-1), value)
        context = op.Transpose(context, perm=[0, 2, 1, 3])
        context = op.Reshape(context, op.Shape(hidden_states))
        return self.out_proj(op, context)


class _MimoEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        eps: float,
        *,
        q_bias: bool,
        k_bias: bool,
        v_bias: bool,
        out_bias: bool,
    ):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps=eps)
        self.self_attn = _MimoAttention(
            hidden_size,
            num_heads,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias,
            out_bias=out_bias,
            rope_theta=10_000.0,
        )
        self.norm2 = LayerNorm(hidden_size, eps=eps)
        self.fc1 = Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=True)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_ids: ir.Value,
        attention_bias: ir.Value,
    ) -> ir.Value:
        hidden_states = op.Add(
            hidden_states,
            self.self_attn(
                op,
                self.norm1(op, hidden_states),
                position_ids,
                attention_bias,
            ),
        )
        feed_forward = self.fc1(op, self.norm2(op, hidden_states))
        feed_forward = self.fc2(op, op.Gelu(feed_forward))
        return op.Add(hidden_states, feed_forward)


class _MimoLocalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        eps: float,
        shapes: TensorShapes,
        index: int,
    ):
        super().__init__()
        prefix = f"mm.a.local_blk.{index}."
        self.norm1 = RMSNorm(hidden_size, eps=eps)
        self.self_attn = _MimoAttention(
            hidden_size,
            num_heads,
            q_bias=prefix + "attn_q.bias" in shapes,
            k_bias=prefix + "attn_k.bias" in shapes,
            v_bias=prefix + "attn_v.bias" in shapes,
            out_bias=False,
            rope_theta=640_000.0,
        )
        self.norm2 = RMSNorm(hidden_size, eps=eps)
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_ids: ir.Value,
        attention_bias: ir.Value,
    ) -> ir.Value:
        hidden_states = op.Add(
            hidden_states,
            self.self_attn(
                op,
                self.norm1(op, hidden_states),
                position_ids,
                attention_bias,
            ),
        )
        normalized = self.norm2(op, hidden_states)
        feed_forward = op.Mul(
            op.Swish(self.gate_proj(op, normalized)),
            self.up_proj(op, normalized),
        )
        return op.Add(hidden_states, self.down_proj(op, feed_forward))


class _MimoRVQBridge(nn.Module):
    """Residual nearest-code lookup followed by summed LLM code embeddings."""

    def __init__(
        self,
        codebook_shape: tuple[int, int, int],
        code_embedding_shape: tuple[int, int, int],
        codebook_sizes: tuple[int, ...],
    ):
        super().__init__()
        self.codebook = nn.Parameter(codebook_shape)
        self.code_embeddings = nn.Parameter(code_embedding_shape)
        self._codebook_sizes = codebook_sizes

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        residual = hidden_states
        embedded = None
        for index, bins in enumerate(self._codebook_sizes):
            codebook = op.Squeeze(
                op.Slice(
                    self.codebook,
                    op.Constant(value_ints=[index]),
                    op.Constant(value_ints=[index + 1]),
                    op.Constant(value_ints=[0]),
                ),
                [0],
            )
            codebook = op.Slice(
                codebook,
                op.Constant(value_ints=[0]),
                op.Constant(value_ints=[bins]),
                op.Constant(value_ints=[0]),
            )
            codebook_norm = op.ReduceSum(
                op.Mul(codebook, codebook),
                axes=[1],
                keepdims=0,
            )
            scores = op.Sub(
                op.Mul(
                    op.MatMul(residual, op.Transpose(codebook, perm=[1, 0])),
                    op.CastLike(2.0, residual),
                ),
                codebook_norm,
            )
            codes = op.ArgMax(scores, axis=-1, keepdims=0)
            quantized = op.Gather(codebook, codes)
            residual = op.Sub(residual, quantized)

            table = op.Squeeze(
                op.Slice(
                    self.code_embeddings,
                    op.Constant(value_ints=[index]),
                    op.Constant(value_ints=[index + 1]),
                    op.Constant(value_ints=[0]),
                ),
                [0],
            )
            table = op.Slice(
                table,
                op.Constant(value_ints=[0]),
                op.Constant(value_ints=[bins]),
                op.Constant(value_ints=[0]),
            )
            current = op.Gather(table, codes)
            embedded = current if embedded is None else op.Add(embedded, current)
        if embedded is None:
            raise RuntimeError("MiMo RVQ requires at least one quantizer.")
        return embedded


class GGUFMimoAudioProjector(nn.Module):
    """MiMo audio tokenizer, RVQ bridge, local transformer, and projection."""

    def __init__(self, metadata: Mapping[str, object], shapes: TensorShapes):
        super().__init__()
        hidden_size = _metadata_int(metadata, "clip.audio.embedding_length")
        _metadata_int(metadata, "clip.audio.feed_forward_length")
        num_layers = _metadata_int(metadata, "clip.audio.block_count")
        num_heads = _metadata_int(metadata, "clip.audio.attention.head_count")
        eps = _metadata_float(metadata, "clip.audio.attention.layer_norm_epsilon")
        self.num_mel_bins = _metadata_int(metadata, "clip.audio.num_mel_bins")
        self.input_schema = (
            (
                "input_features",
                ir.DataType.FLOAT,
                (ir.SymbolicDim("frames"), self.num_mel_bins),
            ),
        )
        self._window_size = _metadata_int(metadata, "clip.audio.window_size")
        pattern = metadata.get("clip.audio.wa_pattern_mode")
        if not isinstance(pattern, (list, tuple)) or len(pattern) != num_layers:
            raise ValueError(
                "clip.audio.wa_pattern_mode must contain one entry per MiMo audio layer."
            )
        self._windowed_layers = tuple(int(value) != -1 for value in pattern)
        self._group_size = _metadata_int(metadata, "clip.audio.local_group_size")
        local_layers = _metadata_int(metadata, "clip.audio.local_block_count")

        conv1_shape = _shape(shapes, "a.conv1d.1.weight", 3)
        conv2_shape = _shape(shapes, "a.conv1d.2.weight", 3)
        self.conv1 = Conv1d(
            self.num_mel_bins,
            hidden_size,
            conv1_shape[-1],
            stride=1,
            padding=conv1_shape[-1] // 2,
            bias=True,
        )
        self.conv2 = Conv1d(
            hidden_size,
            hidden_size,
            conv2_shape[-1],
            stride=2,
            padding=conv2_shape[-1] // 2,
            bias=True,
        )
        encoder_intermediate = _shape(shapes, "a.blk.0.ffn_up.weight", 2)[0]
        self.layers = nn.ModuleList(
            [
                _MimoEncoderLayer(
                    hidden_size,
                    encoder_intermediate,
                    num_heads,
                    eps,
                    q_bias=f"a.blk.{index}.attn_q.bias" in shapes,
                    k_bias=f"a.blk.{index}.attn_k.bias" in shapes,
                    v_bias=f"a.blk.{index}.attn_v.bias" in shapes,
                    out_bias=f"a.blk.{index}.attn_out.bias" in shapes,
                )
                for index in range(num_layers)
            ]
        )
        self.post_layernorm = LayerNorm(hidden_size, eps=eps)
        downsample_shape = _shape(shapes, "a.downsample.conv.weight", 3)
        self.downsample_conv = Conv1d(
            hidden_size,
            hidden_size,
            downsample_shape[-1],
            stride=2,
            padding=0,
            bias=False,
        )
        self.downsample_norm = LayerNorm(hidden_size, eps=eps)

        codebook_shape = _shape(shapes, "a.rvq.codebook.weight", 3)
        code_embedding_shape = _shape(shapes, "mm.a.code_embd.weight", 3)
        num_quantizers = _metadata_int(metadata, "clip.audio.rvq.num_quantizers")
        raw_sizes = metadata.get("clip.audio.rvq.codebook_size")
        if not isinstance(raw_sizes, (list, tuple)) or len(raw_sizes) != num_quantizers:
            raise ValueError(
                "clip.audio.rvq.codebook_size must contain one size per quantizer."
            )
        if codebook_shape[0] != num_quantizers or code_embedding_shape[0] != num_quantizers:
            raise ValueError(
                f"MiMo RVQ tensors {codebook_shape}/{code_embedding_shape} do not contain "
                f"{num_quantizers} quantizers."
            )
        codebook_sizes = tuple(int(value) for value in raw_sizes)
        if any(
            size <= 0 or size > codebook_shape[1] or size > code_embedding_shape[1]
            for size in codebook_sizes
        ):
            raise ValueError("MiMo RVQ codebook sizes exceed the serialized tensors.")
        self.rvq = _MimoRVQBridge(
            codebook_shape,
            code_embedding_shape,
            codebook_sizes,
        )

        local_intermediate = _shape(shapes, "mm.a.local_blk.0.ffn_up.weight", 2)[0]
        self.local_layers = nn.ModuleList(
            [
                _MimoLocalLayer(
                    hidden_size,
                    local_intermediate,
                    num_heads,
                    eps,
                    shapes,
                    index,
                )
                for index in range(local_layers)
            ]
        )
        self.local_norm = RMSNorm(hidden_size, eps=eps)
        first_projection = _shape(shapes, "mm.a.mlp.1.weight", 2)
        second_projection = _shape(shapes, "mm.a.mlp.2.weight", 2)
        if first_projection[1] != hidden_size * self._group_size:
            raise ValueError(
                f"MiMo projection expects {first_projection[1]} inputs, not "
                f"{hidden_size * self._group_size}."
            )
        self.projector = _GeluProjector(
            first_projection[1],
            first_projection[0],
            second_projection[0],
            first_bias=False,
            second_bias=False,
        )

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        hidden_states = op.Transpose(input_features, perm=[1, 0])
        hidden_states = op.Unsqueeze(hidden_states, [0])
        hidden_states = _cast_boundary(op, hidden_states, self.conv1.weight)
        hidden_states = op.Gelu(self.conv1(op, hidden_states))
        hidden_states = op.Gelu(self.conv2(op, hidden_states))
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        time = op.Shape(hidden_states, start=1, end=2)
        position_ids = op.Unsqueeze(
            op.Range(op.Constant(value_int=0), op.Squeeze(time), 1),
            [0],
        )
        full_bias = _causal_audio_mask(op, time, hidden_states, window_size=None)
        window_bias = _causal_audio_mask(
            op,
            time,
            hidden_states,
            window_size=self._window_size,
        )
        skip_hidden = None
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                op,
                hidden_states,
                position_ids,
                window_bias if self._windowed_layers[index] else full_bias,
            )
            if index == 2:
                skip_hidden = hidden_states
        if skip_hidden is None:
            raise ValueError("MiMo audio encoder requires at least three layers for its skip.")
        hidden_states = self.post_layernorm(op, op.Add(hidden_states, skip_hidden))

        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = op.Gelu(self.downsample_conv(op, hidden_states))
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = self.downsample_norm(op, hidden_states)
        hidden_states = self.rvq(op, hidden_states)

        time = op.Shape(hidden_states, start=1, end=2)
        group = op.Constant(value_ints=[self._group_size])
        groups = op.Div(
            op.Add(time, op.Constant(value_ints=[self._group_size - 1])),
            group,
        )
        padded_time = op.Mul(groups, group)
        hidden_states = op.Pad(
            hidden_states,
            op.Concat(
                op.Constant(value_ints=[0, 0, 0, 0]),
                op.Sub(padded_time, time),
                op.Constant(value_ints=[0]),
                axis=0,
            ),
        )
        positions = op.Range(op.Constant(value_int=0), op.Squeeze(padded_time), 1)
        position_ids = op.Unsqueeze(op.Mod(positions, self._group_size), [0])
        group_ids = op.Div(positions, self._group_size)
        allowed = op.Equal(op.Unsqueeze(group_ids, [0]), op.Unsqueeze(group_ids, [1]))
        attention_bias = op.Unsqueeze(
            op.Unsqueeze(
                op.Where(
                    allowed,
                    op.CastLike(0.0, hidden_states),
                    op.CastLike(float("-inf"), hidden_states),
                ),
                [0],
            ),
            [0],
        )
        for layer in self.local_layers:
            hidden_states = layer(op, hidden_states, position_ids, attention_bias)
        hidden_states = self.local_norm(op, hidden_states)
        hidden_states = op.Reshape(
            hidden_states,
            op.Concat(
                op.Constant(value_ints=[1]),
                groups,
                op.Constant(value_ints=[-1]),
                axis=0,
            ),
        )
        return op.Squeeze(self.projector(op, hidden_states), [0])


class _PocketCausalConv1d(nn.Module):
    """PocketTTS causal convolution with ceil-length trailing padding."""

    def __init__(
        self,
        shape: Sequence[int],
        *,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
    ):
        super().__init__()
        if len(shape) != 3:
            raise ValueError(f"PocketTTS convolution shape must be rank 3, got {shape}.")
        self.weight = nn.Parameter(list(shape))
        self.bias = nn.Parameter([int(shape[0])]) if bias else None
        self._kernel = int(shape[2])
        self._stride = stride
        self._dilation = dilation
        self._pad_mode = pad_mode

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        length = op.Shape(hidden_states, start=2, end=3)
        stride = op.Constant(value_ints=[self._stride])
        padded_length = op.Mul(
            op.Div(op.Add(length, op.Constant(value_ints=[self._stride - 1])), stride),
            stride,
        )
        extra = op.Sub(padded_length, length)
        effective_kernel = (self._kernel - 1) * self._dilation + 1
        left = effective_kernel - self._stride
        pads = op.Concat(
            op.Constant(value_ints=[0, 0, left, 0, 0]),
            extra,
            axis=0,
        )
        hidden_states = op.Pad(hidden_states, pads, mode=self._pad_mode)
        args = (
            (hidden_states, self.weight)
            if self.bias is None
            else (hidden_states, self.weight, self.bias)
        )
        return op.Conv(
            *args,
            kernel_shape=[self._kernel],
            strides=[self._stride],
            dilations=[self._dilation],
            pads=[0, 0],
        )


class _PocketResidualUnit(nn.Module):
    def __init__(self, first_shape: Sequence[int], second_shape: Sequence[int]):
        super().__init__()
        self.conv1 = _PocketCausalConv1d(first_shape, dilation=1, bias=True)
        self.conv2 = _PocketCausalConv1d(second_shape, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        residual = hidden_states
        hidden_states = self.conv1(op, op.Elu(hidden_states, alpha=1.0))
        hidden_states = self.conv2(op, op.Elu(hidden_states, alpha=1.0))
        return op.Add(residual, hidden_states)


class _PocketSEANetStage(nn.Module):
    def __init__(
        self,
        residual_first: Sequence[int],
        residual_second: Sequence[int],
        scale_shape: Sequence[int],
    ):
        super().__init__()
        self.residual = _PocketResidualUnit(residual_first, residual_second)
        kernel = int(scale_shape[-1])
        if kernel % 2:
            raise ValueError(f"PocketTTS scale convolution kernel must be even, got {kernel}.")
        self.scale = _PocketCausalConv1d(
            scale_shape,
            stride=kernel // 2,
            bias=True,
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = self.residual(op, hidden_states)
        return self.scale(op, op.Elu(hidden_states, alpha=1.0))


class _PocketSEANetEncoder(nn.Module):
    def __init__(self, shapes: TensorShapes):
        super().__init__()
        self.conv_in = _PocketCausalConv1d(
            _shape(shapes, "a.seanet.conv_in.weight", 3),
            bias="a.seanet.conv_in.bias" in shapes,
        )
        stages = []
        index = 0
        while f"a.seanet.blk.{index}.scale_conv.weight" in shapes:
            stages.append(
                _PocketSEANetStage(
                    _shape(shapes, f"a.seanet.blk.{index}.res_conv1.weight", 3),
                    _shape(shapes, f"a.seanet.blk.{index}.res_conv2.weight", 3),
                    _shape(shapes, f"a.seanet.blk.{index}.scale_conv.weight", 3),
                )
            )
            index += 1
        if not stages:
            raise ValueError("PocketTTS sidecar has no SEANet encoder stages.")
        self.stages = nn.ModuleList(stages)
        self.conv_out = _PocketCausalConv1d(
            _shape(shapes, "a.seanet.conv_out.weight", 3),
            bias="a.seanet.conv_out.bias" in shapes,
        )
        self.frame_hop = math.prod(stage.scale._stride for stage in stages)

    def forward(self, op: OpBuilder, waveform: ir.Value) -> ir.Value:
        hidden_states = self.conv_in(op, waveform)
        for stage in self.stages:
            hidden_states = stage(op, hidden_states)
        return self.conv_out(op, op.Elu(hidden_states, alpha=1.0))


def _guard_false(
    op: OpBuilder,
    invalid: ir.Value,
    reference: ir.Value,
) -> ir.Value:
    """Make an invalid dynamic ABI condition raise through an out-of-range Gather."""
    index = op.Cast(invalid, to=ir.DataType.INT64)
    sentinel = op.Gather(op.Constant(value_ints=[0]), index, axis=0)
    return op.Add(reference, op.CastLike(sentinel, reference))


class GGUFPocketTTSSpeakerEncoder(nn.Module):
    """PocketTTS raw-waveform Mimi speaker encoder; excludes ``pockettts_gen``."""

    def __init__(self, metadata: Mapping[str, object], shapes: TensorShapes):
        super().__init__()
        hidden_size = _metadata_int(metadata, "clip.audio.embedding_length")
        intermediate_size = _metadata_int(metadata, "clip.audio.feed_forward_length")
        num_layers = _metadata_int(metadata, "clip.audio.block_count")
        num_heads = _metadata_int(metadata, "clip.audio.attention.head_count")
        eps = _metadata_float(metadata, "clip.audio.attention.layer_norm_epsilon")
        if _metadata_int(metadata, "clip.audio.num_mel_bins") != 1:
            raise ValueError(
                "PocketTTS speaker input is one raw waveform channel, not mel bins."
            )
        head_dim = hidden_size // num_heads
        if head_dim != 64:
            raise ValueError(
                f"PocketTTS speaker encoder requires head_dim=64, got {head_dim}."
            )
        self.input_schema = (
            (
                "input_values",
                ir.DataType.FLOAT,
                (ir.SymbolicDim("samples"),),
            ),
        )
        self.seanet = _PocketSEANetEncoder(shapes)
        self.transformer = CodecEncoderTransformerModel(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            rope_theta=10_000.0,
            max_position_embeddings=10_000,
            layer_norm_eps=eps,
        )
        downsample_shape = _shape(shapes, "a.downsample.conv.weight", 3)
        self.downsample = _PocketCausalConv1d(
            downsample_shape,
            stride=16,
            bias=False,
            pad_mode="edge",
        )
        speaker_shape = _shape(shapes, "a.speaker_proj.weight", 2)
        self.speaker_projection = Linear(
            speaker_shape[1],
            speaker_shape[0],
            bias=False,
        )
        self._frame_hop = self.seanet.frame_hop * 16
        self._max_samples = 30 * 24_000
        if self._frame_hop != 1_920:
            raise ValueError(
                f"PocketTTS speaker encoder frame hop must be 1920 samples, got "
                f"{self._frame_hop}."
            )

    def forward(self, op: OpBuilder, input_values: ir.Value) -> ir.Value:
        samples = op.Shape(input_values, start=0, end=1)
        invalid = op.Or(
            op.Not(op.Equal(op.Mod(samples, self._frame_hop), 0)),
            op.Greater(samples, op.Constant(value_ints=[self._max_samples])),
        )
        waveform = op.Unsqueeze(op.Unsqueeze(input_values, [0]), [0])
        waveform = _cast_boundary(op, waveform, self.seanet.conv_in.weight)
        waveform = _guard_false(op, op.Squeeze(invalid), waveform)
        hidden_states = self.seanet(op, waveform)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])

        time = op.Shape(hidden_states, start=1, end=2)
        position_ids = op.Unsqueeze(
            op.Range(op.Constant(value_int=0), op.Squeeze(time), 1),
            [0],
        )
        positions = op.Range(op.Constant(value_int=0), op.Squeeze(time), 1)
        query = op.Unsqueeze(positions, [1])
        key = op.Unsqueeze(positions, [0])
        allowed = op.And(
            op.LessOrEqual(key, query),
            op.Less(op.Sub(query, key), op.Constant(value_int=250)),
        )
        attention_mask = op.Unsqueeze(op.Unsqueeze(allowed, [0]), [0])
        hidden_states = self.transformer(
            op,
            hidden_states,
            position_ids,
            attention_mask,
        )
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = self.downsample(op, hidden_states)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = self.speaker_projection(op, hidden_states)
        return op.Squeeze(hidden_states, [0])


def create_gguf_audio_projector(
    projector_type: str,
    metadata: Mapping[str, object],
    tensor_shapes: TensorShapes,
) -> nn.Module:
    """Create the exact reusable graph family for one supported audio route."""
    if projector_type in {"ultravox", "voxtral", "musicflamingo"}:
        encoder: nn.Module = GGUFWhisperAudioProjector(
            projector_type,
            metadata,
            tensor_shapes,
        )
    elif projector_type == "parakeet":
        encoder = GGUFParakeetAudioProjector(metadata, tensor_shapes)
    elif projector_type == "lfm2a":
        encoder = GGUFLFM2AudioProjector(metadata, tensor_shapes)
    elif projector_type == "granite_speech":
        encoder = GGUFGraniteSpeechProjector(metadata, tensor_shapes)
    elif projector_type == "mimo_audio":
        encoder = GGUFMimoAudioProjector(metadata, tensor_shapes)
    elif projector_type == "pockettts_spkenc":
        encoder = GGUFPocketTTSSpeakerEncoder(metadata, tensor_shapes)
    elif projector_type == "meralion":
        hidden_size = _metadata_int(metadata, "clip.audio.embedding_length")
        projection_dim = _metadata_int(metadata, "clip.audio.projection_dim")
        stack_factor = _metadata_int(metadata, "clip.audio.projector.stack_factor")
        num_mel_bins = _metadata_int(metadata, "clip.audio.num_mel_bins")
        position_shape = _shape(tensor_shapes, "a.position_embd.weight", 2)
        first_shape = _shape(tensor_shapes, "mm.a.mlp.0.weight", 2)
        gate_shape = _shape(tensor_shapes, "mm.a.mlp.1.weight", 2)
        pool_shape = _shape(tensor_shapes, "mm.a.mlp.2.weight", 2)
        output_shape = _shape(tensor_shapes, "mm.a.mlp.3.weight", 2)
        if (
            num_mel_bins != 128
            or position_shape != (1500, hidden_size)
            or stack_factor != 15
            or first_shape[1] != hidden_size * stack_factor
            or gate_shape != (first_shape[0], first_shape[0])
            or pool_shape != gate_shape
            or output_shape != (projection_dim, first_shape[0])
        ):
            raise ValueError(
                "meralion position/projector shapes do not form the pinned "
                "Whisper -> stack -> gated adapter contract"
            )
        encoder = MeralionAudioSidecar(
            num_mel_bins=num_mel_bins,
            d_model=hidden_size,
            encoder_layers=_metadata_int(metadata, "clip.audio.block_count"),
            encoder_heads=_metadata_int(metadata, "clip.audio.attention.head_count"),
            encoder_ffn_dim=_metadata_int(metadata, "clip.audio.feed_forward_length"),
            max_source_positions=position_shape[0],
            projector_hidden_size=first_shape[0],
            output_size=output_shape[0],
            stack_factor=stack_factor,
            eps=_metadata_float(metadata, "clip.audio.attention.layer_norm_epsilon"),
        )
    else:
        raise NotImplementedError(
            f"GGUF audio projector {projector_type!r} is not implemented."
        )
    return encoder
