# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable Yasa2 vision and MERaLiON audio clip-sidecar graphs.

The layouts follow llama.cpp commit 8d9af256337d1a501250f9bbf4c0859a654bddd6.
``Yasa2VisionSidecar`` has a truthful single-tile NCHW contract; it does not
implement the checkpoint processor's multi-tile llava-uhd composition.
"""

from __future__ import annotations

from collections.abc import Sequence

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._common import LayerNorm, Linear
from mobius.components._conv import Conv2d
from mobius.components._whisper import Conv1d, WhisperEncoderLayer


class Yasa2ChannelsFirstLayerNorm(nn.Module):
    """ConvNeXt channels-first LayerNorm for an NCHW activation."""

    def __init__(self, channels: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter([channels])
        self.bias = nn.Parameter([channels])
        self._eps = eps

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # llama.cpp reduces C and clamps the variance for allocation-warmup safety.
        x_f32 = op.Cast(x, to=ir.DataType.FLOAT)
        mean = op.ReduceMean(x_f32, [1], keepdims=1)
        centered = op.Sub(x_f32, mean)
        variance = op.ReduceMean(op.Mul(centered, centered), [1], keepdims=1)
        denominator = op.Sqrt(op.Clip(variance, self._eps, 1e30))
        normalized = op.CastLike(op.Div(centered, denominator), x)
        scale = op.Reshape(op.CastLike(self.weight, x), [1, -1, 1, 1])
        bias = op.Reshape(op.CastLike(self.bias, x), [1, -1, 1, 1])
        return op.Add(op.Mul(normalized, scale), bias)


class Yasa2GlobalResponseNorm(nn.Module):
    """ConvNeXtV2 global response normalization with float32 accumulation."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter([channels])
        self.bias = nn.Parameter([channels])

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # Gx = ||x|| over H,W; Nx = Gx / clamp(mean_C(Gx), 1e-6).
        x_f32 = op.Cast(x, to=ir.DataType.FLOAT)
        gx = op.Sqrt(op.ReduceSum(op.Mul(x_f32, x_f32), [2, 3], keepdims=1))
        gx_mean = op.ReduceMean(gx, [1], keepdims=1)
        nx = op.Div(gx, op.Clip(gx_mean, 1e-6, 1e30))
        scale = op.Reshape(op.Cast(self.weight, to=ir.DataType.FLOAT), [1, -1, 1, 1])
        bias = op.Reshape(op.Cast(self.bias, to=ir.DataType.FLOAT), [1, -1, 1, 1])
        response = op.Add(op.Mul(op.Mul(x_f32, nx), scale), bias)
        return op.CastLike(op.Add(x_f32, response), x)


class _Yasa2Block(nn.Module):
    def __init__(self, channels: int, eps: float):
        super().__init__()
        expanded = 4 * channels
        self.depthwise = Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.layer_norm = Yasa2ChannelsFirstLayerNorm(channels, eps)
        self.pointwise_up = Linear(channels, expanded)
        self.grn = Yasa2GlobalResponseNorm(expanded)
        self.pointwise_down = Linear(expanded, channels)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        residual = x
        x = self.depthwise(op, x)
        x = self.layer_norm(op, x)
        # ConvNeXt pointwise layers are Linear operators over NHWC channels.
        x = op.Transpose(x, perm=[0, 2, 3, 1])
        x = op.Gelu(self.pointwise_up(op, x))
        x = self.grn(op, op.Transpose(x, perm=[0, 3, 1, 2]))
        x = op.Transpose(x, perm=[0, 2, 3, 1])
        x = self.pointwise_down(op, x)
        return op.Add(residual, op.Transpose(x, perm=[0, 3, 1, 2]))


class _Yasa2Stage(nn.Module):
    def __init__(self, in_channels: int, channels: int, depth: int, eps: float):
        super().__init__()
        self.downsample_norm = (
            Yasa2ChannelsFirstLayerNorm(in_channels, eps) if in_channels != channels else None
        )
        self.downsample_conv = (
            Conv2d(in_channels, channels, kernel_size=2, stride=2)
            if in_channels != channels
            else None
        )
        self.blocks = nn.ModuleList([_Yasa2Block(channels, eps) for _ in range(depth)])

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        if self.downsample_norm is not None:
            x = self.downsample_norm(op, x)
            assert self.downsample_conv is not None
            x = self.downsample_conv(op, x)
        for block in self.blocks:
            x = block(op, x)
        return x


class Yasa2VisionSidecar(nn.Module):
    """Yasa2 ConvNeXtV2 single-tile vision sidecar.

    Input is ``pixel_values`` with shape ``[B, 3, image_size, image_size]``.
    Production Yasa2 uses a 512x512 tile and returns 64 projected tokens.
    """

    def __init__(
        self,
        depths: Sequence[int],
        hidden_sizes: Sequence[int],
        projector_hidden_size: int,
        output_size: int,
        *,
        image_size: int = 512,
        in_channels: int = 3,
        eps: float = 1e-12,
    ):
        super().__init__()
        if not depths or len(depths) != len(hidden_sizes):
            raise ValueError("depths and hidden_sizes must be non-empty and have equal length")
        final_grid = image_size // (4 * 2 ** (len(depths) - 1))
        if image_size % (4 * 2 ** (len(depths) - 1)) or final_grid < 8 or final_grid % 8:
            raise ValueError("image_size must produce a final spatial grid divisible by 8")

        first = int(hidden_sizes[0])
        self.patch_embedding = Conv2d(in_channels, first, kernel_size=4, stride=4)
        self.patch_layer_norm = Yasa2ChannelsFirstLayerNorm(first, eps)
        stages: list[nn.Module] = []
        previous = first
        for depth, channels in zip(depths, hidden_sizes, strict=True):
            stages.append(_Yasa2Stage(previous, int(channels), int(depth), eps))
            previous = int(channels)
        self.stages = nn.ModuleList(stages)
        self.vision_position_embedding = nn.Parameter([final_grid * final_grid, previous])
        self.projector_up = Linear(previous, projector_hidden_size)
        self.projector_down = Linear(projector_hidden_size, output_size)
        self._final_grid = final_grid
        self._pool_kernel = final_grid // 8

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        pixel_values = op.CastLike(pixel_values, self.patch_embedding.weight)
        x = self.patch_layer_norm(op, self.patch_embedding(op, pixel_values))
        for stage in self.stages:
            x = stage(op, x)

        # Add positions on the full final-stage grid before the 8x8 average pool.
        x = op.Transpose(x, perm=[0, 2, 3, 1])
        x = op.Reshape(x, [0, self._final_grid * self._final_grid, -1])
        x = op.Add(x, op.CastLike(self.vision_position_embedding, x))
        x = op.Reshape(x, [0, self._final_grid, self._final_grid, -1])
        x = op.Transpose(x, perm=[0, 3, 1, 2])
        x = op.AveragePool(
            x,
            kernel_shape=[self._pool_kernel, self._pool_kernel],
            strides=[self._pool_kernel, self._pool_kernel],
        )
        x = op.Transpose(x, perm=[0, 2, 3, 1])
        x = op.Reshape(x, [0, 64, -1])
        return self.projector_down(op, op.Gelu(self.projector_up(op, x)))


class MeralionProjector(nn.Module):
    """Pinned llama.cpp MERaLiON stack-then-norm gated audio projector."""

    def __init__(
        self,
        d_model: int,
        projector_hidden_size: int,
        output_size: int,
        *,
        stack_factor: int = 15,
        eps: float = 1e-5,
    ):
        super().__init__()
        if stack_factor <= 0:
            raise ValueError("stack_factor must be positive")
        self.norm_weight = nn.Parameter([d_model])
        self.norm_bias = nn.Parameter([d_model])
        self.linear0 = Linear(d_model * stack_factor, projector_hidden_size)
        self.linear1 = Linear(projector_hidden_size, projector_hidden_size)
        self.linear2 = Linear(projector_hidden_size, projector_hidden_size)
        self.linear3 = Linear(projector_hidden_size, output_size)
        self._d_model = d_model
        self._stack_factor = stack_factor
        self._eps = eps

    def forward(self, op: OpBuilder, encoder_output: ir.Value) -> ir.Value:
        # Stack consecutive frames first. Norm statistics span all S*C values,
        # while the C-element affine repeats over each of the S frames.
        grouped = op.Reshape(
            encoder_output,
            [0, -1, self._stack_factor, self._d_model],
        )
        grouped_f32 = op.Cast(grouped, to=ir.DataType.FLOAT)
        mean = op.ReduceMean(grouped_f32, [-2, -1], keepdims=1)
        centered = op.Sub(grouped_f32, mean)
        variance = op.ReduceMean(op.Mul(centered, centered), [-2, -1], keepdims=1)
        normalized = op.Div(centered, op.Sqrt(op.Add(variance, self._eps)))
        normalized = op.CastLike(normalized, grouped)
        normalized = op.Add(
            op.Mul(normalized, op.CastLike(self.norm_weight, grouped)),
            op.CastLike(self.norm_bias, grouped),
        )
        stacked = op.Reshape(
            normalized,
            [0, -1, self._stack_factor * self._d_model],
        )
        hidden = self.linear0(op, stacked)
        hidden = op.Mul(hidden, op.Sigmoid(hidden))
        gate = self.linear1(op, hidden)
        gate = op.Mul(gate, op.Sigmoid(gate))
        pool = self.linear2(op, hidden)
        return self.linear3(op, op.Mul(gate, pool))


class MeralionAudioSidecar(nn.Module):
    """Whisper encoder followed by the pinned MERaLiON projector."""

    def __init__(
        self,
        *,
        num_mel_bins: int,
        d_model: int,
        encoder_layers: int,
        encoder_heads: int,
        encoder_ffn_dim: int,
        max_source_positions: int,
        projector_hidden_size: int,
        output_size: int,
        stack_factor: int = 15,
        eps: float = 1e-5,
    ):
        super().__init__()
        if max_source_positions % stack_factor:
            raise ValueError("max_source_positions must be divisible by stack_factor")
        self.input_schema = (
            (
                "input_features",
                ir.DataType.FLOAT,
                (max_source_positions * 2, num_mel_bins),
            ),
        )
        self.conv1 = Conv1d(num_mel_bins, d_model, kernel_size=3, padding=1)
        self.conv2 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.position_embeddings = nn.Parameter([max_source_positions, d_model])
        self.layers = nn.ModuleList(
            [
                WhisperEncoderLayer(d_model, encoder_heads, encoder_ffn_dim, "gelu", eps)
                for _ in range(encoder_layers)
            ]
        )
        self.layer_norm = LayerNorm(d_model, eps)
        self.projector = MeralionProjector(
            d_model,
            projector_hidden_size,
            output_size,
            stack_factor=stack_factor,
            eps=eps,
        )

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        # Processor boundary is (frames, mel); Conv1d consumes (1, mel, frames).
        x = op.Unsqueeze(op.Transpose(input_features, perm=[1, 0]), [0])
        x = op.CastLike(x, self.conv1.weight)
        x = op.Gelu(self.conv1(op, x))
        x = op.Gelu(self.conv2(op, x))
        x = op.Transpose(x, perm=[0, 2, 1])
        x = op.Add(x, op.CastLike(self.position_embeddings, x))
        for layer in self.layers:
            x = layer(op, x)
        return op.Squeeze(self.projector(op, self.layer_norm(op, x)), [0])
