# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable audio encoder/projectors used by GGUF ``clip`` sidecars."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._activations import get_activation
from mobius.components._common import LayerNorm, Linear
from mobius.components._whisper import Conv1d, WhisperEncoderLayer


class GGUFWhisperAudioTower(nn.Module):
    """Whisper encoder stored by llama.cpp audio-projector sidecars.

    The processor boundary is a log-mel tensor ``[B, n_mels, frames]``.
    Qwen2-Audio enables the final two-frame average pool; legacy GLM-ASR
    leaves the post-convolution sequence unpooled before its own frame stack.
    """

    def __init__(
        self,
        *,
        num_mel_bins: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        max_source_positions: int,
        norm_eps: float,
        average_pool: bool,
    ) -> None:
        super().__init__()
        self.conv1 = Conv1d(num_mel_bins, hidden_size, kernel_size=3, padding=1)
        self.conv2 = Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.position_embeddings = nn.Parameter([max_source_positions, hidden_size])
        self.layers = nn.ModuleList(
            [
                WhisperEncoderLayer(
                    hidden_size,
                    num_attention_heads,
                    intermediate_size,
                    activation="gelu",
                    eps=norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.post_layernorm = LayerNorm(hidden_size, eps=norm_eps)
        self._average_pool = average_pool

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        input_features = op.CastLike(input_features, self.conv1.weight)
        hidden_states = op.Gelu(self.conv1(op, input_features), approximate="none")
        hidden_states = op.Gelu(self.conv2(op, hidden_states), approximate="none")
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        # [B, ceil(frames / 2), hidden]

        sequence_length = op.Shape(hidden_states, start=1, end=2)
        positions = op.Slice(
            self.position_embeddings,
            op.Constant(value_ints=[0]),
            sequence_length,
            op.Constant(value_ints=[0]),
        )
        hidden_states = op.Add(hidden_states, op.CastLike(positions, hidden_states))
        for layer in self.layers:
            hidden_states = layer(op, hidden_states)

        if self._average_pool:
            hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
            hidden_states = op.AveragePool(
                hidden_states,
                kernel_shape=[2],
                strides=[2],
            )
            hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
            # [B, floor(ceil(frames / 2) / 2), hidden]
        return self.post_layernorm(op, hidden_states)


class GGUFQwen2AudioProjector(nn.Module):
    """Qwen2-Audio Whisper tower followed by one text-width affine projection."""

    def __init__(
        self,
        *,
        num_mel_bins: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        max_source_positions: int,
        output_size: int,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.audio_tower = GGUFWhisperAudioTower(
            num_mel_bins=num_mel_bins,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_source_positions=max_source_positions,
            norm_eps=norm_eps,
            average_pool=True,
        )
        self.projection = Linear(hidden_size, output_size, bias=True)
        self.input_schema = (
            (
                "input_features",
                ir.DataType.FLOAT,
                (1, num_mel_bins, 2 * max_source_positions),
            ),
        )

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        projected = self.projection(op, self.audio_tower(op, input_features))
        return op.Squeeze(projected, [0])


class GGUFLegacyGlmAudioProjector(nn.Module):
    """Legacy llama.cpp GLMA Whisper tower and boundary-token projector.

    This component documents and tests the serialized graph. The current
    GLM-ASR checkpoint instead uses partial RoPE and cannot be converted into
    this legacy topology, so registry dispatch remains fail-closed.
    """

    def __init__(
        self,
        *,
        num_mel_bins: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        max_source_positions: int,
        stack_factor: int,
        projector_intermediate_size: int,
        output_size: int,
        norm_eps: float = 1e-5,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if stack_factor <= 0:
            raise ValueError("GLMA stack factor must be positive")
        self.audio_tower = GGUFWhisperAudioTower(
            num_mel_bins=num_mel_bins,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_source_positions=max_source_positions,
            norm_eps=norm_eps,
            average_pool=False,
        )
        self.pre_projector_norm = LayerNorm(hidden_size, eps=norm_eps)
        self.linear_1 = Linear(
            hidden_size * stack_factor,
            projector_intermediate_size,
            bias=True,
        )
        self.linear_2 = Linear(projector_intermediate_size, output_size, bias=True)
        self.boi = nn.Parameter([output_size])
        self.eoi = nn.Parameter([output_size])
        self._activation = get_activation(activation)
        self._stack_factor = stack_factor
        self._hidden_size = hidden_size
        self._output_size = output_size
        self.input_schema = (
            (
                "input_features",
                ir.DataType.FLOAT,
                (1, num_mel_bins, 2 * max_source_positions),
            ),
        )

    def _stack_frames(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        sequence_length = op.Squeeze(op.Shape(hidden_states, start=1, end=2), [0])
        factor = op.Constant(value_int=self._stack_factor)
        remainder = op.Mod(sequence_length, factor)
        pad_length = op.Mod(op.Sub(factor, remainder), factor)
        pads = op.Concat(
            op.Constant(value_ints=[0, 0, 0, 0]),
            op.Reshape(pad_length, [1]),
            op.Constant(value_ints=[0]),
            axis=0,
        )
        hidden_states = op.Pad(hidden_states, pads)
        return op.Reshape(
            hidden_states,
            [0, -1, self._hidden_size * self._stack_factor],
        )

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        hidden_states = self.pre_projector_norm(
            op,
            self.audio_tower(op, input_features),
        )
        hidden_states = self._stack_frames(op, hidden_states)
        hidden_states = self.linear_2(
            op,
            self._activation(op, self.linear_1(op, hidden_states)),
        )

        batch = op.Shape(hidden_states, start=0, end=1)
        boundary_shape = op.Concat(batch, [1, self._output_size], axis=0)
        boi = op.Expand(op.Reshape(self.boi, [1, 1, self._output_size]), boundary_shape)
        eoi = op.Expand(op.Reshape(self.eoi, [1, 1, self._output_size]), boundary_shape)
        return op.Squeeze(op.Concat(boi, hidden_states, eoi, axis=1), [0])
