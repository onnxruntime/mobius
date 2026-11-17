# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native neural components for the MiniMax Music 3 modular diffusion pipeline."""

from __future__ import annotations

import math

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius.components import Embedding, LayerNorm, Linear, RMSNorm, TimestepEmbedding
from mobius.integrations.diffusers._configs import (
    MiniMaxMusic3ConditionConfig,
    MiniMaxMusic3RVQConfig,
    MiniMaxMusic3TransformerConfig,
    MiniMaxMusic3VocoderConfig,
)
from mobius.models.qwen import Qwen3CausalLMModel


class _Conv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter([out_channels, in_channels, kernel_size])
        self.bias = nn.Parameter([out_channels]) if bias else None
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        args = (hidden_states, self.weight)
        if self.bias is not None:
            args += (self.bias,)
        return op.Conv(
            *args,
            kernel_shape=[self._kernel_size],
            strides=[self._stride],
            pads=[self._padding, self._padding],
            dilations=[self._dilation],
        )


class _ConvTranspose1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.weight = nn.Parameter([in_channels, out_channels, 2 * stride])
        self.bias = nn.Parameter([out_channels])
        self._stride = stride

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        padding = math.ceil(self._stride / 2)
        return op.ConvTranspose(
            hidden_states,
            self.weight,
            self.bias,
            kernel_shape=[2 * self._stride],
            strides=[self._stride],
            pads=[padding, padding],
        )


class _DepthAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.to_q = Linear(dim, dim, bias=False)
        self.to_k = Linear(dim, dim, bias=False)
        self.to_v = Linear(dim, dim, bias=False)
        self.to_out = Linear(dim, dim, bias=False)
        self._heads = heads
        self._head_dim = dim // heads

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        query = self.to_q(op, hidden_states)
        key = self.to_k(op, hidden_states)
        value = self.to_v(op, hidden_states)
        seq_len = op.Squeeze(op.Shape(hidden_states, start=1, end=2), [0])
        positions = op.Range(op.Constant(value_int=0), seq_len, op.Constant(value_int=1))
        query_positions = op.Unsqueeze(positions, [1])
        key_positions = op.Unsqueeze(positions, [0])
        causal_mask = op.LessOrEqual(key_positions, query_positions)
        shape = op.Shape(query, start=0, end=2)
        qkv_shape = op.Concat(
            shape,
            op.Constant(value_ints=[self._heads, self._head_dim]),
            axis=0,
        )
        query = op.Transpose(op.Reshape(query, qkv_shape), perm=[0, 2, 1, 3])
        key = op.Transpose(op.Reshape(key, qkv_shape), perm=[0, 2, 1, 3])
        value = op.Transpose(op.Reshape(value, qkv_shape), perm=[0, 2, 1, 3])
        scores = op.Mul(
            op.MatMul(query, op.Transpose(key, perm=[0, 1, 3, 2])),
            self._head_dim**-0.5,
        )
        neg_inf = op.CastLike(float("-inf"), scores)
        scores = op.Where(causal_mask, scores, neg_inf)
        attended = op.MatMul(op.Softmax(scores, axis=-1), value)
        attended = op.Transpose(attended, perm=[0, 2, 1, 3])
        attended = op.Reshape(
            attended,
            op.Concat(
                shape,
                op.Constant(value_ints=[self._heads * self._head_dim]),
                axis=0,
            ),
        )
        return self.to_out(op, attended)


class _DepthDecoderBlock(nn.Module):
    def __init__(self, config: MiniMaxMusic3RVQConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = _DepthAttention(config.hidden_size, config.num_attention_heads)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=1e-6)
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states = op.Add(residual, self.attn(op, hidden_states))
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        gate = self.gate_proj(op, hidden_states)
        gate = op.Mul(gate, op.Sigmoid(gate))
        hidden_states = self.down_proj(op, op.Mul(gate, self.up_proj(op, hidden_states)))
        return op.Add(residual, hidden_states)


class MiniMaxMusic3RVQDepthDecoder(nn.Module):
    """Local causal decoder rerun at growing lengths 2..8 for residual RVQ codes."""

    default_task = "minimax-music3-rvq"
    category = "Audio"

    def __init__(self, config: MiniMaxMusic3RVQConfig):
        super().__init__()
        self.audio_embeddings = Embedding(
            config.audio_vocab_size * (config.num_codebooks - 1), config.hidden_size
        )
        self.projection = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.pos_embedding = Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList(
            [_DepthDecoderBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=1e-6)
        self.audio_heads = nn.ModuleList(
            [
                Linear(config.hidden_size, config.audio_vocab_size, bias=False)
                for _ in range(config.num_codebooks - 1)
            ]
        )

    def forward(self, op: OpBuilder, inputs_embeds: ir.Value):
        # Add learned positions to the projected within-frame sequence.
        seq_len = op.Shape(inputs_embeds, start=1, end=2)
        positions = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(seq_len, [0]),
            op.Constant(value_int=1),
        )
        hidden_states = op.Add(
            inputs_embeds, op.Unsqueeze(self.pos_embedding(op, positions), [0])
        )
        for layer in self.layers:
            hidden_states = layer(op, hidden_states)
        return self.norm(op, hidden_states)


class MiniMaxMusic3ConditionEncoder(nn.Module):
    """Mix one global-final plus seven RVQ-step hiddens onto the Flow-VAE timeline."""

    default_task = "minimax-music3-condition"
    category = "Audio"

    def __init__(self, config: MiniMaxMusic3ConditionConfig):
        super().__init__()
        self.layer_weight_logits = nn.Parameter([config.num_condition_layers])
        self.layer_scale = nn.Parameter([1])
        self.proj = _Conv1d(config.condition_hidden_dim, config.out_dim, 3, padding=1)
        self._config = config

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        # [B, frames, layers*hidden] -> [B, layers, hidden, frames].
        batch = op.Shape(hidden_states, start=0, end=1)
        frames = op.Shape(hidden_states, start=1, end=2)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        target_shape = op.Concat(
            batch,
            op.Constant(
                value_ints=[
                    self._config.num_condition_layers,
                    self._config.condition_hidden_dim,
                ]
            ),
            frames,
            axis=0,
        )
        hidden_states = op.Reshape(hidden_states, target_shape)
        weights = op.Softmax(self.layer_weight_logits, axis=0)
        weights = op.Unsqueeze(weights, op.Constant(value_ints=[0, 2, 3]))
        hidden_states = op.ReduceSum(
            op.Mul(hidden_states, weights),
            op.Constant(value_ints=[1]),
            keepdims=0,
        )
        hidden_states = self.proj(op, op.Mul(hidden_states, self.layer_scale))

        # Match Python int() exactly with positive integer arithmetic (floor).
        numerator = self._config.output_sampling_rate * self._config.input_hop_length
        denominator = self._config.input_sampling_rate * self._config.output_hop_length
        latent_length = op.Div(
            op.Mul(frames, op.Constant(value_int=numerator)),
            op.Constant(value_int=denominator),
        )
        latent_length = op.Max(latent_length, op.Constant(value_int=1))
        hidden_states = op.Resize(
            hidden_states,
            None,
            None,
            op.Concat(op.Shape(hidden_states, start=0, end=2), latent_length, axis=0),
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )
        return op.Transpose(hidden_states, perm=[0, 2, 1])


class _FourierEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter([embedding_dim // 2, 1])

    def forward(self, op: OpBuilder, timestep: ir.Value):
        angles = op.Mul(
            op.MatMul(op.Unsqueeze(timestep, [-1]), op.Transpose(self.weight)),
            2.0 * math.pi,
        )
        return op.Concat(op.Cos(angles), op.Sin(angles), axis=-1)


def _partial_rope(
    op: OpBuilder,
    hidden_states: ir.Value,
    cos: ir.Value,
    sin: ir.Value,
    *,
    heads: int,
    head_dim: int,
    rotary_dim: int,
):
    batch_seq = op.Shape(hidden_states, start=0, end=2)
    states = op.Reshape(
        hidden_states,
        op.Concat(batch_seq, op.Constant(value_ints=[heads, head_dim]), axis=0),
    )
    rotated, passthrough = op.Split(
        states,
        op.Constant(value_ints=[rotary_dim, head_dim - rotary_dim]),
        axis=3,
        _outputs=2,
    )
    first, second = op.Split(rotated, num_outputs=2, axis=3, _outputs=2)
    rotate_half = op.Concat(op.Neg(second), first, axis=3)
    cos = op.CastLike(op.Unsqueeze(cos, op.Constant(value_ints=[0, 2])), rotated)
    sin = op.CastLike(op.Unsqueeze(sin, op.Constant(value_ints=[0, 2])), rotated)
    states = op.Concat(
        op.Add(op.Mul(rotated, cos), op.Mul(rotate_half, sin)), passthrough, axis=3
    )
    return op.Reshape(
        states, op.Concat(batch_seq, op.Constant(value_ints=[heads * head_dim]), axis=0)
    )


class _FlowAttention(nn.Module):
    def __init__(self, config: MiniMaxMusic3TransformerConfig):
        super().__init__()
        dim = config.num_attention_heads * config.attention_head_dim
        self.to_q = Linear(dim, dim, bias=False)
        self.to_k = Linear(dim, dim, bias=False)
        self.to_v = Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([Linear(dim, dim, bias=False)])
        self._heads = config.num_attention_heads
        self._head_dim = config.attention_head_dim
        self._rotary_dim = config.rotary_dim

    def forward(self, op: OpBuilder, hidden_states: ir.Value, cos: ir.Value, sin: ir.Value):
        query = _partial_rope(
            op,
            self.to_q(op, hidden_states),
            cos,
            sin,
            heads=self._heads,
            head_dim=self._head_dim,
            rotary_dim=self._rotary_dim,
        )
        key = _partial_rope(
            op,
            self.to_k(op, hidden_states),
            cos,
            sin,
            heads=self._heads,
            head_dim=self._head_dim,
            rotary_dim=self._rotary_dim,
        )
        value = self.to_v(op, hidden_states)
        attended = op.Attention(
            query,
            key,
            value,
            q_num_heads=self._heads,
            kv_num_heads=self._heads,
        )
        return self.to_out[0](op, attended)


class _TransformerBlock(nn.Module):
    def __init__(self, config: MiniMaxMusic3TransformerConfig):
        super().__init__()
        dim = config.num_attention_heads * config.attention_head_dim
        self.norm1 = LayerNorm(dim, eps=1e-5)
        self.attn = _FlowAttention(config)
        self.norm2 = LayerNorm(dim, eps=1e-5)
        self.ff_in = Linear(dim, config.ff_inner_dim * 2)
        self.ff_out = Linear(config.ff_inner_dim, dim)

    def forward(self, op: OpBuilder, hidden_states: ir.Value, cos: ir.Value, sin: ir.Value):
        hidden_states = op.Add(
            hidden_states, self.attn(op, self.norm1(op, hidden_states), cos, sin)
        )
        gate_states, gate = op.Split(
            self.ff_in(op, self.norm2(op, hidden_states)),
            num_outputs=2,
            axis=-1,
            _outputs=2,
        )
        gate = op.Mul(gate, op.Sigmoid(gate))
        return op.Add(hidden_states, self.ff_out(op, op.Mul(gate_states, gate)))


class MiniMaxMusic3Transformer1DModel(nn.Module):
    """Bidirectional 1D flow transformer used by MiniMax Music 3."""

    default_task = "minimax-music3-denoising"
    category = "Diffusion"

    def __init__(self, config: MiniMaxMusic3TransformerConfig):
        super().__init__()
        inner_dim = config.num_attention_heads * config.attention_head_dim
        concat_channels = 2 * config.in_channels + config.condition_dim
        self.time_proj = _FourierEmbedding(config.fourier_embedding_dim)
        self.time_embed = TimestepEmbedding(config.fourier_embedding_dim, inner_dim)
        self.preprocess_conv = _Conv1d(concat_channels, concat_channels, 1, bias=False)
        self.proj_in = Linear(concat_channels, inner_dim, bias=False)
        self.transformer_blocks = nn.ModuleList(
            [_TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.proj_out = Linear(inner_dim, config.in_channels, bias=False)
        self.postprocess_conv = _Conv1d(config.in_channels, config.in_channels, 1, bias=False)
        self._config = config

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        timestep: ir.Value,
        encoder_hidden_states: ir.Value,
    ):
        condition = op.Transpose(encoder_hidden_states, perm=[0, 2, 1])
        hidden_states = op.Concat(hidden_states, op.Mul(hidden_states, 0.0), condition, axis=1)
        hidden_states = op.Add(hidden_states, self.preprocess_conv(op, hidden_states))
        hidden_states = self.proj_in(op, op.Transpose(hidden_states, perm=[0, 2, 1]))
        temb = op.Unsqueeze(self.time_embed(op, self.time_proj(op, timestep)), [1])
        hidden_states = op.Concat(temb, hidden_states, axis=1)

        seq_len = op.Squeeze(op.Shape(hidden_states, start=1, end=2))
        steps = op.Cast(
            op.Range(
                op.Constant(value_int=0),
                seq_len,
                op.Constant(value_int=1),
            ),
            to=ir.DataType.FLOAT,
        )
        inv_idx = op.Cast(
            op.Range(
                op.Constant(value_int=0),
                op.Constant(value_int=self._config.rotary_dim),
                op.Constant(value_int=2),
            ),
            to=ir.DataType.FLOAT,
        )
        inv_freq = op.Reciprocal(
            op.Pow(10000.0, op.Div(inv_idx, float(self._config.rotary_dim)))
        )
        freqs = op.Mul(
            op.Unsqueeze(steps, op.Constant(value_ints=[1])),
            op.Unsqueeze(inv_freq, op.Constant(value_ints=[0])),
        )
        freqs = op.Concat(freqs, freqs, axis=-1)
        cos, sin = op.Cos(freqs), op.Sin(freqs)
        for block in self.transformer_blocks:
            hidden_states = block(op, hidden_states, cos, sin)

        hidden_states = op.Slice(
            hidden_states,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[9223372036854775807]),
            op.Constant(value_ints=[1]),
        )
        hidden_states = op.Transpose(self.proj_out(op, hidden_states), perm=[0, 2, 1])
        return op.Add(hidden_states, self.postprocess_conv(op, hidden_states))


class _Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter([1, channels, 1])

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        # Checkpoint alpha values can be smaller than fp16's finite reciprocal
        # range. Evaluate Snake in fp32, then return to the surrounding model dtype.
        hidden_states_f32 = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        alpha_f32 = op.Cast(self.alpha, to=ir.DataType.FLOAT)
        angle = op.Mul(alpha_f32, hidden_states_f32)
        output = op.Add(
            hidden_states_f32,
            op.Mul(
                op.Reciprocal(op.Add(alpha_f32, 1e-9)),
                op.Pow(op.Sin(angle), 2.0),
            ),
        )
        return op.Cast(output, to=hidden_states.dtype)


class _VocoderResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.snake1 = _Snake1d(dim)
        self.conv1 = _Conv1d(dim, dim, 7, padding=3 * dilation, dilation=dilation)
        self.snake2 = _Snake1d(dim)
        self.conv2 = _Conv1d(dim, dim, 1)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        residual = self.conv2(
            op, self.snake2(op, self.conv1(op, self.snake1(op, hidden_states)))
        )
        return op.Add(hidden_states, residual)


class _VocoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.snake1 = _Snake1d(input_dim)
        self.conv_t1 = _ConvTranspose1d(input_dim, output_dim, stride)
        self.res_unit1 = _VocoderResidualUnit(output_dim, 1)
        self.res_unit2 = _VocoderResidualUnit(output_dim, 3)
        self.res_unit3 = _VocoderResidualUnit(output_dim, 9)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        hidden_states = self.conv_t1(op, self.snake1(op, hidden_states))
        hidden_states = self.res_unit1(op, hidden_states)
        hidden_states = self.res_unit2(op, hidden_states)
        return self.res_unit3(op, hidden_states)


class MiniMaxMusic3Vocoder(nn.Module):
    """Stereo DAC-style waveform decoder used by MiniMax Music 3."""

    default_task = "minimax-music3-vocoder"
    category = "Audio"

    def __init__(self, config: MiniMaxMusic3VocoderConfig):
        super().__init__()
        self.dec_in_proj = _Conv1d(config.latent_channels // 2, config.decoder_input_dim, 1)
        self.conv_in = _Conv1d(
            config.decoder_input_dim, config.decoder_hidden_dim, 7, padding=3
        )
        blocks = []
        output_dim = config.decoder_hidden_dim
        for index, stride in enumerate(config.upsampling_ratios):
            input_dim = config.decoder_hidden_dim // (2**index)
            output_dim = config.decoder_hidden_dim // (2 ** (index + 1))
            blocks.append(_VocoderBlock(input_dim, output_dim, stride))
        self.blocks = nn.ModuleList(blocks)
        self.snake_out = _Snake1d(output_dim)
        self.conv_out = _Conv1d(output_dim, 1, 7, padding=3)
        self._latent_channels = config.latent_channels

    def forward(self, op: OpBuilder, latents: ir.Value):
        batch = op.Shape(latents, start=0, end=1)
        length = op.Shape(latents, start=2, end=3)
        folded_shape = op.Concat(
            op.Mul(batch, op.Constant(value_int=2)),
            op.Constant(value_ints=[self._latent_channels // 2]),
            length,
            axis=0,
        )
        hidden_states = op.Reshape(latents, folded_shape)
        hidden_states = self.conv_in(op, self.dec_in_proj(op, hidden_states))
        for block in self.blocks:
            hidden_states = block(op, hidden_states)
        waveform = op.Tanh(self.conv_out(op, self.snake_out(op, hidden_states)))
        output_shape = op.Concat(batch, op.Constant(value_ints=[2, -1]), axis=0)
        return op.Reshape(waveform, output_shape)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Fold PyTorch weight-normalization tensors into ordinary convolution weights."""
        result = dict(state_dict)
        for name in list(state_dict):
            if not name.endswith(".weight_v"):
                continue
            base = name[: -len("_v")]
            g_name = f"{base}_g"
            value = state_dict[name]
            scale = state_dict[g_name] / torch.linalg.vector_norm(
                value, dim=tuple(range(1, value.ndim)), keepdim=True
            )
            result[base] = value * scale
            del result[name]
            del result[g_name]
        return result


class MiniMaxMusic3LanguageModel(Qwen3CausalLMModel):
    """Qwen3 global language model exposing normalized hidden states for Music 3."""

    default_task = "minimax-music3-language"
    category = "Audio"

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        result = self.model(
            op,
            input_ids=inputs_embeds,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        if len(result) == 3:
            hidden_states, present_key_values, _ = result
        else:
            hidden_states, present_key_values = result
        return self.lm_head(op, hidden_states), hidden_states, present_key_values
