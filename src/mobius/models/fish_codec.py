# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Declarative Fish Speech DAC ``from_indices`` waveform decoder.

Replicates the decode path of pinned Fish Speech ``modded_dac.DAC`` and
``rvq.DownsampleResidualVectorQuantize``. Input is INT64 [B,10,T] code IDs
(without the main text token); output is FP32 [B,1,T*2048] at 44100 Hz.
This is a full-sequence decoder, not a streaming convolution/KV-cache API.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs.fish_codec import FishCodecConfig
from mobius.components import (
    CausalConv1d,
    CausalTransConv1d,
    Conv1d,
    ConvNeXtBlock,
    Embedding,
    Linear,
    RMSNorm,
)


class _Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter([1, channels, 1])

    def forward(self, op: OpBuilder, x: ir.Value):
        # DAC Snake has one direct alpha parameter, unlike log-space SnakeBeta.
        sine = op.Sin(op.Mul(x, self.alpha))
        return op.Add(x, op.Div(op.Mul(sine, sine), op.Add(self.alpha, 1e-9)))


class _ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            _Snake1d(channels),
            CausalConv1d(channels, channels, 7, dilation=dilation),
            _Snake1d(channels),
            CausalConv1d(channels, channels, 1),
        )

    def forward(self, op: OpBuilder, x: ir.Value):
        return op.Add(x, self.block(op, x))


class _DecoderBlock(nn.Module):
    def __init__(self, channels: int, rate: int):
        super().__init__()
        self.block = nn.Sequential(
            _Snake1d(channels),
            CausalTransConv1d(channels, channels // 2, 2 * rate, rate),
            *[_ResidualUnit(channels // 2, dilation) for dilation in (1, 3, 9)],
        )

    def forward(self, op: OpBuilder, x: ir.Value):
        return self.block(op, x)


class _WaveformDecoder(nn.Module):
    def __init__(self, config: FishCodecConfig):
        super().__init__()
        channels = config.decoder_dim
        layers: list[nn.Module] = [CausalConv1d(config.latent_dim, channels, 7)]
        for rate in config.decoder_rates:
            layers.append(_DecoderBlock(channels, rate))
            channels //= 2
        layers.extend([_Snake1d(channels), CausalConv1d(channels, 1, 7)])
        self.model = nn.Sequential(*layers)

    def forward(self, op: OpBuilder, x: ir.Value):
        return op.Tanh(self.model(op, x))


class _Codebook(nn.Module):
    def __init__(self, config: FishCodecConfig, size: int):
        super().__init__()
        self.codebook = Embedding(size, config.codebook_dim)
        # DAC out_proj is a weight-normalized 1x1 convolution, not a linear.
        self.out_proj = Conv1d(config.codebook_dim, config.latent_dim, 1)

    def forward(self, op: OpBuilder, codes: ir.Value):
        embedded = op.Transpose(self.codebook(op, codes), perm=[0, 2, 1])
        return self.out_proj(op, embedded)


class _RVQ(nn.Module):
    def __init__(self, config: FishCodecConfig, count: int, size: int):
        super().__init__()
        self.quantizers = nn.ModuleList([_Codebook(config, size) for _ in range(count)])
        self.size = size

    def forward(self, op: OpBuilder, codes: ir.Value):
        # The token tower has 4096 outputs for every book; upstream clamps only
        # the upper bound before embedding lookup (residual books have 1024 rows).
        codes = op.Min(codes, op.Constant(value_int=self.size - 1))
        values = [
            book(op, op.Gather(codes, i, axis=1)) for i, book in enumerate(self.quantizers)
        ]
        result = values[0]
        for value in values[1:]:
            result = op.Add(result, value)
        return result


class _Scale(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter([channels])

    def forward(self, op: OpBuilder, x: ir.Value):
        return op.Mul(x, self.gamma)


class _Attention(nn.Module):
    def __init__(self, config: FishCodecConfig):
        super().__init__()
        self.config = config
        self.wqkv = Linear(config.latent_dim, 3 * config.latent_dim, bias=False)
        self.wo = Linear(config.latent_dim, config.latent_dim, bias=False)

    def forward(self, op: OpBuilder, x: ir.Value, mask: ir.Value, cos, sin):
        cfg = self.config
        q, k, v = op.Split(self.wqkv(op, x), [cfg.latent_dim] * 3, axis=-1, _outputs=3)

        def rotate(value):
            pairs = op.Reshape(value, [0, 0, cfg.transformer_heads, cfg.head_dim // 2, 2])
            even, odd = op.Gather(pairs, 0, axis=-1), op.Gather(pairs, 1, axis=-1)
            real = op.Sub(op.Mul(even, cos), op.Mul(odd, sin))
            imag = op.Add(op.Mul(odd, cos), op.Mul(even, sin))
            return op.Reshape(
                op.Concat(op.Unsqueeze(real, [-1]), op.Unsqueeze(imag, [-1]), axis=-1),
                [0, 0, cfg.latent_dim],
            )

        q, k = rotate(q), rotate(k)
        attended = op.Attention(
            q,
            k,
            v,
            mask,
            q_num_heads=cfg.transformer_heads,
            kv_num_heads=cfg.transformer_heads,
        )
        return self.wo(op, attended)


class _FeedForward(nn.Module):
    def __init__(self, config: FishCodecConfig):
        super().__init__()
        self.w1 = Linear(config.latent_dim, config.transformer_intermediate_size, bias=False)
        self.w3 = Linear(config.latent_dim, config.transformer_intermediate_size, bias=False)
        self.w2 = Linear(config.transformer_intermediate_size, config.latent_dim, bias=False)

    def forward(self, op: OpBuilder, x: ir.Value):
        gate = self.w1(op, x)
        return self.w2(op, op.Mul(op.Mul(gate, op.Sigmoid(gate)), self.w3(op, x)))


class _TransformerBlock(nn.Module):
    def __init__(self, config: FishCodecConfig):
        super().__init__()
        self.attention = _Attention(config)
        self.feed_forward = _FeedForward(config)
        self.attention_norm = RMSNorm(config.latent_dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.latent_dim, eps=config.norm_eps)
        self.attention_layer_scale = _Scale(config.latent_dim)
        self.ffn_layer_scale = _Scale(config.latent_dim)

    def forward(self, op: OpBuilder, x: ir.Value, mask: ir.Value, cos, sin):
        attended = self.attention(op, self.attention_norm(op, x), mask, cos, sin)
        x = op.Add(x, self.attention_layer_scale(op, attended))
        return op.Add(x, self.ffn_layer_scale(op, self.feed_forward(op, self.ffn_norm(op, x))))


class _PostTransformer(nn.Module):
    def __init__(self, config: FishCodecConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [_TransformerBlock(config) for _ in range(config.transformer_layers)]
        )
        self.norm = RMSNorm(config.latent_dim, eps=config.norm_eps)
        # Upstream precompute_freqs_cis rounds BOTH sin and cos to BF16 even
        # when all weights and activations are FP32. Keep those exact values.
        frequencies = 1.0 / (
            config.rope_base ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim)
        )
        angles = torch.outer(torch.arange(config.max_frames).float(), frequencies)
        polar = torch.polar(torch.ones_like(angles), angles)
        self._rope = (
            torch.stack((polar.real, polar.imag), dim=-1).to(torch.bfloat16).float().numpy()
        )

    def forward(self, op: OpBuilder, x: ir.Value):
        x = op.Transpose(x, perm=[0, 2, 1])  # (B,T,D)
        length = op.Squeeze(op.Shape(x, start=1, end=2), [0])
        positions = op.Range(op.Constant(value_int=0), length, op.Constant(value_int=1))
        rows, columns = op.Unsqueeze(positions, [1]), op.Unsqueeze(positions, [0])
        mask = op.And(
            op.LessOrEqual(columns, rows),
            op.Greater(columns, op.Sub(rows, self.config.transformer_window)),
        )
        mask = op.Unsqueeze(mask, [0, 1])
        rope = op.Gather(op.Constant(value=ir.tensor(self._rope)), positions)
        cos = op.Unsqueeze(op.Gather(rope, 0, axis=-1), [0, 2])
        sin = op.Unsqueeze(op.Gather(rope, 1, axis=-1), [0, 2])
        for layer in self.layers:
            x = layer(op, x, mask, cos, sin)
        return op.Transpose(self.norm(op, x), perm=[0, 2, 1])


class _Quantizer(nn.Module):
    def __init__(self, config: FishCodecConfig):
        super().__init__()
        self.semantic_quantizer = _RVQ(config, 1, config.semantic_codebook_size)
        self.quantizer = _RVQ(config, config.residual_codebooks, config.residual_codebook_size)
        self.post_module = _PostTransformer(config)
        self.upsample = nn.Sequential(
            *[
                nn.Sequential(
                    CausalTransConv1d(config.latent_dim, config.latent_dim, rate, rate),
                    ConvNeXtBlock(config.latent_dim),
                )
                for rate in reversed(config.quantizer_rates)
            ]
        )

    def forward(self, op: OpBuilder, codes: ir.Value):
        semantic = self.semantic_quantizer(op, op.Slice(codes, [0], [1], [1]))
        residual = self.quantizer(op, op.Slice(codes, [1], [np.iinfo(np.int64).max], [1]))
        return self.upsample(op, self.post_module(op, op.Add(semantic, residual)))


class FishCodecDecoder(nn.Module):
    """Fish Speech DAC decode-only model; use ``build_fish_codec_decoder`` to export."""

    category = "Audio"
    default_task = "fish-codec-decode"

    def __init__(self, config: FishCodecConfig):
        super().__init__()
        self.config = config
        self.quantizer = _Quantizer(config)
        self.decoder = _WaveformDecoder(config)

    def forward(self, op: OpBuilder, codes: ir.Value):
        return self.decoder(op, self.quantizer(op, codes))

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]):
        """Select decode weights and materialize both upstream weight-norm formats.

        At most the decode subset plus one convolution normalization temporary
        is materialized; the checkpoint can remain memory-mapped on CPU.
        """
        expected = dict(self.named_parameters())
        result: dict[str, torch.Tensor] = {}
        for name, parameter in expected.items():
            source = name
            if source in state_dict:
                tensor = state_dict[source]
            elif source.endswith(".weight"):
                prefix = source.removesuffix(".weight")
                modern = prefix + ".parametrizations.weight.original"
                if modern + "0" in state_dict and modern + "1" in state_dict:
                    g, v = state_dict[modern + "0"], state_dict[modern + "1"]
                elif prefix + ".weight_g" in state_dict and prefix + ".weight_v" in state_dict:
                    g, v = state_dict[prefix + ".weight_g"], state_dict[prefix + ".weight_v"]
                else:
                    raise ValueError(f"Missing codec weight {source!r}")
                tensor = torch._weight_norm(v.float(), g.float(), 0)
            else:
                raise ValueError(f"Missing codec weight {source!r}")
            if tuple(tensor.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"Codec weight {source!r}: expected {tuple(parameter.shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
            result[name] = tensor.float()
        return result
