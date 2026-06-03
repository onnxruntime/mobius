"""PyTorch re-implementation of the mobius `_NeMoConformerEncoder` math.

This script mirrors the ONNX graph produced by ``src/mobius/models/lfm2_audio.py``
*exactly*, but in pure PyTorch so we can:

  1. Run intermediate sub-block outputs and bisect divergence with HF.
  2. Tweak/fix individual pieces (e.g. the rel-pos shift) without re-exporting
     ONNX every iteration.

The semantic mapping is one-to-one with the mobius modules:
    Subsampling   → MobConvSubsampling
    FeedForward   → MobFF
    ConvBlock     → MobConv
    RelPosAtten   → MobRelPosAttn
    Layer         → MobLayer
    Encoder       → MobEncoder

It deliberately copies HF weights via parameter-name remapping.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MobConvSubsampling(nn.Module):
    def __init__(self, n_mels: int, c: int, d_model: int):
        super().__init__()
        self.conv0 = nn.Conv2d(1, c, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c)
        self.conv3 = nn.Conv2d(c, c, 1)
        self.conv5 = nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c)
        self.conv6 = nn.Conv2d(c, c, 1)
        freq = n_mels
        for _ in range(3):
            freq = (freq + 2 - 3) // 2 + 1
        self.out = nn.Linear(c * freq, d_model)
        self.c = c
        self.freq = freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, n_mels]
        x = x.unsqueeze(1)  # [B, 1, T, n_mels]
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv3(self.conv2(x)))
        x = F.relu(self.conv6(self.conv5(x)))
        # x: [B, C, T', F']
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T', C, F']
        x = x.reshape(x.shape[0], x.shape[1], self.c * self.freq)
        return self.out(x)


class MobFF(nn.Module):
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_inner)
        self.linear2 = nn.Linear(d_inner, d_model)

    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x)))


class MobConvBlock(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(c, 2 * c, 1)
        self.depthwise_conv = nn.Conv1d(c, c, k, padding=(k - 1) // 2, groups=c)
        self.batch_norm = nn.BatchNorm1d(c)
        self.pointwise_conv2 = nn.Conv1d(c, c, 1)

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.pointwise_conv1(x)  # [B, 2C, T]
        a, b = x.chunk(2, dim=1)
        x = a * torch.sigmoid(b)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class MobRelPosAttn(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h = num_heads
        self.d = d_model // num_heads
        self.d_model = d_model
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(num_heads, self.d))
        self.pos_bias_v = nn.Parameter(torch.zeros(num_heads, self.d))

    def _build_pe(self, T, dtype, device):
        # positions: [T-1, T-2, ..., -(T-1)] length 2T-1
        positions = torch.arange(T - 1, -T, -1, dtype=torch.float32, device=device).unsqueeze(1)
        d_model = self.d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
            * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(positions.size(0), d_model, dtype=torch.float32, device=device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0).to(dtype)

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, 2T-1]
        B, H, T, P = x.shape  # P = 2T-1
        zero_pad = x.new_zeros(B, H, T, 1)
        x_pad = torch.cat([zero_pad, x], dim=-1)  # [B, H, T, 2T]
        x_pad = x_pad.view(B, H, 2 * T, T)
        x_pad = x_pad[:, :, 1:]  # [B, H, 2T-1, T]
        x_pad = x_pad.view(B, H, T, 2 * T - 1)
        return x_pad[:, :, :, :T]

    def forward(self, x):
        B, T, _ = x.shape
        q = self.linear_q(x).view(B, T, self.h, self.d)
        k = self.linear_k(x).view(B, T, self.h, self.d)
        v = self.linear_v(x).view(B, T, self.h, self.d)
        q_u = (q + self.pos_bias_u).transpose(1, 2)  # [B, H, T, D]
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        k_t = k.transpose(1, 2)  # [B, H, T, D]
        v_t = v.transpose(1, 2)
        ac = torch.matmul(q_u, k_t.transpose(-2, -1))  # [B, H, T, T]

        pe = self._build_pe(T, x.dtype, x.device)  # [1, 2T-1, d_model]
        p = self.linear_pos(pe).view(1, -1, self.h, self.d).permute(0, 2, 3, 1)  # [1, H, D, 2T-1]
        bd = torch.matmul(q_v, p)  # [B, H, T, 2T-1]
        bd = self._rel_shift(bd)  # [B, H, T, T]

        scores = (ac + bd) * (self.d ** -0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v_t)  # [B, H, T, D]
        context = context.transpose(1, 2).contiguous().view(B, T, self.h * self.d)
        return self.linear_out(context)


class MobLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_inner, k):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(d_model, eps=1e-5)
        self.feed_forward1 = MobFF(d_model, d_inner)
        self.norm_self_att = nn.LayerNorm(d_model, eps=1e-5)
        self.self_attn = MobRelPosAttn(d_model, num_heads)
        self.norm_conv = nn.LayerNorm(d_model, eps=1e-5)
        self.conv = MobConvBlock(d_model, k)
        self.norm_feed_forward2 = nn.LayerNorm(d_model, eps=1e-5)
        self.feed_forward2 = MobFF(d_model, d_inner)
        self.norm_out = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x, dump=None):
        if dump is not None:
            dump['in'] = x.detach().clone()
        ff1 = self.feed_forward1(self.norm_feed_forward1(x))
        x = x + 0.5 * ff1
        if dump is not None:
            dump['after_ff1'] = x.detach().clone()
        attn = self.self_attn(self.norm_self_att(x))
        x = x + attn
        if dump is not None:
            dump['after_attn'] = x.detach().clone()
        c = self.conv(self.norm_conv(x))
        x = x + c
        if dump is not None:
            dump['after_conv'] = x.detach().clone()
        ff2 = self.feed_forward2(self.norm_feed_forward2(x))
        x = x + 0.5 * ff2
        if dump is not None:
            dump['after_ff2'] = x.detach().clone()
        x = self.norm_out(x)
        if dump is not None:
            dump['after_norm_out'] = x.detach().clone()
        return x


class MobEncoder(nn.Module):
    def __init__(self, n_mels, d_model, num_heads, d_inner, num_layers, k, c):
        super().__init__()
        self.pre_encode = MobConvSubsampling(n_mels, c, d_model)
        self.layers = nn.ModuleList(
            [MobLayer(d_model, num_heads, d_inner, k) for _ in range(num_layers)]
        )

    def forward(self, x, layer_dumps=None):
        x = self.pre_encode(x)
        for i, layer in enumerate(self.layers):
            d = None
            if layer_dumps is not None:
                d = {}
                layer_dumps.append(d)
            x = layer(x, dump=d)
        return x


def load_from_hf(mob: MobEncoder, hf_conformer) -> None:
    """Copy weights from HF conformer to MobEncoder."""
    sd = hf_conformer.state_dict()

    def cp(target, name):
        target.data.copy_(sd[name])

    # Subsampling: pre_encode.conv.{0,2,3,5,6} (Conv2d), pre_encode.out (Linear)
    pe = mob.pre_encode
    cp(pe.conv0.weight, 'pre_encode.conv.0.weight'); cp(pe.conv0.bias, 'pre_encode.conv.0.bias')
    cp(pe.conv2.weight, 'pre_encode.conv.2.weight'); cp(pe.conv2.bias, 'pre_encode.conv.2.bias')
    cp(pe.conv3.weight, 'pre_encode.conv.3.weight'); cp(pe.conv3.bias, 'pre_encode.conv.3.bias')
    cp(pe.conv5.weight, 'pre_encode.conv.5.weight'); cp(pe.conv5.bias, 'pre_encode.conv.5.bias')
    cp(pe.conv6.weight, 'pre_encode.conv.6.weight'); cp(pe.conv6.bias, 'pre_encode.conv.6.bias')
    cp(pe.out.weight, 'pre_encode.out.weight'); cp(pe.out.bias, 'pre_encode.out.bias')

    for i, layer in enumerate(mob.layers):
        p = f'layers.{i}'
        for nm in ('norm_feed_forward1', 'norm_self_att', 'norm_conv', 'norm_feed_forward2', 'norm_out'):
            cp(getattr(layer, nm).weight, f'{p}.{nm}.weight')
            cp(getattr(layer, nm).bias, f'{p}.{nm}.bias')
        for ff in ('feed_forward1', 'feed_forward2'):
            cp(getattr(layer, ff).linear1.weight, f'{p}.{ff}.linear1.weight')
            cp(getattr(layer, ff).linear1.bias, f'{p}.{ff}.linear1.bias')
            cp(getattr(layer, ff).linear2.weight, f'{p}.{ff}.linear2.weight')
            cp(getattr(layer, ff).linear2.bias, f'{p}.{ff}.linear2.bias')
        sa = layer.self_attn
        cp(sa.linear_q.weight, f'{p}.self_attn.linear_q.weight'); cp(sa.linear_q.bias, f'{p}.self_attn.linear_q.bias')
        cp(sa.linear_k.weight, f'{p}.self_attn.linear_k.weight'); cp(sa.linear_k.bias, f'{p}.self_attn.linear_k.bias')
        cp(sa.linear_v.weight, f'{p}.self_attn.linear_v.weight'); cp(sa.linear_v.bias, f'{p}.self_attn.linear_v.bias')
        cp(sa.linear_out.weight, f'{p}.self_attn.linear_out.weight'); cp(sa.linear_out.bias, f'{p}.self_attn.linear_out.bias')
        cp(sa.linear_pos.weight, f'{p}.self_attn.linear_pos.weight')
        cp(sa.pos_bias_u, f'{p}.self_attn.pos_bias_u')
        cp(sa.pos_bias_v, f'{p}.self_attn.pos_bias_v')
        cb = layer.conv
        cp(cb.pointwise_conv1.weight, f'{p}.conv.pointwise_conv1.weight'); cp(cb.pointwise_conv1.bias, f'{p}.conv.pointwise_conv1.bias')
        cp(cb.depthwise_conv.weight, f'{p}.conv.depthwise_conv.weight'); cp(cb.depthwise_conv.bias, f'{p}.conv.depthwise_conv.bias')
        cp(cb.batch_norm.weight, f'{p}.conv.batch_norm.weight'); cp(cb.batch_norm.bias, f'{p}.conv.batch_norm.bias')
        cp(cb.batch_norm.running_mean, f'{p}.conv.batch_norm.running_mean')
        cp(cb.batch_norm.running_var, f'{p}.conv.batch_norm.running_var')
        cp(cb.pointwise_conv2.weight, f'{p}.conv.pointwise_conv2.weight'); cp(cb.pointwise_conv2.bias, f'{p}.conv.pointwise_conv2.bias')
