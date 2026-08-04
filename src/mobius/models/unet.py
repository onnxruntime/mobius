# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""UNet2DConditionModel for Stable Diffusion denoisers.

Architecture:
1. Time embedding: sinusoidal → MLP
2. Conv_in: projects noisy latent
3. Down blocks: ResNet + cross-attention + downsample
4. Mid block: ResNet + cross-attention + ResNet
5. Up blocks: ResNet + cross-attention + upsample
6. Conv_out: projects to noise prediction

Supports: SD 1.x, SD 2.x, SDXL UNet
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from onnxscript import OpBuilder, nn

from mobius._diffusers_configs import UNet2DConfig
from mobius.components import Conv2d as _Conv2d
from mobius.components import GroupNorm as _GroupNorm
from mobius.components import Linear as _Linear
from mobius.components import LoRALinear as _LoRALinear
from mobius.components import SiLU as _SiLU

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Time Embedding
# ---------------------------------------------------------------------------


class _TimestepEmbedding(nn.Module):
    """Projects timestep embedding to model hidden dim: Linear → SiLU → Linear."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = _Linear(in_channels, time_embed_dim)
        self.linear_2 = _Linear(time_embed_dim, time_embed_dim)
        self._silu = _SiLU()

    def forward(self, op: OpBuilder, sample: ir.Value):
        sample = self.linear_1(op, sample)
        sample = self._silu(op, sample)
        sample = self.linear_2(op, sample)
        return sample


# ---------------------------------------------------------------------------
# ResNet with time embedding
# ---------------------------------------------------------------------------


class _ResNetBlock2DWithTime(nn.Module):
    """ResNet block with time embedding injection.

    GroupNorm → SiLU → Conv → time_proj → GroupNorm → SiLU → Conv + skip.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self.norm1 = _GroupNorm(norm_num_groups, in_channels)
        self.conv1 = _Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb_proj = _Linear(time_embed_dim, out_channels)
        self.norm2 = _GroupNorm(norm_num_groups, out_channels)
        self.conv2 = _Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self._silu = _SiLU()

        if in_channels != out_channels:
            self.conv_shortcut = _Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.conv_shortcut = None

    def forward(self, op: OpBuilder, hidden_states: ir.Value, temb: ir.Value):
        residual = hidden_states

        hidden_states = self.norm1(op, hidden_states)
        hidden_states = self._silu(op, hidden_states)
        hidden_states = self.conv1(op, hidden_states)

        # Add time embedding: [B, C] → [B, C, 1, 1]
        temb_proj = self._silu(op, temb)
        temb_proj = self.time_emb_proj(op, temb_proj)
        temb_proj = op.Unsqueeze(temb_proj, [-1, -2])
        hidden_states = op.Add(hidden_states, temb_proj)

        hidden_states = self.norm2(op, hidden_states)
        hidden_states = self._silu(op, hidden_states)
        hidden_states = self.conv2(op, hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(op, residual)

        return op.Add(hidden_states, residual)


# ---------------------------------------------------------------------------
# Cross-Attention for conditioning
# ---------------------------------------------------------------------------


class _CrossAttentionBlock(nn.Module):
    """Cross-attention block: self-attention + cross-attention + FFN.

    Processes latent features conditioned on text encoder hidden states.
    """

    def __init__(
        self,
        channels: int,
        cross_attention_dim: int,
        num_heads: int,
        norm_num_groups: int = 32,
        linear_class=_Linear,
        use_linear_projection: bool = False,
    ):
        super().__init__()
        self._use_linear_projection = use_linear_projection
        self.norm = _GroupNorm(norm_num_groups, channels)
        # Transformer2DModel input/output projection: a 1x1 Conv (SD 1.x) or a
        # Linear (SD 2.x / SDXL), around the flattened-spatial transformer block.
        if use_linear_projection:
            self.proj_in = _Linear(channels, channels)
            self.proj_out = _Linear(channels, channels)
        else:
            self.proj_in = _Conv2d(channels, channels, kernel_size=1)
            self.proj_out = _Conv2d(channels, channels, kernel_size=1)
        # Self-attention
        self.attn1 = _BasicAttention(channels, channels, num_heads, linear_class=linear_class)
        # Cross-attention (K, V from encoder_hidden_states)
        self.attn2 = _BasicAttention(
            channels, cross_attention_dim, num_heads, linear_class=linear_class
        )
        # FFN
        self.ff = _FeedForward(channels, channels * 4)
        self.norm1 = _LayerNorm1D(channels)
        self.norm2 = _LayerNorm1D(channels)
        self.norm3 = _LayerNorm1D(channels)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value | None = None,
    ):
        residual = hidden_states
        batch = op.Shape(hidden_states, start=0, end=1)
        channels = op.Shape(hidden_states, start=1, end=2)
        height = op.Shape(hidden_states, start=2, end=3)
        width = op.Shape(hidden_states, start=3, end=4)

        hidden_states = self.norm(op, hidden_states)

        # 1x1 Conv proj_in runs on [B, C, H, W] before the spatial flatten.
        if not self._use_linear_projection:
            hidden_states = self.proj_in(op, hidden_states)

        # Reshape [B, C, H, W] → [B, H*W, C]
        spatial = op.Mul(height, width)
        hidden_states = op.Reshape(hidden_states, op.Concat(batch, channels, spatial, axis=0))
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])

        # Linear proj_in runs on the flattened [B, H*W, C].
        if self._use_linear_projection:
            hidden_states = self.proj_in(op, hidden_states)

        # Self-attention
        norm_hs = self.norm1(op, hidden_states)
        hidden_states = op.Add(self.attn1(op, norm_hs, norm_hs), hidden_states)

        # Cross-attention
        norm_hs = self.norm2(op, hidden_states)
        context = encoder_hidden_states if encoder_hidden_states is not None else norm_hs
        hidden_states = op.Add(self.attn2(op, norm_hs, context), hidden_states)

        # FFN
        norm_hs = self.norm3(op, hidden_states)
        hidden_states = op.Add(self.ff(op, norm_hs), hidden_states)

        # Linear proj_out runs before the reshape back.
        if self._use_linear_projection:
            hidden_states = self.proj_out(op, hidden_states)

        # Reshape back [B, H*W, C] → [B, C, H, W]
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = op.Reshape(
            hidden_states, op.Concat(batch, channels, height, width, axis=0)
        )

        # 1x1 Conv proj_out runs on [B, C, H, W] after the reshape back.
        if not self._use_linear_projection:
            hidden_states = self.proj_out(op, hidden_states)

        return op.Add(hidden_states, residual)


class _BasicAttention(nn.Module):
    """Simple multi-head attention: Q from input, K/V from context.

    ``linear_class`` is the factory used for the q/k/v/out projections; pass a
    ``LoRALinear`` factory (see :mod:`mobius.components._lora`) to build
    LoRA-adapted, optionally runtime-gated projections. Defaults to plain
    ``Linear``.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int,
        linear_class=_Linear,
    ):
        super().__init__()
        # Stable Diffusion attention uses no bias on q/k/v; only the output
        # projection (to_out.0) is biased.
        self.to_q = linear_class(query_dim, query_dim, bias=False)
        self.to_k = linear_class(context_dim, query_dim, bias=False)
        self.to_v = linear_class(context_dim, query_dim, bias=False)
        self.to_out = nn.Sequential(linear_class(query_dim, query_dim, bias=True))
        self._num_heads = num_heads
        self._head_dim = query_dim // num_heads

    def forward(self, op: OpBuilder, hidden_states: ir.Value, context: ir.Value):
        q = self.to_q(op, hidden_states)
        k = self.to_k(op, context)
        v = self.to_v(op, context)

        attn_out = op.Attention(
            q,
            k,
            v,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            is_causal=0,
            scale=float(self._head_dim**-0.5),
        )
        return self.to_out(op, attn_out)


class _LayerNorm1D(nn.Module):
    """Layer normalization for sequence data [B, T, C]."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter((dim,))
        self.bias = nn.Parameter((dim,))
        self._eps = eps

    def forward(self, op: OpBuilder, x: ir.Value):
        return op.LayerNormalization(x, self.weight, self.bias, axis=-1, epsilon=self._eps)


class _FeedForward(nn.Module):
    """GEGLU feed-forward network for transformer blocks."""

    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        # GEGLU: projects to 2*inner_dim, splits into value and gate
        self.proj_in = _Linear(dim, inner_dim * 2)
        self.proj_out = _Linear(inner_dim, dim)

    def forward(self, op: OpBuilder, x: ir.Value):
        projected = self.proj_in(op, x)
        # Split into value and gate
        x1, gate = op.Split(projected, num_outputs=2, axis=-1, _outputs=2)
        # GELU gate
        gate = op.Gelu(gate)
        hidden_states = op.Mul(x1, gate)
        return self.proj_out(op, hidden_states)


# ---------------------------------------------------------------------------
# Down / Up blocks
# ---------------------------------------------------------------------------


class _DownBlock2D(nn.Module):
    """UNet down block: N (ResNet + optional attention) + downsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_layers: int = 2,
        norm_num_groups: int = 32,
        cross_attention_dim: int | None = None,
        attention_head_dim: int = 8,
        add_downsample: bool = True,
        linear_class=_Linear,
        use_linear_projection: bool = False,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList() if cross_attention_dim else None

        for i in range(num_layers):
            res_in = in_channels if i == 0 else out_channels
            self.resnets.append(
                _ResNetBlock2DWithTime(res_in, out_channels, time_embed_dim, norm_num_groups)
            )
            if cross_attention_dim:
                num_heads = attention_head_dim
                self.attentions.append(
                    _CrossAttentionBlock(
                        out_channels,
                        cross_attention_dim,
                        num_heads,
                        norm_num_groups,
                        linear_class=linear_class,
                        use_linear_projection=use_linear_projection,
                    )
                )

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList()
            self.downsamplers.append(_Downsample2D(out_channels))

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        temb: ir.Value,
        encoder_hidden_states: ir.Value | None = None,
    ):
        output_states = []
        for i, resnet in enumerate(self.resnets):
            hidden_states = resnet(op, hidden_states, temb)
            if self.attentions is not None:
                hidden_states = self.attentions[i](op, hidden_states, encoder_hidden_states)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(op, hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class _UpBlock2D(nn.Module):
    """UNet up block: N (ResNet + optional attention) + upsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channels: int,
        time_embed_dim: int,
        num_layers: int = 3,
        norm_num_groups: int = 32,
        cross_attention_dim: int | None = None,
        attention_head_dim: int = 8,
        add_upsample: bool = True,
        linear_class=_Linear,
        use_linear_projection: bool = False,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList() if cross_attention_dim else None

        for i in range(num_layers):
            # Skip-connection channels: the last resnet takes the up block's
            # input channels, the others take out_channels (diffusers convention).
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channels if i == 0 else out_channels
            self.resnets.append(
                _ResNetBlock2DWithTime(
                    resnet_in_channels + res_skip_channels,
                    out_channels,
                    time_embed_dim,
                    norm_num_groups,
                )
            )
            if cross_attention_dim:
                num_heads = attention_head_dim
                self.attentions.append(
                    _CrossAttentionBlock(
                        out_channels,
                        cross_attention_dim,
                        num_heads,
                        norm_num_groups,
                        linear_class=linear_class,
                        use_linear_projection=use_linear_projection,
                    )
                )

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList()
            self.upsamplers.append(_Upsample2D(out_channels))

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        temb: ir.Value,
        skip_connections: ir.Value,
        encoder_hidden_states: ir.Value | None = None,
    ):
        for i, resnet in enumerate(self.resnets):
            skip = skip_connections.pop()
            hidden_states = op.Concat(hidden_states, skip, axis=1)
            hidden_states = resnet(op, hidden_states, temb)
            if self.attentions is not None:
                hidden_states = self.attentions[i](op, hidden_states, encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(op, hidden_states)

        return hidden_states


class _Downsample2D(nn.Module):
    """Strided convolution downsampler."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = _Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, op: OpBuilder, x: ir.Value):
        return self.conv(op, x)


class _Upsample2D(nn.Module):
    """Nearest-neighbor upsampling + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = _Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        # Upsample 2x using nearest neighbor. Resize inputs are (X, roi, scales);
        # pass scales as the 3rd argument (an extra None would push it into the
        # int64 `sizes` slot and fail type checking).
        hidden_states = op.Resize(
            hidden_states,
            None,
            op.Constant(value_floats=[1.0, 1.0, 2.0, 2.0]),
            mode="nearest",
        )
        return self.conv(op, hidden_states)


# ---------------------------------------------------------------------------
# Mid block
# ---------------------------------------------------------------------------


class _UNetMidBlock2DCrossAttn(nn.Module):
    """UNet mid block: ResNet + cross-attention + ResNet."""

    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        cross_attention_dim: int,
        attention_head_dim: int = 8,
        norm_num_groups: int = 32,
        linear_class=_Linear,
        use_linear_projection: bool = False,
    ):
        super().__init__()
        num_heads = attention_head_dim
        self.resnets = nn.ModuleList()
        self.resnets.append(
            _ResNetBlock2DWithTime(channels, channels, time_embed_dim, norm_num_groups)
        )
        self.resnets.append(
            _ResNetBlock2DWithTime(channels, channels, time_embed_dim, norm_num_groups)
        )
        self.attentions = nn.ModuleList()
        self.attentions.append(
            _CrossAttentionBlock(
                channels,
                cross_attention_dim,
                num_heads,
                norm_num_groups,
                linear_class=linear_class,
                use_linear_projection=use_linear_projection,
            )
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        temb: ir.Value,
        encoder_hidden_states: ir.Value | None = None,
    ):
        hidden_states = self.resnets[0](op, hidden_states, temb)
        hidden_states = self.attentions[0](op, hidden_states, encoder_hidden_states)
        hidden_states = self.resnets[1](op, hidden_states, temb)
        return hidden_states


# ---------------------------------------------------------------------------
# Full UNet model
# ---------------------------------------------------------------------------


class UNet2DConditionModel(nn.Module):
    """UNet2D conditional denoiser for Stable Diffusion.

    Takes noisy latent + timestep + text encoder hidden states, outputs noise prediction.
    """

    default_task: str = "denoising"
    category: str = "Diffusion"

    def __init__(self, config: UNet2DConfig):
        super().__init__()
        self.config = config

        # Runtime LoRA: bake the declared adapters into every attention
        # projection and gate each at run time via a shared `_lora_gates`
        # mapping (populated in `forward` from the `lora_gate.{name}` inputs).
        lora_adapters = list(getattr(config, "lora_adapters", ()) or ())
        self._lora_adapter_names = [name for name, _, _ in lora_adapters]
        self._lora_gates: dict = {}
        if lora_adapters:
            gates = self._lora_gates

            def _lora_linear(in_features: int, out_features: int, bias: bool = True):
                return _LoRALinear(
                    in_features,
                    out_features,
                    bias=bias,
                    lora_adapters=lora_adapters,
                    gate_holder=gates,
                )

            linear_class = _lora_linear
        else:
            linear_class = _Linear

        block_out_channels = config.block_out_channels
        time_embed_dim = block_out_channels[0] * 4

        # Time embedding
        self.time_proj_dim = block_out_channels[0]
        self.time_embedding = _TimestepEmbedding(block_out_channels[0], time_embed_dim)

        # Input convolution
        self.conv_in = _Conv2d(
            config.in_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        # Down blocks
        self.down_blocks = nn.ModuleList()
        output_channel = block_out_channels[0]
        down_block_types = config.down_block_types or tuple(
            "CrossAttnDownBlock2D" for _ in block_out_channels
        )
        for i, ch in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = ch
            is_final = i == len(block_out_channels) - 1
            has_cross_attention = "CrossAttn" in down_block_types[i]
            self.down_blocks.append(
                _DownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    time_embed_dim=time_embed_dim,
                    num_layers=config.layers_per_block,
                    norm_num_groups=config.norm_num_groups,
                    cross_attention_dim=(
                        config.cross_attention_dim if has_cross_attention else None
                    ),
                    attention_head_dim=config.attention_head_dim,
                    add_downsample=not is_final,
                    linear_class=linear_class,
                    use_linear_projection=config.use_linear_projection,
                )
            )

        # Mid block
        self.mid_block = _UNetMidBlock2DCrossAttn(
            channels=block_out_channels[-1],
            time_embed_dim=time_embed_dim,
            cross_attention_dim=config.cross_attention_dim,
            attention_head_dim=config.attention_head_dim,
            norm_num_groups=config.norm_num_groups,
            linear_class=linear_class,
            use_linear_projection=config.use_linear_projection,
        )

        # Up blocks (reversed)
        reversed_channels = list(reversed(block_out_channels))
        self.up_blocks = nn.ModuleList()
        output_channel = reversed_channels[0]
        up_block_types = config.up_block_types or tuple(
            "CrossAttnUpBlock2D" for _ in block_out_channels
        )
        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_channels[i]
            input_channel = reversed_channels[min(i + 1, len(block_out_channels) - 1)]
            is_final = i == len(block_out_channels) - 1
            has_cross_attention = "CrossAttn" in up_block_types[i]
            self.up_blocks.append(
                _UpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channels=prev_output_channel,
                    time_embed_dim=time_embed_dim,
                    num_layers=config.layers_per_block + 1,
                    norm_num_groups=config.norm_num_groups,
                    cross_attention_dim=(
                        config.cross_attention_dim if has_cross_attention else None
                    ),
                    attention_head_dim=config.attention_head_dim,
                    add_upsample=not is_final,
                    linear_class=linear_class,
                    use_linear_projection=config.use_linear_projection,
                )
            )

        # Output
        self.conv_norm_out = _GroupNorm(config.norm_num_groups, block_out_channels[0])
        self.conv_out = _Conv2d(
            block_out_channels[0], config.out_channels, kernel_size=3, padding=1
        )
        self._silu = _SiLU()

    def forward(
        self,
        op: OpBuilder,
        sample: ir.Value,
        timestep: ir.Value,
        encoder_hidden_states: ir.Value,
        lora_gates: dict | None = None,
    ):
        """Forward pass for denoising.

        Args:
            op: ONNX op builder.
            sample: Noisy latent [batch, in_channels, height, width]
            timestep: Diffusion timestep [batch]
            encoder_hidden_states: Text encoder output [batch, seq_len, cross_dim]
            lora_gates: Optional ``{adapter_name: scalar ir.Value}`` runtime gates
                for the baked LoRA adapters (1.0 = active, 0.0 = inactive, or a
                blend strength). Shared with every ``LoRALinear`` via the module's
                ``_lora_gates`` mapping.

        Returns:
            noise_pred: Predicted noise [batch, out_channels, height, width]
        """
        # Publish the runtime LoRA gates so every baked LoRALinear reads them.
        if lora_gates:
            self._lora_gates.update(lora_gates)

        # Time embedding: sinusoidal position encoding + MLP
        # Using half_dim = dim // 2 sinusoidal encoding
        t_emb = self._get_timestep_embedding(op, timestep)
        emb = self.time_embedding(op, t_emb)

        # Input conv
        sample = self.conv_in(op, sample)

        # Down
        down_block_res_samples = [sample]
        for down_block in self.down_blocks:
            sample, res_samples = down_block(op, sample, emb, encoder_hidden_states)
            down_block_res_samples.extend(res_samples)

        # Mid
        sample = self.mid_block(op, sample, emb, encoder_hidden_states)

        # Up
        for up_block in self.up_blocks:
            sample = up_block(op, sample, emb, down_block_res_samples, encoder_hidden_states)

        # Output
        sample = self.conv_norm_out(op, sample)
        sample = self._silu(op, sample)
        sample = self.conv_out(op, sample)

        return sample

    def _get_timestep_embedding(self, op: OpBuilder, timestep):
        """Sinusoidal timestep embedding."""
        half_dim = self.time_proj_dim // 2
        exponent = -math.log(10000.0) / half_dim
        # Create frequency array as constant
        freqs = np.exp(np.arange(half_dim) * exponent).astype(np.float32)
        freq_const = op.Constant(value_floats=freqs.tolist())

        # timestep: [batch] → [batch, 1]
        t = op.Cast(timestep, to=1)  # FLOAT
        t = op.Unsqueeze(t, [1])
        args = op.Mul(t, op.Unsqueeze(freq_const, [0]))
        embedding = op.Concat(op.Cos(args), op.Sin(args), axis=-1)
        return embedding

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Align diffusers UNet weight names to this from-scratch UNet.

        The from-scratch attention flattens diffusers' per-attention
        ``transformer_blocks.{i}`` level, and names the GEGLU feed-forward
        ``ff.proj_in`` / ``ff.proj_out`` (diffusers: ``ff.net.0.proj`` /
        ``ff.net.2``). Everything else matches diffusers directly.
        """
        import re

        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = re.sub(r"transformer_blocks\.\d+\.", "", key)
            new_key = new_key.replace("ff.net.0.proj.", "ff.proj_in.").replace(
                "ff.net.2.", "ff.proj_out."
            )
            renamed[new_key] = value
        return renamed


def remap_diffusers_unet_lora(lora_state_dict: dict, adapter_name: str) -> dict:
    """Remap a diffusers-format UNet LoRA state dict to this UNet's baked param names.

    diffusers (via ``convert_state_dict_to_diffusers`` / PEFT) names the low-rank
    factors under a per-attention ``transformer_blocks.{i}`` level and without an
    adapter name, e.g.::

        down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight
        down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight

    The from-scratch UNet flattens the single transformer block into
    ``attn1``/``attn2`` and names each baked adapter ``lora_A.{name}.weight`` /
    ``lora_B.{name}.weight`` (``down`` -> ``A`` = ``[rank, in]``, ``up`` -> ``B``
    = ``[out, rank]``). This maps the former onto the latter for
    ``adapter_name`` so the loaded weights land on the right initializers.

    Both the classic ``lora.down``/``lora.up`` and the newer PEFT
    ``lora_A``/``lora_B`` source spellings are accepted. Keys that are not LoRA
    factors are passed through unchanged.
    """
    import re

    remapped: dict = {}
    for key, value in lora_state_dict.items():
        new_key = re.sub(r"\.transformer_blocks\.\d+", "", key)
        if new_key.endswith(".lora.down.weight"):
            new_key = new_key[: -len(".lora.down.weight")] + f".lora_A.{adapter_name}.weight"
        elif new_key.endswith(".lora.up.weight"):
            new_key = new_key[: -len(".lora.up.weight")] + f".lora_B.{adapter_name}.weight"
        elif new_key.endswith(".lora_A.weight"):
            new_key = new_key[: -len(".lora_A.weight")] + f".lora_A.{adapter_name}.weight"
        elif new_key.endswith(".lora_B.weight"):
            new_key = new_key[: -len(".lora_B.weight")] + f".lora_B.{adapter_name}.weight"
        else:
            # Not a recognized LoRA factor (e.g. an alpha scalar); pass through.
            remapped[new_key] = value
            continue
        remapped[new_key] = value
    return remapped


def load_unet_lora_safetensors(path: str, adapter_name: str) -> dict:
    """Load a diffusers-format UNet LoRA ``.safetensors`` and remap onto baked params.

    Remaps for ``adapter_name`` onto this UNet's baked ``LoRALinear`` params.

    The returned dict is ready to merge into the UNet component's state dict
    (baked slots come from ``UNet2DConfig.lora_adapters``) before
    :func:`mobius.apply_weights`.
    """
    from safetensors.torch import load_file

    return remap_diffusers_unet_lora(load_file(path), adapter_name)
