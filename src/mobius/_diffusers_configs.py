# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration for diffusers models."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mobius._configs import ArchitectureConfig


class CLIPTextConfig:
    """Adapter that builds an :class:`ArchitectureConfig` for a CLIP text encoder.

    Classic Stable Diffusion 1.x/2.x pipelines use a ``transformers``
    ``CLIPTextModel`` as their prompt encoder. Its ``text_encoder/config.json``
    uses transformers field names, which this adapter maps onto the generic
    :class:`mobius._configs.ArchitectureConfig` consumed by
    :class:`mobius.models.clip.CLIPTextModel`.
    """

    @classmethod
    def from_diffusers(cls, config: dict) -> ArchitectureConfig:
        """Create an :class:`ArchitectureConfig` from a diffusers text-encoder config.

        Args:
            config: Parsed ``text_encoder/config.json`` dictionary.

        Returns:
            An :class:`ArchitectureConfig` describing the CLIP text encoder. The
            transformers ``layer_norm_eps`` is mapped onto ``rms_norm_eps`` (the
            field the from-scratch CLIP layers read for their ``LayerNorm`` eps),
            and ``head_dim`` is derived from ``hidden_size / num_attention_heads``.
        """
        from mobius._configs import ArchitectureConfig

        if hasattr(config, "to_dict"):
            config = dict(config.items())
        hidden_size = config.get("hidden_size", 768)
        num_attention_heads = config.get("num_attention_heads", 12)
        return ArchitectureConfig(
            vocab_size=config.get("vocab_size", 49408),
            hidden_size=hidden_size,
            intermediate_size=config.get("intermediate_size", 3072),
            num_hidden_layers=config.get("num_hidden_layers", 12),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            head_dim=hidden_size // num_attention_heads,
            max_position_embeddings=config.get("max_position_embeddings", 77),
            rms_norm_eps=config.get("layer_norm_eps", 1e-5),
            hidden_act=config.get("hidden_act", "quick_gelu"),
        )


@dataclasses.dataclass
class VAEConfig:
    """Configuration for AutoencoderKL (VAE) models."""

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"
    sample_size: int = 256
    scaling_factor: float = 0.18215
    mid_block_add_attention: bool = True
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True

    @classmethod
    def from_diffusers(cls, config: dict) -> VAEConfig:
        """Create a VAEConfig from a diffusers config dict."""
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 3),
            latent_channels=config.get("latent_channels", 4),
            block_out_channels=tuple(config.get("block_out_channels", [128, 256, 512, 512])),
            layers_per_block=config.get("layers_per_block", 2),
            norm_num_groups=config.get("norm_num_groups", 32),
            act_fn=config.get("act_fn", "silu"),
            sample_size=config.get("sample_size", 256),
            scaling_factor=config.get("scaling_factor", 0.18215),
            mid_block_add_attention=config.get("mid_block_add_attention", True),
            use_quant_conv=config.get("use_quant_conv", True),
            use_post_quant_conv=config.get("use_post_quant_conv", True),
        )


@dataclasses.dataclass
class UNet2DConfig:
    """Configuration for UNet2DConditionModel."""

    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    cross_attention_dim: int = 768
    attention_head_dim: int = 8
    act_fn: str = "silu"
    sample_size: int = 64
    # Per-block type names from diffusers. A block gets cross-attention only when
    # its type name contains ``CrossAttn`` (Stable Diffusion 1.x uses a plain
    # ``DownBlock2D`` for the last down block and ``UpBlock2D`` for the first up
    # block). ``None`` = every block has cross-attention (legacy behavior).
    down_block_types: tuple[str, ...] | None = None
    up_block_types: tuple[str, ...] | None = None
    addition_embed_type: str | None = None
    addition_time_embed_dim: int | None = None
    projection_class_embeddings_input_dim: int | None = None
    # Whether the Transformer2D blocks use a Linear (True) or 1x1 Conv (False,
    # Stable Diffusion 1.x default) for their proj_in/proj_out.
    use_linear_projection: bool = False
    # Runtime LoRA adapters to bake into the attention projections as
    # `(name, rank, scale)`. Each becomes a low-rank branch gated at run time by
    # a scalar `lora_gate.{name}` input (1.0 = on, 0.0 = off, or a blend
    # strength). Empty = no LoRA (plain projections).
    lora_adapters: tuple[tuple[str, int, float], ...] = ()

    @classmethod
    def from_diffusers(cls, config: dict) -> UNet2DConfig:
        """Create a UNet2DConfig from a diffusers config dict."""
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            in_channels=config.get("in_channels", 4),
            out_channels=config.get("out_channels", 4),
            block_out_channels=tuple(config.get("block_out_channels", [320, 640, 1280, 1280])),
            layers_per_block=config.get("layers_per_block", 2),
            norm_num_groups=config.get("norm_num_groups", 32),
            cross_attention_dim=config.get("cross_attention_dim", 768),
            attention_head_dim=config.get("attention_head_dim", 8),
            act_fn=config.get("act_fn", "silu"),
            sample_size=config.get("sample_size", 64),
            down_block_types=(
                tuple(config["down_block_types"])
                if config.get("down_block_types") is not None
                else None
            ),
            up_block_types=(
                tuple(config["up_block_types"])
                if config.get("up_block_types") is not None
                else None
            ),
            addition_embed_type=config.get("addition_embed_type"),
            addition_time_embed_dim=config.get("addition_time_embed_dim"),
            projection_class_embeddings_input_dim=config.get(
                "projection_class_embeddings_input_dim"
            ),
            use_linear_projection=config.get("use_linear_projection", False),
        )


@dataclasses.dataclass
class CogVideoXConfig:
    """Configuration for CogVideoXTransformer3DModel.

    3D video diffusion transformer with dual-stream joint attention.
    """

    num_attention_heads: int = 30
    attention_head_dim: int = 64
    in_channels: int = 16
    out_channels: int = 16
    time_embed_dim: int = 512
    text_embed_dim: int = 4096
    num_layers: int = 30
    patch_size: int = 2
    patch_size_t: int | None = None
    sample_height: int = 60
    sample_width: int = 90
    sample_frames: int = 49
    temporal_compression_ratio: int = 4
    max_text_seq_length: int = 226
    spatial_interpolation_scale: float = 1.875
    temporal_interpolation_scale: float = 1.0
    norm_eps: float = 1e-5
    # cross_attention_dim used by VideoDenoisingTask for text conditioning
    cross_attention_dim: int = 4096

    @classmethod
    def from_diffusers(cls, config: dict) -> CogVideoXConfig:
        """Create from a HF diffusers config dict."""
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        heads = config.get("num_attention_heads", 30)
        head_dim = config.get("attention_head_dim", 64)
        text_dim = config.get("text_embed_dim", 4096)
        return cls(
            num_attention_heads=heads,
            attention_head_dim=head_dim,
            in_channels=config.get("in_channels", 16),
            out_channels=config.get("out_channels", 16),
            time_embed_dim=config.get("time_embed_dim", 512),
            text_embed_dim=text_dim,
            num_layers=config.get("num_layers", 30),
            patch_size=config.get("patch_size", 2),
            patch_size_t=config.get("patch_size_t"),
            sample_height=config.get("sample_height", 60),
            sample_width=config.get("sample_width", 90),
            sample_frames=config.get("sample_frames", 49),
            temporal_compression_ratio=config.get("temporal_compression_ratio", 4),
            max_text_seq_length=config.get("max_text_seq_length", 226),
            spatial_interpolation_scale=config.get("spatial_interpolation_scale", 1.875),
            temporal_interpolation_scale=config.get("temporal_interpolation_scale", 1.0),
            norm_eps=config.get("norm_eps", 1e-5),
            cross_attention_dim=text_dim,
        )


@dataclasses.dataclass
class QwenImageConfig:
    """Configuration for QwenImageTransformer2DModel."""

    in_channels: int = 64
    out_channels: int = 16
    patch_size: int = 2
    num_layers: int = 60
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 3584
    guidance_embeds: bool = False
    axes_dims_rope: tuple[int, ...] = (16, 56, 56)
    norm_eps: float = 1e-6
    # cross_attention_dim is used by DenoisingTask for encoder_hidden_states shape
    cross_attention_dim: int = 3584

    @classmethod
    def from_diffusers(cls, config: dict) -> QwenImageConfig:
        """Create a QwenImageConfig from a diffusers config dict."""
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            in_channels=config.get("in_channels", 64),
            out_channels=config.get("out_channels", 16),
            patch_size=config.get("patch_size", 2),
            num_layers=config.get("num_layers", 60),
            attention_head_dim=config.get("attention_head_dim", 128),
            num_attention_heads=config.get("num_attention_heads", 24),
            joint_attention_dim=config.get("joint_attention_dim", 3584),
            guidance_embeds=config.get("guidance_embeds", False),
            axes_dims_rope=tuple(config.get("axes_dims_rope", [16, 56, 56])),
            cross_attention_dim=config.get("joint_attention_dim", 3584),
        )


@dataclasses.dataclass
class QwenImageVAEConfig:
    """Configuration for AutoencoderKLQwenImage (3D causal VAE)."""

    base_dim: int = 96
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0

    @classmethod
    def from_diffusers(cls, config: dict) -> QwenImageVAEConfig:
        """Create a QwenImageVAEConfig from a diffusers config dict."""
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            base_dim=config.get("base_dim", 96),
            z_dim=config.get("z_dim", 16),
            dim_mult=tuple(config.get("dim_mult", [1, 2, 4, 4])),
            num_res_blocks=config.get("num_res_blocks", 2),
            attn_scales=tuple(config.get("attn_scales", [])),
            temperal_downsample=tuple(config.get("temperal_downsample", [False, True, True])),
            dropout=config.get("dropout", 0.0),
        )
