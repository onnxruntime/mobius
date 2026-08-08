# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the diffusers-config adapters (``mobius._diffusers_configs``).

Focuses on the classic Stable Diffusion adapters added for the from-scratch SD
build: the CLIP text-encoder adapter (transformers field names -> generic
``ArchitectureConfig``) and the UNet adapter.
"""

from __future__ import annotations

import pytest

from mobius._diffusers_configs import (
    CLIPTextConfig,
    QwenImageConfig,
    QwenImageVAEConfig,
    UNet2DConfig,
)


class TestCLIPTextConfigFromDiffusers:
    """CLIPTextConfig.from_diffusers maps a transformers text-encoder config."""

    def test_maps_sd15_text_encoder_config(self):
        # A realistic Stable Diffusion 1.5 text_encoder/config.json subset.
        config = {
            "vocab_size": 49408,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "max_position_embeddings": 77,
            "layer_norm_eps": 1e-5,
            "hidden_act": "quick_gelu",
        }
        arch = CLIPTextConfig.from_diffusers(config)
        assert arch.vocab_size == 49408
        assert arch.hidden_size == 768
        assert arch.intermediate_size == 3072
        assert arch.num_hidden_layers == 12
        assert arch.num_attention_heads == 12
        # CLIP text attention is full multi-head (no GQA): kv heads == heads.
        assert arch.num_key_value_heads == 12
        # head_dim is derived from hidden_size / num_attention_heads.
        assert arch.head_dim == 64
        assert arch.max_position_embeddings == 77
        # transformers `layer_norm_eps` feeds the from-scratch LayerNorm eps.
        assert arch.rms_norm_eps == pytest.approx(1e-5)
        assert arch.hidden_act == "quick_gelu"

    def test_defaults_are_sd_clip(self):
        arch = CLIPTextConfig.from_diffusers({})
        assert arch.vocab_size == 49408
        assert arch.hidden_size == 768
        assert arch.num_hidden_layers == 12
        assert arch.num_attention_heads == 12
        assert arch.head_dim == 64
        assert arch.max_position_embeddings == 77
        assert arch.hidden_act == "quick_gelu"

    def test_head_dim_tracks_hidden_size_and_heads(self):
        # SD 2.x uses a larger text encoder (hidden 1024, 16 heads -> head_dim 64).
        arch = CLIPTextConfig.from_diffusers(
            {"hidden_size": 1024, "num_attention_heads": 16, "num_hidden_layers": 23}
        )
        assert arch.hidden_size == 1024
        assert arch.num_attention_heads == 16
        assert arch.num_key_value_heads == 16
        assert arch.head_dim == 64
        assert arch.num_hidden_layers == 23


class TestUNet2DConfigFromDiffusers:
    """UNet2DConfig.from_diffusers maps a diffusers UNet config."""

    def test_maps_core_unet_fields(self):
        config = {
            "in_channels": 4,
            "out_channels": 4,
            "block_out_channels": [320, 640, 1280, 1280],
            "cross_attention_dim": 768,
            "layers_per_block": 2,
        }
        unet = UNet2DConfig.from_diffusers(config)
        assert unet.in_channels == 4
        assert unet.out_channels == 4
        assert tuple(unet.block_out_channels) == (320, 640, 1280, 1280)
        assert unet.cross_attention_dim == 768
        assert unet.layers_per_block == 2


class TestQwenImageEditOfficialConfigs:
    """L2 coverage for the official Qwen-Image-Edit-2509 component configs."""

    def test_transformer_config(self):
        config = QwenImageConfig.from_diffusers(
            {
                "_class_name": "QwenImageTransformer2DModel",
                "attention_head_dim": 128,
                "axes_dims_rope": [16, 56, 56],
                "guidance_embeds": False,
                "in_channels": 64,
                "joint_attention_dim": 3584,
                "num_attention_heads": 24,
                "num_layers": 60,
                "out_channels": 16,
                "patch_size": 2,
            }
        )
        assert config.in_channels == 64
        assert config.out_channels == 16
        assert config.num_layers == 60
        assert config.axes_dims_rope == (16, 56, 56)

    def test_vae_config_retains_latent_normalization(self):
        means = [float(i) / 10 for i in range(16)]
        stds = [1.0 + float(i) / 10 for i in range(16)]
        config = QwenImageVAEConfig.from_diffusers(
            {
                "_class_name": "AutoencoderKLQwenImage",
                "base_dim": 96,
                "z_dim": 16,
                "dim_mult": [1, 2, 4, 4],
                "num_res_blocks": 2,
                "temperal_downsample": [False, True, True],
                "latents_mean": means,
                "latents_std": stds,
            }
        )
        assert config.latents_mean == tuple(means)
        assert config.latents_std == tuple(stds)
