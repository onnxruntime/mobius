# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration for the Transformers-native VibeVoice text-to-speech model."""

from __future__ import annotations

import dataclasses

from mobius._configs._base import ArchitectureConfig, _as_attribute_config


@dataclasses.dataclass
class VibeVoiceTokenizerConfig:
    """Continuous acoustic or semantic tokenizer geometry."""

    channels: int = 1
    hidden_size: int = 64
    kernel_size: int = 7
    num_filters: int = 32
    downsampling_ratios: list[int] = dataclasses.field(
        default_factory=lambda: [2, 2, 4, 5, 5, 8]
    )
    depths: list[int] = dataclasses.field(default_factory=lambda: [3, 3, 3, 3, 3, 3, 8])
    ffn_expansion: int = 4
    hidden_act: str = "gelu"
    rms_norm_eps: float = 1e-5
    layer_scale_init_value: float = 1e-6
    vae_std: float = 0.625

    @property
    def hop_length(self) -> int:
        result = 1
        for ratio in self.downsampling_ratios:
            result *= ratio
        return result


@dataclasses.dataclass
class VibeVoiceDiffusionConfig:
    """Token-level diffusion-head geometry."""

    hidden_size: int = 1536
    intermediate_size: int = 4608
    latent_size: int = 64
    num_hidden_layers: int = 4
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    frequency_embedding_size: int = 256
    diffusion_max_period: int = 10_000
    mlp_bias: bool = False


def _tokenizer_config(config, *, default_hidden_size: int) -> VibeVoiceTokenizerConfig:
    config = _as_attribute_config(config)
    return VibeVoiceTokenizerConfig(
        channels=int(getattr(config, "channels", 1)),
        hidden_size=int(getattr(config, "hidden_size", default_hidden_size)),
        kernel_size=int(getattr(config, "kernel_size", 7)),
        num_filters=int(getattr(config, "num_filters", 32)),
        downsampling_ratios=list(
            getattr(config, "downsampling_ratios", [2, 2, 4, 5, 5, 8])
        ),
        depths=list(getattr(config, "depths", [3, 3, 3, 3, 3, 3, 8])),
        ffn_expansion=int(getattr(config, "ffn_expansion", 4)),
        hidden_act=str(getattr(config, "hidden_act", "gelu")),
        rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-5)),
        layer_scale_init_value=float(getattr(config, "layer_scale_init_value", 1e-6)),
        vae_std=float(getattr(config, "vae_std", 0.625)),
    )


@dataclasses.dataclass
class VibeVoiceConfig(ArchitectureConfig):
    """Mobius configuration for VibeVoice's LM, tokenizers, and diffusion head."""

    acoustic_tokenizer: VibeVoiceTokenizerConfig = dataclasses.field(
        default_factory=VibeVoiceTokenizerConfig
    )
    semantic_tokenizer: VibeVoiceTokenizerConfig = dataclasses.field(
        default_factory=lambda: VibeVoiceTokenizerConfig(hidden_size=128)
    )
    diffusion_head: VibeVoiceDiffusionConfig = dataclasses.field(
        default_factory=VibeVoiceDiffusionConfig
    )
    audio_bos_token_id: int = 151652
    audio_eos_token_id: int = 151653
    audio_token_id: int = 151654
    sampling_rate: int = 24_000
    guidance_scale: float = 1.3
    num_diffusion_steps: int = 10

    @classmethod
    def from_transformers(
        cls,
        config,
        parent_config=None,
        *,
        allow_block_fp8_dense_fallback: bool = False,
    ) -> VibeVoiceConfig:
        """Extract the pinned native HF composite while preserving Qwen2 semantics."""
        parent = _as_attribute_config(parent_config or config)
        result = super().from_transformers(
            config,
            parent_config=parent,
            allow_block_fp8_dense_fallback=allow_block_fp8_dense_fallback,
        )
        audio = _tokenizer_config(
            getattr(parent, "audio_config", None),
            default_hidden_size=64,
        )
        semantic = _tokenizer_config(
            getattr(parent, "semantic_model_config", None),
            default_hidden_size=128,
        )
        diffusion_config = _as_attribute_config(
            getattr(parent, "diffusion_head_config", None)
        )
        diffusion = VibeVoiceDiffusionConfig(
            hidden_size=int(getattr(diffusion_config, "hidden_size", result.hidden_size)),
            intermediate_size=int(
                getattr(diffusion_config, "intermediate_size", 3 * result.hidden_size)
            ),
            latent_size=int(getattr(diffusion_config, "latent_size", audio.hidden_size)),
            num_hidden_layers=int(getattr(diffusion_config, "num_hidden_layers", 4)),
            rms_norm_eps=float(getattr(diffusion_config, "rms_norm_eps", 1e-5)),
            hidden_act=str(getattr(diffusion_config, "hidden_act", "silu")),
            frequency_embedding_size=int(
                getattr(diffusion_config, "frequency_embedding_size", 256)
            ),
            diffusion_max_period=int(
                getattr(diffusion_config, "diffusion_max_period", 10_000)
            ),
            mlp_bias=bool(getattr(diffusion_config, "mlp_bias", False)),
        )
        if diffusion.hidden_size != result.hidden_size:
            raise ValueError(
                "VibeVoice diffusion hidden_size must match the Qwen2 hidden_size"
            )
        if diffusion.latent_size != audio.hidden_size:
            raise ValueError(
                "VibeVoice diffusion latent_size must match the acoustic tokenizer hidden_size"
            )
        return dataclasses.replace(
            result,
            acoustic_tokenizer=audio,
            semantic_tokenizer=semantic,
            diffusion_head=diffusion,
            audio_bos_token_id=int(getattr(parent, "audio_bos_token_id", 151652)),
            audio_eos_token_id=int(getattr(parent, "audio_eos_token_id", 151653)),
            audio_token_id=int(getattr(parent, "audio_token_id", 151654)),
            eos_token_id=getattr(parent, "eos_token_id", 151643),
            pad_token_id=getattr(parent, "pad_token_id", 151643),
            sampling_rate=24_000,
        )
