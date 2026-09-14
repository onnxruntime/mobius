# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration for the Transformers-native VibeVoice text-to-speech model."""

from __future__ import annotations

import dataclasses

from mobius._configs._base import (
    ArchitectureConfig,
    _as_attribute_config,
    _resolve_dtype_value,
)


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


@dataclasses.dataclass
class VibeVoiceStreamingTokenizerConfig:
    """Source-faithful acoustic decoder configuration for VibeVoice Realtime."""

    channels: int = 1
    vae_dim: int = 64
    decoder_n_filters: int = 32
    decoder_ratios: list[int] = dataclasses.field(default_factory=lambda: [8, 5, 5, 4, 2, 2])
    encoder_depths: list[int] = dataclasses.field(
        default_factory=lambda: [3, 3, 3, 3, 3, 3, 8]
    )
    decoder_depths: list[int] | None = None
    kernel_size: int = 7
    ffn_expansion: int = 4
    layernorm_eps: float = 1e-5
    layer_scale_init_value: float = 1e-6
    causal: bool = True
    conv_bias: bool = True
    conv_norm: str = "none"
    pad_mode: str = "constant"
    layernorm: str = "RMSNorm"
    mixer_layer: str = "depthwise_conv"
    disable_last_norm: bool = True

    def as_decoder_config(self) -> VibeVoiceTokenizerConfig:
        """Adapt the source decoder fields to the shared ONNX decoder primitive."""
        decoder_depths = self.decoder_depths or list(reversed(self.encoder_depths))
        return VibeVoiceTokenizerConfig(
            channels=self.channels,
            hidden_size=self.vae_dim,
            kernel_size=self.kernel_size,
            num_filters=self.decoder_n_filters,
            # The shared primitive reverses depths itself, matching the source
            # tokenizer construction, while Realtime preserves ratio order.
            downsampling_ratios=list(self.decoder_ratios),
            depths=list(reversed(decoder_depths)),
            ffn_expansion=self.ffn_expansion,
            hidden_act="gelu",
            rms_norm_eps=self.layernorm_eps,
            layer_scale_init_value=self.layer_scale_init_value,
        )

    def validate(self) -> None:
        """Reject tokenizer variants the shared causal decoder cannot represent."""
        unsupported = {
            "causal": self.causal is not True,
            "conv_bias": self.conv_bias is not True,
            "conv_norm": self.conv_norm != "none",
            "pad_mode": self.pad_mode != "constant",
            "layernorm": self.layernorm != "RMSNorm",
            "mixer_layer": self.mixer_layer != "depthwise_conv",
            "disable_last_norm": self.disable_last_norm is not True,
        }
        selected = sorted(name for name, enabled in unsupported.items() if enabled)
        if selected:
            raise ValueError(
                "VibeVoice Realtime only supports the source acoustic decoder "
                f"topology; unsupported tokenizer fields: {', '.join(selected)}"
            )
        if len(self.decoder_ratios) + 1 != len(
            self.decoder_depths or list(reversed(self.encoder_depths))
        ):
            raise ValueError(
                "VibeVoice Realtime decoder depths must contain one stage for the stem "
                "and one for every decoder ratio"
            )


@dataclasses.dataclass
class VibeVoiceStreamingDiffusionConfig(VibeVoiceDiffusionConfig):
    """Diffusion and scheduler settings owned by Realtime host orchestration."""

    num_train_timesteps: int = 1000
    num_inference_steps: int = 20
    beta_schedule: str = "cosine"
    prediction_type: str = "v_prediction"
    ddpm_batch_mul: int = 4


def _tokenizer_config(config, *, default_hidden_size: int) -> VibeVoiceTokenizerConfig:
    config = _as_attribute_config(config)
    return VibeVoiceTokenizerConfig(
        channels=int(getattr(config, "channels", 1)),
        hidden_size=int(getattr(config, "hidden_size", default_hidden_size)),
        kernel_size=int(getattr(config, "kernel_size", 7)),
        num_filters=int(getattr(config, "num_filters", 32)),
        downsampling_ratios=list(getattr(config, "downsampling_ratios", [2, 2, 4, 5, 5, 8])),
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
        diffusion_config = _as_attribute_config(getattr(parent, "diffusion_head_config", None))
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


def _streaming_depths(value) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [int(part) for part in value.split("-")]
    return [int(part) for part in value]


@dataclasses.dataclass
class VibeVoiceStreamingConfig(ArchitectureConfig):
    """Mobius configuration for the two-backbone VibeVoice Realtime pipeline."""

    acoustic_tokenizer: VibeVoiceStreamingTokenizerConfig = dataclasses.field(
        default_factory=VibeVoiceStreamingTokenizerConfig
    )
    diffusion_head: VibeVoiceStreamingDiffusionConfig = dataclasses.field(
        default_factory=VibeVoiceStreamingDiffusionConfig
    )
    tts_backbone_num_hidden_layers: int = 20
    text_window_size: int = 5
    speech_window_size: int = 6
    sampling_rate: int = 24_000

    @property
    def lm_backbone_num_hidden_layers(self) -> int:
        """Lower Qwen2 text layers preceding the upper TTS backbone."""
        return self.num_hidden_layers - self.tts_backbone_num_hidden_layers

    @classmethod
    def from_transformers(
        cls,
        config,
        parent_config=None,
        *,
        allow_block_fp8_dense_fallback: bool = False,
    ) -> VibeVoiceStreamingConfig:
        """Extract the source composite while retaining its Qwen2 decoder settings."""
        parent = _as_attribute_config(parent_config or config)
        decoder = _as_attribute_config(getattr(parent, "decoder_config", config))
        result = super().from_transformers(
            decoder,
            parent_config=parent,
            allow_block_fp8_dense_fallback=allow_block_fp8_dense_fallback,
        )
        source_dtype = _resolve_dtype_value(getattr(parent, "torch_dtype", None))
        if source_dtype is not None:
            result = dataclasses.replace(result, dtype=source_dtype)
        tokenizer_config = _as_attribute_config(
            getattr(parent, "acoustic_tokenizer_config", None)
        )
        encoder_depths = _streaming_depths(
            getattr(tokenizer_config, "encoder_depths", "3-3-3-3-3-3-8")
        )
        assert encoder_depths is not None
        decoder_depths = _streaming_depths(getattr(tokenizer_config, "decoder_depths", None))
        tokenizer = VibeVoiceStreamingTokenizerConfig(
            channels=int(getattr(tokenizer_config, "channels", 1)),
            vae_dim=int(getattr(tokenizer_config, "vae_dim", 64)),
            decoder_n_filters=int(getattr(tokenizer_config, "decoder_n_filters", 32)),
            decoder_ratios=list(
                getattr(tokenizer_config, "decoder_ratios", [8, 5, 5, 4, 2, 2])
            ),
            encoder_depths=encoder_depths,
            decoder_depths=decoder_depths,
            kernel_size=int(getattr(tokenizer_config, "kernel_size", 7)),
            layernorm_eps=float(getattr(tokenizer_config, "layernorm_eps", 1e-5)),
            layer_scale_init_value=float(
                getattr(tokenizer_config, "layer_scale_init_value", 1e-6)
            ),
            causal=bool(getattr(tokenizer_config, "causal", True)),
            conv_bias=bool(getattr(tokenizer_config, "conv_bias", True)),
            conv_norm=str(getattr(tokenizer_config, "conv_norm", "none")),
            pad_mode=str(getattr(tokenizer_config, "pad_mode", "constant")),
            layernorm=str(getattr(tokenizer_config, "layernorm", "RMSNorm")),
            mixer_layer=str(getattr(tokenizer_config, "mixer_layer", "depthwise_conv")),
            disable_last_norm=bool(getattr(tokenizer_config, "disable_last_norm", True)),
        )
        diffusion_config = _as_attribute_config(getattr(parent, "diffusion_head_config", None))
        diffusion_hidden_size = int(
            getattr(diffusion_config, "hidden_size", result.hidden_size)
        )
        diffusion = VibeVoiceStreamingDiffusionConfig(
            hidden_size=diffusion_hidden_size,
            intermediate_size=int(
                diffusion_hidden_size * float(getattr(diffusion_config, "head_ffn_ratio", 3.0))
            ),
            latent_size=int(getattr(diffusion_config, "latent_size", tokenizer.vae_dim)),
            num_hidden_layers=int(getattr(diffusion_config, "head_layers", 4)),
            rms_norm_eps=float(getattr(diffusion_config, "rms_norm_eps", 1e-5)),
            hidden_act="silu",
            frequency_embedding_size=256,
            diffusion_max_period=10_000,
            mlp_bias=False,
            num_train_timesteps=int(getattr(diffusion_config, "ddpm_num_steps", 1000)),
            num_inference_steps=int(getattr(diffusion_config, "ddpm_num_inference_steps", 20)),
            beta_schedule=str(getattr(diffusion_config, "ddpm_beta_schedule", "cosine")),
            prediction_type=str(getattr(diffusion_config, "prediction_type", "v_prediction")),
            ddpm_batch_mul=int(getattr(diffusion_config, "ddpm_batch_mul", 4)),
        )
        if diffusion.hidden_size != result.hidden_size:
            raise ValueError(
                "VibeVoice Realtime diffusion hidden_size must match Qwen2 hidden_size"
            )
        if diffusion.latent_size != tokenizer.vae_dim:
            raise ValueError(
                "VibeVoice Realtime diffusion latent_size must match acoustic vae_dim"
            )
        return dataclasses.replace(
            result,
            model_type=str(getattr(parent, "model_type", "vibevoice_streaming")),
            acoustic_tokenizer=tokenizer,
            diffusion_head=diffusion,
            tts_backbone_num_hidden_layers=int(
                getattr(parent, "tts_backbone_num_hidden_layers", 20)
            ),
            text_window_size=5,
            speech_window_size=6,
            sampling_rate=24_000,
            tie_word_embeddings=False,
        )

    def validate(self) -> None:
        """Validate source-fixed Realtime stage boundaries before graph construction."""
        super().validate()
        self.acoustic_tokenizer.validate()
        if not 0 < self.tts_backbone_num_hidden_layers < self.num_hidden_layers:
            raise ValueError(
                "VibeVoice Realtime tts_backbone_num_hidden_layers must leave at least "
                "one lower Qwen2 text layer"
            )
        if self.diffusion_head.hidden_size != self.hidden_size:
            raise ValueError(
                "VibeVoice Realtime diffusion hidden_size must match Qwen2 hidden_size"
            )
        if self.diffusion_head.latent_size != self.acoustic_tokenizer.vae_dim:
            raise ValueError(
                "VibeVoice Realtime diffusion latent_size must match acoustic vae_dim"
            )
