# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration extraction for the original VibeVoice offline ASR checkpoint."""

from __future__ import annotations

import dataclasses

from mobius._configs._base import ArchitectureConfig, _as_attribute_config
from mobius._configs.vibevoice import VibeVoiceTokenizerConfig


def _tokenizer_config(config, *, default_hidden_size: int) -> VibeVoiceTokenizerConfig:
    """Translate the original tokenizer schema to its encoder-only geometry."""
    source = _as_attribute_config(config)
    ratios = getattr(source, "downsampling_ratios", None)
    if ratios is None:
        # The original checkpoint stores ratios in decoder order; Transformers'
        # converter reverses them before constructing the causal encoder.
        ratios = reversed(getattr(source, "encoder_ratios", (8, 5, 5, 4, 2, 2)))
    depths = getattr(source, "depths", None)
    if depths is None:
        depths = getattr(source, "encoder_depths", (3, 3, 3, 3, 3, 3, 8))
    if isinstance(depths, str):
        depths = [int(value) for value in depths.split("-")]

    vae_std = getattr(source, "vae_std", None)
    if vae_std is None:
        # Microsoft VibeVoice samples with 0.8 * vae_std. The Transformers
        # conversion records the equivalent standard deviation in ``vae_std``.
        vae_std = float(getattr(source, "fix_std", 0.5)) / 0.8
    return VibeVoiceTokenizerConfig(
        channels=int(getattr(source, "channels", 1)),
        hidden_size=int(
            getattr(source, "hidden_size", getattr(source, "vae_dim", default_hidden_size))
        ),
        kernel_size=int(getattr(source, "kernel_size", 7)),
        num_filters=int(getattr(source, "num_filters", getattr(source, "encoder_n_filters", 32))),
        downsampling_ratios=list(ratios),
        depths=list(depths),
        ffn_expansion=int(getattr(source, "ffn_expansion", 4)),
        hidden_act=str(getattr(source, "hidden_act", "gelu")),
        rms_norm_eps=float(getattr(source, "rms_norm_eps", getattr(source, "layernorm_eps", 1e-5))),
        layer_scale_init_value=float(getattr(source, "layer_scale_init_value", 1e-6)),
        vae_std=float(vae_std),
    )


@dataclasses.dataclass
class VibeVoiceASRConfig(ArchitectureConfig):
    """Offline VibeVoice ASR geometry: two causal audio encoders and Qwen2."""

    acoustic_tokenizer: VibeVoiceTokenizerConfig = dataclasses.field(
        default_factory=VibeVoiceTokenizerConfig
    )
    semantic_tokenizer: VibeVoiceTokenizerConfig = dataclasses.field(
        default_factory=lambda: VibeVoiceTokenizerConfig(hidden_size=128, vae_std=0.0)
    )
    acoustic_tokenizer_chunk_size: int = 1_440_000
    audio_token_id: int = 151648
    audio_bos_token_id: int = 151646
    audio_eos_token_id: int = 151647
    sampling_rate: int = 24_000

    @classmethod
    def from_transformers(
        cls,
        config,
        parent_config=None,
        *,
        allow_block_fp8_dense_fallback: bool = False,
    ) -> VibeVoiceASRConfig:
        """Extract both original and Transformers-converted ASR configurations."""
        parent = _as_attribute_config(parent_config or config)
        decoder = _as_attribute_config(
            getattr(parent, "decoder_config", None) or getattr(parent, "text_config", config)
        )
        result = super().from_transformers(
            decoder,
            parent_config=parent,
            allow_block_fp8_dense_fallback=allow_block_fp8_dense_fallback,
        )
        acoustic_source = getattr(
            parent,
            "acoustic_tokenizer_encoder_config",
            getattr(parent, "acoustic_tokenizer_config", None),
        )
        semantic_source = getattr(
            parent,
            "semantic_tokenizer_encoder_config",
            getattr(parent, "semantic_tokenizer_config", None),
        )
        acoustic = _tokenizer_config(acoustic_source, default_hidden_size=64)
        semantic = _tokenizer_config(semantic_source, default_hidden_size=128)
        if acoustic.hop_length != semantic.hop_length:
            raise ValueError(
                "VibeVoice ASR acoustic and semantic tokenizers must have the same hop length"
            )
        chunk_size = int(getattr(parent, "acoustic_tokenizer_chunk_size", 1_440_000))
        if chunk_size % acoustic.hop_length:
            raise ValueError(
                "VibeVoice ASR acoustic_tokenizer_chunk_size must be divisible by the "
                f"{acoustic.hop_length}-sample tokenizer hop"
            )
        return dataclasses.replace(
            result,
            model_type="vibevoice_asr",
            acoustic_tokenizer=acoustic,
            semantic_tokenizer=semantic,
            acoustic_tokenizer_chunk_size=chunk_size,
            audio_token_id=int(getattr(parent, "audio_token_id", 151648)),
            audio_bos_token_id=int(getattr(parent, "audio_bos_token_id", 151646)),
            audio_eos_token_id=int(getattr(parent, "audio_eos_token_id", 151647)),
            eos_token_id=getattr(parent, "eos_token_id", 151643),
            pad_token_id=getattr(parent, "pad_token_id", 151643),
            sampling_rate=24_000,
        )
