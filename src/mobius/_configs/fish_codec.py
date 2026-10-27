# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned Drama 3 waveform codec configuration, independent of the token tower."""

from __future__ import annotations

import math
from dataclasses import dataclass

DRAMA3_CODEC_REPO = "fishaudio/drama-3"
DRAMA3_CODEC_REVISION = "b2b5ee998034b2975720906c815e7fb0e9f007bd"
FISH_CODEC_SOURCE_REVISION = "541566932abbaebcb0a40e562a1f846738383f38"


@dataclass(frozen=True)
class FishCodecConfig:
    """Decode-only configuration of upstream ``modded_dac_vq.yaml``.

    The encoder and pre-quantization transformer are deliberately excluded.
    Upstream constructs, but never calls, the configured decoder transformer.
    ``max_frames`` bounds the precomputed, BF16-rounded RoPE table.
    """

    sample_rate: int = 44100
    latent_dim: int = 1024
    residual_codebooks: int = 9
    semantic_codebook_size: int = 4096
    residual_codebook_size: int = 1024
    codebook_dim: int = 8
    transformer_layers: int = 8
    transformer_heads: int = 16
    transformer_intermediate_size: int = 3072
    transformer_window: int = 128
    norm_eps: float = 1e-5
    rope_base: float = 10000.0
    max_frames: int = 327680
    quantizer_rates: tuple[int, ...] = (2, 2)
    decoder_dim: int = 1536
    decoder_rates: tuple[int, ...] = (8, 8, 4, 2)

    def __post_init__(self):
        positive = (
            self.sample_rate,
            self.latent_dim,
            self.residual_codebooks,
            self.semantic_codebook_size,
            self.residual_codebook_size,
            self.codebook_dim,
            self.transformer_heads,
            self.transformer_intermediate_size,
            self.transformer_window,
            self.max_frames,
            self.decoder_dim,
            *self.quantizer_rates,
            *self.decoder_rates,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Codec dimensions, rates and window must be positive")
        if self.transformer_layers < 0:
            raise ValueError("transformer_layers must be nonnegative")
        if self.latent_dim % self.transformer_heads or self.head_dim % 2:
            raise ValueError("latent_dim must divide into heads with even head_dim")
        if not self.decoder_rates or self.decoder_dim % (2 ** len(self.decoder_rates)):
            raise ValueError("decoder_dim must be divisible by 2**len(decoder_rates)")
        if self.norm_eps <= 0 or self.rope_base <= 0:
            raise ValueError("norm_eps and rope_base must be positive")

    @property
    def head_dim(self) -> int:
        return self.latent_dim // self.transformer_heads

    @property
    def num_codebooks(self) -> int:
        return self.residual_codebooks + 1

    @property
    def samples_per_frame(self) -> int:
        return math.prod(self.quantizer_rates) * math.prod(self.decoder_rates)
