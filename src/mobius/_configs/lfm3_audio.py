# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration extraction for the released and normalized LFM3 Audio layouts."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from mobius._configs._base import Lfm2Config, _shallow_fields
from mobius._configs.lfm3 import Lfm3Config


def _namespace(value):
    return SimpleNamespace(**value) if isinstance(value, dict) else value


def _field(config, name, legacy, default=None):
    config = _namespace(config)
    return getattr(config, name, getattr(config, legacy, default))


@dataclasses.dataclass
class Lfm3AudioConfig(Lfm3Config):
    """LFM3 Audio's language backbone, causal encoder and speech depth decoder."""

    model_type: str = "lfm3_audio"
    encoder_config: Lfm2Config | None = None
    depth_config: Lfm3Config | None = None
    num_mel_bins: int = 128
    conv_channels: int = 256
    encoder_sliding_window: int = 32
    num_codebooks: int = 12
    codebook_size: int = 2017
    audio_eos_code: int = 2016
    maskgit_schedule: tuple[int, ...] = (4, 4, 4)
    audio_token_id: int = 125031
    audio_out_token_id: int = 125032
    audio_start_token_id: int = 125012
    text_end_token_id: int = 125014
    mixed_start_token_id: int = 125015
    interleaved_n_text: int = 6
    interleaved_n_audio: int = 12

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_codebooks <= 0 or self.codebook_size <= 1:
            raise ValueError(
                "LFM3 Audio requires positive num_codebooks and codebook_size > 1."
            )
        if (
            not self.maskgit_schedule
            or min(self.maskgit_schedule) <= 0
            or sum(self.maskgit_schedule) != self.num_codebooks
        ):
            raise ValueError(
                f"LFM3 Audio maskgit_schedule={self.maskgit_schedule} must contain "
                f"positive counts summing to num_codebooks={self.num_codebooks}."
            )
        if not 0 <= self.audio_eos_code < self.codebook_size:
            raise ValueError(
                f"LFM3 Audio audio_eos_code={self.audio_eos_code} must be in "
                f"[0, codebook_size={self.codebook_size})."
            )
        for name in (
            "audio_token_id",
            "audio_out_token_id",
            "audio_start_token_id",
            "text_end_token_id",
            "mixed_start_token_id",
        ):
            if (
                not isinstance(getattr(self, name), int)
                or not 0 <= getattr(self, name) < self.vocab_size
            ):
                raise ValueError(
                    f"LFM3 Audio {name}={getattr(self, name)} must be in "
                    f"[0, vocab_size={self.vocab_size}); use the matching audio tokenizer."
                )

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Lfm3AudioConfig:
        source = _namespace(parent_config if parent_config is not None else config)
        text = _namespace(_field(source, "text_config", "lfm", config))
        encoder = _namespace(_field(source, "audio_encoder_config", "encoder"))
        depth = _namespace(_field(source, "depth_config", "depthformer"))
        if encoder is None or depth is None:
            raise ValueError(
                "LFM3 Audio config is missing its encoder or depth decoder; "
                "load the composite checkpoint config, not its text sub-config."
            )
        fields = _shallow_fields(Lfm3Config.from_transformers(text))
        codebooks = _field(source, "num_codebooks", "codebooks", 12)
        vocab = _field(source, "codebook_size", "audio_vocab_size", 2017)
        schedule = getattr(source, "maskgit_schedule", None)
        if schedule is None:
            steps = getattr(source, "maskgit_steps", 3)
            if not 1 <= steps <= codebooks:
                raise ValueError(
                    f"LFM3 Audio maskgit_steps={steps} must be in [1, {codebooks}]."
                )
            schedule = [codebooks // steps] * steps
            schedule[-1] += codebooks - sum(schedule)
        fields.update(
            model_type="lfm3_audio",
            encoder_config=Lfm2Config.from_transformers(
                _namespace(_field(encoder, "text_config", "lfm"))
            ),
            depth_config=Lfm3Config.from_transformers(depth),
            num_mel_bins=_field(encoder, "num_mel_bins", "dim_in", 128),
            conv_channels=_field(encoder, "conv_channels", "dim_conv", 256),
            encoder_sliding_window=getattr(encoder, "sliding_window", 32),
            num_codebooks=codebooks,
            codebook_size=vocab,
            audio_eos_code=getattr(source, "audio_eos_code", vocab - 1),
            maskgit_schedule=tuple(schedule),
        )
        for name in (
            "audio_token_id",
            "audio_out_token_id",
            "audio_start_token_id",
            "text_end_token_id",
            "mixed_start_token_id",
            "interleaved_n_text",
            "interleaved_n_audio",
        ):
            fields[name] = getattr(source, name, cls.__dataclass_fields__[name].default)
        return cls(**fields)
