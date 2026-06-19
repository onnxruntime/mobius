# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model configuration dataclasses and HuggingFace → mobius conversion.

This package re-exports the public surface that used to live in
``mobius._configs`` (a single 2.6k-line module). The split keeps that
import path working while letting each concern grow in its own file:

* :mod:`._sub_configs` — pure-data dataclasses for sub-configs
  (RoPE, vision, audio, codec, TTS) shared across model families.
* :mod:`._quantization` — :class:`QuantizationConfig` and its HF parser.
* :mod:`._base` — :class:`BaseModelConfig`, :class:`ArchitectureConfig`,
  the per-model config subclasses, and the legacy ``_extract_*``
  helpers that build sub-configs from HuggingFace fields.

Follow-up PRs in this series convert the model-type switches inside the
``_extract_*`` helpers into a registry-based dispatch, and move per-model
config subclasses into their own files under ``per_model/``.
"""

from __future__ import annotations

from mobius._configs._base import (
    DEFAULT_INT,
    ArchitectureConfig,
    BambaConfig,
    BaseModelConfig,
    CausalLMConfig,
    DepthAnythingConfig,
    DFlashConfig,
    EncoderConfig,
    Gemma2Config,
    Gemma3nConfig,
    Gemma4AssistantConfig,
    Gemma4Config,
    GraniteMoeHybridConfig,
    JambaConfig,
    JetMoeConfig,
    LongcatFlashConfig,
    Mamba2Config,
    MambaConfig,
    MllamaConfig,
    MMSConfig,
    NanoChatConfig,
    NemotronHConfig,
    Qwen35MtpConfig,
    Sam2Config,
    SegformerConfig,
    VisionLanguageConfig,
    WhisperConfig,
    YolosConfig,
    Zamba2Config,
    _as_int,
    _extract_audio_config,
    _extract_mrope_fields,
    _extract_rope_config,
    _extract_vision_config,
    _first,
    _first_not_none,
    _nested_rope_theta,
    _nested_rope_type,
    _normalize_rope_scaling,
    _resolve_dtype,
    _resolve_hidden_act,
    _shallow_fields,
    _shared_expert_size,
)
from mobius._configs._quantization import QuantizationConfig
from mobius._configs._sub_configs import (
    AudioConfig,
    CodecDecoderConfig,
    CodecEncoderConfig,
    CodePredictorConfig,
    Gemma4AudioConfig,
    RoPEConfig,
    SpeakerEncoderConfig,
    TTSConfig,
    VisionConfig,
)

__all__ = [
    "DEFAULT_INT",
    "ArchitectureConfig",
    "AudioConfig",
    "BambaConfig",
    "BaseModelConfig",
    "CausalLMConfig",
    "CodePredictorConfig",
    "CodecDecoderConfig",
    "CodecEncoderConfig",
    "DepthAnythingConfig",
    "EncoderConfig",
    "Gemma2Config",
    "Gemma3nConfig",
    "Gemma4AudioConfig",
    "Gemma4Config",
    "GraniteMoeHybridConfig",
    "JambaConfig",
    "JetMoeConfig",
    "LongcatFlashConfig",
    "Mamba2Config",
    "MambaConfig",
    "MllamaConfig",
    "MMSConfig",
    "NanoChatConfig",
    "NemotronHConfig",
    "QuantizationConfig",
    "Qwen35MtpConfig",
    "RoPEConfig",
    "Sam2Config",
    "SegformerConfig",
    "SpeakerEncoderConfig",
    "TTSConfig",
    "VisionConfig",
    "VisionLanguageConfig",
    "WhisperConfig",
    "YolosConfig",
    "Zamba2Config",
    "_as_int",
    "_extract_audio_config",
    "_extract_mrope_fields",
    "_extract_rope_config",
    "_extract_vision_config",
    "_first",
    "_first_not_none",
    "_nested_rope_theta",
    "_nested_rope_type",
    "_normalize_rope_scaling",
    "_resolve_dtype",
    "_resolve_hidden_act",
    "_shallow_fields",
    "_shared_expert_size",
    "DFlashConfig",
    "Gemma4AssistantConfig",
]
