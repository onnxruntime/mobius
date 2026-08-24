# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GLM-ASR audio encoder config extraction."""

from __future__ import annotations

from mobius._configs._extractors import register_audio_hook


@register_audio_hook
def _glmasr_audio(config, parent_config, model_type: str, fields: dict):
    composite = parent_config or config
    if getattr(composite, "model_type", None) != "glmasr":
        return None

    audio_config = getattr(composite, "audio_config", None)
    if audio_config is None:
        return None
    if isinstance(audio_config, dict):
        audio_config = type("GlmAsrAudioConfig", (), audio_config)()

    rope_parameters = getattr(audio_config, "rope_parameters", None) or {}
    fields.update(
        d_model=getattr(audio_config, "hidden_size", None),
        encoder_layers=getattr(audio_config, "num_hidden_layers", None),
        encoder_attention_heads=getattr(audio_config, "num_attention_heads", None),
        encoder_num_key_value_heads=getattr(
            audio_config,
            "num_key_value_heads",
            getattr(audio_config, "num_attention_heads", None),
        ),
        encoder_head_dim=getattr(audio_config, "head_dim", None),
        encoder_ffn_dim=getattr(audio_config, "intermediate_size", None),
        encoder_partial_rotary_factor=getattr(
            audio_config,
            "partial_rotary_factor",
            rope_parameters.get("partial_rotary_factor", 0.5),
        ),
        encoder_rope_theta=rope_parameters.get("rope_theta", 10_000.0),
        encoder_layer_norm_eps=getattr(audio_config, "layer_norm_eps", 1e-5),
        num_mel_bins=getattr(audio_config, "num_mel_bins", None),
        max_source_positions=getattr(audio_config, "max_position_embeddings", None),
        activation_function=getattr(audio_config, "hidden_act", "gelu"),
        audio_token_id=getattr(composite, "audio_token_id", None),
        output_dim=getattr(getattr(composite, "text_config", None), "hidden_size", None),
    )
    return None
