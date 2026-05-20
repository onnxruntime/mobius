# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SenseVoice (Fun-ASR family) audio extractor hook."""

from __future__ import annotations

from mobius._configs._extractors import register_audio_hook


@register_audio_hook
def _sensevoice_audio(config, parent_config, model_type: str, fields: dict):
    """Map SenseVoice ``encoder_conf`` + ``frontend_conf`` to :class:`AudioConfig`."""
    if model_type != "sensevoice":
        return None
    encoder_conf = getattr(config, "encoder_conf", None) or {}
    frontend_conf = getattr(config, "frontend_conf", None) or {}
    if not isinstance(encoder_conf, dict) or not encoder_conf:
        return None
    fields.update(
        attention_dim=encoder_conf.get("output_size"),
        attention_heads=encoder_conf.get("attention_heads"),
        num_blocks=encoder_conf.get("num_blocks"),
        tp_num_blocks=encoder_conf.get("tp_blocks"),
        linear_units=encoder_conf.get("linear_units"),
        kernel_size=encoder_conf.get("kernel_size"),
        input_size=getattr(config, "input_size", None),
    )
    if isinstance(frontend_conf, dict):
        fields["num_mel_bins"] = frontend_conf.get("n_mels")
    return None
