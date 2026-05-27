# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Default audio-extraction hooks shared across many multimodal models.

Covers the generic ``audio_processor`` / ``embd_layer`` / ``speech_lora``
fields that the original :func:`_extract_audio_config` always inspected,
regardless of model_type.
"""

from __future__ import annotations

from mobius._configs._extractors import DEFAULT_PRIORITY, register_audio_hook


@register_audio_hook(priority=DEFAULT_PRIORITY)
def _audio_processor(config, parent_config, model_type: str, fields: dict):
    """Pull encoder dims from an ``audio_processor`` dict if present."""
    ap = getattr(config, "audio_processor", None)
    if not isinstance(ap, dict) or "config" not in ap:
        return None
    ac = ap["config"]
    nemo = ac.get("nemo_conv_settings", {})
    rel_bias = ac.get("relative_attention_bias_args", {})
    fields.update(
        attention_dim=ac.get("attention_dim"),
        attention_heads=ac.get("attention_heads"),
        num_blocks=ac.get("num_blocks"),
        linear_units=ac.get("linear_units"),
        kernel_size=ac.get("kernel_size"),
        input_size=ac.get("input_size"),
        conv_channels=nemo.get("conv_channels", ac.get("attention_dim")),
        t5_bias_max_distance=rel_bias.get("t5_bias_max_distance"),
    )
    return None


@register_audio_hook(priority=DEFAULT_PRIORITY)
def _projection_hidden_size(config, parent_config, model_type: str, fields: dict):
    """Pull projection_hidden_size from an ``embd_layer`` dict if present."""
    embd_layer = getattr(config, "embd_layer", None)
    if isinstance(embd_layer, dict):
        fields["projection_hidden_size"] = config.hidden_size
    return None


@register_audio_hook(priority=DEFAULT_PRIORITY)
def _speech_lora(config, parent_config, model_type: str, fields: dict):
    """Pull a ``speech_lora`` adapter dict if present."""
    speech_lora = getattr(config, "speech_lora", None)
    if speech_lora is not None:
        fields["lora"] = speech_lora if isinstance(speech_lora, dict) else vars(speech_lora)
    return None
