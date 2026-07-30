# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Default audio-extraction first pass shared across many multimodal models.

Covers the generic ``audio_processor`` / ``embd_layer`` / ``speech_lora``
fields that the original :func:`_extract_audio_config` always inspected,
regardless of model_type. Called explicitly by
:func:`mobius._configs._extractors.extract_audio_config` as the first
step of the pipeline, before any per-model hook runs.
"""

from __future__ import annotations


def apply_audio_defaults(config, parent_config, model_type: str, fields: dict) -> None:
    """Populate generic HF audio-config fields.

    This is the "first pass" of the audio-extraction pipeline: it always
    runs before any per-model hook registered via
    :func:`register_audio_hook`. Per-model hooks may freely overwrite
    fields produced here.

    Called explicitly by :func:`extract_audio_config` rather than being
    registered as a hook — keeps the architecture self-evident
    (defaults are a builtin first pass, hooks are overrides) and removes
    any dependence on hook-registration order.
    """
    # audio_processor dict (Phi4MM-style nested config)
    ap = getattr(config, "audio_processor", None)
    if isinstance(ap, dict) and "config" in ap:
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

    # projection_hidden_size: derived from top-level hidden_size when an
    # embd_layer is present (Phi4MM convention).
    embd_layer = getattr(config, "embd_layer", None)
    if isinstance(embd_layer, dict):
        fields["projection_hidden_size"] = config.hidden_size

    # speech_lora adapter dict (Phi4-MM).
    speech_lora = getattr(config, "speech_lora", None)
    if speech_lora is not None:
        fields["lora"] = speech_lora if isinstance(speech_lora, dict) else vars(speech_lora)
