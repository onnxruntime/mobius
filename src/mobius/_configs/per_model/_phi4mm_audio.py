# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Phi4MM audio extractor hook."""

from __future__ import annotations

from mobius._configs._extractors import register_audio_hook


@register_audio_hook
def _phi4mm_audio_token_id(config, parent_config, model_type: str, fields: dict):
    if model_type != "phi4mm":
        return None
    audio_config_dict = getattr(config, "audio_config", None)
    if audio_config_dict is None:
        return None
    ac_dict = (
        audio_config_dict if isinstance(audio_config_dict, dict) else vars(audio_config_dict)
    )
    fields["token_id"] = ac_dict.get("audio_token_id")
    return None
