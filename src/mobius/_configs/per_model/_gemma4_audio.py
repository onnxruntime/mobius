# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma4 audio extractor hook (uses :class:`Gemma4AudioConfig` subclass)."""

from __future__ import annotations

from mobius._configs._extractors import register_audio_hook
from mobius._configs._sub_configs import Gemma4AudioConfig


@register_audio_hook
def _gemma4_audio(config, parent_config, model_type: str, fields: dict):
    # No decorator filter: this hook also needs to fire when the *parent* is
    # gemma4 (e.g. build() has resolved to a text sub-config whose model_type
    # is no longer "gemma4*"). The body's predicate covers both cases.
    parent_model_type = getattr(parent_config, "model_type", "") if parent_config else ""
    if model_type not in ("gemma4", "gemma4_text") and parent_model_type != "gemma4":
        return None
    composite = parent_config or config
    hf_audio_config = getattr(composite, "audio_config", None)
    if hf_audio_config is None:
        return None
    ac = (
        hf_audio_config
        if not isinstance(hf_audio_config, dict)
        else type("AC", (), hf_audio_config)()
    )
    subsampling = getattr(ac, "subsampling_conv_channels", None)
    return {
        "audio": Gemma4AudioConfig(
            num_layers=getattr(ac, "num_hidden_layers", 12),
            hidden_size=getattr(ac, "hidden_size", 1024),
            subsampling_conv_channels=(list(subsampling) if subsampling is not None else None),
            use_causal_chunked_attn=getattr(ac, "use_causal_chunked_attn", False),
            output_dim=getattr(ac, "output_dim", None),
            output_proj_dims=getattr(ac, "output_proj_dims", None),
            audio_token_id=getattr(composite, "audio_token_id", None),
        )
    }
