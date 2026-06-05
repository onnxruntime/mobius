# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma4-unified (gemma-4-12B) audio extractor hook.

The ``gemma4_unified`` audio config describes an *encoder-free* embedder (no
Conformer tower).  It exposes only ``audio_embed_dim`` (input feature size for
the projection) and ``rms_norm_eps``.  This hook maps those onto
:class:`Gemma4AudioConfig` so
:class:`~mobius.models.gemma4._Gemma4UnifiedAudioEmbedderModel` can read them.
"""

from __future__ import annotations

from mobius._configs._extractors import register_audio_hook
from mobius._configs._sub_configs import Gemma4AudioConfig

_UNIFIED_TYPES = ("gemma4_unified", "gemma4_unified_text", "gemma4_unified_audio")


@register_audio_hook
def _gemma4_unified_audio(config, parent_config, model_type: str, fields: dict):
    composite = parent_config or config
    parent_model_type = getattr(composite, "model_type", "")
    if model_type not in _UNIFIED_TYPES and parent_model_type != "gemma4_unified":
        return None
    hf_audio = getattr(composite, "audio_config", None)
    if hf_audio is None:
        return None
    audio_embed_dim = getattr(hf_audio, "audio_embed_dim", 640)
    return {
        "audio": Gemma4AudioConfig(
            hidden_size=audio_embed_dim,
            output_proj_dims=audio_embed_dim,
            audio_token_id=getattr(composite, "audio_token_id", None),
        )
    }
