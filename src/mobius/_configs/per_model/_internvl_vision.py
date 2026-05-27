# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""InternVL vision extractor hook.

InternVL2 doesn't expose ``image_token_id`` in its config — default to
the Qwen2 ``<IMG_CONTEXT>`` token id used by InternVL2-* models.
"""

from __future__ import annotations

from mobius._configs._extractors import register_vision_hook

_INTERNVL_TYPES = ("internvl_chat", "internvl2", "internvl")


@register_vision_hook
def _internvl_vision(config, parent_config, model_type: str, fields: dict):
    vision_source = parent_config or config
    parent_model_type = getattr(vision_source, "model_type", None)
    if parent_model_type not in _INTERNVL_TYPES and model_type not in _INTERNVL_TYPES:
        return None
    if fields.get("image_token_id") is not None:
        return None
    fields["image_token_id"] = getattr(vision_source, "img_context_token_id", 151667)
    return None
