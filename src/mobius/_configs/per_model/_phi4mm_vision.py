# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Phi4MM vision extractor hook.

The SigLIP vision encoder params are baked into HF model code, not into
the config JSON, so we hard-code them here.

TODO: Move these to a config subclass override once the per-model config
classes have migrated to this package.
"""

from __future__ import annotations

from mobius._configs._extractors import register_vision_hook


@register_vision_hook("phi4mm")
def _phi4mm_vision(config, parent_config, model_type: str, fields: dict):
    fields.update(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=(fields.get("image_crop_size") or 448),
        patch_size=14,
        norm_eps=1e-6,
        image_token_id=getattr(config, "special_image_token_id", 200010),
    )
    return None
