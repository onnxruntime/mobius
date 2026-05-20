# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HunYuan VL-MoT vision extractor hook.

The HF config.json is flat and doesn't include a vision_config sub-object.
The vision encoder params are hardcoded for an InternViT-style ViT.
"""

from __future__ import annotations

from mobius._configs._extractors import register_vision_hook


@register_vision_hook
def _hunyuan_vl_mot_vision(config, parent_config, model_type: str, fields: dict):
    if model_type != "hunyuan_vl_mot":
        return None
    if fields.get("hidden_size"):
        return None
    vision_source = parent_config or config
    fields.update(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=2048,
        patch_size=16,
        norm_eps=1e-6,
        spatial_merge_size=2,
        image_token_id=getattr(vision_source, "mask_init_id", 12),
    )
    return None
