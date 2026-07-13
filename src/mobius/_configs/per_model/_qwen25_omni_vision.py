# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen2.5-Omni vision extractor (vision config lives under thinker_config)."""

from __future__ import annotations

from mobius._configs._extractors import register_vision_hook


@register_vision_hook("qwen2_5_omni_text")
def _qwen25_omni_vision(config, parent_config, model_type: str, fields: dict):
    thinker = getattr(parent_config, "thinker_config", None)
    if thinker is None:
        return None
    if isinstance(thinker, dict):
        thinker = type("ThinkerConfig", (), thinker)()
    vision = getattr(thinker, "vision_config", None)
    if vision is None:
        return None
    if isinstance(vision, dict):
        vision = type("VisionConfig", (), vision)()

    fields.update(
        hidden_size=getattr(vision, "hidden_size", None),
        intermediate_size=getattr(vision, "intermediate_size", None),
        num_hidden_layers=getattr(vision, "depth", None),
        num_attention_heads=getattr(vision, "num_heads", None),
        patch_size=getattr(vision, "patch_size", None),
        out_hidden_size=getattr(vision, "out_hidden_size", None),
        in_channels=getattr(vision, "in_channels", 3),
        spatial_merge_size=getattr(vision, "spatial_merge_size", 2),
        temporal_patch_size=getattr(vision, "temporal_patch_size", 2),
        fullatt_block_indexes=getattr(vision, "fullatt_block_indexes", None),
        window_size=getattr(vision, "window_size", 112),
        image_token_id=getattr(thinker, "image_token_id", None),
    )
    fields["video_token_id"] = getattr(thinker, "video_token_id", None)
    return None
