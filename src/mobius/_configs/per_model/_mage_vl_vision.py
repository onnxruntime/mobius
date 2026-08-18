# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from mobius._configs._extractors import register_vision_hook


@register_vision_hook
def _mage_vl_vision(config, parent_config, model_type: str, fields: dict):
    """Extract the custom Mage-ViT configuration from a Mage-VL parent."""
    del config, model_type
    if parent_config is None or getattr(parent_config, "model_type", None) != "mage_vl":
        return None

    vision = getattr(parent_config, "vision_config", None)
    if vision is None:
        return None

    fields.update(
        model_type=getattr(vision, "model_type", "mage_vl_vision"),
        hidden_size=getattr(vision, "hidden_size", 1024),
        intermediate_size=getattr(vision, "intermediate_size", 4096),
        num_hidden_layers=getattr(vision, "num_hidden_layers", 24),
        num_attention_heads=getattr(vision, "num_attention_heads", 16),
        image_size=getattr(vision, "image_size", 448),
        patch_size=getattr(vision, "patch_size", 16),
        in_channels=getattr(vision, "num_channels", 3),
        out_hidden_size=getattr(vision, "out_hidden_size", 2560),
        spatial_merge_size=getattr(vision, "spatial_merge_size", 2),
        temporal_patch_size=getattr(vision, "temporal_patch_size", 1),
        frame_windows_size=getattr(vision, "frame_windows_size", 4),
        norm_eps=getattr(vision, "layer_norm_eps", 1e-6),
        rope_theta=getattr(vision, "rope_theta", 10_000.0),
        hidden_act=getattr(vision, "hidden_act", "gelu"),
        image_token_id=getattr(parent_config, "image_token_id", None),
        video_token_id=getattr(parent_config, "video_token_id", None),
        vision_start_token_id=getattr(parent_config, "vision_start_token_id", None),
        vision_end_token_id=getattr(parent_config, "vision_end_token_id", None),
        tokens_per_second=getattr(parent_config, "tokens_per_second", 1.0),
    )
    return None
