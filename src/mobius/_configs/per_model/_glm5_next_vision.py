# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GLM-5.3 vision configuration extraction."""

from __future__ import annotations

from mobius._configs._extractors import register_vision_hook


@register_vision_hook("glm5_next", "glm5_next_text")
def _glm5_next_vision(config, parent_config, model_type: str, fields: dict) -> None:
    composite = parent_config or config
    vision = getattr(composite, "vision_config", None)
    if vision is None:
        return
    if isinstance(vision, dict):
        vision = type("Glm5NextVisionConfig", (), vision)()
    fields.update(
        hidden_size=getattr(vision, "hidden_size", None),
        intermediate_size=getattr(vision, "intermediate_size", None),
        num_hidden_layers=getattr(vision, "depth", None),
        num_attention_heads=getattr(vision, "num_heads", None),
        image_size=getattr(vision, "image_size", None),
        patch_size=getattr(vision, "patch_size", None),
        norm_eps=getattr(vision, "rms_norm_eps", 1e-5),
        model_type=getattr(vision, "model_type", "glm5_next_vision"),
        out_hidden_size=getattr(vision, "out_hidden_size", None),
        in_channels=getattr(vision, "in_channels", 3),
        spatial_merge_size=getattr(vision, "spatial_merge_size", 2),
        temporal_patch_size=getattr(vision, "temporal_patch_size", 2),
        hidden_act=getattr(vision, "hidden_act", "silu"),
        projector_intermediate_size=getattr(vision, "projection_intermediate_size", None),
        swiglu_limit=getattr(vision, "swiglu_limit", 10.0),
        image_token_id=getattr(composite, "image_token_id", 154854),
        video_token_id=getattr(composite, "video_token_id", 154855),
        vision_start_token_id=getattr(composite, "image_start_token_id", 154830),
        vision_end_token_id=getattr(composite, "image_end_token_id", 154831),
    )
