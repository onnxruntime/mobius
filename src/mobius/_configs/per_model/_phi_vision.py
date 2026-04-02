# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vision extractor hooks for Phi vision-language models."""

from __future__ import annotations

from mobius._configs._extractors import register_vision_hook


@register_vision_hook("phi3_v")
def _phi3_v_vision(config, parent_config, model_type: str, fields: dict):
    """Extract Phi-3 Vision's CLIP encoder fields from ``img_processor``."""
    vision_source = parent_config or config
    img_processor = getattr(vision_source, "img_processor", None) or {}
    if isinstance(img_processor, dict):
        fields.update(
            hidden_size=img_processor.get("image_dim_out", 1024),
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            norm_eps=1e-5,
        )
    fields.setdefault("image_token_id", 32044)
    return None


@register_vision_hook("phi4-siglip")
def _phi4_siglip_vision(config, parent_config, model_type: str, fields: dict):
    """Supply the fixed SigLIP-2 image geometry omitted by Phi-4's config."""
    fields.setdefault("image_size", 384)
    fields.setdefault("patch_size", 16)
    fields.setdefault("image_token_id", -200)
    return None
