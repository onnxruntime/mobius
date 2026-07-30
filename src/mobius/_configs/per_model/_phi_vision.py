# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vision extractor hooks for Phi vision-language models."""

from __future__ import annotations

from mobius._configs._extractors import register_vision_hook


@register_vision_hook("phi3_v")
def _phi3_v_vision(config, parent_config, model_type: str, fields: dict):
    """Extract Phi-3 Vision's CLIP encoder fields from ``img_processor``.

    Phi-3/3.5-Vision store the CLIP ViT-L/14-336 geometry in a non-standard
    ``img_processor`` dict rather than a standard ``vision_config``. The encoder
    itself is the fixed openai/clip-vit-large-patch14-336 architecture, so the
    geometry is constant; only ``image_dim_out`` and the feature ``layer_idx``
    are read from the checkpoint.
    """
    vision_source = parent_config or config
    img_processor = getattr(vision_source, "img_processor", None)
    if not isinstance(img_processor, dict):
        img_processor = {}
    image_dim_out = img_processor.get("image_dim_out", 1024)
    fields.update(
        hidden_size=image_dim_out,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        image_size=336,
        patch_size=14,
        norm_eps=1e-5,
        hidden_act="quick_gelu",
        # HuggingFace Phi3V selects hidden_states[layer_idx] (default -2) and
        # keeps only the patch tokens (type_feature="patch").
        feature_layer=img_processor.get("layer_idx", -2),
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
