# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LiquidAI LFM2-VL vision extractor hook (SigLIP2 NaFlex tower)."""

from __future__ import annotations

import math

from mobius._configs._extractors import register_vision_hook

# SigLIP2 NaFlex checkpoints omit ``image_size`` because the tower accepts any
# aspect ratio. The learned position table is still a square grid, so the
# native training resolution is ``sqrt(num_patches) * patch_size``.
_DEFAULT_NUM_PATCHES = 256


@register_vision_hook("lfm2")
def _lfm2_vl_vision(config, parent_config, model_type: str, fields: dict):
    """Add the NaFlex-specific geometry the generic extractor cannot infer."""
    if getattr(parent_config, "model_type", None) != "lfm2_vl":
        return None

    vision = getattr(parent_config, "vision_config", None)
    if vision is None:
        return None

    num_patches = getattr(vision, "num_patches", None) or _DEFAULT_NUM_PATCHES
    patch_size = getattr(vision, "patch_size", 16)
    fields.update(
        # HF stores the position-table size as ``num_patches``; mobius keeps
        # learned 2D position grids under ``num_position_embeddings``.
        num_position_embeddings=num_patches,
        hidden_act=getattr(vision, "hidden_act", "gelu_pytorch_tanh"),
        image_size=patch_size * math.isqrt(num_patches),
        image_token_id=getattr(parent_config, "image_token_id", None),
    )
    return None
