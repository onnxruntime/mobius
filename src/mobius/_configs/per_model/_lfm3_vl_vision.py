# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LiquidAI LFM3-VL vision extractor hook (gated conv / attention tower)."""

from __future__ import annotations

from mobius._configs._extractors import register_vision_hook


@register_vision_hook("lfm3")
def _lfm3_vl_vision(config, parent_config, model_type: str, fields: dict):
    """Read the axial 2D RoPE theta, which HF nests under ``rope_parameters``."""
    if fields.get("model_type") != "lfm3_vl_vision":
        return None
    vision = (parent_config or config).vision_config
    fields["rope_theta"] = vision.rope_parameters["rope_theta"]
    return None
