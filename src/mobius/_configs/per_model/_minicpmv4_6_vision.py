# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MiniCPM-V-4.6 vision extractor hook."""

from __future__ import annotations

from mobius._configs._extractors import register_vision_hook


@register_vision_hook("qwen3_5_text")
def _minicpmv4_6_vision(config, parent_config, model_type: str, fields: dict):
    """Extract the packed SigLIP2 geometry from the composite MiniCPM config."""
    if getattr(parent_config, "model_type", None) != "minicpmv4_6":
        return None

    vision = parent_config.vision_config
    image_size = getattr(vision, "image_size", 980)
    patch_size = getattr(vision, "patch_size", 14)
    fields.update(
        model_type="minicpmv4_6_vision",
        hidden_size=vision.hidden_size,
        intermediate_size=vision.intermediate_size,
        num_hidden_layers=vision.num_hidden_layers,
        num_attention_heads=vision.num_attention_heads,
        image_size=image_size,
        patch_size=patch_size,
        norm_eps=vision.layer_norm_eps,
        hidden_act=getattr(vision, "hidden_act", "gelu_pytorch_tanh"),
        in_channels=getattr(vision, "num_channels", 3),
        num_position_embeddings=(image_size // patch_size) ** 2,
        image_token_id=parent_config.image_token_id,
        insert_layer_id=getattr(parent_config, "insert_layer_id", 6),
        window_kernel_size=tuple(getattr(vision, "window_kernel_size", (2, 2))),
        merge_kernel_size=tuple(getattr(parent_config, "merge_kernel_size", (2, 2))),
        merger_times=getattr(parent_config, "merger_times", 1),
    )
    return None
