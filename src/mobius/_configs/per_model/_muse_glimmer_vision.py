# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vision configuration extraction for Muse Glimmer."""

from __future__ import annotations

from mobius._configs._extractors import register_vision_hook


def _get(config, name: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


@register_vision_hook("muse_glimmer", "muse_glimmer_text", "muse_glimmer_vision")
def _muse_glimmer_vision(config, parent_config, model_type: str, fields: dict):
    composite = parent_config or config
    hf_vision = _get(composite, "vision_config") or _get(config, "vision_config")
    if hf_vision is None:
        return None

    position_height = _get(hf_vision, "pos_emb_height")
    position_width = _get(hf_vision, "pos_emb_width")
    patch_size = _get(hf_vision, "patch_size")
    layer_types = _get(hf_vision, "layer_types") or []
    hidden_size = _get(hf_vision, "hidden_size")
    num_attention_heads = _get(hf_vision, "num_attention_heads")

    fields.update(
        hidden_act=_get(hf_vision, "hidden_act"),
        head_dim=(
            _get(hf_vision, "head_dim")
            or (
                hidden_size // num_attention_heads
                if hidden_size is not None and num_attention_heads is not None
                else None
            )
        ),
        spatial_merge_size=_get(hf_vision, "merge_size", 2),
        temporal_patch_size=_get(hf_vision, "patch_temporal", 2),
        position_embedding_height=position_height,
        position_embedding_width=position_width,
        num_position_embeddings=(
            position_height * position_width
            if position_height is not None and position_width is not None
            else None
        ),
        fullatt_block_indexes=[
            layer_idx
            for layer_idx, layer_type in enumerate(layer_types)
            if layer_type == "full_attention"
        ],
        window_size=(
            position_height * patch_size
            if position_height is not None and patch_size is not None
            else None
        ),
        projector_intermediate_size=_get(composite, "projector_hidden_size"),
        out_hidden_size=_get(composite, "out_hidden_size"),
    )
    if (
        fields.get("image_size") is None
        and position_height is not None
        and patch_size is not None
    ):
        fields["image_size"] = position_height * patch_size
    return None
