# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Default vision-extraction hook (HF ``vision_config`` + shared bits).

This hook handles every model whose HuggingFace config exposes a
``vision_config`` sub-object — the vast majority of multimodal models.
Per-model files in the same package can register additional hooks to
override or extend these defaults for specific architectures.
"""

from __future__ import annotations

from mobius._configs._base import _first, _first_not_none
from mobius._configs._extractors import DEFAULT_PRIORITY, register_vision_hook


@register_vision_hook(priority=DEFAULT_PRIORITY)
def _vision_default(config, parent_config, model_type: str, fields: dict):
    """Pull the canonical HF ``vision_config`` fields into ``fields``."""
    vision_source = parent_config or config
    hf_vision_config = getattr(vision_source, "vision_config", None)
    if hf_vision_config is None:
        hf_vision_config = getattr(config, "vision_config", None)

    if hf_vision_config is not None:
        vc = (
            hf_vision_config
            if not isinstance(hf_vision_config, dict)
            else type("VC", (), hf_vision_config)()
        )
        # Qwen2-VL uses ``embed_dim`` for the actual block hidden size and
        # ``hidden_size`` for the projection output. When ``embed_dim``
        # exists and ``out_hidden_size`` does not, remap so the block
        # dimension is used as ``hidden_size``.
        _embed_dim = getattr(vc, "embed_dim", None)
        _hf_hidden = getattr(vc, "hidden_size", None)
        _out_hidden = getattr(vc, "out_hidden_size", None)
        if _embed_dim is not None and _out_hidden is None and _embed_dim != _hf_hidden:
            _vis_hidden = _embed_dim
            _vis_out_hidden = _hf_hidden
        else:
            _vis_hidden = _hf_hidden
            _vis_out_hidden = _out_hidden

        _intermediate = getattr(vc, "intermediate_size", None)
        if _intermediate is None:
            _mlp_ratio = getattr(vc, "mlp_ratio", None)
            # _vis_hidden may be a list for multi-stage models (Florence2)
            _scalar_hidden = _first(_vis_hidden) if _vis_hidden is not None else None
            if _mlp_ratio is not None and _scalar_hidden is not None:
                _intermediate = int(_scalar_hidden * _mlp_ratio)

        fields.update(
            hidden_size=_vis_hidden,
            intermediate_size=_intermediate,
            num_hidden_layers=(
                getattr(vc, "num_hidden_layers", None) or getattr(vc, "depth", None)
            ),
            num_attention_heads=(
                getattr(vc, "num_attention_heads", None)
                or getattr(vc, "num_heads", None)
                or getattr(vc, "attention_heads", None)
            ),
            image_size=getattr(vc, "image_size", None),
            patch_size=getattr(vc, "patch_size", None),
            norm_eps=_first_not_none(
                getattr(vc, "layer_norm_eps", None),
                getattr(vc, "norm_eps", None),
                default=1e-6,
            ),
            model_type=getattr(vc, "model_type", None),
            head_dim=getattr(vc, "head_dim", None),
            rope_theta=getattr(vc, "rope_theta", None),
            out_hidden_size=_vis_out_hidden,
            in_channels=_first_not_none(
                getattr(vc, "in_channels", None),
                getattr(vc, "num_channels", None),
                default=3,
            ),
            spatial_merge_size=getattr(vc, "spatial_merge_size", 2),
            temporal_patch_size=getattr(vc, "temporal_patch_size", 2),
            num_position_embeddings=getattr(vc, "num_position_embeddings", None),
            deepstack_visual_indexes=getattr(vc, "deepstack_visual_indexes", None),
            fullatt_block_indexes=getattr(vc, "fullatt_block_indexes", None),
            window_size=getattr(vc, "window_size", None),
            use_clipped_linears=getattr(vc, "use_clipped_linears", False),
            position_embedding_size=getattr(vc, "position_embedding_size", None),
        )
    # Only fill in shared fields when not already populated. Per-model hooks
    # may run before this one (e.g. via direct registration order) — when
    # they do, their values must survive. Use setdefault semantics so
    # the default only supplies a value when nothing better is available.
    fields.setdefault(
        "mm_tokens_per_image", getattr(vision_source, "mm_tokens_per_image", None)
    )
    fields.setdefault("image_token_id", getattr(vision_source, "image_token_id", None))

    # MRoPE section — only for composite VL models (parent_config != config).
    if parent_config is not None and parent_config is not config:
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        mrope_section = rope_scaling.get("mrope_section", None) or rope_parameters.get(
            "mrope_section", None
        )
        if mrope_section is not None:
            fields["mrope_section"] = mrope_section

    # LoRA config (e.g. Phi4-MM)
    vision_lora = getattr(config, "vision_lora", None)
    if vision_lora is not None:
        fields["lora"] = vision_lora if isinstance(vision_lora, dict) else vars(vision_lora)

    # Phi4MM image embedding config
    embd_layer = getattr(config, "embd_layer", None)
    if isinstance(embd_layer, dict):
        img_cfg = embd_layer.get("image_embd_layer", {})
        fields["image_crop_size"] = img_cfg.get("crop_size")

    return None
