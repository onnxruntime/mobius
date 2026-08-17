# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vision extractor hook for NVIDIA Cosmos3-Edge.

Cosmos3-Edge (``cosmos3_edge`` / ``Cosmos3EdgeForConditionalGeneration``) pairs
a **variable-resolution** SigLIP2 vision tower (``cosmos3_edge_vision``) with a
pixel-shuffle merger projector (``cosmos3_edge_projector``). Several config
quirks need bridging into :class:`~mobius._configs.VisionConfig`:

- The vision config declares ``num_patches`` (256), the size of the *learned
  position-embedding reference grid*, not a fixed input resolution. The tower
  resamples that 16x16 grid to each image's own patch grid, so ``image_size``
  is only a nominal value (``sqrt(num_patches) * patch_size``) retained for
  metadata — it does not constrain the exported graph.
- The projector's intermediate width and norm placement live in a sibling
  ``projector_config`` (``merger_intermediate_size``,
  ``use_postshuffle_norm``), not in ``vision_config``.
- Cosmos3-Edge only supports ``temporal_patch_size=1`` (each video frame is an
  independent patch run), unlike the Qwen-VL default of 2.
- The text decoder uses **interleaved** 3D M-RoPE (T, H, W, T, H, W, ...)
  rather than the contiguous ``mrope_section`` chunking used by Qwen-VL. HF
  reference: ``Cosmos3EdgeTextRotaryEmbedding.compute_default_rope_parameters``.
"""

from __future__ import annotations

import math

from mobius._configs._extractors import register_vision_hook

_DEFAULT_IMAGE_TOKEN_ID = 19
_DEFAULT_VIDEO_TOKEN_ID = 18
_DEFAULT_VISION_START_TOKEN_ID = 20
_DEFAULT_VISION_END_TOKEN_ID = 21


@register_vision_hook("cosmos3_edge", "cosmos3_edge_text", "cosmos3_edge_vision")
def _cosmos3_edge_vision(config, parent_config, model_type: str, fields: dict):
    vision_source = parent_config or config
    hf_vision = getattr(vision_source, "vision_config", None) or getattr(
        config, "vision_config", None
    )

    if hf_vision is not None:
        num_patches = getattr(hf_vision, "num_patches", None)
        patch_size = getattr(hf_vision, "patch_size", None) or fields.get("patch_size")
        if num_patches is not None:
            grid = math.isqrt(num_patches)
            if grid * grid != num_patches:
                raise ValueError(
                    "Cosmos3-Edge vision num_patches must form a square grid, "
                    f"got {num_patches}"
                )
            fields["num_patches"] = num_patches
            # Nominal resolution of the learned position grid (16 * 16 = 256).
            if fields.get("image_size") is None and patch_size is not None:
                fields["image_size"] = grid * patch_size
        fields["hidden_act"] = getattr(hf_vision, "hidden_act", None)

    # Cosmos3-Edge patchifies one frame at a time (``temporal_patch_size=1``).
    fields["temporal_patch_size"] = 1

    # Pixel-shuffle projector settings from projector_config.
    projector_cfg = getattr(vision_source, "projector_config", None) or getattr(
        config, "projector_config", None
    )
    if projector_cfg is not None:
        get = (
            projector_cfg.get
            if isinstance(projector_cfg, dict)
            else lambda k, d=None: getattr(projector_cfg, k, d)
        )
        fields["projector_intermediate_size"] = get("merger_intermediate_size")
        fields["use_postshuffle_norm"] = bool(get("use_postshuffle_norm", False))
        # out_hidden_size drives the projector output (text hidden size).
        if fields.get("out_hidden_size") is None:
            fields["out_hidden_size"] = get("out_hidden_size")
        merge = get("spatial_merge_size")
        if merge is not None:
            fields["spatial_merge_size"] = merge

    # Cosmos3-Edge places the multimodal token ids at the top level.
    if fields.get("image_token_id") is None:
        fields["image_token_id"] = getattr(
            vision_source, "image_token_id", _DEFAULT_IMAGE_TOKEN_ID
        )
    fields["video_token_id"] = getattr(
        vision_source, "video_token_id", _DEFAULT_VIDEO_TOKEN_ID
    )
    fields["vision_start_token_id"] = getattr(
        vision_source, "vision_start_token_id", _DEFAULT_VISION_START_TOKEN_ID
    )
    fields["vision_end_token_id"] = getattr(
        vision_source, "vision_end_token_id", _DEFAULT_VISION_END_TOKEN_ID
    )

    # Interleaved 3D M-RoPE — the Edge checkpoint stores ``mrope_section`` under
    # ``rope_parameters`` without an explicit interleave flag, but
    # ``Cosmos3EdgeTextRotaryEmbedding`` builds its axis masks from ``i % 3``.
    fields["mrope_interleaved"] = True
    return None
