# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vision extractor hook for NVIDIA Cosmos3-Edge.

Cosmos3-Edge (``cosmos3_edge`` / ``Cosmos3EdgeForConditionalGeneration``) pairs
a SigLIP-style vision encoder (``cosmos3_edge_vision``) with a pixel-shuffle
merger projector (``cosmos3_edge_projector``). Two config quirks need bridging
into :class:`~mobius._configs.VisionConfig`:

- The vision config declares ``num_patches`` (256) instead of ``image_size``.
  The standard :class:`~mobius.components.PatchEmbedding` derives the patch
  count from ``image_size // patch_size``, so we reconstruct
  ``image_size = sqrt(num_patches) * patch_size`` (16 * 16 = 256).
- The projector's intermediate width lives in a sibling ``projector_config``
  (``merger_intermediate_size``), not in ``vision_config``.
"""

from __future__ import annotations

import math

from mobius._configs._extractors import register_vision_hook


@register_vision_hook("cosmos3_edge", "cosmos3_edge_vision")
def _cosmos3_edge_vision(config, parent_config, model_type: str, fields: dict):
    vision_source = parent_config or config
    hf_vision = getattr(vision_source, "vision_config", None) or getattr(
        config, "vision_config", None
    )

    # Reconstruct image_size from num_patches (256 -> 16x16 grid -> 256 px).
    if fields.get("image_size") is None and hf_vision is not None:
        num_patches = getattr(hf_vision, "num_patches", None)
        patch_size = getattr(hf_vision, "patch_size", None) or fields.get("patch_size")
        if num_patches is not None and patch_size is not None:
            grid = math.isqrt(num_patches)
            if grid * grid != num_patches:
                raise ValueError(
                    "Cosmos3-Edge vision num_patches must form a square grid, "
                    f"got {num_patches}"
                )
            fields["image_size"] = grid * patch_size

    # Pixel-shuffle projector intermediate size from projector_config.
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
        # out_hidden_size drives the projector output (text hidden size).
        if fields.get("out_hidden_size") is None:
            fields["out_hidden_size"] = get("out_hidden_size")
        merge = get("spatial_merge_size")
        if merge is not None:
            fields["spatial_merge_size"] = merge

    # Cosmos3-Edge places image_token_id at the top level (default 19).
    if fields.get("image_token_id") is None:
        fields["image_token_id"] = getattr(vision_source, "image_token_id", 19)
    return None
