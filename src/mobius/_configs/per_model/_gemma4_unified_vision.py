# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma4-unified (gemma-4-12B) vision extractor hook.

The ``gemma4_unified`` vision config describes an *encoder-free* embedder, not
a SigLIP tower.  Its fields differ from the generic ``vision_config``:

- ``patch_size`` / ``pooling_kernel_size`` → merged ``model_patch_size``
- ``mm_embed_dim`` → embedder hidden size (``VisionConfig.hidden_size``)
- ``mm_posemb_size`` → factorized positional-embedding table size
  (``VisionConfig.position_embedding_size``)
- ``output_proj_dims`` → projection input dim (``VisionConfig.out_hidden_size``)

This hook maps those onto :class:`VisionConfig` so
:class:`~mobius.models.gemma4._Gemma4UnifiedVisionEmbedderModel` can read them.
"""

from __future__ import annotations

from mobius._configs._extractors import register_vision_hook

_UNIFIED_TYPES = ("gemma4_unified", "gemma4_unified_text", "gemma4_unified_vision")


@register_vision_hook
def _gemma4_unified_vision(config, parent_config, model_type: str, fields: dict):
    composite = parent_config or config
    parent_model_type = getattr(composite, "model_type", "")
    if model_type not in _UNIFIED_TYPES and parent_model_type != "gemma4_unified":
        return None
    hf_vision = getattr(composite, "vision_config", None)
    if hf_vision is None:
        return None

    def _get(name, default=None):
        return getattr(hf_vision, name, default)

    fields.update(
        model_type="gemma4_unified_vision",
        hidden_size=_get("mm_embed_dim", 3840),
        patch_size=_get("patch_size", 16),
        pooling_kernel_size=_get("pooling_kernel_size", 3),
        position_embedding_size=_get("mm_posemb_size", 1120),
        out_hidden_size=_get("output_proj_dims", _get("mm_embed_dim", 3840)),
        norm_eps=_get("rms_norm_eps", 1e-6),
    )
    fields["image_token_id"] = getattr(composite, "image_token_id", None)
    return None
