# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma 3n vision extractor hook.

Gemma 3n's vision tower is a timm **MobileNet-V5-300m** encoder, not a SigLIP
transformer, so its HF ``vision_config`` carries almost none of the fields the
generic extractor looks for — no ``patch_size``, ``num_hidden_layers``, or
``num_attention_heads``.  What it does carry:

- ``architecture``: the timm spec name (``"mobilenetv5_300m_enc"``), which is
  the only handle on the block layout (there is no per-layer config).
- ``do_pooling``: ``False`` for Gemma 3n — the decoder needs the 16x16 spatial
  feature map, not a pooled vector.
- ``vocab_offset`` / ``vocab_size``: the image soft-token id range
  ([262144, 262272) for E4B) used by ``Gemma3nMultimodalEmbedder``.

``image_size`` is absent from the config as well: MobileNet-V5 has no
dynamic-resolution path, and the checkpoint's ``SiglipImageProcessorFast``
resizes to a fixed 768x768.  That resolution is pinned here so the exported
graph's ``pixel_values`` shape and the generated preprocessing pipeline agree.
"""

from __future__ import annotations

from mobius._configs._extractors import register_vision_hook

_GEMMA3N_TYPES = ("gemma3n", "gemma3n_text", "gemma3n_vision")

# MobileNet-V5-300m consumes a fixed 768x768 image (SiglipImageProcessorFast
# ``size``), yielding a 16x16 grid = 256 soft tokens.
_GEMMA3N_IMAGE_SIZE = 768


@register_vision_hook
def _gemma3n_vision(config, parent_config, model_type: str, fields: dict):
    # No decorator filter: this hook must also fire when build() has resolved
    # to the text sub-config, whose model_type is "gemma3n_text" while the
    # parent is "gemma3n". The body's predicate covers both cases.
    #
    # Read the parent's model_type off ``parent_config`` rather than the
    # ``parent_config or config`` composite: falling back to *config* would
    # make the hook fire for any dispatched model_type whenever the config
    # itself happens to be a gemma3n one.
    parent_model_type = getattr(parent_config, "model_type", "") if parent_config else ""
    if model_type not in _GEMMA3N_TYPES and parent_model_type != "gemma3n":
        return None
    composite = parent_config or config
    hf_vision = getattr(composite, "vision_config", None)
    if hf_vision is None:
        return None

    def _get(name, default=None):
        return getattr(hf_vision, name, default)

    rms_norm_eps = _get("rms_norm_eps", 1e-6)
    fields.update(
        model_type="gemma3n_vision",
        architecture=_get("architecture", "mobilenetv5_300m_enc"),
        hidden_size=_get("hidden_size", 2048),
        image_size=_GEMMA3N_IMAGE_SIZE,
        do_pooling=bool(_get("do_pooling", False)),
        vocab_offset=_get("vocab_offset"),
        vocab_size=_get("vocab_size"),
        rms_norm_eps=rms_norm_eps,
        norm_eps=rms_norm_eps,
    )
    fields["image_token_id"] = getattr(composite, "image_token_id", None)
    fields["mm_tokens_per_image"] = getattr(composite, "vision_soft_tokens_per_image", None)
    return None
