# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GGUF ``clip`` mmproj → HuggingFace tensor name mapping (Gemma4 vision/audio).

Gemma4's vision and audio encoders ship in a *companion* ``mmproj-*.gguf`` file
whose ``general.architecture`` is ``clip`` (not ``gemma4``).  Its tensors use a
different naming scheme than the text backbone:

- Vision tower tensors are prefixed ``v.`` (e.g. ``v.blk.0.attn_q.weight``,
  ``v.patch_embd.weight``, ``v.position_embd.weight``).
- Audio tower tensors are prefixed ``a.`` (e.g. ``a.blk.0.attn_q.weight``,
  ``a.conv1d.0.weight``, ``a.pre_encode.out.weight``).
- Cross-modal projectors are ``mm.input_projection.weight`` (vision→text) and
  ``mm.a.input_projection.weight`` (audio→text).

This module maps those names onto the **HuggingFace** names that
:meth:`mobius.models.gemma4.Gemma4Model.preprocess_weights` consumes
(``vision_tower.*`` / ``embed_vision.*`` / ``audio_tower.*`` / ``embed_audio.*``),
so the mmproj weights flow through the *same* tested preprocessing path as a
real HF Gemma4 checkpoint before being applied to the ONNX graph.

Companion activation-range tensors (``.input_max`` / ``.input_min`` /
``.output_max`` / ``.output_min``) that llama.cpp stores next to each weight are
**not** model weights and are skipped — see :func:`is_mmproj_stat_tensor`.

The name derivation was verified against ``unsloth/gemma-4-E2B-it-GGUF``'s
``mmproj-F16.gguf`` (``clip.vision.projector_type = gemma4v``,
``clip.audio.projector_type = gemma4a``) by round-tripping through
``preprocess_weights`` and checking every ONNX initializer receives a weight.
"""

from __future__ import annotations

__all__ = [
    "is_mmproj_stat_tensor",
    "map_mmproj_audio_to_hf",
    "map_mmproj_muse_glimmer_vision_to_hf",
    "map_mmproj_vision_to_hf",
]

import re

# ---------------------------------------------------------------------------
# Vision tower (v.* / mm.input_projection)
# ---------------------------------------------------------------------------
# Per-block stem → HF Gemma4VisionEncoderLayer stem. The mmproj SigLIP-style
# block has a 4-norm structure (ln1 = pre-attention, ln2 = pre-feedforward,
# attn_post_norm = post-attention, ffn_post_norm = post-feedforward), SwiGLU
# gate/up/down FFN, and per-head Q/K RMSNorms.
_VISION_BLOCK_STEMS: dict[str, str] = {
    "ln1.weight": "input_layernorm.weight",
    "ln2.weight": "pre_feedforward_layernorm.weight",
    "attn_post_norm.weight": "post_attention_layernorm.weight",
    "ffn_post_norm.weight": "post_feedforward_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_out.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

# ---------------------------------------------------------------------------
# Audio tower (a.* / mm.a.input_projection)
# ---------------------------------------------------------------------------
# Per-block stem → HF Gemma4 Conformer block stem. Mapped against the ONNX
# module parameter names of ``_Gemma4AudioEncoderModel`` (feed_forward1/2,
# self_attn, lconv1d, norm_*). NOTE: the Conformer forward-pass wiring for the
# mmproj audio tower is still being validated (see build_gemma4_vlm_from_gguf),
# so this mapping is provided for completeness and unit-tested for name shape
# but not yet applied to a built audio graph.
_AUDIO_BLOCK_STEMS: dict[str, str] = {
    "ffn_norm.weight": "feed_forward1.pre_layer_norm.weight",
    "ffn_up.weight": "feed_forward1.ffw_layer_1.weight",
    "ffn_down.weight": "feed_forward1.ffw_layer_2.weight",
    "ffn_post_norm.weight": "feed_forward1.post_layer_norm.weight",
    "ffn_norm_1.weight": "feed_forward2.pre_layer_norm.weight",
    "ffn_up_1.weight": "feed_forward2.ffw_layer_1.weight",
    "ffn_down_1.weight": "feed_forward2.ffw_layer_2.weight",
    "ffn_post_norm_1.weight": "feed_forward2.post_layer_norm.weight",
    "attn_pre_norm.weight": "norm_pre_attn.weight",
    "attn_post_norm.weight": "norm_post_attn.weight",
    "ln2.weight": "norm_out.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_out.weight": "self_attn.post.weight",
    "attn_k_rel.weight": "self_attn.relative_k_proj.weight",
    "per_dim_scale.weight": "self_attn.per_dim_scale",
    "norm_conv.weight": "lconv1d.pre_layer_norm.weight",
    "conv_pw1.weight": "lconv1d.linear_start.weight",
    "conv_dw.weight": "lconv1d.depthwise_conv1d.weight",
    "conv_norm.weight": "lconv1d.conv_norm.weight",
    "conv_pw2.weight": "lconv1d.linear_end.weight",
}

_VISION_BLK = re.compile(r"^v\.blk\.(\d+)\.(.+)$")
_AUDIO_BLK = re.compile(r"^a\.blk\.(\d+)\.(.+)$")

_STAT_SUFFIXES = (".input_max", ".input_min", ".output_max", ".output_min")


def is_mmproj_stat_tensor(name: str) -> bool:
    """Return ``True`` for llama.cpp activation-range stats (not weights).

    Each quantizable linear in the mmproj carries companion
    ``.input_max``/``.input_min``/``.output_max``/``.output_min`` scalar
    tensors describing the observed activation range. These are calibration
    statistics, not learnable parameters, and must be skipped.
    """
    return name.endswith(_STAT_SUFFIXES)


def map_mmproj_vision_to_hf(name: str) -> str | None:
    """Map a ``clip`` mmproj vision tensor name to its HF Gemma4 name.

    Returns the HuggingFace name consumed by
    :meth:`Gemma4Model.preprocess_weights` (``vision_tower.*`` /
    ``embed_vision.*``), or ``None`` if the tensor is skipped (activation-range
    stats or the audio tower).

    The mapping mirrors :mod:`_tensor_mapping` for the text backbone.
    """
    if is_mmproj_stat_tensor(name):
        return None

    blk = _VISION_BLK.match(name)
    if blk is not None:
        idx, stem = blk.group(1), blk.group(2)
        hf_stem = _VISION_BLOCK_STEMS.get(stem)
        if hf_stem is None:
            return None
        return f"vision_tower.encoder.layers.{idx}.{hf_stem}"

    if name == "v.patch_embd.weight":
        # Conv patch embedding [out, in_ch, kh, kw] → HF input_proj Linear
        # [out, in_ch*kh*kw]. The 4D→2D flattening is applied by the builder.
        return "vision_tower.patch_embedder.input_proj.weight"
    if name == "v.position_embd.weight":
        # [2, pos_emb_size, hidden] x/y coordinate position table.
        return "vision_tower.patch_embedder.position_embedding_table"
    if name == "mm.input_projection.weight":
        return "embed_vision.embedding_projection.weight"
    return None


# ---------------------------------------------------------------------------
# Muse Glimmer vision tower (v.* / mm.N)
# ---------------------------------------------------------------------------
# Muse Glimmer's tower is a dynamic-resolution ViT with a plain pre-norm block
# (LayerNorm + biased Q/K/V/out + GELU MLP), so unlike Gemma4 there are no
# post-norms, no SwiGLU gate and no QK norms. The projector is three separate
# `mm.N` matrices rather than a single input projection: two adapter layers on
# the pixel-shuffled features and one projection into the text hidden size.
_MUSE_GLIMMER_VISION_BLOCK_STEMS: dict[str, str] = {
    "ln1.weight": "norm1.weight",
    "ln1.bias": "norm1.bias",
    "ln2.weight": "norm2.weight",
    "ln2.bias": "norm2.bias",
    "attn_q.weight": "attn.q_proj.weight",
    "attn_q.bias": "attn.q_proj.bias",
    "attn_k.weight": "attn.k_proj.weight",
    "attn_k.bias": "attn.k_proj.bias",
    "attn_v.weight": "attn.v_proj.weight",
    "attn_v.bias": "attn.v_proj.bias",
    "attn_out.weight": "attn.proj.weight",
    "attn_out.bias": "attn.proj.bias",
    "ffn_up.weight": "mlp.fc1.weight",
    "ffn_up.bias": "mlp.fc1.bias",
    "ffn_down.weight": "mlp.fc2.weight",
    "ffn_down.bias": "mlp.fc2.bias",
}

# The three projector matrices, in file order. mm.0/mm.1 are the adapter that
# runs on merge_size**2-shuffled patches; mm.2 projects into the text model.
_MUSE_GLIMMER_PROJECTOR: dict[str, str] = {
    "mm.0.weight": "model.vision_adapter.fc1.weight",
    "mm.1.weight": "model.vision_adapter.fc2.weight",
    "mm.2.weight": "model.vision_projection.weight",
}


def map_mmproj_muse_glimmer_vision_to_hf(name: str) -> str | None:
    """Map a Muse Glimmer ``clip`` mmproj tensor name to its HF name.

    Returns the name consumed by
    :meth:`mobius.models.muse_glimmer.MuseGlimmerForConditionalGeneration.preprocess_weights`
    (``model.vision_tower.*``, ``model.vision_adapter.*``,
    ``model.vision_projection.*``), or ``None`` for tensors that are skipped.

    The projector matrices are named positionally in the GGUF (``mm.0`` …
    ``mm.2``); their roles are fixed by the published ``clip.projector_type =
    muse-glimmer`` layout and confirmed by their shapes — ``mm.0`` consumes
    ``hidden_size * merge_size**2``, ``mm.2`` produces the text hidden size.
    """
    if is_mmproj_stat_tensor(name):
        return None

    blk = _VISION_BLK.match(name)
    if blk is not None:
        idx, stem = blk.group(1), blk.group(2)
        hf_stem = _MUSE_GLIMMER_VISION_BLOCK_STEMS.get(stem)
        if hf_stem is None:
            return None
        return f"model.vision_tower.layers.{idx}.{hf_stem}"

    top: dict[str, str] = {
        # Conv patch embedding [out, in_ch, kh, kw]; flattened to the Linear
        # the encoder uses by the builder.
        "v.patch_embd.weight": ("model.vision_tower.patch_embedder.patch_embedding.weight"),
        # [pos_grid**2, hidden] learned table, interpolated per resolution.
        "v.position_embd.weight": (
            "model.vision_tower.patch_embedder.position_embedding_table.weight"
        ),
        "v.pre_ln.weight": "model.vision_tower.ln_pre.weight",
        "v.pre_ln.bias": "model.vision_tower.ln_pre.bias",
        "v.post_ln.weight": "model.vision_tower.ln_post.weight",
        "v.post_ln.bias": "model.vision_tower.ln_post.bias",
        **_MUSE_GLIMMER_PROJECTOR,
    }
    return top.get(name)


def map_mmproj_audio_to_hf(name: str) -> str | None:
    """Map a ``clip`` mmproj audio tensor name to its HF Gemma4 name.

    Returns the HuggingFace name (``audio_tower.*`` / ``embed_audio.*``), or
    ``None`` if skipped. See the module docstring: the audio Conformer
    forward-pass wiring is not yet applied to a built graph, so this is
    provided and name-tested but treated as experimental by the builder.
    """
    if is_mmproj_stat_tensor(name):
        return None

    blk = _AUDIO_BLK.match(name)
    if blk is not None:
        idx, stem = blk.group(1), blk.group(2)
        hf_stem = _AUDIO_BLOCK_STEMS.get(stem)
        if hf_stem is None:
            return None
        return f"audio_tower.layers.{idx}.{hf_stem}"

    audio_top: dict[str, str] = {
        "a.conv1d.0.weight": "audio_tower.subsample_conv_projection.conv0.weight",
        "a.conv1d.0.norm.weight": "audio_tower.subsample_conv_projection.norm0.weight",
        "a.conv1d.1.weight": "audio_tower.subsample_conv_projection.conv1.weight",
        "a.conv1d.1.norm.weight": "audio_tower.subsample_conv_projection.norm1.weight",
        "a.input_projection.weight": (
            "audio_tower.subsample_conv_projection.input_proj_linear.weight"
        ),
        "mm.a.input_projection.weight": "embed_audio.embedding_projection.weight",
    }
    return audio_top.get(name)
