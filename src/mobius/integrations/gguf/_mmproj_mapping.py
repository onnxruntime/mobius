# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GGUF ``clip`` mmproj → HuggingFace tensor name mapping.

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
``.output_max`` / ``.output_min``) are learned clipping bounds. Gemma4's
``ClippableLinear`` graph consumes them; dropping them changes model semantics.

The name derivation was verified against ``unsloth/gemma-4-E2B-it-GGUF``'s
``mmproj-F16.gguf`` (``clip.vision.projector_type = gemma4v``,
``clip.audio.projector_type = gemma4a``) by round-tripping through
``preprocess_weights`` and checking every ONNX initializer receives a weight.
"""

from __future__ import annotations

__all__ = [
    "is_mmproj_stat_tensor",
    "map_generic_projector_to_onnx",
    "map_generic_vision_to_onnx",
    "map_mmproj_gemma3_vision_to_hf",
    "map_mmproj_qwen_vision_to_hf",
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

for _gguf_stem, _hf_stem in {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_out": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}.items():
    for _bound in ("input_min", "input_max", "output_min", "output_max"):
        _VISION_BLOCK_STEMS[f"{_gguf_stem}.{_bound}"] = f"{_hf_stem}.{_bound}"

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
    """Return ``True`` for llama.cpp activation clipping bounds.

    These tensors are model parameters for Gemma4 ``ClippableLinear`` layers.
    The predicate remains public for callers that need to classify their role;
    mapping functions decide whether a specific projector consumes them.
    """
    return name.endswith(_STAT_SUFFIXES)


_GENERIC_VISION_BLOCK_STEMS: dict[str, str] = {
    "ln1.weight": "layer_norm1.weight",
    "ln1.bias": "layer_norm1.bias",
    "ln2.weight": "layer_norm2.weight",
    "ln2.bias": "layer_norm2.bias",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_out.weight": "self_attn.out_proj.weight",
    "attn_out.bias": "self_attn.out_proj.bias",
    "ffn_down.weight": "mlp.up_proj.weight",
    "ffn_down.bias": "mlp.up_proj.bias",
    "ffn_up.weight": "mlp.down_proj.weight",
    "ffn_up.bias": "mlp.down_proj.bias",
}


def map_generic_vision_to_onnx(name: str) -> str | None:
    """Map a legacy CLIP/SigLIP sidecar tower directly to ONNX module names."""
    blk = _VISION_BLK.match(name)
    if blk is not None:
        idx, stem = blk.group(1), blk.group(2)
        mapped = _GENERIC_VISION_BLOCK_STEMS.get(stem)
        return None if mapped is None else f"vision_tower.encoder.{idx}.{mapped}"

    top = {
        "v.class_embd": "vision_tower.embeddings.class_embedding",
        "v.patch_embd.weight": ("vision_tower.embeddings.patch_embedding.projection.weight"),
        "v.patch_embd.bias": "vision_tower.embeddings.patch_embedding.projection.bias",
        "v.position_embd.weight": "vision_tower.embeddings.position_embedding.weight",
        "v.pre_ln.weight": "vision_tower.pre_layrnorm.weight",
        "v.pre_ln.bias": "vision_tower.pre_layrnorm.bias",
        "v.post_ln.weight": "vision_tower.post_layernorm.weight",
        "v.post_ln.bias": "vision_tower.post_layernorm.bias",
    }
    return top.get(name)


def _map_ldp_block(name: str) -> str | None:
    match = re.match(r"^mm\.model\.mb_block\.([12])\.block\.(.+)$", name)
    if match is None:
        return None
    block = f"block_{match.group(1)}"
    suffix = match.group(2)
    mapped = {
        "0.0.weight": "depthwise.weight",
        "0.1.weight": "depthwise_norm.weight",
        "0.1.bias": "depthwise_norm.bias",
        "1.fc1.weight": "se_fc1.weight",
        "1.fc1.bias": "se_fc1.bias",
        "1.fc2.weight": "se_fc2.weight",
        "1.fc2.bias": "se_fc2.bias",
        "2.0.weight": "pointwise.weight",
        "2.1.weight": "pointwise_norm.weight",
        "2.1.bias": "pointwise_norm.bias",
    }.get(suffix)
    return None if mapped is None else f"projector.{block}.{mapped}"


def map_generic_projector_to_onnx(name: str, projector_type: str) -> str | None:
    """Map one exact generic projector closure directly to ONNX module names."""
    if projector_type == "mlp":
        return {
            "mm.0.weight": "projector.linear_0.weight",
            "mm.0.bias": "projector.linear_0.bias",
            "mm.2.weight": "projector.linear_2.weight",
            "mm.2.bias": "projector.linear_2.bias",
        }.get(name)
    if projector_type == "ldp":
        top = {
            "mm.model.mlp.1.weight": "projector.mlp_1.weight",
            "mm.model.mlp.1.bias": "projector.mlp_1.bias",
            "mm.model.mlp.3.weight": "projector.mlp_3.weight",
            "mm.model.mlp.3.bias": "projector.mlp_3.bias",
        }
        return top.get(name) or _map_ldp_block(name)
    if projector_type == "ldpv2":
        return {
            "mm.model.mlp.0.weight": "projector.mlp_0.weight",
            "mm.model.mlp.0.bias": "projector.mlp_0.bias",
            "mm.model.mlp.2.weight": "projector.mlp_2.weight",
            "mm.model.mlp.2.bias": "projector.mlp_2.bias",
            "mm.model.peg.0.weight": "projector.peg_0.weight",
            "mm.model.peg.0.bias": "projector.peg_0.bias",
        }.get(name)
    if projector_type == "adapter":
        return {
            "adapter.boi": "projector.boi",
            "adapter.eoi": "projector.eoi",
            "adapter.conv.weight": "projector.conv.weight",
            "adapter.conv.bias": "projector.conv.bias",
            "adapter.linear.linear.weight": "projector.linear.weight",
            "adapter.linear.norm1.weight": "projector.norm1.weight",
            "adapter.linear.norm1.bias": "projector.norm1.bias",
            "adapter.linear.dense_h_to_4h.weight": "projector.dense_h_to_4h.weight",
            "adapter.linear.gate.weight": "projector.gate.weight",
            "adapter.linear.dense_4h_to_h.weight": "projector.dense_4h_to_h.weight",
        }.get(name)
    if projector_type == "resampler":
        return {
            "resampler.query": "projector.query",
            "resampler.kv.weight": "projector.kv.weight",
            "resampler.attn.q.weight": "projector.attn_q.weight",
            "resampler.attn.q.bias": "projector.attn_q.bias",
            "resampler.attn.k.weight": "projector.attn_k.weight",
            "resampler.attn.k.bias": "projector.attn_k.bias",
            "resampler.attn.v.weight": "projector.attn_v.weight",
            "resampler.attn.v.bias": "projector.attn_v.bias",
            "resampler.attn.out.weight": "projector.attn_out.weight",
            "resampler.attn.out.bias": "projector.attn_out.bias",
            "resampler.ln_q.weight": "projector.ln_q.weight",
            "resampler.ln_q.bias": "projector.ln_q.bias",
            "resampler.ln_kv.weight": "projector.ln_kv.weight",
            "resampler.ln_kv.bias": "projector.ln_kv.bias",
            "resampler.ln_post.weight": "projector.ln_post.weight",
            "resampler.ln_post.bias": "projector.ln_post.bias",
            "resampler.proj.weight": "projector.proj.weight",
        }.get(name)
    raise ValueError(f"Unknown generic GGUF projector type {projector_type!r}")


def map_mmproj_vision_to_hf(name: str) -> str | None:
    """Map a ``clip`` mmproj vision tensor name to its HF Gemma4 name.

    Returns the HuggingFace name consumed by
    :meth:`Gemma4Model.preprocess_weights` (``vision_tower.*`` /
    ``embed_vision.*``), or ``None`` if the tensor is outside the Gemma4 vision
    closure (for example, the audio tower).

    The mapping mirrors :mod:`_tensor_mapping` for the text backbone.
    """
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


_GEMMA3_VISION_BLOCK_STEMS: dict[str, str] = {
    "ln1.weight": "layer_norm1.weight",
    "ln1.bias": "layer_norm1.bias",
    "ln2.weight": "layer_norm2.weight",
    "ln2.bias": "layer_norm2.bias",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_out.weight": "self_attn.out_proj.weight",
    "attn_out.bias": "self_attn.out_proj.bias",
    # llama.cpp names these for their data-flow direction. In the Gemma3
    # artifact ffn_down expands hidden→intermediate and ffn_up contracts back.
    "ffn_down.weight": "mlp.fc1.weight",
    "ffn_down.bias": "mlp.fc1.bias",
    "ffn_up.weight": "mlp.fc2.weight",
    "ffn_up.bias": "mlp.fc2.bias",
}


def map_mmproj_gemma3_vision_to_hf(name: str) -> str | None:
    """Map the pinned Gemma3 sidecar closure to names its HF importer consumes."""
    blk = _VISION_BLK.match(name)
    if blk is not None:
        idx, stem = blk.group(1), blk.group(2)
        hf_stem = _GEMMA3_VISION_BLOCK_STEMS.get(stem)
        if hf_stem is None:
            return None
        return f"vision_tower.vision_model.encoder.layers.{idx}.{hf_stem}"

    top = {
        "v.patch_embd.weight": ("vision_tower.vision_model.embeddings.patch_embedding.weight"),
        "v.patch_embd.bias": "vision_tower.vision_model.embeddings.patch_embedding.bias",
        "v.position_embd.weight": (
            "vision_tower.vision_model.embeddings.position_embedding.weight"
        ),
        "v.post_ln.weight": "vision_tower.vision_model.post_layernorm.weight",
        "v.post_ln.bias": "vision_tower.vision_model.post_layernorm.bias",
        "mm.soft_emb_norm.weight": "multi_modal_projector.mm_soft_emb_norm.weight",
        "mm.input_projection.weight": "multi_modal_projector.mm_input_projection_weight",
    }
    return top.get(name)


_QWEN_VISION_BLOCK_STEMS: dict[str, str] = {
    "attn_out.weight": "attn.proj.weight",
    "attn_out.bias": "attn.proj.bias",
    "ln1.weight": "norm1.weight",
    "ln1.bias": "norm1.bias",
    "ln2.weight": "norm2.weight",
    "ln2.bias": "norm2.bias",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_gate.bias": "mlp.gate_proj.bias",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_up.bias": "mlp.up_proj.bias",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_down.bias": "mlp.down_proj.bias",
}


def map_mmproj_qwen_vision_to_hf(name: str) -> str | None:
    """Map non-fused Qwen2/Qwen2.5-VL sidecar tensors to HF names.

    Split Q/K/V and temporal patch halves are fused by the builder because one
    ONNX initializer consumes each group.
    """
    blk = _VISION_BLK.match(name)
    if blk is not None:
        idx, stem = blk.group(1), blk.group(2)
        hf_stem = _QWEN_VISION_BLOCK_STEMS.get(stem)
        return None if hf_stem is None else f"visual.blocks.{idx}.{hf_stem}"

    top = {
        "v.post_ln.weight": "visual.merger.ln_q.weight",
        "v.post_ln.bias": "visual.merger.ln_q.bias",
        "mm.0.weight": "visual.merger.mlp.0.weight",
        "mm.0.bias": "visual.merger.mlp.0.bias",
        "mm.2.weight": "visual.merger.mlp.2.weight",
        "mm.2.bias": "visual.merger.mlp.2.bias",
    }
    return top.get(name)


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
