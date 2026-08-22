# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GGUF → HuggingFace tensor name mapping.

Maps GGUF tensor names (e.g. ``blk.0.attn_q.weight``) to their
HuggingFace equivalents (e.g. ``model.layers.0.self_attn.q_proj.weight``).
Mappings are architecture-specific because different HF models use
different naming conventions.

The GGUF standard tensor names are defined in
https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

Usage::

    from mobius.integrations.gguf._tensor_mapping import (
        map_gguf_to_hf_names,
        build_gguf_to_hf_map,
    )

    hf_name = map_gguf_to_hf_names("blk.0.attn_q.weight", "llama")
    # Returns: "model.layers.0.self_attn.q_proj.weight"

    hf_name = map_gguf_to_hf_names("tokenizer.ggml.tokens", "llama")
    # Returns: None  (tokenizer tensors are skipped)
"""

from __future__ import annotations

__all__ = [
    "build_gguf_to_hf_map",
    "is_known_skip",
    "map_gguf_to_hf_names",
]

import functools
import re
from types import MappingProxyType

# ---------------------------------------------------------------------------
# Architecture-specific GGUF → HF stem mappings
# ---------------------------------------------------------------------------
# Keys are GGUF tensor name stems (without .weight/.bias suffix).
# ``{bid}`` is a placeholder for the block/layer index.
# Values are the corresponding HuggingFace tensor name stems.
#
# Verified against HuggingFace transformers model implementations
# (e.g. LlamaForCausalLM, Qwen2ForCausalLM, Gemma2ForCausalLM, etc.).
# ---------------------------------------------------------------------------

# Llama-family naming convention. Used by Llama, Mistral, Qwen2, Qwen3,
# StarCoder2, InternLM2, Nemotron, StableLM, and DeciLM.
_LLAMA_MAPPING: dict[str, str] = {
    "token_embd": "model.embed_tokens",
    "output": "lm_head",
    "output_norm": "model.norm",
    "blk.{bid}.attn_q": "model.layers.{bid}.self_attn.q_proj",
    "blk.{bid}.attn_k": "model.layers.{bid}.self_attn.k_proj",
    "blk.{bid}.attn_v": "model.layers.{bid}.self_attn.v_proj",
    "blk.{bid}.attn_output": ("model.layers.{bid}.self_attn.o_proj"),
    "blk.{bid}.attn_norm": ("model.layers.{bid}.input_layernorm"),
    "blk.{bid}.ffn_gate": "model.layers.{bid}.mlp.gate_proj",
    "blk.{bid}.ffn_up": "model.layers.{bid}.mlp.up_proj",
    "blk.{bid}.ffn_down": "model.layers.{bid}.mlp.down_proj",
    "blk.{bid}.ffn_norm": ("model.layers.{bid}.post_attention_layernorm"),
}

# Gemma2 uses the llama.cpp Gemma tensor names (the same norm subset as
# Gemma3/4, minus the per-head Q/K norms): ``ffn_norm`` is the pre-feedforward
# norm (overriding the Llama post-attention mapping), ``post_attention_norm`` is
# the post-attention sandwich norm, and ``post_ffw_norm`` is the post-feedforward
# norm. The older ``pre_ffn_norm``/``post_ffn_norm`` names never appear in real
# llama.cpp Gemma2 GGUFs, so mapping them left the sandwich norms unloaded.
_GEMMA2_EXTRAS: dict[str, str] = {
    "blk.{bid}.ffn_norm": ("model.layers.{bid}.pre_feedforward_layernorm"),
    "blk.{bid}.post_attention_norm": ("model.layers.{bid}.post_attention_layernorm"),
    "blk.{bid}.post_ffw_norm": ("model.layers.{bid}.post_feedforward_layernorm"),
}

# Gemma3 uses the llama.cpp Gemma tensor names (ffn_norm as the pre-feedforward
# norm, plus post_attention/post_ffw norms) and additionally carries per-head
# Q/K norms that Gemma2 lacks.
_GEMMA3_EXTRAS: dict[str, str] = {
    "blk.{bid}.ffn_norm": ("model.layers.{bid}.pre_feedforward_layernorm"),
    "blk.{bid}.post_attention_norm": ("model.layers.{bid}.post_attention_layernorm"),
    "blk.{bid}.post_ffw_norm": ("model.layers.{bid}.post_feedforward_layernorm"),
    "blk.{bid}.attn_q_norm": "model.layers.{bid}.self_attn.q_norm",
    "blk.{bid}.attn_k_norm": "model.layers.{bid}.self_attn.k_norm",
}

# Gemma 4 extras on top of the Llama base + Gemma2 extras.
# Gemma 4 GGUF tensor names are taken from llama.cpp constants (gguf-py/gguf/constants.py).
#
# Key differences from Gemma 2/3:
#   - blk.{bid}.ffn_norm → pre_feedforward_layernorm (overrides Llama mapping of post_attn_layernorm)
#   - blk.{bid}.post_attention_norm → post_attention_layernorm (ATTN_POST_NORM, new in Gemma 4)
#   - Q/K/V norms on attention heads
#   - Per-layer scalar gate (all variants)
#   - MoE path norms (26B-A4B with enable_moe_block=True)
#   - Per-layer input embeddings (E2B/E4B with hidden_size_per_layer_input>0)
#
# Note: Gemma 4 GGUF contains the text backbone only. Vision and audio encoders
# are not present in the GGUF file and must be loaded from the HF checkpoint.
_GEMMA4_EXTRAS: dict[str, str] = {
    # Override _LLAMA_MAPPING: ffn_norm is the pre-feedforward norm, not post-attention.
    "blk.{bid}.ffn_norm": ("model.layers.{bid}.pre_feedforward_layernorm"),
    # Post-attention layernorm (ATTN_POST_NORM, present in all Gemma 4 variants).
    "blk.{bid}.post_attention_norm": ("model.layers.{bid}.post_attention_layernorm"),
    # Post-feedforward layernorm (FFN_POST_NORM).
    "blk.{bid}.post_ffw_norm": ("model.layers.{bid}.post_feedforward_layernorm"),
    # Per-head Q/K/V norms (all variants).
    "blk.{bid}.attn_q_norm": "model.layers.{bid}.self_attn.q_norm",
    "blk.{bid}.attn_k_norm": "model.layers.{bid}.self_attn.k_norm",
    # Per-layer scalar gate applied after the residual (all variants).
    # GGUF: LAYER_OUT_SCALE → HF: Gemma4TextDecoderLayer.layer_scalar
    "blk.{bid}.layer_output_scale": "model.layers.{bid}.layer_scalar",
    # MoE path norms (models with enable_moe_block=True, e.g. 26B-A4B).
    "blk.{bid}.pre_ffw_norm_2": ("model.layers.{bid}.pre_feedforward_layernorm_2"),
    "blk.{bid}.post_ffw_norm_1": ("model.layers.{bid}.post_feedforward_layernorm_1"),
    "blk.{bid}.post_ffw_norm_2": ("model.layers.{bid}.post_feedforward_layernorm_2"),
    # Per-layer input embedding path (models with hidden_size_per_layer_input>0, e.g. E2B/E4B).
    "blk.{bid}.inp_gate": "model.layers.{bid}.per_layer_input_gate",
    "blk.{bid}.proj": "model.layers.{bid}.per_layer_projection",
    "blk.{bid}.post_norm": "model.layers.{bid}.post_per_layer_input_norm",
    # Top-level per-layer input tensors (E2B/E4B). GGUF stores the per-layer
    # token embedding table, its projection, and projection norm as global
    # (non-block) tensors that live inside the text backbone (model.*).
    "per_layer_token_embd": "model.embed_tokens_per_layer",
    "per_layer_model_proj": "model.per_layer_model_projection",
    "per_layer_proj_norm": "model.per_layer_projection_norm",
}

# Phi-3 uses fused QKV and gate-up projections.
_PHI3_MAPPING: dict[str, str] = {
    "token_embd": "model.embed_tokens",
    "output": "lm_head",
    "output_norm": "model.norm",
    "blk.{bid}.attn_qkv": ("model.layers.{bid}.self_attn.qkv_proj"),
    "blk.{bid}.attn_output": ("model.layers.{bid}.self_attn.o_proj"),
    "blk.{bid}.attn_norm": ("model.layers.{bid}.input_layernorm"),
    "blk.{bid}.ffn_up": ("model.layers.{bid}.mlp.gate_up_proj"),
    "blk.{bid}.ffn_down": "model.layers.{bid}.mlp.down_proj",
    "blk.{bid}.ffn_norm": ("model.layers.{bid}.post_attention_layernorm"),
}

# Falcon uses transformer.h.* naming.
_FALCON_MAPPING: dict[str, str] = {
    "token_embd": "transformer.word_embeddings",
    "output": "lm_head",
    "output_norm": "transformer.ln_f",
    "blk.{bid}.attn_qkv": ("transformer.h.{bid}.self_attention.query_key_value"),
    "blk.{bid}.attn_output": ("transformer.h.{bid}.self_attention.dense"),
    "blk.{bid}.attn_norm": ("transformer.h.{bid}.input_layernorm"),
    "blk.{bid}.ffn_up": ("transformer.h.{bid}.mlp.dense_h_to_4h"),
    "blk.{bid}.ffn_down": ("transformer.h.{bid}.mlp.dense_4h_to_h"),
    "blk.{bid}.ffn_norm": "transformer.h.{bid}.ln_2",
}

# GPT-2 uses transformer.h.* with c_attn/c_proj naming.
_GPT2_MAPPING: dict[str, str] = {
    "token_embd": "transformer.wte",
    "position_embd": "transformer.wpe",
    "output": "lm_head",
    "output_norm": "transformer.ln_f",
    "blk.{bid}.attn_qkv": "transformer.h.{bid}.attn.c_attn",
    "blk.{bid}.attn_output": ("transformer.h.{bid}.attn.c_proj"),
    "blk.{bid}.attn_norm": "transformer.h.{bid}.ln_1",
    "blk.{bid}.ffn_up": "transformer.h.{bid}.mlp.c_fc",
    "blk.{bid}.ffn_down": "transformer.h.{bid}.mlp.c_proj",
    "blk.{bid}.ffn_norm": "transformer.h.{bid}.ln_2",
}

# Mamba uses backbone.* naming.
_MAMBA_MAPPING: dict[str, str] = {
    "token_embd": "backbone.embeddings",
    "output": "lm_head",
    "output_norm": "backbone.norm_f",
    "blk.{bid}.attn_norm": "backbone.layers.{bid}.norm",
    "blk.{bid}.ssm_in": ("backbone.layers.{bid}.mixer.in_proj"),
    "blk.{bid}.ssm_out": ("backbone.layers.{bid}.mixer.out_proj"),
    "blk.{bid}.ssm_conv1d": ("backbone.layers.{bid}.mixer.conv1d"),
    "blk.{bid}.ssm_dt": ("backbone.layers.{bid}.mixer.dt_proj"),
    "blk.{bid}.ssm_a": "backbone.layers.{bid}.mixer.A_log",
    "blk.{bid}.ssm_d": "backbone.layers.{bid}.mixer.D",
    "blk.{bid}.ssm_x": ("backbone.layers.{bid}.mixer.x_proj"),
}

# MoE extensions for Qwen2MoE/Qwen3MoE/DeepSeek.
_MOE_EXTRAS: dict[str, str] = {
    "blk.{bid}.ffn_gate_inp": ("model.layers.{bid}.mlp.gate"),
    "blk.{bid}.ffn_gate_exps": ("model.layers.{bid}.mlp.experts.gate_proj"),
    "blk.{bid}.ffn_up_exps": ("model.layers.{bid}.mlp.experts.up_proj"),
    "blk.{bid}.ffn_down_exps": ("model.layers.{bid}.mlp.experts.down_proj"),
    "blk.{bid}.ffn_gate_inp_shexp": ("model.layers.{bid}.mlp.shared_expert_gate"),
    "blk.{bid}.ffn_gate_shexp": ("model.layers.{bid}.mlp.shared_expert.gate_proj"),
    "blk.{bid}.ffn_up_shexp": ("model.layers.{bid}.mlp.shared_expert.up_proj"),
    "blk.{bid}.ffn_down_shexp": ("model.layers.{bid}.mlp.shared_expert.down_proj"),
}

# Qwen3.5 hybrid extensions: DeltaNet (SSM) + full-attention.
# DeltaNet layers use linear_attn.* naming; full-attention layers add
# q_norm/k_norm under self_attn; both use post_attention_layernorm.
_QWEN35_HYBRID_EXTRAS: dict[str, str] = {
    # DeltaNet (linear attention) layers
    "blk.{bid}.attn_qkv": "model.layers.{bid}.linear_attn.in_proj_qkv",
    "blk.{bid}.attn_gate": "model.layers.{bid}.linear_attn.in_proj_z",
    "blk.{bid}.ssm_beta": "model.layers.{bid}.linear_attn.in_proj_b",
    "blk.{bid}.ssm_alpha": "model.layers.{bid}.linear_attn.in_proj_a",
    "blk.{bid}.ssm_conv1d": "model.layers.{bid}.linear_attn.conv1d",
    "blk.{bid}.ssm_dt": "model.layers.{bid}.linear_attn.dt_bias",
    "blk.{bid}.ssm_a": "model.layers.{bid}.linear_attn.A_log",
    "blk.{bid}.ssm_norm": "model.layers.{bid}.linear_attn.norm",
    "blk.{bid}.ssm_out": "model.layers.{bid}.linear_attn.out_proj",
    # Full-attention layers — QK norms
    "blk.{bid}.attn_q_norm": "model.layers.{bid}.self_attn.q_norm",
    "blk.{bid}.attn_k_norm": "model.layers.{bid}.self_attn.k_norm",
    # Both layer types — post-attention layernorm
    "blk.{bid}.post_attention_norm": ("model.layers.{bid}.post_attention_layernorm"),
}

_DEEPSEEK4_MAPPING: dict[str, str] = {
    "token_embd": "model.embed_tokens",
    "output": "lm_head",
    "output_norm": "model.norm",
    "output_hc_fn": "model.hc_head_fn",
    "output_hc_base": "model.hc_head_base@",
    "output_hc_scale": "model.hc_head_scale@",
    "blk.{bid}.attn_q_a": "model.layers.{bid}.self_attn.q_a_proj",
    "blk.{bid}.attn_q_a_norm": "model.layers.{bid}.self_attn.q_a_layernorm",
    "blk.{bid}.attn_q_b": "model.layers.{bid}.self_attn.q_b_proj",
    "blk.{bid}.attn_kv": "model.layers.{bid}.self_attn.kv_proj",
    "blk.{bid}.attn_kv_a_norm": "model.layers.{bid}.self_attn.kv_layernorm",
    "blk.{bid}.attn_output_a": "model.layers.{bid}.self_attn.o_a_proj",
    "blk.{bid}.attn_output_b": "model.layers.{bid}.self_attn.o_b_proj",
    "blk.{bid}.attn_sinks": "model.layers.{bid}.self_attn.attn_sink@",
    "blk.{bid}.attn_compressor_kv": "model.layers.{bid}.self_attn.compressor.wkv",
    "blk.{bid}.attn_compressor_gate": "model.layers.{bid}.self_attn.compressor.wgate",
    "blk.{bid}.attn_compressor_ape": "model.layers.{bid}.self_attn.compressor.ape@",
    "blk.{bid}.attn_compressor_norm": "model.layers.{bid}.self_attn.compressor.norm",
    "blk.{bid}.indexer.attn_q_b": "model.layers.{bid}.self_attn.indexer.wq_b",
    "blk.{bid}.indexer.proj": "model.layers.{bid}.self_attn.indexer.weights_proj",
    "blk.{bid}.indexer_compressor_kv": ("model.layers.{bid}.self_attn.indexer.compressor.wkv"),
    "blk.{bid}.indexer_compressor_gate": (
        "model.layers.{bid}.self_attn.indexer.compressor.wgate"
    ),
    "blk.{bid}.indexer_compressor_ape": (
        "model.layers.{bid}.self_attn.indexer.compressor.ape@"
    ),
    "blk.{bid}.indexer_compressor_norm": (
        "model.layers.{bid}.self_attn.indexer.compressor.norm"
    ),
    "blk.{bid}.attn_norm": "model.layers.{bid}.input_layernorm",
    "blk.{bid}.ffn_norm": "model.layers.{bid}.post_attention_layernorm",
    # DeepSeekV4MoE now composes the shared MoELayer (mobius.components._moe),
    # so the gate lives one level deeper at mlp.moe.gate.* than the bare
    # mlp.gate.* used before the QMoE export change.
    "blk.{bid}.ffn_gate_inp": "model.layers.{bid}.mlp.moe.gate",
    "blk.{bid}.exp_probs_b": "model.layers.{bid}.mlp.moe.gate",
    "blk.{bid}.ffn_gate_tid2eid": "model.layers.{bid}.mlp.moe.gate.tid2eid@",
    "blk.{bid}.ffn_gate_exps": "model.layers.{bid}.mlp.experts.gate_proj",
    "blk.{bid}.ffn_up_exps": "model.layers.{bid}.mlp.experts.up_proj",
    "blk.{bid}.ffn_down_exps": "model.layers.{bid}.mlp.experts.down_proj",
    "blk.{bid}.ffn_gate_shexp": "model.layers.{bid}.mlp.shared_experts.gate_proj",
    "blk.{bid}.ffn_up_shexp": "model.layers.{bid}.mlp.shared_experts.up_proj",
    "blk.{bid}.ffn_down_shexp": "model.layers.{bid}.mlp.shared_experts.down_proj",
    "blk.{bid}.hc_attn_fn": "model.layers.{bid}.hc_attn_fn",
    "blk.{bid}.hc_attn_base": "model.layers.{bid}.hc_attn_base@",
    "blk.{bid}.hc_attn_scale": "model.layers.{bid}.hc_attn_scale@",
    "blk.{bid}.hc_ffn_fn": "model.layers.{bid}.hc_ffn_fn",
    "blk.{bid}.hc_ffn_base": "model.layers.{bid}.hc_ffn_base@",
    "blk.{bid}.hc_ffn_scale": "model.layers.{bid}.hc_ffn_scale@",
}

# Architectures sharing the llama HF naming convention.
_LLAMA_FAMILY = frozenset(
    {
        "llama",
        "mistral",
        "qwen2",
        "qwen3",
        "qwen35",
        "starcoder2",
        "internlm2",
        "nemotron",
        "stablelm",
        "deci",
    }
)

_GEMMA_FAMILY = frozenset({"gemma", "gemma2", "gemma3"})

# HunYuan-v1 dense uses the Llama base but adds per-head Q/K layer-norms
# (HF: ``query_layernorm`` / ``key_layernorm``), which mobius renames to
# ``q_norm`` / ``k_norm`` inside the Attention component.
_HUNYUAN_EXTRAS: dict[str, str] = {
    "blk.{bid}.attn_q_norm": "model.layers.{bid}.self_attn.q_norm",
    "blk.{bid}.attn_k_norm": "model.layers.{bid}.self_attn.k_norm",
}

_HUNYUAN_FAMILY = frozenset({"hunyuan-dense", "hunyuan_v1_dense"})

# Muse Glimmer keeps the Llama projection names but rearranges the norms and
# adds a sigmoid gate on the attention output.
#
# Two GGUF tensors have no HuggingFace counterpart and are deliberately absent
# from this mapping, which makes `map_gguf_to_hf_names` skip them:
#
#   blk.{bid}.attn_q_norm / blk.{bid}.attn_k_norm
#       Muse Glimmer's QK normalization is scale-free — the HF checkpoint has no
#       q_norm/k_norm parameters at all. llama.cpp still materializes the tensors
#       so its generic attention path can multiply by something, so attn_k_norm
#       is a vector of ones and attn_q_norm is the `qk_scale_factor` scalar
#       broadcast over head_dim. The scalar is recovered in `_config_mapping`,
#       which reads it back out of attn_q_norm; dropping the tensors here would
#       otherwise lose it.
_MUSE_GLIMMER_EXTRAS: dict[str, str] = {
    # Muse Glimmer has four norms per block. `ffn_norm` is the *pre*-feedforward
    # norm, so it overrides the Llama mapping onto post_attention_layernorm.
    "blk.{bid}.ffn_norm": "model.layers.{bid}.pre_feedforward_layernorm",
    "blk.{bid}.post_attention_norm": ("model.layers.{bid}.post_attention_layernorm"),
    "blk.{bid}.post_ffw_norm": ("model.layers.{bid}.post_feedforward_layernorm"),
    # Sigmoid gate applied to the attention output before o_proj.
    "blk.{bid}.attn_gate": "model.layers.{bid}.self_attn.gate_proj",
}

_MUSE_GLIMMER_FAMILY = frozenset({"muse-glimmer", "muse_glimmer"})

_MOE_FAMILY = frozenset(
    {
        "qwen2moe",
        "qwen2_moe",
        "qwen3moe",
        "qwen3_moe",
        "qwen35moe",
    }
)


def is_known_skip(gguf_name: str) -> bool:
    """Return ``True`` if *gguf_name* is intentionally skipped.

    Known skip patterns include tokenizer tensors and rotary
    embedding frequency tensors that are computed, not loaded.
    """
    if gguf_name.startswith("tokenizer."):
        return True
    if "rope_freqs" in gguf_name or "attn_rot_embd" in gguf_name:
        return True
    return False


@functools.lru_cache(maxsize=16)
def _build_mapping(
    architecture: str,
) -> MappingProxyType[str, str]:
    """Return the GGUF→HF stem mapping for *architecture*.

    Cached per architecture to avoid rebuilding on every tensor.
    Returns an immutable proxy to prevent mutation of the cache.
    """
    arch = architecture.lower()

    if arch in _LLAMA_FAMILY:
        result = dict(_LLAMA_MAPPING)
        if arch == "qwen35":
            result.update(_QWEN35_HYBRID_EXTRAS)
    elif arch == "gemma3":
        # Gemma3 uses the llama.cpp Gemma tensor names (ffn_norm as the
        # pre-feedforward norm, plus post_attention/post_ffw norms and Q/K
        # norms), distinct from the older _GEMMA2_EXTRAS names — keep it out of
        # the shared _GEMMA_FAMILY path.
        result = dict(_LLAMA_MAPPING)
        result.update(_GEMMA3_EXTRAS)
    elif arch == "gemma2":
        # Gemma2 uses the llama.cpp Gemma tensor names (ffn_norm as the
        # pre-feedforward norm, plus post_attention/post_ffw sandwich norms),
        # distinct from the plain-Llama norm layout of Gemma v1. It has no
        # per-head Q/K norms (those are Gemma3+).
        result = dict(_LLAMA_MAPPING)
        result.update(_GEMMA2_EXTRAS)
    elif arch in _GEMMA_FAMILY:
        # Gemma v1: standard two-norm (input + post-attention) Llama layout with
        # no sandwich feedforward norms, so the plain Llama base is correct.
        result = dict(_LLAMA_MAPPING)
    elif arch == "gemma4":
        # Gemma 4 starts from the Llama base but needs several overrides and
        # many new tensor types for Q/K norms, per-layer scalars, MoE norms,
        # and per-layer input embeddings. Use a dedicated extras dict rather
        # than extending _GEMMA_FAMILY to avoid contaminating Gemma 2/3.
        result = dict(_LLAMA_MAPPING)
        result.update(_GEMMA4_EXTRAS)
    elif arch == "phi3":
        result = dict(_PHI3_MAPPING)
    elif arch == "falcon":
        result = dict(_FALCON_MAPPING)
    elif arch == "gpt2":
        result = dict(_GPT2_MAPPING)
    elif arch == "mamba":
        result = dict(_MAMBA_MAPPING)
    elif arch in _HUNYUAN_FAMILY:
        result = dict(_LLAMA_MAPPING)
        result.update(_HUNYUAN_EXTRAS)
    elif arch in _MUSE_GLIMMER_FAMILY:
        result = dict(_LLAMA_MAPPING)
        result.update(_MUSE_GLIMMER_EXTRAS)
    elif arch in _MOE_FAMILY:
        result = dict(_LLAMA_MAPPING)
        result.update(_MOE_EXTRAS)
        if arch == "qwen35moe":
            result.update(_QWEN35_HYBRID_EXTRAS)
    elif arch == "deepseek4":
        result = dict(_DEEPSEEK4_MAPPING)
    else:
        supported = sorted(
            _LLAMA_FAMILY
            | _GEMMA_FAMILY
            | _MOE_FAMILY
            | _HUNYUAN_FAMILY
            | _MUSE_GLIMMER_FAMILY
            | {"deepseek4", "gemma4", "phi3", "falcon", "gpt2", "mamba"}
        )
        raise ValueError(
            f"Unsupported GGUF architecture: {architecture!r}. "
            f"Supported: {', '.join(supported)}"
        )
    return MappingProxyType(result)


# Regex to extract the block index from "blk.0.attn_q" etc.
_BLK_PATTERN = re.compile(r"blk\.(\d+)\.")
_BLK_TEMPLATE = "blk.{bid}."


def _split_suffix(name: str) -> tuple[str, str]:
    """Split ``"blk.0.attn_q.weight"`` → ``("blk.0.attn_q", ".weight")``.

    Returns ``("blk.0.attn_q", "")`` if no suffix is found.
    """
    for suffix in (".weight", ".bias"):
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return name, ""


def map_gguf_to_hf_names(
    gguf_name: str,
    architecture: str,
) -> str | None:
    """Map a GGUF tensor name to the equivalent HuggingFace name.

    Args:
        gguf_name: Full GGUF tensor name
            (e.g. ``"blk.0.attn_q.weight"``).
        architecture: GGUF architecture string
            (e.g. ``"llama"``, ``"qwen2"``).

    Returns:
        The corresponding HuggingFace tensor name, or ``None``
        if the tensor should be skipped (e.g. tokenizer tensors,
        rotary embedding frequencies).
    """
    # Skip known non-model tensors (tokenizer, rope freqs, etc.)
    if is_known_skip(gguf_name):
        return None

    stem, suffix = _split_suffix(gguf_name)
    mapping = _build_mapping(architecture)

    # Block-indexed tensors: blk.{N}.xxx → model.layers.{N}.xxx
    blk_match = _BLK_PATTERN.match(stem)
    if blk_match:
        bid = blk_match.group(1)
        lookup = _BLK_PATTERN.sub(_BLK_TEMPLATE, stem)
        hf_pattern = mapping.get(lookup)
        if hf_pattern is not None:
            hf_pattern = hf_pattern.replace("{bid}", bid)
            if hf_pattern.endswith("@"):
                return hf_pattern[:-1]
            return hf_pattern + suffix
    else:
        hf_stem = mapping.get(stem)
        if hf_stem is not None:
            if hf_stem.endswith("@"):
                return hf_stem[:-1]
            return hf_stem + suffix

    return None


def build_gguf_to_hf_map(
    gguf_names: list[str],
    architecture: str,
) -> dict[str, str]:
    """Build a complete GGUF→HF name mapping for a list of tensors.

    Convenience function that calls :func:`map_gguf_to_hf_names`
    for each name and collects the results.

    Args:
        gguf_names: All GGUF tensor names from the file.
        architecture: GGUF architecture string.

    Returns:
        Dict mapping GGUF tensor names → HF tensor names.
        Tensors that should be skipped are omitted.
    """
    result: dict[str, str] = {}
    for name in gguf_names:
        hf_name = map_gguf_to_hf_names(name, architecture)
        if hf_name is not None:
            result[name] = hf_name
    return result
