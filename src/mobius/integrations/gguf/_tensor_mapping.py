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

from mobius.integrations.gguf._arch_registry import get_arch_spec, try_get_arch_spec
from mobius.integrations.gguf._errors import (
    DisabledGGUFArchitectureError,
    UnsupportedGGUFArchitectureError,
)

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

# OLMo 1 uses weight-free LayerNorm throughout. Its GGUF therefore has no
# normalization tensors: using the broader Llama recipe would obscure that
# architectural invariant even though no extra source tensors happen to arrive.
_OLMO_MAPPING: dict[str, str] = {
    "token_embd": "model.embed_tokens",
    "output": "lm_head",
    "blk.{bid}.attn_q": "model.layers.{bid}.self_attn.q_proj",
    "blk.{bid}.attn_k": "model.layers.{bid}.self_attn.k_proj",
    "blk.{bid}.attn_v": "model.layers.{bid}.self_attn.v_proj",
    "blk.{bid}.attn_output": "model.layers.{bid}.self_attn.o_proj",
    "blk.{bid}.ffn_gate": "model.layers.{bid}.mlp.gate_proj",
    "blk.{bid}.ffn_up": "model.layers.{bid}.mlp.up_proj",
    "blk.{bid}.ffn_down": "model.layers.{bid}.mlp.down_proj",
}

# Arcee uses a non-gated ReLU-squared MLP, so its exact recipe deliberately has
# no ffn_gate mapping.
_ARCEE_MAPPING: dict[str, str] = {
    "token_embd": "model.embed_tokens",
    "output": "lm_head",
    "output_norm": "model.norm",
    "blk.{bid}.attn_q": "model.layers.{bid}.self_attn.q_proj",
    "blk.{bid}.attn_k": "model.layers.{bid}.self_attn.k_proj",
    "blk.{bid}.attn_v": "model.layers.{bid}.self_attn.v_proj",
    "blk.{bid}.attn_output": "model.layers.{bid}.self_attn.o_proj",
    "blk.{bid}.attn_norm": "model.layers.{bid}.input_layernorm",
    "blk.{bid}.ffn_up": "model.layers.{bid}.mlp.up_proj",
    "blk.{bid}.ffn_down": "model.layers.{bid}.mlp.down_proj",
    "blk.{bid}.ffn_norm": "model.layers.{bid}.post_attention_layernorm",
}

# OLMo 2/3 apply RMSNorm after each branch, not before it.
_OLMO2_EXTRAS: dict[str, str] = {
    "blk.{bid}.post_attention_norm": ("model.layers.{bid}.post_attention_layernorm"),
    "blk.{bid}.post_ffw_norm": "model.layers.{bid}.post_feedforward_layernorm",
    "blk.{bid}.attn_q_norm": "model.layers.{bid}.self_attn.q_norm",
    "blk.{bid}.attn_k_norm": "model.layers.{bid}.self_attn.k_norm",
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

# Mamba uses model.* in Mobius; preprocess_weights nests the selective-scan
# parameters under mixer.ssm after this GGUF-to-HF stage.
_MAMBA_MAPPING: dict[str, str] = {
    "token_embd": "model.embeddings",
    "output": "lm_head",
    "output_norm": "model.norm_f",
    "blk.{bid}.attn_norm": "model.layers.{bid}.norm",
    "blk.{bid}.ssm_in": "model.layers.{bid}.mixer.in_proj",
    "blk.{bid}.ssm_out": "model.layers.{bid}.mixer.out_proj",
    "blk.{bid}.ssm_conv1d": "model.layers.{bid}.mixer.conv1d",
    "blk.{bid}.ssm_dt": "model.layers.{bid}.mixer.dt_proj",
    "blk.{bid}.ssm_a": "model.layers.{bid}.mixer.A_log",
    "blk.{bid}.ssm_d": "model.layers.{bid}.mixer.D",
    "blk.{bid}.ssm_x": "model.layers.{bid}.mixer.x_proj",
}

_MAMBA2_MAPPING: dict[str, str] = {
    "token_embd": "backbone.embeddings",
    "output": "lm_head",
    "output_norm": "backbone.norm_f",
    "blk.{bid}.attn_norm": "backbone.layers.{bid}.norm",
    "blk.{bid}.ssm_in": "backbone.layers.{bid}.mixer.in_proj",
    "blk.{bid}.ssm_out": "backbone.layers.{bid}.mixer.out_proj",
    "blk.{bid}.ssm_conv1d": "backbone.layers.{bid}.mixer.conv1d",
    "blk.{bid}.ssm_dt": "backbone.layers.{bid}.mixer.dt_bias@",
    "blk.{bid}.ssm_a": "backbone.layers.{bid}.mixer.A_log",
    "blk.{bid}.ssm_d": "backbone.layers.{bid}.mixer.D",
    "blk.{bid}.ssm_norm": "backbone.layers.{bid}.mixer.norm",
}

_BERT_MAPPING: dict[str, str] = {
    "token_embd": "bert.embeddings.word_embeddings",
    "position_embd": "bert.embeddings.position_embeddings",
    "token_types": "bert.embeddings.token_type_embeddings",
    "token_embd_norm": "bert.embeddings.LayerNorm",
    "cls": "bert.pooler.dense",
    "cls_out": "classifier",
    "blk.{bid}.attn_qkv": "bert.encoder.layer.{bid}.attention.self.qkv",
    "blk.{bid}.attn_q": "bert.encoder.layer.{bid}.attention.self.query",
    "blk.{bid}.attn_k": "bert.encoder.layer.{bid}.attention.self.key",
    "blk.{bid}.attn_v": "bert.encoder.layer.{bid}.attention.self.value",
    "blk.{bid}.attn_output": "bert.encoder.layer.{bid}.attention.output.dense",
    "blk.{bid}.attn_output_norm": "bert.encoder.layer.{bid}.attention.output.LayerNorm",
    "blk.{bid}.ffn_up": "bert.encoder.layer.{bid}.intermediate.dense",
    "blk.{bid}.ffn_down": "bert.encoder.layer.{bid}.output.dense",
    "blk.{bid}.layer_output_norm": "bert.encoder.layer.{bid}.output.LayerNorm",
}

_MODERN_BERT_MAPPING: dict[str, str] = {
    "token_embd": "model.embeddings.tok_embeddings",
    "token_embd_norm": "model.embeddings.norm",
    "output_norm": "model.final_norm",
    "cls": "head.dense",
    "cls_norm": "head.norm",
    "cls_out": "classifier",
    "blk.{bid}.attn_norm": "model.layers.{bid}.attn_norm",
    "blk.{bid}.attn_qkv": "model.layers.{bid}.attn.Wqkv",
    "blk.{bid}.attn_output": "model.layers.{bid}.attn.Wo",
    "blk.{bid}.ffn_up": "model.layers.{bid}.mlp.Wi",
    "blk.{bid}.ffn_down": "model.layers.{bid}.mlp.Wo",
    "blk.{bid}.ffn_norm": "model.layers.{bid}.mlp_norm",
}

_RECURRENT_SUFFIXES: dict[str, dict[str, frozenset[str]]] = {
    "mamba": {
        "token_embd": frozenset({".weight", ".scale", ".input_scale"}),
        "output": frozenset({".weight", ".scale", ".input_scale"}),
        "output_norm": frozenset({".weight", ".scale", ".input_scale"}),
        "blk.{bid}.attn_norm": frozenset({".weight", ".scale", ".input_scale"}),
        "blk.{bid}.ssm_in": frozenset({".weight", ".scale", ".input_scale"}),
        "blk.{bid}.ssm_conv1d": frozenset({".weight", ".bias", ".scale", ".input_scale"}),
        "blk.{bid}.ssm_x": frozenset({".weight", ".scale", ".input_scale"}),
        "blk.{bid}.ssm_dt": frozenset({".weight", ".bias", ".scale", ".input_scale"}),
        "blk.{bid}.ssm_a": frozenset({""}),
        "blk.{bid}.ssm_d": frozenset({""}),
        "blk.{bid}.ssm_out": frozenset({".weight", ".scale", ".input_scale"}),
    },
    "mamba2": {
        "token_embd": frozenset({".weight", ".scale", ".input_scale"}),
        "output": frozenset({".weight", ".scale", ".input_scale"}),
        "output_norm": frozenset({".weight", ".scale", ".input_scale"}),
        "blk.{bid}.attn_norm": frozenset({".weight", ".scale", ".input_scale"}),
        "blk.{bid}.ssm_in": frozenset({".weight", ".scale", ".input_scale"}),
        "blk.{bid}.ssm_conv1d": frozenset({".weight", ".bias", ".scale", ".input_scale"}),
        "blk.{bid}.ssm_dt": frozenset({".bias", ".scale", ".input_scale"}),
        "blk.{bid}.ssm_a": frozenset({""}),
        "blk.{bid}.ssm_d": frozenset({""}),
        "blk.{bid}.ssm_norm": frozenset({".weight", ".scale", ".input_scale"}),
        "blk.{bid}.ssm_out": frozenset({".weight", ".scale", ".input_scale"}),
    },
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

# Per-head/full-projection Q/K norms used by Qwen3-MoE and OLMoE. The
# ArchitectureConfig selects the logical width; the GGUF tensor family is the
# same for both representations.
_MOE_QK_NORM_EXTRAS: dict[str, str] = {
    "blk.{bid}.attn_q_norm": "model.layers.{bid}.self_attn.q_norm",
    "blk.{bid}.attn_k_norm": "model.layers.{bid}.self_attn.k_norm",
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

# Architectures sharing the llama HF naming convention are declared in
# ``_arch_registry`` via ``tensor_map_recipe=("llama", ...)`` rather than by a
# frozenset here, so the "which architectures does this cover?" question has one
# answer instead of one per module.

# HunYuan-v1 dense uses the Llama base but adds per-head Q/K layer-norms
# (HF: ``query_layernorm`` / ``key_layernorm``), which mobius renames to
# ``q_norm`` / ``k_norm`` inside the Attention component.
_HUNYUAN_EXTRAS: dict[str, str] = {
    "blk.{bid}.attn_q_norm": "model.layers.{bid}.self_attn.q_norm",
    "blk.{bid}.attn_k_norm": "model.layers.{bid}.self_attn.k_norm",
}

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


#: Named mapping tables that :data:`GGUFArchitectureSpec.tensor_map_recipe`
#: composes, in order. Later tables in a recipe override earlier ones, which is
#: how the Gemma variants replace the Llama ``ffn_norm`` mapping.
#:
#: Every name here must be referenced by at least one spec, and every name a
#: spec references must exist here. Both directions are asserted by
#: ``_arch_registry_test``, so an orphaned table or a typo in a recipe fails the
#: suite instead of silently producing an unmapped tensor.
_MAPPING_TABLES: MappingProxyType[str, dict[str, str]] = MappingProxyType(
    {
        "llama": _LLAMA_MAPPING,
        "olmo": _OLMO_MAPPING,
        "arcee": _ARCEE_MAPPING,
        "olmo2_extras": _OLMO2_EXTRAS,
        "phi3": _PHI3_MAPPING,
        "falcon": _FALCON_MAPPING,
        "gpt2": _GPT2_MAPPING,
        "mamba": _MAMBA_MAPPING,
        "mamba2": _MAMBA2_MAPPING,
        "bert": _BERT_MAPPING,
        "modern_bert": _MODERN_BERT_MAPPING,
        "deepseek4": _DEEPSEEK4_MAPPING,
        "gemma2_extras": _GEMMA2_EXTRAS,
        "gemma3_extras": _GEMMA3_EXTRAS,
        "gemma4_extras": _GEMMA4_EXTRAS,
        "moe_extras": _MOE_EXTRAS,
        "moe_qk_norm_extras": _MOE_QK_NORM_EXTRAS,
        "qwen35_hybrid_extras": _QWEN35_HYBRID_EXTRAS,
        "hunyuan_extras": _HUNYUAN_EXTRAS,
        "muse_glimmer_extras": _MUSE_GLIMMER_EXTRAS,
    }
)


def is_known_skip(gguf_name: str) -> bool:
    """Return ``True`` if *gguf_name* is intentionally skipped.

    Known skip patterns include tokenizer tensors and rotary
    embedding frequency tensors that are computed, not loaded.
    """
    if gguf_name.startswith("tokenizer."):
        return True
    if (
        "rope_freqs" in gguf_name
        or "attn_rot_embd" in gguf_name
        or gguf_name.startswith(("rope_factors_long", "rope_factors_short"))
    ):
        return True
    return False


@functools.lru_cache(maxsize=16)
def _build_mapping(
    architecture: str,
) -> MappingProxyType[str, str]:
    """Return the GGUF→HF stem mapping for *architecture*.

    The recipe comes from the architecture registry rather than an ``if/elif``
    chain here, so adding an architecture never means editing this function and
    the set of mappable architectures cannot drift from the set of configurable
    ones. Tables are layered in recipe order, so a later table overrides an
    earlier one — which is how the Gemma variants replace the Llama
    ``ffn_norm`` mapping with their pre-feedforward norm.

    Cached per architecture to avoid rebuilding on every tensor. Returns an
    immutable proxy to prevent mutation of the cache.

    Raises:
        UnsupportedGGUFArchitectureError: The architecture has no tensor
            mapping, whether because it is unregistered, deliberately disabled,
            or registered without one. This gate has always reported every such
            case as a ``ValueError``, so it reports disabled architectures that
            way too rather than leaking the ``NotImplementedError`` base that
            :func:`get_arch_spec` uses.
    """
    spec = try_get_arch_spec(architecture.lower())
    if spec is None or not spec.is_importable:
        try:
            get_arch_spec(architecture.lower())
        except DisabledGGUFArchitectureError as error:
            raise UnsupportedGGUFArchitectureError(str(error)) from None
        except UnsupportedGGUFArchitectureError:
            raise
        raise AssertionError("unreachable: spec is importable after all")
    result: dict[str, str] = {}
    for table_name in spec.tensor_map_recipe:
        table = _MAPPING_TABLES.get(table_name)
        if table is None:
            raise ValueError(
                f"Architecture {spec.gguf_arch!r} references unknown tensor mapping "
                f"table {table_name!r}. Known tables: {sorted(_MAPPING_TABLES)}"
            )
        result.update(table)
    return MappingProxyType(result)


# Regex to extract the block index from "blk.0.attn_q" etc.
_BLK_PATTERN = re.compile(r"blk\.(\d+)\.")
_BLK_TEMPLATE = "blk.{bid}."


def _split_suffix(name: str) -> tuple[str, str]:
    """Split ``"blk.0.attn_q.weight"`` → ``("blk.0.attn_q", ".weight")``.

    Returns ``("blk.0.attn_q", "")`` if no suffix is found.
    """
    # llama.cpp's generic model loader accepts sidecar quantization scales for
    # every projection family. Keep these names visible to pre-build validation
    # rather than silently treating them as unknown tensors.
    for suffix in (".input_scale", ".weight", ".scale", ".bias"):
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
        allowed_suffixes = _RECURRENT_SUFFIXES.get(architecture, {}).get(lookup)
        if allowed_suffixes is not None and suffix not in allowed_suffixes:
            return None
        hf_pattern = mapping.get(lookup)
        if hf_pattern is not None:
            hf_pattern = hf_pattern.replace("{bid}", bid)
            if hf_pattern.endswith("@"):
                return hf_pattern[:-1]
            return hf_pattern + suffix
    else:
        allowed_suffixes = _RECURRENT_SUFFIXES.get(architecture, {}).get(stem)
        if allowed_suffixes is not None and suffix not in allowed_suffixes:
            return None
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
