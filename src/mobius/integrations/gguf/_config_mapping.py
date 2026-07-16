# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GGUF metadata → ArchitectureConfig mapping.

Maps GGUF key-value metadata to :class:`ArchitectureConfig` fields,
producing a config suitable for the standard build pipeline. Leverages
HuggingFace's ``GGUF_CONFIG_MAPPING`` for the per-architecture key
mapping where available, with fallback to the GGUF standard key names.

Example::

    from mobius.integrations.gguf._reader import GGUFModel
    from mobius.integrations.gguf._config_mapping import gguf_to_config

    model = GGUFModel("model.gguf")
    config = gguf_to_config(model)
    print(config.hidden_size, config.num_hidden_layers)
"""

from __future__ import annotations

__all__ = ["gguf_to_config"]

import dataclasses
import logging
from typing import Any

import numpy as np

from mobius._configs import ArchitectureConfig, Gemma4Config

logger = logging.getLogger(__name__)


# Map GGUF architecture names → our registry model_type strings.
# Most names match; a few need remapping.
GGUF_ARCH_TO_MODEL_TYPE: dict[str, str] = {
    "llama": "llama",
    "mistral": "llama",  # Mistral uses Llama architecture
    "qwen2": "qwen2",
    "qwen2_moe": "qwen2_moe",
    "qwen3": "qwen3",
    "qwen3_moe": "qwen3_moe",
    "qwen35moe": "qwen3_5_moe",
    "glm-dsa": "glm_moe_dsa",
    "gemma2": "gemma2",
    "gemma3": "gemma3_text",
    # Gemma 4 GGUF contains the text backbone only — no vision or audio encoder.
    # Vision and audio encoders are exported separately from the HuggingFace checkpoint.
    "gemma4": "gemma4_text",
    "phi3": "phi3",
    "falcon": "falcon",
    "gpt2": "gpt2",
    "mamba": "mamba",
    "bloom": "bloom",
    "starcoder2": "starcoder2",
    "stablelm": "stablelm",
    "nemotron": "nemotron",
    "t5": "t5",
    "hunyuan-dense": "hunyuan_v1_dense",
    "deci": "llama",  # DeciLM uses Llama architecture
}


# Standard GGUF metadata keys → HuggingFace config field names.
# Used as fallback when HF's GGUF_CONFIG_MAPPING is not available
# for a given architecture.
_DEFAULT_KEY_MAP: dict[str, str] = {
    "embedding_length": "hidden_size",
    "feed_forward_length": "intermediate_size",
    "block_count": "num_hidden_layers",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "rope.freq_base": "rope_theta",
    "context_length": "max_position_embeddings",
    "vocab_size": "vocab_size",
    "rope.dimension_count": "head_dim",
    # MoE fields
    "expert_count": "num_local_experts",
    "expert_used_count": "num_experts_per_tok",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_shared_feed_forward_length": "shared_expert_intermediate_size",
    # Hybrid (DeltaNet / Mamba + Attention) fields
    "full_attention_interval": "full_attention_interval",
    # SSM/DeltaNet fields (used for linear attention in hybrid models)
    "ssm.group_count": "linear_num_key_heads",
    "ssm.time_step_rank": "linear_num_value_heads",
    "ssm.conv_kernel": "linear_conv_kernel_dim",
}

_ARCH_KEY_MAPS: dict[str, dict[str, str]] = {
    "glm-dsa": {
        "leading_dense_block_count": "first_k_dense_replace",
        "expert_shared_count": "n_shared_experts",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "expert_weights_scale": "routed_scaling_factor",
        "expert_weights_norm": "norm_topk_prob",
        "expert_gating_func": "expert_gating_func",
        "attention.q_lora_rank": "q_lora_rank",
        "attention.kv_lora_rank": "kv_lora_rank",
        "attention.key_length_mla": "qk_head_dim",
        "attention.value_length_mla": "v_head_dim",
        "attention.indexer.head_count": "index_n_heads",
        "attention.indexer.key_length": "index_head_dim",
        "attention.indexer.top_k": "index_topk",
        "nextn_predict_layers": "num_nextn_predict_layers",
    },
}


# GGUF hidden_act values → HuggingFace activation function names
_ACTIVATION_MAP: dict[str, str] = {
    "gelu": "gelu",
    "silu": "silu",
    "relu": "relu",
    "swiglu": "silu",  # SwiGLU uses SiLU as the gate activation
}


def _get_hf_config_mapping(gguf_arch: str) -> dict[str, str] | None:
    """Try to get HF's GGUF config mapping for an architecture.

    Returns the mapping dict if available, else ``None``.
    """
    try:
        from transformers.integrations.ggml import GGUF_CONFIG_MAPPING

        return GGUF_CONFIG_MAPPING.get(gguf_arch)
    except ImportError:
        return None


def _extract_config_fields(
    gguf_arch: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Extract HuggingFace config fields from GGUF metadata.

    Tries HF's architecture-specific mapping first, then falls back
    to the standard GGUF key names.

    Args:
        gguf_arch: GGUF architecture name (e.g. ``'llama'``).
        metadata: Parsed GGUF metadata dict.

    Returns:
        Dict of HuggingFace config field names → values.
    """
    hf_fields: dict[str, Any] = {}

    # Try HF's mapping first
    hf_mapping = _get_hf_config_mapping(gguf_arch)
    if hf_mapping is not None:
        for gguf_key, hf_key in hf_mapping.items():
            full_key = f"{gguf_arch}.{gguf_key}"
            if full_key in metadata:
                hf_fields[hf_key] = metadata[full_key]
        logger.debug(
            "Used HF GGUF_CONFIG_MAPPING for '%s': %d fields",
            gguf_arch,
            len(hf_fields),
        )
    else:
        # Fallback to standard key names
        for gguf_suffix, hf_key in _DEFAULT_KEY_MAP.items():
            full_key = f"{gguf_arch}.{gguf_suffix}"
            if full_key in metadata:
                hf_fields[hf_key] = metadata[full_key]
        logger.debug(
            "Used default GGUF key mapping for '%s': %d fields",
            gguf_arch,
            len(hf_fields),
        )

    for gguf_suffix, hf_key in _ARCH_KEY_MAPS.get(gguf_arch, {}).items():
        full_key = f"{gguf_arch}.{gguf_suffix}"
        if full_key in metadata:
            hf_fields[hf_key] = metadata[full_key]

    # Extract vocab_size from tokenizer token list if not in metadata
    if "vocab_size" not in hf_fields:
        tokens = metadata.get("tokenizer.ggml.tokens")
        if isinstance(tokens, list):
            hf_fields["vocab_size"] = len(tokens)

    return hf_fields


def gguf_to_config(
    model: Any,  # GGUFModel — typed as Any to avoid circular import
) -> ArchitectureConfig:
    """Convert GGUF metadata to an :class:`ArchitectureConfig`.

    Reads the GGUF architecture name, maps metadata keys to config
    fields, and constructs an ``ArchitectureConfig`` suitable for
    the standard build pipeline.

    Args:
        model: A :class:`GGUFModel` instance.

    Returns:
        An :class:`ArchitectureConfig` populated from the GGUF metadata.

    Raises:
        ValueError: If the GGUF architecture is not recognized or
            if required metadata fields (``hidden_size``,
            ``num_hidden_layers``, ``num_attention_heads``) are missing.
    """
    gguf_arch = model.architecture
    metadata = model.metadata

    # Resolve model_type
    model_type = GGUF_ARCH_TO_MODEL_TYPE.get(gguf_arch, gguf_arch)

    # Extract config fields from metadata
    hf_fields = _extract_config_fields(gguf_arch, metadata)

    # Validate required fields — raise instead of silently defaulting
    required_fields = ("hidden_size", "num_hidden_layers", "num_attention_heads")
    for field in required_fields:
        if field not in hf_fields or hf_fields[field] is None:
            raise ValueError(
                f"GGUF file missing required metadata for '{field}'. Architecture: {gguf_arch}"
            )

    # Derive head_dim if not explicitly provided.
    # Prefer attention.key_length (the actual head dimension) over
    # rope.dimension_count (which may be just the rotary embedding
    # dimension for partial-RoPE models like Qwen3.5).
    head_dim = hf_fields.get("head_dim")
    hidden_size = hf_fields["hidden_size"]
    num_attention_heads = hf_fields["num_attention_heads"]
    key_length = metadata.get(f"{gguf_arch}.attention.key_length")
    if key_length is not None:
        head_dim = int(key_length)
    elif head_dim is None:
        head_dim = hidden_size // num_attention_heads

    # Handle num_key_value_heads defaulting to num_attention_heads.
    # Gemma4 GGUF stores per-layer KV head counts as an array; use the
    # most common (mode) value as the scalar config value.
    num_kv_heads = hf_fields.get("num_key_value_heads", num_attention_heads)
    if isinstance(num_kv_heads, (list, np.ndarray)):
        # Per-layer array → pick the majority value (sliding layers dominate)
        values = list(num_kv_heads)
        num_kv_heads = max(set(values), key=values.count)

    # Map activation function
    hidden_act = hf_fields.get("hidden_act")
    if hidden_act is None:
        # Try GGUF-specific activation key
        act_raw = model.get_metadata(f"{gguf_arch}.feed_forward.activation", None)
        if act_raw is not None:
            hidden_act = _ACTIVATION_MAP.get(act_raw, act_raw)
        else:
            # Default activation by architecture
            hidden_act = _default_activation(model_type)

    # Derive layer_types from full_attention_interval for hybrid models
    full_attention_interval = hf_fields.get("full_attention_interval")
    num_hidden_layers = hf_fields["num_hidden_layers"]
    layer_types: list[str] | None = None
    if full_attention_interval is not None:
        layer_types = [
            "full_attention" if (i + 1) % full_attention_interval == 0 else "linear_attention"
            for i in range(num_hidden_layers)
        ]

    # Derive DeltaNet head dimensions from SSM metadata.
    # ssm.state_size = head dimension for both K and V heads.
    ssm_state_size = metadata.get(f"{gguf_arch}.ssm.state_size")
    linear_key_head_dim = int(ssm_state_size) if ssm_state_size else None
    linear_value_head_dim = int(ssm_state_size) if ssm_state_size else None

    # Derive partial_rotary_factor from rope.dimension_count / head_dim.
    rope_dim = hf_fields.get("head_dim")  # from rope.dimension_count
    if rope_dim is not None and head_dim > 0 and rope_dim != head_dim:
        partial_rotary_factor = rope_dim / head_dim
    else:
        partial_rotary_factor = 1.0

    # Derive rope_interleave from rope.dimension_sections metadata.
    rope_sections = metadata.get(f"{gguf_arch}.rope.dimension_sections")
    rope_interleave = rope_sections is not None and any(s > 0 for s in rope_sections)

    # Derive rope_type from rope.scaling.type. GGUF stores the scaling
    # variant under ``<arch>.rope.scaling.type`` (or omits the key for the
    # plain non-scaled case). ``ArchitectureConfig.rope_type`` defaults to
    # ``None``, which would disable RoPE entirely — but every GGUF whose
    # metadata contains ``rope.freq_base`` is a RoPE model, so we promote
    # the absence/``"none"`` case to the default RoPE variant.
    rope_scaling_type = metadata.get(f"{gguf_arch}.rope.scaling.type")
    rope_freq_base = metadata.get(f"{gguf_arch}.rope.freq_base")
    rope_type: str | None = None
    rope_scaling: dict | None = None
    if rope_freq_base is not None or rope_scaling_type is not None:
        if rope_scaling_type in (None, "none", ""):
            rope_type = "default"
        elif rope_scaling_type in ("linear", "yarn", "longrope"):
            rope_type = rope_scaling_type
        elif rope_scaling_type == "dynamic":
            rope_type = "dynamic"
        else:
            # Unknown scaling type — fall back to default rather than disabling
            # RoPE so the model at least runs.
            logger.warning(
                "Unrecognized GGUF rope.scaling.type=%r; using rope_type='default'",
                rope_scaling_type,
            )
            rope_type = "default"

    # HunYuan-V1-Dense: HF runs dynamic-NTK RoPE with rope_theta=10000 and
    # alpha=1000. The Tencent quantization pipeline bakes those into a
    # static rope.freq_base (~1.1e7) and sets rope.scaling.type='none' in
    # the GGUF. That works for short contexts but diverges for long
    # prompts (the dynamic exponent changes with position). Restore the
    # HF dynamic-NTK config so the ONNX model behaves correctly on
    # long-context inputs.
    if (
        gguf_arch == "hunyuan-dense"
        and rope_type == "default"
        and rope_freq_base is not None
        and float(rope_freq_base) > 1e6
    ):
        logger.info(
            "hunyuan-dense GGUF freq_base=%s exceeds 1e6 — restoring HF "
            "dynamic-NTK RoPE (rope_theta=10000, alpha=1000)",
            rope_freq_base,
        )
        rope_type = "dynamic"
        rope_scaling = {
            "alpha": 1000.0,
            "factor": 1.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "type": "dynamic",
        }
        hf_fields["rope_theta"] = 10000.0

    # Build config — required fields validated above, optional fields
    # use safe defaults
    config = ArchitectureConfig(
        hidden_size=hidden_size,
        intermediate_size=hf_fields.get("intermediate_size", 4 * hidden_size),
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=hf_fields.get("vocab_size", 32000),
        max_position_embeddings=hf_fields.get("max_position_embeddings", 2048),
        rope_theta=hf_fields.get("rope_theta", 10000.0),
        rope_type=rope_type,
        rope_scaling=rope_scaling,
        rms_norm_eps=hf_fields.get("rms_norm_eps", 1e-5),
        hidden_act=hidden_act,
        tie_word_embeddings=_infer_tie_embeddings(model),
        # Projection biases are not in GGUF metadata; infer from tensor
        # presence. Qwen2/Qwen3 carry Q/K/V biases — omitting them breaks
        # attention and yields garbage output.
        attn_qkv_bias=_infer_attn_qkv_bias(model),
        attn_o_bias=_infer_attn_o_bias(model),
        mlp_bias=_infer_mlp_bias(model),
        partial_rotary_factor=partial_rotary_factor,
        rope_interleave=rope_interleave,
        # MoE fields (None when not present → non-MoE model)
        num_local_experts=hf_fields.get("num_local_experts"),
        num_experts_per_tok=hf_fields.get("num_experts_per_tok"),
        moe_intermediate_size=hf_fields.get("moe_intermediate_size"),
        shared_expert_intermediate_size=hf_fields.get("shared_expert_intermediate_size"),
        norm_topk_prob=hf_fields.get("norm_topk_prob", True),
        n_group=hf_fields.get("n_group", 1),
        topk_group=hf_fields.get("topk_group", 1),
        routed_scaling_factor=hf_fields.get("routed_scaling_factor", 1.0),
        scoring_func=hf_fields.get("scoring_func", "softmax"),
        topk_method=hf_fields.get("topk_method", "greedy"),
        first_k_dense_replace=hf_fields.get("first_k_dense_replace", 0),
        n_shared_experts=hf_fields.get("n_shared_experts"),
        # Multi-head Latent Attention fields
        q_lora_rank=hf_fields.get("q_lora_rank"),
        kv_lora_rank=hf_fields.get("kv_lora_rank"),
        qk_nope_head_dim=hf_fields.get("qk_nope_head_dim"),
        qk_rope_head_dim=hf_fields.get("qk_rope_head_dim"),
        v_head_dim=hf_fields.get("v_head_dim"),
        # DSA / MTP metadata
        index_topk=hf_fields.get("index_topk"),
        index_head_dim=hf_fields.get("index_head_dim"),
        index_n_heads=hf_fields.get("index_n_heads"),
        num_nextn_predict_layers=hf_fields.get("num_nextn_predict_layers", 0),
        # Hybrid architecture fields
        layer_types=layer_types,
        full_attention_interval=full_attention_interval,
        # DeltaNet / linear attention fields
        linear_num_key_heads=hf_fields.get("linear_num_key_heads"),
        linear_num_value_heads=hf_fields.get("linear_num_value_heads"),
        linear_key_head_dim=linear_key_head_dim,
        linear_value_head_dim=linear_value_head_dim,
        linear_conv_kernel_dim=(hf_fields.get("linear_conv_kernel_dim") or 4),
    )

    # Store model_type for registry lookup and tensor processor dispatch.
    config._gguf_model_type = model_type
    config.model_type = model_type

    # Apply architecture-specific postprocessing to produce the correct
    # config subclass (e.g. Gemma4Config instead of plain ArchitectureConfig).
    postprocessor = _CONFIG_POSTPROCESSORS.get(model_type)
    if postprocessor is not None:
        config = postprocessor(config, metadata)
        config._gguf_model_type = model_type
        config.model_type = model_type

    logger.info(
        "Extracted config from GGUF: arch=%s, model_type=%s, "
        "hidden=%d, layers=%d, heads=%d, vocab=%d",
        gguf_arch,
        model_type,
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.vocab_size,
    )

    return config


def _gemma4_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
) -> Gemma4Config:
    """Convert a base config to Gemma4Config with architecture-specific fields.

    Gemma4 GGUF metadata uses dual-regime keys (global/full-attention vs
    SWA/sliding-window) for head_dim, RoPE theta, and RoPE rotary count.
    The base extractor picks up the global values; this postprocessor
    corrects them for the sliding-window default and stores the global
    values in Gemma4Config's dedicated fields.

    GGUF key mapping (``gemma4.`` prefix omitted for readability):

    =================================== ======================================
    GGUF key                            Gemma4Config / ArchitectureConfig field
    =================================== ======================================
    attention.key_length                global_head_dim
    attention.key_length_swa            head_dim (sliding-window, default)
    rope.freq_base                      global_rope_theta
    rope.freq_base_swa                  rope_theta (sliding-window, default)
    rope.dimension_count                (global rotary dim → partial_rotary_factor)
    rope.dimension_count_swa            (local rotary dim — full rotation)
    attention.sliding_window            sliding_window
    attention.sliding_window_pattern    layer_types (bool[] → str[])
    final_logit_softcapping             final_logit_softcapping
    attention.shared_kv_layers          num_kv_shared_layers
    embedding_length_per_layer_input    hidden_size_per_layer_input
    =================================== ======================================
    """
    arch = "gemma4"

    # --- Dual head_dim ---
    # Base extractor sets head_dim from attention.key_length (global, 512).
    # Override with sliding-window head_dim (256) as the default.
    swa_head_dim = metadata.get(f"{arch}.attention.key_length_swa")
    global_head_dim = metadata.get(f"{arch}.attention.key_length")
    if swa_head_dim is not None:
        config = dataclasses.replace(config, head_dim=int(swa_head_dim))
    elif global_head_dim is not None:
        logger.warning(
            "GGUF file missing non-standard key '%s.attention.key_length_swa'. "
            "Using global head_dim (%d) for sliding-window layers — "
            "this may be incorrect for Gemma4.",
            arch,
            int(global_head_dim),
        )

    # --- Dual RoPE theta ---
    # Base extractor sets rope_theta from rope.freq_base (global, 1M).
    # Override with sliding-window theta (10K) as the default.
    swa_rope_theta = metadata.get(f"{arch}.rope.freq_base_swa")
    global_rope_theta = metadata.get(f"{arch}.rope.freq_base")
    if swa_rope_theta is not None:
        config = dataclasses.replace(config, rope_theta=float(swa_rope_theta))
    elif global_rope_theta is not None:
        logger.warning(
            "GGUF file missing non-standard key '%s.rope.freq_base_swa'. "
            "Using global rope_theta (%.1f) for sliding-window layers — "
            "this may be incorrect for Gemma4.",
            arch,
            float(global_rope_theta),
        )

    # --- Partial rotary factor ---
    # Global layers use partial rotation: rotary_dim / global_head_dim.
    # GGUF provides rope.dimension_count (global rotary dim, e.g. 512)
    # but the actual HF partial_rotary_factor is 0.25, meaning only 128
    # of 512 dims are rotated.  This isn't directly in GGUF metadata,
    # so use the known Gemma4 default.
    # Source: HF Gemma4Config.global_partial_rotary_factor default value
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/configuration_gemma4.py
    global_partial_rotary_factor = 0.25

    # SWA layers use full rotation (partial_rotary_factor = 1.0), which
    # is already the base config default.  Reset the base partial_rotary_factor
    # to 1.0 since the base extractor may have derived it incorrectly.
    config = dataclasses.replace(config, partial_rotary_factor=1.0)

    # --- Layer types from sliding_window_pattern ---
    # GGUF stores a bool array: True = sliding, False = full_attention
    sliding_pattern = metadata.get(f"{arch}.attention.sliding_window_pattern")
    layer_types: list[str] | None = None
    if sliding_pattern is not None:
        if len(sliding_pattern) != config.num_hidden_layers:
            raise ValueError(
                f"GGUF metadata length mismatch: "
                f"attention.sliding_window_pattern has "
                f"{len(sliding_pattern)} entries but "
                f"num_hidden_layers is {config.num_hidden_layers}."
            )
        layer_types = [
            "sliding_attention" if is_sliding else "full_attention"
            for is_sliding in sliding_pattern
        ]
    config = dataclasses.replace(config, layer_types=layer_types)

    # --- Sliding window size ---
    sliding_window = metadata.get(f"{arch}.attention.sliding_window")
    if sliding_window is not None:
        config = dataclasses.replace(config, sliding_window=int(sliding_window))

    # --- Softcapping ---
    final_logit_softcapping = metadata.get(f"{arch}.final_logit_softcapping")
    attn_logit_softcapping = metadata.get(f"{arch}.attention.logit_softcapping")

    # --- KV sharing ---
    num_kv_shared_layers = metadata.get(f"{arch}.attention.shared_kv_layers")

    # --- Per-layer input gating ---
    hidden_size_per_layer_input = metadata.get(f"{arch}.embedding_length_per_layer_input")

    # --- Double-wide MLP (per-layer feed-forward length) ---
    # Gemma4 E2B/E4B store feed_forward_length as a per-layer array: the base
    # intermediate size for standalone layers and 2x that for the KV-shared
    # layers (use_double_wide_mlp).  Gemma4DecoderLayer expects a scalar base
    # size and re-derives the doubling from use_double_wide_mlp + is_kv_shared,
    # so collapse the array back to (base, use_double_wide_mlp) here.
    use_double_wide_mlp = False
    intermediate_size = config.intermediate_size
    if isinstance(intermediate_size, (list, np.ndarray)):
        values = [int(v) for v in intermediate_size]
        distinct = sorted(set(values))
        if len(distinct) == 1:
            intermediate_size = distinct[0]
        elif len(distinct) == 2 and distinct[1] == 2 * distinct[0]:
            base, wide = distinct
            shared = int(num_kv_shared_layers) if num_kv_shared_layers is not None else 0
            first_shared = config.num_hidden_layers - shared
            expected = [
                wide if (shared > 0 and i >= first_shared) else base
                for i in range(config.num_hidden_layers)
            ]
            if values != expected:
                raise ValueError(
                    "Gemma4 per-layer feed_forward_length does not match the "
                    "double-wide-MLP pattern (wide layers must be the last "
                    f"{shared} KV-shared layers): {values}"
                )
            intermediate_size = base
            use_double_wide_mlp = True
        else:
            raise ValueError(
                f"Unexpected Gemma4 per-layer feed_forward_length array: {values}"
            )
        config = dataclasses.replace(config, intermediate_size=intermediate_size)

    # --- Per-layer KV heads (num_global_key_value_heads) ---
    # GGUF stores per-layer KV head counts as an array.  When full-attention
    # layers use fewer KV heads than sliding layers, extract the minority
    # value as num_global_key_value_heads.
    num_global_key_value_heads: int | None = None
    raw_kv_heads = metadata.get(f"{arch}.attention.head_count_kv")
    if isinstance(raw_kv_heads, (list, np.ndarray)) and sliding_pattern is not None:
        if len(raw_kv_heads) != len(sliding_pattern):
            raise ValueError(
                f"GGUF metadata length mismatch: "
                f"attention.head_count_kv has {len(raw_kv_heads)} entries "
                f"but attention.sliding_window_pattern has "
                f"{len(sliding_pattern)} entries. "
                f"Both must equal num_hidden_layers."
            )
        full_kv_heads = {
            int(raw_kv_heads[i])
            for i, is_sliding in enumerate(sliding_pattern)
            if not is_sliding
        }
        if len(full_kv_heads) == 1:
            global_kv = full_kv_heads.pop()
            if global_kv != config.num_key_value_heads:
                num_global_key_value_heads = global_kv

    return Gemma4Config(
        # Inherit all base ArchitectureConfig fields
        **{f.name: getattr(config, f.name) for f in dataclasses.fields(ArchitectureConfig)},
        # Gemma4-specific fields
        global_head_dim=int(global_head_dim) if global_head_dim is not None else None,
        global_rope_theta=float(global_rope_theta)
        if global_rope_theta is not None
        else 1_000_000.0,
        global_partial_rotary_factor=global_partial_rotary_factor,
        num_global_key_value_heads=num_global_key_value_heads,
        # attention_k_eq_v: derive from per-layer KV head counts. When
        # full-attention layers use fewer KV heads, V = K (no v_proj).
        attention_k_eq_v=num_global_key_value_heads is not None,
        final_logit_softcapping=float(final_logit_softcapping or 0.0),
        attn_logit_softcapping=float(attn_logit_softcapping or 0.0),
        num_kv_shared_layers=int(num_kv_shared_layers)
        if num_kv_shared_layers is not None
        else 0,
        hidden_size_per_layer_input=int(hidden_size_per_layer_input)
        if hidden_size_per_layer_input is not None
        else 0,
        use_double_wide_mlp=use_double_wide_mlp,
        # Fields without GGUF metadata — use Gemma4Config defaults
        vocab_size_per_layer_input=config.vocab_size
        if (hidden_size_per_layer_input or 0) > 0
        else 0,
    )


# Gemma3 interleaves sliding-window (local) and full (global) attention. The
# local RoPE base frequency is a fixed architectural constant that GGUF does not
# store (only the global ``gemma3.rope.freq_base`` is present), so it must be
# defaulted. Source: HF Gemma3TextConfig.rope_local_base_freq default value.
_GEMMA3_DEFAULT_ROPE_LOCAL_BASE_FREQ = 10_000.0

# Gemma3 interleaves sliding-window (local) and full (global) attention on a
# fixed period: every ``sliding_window_pattern``-th layer is full attention.
# GGUF stores neither the per-layer types nor the period, so default to the HF
# Gemma3 value. Source: HF Gemma3TextConfig.sliding_window_pattern default.
_GEMMA3_DEFAULT_SLIDING_WINDOW_PATTERN = 6


def _gemma3_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
) -> ArchitectureConfig:
    """Populate Gemma3 fields that GGUF omits.

    GGUF carries only the global RoPE base (``gemma3.rope.freq_base``) and the
    sliding-window size, but not the local RoPE base or the per-layer
    local/global attention pattern that ``Gemma3TextModel`` requires. Without
    them the model builds its local rotary embedding from
    ``rope_local_base_freq = None`` (crash) and iterates ``layer_types = None``
    (crash). Default both to the known Gemma3 constants when GGUF does not
    provide them.

    Args:
        config: The base config extracted from GGUF metadata.
        metadata: The raw GGUF key-value metadata.

    Returns:
        The config with ``rope_local_base_freq`` and ``layer_types`` populated.
    """
    if getattr(config, "rope_local_base_freq", None) is None:
        local_freq_base = metadata.get("gemma3.rope.local_freq_base") or metadata.get(
            "gemma3.rope.freq_base_local"
        )
        config.rope_local_base_freq = (
            float(local_freq_base)
            if local_freq_base is not None
            else _GEMMA3_DEFAULT_ROPE_LOCAL_BASE_FREQ
        )
    if getattr(config, "layer_types", None) is None:
        pattern = (
            metadata.get("gemma3.attention.sliding_window_pattern")
            or _GEMMA3_DEFAULT_SLIDING_WINDOW_PATTERN
        )
        config.layer_types = [
            "full_attention" if (index + 1) % pattern == 0 else "sliding_attention"
            for index in range(config.num_hidden_layers)
        ]
    return config


def _glm_moe_dsa_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
) -> ArchitectureConfig:
    """Restore GLM-5.2 fields that GGUF represents with MLA-specific keys."""
    arch = "glm-dsa"
    n_mtp = int(metadata.get(f"{arch}.nextn_predict_layers", 0))
    num_hidden_layers = config.num_hidden_layers - n_mtp
    qk_head_dim = int(metadata[f"{arch}.attention.key_length_mla"])
    qk_rope_head_dim = int(metadata[f"{arch}.rope.dimension_count"])
    qk_nope_head_dim = qk_head_dim - qk_rope_head_dim
    n_shared_experts = int(metadata.get(f"{arch}.expert_shared_count", 0))
    first_k_dense_replace = int(metadata.get(f"{arch}.leading_dense_block_count", 0))

    # GLM-5.2 shares one full indexer across each four-layer DSA group.
    indexer_types = [
        "full" if i < first_k_dense_replace or (i - 2) % 4 == 0 else "shared"
        for i in range(num_hidden_layers)
    ]
    mlp_layer_types = [
        "dense" if i < first_k_dense_replace else "sparse" for i in range(num_hidden_layers)
    ]

    return dataclasses.replace(
        config,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=config.num_attention_heads,
        head_dim=qk_rope_head_dim,
        partial_rotary_factor=1.0,
        rope_interleave=True,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=int(metadata[f"{arch}.attention.value_length_mla"]),
        q_lora_rank=int(metadata[f"{arch}.attention.q_lora_rank"]),
        kv_lora_rank=int(metadata[f"{arch}.attention.kv_lora_rank"]),
        n_shared_experts=n_shared_experts,
        shared_expert_intermediate_size=(
            config.moe_intermediate_size * n_shared_experts
            if config.moe_intermediate_size is not None
            else None
        ),
        first_k_dense_replace=first_k_dense_replace,
        mlp_layer_types=mlp_layer_types,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        indexer_types=indexer_types,
        num_nextn_predict_layers=n_mtp,
    )


# Architecture-specific config postprocessors.
# Each takes a base ArchitectureConfig + raw metadata and returns
# an architecture-specific config subclass.
_CONFIG_POSTPROCESSORS: dict[str, Any] = {
    "gemma3_text": _gemma3_postprocess,
    "gemma4_text": _gemma4_postprocess,
    "glm_moe_dsa": _glm_moe_dsa_postprocess,
}


def _default_activation(model_type: str) -> str:
    """Return the default activation function for a model type."""
    # Most modern models use SiLU/Swish
    gelu_models = {"gpt2", "bloom", "starcoder2", "t5"}
    if model_type in gelu_models:
        return "gelu"
    return "silu"


def _infer_tie_embeddings(model: Any) -> bool:
    """Infer tie_word_embeddings from tensor presence.

    If the GGUF file has no ``output.weight`` tensor, the
    model likely ties embeddings (shares ``token_embd.weight``
    for both input and output).
    """
    return "output.weight" not in model.tensor_names


def _infer_attn_qkv_bias(model: Any) -> bool:
    """Infer whether Q/K/V projections have bias from tensor presence.

    llama.cpp names attention projection biases ``blk.N.attn_{q,k,v}.bias``
    (or a fused ``blk.N.attn_qkv.bias``). Models such as Qwen2/Qwen3 carry
    these biases; if the config default (``False``) is used instead, the
    graph builder omits the bias ``Add`` after each projection and the
    model produces garbage output.
    """
    return any(
        n.endswith(("attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_qkv.bias"))
        for n in model.tensor_names
    )


def _infer_attn_o_bias(model: Any) -> bool:
    """Infer whether the attention output projection has a bias tensor."""
    return any(n.endswith("attn_output.bias") for n in model.tensor_names)


def _infer_mlp_bias(model: Any) -> bool:
    """Infer whether MLP/FFN projections have bias tensors."""
    return any(
        n.endswith(("ffn_up.bias", "ffn_down.bias", "ffn_gate.bias"))
        for n in model.tensor_names
    )
