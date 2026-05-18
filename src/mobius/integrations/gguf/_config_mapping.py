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
        rms_norm_eps=hf_fields.get("rms_norm_eps", 1e-5),
        hidden_act=hidden_act,
        tie_word_embeddings=_infer_tie_embeddings(model),
        partial_rotary_factor=partial_rotary_factor,
        rope_interleave=rope_interleave,
        # MoE fields (None when not present → non-MoE model)
        num_local_experts=hf_fields.get("num_local_experts"),
        num_experts_per_tok=hf_fields.get("num_experts_per_tok"),
        moe_intermediate_size=hf_fields.get("moe_intermediate_size"),
        shared_expert_intermediate_size=hf_fields.get("shared_expert_intermediate_size"),
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
        # Fields without GGUF metadata — use Gemma4Config defaults
        vocab_size_per_layer_input=config.vocab_size
        if (hidden_size_per_layer_input or 0) > 0
        else 0,
    )


# Architecture-specific config postprocessors.
# Each takes a base ArchitectureConfig + raw metadata and returns
# an architecture-specific config subclass.
_CONFIG_POSTPROCESSORS: dict[str, Any] = {
    "gemma4_text": _gemma4_postprocess,
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
