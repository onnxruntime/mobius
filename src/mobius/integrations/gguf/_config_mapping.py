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

__all__ = ["gguf_to_config", "resolve_model_type", "assert_glm_moe_dsa_resolvable"]

import dataclasses
import logging
from types import MappingProxyType
from typing import Any

import numpy as np

from mobius._configs import (
    ArchitectureConfig,
    Gemma2Config,
    Gemma4Config,
    MuseGlimmerConfig,
    _shallow_fields,
)
from mobius.integrations.gguf._arch_registry import iter_arch_specs, try_get_arch_spec

logger = logging.getLogger(__name__)


# Map GGUF architecture names → our registry model_type strings.
#
# Derived from :mod:`mobius.integrations.gguf._arch_registry`, which is the
# single source of truth. It stays exported under this name because callers
# outside this module read it, but it is now read-only: mutating it here would
# desynchronize it from the tensor mapping, the weight processors, and the
# capability verdicts that are all built from the same specs.
GGUF_ARCH_TO_MODEL_TYPE: MappingProxyType[str, str] = MappingProxyType(
    {
        name: spec.model_type
        for spec in iter_arch_specs()
        if spec.model_type is not None
        for name in sorted(spec.names)
    }
)


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
    "attention.layer_norm_epsilon": "rms_norm_eps",
    "rope.freq_base": "rope_theta",
    "context_length": "max_position_embeddings",
    "vocab_size": "vocab_size",
    "rope.dimension_count": "head_dim",
    "attention.sliding_window": "sliding_window",
    "logit_scale": "logit_scale",
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

_MUSE_GLIMMER_KEY_MAP = {
    "attention.key_length": "head_dim",
    "attention.sliding_window": "sliding_window",
}

_DEEPSEEK4_KEY_MAP = {
    "attention.key_length": "head_dim",
    "rope.dimension_count": "qk_rope_head_dim",
    "attention.q_lora_rank": "q_lora_rank",
    "attention.sliding_window": "sliding_window",
    "expert_count": "num_local_experts",
    "expert_used_count": "num_experts_per_tok",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_shared_count": "n_shared_experts",
    "expert_weights_scale": "routed_scaling_factor",
    "expert_weights_norm": "norm_topk_prob",
    "swiglu_clamp_exp": "swiglu_limit",
    "attention.indexer.head_count": "index_n_heads",
    "attention.indexer.key_length": "index_head_dim",
    "attention.indexer.top_k": "index_topk",
    "attention.output_group_count": "o_groups",
    "attention.output_lora_rank": "o_lora_rank",
    "attention.compress_ratios": "compress_ratios",
    "attention.compress_rope_freq_base": "compress_rope_theta",
    "hyper_connection.count": "hc_mult",
    "hyper_connection.sinkhorn_iterations": "hc_sinkhorn_iters",
    "hyper_connection.epsilon": "hc_eps",
    "hash_layer_count": "num_hash_layers",
}

# GLM-5.2 ('glm-dsa') shares DeepSeek's MLA + MoE + DSA-indexer metadata layout
# but omits the DeepSeek-V4-only hyper-connection / hash-routing / output-group
# extensions. Keep the map to the discriminating MLA/MoE/DSA keys only, so the
# extracted config matches what GlmMoeDsaCausalLMModel (a DeepSeek-V3 subclass)
# consumes. Both spellings of the architecture string are accepted.
_GLM_DSA_KEY_MAP = {
    "attention.key_length": "head_dim",
    "rope.dimension_count": "qk_rope_head_dim",
    "attention.q_lora_rank": "q_lora_rank",
    "attention.kv_lora_rank": "kv_lora_rank",
    "attention.sliding_window": "sliding_window",
    "expert_count": "num_local_experts",
    "expert_used_count": "num_experts_per_tok",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_shared_count": "n_shared_experts",
    "expert_weights_scale": "routed_scaling_factor",
    "expert_weights_norm": "norm_topk_prob",
    "attention.indexer.head_count": "index_n_heads",
    "attention.indexer.key_length": "index_head_dim",
    "attention.indexer.top_k": "index_topk",
}

#: Named architecture-specific key maps that :attr:`GGUFArchitectureSpec.
#: config_key_map` selects. Every name here must be referenced by a spec and
#: every name a spec references must exist here; ``_arch_registry_test`` checks
#: both directions so a typo cannot silently drop config fields.
_KEY_MAP_TABLES: MappingProxyType[str, dict[str, str]] = MappingProxyType(
    {
        "muse_glimmer": _MUSE_GLIMMER_KEY_MAP,
        "deepseek4": _DEEPSEEK4_KEY_MAP,
        "glm_dsa": _GLM_DSA_KEY_MAP,
    }
)


def _arch_key_map(gguf_arch: str) -> dict[str, str]:
    """Return the extra GGUF-key → config-field map for *gguf_arch*."""
    spec = try_get_arch_spec(gguf_arch)
    if spec is None or spec.config_key_map is None:
        return {}
    return _KEY_MAP_TABLES[spec.config_key_map]


#: Per-architecture key maps expanded over every canonical name and alias.
#: Derived from the registry rather than declared, so a spelling accepted by the
#: tensor mapping cannot be missing here.
_ARCH_KEY_MAPS: MappingProxyType[str, dict[str, str]] = MappingProxyType(
    {
        name: _KEY_MAP_TABLES[spec.config_key_map]
        for spec in iter_arch_specs()
        if spec.config_key_map is not None
        for name in sorted(spec.names)
    }
)


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
    # Always apply standard and mobius architecture-specific mappings. This
    # fills fields omitted by Transformers mappings and supports new GGUF
    # architectures before they are added upstream.
    fallback_mapping = {
        **_DEFAULT_KEY_MAP,
        **_arch_key_map(gguf_arch),
    }
    for gguf_suffix, hf_key in fallback_mapping.items():
        full_key = f"{gguf_arch}.{gguf_suffix}"
        if full_key in metadata:
            hf_fields[hf_key] = metadata[full_key]
    if hf_mapping is None:
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

    # Resolve the architecture spec once. It is deliberately *not* required
    # here: config extraction is a separate capability from tensor mapping, and
    # some architectures (bloom, t5) can be configured but not mapped. The
    # tensor-mapping gate raises for those, with a reason.
    spec = try_get_arch_spec(gguf_arch)
    canonical_arch = spec.gguf_arch if spec is not None else gguf_arch
    if spec is not None:
        missing_metadata = [
            suffix
            for suffix in spec.required_metadata
            if f"{gguf_arch}.{suffix}" not in metadata
        ]
        if missing_metadata:
            raise ValueError(
                f"GGUF architecture {gguf_arch!r} is missing required metadata: "
                f"{', '.join(missing_metadata)}"
            )

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

    # Exclude Multi-Token-Prediction (MTP / "nextn") blocks from the decoder
    # layer count. GGUF's ``block_count`` counts the trailing MTP prediction
    # block(s) alongside the regular decoder layers (e.g. Qwen3.5/3.8 store
    # ``block_count = num_hidden_layers + nextn_predict_layers``), but the base
    # decode model does not build them. Their weights (``blk.<n>.nextn.*`` and
    # the accompanying attention/FFN tensors of the trailing block) are skipped
    # during tensor mapping. Without this correction the builder would create
    # an extra decoder layer whose linear-attention / GQA initializers have no
    # backing GGUF weights and fail the ``_check_weights`` invariant on save.
    nextn_layers = metadata.get(f"{gguf_arch}.nextn_predict_layers")
    mtp_predict_layers = 0
    mtp_block_indices: list[int] = []
    if nextn_layers is not None and int(nextn_layers) > 0:
        mtp_count = int(nextn_layers)
        if mtp_count > 1:
            raise ValueError(
                f"GGUF architecture {gguf_arch!r} declares nextn_predict_layers="
                f"{mtp_count}, but mobius can export exactly one MTP sidecar head. "
                "Multi-head MTP export is not supported; use a checkpoint with one "
                "nextn prediction layer."
            )
        decoder_layers = int(hf_fields["num_hidden_layers"]) - mtp_count
        if decoder_layers <= 0:
            raise ValueError(
                f"GGUF metadata inconsistent: block_count "
                f"({hf_fields['num_hidden_layers']}) <= nextn_predict_layers "
                f"({mtp_count}) for architecture {gguf_arch}."
            )
        hf_fields["num_hidden_layers"] = decoder_layers
        # The trailing ``mtp_count`` GGUF blocks (indices ``decoder_layers`` ..
        # ``decoder_layers + mtp_count - 1``) hold the self-speculative MTP head
        # weights (``blk.<n>.nextn.*`` plus the head's own attention/FFN block).
        # Surface them so the builder can emit the MTP sidecar instead of
        # silently dropping the tensors.
        mtp_predict_layers = mtp_count
        mtp_block_indices = list(range(decoder_layers, decoder_layers + mtp_count))

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
    elif canonical_arch == "cohere2":
        # Cohere2's rope.dimension_count is the rotated prefix, not the full
        # attention head width. The graph still projects hidden_size / heads.
        head_dim = hidden_size // num_attention_heads
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
    rope_dim = metadata.get(f"{gguf_arch}.rope.dimension_count")
    if rope_dim is not None and head_dim > 0 and rope_dim != head_dim:
        partial_rotary_factor = int(rope_dim) / head_dim
    else:
        partial_rotary_factor = 1.0

    # Derive rope_interleave.
    #
    # ``rope.dimension_sections`` encodes M-RoPE *section* sizes (the Qwen-VL
    # family splits the rotary dimension across the temporal/height/width
    # position axes, e.g. ``[11, 11, 10, 0]``). It does NOT select the GPT-J
    # style adjacent-pair rotation that the flat ``rope_interleave`` flag
    # controls: Qwen3.5 (and every other section-carrying arch here) rotates
    # with split-half (NEOX / ``rotate_half``) semantics. Deriving
    # ``rope_interleave`` from section presence therefore corrupts RoPE — the
    # exported GroupQueryAttention/RotaryEmbedding gets ``rotary_interleaved=1``
    # and the full-attention layers produce garbage tokens. Section interleave,
    # when a model needs it, is a distinct ``mrope_interleaved`` signal handled
    # via ``mrope_section``. Only architectures that genuinely use adjacent-pair
    # rotation declare the flag on their spec.
    rope_interleave = spec is not None and spec.rope_interleave

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

    if rope_type == "yarn":
        rope_scaling = {
            "type": "yarn",
            "factor": metadata.get(f"{gguf_arch}.rope.scaling.factor", 1.0),
            "original_max_position_embeddings": metadata.get(
                f"{gguf_arch}.rope.scaling.original_context_length"
            ),
            "beta_fast": metadata.get(f"{gguf_arch}.rope.scaling.yarn_beta_fast", 32.0),
            "beta_slow": metadata.get(f"{gguf_arch}.rope.scaling.yarn_beta_slow", 1.0),
        }

    # HunYuan-V1-Dense: HF runs dynamic-NTK RoPE with rope_theta=10000 and
    # alpha=1000. The Tencent quantization pipeline bakes those into a
    # static rope.freq_base (~1.1e7) and sets rope.scaling.type='none' in
    # the GGUF. That works for short contexts but diverges for long
    # prompts (the dynamic exponent changes with position). Restore the
    # HF dynamic-NTK config so the ONNX model behaves correctly on
    # long-context inputs.
    if (
        canonical_arch == "hunyuan-dense"
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
    swiglu_limit = hf_fields.get("swiglu_limit", 0.0)
    if isinstance(swiglu_limit, (list, np.ndarray)):
        swiglu_limit = swiglu_limit[0] if len(swiglu_limit) else 0.0

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
        n_shared_experts=hf_fields.get("n_shared_experts"),
        norm_topk_prob=hf_fields.get("norm_topk_prob", True),
        routed_scaling_factor=hf_fields.get("routed_scaling_factor", 1.0),
        scoring_func=(
            "sqrtsoftplus"
            if canonical_arch == "deepseek4"
            else hf_fields.get("scoring_func", "softmax")
        ),
        q_lora_rank=hf_fields.get("q_lora_rank"),
        qk_rope_head_dim=hf_fields.get("qk_rope_head_dim"),
        o_groups=hf_fields.get("o_groups", 1),
        o_lora_rank=hf_fields.get("o_lora_rank"),
        index_n_heads=hf_fields.get("index_n_heads"),
        index_head_dim=hf_fields.get("index_head_dim"),
        index_topk=hf_fields.get("index_topk"),
        compress_ratios=hf_fields.get("compress_ratios"),
        compress_rope_theta=hf_fields.get("compress_rope_theta"),
        hc_mult=hf_fields.get("hc_mult", 1),
        hc_sinkhorn_iters=hf_fields.get("hc_sinkhorn_iters", 1),
        hc_eps=hf_fields.get("hc_eps", 1e-6),
        num_hash_layers=hf_fields.get("num_hash_layers", 0),
        swiglu_limit=swiglu_limit,
        sliding_window=hf_fields.get("sliding_window"),
        logit_scale=hf_fields.get("logit_scale", 1.0),
        original_max_position_embeddings=(
            rope_scaling.get("original_max_position_embeddings")
            if rope_scaling is not None
            else None
        ),
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

    # Store the source architecture and model_type for registry lookup and for
    # weight-processor dispatch. ``_gguf_arch`` is what lets downstream
    # dispatch key on the architecture spec instead of guessing from
    # ``model_type``, which is where the Gemma 3 processor used to get lost.
    config._gguf_arch = spec.gguf_arch if spec is not None else gguf_arch
    config._gguf_model_type = model_type
    config.model_type = model_type

    # Apply architecture-specific postprocessing to produce the correct
    # config subclass (e.g. Gemma4Config instead of plain ArchitectureConfig).
    # Postprocessors take the GGUF model too, because a few architectures store
    # config scalars inside tensors rather than in the key-value metadata.
    postprocessor_name = None if spec is None else spec.config_postprocessor
    if postprocessor_name is not None:
        postprocessor = _CONFIG_POSTPROCESSORS[postprocessor_name]
        config = postprocessor(config, metadata, model)
        config._gguf_arch = spec.gguf_arch
        config._gguf_model_type = model_type
        config.model_type = model_type

    # Re-surface after any postprocessor swap so the MTP metadata survives on
    # the final config instance.
    config._gguf_nextn_predict_layers = mtp_predict_layers
    config._gguf_mtp_block_indices = mtp_block_indices

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


def _gemma2_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> Gemma2Config:
    """Convert a base config to Gemma2Config with architecture-specific fields.

    Gemma2 applies tanh soft-capping to both the attention logits and the final
    logits, and uses ``query_pre_attn_scalar`` for the attention scale. These
    fields live on :class:`Gemma2Config`; without this postprocessor the GGUF
    path hands the Gemma2 module a plain :class:`ArchitectureConfig` and crashes
    in ``Gemma2Attention.__init__`` on ``config.query_pre_attn_scalar`` /
    ``config.attn_logit_softcapping``.

    GGUF key mapping (``gemma2.`` prefix omitted for readability):

    =================================== ======================================
    GGUF key                            Gemma2Config field
    =================================== ======================================
    attn_logit_softcapping              attn_logit_softcapping
    final_logit_softcapping             final_logit_softcapping
    attention.key_length                head_dim (already on the base config)
    =================================== ======================================

    GGUF does not carry ``query_pre_attn_scalar``. For every released Gemma2
    checkpoint except 27B it equals ``head_dim``, so the default
    ``1/sqrt(head_dim)`` scale is numerically exact; we therefore leave it
    ``None`` (which selects that default) unless a checkpoint provides the key.
    """
    arch = "gemma2"
    attn_logit_softcapping = metadata.get(f"{arch}.attn_logit_softcapping")
    final_logit_softcapping = metadata.get(f"{arch}.final_logit_softcapping")
    query_pre_attn_scalar = metadata.get(f"{arch}.attention.query_pre_attn_scalar")

    # Gemma2 always uses standard (default) RoPE, but its GGUF omits every
    # ``rope.*`` key (llama.cpp hardcodes theta=10000). The base extractor only
    # promotes ``rope_type`` to ``"default"`` when ``rope.freq_base`` is present,
    # so it is left as ``None`` here — which would disable RoPE and make
    # ``initialize_rope`` return ``None``. Force the default variant; the base
    # ``rope_theta`` default (10000.0) already matches the architecture.
    if config.rope_type is None:
        config = dataclasses.replace(config, rope_type="default")

    return Gemma2Config(
        # Inherit all base ArchitectureConfig fields
        **{f.name: getattr(config, f.name) for f in dataclasses.fields(ArchitectureConfig)},
        # Gemma2-specific fields
        attn_logit_softcapping=float(attn_logit_softcapping or 0.0),
        final_logit_softcapping=float(final_logit_softcapping or 0.0),
        query_pre_attn_scalar=float(query_pre_attn_scalar)
        if query_pre_attn_scalar is not None
        else None,
    )


def _gemma4_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
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
    model: Any = None,
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


def _olmo_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> ArchitectureConfig:
    """Reject OLMo variants whose QKV clamp the current graph cannot express."""
    clamp = metadata.get("olmo.attention.clamp_kqv")
    if clamp is not None and float(clamp) > 0:
        raise ValueError(
            "GGUF metadata olmo.attention.clamp_kqv is non-zero, but mobius's "
            "OLMo attention graph does not implement QKV activation clamping."
        )
    return config


def _dense_sliding_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> ArchitectureConfig:
    """Apply dense-architecture attention semantics omitted by generic GGUF metadata."""
    arch = config._gguf_arch
    if arch == "olmo2":
        config.attn_qk_norm = True
        config.attn_qk_norm_full = True
    elif arch == "cohere2":
        # Cohere2 rotates adjacent even/odd pairs rather than split halves.
        config.rope_interleave = True
        config.layer_types = [
            "full_attention" if (layer_index + 1) % 4 == 0 else "sliding_attention"
            for layer_index in range(config.num_hidden_layers)
        ]
        config.no_rope_layers = [
            0 if layer_type == "full_attention" else 1 for layer_type in config.layer_types
        ]
    elif arch == "smollm3":
        # llama.cpp does not serialize no_rope_layer_interval. SmolLM3 fixes it
        # at four: every fourth layer skips RoPE.
        config.no_rope_layers = [
            0 if (layer_index + 1) % 4 == 0 else 1
            for layer_index in range(config.num_hidden_layers)
        ]

    pattern = metadata.get(f"{arch}.attention.sliding_window_pattern")
    if pattern is None:
        return config
    if arch == "olmo2":
        raise ValueError(
            "GGUF architecture olmo2 carries an attention.sliding_window_pattern "
            "(OLMo3 semantics), but mobius's OLMo2 graph does not yet implement "
            "per-layer sliding attention."
        )
    if not isinstance(pattern, (list, tuple, np.ndarray)):
        raise TypeError(
            f"GGUF metadata {arch}.attention.sliding_window_pattern must be a "
            f"per-layer bool array, got {type(pattern).__name__}."
        )
    if len(pattern) != config.num_hidden_layers:
        raise ValueError(
            f"GGUF metadata {arch}.attention.sliding_window_pattern has "
            f"{len(pattern)} entries for {config.num_hidden_layers} layers."
        )
    config.layer_types = [
        "sliding_attention" if bool(is_sliding) else "full_attention" for is_sliding in pattern
    ]
    return config


def _muse_glimmer_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> MuseGlimmerConfig:
    """Convert a base config to :class:`MuseGlimmerConfig`.

    GGUF key mapping (``muse-glimmer.`` prefix omitted for readability):

    ===================================  ====================================
    GGUF key                             MuseGlimmerConfig field
    ===================================  ====================================
    logit_scale                          output_multiplier
    final_logit_softcapping              final_logit_softcapping
    attention.sliding_window             sliding_window
    attention.sliding_window_pattern     layer_types, layer_rope_theta
    ===================================  ====================================

    Two values are not in the metadata at all:

    ``qk_scale_factor``
        Recovered from the ``blk.0.attn_q_norm`` tensor. Muse Glimmer's QK
        normalization is scale-free, so llama.cpp materializes the constant as
        a head_dim-wide vector rather than storing a scalar in the metadata.

    ``post_norm_eps``
        Not represented in GGUF; the dataclass default (1e-8, matching the
        published checkpoints) is kept.
    """
    arch = getattr(model, "architecture", None) or "muse-glimmer"

    # --- Layer types and NoPE layers from the sliding-window pattern ---
    # Unlike Gemma 4, which stores a per-layer bool array, Muse Glimmer stores a
    # single stride: every `pattern`-th layer is a full-attention layer and the
    # rest are sliding. Full-attention layers are also the NoPE layers -- the HF
    # checkpoint expresses this as layer_rope_theta[i] == 0.
    #
    # There is no defensible fallback if the stride is missing. Guessing leaves
    # every layer sliding and rotated, which is a different architecture that
    # happens to load, so refuse the conversion instead of emitting one.
    key = f"{arch}.attention.sliding_window_pattern"
    pattern = metadata.get(key)
    if pattern is None:
        raise ValueError(
            f"GGUF metadata is missing {key}. Muse Glimmer needs it to place "
            f"the full-attention and NoPE layers; without it the converted "
            f"model would silently be a different architecture."
        )
    pattern = int(pattern)
    if pattern <= 0:
        raise ValueError(f"GGUF metadata {key} must be positive, got {pattern}.")
    layer_types = [
        "full_attention" if (index + 1) % pattern == 0 else "sliding_attention"
        for index in range(config.num_hidden_layers)
    ]
    layer_rope_theta: list[float | int] = [
        0 if layer_type == "full_attention" else config.rope_theta
        for layer_type in layer_types
    ]
    config = dataclasses.replace(
        config,
        layer_types=layer_types,
        no_rope_layers=[index for index, theta in enumerate(layer_rope_theta) if theta == 0],
    )

    sliding_window = metadata.get(f"{arch}.attention.sliding_window")
    if sliding_window is not None:
        config = dataclasses.replace(config, sliding_window=int(sliding_window))

    config = dataclasses.replace(config, attn_qk_norm=True)

    defaults = MuseGlimmerConfig()
    softcapping = metadata.get(f"{arch}.final_logit_softcapping")
    logit_scale = metadata.get(f"{arch}.logit_scale")

    return MuseGlimmerConfig(
        **_shallow_fields(config),
        qk_scale_factor=_muse_glimmer_qk_scale_factor(model, defaults.qk_scale_factor),
        output_multiplier=(
            float(logit_scale) if logit_scale is not None else defaults.output_multiplier
        ),
        final_logit_softcapping=(
            float(softcapping) if softcapping is not None else defaults.final_logit_softcapping
        ),
        post_norm_eps=defaults.post_norm_eps,
        layer_rope_theta=layer_rope_theta,
    )


def _muse_glimmer_qk_scale_factor(model: Any, default: float) -> float:
    """Read Muse Glimmer's ``qk_scale_factor`` out of ``blk.0.attn_q_norm``.

    Muse Glimmer normalizes Q and K without a learned scale and then multiplies
    Q by a single constant. llama.cpp has no place for that constant in the
    metadata, so it broadcasts it across a head_dim-wide ``attn_q_norm`` tensor
    (``attn_k_norm`` is the matching all-ones vector). Reading element 0 back is
    therefore the only way to recover the value from a GGUF file.

    Falls back to *default* when the tensor is missing, and rejects a tensor
    that is not constant, because a non-constant vector would mean the file
    carries a genuine per-channel norm this importer would silently drop.
    """
    if model is None:
        return default
    try:
        weight = model.get_tensor("blk.0.attn_q_norm.weight")
    except (KeyError, ValueError):
        logger.warning(
            "Muse Glimmer GGUF has no blk.0.attn_q_norm tensor; "
            "falling back to qk_scale_factor=%s",
            default,
        )
        return default
    values = np.asarray(weight, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return default
    if not np.allclose(values, values[0]):
        raise ValueError(
            "Muse Glimmer blk.0.attn_q_norm is not a constant vector "
            f"(min={values.min()}, max={values.max()}), so it carries a real "
            "per-channel QK norm that this importer does not model."
        )
    return float(values[0])


# Architecture-specific config postprocessors, keyed by the name a
# :class:`GGUFArchitectureSpec` refers to. Each takes a base ArchitectureConfig
# + raw metadata and returns an architecture-specific config subclass.
#
# Keyed by postprocessor name rather than by ``model_type``: the old
# model_type keying is what let the Gemma weight processor drift out of reach
# when an architecture's model_type gained a ``_text`` suffix.
_CONFIG_POSTPROCESSORS: dict[str, Any] = {
    "olmo": _olmo_postprocess,
    "dense_sliding": _dense_sliding_postprocess,
    "gemma2": _gemma2_postprocess,
    "gemma3": _gemma3_postprocess,
    "gemma4": _gemma4_postprocess,
    "muse_glimmer": _muse_glimmer_postprocess,
}


def _default_activation(model_type: str) -> str:
    """Return the default activation function for a model type."""
    # Gemma models use the approximate-tanh GELU (GeGLU) in every MLP block.
    # Gemma GGUFs typically omit the activation metadata key, so guard against
    # the generic SiLU default below (using SiLU here silently degrades Gemma
    # output to near-garbage).
    if model_type.startswith("gemma"):
        return "gelu_pytorch_tanh"
    # Most modern models use SiLU/Swish
    if model_type == "arcee":
        return "relu2"
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


class GgufArchResolutionError(ValueError):
    """A GGUF architecture could not be resolved to a buildable model type.

    Raised when the GGUF architecture string maps to a canonical registry key
    (e.g. ``glm-dsa`` → ``glm_moe_dsa``) but the metadata-derived config is
    missing the head/layer/expert/attention properties that model requires, so
    dispatching to it would build the wrong graph. The message lists every
    failed property (precise rejection reasons) rather than the first one.
    """


def resolve_model_type(gguf_arch: str) -> str:
    """Resolve a GGUF architecture string to a canonical registry model type.

    This is the single, explicit format-bridge lookup. It is driven purely by
    the authoritative ``general.architecture`` metadata value — never by the
    filename or ``general.name`` — so an arbitrarily-named GGUF cannot be
    coerced into a different architecture. Unknown architectures fall through
    unchanged, matching the pre-existing default behaviour.
    """
    return GGUF_ARCH_TO_MODEL_TYPE.get(gguf_arch, gguf_arch)


def _positive(value: Any) -> bool:
    try:
        return value is not None and int(value) > 0
    except (TypeError, ValueError):
        return False


def assert_glm_moe_dsa_resolvable(
    config: ArchitectureConfig,
    gguf_arch: str,
    *,
    source: str,
) -> None:
    """Verify a ``glm-dsa`` GGUF really is a buildable GLM-5.2 ``glm_moe_dsa``.

    GLM-5.2 (``glm_moe_dsa``) is MLA + DeepSeek Sparse Attention (DSA) + MoE.
    Before the builder selects :class:`GlmMoeDsaCausalLMModel`, confirm the
    metadata-derived config carries the discriminating head/layer/expert/DSA
    properties. A bare decoder GGUF mislabelled ``glm-dsa`` (or one whose DSA /
    MoE keys use unexpected suffixes) is rejected with the exact list of missing
    properties instead of silently building an incorrect graph.

    Only invoked for the ``glm-dsa`` → ``glm_moe_dsa`` bridge; other
    architectures are unaffected.
    """
    reasons: list[str] = []

    if not _positive(config.num_hidden_layers):
        reasons.append(f"num_hidden_layers must be > 0 (got {config.num_hidden_layers!r})")
    if not _positive(config.num_attention_heads):
        reasons.append(f"num_attention_heads must be > 0 (got {config.num_attention_heads!r})")
    if not _positive(config.hidden_size):
        reasons.append(f"hidden_size must be > 0 (got {config.hidden_size!r})")

    # MoE expert stack.
    if not _positive(config.num_local_experts):
        reasons.append(
            "missing routed-expert count (GGUF '<arch>.expert_count'); "
            f"num_local_experts={config.num_local_experts!r}"
        )
    if not _positive(config.num_experts_per_tok):
        reasons.append(
            "missing experts-per-token (GGUF '<arch>.expert_used_count'); "
            f"num_experts_per_tok={config.num_experts_per_tok!r}"
        )
    if not _positive(config.moe_intermediate_size):
        reasons.append(
            "missing expert FFN width (GGUF '<arch>.expert_feed_forward_length'); "
            f"moe_intermediate_size={config.moe_intermediate_size!r}"
        )

    # MLA (latent attention) evidence: GLM-5.2 uses low-rank Q/KV projections.
    if not (_positive(config.q_lora_rank) or _positive(config.kv_lora_rank)):
        reasons.append(
            "no MLA low-rank projection found (GGUF '<arch>.attention.q_lora_rank' "
            "or '<arch>.attention.kv_lora_rank'); GLM-5.2 requires latent attention"
        )

    # DSA indexer — only required on the default sparse-attention export path.
    if getattr(config, "use_dsa", True):
        for field_name, gguf_key in (
            ("index_n_heads", "attention.indexer.head_count"),
            ("index_head_dim", "attention.indexer.key_length"),
            ("index_topk", "attention.indexer.top_k"),
        ):
            if not _positive(getattr(config, field_name, None)):
                reasons.append(
                    f"missing DSA indexer property {field_name} "
                    f"(GGUF '<arch>.{gguf_key}'); required for use_dsa=True "
                    "(pass use_dsa=False / --glm-full-attention for dense MLA)"
                )

    if reasons:
        joined = "\n  - ".join(reasons)
        raise GgufArchResolutionError(
            f"GGUF architecture {gguf_arch!r} in {source!r} resolves to the "
            "canonical model type 'glm_moe_dsa' (GLM-5.2), but the metadata does "
            "not describe a complete MLA + DSA + MoE model:\n  - "
            f"{joined}\n"
            "No ONNX artifacts were emitted. Re-export the GGUF from an "
            "authoritative GLM-5.2 checkpoint, or if this file is a different "
            "architecture, do not tag it 'glm-dsa'."
        )
