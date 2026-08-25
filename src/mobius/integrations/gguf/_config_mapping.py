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
from typing import TYPE_CHECKING, Any

import numpy as np

from mobius._configs import (
    ArchitectureConfig,
    Gemma2Config,
    Gemma4Config,
    GraniteMoeHybridConfig,
    JambaConfig,
    Mamba2Config,
    MambaConfig,
    MuseGlimmerConfig,
    NemotronHConfig,
    _shallow_fields,
)
from mobius.integrations.gguf._arch_registry import iter_arch_specs, try_get_arch_spec

if TYPE_CHECKING:
    from mobius._configs import DFlashConfig, Eagle3Config

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
    "hidden_activation": "hidden_act",
    # MoE fields
    "expert_count": "num_local_experts",
    "expert_used_count": "num_experts_per_tok",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_shared_feed_forward_length": "shared_expert_intermediate_size",
    "expert_weights_scale": "routed_scaling_factor",
    "expert_weights_norm": "norm_topk_prob",
    "expert_group_count": "n_group",
    "expert_group_used_count": "topk_group",
    # Hybrid (DeltaNet / Mamba + Attention) fields
    "full_attention_interval": "full_attention_interval",
    # SSM/DeltaNet fields (used for linear attention in hybrid models)
    "ssm.group_count": "linear_num_key_heads",
    "ssm.time_step_rank": "linear_num_value_heads",
    "ssm.conv_kernel": "linear_conv_kernel_dim",
    "ssm.inner_size": "linear_inner_size",
    "ssm.state_size": "linear_key_head_dim",
    "shortconv.l_cache": "short_conv_kernel",
}

_DRAFT_KEY_MAP = {
    "block_size": "block_size",
    "target_hidden_size": "target_hidden_size",
    "target_layers": "target_layer_ids",
    "norm_before_residual": "norm_before_residual",
    "norm_before_fc": "norm_before_fc",
}

_MUSE_GLIMMER_KEY_MAP = {
    "attention.key_length": "head_dim",
    "attention.sliding_window": "sliding_window",
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

_MAMBA_KEY_MAP = {
    "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
    "ssm.conv_kernel": "conv_kernel",
    "ssm.group_count": "n_groups",
    "ssm.inner_size": "intermediate_size",
    "ssm.state_size": "state_size",
    "ssm.time_step_rank": "time_step_rank",
}

_JAMBA_KEY_MAP = {
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "ssm.conv_kernel": "mamba_d_conv",
    "ssm.inner_size": "mamba_d_inner",
    "ssm.state_size": "mamba_d_state",
    "ssm.time_step_rank": "mamba_dt_rank",
}

_NEMOTRON_H_KEY_MAP = {
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "ssm.conv_kernel": "conv_kernel",
    "ssm.group_count": "n_groups",
    "ssm.state_size": "state_size",
    "ssm.time_step_rank": "mamba_num_heads",
}

_GRANITEHYBRID_KEY_MAP = {
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "ssm.conv_kernel": "conv_kernel",
    "ssm.group_count": "n_groups",
    "ssm.inner_size": "mamba_intermediate_size",
    "ssm.state_size": "state_size",
    "ssm.time_step_rank": "mamba_num_heads",
}

_T5_KEY_MAP = {
    "attention.key_length": "head_dim",
    "attention.relative_buckets_count": "relative_attention_num_buckets",
    "decoder_block_count": "num_decoder_layers",
    "decoder_start_token_id": "decoder_start_token_id",
}

#: Named architecture-specific key maps that :attr:`GGUFArchitectureSpec.
#: config_key_map` selects. Every name here must be referenced by a spec and
#: every name a spec references must exist here; ``_arch_registry_test`` checks
#: both directions so a typo cannot silently drop config fields.
_KEY_MAP_TABLES: MappingProxyType[str, dict[str, str]] = MappingProxyType(
    {
        "draft": _DRAFT_KEY_MAP,
        "muse_glimmer": _MUSE_GLIMMER_KEY_MAP,
        "glm_dsa": _GLM_DSA_KEY_MAP,
        "mamba": _MAMBA_KEY_MAP,
        "jamba": _JAMBA_KEY_MAP,
        "nemotron_h": _NEMOTRON_H_KEY_MAP,
        "granitehybrid": _GRANITEHYBRID_KEY_MAP,
        "t5": _T5_KEY_MAP,
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


_DELTA_NET_ARCHITECTURES = frozenset({"qwen35", "qwen35moe", "qwen3next"})


def _nextn_predict_layers(gguf_arch: str, metadata: dict[str, Any]) -> int:
    """Read the exact nextn spelling from pinned llama.cpp."""
    dotted_key = f"{gguf_arch}.nextn.predict_layers"
    if dotted_key in metadata:
        raise ValueError(
            f"Unsupported non-pinned MTP metadata key {dotted_key!r}; expected "
            f"{gguf_arch}.nextn_predict_layers"
        )
    key = f"{gguf_arch}.nextn_predict_layers"
    if key not in metadata:
        return 0
    count = int(metadata[key])
    if count < 0:
        raise ValueError(f"{key} must be non-negative, got {count}")
    return count


def _derive_hybrid_layout(
    gguf_arch: str,
    metadata: dict[str, Any],
) -> tuple[int, list[str] | None, int]:
    """Derive the trunk layer count and exact mixer schedule from GGUF metadata."""
    total_layers = int(metadata[f"{gguf_arch}.block_count"])
    mtp_count = _nextn_predict_layers(gguf_arch, metadata)
    trunk_layers = total_layers - mtp_count
    if trunk_layers <= 0:
        raise ValueError(
            f"GGUF metadata inconsistent: block_count ({total_layers}) <= "
            f"nextn predict layers ({mtp_count}) for architecture {gguf_arch}."
        )

    if gguf_arch in {"lfm2", "jamba", "granitehybrid"}:
        raw_kv_heads = metadata.get(f"{gguf_arch}.attention.head_count_kv")
        if not isinstance(raw_kv_heads, (list, tuple, np.ndarray)):
            raise ValueError(
                f"{gguf_arch}.attention.head_count_kv must be a per-layer array; "
                "a scalar cannot reconstruct the hybrid attention/conv schedule"
            )
        kv_heads = [int(value) for value in raw_kv_heads]
        if len(kv_heads) != total_layers:
            raise ValueError(
                f"{gguf_arch}.attention.head_count_kv must contain exactly "
                f"{total_layers} entries, got {len(kv_heads)}"
            )
        if any(value < 0 for value in kv_heads):
            raise ValueError(
                f"{gguf_arch}.attention.head_count_kv entries must be non-negative"
            )
        recurrent_type = {
            "lfm2": "conv",
            "jamba": "mamba",
            "granitehybrid": "mamba2",
        }[gguf_arch]
        return (
            trunk_layers,
            [
                recurrent_type if value == 0 else "full_attention"
                for value in kv_heads[:trunk_layers]
            ],
            mtp_count,
        )

    if gguf_arch == "nemotron_h":
        if mtp_count:
            raise ValueError("nemotron_h GGUF import does not support folded MTP blocks")
        kv_raw = metadata.get(f"{gguf_arch}.attention.head_count_kv")
        ffn_raw = metadata.get(f"{gguf_arch}.feed_forward_length")
        if not isinstance(kv_raw, (list, tuple, np.ndarray)) or not isinstance(
            ffn_raw, (list, tuple, np.ndarray)
        ):
            raise ValueError(
                "nemotron_h requires per-layer attention.head_count_kv and "
                "feed_forward_length arrays"
            )
        kv_heads = [int(value) for value in kv_raw]
        ffn_lengths = [int(value) for value in ffn_raw]
        if len(kv_heads) != total_layers or len(ffn_lengths) != total_layers:
            raise ValueError(
                f"nemotron_h schedule arrays must each contain exactly {total_layers} entries"
            )
        if any(value < 0 for value in (*kv_heads, *ffn_lengths)):
            raise ValueError("nemotron_h schedule entries must be non-negative")
        uses_moe = int(metadata.get(f"{gguf_arch}.expert_count", 0)) > 0
        layer_types = []
        for kv_heads_i, ffn_length_i in zip(kv_heads, ffn_lengths):
            if ffn_length_i:
                layer_types.append("moe" if uses_moe else "mlp")
            elif kv_heads_i == 0:
                layer_types.append("mamba2")
            else:
                layer_types.append("full_attention")
        return trunk_layers, layer_types, mtp_count

    if gguf_arch not in _DELTA_NET_ARCHITECTURES:
        return trunk_layers, None, mtp_count

    recurrent_key = f"{gguf_arch}.attention.recurrent_layers"
    recurrent = metadata.get(recurrent_key)
    if recurrent is not None:
        if isinstance(recurrent, (list, tuple, np.ndarray)):
            if any(
                not isinstance(value, (bool, np.bool_, int, np.integer))
                or int(value) not in (0, 1)
                for value in recurrent
            ):
                raise ValueError(f"{recurrent_key} entries must be booleans or 0/1")
            recurrent_layers = [bool(value) for value in recurrent]
            if len(recurrent_layers) != total_layers:
                raise ValueError(
                    f"{recurrent_key} must contain exactly {total_layers} entries, "
                    f"got {len(recurrent_layers)}"
                )
        elif isinstance(recurrent, (bool, np.bool_)):
            recurrent_layers = [bool(recurrent)] * total_layers
        else:
            raise ValueError(f"{recurrent_key} must be a boolean or boolean array")
        if any(recurrent_layers[trunk_layers:]):
            raise ValueError(f"{recurrent_key} marks an appended MTP block as recurrent")
    else:
        interval = int(metadata.get(f"{gguf_arch}.full_attention_interval", 4))
        if interval <= 0:
            raise ValueError(
                f"{gguf_arch}.full_attention_interval must be positive, got {interval}"
            )
        recurrent_layers = [
            i < trunk_layers and (i + 1) % interval != 0 for i in range(total_layers)
        ]

    return (
        trunk_layers,
        [
            "linear_attention" if value else "full_attention"
            for value in recurrent_layers[:trunk_layers]
        ],
        mtp_count,
    )


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
    nextn_layers = _nextn_predict_layers(gguf_arch, metadata)
    mtp_predict_layers = 0
    mtp_block_indices: list[int] = []
    if nextn_layers > 0:
        mtp_count = nextn_layers
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
    if canonical_arch in {"mamba", "mamba2"}:
        # Pure recurrent GGUFs deliberately write attention.head_count=0.
        # The temporary ArchitectureConfig still needs a nonzero placeholder;
        # the postprocessor replaces it with the real SSM head geometry.
        num_attention_heads = int(num_attention_heads) or 1
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
        values = [int(value) for value in num_kv_heads]
        if canonical_arch in {"lfm2", "jamba", "nemotron_h", "granitehybrid"}:
            nonzero = {value for value in values if value}
            if len(nonzero) != 1:
                raise ValueError(
                    f"{canonical_arch} attention layers must use one consistent non-zero "
                    "KV-head count, "
                    f"got {sorted(nonzero)}"
                )
            if not nonzero:
                raise ValueError(f"{canonical_arch} GGUF has no attention layer KV-head count")
            num_kv_heads = nonzero.pop()
        else:
            # Per-layer array → pick the majority value (sliding layers dominate)
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

    # Derive the exact hybrid schedule from the same serialized metadata used
    # by pinned llama.cpp. Explicit recurrent arrays always win over intervals.
    full_attention_interval = hf_fields.get("full_attention_interval")
    num_hidden_layers = hf_fields["num_hidden_layers"]
    layer_types: list[str] | None = None
    if (
        canonical_arch
        in {
            "lfm2",
            "jamba",
            "nemotron_h",
            "granitehybrid",
        }
        or canonical_arch in _DELTA_NET_ARCHITECTURES
    ):
        derived_layers, layer_types, derived_mtp_count = _derive_hybrid_layout(
            canonical_arch, metadata
        )
        if derived_layers != int(num_hidden_layers) or derived_mtp_count != mtp_predict_layers:
            raise ValueError("Hybrid schedule and decoder layer metadata disagree")
        if canonical_arch in _DELTA_NET_ARCHITECTURES:
            full_attention_interval = int(
                metadata.get(f"{canonical_arch}.full_attention_interval", 4)
            )

    if canonical_arch == "nemotron_h":
        ffn_lengths = metadata[f"{gguf_arch}.feed_forward_length"]
        nonzero_ffn_lengths = {int(value) for value in ffn_lengths if int(value)}
        if len(nonzero_ffn_lengths) > 1:
            raise ValueError(
                "nemotron_h dense FFN layers must use one consistent feed-forward length"
            )
        hf_fields["intermediate_size"] = (
            nonzero_ffn_lengths.pop() if nonzero_ffn_lengths else 4 * hidden_size
        )

    # Derive DeltaNet head dimensions from SSM metadata.
    # Key width comes from state_size. Value width is independently derived from
    # inner_size / time_step_rank and must not be guessed from key width.
    ssm_state_size = metadata.get(f"{gguf_arch}.ssm.state_size")
    linear_key_head_dim = int(ssm_state_size) if ssm_state_size else None
    linear_value_head_dim = None
    if canonical_arch in _DELTA_NET_ARCHITECTURES:
        inner_size = int(metadata[f"{gguf_arch}.ssm.inner_size"])
        value_heads = int(metadata[f"{gguf_arch}.ssm.time_step_rank"])
        key_heads = int(metadata[f"{gguf_arch}.ssm.group_count"])
        conv_kernel = int(metadata[f"{gguf_arch}.ssm.conv_kernel"])
        if min(inner_size, value_heads, key_heads, conv_kernel, int(ssm_state_size)) <= 0:
            raise ValueError(f"{gguf_arch} DeltaNet dimensions must all be positive")
        if inner_size % value_heads:
            raise ValueError(
                f"{gguf_arch}.ssm.inner_size ({inner_size}) must be divisible by "
                f"ssm.time_step_rank ({value_heads})"
            )
        if inner_size != int(ssm_state_size) * value_heads:
            raise ValueError(
                f"{gguf_arch}.ssm.inner_size ({inner_size}) must equal "
                f"ssm.state_size * ssm.time_step_rank "
                f"({int(ssm_state_size) * value_heads}) for the pinned DeltaNet loader"
            )
        if value_heads % key_heads:
            raise ValueError(
                f"{gguf_arch}.ssm.time_step_rank ({value_heads}) must be divisible by "
                f"ssm.group_count ({key_heads})"
            )
        linear_value_head_dim = inner_size // value_heads

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

    if canonical_arch in {"qwen35", "qwen35moe"}:
        mrope_section = metadata[f"{gguf_arch}.rope.dimension_sections"]
        if not isinstance(mrope_section, (list, tuple, np.ndarray)) or len(mrope_section) != 4:
            raise ValueError(
                f"{gguf_arch}.rope.dimension_sections must contain exactly four entries"
            )
        mrope_section = [int(value) for value in mrope_section]
    else:
        mrope_section = None

    if canonical_arch in {"qwen35moe", "qwen3next"}:
        top_k = int(hf_fields["num_experts_per_tok"])
        experts = int(hf_fields["num_local_experts"])
        intermediate = int(hf_fields.get("intermediate_size", 4 * hidden_size))
        if min(top_k, experts) <= 0 or top_k > experts:
            raise ValueError(
                f"{gguf_arch} expert counts are invalid: expert_count={experts}, "
                f"expert_used_count={top_k}"
            )
        hf_fields.setdefault("moe_intermediate_size", intermediate // top_k)
        hf_fields.setdefault("shared_expert_intermediate_size", intermediate)

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
        pad_token_id=int(metadata.get("tokenizer.ggml.padding_token_id", 0)),
        eos_token_id=metadata.get("tokenizer.ggml.eos_token_id"),
        tie_word_embeddings=_infer_tie_embeddings(model),
        # Projection biases are not in GGUF metadata; infer from tensor
        # presence. Qwen2/Qwen3 carry Q/K/V biases — omitting them breaks
        # attention and yields garbage output.
        attn_qkv_bias=_infer_attn_qkv_bias(model),
        attn_o_bias=_infer_attn_o_bias(model),
        mlp_bias=_infer_mlp_bias(model),
        partial_rotary_factor=partial_rotary_factor,
        rope_interleave=rope_interleave,
        mrope_section=mrope_section,
        num_decoder_layers=hf_fields.get("num_decoder_layers"),
        decoder_start_token_id=hf_fields.get("decoder_start_token_id"),
        relative_attention_num_buckets=hf_fields.get("relative_attention_num_buckets", 32),
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
        short_conv_kernel=(hf_fields.get("short_conv_kernel") or 3),
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
        model_type = config.model_type or model_type
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


def _moe_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> ArchitectureConfig:
    """Validate and complete conventional-attention MoE metadata."""
    del model
    arch = getattr(config, "_gguf_arch", config.model_type)
    num_experts = config.num_local_experts
    top_k = config.num_experts_per_tok
    if num_experts is None or int(num_experts) <= 0:
        raise ValueError(f"{arch}.expert_count must be greater than zero")
    if top_k is None or int(top_k) <= 0 or int(top_k) > int(num_experts):
        raise ValueError(
            f"{arch}.expert_used_count must be in [1, {num_experts}], got {top_k}"
        )
    if config.intermediate_size <= 0:
        raise ValueError(f"{arch}.feed_forward_length must be greater than zero")
    if config.moe_intermediate_size is not None and config.moe_intermediate_size <= 0:
        raise ValueError(f"{arch}.expert_feed_forward_length must be greater than zero")
    if (
        config.shared_expert_intermediate_size is not None
        and config.shared_expert_intermediate_size <= 0
    ):
        raise ValueError(f"{arch}.expert_shared_feed_forward_length must be greater than zero")

    updates: dict[str, Any] = {}
    if arch in ("qwen2moe", "qwen3moe"):
        if config.moe_intermediate_size is None:
            if config.intermediate_size % int(top_k):
                raise ValueError(
                    f"{arch}.feed_forward_length ({config.intermediate_size}) must be "
                    f"divisible by expert_used_count ({top_k}) when "
                    "expert_feed_forward_length is absent"
                )
            updates["moe_intermediate_size"] = config.intermediate_size // int(top_k)
    if arch == "qwen2moe":
        updates["norm_topk_prob"] = False
        if config.shared_expert_intermediate_size is None:
            updates["shared_expert_intermediate_size"] = config.intermediate_size
    elif arch == "qwen3moe":
        updates["attn_qk_norm"] = True
        updates["attn_qk_norm_full"] = False
        updates["norm_topk_prob"] = True
    elif arch == "olmoe":
        updates["attn_qk_norm"] = True
        updates["attn_qk_norm_full"] = True
        updates["norm_topk_prob"] = False

    return dataclasses.replace(config, **updates)


def _diffusion_common_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Validate the common no-cache masked-diffusion contract."""
    arch = model.architecture
    mask_token_id = metadata.get("tokenizer.ggml.mask_token_id")
    if mask_token_id is None:
        raise ValueError(
            "tokenizer.ggml.mask_token_id is required for masked-diffusion generation"
        )
    mask_token_id = int(mask_token_id)
    if not 0 <= mask_token_id < config.vocab_size:
        raise ValueError(
            f"tokenizer.ggml.mask_token_id must be in [0, {config.vocab_size}), "
            f"got {mask_token_id}"
        )
    if config.hidden_act not in (None, "silu"):
        raise ValueError(f"{arch}.hidden_activation must be silu, got {config.hidden_act!r}")

    result = dataclasses.replace(
        config,
        hidden_act="silu",
        rope_type="default",
        mlp_bias=False,
        mask_token_id=mask_token_id,
        diffusion_shift_logits=bool(
            metadata.get("diffusion.shift_logits", arch in {"dream", "rnd1"})
        ),
    )
    return result


def _dense_diffusion_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
    *,
    dream: bool,
) -> ArchitectureConfig:
    result = _diffusion_common_postprocess(config, metadata, model)
    names = set(model.tensor_names)
    output_present = "output.weight" in names
    qkv_bias = _diffusion_qkv_bias(config, names, allow_fused=dream)
    if qkv_bias and not dream:
        raise ValueError("llada GGUF does not support Q/K/V projection biases")
    return dataclasses.replace(
        result,
        attn_qkv_bias=qkv_bias,
        attn_o_bias=False,
        tie_word_embeddings=not output_present,
    )


def _diffusion_qkv_bias(
    config: ArchitectureConfig,
    names: set[str],
    *,
    allow_fused: bool,
) -> bool:
    """Validate per-layer fused/separate QKV alternatives and bias consistency."""
    layer_biases: list[bool] = []
    for layer in range(config.num_hidden_layers):
        fused_weight = f"blk.{layer}.attn_qkv.weight"
        fused_bias = f"blk.{layer}.attn_qkv.bias"
        separate_weights = {
            f"blk.{layer}.attn_{projection}.weight" for projection in ("q", "k", "v")
        }
        separate_biases = {
            f"blk.{layer}.attn_{projection}.bias" for projection in ("q", "k", "v")
        }
        has_fused = fused_weight in names
        present_weights = separate_weights & names
        if has_fused and not allow_fused:
            raise ValueError(f"llada does not support fused QKV tensor {fused_weight!r}")
        if has_fused and present_weights:
            raise ValueError(
                f"layer {layer} provides both fused and separate QKV projection weights"
            )
        if not has_fused and present_weights != separate_weights:
            missing = sorted(separate_weights - present_weights)
            raise ValueError(f"masked-diffusion GGUF is missing {missing[0]!r}")

        if has_fused:
            if separate_biases & names:
                raise ValueError(
                    f"layer {layer} mixes a fused QKV weight with separate Q/K/V biases"
                )
            layer_biases.append(fused_bias in names)
        else:
            present_biases = separate_biases & names
            if present_biases and present_biases != separate_biases:
                missing = sorted(separate_biases - present_biases)
                raise ValueError(
                    f"Q/K/V bias must be present consistently; missing {missing[0]!r}"
                )
            if fused_bias in names:
                raise ValueError(
                    f"layer {layer} provides fused QKV bias without a fused weight"
                )
            layer_biases.append(bool(present_biases))
    if len(set(layer_biases)) > 1:
        raise ValueError("Q/K/V bias presence must be consistent across every layer")
    return layer_biases[0] if layer_biases else False


def _dream_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    return _dense_diffusion_postprocess(config, metadata, model, dream=True)


def _llada_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    return _dense_diffusion_postprocess(config, metadata, model, dream=False)


def _diffusion_moe_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
    *,
    normalize_topk: bool,
    require_output: bool,
) -> ArchitectureConfig:
    result = _diffusion_common_postprocess(config, metadata, model)
    result = _moe_postprocess(result, metadata, model)
    assert result.num_experts_per_tok is not None
    if result.moe_intermediate_size is None:
        if result.intermediate_size % result.num_experts_per_tok:
            raise ValueError(
                f"{model.architecture}.feed_forward_length ({result.intermediate_size}) "
                "must be divisible by expert_used_count "
                f"({result.num_experts_per_tok}) when expert_feed_forward_length is absent"
            )
        expert_width = result.intermediate_size // result.num_experts_per_tok
    else:
        expert_width = result.moe_intermediate_size

    output_present = "output.weight" in set(model.tensor_names)
    if require_output and not output_present:
        raise ValueError(f"{model.architecture} requires output.weight")
    return dataclasses.replace(
        result,
        attn_qk_norm=True,
        attn_qk_norm_full=False,
        attn_qkv_bias=_diffusion_qkv_bias(config, set(model.tensor_names), allow_fused=True),
        attn_o_bias=False,
        moe_intermediate_size=expert_width,
        norm_topk_prob=normalize_topk,
        tie_word_embeddings=not output_present,
    )


def _llada_moe_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    return _diffusion_moe_postprocess(
        config, metadata, model, normalize_topk=False, require_output=True
    )


def _rnd1_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    return _diffusion_moe_postprocess(
        config, metadata, model, normalize_topk=True, require_output=False
    )


def _granitemoe_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> ArchitectureConfig:
    """Apply Granite scaling and select its dense or MoE graph exactly."""
    del model
    arch = "granitemoe"
    num_experts = config.num_local_experts
    top_k = config.num_experts_per_tok
    if config.intermediate_size <= 0:
        raise ValueError("granitemoe.feed_forward_length must be greater than zero")
    if num_experts is None or int(num_experts) == 0:
        if top_k not in (None, 0):
            raise ValueError(
                "granitemoe.expert_used_count must be absent or zero when expert_count is zero"
            )
        model_type = "granite"
        num_experts = None
        top_k = None
    else:
        if int(num_experts) < 0:
            raise ValueError("granitemoe.expert_count must not be negative")
        if top_k is None or int(top_k) <= 0 or int(top_k) > int(num_experts):
            raise ValueError(
                f"granitemoe.expert_used_count must be in [1, {num_experts}], got {top_k}"
            )
        model_type = "granitemoe"

    logit_scale = float(metadata[f"{arch}.logit_scale"])
    if not logit_scale:
        raise ValueError("granitemoe.logit_scale must be nonzero")
    return dataclasses.replace(
        config,
        model_type=model_type,
        num_local_experts=num_experts,
        num_experts_per_tok=top_k,
        embedding_multiplier=float(metadata.get(f"{arch}.embedding_scale", 1.0)),
        attention_multiplier=(
            float(metadata[f"{arch}.attention.scale"])
            if f"{arch}.attention.scale" in metadata
            else None
        ),
        logits_scaling=logit_scale,
        residual_multiplier=float(metadata.get(f"{arch}.residual_scale", 1.0)),
        norm_topk_prob=True,
    )


def _jamba_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> JambaConfig:
    """Build the dense Jamba subset from the serialized per-layer schedule."""
    inner_size = int(metadata["jamba.ssm.inner_size"])
    if inner_size != 2 * config.hidden_size:
        raise ValueError(
            "jamba.ssm.inner_size must equal 2 * embedding_length for the pinned loader"
        )
    if int(metadata.get("jamba.expert_count", 0)) or any(
        ".ffn_gate_inp." in name for name in model.tensor_names
    ):
        raise ValueError(
            "Jamba GGUF MoE layers are deferred until stacked-expert parity is established"
        )
    return JambaConfig(
        **_shallow_fields(config),
        mamba_d_state=int(metadata["jamba.ssm.state_size"]),
        mamba_d_conv=int(metadata["jamba.ssm.conv_kernel"]),
        mamba_expand=2,
        mamba_dt_rank=int(metadata["jamba.ssm.time_step_rank"]),
        mamba_conv_bias=any(".ssm_conv1d.bias" in name for name in model.tensor_names),
        mamba_proj_bias=False,
    )


def _nemotron_h_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> NemotronHConfig:
    """Build the dense, no-MTP Nemotron-H subset."""
    if int(metadata.get("nemotron_h.expert_count", 0)):
        raise ValueError(
            "Nemotron-H MoE GGUF import is deferred; fused softmax MoE is incompatible "
            "with its sigmoid correction-bias routing"
        )
    inner_size = int(metadata["nemotron_h.ssm.inner_size"])
    num_heads = int(metadata["nemotron_h.ssm.time_step_rank"])
    if min(inner_size, num_heads) <= 0 or inner_size % num_heads:
        raise ValueError(
            "nemotron_h.ssm.inner_size must be positive and divisible by ssm.time_step_rank"
        )
    fields = _shallow_fields(config)
    fields["hidden_act"] = "relu2"
    return NemotronHConfig(
        **fields,
        mamba_n_heads=num_heads,
        mamba_d_head=inner_size // num_heads,
        mamba_d_state=int(metadata["nemotron_h.ssm.state_size"]),
        mamba_n_groups=int(metadata["nemotron_h.ssm.group_count"]),
        mamba_d_conv=int(metadata["nemotron_h.ssm.conv_kernel"]),
        mamba_expand=inner_size // config.hidden_size,
        mamba_conv_bias=any(".ssm_conv1d.bias" in name for name in model.tensor_names),
        mamba_proj_bias=False,
    )


def _granitehybrid_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> GraniteMoeHybridConfig:
    """Build the dense GraniteHybrid subset with exact scaling metadata."""
    if int(metadata.get("granitehybrid.expert_count", 0)):
        raise ValueError(
            "GraniteHybrid MoE GGUF import is deferred until 3-D expert fusion and "
            "quantized expert ordering have independent value tests"
        )
    inner_size = int(metadata["granitehybrid.ssm.inner_size"])
    num_heads = int(metadata["granitehybrid.ssm.time_step_rank"])
    groups = int(metadata["granitehybrid.ssm.group_count"])
    if inner_size != 2 * config.hidden_size:
        raise ValueError("granitehybrid.ssm.inner_size must equal 2 * embedding_length")
    if min(num_heads, groups) <= 0 or inner_size % num_heads or num_heads % groups:
        raise ValueError("GraniteHybrid Mamba2 head and group dimensions are inconsistent")

    fields = _shallow_fields(config)
    fields.update(
        num_local_experts=None,
        num_experts_per_tok=None,
        rope_type=(
            fields["rope_type"]
            if bool(metadata.get("granitehybrid.rope.scaling.finetuned", True))
            else None
        ),
        embedding_multiplier=float(metadata.get("granitehybrid.embedding_scale", 0.0)) or 1.0,
        residual_multiplier=float(metadata.get("granitehybrid.residual_scale", 0.0)) or 1.0,
        attention_multiplier=(
            float(metadata.get("granitehybrid.attention.scale", 0.0)) or None
        ),
        logits_scaling=float(metadata.get("granitehybrid.logit_scale", 0.0)) or 1.0,
    )
    return GraniteMoeHybridConfig(
        **fields,
        mamba_n_heads=num_heads,
        mamba_d_head=inner_size // num_heads,
        mamba_d_state=int(metadata["granitehybrid.ssm.state_size"]),
        mamba_n_groups=groups,
        mamba_d_conv=int(metadata["granitehybrid.ssm.conv_kernel"]),
        mamba_expand=2,
        mamba_conv_bias=any(".ssm_conv1d.bias" in name for name in model.tensor_names),
        mamba_proj_bias=False,
        shared_intermediate_size=config.intermediate_size,
    )


def _phimoe_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> ArchitectureConfig:
    """Complete PhiMoE metadata, including tensor-backed LongRoPE factors."""
    config = _moe_postprocess(config, metadata, model)
    tensor_names = set(getattr(model, "tensor_names", ()) or ())
    long_name = "rope_factors_long.weight"
    short_name = "rope_factors_short.weight"
    if long_name not in tensor_names and short_name not in tensor_names:
        return config
    if long_name not in tensor_names or short_name not in tensor_names:
        raise ValueError(
            "phimoe LongRoPE requires both rope_factors_long.weight and "
            "rope_factors_short.weight"
        )
    if model is None or not hasattr(model, "get_tensor"):
        raise ValueError("phimoe LongRoPE factors could not be read from the GGUF model")

    original_context = metadata.get("phimoe.rope.scaling.original_context_length")
    if original_context is None:
        raise ValueError(
            "phimoe.rope.scaling.original_context_length is required with LongRoPE factors"
        )
    return dataclasses.replace(
        config,
        rope_type="longrope",
        rope_scaling={
            "long_factor": model.get_tensor(long_name).reshape(-1).tolist(),
            "short_factor": model.get_tensor(short_name).reshape(-1).tolist(),
        },
        original_max_position_embeddings=int(original_context),
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


def _base_model_fields(config: ArchitectureConfig, cls: type) -> dict[str, Any]:
    """Copy only fields accepted by a recurrent BaseModelConfig subclass."""
    return {
        field.name: getattr(config, field.name)
        for field in dataclasses.fields(cls)
        if hasattr(config, field.name)
    }


def _mamba_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> MambaConfig:
    """Build the Mamba config from llama.cpp's SSM metadata."""
    del model
    arch = "mamba"
    if bool(metadata.get(f"{arch}.ssm.dt_b_c_rms")):
        raise ValueError(
            "mamba.ssm.dt_b_c_rms=true requires FalconMamba's extra B/C/dt norms, "
            "which the pure Mamba graph does not implement"
        )
    fields = _base_model_fields(config, MambaConfig)
    fields.update(
        intermediate_size=int(metadata[f"{arch}.ssm.inner_size"]),
        state_size=int(metadata[f"{arch}.ssm.state_size"]),
        conv_kernel=int(metadata[f"{arch}.ssm.conv_kernel"]),
        time_step_rank=int(metadata[f"{arch}.ssm.time_step_rank"]),
        layer_norm_epsilon=float(metadata[f"{arch}.attention.layer_norm_rms_epsilon"]),
        use_conv_bias=True,
        expand=int(metadata[f"{arch}.ssm.inner_size"]) // config.hidden_size,
    )
    result = MambaConfig(**fields)
    result.model_type = "mamba"
    return result


def _mamba2_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> Mamba2Config:
    """Build Mamba2 head/group geometry from pinned GGUF SSM metadata."""
    del model
    arch = "mamba2"
    d_inner = int(metadata[f"{arch}.ssm.inner_size"])
    num_heads = int(metadata[f"{arch}.ssm.time_step_rank"])
    n_groups = int(metadata[f"{arch}.ssm.group_count"])
    if num_heads <= 0 or d_inner % num_heads:
        raise ValueError(
            f"{arch}.ssm.inner_size ({d_inner}) must be divisible by "
            f"ssm.time_step_rank/head_count ({num_heads})"
        )
    if n_groups <= 0 or num_heads % n_groups or d_inner % n_groups:
        raise ValueError(
            f"{arch}.ssm.group_count ({n_groups}) must divide both head_count "
            f"({num_heads}) and inner_size ({d_inner})"
        )
    fields = _base_model_fields(config, Mamba2Config)
    fields.update(
        intermediate_size=d_inner,
        num_heads=num_heads,
        head_dim=d_inner // num_heads,
        state_size=int(metadata[f"{arch}.ssm.state_size"]),
        n_groups=n_groups,
        conv_kernel=int(metadata[f"{arch}.ssm.conv_kernel"]),
        expand=d_inner // config.hidden_size,
        layer_norm_epsilon=float(metadata[f"{arch}.attention.layer_norm_rms_epsilon"]),
        use_conv_bias=True,
        chunk_size=256,
    )
    result = Mamba2Config(**fields)
    result.model_type = "mamba2"
    return result


def _validate_encoder_metadata(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    arch: str,
) -> None:
    """Validate metadata that changes an encoder's externally visible contract."""
    if config.num_key_value_heads != config.num_attention_heads:
        raise ValueError(
            f"{arch} GGUF grouped-query attention is not supported: "
            f"attention.head_count={config.num_attention_heads}, "
            f"attention.head_count_kv={config.num_key_value_heads}"
        )
    if bool(metadata[f"{arch}.attention.causal"]):
        raise ValueError(f"{arch}.attention.causal must be false for encoder import")
    # llama_hparams defaults omitted pooling metadata to NONE.
    pooling_type = int(metadata.get(f"{arch}.pooling_type", 0))
    if pooling_type != 0:
        raise ValueError(
            f"{arch}.pooling_type={pooling_type} requests pooled/reranker output, but "
            "Mobius currently exports token embeddings only (pooling_type=NONE/0)"
        )
    labels = metadata.get(f"{arch}.classifier.output_labels")
    if labels:
        raise ValueError(
            f"{arch}.classifier.output_labels declares a classifier head, but the "
            "feature-extraction graph exports token embeddings only"
        )


def _t5_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Apply the pinned llama.cpp T5 defaults and reject unrepresentable variants."""
    arch = model.architecture
    heads = int(metadata[f"{arch}.attention.head_count"])
    kv_heads = int(metadata.get(f"{arch}.attention.head_count_kv", heads))
    if heads <= 0 or kv_heads != heads:
        raise ValueError(
            f"{arch} requires multi-head attention with head_count_kv == head_count; "
            f"got {kv_heads} and {heads}"
        )
    hidden = int(metadata[f"{arch}.embedding_length"])
    key_length = int(metadata.get(f"{arch}.attention.key_length", hidden // heads))
    value_length = int(metadata.get(f"{arch}.attention.value_length", hidden // heads))
    if key_length <= 0 or value_length != key_length:
        raise ValueError(
            f"{arch} requires equal positive attention key/value lengths; "
            f"got key={key_length}, value={value_length}"
        )
    if int(metadata[f"{arch}.feed_forward_length"]) <= 0:
        raise ValueError(f"{arch}.feed_forward_length must be greater than zero")

    encoder_layers = int(metadata[f"{arch}.block_count"])
    decoder_layers = (
        int(metadata.get("t5.decoder_block_count", encoder_layers)) if arch == "t5" else None
    )
    names = set(model.tensor_names)
    encoder_bias_layers = [
        i for i in range(encoder_layers) if f"enc.blk.{i}.attn_rel_b.weight" in names
    ]
    decoder_bias_layers = (
        [i for i in range(decoder_layers or 0) if f"dec.blk.{i}.attn_rel_b.weight" in names]
        if arch == "t5"
        else None
    )
    if not encoder_bias_layers or encoder_bias_layers[0] != 0:
        raise ValueError(f"{arch} GGUF must contain enc.blk.0.attn_rel_b.weight")
    if arch == "t5" and (not decoder_bias_layers or decoder_bias_layers[0] != 0):
        raise ValueError("t5 GGUF must contain dec.blk.0.attn_rel_b.weight")

    layer_prefixes = [
        *(f"enc.blk.{i}" for i in range(encoder_layers)),
        *(f"dec.blk.{i}" for i in range(decoder_layers or 0)),
    ]
    gated = [f"{prefix}.ffn_gate.weight" in names for prefix in layer_prefixes]
    if any(gated) and not all(gated):
        raise ValueError(
            f"{arch} GGUF mixes gated and non-gated FFN layers; one global T5 "
            "activation contract cannot represent that layout"
        )
    is_gated = bool(gated and gated[0])
    if is_gated:
        raise ValueError(
            f"{arch} GGUF gated FFNs are ambiguous: the pinned converter does not "
            "serialize feed_forward_proj or dense_act_fn, so identical metadata and "
            "tensor shapes can represent gated-gelu, gated-silu, or other activations. "
            "Pinned llama.cpp always executes these tensors as tanh-approximate GELU, "
            "but Mobius cannot prove that this matches the source checkpoint."
        )
    return dataclasses.replace(
        config,
        head_dim=key_length,
        num_key_value_heads=heads,
        num_decoder_layers=decoder_layers,
        hidden_act="relu",
        is_gated_act=False,
        scale_decoder_outputs=False,
        relative_attention_max_distance=128,
        encoder_relative_attention_bias_layers=encoder_bias_layers,
        decoder_relative_attention_bias_layers=decoder_bias_layers,
    )


def _bert_encoder_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Apply pinned BERT defaults and tokenizer-owned token-type metadata."""
    _validate_encoder_metadata(config, metadata, "bert")
    token_types = metadata.get("tokenizer.ggml.token_type_count")
    if token_types is None or int(token_types) <= 0:
        raise ValueError(
            "BERT GGUF requires tokenizer.ggml.token_type_count > 0 for token-type embeddings"
        )
    config.type_vocab_size = int(token_types)
    config.hidden_act = "gelu"
    config.rope_type = None
    config.attn_qkv_bias = _infer_attn_qkv_bias(model)
    config.attn_o_bias = _infer_attn_o_bias(model)
    config.mlp_bias = _infer_mlp_bias(model)
    return config


def _modern_bert_encoder_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Apply pinned ModernBERT bias-free GeGLU/RoPE semantics."""
    _validate_encoder_metadata(config, metadata, "modern-bert")
    del model
    sliding_window = int(metadata.get("modern-bert.attention.sliding_window", 0))
    if sliding_window:
        raise ValueError(
            "modern-bert GGUF requests symmetric sliding-window attention "
            f"(window={sliding_window}), which the current encoder graph does not implement"
        )
    config.type_vocab_size = 0
    config.hidden_act = config.hidden_act or "gelu"
    config.attn_qkv_bias = False
    config.attn_o_bias = False
    config.mlp_bias = False
    return config


def _dflash_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> DFlashConfig:
    from mobius._configs import DFlashConfig

    # The pinned Qwen converter serializes layer-input indices (HF output index + 1).
    # Mobius consumes decoder-layer outputs, so normalize back to zero-based ids.
    target_layers = [int(value) - 1 for value in metadata["dflash.target_layers"]]
    raw_shapes = {name: shape for name, _raw, _qtype, shape in model.tensor_items_raw()}
    draft_vocab = (
        int(raw_shapes["d2t"][0])
        if "d2t" in raw_shapes
        else len(metadata.get("tokenizer.ggml.tokens", ()))
    )
    fields = {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
    fields.update(
        model_type="DFlashDraftModel",
        vocab_size=len(metadata.get("tokenizer.ggml.tokens", ())),
        target_layer_ids=target_layers,
        block_size=int(metadata["dflash.block_size"]),
        num_target_layers=None,
        draft_vocab_size=draft_vocab,
        use_draft_lm_head="output.weight" in model.tensor_names,
    )
    return DFlashConfig(**fields)


def _eagle3_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> Eagle3Config:
    from mobius._configs import Eagle3Config

    target_layers = [int(value) for value in metadata["eagle3.target_layers"]]
    raw_shapes = {name: shape for name, _raw, _qtype, shape in model.tensor_items_raw()}
    target_vocab = len(metadata.get("tokenizer.ggml.tokens", ()))
    draft_vocab = int(raw_shapes["d2t"][0]) if "d2t" in raw_shapes else target_vocab
    fields = {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
    fields.update(
        model_type="Eagle3DraftModel",
        vocab_size=target_vocab,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        draft_vocab_size=draft_vocab,
        target_hidden_size=int(metadata["eagle3.target_hidden_size"]),
        target_layer_ids=target_layers,
        norm_before_residual=bool(metadata.get("eagle3.norm_before_residual")),
        norm_before_fc=bool(metadata.get("eagle3.norm_before_fc")),
        fc_norm=False,
        use_target_lm_head="output.weight" not in model.tensor_names,
    )
    return Eagle3Config(**fields)


# Architecture-specific config postprocessors, keyed by the name a
# :class:`GGUFArchitectureSpec` refers to. Each takes a base ArchitectureConfig
# + raw metadata and returns an architecture-specific config subclass.
#
# Keyed by postprocessor name rather than by ``model_type``: the old
# model_type keying is what let the Gemma weight processor drift out of reach
# when an architecture's model_type gained a ``_text`` suffix.
_CONFIG_POSTPROCESSORS: dict[str, Any] = {
    "dflash": _dflash_postprocess,
    "eagle3": _eagle3_postprocess,
    "dream": _dream_postprocess,
    "llada": _llada_postprocess,
    "llada_moe": _llada_moe_postprocess,
    "rnd1": _rnd1_postprocess,
    "olmo": _olmo_postprocess,
    "moe": _moe_postprocess,
    "granitemoe": _granitemoe_postprocess,
    "phimoe": _phimoe_postprocess,
    "dense_sliding": _dense_sliding_postprocess,
    "gemma2": _gemma2_postprocess,
    "gemma3": _gemma3_postprocess,
    "gemma4": _gemma4_postprocess,
    "muse_glimmer": _muse_glimmer_postprocess,
    "mamba": _mamba_postprocess,
    "mamba2": _mamba2_postprocess,
    "jamba": _jamba_postprocess,
    "nemotron_h": _nemotron_h_postprocess,
    "granitehybrid": _granitehybrid_postprocess,
    "bert_encoder": _bert_encoder_postprocess,
    "modern_bert_encoder": _modern_bert_encoder_postprocess,
    "t5": _t5_postprocess,
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
    gelu_models = {"bert", "gpt2", "bloom", "modernbert", "starcoder2", "t5"}
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
