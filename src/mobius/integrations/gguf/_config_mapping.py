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
import math
import re
from collections.abc import Iterable
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from mobius._configs import (
    ArchitectureConfig,
    FalconH1Config,
    Gemma2Config,
    Gemma4Config,
    GraniteMoeHybridConfig,
    GrokGGUFConfig,
    GroveMoEGGUFConfig,
    HyV3Config,
    JambaConfig,
    KimiK3Config,
    KimiLinearConfig,
    Lfm2MoeConfig,
    Mamba2Config,
    MambaConfig,
    MiniMaxConfig,
    MuseGlimmerConfig,
    NemotronHConfig,
    Plamo2Config,
    Qwen4ExpConfig,
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
    "expert_shared_count": "n_shared_experts",
    "expert_weights_scale": "routed_scaling_factor",
    "expert_weights_norm": "norm_topk_prob",
    "expert_group_count": "n_group",
    "expert_group_used_count": "topk_group",
    "moe_latent_size": "moe_latent_size",
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
    "attention.key_length_mla": "head_dim",
    "rope.dimension_count": "qk_rope_head_dim",
    "attention.q_lora_rank": "q_lora_rank",
    "attention.kv_lora_rank": "kv_lora_rank",
    "attention.value_length_mla": "v_head_dim",
    "attention.sliding_window": "sliding_window",
    "expert_count": "num_local_experts",
    "expert_used_count": "num_experts_per_tok",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_shared_count": "n_shared_experts",
    "expert_weights_scale": "routed_scaling_factor",
    "expert_weights_norm": "norm_topk_prob",
    "expert_group_count": "n_group",
    "expert_group_used_count": "topk_group",
    "leading_dense_block_count": "first_k_dense_replace",
    "attention.indexer.head_count": "index_n_heads",
    "attention.indexer.key_length": "index_head_dim",
    "attention.indexer.top_k": "index_topk",
}

_MINIMAX_M2_KEY_MAP = {
    "attention.key_length": "head_dim",
    "expert_feed_forward_length": "moe_intermediate_size",
}

_MISTRAL4_KEY_MAP = {
    "attention.key_length_mla": "head_dim",
    "attention.q_lora_rank": "q_lora_rank",
    "attention.kv_lora_rank": "kv_lora_rank",
    "expert_feed_forward_length": "moe_intermediate_size",
    "leading_dense_block_count": "first_k_dense_replace",
}

_MAMBA_KEY_MAP = {
    "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
    "ssm.conv_kernel": "conv_kernel",
    "ssm.group_count": "n_groups",
    "ssm.inner_size": "intermediate_size",
    "ssm.state_size": "state_size",
    "ssm.time_step_rank": "time_step_rank",
}

_FALCON_H1_KEY_MAP = {
    "attention.key_length": "head_dim",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "ssm.conv_kernel": "mamba_d_conv",
    "ssm.group_count": "mamba_n_groups",
    "ssm.inner_size": "mamba_d_ssm",
    "ssm.state_size": "mamba_d_state",
    "ssm.time_step_rank": "mamba_n_heads",
}

_PLAMO2_KEY_MAP = {
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "ssm.conv_kernel": "mamba_d_conv",
    "ssm.group_count": "mamba_group_count",
    "ssm.state_size": "mamba_d_state",
    "ssm.time_step_rank": "mamba_num_heads",
}

_PLM_KEY_MAP = {
    "attention.key_length": "head_dim",
    "attention.value_length": "v_head_dim",
    "attention.kv_lora_rank": "kv_lora_rank",
    "rope.dimension_count": "qk_rope_head_dim",
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

_MINIMAX_KEY_MAP = {
    "attention.key_length": "head_dim",
    "attention.value_length": "value_head_dim",
    "residual_scale": "residual_scale",
}

_KIMI_LINEAR_KEY_MAP = {
    "attention.key_length_mla": "qk_nope_head_dim",
    "attention.value_length_mla": "v_head_dim",
    "attention.kv_lora_rank": "kv_lora_rank",
    "rope.dimension_count": "qk_rope_head_dim",
    "ssm.conv_kernel": "linear_conv_kernel_dim",
    "kda.head_dim": "linear_key_head_dim",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_shared_count": "n_shared_experts",
    "leading_dense_block_count": "first_k_dense_replace",
    "expert_weights_scale": "routed_scaling_factor",
}

_KIMI_K3_KEY_MAP = {
    **_KIMI_LINEAR_KEY_MAP,
    "attention.q_lora_rank": "q_lora_rank",
    "kda.gate_lower_bound": "linear_gate_lower_bound",
    "expert_latent_length": "routed_expert_hidden_size",
    "expert_weights_norm": "norm_topk_prob",
    "attn_res.block_size": "attn_res_block_size",
    "activation.situ_beta": "activation_situ_beta",
    "activation.situ_linear_beta": "activation_situ_linear_beta",
}

_MINICPM_KEY_MAP = {
    "embedding_scale": "embedding_multiplier",
    "residual_scale": "residual_multiplier",
    # The pinned converter serializes hidden_size / dim_model_base. The graph
    # divides the normalized hidden state by this value before the LM head.
    "logit_scale": "logits_scaling",
}

_MINICPM3_KEY_MAP = {
    "attention.q_lora_rank": "q_lora_rank",
    "attention.kv_lora_rank": "kv_lora_rank",
    "rope.dimension_count": "qk_rope_head_dim",
}

_CONVENTIONAL_SHARED_MOE_KEY_MAP = {
    "leading_dense_block_count": "first_k_dense_replace",
}

_ERNIE45_MOE_KEY_MAP = {
    "leading_dense_block_count": "first_k_dense_replace",
    "interleave_moe_layer_step": "moe_layer_frequency",
}

_QWEN4EXP_KEY_MAP = {
    "attention.key_length": "head_dim",
    "expert_count": "num_local_experts",
    "expert_used_count": "num_experts_per_tok",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_shared_feed_forward_length": "shared_expert_intermediate_size",
    "hyper_connection.count": "hc_count",
    "hyper_connection.low_rank": "hc_lowrank",
    "attention.indexer.head_count": "indexer_n_heads",
    "attention.indexer.key_length": "indexer_head_dim",
    "attention.indexer.top_k": "indexer_budget",
    "ple.ngram_size": "ngram_size",
    "ple.heads_per_ngram": "heads_per_ngram",
    "ple.conv_kernel": "ple_conv_kernel_size",
    "ple.eos_token_id": "eos_token_id",
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
        "minimax_m2": _MINIMAX_M2_KEY_MAP,
        "mistral4": _MISTRAL4_KEY_MAP,
        "mamba": _MAMBA_KEY_MAP,
        "falcon_h1": _FALCON_H1_KEY_MAP,
        "plamo2": _PLAMO2_KEY_MAP,
        "plm": _PLM_KEY_MAP,
        "jamba": _JAMBA_KEY_MAP,
        "nemotron_h": _NEMOTRON_H_KEY_MAP,
        "granitehybrid": _GRANITEHYBRID_KEY_MAP,
        "minimax": _MINIMAX_KEY_MAP,
        "kimi_linear": _KIMI_LINEAR_KEY_MAP,
        "kimi_k3": _KIMI_K3_KEY_MAP,
        "minicpm": _MINICPM_KEY_MAP,
        "minicpm3": _MINICPM3_KEY_MAP,
        "conventional_shared_moe": _CONVENTIONAL_SHARED_MOE_KEY_MAP,
        "ernie45_moe": _ERNIE45_MOE_KEY_MAP,
        "qwen4exp": _QWEN4EXP_KEY_MAP,
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


_DELTA_NET_ARCHITECTURES = frozenset({"qwen35", "qwen35moe", "qwen3next", "qwen4exp"})


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


def _validate_closed_rope_scaling_metadata(
    metadata: dict[str, Any],
    arch: str,
    *,
    allowed_suffixes: set[str] | None = None,
) -> None:
    """Reject RoPE-scaling metadata outside an architecture's exact subset."""
    allowed = {f"{arch}.rope.scaling.{suffix}" for suffix in (allowed_suffixes or set())}
    unsupported = {
        key
        for key in metadata
        if key.startswith(f"{arch}.rope.scaling.") and key not in allowed
    }
    unsupported.update(
        key
        for key in (
            f"{arch}.rope.scale_linear",
            f"{arch}.rope.factor",
            f"{arch}.rope.original_context",
        )
        if key in metadata
    )
    if unsupported:
        raise ValueError(
            f"{arch} has unsupported RoPE scaling metadata: {', '.join(sorted(unsupported))}"
        )
    scaling_type = metadata.get(f"{arch}.rope.scaling.type")
    if "type" in (allowed_suffixes or set()) and scaling_type not in (None, "", "none"):
        raise ValueError(
            f"{arch} rope.scaling.type={scaling_type!r} is not in the exact supported subset"
        )


def _derive_hybrid_layout(
    gguf_arch: str,
    metadata: dict[str, Any],
    tensor_names: Iterable[str] | None = None,
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

    if gguf_arch in {
        "lfm2",
        "lfm2moe",
        "jamba",
        "granitehybrid",
        "plamo2",
        "kimi-linear",
        "kimi-k3",
    }:
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
            "lfm2moe": "conv",
            "jamba": "mamba",
            "granitehybrid": "mamba2",
            "plamo2": "mamba",
            "kimi-linear": "kimi_linear_attention",
            "kimi-k3": "kimi_k3_attention",
        }[gguf_arch]
        return (
            trunk_layers,
            [
                recurrent_type if value == 0 else "full_attention"
                for value in kv_heads[:trunk_layers]
            ],
            mtp_count,
        )

    if gguf_arch in {"nemotron_h", "nemotron_h_moe"}:
        kv_raw = metadata.get(f"{gguf_arch}.attention.head_count_kv")
        ffn_raw = metadata.get(f"{gguf_arch}.feed_forward_length")
        if not isinstance(kv_raw, (list, tuple, np.ndarray)) or not isinstance(
            ffn_raw, (list, tuple, np.ndarray)
        ):
            raise ValueError(
                f"{gguf_arch} requires per-layer attention.head_count_kv and "
                "feed_forward_length arrays"
            )
        kv_heads = [int(value) for value in kv_raw]
        ffn_lengths = [int(value) for value in ffn_raw]
        if len(kv_heads) != total_layers or len(ffn_lengths) != total_layers:
            raise ValueError(
                f"{gguf_arch} schedule arrays must each contain exactly {total_layers} entries"
            )
        if any(value < 0 for value in (*kv_heads, *ffn_lengths)):
            raise ValueError(f"{gguf_arch} schedule entries must be non-negative")
        uses_moe = int(metadata.get(f"{gguf_arch}.expert_count", 0)) > 0
        names = set(tensor_names) if tensor_names is not None else None
        layer_types = []
        for layer, (kv_heads_i, ffn_length_i) in enumerate(
            zip(kv_heads[:trunk_layers], ffn_lengths[:trunk_layers])
        ):
            if ffn_length_i:
                has_router = names is not None and f"blk.{layer}.ffn_gate_inp.weight" in names
                layer_types.append(
                    "moe" if has_router or (names is None and uses_moe) else "mlp"
                )
            elif kv_heads_i == 0:
                layer_types.append("mamba2")
            else:
                layer_types.append("full_attention")
        return trunk_layers, layer_types, mtp_count

    if gguf_arch == "minimax-01":
        recurrent_key = f"{gguf_arch}.attention.recurrent_layers"
        raw_recurrent = metadata.get(recurrent_key)
        if raw_recurrent is None:
            interval = int(metadata.get(f"{gguf_arch}.full_attention_interval", 8))
            if interval <= 0:
                raise ValueError(
                    f"{gguf_arch}.full_attention_interval must be positive, got {interval}"
                )
            recurrent = [(layer + 1) % interval != 0 for layer in range(total_layers)]
        else:
            if not isinstance(raw_recurrent, (list, tuple, np.ndarray)):
                raise ValueError(f"{recurrent_key} must be a boolean array")
            if len(raw_recurrent) != total_layers:
                raise ValueError(
                    f"{recurrent_key} must contain exactly {total_layers} entries, "
                    f"got {len(raw_recurrent)}"
                )
            if any(
                not isinstance(value, (bool, np.bool_, int, np.integer))
                or int(value) not in (0, 1)
                for value in raw_recurrent
            ):
                raise ValueError(f"{recurrent_key} entries must be booleans or 0/1")
            recurrent = [bool(value) for value in raw_recurrent]
            if f"{gguf_arch}.full_attention_interval" in metadata:
                interval = int(metadata[f"{gguf_arch}.full_attention_interval"])
                if interval <= 0:
                    raise ValueError(
                        f"{gguf_arch}.full_attention_interval must be positive, got {interval}"
                    )
                periodic = [(layer + 1) % interval != 0 for layer in range(total_layers)]
                if recurrent != periodic:
                    raise ValueError(
                        "MiniMax-01 recurrent_layers contradicts full_attention_interval"
                    )
        return (
            trunk_layers,
            [
                "lightning_attention" if value else "full_attention"
                for value in recurrent[:trunk_layers]
            ],
            mtp_count,
        )

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
    if isinstance(num_attention_heads, (list, np.ndarray)):
        values = [int(value) for value in num_attention_heads]
        if canonical_arch not in {"openelm", "plamo2", "nemotron_h", "nemotron_h_moe"}:
            raise ValueError(
                f"{canonical_arch} has unsupported per-layer attention head counts"
            )
        nonzero = {value for value in values if value}
        if canonical_arch == "openelm":
            if not values or min(values) <= 0:
                raise ValueError("openelm attention head counts must all be positive")
            num_attention_heads = values[0]
            nonzero = set()
        if nonzero and len(nonzero) != 1:
            raise ValueError(
                f"{canonical_arch} attention layers must use one consistent non-zero "
                "head count"
            )
        if canonical_arch != "openelm" and not nonzero:
            raise ValueError(f"{canonical_arch} GGUF has no attention layer")
        if nonzero:
            num_attention_heads = nonzero.pop()
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
        if canonical_arch in {
            "lfm2",
            "lfm2moe",
            "jamba",
            "nemotron_h",
            "nemotron_h_moe",
            "granitehybrid",
            "plamo2",
            "minimax-01",
            "kimi-linear",
        }:
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
        elif canonical_arch == "openelm":
            if not values or min(values) <= 0:
                raise ValueError("openelm KV-head counts must all be positive")
            num_kv_heads = values[0]
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
            "lfm2moe",
            "jamba",
            "nemotron_h",
            "nemotron_h_moe",
            "granitehybrid",
            "plamo2",
            "kimi-linear",
        }
        or canonical_arch in _DELTA_NET_ARCHITECTURES
    ):
        derived_layers, layer_types, derived_mtp_count = _derive_hybrid_layout(
            canonical_arch, metadata, model.tensor_names
        )
        if derived_layers != int(num_hidden_layers) or derived_mtp_count != mtp_predict_layers:
            raise ValueError("Hybrid schedule and decoder layer metadata disagree")
        if canonical_arch in _DELTA_NET_ARCHITECTURES:
            full_attention_interval = int(
                metadata.get(f"{canonical_arch}.full_attention_interval", 4)
            )

    if canonical_arch in {"nemotron_h", "nemotron_h_moe"}:
        ffn_lengths = metadata[f"{gguf_arch}.feed_forward_length"]
        nonzero_ffn_lengths = {int(value) for value in ffn_lengths if int(value)}
        if len(nonzero_ffn_lengths) > 1:
            raise ValueError(
                f"{gguf_arch} FFN layers must use one consistent feed-forward length"
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

    if canonical_arch in {"qwen2vl", "qwen35", "qwen35moe", "qwen4exp"}:
        mrope_section = metadata[f"{gguf_arch}.rope.dimension_sections"]
        expected_sections = 3 if canonical_arch == "qwen4exp" else 4
        if (
            not isinstance(mrope_section, (list, tuple, np.ndarray))
            or len(mrope_section) != expected_sections
        ):
            raise ValueError(
                f"{gguf_arch}.rope.dimension_sections must contain exactly "
                f"{expected_sections} entries"
            )
        mrope_section = [int(value) for value in mrope_section]
        if canonical_arch == "qwen2vl":
            if mrope_section[-1] != 0:
                raise ValueError("qwen2vl.rope.dimension_sections reserved entry must be zero")
            mrope_section = mrope_section[:3]
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

    if canonical_arch == "openelm" and isinstance(
        hf_fields.get("intermediate_size"), (list, tuple, np.ndarray)
    ):
        hf_fields["intermediate_size"] = int(hf_fields["intermediate_size"][0])

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
        mrope_interleaved=canonical_arch in {"qwen35", "qwen35moe"},
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
    if gguf_arch == "hy_v3" and mtp_block_indices:
        probe = f"blk.{mtp_block_indices[0]}.nextn.eh_proj.weight"
        if probe not in set(model.tensor_names):
            # llama.cpp permits a target-only split file whose metadata retains
            # the appended block count while the MTP tensors live elsewhere.
            mtp_block_indices = []
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


def _lfm2moe_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> Lfm2MoeConfig:
    """Restore LFM2MoE fields serialized by the pinned llama.cpp converter.

    The pinned loader defaults a missing dense-prefix length to zero and
    requires the SIGMOID gating enum. Its graph always normalizes selected
    probabilities, and the architecture loader does not read a scaling
    override, so metadata that conflicts with those invariants is rejected.
    """
    del model
    arch = "lfm2moe"
    raw_gating = metadata[f"{arch}.expert_gating_func"]
    gating_value = float(raw_gating)
    if not math.isfinite(gating_value) or not gating_value.is_integer():
        raise ValueError(f"{arch}.expert_gating_func must be an integer")
    gating = int(gating_value)
    if gating != 2:
        raise ValueError(f"{arch}.expert_gating_func must be SIGMOID (2), got {gating}")
    num_dense_layers = int(metadata.get(f"{arch}.leading_dense_block_count", 0))
    if not 0 <= num_dense_layers <= config.num_hidden_layers:
        raise ValueError(
            f"{arch}.leading_dense_block_count must be in [0, "
            f"{config.num_hidden_layers}], got {num_dense_layers}"
        )
    if (
        config.num_local_experts is None
        or config.num_experts_per_tok is None
        or config.moe_intermediate_size is None
    ):
        raise ValueError("lfm2moe requires expert count, top-k, and expert FFN width")
    if not 0 < config.num_experts_per_tok <= config.num_local_experts:
        raise ValueError(
            "lfm2moe expert_used_count must be positive and no greater than expert_count"
        )

    if metadata.get(f"{arch}.expert_weights_norm", True) is not True:
        raise ValueError(
            "lfm2moe.expert_weights_norm=False is incompatible with the pinned "
            "llama.cpp graph, which always normalizes selected expert weights"
        )
    expert_scale = float(metadata.get(f"{arch}.expert_weights_scale", 1.0))
    if expert_scale != 1.0:  # noqa: RUF069
        raise ValueError(
            "lfm2moe.expert_weights_scale must be 1.0 because the pinned loader "
            "does not read an architecture-specific override"
        )

    fields = _shallow_fields(config)
    fields.update(
        hidden_act="silu",
        attn_qk_norm=True,
        short_conv_bias=False,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        use_expert_bias=True,
    )
    return Lfm2MoeConfig(
        **fields,
        num_dense_layers=num_dense_layers,
    )


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


def _baichuan_postprocess(
    config: ArchitectureConfig, metadata: dict[str, Any], model: Any
) -> ArchitectureConfig:
    del metadata, model
    if config.num_hidden_layers != 32:
        raise ValueError(
            "Baichuan GGUF import supports only the pinned 32-layer/7B RoPE profile; "
            f"got block_count={config.num_hidden_layers}. The 40-layer/13B loader uses "
            "a hardcoded ALiBi path that the Mobius graph does not represent."
        )
    return dataclasses.replace(
        config,
        num_key_value_heads=config.num_attention_heads,
        rope_type="default",
        partial_rotary_factor=1.0,
        tie_word_embeddings=False,
        attn_qkv_bias=False,
        attn_o_bias=False,
        mlp_bias=False,
        hidden_act="silu",
    )


def _chatglm_postprocess(
    config: ArchitectureConfig, metadata: dict[str, Any], model: Any
) -> ArchitectureConfig:
    names = set(model.tensor_names)
    head_dim = int(
        metadata.get(
            "chatglm.attention.key_length",
            config.hidden_size // config.num_attention_heads,
        )
    )
    rope_dim = int(metadata.get("chatglm.rope.dimension_count", head_dim))
    qkv_biases = [
        f"blk.{layer}.attn_qkv.bias" in names
        or all(
            f"blk.{layer}.attn_{projection}.bias" in names for projection in ("q", "k", "v")
        )
        for layer in range(config.num_hidden_layers)
    ]
    return dataclasses.replace(
        config,
        head_dim=head_dim,
        partial_rotary_factor=rope_dim / head_dim,
        rope_type="default",
        hidden_act="silu",
        tie_word_embeddings="output.weight" not in names,
        attn_qkv_bias=all(qkv_biases),
        attn_o_bias=False,
        mlp_bias=False,
    )


def _phi2_postprocess(
    config: ArchitectureConfig, metadata: dict[str, Any], model: Any
) -> ArchitectureConfig:
    del model
    head_dim = int(
        metadata.get(
            "phi2.attention.key_length",
            config.hidden_size // config.num_attention_heads,
        )
    )
    rope_dim = int(metadata.get("phi2.rope.dimension_count", head_dim))
    return dataclasses.replace(
        config,
        head_dim=head_dim,
        partial_rotary_factor=rope_dim / head_dim,
        intermediate_size=4 * config.hidden_size,
        num_key_value_heads=config.num_attention_heads,
        rope_type="default",
        hidden_act="gelu_new",
        tie_word_embeddings=False,
        attn_qkv_bias=True,
        attn_o_bias=True,
        mlp_bias=True,
    )


def _seed_oss_postprocess(
    config: ArchitectureConfig, metadata: dict[str, Any], model: Any
) -> ArchitectureConfig:
    del metadata
    return dataclasses.replace(
        config,
        rope_type="default",
        partial_rotary_factor=1.0,
        tie_word_embeddings="output.weight" not in set(model.tensor_names),
        attn_qkv_bias=_infer_attn_qkv_bias(model),
        attn_o_bias=False,
        mlp_bias=False,
        hidden_act="silu",
    )


def _apertus_postprocess(
    config: ArchitectureConfig, metadata: dict[str, Any], model: Any
) -> ArchitectureConfig:
    """Restore Apertus values serialized outside ordinary GGUF weight tensors."""
    arch = "apertus"
    layers = config.num_hidden_layers

    def per_layer(suffix: str) -> tuple[float, ...]:
        raw = metadata[f"{arch}.xielu.{suffix}"]
        values = list(raw) if isinstance(raw, (list, tuple, np.ndarray)) else [raw] * layers
        if len(values) != layers:
            raise ValueError(
                f"{arch}.xielu.{suffix} must be a scalar or contain {layers} values, "
                f"got {len(values)}"
            )
        result = tuple(float(value) for value in values)
        if not all(np.isfinite(result)):
            raise ValueError(f"{arch}.xielu.{suffix} must contain only finite values")
        return result

    names = set(model.tensor_names)
    rope_names = names & {
        "rope_freqs.weight",
        "rope_factors_long.weight",
        "rope_factors_short.weight",
    }
    raw_types = {
        name: getattr(qtype, "value", qtype)
        for name, _raw, qtype, _shape in model.tensor_items_raw()
        if name in rope_names
    }
    if any(type_id not in {0, 1, 30} for type_id in raw_types.values()):
        raise ValueError("Apertus serialized RoPE factors must use F32/F16/BF16 storage")

    if rope_names == {"rope_freqs.weight"}:
        factors = np.asarray(model.get_tensor("rope_freqs.weight"), dtype=np.float32).reshape(
            -1
        )
        short_factors = long_factors = factors
        original_context = config.max_position_embeddings
    elif rope_names == {"rope_factors_long.weight", "rope_factors_short.weight"}:
        original_context_raw = metadata.get(f"{arch}.rope.scaling.original_context_length")
        if original_context_raw is None:
            raise ValueError(
                "Apertus LongRoPE factors require apertus.rope.scaling.original_context_length"
            )
        original_context = int(original_context_raw)
        if original_context <= 1 or original_context > config.max_position_embeddings:
            raise ValueError(
                "Apertus LongRoPE original context length must be greater than 1 "
                "and no larger than the configured context length"
            )
        long_factors = np.asarray(
            model.get_tensor("rope_factors_long.weight"), dtype=np.float32
        ).reshape(-1)
        short_factors = np.asarray(
            model.get_tensor("rope_factors_short.weight"), dtype=np.float32
        ).reshape(-1)
    else:
        raise ValueError(
            "Apertus GGUF must contain exactly rope_freqs.weight or the complete "
            "rope_factors_long.weight/rope_factors_short.weight pair"
        )

    expected = config.head_dim // 2
    for name, factors in (("short", short_factors), ("long", long_factors)):
        if (
            factors.shape != (expected,)
            or not np.all(np.isfinite(factors))
            or np.any(factors <= 0)
        ):
            raise ValueError(
                f"Apertus {name} RoPE factors must have shape ({expected},) and be "
                "finite positive values"
            )

    q_biases = tuple(f"blk.{layer}.attn_q_norm.bias" in names for layer in range(layers))
    k_biases = tuple(f"blk.{layer}.attn_k_norm.bias" in names for layer in range(layers))
    return dataclasses.replace(
        config,
        hidden_act="xielu",
        attn_qk_norm=True,
        attn_q_norm_biases=q_biases,
        attn_k_norm_biases=k_biases,
        rope_type="longrope",
        rope_scaling={
            "short_factor": short_factors.tolist(),
            "long_factor": long_factors.tolist(),
        },
        original_max_position_embeddings=original_context,
        xielu_alpha_p=per_layer("alpha_p"),
        xielu_alpha_n=per_layer("alpha_n"),
        xielu_beta=per_layer("beta"),
        xielu_eps=per_layer("eps"),
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


def _qwen3_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> ArchitectureConfig:
    """Apply Qwen3's per-head Q/K normalization omitted by GGUF metadata."""
    return dataclasses.replace(config, attn_qk_norm=True, attn_qk_norm_full=False)


def _starcoder2_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> ArchitectureConfig:
    """Restore StarCoder2's architecture-owned uniform sliding window."""
    del metadata, model
    if config.sliding_window is not None:
        return config
    return dataclasses.replace(config, sliding_window=4096)


def _dbrx_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> ArchitectureConfig:
    """Restore the fused-clamped attention and LayerNorm DBRX profile."""
    config = _moe_postprocess(config, metadata, model)
    arch = "dbrx"
    clamp = float(metadata[f"{arch}.attention.clamp_kqv"])
    if not math.isfinite(clamp) or clamp < 0:
        raise ValueError(f"{arch}.attention.clamp_kqv must be finite and non-negative")
    return dataclasses.replace(
        config,
        hidden_act="silu",
        attention_clamp=clamp,
        attn_qkv_bias=False,
        attn_o_bias=False,
        mlp_bias=False,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        tie_word_embeddings=False,
        rope_type="default",
    )


def _arctic_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> ArchitectureConfig:
    """Restore Arctic's dual-branch routing and optional tied output."""
    config = _moe_postprocess(config, metadata, model)
    route_scale = float(metadata.get("arctic.expert_weights_scale", 1.0))
    if math.isclose(route_scale, 0.0):
        route_scale = 1.0
    if not math.isfinite(route_scale) or route_scale <= 0:
        raise ValueError("arctic.expert_weights_scale must be finite and positive")
    return dataclasses.replace(
        config,
        hidden_act="silu",
        attn_qkv_bias=False,
        attn_o_bias=False,
        mlp_bias=False,
        norm_topk_prob=True,
        routed_scaling_factor=route_scale,
        tie_word_embeddings="output.weight" not in model.tensor_names,
        rope_type="default",
    )


def _grok_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> GrokGGUFConfig:
    """Restore Grok's defaults, sandwich norms, and optional dense branch."""
    arch = "grok"
    _validate_conventional_moe_rope_scaling(metadata, arch)
    expert_width = int(metadata.get(f"{arch}.expert_feed_forward_length", 0))
    if expert_width < 0:
        raise ValueError("grok.expert_feed_forward_length must be non-negative")
    config = dataclasses.replace(
        config,
        moe_intermediate_size=expert_width or config.intermediate_size,
    )
    config = _moe_postprocess(config, metadata, model)

    def finite(name: str, default: float) -> float:
        value = float(metadata.get(f"{arch}.{name}", default))
        if not math.isfinite(value):
            raise ValueError(f"{arch}.{name} must be finite")
        return value

    embedding_scale = finite("embedding_scale", 78.38367176906169)
    rms_norm_eps = finite(
        "attention.layer_norm_rms_epsilon",
        config.rms_norm_eps,
    )
    attention_output_scale = finite(
        "attention.output_scale",
        0.08838834764831845,
    )
    logit_output_scale = finite("logit_scale", 0.5773502691896257)
    attn_softcap = finite("attn_logit_softcapping", 30.0)
    router_softcap = finite("router_logit_softcapping", 30.0)
    final_softcap = finite("final_logit_softcapping", 0.0)
    temperature_length = int(metadata.get(f"{arch}.attention.temperature_length", 0))
    if math.isclose(logit_output_scale, 0.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("grok.logit_scale must be nonzero")
    if rms_norm_eps <= 0:
        raise ValueError("grok.attention.layer_norm_rms_epsilon must be positive")
    if attn_softcap <= 0:
        raise ValueError("grok.attn_logit_softcapping must be positive")
    if final_softcap < 0:
        raise ValueError("grok.final_logit_softcapping must be non-negative")
    if temperature_length < 0:
        raise ValueError("grok.attention.temperature_length must be non-negative")

    tensor_names = set(model.tensor_names)
    has_dense_ffn = any(re.match(r"^blk\.\d+\.ffn_up\.weight$", name) for name in tensor_names)
    has_gated_dense_ffn = any(
        re.match(r"^blk\.\d+\.ffn_gate\.weight$", name) for name in tensor_names
    )
    has_gated_experts = any(
        re.match(r"^blk\.\d+\.ffn_gate_exps\.weight$", name) for name in tensor_names
    )

    rope_scaling = dict(config.rope_scaling or {})
    if config.rope_type == "yarn":
        rope_scaling.update(
            beta_fast=finite("rope.scaling.yarn_beta_fast", 8.0),
            beta_slow=finite("rope.scaling.yarn_beta_slow", 1.0),
        )

    fields = _shallow_fields(config)
    fields.update(
        model_type="grok_gguf",
        hidden_act="gelu_new",
        rms_norm_eps=rms_norm_eps,
        mlp_bias=False,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        tie_word_embeddings="output.weight" not in tensor_names,
        rope_scaling=rope_scaling or None,
        embedding_scale=embedding_scale,
        attention_output_scale=attention_output_scale,
        logit_output_scale=logit_output_scale,
        attn_logit_softcapping=attn_softcap,
        router_logit_softcapping=router_softcap,
        final_logit_softcapping=final_softcap,
        attention_temperature_length=temperature_length,
        has_dense_ffn=has_dense_ffn,
        has_gated_dense_ffn=has_gated_dense_ffn,
        has_gated_experts=has_gated_experts,
    )
    return GrokGGUFConfig(**fields)


def _grovemoe_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> GroveMoEGGUFConfig:
    """Restore GroveMoE's sigmoid-selected primary and chunk expert banks."""
    arch = "grovemoe"
    config = _moe_postprocess(config, metadata, model)
    num_experts = config.num_local_experts
    if num_experts is None:
        raise ValueError("grovemoe.expert_count must be positive")
    chunk_width = int(metadata.get(f"{arch}.expert_chunk_feed_forward_length", 0))
    if chunk_width == 0:
        chunk_width = config.head_dim
    experts_per_group = int(metadata[f"{arch}.experts_per_group"])
    group_scale = float(metadata[f"{arch}.expert_group_scale"])
    if chunk_width <= 0:
        raise ValueError("grovemoe.expert_chunk_feed_forward_length must be positive")
    if experts_per_group <= 0 or num_experts % experts_per_group:
        raise ValueError("grovemoe.experts_per_group must be positive and divide expert_count")
    if not math.isfinite(group_scale):
        raise ValueError("grovemoe.expert_group_scale must be finite")

    fields = _shallow_fields(config)
    fields.update(
        model_type="grovemoe_gguf",
        hidden_act="silu",
        attn_qk_norm=True,
        attn_qk_norm_full=False,
        mlp_bias=False,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        tie_word_embeddings="output.weight" not in model.tensor_names,
        chunk_expert_intermediate_size=chunk_width,
        experts_per_group=experts_per_group,
        expert_group_scale=group_scale,
    )
    return GroveMoEGGUFConfig(**fields)


def _hunyuan_moe_gguf_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Restore Hunyuan-MoE's post-RoPE Q/K norm and shared-expert profile."""
    arch = "hunyuan-moe"
    config = _moe_postprocess(config, metadata, model)
    expert_width = int(metadata[f"{arch}.expert_feed_forward_length"])
    if expert_width != config.intermediate_size:
        raise ValueError(
            "hunyuan-moe.expert_feed_forward_length must equal feed_forward_length "
            "because the pinned loader uses feed_forward_length for routed expert tensors"
        )
    shared_width = (
        int(metadata.get(f"{arch}.expert_shared_feed_forward_length", 0))
        or config.intermediate_size
    )
    shared_count = int(metadata.get(f"{arch}.expert_shared_count", 1))
    if shared_width <= 0:
        raise ValueError(
            "hunyuan-moe.expert_shared_feed_forward_length must be positive when present"
        )
    if shared_count != 1:
        raise ValueError(
            "hunyuan-moe.expert_shared_count must be one; the pinned graph owns one "
            "ungated shared expert"
        )
    return dataclasses.replace(
        config,
        model_type="hunyuan_moe_gguf",
        hidden_act="silu",
        attn_qk_norm=True,
        attn_qk_norm_full=False,
        mlp_bias=False,
        moe_intermediate_size=config.intermediate_size,
        shared_expert_intermediate_size=shared_width,
        n_shared_experts=None,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        tie_word_embeddings="output.weight" not in model.tensor_names,
    )


def _ernie45_shared_expert_width(
    metadata: dict[str, Any],
    tensor_names: Iterable[str],
    routed_layers: Iterable[int],
) -> tuple[int | None, int | None]:
    """Resolve ERNIE's merged shared MLP width and optional width multiplier."""
    width = int(metadata.get("ernie4_5-moe.expert_shared_feed_forward_length", 0))
    count_value = metadata.get("ernie4_5-moe.expert_shared_count")
    count = None if count_value is None else int(count_value)
    expected = {
        f"blk.{layer}.ffn_{projection}_shexp.weight"
        for layer in routed_layers
        for projection in ("gate", "up", "down")
    }
    present = expected & set(tensor_names)

    if width < 0:
        raise ValueError("ernie4_5-moe.expert_shared_feed_forward_length must be non-negative")
    if width == 0:
        if present:
            raise ValueError(
                "ernie4_5-moe shared-expert tensors require a positive shared FFN width"
            )
        if count not in {None, 0}:
            raise ValueError(
                "ernie4_5-moe.expert_shared_count requires a positive shared FFN width"
            )
        return None, count

    if present != expected:
        raise ValueError(
            "ernie4_5-moe positive shared FFN width requires complete shared-expert tensors"
        )
    if count is not None:
        expert_width = int(metadata["ernie4_5-moe.expert_feed_forward_length"])
        if count <= 0 or width != count * expert_width:
            expected_width = count * expert_width
            raise ValueError(
                "ernie4_5-moe.expert_shared_count is inconsistent with its merged shared "
                f"FFN width: expected {expected_width}, got {width}"
            )
    return width, count


def _ernie45_moe_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Restore ERNIE's periodic bias-corrected routed/shared MoE profile."""
    config = _moe_postprocess(config, metadata, model)
    frequency = int(metadata["ernie4_5-moe.interleave_moe_layer_step"])
    dense_prefix = int(metadata.get("ernie4_5-moe.leading_dense_block_count", 0))
    if frequency <= 0:
        raise ValueError("ernie4_5-moe.interleave_moe_layer_step must be positive")
    if not 0 <= dense_prefix <= config.num_hidden_layers:
        raise ValueError("ernie4_5-moe.leading_dense_block_count is out of range")
    routed_layers = [
        layer
        for layer in range(config.num_hidden_layers)
        if layer >= dense_prefix and (layer + 1) % frequency == 0
    ]
    if not routed_layers:
        raise ValueError("ernie4_5-moe schedule must select at least one routed layer")
    names = set(model.tensor_names)
    correction = {f"blk.{layer}.exp_probs_b.bias" for layer in routed_layers}
    present_correction = correction & names
    if present_correction and present_correction != correction:
        raise ValueError("ernie4_5-moe correction bias must be complete across routed layers")
    shared_width, shared_count = _ernie45_shared_expert_width(
        metadata,
        names,
        routed_layers,
    )
    return dataclasses.replace(
        config,
        hidden_act="silu",
        first_k_dense_replace=dense_prefix,
        moe_layer_frequency=frequency,
        scoring_func="softmax",
        topk_method="greedy",
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routing_weight_normalization_floor=6.103515625e-5,
        routed_scaling_factor=1.0,
        use_expert_bias=bool(present_correction),
        shared_expert_intermediate_size=shared_width,
        n_shared_experts=shared_count,
        attn_qkv_bias=False,
        attn_o_bias=False,
        mlp_bias=False,
        tie_word_embeddings="output.weight" not in names,
        rope_type="default",
        rope_interleave=True,
    )


def _smallthinker_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Restore the pinned llama.cpp SmallThinker routing and layer schedule."""
    arch = "smallthinker"
    _validate_closed_rope_scaling_metadata(metadata, arch, allowed_suffixes={"type"})
    raw_gating = metadata[f"{arch}.expert_gating_func"]
    gating_value = float(raw_gating)
    if not math.isfinite(gating_value) or not gating_value.is_integer():
        raise ValueError("smallthinker.expert_gating_func must be an integer")
    gating = int(gating_value)
    if gating not in (1, 2):
        raise ValueError(
            f"smallthinker.expert_gating_func must be SOFTMAX (1) or SIGMOID (2), got {gating}"
        )
    if config.num_local_experts is None or config.num_experts_per_tok is None:
        raise ValueError("SmallThinker requires expert_count and expert_used_count")
    if not 1 <= config.num_experts_per_tok <= config.num_local_experts:
        raise ValueError(
            "smallthinker.expert_used_count must be in "
            f"[1, {config.num_local_experts}], got {config.num_experts_per_tok}"
        )
    if config.moe_intermediate_size is None or config.moe_intermediate_size <= 0:
        raise ValueError("smallthinker.expert_feed_forward_length must be positive")
    if config.intermediate_size != config.moe_intermediate_size:
        raise ValueError(
            "smallthinker.feed_forward_length must equal "
            "smallthinker.expert_feed_forward_length"
        )
    if metadata.get(f"{arch}.expert_weights_norm", True) is not True:
        raise ValueError("SmallThinker requires normalized top-k expert weights")

    raw_route_scale = metadata.get(f"{arch}.expert_weights_scale", 0.0)
    route_scale = float(raw_route_scale)
    if not math.isfinite(route_scale) or not math.isclose(
        route_scale, 0.0, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(
            "smallthinker.expert_weights_scale must be absent or the zero sentinel "
            "because the pinned loader does not consume a routing scale"
        )
    route_scale = 1.0

    scaling_type = metadata.get(f"{arch}.rope.scaling.type")
    if scaling_type not in (None, "", "none"):
        raise ValueError(
            "SmallThinker supports only unscaled/default RoPE, "
            f"got rope.scaling.type={scaling_type!r}"
        )
    rope_base = float(metadata[f"{arch}.rope.freq_base"])
    if not math.isfinite(rope_base) or rope_base <= 0:
        raise ValueError("smallthinker.rope.freq_base must be finite and positive")

    layers = config.num_hidden_layers
    raw_window = metadata.get(f"{arch}.attention.sliding_window")
    if isinstance(raw_window, (list, tuple, np.ndarray)):
        raise TypeError("smallthinker.attention.sliding_window must be a scalar")
    window_value = 0.0 if raw_window is None else float(raw_window)
    if not math.isfinite(window_value) or not window_value.is_integer() or window_value < 0:
        raise ValueError(
            "smallthinker.attention.sliding_window must be a non-negative integer"
        )
    window_enabled = window_value > 0
    if window_enabled:
        raw_period = metadata.get(f"{arch}.attention.sliding_window_pattern", 4)
        if isinstance(raw_period, (list, tuple, np.ndarray)):
            raise TypeError("smallthinker.attention.sliding_window_pattern must be a scalar")
        period_value = float(raw_period)
        if (
            not math.isfinite(period_value)
            or not period_value.is_integer()
            or period_value < 0
        ):
            raise ValueError(
                "smallthinker.attention.sliding_window_pattern must be a non-negative integer"
            )
        period = int(period_value)
        layer_types = [
            ("sliding_attention" if period == 0 or layer % period != 0 else "full_attention")
            for layer in range(layers)
        ]
        # The pinned loader forces 4096 regardless of the serialized positive value.
        sliding_window = 4096
        # With SWA enabled, llama.cpp disables RoPE on layers 0, 4, 8, ...
        use_rope_layers = [0 if layer % 4 == 0 else 1 for layer in range(layers)]
        local_rope_base = float(metadata.get(f"{arch}.rope.freq_base_swa", rope_base))
        if not math.isfinite(local_rope_base) or local_rope_base <= 0:
            raise ValueError("smallthinker.rope.freq_base_swa must be finite and positive")
    else:
        if f"{arch}.attention.sliding_window_pattern" in metadata:
            raise ValueError(
                "smallthinker.attention.sliding_window_pattern requires a positive "
                "attention.sliding_window"
            )
        if f"{arch}.rope.freq_base_swa" in metadata:
            raise ValueError(
                "smallthinker.rope.freq_base_swa requires a positive attention.sliding_window"
            )
        layer_types = ["full_attention"] * layers
        sliding_window = None
        use_rope_layers = [1] * layers
        local_rope_base = rope_base

    return dataclasses.replace(
        config,
        hidden_act="relu",
        tie_word_embeddings="output.weight" not in set(model.tensor_names),
        attn_qkv_bias=any(
            name.endswith(("attn_q.bias", "attn_qkv.bias")) for name in model.tensor_names
        ),
        attn_o_bias=False,
        mlp_bias=False,
        rope_type="default",
        rope_interleave=False,
        rope_theta=rope_base,
        rope_local_base_freq=local_rope_base,
        partial_rotary_factor=1.0,
        layer_types=layer_types,
        no_rope_layers=use_rope_layers,
        sliding_window=sliding_window,
        scoring_func="softmax" if gating == 1 else "sigmoid",
        norm_topk_prob=True,
        routed_scaling_factor=route_scale,
        routing_weight_normalization_floor=6.103515625e-5,
        n_group=1,
        topk_group=1,
        use_expert_bias=False,
        disable_qmoe=True,
    )


def _talkie_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Restore Talkie's causal scalar-gain and inverse-NeoX-RoPE profile."""
    arch = model.architecture
    return dataclasses.replace(
        config,
        hidden_act="silu",
        tie_word_embeddings=False,
        attn_qk_norm=False,
        logit_scale=float(metadata[f"{arch}.logit_scale"]),
        rope_type="default",
        rope_interleave=False,
    )


def _maincoder_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Restore Maincoder's tied, adjacent-pair, post-RoPE QK-norm profile."""
    _validate_closed_rope_scaling_metadata(metadata, "maincoder")
    return dataclasses.replace(
        config,
        hidden_act="silu",
        tie_word_embeddings=True,
        attn_qk_norm=True,
        attn_qk_norm_full=False,
        attn_qkv_bias=False,
        attn_o_bias=False,
        mlp_bias=False,
        rope_type="default",
        rope_interleave=True,
    )


def _hy_v3_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> HyV3Config:
    """Restore the exact llama.cpp HYV3 full-attention and MoE contract."""
    arch = "hy_v3"
    names = set(model.tensor_names)
    total_layers = int(metadata[f"{arch}.block_count"])
    trunk_layers = int(config.num_hidden_layers)
    if total_layers - trunk_layers not in (0, 1):
        raise ValueError(
            "hy_v3 supports a trunk-only file or exactly one appended NextN block"
        )

    gating = int(metadata.get(f"{arch}.expert_gating_func", 2))
    if gating != 2:
        raise ValueError(f"{arch}.expert_gating_func must be SIGMOID (2), got {gating}")

    block_kinds = [
        "sparse" if f"blk.{layer}.ffn_gate_inp.weight" in names else "dense"
        for layer in range(trunk_layers)
    ]
    dense_prefix = 0
    while dense_prefix < trunk_layers and block_kinds[dense_prefix] == "dense":
        dense_prefix += 1
    if any(kind != "sparse" for kind in block_kinds[dense_prefix:]):
        raise ValueError("hy_v3 dense FFN blocks must form one contiguous leading prefix")
    if dense_prefix == trunk_layers:
        raise ValueError("hy_v3 requires at least one routed expert block")

    num_experts = int(config.num_local_experts or 0)
    top_k = int(config.num_experts_per_tok or 0)
    moe_width = int(config.moe_intermediate_size or 0)
    if min(num_experts, top_k, moe_width) <= 0 or top_k > num_experts:
        raise ValueError("hy_v3 requires valid expert count, top-k, and expert FFN width")
    shared_width = int(config.shared_expert_intermediate_size or moe_width)
    if shared_width <= 0:
        raise ValueError("hy_v3 shared expert width must be positive")

    correction_biases = [
        f"blk.{layer}.exp_probs_b" in names for layer in range(dense_prefix, trunk_layers)
    ]
    if any(correction_biases) and not all(correction_biases):
        raise ValueError(
            "hy_v3 expert selection bias must be present in every routed trunk layer or absent"
        )

    has_mtp_tensors = (
        total_layers > trunk_layers and f"blk.{total_layers - 1}.nextn.eh_proj.weight" in names
    )
    serialized_layers = total_layers if has_mtp_tensors else trunk_layers
    qkv_biases = []
    for layer in range(serialized_layers):
        fused = f"blk.{layer}.attn_qkv.bias" in names
        split_count = sum(
            f"blk.{layer}.attn_{projection}.bias" in names for projection in ("q", "k", "v")
        )
        if split_count not in (0, 3) or (fused and split_count):
            raise ValueError(f"hy_v3 layer {layer} has a partial or ambiguous Q/K/V bias set")
        qkv_biases.append(fused or split_count == 3)
    if any(qkv_biases) and not all(qkv_biases):
        raise ValueError("hy_v3 Q/K/V bias presence must be uniform across all blocks")

    route_scale = float(metadata.get(f"{arch}.expert_weights_scale", 1.0))
    if math.isclose(route_scale, 0.0):
        route_scale = 1.0
    if not math.isfinite(route_scale) or route_scale <= 0:
        raise ValueError("hy_v3 expert_weights_scale must be finite and positive")

    fields = _shallow_fields(config)
    fields.update(
        model_type="hy_v3",
        hidden_act="silu",
        tie_word_embeddings="output.weight" not in names,
        attn_qkv_bias=all(qkv_biases),
        attn_o_bias=False,
        mlp_bias=False,
        attn_qk_norm=True,
        attn_qk_norm_full=False,
        rope_type="default",
        partial_rotary_factor=1.0,
        first_k_dense_replace=dense_prefix,
        n_shared_experts=max(shared_width // moe_width, 1),
        shared_expert_intermediate_size=shared_width,
        norm_topk_prob=bool(metadata.get(f"{arch}.expert_weights_norm")),
        routed_scaling_factor=route_scale,
        routing_weight_normalization_floor=6.103515625e-5,
        routing_weight_normalization_epsilon=None,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        use_expert_bias=all(correction_biases),
        disable_qmoe=True,
        enable_moe_fp32_combine=True,
    )
    return HyV3Config(**fields)


def _validate_conventional_moe_rope_scaling(metadata: dict[str, Any], arch: str) -> None:
    scaling_type = metadata.get(f"{arch}.rope.scaling.type")
    if scaling_type in (None, "", "none"):
        return
    if scaling_type != "yarn":
        raise ValueError(
            f"{arch}.rope.scaling.type={scaling_type!r} is unsupported; "
            "only unscaled and YaRN RoPE are exact for this architecture"
        )

    required = ("factor", "original_context_length")
    missing = [
        f"{arch}.rope.scaling.{suffix}"
        for suffix in required
        if f"{arch}.rope.scaling.{suffix}" not in metadata
    ]
    if missing:
        raise ValueError(f"{arch} YaRN scaling is missing required metadata: {missing}")

    expected_beta_fast = 8.0 if arch == "grok" else 32.0
    positive_values = {
        "factor": metadata[f"{arch}.rope.scaling.factor"],
        "original_context_length": metadata[f"{arch}.rope.scaling.original_context_length"],
        "yarn_beta_fast": metadata.get(
            f"{arch}.rope.scaling.yarn_beta_fast",
            expected_beta_fast,
        ),
        "yarn_beta_slow": metadata.get(f"{arch}.rope.scaling.yarn_beta_slow", 1.0),
        "attn_factor": metadata.get(f"{arch}.rope.scaling.attn_factor", 1.0),
    }
    if any(
        not math.isfinite(float(value)) or float(value) <= 0
        for value in positive_values.values()
    ):
        raise ValueError(f"{arch} YaRN scaling values must be finite and positive")
    if (
        not math.isclose(
            float(positive_values["yarn_beta_fast"]),
            expected_beta_fast,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(positive_values["yarn_beta_slow"]), 1.0, rel_tol=0.0, abs_tol=0.0
        )
        or not math.isclose(
            float(positive_values["attn_factor"]), 1.0, rel_tol=0.0, abs_tol=0.0
        )
    ):
        raise ValueError(
            f"{arch} YaRN metadata must retain the supported pinned loader defaults "
            f"yarn_beta_fast={expected_beta_fast:g}, yarn_beta_slow=1, and attn_factor=1"
        )
    for suffix in ("yarn_ext_factor", "yarn_attn_factor"):
        value = metadata.get(f"{arch}.rope.scaling.{suffix}")
        if value is not None and not math.isclose(
            float(value),
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"{arch}.rope.scaling.{suffix} must be absent or 1 for the "
                "supported YaRN graph"
            )


def _conventional_shared_moe_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Restore the exact BailingMoE/DeepSeek/Dots1 routed/shared MoE contract."""
    arch = model.architecture
    names = set(model.tensor_names)
    _validate_conventional_moe_rope_scaling(metadata, arch)

    dense_prefix = int(metadata.get(f"{arch}.leading_dense_block_count", 0))
    if arch == "bailingmoe":
        # The pinned graph is all-MoE and does not branch on this optional key.
        # Reject contradictory metadata instead of silently changing its meaning.
        if dense_prefix != 0:
            raise ValueError(
                "bailingmoe.leading_dense_block_count must be zero for the "
                "pinned all-layer MoE graph"
            )
    if not 0 <= dense_prefix <= config.num_hidden_layers:
        raise ValueError(
            f"{arch}.leading_dense_block_count must be in [0, "
            f"{config.num_hidden_layers}], got {dense_prefix}"
        )
    all_dense = dense_prefix == config.num_hidden_layers

    qkv_bias_layers = []
    for layer in range(config.num_hidden_layers):
        fused = f"blk.{layer}.attn_qkv.bias" in names
        split_count = sum(
            f"blk.{layer}.attn_{projection}.bias" in names for projection in ("q", "k", "v")
        )
        if split_count not in (0, 3) or (fused and split_count):
            raise ValueError(f"{arch} layer {layer} has a partial or ambiguous Q/K/V bias set")
        qkv_bias_layers.append(fused or split_count == 3)
    if any(qkv_bias_layers) and not all(qkv_bias_layers):
        raise ValueError(f"{arch} Q/K/V bias presence must be uniform across all layers")

    common_updates: dict[str, Any] = {
        "rope_type": config.rope_type or "default",
        "hidden_act": "silu",
        "tie_word_embeddings": arch == "deepseek" and "output.weight" not in names,
        "attn_qkv_bias": all(qkv_bias_layers),
        "attn_o_bias": False,
        "mlp_bias": False,
        "first_k_dense_replace": dense_prefix,
        "attn_qk_norm": arch == "dots1",
        "attn_qk_norm_full": False,
    }
    if all_dense:
        return dataclasses.replace(
            config,
            **common_updates,
            num_local_experts=None,
            num_experts_per_tok=None,
            moe_intermediate_size=None,
            n_shared_experts=None,
            shared_expert_intermediate_size=None,
            use_expert_bias=False,
            disable_qmoe=True,
        )

    routed_suffixes = [
        "expert_count",
        "expert_used_count",
        "expert_feed_forward_length",
        "expert_shared_count",
    ]
    if arch == "dots1":
        routed_suffixes.append("expert_gating_func")
    missing_routed_metadata = [
        f"{arch}.{suffix}" for suffix in routed_suffixes if f"{arch}.{suffix}" not in metadata
    ]
    if missing_routed_metadata:
        raise ValueError(
            f"{arch} routed layers require MoE metadata: {missing_routed_metadata}"
        )

    config = _moe_postprocess(config, metadata, model)
    assert config.num_local_experts is not None
    assert config.num_experts_per_tok is not None
    assert config.moe_intermediate_size is not None

    n_shared = int(metadata[f"{arch}.expert_shared_count"])
    if n_shared <= 0:
        raise ValueError(f"{arch}.expert_shared_count must be greater than zero")
    shared_width = config.moe_intermediate_size * n_shared
    serialized_shared_width = metadata.get(f"{arch}.expert_shared_feed_forward_length")
    if serialized_shared_width is not None and int(serialized_shared_width) != shared_width:
        raise ValueError(
            f"{arch}.expert_shared_feed_forward_length must equal "
            f"expert_feed_forward_length * expert_shared_count ({shared_width}), "
            f"got {serialized_shared_width}"
        )

    gating = int(metadata.get(f"{arch}.expert_gating_func", 1))
    if gating not in (1, 2):
        raise ValueError(
            f"{arch}.expert_gating_func must be SOFTMAX (1) or SIGMOID (2), got {gating}"
        )
    if arch in {"bailingmoe", "deepseek"} and gating != 1:
        raise ValueError(f"{arch}.expert_gating_func must be SOFTMAX (1), got {gating}")

    all_bias_layers = {
        layer
        for layer in range(config.num_hidden_layers)
        if f"blk.{layer}.exp_probs_b.bias" in names
    }
    if arch != "dots1" and all_bias_layers:
        raise ValueError(f"{arch} does not consume exp_probs_b correction-bias tensors")
    dense_bias_layers = all_bias_layers & set(range(dense_prefix))
    if dense_bias_layers:
        raise ValueError(
            f"{arch} correction bias is invalid on dense layers {sorted(dense_bias_layers)}"
        )
    bias_layers = {
        layer
        for layer in range(dense_prefix, config.num_hidden_layers)
        if layer in all_bias_layers
    }
    routed_layers = set(range(dense_prefix, config.num_hidden_layers))
    if bias_layers and bias_layers != routed_layers:
        raise ValueError(
            f"{arch} correction bias must be present for every routed layer or none; "
            f"found layers {sorted(bias_layers)}, expected {sorted(routed_layers)}"
        )
    use_expert_bias = bool(bias_layers)

    norm_topk_prob = (
        False if arch == "deepseek" else bool(metadata.get(f"{arch}.expert_weights_norm"))
    )
    if arch == "deepseek" and metadata.get(f"{arch}.expert_weights_norm"):
        raise ValueError("deepseek.expert_weights_norm=True is not used by the pinned graph")
    route_scale = float(metadata.get(f"{arch}.expert_weights_scale", 1.0))
    # llama.cpp uses zero as the serialized default sentinel.
    if math.isclose(route_scale, 0.0):
        route_scale = 1.0
    if not math.isfinite(route_scale) or route_scale <= 0:
        raise ValueError(
            f"{arch}.expert_weights_scale must resolve to a finite positive value, "
            f"got {route_scale!r}"
        )

    return dataclasses.replace(
        config,
        **common_updates,
        n_shared_experts=n_shared,
        shared_expert_intermediate_size=shared_width,
        scoring_func="softmax" if gating == 1 else "sigmoid",
        topk_method="greedy",
        n_group=1,
        topk_group=1,
        routing_weight_normalization_floor=(6.103515625e-5 if arch == "dots1" else None),
        use_expert_bias=use_expert_bias,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=route_scale,
        # QMoE's CUDA ABI cannot preserve sigmoid/correction-bias aggregation.
        # Keep quantized Dots1 on the gate-agnostic dense/block fallback.
        disable_qmoe=arch == "dots1",
    )


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


def _granite_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Select the exact dense or MoE Granite graph and restore GGUF scaling."""
    arch = "granite"
    _validate_closed_rope_scaling_metadata(
        metadata, arch, allowed_suffixes={"type", "finetuned"}
    )
    num_experts = int(config.num_local_experts or 0)
    top_k = int(config.num_experts_per_tok or 0)
    if num_experts < 0 or top_k < 0 or bool(num_experts) != bool(top_k):
        raise ValueError(
            "granite expert_count and expert_used_count must both be zero or both positive"
        )
    if num_experts and top_k > num_experts:
        raise ValueError(
            f"granite.expert_used_count must be in [1, {num_experts}], got {top_k}"
        )

    shared_width = int(metadata.get(f"{arch}.expert_shared_feed_forward_length", 0))
    if shared_width < 0:
        raise ValueError("granite.expert_shared_feed_forward_length must not be negative")
    if not num_experts and shared_width:
        raise ValueError("granite.expert_shared_feed_forward_length requires routed experts")

    deepstack = metadata.get(f"{arch}.deepstack_mapping")
    if deepstack is not None:
        if not isinstance(deepstack, (list, tuple, np.ndarray)):
            raise ValueError("granite.deepstack_mapping must be an integer array")
        mapping = [int(value) for value in deepstack]
        if mapping and len(mapping) != config.num_hidden_layers:
            raise ValueError("granite.deepstack_mapping must match block_count")
        if any(value != -1 for value in mapping):
            raise NotImplementedError(
                "granite deep-stack embedding injection is unsupported by the text task; "
                "only an absent, empty, or all -1 mapping is accepted"
            )

    finetuned = metadata.get(f"{arch}.rope.scaling.finetuned", True)
    if not isinstance(finetuned, (bool, np.bool_)):
        raise TypeError("granite.rope.scaling.finetuned must be boolean")
    tensor_names = set(getattr(model, "tensor_names", ()) or ())
    if "rope_freqs.weight" in tensor_names:
        raise ValueError(
            "granite serialized rope_freqs.weight is unsupported because the exact "
            "per-dimension frequency factors are not representable by the current rotary graph"
        )
    longrope_names = {"rope_factors_long.weight", "rope_factors_short.weight"}
    present_longrope = tensor_names & longrope_names
    scaling_type = metadata.get(f"{arch}.rope.scaling.type")
    if scaling_type not in (None, "", "none"):
        raise ValueError(
            f"granite rope.scaling.type={scaling_type!r} is not in the exact supported subset"
        )
    if present_longrope:
        raise ValueError(
            "granite tensor-backed LongRoPE is unsupported because the current rotary graph "
            "cannot preserve rope.scaling.attn_factor exactly"
        )

    def finite_scale(suffix: str, default: float) -> float:
        value = float(metadata.get(f"{arch}.{suffix}", default))
        if not math.isfinite(value):
            raise ValueError(f"granite.{suffix} must be finite")
        return value

    logit_scale = finite_scale("logit_scale", 0.0)
    if math.isclose(logit_scale, 0.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("granite.logit_scale must be nonzero")
    embedding_scale = finite_scale("embedding_scale", 0.0) or 1.0
    residual_scale = finite_scale("residual_scale", 0.0) or 1.0
    attention_scale = finite_scale("attention.scale", 0.0) or None
    raw_expert_width = metadata.get(f"{arch}.expert_feed_forward_length")
    if raw_expert_width is not None:
        if isinstance(raw_expert_width, (bool, np.bool_)):
            serialized_expert_width = None
        elif isinstance(raw_expert_width, (int, np.integer)):
            serialized_expert_width = int(raw_expert_width)
        elif isinstance(raw_expert_width, (float, np.floating)):
            numeric_expert_width = float(raw_expert_width)
            serialized_expert_width = (
                int(numeric_expert_width)
                if math.isfinite(numeric_expert_width) and numeric_expert_width.is_integer()
                else None
            )
        else:
            serialized_expert_width = None
        if serialized_expert_width != config.intermediate_size:
            raise ValueError(
                "granite.expert_feed_forward_length must equal feed_forward_length "
                "because the pinned loader sizes routed experts from feed_forward_length"
            )
    expert_width = config.intermediate_size

    return dataclasses.replace(
        config,
        model_type="granitemoe" if num_experts else "granite",
        hidden_act="silu",
        num_local_experts=num_experts or None,
        num_experts_per_tok=top_k or None,
        moe_intermediate_size=expert_width if num_experts else None,
        shared_expert_intermediate_size=shared_width or None,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        embedding_multiplier=embedding_scale,
        residual_multiplier=residual_scale,
        attention_multiplier=attention_scale,
        logits_scaling=logit_scale,
        rope_type=config.rope_type if finetuned else None,
        rope_theta=config.rope_theta if finetuned else None,
        rope_scaling=config.rope_scaling if finetuned else None,
        partial_rotary_factor=config.partial_rotary_factor if finetuned else None,
    )


def _jamba_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> JambaConfig:
    """Build exact Jamba mixer and routed-FFN schedules from serialized tensors."""
    inner_size = int(metadata["jamba.ssm.inner_size"])
    if inner_size != 2 * config.hidden_size:
        raise ValueError(
            "jamba.ssm.inner_size must equal 2 * embedding_length for the pinned loader"
        )
    if config.hidden_act not in {"silu", "swish"}:
        raise ValueError("Jamba GGUF requires the pinned SiLU feed-forward activation")
    num_experts = int(metadata.get("jamba.expert_count", 0))
    top_k = int(metadata.get("jamba.expert_used_count", 0))
    expert_layers = sorted(
        {
            int(match.group(1))
            for name in model.tensor_names
            if (match := re.fullmatch(r"blk\.(\d+)\.ffn_gate_inp\.weight", name))
        }
    )
    if num_experts:
        if num_experts == 1:
            raise ValueError(
                "Jamba expert_count=1 is not a routed-MoE layout; use dense FFN tensors"
            )
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                f"jamba.expert_used_count must be in [1, {num_experts}], got {top_k}"
            )
        if not expert_layers:
            raise ValueError("Jamba expert metadata requires at least one routed MoE layer")
    elif top_k or expert_layers:
        raise ValueError(
            "Jamba routed tensors require positive expert_count and expert_used_count"
        )
    output_present = "output.weight" in set(model.tensor_names)
    fields = _shallow_fields(config)
    fields.update(
        num_local_experts=num_experts or None,
        num_experts_per_tok=top_k or None,
        moe_intermediate_size=config.intermediate_size if num_experts else None,
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
        tie_word_embeddings=not output_present,
        rope_type=None,
    )
    return JambaConfig(
        **fields,
        mamba_d_state=int(metadata["jamba.ssm.state_size"]),
        mamba_d_conv=int(metadata["jamba.ssm.conv_kernel"]),
        mamba_expand=2,
        mamba_dt_rank=int(metadata["jamba.ssm.time_step_rank"]),
        mamba_conv_bias=any(".ssm_conv1d.bias" in name for name in model.tensor_names),
        mamba_proj_bias=False,
        expert_layer_indices=expert_layers,
    )


def _nemotron_h_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> NemotronHConfig:
    """Build the exact dense or routed-MoE Nemotron-H backbone."""
    arch = model.architecture
    if arch not in {"nemotron_h", "nemotron_h_moe"}:
        raise ValueError(f"Unexpected GGUF architecture for Nemotron-H: {arch!r}")
    inner_size = int(metadata[f"{arch}.ssm.inner_size"])
    num_heads = int(metadata[f"{arch}.ssm.time_step_rank"])
    groups = int(metadata[f"{arch}.ssm.group_count"])
    if min(inner_size, num_heads) <= 0 or inner_size % num_heads:
        raise ValueError(
            f"{arch}.ssm.inner_size must be positive and divisible by ssm.time_step_rank"
        )
    if groups <= 0 or num_heads % groups:
        raise ValueError(f"{arch}.ssm.time_step_rank must be divisible by ssm.group_count")

    num_experts = int(metadata.get(f"{arch}.expert_count", 0))
    top_k = int(metadata.get(f"{arch}.expert_used_count", 0))
    layer_types = list(config.layer_types or ())
    routed_layers = [i for i, layer_type in enumerate(layer_types) if layer_type == "moe"]
    if arch == "nemotron_h_moe":
        if num_experts <= 1 or not 1 <= top_k <= num_experts:
            raise ValueError(
                f"{arch} requires expert_count > 1 and expert_used_count in "
                f"[1, expert_count], got {num_experts} and {top_k}"
            )
        if not routed_layers:
            raise ValueError(f"{arch} metadata declares experts but has no routed MoE layer")
        if config.moe_intermediate_size is None or config.moe_intermediate_size <= 0:
            raise ValueError(f"{arch}.expert_feed_forward_length must be greater than zero")
        if (
            config.shared_expert_intermediate_size is None
            or config.shared_expert_intermediate_size <= 0
        ):
            raise ValueError(
                f"{arch}.expert_shared_feed_forward_length must be greater than zero"
            )
        shared_count = int(metadata.get(f"{arch}.expert_shared_count", 1))
        if shared_count != 1:
            raise ValueError(
                f"{arch}.expert_shared_count must be exactly 1, got {shared_count}"
            )
        n_group = int(metadata.get(f"{arch}.expert_group_count", 1))
        topk_group = int(metadata.get(f"{arch}.expert_group_used_count", 1))
        if (n_group, topk_group) != (1, 1):
            raise ValueError(
                f"{arch} grouped expert routing is unsupported; "
                f"expert_group_count and expert_group_used_count must both be 1, "
                f"got {n_group} and {topk_group}"
            )
        latent_size = metadata.get(f"{arch}.moe_latent_size")
        if latent_size is not None and int(latent_size) <= 0:
            raise ValueError(f"{arch}.moe_latent_size must be greater than zero when present")
    elif num_experts or top_k or routed_layers:
        raise ValueError(
            "nemotron_h uses the dense architecture contract; routed experts require "
            "general.architecture='nemotron_h_moe'"
        )

    fields = _shallow_fields(config)
    fields.update(
        hidden_act="relu2",
        layer_types=layer_types,
        num_local_experts=num_experts or None,
        num_experts_per_tok=top_k or None,
        norm_topk_prob=bool(metadata.get(f"{arch}.expert_weights_norm", True)),
        routed_scaling_factor=float(metadata.get(f"{arch}.expert_weights_scale", 1.0)),
        n_group=1,
        topk_group=1,
    )
    return NemotronHConfig(
        **fields,
        mamba_n_heads=num_heads,
        mamba_d_head=inner_size // num_heads,
        mamba_d_state=int(metadata[f"{arch}.ssm.state_size"]),
        mamba_n_groups=groups,
        mamba_d_conv=int(metadata[f"{arch}.ssm.conv_kernel"]),
        mamba_expand=inner_size // config.hidden_size,
        mamba_conv_bias=any(".ssm_conv1d.bias" in name for name in model.tensor_names),
        mamba_proj_bias=False,
        moe_latent_size=(
            int(metadata[f"{arch}.moe_latent_size"])
            if f"{arch}.moe_latent_size" in metadata
            else None
        ),
    )


def _granitehybrid_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> GraniteMoeHybridConfig:
    """Build GraniteHybrid with its exact shared-MLP and routed-expert geometry."""
    arch = "granitehybrid"
    inner_size = int(metadata["granitehybrid.ssm.inner_size"])
    num_heads = int(metadata["granitehybrid.ssm.time_step_rank"])
    groups = int(metadata["granitehybrid.ssm.group_count"])
    if inner_size != 2 * config.hidden_size:
        raise ValueError("granitehybrid.ssm.inner_size must equal 2 * embedding_length")
    if min(num_heads, groups) <= 0 or inner_size % num_heads or num_heads % groups:
        raise ValueError("GraniteHybrid Mamba2 head and group dimensions are inconsistent")

    num_experts = int(metadata.get(f"{arch}.expert_count", 0))
    top_k = int(metadata.get(f"{arch}.expert_used_count", 0))
    if bool(num_experts) != bool(top_k):
        raise ValueError(
            "GraniteHybrid expert_count and expert_used_count must both be zero "
            "or both positive"
        )
    expert_width = config.intermediate_size
    if expert_width <= 0:
        raise ValueError("granitehybrid.feed_forward_length must be greater than zero")
    if num_experts:
        if num_experts <= 1:
            raise ValueError(
                "GraniteHybrid expert_count=1 is not a routed-MoE layout; "
                "use the dense shared-MLP tensors"
            )
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                f"GraniteHybrid expert_used_count must be in [1, {num_experts}], got {top_k}"
            )
        shared_width = int(metadata.get(f"{arch}.expert_shared_feed_forward_length", 0))
        if shared_width < 0:
            raise ValueError(
                "granitehybrid.expert_shared_feed_forward_length must be non-negative"
            )
        if not bool(metadata.get(f"{arch}.expert_weights_norm", True)):
            raise ValueError("GraniteHybrid requires normalized top-k routing weights")
        routed_scale = float(metadata.get(f"{arch}.expert_weights_scale", 1.0))
        if not math.isclose(routed_scale, 1.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                "GraniteHybrid does not define an additional routed-expert output scale; "
                f"got expert_weights_scale={routed_scale}"
            )
    else:
        shared_width = config.intermediate_size
        if f"{arch}.expert_shared_feed_forward_length" in metadata:
            raise ValueError(
                "granitehybrid.expert_shared_feed_forward_length requires routed experts"
            )

    if config.hidden_act not in (None, "silu"):
        raise ValueError(
            f"granitehybrid.hidden_activation must be silu, got {config.hidden_act!r}"
        )
    logit_scale = float(metadata.get(f"{arch}.logit_scale", 1.0))
    if math.isclose(logit_scale, 0.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("granitehybrid.logit_scale must be nonzero")
    tensor_names = set(model.tensor_names)
    has_attention_bias = any(".attn_q.bias" in name for name in tensor_names)
    has_attention_output_bias = any(".attn_output.bias" in name for name in tensor_names)
    has_mlp_bias = any(
        name.endswith((".ffn_gate.bias", ".ffn_up.bias", ".ffn_down.bias"))
        for name in tensor_names
    )
    fields = _shallow_fields(config)
    fields.update(
        hidden_act="silu",
        intermediate_size=int(expert_width),
        num_local_experts=num_experts or None,
        num_experts_per_tok=top_k or None,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        attn_qkv_bias=has_attention_bias,
        attn_o_bias=has_attention_output_bias,
        mlp_bias=has_mlp_bias,
        rope_type=(
            fields["rope_type"]
            if bool(metadata.get("granitehybrid.rope.scaling.finetuned", True))
            else None
        ),
        embedding_multiplier=float(metadata.get("granitehybrid.embedding_scale", 1.0)),
        residual_multiplier=float(metadata.get("granitehybrid.residual_scale", 1.0)),
        attention_multiplier=(
            float(metadata["granitehybrid.attention.scale"]) or None
            if "granitehybrid.attention.scale" in metadata
            else None
        ),
        logits_scaling=logit_scale,
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
        shared_intermediate_size=int(shared_width),
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


def _pangu_embedded_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> ArchitectureConfig:
    """Restore the exact ordinary-RoPE Pangu-Embedded decoder contract."""
    arch = "pangu-embedded"
    tensor_names = set(getattr(model, "tensor_names", ()) or ())
    factor_tensors = tensor_names & {
        "rope_freqs.weight",
        "rope_factors_long.weight",
        "rope_factors_short.weight",
    }
    scaling_type = metadata.get(f"{arch}.rope.scaling.type")
    if factor_tensors or scaling_type not in (None, "", "none"):
        raise ValueError(
            "pangu-embedded tensor-backed or scaled RoPE is not supported; "
            "ordinary full-head RoPE is required"
        )

    head_dim = config.hidden_size // config.num_attention_heads
    for suffix in ("attention.key_length", "attention.value_length", "rope.dimension_count"):
        value = metadata.get(f"{arch}.{suffix}")
        if value is not None and int(value) != head_dim:
            raise ValueError(f"{arch}.{suffix} must equal head_dim ({head_dim}), got {value}")
    if not config.attn_o_bias:
        raise ValueError("pangu-embedded requires attn_output.bias in every layer")
    if config.mlp_bias:
        raise ValueError("pangu-embedded does not support FFN projection biases")
    return dataclasses.replace(
        config,
        hidden_act="silu",
        rope_type="default",
        rope_scaling=None,
        partial_rotary_factor=1.0,
        attn_qk_norm=False,
    )


def _minicpm_longrope(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
    *,
    rope_dim: int,
    label: str = "MiniCPM",
) -> ArchitectureConfig:
    tensor_names = set(getattr(model, "tensor_names", ()) or ())
    long_name = "rope_factors_long.weight"
    short_name = "rope_factors_short.weight"
    present = {name for name in (long_name, short_name) if name in tensor_names}
    if not present:
        return config
    if present != {long_name, short_name}:
        raise ValueError(
            f"{label} LongRoPE requires both rope_factors_long.weight and "
            "rope_factors_short.weight"
        )
    if model is None or not hasattr(model, "get_tensor"):
        raise ValueError(f"{label} LongRoPE factors could not be read from the GGUF model")

    raw_types = {
        name: getattr(qtype, "value", qtype)
        for name, _raw, qtype, _shape in model.tensor_items_raw()
        if name in present
    }
    if any(type_id not in {0, 1, 30} for type_id in raw_types.values()):
        raise ValueError(f"{label} LongRoPE factors must use F32/F16/BF16 storage")

    long_factor = np.asarray(model.get_tensor(long_name), dtype=np.float32).reshape(-1)
    short_factor = np.asarray(model.get_tensor(short_name), dtype=np.float32).reshape(-1)
    expected = rope_dim // 2
    if (
        rope_dim <= 0
        or rope_dim % 2
        or long_factor.shape != (expected,)
        or short_factor.shape != (expected,)
        or not np.all(np.isfinite(long_factor))
        or not np.all(np.isfinite(short_factor))
        or np.any(long_factor <= 0)
        or np.any(short_factor <= 0)
    ):
        raise ValueError(
            f"{label} LongRoPE factors must be finite positive vectors of length {expected}"
        )

    arch = model.architecture
    original_context = metadata.get(f"{arch}.rope.scaling.original_context_length")
    if original_context is None:
        # MiniCPM3's pinned converter omits scaling metadata. Its published
        # checkpoint has original_context == context and identical factor tables.
        if not np.array_equal(long_factor, short_factor):
            raise ValueError(
                f"{label} LongRoPE with distinct long/short factors requires "
                "rope.scaling.original_context_length"
            )
        original_context = config.max_position_embeddings
    original_context = int(original_context)
    if not 0 < original_context <= config.max_position_embeddings:
        raise ValueError(f"{label} LongRoPE original context is outside the model context")

    return dataclasses.replace(
        config,
        rope_type="longrope",
        rope_scaling={
            "long_factor": long_factor.tolist(),
            "short_factor": short_factor.tolist(),
        },
        original_max_position_embeddings=original_context,
    )


def _minicpm_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    arch = model.architecture
    scales = {
        "embedding_multiplier": float(metadata[f"{arch}.embedding_scale"]),
        "residual_multiplier": float(metadata[f"{arch}.residual_scale"]),
        "logits_scaling": float(metadata[f"{arch}.logit_scale"]),
    }
    if any(not math.isfinite(value) or value <= 0 for value in scales.values()):
        raise ValueError(
            "MiniCPM embedding, residual, and logit scales must be finite positive"
        )
    if int(metadata.get(f"{arch}.expert_count", 0)):
        raise ValueError("MiniCPM routed-expert GGUF is outside the exact dense graph subset")
    config = dataclasses.replace(
        config,
        rope_type=config.rope_type or "default",
        embedding_multiplier=scales["embedding_multiplier"],
        residual_multiplier=scales["residual_multiplier"],
        logits_scaling=scales["logits_scaling"],
    )
    rope_dim = int(
        metadata.get(
            f"{arch}.rope.dimension_count",
            config.head_dim,
        )
    )
    return _minicpm_longrope(
        config,
        metadata,
        model,
        rope_dim=rope_dim,
    )


def _minicpm3_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    arch = model.architecture
    heads = config.num_attention_heads
    qk_dim = int(metadata[f"{arch}.attention.key_length"])
    rope_dim = int(metadata[f"{arch}.rope.dimension_count"])
    value_dim = config.hidden_size // heads
    q_rank = int(metadata[f"{arch}.attention.q_lora_rank"])
    kv_rank = int(metadata[f"{arch}.attention.kv_lora_rank"])
    if config.hidden_size % heads or not 0 < rope_dim < qk_dim:
        raise ValueError("MiniCPM3 has invalid MLA head geometry")
    if min(q_rank, kv_rank, value_dim) <= 0:
        raise ValueError("MiniCPM3 LoRA ranks and value head width must be positive")
    if int(metadata.get(f"{arch}.expert_count", 0)):
        raise ValueError("MiniCPM3 pinned GGUF graph is dense-only")

    config = dataclasses.replace(
        config,
        head_dim=qk_dim,
        num_key_value_heads=heads,
        q_lora_rank=q_rank,
        kv_lora_rank=kv_rank,
        qk_nope_head_dim=qk_dim - rope_dim,
        qk_rope_head_dim=rope_dim,
        v_head_dim=value_dim,
        partial_rotary_factor=1.0,
        rope_type=config.rope_type or "default",
        rope_interleave=False,
        embedding_multiplier=12.0,
        residual_multiplier=1.4 / math.sqrt(config.num_hidden_layers),
        logits_scaling=config.hidden_size / 256.0,
        hidden_act="silu",
    )
    return _minicpm_longrope(config, metadata, model, rope_dim=rope_dim)


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


def _falcon_h1_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> FalconH1Config:
    """Build exact Falcon-H1 attention and Mamba2 geometry from pinned metadata."""
    arch = "falcon-h1"
    d_inner = int(metadata[f"{arch}.ssm.inner_size"])
    num_heads = int(metadata[f"{arch}.ssm.time_step_rank"])
    n_groups = int(metadata[f"{arch}.ssm.group_count"])
    key_dim = int(metadata[f"{arch}.attention.key_length"])
    value_dim = int(metadata[f"{arch}.attention.value_length"])
    if key_dim <= 0 or value_dim != key_dim:
        raise ValueError(
            "falcon-h1 requires equal positive attention key/value head dimensions"
        )
    if config.num_attention_heads * key_dim != config.hidden_size:
        raise ValueError(
            "falcon-h1 attention.head_count * attention.key_length must equal embedding_length"
        )
    if num_heads <= 0 or d_inner <= 0 or d_inner % num_heads:
        raise ValueError(
            "falcon-h1 ssm.time_step_rank must divide the positive ssm.inner_size"
        )
    if n_groups <= 0 or num_heads % n_groups or d_inner % n_groups:
        raise ValueError(
            "falcon-h1 ssm.group_count must divide both the SSM head count and inner size"
        )

    names = set(model.tensor_names) if model is not None else set()
    layers = config.num_hidden_layers
    conv_biases = {f"blk.{layer}.ssm_conv1d.bias" for layer in range(layers)}
    ssm_norms = {f"blk.{layer}.ssm_norm.weight" for layer in range(layers)}
    fields = _base_model_fields(config, FalconH1Config)
    fields.update(
        hidden_act="silu",
        head_dim=key_dim,
        mamba_d_ssm=d_inner,
        mamba_n_heads=num_heads,
        mamba_d_head=d_inner // num_heads,
        mamba_n_groups=n_groups,
        mamba_d_state=int(metadata[f"{arch}.ssm.state_size"]),
        mamba_d_conv=int(metadata[f"{arch}.ssm.conv_kernel"]),
        mamba_expand=2,
        mamba_chunk_size=256,
        mamba_conv_bias=conv_biases.issubset(names),
        mamba_proj_bias=False,
        mamba_norm_before_gate=True,
        mamba_rms_norm=ssm_norms.issubset(names),
        attention_bias=config.attn_qkv_bias,
        projectors_bias=False,
        # The pinned converter folds every Falcon-H1 multiplier into its tensor.
        embedding_multiplier=1.0,
        lm_head_multiplier=1.0,
        mlp_multipliers=(1.0, 1.0),
        key_multiplier=1.0,
        attention_in_multiplier=1.0,
        attention_out_multiplier=1.0,
        ssm_multipliers=(1.0, 1.0, 1.0, 1.0, 1.0),
        ssm_in_multiplier=1.0,
        ssm_out_multiplier=1.0,
    )
    result = FalconH1Config(**fields)
    result.model_type = "falcon_h1"
    return result


def _infer_plamo2_attention_widths(
    model: Any,
    head_counts: tuple[int, ...],
    kv_head_counts: tuple[int, ...],
) -> tuple[int, int]:
    """Infer PLaMo2 key/value widths from every attention layer's tensors."""
    inferred: set[tuple[int, int]] = set()
    tensor_names = set(model.tensor_names)
    for layer, (q_heads, kv_heads) in enumerate(zip(head_counts, kv_head_counts)):
        if kv_heads == 0:
            continue
        qkv_name = f"blk.{layer}.attn_qkv.weight"
        output_name = f"blk.{layer}.attn_output.weight"
        if qkv_name not in tensor_names or output_name not in tensor_names:
            raise ValueError(
                f"PLaMo2 layer {layer} is missing attention tensors required "
                "to infer key/value widths"
            )
        qkv_shape = model.get_tensor_shape(qkv_name)
        output_shape = model.get_tensor_shape(output_name)
        value_width, value_remainder = divmod(int(output_shape[1]), q_heads)
        key_width, key_remainder = divmod(
            int(qkv_shape[0]) - kv_heads * value_width,
            q_heads + kv_heads,
        )
        if value_remainder or key_remainder or min(key_width, value_width) <= 0:
            raise ValueError(
                f"PLaMo2 layer {layer} key/value widths cannot be inferred exactly"
            )
        inferred.add((key_width, value_width))
    if len(inferred) != 1:
        raise ValueError("PLaMo2 attention tensor widths are missing or contradictory")
    return inferred.pop()


def _plamo2_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any = None,
) -> Plamo2Config:
    """Build PLaMo2 from its exact serialized schedule and tensor geometry."""
    arch = "plamo2"
    layers = config.num_hidden_layers
    head_counts = tuple(int(value) for value in metadata[f"{arch}.attention.head_count"])
    kv_head_counts = tuple(int(value) for value in metadata[f"{arch}.attention.head_count_kv"])
    if len(head_counts) != layers or len(kv_head_counts) != layers:
        raise ValueError("PLaMo2 attention head arrays must match block_count")

    inner = int(metadata[f"{arch}.ssm.inner_size"])
    ssm_heads = int(metadata[f"{arch}.ssm.time_step_rank"])
    if ssm_heads <= 0 or inner % ssm_heads:
        raise ValueError("PLaMo2 SSM head count must divide ssm.inner_size")
    if int(metadata[f"{arch}.ssm.group_count"]) != 0:
        raise ValueError("PLaMo2 supports only ssm.group_count=0")

    key_length = metadata.get(f"{arch}.attention.key_length")
    value_length = metadata.get(f"{arch}.attention.value_length")
    if model is not None:
        inferred_key, inferred_value = _infer_plamo2_attention_widths(
            model,
            head_counts,
            kv_head_counts,
        )
        if key_length is not None and int(key_length) != inferred_key:
            raise ValueError("PLaMo2 attention.key_length contradicts tensor shapes")
        if value_length is not None and int(value_length) != inferred_value:
            raise ValueError("PLaMo2 attention.value_length contradicts tensor shapes")
        key_length = inferred_key
        value_length = inferred_value
    if key_length is None or value_length is None:
        raise ValueError("PLaMo2 key/value widths require exact tensor-shape evidence")
    key_length = int(key_length)
    value_length = int(value_length)
    if (
        key_length != value_length
        or key_length * config.num_attention_heads != config.hidden_size
    ):
        raise ValueError(
            "PLaMo2 requires equal key/value widths and head_count * width == embedding_length"
        )
    if inner != ssm_heads * key_length:
        raise ValueError(
            "PLaMo2 ssm.inner_size must equal ssm.time_step_rank * attention key width"
        )

    fields = _base_model_fields(config, Plamo2Config)
    fields.pop("layer_types", None)
    fields.update(
        hidden_act="silu",
        head_dim=key_length,
        # The released PLaMo2 converter wrote 1e6 here even though the pinned
        # reference architecture uses its 1e4 local-RoPE default for every
        # attention layer. Preserve other explicit bases for future variants.
        rope_theta=(
            10_000.0
            if float(metadata[f"{arch}.rope.freq_base"]) == 1_000_000.0  # noqa: RUF069
            else config.rope_theta
        ),
        attention_head_counts=head_counts,
        attention_kv_head_counts=kv_head_counts,
        mamba_num_heads=ssm_heads,
        mamba_d_state=int(metadata[f"{arch}.ssm.state_size"]),
        mamba_d_conv=int(metadata[f"{arch}.ssm.conv_kernel"]),
        mamba_dt_rank=max(64, config.hidden_size // 16),
        mamba_group_count=0,
        attention_window_size=min(config.max_position_embeddings, 2048),
        use_predefined_initial_state=False,
        attn_qkv_bias=False,
        attn_o_bias=False,
        mlp_bias=False,
    )
    result = Plamo2Config(**fields)
    result.model_type = "plamo2"
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


def _minimax_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> MiniMaxConfig:
    """Restore the exact pinned MiniMax-01 GGUF execution contract."""
    arch = model.architecture
    head_dim = int(metadata[f"{arch}.attention.key_length"])
    value_dim = int(metadata[f"{arch}.attention.value_length"])
    rope_dim = int(metadata[f"{arch}.rope.dimension_count"])
    residual_scale = float(metadata[f"{arch}.residual_scale"])
    experts = int(metadata[f"{arch}.expert_count"])
    top_k = int(metadata[f"{arch}.expert_used_count"])
    if head_dim <= 0 or value_dim != head_dim:
        raise ValueError(
            f"MiniMax-01 requires equal positive key/value lengths, got {head_dim}/{value_dim}"
        )
    if rope_dim <= 0 or rope_dim > head_dim or rope_dim % 2:
        raise ValueError(
            f"MiniMax-01 rope.dimension_count must be positive, even, and <= {head_dim}"
        )
    if not math.isfinite(residual_scale) or residual_scale <= 0:
        raise ValueError("MiniMax-01 residual_scale must be finite and positive")
    if experts <= 1 or not 1 <= top_k <= experts:
        raise ValueError(
            f"MiniMax-01 expert counts are invalid: expert_count={experts}, "
            f"expert_used_count={top_k}"
        )
    if any(
        key in metadata
        for key in (
            f"{arch}.expert_shared_count",
            f"{arch}.expert_shared_feed_forward_length",
        )
    ):
        raise ValueError("MiniMax-01 pinned GGUF does not support shared experts")

    _, layer_types, _ = _derive_hybrid_layout(arch, metadata, model.tensor_names)
    assert layer_types is not None
    fields = _shallow_fields(config)
    fields.update(
        model_type="minimax",
        head_dim=head_dim,
        partial_rotary_factor=rope_dim / head_dim,
        layer_types=layer_types,
        hidden_act="silu",
        norm_topk_prob=True,
        disable_qmoe=True,
        lightning_norm_eps=config.rms_norm_eps,
        full_attn_alpha_factor=residual_scale,
        full_attn_beta_factor=1.0,
        linear_attn_alpha_factor=residual_scale,
        linear_attn_beta_factor=1.0,
        mlp_alpha_factor=residual_scale,
        mlp_beta_factor=1.0,
    )
    return MiniMaxConfig(**fields)


def _kimi_linear_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> KimiLinearConfig:
    """Restore the pinned llama.cpp Kimi Linear configuration."""
    arch = model.architecture
    _, layer_types, mtp_count = _derive_hybrid_layout(arch, metadata, model.tensor_names)
    assert layer_types is not None
    if mtp_count:
        raise ValueError("Kimi Linear GGUF does not support appended NextN blocks")
    gating = int(metadata[f"{arch}.expert_gating_func"])
    if gating != 2:
        raise ValueError(f"{arch}.expert_gating_func must be SIGMOID (2), got {gating}")
    heads = int(metadata[f"{arch}.attention.head_count"])
    kda_dim = int(metadata[f"{arch}.kda.head_dim"])
    qk_dim = int(metadata[f"{arch}.attention.key_length_mla"])
    extra_dim = int(metadata[f"{arch}.rope.dimension_count"])
    if qk_dim <= extra_dim:
        raise ValueError("Kimi Linear MLA key length must exceed the nominal extra-key width")
    fields = _shallow_fields(config)
    fields.update(
        model_type="kimi_linear",
        num_key_value_heads=1,
        head_dim=qk_dim,
        qk_nope_head_dim=qk_dim - extra_dim,
        qk_rope_head_dim=extra_dim,
        v_head_dim=int(metadata[f"{arch}.attention.value_length_mla"]),
        kv_lora_rank=int(metadata[f"{arch}.attention.kv_lora_rank"]),
        intermediate_size=int(metadata[f"{arch}.feed_forward_length"]),
        moe_intermediate_size=int(metadata[f"{arch}.expert_feed_forward_length"]),
        n_shared_experts=int(metadata[f"{arch}.expert_shared_count"]),
        first_k_dense_replace=int(metadata[f"{arch}.leading_dense_block_count"]),
        layer_types=layer_types,
        linear_num_key_heads=heads,
        linear_num_value_heads=heads,
        linear_key_head_dim=kda_dim,
        linear_value_head_dim=kda_dim,
        linear_conv_kernel_dim=int(metadata[f"{arch}.ssm.conv_kernel"]),
        hidden_act="silu",
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        disable_qmoe=True,
        q_lora_rank=None,
        rope_type=None,
        rope_theta=None,
        rope_scaling=None,
        partial_rotary_factor=None,
    )
    return KimiLinearConfig(**fields)


def _glm_dsa_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Restore the exact pinned GLM-5.2 MLA, DSA, and routed-MoE config."""
    arch = model.architecture
    _validate_closed_rope_scaling_metadata(metadata, arch)
    raw_gating = metadata.get(f"{arch}.expert_gating_func")
    gating = int(raw_gating) if raw_gating is not None else 2
    if gating != 2:
        raise ValueError(f"{arch}.expert_gating_func must be SIGMOID (2), got {gating}")

    qk_dim = int(metadata[f"{arch}.attention.key_length_mla"])
    rope_dim = int(metadata[f"{arch}.rope.dimension_count"])
    nope_dim = qk_dim - rope_dim
    value_dim = int(metadata[f"{arch}.attention.value_length_mla"])
    raw_kv_rank = metadata.get(f"{arch}.attention.kv_lora_rank")
    kv_rank = int(raw_kv_rank) if raw_kv_rank is not None else None
    compressed_key_dim = int(metadata[f"{arch}.attention.key_length"])
    if min(nope_dim, rope_dim, value_dim) <= 0 or (kv_rank is not None and kv_rank <= 0):
        raise ValueError("GLM-5.2 requires positive NoPE, RoPE, value, and KV-LoRA dimensions")
    if kv_rank is not None and compressed_key_dim != kv_rank + rope_dim:
        raise ValueError(
            f"{arch}.attention.key_length must equal kv_lora_rank + rope.dimension_count "
            f"({kv_rank + rope_dim}), got {compressed_key_dim}"
        )

    dense_prefix = int(metadata.get(f"{arch}.leading_dense_block_count", 0))
    if not 0 <= dense_prefix <= config.num_hidden_layers:
        raise ValueError(
            f"{arch}.leading_dense_block_count must be in [0, "
            f"{config.num_hidden_layers}], got {dense_prefix}"
        )

    indexer_types = _glm_dsa_indexer_types(config, metadata)
    names = set(model.tensor_names)
    routed_layers = set(range(dense_prefix, config.num_hidden_layers))
    bias_layers = {
        layer for layer in routed_layers if f"blk.{layer}.exp_probs_b.bias" in names
    }
    if bias_layers and bias_layers != routed_layers:
        raise ValueError(
            f"{arch} correction bias must be present for every routed layer or none; "
            f"found {sorted(bias_layers)}, expected {sorted(routed_layers)}"
        )
    route_scale = float(metadata.get(f"{arch}.expert_weights_scale", 0.0))
    if math.isclose(route_scale, 0.0):
        route_scale = 1.0
    if not math.isfinite(route_scale) or route_scale <= 0:
        raise ValueError(f"{arch}.expert_weights_scale must resolve to a positive value")

    fields = _shallow_fields(config)
    fields.update(
        model_type="glm_moe_dsa",
        num_key_value_heads=config.num_attention_heads,
        head_dim=nope_dim,
        q_lora_rank=(
            int(metadata[f"{arch}.attention.q_lora_rank"])
            if f"{arch}.attention.q_lora_rank" in metadata
            else None
        ),
        kv_lora_rank=kv_rank,
        qk_nope_head_dim=nope_dim,
        qk_rope_head_dim=rope_dim,
        v_head_dim=value_dim,
        intermediate_size=int(metadata[f"{arch}.feed_forward_length"]),
        moe_intermediate_size=config.moe_intermediate_size,
        n_shared_experts=config.n_shared_experts,
        first_k_dense_replace=dense_prefix,
        n_group=int(metadata.get(f"{arch}.expert_group_count", 1)),
        topk_group=int(metadata.get(f"{arch}.expert_group_used_count", 1)),
        routed_scaling_factor=route_scale,
        norm_topk_prob=bool(metadata.get(f"{arch}.expert_weights_norm")),
        routing_weight_normalization_floor=(
            6.103515625e-5 if bool(metadata.get(f"{arch}.expert_weights_norm")) else None
        ),
        hidden_act="silu",
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        use_expert_bias=bool(bias_layers),
        disable_qmoe=True,
        rope_interleave=True,
        indexer_rope_interleave=True,
        indexer_types=indexer_types,
        index_topk_freq=4 if config.max_position_embeddings >= 1_048_576 else 1,
        index_skip_topk_offset=3 if config.max_position_embeddings >= 1_048_576 else 0,
        partial_rotary_factor=1.0,
    )
    return ArchitectureConfig(**fields)


def _glm_dsa_indexer_types(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
) -> list[str]:
    """Resolve the pinned bool/scalar indexer schedule without model-ID heuristics."""
    arch = "glm-dsa"
    layers = config.num_hidden_layers
    raw = metadata.get(f"{arch}.attention.indexer.types")
    if raw is None:
        if config.max_position_embeddings < 1_048_576:
            result = ["full"] * layers
        else:
            if layers > 78:
                raise ValueError(
                    "glm-dsa models with more than 78 trunk layers must serialize "
                    "attention.indexer.types"
                )
            result = [
                "full" if index < 3 or (index - 2) % 4 == 0 else "shared"
                for index in range(layers)
            ]
    elif isinstance(raw, (list, tuple, np.ndarray)):
        if len(raw) != layers:
            raise ValueError(
                f"{arch}.attention.indexer.types has wrong array length; "
                f"expected {layers}, got {len(raw)}"
            )
        if any(value not in (0, 1, False, True) for value in raw):
            raise ValueError(f"{arch}.attention.indexer.types entries must be bool or 0/1")
        result = ["full" if bool(value) else "shared" for value in raw]
    else:
        if raw not in (0, 1, False, True):
            raise ValueError(f"{arch}.attention.indexer.types scalar must be bool or 0/1")
        result = ["full" if bool(raw) else "shared"] * layers
    if not result or result[0] != "full":
        raise ValueError("glm-dsa layer 0 must own a full indexer")
    return result


def _minimax_m2_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Restore MiniMax-M2's full-vector Q/K norms and exact sigmoid router."""
    arch = model.architecture
    _validate_closed_rope_scaling_metadata(metadata, arch)
    gating = metadata[f"{arch}.expert_gating_func"]
    if isinstance(gating, bool) or not isinstance(gating, (int, np.integer)):
        raise TypeError(f"{arch}.expert_gating_func must be the integer SIGMOID enum")
    if int(gating) != 2:
        raise ValueError(f"{arch}.expert_gating_func must be SIGMOID (2), got {gating}")

    head_dim = int(metadata[f"{arch}.attention.key_length"])
    value_dim = int(metadata[f"{arch}.attention.value_length"])
    rope_dim = int(metadata[f"{arch}.rope.dimension_count"])
    intermediate = int(metadata[f"{arch}.feed_forward_length"])
    expert_intermediate = int(metadata[f"{arch}.expert_feed_forward_length"])
    if value_dim != head_dim:
        raise ValueError(f"{arch}.attention.value_length must equal key_length")
    if rope_dim <= 0 or rope_dim > head_dim or rope_dim % 2:
        raise ValueError(
            f"{arch}.rope.dimension_count must be positive, even, and <= head_dim"
        )
    if expert_intermediate != intermediate:
        raise ValueError(
            f"{arch}.expert_feed_forward_length must equal feed_forward_length "
            f"({intermediate}), got {expert_intermediate}"
        )
    if int(metadata.get(f"{arch}.expert_shared_count", 0)):
        raise ValueError("MiniMax-M2 does not support shared experts")
    if int(metadata.get(f"{arch}.leading_dense_block_count", 0)):
        raise ValueError("MiniMax-M2 uses routed experts in every layer")
    if int(metadata.get(f"{arch}.attention.sliding_window", 0)):
        raise ValueError("MiniMax-M2 does not use sliding-window attention")
    if int(metadata.get(f"{arch}.expert_group_count", 0)) not in (0, 1):
        raise ValueError("MiniMax-M2 does not use grouped expert selection")
    if int(metadata.get(f"{arch}.nextn_predict_layers", 0)):
        raise ValueError("MiniMax-M2 GGUF does not define an executable NextN graph")
    route_scale = float(metadata.get(f"{arch}.expert_weights_scale", 0.0))
    if math.isclose(route_scale, 0.0):
        route_scale = 1.0
    if not math.isfinite(route_scale) or route_scale <= 0:
        raise ValueError("MiniMax-M2 expert_weights_scale must resolve to a positive value")

    return dataclasses.replace(
        config,
        model_type="minimax_m2_gguf",
        head_dim=head_dim,
        intermediate_size=intermediate,
        moe_intermediate_size=intermediate,
        hidden_act="silu",
        tie_word_embeddings=False,
        attn_qkv_bias=False,
        attn_o_bias=False,
        mlp_bias=False,
        attn_qk_norm=True,
        attn_qk_norm_full=True,
        rope_type="default",
        rope_interleave=False,
        partial_rotary_factor=rope_dim / head_dim,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=route_scale,
        routing_weight_normalization_floor=6.103515625e-5,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        use_expert_bias=True,
        disable_qmoe=True,
    )


def _mistral4_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Restore the pinned DeepSeek-V2 MLA/MoE contract serialized as Mistral4."""
    arch = model.architecture
    _validate_conventional_moe_rope_scaling(metadata, arch)
    q_lora_rank = int(metadata[f"{arch}.attention.q_lora_rank"])
    kv_lora_rank = int(metadata[f"{arch}.attention.kv_lora_rank"])
    qk_head_dim = int(metadata[f"{arch}.attention.key_length_mla"])
    rope_dim = int(metadata[f"{arch}.rope.dimension_count"])
    nope_dim = qk_head_dim - rope_dim
    value_dim = int(metadata[f"{arch}.attention.value_length_mla"])
    serialized_key = int(metadata[f"{arch}.attention.key_length"])
    serialized_value = int(metadata[f"{arch}.attention.value_length"])
    if min(q_lora_rank, kv_lora_rank, nope_dim, rope_dim, value_dim) <= 0:
        raise ValueError("Mistral4 requires positive Q/KV-LoRA and MLA head dimensions")
    if serialized_key != kv_lora_rank + rope_dim or serialized_value != kv_lora_rank:
        raise ValueError(
            "Mistral4 compressed cache geometry must satisfy "
            "key_length=kv_lora_rank+rope_dim and value_length=kv_lora_rank"
        )
    if int(config.num_key_value_heads) != 1:
        raise ValueError("Mistral4 GGUF must serialize attention.head_count_kv=1")
    if int(metadata.get(f"{arch}.nextn_predict_layers", 0)):
        raise ValueError("Mistral4's pinned graph does not execute a NextN sidecar")
    temperature_scale = float(metadata.get(f"{arch}.attention.temperature_scale", 0.0))
    if not math.isclose(temperature_scale, 0.0):
        raise ValueError(
            "Mistral4 attention.temperature_scale is loader-optional but is not emitted "
            "by the pinned converter; nonzero temperature scaling is outside this route"
        )

    layers = config.num_hidden_layers
    dense_prefix = int(metadata.get(f"{arch}.leading_dense_block_count", 0))
    if not 0 <= dense_prefix < layers:
        raise ValueError("Mistral4 requires a valid dense prefix and at least one MoE layer")
    experts = int(config.num_local_experts or 0)
    top_k = int(config.num_experts_per_tok or 0)
    shared = int(config.n_shared_experts or 0)
    n_group = int(metadata.get(f"{arch}.expert_group_count", 1))
    topk_group = int(metadata.get(f"{arch}.expert_group_used_count", 1))
    if min(experts, top_k, shared) <= 0 or top_k > experts:
        raise ValueError("Mistral4 requires routed experts, top-k, and shared experts")
    if n_group != 1 or topk_group != 1:
        raise ValueError("Mistral4's proven route requires one expert group")

    raw_gating = metadata.get(f"{arch}.expert_gating_func", 1)
    if isinstance(raw_gating, bool) or not isinstance(raw_gating, (int, np.integer)):
        raise TypeError(f"{arch}.expert_gating_func must be an integer enum")
    gating = int(raw_gating)
    if gating not in (1, 2):
        raise ValueError(f"{arch}.expert_gating_func must be SOFTMAX (1) or SIGMOID (2)")

    routed_layers = set(range(dense_prefix, layers))
    names = set(model.tensor_names)
    bias_layers = {
        layer for layer in routed_layers if f"blk.{layer}.exp_probs_b.bias" in names
    }
    if bias_layers and bias_layers != routed_layers:
        raise ValueError(
            f"Mistral4 correction bias must be present in every routed layer or none; "
            f"found {sorted(bias_layers)}, expected {sorted(routed_layers)}"
        )
    route_scale = float(metadata.get(f"{arch}.expert_weights_scale", 0.0))
    if math.isclose(route_scale, 0.0):
        route_scale = 1.0
    if not math.isfinite(route_scale) or route_scale <= 0:
        raise ValueError("Mistral4 expert_weights_scale must resolve to a positive value")
    norm_topk_prob = bool(metadata.get(f"{arch}.expert_weights_norm"))

    rope_scaling = None if config.rope_scaling is None else dict(config.rope_scaling)
    yarn_log_multiplier = metadata.get(f"{arch}.rope.scaling.yarn_log_multiplier")
    if yarn_log_multiplier is not None:
        if rope_scaling is None or config.rope_type != "yarn":
            raise ValueError("Mistral4 yarn_log_multiplier requires YaRN rope metadata")
        rope_scaling["mscale"] = 1.0
        rope_scaling["mscale_all_dim"] = float(yarn_log_multiplier) / 0.1

    return dataclasses.replace(
        config,
        model_type="mistral4_gguf",
        head_dim=qk_head_dim,
        num_key_value_heads=1,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=nope_dim,
        qk_rope_head_dim=rope_dim,
        v_head_dim=value_dim,
        first_k_dense_replace=dense_prefix,
        hidden_act="silu",
        attn_qkv_bias=False,
        attn_o_bias=False,
        mlp_bias=False,
        rope_scaling=rope_scaling,
        rope_interleave=True,
        partial_rotary_factor=None,
        scoring_func="softmax" if gating == 1 else "sigmoid",
        topk_method="greedy",
        n_group=n_group,
        topk_group=topk_group,
        use_expert_bias=bool(bias_layers),
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=route_scale,
        routing_weight_normalization_floor=(6.103515625e-5 if norm_topk_prob else None),
        disable_qmoe=True,
        num_nextn_predict_layers=0,
    )


def _kimi_k3_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> KimiK3Config:
    """Restore the exact pinned llama.cpp Kimi-K3 text configuration."""
    arch = model.architecture
    _, layer_types, mtp_count = _derive_hybrid_layout(arch, metadata, model.tensor_names)
    assert layer_types is not None
    if mtp_count:
        raise ValueError("Kimi-K3 GGUF does not support appended NextN blocks")
    gating = int(metadata[f"{arch}.expert_gating_func"])
    if gating != 2:
        raise ValueError(f"{arch}.expert_gating_func must be SIGMOID (2), got {gating}")
    heads = int(metadata[f"{arch}.attention.head_count"])
    kda_dim = int(metadata[f"{arch}.kda.head_dim"])
    qk_dim = int(metadata[f"{arch}.attention.key_length_mla"])
    extra_dim = int(metadata[f"{arch}.rope.dimension_count"])
    if qk_dim <= extra_dim:
        raise ValueError("Kimi-K3 MLA key length must exceed the nominal extra-key width")
    fields = _shallow_fields(config)
    fields.update(
        model_type="kimi_k3",
        num_key_value_heads=1,
        head_dim=qk_dim,
        q_lora_rank=int(metadata[f"{arch}.attention.q_lora_rank"]),
        qk_nope_head_dim=qk_dim - extra_dim,
        qk_rope_head_dim=extra_dim,
        v_head_dim=int(metadata[f"{arch}.attention.value_length_mla"]),
        kv_lora_rank=int(metadata[f"{arch}.attention.kv_lora_rank"]),
        intermediate_size=int(metadata[f"{arch}.feed_forward_length"]),
        moe_intermediate_size=int(metadata[f"{arch}.expert_feed_forward_length"]),
        n_shared_experts=int(metadata.get(f"{arch}.expert_shared_count", 0)),
        first_k_dense_replace=int(metadata.get(f"{arch}.leading_dense_block_count", 0)),
        routed_scaling_factor=float(metadata.get(f"{arch}.expert_weights_scale", 0.0)),
        norm_topk_prob=bool(metadata.get(f"{arch}.expert_weights_norm")),
        layer_types=layer_types,
        linear_num_key_heads=heads,
        linear_num_value_heads=heads,
        linear_key_head_dim=kda_dim,
        linear_value_head_dim=kda_dim,
        linear_conv_kernel_dim=int(metadata[f"{arch}.ssm.conv_kernel"]),
        linear_gate_lower_bound=-float(
            metadata.get(f"{arch}.kda.gate_lower_bound", -math.inf)
        ),
        linear_use_full_rank_gate=True,
        routed_expert_hidden_size=int(
            metadata.get(f"{arch}.expert_latent_length", config.hidden_size)
        ),
        attn_res_block_size=int(metadata.get(f"{arch}.attn_res.block_size", 0)),
        latent_moe_use_norm=True,
        mla_use_output_gate=True,
        activation_situ_beta=float(metadata.get(f"{arch}.activation.situ_beta", 1.0)),
        activation_situ_linear_beta=float(
            metadata.get(f"{arch}.activation.situ_linear_beta", 0.0)
        ),
        hidden_act="situ",
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        disable_qmoe=True,
        tie_word_embeddings=False,
        rope_type=None,
        rope_theta=None,
        rope_scaling=None,
        partial_rotary_factor=None,
    )
    return KimiK3Config(**fields)


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


def _specialized_encoder_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Resolve tensor-selected variants for the promoted stateless GGUF encoders."""
    arch = model.architecture
    if config.num_key_value_heads != config.num_attention_heads:
        raise ValueError(f"{arch} GGUF requires head_count_kv == head_count")
    if bool(metadata[f"{arch}.attention.causal"]):
        raise ValueError(f"{arch}.attention.causal must be false for encoder import")

    pooling_type = int(metadata.get(f"{arch}.pooling_type", 0))
    if pooling_type not in {0, 1, 2}:
        raise ValueError(f"{arch}.pooling_type={pooling_type} is not a known encoder pooling")
    if metadata.get(f"{arch}.classifier.output_labels"):
        raise ValueError(f"{arch} classifier heads are not part of feature extraction")

    names = set(model.tensor_names)
    layers = config.num_hidden_layers

    if arch == "nomic-bert-moe":
        frequency = int(metadata[f"{arch}.moe_every_n_layers"])
        if frequency < 2:
            raise ValueError("nomic-bert-moe.moe_every_n_layers must be at least 2")
        token_types = int(metadata.get("tokenizer.ggml.token_type_count", 0))
        if token_types <= 0:
            raise ValueError("nomic-bert-moe requires tokenizer.ggml.token_type_count")

        def _scheduled_family(suffix: str, layer_ids: list[int]) -> bool:
            expected = {f"blk.{layer}.{suffix}" for layer in layer_ids}
            present = expected & names
            if present and present != expected:
                raise ValueError(f"{arch} optional tensor family {suffix!r} must be complete")
            return bool(present)

        all_layers = list(range(layers))
        dense_layers = [layer for layer in all_layers if layer % frequency != 1]
        config.pooling_type = pooling_type
        config.moe_layer_frequency = frequency
        config.hidden_act = "gelu_pytorch_tanh"
        config.norm_topk_prob = False
        config.routed_scaling_factor = 1.0
        config.type_vocab_size = token_types
        config.encoder_use_token_type_embeddings = "token_types.weight" in names
        fused_bias = _scheduled_family("attn_qkv.bias", all_layers)
        config.encoder_q_bias = fused_bias
        config.encoder_k_bias = fused_bias
        config.encoder_v_bias = fused_bias
        config.attn_qkv_bias = fused_bias
        config.attn_o_bias = _scheduled_family("attn_output.bias", all_layers)
        config.encoder_ffn_up_bias = _scheduled_family("ffn_up.bias", dense_layers)
        config.encoder_ffn_down_bias = _scheduled_family("ffn_down.bias", dense_layers)
        config.mlp_bias = config.encoder_ffn_up_bias or config.encoder_ffn_down_bias
        return config

    def _all_or_none(suffix: str) -> bool:
        expected = {f"blk.{layer}.{suffix}" for layer in range(layers)}
        present = expected & names
        if present and present != expected:
            raise ValueError(
                f"{arch} optional tensor family {suffix!r} must be present in every layer "
                "or absent entirely"
            )
        return bool(present)

    config.pooling_type = pooling_type
    config.encoder_q_bias = _all_or_none("attn_q.bias")
    config.encoder_k_bias = _all_or_none("attn_k.bias")
    config.encoder_v_bias = _all_or_none("attn_v.bias")
    config.attn_qkv_bias = (
        config.encoder_q_bias or config.encoder_k_bias or config.encoder_v_bias
    )
    config.attn_o_bias = _all_or_none("attn_output.bias")
    config.encoder_ffn_up_bias = _all_or_none("ffn_up.bias")
    config.encoder_ffn_down_bias = _all_or_none("ffn_down.bias")
    config.mlp_bias = config.encoder_ffn_up_bias or config.encoder_ffn_down_bias

    token_types = int(metadata.get("tokenizer.ggml.token_type_count", 0))
    config.type_vocab_size = token_types
    config.encoder_use_token_type_embeddings = "token_types.weight" in names

    if arch in {"eurobert", "neo-bert"}:
        config.type_vocab_size = 0
        config.encoder_use_token_type_embeddings = False
        config.hidden_act = "silu"
        config.attn_qkv_bias = False
        config.encoder_q_bias = False
        config.encoder_k_bias = False
        config.encoder_v_bias = False
        config.attn_o_bias = False
        config.encoder_ffn_up_bias = False
        config.encoder_ffn_down_bias = False
        config.mlp_bias = False
    elif arch == "nomic-bert":
        if token_types <= 0:
            raise ValueError(
                "nomic-bert requires tokenizer.ggml.token_type_count even when the "
                "optional token_types tensor is absent"
            )
        config.hidden_act = "silu"
    else:
        if token_types <= 0 or not config.encoder_use_token_type_embeddings:
            raise ValueError("jina-bert-v2 requires its token-type embedding table")
        config.hidden_act = "gelu"
        config.rope_type = None
        config.rope_theta = None
        config.rope_scaling = None
        config.partial_rotary_factor = None
        q_norm = _all_or_none("attn_q_norm.weight")
        k_norm = _all_or_none("attn_k_norm.weight")
        if q_norm != k_norm:
            raise ValueError("jina-bert-v2 Q and K norm families must appear together")
        config.encoder_qk_norm = q_norm
        config.encoder_extra_attention_norm = _all_or_none("attn_norm_2.weight")
        has_gate = _all_or_none("ffn_gate.weight")
        config.encoder_fused_geglu = not has_gate
    return config


def _jina_bert_v3_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Resolve JinaBERT-v3's QKV representation and exact dense/MoE schedule."""
    arch = model.architecture
    _validate_closed_rope_scaling_metadata(metadata, arch, allowed_suffixes={"type"})
    if config.num_key_value_heads != config.num_attention_heads:
        raise ValueError(f"{arch} GGUF requires head_count_kv == head_count")
    if bool(metadata[f"{arch}.attention.causal"]):
        raise ValueError(f"{arch}.attention.causal must be false for encoder import")

    pooling_type = int(metadata.get(f"{arch}.pooling_type", 0))
    if pooling_type not in {0, 1, 2}:
        raise ValueError(f"{arch}.pooling_type={pooling_type} is not a known encoder pooling")
    if metadata.get(f"{arch}.classifier.output_labels"):
        raise ValueError(f"{arch} classifier heads are not part of feature extraction")

    names = set(model.tensor_names)
    layer_count = config.num_hidden_layers
    layers = range(layer_count)

    def _all_or_none(suffix: str, selected_layers=layers) -> bool:
        expected = {f"blk.{layer}.{suffix}" for layer in selected_layers}
        present = expected & names
        if present and present != expected:
            raise ValueError(
                f"{arch} optional tensor family {suffix!r} must be all-layers or absent"
            )
        return bool(present)

    fused = {f"blk.{layer}.attn_qkv.weight" for layer in layers}
    split = {
        f"blk.{layer}.attn_{projection}.weight"
        for layer in layers
        for projection in ("q", "k", "v")
    }
    if fused <= names and not names & split:
        config.encoder_fused_qkv = True
        config.attn_qkv_bias = _all_or_none("attn_qkv.bias")
        config.encoder_q_bias = False
        config.encoder_k_bias = False
        config.encoder_v_bias = False
    elif split <= names and not names & fused:
        config.encoder_fused_qkv = False
        config.attn_qkv_bias = False
        config.encoder_q_bias = _all_or_none("attn_q.bias")
        config.encoder_k_bias = _all_or_none("attn_k.bias")
        config.encoder_v_bias = _all_or_none("attn_v.bias")
    else:
        raise ValueError(
            f"{arch} requires a uniform complete fused-QKV or split-Q/K/V tensor family"
        )

    config.attn_o_bias = _all_or_none("attn_output.bias")
    config.pooling_type = pooling_type
    token_types = int(metadata.get("tokenizer.ggml.token_type_count", 0))
    if token_types <= 0:
        raise ValueError(f"{arch} tokenizer.ggml.token_type_count must be positive")
    config.type_vocab_size = token_types
    config.encoder_use_token_type_embeddings = "token_types.weight" in names
    config.hidden_act = "gelu_pytorch_tanh"

    interval = int(metadata.get(f"{arch}.moe_every_n_layers", 0))
    expert_metadata = tuple(
        suffix
        for suffix in (
            "expert_count",
            "expert_used_count",
            "expert_feed_forward_length",
            "expert_weights_norm",
            "expert_weights_scale",
        )
        if f"{arch}.{suffix}" in metadata
    )
    if interval != 0 or expert_metadata:
        raise ValueError(
            f"{arch} MoE metadata is unsupported: the pinned loader does not read "
            "moe_every_n_layers, so only its reachable dense tensor path is importable"
        )

    config.encoder_ffn_up_bias = _all_or_none("ffn_up.bias")
    config.encoder_ffn_down_bias = _all_or_none("ffn_down.bias")
    config.mlp_bias = config.encoder_ffn_up_bias or config.encoder_ffn_down_bias
    return config


def _gguf_embedding_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> ArchitectureConfig:
    """Select only exact stateless profiles from the two conditional loaders."""
    arch = model.architecture
    if bool(metadata[f"{arch}.attention.causal"]):
        raise ValueError(f"{arch}.attention.causal must be false")
    pooling_type = int(metadata.get(f"{arch}.pooling_type", 0))
    if pooling_type not in {0, 1, 2, 3}:
        raise ValueError(f"{arch}.pooling_type={pooling_type} is unsupported")
    if metadata.get(f"{arch}.classifier.output_labels"):
        raise ValueError(f"{arch} classifier/reranker heads are not feature extraction")

    head_dim = config.hidden_size // config.num_attention_heads
    if config.head_dim != head_dim:
        raise ValueError(f"{arch} requires full-head RoPE")
    key_length = int(metadata.get(f"{arch}.attention.key_length", head_dim))
    value_length = int(metadata.get(f"{arch}.attention.value_length", head_dim))
    if key_length != head_dim or value_length != head_dim:
        raise ValueError(f"{arch} requires equal full-width Q/K/V heads")

    fields = _shallow_fields(config)
    fields.update(
        pooling_type=pooling_type,
        hidden_act="gelu_pytorch_tanh" if arch == "gemma-embedding" else "silu",
        attention_multiplier=(
            head_dim**-0.5
            if arch == "gemma-embedding"
            else float(metadata.get(f"{arch}.attention.scale", head_dim**-0.5))
        ),
    )
    if arch == "gemma-embedding":
        pattern = int(metadata.get(f"{arch}.attention.sliding_window_pattern", 6))
        if pattern <= 0:
            raise ValueError("gemma-embedding sliding_window_pattern must be positive")
        fields.update(
            layer_types=[
                ("sliding_attention" if layer % pattern < pattern - 1 else "full_attention")
                for layer in range(config.num_hidden_layers)
            ],
            rope_local_base_freq=float(metadata.get(f"{arch}.rope.freq_base_swa", 10_000.0)),
            embedding_dense_2_out=(
                int(metadata[f"{arch}.dense_2_feat_out"])
                if "dense_2.weight" in model.tensor_names
                else None
            ),
            embedding_dense_3_in=(
                int(metadata[f"{arch}.dense_3_feat_in"])
                if "dense_3.weight" in model.tensor_names
                else None
            ),
        )
    else:
        fields.update(
            layer_types=["full_attention"] * config.num_hidden_layers,
            sliding_window=None,
            rope_local_base_freq=None,
            embedding_dense_2_out=None,
            embedding_dense_3_in=None,
        )
    return ArchitectureConfig(**fields)


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


def _conventional_legacy_postprocess(
    config: ArchitectureConfig, metadata: dict[str, Any], model: Any
) -> ArchitectureConfig:
    """Apply only pinned, architecture-owned defaults for legacy decoders."""
    del metadata
    has_command_qk_norm = model.architecture == "command-r" and any(
        name.endswith(("attn_q_norm.weight", "attn_k_norm.weight"))
        for name in model.tensor_names
    )
    if has_command_qk_norm:
        raise ValueError(
            "command-r GGUF with per-head Q/K LayerNorm tensors is not supported: "
            "the current Attention graph shares one norm vector across heads"
        )
    return dataclasses.replace(
        config,
        hidden_act={
            "codeshell": "gelu_pytorch_tanh",
            "jais2": "relu2",
            "starcoder": "gelu_pytorch_tanh",
        }.get(model.architecture, config.hidden_act),
        intermediate_size=(
            config.intermediate_size // 2
            if model.architecture == "qwen"
            else config.intermediate_size
        ),
        tie_word_embeddings=(
            not {"token_embd.weight", "output.weight"}.issubset(model.tensor_names)
            if model.architecture == "codeshell"
            else config.tie_word_embeddings
        ),
    )


def _exact_legacy_gguf_postprocess(
    config: ArchitectureConfig, metadata: dict[str, Any], model: Any
) -> ArchitectureConfig:
    """Materialize only the architecture variants admitted by strict validation."""
    architecture = model.architecture
    fields: dict[str, Any] = {
        "model_type": {
            "gptneox": "gpt_neox",
            "jais": "jais",
            "mpt": "mpt",
            "refact": "refact",
            "ernie4_5": "ernie4_5",
            "openelm": "openelm",
        }[architecture],
        "hidden_act": {
            "gptneox": "gelu",
            "jais": "silu",
            "mpt": "gelu",
            "refact": "silu",
            "ernie4_5": "silu",
            "openelm": "silu",
        }[architecture],
        "use_parallel_residual": architecture == "gptneox",
        # Older conventional GGUFs may serialize the rotary dimension while
        # omitting the default 10,000 frequency base. These architectures
        # still execute RoPE in the pinned loaders.
        "rope_type": (
            "default" if architecture in {"gptneox", "ernie4_5", "openelm"} else None
        ),
        "rope_interleave": architecture == "ernie4_5",
        "alibi_max_bias": (
            float(metadata[f"{architecture}.attention.max_alibi_bias"])
            if architecture in {"jais", "mpt"}
            else 8.0
            if architecture == "refact"
            else None
        ),
        "attention_scale": (1.0 / config.head_dim if architecture == "jais" else None),
        "tie_word_embeddings": (
            architecture == "openelm"
            or (
                architecture in {"mpt", "refact", "ernie4_5"}
                and "output.weight" not in model.tensor_names
            )
        ),
        "attn_qkv_bias": architecture in {"gptneox", "jais"}
        or (architecture == "mpt" and config.attn_qkv_bias),
        "attn_o_bias": architecture in {"gptneox", "jais"}
        or (architecture == "mpt" and config.attn_o_bias),
        "mlp_bias": architecture in {"gptneox", "jais"}
        or (architecture == "mpt" and config.mlp_bias),
    }
    if architecture == "gptneox":
        heads = int(metadata["gptneox.attention.head_count"])
        head_dim = int(metadata["gptneox.embedding_length"]) // heads
        rotary_dim = int(metadata.get("gptneox.rope.dimension_count", head_dim))
        fields.update(
            head_dim=head_dim,
            partial_rotary_factor=rotary_dim / head_dim,
        )
    if architecture == "openelm":
        heads = tuple(int(value) for value in metadata["openelm.attention.head_count"])
        kv_heads = tuple(int(value) for value in metadata["openelm.attention.head_count_kv"])
        intermediate = tuple(int(value) for value in metadata["openelm.feed_forward_length"])
        fields.update(
            num_attention_heads=heads[0],
            num_key_value_heads=kv_heads[0],
            intermediate_size=intermediate[0],
            layer_attention_head_counts=heads,
            layer_attention_kv_head_counts=kv_heads,
            layer_intermediate_sizes=intermediate,
            attn_qk_norm=True,
        )
    return dataclasses.replace(config, **fields)


def _plamo_postprocess(
    config: ArchitectureConfig, metadata: dict[str, Any], model: Any
) -> ArchitectureConfig:
    """Validate and materialize the fixed PLaMo-13B converter contract."""
    prefix = "plamo."
    _validate_closed_rope_scaling_metadata(metadata, "plamo")
    expected_ints = {
        "context_length": 4096,
        "embedding_length": 5120,
        "block_count": 40,
        "feed_forward_length": 16640,
        "attention.head_count": 40,
        "attention.head_count_kv": 5,
    }
    for suffix, expected in expected_ints.items():
        actual = int(metadata[f"{prefix}{suffix}"])
        if actual != expected:
            raise ValueError(f"PLaMo requires {prefix}{suffix}={expected}, got {actual}")

    epsilon = float(metadata[f"{prefix}attention.layer_norm_rms_epsilon"])
    if not math.isclose(epsilon, 1e-6, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"PLaMo requires RMSNorm epsilon 1e-6, got {epsilon}")
    rope_theta = float(metadata.get(f"{prefix}rope.freq_base", 10000.0))
    if not math.isclose(rope_theta, 10000.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"PLaMo requires rope.freq_base=10000, got {rope_theta}")
    rope_dim = int(metadata.get(f"{prefix}rope.dimension_count", 128))
    if rope_dim != 128:
        raise ValueError(f"PLaMo requires full-head rope.dimension_count=128, got {rope_dim}")
    return dataclasses.replace(
        config,
        model_type="plamo",
        hidden_size=5120,
        intermediate_size=16640,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=5,
        head_dim=128,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_type="default",
        partial_rotary_factor=1.0,
        rope_interleave=False,
        hidden_act="silu",
        attn_qkv_bias=False,
        attn_o_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
    )


def _plm_postprocess(
    config: ArchitectureConfig, metadata: dict[str, Any], model: Any
) -> ArchitectureConfig:
    """Validate and materialize the exact pinned PLM GGUF contract."""
    arch = model.architecture
    key_dim = int(metadata[f"{arch}.attention.key_length"])
    value_dim = int(metadata[f"{arch}.attention.value_length"])
    rope_dim = int(metadata[f"{arch}.rope.dimension_count"])
    kv_rank = int(metadata[f"{arch}.attention.kv_lora_rank"])
    nope_dim = key_dim - rope_dim
    heads = config.num_attention_heads

    if min(nope_dim, rope_dim, value_dim, kv_rank) <= 0:
        raise ValueError("PLM requires positive NoPE, RoPE, value, and KV-LoRA dimensions")
    if rope_dim % 2:
        raise ValueError("PLM rope.dimension_count must be even")
    if "output.weight" in model.tensor_names:
        raise ValueError(
            "PLM uses token_embd.weight as the sole tied output owner; "
            "standalone output.weight is not accepted"
        )

    expected: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (config.vocab_size, config.hidden_size),
        "output_norm.weight": (config.hidden_size,),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"blk.{layer}."
        expected.update(
            {
                prefix + "attn_norm.weight": (config.hidden_size,),
                prefix + "attn_q.weight": (heads * key_dim, config.hidden_size),
                prefix + "attn_kv_a_mqa.weight": (
                    kv_rank + rope_dim,
                    config.hidden_size,
                ),
                prefix + "attn_kv_a_norm.weight": (kv_rank,),
                prefix + "attn_kv_b.weight": (
                    heads * (nope_dim + value_dim),
                    kv_rank,
                ),
                prefix + "attn_output.weight": (
                    config.hidden_size,
                    heads * value_dim,
                ),
                prefix + "ffn_norm.weight": (config.hidden_size,),
                prefix + "ffn_up.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                prefix + "ffn_down.weight": (
                    config.hidden_size,
                    config.intermediate_size,
                ),
            }
        )

    raw_shapes = {
        name: tuple(int(dim) for dim in shape)
        for name, _raw, _qtype, shape in model.tensor_items_raw()
    }
    missing = sorted(set(expected) - set(raw_shapes))
    mismatched = sorted(
        f"{name}: expected {shape}, got {raw_shapes[name]}"
        for name, shape in expected.items()
        if name in raw_shapes and raw_shapes[name] != shape
    )
    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if mismatched:
            details.append(f"shape_mismatches={mismatched}")
        raise ValueError("Invalid PLM tensor contract: " + "; ".join(details))

    return dataclasses.replace(
        config,
        head_dim=key_dim,
        num_key_value_heads=heads,
        q_lora_rank=None,
        kv_lora_rank=kv_rank,
        qk_nope_head_dim=nope_dim,
        qk_rope_head_dim=rope_dim,
        v_head_dim=value_dim,
        hidden_act="relu2",
        tie_word_embeddings=True,
        attn_qkv_bias=False,
        attn_o_bias=False,
        mlp_bias=False,
        rope_interleave=True,
        partial_rotary_factor=1.0,
    )


def _qwen4exp_postprocess(
    config: ArchitectureConfig,
    metadata: dict[str, Any],
    model: Any,
) -> Qwen4ExpConfig:
    """Construct the exact Qwen3.8 Flash-Next text config from GGUF metadata."""
    arch = model.architecture
    prefix = f"{arch}."
    ratios_raw = metadata[f"{prefix}attention.compress_ratios"]
    if not isinstance(ratios_raw, (list, tuple, np.ndarray)):
        raise TypeError(f"{prefix}attention.compress_ratios must be an integer array")
    ratios = [int(value) for value in ratios_raw]
    if len(ratios) != config.num_hidden_layers:
        raise ValueError(
            f"{prefix}attention.compress_ratios must contain exactly "
            f"{config.num_hidden_layers} entries, got {len(ratios)}"
        )
    nonzero_ratios = {value for value in ratios if value > 0}
    if any(value < 0 for value in ratios) or len(nonzero_ratios) != 1:
        raise ValueError(
            f"{prefix}attention.compress_ratios must contain zero for DeltaNet "
            "layers and one consistent positive QSA ratio"
        )
    compress_ratio = nonzero_ratios.pop()
    layer_types = [
        "linear_attention" if ratio == 0 else "qwen_sparse_attention" for ratio in ratios
    ]

    ple_layers_raw = metadata[f"{prefix}ple.layers"]
    if not isinstance(ple_layers_raw, (list, tuple, np.ndarray)):
        raise TypeError(f"{prefix}ple.layers must be an integer array")
    ple_layer_ids = [int(layer) + 1 for layer in ple_layers_raw]
    if len(set(ple_layer_ids)) != len(ple_layer_ids):
        raise ValueError(f"{prefix}ple.layers contains duplicate layer indices")

    fields = _shallow_fields(config)
    fields.update(
        model_type="qwen4_exp_text",
        layer_types=layer_types,
        hc_count=int(metadata[f"{prefix}hyper_connection.count"]),
        hc_lowrank=int(metadata[f"{prefix}hyper_connection.low_rank"]),
        ple_layer_ids=ple_layer_ids,
        ple_embed_dim=config.hidden_size,
        ple_conv_kernel_size=int(metadata[f"{prefix}ple.conv_kernel"]),
        ngram_size=int(metadata[f"{prefix}ple.ngram_size"]),
        heads_per_ngram=int(metadata[f"{prefix}ple.heads_per_ngram"]),
        # These source-owned values are absent from GGUF because its concrete
        # hash arrays and combined PLE table already embody them.
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        seed=1234,
        split_ngram_parts=128,
        indexer_n_heads=int(metadata[f"{prefix}attention.indexer.head_count"]),
        indexer_kv_heads=1,
        indexer_head_dim=int(metadata[f"{prefix}attention.indexer.key_length"]),
        indexer_budget=int(metadata[f"{prefix}attention.indexer.top_k"]),
        indexer_compress_ratio=compress_ratio,
        output_gate_type="sigmoid",
        eos_token_id=int(metadata[f"{prefix}ple.eos_token_id"]),
        mrope_section=[int(value) for value in metadata[f"{prefix}rope.dimension_sections"]],
        mrope_interleaved=True,
        norm_topk_prob=True,
        mtp_num_hidden_layers=0,
        mtp_use_dedicated_embeddings=False,
    )
    return Qwen4ExpConfig(**fields)


# Architecture-specific config postprocessors, keyed by the name a
# :class:`GGUFArchitectureSpec` refers to. Each takes a base ArchitectureConfig
# + raw metadata and returns an architecture-specific config subclass.
#
# Keyed by postprocessor name rather than by ``model_type``: the old
# model_type keying is what let the Gemma weight processor drift out of reach
# when an architecture's model_type gained a ``_text`` suffix.
_CONFIG_POSTPROCESSORS: dict[str, Any] = {
    "conventional_legacy": _conventional_legacy_postprocess,
    "exact_legacy_gguf": _exact_legacy_gguf_postprocess,
    "dflash": _dflash_postprocess,
    "eagle3": _eagle3_postprocess,
    "dream": _dream_postprocess,
    "llada": _llada_postprocess,
    "llada_moe": _llada_moe_postprocess,
    "rnd1": _rnd1_postprocess,
    "olmo": _olmo_postprocess,
    "moe": _moe_postprocess,
    "dbrx": _dbrx_postprocess,
    "arctic": _arctic_postprocess,
    "grok": _grok_postprocess,
    "grovemoe": _grovemoe_postprocess,
    "hunyuan_moe_gguf": _hunyuan_moe_gguf_postprocess,
    "ernie45_moe": _ernie45_moe_postprocess,
    "smallthinker": _smallthinker_postprocess,
    "conventional_shared_moe": _conventional_shared_moe_postprocess,
    "granite": _granite_postprocess,
    "granitemoe": _granitemoe_postprocess,
    "hy_v3": _hy_v3_postprocess,
    "phimoe": _phimoe_postprocess,
    "pangu_embedded": _pangu_embedded_postprocess,
    "dense_sliding": _dense_sliding_postprocess,
    "qwen3": _qwen3_postprocess,
    "starcoder2": _starcoder2_postprocess,
    "gemma2": _gemma2_postprocess,
    "baichuan": _baichuan_postprocess,
    "chatglm": _chatglm_postprocess,
    "phi2": _phi2_postprocess,
    "seed_oss": _seed_oss_postprocess,
    "apertus": _apertus_postprocess,
    "gemma3": _gemma3_postprocess,
    "gemma4": _gemma4_postprocess,
    "muse_glimmer": _muse_glimmer_postprocess,
    "mamba": _mamba_postprocess,
    "mamba2": _mamba2_postprocess,
    "falcon_h1": _falcon_h1_postprocess,
    "plamo": _plamo_postprocess,
    "plamo2": _plamo2_postprocess,
    "plm": _plm_postprocess,
    "jamba": _jamba_postprocess,
    "lfm2moe": _lfm2moe_postprocess,
    "nemotron_h": _nemotron_h_postprocess,
    "nemotron_h_moe": _nemotron_h_postprocess,
    "granitehybrid": _granitehybrid_postprocess,
    "bert_encoder": _bert_encoder_postprocess,
    "modern_bert_encoder": _modern_bert_encoder_postprocess,
    "specialized_encoder": _specialized_encoder_postprocess,
    "jina_bert_v3_encoder": _jina_bert_v3_postprocess,
    "gguf_embedding": _gguf_embedding_postprocess,
    "talkie": _talkie_postprocess,
    "maincoder": _maincoder_postprocess,
    "t5": _t5_postprocess,
    "minimax": _minimax_postprocess,
    "minimax_m2": _minimax_m2_postprocess,
    "mistral4": _mistral4_postprocess,
    "glm_dsa": _glm_dsa_postprocess,
    "kimi_linear": _kimi_linear_postprocess,
    "kimi_k3": _kimi_k3_postprocess,
    "minicpm": _minicpm_postprocess,
    "minicpm3": _minicpm3_postprocess,
    "qwen4exp": _qwen4exp_postprocess,
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
    if model_type in {"gpt2", "starcoder2"}:
        # These architectures use tanh-approximate GELU by default. Their GGUF
        # metadata omits that architecture-owned choice, so exact GELU is not equivalent.
        return "gelu_pytorch_tanh"
    gelu_models = {
        "bert",
        "bloom",
        "gpt_bigcode",
        "jais2",
        "kclgpt",
        "modernbert",
        "t5",
    }
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
    if config.scoring_func != "sigmoid":
        reasons.append(
            "missing SIGMOID expert gate (GGUF '<arch>.expert_gating_func'=2); "
            f"scoring_func={config.scoring_func!r}"
        )
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
