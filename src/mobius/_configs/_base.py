# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import onnx_ir as ir
import torch
from onnx_ir import tensor_adapters

from mobius._configs._quantization import QuantizationConfig
from mobius._configs._sub_configs import (
    AudioConfig,
    CodecDecoderConfig,
    CodecEncoderConfig,
    CodePredictorConfig,
    RoPEConfig,
    SpeakerEncoderConfig,
    TTSConfig,
    VisionConfig,
)

if TYPE_CHECKING:
    from mobius.integrations._block_quant import BlockQuantScheme

DEFAULT_INT = -42


def _resolve_dtype_value(torch_dtype) -> ir.DataType | None:
    """Convert a HuggingFace dtype value to an ONNX IR dtype."""
    if torch_dtype is not None and torch_dtype != "auto":
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype, None)
        if torch_dtype is not None:
            return tensor_adapters.from_torch_dtype(torch_dtype)
    return None


def _resolve_dtype(config) -> ir.DataType | None:
    """Extract model dtype from a HuggingFace config."""
    return _resolve_dtype_value(getattr(config, "dtype", None))


def _resolve_hidden_act(config, model_type: str) -> str | None:
    """Resolve the hidden activation function from HF config patterns.

    Fallback order (first truthy value wins):
      hidden_act            — standard field (most models)
      hidden_activation     — some encoder models
      activation_function   — GPT-2 family
      ff_activation         — XLNet
      dense_act_fn          — some BERT variants
      activation            — generic fallback
      afn                   — older BERT configs
      "silu"  (qwen and chatglm) — Qwen v1 and ChatGLM hardcode silu; no activation attr
      "gelu"  (XLM)         — gelu_activation=True is a boolean flag
      "relu"  (ctrl)        — CTRL hardcodes relu; no hidden_act attr
    """
    return (
        getattr(config, "hidden_act", None)
        or getattr(config, "hidden_activation", None)
        or getattr(config, "activation_function", None)
        or getattr(config, "ff_activation", None)
        or getattr(config, "dense_act_fn", None)
        or getattr(config, "activation", None)
        or getattr(config, "afn", None)
        # LLaDA/OLMo expose the activation as ``activation_type`` (e.g. "silu").
        or getattr(config, "activation_type", None)
        or ("silu" if model_type in ("qwen", "chatglm", "lfm2") else None)
        # gelu_activation is a boolean (XLM) — must be after all string
        # attrs so it cannot override an explicit hidden_act.
        or ("gelu" if getattr(config, "gelu_activation", False) else None)
        or ("relu" if model_type in ("ctrl",) else None)
    )


def _resolve_sliding_window(config) -> int | None:
    """Resolve the effective sliding-window size, honoring HF's enable gate.

    Qwen2/Qwen3 keep a non-null ``sliding_window`` in the config even when the
    window is disabled, signalling activation through the separate
    ``use_sliding_window`` flag (HF's ``__post_init__`` nulls ``sliding_window``
    when it is ``False``). A raw ``config.json`` fallback that bypasses
    ``__post_init__`` would otherwise leak a window onto a model that does not
    use one, so the gate must be re-applied here. ``use_sliding_window`` defaults
    to ``True`` so models without the flag (e.g. Mistral) are unaffected.
    """
    window = getattr(config, "sliding_window", None) or getattr(config, "window_size", None)
    if window is None:
        return None
    if getattr(config, "use_sliding_window", True) is False:
        return None
    return window


def _nested_rope_theta(rope_scaling: dict, key: str) -> float | None:
    """Extract rope_theta from a nested rope_scaling dict (e.g. Gemma3)."""
    sub = rope_scaling.get(key)
    if isinstance(sub, dict):
        return sub.get("rope_theta")
    return None


def _nested_rope_type(rope_scaling: dict, key: str) -> str | None:
    """Extract rope_type from a nested rope_scaling dict (e.g. Gemma3)."""
    sub = rope_scaling.get(key)
    if isinstance(sub, dict):
        return sub.get("rope_type")
    return None


def _normalize_rope_scaling(rope_scaling: dict) -> dict:
    """Flatten nested rope_scaling dicts (e.g. Gemma3).

    Gemma3 stores per-attention-type configs::

        {"full_attention": {"rope_type": "linear", "factor": 8.0, ...},
         "sliding_attention": {"rope_type": "default", ...}}

    This normalizes to the ``full_attention`` sub-dict so downstream
    code (e.g. ``LinearRope``) can find ``rope_scaling["factor"]``.
    """
    if not rope_scaling:
        return rope_scaling
    if "full_attention" in rope_scaling and isinstance(rope_scaling["full_attention"], dict):
        return rope_scaling["full_attention"]
    return rope_scaling


def _first_not_none(*values, default=None):
    """Return the first value that is not None, or *default*."""
    for v in values:
        if v is not None:
            return v
    return default


# Legacy rope_type spellings that name the same algorithm as a canonical
# rope_type understood by ``initialize_rope``. Phi-3/Phi-3.5 checkpoints
# label LongRoPE as ``"su"`` (short/long-factor scaled rotary embeddings);
# newer HuggingFace configs call the identical algorithm ``"longrope"``.
# Both must resolve to the same code path, so we canonicalize the alias at
# extraction time rather than special-casing ``"su"`` downstream.
_ROPE_TYPE_ALIASES: dict[str, str] = {
    "su": "longrope",
}


def _canonical_rope_type(rope_type: str | None) -> str | None:
    """Map legacy rope_type aliases to their canonical spelling.

    Returns the input unchanged when it is not a known alias (including
    ``None`` and ``"default"``).
    """
    if rope_type is None:
        return None
    return _ROPE_TYPE_ALIASES.get(rope_type, rope_type)


def _leading_layer_type_count(layer_types, target: str) -> int:
    count = 0
    for layer_type in layer_types or ():
        if layer_type != target:
            break
        count += 1
    return count


def _deepseek_v4_compress_ratios(config) -> list[int] | None:
    ratios = getattr(config, "compress_ratios", None)
    if ratios is not None:
        return list(ratios)
    layer_types = getattr(config, "layer_types", None)
    rates = getattr(config, "compress_rates", None)
    if not layer_types or not rates:
        return None
    return [
        (
            rates.get("compressed_sparse_attention", 4)
            if layer_type == "compressed_sparse_attention"
            else rates.get("heavily_compressed_attention", 128)
            if layer_type == "heavily_compressed_attention"
            else 0
        )
        for layer_type in layer_types
    ]


# Models that use RoPE but hardcode rope_theta entirely in model __init__,
# not in config JSON.  These are NOT detectable via config introspection
# without trust_remote_code.  When a new model fails L2 with missing RoPE:
# 1. Confirm the model's HF source uses rotary embeddings
# 2. Find the hardcoded rope_theta in the transformers source
# 3. Add an entry here: model_type → rope_theta
# TODO: migrate to a registry annotation (uses_rope: bool, rope_theta: float)
# once the registry schema supports per-model capability flags.
_IMPLICIT_ROPE_DEFAULTS: dict[str, float] = {
    # arctic: config JSON has rope_theta=10000 (default) and rope_scaling=null;
    # no signal triggers _extract_rope_config, but the model uses RoPE.
    "arctic": 10_000.0,
    # chatglm: config JSON has no rope_theta/rope_scaling/rotary_* attrs;
    # uses default rope_theta=10000.0 hardcoded in modeling code.
    "chatglm": 10_000.0,
    # deepseek_vl_v2: config JSON has no rope_theta attr;
    # uses default rope_theta=10000.0 hardcoded in modeling code.
    "deepseek_vl_v2": 10_000.0,
    # jamba: config JSON has no rope_theta/rope_scaling/rotary_* attrs;
    # uses rope_theta=8000.0 hardcoded in modeling code.
    "jamba": 8_000.0,
    # NOTE: qwen3_omni_moe removed — its config JSON contains
    # rope_scaling (with embedded rope_theta=1e6), so
    # _extract_rope_config() already handles it via the rope_scaling path.
}


def _extract_rope_config(config) -> RoPEConfig | None:
    """Extract and normalize RoPE-related config fields.

    Reads ``rope_scaling``, ``rope_parameters``, and related attributes
    from a HuggingFace config and returns a :class:`RoPEConfig`.

    Returns ``None`` when the source config has no RoPE signal at all —
    i.e. it declares neither the modern ``rope_parameters``/``rope_scaling``
    fields, nor the legacy ``rotary_dim``/``rotary_pct``/``rotary_emb_base``
    fields, nor a non-default ``rope_theta``.  This is the "NoPE" case
    (e.g. NemotronH, GraniteMoeHybrid, GPT-2 family, BERT family, OPT) —
    the model does not use rotary position embeddings at all, and callers
    should treat RoPE as absent rather than manufacturing defaults that
    would silently introduce RoPE operations into the ONNX graph.

    RoPE signal detection:
      - ``rope_parameters`` / ``rope_scaling``: authoritative modern signal
        (populated by ``PretrainedConfig.__post_init__``).
      - ``rotary_dim`` / ``rotary_pct`` / ``rotary_emb_base``: legacy
        signal used by GPT-J / GPT-NeoX / CodeGen models that predate
        ``rope_parameters``.
      - ``rope_theta`` with a non-default value (≠ 10000.0): treated as a
        RoPE indicator for models like Arctic and Jamba that set a custom
        ``rope_theta`` without exposing ``rope_scaling``.  The HF default
        of 10000.0 is excluded via ``math.isclose()`` because NoPE models
        (e.g. NemotronH) inherit it as dead config data.
    """
    # Check for RoPE signals BEFORE the `or {}` fallback below —
    # `or {}` converts None to empty dict, destroying the absence signal.
    raw_rope_scaling = getattr(config, "rope_scaling", None)
    raw_rope_parameters = getattr(config, "rope_parameters", None)
    has_legacy_rope = (
        getattr(config, "rotary_dim", None) is not None
        or getattr(config, "rotary_pct", None) is not None
        or getattr(config, "rotary_emb_base", None) is not None
    )
    # rope_theta alone is a weaker signal but still indicates RoPE intent
    # for many models (Arctic, Jamba, etc.) that don't set rope_scaling.
    # Only treat it as a signal when it differs from the common default
    # (10000.0) to avoid false positives on NoPE models that inherit
    # rope_theta as dead config data.
    raw_rope_theta = getattr(config, "rope_theta", None)
    has_nondefault_rope_theta = raw_rope_theta is not None and not math.isclose(
        raw_rope_theta, 10_000.0
    )
    if (
        raw_rope_scaling is None
        and raw_rope_parameters is None
        and not has_legacy_rope
        and not has_nondefault_rope_theta
    ):
        return None

    rope_scaling = raw_rope_scaling or {}
    rope_parameters = raw_rope_parameters or {}

    return RoPEConfig(
        rope_type=_canonical_rope_type(
            _first_not_none(
                rope_scaling.get("rope_type", None),
                rope_scaling.get("type", None),
                rope_parameters.get("rope_type", None),
                _nested_rope_type(rope_scaling, "full_attention"),
                default="default",
            )
        ),
        rope_theta=_first_not_none(
            getattr(config, "rope_theta", None),
            rope_scaling.get("rope_theta", None),
            rope_parameters.get("rope_theta", None),
            _nested_rope_theta(rope_scaling, "full_attention"),
            default=10_000.0,
        ),
        # Some models (e.g. Ministral-3) store YaRN config under
        # rope_parameters instead of rope_scaling; fall back accordingly.
        rope_scaling=(
            _normalize_rope_scaling(rope_scaling)
            or _normalize_rope_scaling(rope_parameters)
            or None
        ),
        partial_rotary_factor=_first_not_none(
            getattr(config, "partial_rotary_factor", None),
            rope_scaling.get("partial_rotary_factor", None),
            rope_parameters.get("partial_rotary_factor", None),
            default=1.0,
        ),
        rope_local_base_freq=_first_not_none(
            getattr(config, "rope_local_base_freq", None),
            _nested_rope_theta(rope_scaling, "sliding_attention"),
        ),
        original_max_position_embeddings=_first_not_none(
            getattr(config, "original_max_position_embeddings", None),
            rope_scaling.get("original_max_position_embeddings", None),
            # Also check rope_parameters (see rope_scaling comment above).
            rope_parameters.get("original_max_position_embeddings", None),
        ),
    )


def _extract_mrope_fields(config) -> dict:
    """Extract MRoPE fields from a HuggingFace config.

    Returns a dict with ``mrope_interleaved`` and ``mrope_section``
    keys (only if present).  These are separate from :class:`RoPEConfig`
    because they affect multimodal position encoding and are shared
    with :class:`VisionConfig`.
    """
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    result: dict = {}
    # Some models (e.g. Qwen3-TTS talker) spell the interleaved flag as the
    # bare ``interleaved`` key inside ``rope_scaling`` rather than the
    # ``mrope_interleaved`` name that Qwen3-VL uses. Accept both spellings.
    mrope_interleaved = (
        rope_scaling.get("mrope_interleaved", False)
        or rope_scaling.get("interleaved", False)
        or rope_parameters.get("mrope_interleaved", False)
        or rope_parameters.get("interleaved", False)
    )
    if mrope_interleaved:
        result["mrope_interleaved"] = True
        section = rope_scaling.get("mrope_section", None) or rope_parameters.get(
            "mrope_section", None
        )
        if section is not None:
            result["mrope_section"] = section
    return result


def _extract_vision_config(config, parent_config, model_type: str) -> dict:
    """Extract vision sub-config from a HuggingFace config.

    Thin shim that delegates to the per-model registry. The actual
    hooks live under :mod:`mobius._configs.per_model` and are
    registered with :mod:`mobius._configs._extractors` at import time.
    """
    from mobius._configs import per_model  # noqa: F401 - imported for registration side effect
    from mobius._configs._extractors import extract_vision_config as _dispatch

    return _dispatch(config, parent_config, model_type)


def _extract_audio_config(config, parent_config, model_type: str) -> dict:
    """Extract audio sub-config from a HuggingFace config.

    Thin shim that delegates to the per-model registry. The actual
    hooks live under :mod:`mobius._configs.per_model` and are
    registered with :mod:`mobius._configs._extractors` at import time.
    """
    from mobius._configs import per_model  # noqa: F401 - imported for registration side effect
    from mobius._configs._extractors import extract_audio_config as _dispatch

    return _dispatch(config, parent_config, model_type)


_COMPONENT_QUANTIZATION_ALIASES = {
    "text": "decoder",
    "vision": "vision_encoder",
    "audio": "audio_encoder",
}


def _get_config_value(config: object, name: str) -> object | None:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _parse_component_quantization_mapping(
    value: object,
    *,
    expert_dtype: object | None,
) -> dict[str, QuantizationConfig]:
    """Parse an explicit component-name to quantization-config mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(
            "component_quantization must be a mapping of component names to "
            f"quantization configs, got {type(value).__name__}"
        )

    result: dict[str, QuantizationConfig] = {}
    for raw_name, raw_config in value.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("component_quantization keys must be non-empty strings")
        name = _COMPONENT_QUANTIZATION_ALIASES.get(raw_name, raw_name)
        quantization = QuantizationConfig.from_value(
            raw_config,
            expert_dtype=expert_dtype,
        )
        if quantization is None:
            continue
        if name in result:
            raise ValueError(
                f"component_quantization declares component {name!r} more than once"
            )
        result[name] = quantization
    return result


def _extract_component_quantization(
    config: object,
    parent_config: object | None,
    decoder_quantization: QuantizationConfig | None,
) -> dict[str, QuantizationConfig] | None:
    """Extract explicit or nested per-component quantization metadata."""
    sources = []
    for source in (parent_config, config):
        if source is not None and all(id(source) != id(item) for item in sources):
            sources.append(source)

    # Explicit mapping is authoritative. Accept a top-level field or a
    # ``components`` mapping nested in the traditional quantization_config.
    for source in sources:
        declaration = _get_config_value(source, "component_quantization")
        if declaration is None:
            declaration = _get_config_value(source, "component_quantization_config")
        if declaration is None:
            root_quantization = _get_config_value(source, "quantization_config")
            if hasattr(root_quantization, "to_dict"):
                root_quantization = root_quantization.to_dict()
            if isinstance(root_quantization, Mapping):
                declaration = root_quantization.get("components")
        if declaration is not None:
            return _parse_component_quantization_mapping(
                declaration,
                expert_dtype=_get_config_value(source, "expert_dtype"),
            )

    # Composite checkpoints may instead put independent quantization_config
    # dictionaries directly on their vision/audio sub-configs.
    composite = parent_config or config
    nested: dict[str, QuantizationConfig] = {}
    found_nested_declaration = False
    for field_name, component_name in (
        ("vision_config", "vision_encoder"),
        ("audio_config", "audio_encoder"),
    ):
        sub_config = _get_config_value(composite, field_name)
        if sub_config is None:
            continue
        raw_quantization = _get_config_value(sub_config, "quantization_config")
        if raw_quantization is None:
            continue
        found_nested_declaration = True
        quantization = QuantizationConfig.from_value(
            raw_quantization,
            expert_dtype=_get_config_value(sub_config, "expert_dtype"),
        )
        if quantization is not None:
            nested[component_name] = quantization

    if not found_nested_declaration:
        return None
    if decoder_quantization is not None:
        nested["decoder"] = dataclasses.replace(decoder_quantization)
        nested["embedding"] = dataclasses.replace(decoder_quantization)
    return nested


@dataclasses.dataclass
class BaseModelConfig:
    """Base configuration shared by all model architectures.

    Contains the minimal set of fields needed by the task/exporter
    infrastructure (KV cache shapes, dtype casting, etc.).
    """

    vocab_size: int = DEFAULT_INT
    hidden_size: int = DEFAULT_INT
    intermediate_size: int = DEFAULT_INT
    num_hidden_layers: int = DEFAULT_INT
    num_attention_heads: int = DEFAULT_INT
    num_key_value_heads: int = DEFAULT_INT
    head_dim: int = DEFAULT_INT
    hidden_act: str | None = None
    pad_token_id: int = DEFAULT_INT
    tie_word_embeddings: bool = False
    attn_qkv_bias: bool = False
    attn_o_bias: bool = False

    # Model dtype (from HF config dtype)
    dtype: ir.DataType = ir.DataType.FLOAT
    quantization: QuantizationConfig | None = None
    # ``None`` preserves the legacy model-wide quantization behavior. A mapping
    # is authoritative: omitted components remain floating point.
    component_quantization: dict[str, QuantizationConfig] | None = None

    # HuggingFace identity and token metadata used by package persistence.
    model_type: str | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    mask_token_id: int | None = None
    diffusion_shift_logits: bool = False

    def quantization_for(self, component: str) -> QuantizationConfig | None:
        """Return the effective quantization config for one package component."""
        if self.component_quantization is None:
            return self.quantization
        candidates = {
            "decoder": ("decoder", "model"),
            "model": ("model", "decoder"),
            "vision_encoder": ("vision_encoder", "vision"),
            "vision": ("vision", "vision_encoder"),
            "audio_encoder": ("audio_encoder", "audio"),
            "audio": ("audio", "audio_encoder"),
        }.get(component, (component,))
        return next(
            (
                self.component_quantization[name]
                for name in candidates
                if name in self.component_quantization
            ),
            None,
        )

    def quantization_for_source_paths(
        self,
        component: str,
        source_paths: tuple[str, ...],
        *,
        ignored_source_names: tuple[str, ...] = ("lm_head", "embed_tokens"),
    ) -> QuantizationConfig | None:
        """Resolve module-level rules using one component's Hugging Face paths."""
        quantization = self.quantization_for(component)
        if quantization is None or not quantization.has_module_plan:
            return quantization
        effective_paths = tuple(
            path
            for path in source_paths
            if path.rsplit(".", 1)[-1] not in ignored_source_names
        )
        return quantization.for_source_paths(
            effective_paths or source_paths,
            component=component,
        )


@dataclasses.dataclass
class ArchitectureConfig(BaseModelConfig):
    """Configuration for decoder-only model architectures."""

    max_position_embeddings: int = DEFAULT_INT

    # attention config
    layer_types: list[str] | None = None
    no_rope_layers: list[int] | None = None
    full_attention_interval: int | None = None
    sliding_window: int | None = None

    # Linear attention (DeltaNet) config
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_gate_lower_bound: float | None = None
    linear_use_full_rank_gate: bool = False

    # Double-gated short-convolution config (LFM2-style hybrid layers).
    short_conv_kernel: int = 3
    short_conv_bias: bool = False

    rms_norm_eps: float = 1e-6

    # Rotary embedding config.
    #
    # ``rope_type`` is the structural signal: ``None`` means "this model
    # does not use RoPE". ``from_transformers`` populates ``rope_type``
    # (and the other flat RoPE fields below) from the HuggingFace config
    # only when RoPE is actually declared — see :func:`_extract_rope_config`.
    # For NoPE models (NemotronH, GraniteMoeHybrid, GPT-2 family, BERT, OPT,
    # ...) ``from_transformers`` sets every flat RoPE field to ``None`` so
    # downstream code (``initialize_rope``, ``TextModel``, ``Attention``) can
    # structurally detect the absence of RoPE instead of spuriously applying
    # a "default" rotary encoding.
    #
    # The non-``rope_type`` fields keep inert numeric defaults at the
    # dataclass level so that code that constructs ``ArchitectureConfig``
    # directly with just ``rope_type="default"`` (e.g. tests, small reproducer
    # configs) works without having to spell out every RoPE parameter. These
    # defaults are only consumed when ``rope_type`` is non-``None``.
    rope_type: str | None = None
    rope_theta: float | None = 10_000.0
    rope_scaling: dict | None = None
    partial_rotary_factor: float | None = 1.0
    rope_local_base_freq: float | None = None
    original_max_position_embeddings: int | None = None

    attn_qk_norm: bool = False
    attn_qk_norm_full: bool = False
    attn_q_norm_biases: tuple[bool, ...] | None = None
    attn_k_norm_biases: tuple[bool, ...] | None = None
    mlp_bias: bool = False
    attention_scale: float | None = None
    attention_clamp: float | None = None
    alibi_max_bias: float | None = None
    use_parallel_residual: bool = False
    layer_intermediate_sizes: tuple[int, ...] = ()
    layer_attention_head_counts: tuple[int, ...] = ()
    layer_attention_kv_head_counts: tuple[int, ...] = ()

    # Apertus xIELU parameters. GGUF stores these learned per-layer scalars as
    # metadata rather than tensors, so they must survive config replacement.
    xielu_alpha_p: tuple[float, ...] | None = None
    xielu_alpha_n: tuple[float, ...] | None = None
    xielu_beta: tuple[float, ...] | None = None
    xielu_eps: tuple[float, ...] | None = None

    # Encoder-specific config
    type_vocab_size: int = 0
    encoder_use_token_type_embeddings: bool = False
    encoder_q_bias: bool = False
    encoder_k_bias: bool = False
    encoder_v_bias: bool = False
    encoder_ffn_up_bias: bool = False
    encoder_ffn_down_bias: bool = False
    encoder_qk_norm: bool = False
    encoder_extra_attention_norm: bool = False
    encoder_fused_geglu: bool = False
    encoder_fused_qkv: bool = False
    pooling_type: int = 0
    embedding_dense_2_out: int | None = None
    embedding_dense_3_in: int | None = None

    # Encoder-decoder config
    num_decoder_layers: int | None = None
    decoder_start_token_id: int | None = None
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    encoder_relative_attention_bias_layers: list[int] | None = None
    decoder_relative_attention_bias_layers: list[int] | None = None
    is_gated_act: bool = False
    scale_decoder_outputs: bool | None = None

    # MoE config
    num_local_experts: int | None = None
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    norm_topk_prob: bool = True
    # When True, the decoder layer uses post-norm style (FlexOLMo): norms are applied
    # to sub-layer outputs instead of inputs, with an extra post_feedforward_layernorm.
    post_feedforward_norm: bool = False
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    routing_weight_normalization_floor: float | None = None
    scoring_func: str = "softmax"
    topk_method: str = "greedy"
    first_k_dense_replace: int = 0
    moe_layer_frequency: int = 0
    n_shared_experts: int | None = None
    use_expert_bias: bool | None = None
    disable_qmoe: bool = False

    # MiniMax-01 hybrid attention and normalized-residual scaling.
    lightning_norm_eps: float | None = None
    full_attn_alpha_factor: float = 1.0
    full_attn_beta_factor: float = 1.0
    linear_attn_alpha_factor: float = 1.0
    linear_attn_beta_factor: float = 1.0
    mlp_alpha_factor: float = 1.0
    mlp_beta_factor: float = 1.0

    # Multi-head Latent Attention (MLA) config — DeepSeek-V2/V3
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None
    rope_interleave: bool = False
    mla_use_output_gate: bool = False

    # Kimi-K3 attention-residual and latent-expert configuration.
    attn_res_block_size: int | None = None
    routed_expert_hidden_size: int | None = None
    latent_moe_use_norm: bool = False
    activation_situ_beta: float = 1.0
    activation_situ_linear_beta: float | None = None

    # DeepSeek-V4 compressed sparse attention / Hyper-Connections.
    o_groups: int = 1
    o_lora_rank: int | None = None
    index_n_heads: int | None = None
    index_head_dim: int | None = None
    index_topk: int | None = None
    compress_ratios: list[int] | None = None
    compress_rope_theta: float | None = None
    hc_mult: int = 1
    hc_sinkhorn_iters: int = 1
    hc_eps: float = 1e-6
    num_hash_layers: int = 0
    swiglu_limit: float = 0.0
    num_nextn_predict_layers: int = 0
    # DeepSeek-V4 native compressed-sparse-attention export toggle. Not an
    # upstream HF field -- a Mobius export-time opt-in (default False) that
    # replaces the dense CSA/HCA correctness fallback with the frozen
    # ``pkg.nxrt::CompressedSparseAttention`` v1 op for property-matching
    # ratio-128 (HCA) layers. Off by default so every shipped graph stays
    # byte-identical and ``pkg.nxrt``-free unless explicitly requested via
    # ``config_overrides={"native_csa": True}``. When requested, layers that
    # do not satisfy the frozen op contract raise a typed export error rather
    # than silently emitting the dense fallback (see
    # ``mobius.models._deepseek_v4_csa``).
    native_csa: bool = False

    # GLM-5.2 (``glm_moe_dsa``) DeepSeek Sparse Attention (DSA).
    #
    # ``use_dsa`` is not an upstream HF field -- it is a Mobius export-time
    # toggle (default True) flipped to False by ``--glm-full-attention`` /
    # ``config_overrides={"use_dsa": False}`` to fall back to plain dense MLA
    # (reusing ``DeepSeekV3TextModel`` unchanged) on runtimes that cannot yet
    # execute ``pkg.nxrt::IndexShare``.
    use_dsa: bool = True
    # Opt-in export of ``com.microsoft::PagedAttention`` (LATENT / absorbed-MLA
    # mode) for property-compatible dense MLA. Default off. When off, exports are
    # byte-identical to the current dense-MLA graph. Eligibility is decided from
    # semantic geometry (see ``mobius.components._paged_mla``), never model names;
    # an incompatible geometry raises rather than silently falling back.
    export_paged_attention: bool = False
    # Per-layer indexer schedule ("full" runs the indexer; "shared" reuses the
    # top-k selection from the closest preceding "full" layer). When the
    # checkpoint config omits this list, it is derived from
    # ``index_topk_freq`` / ``index_skip_topk_offset`` -- see
    # ``mobius.models.glm_moe_dsa._indexer_types``, which mirrors HF
    # ``GlmMoeDsaConfig.__post_init__`` exactly.
    indexer_types: list[str] | None = None
    index_topk_freq: int | None = None
    index_skip_topk_offset: int | None = None
    indexer_rope_interleave: bool = False
    # Whether the (currently unexported, see ``glm_moe_dsa.py``) MTP layer is
    # meant to reuse the target's shared top-k indices rather than running its
    # own indexer pass. Recorded for round-tripping/documentation; not yet
    # acted on since MTP export itself is out of scope this cycle.
    index_share_for_mtp_iteration: bool = False

    # Vision shared fields (accessed as top-level config.X by tasks)
    mm_tokens_per_image: int | None = None
    image_token_id: int | None = None
    video_token_id: int | None = None
    vision_start_token_id: int | None = None
    vision_end_token_id: int | None = None
    downsample_mode: str = "16x"
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    frame_windows_size: int = 4
    tokens_per_second: float = 1.0
    deepstack_visual_indexes: list[int] | None = None
    fullatt_block_indexes: list[int] | None = None
    window_size: int = 112

    # Q-Former config (for BLIP-2 style models)
    num_query_tokens: int | None = None
    qformer_hidden_size: int | None = None
    qformer_num_hidden_layers: int | None = None
    qformer_num_attention_heads: int | None = None
    qformer_intermediate_size: int | None = None

    # MRoPE config (for multimodal position encoding)
    mrope_section: list[int] | None = None
    mrope_interleaved: bool = False

    # Standalone vision config
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3

    # Audio config (for multimodal models like Phi4-MM)
    audio_attention_dim: int | None = None
    audio_attention_heads: int | None = None
    audio_num_blocks: int | None = None
    audio_linear_units: int | None = None
    audio_kernel_size: int | None = None
    audio_input_size: int | None = None
    audio_conv_channels: int | None = None
    audio_t5_bias_max_distance: int | None = None
    audio_token_id: int | None = None

    # FastConformer-RNNT (NeMo) config — see models/nemo_rnnt.py.
    # Encoder
    fastconformer_subsampling_factor: int = 8
    fastconformer_subsampling_conv_channels: int = 256
    fastconformer_conv_kernel_size: int = 9
    fastconformer_pos_emb_max_len: int = 5000
    fastconformer_xscaling: bool = False
    # Number of input mel features to the encoder.
    fastconformer_feat_in: int = 128
    # Chunked-limited attention context: [left_context, right_context] in frames.
    fastconformer_att_context_size: tuple[int, int] = (70, 13)
    # Cache-aware streaming: per-layer last-channel (attention) cache length in
    # subsampled frames (NeMo ``last_channel_cache_size`` = att_context left) and
    # the number of leading subsampled frames dropped per chunk (NeMo
    # ``drop_extra_pre_encoded``).
    fastconformer_streaming_cache_size: int = 70
    fastconformer_streaming_drop_extra: int = 2
    # RNN-T prediction network + joint
    rnnt_pred_hidden: int | None = None
    rnnt_pred_rnn_layers: int = 1
    rnnt_joint_hidden: int | None = None
    # Number of acoustic classes excluding the blank symbol (vocab without blank).
    rnnt_num_classes: int | None = None

    # LoRA config (for multimodal models like Phi4-MM)
    speech_lora: dict | None = None

    # Phi4MM image embedding config
    image_crop_size: int | None = None

    # Falcon config
    alibi: bool = False
    parallel_attn: bool = False
    # True for models with two separate norms in parallel layers (MPT, GPT-NeoX-Falcon)
    dual_ln: bool = False

    # Post-norm vs pre-norm architecture toggle (used by OpenAI-GPT vs standard GPT-2)
    post_norm: bool = False

    # Granite scaling multipliers
    embedding_multiplier: float = 1.0
    attention_multiplier: float | None = None
    logits_scaling: float = 1.0
    residual_multiplier: float = 1.0

    # Cohere logit scale: multiplied into the final logits before softmax
    logit_scale: float = 1.0

    # YOLOS object detection config
    num_labels: int = 91

    # Composed sub-configs
    rope: RoPEConfig | None = None
    vision: VisionConfig | None = None
    audio: AudioConfig | None = None
    tts: TTSConfig | None = None
    codec_decoder: CodecDecoderConfig | None = None
    codec_encoder: CodecEncoderConfig | None = None
    quantization: QuantizationConfig | None = None
    # Deferred block-scaled FP8 / packed-FP4 scheme (DeepSeek-V4-Flash native
    # CSA). Recorded by ``from_transformers`` when ``native_csa`` opts into
    # deferring #602's config-resolution block-quant reject so that graph
    # construction can progress past the former generic weight-shape mismatch.
    # The runtime-capability gate
    # (``mobius.models._deepseek_v4_csa.assert_native_runtime_supports_block_quant``)
    # then fails closed on the *runnable* full export until nxrt advertises real
    # block-FP8 / planar-FP4 format strings. ``None`` for every ordinary,
    # per-tensor-fp8, or non-native path.
    block_quant_scheme: BlockQuantScheme | None = None
    # HuggingFace model_type and special token IDs — populated by from_transformers()
    # so that genai_config.json can be written without re-fetching the HF config.
    model_type: str | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    # Speculative-decoding draft-target support.
    # When set to a non-empty list, ``TextModel`` captures the post-residual
    # output of ``self.layers[k]`` for each ``k`` in the list (before the
    # final norm), and ``CausalLMTask`` registers them as additional ONNX
    # outputs named ``hidden_states.{k}``.  Indices follow the HF
    # convention used by drafters such as DFlash: ``k`` refers to the
    # 0-based decoder layer whose output you want — equivalent to
    # ``model(..., output_hidden_states=True).hidden_states[k + 1]`` in
    # transformers (where index 0 is the embedding output).
    output_layer_indices: list[int] | None = None
    # Emit the post-final-norm state as ``mtp_seed``. This is separate from
    # output_layer_indices, whose hidden_states.N outputs remain pre-final-norm.
    output_final_hidden_state: bool = False

    @classmethod
    def from_transformers(
        cls,
        config,
        parent_config=None,
        *,
        allow_block_fp8_dense_fallback: bool = False,
    ) -> ArchitectureConfig:
        model_type = config.model_type
        rope_config = _extract_rope_config(config)
        mrope_fields = _extract_mrope_fields(config)

        # Models that use RoPE but don't expose rope_scaling/rope_parameters
        # in their HF config (e.g. loaded without trust_remote_code, or
        # because the HF code hardcodes defaults internally).
        if rope_config is None and model_type in _IMPLICIT_ROPE_DEFAULTS:
            rope_config = RoPEConfig(
                rope_type="default",
                rope_theta=_IMPLICIT_ROPE_DEFAULTS[model_type],
                partial_rotary_factor=0.5 if model_type == "chatglm" else 1.0,
            )

        # Some hierarchical models (Segformer, Swin) use plural list attrs
        # instead of scalar ones.  Resolve to a scalar for the base config.
        hidden_size = (
            getattr(config, "hidden_size", None)
            or getattr(config, "d_model", None)
            or _first(getattr(config, "hidden_sizes", None))
            or 0
        )
        num_attention_heads = (
            getattr(config, "num_attention_heads", None)
            or getattr(config, "n_heads", None)
            or _first(getattr(config, "num_heads", None))
            or 1
        )
        num_hidden_layers = (
            getattr(config, "num_hidden_layers", None)
            or getattr(config, "n_layers", None)
            or getattr(config, "num_layers", None)
            or getattr(config, "num_encoder_blocks", None)
            or 0
        )

        # rope_interleave depends on model_type / qk_rope_head_dim.
        # Only compute it when RoPE is actually in use — for NoPE models
        # (rope_config is None) we leave the flat ``rope_interleave`` at
        # its inert ``False`` default.
        rope_interleave = getattr(
            config,
            "rope_interleave",
            (getattr(config, "qk_rope_head_dim", None) or 0) > 0
            or model_type in ("glm", "glm4", "glm4_moe", "glm_ocr_text", "chatglm"),
        )
        if rope_config is not None:
            rope_config = dataclasses.replace(rope_config, rope_interleave=rope_interleave)

        per_layer_config = getattr(config, "per_layer_config", None)
        layer_configs = []
        layer_types = getattr(config, "layer_types", None)
        if per_layer_config:
            layer_configs = list(
                per_layer_config.values()
                if isinstance(per_layer_config, dict)
                else per_layer_config
            )

        def _per_layer_value(attribute: str) -> int | None:
            values = set()
            values_by_layer_type: dict[str, set[int]] = {}
            for index, layer_config in enumerate(layer_configs):
                value = (
                    layer_config.get(attribute)
                    if isinstance(layer_config, dict)
                    else getattr(layer_config, attribute, None)
                )
                if value is None:
                    continue
                values.add(value)
                if layer_types and len(layer_types) == len(layer_configs):
                    values_by_layer_type.setdefault(layer_types[index], set()).add(value)
            if not values:
                return None
            if len(values) == 1:
                return next(iter(values))
            supported_layer_types = {"sliding_attention", "full_attention"}
            if (
                model_type not in {"gemma4_text", "gemma4_unified_text"}
                or set(values_by_layer_type) != supported_layer_types
                or any(len(type_values) != 1 for type_values in values_by_layer_type.values())
            ):
                raise ValueError(
                    f"Mobius does not support these heterogeneous per-layer {attribute} "
                    f"values: {sorted(values)}."
                )
            return next(iter(values_by_layer_type["sliding_attention"]))

        head_dim = _per_layer_value("head_dim")
        if head_dim is None:
            head_dim = getattr(config, "head_dim", None)
        num_key_value_heads = _per_layer_value("num_key_value_heads")
        if num_key_value_heads is None:
            num_key_value_heads = getattr(config, "num_key_value_heads", None)

        options = dict(
            head_dim=(
                head_dim
                if head_dim is not None
                else getattr(config, "d_kv", None)
                or getattr(config, "kv_channels", None)
                or _as_int(hidden_size) // _as_int(num_attention_heads)
            ),
            num_attention_heads=_as_int(num_attention_heads),
            num_key_value_heads=_as_int(
                num_key_value_heads
                or getattr(config, "n_kv_heads", None)
                or (
                    getattr(config, "multi_query_group_num", None)
                    if getattr(config, "multi_query_attention", False)
                    else None
                )
                or num_attention_heads
            ),
            num_hidden_layers=_as_int(num_hidden_layers),
            vocab_size=(
                getattr(config, "vocab_size", None)
                or getattr(config, "embedding_size", None)
                or 0
            ),
            hidden_size=_as_int(hidden_size),
            intermediate_size=_as_int(
                getattr(config, "intermediate_size", None)
                or getattr(config, "mlp_hidden_size", None)
                or getattr(config, "n_inner", None)
                or getattr(config, "d_ff", None)
                or getattr(config, "ffn_dim", None)
                or getattr(config, "ffn_hidden_size", None)
                or getattr(config, "encoder_ffn_dim", None)
                or getattr(config, "decoder_ffn_dim", None)
                or 4 * _as_int(hidden_size)
            ),
            hidden_act=_resolve_hidden_act(config, model_type),
            layer_types=(
                getattr(config, "layer_types", None)
                or getattr(config, "attention_layers", None)
            ),
            no_rope_layers=getattr(config, "no_rope_layers", None),
            full_attention_interval=(getattr(config, "full_attention_interval", None)),
            sliding_window=_resolve_sliding_window(config),
            # Linear attention (DeltaNet) parameters
            linear_conv_kernel_dim=(getattr(config, "linear_conv_kernel_dim", 4)),
            linear_key_head_dim=(getattr(config, "linear_key_head_dim", None)),
            linear_value_head_dim=(getattr(config, "linear_value_head_dim", None)),
            linear_num_key_heads=(getattr(config, "linear_num_key_heads", None)),
            linear_num_value_heads=(getattr(config, "linear_num_value_heads", None)),
            short_conv_kernel=getattr(config, "conv_L_cache", 3),
            short_conv_bias=getattr(config, "conv_bias", False),
            pad_token_id=(getattr(config, "pad_token_id", 0)),
            model_type=model_type,
            bos_token_id=getattr(config, "bos_token_id", None),
            eos_token_id=getattr(config, "eos_token_id", None),
            mask_token_id=getattr(config, "mask_token_id", None),
            diffusion_shift_logits=getattr(config, "diffusion_shift_logits", False),
            video_token_id=getattr(
                parent_config or config,
                "video_token_id",
                getattr(config, "video_token_id", None),
            ),
            rms_norm_eps=(
                getattr(config, "rms_norm_eps", None)
                or getattr(config, "layer_norm_eps", None)
                or getattr(config, "layer_norm_epsilon", None)
                or getattr(config, "norm_epsilon", None)
                or getattr(config, "norm_eps", None)
                or 1e-6
            ),
            attn_qkv_bias=(
                getattr(config, "add_qkv_bias", False)
                or getattr(config, "add_bias_linear", False)
                or getattr(
                    config,
                    "attention_bias",
                    getattr(
                        config,
                        "enable_bias",
                        getattr(
                            config,
                            "bias",
                            getattr(
                                config,
                                "use_qkv_bias",
                                getattr(
                                    config,
                                    "use_bias",
                                    model_type
                                    in (
                                        "gpt2",
                                        "gpt_bigcode",
                                        "openai-gpt",
                                        "phi",
                                        "bloom",
                                        "qwen2",
                                        "qwen2_5_vl_text",
                                        "qwen2_moe",
                                        "qwen2_vl_text",
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            ),
            attn_o_bias=(
                getattr(config, "add_bias_linear", False)
                or getattr(
                    config,
                    "attention_bias",
                    getattr(
                        config,
                        "enable_bias",
                        getattr(
                            config,
                            "bias",
                            getattr(
                                config,
                                "use_bias",
                                model_type
                                in (
                                    "gpt2",
                                    "gpt_bigcode",
                                    "gpt_neo",
                                    "openai-gpt",
                                    "phi",
                                    "bloom",
                                ),
                            ),
                        ),
                    ),
                )
            ),
            attn_qk_norm=(
                model_type
                in (
                    "gemma3_text",
                    "gemma3n",
                    "gemma3n_text",
                    "lfm2",
                    "flex_olmo",
                    "olmoe",
                    "olmo2",
                    "olmo3",
                    "qwen3",
                    "qwen3_moe",
                    "qwen3_tts_talker",
                    "qwen3_5_vl",
                    "qwen3_vl",
                    "qwen3_vl_text",
                )
                or getattr(config, "use_qk_norm", False)
            ),
            attn_qk_norm_full=(model_type in ("flex_olmo", "olmoe", "olmo2", "olmo3")),
            mlp_bias=(
                getattr(config, "add_bias_linear", False)
                or getattr(
                    config,
                    "use_mlp_bias",
                    getattr(
                        config,
                        "use_bias",
                        model_type
                        in (
                            "gpt_neo",
                            "gpt_bigcode",
                            "gpt_neox",
                            "gpt_neox_japanese",
                            "openai-gpt",
                            "phi",
                        ),
                    ),
                )
            ),
            rope=rope_config,
            # Flat rope field copies: ``None`` for NoPE models so that
            # ``initialize_rope`` / ``TextModel`` / ``Attention`` can detect
            # "this model has no RoPE" structurally.
            rope_type=rope_config.rope_type if rope_config is not None else None,
            rope_theta=rope_config.rope_theta if rope_config is not None else None,
            rope_scaling=rope_config.rope_scaling if rope_config is not None else None,
            partial_rotary_factor=(
                rope_config.partial_rotary_factor if rope_config is not None else None
            ),
            rope_local_base_freq=(
                rope_config.rope_local_base_freq if rope_config is not None else None
            ),
            original_max_position_embeddings=(
                rope_config.original_max_position_embeddings
                if rope_config is not None
                else None
            ),
            rope_interleave=(
                rope_config.rope_interleave if rope_config is not None else False
            ),
            **mrope_fields,
            max_position_embeddings=(
                getattr(config, "max_position_embeddings", None)
                or getattr(config, "max_sequence_length", None)
                or getattr(config, "seq_length", None)
                or 0
            ),
            tie_word_embeddings=(
                getattr(config, "tie_word_embeddings", None)
                if getattr(config, "tie_word_embeddings", None) is not None
                else getattr(config, "weight_tying", None)
                if getattr(config, "weight_tying", None) is not None
                else getattr(parent_config, "tie_word_embeddings", False)
            ),
            # MoE
            num_local_experts=(
                _first(getattr(config, "num_local_experts", None))
                or _first(getattr(config, "num_experts", None))
                or _first(getattr(config, "n_routed_experts", None))
                or _first(getattr(config, "moe_num_experts", None))
            ),
            num_experts_per_tok=(
                _first(getattr(config, "num_experts_per_tok", None))
                or _first(getattr(config, "moe_k", None))
                or _first(getattr(config, "moe_topk", None))
            ),
            moe_intermediate_size=_first(getattr(config, "moe_intermediate_size", None)),
            shared_expert_intermediate_size=(
                getattr(config, "shared_expert_intermediate_size", None)
                or _shared_expert_size(config)
            ),
            norm_topk_prob=(getattr(config, "norm_topk_prob", True)),
            post_feedforward_norm=(model_type in ("flex_olmo",)),
            n_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
            # ``scoring_func``/``topk_method`` are not real HF dataclass
            # fields on ``GlmMoeDsaConfig`` -- transformers hardcodes
            # sigmoid + e_score_correction_bias + (degenerate, n_group=1)
            # top-k routing for GLM-5.2 rather than exposing it as a config
            # knob. A checkpoint that omits these keys entirely (unlike the
            # real zai-org/GLM-5.2 config.json, which sets them explicitly)
            # would otherwise silently default to the wrong (softmax/greedy)
            # DeepSeek-V2 routing.
            scoring_func=getattr(
                config, "scoring_func", "sigmoid" if model_type == "glm_moe_dsa" else "softmax"
            ),
            topk_method=getattr(
                config, "topk_method", "noaux_tc" if model_type == "glm_moe_dsa" else "greedy"
            ),
            first_k_dense_replace=getattr(config, "first_k_dense_replace", 0),
            n_shared_experts=getattr(config, "n_shared_experts", None),
            # Multi-head Latent Attention (MLA)
            q_lora_rank=getattr(config, "q_lora_rank", None),
            kv_lora_rank=getattr(config, "kv_lora_rank", None),
            qk_nope_head_dim=getattr(config, "qk_nope_head_dim", None),
            qk_rope_head_dim=getattr(config, "qk_rope_head_dim", None),
            v_head_dim=getattr(config, "v_head_dim", None),
            # DeepSeek-V4 compressed sparse attention / Hyper-Connections
            o_groups=getattr(config, "o_groups", 1),
            o_lora_rank=getattr(config, "o_lora_rank", None),
            index_n_heads=getattr(config, "index_n_heads", None),
            index_head_dim=getattr(config, "index_head_dim", None),
            index_topk=getattr(config, "index_topk", None),
            compress_ratios=_deepseek_v4_compress_ratios(config),
            compress_rope_theta=getattr(config, "compress_rope_theta", None),
            hc_mult=getattr(config, "hc_mult", 1),
            hc_sinkhorn_iters=getattr(config, "hc_sinkhorn_iters", 1),
            hc_eps=getattr(config, "hc_eps", 1e-6),
            num_hash_layers=(
                getattr(config, "num_hash_layers", None)
                or getattr(config, "n_hash_layers", 0)
                or _leading_layer_type_count(
                    getattr(config, "mlp_layer_types", None), "hash_moe"
                )
            ),
            swiglu_limit=getattr(config, "swiglu_limit", 0.0),
            num_nextn_predict_layers=getattr(config, "num_nextn_predict_layers", 0),
            # DeepSeek-V4 native CSA/HCA export opt-in. No HF equivalent --
            # every checkpoint defaults to the dense correctness fallback;
            # ``config_overrides={"native_csa": True}`` opts a build into the
            # frozen ``pkg.nxrt::CompressedSparseAttention`` v1 ratio-128 path.
            native_csa=getattr(config, "native_csa", False),
            # GLM-5.2 DSA. ``use_dsa`` has no HF equivalent -- every checkpoint
            # defaults to the sparse path; ``--glm-full-attention`` overrides
            # it afterwards via ``dataclasses.replace``/``config_overrides``.
            use_dsa=getattr(config, "use_dsa", True),
            indexer_types=(
                list(config.indexer_types)
                if getattr(config, "indexer_types", None) is not None
                else None
            ),
            index_topk_freq=getattr(config, "index_topk_freq", None),
            index_skip_topk_offset=getattr(config, "index_skip_topk_offset", None),
            indexer_rope_interleave=getattr(config, "indexer_rope_interleave", False),
            index_share_for_mtp_iteration=getattr(
                config, "index_share_for_mtp_iteration", False
            ),
            # Encoder-specific
            type_vocab_size=getattr(config, "type_vocab_size", 0),
            # Encoder-decoder
            num_decoder_layers=(
                getattr(config, "num_decoder_layers", None)
                or getattr(config, "decoder_layers", None)
            ),
            decoder_start_token_id=getattr(config, "decoder_start_token_id", None),
            relative_attention_num_buckets=getattr(
                config, "relative_attention_num_buckets", 32
            ),
            relative_attention_max_distance=getattr(
                config, "relative_attention_max_distance", 128
            ),
            encoder_relative_attention_bias_layers=getattr(
                config, "encoder_relative_attention_bias_layers", None
            ),
            decoder_relative_attention_bias_layers=getattr(
                config, "decoder_relative_attention_bias_layers", None
            ),
            is_gated_act=getattr(config, "is_gated_act", False),
            scale_decoder_outputs=getattr(config, "scale_decoder_outputs", None),
            # Standalone vision (coerce list to int — some HF configs use [H, W])
            image_size=_as_int(getattr(config, "image_size", 224)),
            patch_size=_as_int(getattr(config, "patch_size", 16)),
            num_channels=getattr(config, "num_channels", 3),
            # OpenAI-GPT uses post-norm (no final LayerNorm); GPT-2 uses pre-norm.
            post_norm=model_type == "openai-gpt",
            # Granite scaling multipliers
            embedding_multiplier=getattr(config, "embedding_multiplier", 1.0),
            attention_multiplier=(
                getattr(config, "attention_multiplier", None)
                # GPT-Neo computes Q @ K^T without 1/sqrt(head_dim) scaling
                or (1.0 if model_type == "gpt_neo" else None)
            ),
            logits_scaling=getattr(config, "logits_scaling", 1.0),
            residual_multiplier=getattr(config, "residual_multiplier", 1.0),
            # Cohere logit scale
            logit_scale=getattr(config, "logit_scale", 1.0),
            # Falcon config
            alibi=getattr(config, "alibi", False),
            parallel_attn=getattr(config, "parallel_attn", False),
        )

        # Falcon/Bloom model-specific overrides
        if model_type == "falcon":
            # Falcon MQA: multi_query=True with old architecture → 1 KV head
            if getattr(config, "multi_query", False) and not getattr(
                config, "new_decoder_architecture", False
            ):
                options["num_key_value_heads"] = 1
            # Falcon uses config.bias for both attention and MLP
            options["mlp_bias"] = getattr(config, "bias", False)
        elif model_type == "bloom":
            options["alibi"] = True
            options["mlp_bias"] = True

        # Convert rotary_dim to partial_rotary_factor (GPT-J, CodeGen, etc.)
        rotary_dim = getattr(config, "rotary_dim", None)
        if rotary_dim is not None and options["head_dim"] > 0:
            options["partial_rotary_factor"] = rotary_dim / options["head_dim"]
            rope_config = dataclasses.replace(
                rope_config,
                partial_rotary_factor=options["partial_rotary_factor"],
            )
            options["rope"] = rope_config

        # Compute layer_types from full_attention_interval if not provided
        if options.get("layer_types") is None:
            full_attention_interval = options.get("full_attention_interval")
            if full_attention_interval is not None:
                num_hidden_layers = options["num_hidden_layers"]
                layer_types = []
                for i in range(num_hidden_layers):
                    if (i + 1) % full_attention_interval == 0:
                        layer_types.append("full_attention")
                    else:
                        layer_types.append("linear_attention")
                options["layer_types"] = layer_types

        # Vision config (from multimodal models)
        options.update(_extract_vision_config(config, parent_config, model_type))
        if getattr(parent_config, "model_type", None) == "mage_vl":
            options["model_type"] = "mage_vl"

        # Audio config
        options.update(_extract_audio_config(config, parent_config, model_type))

        # Q-Former config (for BLIP-2 style models)
        qformer_source = parent_config or config
        hf_qformer_config = getattr(qformer_source, "qformer_config", None)
        if hf_qformer_config is not None:
            qc = (
                hf_qformer_config
                if not isinstance(hf_qformer_config, dict)
                else type("QC", (), hf_qformer_config)()
            )
            options["num_query_tokens"] = getattr(qformer_source, "num_query_tokens", 32)
            options["qformer_hidden_size"] = getattr(qc, "hidden_size", 768)
            options["qformer_num_hidden_layers"] = getattr(qc, "num_hidden_layers", 12)
            options["qformer_num_attention_heads"] = getattr(qc, "num_attention_heads", 12)
            options["qformer_intermediate_size"] = getattr(qc, "intermediate_size", 3072)

        # TTS config (Qwen3-TTS talker + code predictor + speaker encoder)
        tts_source = parent_config or config
        talker_cfg = getattr(tts_source, "talker_config", None)
        if talker_cfg is not None:
            tc = (
                talker_cfg
                if not isinstance(talker_cfg, dict)
                else type("TC", (), talker_cfg)()
            )
            tts_fields: dict = {}
            tts_fields["text_hidden_size"] = getattr(tc, "text_hidden_size", 2048)
            tts_fields["text_vocab_size"] = getattr(tc, "text_vocab_size", 151936)
            tts_fields["num_code_groups"] = getattr(tc, "num_code_groups", 16)
            tts_fields["codec_bos_id"] = getattr(tc, "codec_bos_id", 2149)
            tts_fields["codec_eos_token_id"] = getattr(tc, "codec_eos_token_id", 2150)
            tts_fields["codec_pad_id"] = getattr(tc, "codec_pad_id", 2148)
            tts_fields["codec_think_id"] = getattr(tc, "codec_think_id", 2154)
            tts_fields["codec_nothink_id"] = getattr(tc, "codec_nothink_id", 2155)

            # Code predictor config
            cp_cfg = getattr(tc, "code_predictor_config", None)
            if cp_cfg is not None:
                cp = cp_cfg if not isinstance(cp_cfg, dict) else type("CP", (), cp_cfg)()
                tts_fields["code_predictor"] = CodePredictorConfig(
                    hidden_size=getattr(cp, "hidden_size", 1024),
                    intermediate_size=getattr(cp, "intermediate_size", 3072),
                    num_hidden_layers=getattr(cp, "num_hidden_layers", 5),
                    num_attention_heads=getattr(cp, "num_attention_heads", 16),
                    num_key_value_heads=getattr(cp, "num_key_value_heads", 8),
                    head_dim=getattr(cp, "head_dim", 128),
                    vocab_size=getattr(cp, "vocab_size", 2048),
                    num_code_groups=getattr(cp, "num_code_groups", 16),
                    rms_norm_eps=getattr(cp, "rms_norm_eps", 1e-6),
                    rope_theta=getattr(cp, "rope_theta", 1_000_000.0),
                    hidden_act=getattr(cp, "hidden_act", "silu"),
                    layer_types=getattr(cp, "layer_types", None),
                )

            # Speaker encoder config
            se_cfg = getattr(tts_source, "speaker_encoder_config", None)
            if se_cfg is not None:
                se = se_cfg if not isinstance(se_cfg, dict) else type("SE", (), se_cfg)()
                tts_fields["speaker_encoder"] = SpeakerEncoderConfig(
                    mel_dim=getattr(se, "mel_dim", 128),
                    enc_dim=getattr(se, "enc_dim", 1024),
                    enc_channels=getattr(se, "enc_channels", [512, 512, 512, 512, 1536]),
                    enc_kernel_sizes=getattr(se, "enc_kernel_sizes", [5, 3, 3, 3, 1]),
                    enc_dilations=getattr(se, "enc_dilations", [1, 2, 3, 4, 1]),
                    enc_attention_channels=getattr(se, "enc_attention_channels", 128),
                    enc_res2net_scale=getattr(se, "enc_res2net_scale", 8),
                    enc_se_channels=getattr(se, "enc_se_channels", 128),
                )

            options["tts"] = TTSConfig(**tts_fields)

        # Codec tokenizer config (Qwen3-TTS-Tokenizer-12Hz)
        codec_source = parent_config or config
        hf_decoder_cfg = getattr(codec_source, "decoder_config", None)
        hf_encoder_cfg = getattr(codec_source, "encoder_config", None)
        if hf_decoder_cfg is not None and model_type == "qwen3_tts_tokenizer_12hz":
            dc = (
                hf_decoder_cfg
                if not isinstance(hf_decoder_cfg, dict)
                else type("DC", (), hf_decoder_cfg)()
            )
            options["codec_decoder"] = CodecDecoderConfig(
                codebook_dim=getattr(dc, "codebook_dim", 512),
                codebook_size=getattr(dc, "codebook_size", 2048),
                latent_dim=getattr(dc, "latent_dim", 1024),
                hidden_size=getattr(dc, "hidden_size", 512),
                intermediate_size=getattr(dc, "intermediate_size", 1024),
                num_hidden_layers=getattr(dc, "num_hidden_layers", 8),
                num_attention_heads=getattr(dc, "num_attention_heads", 16),
                num_key_value_heads=getattr(dc, "num_key_value_heads", 16),
                head_dim=getattr(dc, "head_dim", 64),
                rms_norm_eps=getattr(dc, "rms_norm_eps", 1e-5),
                rope_theta=getattr(dc, "rope_theta", 10000.0),
                max_position_embeddings=getattr(dc, "max_position_embeddings", 8000),
                decoder_dim=getattr(dc, "decoder_dim", 1536),
                num_quantizers=getattr(dc, "num_quantizers", 16),
                upsample_rates=getattr(dc, "upsample_rates", [8, 5, 4, 3]),
                upsampling_ratios=getattr(dc, "upsampling_ratios", [2, 2]),
            )
        if hf_encoder_cfg is not None and model_type == "qwen3_tts_tokenizer_12hz":
            ec = (
                hf_encoder_cfg
                if not isinstance(hf_encoder_cfg, dict)
                else type("EC", (), hf_encoder_cfg)()
            )
            options["codec_encoder"] = CodecEncoderConfig(
                codebook_dim=getattr(ec, "codebook_dim", 256),
                codebook_size=getattr(ec, "codebook_size", 2048),
                hidden_size=getattr(ec, "hidden_size", 512),
                intermediate_size=getattr(ec, "intermediate_size", 2048),
                num_hidden_layers=getattr(ec, "num_hidden_layers", 8),
                num_attention_heads=getattr(ec, "num_attention_heads", 8),
                num_key_value_heads=getattr(ec, "num_key_value_heads", 8),
                head_dim=getattr(ec, "head_dim", 64),
                rope_theta=getattr(ec, "rope_theta", 10000.0),
                max_position_embeddings=getattr(ec, "max_position_embeddings", 8000),
                num_quantizers=getattr(ec, "num_quantizers", 32),
                num_semantic_quantizers=getattr(ec, "num_semantic_quantizers", 1),
                audio_channels=getattr(ec, "audio_channels", 1),
                num_filters=getattr(ec, "num_filters", 64),
                num_residual_layers=getattr(ec, "num_residual_layers", 1),
                kernel_size=getattr(ec, "kernel_size", 7),
                last_kernel_size=getattr(ec, "last_kernel_size", 3),
                residual_kernel_size=getattr(ec, "residual_kernel_size", 3),
                compress=getattr(ec, "compress", 2),
                upsampling_ratios=list(getattr(ec, "upsampling_ratios", [8, 6, 5, 4])),
            )

        # Model dtype
        resolved = _resolve_dtype(config)
        if resolved is not None:
            options["dtype"] = resolved

        # Quantization config. Block-scaled FP8 / packed-FP4 checkpoints
        # (DeepSeek-V4-Flash) are rejected here by #602's typed
        # ``BlockQuantExportError`` (the INT4/per-tensor path cannot load them).
        # For a native-CSA export we *defer* that reject so graph construction
        # can progress past the former generic weight-shape mismatch; the
        # runtime-capability gate
        # (``mobius.models._deepseek_v4_csa.assert_native_runtime_supports_block_quant``,
        # invoked at weight-load / full-export) then fails closed until nxrt
        # advertises real block-FP8 / planar-FP4 format strings. Every non-native
        # path keeps #602's early, loud reject.
        from mobius.integrations._block_quant import (
            BlockQuantExportError,
            BlockQuantScheme,
        )

        def parse_quantization(source):
            try:
                return QuantizationConfig.from_transformers(source)
            except BlockQuantExportError:
                if not (
                    getattr(source, "native_csa", False) or allow_block_fp8_dense_fallback
                ):
                    raise
                scheme = BlockQuantScheme.from_hf_config(source)
                if scheme is None:
                    raise
                options["block_quant_scheme"] = scheme
                return None

        quant = parse_quantization(config)
        if quant is None and parent_config is not None:
            quant = parse_quantization(parent_config)
        component_quantization = _extract_component_quantization(
            config,
            parent_config,
            quant,
        )
        if component_quantization is not None:
            options["component_quantization"] = component_quantization
            quant = component_quantization.get(
                "decoder",
                component_quantization.get("model"),
            )
        if quant is not None:
            options["quantization"] = quant

        return cls(**options)

    @classmethod
    def from_file(cls, path: str, parent_config=None) -> ArchitectureConfig:
        """Create config from a local model directory or config.json file.

        Args:
            path: Path to a directory containing ``config.json``, or a
                direct path to a JSON config file.
            parent_config: Optional parent HF config (for composite models).

        Returns:
            An ``ArchitectureConfig`` instance.
        """
        import transformers

        config = transformers.AutoConfig.from_pretrained(path)
        hf_config = config
        if parent_config is None:
            parent_config = config
        if hasattr(config, "text_config"):
            hf_config = config.text_config
        elif hasattr(config, "language_config"):
            hf_config = config.language_config
        return cls.from_transformers(hf_config, parent_config=parent_config)

    def validate(self) -> None:
        """Validate config field consistency.

        Raises:
            ValueError: If the config has invalid or inconsistent fields.
        """
        errors: list[str] = []
        if self.hidden_size <= 0:
            errors.append(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_attention_heads <= 0:
            errors.append(
                f"num_attention_heads must be positive, got {self.num_attention_heads}"
            )
        if self.num_hidden_layers <= 0:
            errors.append(f"num_hidden_layers must be positive, got {self.num_hidden_layers}")
        if self.vocab_size < 0:
            errors.append(f"vocab_size must be non-negative, got {self.vocab_size}")
        if self.head_dim <= 0:
            errors.append(f"head_dim must be positive, got {self.head_dim}")
        if (
            self.num_key_value_heads > 0
            and self.num_attention_heads % self.num_key_value_heads != 0
        ):
            errors.append(
                f"num_attention_heads ({self.num_attention_heads}) must be "
                f"divisible by num_key_value_heads ({self.num_key_value_heads})"
            )
        if (
            self.hidden_size > 0
            and self.num_attention_heads > 0
            and self.head_dim == DEFAULT_INT
            and self.hidden_size % self.num_attention_heads != 0
        ):
            errors.append(
                f"hidden_size ({self.hidden_size}) must be "
                f"divisible by num_attention_heads ({self.num_attention_heads})"
            )
        if self.intermediate_size is not None and self.intermediate_size <= 0:
            errors.append(f"intermediate_size must be positive, got {self.intermediate_size}")
        if errors:
            raise ValueError(
                "Invalid ArchitectureConfig:\n" + "\n".join(f"  - {e}" for e in errors)
            )


def _as_attribute_config(value: object) -> object:
    """Recursively give a plain ``dict`` HF sub-config attribute access.

    ``transformers`` 5.x increasingly leaves nested sub-configs (a decoder,
    a vision tower) as plain dicts on the parent config rather than as
    ``PretrainedConfig`` instances. Every ``from_transformers`` here reads its
    input with ``getattr``, so a dict silently degrades: ``getattr(d, "x", None)``
    is always ``None``, which turns into a wrong default rather than an error,
    or into ``AttributeError`` on a required field.

    Non-dict values, including real config objects, are returned unchanged.
    """
    import types

    if isinstance(value, dict):
        return types.SimpleNamespace(
            **{key: _as_attribute_config(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_as_attribute_config(item) for item in value]
    return value


def _shallow_fields(config) -> dict:
    """Extract fields from a dataclass without recursive conversion.

    Unlike ``dataclasses.asdict()``, this preserves nested dataclass
    instances (:class:`VisionConfig`, :class:`AudioConfig`, etc.) as-is.
    """
    return {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}


def _as_int(value) -> int:
    """Coerce *value* to int, taking the first element if it is a list/tuple.

    Some HuggingFace configs express ``image_size`` or ``patch_size`` as
    ``[H, W]`` lists or ``{"height": H, "width": W}`` dicts.
    We take the first element (height) for simplicity.
    """
    if isinstance(value, dict):
        # PVT-v2 etc. use {"height": H, "width": W} for image_size
        if "height" in value:
            return int(value["height"])
        return int(next(iter(value.values())))
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def _first(value):
    """Return the first element of a list/tuple, or *value* unchanged."""
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def _shared_expert_size(config) -> int | None:
    """Compute shared_expert_intermediate_size from moe_intermediate_size * n_shared_experts.

    ERNIE-4.5 uses ``moe_num_shared_experts``, GLM-4.5 uses ``n_shared_experts``,
    and Hunyuan uses ``num_shared_expert`` (singular, per-layer list).
    Both share ``moe_intermediate_size`` as the per-expert FFN hidden size.
    """
    moe_dim = _first(getattr(config, "moe_intermediate_size", None))
    n_shared = _first(
        getattr(config, "moe_num_shared_experts", None)
        or getattr(config, "n_shared_experts", None)
        or getattr(config, "num_shared_expert", None)
    )
    if moe_dim is not None and n_shared is not None:
        return int(moe_dim) * int(n_shared)
    return None


# ---------------------------------------------------------------------------
# Category subclasses — type markers for model categories
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CausalLMConfig(ArchitectureConfig):
    """Configuration for decoder-only causal language models.

    Used by Llama, Mistral, Qwen, GPT-2, and similar architectures.
    Inherits all shared transformer fields from :class:`ArchitectureConfig`.
    """


@dataclasses.dataclass
class GrokGGUFConfig(CausalLMConfig):
    """GGUF-only Grok graph settings loaded by the pinned llama.cpp implementation."""

    embedding_scale: float = 78.38367176906169
    attention_output_scale: float = 0.08838834764831845
    logit_output_scale: float = 0.5773502691896257
    attn_logit_softcapping: float = 30.0
    router_logit_softcapping: float = 30.0
    final_logit_softcapping: float = 0.0
    attention_temperature_length: int = 0
    has_dense_ffn: bool = False
    has_gated_dense_ffn: bool = False
    has_gated_experts: bool = True


@dataclasses.dataclass
class GroveMoEGGUFConfig(CausalLMConfig):
    """GGUF-only GroveMoE chunk-expert routing settings."""

    chunk_expert_intermediate_size: int = DEFAULT_INT
    experts_per_group: int = DEFAULT_INT
    expert_group_scale: float = 0.05


@dataclasses.dataclass
class Qwen4ExpConfig(CausalLMConfig):
    """Exact configuration for experimental Qwen4/Qwen3.8 Flash-Next."""

    hc_count: int = 4
    hc_lowrank: int = 320
    ple_layer_ids: list[int] | None = None
    ple_embed_dim: int | None = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    split_ngram_parts: int = 512
    indexer_n_heads: int | None = None
    indexer_kv_heads: int | None = None
    indexer_head_dim: int | None = None
    indexer_budget: int | None = None
    indexer_compress_ratio: int | None = None
    output_gate_type: str | None = None
    linear_qk_l2norm_eps: float = 1e-6
    mamba_ssm_dtype: ir.DataType = ir.DataType.FLOAT
    mtp_num_hidden_layers: int = 0
    mtp_use_dedicated_embeddings: bool = False
    unsupported_video_token_id: int | None = None

    def __post_init__(self) -> None:
        self.ple_layer_ids = sorted(set(self.ple_layer_ids or []))
        if self.ple_embed_dim is None:
            self.ple_embed_dim = self.hidden_size
        if self.layer_types is None:
            interval = self.full_attention_interval or 4
            self.layer_types = [
                "linear_attention" if (index + 1) % interval else "qwen_sparse_attention"
                for index in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = [
                "qwen_sparse_attention" if layer_type == "full_attention" else layer_type
                for layer_type in self.layer_types
            ]
        self._validate_architecture()

    def _validate_architecture(self) -> None:
        layer_types = self.layer_types or []
        if len(layer_types) != self.num_hidden_layers:
            raise ValueError(
                "Qwen4-Exp layer_types must contain exactly num_hidden_layers entries "
                f"(expected {self.num_hidden_layers}, got {len(layer_types)})"
            )
        unsupported = sorted(set(layer_types) - {"linear_attention", "qwen_sparse_attention"})
        if unsupported:
            raise ValueError(f"Unsupported Qwen4-Exp layer types: {unsupported}")
        output_gate_type = self.output_gate_type or self.hidden_act
        if output_gate_type not in {"sigmoid", "silu"}:
            raise ValueError(
                f"Unsupported Qwen4-Exp output gate activation: {output_gate_type}"
            )
        if self.hc_count <= 1:
            raise ValueError(f"Qwen4-Exp requires hc_count > 1, got {self.hc_count}")
        if self.hc_lowrank <= 0:
            raise ValueError(f"Qwen4-Exp requires hc_lowrank > 0, got {self.hc_lowrank}")
        if self.mamba_ssm_dtype != ir.DataType.FLOAT:
            raise ValueError(
                "Qwen4-Exp requires mamba_ssm_dtype=float32 for the pinned "
                "Gated-DeltaNet recurrence"
            )
        if self.rope_interleave:
            raise ValueError("Qwen4-Exp uses half-split RoPE; rope_interleave must be false")
        if not self.num_local_experts or self.num_local_experts <= 0:
            raise ValueError("Qwen4-Exp num_local_experts must be > 0")
        if not self.num_experts_per_tok or not (
            0 < self.num_experts_per_tok <= self.num_local_experts
        ):
            raise ValueError("Qwen4-Exp num_experts_per_tok must be in [1, num_local_experts]")
        if not self.moe_intermediate_size or self.moe_intermediate_size <= 0:
            raise ValueError("Qwen4-Exp moe_intermediate_size must be > 0")
        if (
            not self.shared_expert_intermediate_size
            or self.shared_expert_intermediate_size <= 0
        ):
            raise ValueError("Qwen4-Exp shared_expert_intermediate_size must be > 0")
        if "linear_attention" in layer_types:
            linear_fields = {
                "linear_num_key_heads": self.linear_num_key_heads,
                "linear_num_value_heads": self.linear_num_value_heads,
                "linear_key_head_dim": self.linear_key_head_dim,
                "linear_value_head_dim": self.linear_value_head_dim,
                "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            }
            if any(value is None or value <= 0 for value in linear_fields.values()):
                raise ValueError(
                    f"Qwen4-Exp linear-attention config values must be positive: {linear_fields}"
                )
            assert self.linear_num_key_heads is not None
            assert self.linear_num_value_heads is not None
            if self.linear_num_value_heads % self.linear_num_key_heads:
                raise ValueError(
                    "Qwen4-Exp linear_num_value_heads must be divisible by "
                    "linear_num_key_heads"
                )

        qsa_fields = {
            "indexer_n_heads": self.indexer_n_heads,
            "indexer_kv_heads": self.indexer_kv_heads,
            "indexer_head_dim": self.indexer_head_dim,
            "indexer_budget": self.indexer_budget,
            "indexer_compress_ratio": self.indexer_compress_ratio,
        }
        if any(value is not None for value in qsa_fields.values()):
            missing = [name for name, value in qsa_fields.items() if value is None]
            if missing:
                raise ValueError(f"Qwen4-Exp QSA config is missing required fields: {missing}")
            if any(value is not None and value <= 0 for value in qsa_fields.values()):
                raise ValueError(f"Qwen4-Exp QSA config values must be positive: {qsa_fields}")
            if self.indexer_kv_heads != 1:
                raise ValueError("Qwen4-Exp QSA requires indexer_kv_heads=1")
            assert self.indexer_budget is not None
            assert self.indexer_compress_ratio is not None
            if self.indexer_budget % self.indexer_compress_ratio:
                raise ValueError(
                    "Qwen4-Exp indexer_budget must be divisible by indexer_compress_ratio"
                )
            rotary_dim = int(self.head_dim * (self.partial_rotary_factor or 1.0))
            assert self.indexer_head_dim is not None
            if rotary_dim > self.indexer_head_dim:
                raise ValueError(
                    "Qwen4-Exp attention RoPE dimensions must fit the QSA index head: "
                    f"rotary_dim={rotary_dim}, indexer_head_dim={self.indexer_head_dim}"
                )
        elif "qwen_sparse_attention" in layer_types:
            raise ValueError("Qwen4-Exp sparse-attention layers require a complete QSA config")

        if self.ple_layer_ids:
            ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
            assert self.ple_embed_dim is not None
            if ngram_heads <= 0 or self.ple_embed_dim <= 0 or self.ple_embed_dim % ngram_heads:
                raise ValueError(
                    "Qwen4-Exp ple_embed_dim must be positive and divisible by the "
                    f"number of n-gram heads ({ngram_heads})"
                )
            invalid = [
                layer_id
                for layer_id in self.ple_layer_ids
                if layer_id < 1 or layer_id > self.num_hidden_layers
            ]
            if invalid:
                raise ValueError(
                    "Qwen4-Exp ple_layer_ids must be one-indexed decoder layer ids; "
                    f"invalid ids: {invalid}"
                )
            non_linear = [
                layer_id
                for layer_id in self.ple_layer_ids
                if layer_types[layer_id - 1] != "linear_attention"
            ]
            if non_linear:
                raise ValueError(
                    "Qwen4-Exp PLE is only supported on linear_attention layers; "
                    f"got PLE on layers {non_linear}"
                )
            if self.eos_token_id is None or (
                isinstance(self.eos_token_id, list) and not self.eos_token_id
            ):
                raise ValueError("Qwen4-Exp eos_token_id must be set when PLE is enabled")
            if self.ple_conv_kernel_size <= 0:
                raise ValueError("Qwen4-Exp ple_conv_kernel_size must be > 0")
            if self.ngram_vocab_size_base < 2:
                raise ValueError("Qwen4-Exp ngram_vocab_size_base must be >= 2")
            if self.make_ngram_vocab_size_divisible_by <= 0:
                raise ValueError("Qwen4-Exp make_ngram_vocab_size_divisible_by must be > 0")
            if self.split_ngram_parts <= 0:
                raise ValueError("Qwen4-Exp split_ngram_parts must be > 0")

        if self.mtp_use_dedicated_embeddings:
            raise ValueError(
                "Qwen4-Exp dedicated MTP embeddings are unsupported: the pinned "
                "official runtime defines no MTP execution or NextN cache ABI"
            )

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Qwen4ExpConfig:
        text = _as_attribute_config(getattr(config, "text_config", None)) or config
        parent = parent_config or (config if text is not config else None)
        base = ArchitectureConfig.from_transformers(
            text,
            parent,
            allow_block_fp8_dense_fallback=True,
        )
        fields = _shallow_fields(base)
        is_multimodal = (
            getattr(parent, "model_type", None) == "qwen4_exp"
            and getattr(parent, "vision_config", None) is not None
        )
        if is_multimodal:
            if getattr(parent, "language_model_only", False):
                raise ValueError(
                    "Qwen4-Exp multimodal export requires language_model_only=false"
                )
            if base.video_token_id != 248057:
                raise ValueError(
                    "Unsupported Qwen4-Exp video token contract: expected source "
                    f"video_token_id 248057, got {base.video_token_id}"
                )
            vision = base.vision
            if vision is None:
                raise ValueError("Qwen4-Exp multimodal config requires vision_config")
            expected = {
                "num_hidden_layers": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_attention_heads": 16,
                "in_channels": 3,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "num_position_embeddings": 2304,
                "out_hidden_size": 2560,
                "hidden_act": "gelu_pytorch_tanh",
            }
            actual = {
                "num_hidden_layers": vision.num_hidden_layers,
                "hidden_size": vision.hidden_size,
                "intermediate_size": vision.intermediate_size,
                "num_attention_heads": vision.num_attention_heads,
                "in_channels": vision.in_channels,
                "patch_size": vision.patch_size,
                "temporal_patch_size": vision.temporal_patch_size,
                "spatial_merge_size": vision.spatial_merge_size,
                "num_position_embeddings": vision.num_position_embeddings,
                "out_hidden_size": vision.out_hidden_size,
                "hidden_act": vision.hidden_act,
            }
            mismatches = {
                name: (actual[name], value)
                for name, value in expected.items()
                if actual[name] != value
            }
            if mismatches:
                raise ValueError(
                    "Unsupported Qwen4-Exp vision variant; expected the pinned "
                    f"Qwen3/Qwen3.5 tower, got mismatches {mismatches}"
                )
            if vision.deepstack_visual_indexes:
                raise ValueError(
                    "Unsupported Qwen4-Exp vision variant: DeepStack must be disabled"
                )
            if base.hidden_size != 2560:
                raise ValueError(
                    "Unsupported Qwen4-Exp multimodal projection width: "
                    f"expected text hidden_size 2560, got {base.hidden_size}"
                )
        mamba_ssm_dtype = _resolve_dtype_value(getattr(text, "mamba_ssm_dtype", "float32"))
        if mamba_ssm_dtype is None:
            raise ValueError(
                "Qwen4-Exp mamba_ssm_dtype must be a recognized floating-point dtype"
            )
        layer_types = getattr(text, "layer_types", None)
        if layer_types is not None:
            layer_types = [
                "qwen_sparse_attention" if value == "full_attention" else value
                for value in layer_types
            ]
        fields.update(
            model_type="qwen4_exp" if is_multimodal else "qwen4_exp_text",
            layer_types=layer_types,
            hc_count=getattr(text, "hc_count", 4),
            hc_lowrank=getattr(text, "hc_lowrank", 320),
            ple_layer_ids=list(getattr(text, "ple_layer_ids", None) or []),
            ple_embed_dim=getattr(text, "ple_embed_dim", None),
            ple_conv_kernel_size=getattr(text, "ple_conv_kernel_size", 4),
            ngram_size=getattr(text, "ngram_size", 3),
            heads_per_ngram=getattr(text, "heads_per_ngram", 8),
            ngram_vocab_size_base=getattr(text, "ngram_vocab_size_base", 20_000_000),
            make_ngram_vocab_size_divisible_by=getattr(
                text, "make_ngram_vocab_size_divisible_by", 128
            ),
            seed=getattr(text, "seed", 1234),
            split_ngram_parts=getattr(text, "split_ngram_parts", 512),
            indexer_n_heads=getattr(text, "indexer_n_heads", None),
            indexer_kv_heads=getattr(text, "indexer_kv_heads", None),
            indexer_head_dim=getattr(text, "indexer_head_dim", None),
            indexer_budget=getattr(text, "indexer_budget", None),
            indexer_compress_ratio=getattr(text, "indexer_compress_ratio", None),
            output_gate_type=getattr(text, "output_gate_type", None),
            mamba_ssm_dtype=mamba_ssm_dtype,
            mtp_num_hidden_layers=getattr(text, "mtp_num_hidden_layers", 0),
            mtp_use_dedicated_embeddings=getattr(text, "mtp_use_dedicated_embeddings", False),
        )
        if is_multimodal:
            # The current package is explicitly image-only. Preserve validation
            # of the source token above, but do not publish runtime video support
            # until a video producer/preprocessor route exists.
            fields["video_token_id"] = None
            fields["unsupported_video_token_id"] = base.video_token_id
        return cls(**fields)


@dataclasses.dataclass
class Jais2Config(CausalLMConfig):
    """Normalize the published Jais2 projection and LayerNorm configuration."""

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Jais2Config:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        attention_bias = bool(getattr(config, "attention_bias", True))
        fields = _shallow_fields(base)
        fields.update(
            attn_qkv_bias=attention_bias,
            attn_o_bias=attention_bias,
            mlp_bias=bool(getattr(config, "mlp_bias", True)),
        )
        return cls(**fields)


@dataclasses.dataclass
class CodeShellConfig(CausalLMConfig):
    """Canonicalize the published CodeShell HuggingFace configuration."""

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> CodeShellConfig:
        if getattr(config, "position_embedding_type", "rope") != "rope":
            raise ValueError("CodeShell export only supports position_embedding_type='rope'")

        base = ArchitectureConfig.from_transformers(config, parent_config)
        hidden_size = int(getattr(config, "hidden_size", None) or config.n_embd)
        num_heads = int(getattr(config, "num_attention_heads", None) or config.n_head)
        num_kv_heads = (
            int(getattr(config, "num_query_groups", num_heads))
            if getattr(config, "group_query_attention", False)
            else num_heads
        )
        rope = base.rope or RoPEConfig(rope_type="default", rope_theta=10_000.0)
        fields = _shallow_fields(base)
        fields.update(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            num_hidden_layers=int(
                getattr(config, "num_hidden_layers", None) or config.n_layer
            ),
            intermediate_size=int(
                getattr(config, "intermediate_size", None)
                or getattr(config, "n_inner", None)
                or 4 * hidden_size
            ),
            max_position_embeddings=int(
                getattr(config, "max_position_embeddings", None) or config.n_positions
            ),
            hidden_act=getattr(config, "activation_function", "gelu_pytorch_tanh"),
            attn_qkv_bias=True,
            attn_o_bias=True,
            mlp_bias=True,
            tie_word_embeddings=True,
            rope=rope,
            rope_type=rope.rope_type,
            rope_theta=rope.rope_theta,
            rope_scaling=rope.rope_scaling,
            partial_rotary_factor=rope.partial_rotary_factor,
        )
        return cls(**fields)


@dataclasses.dataclass
class XverseConfig(CausalLMConfig):
    """Supply Xverse's source-defined standard RoPE default."""

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> XverseConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        rope = base.rope or RoPEConfig(rope_type="default", rope_theta=10_000.0)
        fields = _shallow_fields(base)
        fields.update(
            rope=rope,
            rope_type=rope.rope_type,
            rope_theta=rope.rope_theta,
            rope_scaling=rope.rope_scaling,
            partial_rotary_factor=rope.partial_rotary_factor,
        )
        return cls(**fields)


@dataclasses.dataclass
class MiniMaxConfig(CausalLMConfig):
    """Exact configuration for MiniMax-Text-01 and MiniMax-M1 backbones."""

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> MiniMaxConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        raw_schedule = getattr(config, "attn_type_list", None)
        if raw_schedule is None:
            raise ValueError("MiniMax-01 config requires an explicit attn_type_list")
        if len(raw_schedule) != base.num_hidden_layers:
            raise ValueError(
                "MiniMax-01 attn_type_list must contain exactly "
                f"{base.num_hidden_layers} entries, got {len(raw_schedule)}"
            )
        if any(value not in (0, 1, False, True) for value in raw_schedule):
            raise ValueError("MiniMax-01 attn_type_list entries must be 0 or 1")
        if not bool(getattr(config, "postnorm", True)):
            raise ValueError("MiniMax-01 requires postnorm=true")
        if int(getattr(config, "shared_intermediate_size", 0) or 0):
            raise ValueError(
                "MiniMax-01 shared experts are not supported by the pinned GGUF architecture"
            )

        beta_names = (
            "layernorm_full_attention_beta",
            "layernorm_linear_attention_beta",
            "layernorm_mlp_beta",
        )
        betas = {name: float(getattr(config, name, 1.0)) for name in beta_names}
        if any(not math.isclose(value, 1.0) for value in betas.values()):
            raise ValueError(f"MiniMax-01 beta residual factors must all equal 1.0: {betas}")

        fields = _shallow_fields(base)
        fields.update(
            model_type="minimax",
            layer_types=[
                "full_attention" if int(value) == 1 else "lightning_attention"
                for value in raw_schedule
            ],
            hidden_act="silu",
            norm_topk_prob=True,
            disable_qmoe=True,
            lightning_norm_eps=float(getattr(config, "lightning_norm_eps", 1e-6)),
            full_attn_alpha_factor=float(
                getattr(config, "layernorm_full_attention_alpha", 1.0)
            ),
            full_attn_beta_factor=betas["layernorm_full_attention_beta"],
            linear_attn_alpha_factor=float(
                getattr(config, "layernorm_linear_attention_alpha", 1.0)
            ),
            linear_attn_beta_factor=betas["layernorm_linear_attention_beta"],
            mlp_alpha_factor=float(getattr(config, "layernorm_mlp_alpha", 1.0)),
            mlp_beta_factor=betas["layernorm_mlp_beta"],
        )
        return cls(**fields)


@dataclasses.dataclass
class KimiLinearConfig(CausalLMConfig):
    """Exact configuration for Moonshot Kimi Linear models."""

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> KimiLinearConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        linear = _as_attribute_config(getattr(config, "linear_attn_config", None))
        if linear is None:
            raise ValueError("Kimi Linear requires linear_attn_config")

        kda_layers = [int(i) for i in getattr(linear, "kda_layers", ())]
        full_layers = [int(i) for i in getattr(linear, "full_attn_layers", ())]
        expected = set(range(1, base.num_hidden_layers + 1))
        if set(kda_layers) & set(full_layers):
            raise ValueError("Kimi Linear KDA and full-attention schedules must not overlap")
        if set(kda_layers) | set(full_layers) != expected:
            raise ValueError(
                "Kimi Linear schedules must exactly partition one-based layer indices "
                f"1..{base.num_hidden_layers}"
            )
        if len(kda_layers) != len(set(kda_layers)) or len(full_layers) != len(
            set(full_layers)
        ):
            raise ValueError("Kimi Linear schedules must not contain duplicate layers")
        if not bool(getattr(config, "mla_use_nope", False)):
            raise ValueError("Kimi Linear requires mla_use_nope=true")
        if getattr(config, "q_lora_rank", None) not in (None, 0):
            raise ValueError("Kimi Linear Q-LoRA is not supported by the authoritative graph")
        if int(getattr(config, "num_nextn_predict_layers", 0)):
            raise ValueError("Kimi Linear NextN prediction layers are not supported")
        if int(getattr(config, "moe_layer_freq", 1)) != 1:
            raise ValueError("Kimi Linear requires moe_layer_freq=1")
        if str(getattr(config, "moe_router_activation_func", "")) != "sigmoid":
            raise ValueError("Kimi Linear requires sigmoid expert routing")
        if not bool(getattr(config, "moe_renormalize", False)):
            raise ValueError("Kimi Linear requires selected expert weight renormalization")
        n_group = int(vars(config).get("num_expert_group", 1))
        topk_group = int(vars(config).get("topk_group", 1))
        if n_group != 1 or topk_group != 1:
            raise ValueError(
                "Kimi Linear supports only the pinned single expert-group profile"
            )

        num_heads = int(vars(linear).get("num_heads", base.num_attention_heads))
        head_dim = int(linear.head_dim)
        conv_kernel = int(linear.short_conv_kernel_size)
        if num_heads != base.num_attention_heads:
            raise ValueError("Kimi Linear KDA and MLA head counts must match")
        if conv_kernel < 2:
            raise ValueError("Kimi Linear short_conv_kernel_size must be at least 2")
        fields = _shallow_fields(base)
        fields.update(
            model_type="kimi_linear",
            layer_types=[
                "kimi_linear_attention" if i + 1 in set(kda_layers) else "full_attention"
                for i in range(base.num_hidden_layers)
            ],
            linear_num_key_heads=num_heads,
            linear_num_value_heads=num_heads,
            linear_key_head_dim=head_dim,
            linear_value_head_dim=head_dim,
            linear_conv_kernel_dim=conv_kernel,
            num_local_experts=int(config.num_experts),
            num_experts_per_tok=int(config.num_experts_per_token),
            n_group=n_group,
            topk_group=topk_group,
            n_shared_experts=int(config.num_shared_experts),
            norm_topk_prob=True,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            disable_qmoe=True,
            rope_type=None,
            rope_theta=None,
            rope_scaling=None,
            partial_rotary_factor=None,
        )
        result = cls(**fields)
        if result.qk_nope_head_dim is None or result.qk_rope_head_dim is None:
            raise ValueError("Kimi Linear requires both MLA key dimension fields")
        if result.v_head_dim is None or result.kv_lora_rank is None:
            raise ValueError("Kimi Linear requires MLA value dimension and KV-LoRA rank")
        return result


@dataclasses.dataclass
class KimiK3Config(CausalLMConfig):
    """Text-decoder configuration extracted from the composite Kimi-K3 config."""

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> KimiK3Config:
        parent = parent_config or config
        text = _as_attribute_config(getattr(config, "text_config", None)) or config
        quantization_config = getattr(text, "quantization_config", None)
        from mobius.integrations.compressed_tensors import is_compressed_tensors_config

        if is_compressed_tensors_config(quantization_config):
            raise NotImplementedError(
                "Kimi-K3 compressed-tensors checkpoints use selective MXFP4 routed "
                "experts, which are not representable by the generic quantized-linear "
                "loader. Use a GGUF checkpoint or an unquantized state dict."
            )
        base = ArchitectureConfig.from_transformers(text, parent)
        linear = _as_attribute_config(getattr(text, "linear_attn_config", None))
        if linear is None:
            raise ValueError("Kimi-K3 requires text_config.linear_attn_config")

        kda_layers = [int(i) for i in getattr(linear, "kda_layers", ())]
        full_layers = [int(i) for i in getattr(linear, "full_attn_layers", ())]
        expected = set(range(1, base.num_hidden_layers + 1))
        if (
            set(kda_layers) & set(full_layers)
            or set(kda_layers) | set(full_layers) != expected
            or len(kda_layers) != len(set(kda_layers))
            or len(full_layers) != len(set(full_layers))
        ):
            raise ValueError(
                "Kimi-K3 KDA and MLA schedules must uniquely partition one-based "
                f"layer indices 1..{base.num_hidden_layers}"
            )
        if not bool(getattr(text, "mla_use_nope", False)):
            raise ValueError("Kimi-K3 requires mla_use_nope=true")
        if not bool(getattr(text, "mla_use_output_gate", False)):
            raise ValueError("Kimi-K3 requires mla_use_output_gate=true")
        if not bool(getattr(linear, "use_full_rank_gate", False)):
            raise ValueError("Kimi-K3 requires a full-rank KDA output gate")
        raw_lower_bound = float(getattr(linear, "gate_lower_bound", 0.0))
        if raw_lower_bound >= 0.0:
            raise ValueError("Kimi-K3 gate_lower_bound must be negative")
        if int(getattr(text, "first_k_dense_replace", 0)) != 1:
            raise ValueError("Kimi-K3 requires exactly one leading dense layer")
        if int(getattr(text, "num_shared_experts", 0)) != 2:
            raise ValueError("Kimi-K3 requires exactly two shared experts")
        if str(getattr(text, "hidden_act", "")) != "situ":
            raise ValueError("Kimi-K3 requires SiTU expert activation")
        if not math.isclose(float(getattr(text, "activation_situ_beta", 0.0)), 4.0):
            raise ValueError("Kimi-K3 requires activation_situ_beta=4")
        if not math.isclose(float(getattr(text, "activation_situ_linear_beta", 0.0)), 25.0):
            raise ValueError("Kimi-K3 requires activation_situ_linear_beta=25")
        if not bool(getattr(text, "latent_moe_use_norm", False)):
            raise ValueError("Kimi-K3 requires latent MoE RMS normalization")
        if getattr(text, "routed_expert_hidden_size", None) is None:
            raise ValueError("Kimi-K3 requires routed_expert_hidden_size")
        if getattr(text, "q_lora_rank", None) in (None, 0):
            raise ValueError("Kimi-K3 requires Q-LoRA")
        if bool(getattr(text, "tie_word_embeddings", False)):
            raise ValueError("Kimi-K3 requires an untied LM head")
        if str(getattr(text, "moe_router_activation_func", "")) != "sigmoid":
            raise ValueError("Kimi-K3 requires sigmoid expert routing")
        if not bool(getattr(text, "moe_renormalize", False)):
            raise ValueError("Kimi-K3 requires selected expert weight renormalization")
        if int(getattr(text, "moe_layer_freq", 1)) != 1:
            raise ValueError("Kimi-K3 requires MoE in every layer after layer zero")
        if str(getattr(text, "topk_method", "")) != "noaux_tc":
            raise ValueError("Kimi-K3 requires correction-bias noaux_tc routing")
        if (
            int(getattr(text, "num_expert_group", 1)) != 1
            or int(getattr(text, "topk_group", 1)) != 1
        ):
            raise ValueError("Kimi-K3 requires the released single expert-group profile")
        if int(getattr(text, "num_nextn_predict_layers", 0)):
            raise ValueError("Kimi-K3 NextN prediction layers are not supported")
        if int(getattr(text, "attn_res_block_size", 0)) <= 0:
            raise ValueError("Kimi-K3 requires a positive attn_res_block_size")

        num_heads = int(vars(linear).get("num_heads", base.num_attention_heads))
        conv_kernel = int(linear.short_conv_kernel_size)
        if num_heads != base.num_attention_heads:
            raise ValueError("Kimi-K3 KDA and MLA head counts must match")
        if conv_kernel < 2:
            raise ValueError("Kimi-K3 short_conv_kernel_size must be at least 2")
        fields = _shallow_fields(base)
        fields.update(
            model_type="kimi_k3",
            layer_types=[
                "kimi_k3_attention" if i + 1 in set(kda_layers) else "full_attention"
                for i in range(base.num_hidden_layers)
            ],
            linear_num_key_heads=num_heads,
            linear_num_value_heads=num_heads,
            linear_key_head_dim=int(linear.head_dim),
            linear_value_head_dim=int(linear.head_dim),
            linear_conv_kernel_dim=conv_kernel,
            linear_gate_lower_bound=-raw_lower_bound,
            linear_use_full_rank_gate=True,
            mla_use_output_gate=True,
            attn_res_block_size=int(text.attn_res_block_size),
            routed_expert_hidden_size=int(text.routed_expert_hidden_size),
            latent_moe_use_norm=True,
            activation_situ_beta=float(text.activation_situ_beta),
            activation_situ_linear_beta=float(text.activation_situ_linear_beta),
            num_local_experts=int(text.num_experts),
            num_experts_per_tok=int(text.num_experts_per_token),
            n_group=int(getattr(text, "num_expert_group", 1)),
            topk_group=int(getattr(text, "topk_group", 1)),
            n_shared_experts=2,
            norm_topk_prob=True,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            disable_qmoe=True,
            rope_type=None,
            rope_theta=None,
            rope_scaling=None,
            partial_rotary_factor=None,
            tie_word_embeddings=False,
        )
        result = cls(**fields)
        required = (
            result.q_lora_rank,
            result.kv_lora_rank,
            result.qk_nope_head_dim,
            result.qk_rope_head_dim,
            result.v_head_dim,
        )
        if any(value is None for value in required):
            raise ValueError("Kimi-K3 requires complete Q-LoRA and MLA dimensions")
        return result


@dataclasses.dataclass
class EncoderConfig(ArchitectureConfig):
    """Configuration for encoder-only models (BERT, ViT, etc.)."""


@dataclasses.dataclass
class VisionLanguageConfig(CausalLMConfig):
    """Configuration for vision-language models (LLaVA, Qwen-VL, etc.).

    Inherits :class:`CausalLMConfig` for the text decoder component.
    Vision-specific fields live in the :class:`VisionConfig` sub-config.
    """


@dataclasses.dataclass
class NemotronParseConfig(ArchitectureConfig):
    """Configuration for NVIDIA Nemotron Parse image-to-text models."""

    image_height: int = 2048
    image_width: int = 1664
    vision_max_grid_size: int = 128
    num_summary_tokens: int = 3
    decoder_start_token_id: int = 2
    scale_embedding: bool = True
    add_final_layer_norm: bool = True

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> NemotronParseConfig:
        del parent_config

        decoder = _as_attribute_config(getattr(config, "decoder", None))
        if decoder is None:
            raise ValueError("Nemotron Parse config is missing its decoder sub-config")
        base = ArchitectureConfig.from_transformers(decoder, parent_config=config)
        fields = _shallow_fields(base)
        num_attention_heads = int(
            getattr(decoder, "decoder_attention_heads", None)
            or getattr(decoder, "num_attention_heads", fields["num_attention_heads"])
        )
        hidden_size = int(fields["hidden_size"])
        if hidden_size % num_attention_heads:
            raise ValueError(
                "Nemotron Parse decoder hidden size must be divisible by its attention heads"
            )
        fields.update(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            head_dim=hidden_size // num_attention_heads,
        )

        raw_image_size = getattr(config, "image_size", (2048, 1664))
        if isinstance(raw_image_size, int):
            image_height = image_width = raw_image_size
        else:
            image_height, image_width = (int(raw_image_size[0]), int(raw_image_size[1]))

        encoder = _as_attribute_config(getattr(config, "encoder", None))
        patch_size = int(getattr(encoder, "patch_size", 16))
        max_resolution = int(
            getattr(encoder, "max_resolution", max(image_height, image_width))
        )
        fields.update(
            model_type="nemotron_parse",
            decoder_start_token_id=int(getattr(config, "decoder_start_token_id", 2)),
            bos_token_id=getattr(config, "bos_token_id", fields.get("bos_token_id")),
            eos_token_id=getattr(config, "eos_token_id", fields.get("eos_token_id")),
            pad_token_id=getattr(config, "pad_token_id", fields["pad_token_id"]),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", True),
            max_position_embeddings=(
                getattr(config, "max_sequence_length", None)
                or fields["max_position_embeddings"]
            ),
            vision=VisionConfig(
                hidden_size=1280,
                intermediate_size=5120,
                num_hidden_layers=32,
                num_attention_heads=16,
                image_size=max_resolution,
                patch_size=patch_size,
                norm_eps=1e-6,
                model_type="radio_v2.5-h",
                in_channels=3,
            ),
        )
        resolved_dtype = _resolve_dtype(config)
        if resolved_dtype is not None:
            fields["dtype"] = resolved_dtype

        return cls(
            **fields,
            image_height=image_height,
            image_width=image_width,
            vision_max_grid_size=max_resolution // patch_size,
            num_summary_tokens=3,
            scale_embedding=bool(getattr(decoder, "scale_embedding", True)),
            add_final_layer_norm=bool(getattr(decoder, "add_final_layer_norm", True)),
        )


# ---------------------------------------------------------------------------
# Model-family subclasses — add model-specific fields
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MuseGlimmerConfig(ArchitectureConfig):
    """Configuration for Muse Glimmer text and vision-language models."""

    qk_scale_factor: float = 3.87
    output_multiplier: float = 0.19611613513818404
    final_logit_softcapping: float = 20.0
    post_norm_eps: float = 1e-8
    layer_rope_theta: list[float | int] | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> MuseGlimmerConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        layer_rope_theta = getattr(config, "layer_rope_theta", None)
        if layer_rope_theta is not None:
            layer_rope_theta = list(layer_rope_theta)
            no_rope_layers = [
                layer_idx
                for layer_idx, rope_theta in enumerate(layer_rope_theta)
                if rope_theta == 0
            ]
            base = dataclasses.replace(base, no_rope_layers=no_rope_layers)
        base = dataclasses.replace(base, attn_qk_norm=True)
        return cls(
            **_shallow_fields(base),
            qk_scale_factor=getattr(config, "qk_scale_factor", 3.87),
            output_multiplier=getattr(config, "output_multiplier", 0.19611613513818404),
            final_logit_softcapping=(getattr(config, "final_logit_softcapping", 20.0) or 0.0),
            post_norm_eps=getattr(config, "post_norm_eps", 1e-8),
            layer_rope_theta=layer_rope_theta,
        )


@dataclasses.dataclass
class Lfm2Config(CausalLMConfig):
    """Configuration for LFM2's automatically adjusted feed-forward width."""

    block_multiple_of: int = 256
    block_ffn_dim_multiplier: float | int | None = 1.0
    block_auto_adjust_ff_dim: bool = True

    @property
    def effective_intermediate_size(self) -> int:
        """The MLP width constructed by HuggingFace's ``Lfm2MLP``."""
        intermediate_size = self.intermediate_size
        if self.block_auto_adjust_ff_dim:
            intermediate_size = int(2 * intermediate_size / 3)
            if self.block_ffn_dim_multiplier is not None:
                intermediate_size = int(self.block_ffn_dim_multiplier * intermediate_size)
                intermediate_size = self.block_multiple_of * (
                    (intermediate_size + self.block_multiple_of - 1) // self.block_multiple_of
                )
        return intermediate_size

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Lfm2Config:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            block_multiple_of=getattr(config, "block_multiple_of", 256),
            block_ffn_dim_multiplier=getattr(config, "block_ffn_dim_multiplier", 1.0),
            block_auto_adjust_ff_dim=getattr(config, "block_auto_adjust_ff_dim", True),
        )


@dataclasses.dataclass
class Lfm2MoeConfig(CausalLMConfig):
    """Configuration for LFM2MoE's dense-prefix and routed-expert feed-forwards."""

    num_dense_layers: int = 2
    use_expert_bias: bool | None = True

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Lfm2MoeConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            num_dense_layers=getattr(config, "num_dense_layers", 2),
            use_expert_bias=getattr(config, "use_expert_bias", True),
        )


@dataclasses.dataclass
class Lfm2VlConfig(Lfm2Config):
    """Configuration for LiquidAI LFM2-VL (SigLIP2 NaFlex + LFM2 decoder).

    The HuggingFace ``Lfm2VlConfig`` is composite: the LFM2 decoder fields
    live under ``text_config`` (which is what ``config`` refers to here) and
    the projector knobs sit on the top-level composite, reachable through
    ``parent_config``.  The SigLIP2 NaFlex geometry is extracted separately
    into :attr:`ArchitectureConfig.vision`.
    """

    downsample_factor: int = 2
    projector_hidden_act: str = "gelu"
    projector_hidden_size: int = 2048
    projector_bias: bool = True
    projector_use_layernorm: bool = False

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Lfm2VlConfig:
        base = Lfm2Config.from_transformers(config, parent_config)
        # Projector fields are top-level on the composite config.
        source = parent_config if parent_config is not None else config
        result = cls(
            **_shallow_fields(base),
            downsample_factor=getattr(source, "downsample_factor", 2),
            projector_hidden_act=getattr(source, "projector_hidden_act", "gelu"),
            projector_hidden_size=getattr(source, "projector_hidden_size", base.hidden_size),
            projector_bias=getattr(source, "projector_bias", True),
            projector_use_layernorm=getattr(source, "projector_use_layernorm", False),
        )
        # Preserve the composite identity; ``config`` is the nested LFM2 text
        # config and would otherwise report ``model_type="lfm2"``.
        result.model_type = "lfm2_vl"
        if result.image_token_id is None:
            result.image_token_id = getattr(source, "image_token_id", None)
        return result


@dataclasses.dataclass
class DFlashConfig(CausalLMConfig):
    """Configuration for the DFlash speculative-decoding draft model.

    DFlash drafters (z-lab/dflash) condition on intermediate target hidden
    states fused into every draft layer.  The HuggingFace ``config.json``
    of a DFlash checkpoint stores DFlash-specific fields under a nested
    ``dflash_config`` dict; we lift them onto the top-level mobius config
    so the rest of the build pipeline (component init, task graph wiring,
    weight loading) can read them through standard attribute access.

    Fields:
        target_layer_ids: 0-based decoder layer indices on the *target*
            model whose post-residual hidden states feed each draft layer.
            Same convention as
            :class:`ArchitectureConfig.output_layer_indices`.
        block_size: Number of mask/draft tokens the drafter consumes per
            speculative step (``b16`` in checkpoint names means 16).
        mask_token_id: Token id used to embed the masked draft positions
            via the target's ``embed_tokens``; ``None`` is allowed and
            indicates a pure noise embedding.
        num_target_layers: Total number of decoder layers on the target
            model.  Used together with ``target_layer_ids`` for runtime
            consistency checks.
    """

    target_layer_ids: list[int] | None = None
    block_size: int | None = None
    mask_token_id: int | None = None
    num_target_layers: int | None = None
    draft_vocab_size: int | None = None
    use_draft_lm_head: bool = False

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> DFlashConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        dflash_cfg = getattr(config, "dflash_config", None) or {}
        if not isinstance(dflash_cfg, dict):
            # Some checkpoints expose ``dflash_config`` as a nested config object.
            dflash_cfg = {
                "target_layer_ids": getattr(dflash_cfg, "target_layer_ids", None),
                "mask_token_id": getattr(dflash_cfg, "mask_token_id", None),
            }
        base_fields = _shallow_fields(base)
        base_fields.pop("mask_token_id", None)
        return cls(
            **base_fields,
            target_layer_ids=dflash_cfg.get("target_layer_ids"),
            block_size=getattr(config, "block_size", None),
            mask_token_id=dflash_cfg.get("mask_token_id"),
            num_target_layers=getattr(config, "num_target_layers", None),
        )


@dataclasses.dataclass
class Qwen35MtpConfig(CausalLMConfig):
    """Configuration for the Qwen3.6 multi-token-prediction (MTP) head.

    The MTP head is a self-speculative drafter shipped inside the dense
    ``Qwen/Qwen3.6-27B`` checkpoint under the ``mtp.*`` weight prefix
    (HuggingFace ``transformers`` discards these on ``from_pretrained``).
    Architecturally it is a single ``full_attention`` Qwen3.5 decoder layer
    preceded by an input projection that fuses the just-emitted token
    embedding with the target model's last hidden state::

        h' = fc(concat[ pre_fc_norm_embedding(embed(input_ids)),
                        pre_fc_norm_hidden(hidden_states) ])

    All standard transformer fields (hidden_size, head_dim, mrope_section,
    partial_rotary_factor, attn_output_gate, …) are read from the parent
    model's ``text_config`` so the reused :class:`Qwen35DecoderLayer`,
    :class:`Qwen35Attention` and mRoPE machinery stay bit-compatible with
    the target.  ``num_hidden_layers`` is forced to ``1`` and
    ``layer_types`` to ``["full_attention"]`` regardless of the parent's
    (64-layer, hybrid) stack — the MTP head has exactly one layer.
    """

    use_dedicated_embeddings: bool = False
    use_dedicated_lm_head: bool = False

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Qwen35MtpConfig:
        hf_config = config
        if hasattr(config, "text_config"):
            hf_config = config.text_config
        base = ArchitectureConfig.from_transformers(
            hf_config, parent_config=parent_config or config
        )
        fields = _shallow_fields(base)
        # The MTP head is a single full-attention layer no matter how deep /
        # hybrid the parent decoder stack is.
        fields["num_hidden_layers"] = 1
        fields["layer_types"] = ["full_attention"]
        return cls(**fields)


@dataclasses.dataclass
class HyV3Config(CausalLMConfig):
    """Configuration for Hunyuan-V3 dense-prefix and routed/shared MoE blocks."""

    enable_moe_fp32_combine: bool = True
    routing_weight_normalization_epsilon: float | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> HyV3Config:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        fields = _shallow_fields(base)
        layer_types = list(
            getattr(config, "mlp_layer_types", None)
            or ["dense"] + ["sparse"] * max(int(base.num_hidden_layers) - 1, 0)
        )
        if len(layer_types) != base.num_hidden_layers:
            raise ValueError("HYV3 mlp_layer_types must have one entry per decoder layer")
        dense_prefix = 0
        while dense_prefix < len(layer_types) and layer_types[dense_prefix] == "dense":
            dense_prefix += 1
        if any(kind != "sparse" for kind in layer_types[dense_prefix:]):
            raise ValueError(
                "HYV3 supports a contiguous leading dense prefix followed by sparse layers"
            )
        num_shared = int(getattr(config, "num_shared_experts", None) or 1)
        moe_width = int(
            getattr(config, "moe_intermediate_size", None) or base.intermediate_size
        )
        fields.update(
            first_k_dense_replace=dense_prefix,
            shared_expert_intermediate_size=moe_width * num_shared,
            n_shared_experts=num_shared,
            routed_scaling_factor=float(
                getattr(config, "router_scaling_factor", None) or base.routed_scaling_factor
            ),
            norm_topk_prob=True,
            routing_weight_normalization_floor=None,
            routing_weight_normalization_epsilon=1e-20,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            use_expert_bias=True,
            disable_qmoe=True,
            attn_qk_norm=True,
            attn_qk_norm_full=False,
            enable_moe_fp32_combine=bool(getattr(config, "enable_moe_fp32_combine", True)),
        )
        return cls(**fields)


@dataclasses.dataclass
class HyV3MtpConfig(HyV3Config):
    """Configuration for one independently cached Hunyuan-V3 NextN block."""

    use_dedicated_embeddings: bool = False
    use_dedicated_lm_head: bool = False


def _speculators_layer_namespace(layer_cfg: dict):
    """Build a config-like namespace from a speculators ``transformer_layer_config``.

    Normalizes the rope fields: speculators checkpoints store either a flat
    ``rope_theta`` (Qwen3) or a nested ``rope_parameters = {rope_theta, ...}``
    (Gemma4), so flatten the latter to ``rope_theta`` for ArchitectureConfig.
    """
    import types

    cfg = dict(layer_cfg)
    if "rope_theta" not in cfg:
        rope = cfg.get("rope_parameters") or cfg.get("rope_scaling")
        if isinstance(rope, dict) and rope.get("rope_theta") is not None:
            cfg["rope_theta"] = rope["rope_theta"]
    return types.SimpleNamespace(**cfg)


@dataclasses.dataclass
class Eagle3Config(CausalLMConfig):
    """Configuration for EAGLE-3 draft checkpoints.

    Two on-disk formats are supported:
      * AngelSlim: a flat llama config with ``draft_vocab_size`` at top level
        (Qwen3-4B/8B); ``norm_before_residual`` absent -> False.
      * speculators (RedHat): the architecture config is nested under
        ``transformer_layer_config``; eagle fields (``draft_vocab_size``,
        ``norm_before_residual``, ``norm_before_fc``, ``target_hidden_size``,
        ``eagle_aux_hidden_state_layer_ids``) sit at the top level.
    """

    draft_vocab_size: int | None = None
    norm_before_residual: bool = False
    norm_before_fc: bool = False
    fc_norm: bool = False
    target_hidden_size: int | None = None
    eagle_aux_hidden_state_layer_ids: list[int] | None = None
    target_layer_ids: list[int] | None = None
    use_target_lm_head: bool = False
    use_draft_token_embedding: bool = False

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Eagle3Config:
        layer_cfg = getattr(config, "transformer_layer_config", None)
        is_speculators_format = layer_cfg is not None
        if layer_cfg is not None:
            # speculators format: arch config nested under transformer_layer_config.
            if isinstance(layer_cfg, dict):
                layer_cfg = _speculators_layer_namespace(layer_cfg)
            base = ArchitectureConfig.from_transformers(layer_cfg, parent_config=config)
        else:
            base = ArchitectureConfig.from_transformers(config, parent_config)
        fields = _shallow_fields(base)
        fields["num_hidden_layers"] = 1
        fields["layer_types"] = ["full_attention"]
        return cls(
            **fields,
            draft_vocab_size=getattr(config, "draft_vocab_size", None),
            norm_before_residual=bool(getattr(config, "norm_before_residual", False)),
            norm_before_fc=bool(getattr(config, "norm_before_fc", False)),
            fc_norm=bool(getattr(config, "fc_norm", False)),
            target_hidden_size=getattr(config, "target_hidden_size", None),
            eagle_aux_hidden_state_layer_ids=getattr(
                config, "eagle_aux_hidden_state_layer_ids", None
            ),
            use_draft_token_embedding=is_speculators_format,
        )


@dataclasses.dataclass
class Gemma2Config(CausalLMConfig):
    """Configuration for Gemma2 models with attention soft-capping.

    Adds ``attn_logit_softcapping``, ``final_logit_softcapping``, and
    ``query_pre_attn_scalar`` used exclusively by :mod:`models.gemma`.
    """

    attn_logit_softcapping: float = 0.0
    final_logit_softcapping: float = 0.0
    query_pre_attn_scalar: float | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Gemma2Config:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            attn_logit_softcapping=(getattr(config, "attn_logit_softcapping", 0.0) or 0.0),
            final_logit_softcapping=(getattr(config, "final_logit_softcapping", 0.0) or 0.0),
            query_pre_attn_scalar=getattr(config, "query_pre_attn_scalar", None),
        )


def _as_int_or_default(value, default: int) -> int:
    """Return ``int(value)``, falling back to *default* only when unset.

    Unlike ``int(value or default)`` this preserves an explicit ``0`` so that
    an out-of-range checkpoint value reaches the config's own validation
    instead of being silently rewritten to the default.
    """
    return default if value is None else int(value)


def _as_float_or_default(value, default: float) -> float:
    """Float counterpart of :func:`_as_int_or_default`."""
    return default if value is None else float(value)


@dataclasses.dataclass
class HrmTextConfig(CausalLMConfig):
    """Configuration for HRM-Text hierarchical recurrent models.

    Mirrors HuggingFace ``HrmTextConfig``. The checkpoint stores the real
    per-stack block count in ``num_hidden_layers``; HuggingFace's
    ``__post_init__`` moves it to ``num_layers_per_stack`` and rewrites
    ``num_hidden_layers`` to the *inflated* total number of unique attention
    invocations under the H/L recurrence::

        num_hidden_layers = num_layers_per_stack * H_cycles * (L_cycles + 1)

    That inflated count is what drives KV-cache slot allocation, so it is what
    :class:`~mobius.tasks.CausalLMTask` must see. This dataclass reproduces the
    same split so that both a trusted ``HrmTextConfig`` instance (already
    inflated, ``num_layers_per_stack`` set) and a raw pinned ``config.json``
    (not inflated, ``num_layers_per_stack`` absent) resolve identically.

    ``embedding_scale`` defaults to ``1 / initializer_range`` exactly as
    upstream does when the checkpoint leaves it unset.
    """

    H_cycles: int = 2
    L_cycles: int = 3
    num_layers_per_stack: int | None = None
    embedding_scale: float | None = None
    initializer_range: float = 0.02
    prefix_lm: bool = True

    def __post_init__(self):
        if self.H_cycles <= 0 or self.L_cycles <= 0:
            raise ValueError(
                f"HrmTextConfig requires positive H_cycles/L_cycles, got "
                f"H_cycles={self.H_cycles}, L_cycles={self.L_cycles}"
            )
        # HRM-Text attention is always MHA: upstream hardcodes
        # ``num_key_value_groups = 1`` and sizes k_proj/v_proj by
        # ``num_attention_heads * head_dim``, ignoring any
        # ``num_key_value_heads`` the checkpoint happens to carry.
        if self.num_attention_heads != DEFAULT_INT:
            self.num_key_value_heads = self.num_attention_heads
        if self.embedding_scale is None:
            if not self.initializer_range:
                raise ValueError(
                    "HrmTextConfig needs a non-zero initializer_range to derive "
                    "embedding_scale when the checkpoint does not supply one."
                )
            self.embedding_scale = 1.0 / self.initializer_range
        if self.num_layers_per_stack is None and self.num_hidden_layers != DEFAULT_INT:
            # Raw-config path: ``num_hidden_layers`` still carries the real
            # per-stack count. Remember it, then inflate exactly as upstream.
            self.num_layers_per_stack = self.num_hidden_layers
            self.num_hidden_layers = (
                self.num_layers_per_stack * self.H_cycles * (self.L_cycles + 1)
            )

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> HrmTextConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        fields = _shallow_fields(base)
        # ``HrmTextRotaryEmbedding`` is unconditional upstream, so this family
        # always uses RoPE. A raw ``config.json`` only carries the default
        # ``rope_theta`` of 10000.0, which the generic extractor deliberately
        # ignores as a RoPE signal (NoPE models inherit it as dead config
        # data), leaving ``rope_type=None`` and silently exporting a
        # position-free graph. Pin the default RoPE explicitly instead.
        if fields.get("rope_type") is None:
            fields["rope_type"] = "default"
            if fields.get("rope_theta") is None:
                fields["rope_theta"] = 10_000.0
            if fields.get("partial_rotary_factor") is None:
                # Upstream rotates the full head_dim.
                fields["partial_rotary_factor"] = 1.0
        return cls(
            **fields,
            H_cycles=_as_int_or_default(getattr(config, "H_cycles", None), 2),
            L_cycles=_as_int_or_default(getattr(config, "L_cycles", None), 3),
            # Present only on a trusted HF config that already ran its own
            # ``__post_init__``; ``None`` for a raw ``config.json``.
            num_layers_per_stack=getattr(config, "num_layers_per_stack", None),
            embedding_scale=getattr(config, "embedding_scale", None),
            initializer_range=_as_float_or_default(
                getattr(config, "initializer_range", None), 0.02
            ),
            prefix_lm=bool(getattr(config, "prefix_lm", True)),
        )


@dataclasses.dataclass
class NanoChatConfig(CausalLMConfig):
    """Configuration for NanoChat models with final logit soft-capping.

    Adds ``final_logit_softcapping`` used by :mod:`models.nanochat`.
    """

    final_logit_softcapping: float = 0.0

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> NanoChatConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            final_logit_softcapping=(getattr(config, "final_logit_softcapping", 0.0) or 0.0),
        )


@dataclasses.dataclass
class LongcatFlashConfig(CausalLMConfig):
    """Configuration for LongCat Flash dual-sublayer models.

    Adds ``zero_expert_num`` for identity/pass-through MoE experts.
    Unlike standard MoE, LongCat uses a fixed shortcut MoE block per
    physical layer alongside two dense sub-attentions and two dense MLPs.
    """

    zero_expert_num: int = 0

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> LongcatFlashConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        # LongCat uses ffn_hidden_size for dense MLP (not the generic intermediate_size)
        ffn_hidden_size = getattr(config, "ffn_hidden_size", None)
        if ffn_hidden_size is not None:
            base = dataclasses.replace(base, intermediate_size=ffn_hidden_size)
        # LongCat uses moe_topk (not num_experts_per_tok)
        moe_topk = getattr(config, "moe_topk", None)
        if moe_topk is not None:
            base = dataclasses.replace(base, num_experts_per_tok=moe_topk)
        # LongCat uses expert_ffn_hidden_size (not moe_intermediate_size)
        expert_ffn_hidden_size = getattr(config, "expert_ffn_hidden_size", None)
        if expert_ffn_hidden_size is not None:
            base = dataclasses.replace(base, moe_intermediate_size=expert_ffn_hidden_size)
        return cls(
            **_shallow_fields(base),
            zero_expert_num=getattr(config, "zero_expert_num", 0),
        )


@dataclasses.dataclass
class Gemma3nConfig(CausalLMConfig):
    """Configuration for Gemma3n models with AltUp and Laurel compression.

    Adds AltUp prediction/correction parameters and per-layer input
    dimension fields used exclusively by :mod:`models.gemma3n`.

    ``num_kv_shared_layers`` mirrors the Gemma4 field of the same name: the
    last N decoder layers borrow K,V from the last non-shared layer of the
    same attention type instead of projecting their own, so they own no KV
    cache entry.  E4B ships 15 (of 35 layers), i.e. layers 20..34 are shared.

    ``activation_sparsity_pattern`` holds a per-layer target sparsity for the
    MLP gate branch.  Where it is non-zero the gate activations below a
    Gaussian quantile cutoff are zeroed (see
    :class:`~mobius.models.gemma3n.Gemma3nMLP`).  E4B ships 0.95 for layers
    0..9 and 0.0 for the rest; ``None`` disables sparsity everywhere.

    ``final_logit_softcapping`` tanh-caps the LM head output as in Gemma2
    (``cap * tanh(logits / cap)``).  Every published Gemma 3n config ships
    30.0; ``0.0`` disables it.
    """

    altup_num_inputs: int = 4
    altup_active_idx: int = 0
    altup_correct_scale: bool = True
    laurel_rank: int = 64
    hidden_size_per_layer_input: int = 256
    vocab_size_per_layer_input: int = 262_144
    num_kv_shared_layers: int = 0
    activation_sparsity_pattern: list[float] | None = None
    final_logit_softcapping: float = 0.0

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Gemma3nConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        sparsity = getattr(config, "activation_sparsity_pattern", None)
        return cls(
            **_shallow_fields(base),
            altup_num_inputs=getattr(config, "altup_num_inputs", 4),
            altup_active_idx=getattr(config, "altup_active_idx", 0),
            altup_correct_scale=getattr(config, "altup_correct_scale", True),
            laurel_rank=getattr(config, "laurel_rank", 64),
            hidden_size_per_layer_input=getattr(config, "hidden_size_per_layer_input", 256),
            vocab_size_per_layer_input=getattr(config, "vocab_size_per_layer_input", 262_144),
            num_kv_shared_layers=getattr(config, "num_kv_shared_layers", 0) or 0,
            activation_sparsity_pattern=(
                [float(s) for s in sparsity] if sparsity is not None else None
            ),
            final_logit_softcapping=(getattr(config, "final_logit_softcapping", 0.0) or 0.0),
        )


@dataclasses.dataclass
class Gemma3nMultiModalConfig(Gemma3nConfig):
    """Configuration for the full Gemma 3n image + audio + text model.

    Carries the same text-decoder fields as :class:`Gemma3nConfig` (from which
    it inherits, since the decoder is unchanged) plus the multimodal wiring.
    The towers live in the inherited ``vision`` (:class:`VisionConfig`, a
    MobileNet-V5 encoder rather than SigLIP) and ``audio``
    (:class:`Gemma3nAudioConfig`, a USM Conformer) sub-configs, populated by
    the ``gemma3n`` extractor hooks.

    Both modalities are projected into the decoder's embedding space and
    spliced in at their placeholder token positions, so each needs its token
    id and its fixed soft-token count.  The token ids come from the inherited
    ``image_token_id`` / ``audio_token_id`` (262145 and 262273 for E4B, also
    mirrored on the sub-configs per the Gemma4 convention); the per-image
    counts are fixed because both towers emit a fixed-size feature map:

    - ``vision_soft_tokens_per_image``: 256 — a 768x768 image becomes a 16x16
      grid.
    - ``audio_soft_tokens_per_image``: 188.

    Audio is optional: checkpoints without an ``audio_config`` leave ``audio``
    as ``None`` and the exported package omits the audio encoder.
    """

    vision_soft_tokens_per_image: int = 256
    audio_soft_tokens_per_image: int = 188

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Gemma3nMultiModalConfig:
        # The HF Gemma3nConfig wraps a text_config while the multimodal token
        # ids and soft-token counts live on the outer config.  build() may hand
        # us either one, so resolve in both directions.
        text_config = getattr(config, "text_config", None) or config
        composite = parent_config if parent_config is not None else config
        base = Gemma3nConfig.from_transformers(text_config, composite)
        # audio_token_id is an ArchitectureConfig field that no generic
        # extractor populates for gemma3n (the audio hook puts it on the
        # sub-config); lift it so tasks can read it at the top level.
        if base.audio_token_id is None:
            base = dataclasses.replace(
                base, audio_token_id=getattr(composite, "audio_token_id", None)
            )
        return cls(
            **_shallow_fields(base),
            vision_soft_tokens_per_image=int(
                getattr(composite, "vision_soft_tokens_per_image", 256) or 256
            ),
            audio_soft_tokens_per_image=int(
                getattr(composite, "audio_soft_tokens_per_image", 188) or 188
            ),
        )


@dataclasses.dataclass
class MllamaConfig(VisionLanguageConfig):
    """Configuration for Mllama (Llama 3.2 Vision) cross-attention models.

    Adds ``cross_attention_layers`` specifying which decoder layers use
    cross-attention with vision features.
    """

    cross_attention_layers: list[int] | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> MllamaConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            cross_attention_layers=getattr(config, "cross_attention_layers", None),
        )


@dataclasses.dataclass
class Gemma4Config(VisionLanguageConfig):
    """Configuration for Gemma4 multimodal models.

    Extends :class:`VisionLanguageConfig` with Gemma4-specific text decoder
    fields.  Vision config lives in the inherited ``vision`` sub-config
    (:class:`VisionConfig`) and audio config in the ``audio`` sub-config
    (:class:`Gemma4AudioConfig`).

    Text decoder specifics:
    - ``global_head_dim``: head dimension for full-attention (global) layers
      (512 for Gemma4), which differs from the local-sliding head_dim (256).
    - ``global_rope_theta``: RoPE base frequency for full-attention layers
      (1_000_000 for Gemma4).
    - ``global_partial_rotary_factor``: fraction of head_dim to rotate for
      global attention (0.25, so rotary_dim = 512 * 0.25 = 128).
    - ``num_global_key_value_heads``: KV head count for full-attention layers
      (4 for Gemma4 31B).  When ``None``, all layers use
      ``num_key_value_heads``.  HF calls this ``num_global_key_value_heads``
      and gates it behind ``attention_k_eq_v``.
    - ``hidden_size_per_layer_input``: per-layer input gating dimension
      (256 for Gemma4); 0 disables per-layer input entirely.
    - ``vocab_size_per_layer_input``: vocabulary size for per-layer embeddings.
    - ``num_kv_shared_layers``: how many layers share KV projections with the
      next layer (20 for the 27B variant; 0 if disabled).
    - ``use_double_wide_mlp``: whether the MLP intermediate size is doubled
      relative to the standard multiplier.
    - ``final_logit_softcapping``: tanh soft-cap applied to final logits
      (30.0 for Gemma4); 0.0 disables it.
    - ``attn_logit_softcapping``: tanh soft-cap applied to attention QK
      logits before softmax (50.0 for Gemma4); 0.0 disables it.
      Maps directly to the ``softcap`` attribute of the ONNX Attention op
      (opset 24), so no manual Tanh/scale ops are needed.
    - ``enable_moe_block``: whether any layers use MoE routing.
      MoE hyper-parameters (``num_local_experts``, ``num_experts_per_tok``,
      ``moe_intermediate_size``) are inherited from :class:`ArchitectureConfig`.
    """

    global_head_dim: int | None = None
    global_rope_theta: float = 1_000_000.0
    global_partial_rotary_factor: float = 0.25
    num_global_key_value_heads: int | None = None
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int = 0
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False
    final_logit_softcapping: float = 0.0
    attn_logit_softcapping: float = 0.0
    enable_moe_block: bool = False
    attention_k_eq_v: bool = False
    boa_token_id: int | None = None
    use_bidirectional_attention: str | None = None
    """Bidirectional attention mode for the text decoder.

    Mirrors HF ``Gemma4TextConfig.use_bidirectional_attention``:
    - ``None``: fully causal (smaller Gemma4 models, e.g. E2B).
    - ``"vision"``: text stays causal, but contiguous image-token blocks
      attend bidirectionally within each block (larger models, e.g.
      12B/26B/32B). Implemented via a per-position ``block_sequence_ids``
      overlay added onto the causal mask. Audio placeholders are *not*
      included (HF marks audio as token-type 3, excluded from the vision
      block mask), so audio tokens keep causal attention.
    - ``"all"``: HF mode where every token attends bidirectionally. Not used
      by any currently supported Gemma4 model and not implemented here; the
      decoder raises ``NotImplementedError`` rather than silently degrading to
      causal attention (only ``None`` and ``"vision"`` are accepted).
    """

    # Set to True by Gemma4Task.build() when the target EP's max_buffer_size
    # is too small for the fused [V, L*D] per-layer embedding table, to split
    # it into L separate [V, D] tables that each fit within the EP's buffer limit.
    split_per_layer_embedding: bool = False

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Gemma4Config:
        base = ArchitectureConfig.from_transformers(config, parent_config)

        # Gemma4 encodes the full/local pattern as a single integer
        # (sliding_window_pattern) rather than a list.  Convert it to
        # layer_types if not already set by the base extractor.
        if base.layer_types is None:
            sliding_window_pattern = getattr(config, "sliding_window_pattern", None)
            if sliding_window_pattern is not None and base.num_hidden_layers:
                # Every sliding_window_pattern-th layer (1-indexed) is full attention;
                # all others use sliding-window attention.
                layer_types = []
                for i in range(base.num_hidden_layers):
                    if (i + 1) % sliding_window_pattern == 0:
                        layer_types.append("full_attention")
                    else:
                        layer_types.append("sliding_attention")
                base = dataclasses.replace(base, layer_types=layer_types)

        # MoE fields — map Gemma4 names to ArchitectureConfig fields
        num_local_experts = getattr(config, "num_experts", None)
        num_experts_per_tok = getattr(config, "top_k_experts", None)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", None)
        if num_local_experts is not None:
            base = dataclasses.replace(base, num_local_experts=num_local_experts)
        if num_experts_per_tok is not None:
            base = dataclasses.replace(base, num_experts_per_tok=num_experts_per_tok)
        if moe_intermediate_size is not None:
            base = dataclasses.replace(base, moe_intermediate_size=moe_intermediate_size)

        # Extract dual RoPE parameters from rope_parameters dict.
        # NOTE: Gemma4TextConfig exposes rope_parameters == rope_scaling, both being a
        # nested dict keyed by layer type.  The generic _extract_rope_config extractor
        # picks up full_attention.rope_theta (1_000_000) via _nested_rope_theta, making
        # the base rope_theta wrong for local/sliding layers.  Correct it here.
        rope_params = getattr(config, "rope_parameters", {}) or {}
        full_rope = (
            rope_params.get("full_attention", {}) if isinstance(rope_params, dict) else {}
        )
        sliding_rope = (
            rope_params.get("sliding_attention", {}) if isinstance(rope_params, dict) else {}
        )
        if "rope_theta" in sliding_rope:
            # Override with the correct sliding-attention theta (e.g. 10_000 for E2B/E4B).
            base = dataclasses.replace(base, rope_theta=float(sliding_rope["rope_theta"]))

        num_global_kv = getattr(config, "num_global_key_value_heads", None)
        global_head_dim = getattr(config, "global_head_dim", None)
        if global_head_dim is None or num_global_kv is None:
            per_layer_config = getattr(config, "per_layer_config", None)
            layer_types = getattr(config, "layer_types", None)
            if per_layer_config and layer_types:
                layer_configs = list(
                    per_layer_config.values()
                    if isinstance(per_layer_config, dict)
                    else per_layer_config
                )
                if len(layer_types) == len(layer_configs):
                    full_head_dims = {
                        (
                            layer_config.get("head_dim")
                            if isinstance(layer_config, dict)
                            else getattr(layer_config, "head_dim", None)
                        )
                        for layer_type, layer_config in zip(
                            layer_types, layer_configs, strict=True
                        )
                        if layer_type == "full_attention"
                    }
                    full_head_dims.discard(None)
                    if global_head_dim is None and len(full_head_dims) == 1:
                        global_head_dim = next(iter(full_head_dims))
                    full_kv_heads = {
                        (
                            layer_config.get("num_key_value_heads")
                            if isinstance(layer_config, dict)
                            else getattr(layer_config, "num_key_value_heads", None)
                        )
                        for layer_type, layer_config in zip(
                            layer_types, layer_configs, strict=True
                        )
                        if layer_type == "full_attention"
                    }
                    full_kv_heads.discard(None)
                    if num_global_kv is None and len(full_kv_heads) == 1:
                        num_global_kv = next(iter(full_kv_heads))

        return cls(
            **_shallow_fields(base),
            global_head_dim=global_head_dim,
            global_rope_theta=float(full_rope.get("rope_theta", 1_000_000.0)),
            global_partial_rotary_factor=float(full_rope.get("partial_rotary_factor", 0.25)),
            num_global_key_value_heads=num_global_kv,
            hidden_size_per_layer_input=int(
                getattr(config, "hidden_size_per_layer_input", 0) or 0
            ),
            vocab_size_per_layer_input=int(
                getattr(config, "vocab_size_per_layer_input", 0) or 0
            ),
            num_kv_shared_layers=getattr(config, "num_kv_shared_layers", 0) or 0,
            use_double_wide_mlp=getattr(config, "use_double_wide_mlp", False),
            final_logit_softcapping=(getattr(config, "final_logit_softcapping", 0.0) or 0.0),
            attn_logit_softcapping=(getattr(config, "attn_logit_softcapping", 0.0) or 0.0),
            enable_moe_block=getattr(config, "enable_moe_block", False),
            attention_k_eq_v=getattr(config, "attention_k_eq_v", False),
            boa_token_id=getattr(parent_config, "boa_token_id", None),
            use_bidirectional_attention=getattr(config, "use_bidirectional_attention", None),
        )


@dataclasses.dataclass
class Gemma4AssistantConfig(Gemma4Config):
    """Configuration for the Gemma4-Assistant MTP draft model.

    The HuggingFace Gemma4-Assistant checkpoint
    (``google/gemma-4-{E2B,E4B,12B,26B,31B}-it-assistant``) is a small
    Gemma4-style decoder that is hooked up to a target Gemma4 model for
    speculative decoding.  Its HF config layout is::

        Gemma4AssistantConfig
        ├── text_config: Gemma4TextConfig   ← all the standard Gemma4 fields
        ├── backbone_hidden_size: int        ← target model's hidden size
        ├── use_ordered_embeddings: bool
        ├── num_centroids: int
        └── centroid_intermediate_top_k: int

    We flatten this into a single mobius dataclass by extracting the
    nested ``text_config`` fields onto the top level (via
    :meth:`Gemma4Config.from_transformers`) and adding the assistant-
    specific fields below.

    Constraints enforced by the HF config (mirrored here):
    - All draft layers must be KV-shared with the target — i.e.
      ``num_kv_shared_layers == num_hidden_layers``.  The drafter has no
      KV cache of its own; per-layer K/V is fed in from the target's
      shared K/V buffers at inference time.
    - ``hidden_size_per_layer_input == 0`` (no per-layer input gating).
    - ``enable_moe_block == False`` (no MoE).
    - ``use_double_wide_mlp == False``.
    - ``vocab_size_per_layer_input == 0``.

    Fields (assistant-specific):
        backbone_hidden_size: Hidden size of the target model the
            assistant was trained against (so the assistant's
            ``pre_projection`` and ``post_projection`` know the right
            input/output dims for the shared hidden state).
        use_ordered_embeddings: When True, the assistant routes its
            output through a centroid-based ordered-embedding LM head
            (``Gemma4AssistantMaskedEmbedder``), built by
            ``Gemma4AssistantCausalLMModel``.
        num_centroids: Number of centroids used by the ordered-embedding
            head when ``use_ordered_embeddings`` is True.
        centroid_intermediate_top_k: Top-K centroid count for the
            ordered-embedding head.
    """

    backbone_hidden_size: int = 1536
    use_ordered_embeddings: bool = False
    num_centroids: int = 2048
    centroid_intermediate_top_k: int = 32

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Gemma4AssistantConfig:
        # The HF Gemma4AssistantConfig nests the Gemma4 text config under
        # ``text_config``.  Drive Gemma4Config.from_transformers on that
        # nested config to lift all the standard text-decoder fields onto
        # the top level, then layer on the assistant-specific knobs.
        #
        # NOTE: ``build()`` unwraps ``config.text_config`` BEFORE calling
        # us when the wrapper has a ``text_config`` attribute (see
        # _builder.py:370-382).  In that case ``config`` is the unwrapped
        # Gemma4TextConfig and the assistant-specific fields
        # (use_ordered_embeddings, backbone_hidden_size, num_centroids,
        # centroid_intermediate_top_k) live on ``parent_config`` instead.
        # Resolve from whichever object has them.
        text_cfg = getattr(config, "text_config", None) or config
        base = Gemma4Config.from_transformers(text_cfg, parent_config=parent_config)

        def _resolve(name, default):
            for src in (config, parent_config):
                if src is not None and hasattr(src, name):
                    val = getattr(src, name)
                    if val is not None:
                        return val
            return default

        return cls(
            **_shallow_fields(base),
            backbone_hidden_size=int(_resolve("backbone_hidden_size", 1536)),
            use_ordered_embeddings=bool(_resolve("use_ordered_embeddings", False)),
            num_centroids=int(_resolve("num_centroids", 2048)),
            centroid_intermediate_top_k=int(_resolve("centroid_intermediate_top_k", 32)),
        )

    def validate(self) -> None:
        super().validate()
        errors: list[str] = []
        if self.num_kv_shared_layers != self.num_hidden_layers:
            errors.append(
                "Gemma4-Assistant requires every layer to be KV-shared with the "
                f"target: num_kv_shared_layers ({self.num_kv_shared_layers}) "
                f"must equal num_hidden_layers ({self.num_hidden_layers})."
            )
        if self.hidden_size_per_layer_input:
            errors.append(
                "Gemma4-Assistant does not support per-layer input gating; "
                f"hidden_size_per_layer_input must be 0, got {self.hidden_size_per_layer_input}."
            )
        if self.enable_moe_block:
            errors.append(
                "Gemma4-Assistant does not support MoE layers; enable_moe_block must be False."
            )
        if self.use_double_wide_mlp:
            errors.append(
                "Gemma4-Assistant does not support double-wide MLP; "
                "use_double_wide_mlp must be False."
            )
        if self.vocab_size_per_layer_input:
            errors.append(
                "Gemma4-Assistant does not support per-layer vocab; "
                f"vocab_size_per_layer_input must be 0, got {self.vocab_size_per_layer_input}."
            )
        if errors:
            raise ValueError(
                "Invalid Gemma4AssistantConfig:\n" + "\n".join(f"  - {e}" for e in errors)
            )


@dataclasses.dataclass
class YolosConfig(EncoderConfig):
    """Configuration for YOLOS object detection models.

    Adds ``num_detection_tokens`` for the learned detection token count.
    ``num_labels`` remains on :class:`ArchitectureConfig` (shared with
    :class:`SegformerConfig`).
    """

    num_detection_tokens: int = 100
    # YOLOS uses a rectangular input resolution (e.g. [800, 1333] for
    # yolos-tiny). The base ArchitectureConfig collapses ``image_size`` to a
    # single int (height), which would size the learned position embeddings
    # for a square image and mismatch the pretrained weights. Preserve both
    # dimensions here so the patch grid (and position-embedding length) is
    # computed correctly.
    image_height: int = 800
    image_width: int = 1333

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> YolosConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        raw_image_size = getattr(config, "image_size", [base.image_size, base.image_size])
        if isinstance(raw_image_size, dict):
            height = int(raw_image_size.get("height", base.image_size))
            width = int(raw_image_size.get("width", base.image_size))
        elif isinstance(raw_image_size, (list, tuple)):
            height = int(raw_image_size[0])
            width = int(raw_image_size[-1])
        else:
            height = width = int(raw_image_size)
        return cls(
            **_shallow_fields(base),
            num_detection_tokens=getattr(config, "num_detection_tokens", 100),
            image_height=height,
            image_width=width,
        )


@dataclasses.dataclass
class DepthAnythingConfig(ArchitectureConfig):
    """Configuration for Depth Anything DPT depth estimation models.

    Adds DPT neck, reassembly, and fusion head parameters.
    """

    neck_hidden_sizes: list[int] | None = None
    reassemble_factors: list[float] | None = None
    fusion_hidden_size: int = 64
    head_hidden_size: int = 32
    backbone_out_indices: list[int] | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> DepthAnythingConfig:
        # HF DepthAnythingConfig stores backbone (ViT) fields on a nested
        # backbone_config (e.g. Dinov2Config).  Extract it so the base
        # ArchitectureConfig resolver can find hidden_size, num_heads, etc.
        backbone = getattr(config, "backbone_config", config)
        base = ArchitectureConfig.from_transformers(backbone, parent_config)
        return cls(
            **_shallow_fields(base),
            neck_hidden_sizes=getattr(config, "neck_hidden_sizes", None),
            reassemble_factors=getattr(config, "reassemble_factors", None),
            fusion_hidden_size=getattr(config, "fusion_hidden_size", 64),
            head_hidden_size=getattr(config, "head_hidden_size", 32),
            # backbone_out_indices lives on the backbone config as out_indices
            backbone_out_indices=(
                getattr(backbone, "out_indices", None)
                or getattr(config, "backbone_out_indices", None)
            ),
        )


@dataclasses.dataclass
class SegformerConfig(EncoderConfig):
    """Configuration for Segformer hierarchical vision transformers.

    Adds per-stage encoder parameters and decode head hidden size.
    ``num_labels`` remains on :class:`ArchitectureConfig`.
    """

    segformer_hidden_sizes: list[int] | None = None
    segformer_num_attention_heads: list[int] | None = None
    segformer_depths: list[int] | None = None
    segformer_sr_ratios: list[int] | None = None
    segformer_mlp_ratios: list[int] | None = None
    segformer_patch_sizes: list[int] | None = None
    segformer_strides: list[int] | None = None
    decoder_hidden_size: int = 256

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> SegformerConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            segformer_hidden_sizes=getattr(config, "segformer_hidden_sizes", None),
            segformer_num_attention_heads=getattr(
                config, "segformer_num_attention_heads", None
            ),
            segformer_depths=getattr(config, "segformer_depths", None),
            segformer_sr_ratios=getattr(config, "segformer_sr_ratios", None),
            segformer_mlp_ratios=getattr(config, "segformer_mlp_ratios", None),
            segformer_patch_sizes=getattr(config, "segformer_patch_sizes", None),
            segformer_strides=getattr(config, "segformer_strides", None),
            decoder_hidden_size=getattr(config, "decoder_hidden_size", 256),
        )


@dataclasses.dataclass
class Sam2Config(ArchitectureConfig):
    """Configuration for SAM2 Hiera backbone vision models.

    Adds Hiera-specific per-stage dimensions, block counts, and FPN
    parameters.
    """

    sam2_embed_dims: list[int] | None = None
    sam2_blocks_per_stage: list[int] | None = None
    sam2_num_heads_per_stage: list[int] | None = None
    sam2_mlp_ratio: float | None = None
    sam2_fpn_hidden_size: int | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Sam2Config:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        # SAM2 uses gelu in its backbone MLP but doesn't expose a single
        # hidden_act field — default to gelu.
        if base.hidden_act is None:
            base = dataclasses.replace(base, hidden_act="gelu")
        return cls(
            **_shallow_fields(base),
            sam2_embed_dims=getattr(config, "sam2_embed_dims", None),
            sam2_blocks_per_stage=getattr(config, "sam2_blocks_per_stage", None),
            sam2_num_heads_per_stage=getattr(config, "sam2_num_heads_per_stage", None),
            sam2_mlp_ratio=getattr(config, "sam2_mlp_ratio", None),
            sam2_fpn_hidden_size=getattr(config, "sam2_fpn_hidden_size", None),
        )


@dataclasses.dataclass
class MambaConfig(BaseModelConfig):
    """Configuration for Mamba SSM (Selective State Space) models.

    Mamba replaces transformer attention with a selective scan mechanism.
    Fields map to HuggingFace ``MambaConfig``.

    State carried per layer:
        conv_state: (batch, d_inner, conv_kernel - 1)
        ssm_state:  (batch, d_inner, state_size)
    """

    state_size: int = 16
    conv_kernel: int = 4
    expand: int = 2
    time_step_rank: int = 48
    layer_norm_epsilon: float = 1e-5
    use_conv_bias: bool = True
    residual_in_fp32: bool = True

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> MambaConfig:
        del parent_config  # unused
        expand = getattr(config, "expand", 2)
        d_inner = getattr(config, "intermediate_size", 0)
        if not d_inner:
            d_inner = config.hidden_size * expand

        tr = getattr(config, "time_step_rank", 48)
        if tr == "auto":
            import math

            tr = math.ceil(config.hidden_size / 16)

        options = dict(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=d_inner,
            num_hidden_layers=config.num_hidden_layers,
            pad_token_id=getattr(config, "pad_token_id", 0),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", True),
            state_size=getattr(config, "state_size", 16),
            conv_kernel=getattr(config, "conv_kernel", 4),
            expand=expand,
            time_step_rank=tr,
            layer_norm_epsilon=getattr(config, "layer_norm_epsilon", 1e-5),
            use_conv_bias=getattr(config, "use_conv_bias", True),
            residual_in_fp32=getattr(config, "residual_in_fp32", True),
        )

        # Model dtype
        resolved = _resolve_dtype(config)
        if resolved is not None:
            options["dtype"] = resolved

        return cls(**options)


@dataclasses.dataclass
class Mamba2Config(BaseModelConfig):
    """Configuration for standalone Mamba2/SSD models.

    Pure Mamba2 (no attention, no MLP). Each layer is:
        RMSNorm -> Mamba2Block -> residual add

    State per layer:
        conv_state: (batch, conv_dim, d_conv - 1)
        ssm_state:  (batch, num_heads, head_dim, state_size)

    HuggingFace reference: ``Mamba2Config``.
    """

    num_heads: int = 128
    head_dim: int = 64
    state_size: int = 128
    n_groups: int = 8
    conv_kernel: int = 4
    expand: int = 2
    layer_norm_epsilon: float = 1e-5
    use_conv_bias: bool = True
    norm_before_gate: bool = True
    chunk_size: int = 256

    def __post_init__(self):
        # Mamba2 requires d_inner = num_heads * head_dim.
        if self.intermediate_size != DEFAULT_INT:
            expected = self.num_heads * self.head_dim
            if self.intermediate_size != expected:
                raise ValueError(
                    f"Mamba2Config: intermediate_size ({self.intermediate_size}) "
                    f"must equal num_heads * head_dim "
                    f"({self.num_heads} * {self.head_dim} = {expected})."
                )

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Mamba2Config:
        del parent_config  # unused
        expand = getattr(config, "expand", 2)
        d_inner = getattr(config, "intermediate_size", 0)
        if not d_inner:
            d_inner = config.hidden_size * expand

        num_heads = getattr(config, "num_heads", 128)
        head_dim = getattr(config, "head_dim", "auto")
        if head_dim == "auto":
            if d_inner % num_heads != 0:
                raise ValueError(
                    f"Mamba2Config: d_inner ({d_inner}) must be divisible "
                    f"by num_heads ({num_heads}) to compute head_dim."
                )
            head_dim = d_inner // num_heads

        options = dict(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=d_inner,
            num_hidden_layers=config.num_hidden_layers,
            pad_token_id=getattr(config, "pad_token_id", 0),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=getattr(config, "state_size", 128),
            n_groups=getattr(config, "n_groups", 8),
            conv_kernel=getattr(config, "conv_kernel", 4),
            expand=expand,
            layer_norm_epsilon=getattr(config, "layer_norm_epsilon", 1e-5),
            use_conv_bias=getattr(config, "use_conv_bias", True),
            norm_before_gate=getattr(config, "norm_before_gate", True),
            chunk_size=getattr(config, "chunk_size", 256),
        )

        # Model dtype
        resolved = _resolve_dtype(config)
        if resolved is not None:
            options["dtype"] = resolved

        return cls(**options)


@dataclasses.dataclass
class JambaConfig(ArchitectureConfig):
    """Configuration for Jamba hybrid SSM+Attention models.

    Jamba interleaves Mamba SSM layers with Transformer attention layers.
    Some layers use MoE (multiple expert MLPs) instead of dense MLP.

    Layer type selection:
        - Attention if ``(i - attn_layer_offset) % attn_layer_period == 0``
        - Mamba otherwise
        - MoE MLP if ``(i - expert_layer_offset) % expert_layer_period == 0``
        - Dense MLP otherwise
    """

    # Mamba SSM parameters
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dt_rank: int = 256
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False

    # Layer interleaving
    attn_layer_period: int = 8
    attn_layer_offset: int = 4
    expert_layer_period: int = 2
    expert_layer_offset: int = 1
    # GGUF serializes the resolved schedule through tensor presence rather than
    # preserving the source period/offset pair.
    expert_layer_indices: list[int] | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> JambaConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)

        # Build layer_types list for HybridCausalLMTask
        n = base.num_hidden_layers
        attn_period = getattr(config, "attn_layer_period", 8)
        attn_offset = getattr(config, "attn_layer_offset", 4)
        expert_period = getattr(config, "expert_layer_period", 2)
        expert_offset = getattr(config, "expert_layer_offset", 1)
        if attn_period <= 0 or not 0 <= attn_offset < attn_period:
            raise ValueError("Jamba attn_layer_offset must be in [0, attn_layer_period)")
        if expert_period <= 0 or not 0 <= expert_offset < expert_period:
            raise ValueError("Jamba expert_layer_offset must be in [0, expert_layer_period)")
        layer_types = []
        for i in range(n):
            if i % attn_period == attn_offset:
                layer_types.append("full_attention")
            else:
                layer_types.append("mamba")

        num_experts = getattr(config, "num_experts", 16)
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)
        dt_rank = getattr(config, "mamba_dt_rank", "auto")
        if dt_rank == "auto":
            dt_rank = math.ceil(base.hidden_size / 16)

        # Exclude fields we set explicitly below to avoid duplicate keyword args
        _exclude = {
            "layer_types",
            "num_local_experts",
            "num_experts_per_tok",
            "norm_topk_prob",
            "rope_type",
        }
        base_fields = {k: v for k, v in _shallow_fields(base).items() if k not in _exclude}
        return cls(
            **base_fields,
            layer_types=layer_types,
            # HF uses "num_experts"; we use inherited "num_local_experts"
            num_local_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            mamba_d_state=getattr(config, "mamba_d_state", 16),
            mamba_d_conv=getattr(config, "mamba_d_conv", 4),
            mamba_expand=getattr(config, "mamba_expand", 2),
            mamba_dt_rank=int(dt_rank),
            mamba_conv_bias=getattr(config, "mamba_conv_bias", True),
            mamba_proj_bias=getattr(config, "mamba_proj_bias", False),
            attn_layer_period=attn_period,
            attn_layer_offset=attn_offset,
            expert_layer_period=expert_period,
            expert_layer_offset=expert_offset,
            expert_layer_indices=[i for i in range(n) if i % expert_period == expert_offset],
            norm_topk_prob=False,
            rope_type=None,
        )


@dataclasses.dataclass
class BambaConfig(ArchitectureConfig):
    """Configuration for Bamba hybrid Mamba2+Attention models.

    Uses multi-head Mamba2/SSD layers interleaved with attention layers.
    """

    mamba_n_heads: int = 128
    mamba_d_head: int = 64
    mamba_d_state: int = 256
    mamba_n_groups: int = 1
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> BambaConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)

        n = base.num_hidden_layers
        attn_indices = set(getattr(config, "attn_layer_indices", None) or [])
        layers_block_type = getattr(config, "layers_block_type", None)

        layer_types: list[str] = []
        for i in range(n):
            if layers_block_type and i < len(layers_block_type):
                ltype = layers_block_type[i]
                layer_types.append("full_attention" if ltype == "attention" else "mamba2")
            elif i in attn_indices:
                layer_types.append("full_attention")
            else:
                layer_types.append("mamba2")

        mamba_expand = getattr(config, "mamba_expand", 2)
        d_inner = config.hidden_size * mamba_expand

        mamba_n_heads = getattr(config, "mamba_n_heads", 128)
        mamba_d_head = getattr(config, "mamba_d_head", "auto")
        if mamba_d_head == "auto":
            mamba_d_head = d_inner // mamba_n_heads

        # Exclude layer_types from base fields — we built it explicitly above
        base_fields = {k: v for k, v in _shallow_fields(base).items() if k != "layer_types"}
        return cls(
            **base_fields,
            layer_types=layer_types,
            mamba_n_heads=mamba_n_heads,
            mamba_d_head=mamba_d_head,
            mamba_d_state=getattr(config, "mamba_d_state", 256),
            mamba_n_groups=getattr(config, "mamba_n_groups", 1),
            mamba_d_conv=getattr(config, "mamba_d_conv", 4),
            mamba_expand=mamba_expand,
            mamba_conv_bias=getattr(config, "mamba_conv_bias", True),
            mamba_proj_bias=getattr(config, "mamba_proj_bias", False),
        )


@dataclasses.dataclass
class FalconH1Config(ArchitectureConfig):
    """Configuration for Falcon-H1 parallel Attention + Mamba2 decoder layers."""

    mamba_d_ssm: int = 1024
    mamba_n_heads: int = 128
    mamba_d_head: int = 8
    mamba_n_groups: int = 1
    mamba_d_state: int = 256
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_chunk_size: int = 256
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False
    mamba_norm_before_gate: bool = True
    mamba_rms_norm: bool = False
    time_step_limit: tuple[float, float] = (0.0, float("inf"))
    attention_bias: bool = False
    projectors_bias: bool = False
    lm_head_multiplier: float = 1.0
    embedding_multiplier: float = 1.0
    mlp_multipliers: tuple[float, float] = (1.0, 1.0)
    key_multiplier: float = 1.0
    attention_out_multiplier: float = 1.0
    attention_in_multiplier: float = 1.0
    ssm_multipliers: tuple[float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    ssm_in_multiplier: float = 1.0
    ssm_out_multiplier: float = 1.0

    def __post_init__(self) -> None:
        # Architecture-specific fields win when enabled; attention_bias remains
        # the Hugging Face compatibility fallback for callers that only expose it.
        self.attn_qkv_bias = self.attn_qkv_bias or self.attention_bias
        self.attn_o_bias = self.attn_o_bias or self.attention_bias
        if self.hidden_act != "silu":
            raise ValueError("Falcon-H1 supports only hidden_act='silu'")
        if (
            self.head_dim <= 0
            or self.num_attention_heads <= 0
            or self.hidden_size != self.num_attention_heads * self.head_dim
        ):
            raise ValueError("Falcon-H1 hidden_size must equal num_attention_heads * head_dim")
        if (
            self.num_key_value_heads <= 0
            or self.num_attention_heads % self.num_key_value_heads
        ):
            raise ValueError("Falcon-H1 num_key_value_heads must divide num_attention_heads")
        if self.mamba_d_ssm <= 0 or self.mamba_n_heads <= 0:
            raise ValueError("Falcon-H1 Mamba dimensions must be positive")
        if self.mamba_d_ssm % self.mamba_n_heads:
            raise ValueError("mamba_n_heads must divide mamba_d_ssm")
        if self.mamba_d_head * self.mamba_n_heads != self.mamba_d_ssm:
            raise ValueError("mamba_d_head * mamba_n_heads must equal mamba_d_ssm")
        if (
            self.mamba_n_groups <= 0
            or self.mamba_n_heads % self.mamba_n_groups
            or self.mamba_d_ssm % self.mamba_n_groups
        ):
            raise ValueError("mamba_n_groups must divide both mamba_n_heads and mamba_d_ssm")
        if self.mamba_d_state <= 0 or self.mamba_d_conv <= 0 or self.mamba_chunk_size <= 0:
            raise ValueError("Falcon-H1 state, convolution, and chunk sizes must be positive")
        if len(self.mlp_multipliers) != 2:
            raise ValueError("mlp_multipliers must contain exactly two values")
        if len(self.ssm_multipliers) != 5:
            raise ValueError("ssm_multipliers must contain exactly five values")
        if len(self.time_step_limit) != 2:
            raise ValueError("time_step_limit must contain exactly two values")
        time_step_min, time_step_max = self.time_step_limit
        if time_step_min < 0 or time_step_max < time_step_min:
            raise ValueError("time_step_limit must be ordered and non-negative")
        multipliers = (
            self.embedding_multiplier,
            self.lm_head_multiplier,
            self.attention_in_multiplier,
            self.attention_out_multiplier,
            self.key_multiplier,
            self.ssm_in_multiplier,
            self.ssm_out_multiplier,
            *self.mlp_multipliers,
            *self.ssm_multipliers,
        )
        if not all(math.isfinite(value) for value in multipliers):
            raise ValueError("Falcon-H1 multipliers must all be finite")

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> FalconH1Config:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        fields = _shallow_fields(base)
        fields.pop("embedding_multiplier", None)
        fields["mlp_bias"] = bool(getattr(config, "mlp_bias", False))
        d_ssm = getattr(config, "mamba_d_ssm", None)
        if d_ssm is None:
            d_ssm = int(getattr(config, "mamba_expand", 2)) * base.hidden_size
        n_heads = int(getattr(config, "mamba_n_heads", 128))
        d_head = getattr(config, "mamba_d_head", "auto")
        if d_head == "auto":
            if d_ssm % n_heads:
                raise ValueError("mamba_n_heads must divide mamba_d_ssm")
            d_head = d_ssm // n_heads
        time_step_limit = getattr(config, "time_step_limit", None) or (
            0.0,
            float("inf"),
        )
        return cls(
            **fields,
            mamba_d_ssm=int(d_ssm),
            mamba_n_heads=n_heads,
            mamba_d_head=int(d_head),
            mamba_n_groups=int(getattr(config, "mamba_n_groups", 1)),
            mamba_d_state=int(getattr(config, "mamba_d_state", 256)),
            mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
            mamba_expand=int(getattr(config, "mamba_expand", 2)),
            mamba_chunk_size=int(getattr(config, "mamba_chunk_size", 256)),
            mamba_conv_bias=bool(getattr(config, "mamba_conv_bias", True)),
            mamba_proj_bias=bool(getattr(config, "mamba_proj_bias", False)),
            mamba_norm_before_gate=bool(getattr(config, "mamba_norm_before_gate", True)),
            mamba_rms_norm=bool(getattr(config, "mamba_rms_norm", False)),
            time_step_limit=tuple(float(value) for value in time_step_limit),
            attention_bias=bool(getattr(config, "attention_bias", False)),
            projectors_bias=bool(getattr(config, "projectors_bias", False)),
            lm_head_multiplier=float(getattr(config, "lm_head_multiplier", 1.0)),
            embedding_multiplier=float(getattr(config, "embedding_multiplier", 1.0)),
            mlp_multipliers=tuple(
                float(value)
                for value in (getattr(config, "mlp_multipliers", None) or (1.0, 1.0))
            ),
            key_multiplier=float(getattr(config, "key_multiplier", 1.0)),
            attention_out_multiplier=float(getattr(config, "attention_out_multiplier", 1.0)),
            attention_in_multiplier=float(getattr(config, "attention_in_multiplier", 1.0)),
            ssm_multipliers=tuple(
                float(value)
                for value in (
                    getattr(config, "ssm_multipliers", None) or (1.0, 1.0, 1.0, 1.0, 1.0)
                )
            ),
            ssm_in_multiplier=float(getattr(config, "ssm_in_multiplier", 1.0)),
            ssm_out_multiplier=float(getattr(config, "ssm_out_multiplier", 1.0)),
        )


@dataclasses.dataclass
class Plamo2Config(ArchitectureConfig):
    """Configuration for PLaMo2 alternating Mamba/attention decoder layers."""

    attention_head_counts: tuple[int, ...] = ()
    attention_kv_head_counts: tuple[int, ...] = ()
    mamba_num_heads: int = 32
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_dt_rank: int = 128
    mamba_group_count: int = 0
    attention_window_size: int = 2048
    use_predefined_initial_state: bool = False

    def __post_init__(self) -> None:
        if self.hidden_act != "silu":
            raise ValueError("PLaMo2 supports only hidden_act='silu'")
        if not math.isclose(self.rms_norm_eps, 1e-6, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("PLaMo2 requires rms_norm_eps=1e-6")
        if self.mamba_group_count != 0:
            raise ValueError("PLaMo2 supports only ssm.group_count=0")
        if self.use_predefined_initial_state:
            raise ValueError("PLaMo2 predefined initial state is unsupported")
        if len(self.attention_head_counts) != self.num_hidden_layers:
            raise ValueError(
                "PLaMo2 attention_head_counts must contain exactly num_hidden_layers entries"
            )
        if len(self.attention_kv_head_counts) != self.num_hidden_layers:
            raise ValueError(
                "PLaMo2 attention_kv_head_counts must contain exactly num_hidden_layers entries"
            )
        layer_types: list[str] = []
        for layer, (heads, kv_heads) in enumerate(
            zip(self.attention_head_counts, self.attention_kv_head_counts)
        ):
            if kv_heads == 0:
                if heads != 0:
                    raise ValueError(
                        f"PLaMo2 layer {layer} has head_count={heads} but head_count_kv=0"
                    )
                layer_types.append("mamba")
            else:
                if heads <= 0 or heads % kv_heads:
                    raise ValueError(
                        f"PLaMo2 layer {layer} has invalid attention head geometry "
                        f"head_count={heads}, head_count_kv={kv_heads}"
                    )
                if heads != self.num_attention_heads or kv_heads != self.num_key_value_heads:
                    raise ValueError(
                        "PLaMo2 currently requires one shared attention geometry across "
                        "all attention layers"
                    )
                layer_types.append("full_attention")
        if (
            not layer_types
            or "mamba" not in layer_types
            or "full_attention" not in layer_types
        ):
            raise ValueError("PLaMo2 requires both Mamba and attention layers")
        self.layer_types = layer_types
        if self.head_dim <= 0 or self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("PLaMo2 hidden_size must equal num_attention_heads * head_dim")
        if self.mamba_num_heads <= 0:
            raise ValueError("PLaMo2 mamba_num_heads must be positive")
        if self.mamba_d_state <= 0 or self.mamba_d_conv <= 1 or self.mamba_dt_rank <= 0:
            raise ValueError(
                "PLaMo2 Mamba state, convolution, and dt dimensions must be positive"
            )
        if self.attention_window_size <= 0:
            raise ValueError("PLaMo2 attention_window_size must be positive")

    @property
    def mamba_inner_size(self) -> int:
        return self.mamba_num_heads * self.head_dim

    @property
    def mamba_head_dim(self) -> int:
        return self.head_dim

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Plamo2Config:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        fields = _shallow_fields(base)
        fields["hidden_act"] = getattr(config, "hidden_act", None) or "silu"
        fields["rope_type"] = getattr(config, "rope_type", None) or "default"
        fields["rope_theta"] = float(
            getattr(config, "rope_local_theta", getattr(config, "rope_theta", 10_000.0))
        )
        fields["head_dim"] = int(getattr(config, "hidden_size_per_head", base.head_dim))
        fields["partial_rotary_factor"] = 1.0
        if getattr(config, "full_attention_idx", None):
            raise ValueError("PLaMo2 full_attention_idx layers are unsupported")
        layers = base.num_hidden_layers
        explicit_heads = getattr(config, "attention_head_counts", None)
        explicit_kv_heads = getattr(config, "attention_kv_head_counts", None)
        if explicit_heads is not None or explicit_kv_heads is not None:
            if explicit_heads is None or explicit_kv_heads is None:
                raise ValueError(
                    "PLaMo2 per-layer attention head arrays must be supplied together"
                )
            head_counts = tuple(int(value) for value in explicit_heads)
            kv_head_counts = tuple(int(value) for value in explicit_kv_heads)
        else:
            mamba_enabled = bool(getattr(config, "mamba_enabled", True))
            mamba_step = int(getattr(config, "mamba_step", 2))
            if mamba_enabled and mamba_step <= 1:
                raise ValueError("PLaMo2 mamba_step must be greater than one")

            def is_mamba(layer: int) -> bool:
                if not mamba_enabled:
                    return False
                if layers <= mamba_step // 2:
                    return layer != layers - 1
                return layer % mamba_step != mamba_step // 2

            head_counts = tuple(
                0 if is_mamba(i) else base.num_attention_heads for i in range(layers)
            )
            kv_head_counts = tuple(
                0 if is_mamba(i) else base.num_key_value_heads for i in range(layers)
            )

        return cls(
            **fields,
            attention_head_counts=head_counts,
            attention_kv_head_counts=kv_head_counts,
            mamba_num_heads=int(getattr(config, "mamba_num_heads", 32)),
            mamba_d_state=int(getattr(config, "mamba_d_state", 64)),
            mamba_d_conv=int(getattr(config, "mamba_d_conv", 4)),
            mamba_dt_rank=int(
                getattr(config, "mamba_dt_rank", max(64, base.hidden_size // 16))
            ),
            mamba_group_count=int(getattr(config, "mamba_group_count", 0)),
            attention_window_size=int(
                getattr(
                    config,
                    "attention_window_size",
                    getattr(config, "sliding_window", 2048),
                )
            ),
            use_predefined_initial_state=bool(
                getattr(config, "use_predefined_initial_state", False)
            ),
        )


@dataclasses.dataclass
class GraniteMoeHybridConfig(BambaConfig):
    """Configuration for GraniteMoeHybrid: Mamba2+Attention hybrid with MoE on all layers.

    Extends BambaConfig with ``shared_intermediate_size`` for the dense shared MLP
    that runs alongside the routed MoE block on every layer.
    """

    shared_intermediate_size: int = 1024

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> GraniteMoeHybridConfig:
        # Reuse BambaConfig.from_transformers for mamba fields, MoE/RoPE/multiplier
        # extraction, then rebuild layer_types from GraniteMoeHybrid's own naming.
        bamba = BambaConfig.from_transformers(config, parent_config)
        bamba_fields = _shallow_fields(bamba)

        # GraniteMoeHybrid names layers "full_attention" / "linear_attention"
        # (linear_attention == Mamba2/SSD), unlike Bamba's "attention" / "mamba".
        raw_layer_types = (
            getattr(config, "layer_types", None)
            or getattr(config, "layers_block_type", None)
            or []
        )
        _attn = {"full_attention", "attention"}
        _mamba = {"linear_attention", "mamba", "mamba2"}
        layer_types: list[str] = []
        for ltype in raw_layer_types:
            if ltype in _attn:
                layer_types.append("full_attention")
            elif ltype in _mamba:
                layer_types.append("mamba2")
            else:
                raise ValueError(f"Unknown GraniteMoeHybrid layer type: {ltype!r}")
        if layer_types:
            bamba_fields["layer_types"] = layer_types

        # Respect position_embedding_type: GraniteMoeHybrid checkpoints ship
        # default ``rope_parameters`` even for the NoPE variant
        # (granite-4.0-tiny-preview: position_embedding_type='nope'). Only apply
        # RoPE when explicitly requested; otherwise disable it so
        # ``initialize_rope`` returns None and attention runs NoPE.
        if getattr(config, "position_embedding_type", "rope") != "rope":
            bamba_fields["rope_type"] = None

        return cls(
            **bamba_fields,
            shared_intermediate_size=getattr(config, "shared_intermediate_size", 1024),
        )


@dataclasses.dataclass
class Zamba2Config(ArchitectureConfig):
    """Configuration for Zamba2 hybrid Mamba2+Attention models.

    Zamba2 is a hybrid architecture where most layers are Mamba2 (SSM) and a
    subset are "hybrid" layers containing BOTH a shared attention block AND a
    Mamba2 block.  The attention block is tied (shared weights) across all
    hybrid layers.

    In the ONNX representation, each physical hybrid layer is expanded into
    two logical layers:
    - ``"full_attention"`` for the shared transformer (attention + MLP + linear)
    - ``"mamba2"`` for the Mamba block with transformer injection

    This keeps the cache system aligned (one entry per logical layer).
    """

    hidden_act: str = "gelu"

    mamba_n_heads: int = 8
    mamba_d_head: int = 64
    mamba_d_state: int = 64
    mamba_n_groups: int = 1
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False
    mamba_time_step_min: float = 0.001

    # Zamba2-specific: attention operates on 2*hidden_size input
    attention_hidden_size: int = DEFAULT_INT

    # Number of physical hybrid layers (for weight sharing bookkeeping)
    num_mem_blocks: int = 1

    # Low-rank adapter rank for per-layer differentiation
    adapter_rank: int = 128

    # Physical layer indices that are hybrid (for preprocess_weights mapping)
    hybrid_layer_indices: list[int] | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Zamba2Config:
        base = ArchitectureConfig.from_transformers(config, parent_config)

        # Extract physical layer types from HF config
        layers_block_type = getattr(config, "layers_block_type", None) or []
        hybrid_layer_indices = [i for i, t in enumerate(layers_block_type) if t == "hybrid"]

        # Expand to logical layer_types: hybrid → [full_attention, mamba2]
        layer_types: list[str] = []
        for t in layers_block_type:
            if t == "hybrid":
                layer_types.append("full_attention")
                layer_types.append("mamba2")
            else:
                layer_types.append("mamba2")

        num_hidden_layers = len(layer_types)

        mamba_expand = getattr(config, "mamba_expand", 2)
        d_inner = config.hidden_size * mamba_expand
        n_mamba_heads = getattr(config, "n_mamba_heads", 8)
        mamba_headdim = getattr(config, "mamba_headdim", None)
        if mamba_headdim is None:
            mamba_headdim = d_inner // n_mamba_heads

        attention_hidden_size = getattr(
            config, "attention_hidden_size", 2 * config.hidden_size
        )
        # head_dim for attention is based on attention_hidden_size
        head_dim = attention_hidden_size // config.num_attention_heads

        base_fields = {
            k: v
            for k, v in _shallow_fields(base).items()
            if k not in ("layer_types", "num_hidden_layers", "head_dim")
        }
        return cls(
            **base_fields,
            num_hidden_layers=num_hidden_layers,
            head_dim=head_dim,
            layer_types=layer_types,
            mamba_n_heads=n_mamba_heads,
            mamba_d_head=mamba_headdim,
            mamba_d_state=getattr(config, "mamba_d_state", 64),
            mamba_n_groups=getattr(config, "mamba_ngroups", 1),
            mamba_d_conv=getattr(config, "mamba_d_conv", 4),
            mamba_expand=mamba_expand,
            mamba_conv_bias=getattr(config, "use_conv_bias", True),
            mamba_proj_bias=getattr(config, "add_bias_linear", False),
            mamba_time_step_min=getattr(config, "time_step_min", 0.001),
            attention_hidden_size=attention_hidden_size,
            num_mem_blocks=getattr(config, "num_mem_blocks", 1),
            adapter_rank=getattr(config, "adapter_rank", 128),
            hybrid_layer_indices=hybrid_layer_indices,
        )


@dataclasses.dataclass
class NemotronHConfig(ArchitectureConfig):
    """Configuration for NemotronH hybrid Mamba2+Attention+MLP models.

    Uses multi-head Mamba2/SSD layers interleaved with attention and
    standalone MLP layers.  Each layer is a single-mixer block
    (RMSNorm → mixer → residual).
    """

    mamba_n_heads: int = 128
    mamba_d_head: int = 64
    mamba_d_state: int = 128
    mamba_n_groups: int = 8
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False
    mamba_time_step_min: float = 0.001
    moe_latent_size: int | None = None

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> NemotronHConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)

        # Get layer types from layers_block_type or hybrid_override_pattern
        layers_block_type = getattr(config, "layers_block_type", None)
        if layers_block_type is None:
            pattern = getattr(config, "hybrid_override_pattern", "")
            # Map pattern chars: M=mamba2, *=full_attention, -=mlp, E=moe
            char_map = {
                "M": "mamba2",
                "*": "full_attention",
                "-": "mlp",
                "E": "moe",
            }
            layers_block_type = [char_map.get(c, "mamba2") for c in pattern]
        else:
            # Convert HF names to mobius names
            type_map = {
                "mamba": "mamba2",
                "attention": "full_attention",
                "moe": "moe",
            }
            layers_block_type = [type_map.get(t, t) for t in layers_block_type]

        # Override num_hidden_layers based on actual pattern length
        n = len(layers_block_type) if layers_block_type else base.num_hidden_layers

        mamba_expand = getattr(config, "expand", getattr(config, "mamba_expand", 2))
        d_inner = config.hidden_size * mamba_expand

        mamba_n_heads = getattr(config, "mamba_num_heads", 128)
        mamba_d_head = getattr(config, "mamba_head_dim", "auto")
        if mamba_d_head == "auto":
            mamba_d_head = d_inner // mamba_n_heads

        # Exclude fields we set explicitly to avoid duplicate keyword args
        base_fields = {
            k: v
            for k, v in _shallow_fields(base).items()
            if k
            not in (
                "layer_types",
                "num_hidden_layers",
                "hidden_act",
                "moe_latent_size",
                "shared_expert_intermediate_size",
            )
        }

        # Extract shared expert intermediate size (NemotronH uses a dedicated
        # field name different from the base ArchitectureConfig default).
        shared_expert_intermediate_size = getattr(
            config,
            "moe_shared_expert_intermediate_size",
            base.shared_expert_intermediate_size,
        )

        return cls(
            **base_fields,
            num_hidden_layers=n,
            layer_types=layers_block_type,
            hidden_act="relu2",
            mamba_n_heads=mamba_n_heads,
            mamba_d_head=mamba_d_head,
            mamba_d_state=getattr(config, "ssm_state_size", 128),
            mamba_n_groups=getattr(config, "n_groups", 8),
            mamba_d_conv=getattr(config, "conv_kernel", 4),
            mamba_expand=mamba_expand,
            mamba_conv_bias=getattr(config, "use_conv_bias", True),
            mamba_proj_bias=getattr(config, "mamba_proj_bias", False),
            mamba_time_step_min=getattr(config, "time_step_min", 0.001),
            moe_latent_size=getattr(config, "moe_latent_size", None),
            shared_expert_intermediate_size=shared_expert_intermediate_size,
        )


@dataclasses.dataclass
class JetMoeConfig(CausalLMConfig):
    """Configuration for JetMoE: Mixture-of-Attention + MoE FFN model.

    JetMoE uses ``kv_channels`` as the per-head key/value dimension rather
    than deriving it from ``hidden_size // num_attention_heads``.  The
    standard formula gives the wrong answer because ``num_attention_heads``
    is the *total* Q head count (``top_k * num_kv_heads``), not the KV head
    count.  We therefore read ``kv_channels`` directly from the HF config
    and store it as ``head_dim``.
    """

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> JetMoeConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        # Override head_dim to use kv_channels directly, not hidden/num_heads.
        kv_channels = getattr(config, "kv_channels", base.head_dim)
        # Also map num_kv_heads → num_key_value_heads if present (HF JetMoE
        # uses num_kv_heads instead of the standard num_key_value_heads).
        num_kv = getattr(config, "num_kv_heads", None)
        base_fields = _shallow_fields(base)
        base_fields["head_dim"] = kv_channels
        if num_kv is not None:
            base_fields["num_key_value_heads"] = num_kv
        return cls(**base_fields)


@dataclasses.dataclass
class GlmAsrConfig(CausalLMConfig):
    """Configuration for GLM-ASR audio-language models."""

    projector_hidden_act: str = "gelu"

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> GlmAsrConfig:
        composite = parent_config or config
        text_config = getattr(composite, "text_config", None) or config
        base = ArchitectureConfig.from_transformers(text_config, parent_config=composite)
        fields = _shallow_fields(base)
        fields.update(
            model_type=getattr(composite, "model_type", "glmasr"),
            audio_token_id=getattr(composite, "audio_token_id", base.audio_token_id),
            tie_word_embeddings=getattr(
                composite, "tie_word_embeddings", base.tie_word_embeddings
            ),
        )
        resolved_dtype = _resolve_dtype(composite)
        if resolved_dtype is not None:
            fields["dtype"] = resolved_dtype
        return cls(
            **fields,
            projector_hidden_act=getattr(composite, "projector_hidden_act", "gelu"),
        )


@dataclasses.dataclass
class SenseNovaU1Config(CausalLMConfig):
    """Configuration for SenseNova-U1.5 ``neo_chat`` (NEO-unify) models.

    NEO-unify is a *native* unified any-to-any architecture: a single
    Qwen3 backbone carries two complete sets of transformer weights
    ("Mixture of Transformers").  The understanding branch consumes text
    and reference-image tokens; the ``_mot_gen`` branch consumes noisy
    image tokens during flow-matching sampling.  The text-decoder fields
    are lifted from the nested ``llm_config``; the extra fields below
    describe the spatial rotary axes and the flow-matching image head.

    Attributes:
        rope_theta_hw: RoPE base for the spatial (height/width) axes.
            ``rope_theta`` (inherited) drives the temporal/text axis.
        max_position_embeddings_hw: Position limit for the spatial axes.
        patch_size: ViT patch size (pixels per patch side).
        downsample_ratio: Patch-merge ratio; ``1 / downsample_ratio`` is
            the merge factor applied by the ``dense_embedding`` conv.
        use_pixel_head: When true the flow-matching head is a
            pixel-shuffle ``ConvDecoder`` predicting RGB directly (no VAE).
        fm_head_dim / fm_head_layers / fm_head_mlp_ratio: Geometry of the
            deep ``SimpleMLPAdaLN`` head, used only when
            ``fm_head_layers > 2`` (not the case for the released model).
        add_noise_scale_embedding: Add a second sinusoidal embedder whose
            input is the resolution-dependent noise scale.
        noise_scale / noise_scale_mode / noise_scale_base_image_seq_len /
        noise_scale_max_value: Noise-scale schedule parameters.
        timestep_shift / time_schedule: Flow-matching timestep warping.
        t_eps: Lower clamp on ``1 - t`` when converting the head's
            x0-prediction into a velocity.
    """

    rope_theta_hw: float = 10_000.0
    max_position_embeddings_hw: int = 10_000
    patch_size: int = 16
    downsample_ratio: float = 0.5
    use_pixel_head: bool = True
    fm_head_dim: int = 1536
    fm_head_layers: int = 2
    fm_head_mlp_ratio: float = 1.0
    add_noise_scale_embedding: bool = True
    noise_scale: float = 1.0
    noise_scale_mode: str = "resolution"
    noise_scale_base_image_seq_len: int = 64
    noise_scale_max_value: float = 16.0
    timestep_shift: float = 1.0
    time_schedule: str = "standard"
    t_eps: float = 0.05
    frequency_embedding_size: int = 256

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> SenseNovaU1Config:
        # ``neo_chat`` is a composite config: the Qwen3 text-decoder fields
        # live under ``llm_config`` while the flow-matching / patchify
        # fields sit at the top level next to ``vision_config``.
        composite = parent_config or config
        llm_config = getattr(composite, "llm_config", None) or config
        base = ArchitectureConfig.from_transformers(llm_config, parent_config=composite)
        fields = _shallow_fields(base)
        fields.update(
            model_type=getattr(composite, "model_type", "neo_chat"),
            tie_word_embeddings=bool(getattr(composite, "tie_word_embeddings", False)),
        )
        resolved_dtype = _resolve_dtype(composite)
        if resolved_dtype is not None:
            fields["dtype"] = resolved_dtype

        vision_config = getattr(composite, "vision_config", None)

        def _pick(name: str, default):
            # Prefer the top-level value, then the vision sub-config, then
            # the llm sub-config; the released config.json spreads these
            # three groups across all three levels.
            for source in (composite, vision_config, llm_config):
                if source is None:
                    continue
                value = getattr(source, name, None)
                if value is not None:
                    return value
            return default

        return cls(
            **{
                **fields,
                "rope_theta_hw": float(_pick("rope_theta_hw", 10_000.0)),
                "max_position_embeddings_hw": int(_pick("max_position_embeddings_hw", 10_000)),
                "patch_size": int(_pick("patch_size", 16)),
                "downsample_ratio": float(_pick("downsample_ratio", 0.5)),
                "use_pixel_head": bool(_pick("use_pixel_head", True)),
                "fm_head_dim": int(_pick("fm_head_dim", 1536)),
                "fm_head_layers": int(_pick("fm_head_layers", 2)),
                "fm_head_mlp_ratio": float(_pick("fm_head_mlp_ratio", 1.0)),
                "add_noise_scale_embedding": bool(_pick("add_noise_scale_embedding", True)),
                "noise_scale": float(_pick("noise_scale", 1.0)),
                "noise_scale_mode": str(_pick("noise_scale_mode", "resolution")),
                "noise_scale_base_image_seq_len": int(
                    _pick("noise_scale_base_image_seq_len", 64)
                ),
                "noise_scale_max_value": float(_pick("noise_scale_max_value", 16.0)),
                "timestep_shift": float(_pick("timestep_shift", 1.0)),
                "time_schedule": str(_pick("time_schedule", "standard")),
                "t_eps": float(_pick("t_eps", 0.05)),
            }
        )

    @property
    def merge_size(self) -> int:
        """Patch-merge factor applied by ``dense_embedding`` (2 for 0.5)."""
        return round(1.0 / self.downsample_ratio)

    @property
    def pixels_per_token(self) -> int:
        """Image-token stride in pixels (``patch_size * merge_size`` = 32)."""
        return self.patch_size * self.merge_size


@dataclasses.dataclass
class SpeechToTextConfig(ArchitectureConfig):
    """Shared configuration contract for encoder-decoder speech models."""

    encoder_input_name: str = "input_features"
    encoder_input_channels: int | None = None
    encoder_uses_attention_mask: bool = False
    decoder_uses_encoder_attention_mask: bool = False
    decoder_start_token_id: int | None = None
    layer_norm_eps: float = 1e-5


@dataclasses.dataclass
class WhisperConfig(SpeechToTextConfig):
    """Configuration for Whisper encoder-decoder models."""

    encoder_layers: int = DEFAULT_INT
    encoder_attention_heads: int = DEFAULT_INT
    encoder_ffn_dim: int = DEFAULT_INT
    num_mel_bins: int = 80
    max_source_positions: int = 1500
    max_target_positions: int = 448
    scale_embedding: bool = False

    def __post_init__(self):
        if self.encoder_input_channels is None:
            self.encoder_input_channels = self.num_mel_bins
        elif self.encoder_input_channels != self.num_mel_bins:
            raise ValueError(
                "WhisperConfig: encoder_input_channels "
                f"({self.encoder_input_channels}) must equal num_mel_bins "
                f"({self.num_mel_bins})."
            )
        # The decoder's learned position table is the model's context bound;
        # Whisper spells it `max_target_positions`, so mirror it onto the
        # architecture-wide field consumers read.
        if self.max_position_embeddings == DEFAULT_INT:
            self.max_position_embeddings = self.max_target_positions

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> WhisperConfig:
        if config.model_type != "whisper":
            raise ValueError(
                f"WhisperConfig expects model_type='whisper', got '{config.model_type}'"
            )

        d_model = getattr(config, "d_model", config.hidden_size)
        decoder_heads = getattr(config, "decoder_attention_heads", config.num_attention_heads)

        options = dict(
            vocab_size=config.vocab_size,
            hidden_size=d_model,
            intermediate_size=getattr(config, "decoder_ffn_dim", 4 * d_model),
            num_hidden_layers=config.decoder_layers,
            num_attention_heads=decoder_heads,
            num_key_value_heads=decoder_heads,
            head_dim=d_model // decoder_heads,
            hidden_act=getattr(config, "activation_function", "gelu"),
            pad_token_id=getattr(config, "pad_token_id", 0),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", True),
            attn_qkv_bias=True,
            attn_o_bias=True,
            encoder_layers=config.encoder_layers,
            encoder_attention_heads=getattr(config, "encoder_attention_heads", decoder_heads),
            encoder_ffn_dim=getattr(config, "encoder_ffn_dim", 4 * d_model),
            num_mel_bins=getattr(config, "num_mel_bins", 80),
            max_source_positions=getattr(config, "max_source_positions", 1500),
            max_target_positions=getattr(config, "max_target_positions", 448),
            scale_embedding=getattr(config, "scale_embedding", False),
            decoder_start_token_id=getattr(config, "decoder_start_token_id", None),
            bos_token_id=getattr(config, "bos_token_id", None),
            eos_token_id=getattr(config, "eos_token_id", None),
        )

        # Model dtype
        resolved = _resolve_dtype(config)
        if resolved is not None:
            options["dtype"] = resolved

        return cls(**options)


@dataclasses.dataclass
class MoonshineConfig(SpeechToTextConfig):
    """Configuration for Moonshine raw-waveform encoder-decoder ASR models."""

    encoder_input_name: str = "input_values"
    encoder_input_channels: int | None = None
    encoder_uses_attention_mask: bool = True
    decoder_uses_encoder_attention_mask: bool = True
    encoder_num_hidden_layers: int = DEFAULT_INT
    encoder_num_attention_heads: int = DEFAULT_INT
    encoder_num_key_value_heads: int = DEFAULT_INT
    encoder_hidden_act: str = "gelu"
    decoder_hidden_act: str = "silu"

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> MoonshineConfig:
        if config.model_type != "moonshine":
            raise ValueError(
                f"MoonshineConfig expects model_type='moonshine', got '{config.model_type}'"
            )

        hidden_size = config.hidden_size
        decoder_heads = config.decoder_num_attention_heads
        encoder_heads = config.encoder_num_attention_heads
        rope_parameters = getattr(config, "rope_parameters", None) or getattr(
            config, "rope_scaling", None
        )
        rope_type = (rope_parameters or {}).get("rope_type", "default")
        options = dict(
            vocab_size=config.vocab_size,
            hidden_size=hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.decoder_num_hidden_layers,
            num_attention_heads=decoder_heads,
            num_key_value_heads=getattr(config, "decoder_num_key_value_heads", decoder_heads),
            head_dim=hidden_size // decoder_heads,
            hidden_act=getattr(config, "decoder_hidden_act", "silu"),
            pad_token_id=getattr(config, "pad_token_id", 2),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", True),
            attn_qkv_bias=getattr(config, "attention_bias", False),
            attn_o_bias=getattr(config, "attention_bias", False),
            max_position_embeddings=getattr(config, "max_position_embeddings", 194),
            rope_type=rope_type,
            rope_theta=getattr(
                config,
                "rope_theta",
                (rope_parameters or {}).get("rope_theta", 10_000.0),
            ),
            rope_scaling=rope_parameters,
            partial_rotary_factor=getattr(
                config,
                "partial_rotary_factor",
                (rope_parameters or {}).get("partial_rotary_factor", 0.9),
            ),
            rope_interleave=True,
            mlp_bias=True,
            encoder_num_hidden_layers=config.encoder_num_hidden_layers,
            encoder_num_attention_heads=encoder_heads,
            encoder_num_key_value_heads=getattr(
                config, "encoder_num_key_value_heads", encoder_heads
            ),
            encoder_hidden_act=getattr(config, "encoder_hidden_act", "gelu"),
            decoder_hidden_act=getattr(config, "decoder_hidden_act", "silu"),
            decoder_start_token_id=getattr(config, "decoder_start_token_id", 1),
            layer_norm_eps=getattr(config, "layer_norm_eps", 1e-5),
            model_type="moonshine",
            bos_token_id=getattr(config, "bos_token_id", 1),
            eos_token_id=getattr(config, "eos_token_id", 2),
        )
        resolved = _resolve_dtype(config)
        if resolved is not None:
            options["dtype"] = resolved
        return cls(**options)


def _conv_widths(config, defaults, hidden_size: int) -> tuple[int, ...]:
    """Per-layer channel widths of a wav2vec2-family convolutional feature encoder.

    ``conv_dim``, ``conv_kernel`` and ``conv_stride`` describe one conv stack, so
    the depth they imply has to agree. M-CTC-T states its single subsampling
    convolution through ``conv_kernel``/``conv_stride`` alone and never publishes
    ``conv_dim``; inheriting wav2vec2's seven-layer default there would describe a
    stack the checkpoint does not have. Size the widths to the declared depth
    instead, preferring the checkpoint's own ``conv_channels`` when present.
    """
    declared = getattr(config, "conv_dim", None)
    if declared:
        return tuple(declared)
    depth = len(tuple(getattr(config, "conv_kernel", None) or defaults.conv_kernel))
    if depth == len(defaults.conv_dim):
        return defaults.conv_dim
    channels = getattr(config, "conv_channels", None)
    if channels is None:
        return (hidden_size,) * depth
    if isinstance(channels, (list, tuple)):
        return tuple(channels)
    return (int(channels),) * depth


@dataclasses.dataclass
class MMSConfig(ArchitectureConfig):
    """Configuration for MMS (Massively Multilingual Speech) CTC models.

    Extends ``ArchitectureConfig`` with the adapter parameters used in
    ``facebook/mms-1b-all`` and related checkpoints.  When ``add_adapter=True``
    the adapter layers are included in the exported ONNX graph; set this after
    calling ``model.load_adapter(lang_code)`` to bake a specific language's
    weights into the model.

    HuggingFace class: ``Wav2Vec2ForCTC`` with ``config.model_type == "wav2vec2"``

    The convolutional feature-encoder geometry (``conv_dim``/``conv_kernel``/
    ``conv_stride``) is *not* boilerplate: it fixes the waveform-to-frame
    downsampling ratio, so it must come from the checkpoint rather than from a
    hard-coded default.  ``facebook/wav2vec2-base-960h`` downsamples by 320
    (``prod(conv_stride)``) and disables conv bias, while
    ``facebook/mms-1b-all`` enables it.
    """

    add_adapter: bool = False
    output_hidden_size: int = 0  # 0 → use hidden_size
    adapter_kernel_size: int = 3
    adapter_stride: int = 2
    num_adapter_layers: int = 3

    # Convolutional feature encoder (raw waveform → frames).
    conv_dim: tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_kernel: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_bias: bool = False
    feat_extract_norm: str = "group"

    # Transformer encoder shape.
    do_stable_layer_norm: bool = False
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    layer_norm_eps: float = 1e-5

    def __post_init__(self):
        if self.output_hidden_size == 0:
            self.output_hidden_size = self.hidden_size
        # Normalize sequence fields so downstream code can index them freely
        # regardless of whether the checkpoint used a list or a tuple.
        self.conv_dim = tuple(self.conv_dim)
        self.conv_kernel = tuple(self.conv_kernel)
        self.conv_stride = tuple(self.conv_stride)
        if not (len(self.conv_dim) == len(self.conv_kernel) == len(self.conv_stride)):
            raise ValueError(
                "conv_dim, conv_kernel and conv_stride must have equal length; got "
                f"{len(self.conv_dim)}, {len(self.conv_kernel)}, {len(self.conv_stride)}"
            )
        if self.feat_extract_norm not in ("group", "layer"):
            raise ValueError(
                f"feat_extract_norm must be 'group' or 'layer', got {self.feat_extract_norm!r}"
            )

    def feature_extract_output_length(self, num_samples: int) -> int:
        """Return the frame count the conv stack emits for *num_samples*.

        Mirrors ``Wav2Vec2PreTrainedModel._get_feat_extract_output_lengths``:
        each conv applies ``floor((L - kernel) / stride) + 1``.  Callers use it
        to segment a padded batch back into per-row transcripts.
        """
        length = num_samples
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            length = (length - kernel) // stride + 1
        if self.add_adapter:
            for _ in range(self.num_adapter_layers):
                length = (length - 1) // self.adapter_stride + 1
        return length

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> MMSConfig:
        """Extract MMSConfig from a HuggingFace Wav2Vec2Config."""
        base = ArchitectureConfig.from_transformers(config, parent_config=parent_config)
        base_fields = _shallow_fields(base)
        defaults = cls(hidden_size=1)
        return cls(
            **base_fields,
            add_adapter=getattr(config, "add_adapter", False) or False,
            output_hidden_size=getattr(
                config, "output_hidden_size", base_fields["hidden_size"]
            ),
            adapter_kernel_size=getattr(config, "adapter_kernel_size", 3),
            adapter_stride=getattr(config, "adapter_stride", 2),
            num_adapter_layers=getattr(config, "num_adapter_layers", 3),
            conv_dim=_conv_widths(config, defaults, base_fields["hidden_size"]),
            conv_kernel=tuple(getattr(config, "conv_kernel", None) or defaults.conv_kernel),
            conv_stride=tuple(getattr(config, "conv_stride", None) or defaults.conv_stride),
            conv_bias=bool(getattr(config, "conv_bias", False)),
            feat_extract_norm=getattr(config, "feat_extract_norm", None) or "group",
            do_stable_layer_norm=bool(getattr(config, "do_stable_layer_norm", False)),
            num_conv_pos_embeddings=getattr(config, "num_conv_pos_embeddings", None) or 128,
            num_conv_pos_embedding_groups=(
                getattr(config, "num_conv_pos_embedding_groups", None) or 16
            ),
            layer_norm_eps=getattr(config, "layer_norm_eps", None) or 1e-5,
        )


@dataclasses.dataclass
class ParakeetCTCConfig(ArchitectureConfig):
    """Configuration for Hugging Face Parakeet FastConformer CTC models."""

    num_mel_bins: int = 80
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    subsampling_conv_kernel_size: int = 3
    subsampling_conv_stride: int = 2
    conv_kernel_size: int = 9
    attention_bias: bool = True
    convolution_bias: bool = True
    scale_input: bool = True
    layer_norm_eps: float = 1e-5

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> ParakeetCTCConfig:
        """Extract the nested Parakeet encoder and parent CTC vocabulary fields."""
        encoder = getattr(config, "encoder_config", config)
        parent = config if encoder is not config else parent_config
        base = ArchitectureConfig.from_transformers(encoder, parent_config=parent)
        fields = _shallow_fields(base)
        fields.update(
            vocab_size=getattr(parent, "vocab_size", fields["vocab_size"]),
            pad_token_id=getattr(parent, "pad_token_id", fields["pad_token_id"]),
            model_type=getattr(parent, "model_type", fields["model_type"]),
        )
        resolved_dtype = _resolve_dtype(parent)
        # ORT CUDA executes this architecture in bf16 but diverges enough to
        # collapse real CTC output to blanks. The checkpoint weights are fp32,
        # so keep the safe fp32 default; callers may explicitly select fp16.
        if resolved_dtype is not None and resolved_dtype != ir.DataType.BFLOAT16:
            fields["dtype"] = resolved_dtype
        return cls(
            **fields,
            num_mel_bins=getattr(encoder, "num_mel_bins", 80),
            subsampling_factor=getattr(encoder, "subsampling_factor", 8),
            subsampling_conv_channels=getattr(encoder, "subsampling_conv_channels", 256),
            subsampling_conv_kernel_size=getattr(encoder, "subsampling_conv_kernel_size", 3),
            subsampling_conv_stride=getattr(encoder, "subsampling_conv_stride", 2),
            conv_kernel_size=getattr(encoder, "conv_kernel_size", 9),
            attention_bias=getattr(encoder, "attention_bias", True),
            convolution_bias=getattr(encoder, "convolution_bias", True),
            scale_input=getattr(encoder, "scale_input", True),
            layer_norm_eps=getattr(encoder, "layer_norm_eps", 1e-5),
        )
