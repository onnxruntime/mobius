# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses
import math

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

DEFAULT_INT = -42


def _resolve_dtype(config) -> ir.DataType | None:
    """Extract model dtype from a HuggingFace config.

    Handles string dtypes (e.g. "float16"), torch.dtype objects,
    and the "auto" sentinel (returns None).
    """
    torch_dtype = getattr(config, "dtype", None)
    if torch_dtype is not None and torch_dtype != "auto":
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype, None)
        if torch_dtype is not None:
            return tensor_adapters.from_torch_dtype(torch_dtype)
    return None


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
        or ("silu" if model_type in ("qwen", "chatglm") else None)
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
    from mobius._configs import (
        per_model,  # ruff: ignore[unused-import] - imported for registration side effect
    )
    from mobius._configs._extractors import extract_vision_config as _dispatch

    return _dispatch(config, parent_config, model_type)


def _extract_audio_config(config, parent_config, model_type: str) -> dict:
    """Extract audio sub-config from a HuggingFace config.

    Thin shim that delegates to the per-model registry. The actual
    hooks live under :mod:`mobius._configs.per_model` and are
    registered with :mod:`mobius._configs._extractors` at import time.
    """
    from mobius._configs import (
        per_model,  # ruff: ignore[unused-import] - imported for registration side effect
    )
    from mobius._configs._extractors import extract_audio_config as _dispatch

    return _dispatch(config, parent_config, model_type)


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
    mlp_bias: bool = False

    # Encoder-specific config
    type_vocab_size: int = 0

    # Encoder-decoder config
    num_decoder_layers: int | None = None
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
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
    scoring_func: str = "softmax"
    topk_method: str = "greedy"
    first_k_dense_replace: int = 0
    n_shared_experts: int | None = None

    # Multi-head Latent Attention (MLA) config — DeepSeek-V2/V3
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None
    rope_interleave: bool = False

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

    # Vision shared fields (accessed as top-level config.X by tasks)
    mm_tokens_per_image: int | None = None
    image_token_id: int | None = None
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
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

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> ArchitectureConfig:
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
            or model_type in ("glm", "glm4", "glm4_moe", "chatglm"),
        )
        if rope_config is not None:
            rope_config = dataclasses.replace(rope_config, rope_interleave=rope_interleave)

        options = dict(
            head_dim=(
                config.head_dim
                if (hasattr(config, "head_dim") and config.head_dim is not None)
                else getattr(config, "d_kv", None)
                or getattr(config, "kv_channels", None)
                or _as_int(hidden_size) // _as_int(num_attention_heads)
            ),
            num_attention_heads=_as_int(num_attention_heads),
            num_key_value_heads=_as_int(
                getattr(config, "num_key_value_heads", None)
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
            intermediate_size=(
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
            pad_token_id=(getattr(config, "pad_token_id", 0)),
            model_type=model_type,
            bos_token_id=getattr(config, "bos_token_id", None),
            eos_token_id=getattr(config, "eos_token_id", None),
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
            scoring_func=getattr(config, "scoring_func", "softmax"),
            topk_method=getattr(config, "topk_method", "greedy"),
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
            # Encoder-specific
            type_vocab_size=getattr(config, "type_vocab_size", 0),
            # Encoder-decoder
            num_decoder_layers=(
                getattr(config, "num_decoder_layers", None)
                or getattr(config, "decoder_layers", None)
            ),
            relative_attention_num_buckets=getattr(
                config, "relative_attention_num_buckets", 32
            ),
            relative_attention_max_distance=getattr(
                config, "relative_attention_max_distance", 128
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
        if model_type in ("falcon", "falcon_h1"):
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
            )

        # Model dtype
        resolved = _resolve_dtype(config)
        if resolved is not None:
            options["dtype"] = resolved

        # Quantization config
        quant = QuantizationConfig.from_transformers(config)
        if quant is None and parent_config is not None:
            quant = QuantizationConfig.from_transformers(parent_config)
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
class EncoderConfig(ArchitectureConfig):
    """Configuration for encoder-only models (BERT, ViT, etc.)."""


@dataclasses.dataclass
class VisionLanguageConfig(CausalLMConfig):
    """Configuration for vision-language models (LLaVA, Qwen-VL, etc.).

    Inherits :class:`CausalLMConfig` for the text decoder component.
    Vision-specific fields live in the :class:`VisionConfig` sub-config.
    """


# ---------------------------------------------------------------------------
# Model-family subclasses — add model-specific fields
# ---------------------------------------------------------------------------


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
        return cls(
            **_shallow_fields(base),
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

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Eagle3Config:
        layer_cfg = getattr(config, "transformer_layer_config", None)
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
    """

    altup_num_inputs: int = 4
    altup_active_idx: int = 0
    altup_correct_scale: bool = True
    laurel_rank: int = 64
    hidden_size_per_layer_input: int = 256
    vocab_size_per_layer_input: int = 262_144

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Gemma3nConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)
        return cls(
            **_shallow_fields(base),
            altup_num_inputs=getattr(config, "altup_num_inputs", 4),
            altup_active_idx=getattr(config, "altup_active_idx", 0),
            altup_correct_scale=getattr(config, "altup_correct_scale", True),
            laurel_rank=getattr(config, "laurel_rank", 64),
            hidden_size_per_layer_input=getattr(config, "hidden_size_per_layer_input", 256),
            vocab_size_per_layer_input=getattr(config, "vocab_size_per_layer_input", 262_144),
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
    # Dedicated quantization settings for the split per-layer embedding table.
    # Kept separate from ``quantization`` so embedding_bits does not affect text
    # token embeddings, LM head, or decoder Linear projection quantization.
    per_layer_embedding_bits: int | None = None
    per_layer_embedding_group_size: int = 32
    per_layer_embedding_sym: bool = False

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

        # num_global_key_value_heads: only set when attention_k_eq_v is True
        # (full-attention layers use fewer KV heads than sliding layers).
        num_global_kv = None
        if getattr(config, "attention_k_eq_v", False):
            num_global_kv = getattr(config, "num_global_key_value_heads", None)

        return cls(
            **_shallow_fields(base),
            global_head_dim=getattr(config, "global_head_dim", None),
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

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> JambaConfig:
        base = ArchitectureConfig.from_transformers(config, parent_config)

        # Build layer_types list for HybridCausalLMTask
        n = base.num_hidden_layers
        attn_period = getattr(config, "attn_layer_period", 8)
        attn_offset = getattr(config, "attn_layer_offset", 4)
        layer_types = []
        for i in range(n):
            if (i - attn_offset) % attn_period == 0:
                layer_types.append("full_attention")
            else:
                layer_types.append("mamba")

        num_experts = getattr(config, "num_experts", 16)
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)

        # Exclude fields we set explicitly below to avoid duplicate keyword args
        _exclude = {"layer_types", "num_local_experts", "num_experts_per_tok"}
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
            mamba_dt_rank=getattr(config, "mamba_dt_rank", 256),
            mamba_conv_bias=getattr(config, "mamba_conv_bias", True),
            mamba_proj_bias=getattr(config, "mamba_proj_bias", False),
            attn_layer_period=attn_period,
            attn_layer_offset=attn_offset,
            expert_layer_period=getattr(config, "expert_layer_period", 2),
            expert_layer_offset=getattr(config, "expert_layer_offset", 1),
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
class WhisperConfig(BaseModelConfig):
    """Configuration for Whisper encoder-decoder models."""

    encoder_layers: int = DEFAULT_INT
    encoder_attention_heads: int = DEFAULT_INT
    encoder_ffn_dim: int = DEFAULT_INT
    num_mel_bins: int = 80
    max_source_positions: int = 1500
    max_target_positions: int = 448
    scale_embedding: bool = False
    decoder_start_token_id: int | None = None
    layer_norm_eps: float = 1e-5

    @classmethod
    def from_transformers(cls, config) -> WhisperConfig:
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
        )

        # Model dtype
        resolved = _resolve_dtype(config)
        if resolved is not None:
            options["dtype"] = resolved

        return cls(**options)


@dataclasses.dataclass
class MMSConfig(ArchitectureConfig):
    """Configuration for MMS (Massively Multilingual Speech) CTC models.

    Extends ``ArchitectureConfig`` with the adapter parameters used in
    ``facebook/mms-1b-all`` and related checkpoints.  When ``add_adapter=True``
    the adapter layers are included in the exported ONNX graph; set this after
    calling ``model.load_adapter(lang_code)`` to bake a specific language's
    weights into the model.

    HuggingFace class: ``Wav2Vec2ForCTC`` with ``config.model_type == "wav2vec2"``
    """

    add_adapter: bool = False
    output_hidden_size: int = 0  # 0 → use hidden_size
    adapter_kernel_size: int = 3
    adapter_stride: int = 2
    num_adapter_layers: int = 3

    def __post_init__(self):
        if self.output_hidden_size == 0:
            self.output_hidden_size = self.hidden_size

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> MMSConfig:
        """Extract MMSConfig from a HuggingFace Wav2Vec2Config."""
        base = ArchitectureConfig.from_transformers(config, parent_config=parent_config)
        base_fields = _shallow_fields(base)
        return cls(
            **base_fields,
            add_adapter=getattr(config, "add_adapter", False),
            output_hidden_size=getattr(
                config, "output_hidden_size", base_fields["hidden_size"]
            ),
            adapter_kernel_size=getattr(config, "adapter_kernel_size", 3),
            adapter_stride=getattr(config, "adapter_stride", 2),
            num_adapter_layers=getattr(config, "num_adapter_layers", 3),
        )
