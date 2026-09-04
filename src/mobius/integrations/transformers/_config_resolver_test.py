# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the Transformers config resolver."""

from __future__ import annotations

from unittest import mock

import pytest

from mobius._configs import (
    ArchitectureConfig,
    MoonshineConfig,
    MoonshineStreamingConfig,
    QuantizationConfig,
    WhisperConfig,
)
from mobius.integrations.transformers._config_resolver import (
    _config_from_hf,
    _default_task_for_model,
    _dict_to_pretrained_config,
)


def _fake_hf_config(model_type: str, **overrides):
    """Create a minimal HF-config-like object for testing.

    ``rope_parameters`` is present by default so the resolver treats the
    fake config as a RoPE-capable model (matching how real HuggingFace
    configs populate this field in ``PretrainedConfig.__post_init__``).
    Pass ``rope_parameters=None`` explicitly to exercise the NoPE path.
    """
    defaults = {
        "model_type": model_type,
        "vocab_size": 100,
        "max_position_embeddings": 32,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "head_dim": 16,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
        "rope_parameters": {"rope_type": "default"},
    }
    defaults.update(overrides)
    return type("FakeHFConfig", (), defaults)()


def test_component_overlay_does_not_reparse_deferred_block_quantization():
    hf = _fake_hf_config(
        "unregistered_block_model",
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
        },
    )
    resolved = ArchitectureConfig(block_quant_scheme=object())

    with (
        mock.patch.object(
            ArchitectureConfig,
            "from_transformers",
            return_value=resolved,
        ),
        mock.patch.object(
            QuantizationConfig,
            "from_value",
            side_effect=AssertionError("block quantization must stay deferred"),
        ),
    ):
        result = _config_from_hf(hf)

    assert result is resolved


# ── Top 5 HF config formats ─────────────────────────────────────────────


class TestConfigFromHfLlama:
    """Llama config resolution — the baseline decoder-only format."""

    def test_resolves_to_architecture_config(self):
        hf = _fake_hf_config("llama")
        result = _config_from_hf(hf)
        assert isinstance(result, ArchitectureConfig)

    def test_core_fields_extracted(self):
        hf = _fake_hf_config(
            "llama",
            vocab_size=32000,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            num_hidden_layers=32,
        )
        result = _config_from_hf(hf)
        assert result.vocab_size == 32000
        assert result.hidden_size == 4096
        assert result.num_attention_heads == 32
        assert result.num_key_value_heads == 8
        assert result.head_dim == 128

    def test_no_qkv_bias(self):
        hf = _fake_hf_config("llama")
        result = _config_from_hf(hf)
        assert result.attn_qkv_bias is False

    def test_default_task_is_text_generation(self):
        assert _default_task_for_model("llama") == "text-generation"


class TestConfigFromHfQwen:
    """Qwen2/3 config resolution — attention bias and QK norm."""

    def test_qwen2_has_qkv_bias(self):
        hf = _fake_hf_config("qwen2")
        result = _config_from_hf(hf)
        assert result.attn_qkv_bias is True

    def test_qwen2_head_dim_inferred(self):
        hf = _fake_hf_config("qwen2", head_dim=None, hidden_size=2048, num_attention_heads=16)
        result = _config_from_hf(hf)
        assert result.head_dim == 2048 // 16

    def test_qwen3_has_qk_norm(self):
        hf = _fake_hf_config("qwen3", head_dim=128)
        result = _config_from_hf(hf)
        assert result.attn_qk_norm is True

    def test_qwen2_no_qk_norm(self):
        hf = _fake_hf_config("qwen2")
        result = _config_from_hf(hf)
        assert result.attn_qk_norm is False


class TestConfigFromHfPhi:
    """Phi3 config resolution — partial rotary and su rope."""

    def test_phi3_resolves(self):
        hf = _fake_hf_config(
            "phi3",
            rope_scaling={"rope_type": "su", "long_factor": [1.0] * 48},
        )
        result = _config_from_hf(hf)
        assert isinstance(result, ArchitectureConfig)
        # The legacy Phi-3 ``"su"`` rope_type is canonicalized to ``"longrope"``
        # (they name the same LongRoPE algorithm); see _canonical_rope_type.
        assert result.rope_type == "longrope"

    def test_phi3_partial_rotary(self):
        hf = _fake_hf_config("phi3", partial_rotary_factor=0.5)
        result = _config_from_hf(hf)
        assert result.partial_rotary_factor == pytest.approx(0.5)


class TestConfigFromHfGemma:
    """Gemma config resolution — including nested rope_scaling."""

    def test_gemma_resolves(self):
        hf = _fake_hf_config("gemma")
        result = _config_from_hf(hf)
        assert isinstance(result, ArchitectureConfig)

    def test_gemma3_text_has_qk_norm(self):
        hf = _fake_hf_config("gemma3_text")
        result = _config_from_hf(hf)
        assert result.attn_qk_norm is True

    def test_gemma3_nested_rope_scaling(self):
        """Gemma3 stores per-attention-type rope configs.

        When config.rope_theta is set, it takes priority.  The nested
        ``full_attention.rope_theta`` is only used as fallback.
        """
        hf = _fake_hf_config(
            "gemma3_text",
            rope_theta=None,  # force fallback to nested lookup
            # Disable the default rope_parameters so the nested
            # rope_scaling entries are used for rope_type resolution.
            rope_parameters=None,
            rope_scaling={
                "full_attention": {
                    "rope_type": "linear",
                    "factor": 8.0,
                    "rope_theta": 500_000.0,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                },
            },
        )
        result = _config_from_hf(hf)
        assert result.rope_type == "linear"
        assert result.rope_theta == pytest.approx(500_000.0)
        assert result.rope_local_base_freq == pytest.approx(10_000.0)


class TestConfigFromHfMistral:
    """Mistral config resolution — sliding window."""

    def test_mistral_resolves(self):
        hf = _fake_hf_config("mistral")
        result = _config_from_hf(hf)
        assert isinstance(result, ArchitectureConfig)

    def test_mistral_sliding_window(self):
        hf = _fake_hf_config("mistral", sliding_window=4096)
        result = _config_from_hf(hf)
        assert result.sliding_window == 4096

    def test_mistral_no_qkv_bias(self):
        hf = _fake_hf_config("mistral")
        result = _config_from_hf(hf)
        assert result.attn_qkv_bias is False


# ── Unknown/malformed config ────────────────────────────────────────────


class TestUnknownMalformedConfig:
    """Unknown model types are handled by the registry, not from_transformers().

    from_transformers() extracts config fields regardless of model_type.
    The registry raises KeyError for unregistered architectures.
    """

    def test_unknown_model_type_still_extracts_config(self):
        hf = _fake_hf_config("totally_unknown_model_xyz")
        config = _config_from_hf(hf)
        assert isinstance(config, ArchitectureConfig)
        assert config.hidden_size == 64

    def test_unknown_model_type_preserves_all_fields(self):
        hf = _fake_hf_config("bogus_arch_42")
        config = _config_from_hf(hf)
        assert isinstance(config, ArchitectureConfig)
        assert config.num_attention_heads == 4

    def test_missing_model_type_attribute(self):
        """Config object without model_type falls through to ArchitectureConfig default."""
        hf = type(
            "NoModelType",
            (),
            {
                "vocab_size": 100,
                "hidden_size": 64,
                "num_attention_heads": 4,
            },
        )()
        # No model_type → registry lookup skipped → ArchitectureConfig.from_transformers
        # which requires model_type, so should raise
        with pytest.raises((ValueError, AttributeError)):
            _config_from_hf(hf)


# ── Gemma OffsetRMSNorm +1.0 config edge case ──────────────────────────


class TestGemmaOffsetRMSNorm:
    """Gemma OffsetRMSNorm +1.0 edge case.

    The OffsetRMSNorm class uses ``1.0 + weight`` for the effective multiplier.
    This is a model-level behavior, but config resolution must correctly
    preserve the rms_norm_eps and related fields that feed into the norm.
    """

    def test_gemma_rms_norm_eps_preserved(self):
        hf = _fake_hf_config("gemma", rms_norm_eps=1e-6)
        result = _config_from_hf(hf)
        assert result.rms_norm_eps == pytest.approx(1e-6)

    def test_gemma_layer_norm_eps_fallback(self):
        """When rms_norm_eps missing, falls back to layer_norm_eps."""
        hf = _fake_hf_config("gemma", rms_norm_eps=None, layer_norm_eps=1e-5)
        result = _config_from_hf(hf)
        assert result.rms_norm_eps == pytest.approx(1e-5)

    def test_gemma_custom_norm_eps(self):
        """Gemma2 uses 1e-6 by default; ensure custom values propagate."""
        hf = _fake_hf_config("gemma2", rms_norm_eps=1e-10)
        result = _config_from_hf(hf)
        assert result.rms_norm_eps == pytest.approx(1e-10)

    def test_gemma3_text_config_for_offset_norm(self):
        """Gemma3_text models use OffsetRMSNorm — config must resolve cleanly."""
        hf = _fake_hf_config(
            "gemma3_text",
            rms_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
        )
        result = _config_from_hf(hf)
        assert result.rms_norm_eps == pytest.approx(1e-6)
        assert result.hidden_act == "gelu_pytorch_tanh"
        assert result.attn_qk_norm is True


# ── DeepSeek MLA-specific config fields ─────────────────────────────────


class TestDeepSeekMLA:
    """DeepSeek-V2/V3 Multi-Latent Attention config extraction."""

    def _deepseek_config(self, **overrides):
        defaults = dict(
            model_type="deepseek_v2",
            vocab_size=102400,
            hidden_size=2048,
            intermediate_size=10944,
            num_hidden_layers=27,
            num_attention_heads=16,
            num_key_value_heads=16,
            head_dim=128,
            hidden_act="silu",
            max_position_embeddings=4096,
            pad_token_id=0,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            rope_scaling=None,
            # Real HF DeepSeek configs populate rope_parameters in
            # __post_init__; include it here so _extract_rope_config
            # treats this as a RoPE-capable model (not NoPE).
            rope_parameters={"rope_type": "default"},
            # MLA-specific fields
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            # MoE fields
            n_routed_experts=64,
            num_experts_per_tok=6,
            moe_intermediate_size=1408,
            shared_expert_intermediate_size=5632,
            n_shared_experts=2,
            first_k_dense_replace=1,
            scoring_func="softmax",
            topk_method="greedy",
        )
        defaults.update(overrides)
        return type("FakeDeepSeekConfig", (), defaults)()

    def test_mla_fields_extracted(self):
        hf = self._deepseek_config()
        result = _config_from_hf(hf)
        assert result.q_lora_rank == 1536
        assert result.kv_lora_rank == 512
        assert result.qk_nope_head_dim == 128
        assert result.qk_rope_head_dim == 64
        assert result.v_head_dim == 128

    def test_rope_interleave_auto_enabled(self):
        """rope_interleave is auto-enabled when qk_rope_head_dim > 0."""
        hf = self._deepseek_config(qk_rope_head_dim=64)
        result = _config_from_hf(hf)
        assert result.rope_interleave is True

    def test_rope_interleave_disabled_without_mla(self):
        """No MLA (qk_rope_head_dim=None) → rope_interleave stays False."""
        hf = self._deepseek_config(qk_rope_head_dim=None)
        result = _config_from_hf(hf)
        assert result.rope_interleave is False

    def test_moe_fields_extracted(self):
        hf = self._deepseek_config()
        result = _config_from_hf(hf)
        assert result.num_local_experts == 64
        assert result.num_experts_per_tok == 6
        assert result.moe_intermediate_size == 1408
        assert result.shared_expert_intermediate_size == 5632
        assert result.n_shared_experts == 2

    def test_deepseek_v3_model_type(self):
        hf = self._deepseek_config(model_type="deepseek_v3")
        result = _config_from_hf(hf)
        assert result.q_lora_rank == 1536
        assert result.kv_lora_rank == 512


# ── Whisper encoder-decoder nesting ─────────────────────────────────────


class TestWhisperEncoderDecoder:
    """Whisper encoder-decoder config resolution via registry config_class."""

    def _whisper_hf_config(self, **overrides):
        defaults = dict(
            model_type="whisper",
            vocab_size=51865,
            hidden_size=512,
            d_model=512,
            num_attention_heads=8,
            encoder_attention_heads=8,
            decoder_attention_heads=8,
            encoder_layers=6,
            decoder_layers=6,
            encoder_ffn_dim=2048,
            decoder_ffn_dim=2048,
            num_hidden_layers=6,
            num_mel_bins=80,
            max_source_positions=1500,
            max_target_positions=448,
            scale_embedding=False,
            decoder_start_token_id=50258,
            activation_function="gelu",
            pad_token_id=0,
            tie_word_embeddings=True,
        )
        defaults.update(overrides)
        return type("FakeWhisperConfig", (), defaults)()

    def test_whisper_routes_to_whisper_config(self):
        """_config_from_hf routes whisper to WhisperConfig via registry."""
        hf = self._whisper_hf_config()
        result = _config_from_hf(hf)
        assert isinstance(result, WhisperConfig)

    def test_encoder_fields_extracted(self):
        hf = self._whisper_hf_config()
        result = _config_from_hf(hf)
        assert result.encoder_layers == 6
        assert result.encoder_attention_heads == 8
        assert result.encoder_ffn_dim == 2048

    def test_decoder_fields_extracted(self):
        hf = self._whisper_hf_config()
        result = _config_from_hf(hf)
        assert result.num_hidden_layers == 6
        assert result.num_attention_heads == 8
        assert result.head_dim == 512 // 8

    def test_whisper_speech_specific_fields(self):
        hf = self._whisper_hf_config()
        result = _config_from_hf(hf)
        assert result.num_mel_bins == 80
        assert result.encoder_input_channels == 80
        assert result.max_source_positions == 1500
        assert result.max_target_positions == 448
        assert result.decoder_start_token_id == 50258

    def test_whisper_128_mel_channels_are_synchronized(self):
        result = _config_from_hf(self._whisper_hf_config(num_mel_bins=128))
        assert result.num_mel_bins == 128
        assert result.encoder_input_channels == 128

    def test_whisper_default_task(self):
        assert _default_task_for_model("whisper") == "speech-to-text"

    def test_whisper_bias_flags(self):
        """Whisper has both QKV and output projection biases."""
        hf = self._whisper_hf_config()
        result = _config_from_hf(hf)
        assert result.attn_qkv_bias is True
        assert result.attn_o_bias is True

    def test_explicit_component_quantization_survives_custom_config_parser(self):
        hf = self._whisper_hf_config(
            component_quantization={
                "encoder": {
                    "quant_method": "olive",
                    "bits": 8,
                    "group_size": 32,
                },
                "decoder": {
                    "quant_method": "olive",
                    "bits": 4,
                    "group_size": 16,
                },
            }
        )

        result = _config_from_hf(hf)

        assert result.component_quantization is not None
        assert result.component_quantization["encoder"].bits == 8
        assert result.component_quantization["decoder"].bits == 4
        assert result.quantization is result.component_quantization["decoder"]

    def test_whisper_tie_word_embeddings(self):
        hf = self._whisper_hf_config(tie_word_embeddings=True)
        result = _config_from_hf(hf)
        assert result.tie_word_embeddings is True


class TestMoonshineEncoderDecoder:
    """Moonshine config extraction preserves its distinct ASR architecture."""

    def _moonshine_hf_config(self):
        return type(
            "FakeMoonshineConfig",
            (),
            {
                "model_type": "moonshine",
                "vocab_size": 32768,
                "hidden_size": 288,
                "intermediate_size": 1152,
                "decoder_num_hidden_layers": 6,
                "decoder_num_attention_heads": 8,
                "decoder_num_key_value_heads": 8,
                "encoder_num_hidden_layers": 6,
                "encoder_num_attention_heads": 8,
                "encoder_num_key_value_heads": 8,
                "decoder_hidden_act": "silu",
                "encoder_hidden_act": "gelu",
                "max_position_embeddings": 194,
                "partial_rotary_factor": 0.9,
                "rope_scaling": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                    "partial_rotary_factor": 0.9,
                },
                "attention_bias": False,
                "pad_token_id": 2,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "decoder_start_token_id": 1,
                "tie_word_embeddings": True,
            },
        )()

    def test_moonshine_routes_to_moonshine_config(self):
        result = _config_from_hf(self._moonshine_hf_config())
        assert isinstance(result, MoonshineConfig)
        assert result.encoder_input_name == "input_values"
        assert result.encoder_input_channels is None
        assert result.encoder_uses_attention_mask is True
        assert result.decoder_uses_encoder_attention_mask is True

    def test_moonshine_attention_and_rope_fields(self):
        result = _config_from_hf(self._moonshine_hf_config())
        assert result.head_dim == 36
        assert result.encoder_num_hidden_layers == 6
        assert result.encoder_num_attention_heads == 8
        assert result.partial_rotary_factor == pytest.approx(0.9)
        assert result.rope_interleave is True
        assert result.attn_qkv_bias is False
        assert result.attn_o_bias is False

    def test_moonshine_default_task(self):
        assert _default_task_for_model("moonshine") == "speech-to-text"


class TestMoonshineStreamingEncoderDecoder:
    """Moonshine Streaming extraction keeps its encoder sub-config semantics."""

    def _hf_config(self, encoder_config):
        return type(
            "FakeMoonshineStreamingConfig",
            (),
            {
                "model_type": "moonshine_streaming",
                "encoder_config": encoder_config,
                "vocab_size": 32768,
                "hidden_size": 320,
                "intermediate_size": 1280,
                "num_hidden_layers": 6,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "head_dim": 40,
                "hidden_act": "silu",
                "max_position_embeddings": 4096,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                    "partial_rotary_factor": 0.8,
                },
                "attention_bias": False,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "decoder_start_token_id": 1,
                "tie_word_embeddings": False,
            },
        )()

    def _encoder_dict(self):
        return {
            "hidden_size": 320,
            "intermediate_size": 1280,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "head_dim": 40,
            "hidden_act": "gelu",
            "sample_rate": 16000,
            "frame_ms": 5.0,
            "sliding_windows": [[16, 4], [16, 4], [16, 0], [16, 0], [16, 4], [16, 4]],
        }

    def test_routes_to_moonshine_streaming_config(self):
        result = _config_from_hf(self._hf_config(self._encoder_dict()))
        assert isinstance(result, MoonshineStreamingConfig)
        assert result.encoder_input_name == "input_values"
        assert result.encoder_uses_attention_mask is True
        assert result.decoder_uses_encoder_attention_mask is True
        assert result.tie_word_embeddings is False

    def test_encoder_sub_config_dict_and_object_agree(self):
        """A dict sub-config and an attribute-style sub-config extract alike."""
        from_dict = _config_from_hf(self._hf_config(self._encoder_dict()))
        encoder_object = type("FakeEncoderConfig", (), self._encoder_dict())()
        from_object = _config_from_hf(self._hf_config(encoder_object))
        assert from_dict == from_object

    def test_streaming_specific_fields(self):
        result = _config_from_hf(self._hf_config(self._encoder_dict()))
        assert result.encoder_sliding_windows == (
            (16, 4),
            (16, 4),
            (16, 0),
            (16, 0),
            (16, 4),
            (16, 4),
        )
        assert result.encoder_hidden_act == "gelu"
        assert result.decoder_hidden_act == "silu"
        assert result.encoder_head_dim == 40
        assert result.encoder_sample_rate == 16000
        assert result.encoder_frame_ms == pytest.approx(5.0)
        assert result.frame_length == 80
        assert result.partial_rotary_factor == pytest.approx(0.8)
        assert result.rope_interleave is True
        assert result.mlp_bias is True
        assert result.attn_qkv_bias is False
        assert result.attn_o_bias is False

    def test_default_task(self):
        assert _default_task_for_model("moonshine_streaming") == "speech-to-text"


# ── _dict_to_pretrained_config ──────────────────────────────────────────


class TestDictToPretrainedConfig:
    """Tests for dict → PretrainedConfig conversion with nested configs."""

    def test_older_hub_without_strict_validation_error(self, monkeypatch):
        import transformers
        from huggingface_hub import errors as hub_errors

        # Resolve the lazy Transformers import before simulating an older,
        # mutually compatible huggingface_hub release.
        assert transformers.PretrainedConfig is not None
        monkeypatch.delattr(
            hub_errors,
            "StrictDataclassClassValidationError",
            raising=False,
        )
        config = _dict_to_pretrained_config({"model_type": "llama", "hidden_size": 64})
        assert config.hidden_size == 64

    def test_flat_dict(self):
        d = {"model_type": "llama", "hidden_size": 4096, "vocab_size": 32000}
        config = _dict_to_pretrained_config(d)
        assert config.model_type == "llama"
        assert config.hidden_size == 4096

    def test_nested_text_config(self):
        d = {
            "model_type": "qwen3_vl",
            "text_config": {"model_type": "qwen3_vl_text", "hidden_size": 2048},
        }
        config = _dict_to_pretrained_config(d)
        assert hasattr(config, "text_config")
        assert config.text_config.model_type == "qwen3_vl_text"
        assert config.text_config.hidden_size == 2048

    def test_nested_vision_config(self):
        d = {
            "model_type": "llava",
            "vision_config": {"hidden_size": 1024, "num_hidden_layers": 24},
        }
        config = _dict_to_pretrained_config(d)
        assert hasattr(config, "vision_config")
        assert config.vision_config.hidden_size == 1024

    def test_non_nested_keys_untouched(self):
        """Keys not in the nested_keys list stay as-is (dicts stay dicts)."""
        d = {"model_type": "test", "custom_config": {"key": "value"}}
        config = _dict_to_pretrained_config(d)
        assert isinstance(config.custom_config, dict)

    def test_thinker_config_nested(self):
        """Qwen3-ASR thinker_config nesting works."""
        d = {
            "model_type": "qwen3_asr",
            "thinker_config": {
                "model_type": "qwen3",
                "text_config": {"model_type": "qwen3", "hidden_size": 1024},
            },
        }
        config = _dict_to_pretrained_config(d)
        assert config.thinker_config.model_type == "qwen3"
        assert config.thinker_config.text_config.model_type == "qwen3"

    def test_composite_config_strips_rope_fields(self):
        """Composite configs strip top-level rope_scaling/rope_parameters."""
        d = {
            "model_type": "composite",
            "rope_scaling": {"type": "longrope"},
            "rope_parameters": {"rope_type": "default"},
            "text_config": {"model_type": "inner", "hidden_size": 256},
        }
        config = _dict_to_pretrained_config(d)
        # Rope fields stripped from top level
        assert not hasattr(config, "rope_scaling") or config.rope_scaling is None
        # Nested config still works
        assert config.text_config.model_type == "inner"

    def test_flat_config_keeps_rope_fields(self):
        """Non-composite (flat) configs keep rope_scaling as attribute."""
        d = {
            "model_type": "flat",
            "hidden_size": 256,
            "max_position_embeddings": 4096,
            # Simple rope_scaling that won't crash PretrainedConfig
            "rope_theta": 10000.0,
        }
        config = _dict_to_pretrained_config(d)
        assert config.rope_theta == pytest.approx(10000.0)

    def test_rope_retry_restores_fields(self):
        """When PretrainedConfig init crashes on rope, fields are restored."""
        # Simulate a config with rope_scaling that crashes standardization
        # by including rope_scaling without the fields needed for
        # standardization (matching Phi4-MM's failure pattern).
        d = {
            "model_type": "phi4mm",
            "rope_scaling": {"type": "longrope", "long_factor": [1.0]},
            "rope_theta": 10000.0,
            "hidden_size": 256,
            "max_position_embeddings": 4096,
        }
        # This should succeed (either directly or via retry)
        config = _dict_to_pretrained_config(d)
        assert config.model_type == "phi4mm"
        # If the retry path ran, rope_scaling should be restored as attr
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None:
            assert rope_scaling["type"] == "longrope"
