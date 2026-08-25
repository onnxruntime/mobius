# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for architecture-specific configuration extraction."""

from __future__ import annotations

import types

import pytest

from mobius._configs import (
    ArchitectureConfig,
    KimiLinearConfig,
    MiniMaxConfig,
    NemotronParseConfig,
)


def _kimi_linear_hf_config(**overrides):
    values = {
        "model_type": "kimi_linear",
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 48,
        "vocab_size": 64,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "linear_attn_config": {
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [4],
            "num_heads": 2,
            "head_dim": 32,
            "short_conv_kernel_size": 4,
        },
        "mla_use_nope": True,
        "q_lora_rank": None,
        "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 16,
        "v_head_dim": 32,
        "kv_lora_rank": 32,
        "first_k_dense_replace": 1,
        "moe_intermediate_size": 32,
        "moe_layer_freq": 1,
        "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid",
        "num_experts": 2,
        "num_experts_per_token": 1,
        "num_expert_group": 1,
        "topk_group": 1,
        "num_shared_experts": 1,
        "num_nextn_predict_layers": 0,
        "routed_scaling_factor": 2.446,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_kimi_linear_config_extracts_exact_schedule() -> None:
    config = KimiLinearConfig.from_transformers(_kimi_linear_hf_config())

    assert config.layer_types == [
        "kimi_linear_attention",
        "kimi_linear_attention",
        "kimi_linear_attention",
        "full_attention",
    ]
    assert config.linear_key_head_dim == 32
    assert config.qk_nope_head_dim == 32
    assert config.qk_rope_head_dim == 16


def test_kimi_linear_config_uses_top_level_head_count_when_nested_value_is_absent() -> None:
    config = _kimi_linear_hf_config()
    del config.linear_attn_config["num_heads"]

    extracted = KimiLinearConfig.from_transformers(config)

    assert extracted.linear_num_key_heads == config.num_attention_heads


def test_kimi_linear_config_rejects_empty_convolution_history() -> None:
    config = _kimi_linear_hf_config()
    config.linear_attn_config["short_conv_kernel_size"] = 1

    with pytest.raises(ValueError, match="at least 2"):
        KimiLinearConfig.from_transformers(config)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (
            {
                "linear_attn_config": {
                    "kda_layers": [1, 2],
                    "full_attn_layers": [4],
                    "num_heads": 2,
                    "head_dim": 32,
                    "short_conv_kernel_size": 4,
                }
            },
            "exactly partition",
        ),
        ({"num_expert_group": 2}, "single expert-group"),
    ],
)
def test_kimi_linear_config_rejects_non_authoritative_profiles(
    override: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        KimiLinearConfig.from_transformers(_kimi_linear_hf_config(**override))


def test_minimax_config_extracts_exact_schedule_head_geometry_and_residuals():
    config = types.SimpleNamespace(
        model_type="MiniMaxText01",
        hidden_size=48,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rotary_dim=8,
        rope_theta=10_000_000.0,
        vocab_size=64,
        attn_type_list=[0, 1],
        num_local_experts=2,
        num_experts_per_tok=1,
        rms_norm_eps=1e-5,
        postnorm=True,
        shared_intermediate_size=0,
        layernorm_full_attention_alpha=3.5,
        layernorm_full_attention_beta=1.0,
        layernorm_linear_attention_alpha=3.5,
        layernorm_linear_attention_beta=1.0,
        layernorm_mlp_alpha=3.5,
        layernorm_mlp_beta=1.0,
    )

    extracted = MiniMaxConfig.from_transformers(config)

    assert extracted.model_type == "minimax"
    assert extracted.head_dim == 16
    assert extracted.partial_rotary_factor == pytest.approx(0.5)
    assert extracted.layer_types == ["lightning_attention", "full_attention"]
    assert extracted.lightning_norm_eps == pytest.approx(1e-6)
    assert extracted.full_attn_alpha_factor == pytest.approx(3.5)
    assert extracted.linear_attn_alpha_factor == pytest.approx(3.5)
    assert extracted.mlp_alpha_factor == pytest.approx(3.5)
    assert extracted.disable_qmoe


def test_nemotron_parse_maps_raw_mbart_decoder_attention_heads():
    """The Hub's non-trusted config exposes MBART's decoder-specific aliases."""
    config = types.SimpleNamespace(
        model_type="nemotron_parse",
        decoder={
            "model_type": "nemotron_parse_text",
            "d_model": 1024,
            "decoder_attention_heads": 16,
            "decoder_ffn_dim": 4096,
            "decoder_layers": 10,
            "num_hidden_layers": 12,
            "vocab_size": 72256,
            "pad_token_id": 1,
        },
        encoder={"patch_size": 16, "max_resolution": 2048},
        image_size=[2048, 1664],
        max_sequence_length=9000,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        tie_word_embeddings=True,
        decoder_start_token_id=2,
    )

    extracted = NemotronParseConfig.from_transformers(config)

    assert extracted.num_attention_heads == 16
    assert extracted.num_key_value_heads == 16
    assert extracted.head_dim == 64
    assert extracted.num_decoder_layers == 10


class _FakeHFConfig:
    """Minimal HuggingFace-config stand-in with attribute access."""

    def __init__(self, model_type: str = "_unrelated_", **kwargs):
        self.model_type = model_type
        self.__dict__.update(kwargs)


def test_scalar_intermediate_size_passes_through():
    cfg = _FakeHFConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_attention_heads=8,
        num_hidden_layers=4,
        vocab_size=256,
    )
    out = ArchitectureConfig.from_transformers(cfg)
    assert out.intermediate_size == 8192


def test_list_intermediate_size_collapses_to_first_element():
    """Gemma 3n expresses intermediate_size as a per-layer list.

    The list is uniform in every shipped checkpoint, so collapsing to the
    first element yields the correct scalar MLP width. Without the coercion
    the list reached ``nn.Parameter``/``ir.Shape`` as a dim and raised.
    """
    cfg = _FakeHFConfig(
        hidden_size=2048,
        intermediate_size=[8192] * 4,
        num_attention_heads=8,
        num_hidden_layers=4,
        vocab_size=256,
    )
    out = ArchitectureConfig.from_transformers(cfg)
    assert out.intermediate_size == 8192
    assert isinstance(out.intermediate_size, int)


def _codec_hf_config(**encoder_overrides):
    """A Qwen3-TTS-Tokenizer-style HF config with nested encoder/decoder."""
    encoder = {
        "codebook_dim": 256,
        "codebook_size": 2048,
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "num_quantizers": 32,
        "num_semantic_quantizers": 1,
        "audio_channels": 1,
        "num_filters": 64,
        "num_residual_layers": 1,
        "kernel_size": 7,
        "last_kernel_size": 3,
        "residual_kernel_size": 3,
        "compress": 2,
        "upsampling_ratios": [8, 6, 5, 4],
    }
    encoder.update(encoder_overrides)
    return _FakeHFConfig(
        model_type="qwen3_tts_tokenizer_12hz",
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=8,
        vocab_size=2048,
        decoder_config={"hidden_size": 512, "codebook_dim": 512},
        encoder_config=encoder,
    )


def test_codec_encoder_conv_fields_extracted_from_nested_config():
    """Nested ``encoder_config`` values drive the derived conv stack.

    These fields are read with ``getattr`` off a nested config, so a wrong
    key or default would silently fall back to the checkpoint defaults and
    build the wrong architecture.
    """
    out = ArchitectureConfig.from_transformers(_codec_hf_config())

    enc = out.codec_encoder
    assert enc is not None
    assert enc.audio_channels == 1
    assert enc.num_filters == 64
    assert enc.num_residual_layers == 1
    assert enc.kernel_size == 7
    assert enc.last_kernel_size == 3
    assert enc.residual_kernel_size == 3
    assert enc.compress == 2
    assert enc.upsampling_ratios == [8, 6, 5, 4]


def test_codec_encoder_conv_fields_honor_non_default_values():
    """Non-default nested values must survive extraction unchanged."""
    out = ArchitectureConfig.from_transformers(
        _codec_hf_config(
            hidden_size=64,
            audio_channels=2,
            num_filters=8,
            num_residual_layers=3,
            kernel_size=5,
            last_kernel_size=1,
            residual_kernel_size=7,
            compress=4,
            upsampling_ratios=[4, 2],
        )
    )

    enc = out.codec_encoder
    assert enc is not None
    assert enc.hidden_size == 64
    assert enc.audio_channels == 2
    assert enc.num_filters == 8
    assert enc.num_residual_layers == 3
    assert enc.kernel_size == 5
    assert enc.last_kernel_size == 1
    assert enc.residual_kernel_size == 7
    assert enc.compress == 4
    assert enc.upsampling_ratios == [4, 2]


def test_codec_encoder_conv_fields_fall_back_to_checkpoint_defaults():
    """A config omitting the conv fields still yields the real architecture."""
    out = ArchitectureConfig.from_transformers(
        _FakeHFConfig(
            model_type="qwen3_tts_tokenizer_12hz",
            hidden_size=512,
            num_attention_heads=8,
            num_hidden_layers=8,
            vocab_size=2048,
            decoder_config={"hidden_size": 512},
            encoder_config={"hidden_size": 512},
        )
    )

    enc = out.codec_encoder
    assert enc is not None
    assert enc.num_filters == 64
    assert enc.upsampling_ratios == [8, 6, 5, 4]
    assert enc.num_residual_layers == 1
