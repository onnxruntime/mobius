# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for architecture-specific GGUF config postprocessing."""

from __future__ import annotations

import pytest


class TestGemma3Postprocess:
    """Gemma3 config postprocessing fills fields GGUF omits."""

    def test_defaults_local_rope_and_layer_types(self) -> None:
        from mobius._configs import ArchitectureConfig
        from mobius.integrations.gguf._config_mapping import _gemma3_postprocess

        config = ArchitectureConfig(
            hidden_size=1152,
            num_hidden_layers=26,
            num_attention_heads=4,
            num_key_value_heads=1,
            vocab_size=262144,
            intermediate_size=6912,
        )
        # GGUF carries only the global rope base and the sliding-window size.
        result = _gemma3_postprocess(config, {"gemma3.rope.freq_base": 1_000_000.0})

        assert result.rope_local_base_freq == pytest.approx(10_000.0)
        assert result.layer_types is not None
        assert len(result.layer_types) == 26
        # Every 6th layer (1-indexed) is full attention; the rest sliding.
        assert result.layer_types[5] == "full_attention"
        assert result.layer_types[0] == "sliding_attention"
        assert result.layer_types[11] == "full_attention"

    def test_respects_explicit_gguf_values(self) -> None:
        from mobius._configs import ArchitectureConfig
        from mobius.integrations.gguf._config_mapping import _gemma3_postprocess

        config = ArchitectureConfig(
            hidden_size=1152,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            vocab_size=256,
            intermediate_size=512,
        )
        result = _gemma3_postprocess(
            config,
            {
                "gemma3.rope.local_freq_base": 20_000.0,
                "gemma3.attention.sliding_window_pattern": 2,
            },
        )

        assert result.rope_local_base_freq == pytest.approx(20_000.0)
        # Pattern of 2 → every other layer (1-indexed even) is full attention.
        assert result.layer_types == [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]


class TestGemma4DoubleWideMlp:
    """Gemma4 per-layer feed_forward_length arrays collapse to a scalar base."""

    def _base_config(self, intermediate_size, num_layers=4):
        from mobius._configs import ArchitectureConfig

        return ArchitectureConfig(
            hidden_size=1536,
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            num_key_value_heads=1,
            vocab_size=262144,
            intermediate_size=intermediate_size,
        )

    def test_uniform_array_collapses_to_scalar(self) -> None:
        from mobius.integrations.gguf._config_mapping import _gemma4_postprocess

        result = _gemma4_postprocess(self._base_config([8192, 8192, 8192, 8192]), {})

        assert result.intermediate_size == 8192
        assert result.use_double_wide_mlp is False

    def test_double_wide_tail_layers_set_flag(self) -> None:
        from mobius.integrations.gguf._config_mapping import _gemma4_postprocess

        # Last 2 KV-shared layers are double-wide (2x the base).
        result = _gemma4_postprocess(
            self._base_config([6144, 6144, 12288, 12288]),
            {"gemma4.attention.shared_kv_layers": 2},
        )

        assert result.intermediate_size == 6144
        assert result.use_double_wide_mlp is True
        assert result.num_kv_shared_layers == 2

    def test_scalar_intermediate_size_unchanged(self) -> None:
        from mobius.integrations.gguf._config_mapping import _gemma4_postprocess

        result = _gemma4_postprocess(self._base_config(8192), {})

        assert result.intermediate_size == 8192
        assert result.use_double_wide_mlp is False

    def test_mismatched_double_wide_pattern_raises(self) -> None:
        import pytest

        from mobius.integrations.gguf._config_mapping import _gemma4_postprocess

        # Double-wide layers must be the trailing KV-shared layers; a leading
        # wide layer does not match the expected pattern.
        with pytest.raises(ValueError, match="double-wide-MLP pattern"):
            _gemma4_postprocess(
                self._base_config([12288, 6144, 6144, 6144]),
                {"gemma4.attention.shared_kv_layers": 2},
            )


class TestDefaultActivation:
    """Tests for _default_activation()."""

    @pytest.mark.parametrize(
        "model_type",
        ["gemma", "gemma2", "gemma3_text", "gemma4_text"],
    )
    def test_gemma_uses_gelu_pytorch_tanh(self, model_type: str) -> None:
        from mobius.integrations.gguf._config_mapping import _default_activation

        assert _default_activation(model_type) == "gelu_pytorch_tanh"

    @pytest.mark.parametrize("model_type", ["gpt2", "bloom", "starcoder2", "t5"])
    def test_gelu_models(self, model_type: str) -> None:
        from mobius.integrations.gguf._config_mapping import _default_activation

        assert _default_activation(model_type) == "gelu"

    @pytest.mark.parametrize("model_type", ["llama", "qwen2", "mistral"])
    def test_silu_default(self, model_type: str) -> None:
        from mobius.integrations.gguf._config_mapping import _default_activation

        assert _default_activation(model_type) == "silu"
