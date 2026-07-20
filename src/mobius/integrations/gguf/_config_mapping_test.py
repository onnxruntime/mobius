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
