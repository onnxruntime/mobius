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


class TestQwen35MtpBlockExclusion:
    """Qwen3.5/3.8 GGUF ``block_count`` includes trailing MTP (nextn) blocks.

    ``gguf_to_config`` must subtract ``nextn_predict_layers`` so the decoder
    builds only the real transformer layers; otherwise it fabricates an extra
    layer whose weights are missing from the GGUF (the ``blk.<n>.nextn.*``
    prediction head is skipped during tensor mapping).
    """

    def _fake_model(self, metadata: dict) -> object:
        class _FakeGGUF:
            architecture = "qwen35"

            def __init__(self, md: dict) -> None:
                self.metadata = md

            def get_metadata(self, key, default=None):
                return self.metadata.get(key, default)

            @property
            def tensor_names(self) -> list[str]:
                return ["output.weight", "blk.0.attn_q.weight"]

        return _FakeGGUF(metadata)

    def _base_metadata(self, block_count: int) -> dict:
        return {
            "qwen35.embedding_length": 5120,
            "qwen35.block_count": block_count,
            "qwen35.attention.head_count": 24,
            "qwen35.attention.head_count_kv": 4,
            "qwen35.attention.key_length": 256,
            "qwen35.attention.value_length": 256,
            "qwen35.feed_forward_length": 17408,
            "qwen35.vocab_size": 248320,
            "qwen35.full_attention_interval": 4,
            "qwen35.rope.dimension_count": 64,
        }

    def test_nextn_layers_excluded_from_decoder_count(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        md = self._base_metadata(block_count=65)
        md["qwen35.nextn_predict_layers"] = 1
        config = gguf_to_config(self._fake_model(md))

        assert config.num_hidden_layers == 64
        assert config.layer_types is not None
        assert len(config.layer_types) == 64
        # 3 linear + 1 full pattern (full at every 4th, 1-indexed).
        assert config.layer_types[3] == "full_attention"
        assert config.layer_types[0] == "linear_attention"

    def test_no_nextn_metadata_leaves_count_unchanged(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        md = self._base_metadata(block_count=64)
        config = gguf_to_config(self._fake_model(md))

        assert config.num_hidden_layers == 64

    def test_nextn_not_greater_than_block_count(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        md = self._base_metadata(block_count=1)
        md["qwen35.nextn_predict_layers"] = 1
        with pytest.raises(ValueError, match="nextn_predict_layers"):
            gguf_to_config(self._fake_model(md))


class TestQwen35RopeInterleave:
    """``rope.dimension_sections`` is M-RoPE section metadata, not a GPT-J
    adjacent-pair rotation signal.

    Qwen3.5/3.8 rotate with split-half (NEOX) semantics. Deriving the flat
    ``rope_interleave`` from section presence corrupts RoPE — the exported
    GroupQueryAttention/RotaryEmbedding gets ``rotary_interleaved=1`` and the
    full-attention layers emit garbage tokens. The mapping must keep
    ``rope_interleave`` False for section-carrying non-interleaving arches.
    """

    def _fake_model(self, metadata: dict, architecture: str = "qwen35") -> object:
        class _FakeGGUF:
            def __init__(self, md: dict, arch: str) -> None:
                self.metadata = md
                self.architecture = arch

            def get_metadata(self, key, default=None):
                return self.metadata.get(key, default)

            @property
            def tensor_names(self) -> list[str]:
                return ["output.weight", "blk.0.attn_q.weight"]

        return _FakeGGUF(metadata, architecture)

    def _base_metadata(self) -> dict:
        return {
            "qwen35.embedding_length": 5120,
            "qwen35.block_count": 64,
            "qwen35.attention.head_count": 24,
            "qwen35.attention.head_count_kv": 4,
            "qwen35.attention.key_length": 256,
            "qwen35.attention.value_length": 256,
            "qwen35.feed_forward_length": 17408,
            "qwen35.vocab_size": 248320,
            "qwen35.full_attention_interval": 4,
            "qwen35.rope.dimension_count": 64,
            "qwen35.rope.freq_base": 1e7,
        }

    def test_dimension_sections_do_not_force_interleave(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        md = self._base_metadata()
        md["qwen35.rope.dimension_sections"] = [11, 11, 10, 0]
        config = gguf_to_config(self._fake_model(md))

        assert config.rope_interleave is False

    def test_no_sections_still_not_interleaved(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        config = gguf_to_config(self._fake_model(self._base_metadata()))

        assert config.rope_interleave is False
