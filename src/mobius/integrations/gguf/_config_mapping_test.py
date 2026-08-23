# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for architecture-specific GGUF config postprocessing."""

from __future__ import annotations

import pytest


class _FakeDenseGGUF:
    def __init__(self, architecture: str, metadata: dict, tensor_names: list[str]):
        self.architecture = architecture
        self.metadata = metadata
        self.tensor_names = tensor_names

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)


def _dense_metadata(architecture: str) -> dict:
    return {
        f"{architecture}.embedding_length": 64,
        f"{architecture}.feed_forward_length": 128,
        f"{architecture}.block_count": 2,
        f"{architecture}.attention.head_count": 4,
        f"{architecture}.attention.head_count_kv": 2,
        f"{architecture}.context_length": 512,
        f"{architecture}.rope.freq_base": 10_000.0,
        f"{architecture}.rope.dimension_count": 16,
        f"{architecture}.vocab_size": 256,
    }


class TestDenseCohortConfig:
    @pytest.mark.parametrize("architecture", ["arcee", "smollm3", "exaone"])
    def test_rmsnorm_dense_configs(self, architecture: str) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _dense_metadata(architecture)
        metadata[f"{architecture}.attention.layer_norm_rms_epsilon"] = 1e-5
        config = gguf_to_config(
            _FakeDenseGGUF(
                architecture,
                metadata,
                ["token_embd.weight", "output.weight", "blk.0.attn_q.weight"],
            )
        )

        assert config.model_type == architecture
        assert config.rms_norm_eps == pytest.approx(1e-5)
        assert config.hidden_act == ("relu2" if architecture == "arcee" else "silu")

    def test_olmo_weight_free_layernorm_config(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _dense_metadata("olmo")
        metadata["olmo.attention.layer_norm_epsilon"] = 1e-5
        config = gguf_to_config(
            _FakeDenseGGUF("olmo", metadata, ["token_embd.weight", "output.weight"])
        )

        assert config.model_type == "olmo"
        assert config.rms_norm_eps == pytest.approx(1e-5)

    def test_olmo_rejects_nonzero_qkv_clamp(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _dense_metadata("olmo")
        metadata["olmo.attention.layer_norm_epsilon"] = 1e-5
        metadata["olmo.attention.clamp_kqv"] = 8.0
        with pytest.raises(ValueError, match="clamp_kqv"):
            gguf_to_config(_FakeDenseGGUF("olmo", metadata, ["token_embd.weight"]))

    def test_olmo2_qk_norm_and_olmo3_pattern_rejection(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _dense_metadata("olmo2")
        metadata["olmo2.attention.layer_norm_rms_epsilon"] = 1e-6
        config = gguf_to_config(
            _FakeDenseGGUF("olmo2", metadata, ["token_embd.weight", "output.weight"])
        )

        assert config.model_type == "olmo2"
        assert config.attn_qk_norm is True
        assert config.attn_qk_norm_full is True

        metadata["olmo2.attention.sliding_window"] = 128
        metadata["olmo2.attention.sliding_window_pattern"] = [True, False]
        with pytest.raises(ValueError, match="OLMo3 semantics"):
            gguf_to_config(
                _FakeDenseGGUF("olmo2", metadata, ["token_embd.weight", "output.weight"])
            )

    def test_cohere2_logit_scale_and_partial_rope(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _dense_metadata("cohere2")
        metadata["cohere2.block_count"] = 4
        metadata["cohere2.attention.layer_norm_epsilon"] = 1e-5
        metadata["cohere2.attention.sliding_window"] = 128
        metadata["cohere2.logit_scale"] = 0.0625
        metadata["cohere2.rope.dimension_count"] = 8
        config = gguf_to_config(
            _FakeDenseGGUF("cohere2", metadata, ["token_embd.weight", "output_norm.weight"])
        )

        assert config.model_type == "cohere2"
        assert config.logit_scale == pytest.approx(0.0625)
        assert config.head_dim == 16
        assert config.partial_rotary_factor == pytest.approx(0.5)
        assert config.rope_interleave is True
        assert config.layer_types == [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        assert config.no_rope_layers == [1, 1, 1, 0]

    def test_smollm3_reconstructs_fixed_no_rope_schedule(self) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _dense_metadata("smollm3")
        metadata["smollm3.block_count"] = 8
        metadata["smollm3.attention.layer_norm_rms_epsilon"] = 1e-6
        config = gguf_to_config(
            _FakeDenseGGUF("smollm3", metadata, ["token_embd.weight", "output.weight"])
        )

        assert config.no_rope_layers == [1, 1, 1, 0, 1, 1, 1, 0]

    @pytest.mark.parametrize(
        ("architecture", "required_suffix"),
        [
            ("olmo", "attention.layer_norm_epsilon"),
            ("olmo2", "attention.layer_norm_rms_epsilon"),
            ("cohere2", "logit_scale"),
            ("arcee", "attention.layer_norm_rms_epsilon"),
        ],
    )
    def test_missing_required_metadata_is_rejected(
        self, architecture: str, required_suffix: str
    ) -> None:
        from mobius.integrations.gguf._arch_registry import get_arch_spec
        from mobius.integrations.gguf._config_mapping import gguf_to_config

        metadata = _dense_metadata(architecture)
        for suffix in get_arch_spec(architecture).required_metadata:
            metadata[f"{architecture}.{suffix}"] = 1
        del metadata[f"{architecture}.{required_suffix}"]

        with pytest.raises(ValueError, match=required_suffix):
            gguf_to_config(_FakeDenseGGUF(architecture, metadata, ["token_embd.weight"]))


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
    """M-RoPE section metadata is not a GPT-J adjacent-pair rotation signal.

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


class TestMuseGlimmerPostprocess:
    """Muse Glimmer config postprocessing.

    Ground truth is the published Muse-Glimmer-30B text config: a stride-4
    sliding-window pattern, NoPE on the full-attention layers,
    ``qk_scale_factor`` 3.87 and ``output_multiplier`` 0.19611613513818404.
    """

    @staticmethod
    def _base_config():
        from mobius._configs import ArchitectureConfig

        return ArchitectureConfig(
            hidden_size=6656,
            num_hidden_layers=8,
            num_attention_heads=32,
            num_key_value_heads=2,
            vocab_size=202048,
            intermediate_size=19968,
            rope_theta=500000.0,
        )

    @staticmethod
    def _metadata():
        return {
            "muse-glimmer.attention.sliding_window": 2048,
            "muse-glimmer.attention.sliding_window_pattern": 4,
            "muse-glimmer.final_logit_softcapping": 20.0,
            "muse-glimmer.logit_scale": 0.1961161345243454,
        }

    class _FakeModel:
        """Stands in for GGUFModel, exposing only what the mapping reads."""

        def __init__(self, q_norm):
            self._q_norm = q_norm

        def get_tensor(self, name: str):
            import numpy as np

            if name == "blk.0.attn_q_norm.weight":
                if self._q_norm is None:
                    raise KeyError(name)
                return np.asarray(self._q_norm, dtype=np.float32)
            raise KeyError(name)

    def test_layer_types_and_nope_layers_follow_the_stride(self) -> None:
        from mobius.integrations.gguf._config_mapping import _muse_glimmer_postprocess

        result = _muse_glimmer_postprocess(
            self._base_config(),
            self._metadata(),
            self._FakeModel([3.87] * 128),
        )

        assert result.layer_types == [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        # Full-attention layers are the NoPE layers.
        assert result.no_rope_layers == [3, 7]
        assert result.layer_rope_theta == [
            500000.0,
            500000.0,
            500000.0,
            0,
            500000.0,
            500000.0,
            500000.0,
            0,
        ]
        assert result.sliding_window == 2048

    def test_scalars_come_from_metadata_and_the_q_norm_tensor(self) -> None:
        from mobius.integrations.gguf._config_mapping import _muse_glimmer_postprocess

        result = _muse_glimmer_postprocess(
            self._base_config(),
            self._metadata(),
            self._FakeModel([3.87] * 128),
        )

        assert result.qk_scale_factor == pytest.approx(3.87)
        assert result.output_multiplier == pytest.approx(0.19611613513818404)
        assert result.final_logit_softcapping == pytest.approx(20.0)
        # GGUF has no key for post_norm_eps; the checkpoint default stands.
        assert result.post_norm_eps == pytest.approx(1e-8)
        assert result.attn_qk_norm is True

    def test_missing_q_norm_falls_back_to_the_default_scale(self) -> None:
        from mobius.integrations.gguf._config_mapping import _muse_glimmer_postprocess

        result = _muse_glimmer_postprocess(
            self._base_config(),
            self._metadata(),
            self._FakeModel(None),
        )

        assert result.qk_scale_factor == pytest.approx(3.87)

    def test_non_constant_q_norm_is_rejected(self) -> None:
        from mobius.integrations.gguf._config_mapping import _muse_glimmer_postprocess

        with pytest.raises(ValueError, match="not a constant vector"):
            _muse_glimmer_postprocess(
                self._base_config(),
                self._metadata(),
                self._FakeModel([3.87] * 127 + [1.0]),
            )

    def test_underscore_spelled_architecture_is_read_the_same_way(self) -> None:
        """Both ``muse-glimmer`` and ``muse_glimmer`` name the same architecture.

        The metadata prefix follows whatever the file calls itself, so the
        postprocessor has to take the prefix from the model rather than assume
        the hyphenated spelling.
        """
        from mobius.integrations.gguf._config_mapping import (
            _ARCH_KEY_MAPS,
            _muse_glimmer_postprocess,
        )

        model = self._FakeModel([3.87] * 128)
        model.architecture = "muse_glimmer"
        metadata = {
            key.replace("muse-glimmer.", "muse_glimmer."): value
            for key, value in self._metadata().items()
        }

        result = _muse_glimmer_postprocess(self._base_config(), metadata, model)

        assert result.sliding_window == 2048
        assert result.no_rope_layers == [3, 7]
        assert result.output_multiplier == pytest.approx(0.19611613513818404)
        # The key map is what turns attention.key_length into head_dim, so it
        # has to answer to both spellings too.
        assert _ARCH_KEY_MAPS["muse_glimmer"] == _ARCH_KEY_MAPS["muse-glimmer"]

    def test_a_missing_sliding_window_pattern_is_rejected(self) -> None:
        """Guessing the stride would yield a different architecture.

        Without it every layer is left sliding and rotated, which loads and
        runs and is not Muse Glimmer. Refusing is the only honest answer.
        """
        from mobius.integrations.gguf._config_mapping import _muse_glimmer_postprocess

        metadata = self._metadata()
        del metadata["muse-glimmer.attention.sliding_window_pattern"]

        with pytest.raises(ValueError, match="sliding_window_pattern"):
            _muse_glimmer_postprocess(
                self._base_config(), metadata, self._FakeModel([3.87] * 128)
            )
