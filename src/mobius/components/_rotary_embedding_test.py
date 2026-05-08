# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for rotary embeddings."""

from __future__ import annotations

import numpy as np
import pytest

from mobius._testing import create_test_builder, create_test_input, make_config
from mobius.components._rotary_embedding import (
    ChunkedMRope,
    DefaultRope,
    DynamicNTKRope,
    InterleavedMRope,
    LinearRope,
    Llama3Rope,
    LongRope,
    _get_cos_sin_cache,
    _get_default_inv_freq,
    apply_rotary_pos_emb,
    get_rotary_pos_emb,
    initialize_rope,
)


class TestInvFreq:
    def test_default_inv_freq_shape(self):
        config = make_config(head_dim=16, partial_rotary_factor=1.0)
        inv_freq = _get_default_inv_freq(config)
        assert inv_freq.shape == (8,)  # dim/2

    def test_partial_rotary_inv_freq_shape(self):
        config = make_config(head_dim=16, partial_rotary_factor=0.5)
        inv_freq = _get_default_inv_freq(config)
        assert inv_freq.shape == (4,)  # (dim * 0.5) / 2

    def test_inv_freq_values_decrease(self):
        config = make_config(head_dim=16)
        inv_freq = _get_default_inv_freq(config)
        for i in range(len(inv_freq) - 1):
            assert inv_freq[i] > inv_freq[i + 1]


class TestCosSinCache:
    def test_cache_shape(self):
        inv_freq = np.array([1.0, 0.5, 0.25, 0.125])
        cos, sin = _get_cos_sin_cache(32, inv_freq)
        assert cos.shape == (32, 4)
        assert sin.shape == (32, 4)

    def test_cache_values_bounded(self):
        inv_freq = np.array([1.0, 0.5])
        cos, sin = _get_cos_sin_cache(16, inv_freq)
        assert np.all(cos >= -1.0) and np.all(cos <= 1.0)
        assert np.all(sin >= -1.0) and np.all(sin <= 1.0)

    def test_attention_scaling(self):
        inv_freq = np.array([1.0, 0.5])
        cos1, sin1 = _get_cos_sin_cache(16, inv_freq, attention_scaling=1.0)
        cos2, sin2 = _get_cos_sin_cache(16, inv_freq, attention_scaling=2.0)
        np.testing.assert_allclose(cos2, cos1 * 2.0)
        np.testing.assert_allclose(sin2, sin1 * 2.0)


class TestRopeVariants:
    def test_default_rope_creates_caches(self):
        config = make_config()
        rope = DefaultRope(config)
        params = list(rope.parameters())
        assert len(params) == 2  # cos_cache, sin_cache

    def test_default_rope_cache_shapes(self):
        config = make_config(max_position_embeddings=64, head_dim=16)
        rope = DefaultRope(config)
        assert list(rope.cos_cache.shape) == [64, 8]
        assert list(rope.sin_cache.shape) == [64, 8]

    def test_default_rope_forward(self):
        config = make_config()
        rope = DefaultRope(config)
        builder, op, _graph = create_test_builder()
        pos_ids = create_test_input(builder, "pos_ids", [2, 4])
        result = rope(op, pos_ids)
        assert len(result) == 2  # (cos_emb, sin_emb)

    def test_linear_rope(self):
        config = make_config(rope_scaling={"factor": 2.0})
        rope = LinearRope(config)
        assert next(iter(rope.cos_cache.shape)) == config.max_position_embeddings

    def test_llama3_rope(self):
        config = make_config(
            max_position_embeddings=131072,
            original_max_position_embeddings=8192,
            rope_scaling={
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
            },
        )
        rope = Llama3Rope(config)
        assert next(iter(rope.cos_cache.shape)) == 131072

    def test_long_rope_short_only(self):
        config = make_config(
            max_position_embeddings=32,
            original_max_position_embeddings=32,
            rope_scaling={
                "short_factor": [1.0] * 8,
                "long_factor": [1.0] * 8,
            },
        )
        rope = LongRope(config)
        assert not rope.has_long_cache

    def test_long_rope_with_long_cache(self):
        config = make_config(
            max_position_embeddings=64,
            original_max_position_embeddings=32,
            rope_scaling={
                "short_factor": [1.0] * 8,
                "long_factor": [1.0] * 8,
            },
        )
        rope = LongRope(config)
        assert rope.has_long_cache
        assert next(iter(rope.cos_cache.shape)) == 96

    def test_dynamic_ntk_rope_with_factor(self):
        """DynamicNTKRope with factor (standard, no alpha) applies NTK scaling."""
        config = make_config(
            rope_type="dynamic",
            rope_scaling={"factor": 4.0},
        )
        rope = DynamicNTKRope(config)
        assert next(iter(rope.cos_cache.shape)) == config.max_position_embeddings

        # Verify NTK scaling: new_theta = theta * factor^(dim/(dim-2))
        dim = config.head_dim
        expected_theta = config.rope_theta * (4.0 ** (dim / (dim - 2)))
        expected_inv = 1.0 / (expected_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        expected_cos1 = np.cos(expected_inv)
        actual_cos1 = rope.cos_cache.const_value.numpy()[1, :dim // 2]
        np.testing.assert_allclose(actual_cos1, expected_cos1, atol=1e-5)

    def test_dynamic_ntk_rope_with_alpha(self):
        """DynamicNTKRope with alpha (HunyuanV1) uses alpha instead of factor."""
        config = make_config(
            rope_type="dynamic",
            rope_scaling={"factor": 1.0, "alpha": 1000.0},
        )
        rope = DynamicNTKRope(config)

        # With alpha=1000, scaling should use 1000 not factor=1.0
        dim = config.head_dim
        expected_theta = config.rope_theta * (1000.0 ** (dim / (dim - 2)))
        expected_inv = 1.0 / (expected_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        expected_cos1 = np.cos(expected_inv)
        actual_cos1 = rope.cos_cache.const_value.numpy()[1, :dim // 2]
        np.testing.assert_allclose(actual_cos1, expected_cos1, atol=1e-5)

        # Verify it differs from factor=1.0 (no scaling)
        default_inv = 1.0 / (config.rope_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        default_cos1 = np.cos(default_inv)
        assert not np.allclose(actual_cos1, default_cos1, atol=1e-3), (
            "alpha=1000 should produce different frequencies than default"
        )

    def test_dynamic_ntk_rope_alpha_matches_hunyuan_hf(self):
        """DynamicNTKRope with HunyuanV1 config matches HF inv_freq exactly."""
        # HunyuanV1 HF formula: base = theta * alpha^(dim/(dim-2))
        # inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
        config = make_config(
            rope_theta=10000.0,
            head_dim=128,
            rope_type="dynamic",
            rope_scaling={
                "factor": 1.0,
                "alpha": 1000.0,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
            },
        )
        rope = DynamicNTKRope(config)

        # Reference: HF HunYuanDenseV1RotaryEmbedding
        dim = 128
        base = 10000.0 * 1000.0 ** (dim / (dim - 2))
        hf_inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))

        # Compare cos at position 1 (cos(1 * inv_freq))
        hf_cos1 = np.cos(hf_inv_freq)
        actual_cos1 = rope.cos_cache.const_value.numpy()[1, :dim // 2]
        np.testing.assert_allclose(actual_cos1, hf_cos1, atol=1e-5)

    def test_dynamic_ntk_rope_factor_only_backward_compatible(self):
        """DynamicNTKRope without alpha still works with factor alone."""
        config = make_config(
            rope_type="dynamic",
            rope_scaling={"factor": 2.0},
        )
        rope = DynamicNTKRope(config)

        dim = config.head_dim
        expected_theta = config.rope_theta * (2.0 ** (dim / (dim - 2)))
        expected_inv = 1.0 / (expected_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        expected_cos1 = np.cos(expected_inv)
        actual_cos1 = rope.cos_cache.const_value.numpy()[1, :dim // 2]
        np.testing.assert_allclose(actual_cos1, expected_cos1, atol=1e-5)


class TestInitializeRope:
    def test_default(self):
        config = make_config(rope_type="default")
        rope = initialize_rope(config)
        assert isinstance(rope, DefaultRope)

    def test_linear(self):
        config = make_config(rope_type="linear", rope_scaling={"factor": 2.0})
        rope = initialize_rope(config)
        assert isinstance(rope, LinearRope)

    def test_llama3(self):
        config = make_config(
            rope_type="llama3",
            original_max_position_embeddings=8192,
            rope_scaling={"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0},
        )
        rope = initialize_rope(config)
        assert isinstance(rope, Llama3Rope)

    def test_longrope(self):
        config = make_config(
            rope_type="longrope",
            rope_scaling={"short_factor": [1.0] * 8, "long_factor": [1.0] * 8},
        )
        rope = initialize_rope(config)
        assert isinstance(rope, LongRope)

    def test_dynamic(self):
        config = make_config(rope_type="dynamic", rope_scaling={"factor": 2.0})
        rope = initialize_rope(config)
        assert isinstance(rope, DynamicNTKRope)

    def test_dynamic_with_alpha(self):
        config = make_config(
            rope_type="dynamic",
            rope_scaling={"factor": 1.0, "alpha": 1000.0},
        )
        rope = initialize_rope(config)
        assert isinstance(rope, DynamicNTKRope)

    def test_unsupported_raises(self):
        config = make_config(rope_type="unknown")
        with pytest.raises(ValueError, match="Unsupported rope type"):
            initialize_rope(config)

    def test_mrope_section_without_interleaved_returns_chunked(self):
        config = make_config(mrope_section=[8, 12, 12])
        rope = initialize_rope(config)
        assert isinstance(rope, ChunkedMRope)

    def test_mrope_section_with_interleaved_returns_interleaved(self):
        config = make_config(mrope_section=[11, 11, 10], mrope_interleaved=True)
        rope = initialize_rope(config)
        assert isinstance(rope, InterleavedMRope)

    def test_nope_returns_none(self):
        """initialize_rope returns None when rope_type is None (NoPE).

        NoPE models like NemotronH and GraniteMoeHybrid leave ``rope_type``
        at its ``None`` default so that callers do not silently apply
        rotary encoding to a model that shouldn't have any. This is the
        Phase 1 fix for the default-value bug.
        """
        config = make_config(rope_type=None)
        assert initialize_rope(config) is None


class TestChunkedMRope:
    def test_creates_caches_and_masks(self):
        config = make_config(head_dim=16, mrope_section=[3, 3, 2])
        rope = ChunkedMRope(config)
        param_names = [n for n, _ in rope.named_parameters()]
        assert "cos_cache" in param_names
        assert "sin_cache" in param_names
        assert "h_mask" in param_names
        assert "w_mask" in param_names

    def test_contiguous_mask_layout(self):
        # mrope_section=[3, 3, 2] with head_dim=16 → rotary_dim=8
        config = make_config(head_dim=16, mrope_section=[3, 3, 2])
        rope = ChunkedMRope(config)
        h_mask = rope.h_mask._const_value.numpy()
        w_mask = rope.w_mask._const_value.numpy()
        # H occupies indices 3,4,5 (contiguous block after T)
        assert list(h_mask) == [False, False, False, True, True, True, False, False]
        # W occupies indices 6,7 (contiguous block after H)
        assert list(w_mask) == [False, False, False, False, False, False, True, True]

    def test_forward_builds_graph(self):
        config = make_config(head_dim=16, mrope_section=[3, 3, 2])
        rope = ChunkedMRope(config)
        builder, op, graph = create_test_builder()
        pos_ids = create_test_input(builder, "pos_ids", [3, 2, 4])
        result = rope(op, pos_ids)
        assert len(result) == 2  # (cos, sin)
        assert graph.num_nodes() > 0


class TestInterleavedMRope:
    def test_creates_caches_and_masks(self):
        config = make_config(head_dim=16, mrope_section=[3, 3, 2], mrope_interleaved=True)
        rope = InterleavedMRope(config)
        param_names = [n for n, _ in rope.named_parameters()]
        assert "cos_cache" in param_names
        assert "sin_cache" in param_names
        assert "h_mask" in param_names
        assert "w_mask" in param_names

    def test_interleaved_mask_layout(self):
        # mrope_section=[3, 3, 2] with head_dim=16 → rotary_dim=8
        # H channels at stride 3 offset 1: positions 1, 4, 7  (h_length=3*3=9)
        # W channels at stride 3 offset 2: positions 2, 5     (w_length=2*3=6)
        config = make_config(head_dim=16, mrope_section=[3, 3, 2], mrope_interleaved=True)
        rope = InterleavedMRope(config)
        h_mask = rope.h_mask._const_value.numpy()
        w_mask = rope.w_mask._const_value.numpy()
        assert list(h_mask) == [False, True, False, False, True, False, False, True]
        assert list(w_mask) == [False, False, True, False, False, True, False, False]

    def test_forward_builds_graph(self):
        config = make_config(head_dim=16, mrope_section=[3, 3, 2], mrope_interleaved=True)
        rope = InterleavedMRope(config)
        builder, op, graph = create_test_builder()
        pos_ids = create_test_input(builder, "pos_ids", [3, 2, 4])
        result = rope(op, pos_ids)
        assert len(result) == 2  # (cos, sin)
        assert graph.num_nodes() > 0


class TestApplyRotaryPosEmb:
    def test_apply_rotary_pos_emb_full(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 4, 64])
        cos = create_test_input(builder, "cos", [2, 4, 8])
        sin = create_test_input(builder, "sin", [2, 4, 8])

        result = apply_rotary_pos_emb(op, x, (cos, sin), num_heads=4, rotary_embedding_dim=0)
        assert result is not None
        assert graph.num_nodes() > 0

    def test_get_rotary_pos_emb(self):
        builder, op, _graph = create_test_builder()
        pos_ids = create_test_input(builder, "pos_ids", [2, 4])
        cos_cache = create_test_input(builder, "cos_cache", [32, 8])
        sin_cache = create_test_input(builder, "sin_cache", [32, 8])

        cos, sin = get_rotary_pos_emb(op, pos_ids, cos_cache, sin_cache)
        assert cos is not None
        assert sin is not None


class TestYarnRopeAttnScale:
    """Tests for YarnRope with llama_4_attn_scale (Ministral3)."""

    def test_yarn_without_attn_scale_returns_2tuple(self):
        """Standard YaRN (no llama_4_scaling_beta) returns (cos, sin)."""
        from mobius.components._rotary_embedding import YarnRope

        config = make_config(
            head_dim=128,
            max_position_embeddings=16384,
            rope_theta=1000000.0,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 16.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 16384,
            },
        )
        rope = YarnRope(config)
        builder, op, _graph = create_test_builder()
        pos_ids = create_test_input(builder, "pos_ids", [1, 4])
        result = rope.forward(op, pos_ids)
        assert len(result) == 2, "Without llama_4_scaling_beta, should return (cos, sin)"

    def test_yarn_with_attn_scale_returns_3tuple(self):
        """YaRN with llama_4_scaling_beta returns (cos, sin, attn_scale)."""
        from mobius.components._rotary_embedding import YarnRope

        config = make_config(
            head_dim=128,
            max_position_embeddings=262144,
            rope_theta=1000000.0,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 16.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 16384,
                "llama_4_scaling_beta": 0.1,
            },
        )
        rope = YarnRope(config)
        builder, op, _graph = create_test_builder()
        pos_ids = create_test_input(builder, "pos_ids", [1, 4])
        result = rope.forward(op, pos_ids)
        assert len(result) == 3, (
            "With llama_4_scaling_beta, should return (cos, sin, attn_scale)"
        )

    def test_apply_rotary_pos_emb_ignores_3rd_element(self):
        """apply_rotary_pos_emb should work with both 2-tuple and 3-tuple."""
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 4, 64])
        cos = create_test_input(builder, "cos", [1, 4, 8])
        sin = create_test_input(builder, "sin", [1, 4, 8])
        scale = create_test_input(builder, "scale", [1, 4, 1])

        # 3-tuple should work — apply_rotary_pos_emb only uses [0] and [1]
        result = apply_rotary_pos_emb(op, x, (cos, sin, scale), num_heads=4)
        assert result is not None
