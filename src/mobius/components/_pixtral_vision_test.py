# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for Pixtral vision encoder components."""

from __future__ import annotations

import numpy as np

from mobius._configs import ArchitectureConfig, VisionConfig
from mobius.components._pixtral_vision import (
    Mistral3MultiModalProjector,
    Mistral3PatchMerger,
    PixtralRoPE2D,
    PixtralTransformerEncoder,
    PixtralVisionTower,
)


def test_pixtral_rope_2d_cache_shapes():
    """2D RoPE produces correct cache shapes."""
    rope = PixtralRoPE2D(head_dim=16, max_grid_size=4)
    # 4x4 grid = 16 positions, cache dim = head_dim/2 = 8
    assert list(rope.cos_cache.shape) == [16, 8]
    assert list(rope.sin_cache.shape) == [16, 8]


def test_pixtral_rope_2d_cache_values():
    """2D RoPE cache has non-trivial values at non-zero positions."""
    rope = PixtralRoPE2D(head_dim=16, max_grid_size=4)
    cos_data = rope.cos_cache.const_value.numpy()
    sin_data = rope.sin_cache.const_value.numpy()
    # Position (0,0) should have cos=1, sin=0 (freq*0=0)
    np.testing.assert_allclose(cos_data[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(sin_data[0], 0.0, atol=1e-6)
    # Non-zero positions should differ
    assert not np.allclose(cos_data[5], cos_data[0])


def test_pixtral_rope_2d_small_grid():
    """2D RoPE works with smallest possible grid (1x1)."""
    rope = PixtralRoPE2D(head_dim=8, max_grid_size=1)
    assert list(rope.cos_cache.shape) == [1, 4]


def test_pixtral_vision_tower_builds():
    """PixtralVisionTower constructs without errors."""
    config = ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=100,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
            model_type="pixtral",
        ),
    )
    tower = PixtralVisionTower(config)
    assert tower.ln_pre is not None
    assert tower.transformer is not None
    assert tower.rope is not None
    assert tower.patch_conv is not None


def test_mistral3_projector_builds():
    """Mistral3MultiModalProjector constructs without errors."""
    proj = Mistral3MultiModalProjector(
        vision_hidden_size=32,
        text_hidden_size=64,
        spatial_merge_size=2,
    )
    assert proj.patch_merger is not None
    assert proj.norm is not None
    assert proj.linear_1 is not None
    assert proj.linear_2 is not None


def test_transformer_encoder_layer_count():
    """Encoder creates the correct number of layers."""
    enc = PixtralTransformerEncoder(
        num_layers=3,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        head_dim=16,
    )
    assert len(list(enc.layers)) == 3


def test_patch_merger_builds():
    """PatchMerger reduces from merged_dim to hidden_size."""
    merger = Mistral3PatchMerger(hidden_size=32, spatial_merge_size=2)
    # input_dim = 32 * 2 * 2 = 128, output_dim = 32
    assert list(merger.merging_layer.weight.shape) == [32, 128]
