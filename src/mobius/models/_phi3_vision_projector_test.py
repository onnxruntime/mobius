# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the Phi-3.5-Vision host-side HD feature transform + projector.

These are L1 (pure NumPy) tests: they verify the 2x2 spatial patch merge, the
learnable separator insertion, the ``img_projection`` MLP math, the ``sub_glb``
token ordering / token counts, and the checkpoint-weight loader — all without
downloading the real checkpoint. Correctness of the transform against the
HuggingFace reference is additionally covered end-to-end by the L4 golden test
(``testdata/cases/vision-language/phi3_5-vision-instruct.yaml``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mobius.models._phi3_vision_projector import (
    Phi3VisionProjectorWeights,
    _add_image_newline,
    _apply_image_projection,
    _reshape_hd_patches_2x2_merge,
    load_phi3_vision_projector_weights,
    phi3_vision_hd_feature_transform,
)

# CLIP ViT-L/14-336 produces a 24x24 = 576 patch grid.
_PATCH_COUNT = 576
_GRID_SIDE = 24


def _make_projector_weights(
    image_dim_out: int, hidden_size: int, seed: int = 0
) -> Phi3VisionProjectorWeights:
    """Build small random projector weights with the real tensor ranks/shapes."""
    rng = np.random.default_rng(seed)
    merged_dim = image_dim_out * 4
    return Phi3VisionProjectorWeights(
        global_separator=rng.standard_normal((1, 1, merged_dim)).astype(np.float32),
        sublayer_separator=rng.standard_normal((1, 1, 1, merged_dim)).astype(np.float32),
        projection_first_weight=rng.standard_normal((hidden_size, merged_dim)).astype(
            np.float32
        ),
        projection_first_bias=rng.standard_normal((hidden_size,)).astype(np.float32),
        projection_second_weight=rng.standard_normal((hidden_size, hidden_size)).astype(
            np.float32
        ),
        projection_second_bias=rng.standard_normal((hidden_size,)).astype(np.float32),
    )


def test_reshape_hd_patches_2x2_merge_orders_the_four_subpatches() -> None:
    """Each merged token concatenates its 2x2 block in (row, col) raster order."""
    # One image, one crop, single channel. Patch value encodes its grid position
    # so we can assert exactly which four patches land in each merged token.
    values = np.arange(_PATCH_COUNT, dtype=np.float32).reshape(1, _PATCH_COUNT, 1)
    merged = _reshape_hd_patches_2x2_merge(values, height_crops=1, width_crops=1)

    # (num_images, h_crop*12, w_crop*12, 4*C) = (1, 12, 12, 4)
    assert merged.shape == (1, 12, 12, 4)
    # Top-left merged token = patches (0,0),(0,1),(1,0),(1,1) = [0, 1, 24, 25].
    np.testing.assert_array_equal(merged[0, 0, 0], [0.0, 1.0, 24.0, 25.0])
    # Merged token (i=1, j=2) = patches at rows 2/3, cols 4/5.
    expected = [
        2 * _GRID_SIDE + 4,
        2 * _GRID_SIDE + 5,
        3 * _GRID_SIDE + 4,
        3 * _GRID_SIDE + 5,
    ]
    np.testing.assert_array_equal(merged[0, 1, 2], expected)


def test_reshape_hd_patches_rejects_wrong_patch_count() -> None:
    bad = np.zeros((1, _PATCH_COUNT - 1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="patches"):
        _reshape_hd_patches_2x2_merge(bad, 1, 1)


def test_reshape_hd_patches_rejects_indivisible_crop_grid() -> None:
    # 3 crops cannot be arranged as a 2x2 grid.
    three_crops = np.zeros((3, _PATCH_COUNT, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="divisible"):
        _reshape_hd_patches_2x2_merge(three_crops, height_crops=2, width_crops=2)


def test_add_image_newline_appends_one_separator_per_row() -> None:
    merged = np.zeros((1, 12, 12, 4), dtype=np.float32)
    separator = np.full((1, 1, 1, 4), 7.0, dtype=np.float32)
    with_newline = _add_image_newline(merged, separator)

    # Row width grows by one (the separator column): 12 -> 13, flattened.
    assert with_newline.shape == (1, 12 * 13, 4)
    # Every 13th token (end of each row) is the separator value.
    separators = with_newline[0].reshape(12, 13, 4)[:, 12, :]
    np.testing.assert_array_equal(separators, np.full((12, 4), 7.0))


def test_apply_image_projection_matches_manual_linear_gelu_linear() -> None:
    weights = _make_projector_weights(image_dim_out=8, hidden_size=16, seed=1)
    tokens = np.random.default_rng(2).standard_normal((5, 32)).astype(np.float32)

    result = _apply_image_projection(tokens, weights)

    hidden = tokens @ weights.projection_first_weight.T + weights.projection_first_bias
    gelu = np.array(
        [0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in hidden.reshape(-1)]
    ).reshape(hidden.shape)
    expected = gelu @ weights.projection_second_weight.T + weights.projection_second_bias

    assert result.shape == (5, 16)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_hd_feature_transform_token_count_and_shape_for_2x2_crops() -> None:
    """A 672x672 image (2x2 HD crops) yields 757 projected image tokens.

    Layout per image = [sub features + newlines, glb_GN, global features +
    newlines] = 600 + 1 + 156 = 757, matching HuggingFace and the golden case.
    """
    image_dim_out, hidden_size = 8, 16
    weights = _make_projector_weights(image_dim_out, hidden_size, seed=3)
    num_crops_including_global = 5  # 1 global + 2x2 sub-crops
    image_features = np.random.default_rng(4).standard_normal(
        (1, num_crops_including_global, _PATCH_COUNT, image_dim_out)
    )
    image_sizes = np.array([[672, 672]])

    projected = phi3_vision_hd_feature_transform(image_features, image_sizes, weights)

    sub_tokens = 24 * (24 + 1)  # 2x2 crops -> 24x24 merged grid + newline column
    global_tokens = 12 * (12 + 1)  # 1x1 global crop -> 12x12 grid + newline column
    expected_tokens = sub_tokens + 1 + global_tokens
    assert expected_tokens == 757
    assert projected.shape == (expected_tokens, hidden_size)


def test_hd_feature_transform_places_global_separator_between_sub_and_global() -> None:
    """The single glb_GN separator token sits exactly after the sub features."""
    image_dim_out, hidden_size = 4, 6
    weights = _make_projector_weights(image_dim_out, hidden_size, seed=5)
    image_features = np.random.default_rng(6).standard_normal(
        (1, 5, _PATCH_COUNT, image_dim_out)
    )
    image_sizes = np.array([[672, 672]])

    # Project only the raw (pre-projection) token stream by using an identity-ish
    # projector is overkill; instead verify the separator lands at the right
    # index by re-deriving the un-projected concatenation length.
    sub_tokens = 24 * (24 + 1)
    projected = phi3_vision_hd_feature_transform(image_features, image_sizes, weights)

    # The projector is deterministic, so the separator token (index sub_tokens)
    # must equal the projection of the flattened glb_GN separator.
    separator_projected = _apply_image_projection(
        weights.global_separator.reshape(1, -1), weights
    )[0]
    np.testing.assert_allclose(
        projected[sub_tokens], separator_projected, rtol=1e-5, atol=1e-5
    )


def test_hd_feature_transform_rejects_non_4d_features() -> None:
    weights = _make_projector_weights(4, 6)
    with pytest.raises(ValueError, match="num_images"):
        phi3_vision_hd_feature_transform(
            np.zeros((5, _PATCH_COUNT, 4)), np.array([[336, 336]]), weights
        )


def test_load_projector_weights_reads_and_validates_safetensors(tmp_path) -> None:
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    image_dim_out, hidden_size = 8, 16
    merged_dim = image_dim_out * 4
    tensors = {
        "model.vision_embed_tokens.glb_GN": np.ones((1, 1, merged_dim), dtype=np.float32),
        "model.vision_embed_tokens.sub_GN": np.full(
            (1, 1, 1, merged_dim), 2.0, dtype=np.float32
        ),
        "model.vision_embed_tokens.img_projection.0.weight": np.zeros(
            (hidden_size, merged_dim), dtype=np.float32
        ),
        "model.vision_embed_tokens.img_projection.0.bias": np.zeros(
            (hidden_size,), dtype=np.float32
        ),
        "model.vision_embed_tokens.img_projection.2.weight": np.zeros(
            (hidden_size, hidden_size), dtype=np.float32
        ),
        "model.vision_embed_tokens.img_projection.2.bias": np.zeros(
            (hidden_size,), dtype=np.float32
        ),
        # An unrelated tensor that must be ignored by the loader.
        "model.layers.0.mlp.weight": np.zeros((2, 2), dtype=np.float32),
    }
    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    safetensors_numpy.save_file(tensors, str(shard_path))

    weights = load_phi3_vision_projector_weights(str(tmp_path))

    assert weights.global_separator.shape == (1, 1, merged_dim)
    assert weights.sublayer_separator.shape == (1, 1, 1, merged_dim)
    assert weights.projection_first_weight.shape == (hidden_size, merged_dim)
    np.testing.assert_array_equal(
        weights.sublayer_separator, tensors["model.vision_embed_tokens.sub_GN"]
    )


def test_load_projector_weights_raises_on_missing_shards(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="safetensors"):
        load_phi3_vision_projector_weights(str(tmp_path))


def test_load_projector_weights_raises_on_missing_projector_tensor(tmp_path) -> None:
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    # Shard exists but lacks the projector tensors.
    safetensors_numpy.save_file(
        {"model.layers.0.mlp.weight": np.zeros((2, 2), dtype=np.float32)},
        str(tmp_path / "model.safetensors"),
    )
    with pytest.raises(KeyError, match="projector"):
        load_phi3_vision_projector_weights(str(tmp_path))
