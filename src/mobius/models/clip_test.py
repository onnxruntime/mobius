# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for CLIP vision weight rename and feature-layer selection.

Covers the pieces exercised by CLIP-based multimodal encoders such as
Phi-3.5-Vision: the HuggingFace → ONNX weight name mapping (including the
bias-free patch convolution wrapping) and the intermediate feature-layer
math. No HF model downloads needed.
"""

from __future__ import annotations

import pytest

from mobius.models.clip import (
    _rename_clip_vision_weight,
    resolve_clip_feature_num_layers,
)


class TestRenameClipVisionWeight:
    """Test _rename_clip_vision_weight for HF CLIP vision weight patterns."""

    def test_class_embedding(self):
        assert (
            _rename_clip_vision_weight("vision_model.embeddings.class_embedding")
            == "embeddings.class_embedding"
        )

    def test_position_embedding(self):
        assert (
            _rename_clip_vision_weight("vision_model.embeddings.position_embedding.weight")
            == "embeddings.position_embedding.weight"
        )

    def test_patch_embedding_wraps_projection(self):
        # HF stores a flat patch_embedding.weight; our module nests it under
        # a .projection Conv sub-module.
        assert (
            _rename_clip_vision_weight("vision_model.embeddings.patch_embedding.weight")
            == "embeddings.patch_embedding.projection.weight"
        )

    def test_pre_and_post_layernorm(self):
        assert (
            _rename_clip_vision_weight("vision_model.pre_layrnorm.weight")
            == "pre_layrnorm.weight"
        )
        assert (
            _rename_clip_vision_weight("vision_model.post_layernorm.bias")
            == "post_layernorm.bias"
        )

    def test_encoder_attention_projection(self):
        assert (
            _rename_clip_vision_weight("vision_model.encoder.layers.5.self_attn.q_proj.weight")
            == "encoder.5.self_attn.q_proj.weight"
        )

    def test_encoder_mlp_renamed(self):
        assert (
            _rename_clip_vision_weight("vision_model.encoder.layers.0.mlp.fc1.weight")
            == "encoder.0.mlp.up_proj.weight"
        )
        assert (
            _rename_clip_vision_weight("vision_model.encoder.layers.0.mlp.fc2.bias")
            == "encoder.0.mlp.down_proj.bias"
        )

    def test_text_weights_dropped(self):
        assert _rename_clip_vision_weight("text_model.encoder.layers.0.mlp.fc1.weight") is None
        assert _rename_clip_vision_weight("visual_projection.weight") is None


class TestResolveClipFeatureNumLayers:
    """Test resolve_clip_feature_num_layers (HF hidden_states indexing)."""

    def test_negative_two_skips_last_layer(self):
        # 24 layers -> 25 hidden states; index -2 == state 23 == run 23 layers.
        assert resolve_clip_feature_num_layers(24, -2) == 23

    def test_negative_one_is_last_layer(self):
        assert resolve_clip_feature_num_layers(24, -1) == 24

    def test_positive_index(self):
        assert resolve_clip_feature_num_layers(24, 10) == 10

    def test_zero_is_pre_encoder(self):
        assert resolve_clip_feature_num_layers(24, 0) == 0

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            resolve_clip_feature_num_layers(24, -30)
        with pytest.raises(ValueError):
            resolve_clip_feature_num_layers(24, 25)
