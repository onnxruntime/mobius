# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for SAM vision encoder weight rename logic."""

from __future__ import annotations

from mobius.components._sam_vision import preprocess_sam_encoder_weights


class TestPreprocessSamEncoderWeights:
    """Test preprocess_sam_encoder_weights with fake state dicts."""

    def test_patch_embed_projection(self):
        sd = {"patch_embed.projection.weight": "w"}
        result = preprocess_sam_encoder_weights(sd)
        assert "patch_embed.proj.weight" in result

    def test_neck_conv1_to_indexed(self):
        sd = {"neck.conv1.weight": "w"}
        result = preprocess_sam_encoder_weights(sd)
        assert "neck.0.weight" in result

    def test_neck_layer_norm1_to_indexed(self):
        sd = {"neck.layer_norm1.weight": "w", "neck.layer_norm1.bias": "b"}
        result = preprocess_sam_encoder_weights(sd)
        assert "neck.1.weight" in result
        assert "neck.1.bias" in result

    def test_neck_conv2_to_indexed(self):
        sd = {"neck.conv2.weight": "w"}
        result = preprocess_sam_encoder_weights(sd)
        assert "neck.2.weight" in result

    def test_neck_layer_norm2_to_indexed(self):
        sd = {"neck.layer_norm2.weight": "w"}
        result = preprocess_sam_encoder_weights(sd)
        assert "neck.3.weight" in result

    def test_layers_to_blocks(self):
        sd = {"layers.0.attn.q_proj.weight": "w"}
        result = preprocess_sam_encoder_weights(sd)
        assert "blocks.0.attn.q_proj.weight" in result

    def test_layer_norm1_to_norm1(self):
        sd = {"layers.0.layer_norm1.weight": "w"}
        result = preprocess_sam_encoder_weights(sd)
        assert "blocks.0.norm1.weight" in result

    def test_layer_norm2_to_norm2(self):
        sd = {"layers.1.layer_norm2.bias": "b"}
        result = preprocess_sam_encoder_weights(sd)
        assert "blocks.1.norm2.bias" in result

    def test_mlp_lin1_to_up_proj(self):
        sd = {"layers.0.mlp.lin1.weight": "w"}
        result = preprocess_sam_encoder_weights(sd)
        assert "blocks.0.mlp.up_proj.weight" in result

    def test_mlp_lin2_to_down_proj(self):
        sd = {"layers.0.mlp.lin2.bias": "b"}
        result = preprocess_sam_encoder_weights(sd)
        assert "blocks.0.mlp.down_proj.bias" in result

    def test_full_state_dict_roundtrip(self):
        """All keys in a typical SAM state dict get renamed."""
        fake_sd = {
            "patch_embed.projection.weight": "w",
            "patch_embed.projection.bias": "b",
            "layers.0.layer_norm1.weight": "w",
            "layers.0.layer_norm1.bias": "b",
            "layers.0.attn.q_proj.weight": "w",
            "layers.0.attn.k_proj.weight": "w",
            "layers.0.attn.v_proj.weight": "w",
            "layers.0.attn.proj.weight": "w",
            "layers.0.layer_norm2.weight": "w",
            "layers.0.mlp.lin1.weight": "w",
            "layers.0.mlp.lin2.weight": "w",
            "neck.conv1.weight": "w",
            "neck.layer_norm1.weight": "w",
            "neck.layer_norm1.bias": "b",
            "neck.conv2.weight": "w",
            "neck.layer_norm2.weight": "w",
            "neck.layer_norm2.bias": "b",
            "pos_embed": "pe",
        }
        result = preprocess_sam_encoder_weights(fake_sd)
        expected_keys = {
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            "blocks.0.norm1.weight",
            "blocks.0.norm1.bias",
            "blocks.0.attn.q_proj.weight",
            "blocks.0.attn.k_proj.weight",
            "blocks.0.attn.v_proj.weight",
            "blocks.0.attn.proj.weight",
            "blocks.0.norm2.weight",
            "blocks.0.mlp.up_proj.weight",
            "blocks.0.mlp.down_proj.weight",
            "neck.0.weight",
            "neck.1.weight",
            "neck.1.bias",
            "neck.2.weight",
            "neck.3.weight",
            "neck.3.bias",
            "pos_embed",
        }
        assert set(result.keys()) == expected_keys
