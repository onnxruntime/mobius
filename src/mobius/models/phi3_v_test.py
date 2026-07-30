# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for Phi-3/3.5-Vision (phi3_v) vision-tower weight mapping.

Verifies that the non-standard ``img_processor``-nested CLIP weights are
mapped onto the CLIP vision-tower initializer names, and that the host-side
img_projection / learnable-separator tensors are dropped. No HF downloads.
"""

from __future__ import annotations

import torch

from mobius.models.phi3_v import (
    _Phi3VVisionEncoderModel,
    _rename_phi3v_vision_weight,
)


class TestRenamePhi3VVisionWeight:
    """Test _rename_phi3v_vision_weight for the img_processor CLIP prefix."""

    _PFX = "model.vision_embed_tokens.img_processor.vision_model."

    def test_class_embedding(self):
        assert (
            _rename_phi3v_vision_weight(self._PFX + "embeddings.class_embedding")
            == "vision_tower.embeddings.class_embedding"
        )

    def test_patch_embedding_wraps_projection(self):
        assert (
            _rename_phi3v_vision_weight(self._PFX + "embeddings.patch_embedding.weight")
            == "vision_tower.embeddings.patch_embedding.projection.weight"
        )

    def test_position_embedding(self):
        assert (
            _rename_phi3v_vision_weight(self._PFX + "embeddings.position_embedding.weight")
            == "vision_tower.embeddings.position_embedding.weight"
        )

    def test_pre_layrnorm(self):
        assert (
            _rename_phi3v_vision_weight(self._PFX + "pre_layrnorm.weight")
            == "vision_tower.pre_layrnorm.weight"
        )

    def test_encoder_layer_mlp(self):
        assert (
            _rename_phi3v_vision_weight(self._PFX + "encoder.layers.7.mlp.fc1.weight")
            == "vision_tower.encoder.7.mlp.up_proj.weight"
        )

    def test_img_projection_dropped(self):
        # Projector is applied host-side (HD transform) — not exported.
        assert (
            _rename_phi3v_vision_weight("model.vision_embed_tokens.img_projection.0.weight")
            is None
        )
        assert (
            _rename_phi3v_vision_weight("model.vision_embed_tokens.img_projection.2.weight")
            is None
        )

    def test_learnable_separators_dropped(self):
        assert _rename_phi3v_vision_weight("model.vision_embed_tokens.sub_GN") is None
        assert _rename_phi3v_vision_weight("model.vision_embed_tokens.glb_GN") is None

    def test_text_decoder_weights_dropped(self):
        assert _rename_phi3v_vision_weight("model.layers.0.self_attn.qkv_proj.weight") is None
        assert _rename_phi3v_vision_weight("lm_head.weight") is None


class TestPhi3VVisionPreprocessWeights:
    """The vision encoder maps a realistic checkpoint slice with no leftovers."""

    def _config(self, feature_layer):
        from mobius._configs import ArchitectureConfig
        from mobius._configs._sub_configs import VisionConfig

        vision = VisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=3,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
            norm_eps=1e-5,
            hidden_act="quick_gelu",
            feature_layer=feature_layer,
        )
        return ArchitectureConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            num_hidden_layers=1,
            vocab_size=32,
            max_position_embeddings=64,
            rms_norm_eps=1e-5,
            rope_type="default",
            rope_theta=10_000.0,
            pad_token_id=0,
            vision=vision,
            image_token_id=32044,
        )

    def test_feature_layer_maps_only_needed_layers(self):
        pfx = "model.vision_embed_tokens.img_processor.vision_model."
        state_dict = {
            pfx + "embeddings.class_embedding": torch.zeros(8),
            pfx + "embeddings.patch_embedding.weight": torch.zeros(8, 3, 14, 14),
            pfx + "embeddings.position_embedding.weight": torch.zeros(5, 8),
            pfx + "pre_layrnorm.weight": torch.zeros(8),
            pfx + "encoder.layers.0.mlp.fc1.weight": torch.zeros(16, 8),
            pfx + "encoder.layers.2.mlp.fc1.weight": torch.zeros(16, 8),
            pfx + "post_layernorm.weight": torch.zeros(8),
            "model.vision_embed_tokens.img_projection.0.weight": torch.zeros(16, 32),
            "model.vision_embed_tokens.sub_GN": torch.zeros(1, 1, 1, 32),
        }
        module = _Phi3VVisionEncoderModel(self._config(feature_layer=-2))
        renamed = module.preprocess_weights(state_dict)

        assert "vision_tower.embeddings.class_embedding" in renamed
        assert "vision_tower.embeddings.patch_embedding.projection.weight" in renamed
        assert "vision_tower.encoder.0.mlp.up_proj.weight" in renamed
        assert "vision_tower.encoder.2.mlp.up_proj.weight" not in renamed
        assert "vision_tower.post_layernorm.weight" not in renamed
        # Projector + separator are host-side.
        assert not any("projection.0" in k or "sub_GN" in k for k in renamed)
        # Only vision_tower.* names are produced.
        assert all(k.startswith("vision_tower.") for k in renamed)
