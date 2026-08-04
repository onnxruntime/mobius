# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for ViT weight rename logic.

Tests _rename_vit_weight with fake state dicts covering standard ViT,
DINOv2, BeiT, and SAM weight naming conventions.  No HF model downloads
needed.
"""

from __future__ import annotations

import pytest

from mobius.models.vit import _rename_vit_weight


class TestRenameVitWeight:
    """Test _rename_vit_weight for various HF ViT weight name patterns."""

    # --- Standard ViT ---

    def test_cls_token(self):
        assert _rename_vit_weight("vit.embeddings.cls_token") == "embeddings.cls_token"

    def test_position_embeddings(self):
        assert (
            _rename_vit_weight("vit.embeddings.position_embeddings")
            == "embeddings.position_embeddings"
        )

    def test_patch_embeddings_projection(self):
        assert (
            _rename_vit_weight("vit.embeddings.patch_embeddings.projection.weight")
            == "embeddings.patch_embeddings.projection.weight"
        )

    def test_layernorm(self):
        assert _rename_vit_weight("vit.layernorm.weight") == "layernorm.weight"
        assert _rename_vit_weight("vit.layernorm.bias") == "layernorm.bias"

    def test_encoder_layer_attention(self):
        result = _rename_vit_weight("vit.encoder.layer.3.attention.attention.query.weight")
        assert result == "encoder.layer.3.self_attn.q_proj.weight"

    def test_encoder_layer_key(self):
        result = _rename_vit_weight("vit.encoder.layer.0.attention.attention.key.bias")
        assert result == "encoder.layer.0.self_attn.k_proj.bias"

    def test_encoder_layer_value(self):
        result = _rename_vit_weight("vit.encoder.layer.1.attention.attention.value.weight")
        assert result == "encoder.layer.1.self_attn.v_proj.weight"

    def test_encoder_layer_output_dense(self):
        result = _rename_vit_weight("vit.encoder.layer.2.attention.output.dense.weight")
        assert result == "encoder.layer.2.self_attn.out_proj.weight"

    def test_encoder_layer_intermediate(self):
        result = _rename_vit_weight("vit.encoder.layer.5.intermediate.dense.weight")
        assert result == "encoder.layer.5.mlp.up_proj.weight"

    def test_encoder_layer_output(self):
        result = _rename_vit_weight("vit.encoder.layer.5.output.dense.bias")
        assert result == "encoder.layer.5.mlp.down_proj.bias"

    def test_encoder_layernorm_before(self):
        result = _rename_vit_weight("vit.encoder.layer.0.layernorm_before.weight")
        assert result == "encoder.layer.0.layernorm_before.weight"

    def test_encoder_layernorm_after(self):
        result = _rename_vit_weight("vit.encoder.layer.1.layernorm_after.bias")
        assert result == "encoder.layer.1.layernorm_after.bias"

    # --- DINOv2 ---

    def test_dinov2_prefix_strip(self):
        result = _rename_vit_weight("dinov2.encoder.layer.0.attention.attention.query.weight")
        assert result == "encoder.layer.0.self_attn.q_proj.weight"

    def test_dinov2_norm1_to_layernorm_before(self):
        result = _rename_vit_weight("dinov2.encoder.layer.2.norm1.weight")
        assert result == "encoder.layer.2.layernorm_before.weight"

    def test_dinov2_norm2_to_layernorm_after(self):
        result = _rename_vit_weight("dinov2.encoder.layer.3.norm2.bias")
        assert result == "encoder.layer.3.layernorm_after.bias"

    def test_dinov2_mlp_fc1_to_up_proj(self):
        result = _rename_vit_weight("dinov2.encoder.layer.0.mlp.fc1.weight")
        assert result == "encoder.layer.0.mlp.up_proj.weight"

    def test_dinov2_mlp_fc2_to_down_proj(self):
        result = _rename_vit_weight("dinov2.encoder.layer.1.mlp.fc2.bias")
        assert result == "encoder.layer.1.mlp.down_proj.bias"

    def test_dinov2_mask_token_skipped(self):
        assert _rename_vit_weight("dinov2.embeddings.mask_token") is None

    def test_dinov2_layer_scale_skipped(self):
        assert _rename_vit_weight("dinov2.encoder.layer.0.layer_scale1") is None
        assert _rename_vit_weight("dinov2.encoder.layer.1.layer_scale2") is None

    # --- BeiT ---

    def test_beit_prefix_strip(self):
        result = _rename_vit_weight("beit.encoder.layer.0.attention.attention.query.weight")
        assert result == "encoder.layer.0.self_attn.q_proj.weight"

    def test_beit_pooler_layernorm(self):
        result = _rename_vit_weight("beit.pooler.layernorm.weight")
        assert result == "layernorm.weight"

    def test_beit_lambda_skipped(self):
        assert _rename_vit_weight("beit.encoder.layer.0.lambda_1") is None
        assert _rename_vit_weight("beit.encoder.layer.0.lambda_2") is None

    def test_beit_relative_position_bias_skipped(self):
        assert (
            _rename_vit_weight(
                "beit.encoder.layer.0.attention.attention.relative_position_bias.weight"
            )
            is None
        )

    # --- Classifier / pooler skips ---

    def test_classifier_skipped(self):
        assert _rename_vit_weight("vit.classifier.weight") is None
        assert _rename_vit_weight("vit.classifier.bias") is None

    def test_pooler_dense_skipped(self):
        assert _rename_vit_weight("vit.pooler.dense.weight") is None


class TestRenameVitWeightTransformers5x:
    """transformers >=5.x flattened ``layers.N.<sub>`` encoder layout.

    The 5.x ``ViTForImageClassification`` checkpoint prefixes encoder layers
    with ``vit.layers.N.*`` (no ``encoder.`` segment) and consolidates the
    attention/MLP projections.  The prefix-strip allowlist must include
    ``layers.`` so these keys are not silently dropped (returned ``None``).
    """

    def test_vit5x_attention_q_proj(self):
        assert (
            _rename_vit_weight("vit.layers.0.attention.q_proj.weight")
            == "encoder.layer.0.self_attn.q_proj.weight"
        )

    def test_vit5x_attention_k_proj(self):
        assert (
            _rename_vit_weight("vit.layers.2.attention.k_proj.bias")
            == "encoder.layer.2.self_attn.k_proj.bias"
        )

    def test_vit5x_attention_v_proj(self):
        assert (
            _rename_vit_weight("vit.layers.1.attention.v_proj.weight")
            == "encoder.layer.1.self_attn.v_proj.weight"
        )

    def test_vit5x_attention_o_proj(self):
        assert (
            _rename_vit_weight("vit.layers.3.attention.o_proj.weight")
            == "encoder.layer.3.self_attn.out_proj.weight"
        )

    def test_vit5x_mlp_fc1(self):
        assert (
            _rename_vit_weight("vit.layers.0.mlp.fc1.weight")
            == "encoder.layer.0.mlp.up_proj.weight"
        )

    def test_vit5x_mlp_fc2(self):
        assert (
            _rename_vit_weight("vit.layers.5.mlp.fc2.bias")
            == "encoder.layer.5.mlp.down_proj.bias"
        )

    def test_vit5x_layernorm_before(self):
        assert (
            _rename_vit_weight("vit.layers.0.layernorm_before.weight")
            == "encoder.layer.0.layernorm_before.weight"
        )

    def test_vit5x_layernorm_after(self):
        assert (
            _rename_vit_weight("vit.layers.4.layernorm_after.bias")
            == "encoder.layer.4.layernorm_after.bias"
        )

    def test_vit5x_layers_not_dropped(self):
        """Regression: ``layers.N.*`` keys must not silently map to None."""
        keys = [
            "vit.layers.0.attention.q_proj.weight",
            "vit.layers.0.attention.k_proj.weight",
            "vit.layers.0.attention.v_proj.weight",
            "vit.layers.0.attention.o_proj.weight",
            "vit.layers.0.mlp.fc1.weight",
            "vit.layers.0.mlp.fc2.weight",
            "vit.layers.0.layernorm_before.weight",
            "vit.layers.0.layernorm_after.weight",
        ]
        for key in keys:
            assert _rename_vit_weight(key) is not None, f"Key {key!r} mapped to None"


class TestRenameVitWeightBatch:
    """Test full fake state_dict rename round-trips."""

    def test_standard_vit_state_dict(self):
        """Standard ViT state dict keys all rename without None."""
        fake_keys = [
            "vit.embeddings.cls_token",
            "vit.embeddings.position_embeddings",
            "vit.embeddings.patch_embeddings.projection.weight",
            "vit.embeddings.patch_embeddings.projection.bias",
            "vit.encoder.layer.0.layernorm_before.weight",
            "vit.encoder.layer.0.layernorm_before.bias",
            "vit.encoder.layer.0.attention.attention.query.weight",
            "vit.encoder.layer.0.attention.attention.key.weight",
            "vit.encoder.layer.0.attention.attention.value.weight",
            "vit.encoder.layer.0.attention.output.dense.weight",
            "vit.encoder.layer.0.intermediate.dense.weight",
            "vit.encoder.layer.0.output.dense.weight",
            "vit.encoder.layer.0.layernorm_after.weight",
            "vit.layernorm.weight",
        ]
        for key in fake_keys:
            result = _rename_vit_weight(key)
            assert result is not None, f"Key {key!r} mapped to None"

    @pytest.mark.parametrize("prefix", ["vit", "beit", "deit", "dinov2", "swin", "hiera"])
    def test_prefix_stripping(self, prefix: str):
        """All model-type prefixes are stripped properly."""
        result = _rename_vit_weight(f"{prefix}.embeddings.cls_token")
        assert result == "embeddings.cls_token"
